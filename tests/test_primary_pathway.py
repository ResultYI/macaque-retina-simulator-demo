"""Necessary synthetic integration checks; no animal fits or validation claims."""
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from retina_sim.contracts import Quantity,SolverConfig,RunMode
from retina_sim.assembly import RetinaBuilder
from retina_sim.network import JointNetwork,InputBundle
from examples.full_target_network import full_target_network

ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope='module')
def full():return full_target_network(ROOT,effective_cones=64,effective_rods=96,duration=.02)

def config():return SolverConfig(method='DOP853',rtol=1e-9,atol=1e-12,max_step=.00003,input_interpolation='analytic')

def test_all_targets_have_connected_actual_equations(full):
    c,ch,a=full
    assert not [x for x in c.spec.required_mechanisms if not any(x in cc.layers for cc in c.components.values())]
    assert len(c.spec.required_mechanisms)==20
    assert all(cc.mechanism not in ('RodMechanismPlugin','RodBipolarMechanismPlugin','AIIMechanismPlugin') for cc in c.components.values())
    graph={cid:set() for cid in c.components}
    for x in c.links:graph[x.specification.source_component].add(x.specification.target_component)
    for g in c.spec.gap_junctions:graph[g.left_component].add(g.right_component);graph[g.right_component].add(g.left_component)
    def reaches(start,target):
        seen=set();pending=[start]
        while pending:
            node=pending.pop()
            if node==target:return True
            if node not in seen:seen.add(node);pending.extend(graph[node]-seen)
        return False
    for src in ('rod','rod_BC','AII_AC','physical_OS','physical_membrane','physical_terminal','outer_native'):
        assert reaches(src,'midget_RGC_ON')
        assert reaches(src,'midget_RGC_OFF')
    assert set(a['cone_types'])>= {'NL','NM','Ns'} and a['cone_types']['Ns']>0
    assert c.clock['scheduler_kind']=='explicit_event_clock'
    assert c.clock['component_periods_seconds']['widefield_AC']==.00073

def test_mixed_author_discrete_parity_and_checkpoint_phase(full):
    c,ch,a=full;runner=c.runner();initial=c.initialize(config())
    one,r=c.advance(initial,ch,.004,sample_times=[0,.004])
    split,_=c.advance(initial,ch,.00137,sample_times=[0,.00137])
    continued,_=c.advance(c.restore(c.checkpoint(split)),ch,.004,sample_times=[.00137,.004])
    assert one.component_tick_indices==continued.component_tick_indices
    for cid in one.spike_states:
        assert one.spike_states[cid]['rng_state']==continued.spike_states[cid]['rng_state']
        assert one.spike_states[cid]['history']==continued.spike_states[cid]['history']
        first=one.spike_states[cid]['pending_events'];second=continued.spike_states[cid]['pending_events']
        assert [e['cell_index'] for e in first]==[e['cell_index'] for e in second]
        np.testing.assert_allclose([e['time_seconds'] for e in first],[e['time_seconds'] for e in second],atol=1e-12,rtol=0)
    for x,y in zip(one.ordered_state_arrays,continued.ordered_state_arrays):np.testing.assert_allclose(x,y,rtol=2e-7,atol=2e-8)
    for cid,channel in [('rod','rod_light'),('physical_OS','physical_cone_light')]:
        native=c.components[cid].native_model;p=native.p
        expected=tuple(q.values.copy() for q in c.components[cid].initial())
        for step in range(40):
            R,P,G,C,S=expected;dt=.0001;rate=ch[channel].at(step*dt).values
            rr=R+dt*(-p['sigma']*R)+p['gamma']*(rate*dt)
            pp=P+dt*(R+p['eta']-p['phi']*P)
            cc=C+dt*(native.q*p['k']*G**p['n']-p['beta']*C)
            ss=native.smax/(1+(cc/p['kGC'])**p['m']);gg=G+dt*(S-P*G)
            expected=(rr,pp,gg,cc,ss)
        actual=runner.model.local_blocks(cid,np.concatenate(one.ordered_state_arrays))
        for x,y in zip(actual,expected):np.testing.assert_allclose(x,y,rtol=0,atol=2e-12)

def test_rod_source_equilibrium_and_units(full):
    c,_,_=full;cc=c.components['rod'];native=cc.native_model;p=native.p
    b=tuple(q.values for q in cc.initial());rate=np.full(len(b[0]),5.)
    evolved=native.transition(b,{'light':Quantity(rate,'R*/s','photoisomerization_rate','synthetic_fixture')})
    for x,y in zip(b,evolved):np.testing.assert_allclose(x,y,atol=1e-12,rtol=0)
    assert native.dark_current==pytest.approx(37.23875)
    assert cc.state_specs[2].unit=='Chen_scaled_concentration'
    assert all(r.evidence_class=='published_fit' for r in cc.parameters.records)
    with pytest.raises(ValueError):native.transition(b,{'light':Quantity(-rate,'R*/s','photoisomerization_rate','synthetic_fixture')})

def test_gap_currents_conservative_and_bidirectional(full):
    c,ch,_=full;model=JointNetwork(c);y=np.concatenate([q.values for q in model.initial_conditions()])
    for k,(cid,cc) in enumerate(c.components.items()):
        if hasattr(cc.native_model,'membrane_capacitance'):
            block=model.slices[cid];j=cc.native_model.gap_state_index;n=len(cc.outputs['voltage'].cell_ids)
            y[block.start+j*n:block.start+(j+1)*n]=-.07+.002*k
    incoming,signals=model.evaluate(0,y,InputBundle({k:v.at(0) for k,v in ch.items()}))
    currents=[v.values for (cid,port),v in signals.items() if port=='gap_current']
    assert abs(sum(float(v.sum()) for v in currents))<1e-23
    voltages=[signals[(cid,'voltage')].values for cid,cc in c.components.items() if hasattr(cc.native_model,'membrane_capacitance')]
    minimum=min(v.min() for v in voltages);maximum=max(v.max() for v in voltages)
    for cid,cc in c.components.items():
        if not hasattr(cc.native_model,'membrane_capacitance'):continue
        V=signals[(cid,'voltage')].values;I=signals[(cid,'gap_current')].values
        assert np.all(I[V==minimum]>=-1e-23) and np.all(I[V==maximum]<=1e-23)
        blocks=model.local_blocks(cid,y);dr=cc.derivative(0,blocks,incoming[cid]);clean=dict(incoming[cid]);clean.pop('__gap_current__')
        no=cc.derivative(0,blocks,clean);j=cc.native_model.gap_state_index
        np.testing.assert_allclose((dr[j].values-no[j].values)*cc.native_model.membrane_capacitance,I,atol=1e-23,rtol=1e-10)
    reverse=tuple(replace(g,left_component=g.right_component,right_component=g.left_component,left_cell_ids=g.right_cell_ids,right_cell_ids=g.left_cell_ids) for g in c.spec.gap_junctions)
    other=JointNetwork(replace(c,spec=replace(c.spec,gap_junctions=reverse)))
    _,swapped=other.evaluate(0,y,InputBundle({k:v.at(0) for k,v in ch.items()}))
    for cid,port in signals:
        if port=='gap_current':np.testing.assert_allclose(signals[(cid,port)].values,swapped[(cid,port)].values,atol=1e-23)

def test_first_synapse_nonlinearity_precedes_pooling(full):
    c,_,_=full;cc=c.components['individual_rod_RBC_contacts'];n=len(cc.state_specs[0].cell_ids)
    drive=np.tile([0.,1.],n//2+1)[:n];zero=np.zeros(n)
    rate=cc.derivative(0,(zero,),{'release':Quantity(drive,'dimensionless','effective_synaptic_drive','synthetic_fixture')})[0].values
    gate=rate*.01
    assert np.mean(gate)>.4
    pooled_gate=1/(1+(.5/.25)**4)
    assert abs(gate.mean()-pooled_gate)>.3

def test_ribbon_depletion_recovery_and_event_peak(full):
    c,_,_=full;cc=c.components['rod_BC'];b=[q.values.copy() for q in cc.initial()]
    b[0][:]=-.03;b[2][:]=1.;n=len(b[0]);input={'excitation':Quantity(np.zeros(n),'S','excitatory_conductance','synthetic_fixture')}
    assert np.all(cc.derivative(0,b,input)[2].values<0)
    b[0][:]=-.1;b[2][:]=.2
    assert np.all(cc.derivative(0,b,input)[2].values>0)
    for name in ('RBC_AII_AMPA','AII_FMB_glycine','AII_DB1_glycine'):
        syn=c.components[name];p=syn.parameters
        rise=p.quantity('tau_rise','s','synaptic_rise_time_constant').values;decay=p.quantity('tau_decay','s','synaptic_decay_time_constant').values
        tp=np.log(decay/rise)/(1/rise-1/decay)
        peak=p.quantity('g_peak','S','single_event_peak_conductance').values
        sites=p.quantity('ncontact','dimensionless','effective_release_site_multiplier').values
        np.testing.assert_allclose(syn.native_model.unit_event_peak(tp),peak*sites,rtol=1e-12)

def test_receptor_ablations_remove_only_named_conductance(full):
    c,ch,a=full
    components=tuple(replace(s,options=dict(s.options,enabled=False)) if s.mechanism=='DoubleExponentialChemicalSynapse' else s for s in c.spec.components)
    muted=RetinaBuilder(a['registry']).build(replace(c.spec,components=components,gap_junctions=()),RunMode.REFERENCE)
    model=JointNetwork(muted);y=np.concatenate([q.values for q in model.initial_conditions()]);incoming,out=model.evaluate(0,y,InputBundle({k:v.at(0) for k,v in ch.items()}))
    for cid in ('RBC_AII_AMPA','AII_FMB_glycine','AII_DB1_glycine'):assert np.all(out[(cid,'conductance')].values==0)
    assert np.any(incoming['midget_BC_OFF']['drive'].values>0)
    for cid,port in out:
        if port=='gap_current':assert np.all(out[(cid,port)].values==0)

def test_measurements_do_not_disguise_project_numbers(full):
    c,_,a=full
    for name in ('AII_FMB_glycine','AII_DB1_glycine'):
        records={r.parameter_id:r for r in c.components[name].parameters.records}
        assert records['g_peak'].evidence_class=='project_approximation'
        assert 'Unverified primary-source quotation' in records['g_peak'].locator
        assert records['tau_rise'].evidence_class=='project_approximation'
        assert records['ncontact'].evidence_class=='project_approximation'
        assert records['decay_minus100'].evidence_class=='project_approximation'
    assert c.components['AII_AC'].parameters.evidence_class=='project_approximation'
    assert all(p.cells.evidence_tags[0]=='synthetic_fixture' for p in c.populations.values())
    assert a['effective_density_scale']!=1 and a['rod_geometry']['class']=='project_approximation'
