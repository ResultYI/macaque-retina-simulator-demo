"""Small manufactured graph checks; these are not real-retina acceptance runs."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from scipy.special import expit

from retina_sim.contracts import ContractError
from retina_sim.group_inner_dynamics import GroupInnerDynamics, key, BC_MECH, MEMBRANE, RODS, RGC_MECH
from retina_sim.rod_pathway import rod_source_values
from retina_sim.spatial_group_model import (GroupModelAssembly, POPULATIONS, BC, AC, RGC,
                                           REQUIRED_ROUTE_PAIRS)


ROOT = Path(__file__).resolve().parents[1]


def fixture_model():
    counts = {n: 1 for n in POPULATIONS}
    counts.update(rods=2, AII_AC=2, midget_BC_ON=2)
    populations = {n: SimpleNamespace(group_count=k, counts=np.arange(1, k+1, dtype=np.int64)) for n,k in counts.items()}
    contact_pairs = {('rods','rod_BC'):'individual_rod_RBC_contacts', ('rod_BC','AII_AC'):'RBC_AII_AMPA',
        ('AII_AC','midget_BC_OFF'):'AII_FMB_glycine', ('AII_AC','diffuse_BC_OFF'):'AII_DB1_glycine'}
    gap_names = ('AII_AC','midget_BC_ON','diffuse_BC_ON','S_ON_BC')
    gap_pairs = {('AII_AC','AII_AC'), *(p for n in gap_names[1:] for p in (('AII_AC',n),(n,'AII_AC')))}
    matrices, arrays, routes = {}, {}, {}
    for pre,post in REQUIRED_ROUTE_PAIRS:
        name=pre+'_to_'+post; comp=contact_pairs.get((pre,post))
        kind='gap' if (pre,post) in gap_pairs else 'native' if (pre,post) in (('cones','HC'),('HC','cones')) else 'primary_chemical' if comp else 'inner_chemical'
        routes[name]=SimpleNamespace(name=name,pre=pre,post=post,kind=kind,contact_component=comp,
                                     contact_groups=1 if comp else 0)
        matrices['routes/'+name+'/mean_weights.npz']=sparse.csr_matrix(np.ones((counts[post],counts[pre]))/counts[pre])
        if comp:
            arrays['routes/'+name+'/contact_post_group.npy']=np.array([0],dtype=np.int64)
            arrays['routes/'+name+'/contact_pre_group.npy']=np.array([0],dtype=np.int64)
            matrices['routes/'+name+'/contact_coefficient_S.npz']=sparse.csr_matrix(([2e-10],([0],[0])),shape=(counts[post],counts[pre]))
    size=sum(counts[n] for n in gap_names)
    adj=sparse.csr_matrix(([1e-10,1e-10],([0,2],[2,0])),shape=(size,size))
    matrices['gap/laplacian_S.npz']=sparse.diags(np.asarray(adj.sum(1)).ravel())-adj
    assembly=SimpleNamespace(populations=populations,routes=routes,
        manifest={'gap':{'population_order':gap_names},'files':{}},artifact_sha256='synthetic-fixture',
        matrix=lambda path:matrices[path],array=lambda path:arrays[path])
    assembly.state_blocks=GroupModelAssembly._states(assembly)
    assembly.probability_blocks=GroupModelAssembly._probability_blocks(assembly)
    params={'populations':{},'contacts':{},'route_gains':{},'rod_phototransduction':{},'identity':'software_fixture'}
    for pop in ('rods','rod_BC','AII_AC'):
        p=dict(C=2e-11,gL=1e-9,EL=-.06,EE=0.,release_half=-.05,release_slope=.005,tau_release=.003,
               current_anchor=0.,rmax=100.,Npool=10.,U=.1,tau_recovery=.05,r_half=-.045,r_slope=.005)
        params['populations'][pop]=p
    for pop in BC:
        params['populations'][pop]=dict(C=2e-11,gL=1e-9,Vref=-.04,native_voltage_gain=.01,native_reference=.5,
            release_half=-.04,release_slope=.005,tau_release=.005,EI=-.07,kernel=[.3,.7,-.2],
            baseline=.5,bias=0.,sample_interval=.001,adaptation_enabled=True,tau_a_seconds=.1,adaptation_scale=.75)
    for pop in AC:
        params['populations'][pop]=dict(reference_input=.5,input_slope=2.,output_scale=1.,tau_A_seconds=.01)
    for pop in RGC:
        params['populations'][pop]=dict(C=5e-11,gL=2e-9,EL=-.06,EE=0.,EI=-.07,alpha=50.,mu=-.04,beta=.005,
            spike_history_kernel=[-.005,-.002,-.001],spike_interval=.001)
    params['contacts']['individual_rod_RBC_contacts']=dict(K=.5,hill=2.,tau=.01)
    params['contacts']['RBC_AII_AMPA']=dict(tau_rise=.0002,tau_decay=.002)
    params['contacts']['AII_FMB_glycine']=dict(tau_rise=.00008,tau_decay=.003,decay_minus100=.0021,decay_0=.004)
    params['contacts']['AII_DB1_glycine']=dict(tau_rise=.00011,tau_decay=.004,decay_minus100=.0027,decay_0=.0053)
    for r in routes.values():
        if r.post in RGC:params['route_gains'][r.name]=1e-10
    p=rod_source_values(ROOT);p['q']=p['beta']*p['cdark']/(p['gdark']**p['n']*p['k'])
    p['smax']=p['eta']/p['phi']*p['gdark']*(1+(p['cdark']/p['kGC'])**p['m'])
    params['rod_phototransduction']=p
    model=GroupInnerDynamics(assembly,params)
    state={k:np.zeros(model.sizes[k],dtype=np.float64) for k in model.state_keys}
    for k in state:
        pop,mech,name=k
        if mech==RODS:
            initial=dict(R=0.,P=p['eta']/p['phi'],G=p['gdark'],C=p['cdark'],S=p['gdark']*p['eta']/p['phi'])
            state[k][:]=initial[name]
        elif name=='V':state[k][:]=-.05
        elif name in ('release','terminal_release'):state[k][:]=.5
        elif name=='x':state[k][:]=.8
        elif name=='A':state[k][:]=.375
        elif name=='h':state[k][:]=.5
        elif name=='m':state[k][:]=.5
        elif name=='activation':state[k][:]=np.logaddexp(0.,0.)
        elif mech=='RGC_expected_history' and name=='history_probability_0':state[k][:]=1.
    return model,state


def test_physical_totals_and_exact_gap_currents_conserve_charge():
    model,state=fixture_model()
    state[key('AII_AC',MEMBRANE,'V')][0]=-.04
    state[key('midget_BC_ON',BC_MECH,'V')][0]=-.06
    out=model.outputs(0.,state)
    assert sum(np.sum(out[n,'gap_current']) for n in model.gap_names)==pytest.approx(0.,abs=1e-27)
    np.testing.assert_array_equal(model.C['AII_AC'],np.array([2e-11,4e-11]))
    assert out['AII_AC','gap_current'][0]==pytest.approx(-2e-12)


def test_ribbon_rate_uses_voltage_resource_not_filtered_release():
    model,state=fixture_model();out=model.outputs(0.,state)
    changed=deepcopy(state);changed[key('AII_AC',MEMBRANE,'release')][:]=.01
    np.testing.assert_array_equal(model.outputs(0.,changed)['AII_AC','event_rate'],out['AII_AC','event_rate'])
    changed[key('AII_AC',MEMBRANE,'x')]*=.5
    np.testing.assert_allclose(model.outputs(0.,changed)['AII_AC','event_rate'],.5*out['AII_AC','event_rate'])


def test_glycine_decay_and_normalization_both_read_current_voltage():
    model,state=fixture_model();component='AII_FMB_glycine'
    state[key('AII_AC',component,'a')][:]=.01
    state[key('AII_AC',component,'b')][:]=.1
    first=model.outputs(0.,state)[component,'conductance']
    rhs1=model.rhs(0.,state)[key('AII_AC',component,'b')]
    state[key('midget_BC_OFF',BC_MECH,'V')][:]=-.08
    second=model.outputs(0.,state)[component,'conductance']
    rhs2=model.rhs(0.,state)[key('AII_AC',component,'b')]
    assert not np.array_equal(first,second)
    assert not np.array_equal(rhs1,rhs2)


def test_rod_old_new_order_and_common_BC_snapshot():
    model,state=fixture_model();original=deepcopy(state)
    light=np.array([100.,200.]);p=model.rod
    new,_,_=model.commit_ticks(.001,state,rod_light=light,cone_release=np.array([.9]),rod_tick=True,bc_tick=True)
    R,P,G,C,S=(original[key('rods',RODS,n)] for n in ('R','P','G','C','S'))
    expected_R=R+p['timeStep']*(-p['sigma']*R)
    expected_R=expected_R+p['gamma']*(light*p['timeStep'])
    np.testing.assert_array_equal(new[key('rods',RODS,'R')],expected_R)
    np.testing.assert_array_equal(new[key('rods',RODS,'G')],G+p['timeStep']*(S-P*G))
    for pop in BC:
        np.testing.assert_array_equal(new[key(pop,BC_MECH,'history_0')],np.full(model.counts[pop],.4))
        sign=-1 if pop.endswith('_ON') or pop=='S_ON_BC' else 1
        np.testing.assert_allclose(new[key(pop,BC_MECH,'activation')],np.logaddexp(0.,sign*.3*.4))
    for k in state:np.testing.assert_array_equal(state[k],original[k])


def test_strict_past_sampler_and_probability_mass_are_transactional():
    model,state=fixture_model()
    auxiliary={}
    for pop in RGC:
        gen=np.random.Generator(np.random.PCG64(82))
        auxiliary[pop]=dict(history=[[0]*model.counts[pop] for _ in range(3)],last_bin_counts=[1]*model.counts[pop],rng_state=deepcopy(gen.bit_generator.state),pending_events=[])
    saved=deepcopy(auxiliary)
    invalid=deepcopy(state)
    invalid[key(RGC[-1],'RGC_expected_history','history_probability_0')][:]=.5
    with pytest.raises(ContractError,match='probability law'):
        model.commit_ticks(.001,invalid,rgc_tick=True,spike_states=auxiliary)
    assert auxiliary==saved
    committed,after,events=model.commit_ticks(.001,state,rgc_tick=True,spike_states=auxiliary)
    assert auxiliary==saved
    for pop in RGC:
        assert after[pop]['history'][0]==[1]*model.counts[pop]
        probabilities=np.stack([committed[key(pop,'RGC_expected_history','history_probability_'+str(i))] for i in range(8)])
        np.testing.assert_allclose(probabilities.sum(0),1.,atol=2e-16,rtol=0)
        assert after[pop]['rng_state']!=saved[pop]['rng_state']


def test_candidate_matches_joint_same_equation_inside_interval():
    model,state=fixture_model()
    stop=.00004
    reference=model.advance_reference(state,0.,stop,rtol=1e-7,atol=1e-10)
    candidate=model.advance_candidate(state,0.,stop,rtol=1e-7,atol=1e-10)
    for result in (reference,candidate):
        assert result.accepted_times[0]==0. and result.accepted_times[-1]==stop
        assert tuple(sorted(set(result.accepted_times)))==result.accepted_times
        assert all(0.<=t<=stop for t in result.accepted_times)
    represented_endpoints={s.t1 for entries in candidate.representations.values() for _,series in entries
        for s in getattr(series,'representations',(series,))}
    assert represented_endpoints.issubset(set(candidate.accepted_times))
    for t,side in ((0.,'right'),(.000017,'left'),(stop,'left')):
        a,b=reference.evaluate(t,side),candidate.evaluate(t,side)
        for k in model.state_keys:
            np.testing.assert_allclose(a[k],b[k],rtol=1e-7,atol=1e-10,err_msg=str(k))
    assert candidate.metrics['fast_blocks']>=4
    assert candidate.metrics['forcing_source_evaluations']>0
    assert candidate.metrics['forcing_error_max_original_scale']<=.25
    assert candidate.metrics['quadrature_error_max_original_scale']<=.25
    original=candidate.rhs_at(.000017,'left')
    represented=candidate.representation_derivative(.000017,'left')
    for k in model.state_keys:
        np.testing.assert_allclose(original[k],represented[k],rtol=2e-5,atol=2e-6,err_msg=str(k))
    with pytest.raises(ContractError,match='wrong side'):
        candidate.evaluate(stop,'right')
    with pytest.raises(ContractError,match='interval'):
        candidate.evaluate(np.nextafter(stop,np.inf),'left')


def test_complete_layout_and_pack_reject_missing_or_nonfinite_state():
    model,state=fixture_model()
    restored=model.unpack(model.pack(state))
    for k in state:np.testing.assert_array_equal(restored[k],state[k])
    incomplete=dict(state);incomplete.pop(model.state_keys[0])
    with pytest.raises(ContractError,match='Complete'):
        model.rhs(0.,incomplete)
    broken=deepcopy(state);broken[model.state_keys[0]][0]=np.nan
    with pytest.raises(ContractError,match='Finite'):
        model.pack(broken)


def test_actual_events_cannot_silently_omit_readout_or_rng_and_compiled_dispatch():
    model,state=fixture_model()
    with pytest.raises(ContractError,match='Cone release'):
        model.commit_ticks(.001,state,bc_tick=True)
    with pytest.raises(ContractError,match='PCG64'):
        model.commit_ticks(.001,state,rgc_tick=True)
    calls=[]
    def compiled_fixture(t,y):
        calls.append(t)
        return model.pack(model.rhs(t,model.unpack(y,copy=False)))
    model.compiled_rhs=compiled_fixture
    result=model.advance_candidate(state,0.,.00002,rtol=1e-7,atol=1e-10)
    assert calls and result.metrics['compiled_inner_rhs'] is True


def test_actual_compiled_candidate_matches_independent_numpy_reference():
    from retina_sim.group_inner_compiled import CompiledGroupInnerRHS
    model,state=fixture_model()
    reference=model.advance_reference(state,0.,.00004,rtol=1e-7,atol=1e-10)
    model.compiled_rhs=CompiledGroupInnerRHS(model)
    candidate=model.advance_candidate(state,0.,.00004,rtol=1e-7,atol=1e-10)
    assert candidate.metrics['compiled_inner_rhs']
    for k in model.state_keys:
        np.testing.assert_allclose(reference.state[k],candidate.state[k],rtol=1e-7,atol=1e-10,err_msg=str(k))


def test_pure_transition_witness_never_invokes_sampler(monkeypatch):
    model,state=fixture_model()
    from retina_sim.inner import HistoryRGC
    def no_sample(*args,**kwargs):raise AssertionError('Monitor must not sample any RNG')
    monkeypatch.setattr(HistoryRGC,'sample_spikes',no_sample)
    before=deepcopy(state)
    result=model.pure_transition_witness(state,np.array([.9]),np.full(model.counts['rods'],100.))
    assert not np.array_equal(result[key('rods',RODS,'R')],state[key('rods',RODS,'R')])
    assert not np.array_equal(result[key(RGC[0],'RGC_expected_history','history_probability_0')],state[key(RGC[0],'RGC_expected_history','history_probability_0')])
    for k in state:np.testing.assert_array_equal(state[k],before[k])
