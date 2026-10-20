"""Synthetic equation/contract tests, not calibrated macaque coefficients."""
from pathlib import Path
from dataclasses import replace
import copy
import json
import hashlib
import sys
import numpy as np
import pytest
from scipy import sparse
from scipy.special import expit
from examples.full_network_fixture import population, fixture_profile, q, link
from examples.full_target_network import membrane_entries
from retina_sim.core_sources import load_core_registry
from retina_sim.assembly import ModelSpec, ComponentSpec, ExternalInput, RetinaBuilder, GapJunction
from retina_sim.contracts import RunMode, ContractError, SolverConfig, canonical_hash
from retina_sim.mechanism_contracts import EvidenceValue
from retina_sim.runtime_profiles import RuntimeProfile
from retina_sim.spatial import InputChannel
from retina_sim.network import JointNetwork, InputBundle
from retina_sim.calibration_modules import build_module, gap_boundary_id, parameter_package, merge_parameter_package
from retina_sim.bc_ac_candidate import CandidateProfile, CandidateParametersMissing, migrate_bc_ac
from retina_sim.bc_ac_receptor_components import BC_TYPES, AC_TYPES, BC_KEY, AC_KEY, POOLING

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT/'configurations/bc_ac_receptor_candidate_v1.json'


def synthetic_profile(path):
    data = json.loads(TEMPLATE.read_bytes())
    data.update(profile_id='synthetic_bc_ac_test_v1', test_only=True)
    # Explicit numerical QA choices, exclusively software_fixture. They are
    # never supplied as defaults or inferred from animal subtype names.
    values = dict(C=20e-12, gL=1e-9, EL=-.06, V0=-.05, K=.25, hill=2.,
        tau_e=.004, gE=.5e-9, EE=0., e0=.3, release_half=-.05,
        release_slope=.005, tau_release=.003, EI=-.076,
        activation_half=-.05, activation_slope=.005)
    for cid, family in data['families'].items():
        family['conditions']={'recording_context':'SYNTHETIC NUMERICAL QA; no physiological data'}
        for key, row in family['parameters'].items():
            row.update(value=values[key], evidence_class='software_fixture', source_id=None,
                locator='Explicit software-test coefficient', uncertainty='Not an animal estimate')
        # Different upstream release levels make pool-before-nonlinearity testable.
        if cid == 'diffuse_BC_ON':
            family['parameters']['V0'].update(binding='per_cell', value=[-.06, -.04])
    path.write_text(json.dumps(data), encoding='utf8')
    return CandidateProfile(path)


@pytest.fixture(scope='module')
def parent():
    registry = load_core_registry(ROOT); n=2; xy=np.array([[.01,0.],[.02,0.]])
    pops={}; specs=[]; links=[]; externals=[]; channels={}
    def external(cid, port, unit, meaning, value):
        name=cid+'_'+port;externals.append(ExternalInput(name,cid,port))
        channels[name]=InputChannel(0.,.002,lambda t,u=unit,m=meaning,v=value:q(np.full(n,v),u,m),
            unit,meaning,'zero_order_hold','synthetic_fixture',{'source':'explicit synthetic constant clamp'})
    for cid, polarity in BC_TYPES.items():
        pid='effective_'+cid
        entries={'kernel':([.5,.25,-.1],'dimensionless','causal_residual_kernel'),
            'sample_interval':(.001,'s','history_sampling_interval'),
            'bias':(np.zeros(n),'dimensionless','activation_bias'),
            'baseline':(np.full(n,.25),'dimensionless','input_baseline'),
            'initial_history':(np.zeros((3,n)),'dimensionless','causal_input_history'),
            'C':(np.full(n,20e-12),'F','membrane_capacitance'),
            'gL':(np.full(n,1e-9),'S','leak_conductance'),
            'Vref':(np.full(n,-.05),'V','terminal_voltage_reference'),
            'native_voltage_gain':(np.full(n,.01),'V','voltage_per_native_activation'),
            'native_reference':(np.full(n,np.log(2.)),'dimensionless','native_activation_reference'),
            'V0':(np.full(n,-.05),'V','initial_membrane_voltage'),
            'release_half':(np.full(n,-.05),'V','activation_half_voltage'),
            'release_slope':(np.full(n,.005),'V','activation_voltage_slope'),
            'tau_release':(np.full(n,.003),'s','release_time_constant')}
        if polarity=='OFF':entries['EI']=(np.full(n,-.076),'V','inhibitory_reversal')
        fixture_profile(registry,pid,n,entries)
        pops[cid]=population(cid,'BC',(cid,)*n,xy,('center',)*n,pid,'Macaque_software_fixture')
        specs.append(ComponentSpec(cid,'HistoryBipolarTerminal',{'cells':cid},pid,{'polarity':polarity},cid))
        external(cid,'drive','dimensionless','effective_synaptic_drive',.25)
        if polarity=='OFF':external(cid,'inhibition','S','inhibitory_conductance',.1e-9)
    W=sparse.csr_matrix([[.8,.2],[.1,.9]])
    for cid, driver in zip(AC_TYPES,('diffuse_BC_ON','diffuse_BC_OFF','diffuse_BC_ON')):
        pid='effective_'+cid
        fixture_profile(registry,pid,n,{'kernel':([.5,.25,-.1],'dimensionless','causal_residual_kernel'),
            'sample_interval':(.00073 if cid=='widefield_AC' else .001,'s','history_sampling_interval'),
            'bias':(np.full(n,-.3),'dimensionless','activation_bias'),
            'baseline':(np.full(n,.5),'dimensionless','input_baseline'),
            'initial_history':(np.zeros((3,n)),'dimensionless','causal_input_history')})
        pops[cid]=population(cid,'AC',(cid,)*n,xy,('center',)*n,pid,'Macaque_software_fixture')
        specs.append(ComponentSpec(cid,'ResidualAC',{'cells':cid},pid,{'polarity_representation':'stored_signed'},cid))
        links.append(link(driver,'activation',cid,'drive',W,pops,'effective_'+driver,'IPL'))
    cid='AII_AC';pid='old_AII'
    fixture_profile(registry,pid,n,membrane_entries(n,'AII'))
    pops[cid]=population(cid,'AC',('AII',)*n,xy,('center',)*n,pid,'Macaque_software_fixture')
    specs.append(ComponentSpec(cid,'ReceptorConductanceMembrane',{'cells':cid},pid,{'cell_kind':'AII'},cid))
    external(cid,'excitation','S','excitatory_conductance',.1e-9)
    gap=GapJunction(cid,pops[cid].cells.cell_id,'midget_BC_ON',pops['midget_BC_ON'].cells.cell_id,
        EvidenceValue('synthetic_gap',q(np.full(n,.1e-9),'S','gap_junction_conductance'),'software_fixture',None,
            'Synthetic reciprocal gap QA',{'fixture':True},'No biological interpretation'))
    spec=ModelSpec('synthetic_bc_ac_parent',tuple(pops.values()),tuple(specs),tuple(links),tuple(externals),
        tuple(pops),clock_policy={'kind':'explicit_event_clock','base_step_seconds':.0001,'sampled_output_hold':'zero_order_hold'},
        gap_junctions=(gap,))
    compiled=RetinaBuilder(registry).build(spec,RunMode.NUMERIC_FIXTURE)
    # Exercise candidate scope admission without relabelling any real population.
    registry.runtime_profile=RuntimeProfile(ROOT,'effective_dynamics_v1')
    return compiled,registry,channels


@pytest.fixture
def migrated(parent,tmp_path):
    profile=synthetic_profile(tmp_path/'synthetic.json')
    compiled,registry,report=migrate_bc_ac(parent[0],parent[1],profile)
    return compiled,registry,report,profile


def test_distributed_profile_is_unbound_and_never_falls_back(parent):
    profile=CandidateProfile(TEMPLATE)
    assert sum(len(f['parameters']) for f in profile.data['families'].values())==103
    assert all(r['value'] is None for f in profile.data['families'].values() for r in f['parameters'].values())
    with pytest.raises(CandidateParametersMissing,match='BC_AC_CANDIDATE_UNBOUND') as error:
        migrate_bc_ac(parent[0],parent[1],profile)
    assert set(error.value.missing)==set(BC_TYPES)|set(AC_TYPES)
    assert {'C','K','tau_e','gE'} <= set(error.value.missing['midget_BC_ON'])


def test_new_states_replace_fir_without_changing_connections_or_identity(parent,migrated):
    new,registry,report,_=migrated;old=parent[0]
    assert report['pooling_order']==POOLING and report['connectivity_and_cell_ids_preserved']
    for cid in BC_TYPES:
        c=new.components[cid]
        assert c.mechanism==BC_KEY
        assert [s.state_name for s in c.state_specs]==['V','release','e']
        assert 'generator_activation' not in c.outputs
    for cid in AC_TYPES:
        c=new.components[cid]
        assert c.mechanism==AC_KEY and [s.state_name for s in c.state_specs]==['V','e']
        assert not hasattr(c.native_model,'sample_spikes')
    for cid,pop in old.populations.items():
        assert new.populations[cid].cells.cell_id==pop.cells.cell_id
        assert np.array_equal(new.populations[cid].cells.rf_position.values,pop.cells.rf_position.values)
    assert new.components['AII_AC'].parameters.parameter_hash==old.components['AII_AC'].parameters.parameter_hash
    assert new.components['AII_AC'].implementation_hash==old.components['AII_AC'].implementation_hash
    assert canonical_hash(new.spec.links)==canonical_hash(old.spec.links)
    assert canonical_hash(new.spec.gap_junctions)==canonical_hash(old.spec.gap_junctions)
    assert new.modeling_scope['equation_revision']=='bc_ac_receptor_candidate_v1'
    assert not new.modeling_scope['physiology_validated']


def test_bc_on_off_gate_polarity_membrane_and_retained_release(migrated):
    c,_,_,_=migrated
    for cid in ('midget_BC_ON','midget_BC_OFF'):
        component=c.components[cid];initial=tuple(v.values for v in component.initial())
        u=np.array([0.,1.]);incoming={'drive':q(u,'dimensionless','effective_synaptic_drive')}
        if cid.endswith('OFF'):incoming['inhibition']=q(np.full(2,.1e-9),'S','inhibitory_conductance')
        rows=component.derivative(0.,initial,incoming)
        occupancy=u*u/(.25**2+u*u)
        expected=1-occupancy if cid.endswith('ON') else occupancy
        np.testing.assert_allclose(rows[2].values,(expected-.3)/.004,rtol=0,atol=1e-12)
        V,r,e=initial
        current=1e-9*(-.06-V)+.5e-9*e*(0.-V)
        if cid.endswith('OFF'):current+=.1e-9*(-.076-V)
        np.testing.assert_allclose(rows[0].values,current/20e-12,rtol=0,atol=1e-12)
        np.testing.assert_allclose(rows[1].values,(expit((V+.05)/.005)-r)/.003,rtol=0,atol=1e-12)
        wrong=dict(incoming,drive=q(u,'uM','glutamate_concentration'))
        with pytest.raises(ContractError,match='dimensionless'):
            component.derivative(0.,initial,wrong)


def test_pooling_order_and_ac_has_exactly_two_states(migrated,parent):
    c,_,report,_=migrated
    network=JointNetwork(c,compute_backend='python')
    y=np.concatenate([v.values for v in network.initial_conditions()])
    incoming,outputs=network.evaluate(0.,y,InputBundle({k:v.at(0.) for k,v in parent[2].items()}))
    cid='local_feedforward_AC';component=c.components[cid];blocks=tuple(v.values for v in component.initial())
    drive=incoming[cid]['drive'].values
    release=outputs[('diffuse_BC_ON','activation')].values
    W=np.array([[.8,.2],[.1,.9]])
    np.testing.assert_allclose(drive,W@release,rtol=0,atol=1e-15)
    receptor=lambda u:u*u/(.25**2+u*u)
    assert not np.allclose(receptor(W@release),W@receptor(release))
    rows=component.derivative(0.,blocks,incoming[cid])
    np.testing.assert_allclose(rows[1].values,(receptor(drive)-.3)/.004,rtol=0,atol=1e-12)
    np.testing.assert_allclose(component.observe('activation',0.,blocks,incoming[cid]).values,
                               expit((blocks[0]+.05)/.005),rtol=0,atol=0)


def test_module_and_circuit_share_candidate_equations_and_short_run(migrated,parent):
    c,registry,_,_=migrated
    conditions=dict(source_id='synthetic_candidate_test',species='software_fixture',retinal_region='fixture',
        baseline_source='explicit synthetic clamp',time_origin='test_start_seconds',adaptation_status='factory_not_adapted')
    for cid,kind in [('midget_BC_ON','bc_clamped'),('local_feedforward_AC','generic_ac_clamped')]:
        component=c.components[cid]
        baseline={name:q(np.full(len(port.cell_ids),.25),'dimensionless',port.meaning)
                  for name,port in component.inputs.items()}
        gaps=[gap_boundary_id(g) for g in c.spec.gap_junctions if cid in (g.left_component,g.right_component)]
        module=build_module(c,registry,cid,kind=kind,mode='declared_input_clamps',conditions=conditions,
                            baselines=baseline,open_gap_boundaries=gaps)
        isolated=module.compiled.components[cid]
        assert isolated.implementation_hash==component.implementation_hash
        blocks=tuple(v.values for v in isolated.initial())
        for a,b in zip(component.derivative(0.,blocks,baseline),isolated.derivative(0.,blocks,baseline)):
            np.testing.assert_array_equal(a.values,b.values)
        package=parameter_package(module,version='synthetic-v1',source_id='fixture',applicability=conditions)
        assert {'V0','e0'} <= set(package['initialization_not_transferable'])
        assert merge_parameter_package(c,registry,package,applicability=conditions).assembly_hash==c.assembly_hash
    solver=SolverConfig(method='DOP853',rtol=1e-8,atol=1e-11,max_step=.0001,input_interpolation='zero_order_hold')
    state=c.initialize(solver)
    final,result=c.advance(state,parent[2],.0002,solver,sample_times=[0.,.0001,.0002])
    assert final.time_seconds==.0002 and all(np.all(np.isfinite(v)) for v in final.ordered_state_arrays)
    assert set(final.spike_states)==set()


def test_invalid_units_and_synthetic_production_use_are_rejected(tmp_path,parent):
    profile=synthetic_profile(tmp_path/'synthetic.json')
    wrong=copy.deepcopy(profile.data);wrong['families']['midget_BC_ON']['parameters']['C']['unit']='pF'
    path=tmp_path/'wrong.json';path.write_text(json.dumps(wrong),encoding='utf8')
    with pytest.raises(ContractError,match='unit/meaning mismatch'):CandidateProfile(path)
    with pytest.raises(ContractError,match='SYNTHETIC_CANDIDATE_REQUIRES_SOFTWARE_FIXTURE'):
        migrate_bc_ac(replace(parent[0],run_mode=RunMode.REFERENCE),parent[1],profile)
    changed=copy.deepcopy(profile.data)
    changed['families']['midget_BC_ON']['parameters']['C'].update(binding='per_cell',value=[20e-12,30e-12])
    path.write_text(json.dumps(changed),encoding='utf8')
    with pytest.raises(ContractError,match='per-cell initialization only'):CandidateProfile(path)
    profile.data['families']['midget_BC_ON']['parameters']['C']['value']=30e-12
    with pytest.raises(ContractError,match='source identity was read'):
        migrate_bc_ac(parent[0],parent[1],profile)


def test_candidate_is_reference_only_original_lowering_sources_unchanged(migrated):
    from retina_sim.compiled_kernel import IMPLEMENTATIONS
    c=migrated[0]
    auto=JointNetwork(c,compute_backend='auto')
    assert auto.compute_metadata['backend']=='python_reference'
    assert 'Candidate' in auto.compute_metadata['fallback_reason']
    with pytest.raises(ContractError,match='Unsupported mechanisms'):JointNetwork(c,compute_backend='numba')
    for name,expected in IMPLEMENTATIONS.items():
        assert hashlib.sha256((ROOT/'retina_sim'/name).read_bytes()).hexdigest()==expected


def test_cli_reports_missing_parameters_before_parent_assembly(monkeypatch,capsys):
    from tools import run_calibration_module as cli
    def unexpected(*args,**kwargs):raise AssertionError('Must not assemble/run an unbound candidate')
    monkeypatch.setattr(cli,'atlas_population_network',unexpected)
    args=['run_calibration_module.py','--component','midget_BC_ON','--kind','bc_clamped',
          '--mode','declared_input_clamps','--equation-route','bc_ac_receptor_candidate_v1']
    monkeypatch.setattr(sys,'argv',args);cli.main()
    assert json.loads(capsys.readouterr().out)['status']=='BC_AC_CANDIDATE_UNBOUND_NO_RUN'
    monkeypatch.setattr(sys,'argv',args+['--run'])
    with pytest.raises(CandidateParametersMissing):cli.main()


@pytest.mark.parametrize('cid,kind',[('midget_BC_ON','bc_clamped'),
                                   ('local_feedforward_AC','generic_ac_clamped')])
def test_candidate_cli_executes_same_equations_with_explicit_synthetic_data(parent,tmp_path,monkeypatch,cid,kind):
    from tools import run_calibration_module as cli
    profile=synthetic_profile(tmp_path/'synthetic.json')
    monkeypatch.setattr(cli,'atlas_population_network',lambda *args,**kw:(parent[0],parent[2],{'registry':parent[1]}))
    conditions=dict(source_id='synthetic_equation_test',species='software_fixture',retinal_region='fixture',
        baseline_source='synthetic_test_clamp',time_origin='test_start_seconds',adaptation_status='factory_not_adapted')
    ids=list(parent[0].components[cid].inputs['drive'].cell_ids)
    meaning=parent[0].components[cid].inputs['drive'].meaning
    contract=dict(conditions=conditions,parameter_snapshot_version='synthetic-candidate-v1',
        open_gap_boundaries=[gap_boundary_id(g) for g in parent[0].spec.gap_junctions
                             if cid in (g.left_component,g.right_component)],
        solver=dict(method='DOP853',rtol=1e-8,atol=1e-11,max_step=.0001,input_interpolation='zero_order_hold'),
        inputs={'drive':dict(edges_seconds=[0.,.0001,.0002],values=[[.25,.25],[.25,.25]],
            baseline=[.25,.25],unit='dimensionless',meaning=meaning,cell_ids=ids,
            recording_kind='effective_drive',source_id='synthetic_test_only')})
    path=tmp_path/'contract.json';path.write_text(json.dumps(contract),encoding='utf8')
    out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['run_calibration_module.py','--component',cid,'--kind',kind,
        '--mode','declared_input_clamps','--equation-route','bc_ac_receptor_candidate_v1',
        '--candidate-profile',str(profile.path),'--contract',str(path),'--output',str(out),'--run'])
    cli.main()
    result=json.loads((out/'result.json').read_bytes())
    assert result['time_seconds']==.0002 and result['status']=='ISOLATED_MODULE_RUN_COMPLETE_NOT_CALIBRATED'
    assert result['module']['equation_revision']=='bc_ac_receptor_candidate_v1'
    assert result['module']['compute_policy']=='python_reference_only'
    assert result['module']['candidate_dynamics']['test_only']
    assert json.loads((out/'parameters.json').read_bytes())['status']=='unfitted_snapshot'


def test_voltage_and_drive_scale_symmetries_do_not_identify_absolute_parameters(parent,tmp_path):
    original=synthetic_profile(tmp_path/'original.json')
    a,_,_=migrate_bc_ac(parent[0],parent[1],original)
    changed=copy.deepcopy(original.data);scale=3.
    for cid in BC_TYPES:
        for key in ('C','gL','gE'):changed['families'][cid]['parameters'][key]['value']*=scale
    path=tmp_path/'scaled_g.json';path.write_text(json.dumps(changed),encoding='utf8')
    b,_,_=migrate_bc_ac(parent[0],parent[1],CandidateProfile(path))
    cid='midget_BC_OFF';blocks=tuple(v.values for v in a.components[cid].initial())
    incoming={'drive':q(np.array([.1,.6]),'dimensionless','effective_synaptic_drive'),
              'inhibition':q(np.full(2,.1e-9),'S','inhibitory_conductance')}
    scaled=dict(incoming,inhibition=q(np.full(2,.1e-9*scale),'S','inhibitory_conductance'))
    for x,y in zip(a.components[cid].derivative(0.,blocks,incoming),b.components[cid].derivative(0.,blocks,scaled)):
        np.testing.assert_allclose(x.values,y.values,rtol=0,atol=1e-10)
    changed=copy.deepcopy(original.data);changed['families'][cid]['parameters']['K']['value']*=4.
    path=tmp_path/'scaled_u.json';path.write_text(json.dumps(changed),encoding='utf8')
    c,_,_=migrate_bc_ac(parent[0],parent[1],CandidateProfile(path))
    scaled=dict(incoming,drive=q(incoming['drive'].values*4.,'dimensionless','effective_synaptic_drive'))
    for x,y in zip(a.components[cid].derivative(0.,blocks,incoming),c.components[cid].derivative(0.,blocks,scaled)):
        np.testing.assert_allclose(x.values,y.values,rtol=0,atol=1e-10)
