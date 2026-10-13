"""New delay/rate/plugin abilities only; every input is explicitly synthetic."""
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from retina_sim.contracts import ContractError,RunMode
from retina_sim.assembly import RetinaBuilder
from retina_sim.delays import EdgeDelayBuffer
from retina_sim.plugins import boundary_status
from retina_sim.mechanism_contracts import register_mechanism_profile
from examples.full_network_fixture import full_fixture,q
from examples.plugin_boundary_fixture import plugin_fixture
from conftest import NUMERIC_METRICS
ROOT=Path(__file__).resolve().parents[1]
def flatten(s):return np.concatenate(s.ordered_state_arrays)

def test_heterogeneous_explicit_delays_use_past_values_not_instantaneous_drive():
    c,ch,b,s=plugin_fixture(ROOT);end,r=c.advance(c.initialize(),ch,.001)
    expected=.1+.1*np.exp(-.001/.002)
    np.testing.assert_allclose(r.outputs_with_units['RBC.bipolar_signal'].values[:,-1],expected,atol=1e-10)
    assert len(end.link_delay_states)==1
    later,lr=c.advance(end,ch,.003)
    values=lr.outputs_with_units['RBC.bipolar_signal'].values[:,-1]
    assert values[0]>values[1] and values[0]>expected
    NUMERIC_METRICS['delay_smoke_heterogeneous_outputs']=values.tolist()

def test_delayed_history_non_grid_checkpoint_and_no_input_state_mutation():
    c,ch,b,s=plugin_fixture(ROOT);initial=c.initialize();whole,_=c.advance(initial,ch,.0047)
    first,_=c.advance(initial,ch,.00235);split,_=c.advance(c.restore(c.checkpoint(first)),ch,.0047)
    error=float(np.max(np.abs(flatten(whole)-flatten(split))))
    assert error<3e-5 and len(initial.link_delay_states['link:0']['times'])==1
    NUMERIC_METRICS['delayed_checkpoint_max_state_difference']=error
    assert whole.link_delay_states['link:0']['max_samples']>=len(whole.link_delay_states['link:0']['times'])

def test_delay_step_refinement_is_numerical_not_calibration():
    a,ch,b,s=plugin_fixture(ROOT,sampling=.0001);a2,ch2,_,_=plugin_fixture(ROOT,sampling=.00005);a4,ch4,_,_=plugin_fixture(ROOT,sampling=.000025)
    x,_=a.advance(a.initialize(),ch,.006);y,_=a2.advance(a2.initialize(),ch2,.006);z,_=a4.advance(a4.initialize(),ch4,.006)
    coarse=np.linalg.norm(flatten(x)-flatten(y));fine=np.linalg.norm(flatten(y)-flatten(z))
    assert fine<coarse*.6
    NUMERIC_METRICS['linear_delay_interpolation_refinement_ratio']=float(fine/coarse)

def test_delay_discontinuity_left_and_right_values_and_checkpoint_payload():
    base=q([0,0],'dimensionless','effective_synaptic_drive')
    buf=EdgeDelayBuffer(.002,[0],[[0,0]],base,0,'linear',[])
    buf.append(.001,q([0,0],'dimensionless','effective_synaptic_drive'));buf.append(.001,q([1,2],'dimensionless','effective_synaptic_drive'))
    buf.append(.002,q([1,2],'dimensionless','effective_synaptic_drive'))
    np.testing.assert_array_equal(buf.query_edges(.002,[.001,.001],[0,1],left_limit=True),[0,0])
    np.testing.assert_array_equal(buf.query_edges(.002,[.001,.001],[0,1]),[1,2])
    assert buf.next_arrival(.001,[.001,.0015])==pytest.approx(.002)
    clone=EdgeDelayBuffer.from_payload(buf.payload());np.testing.assert_array_equal(clone.query_edges(.002,[.001,.001],[0,1]),[1,2])
    with pytest.raises(ContractError,match='unavailable'):buf.query_edges(.01,[.001,.001],[0,1])

def test_delay_prehistory_and_interpolation_are_required():
    c,ch,b,s=plugin_fixture(ROOT)
    with pytest.raises(ContractError,match='prehistory'):b.build(replace(s,links=(replace(s.links[0],delay_prehistory=None),s.links[1])),RunMode.NUMERIC_FIXTURE)
    with pytest.raises(ContractError):b.build(replace(s,links=(replace(s.links[0],delay_interpolation=None),s.links[1])),RunMode.NUMERIC_FIXTURE)
    with pytest.raises(ContractError,match='evidence'):b.build(replace(s,links=(replace(s.links[0],delay_evidence=None),s.links[1])),RunMode.NUMERIC_FIXTURE)

def test_rod_RBC_AII_boundaries_run_only_supplied_engineering_provider():
    c,ch,b,s=plugin_fixture(ROOT);state,result=c.advance(c.initialize(),ch,.003)
    assert np.all(result.outputs_with_units['AII.crossover_signal'].values>0)
    assert not np.allclose(result.outputs_with_units['AII.crossover_signal'].values[:,0],result.outputs_with_units['AII.crossover_signal'].values[:,-1])
    status=boundary_status(b.registry)
    assert len(status)==3 and all(v['bound_backends']==['QA_transport'] for v in status.values())
    NUMERIC_METRICS['rod_plugin_boundaries']=status
    with pytest.raises(ContractError,match='synthetic-only'):b.build(s,RunMode.REFERENCE)
    components=(replace(s.components[0],options={'backend_id':None}),)+s.components[1:]
    with pytest.raises(ContractError,match='unbound'):b.build(replace(s,components=components),RunMode.NUMERIC_FIXTURE)

def slow_fixture():
    c,ch,a=full_fixture(ROOT);old=a['registry'].mechanism_profiles['QA_DB_ON']
    records=tuple(replace(r,value=q(.002,'s','history_sampling_interval')) if r.parameter_id=='sample_interval' else r for r in old.records)
    profile=replace(old,profile_id='QA_DB_ON_2ms',records=records);register_mechanism_profile(a['registry'],profile)
    pops=tuple(replace(p,cells=replace(p.cells,parameter_profile_id=(profile.profile_id,)*len(p.cells.cell_id))) if p.population_id=='DB_ON' else p for p in a['spec'].populations)
    components=tuple(replace(x,parameter_profile_id=profile.profile_id) if x.component_id=='DB_ON' else x for x in a['spec'].components)
    spec=replace(a['spec'],populations=pops,components=components,clock_policy={'kind':'commensurate_minimum_step','base_step_seconds':.001,'sampled_output_hold':'zero_order_hold'})
    return RetinaBuilder(a['registry']).build(spec,RunMode.NUMERIC_FIXTURE),ch,a,spec

@pytest.fixture(scope='module')
def multi():return slow_fixture()

def test_commensurate_multi_rate_keeps_inactive_history_and_checkpoint_phase(multi):
    c,ch,a,spec=multi;initial=c.initialize();sampled,_=c.advance(initial,ch,.001)
    assert c.clock['component_tick_strides']['DB_ON']==2
    model=c.runner().model;i=[s.state_name for s in model.state_specs].index('DB_ON.history_1')
    np.testing.assert_array_equal(sampled.ordered_state_arrays[i],initial.ordered_state_arrays[i])
    whole,_=c.advance(initial,ch,.0043);first,_=c.advance(initial,ch,.00235);split,_=c.advance(c.restore(c.checkpoint(first)),ch,.0043)
    np.testing.assert_allclose(flatten(whole),flatten(split),atol=2e-9,rtol=2e-8)
    for cid in whole.spike_states:assert whole.spike_states[cid]['rng_state']==split.spike_states[cid]['rng_state']
    NUMERIC_METRICS['multi_rate_clock']=c.clock

def test_multi_rate_requires_declared_hold_and_rejects_noncommensurate_periods(multi):
    c,ch,a,spec=multi;b=RetinaBuilder(a['registry'])
    with pytest.raises(ContractError,match='shared clock'):b.build(replace(spec,clock_policy=None),RunMode.NUMERIC_FIXTURE)
    with pytest.raises(ContractError,match='noncommensurate'):b.build(replace(spec,clock_policy={'kind':'commensurate_minimum_step','base_step_seconds':.0007,'sampled_output_hold':'zero_order_hold'}),RunMode.NUMERIC_FIXTURE)

def test_replaceable_mass_conserving_mosaic_backend_does_not_claim_exact_geometry():
    from retina_sim.geometry import MosaicBackend,RetinalDomain
    field=lambda xy:np.full(len(xy),1000.)
    domain=RetinalDomain((0,.1,0,.1),(0,.1,0,.1),lambda xy:np.ones(len(xy),bool),'synthetic mask','QA_backend_domain')
    result=MosaicBackend().generate('greedy_capacitated_equal_rho_mass',field,domain,9,quadrature_shape=(16,16),iterations=1)
    assert len(result.cell_ids)==10 and result.diagnostics['exact_power_diagram'] is False
    assert abs(result.diagnostics['count_minus_density_mass'])<=.5
    assert result.diagnostics['maximum_capacity_relative_residual']<1e-8
    NUMERIC_METRICS['mosaic_backend_audit']=result.diagnostics

def test_required_phenotype_layer_binding_is_executable_and_family_checked(multi):
    c,ch,a,spec=multi;b=RetinaBuilder(a['registry'])
    components=tuple(replace(x,required_layer='diffuse_BC_ON') if x.component_id=='DB_ON' else x for x in spec.components)
    target=replace(spec,components=components,required_mechanisms=('diffuse_BC_ON',))
    built=b.build(target,RunMode.NUMERIC_FIXTURE)
    assert 'diffuse_BC_ON' in built.components['DB_ON'].layers
    invalid=tuple(replace(x,required_layer='rod') if x.component_id=='DB_ON' else x for x in spec.components)
    with pytest.raises(ContractError,match='not supported'):b.build(replace(spec,components=invalid),RunMode.NUMERIC_FIXTURE)
