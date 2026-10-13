"""Short numerical fixtures only: no production 1.25/10-second adaptation run."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from examples.full_network_fixture import full_fixture
from retina_sim.assembly import RetinaBuilder
from retina_sim.contracts import ContractError,RunMode,canonical_hash
from retina_sim.background import BackgroundPolicy,constant_background_channels,prepare_background,timing_requirements,_WindowMonitor,rebase_stimulus_channels
from retina_sim.runtime_profiles import LightContext
from retina_sim.network import NetworkStimulus
from retina_sim.recording import RecordingPlan

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def network():return full_fixture(ROOT)

def outer(network):
    c,_,a=network;spec=a['spec']
    compiled=RetinaBuilder(a['registry']).build(replace(spec,populations=spec.populations[:2],components=spec.components[:1],links=()),RunMode.NUMERIC_FIXTURE)
    return compiled.runner()

def short_policy(**kw):
    return BackgroundPolicy(**dict(window_seconds=.001,stable_windows=3,minimum_seconds=0.,known_tau_multiplier=0.,
        delay_multiplier=0.,max_sim_seconds=.004,monitor_step_seconds=.0001,**kw))

def background(r,stop=.03):
    return constant_background_channels(r.compiled,LightContext(),stop,interpolation='analytic')

def test_standard_lower_bound_is_source_tau_not_rerun(network):
    r=outer(network);requirements=timing_requirements(r,BackgroundPolicy())
    assert requirements['maximum_known_tau_seconds']==.25
    assert requirements['minimum_required_seconds']==1.25
    assert BackgroundPolicy().max_sim_seconds==10.

def test_consecutive_windows_and_absolute_time(network):
    r=outer(network);s=r.initialize(time_seconds=.01)
    s,report=prepare_background(r,s,background(r),short_policy())
    assert report['status']=='CONVERGED' and len(report['windows'])==3
    assert s.time_seconds==pytest.approx(.013) and s.grid_origin_seconds==.01
    assert s.input_position['stimulus_origin_seconds']==s.time_seconds
    assert report['whole_network_steady_state_claim'] is False

def test_partial_window_and_standard_minimum_do_not_pass(network):
    r=outer(network);s,report=prepare_background(r,r.initialize(),background(r),
        BackgroundPolicy(max_sim_seconds=.0002))
    assert report['status']=='NOT_REACHED' and not report['converged']
    assert not report['windows'][0]['complete_window']
    assert report['timing_requirements']['minimum_required_seconds']==1.25
    assert s.time_seconds==pytest.approx(.0002)

def test_window_excursion_catches_equal_endpoints(network):
    r=outer(network);s=r.initialize();p=short_policy();monitor=_WindowMonitor(r,s,p)
    y=r._flat(s);bundle=NetworkStimulus(r.compiled,background(r),s,.004,'analytic',r.model.evidence_tag).at(0)
    monitor.observe(0,y,bundle);middle=y.copy();middle[r.model._state_slices[4]]+=1
    monitor.observe(.0005,middle,bundle);monitor.observe(.001,y,bundle)
    report=monitor.report(.001)
    assert not report['passed'] and report['maximum_norm']>1
    assert next(x for x in report['states'] if x['state']=='outer.Vis')['maximum_observed_excursion_norm']>1

def test_original_rhs_cannot_pass_by_tiny_dt(network):
    r=outer(network);s=r.initialize();y=r._flat(s);y[r.model._state_slices[4]]+=1
    m=_WindowMonitor(r,s,short_policy());bundle=NetworkStimulus(r.compiled,background(r),s,.004,'analytic',r.model.evidence_tag).at(0)
    m.observe(0,y,bundle);m.observe(1e-12,y,bundle)
    report=m.report(.001)
    assert any(row['original_rhs_norm']>1 for row in report['states'])
    assert not report['passed']

def test_discrete_defect_catches_zero_rhs_history(network):
    c,_,_=network;r=c.runner();s=r.initialize();y=r._flat(s)
    index=next(i for i,sp in enumerate(r.model.state_specs) if sp.state_name=='midget_ON.history_0')
    y[r.model._state_slices[index]]=100.
    m=_WindowMonitor(r,s,short_policy());bundle=NetworkStimulus(c,background(r),s,.004,'analytic',r.model.evidence_tag).at(0)
    m.observe(0,y,bundle);report=m.report(.001)
    row=next(row for row in report['states'] if row['state']=='midget_ON.history_0')
    assert row['original_rhs_norm']==0 and row['discrete_defect_norm']>1

def test_unknown_units_are_inconclusive(network):
    r=outer(network);_,report=prepare_background(r,r.initialize(),background(r),short_policy(),unit_scales={})
    assert report['status']=='INCONCLUSIVE' and report['uncovered_states']
    assert not report['converged']

def test_independent_wall_and_observation_budgets(network):
    r=outer(network);s=r.initialize()
    final,report=prepare_background(r,s,background(r),short_policy(max_wall_seconds=1e-12))
    assert final.time_seconds==s.time_seconds and report['budget_exit']=='max_wall_seconds'
    assert not report['converged']
    final,report=prepare_background(r,s,background(r),short_policy(max_observations=1))
    assert report['budget_exit']=='max_observations' and report['observations']==1
    assert not report['converged'] and report['windows'][0]['monitoring_budget_limited']

def test_actual_background_values_and_units_are_independent(network):
    r=outer(network);light=LightContext(native_background_td=73.,cone_background_rstar_s=901.,rod_background_rstar_s=8.)
    channels=constant_background_channels(r.compiled,light,.01,interpolation='analytic')
    assert channels['native_light'].at(.005).unit=='td'
    np.testing.assert_array_equal(channels['native_light'].at(.005).values,np.full(9,73.))
    assert channels['native_light'].input_contract['background_context_hash']==light.background_context_hash

def test_author_preparation_budget_preserves_native_grid_phase():
    from types import SimpleNamespace
    runner=SimpleNamespace(compiled=SimpleNamespace(components={'author':SimpleNamespace(
        time_contract=SimpleNamespace(kind='author_discrete',fixed_step_seconds=.0001))}))
    with pytest.raises(ContractError,match='author input/grid phase'):
        prepare_background(runner,None,{},BackgroundPolicy(max_sim_seconds=.00025))

def test_actual_preparation_metadata_overrides_fixed_request():
    from retina_platform.backend import _record_preparation_metadata
    original=LightContext().manifest(.002)
    manifest={'initialization':original,'input_assumptions':{'light_context':original}}
    resolved={'light_context':original};report={'elapsed_seconds':.0002,'status':'NOT_REACHED'}
    _record_preparation_metadata(manifest,resolved,report)
    assert original['prerun_seconds']==.002
    for value in [manifest['initialization'],manifest['input_assumptions']['light_context'],resolved['light_context']]:
        assert value['prerun_seconds']==.0002 and value['requested_fixed_prerun_seconds']==.002
        assert value['adaptation_status']=='NOT_REACHED' and value['whole_network_steady_state_claim'] is False

def test_checkpoint_preserves_origin_delay_rng_and_continuation(network):
    c,_,_=network;r=c.runner();s=r.initialize(time_seconds=.01)
    s.input_position={'background_context_hash':LightContext().background_context_hash}
    channels=background(r,.03)
    initial_rng=canonical_hash(s.spike_states)
    s,report=prepare_background(r,s,channels,short_policy())
    assert canonical_hash(s.spike_states)!=initial_rng
    assert report['stochastic_status']=='STATISTICS_INSUFFICIENT'
    assert s.grid_origin_seconds==.01 and s.time_seconds>.01
    restored=r.restore(r.checkpoint(s),expected_input_context={'background_context_hash':LightContext().background_context_hash})
    end=s.time_seconds+.001
    a,ra=r.advance(s,channels,end,recording_plan=RecordingPlan())
    b,rb=r.advance(restored,channels,end,recording_plan=RecordingPlan())
    for x,y in zip(a.ordered_state_arrays,b.ordered_state_arrays):np.testing.assert_array_equal(x,y)
    assert a.spike_states==b.spike_states and a.delay_states==b.delay_states and a.link_delay_states==b.link_delay_states
    assert a.component_tick_indices==b.component_tick_indices and a.grid_origin_seconds==b.grid_origin_seconds
    assert a.input_position==b.input_position and a.input_position['stimulus_origin_seconds']==s.time_seconds
    assert ra.metadata['spike_events']==rb.metadata['spike_events']
    with pytest.raises(ContractError,match='context mismatch'):r.restore(r.checkpoint(s),expected_input_context={'background_context_hash':'wrong'})


def test_dense_background_samples_match_mixed_clock_from_zero(network):
    c,_,_=network;r=c.runner();s=r.initialize()
    final,report=prepare_background(r,s,background(r),short_policy())
    assert final.time_seconds==pytest.approx(.004)
    assert len(report['windows'])==4 and report['observations']>40
    assert report['grid_origin_seconds']==0.
    assert all(w['complete_window'] for w in report['windows'])
    assert all(np.isfinite(a).all() for a in final.ordered_state_arrays)


def test_recording_near_history_tick_keeps_requested_stamp_and_rng(network):
    c,_,_=network;r=c.runner();s=r.initialize()
    channels=background(r)
    exact=np.array([0.,.001,.002]);near=exact.copy();near[1]=np.nextafter(near[1],-np.inf)
    plan=RecordingPlan(ports=(),states=(),cell_ids=())
    a,ra=r.advance(s,channels,.002,sample_times=exact,recording_plan=plan)
    b,rb=r.advance(s,channels,.002,sample_times=near,recording_plan=plan)
    np.testing.assert_array_equal(ra.outputs_with_units['time'].values,exact)
    np.testing.assert_array_equal(rb.outputs_with_units['time'].values,near)
    assert a.spike_states==b.spike_states and a.component_tick_indices==b.component_tick_indices

def test_rebase_external_phase_keeps_preceding_background(network):
    r=outer(network);ch=background(r,.03)
    rebased=rebase_stimulus_channels(ch,ch,original_origin_seconds=.002,stimulus_origin_seconds=.01,stop_seconds=.02)
    np.testing.assert_array_equal(rebased['native_light'].at(.0099).values,ch['native_light'].at(.0099).values)
    assert rebased['native_light'].input_contract['absolute_time_preserved']
