"""Full state/monitor/event/queue equivalence for the background-only interface."""
from pathlib import Path
import json
import numpy as np
import pytest
from examples.full_network_fixture import full_fixture
from retina_sim.background import BackgroundPolicy,_WindowMonitor,constant_background_channels
from retina_sim.background_advance import StateOnlyBackgroundRunner
from retina_sim.contracts import ContractError,SolverConfig
from retina_sim.hybrid import HybridRunner
from retina_sim.recording import RecordingPlan
from retina_sim.runtime_profiles import LightContext

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def runner():
    compiled,_,_=full_fixture(ROOT)
    return HybridRunner(compiled,compute_backend='python')


def compare(runner,baseline,channels,stop,times):
    fast=StateOnlyBackgroundRunner(runner);policy=BackgroundPolicy();plan=RecordingPlan(ports=(),states=(),cell_ids=())
    before=runner.checkpoint(baseline);records=[];outputs=[];counts=[];calls=[]
    for target in (runner,fast):
        monitor=_WindowMonitor(target,baseline,policy);points=[];metrics=dict(runner.integration_metrics)
        def observe(t,y,bundle):
            points.append((t,y.copy()));monitor.observe(t,y,bundle)
        state,result=target.advance(baseline,channels,stop,sample_times=times,recording_plan=plan,state_observer=observe)
        assert not monitor.missing
        records.append((points,monitor.report(stop-baseline.time_seconds)))
        outputs.append(state);counts.append({key:runner.integration_metrics[key]-metrics[key] for key in ('solver_calls','nfev')})
        calls.append(result)
    assert calls[0] is not None and calls[1] is None
    assert counts[0]==counts[1]
    assert runner.checkpoint(outputs[0])==runner.checkpoint(outputs[1])
    assert runner.checkpoint(baseline)==before
    assert records[0][1]==records[1][1]
    assert [t for t,y in records[0][0]]==[t for t,y in records[1][0]]
    for (_,a),(_,b) in zip(records[0][0],records[1][0]):np.testing.assert_array_equal(a,b)
    return outputs


@pytest.mark.parametrize('interpolation',['linear','zero_order_hold'])
def test_all_states_monitor_rng_and_delays_and_checkpoint_continuation(runner,interpolation):
    solver=SolverConfig(rtol=1e-6,atol=1e-9,max_step=.0001,input_interpolation=interpolation)
    baseline=runner.initialize(solver)
    channels=constant_background_channels(runner.compiled,LightContext(),.01,interpolation=interpolation)
    times=np.array([0.,.00013,.00047,.001,.00123,.0021,.003])
    left,right=compare(runner,baseline,channels,.003,times)
    restored=runner.restore(runner.checkpoint(left))
    compare(runner,restored,channels,.004,np.array([.003,.0031,.00347,.004]))
    assert left.spike_states and left.delay_states
    for cid in left.spike_states:
        assert left.spike_states[cid]['rng_state']!=baseline.spike_states[cid]['rng_state']


def test_does_not_build_metadata_or_evaluate_empty_readouts(runner,monkeypatch):
    import retina_sim.hybrid as hybrid
    def forbidden(*a,**k):raise AssertionError('Unneeded SimulationResult metadata requested')
    monkeypatch.setattr(hybrid,'execution_metadata',forbidden)
    fast=StateOnlyBackgroundRunner(runner);state=runner.initialize(SolverConfig(input_interpolation='linear'));policy=BackgroundPolicy()
    monitor=_WindowMonitor(fast,state,policy);calls=[];original=runner.model.evaluate
    def traced(*a,**kw):
        import inspect
        calls.append(inspect.currentframe().f_back.f_code.co_name)
        return original(*a,**kw)
    monkeypatch.setattr(runner.model,'evaluate',traced)
    channels=constant_background_channels(runner.compiled,LightContext(),.001)
    _,result=fast.advance(state,channels,.0003,sample_times=np.array([0.,.0001,.0002,.0003]),
        recording_plan=RecordingPlan(ports=(),states=(),cell_ids=()),state_observer=monitor.observe)
    assert result is None and calls and 'record' not in calls


def test_explicit_observer_and_empty_plan_required(runner):
    fast=StateOnlyBackgroundRunner(runner);state=runner.initialize()
    channels=constant_background_channels(runner.compiled,LightContext(),.001)
    with pytest.raises(ContractError,match='empty RecordingPlan'):
        fast.advance(state,channels,.0001,recording_plan=RecordingPlan(),state_observer=lambda *a:None)
    with pytest.raises(ContractError,match='numerical observer'):
        fast.advance(state,channels,.0001,recording_plan=RecordingPlan(ports=(),states=(),cell_ids=()))


def test_unreviewed_readout_is_rejected(runner,monkeypatch):
    component=next(iter(runner.compiled.components.values()))
    monkeypatch.setattr(component,'implementation_hash','unreviewed_side_effectful_observer')
    with pytest.raises(ContractError,match='pure component readouts'):StateOnlyBackgroundRunner(runner)
