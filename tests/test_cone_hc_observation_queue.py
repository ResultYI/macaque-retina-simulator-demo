"""Short CPU queue/identity/codec regressions, not GPU or numerical acceptance."""
from copy import deepcopy
from threading import RLock
from types import SimpleNamespace as NS
import json
import subprocess
import sys

import numpy as np
import pytest

from retina_sim.contracts import ContractError, SolverConfig, canonical_hash
from retina_sim.cone_hc_observation_queue import (
    is_queue_node, queue_contract, upgrade_split_reference,
    queue_time_evidence, compare_queue_times,
)
from retina_sim.cone_hc_split_runner import ConeHCSplitRunner, ConeHCSplitState
from retina_sim.resident_delay import ResidentTimestampedDelay


def delay(torch):
    return ResidentTimestampedDelay.from_payload(dict(delay_seconds=.00282,
        interpolation='linear', prehistory_stop_seconds=0.,
        prehistory=dict(values=[1.,2.],unit='vH_native_voltage',meaning='native_relative_voltage',
                        evidence_tag='synthetic_fixture'),
        history_times=[0.],history_values=[[1.,2.]]),torch,'cpu')


def legacy_fixture():
    import torch
    runner = ConeHCSplitRunner.__new__(ConeHCSplitRunner)
    runner.torch, runner._lock = torch, RLock()
    runner._check_runtime = lambda: None
    runner.validate_solver = lambda config: None
    runner.model = NS(model_hash='legacy-queue-model',_state_spec_hash='spec',
        graph=NS(graph_hash='graph'),parameters=NS(parameter_hash='params'),model_id='fixture',
        version='1',evidence_tag='synthetic_fixture',compute_metadata={},
        _local_slices={'native':(slice(0,2),)})
    runner.compiled = NS(assembly_hash='assembly',components={'native':NS(
        native_model=NS(observation_interval_seconds=.0001),state_specs=[NS(state_name='Vh')],
        outputs={'Vh':NS(unit='vH_native_voltage',meaning='native_relative_voltage')})})
    runner.partition = NS(cid='native',gather=torch.tensor([0]),work=np.zeros(1),statistics={})
    runner.evaluator = NS(runner=runner)
    runner.nstate,runner.observation_dt,runner.periods = 2,.0001,{'inner':.001}
    runner.integration_metrics,runner.split_metrics = {},{}
    runner._binding = {'binding_hash':'original-input'}
    metadata = dict(state_spec_hash='spec',graph_hash='graph',parameter_hash='params',
        model_hash=runner.model.model_hash,model_id='fixture',model_version='1',
        evidence_tag='synthetic_fixture',solver_config=SolverConfig())
    state = ConeHCSplitState(torch.tensor([1.,2.],dtype=torch.float64),.0002,metadata,'assembly',
        0.,1,{}, {'native':delay(torch)}, {},{}, {'inner':1})
    return runner,state


@pytest.mark.parametrize('offset',[-1,1])
def test_unrequested_adjacent_ULP_is_excluded_but_explicit_request_is_retained(offset):
    legacy,state = legacy_fixture()
    runner,state = upgrade_split_reference(legacy,state)
    grid = .0001
    neighbor = float(np.nextafter(grid,np.inf if offset>0 else -np.inf))
    runner._queue_interval_stop = grid
    runner._queue_sample_times = frozenset()
    runner._update_observation_delays(state,neighbor,state.y)
    assert state.delay_states['native'].history_times==[0.]
    runner._queue_sample_times = frozenset((neighbor,))
    for t in sorted((neighbor,grid)):
        assert is_queue_node(t,runner._queue_sample_times,grid)
        runner._update_observation_delays(state,t,state.y)
    assert state.delay_states['native'].history_times==[0.,*sorted((neighbor,grid))]
    with pytest.raises(ContractError,match='lookahead'):
        runner._update_observation_delays(state,.0003,state.y)


def test_historical_38_ULP_native_endpoint_no_longer_adds_a_queue_knot():
    legacy,state = legacy_fixture()
    runner,state = upgrade_split_reference(legacy,state)
    state.time_seconds = 6.4005
    runner._queue_sample_times,runner._queue_interval_stop = frozenset(),6.4005
    runner._update_observation_delays(state,6.400499999999966,state.y)
    runner._update_observation_delays(state,6.4005,state.y)
    assert state.delay_states['native'].history_times==[0.,6.4005]


def test_migration_preserves_history_and_uses_distinct_model_identity():
    legacy,original = legacy_fixture()
    inherited = float(np.nextafter(.0001,-np.inf))
    original.delay_states['native'].append(inherited,original.y,'vH_native_voltage','native_relative_voltage')
    before = original.delay_states['native'].payload()
    runner,state = upgrade_split_reference(legacy,original)
    assert state.delay_states['native'].payload()==before
    assert state.model_hash!=original.model_hash and original.model_hash==legacy.model.model_hash
    assert state.input_position['observation_queue_upgrade']['observation_queue']==queue_contract()
    assert not original.input_position
    assert runner.evaluator.runner is runner and legacy.evaluator.runner is legacy
    runner.partition.work[0] = 7
    assert legacy.partition.work[0]==0
    with pytest.raises(ContractError,match='model_hash'):
        original.validate(runner)
    with pytest.raises(ContractError,match='model_hash'):
        state.validate(legacy)


def test_input_edge_and_adjacent_grid_are_distinct_but_left_right_observation_does_not_duplicate():
    legacy,state = legacy_fixture()
    runner,state = upgrade_split_reference(legacy,state)
    grid=.0001;edge=float(np.nextafter(grid,-np.inf));seen=[]
    runner._queue_sample_times=frozenset()
    stimulus=NS(channels={'light':NS(breakpoints_seconds=(edge,))},
        at=lambda t,left_limit=False:NS(left=left_limit))
    for t in (edge,grid):
        runner._queue_interval_stop=t
        runner._update_observation_delays(state,t,state.y)
        runner._observe(lambda t,y,b:seen.append((t,b.left)),stimulus,t,state.y)
        runner._update_observation_delays(state,t,state.y)
    assert seen==[(edge,True),(edge,False),(grid,False)]
    assert state.delay_states['native'].history_times==[0.,edge,grid]


def test_queue_failure_evidence_includes_complete_times_hex_and_first_difference():
    _,state=legacy_fixture()
    a=queue_time_evidence(state.delay_states,time_seconds=.0002)
    b=deepcopy(a); neighbor=float(np.nextafter(.0001,-np.inf))
    a['components']['native']=dict(seconds=[0.,.0001],hex=[0.0.hex(),(.0001).hex()])
    b['components']['native']=dict(seconds=[0.,neighbor,.0001],hex=[0.0.hex(),neighbor.hex(),(.0001).hex()])
    result=compare_queue_times(a,b)
    assert result['status']=='FAIL'
    first=result['components']['native']['first_difference']
    assert first==dict(index=1,reference=dict(seconds=.0001,hex=(.0001).hex()),
                      candidate=dict(seconds=neighbor,hex=neighbor.hex()),missing_component=None)
    decoded=json.loads(json.dumps(result,allow_nan=False))
    assert decoded['candidate']['components']['native']['hex']==b['components']['native']['hex']
    assert compare_queue_times(a,deepcopy(a))['status']=='PASS'
    b=deepcopy(a);b['components']['native']['seconds'].pop();b['components']['native']['hex'].pop()
    assert compare_queue_times(a,b)['components']['native']['first_difference']['candidate'] is None
    b=deepcopy(a);b['components']={}
    assert compare_queue_times(a,b)['status']=='FAIL'
    b=deepcopy(a);b['time_seconds']=np.nextafter(.0002,np.inf)
    assert not compare_queue_times(a,b)['same_public_time']


def reduced_clock_fixture(monkeypatch):
    """Actual runner/CPU solver/queue/codec, synthetic linear native packets."""
    import torch
    from tests.test_cone_hc_reduced_interfaces import tensor_fixture
    from retina_sim.cone_hc_reduced_native import AnalyticREAnchor,ReducedNativeState,QuadraticNativeSegment
    from retina_sim.cone_hc_reduced_runner import ReducedConeHCRunner,ReducedPoint,ReducedCursor,CONSUMERS
    import retina_sim.network
    runner,state=tensor_fixture()
    engine=ReducedConeHCRunner.__new__(ReducedConeHCRunner);engine.__dict__.update(vars(runner))
    engine.model=NS(evidence_tag='synthetic_fixture')
    engine.evaluate=lambda t,point,bundle:NS(time_seconds=t)
    part=engine.partition
    part.external=[];part.work=part.program=part.weights=part.indices=part.pointers=part.derivatives=np.array([])
    part.fn=lambda y,*args:np.zeros_like(y)
    engine.periods,engine.observation_dt={'inner':.0001},.0001
    edge=.002867
    engine.channels={'clock':NS(evaluator=object(),breakpoints_seconds=(edge,))}
    stimulus=NS(channels=engine.channels,breakpoints_seconds=(edge,),
        at=lambda t,left_limit=False:NS(channels={},delay_left_limit=left_limit))
    monkeypatch.setattr(retina_sim.network,'NetworkStimulus',lambda *args:stimulus)
    engine.metrics=dict(native_seconds=0.,inner_seconds=0.,monitor_seconds=0.,advance_seconds=0.,inner_nfev=0,native_segments=0)
    engine.compiled.components['native']=NS(native_model=NS(observation_interval_seconds=.0001))
    a0=state.point.native.anchor
    a=AnalyticREAnchor(0.,a0.R,a0.E,a0.light,edge)
    native=ReducedNativeState(0.,state.point.native.y,a,engine.native.source_version,False)
    state.point=ReducedPoint(native,state.point.inner)
    state.cursor=ReducedCursor(native,consumed=dict.fromkeys(CONSUMERS,0.))
    state.component_tick_indices,state.sample_index={'inner':0},0
    state.delay_states={'native':delay(torch)}
    def sample(st,stimulus):
        st.component_tick_indices['inner']+=1;st.sample_index+=1
    engine._sample=sample
    def ensure(st):
        n=st.cursor.frontier
        if n.time_seconds>st.time_seconds:return
        if n.time_seconds==n.anchor.input_stop_seconds:
            R,E=n.anchor.at(n.time_seconds)
            anchor=AnalyticREAnchor(n.time_seconds,R,E,n.anchor.light+1,1.,n.anchor.input_epoch+1)
            n=ReducedNativeState(n.time_seconds,n.y,anchor,n.source_version,False)
        end=min(n.time_seconds+.0001,n.anchor.input_stop_seconds)
        f=torch.zeros_like(n.y);f[engine.native.slices['Vh']]=1.
        y1=n.y+(end-n.time_seconds)*f
        segment=QuadraticNativeSegment(n.time_seconds,end,n.y,f,y1,n.anchor.input_epoch,n.anchor,n.source_version,False)
        st.cursor.frontier=ReducedNativeState(end,y1,n.anchor,n.source_version,False)
        st.cursor.segments.append(segment)
        st.cursor.packets.append(engine.selection.project(segment,segment_id=st.cursor.next_id))
        st.cursor.next_id+=1
    engine._ensure_segment=ensure
    return engine,state,edge


def test_reduced_checkpoint_resume_preserves_ULP_queue_input_boundary_and_delayed_readout(monkeypatch,tmp_path):
    from retina_sim.cone_hc_reduced_checkpoint import save_complete,load_complete,SCHEMA
    engine,initial,edge=reduced_clock_fixture(monkeypatch)
    seam=.002831;grid=.0029;stop=.003
    lo,hi=float(np.nextafter(grid,-np.inf)),float(np.nextafter(grid,np.inf))
    samples=np.array([0.,seam,edge,lo,grid,hi,stop])
    recorded=[]
    full=engine.advance(initial,engine.channels,stop,sample_times=samples,state_observer=lambda *args:None,
        record=lambda t,*args:recorded.append(t))
    assert recorded==list(samples)
    prefix=engine.advance(initial,engine.channels,seam,sample_times=samples[samples<=seam],state_observer=lambda *args:None)
    receipt=save_complete(engine,prefix,tmp_path/'checkpoint')
    assert receipt['schema']==SCHEMA=='cone-hc-reduced-complete-binary-checkpoint-2'
    restored=load_complete(engine,tmp_path/'checkpoint',expected_manifest_hash=receipt['manifest_hash'])
    assert restored.delay_states['native'].payload()==prefix.delay_states['native'].payload()
    assert len(restored.cursor.segments)==1 and restored.cursor.frontier.time_seconds==edge
    observed=[];recorded=[]
    resumed=engine.advance(restored,engine.channels,stop,sample_times=samples[samples>=seam],
        state_observer=lambda t,y,b:observed.append((t,b.delay_left_limit)),
        record=lambda t,*args:recorded.append(t))
    assert recorded==list(samples[samples>=seam])
    assert (edge,True) in observed and (edge,False) in observed
    times=resumed.delay_states['native'].history_times
    assert all(t in times for t in (edge,lo,grid,hi))
    assert compare_queue_times(queue_time_evidence(full.delay_states,time_seconds=stop),
                               queue_time_evidence(resumed.delay_states,time_seconds=stop))['status']=='PASS'
    assert full.delay_states['native'].payload()==resumed.delay_states['native'].payload()
    assert engine.torch.equal(full.delay_states['native'].at(stop),resumed.delay_states['native'].at(stop))
    assert engine.torch.equal(full.point.native.y,resumed.point.native.y)
    assert full.component_tick_indices==resumed.component_tick_indices
    assert initial.delay_states['native'].history_times==[0.]


def test_checkpoint_rejects_prior_queue_schema_or_resealed_wrong_contract(tmp_path):
    from tests.test_cone_hc_reduced_interfaces import tensor_fixture
    from retina_sim.cone_hc_reduced_checkpoint import save_complete,load_complete
    runner,state=tensor_fixture();receipt=save_complete(runner,state,tmp_path/'checkpoint')
    path=tmp_path/'checkpoint'/'manifest.json';original=json.loads(path.read_text())
    for key,value in (('schema','cone-hc-reduced-complete-binary-checkpoint-1'),('observation_queue',{})):
        changed=deepcopy(original);changed[key]=value;changed.pop('manifest_hash')
        expected=canonical_hash(changed);changed['manifest_hash']=expected
        path.write_text(json.dumps(changed),encoding='utf-8')
        with pytest.raises(ContractError,match='schema/content'):
            load_complete(runner,tmp_path/'checkpoint',expected_manifest_hash=expected)


def test_queue_preflight_import_stays_standard_library_only():
    result=subprocess.run([sys.executable,'-B','-c',
        "import sys; import tools.validate_cone_hc_reduced; assert 'numpy' not in sys.modules; assert 'torch' not in sys.modules"],
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
