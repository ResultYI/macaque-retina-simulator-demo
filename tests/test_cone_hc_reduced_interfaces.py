"""CPU interface/codec tests; these do not certify a CUDA execution."""
from dataclasses import replace
from types import SimpleNamespace as NS
from threading import RLock
import json

import numpy as np
import pytest

from retina_sim.contracts import ContractError, SolverConfig
from retina_sim.cone_hc_reduced_runner import ReducedPoint, ReducedState, ReducedCursor, CONSUMERS
from retina_sim.cone_hc_reduced_native import ReducedNativeState
from retina_sim.cone_hc_reduced_bridge import BridgeSelection
from retina_sim.cone_hc_reduced_checkpoint import save_complete, load_complete
from tests.test_cone_hc_reduced_native import fixture
from tools.validate_cone_hc_reduced import compare_vector


def tensor_fixture():
    import torch
    from retina_sim.cone_hc_reduced_native import AnalyticREAnchor, QuadraticNativeSegment
    model,native,old,light = fixture()
    initial = native.migrate(old,time_seconds=0.,light=light,input_stop_seconds=.000067)
    final,segment = native.step(initial)
    def tensor(v):
        return torch.as_tensor(np.array(v,copy=True),dtype=torch.float64)
    def anchor(a):
        return AnalyticREAnchor(a.time_seconds,tensor(a.R),tensor(a.E),tensor(a.light),
                                a.input_stop_seconds,a.input_epoch)
    a = anchor(initial.anchor)
    seg = QuadraticNativeSegment(segment.start,segment.stop,tensor(segment.y0),tensor(segment.f0),
        tensor(segment.y1),segment.input_epoch,a,native.source_version,False)
    chosen = BridgeSelection(model.B,np.array([0,2]))
    packet = chosen.project(segment,segment_id=0)
    point = ReducedPoint(ReducedNativeState(.000031,seg.at(.000031),a,native.source_version,False),np.array([.5]))
    cursor = ReducedCursor(ReducedNativeState(final.time_seconds,tensor(final.y),a,native.source_version,False),
                           [seg],[packet],dict.fromkeys(CONSUMERS,.000031),1)
    native.device = torch.device('cpu')
    native._state_guard = lambda state:state.check_owned()
    native._validate = lambda *args:None
    runner = NS(torch=torch,native=native,selection=chosen,bridge=NS(rows=torch.as_tensor(chosen.rows.copy())),
        identity='fixture-reduced-identity',binding_hash='fixture-bound-input',_lock=RLock(),
        _check_runtime=lambda:None,validate_solver=lambda config:None,periods={'inner':.001},
        compiled=NS(components={'inner':NS(native_model=NS())}),partition=NS(cid='native',inner_size=1))
    state = ReducedState(point,cursor,runner.identity,runner.binding_hash,SolverConfig(),0.,1,{}, {},{}, {'inner':1})
    return runner,state


def test_four_consumer_watermarks_and_owned_fork():
    runner,state = tensor_fixture()
    fork = state.fork()
    fork.cursor.packets[0].metadata['segment_id'] = 44
    assert state.cursor.packets[0].segment_id==0
    end = state.cursor.frontier.time_seconds
    for consumer in CONSUMERS[:-1]:
        state.cursor.acknowledge(consumer,end,runner.bridge)
        assert len(state.cursor.segments)==1
    state.cursor.acknowledge(CONSUMERS[-1],end,runner.bridge)
    assert not state.cursor.segments and not state.cursor.packets
    assert len(fork.cursor.segments)==1
    with pytest.raises(ContractError,match='watermark'):
        fork.cursor.acknowledge('inner',0.,runner.bridge)


def test_chunk_return_transfers_pending_packet_ownership_before_slot_reuse():
    from tests.test_cone_hc_reduced_bridge import selection,segment,FakeTorch,gpu_source
    from retina_sim.cone_hc_reduced_bridge import PinnedSegmentBridge
    chosen,fake = selection(),FakeTorch()
    bridge = PinnedSegmentBridge(chosen,fake,'cuda')
    packet = bridge.submit(gpu_source(segment(),fake))
    cursor = ReducedCursor(None,packets=[packet])
    cursor.own_packets(bridge)
    assert packet._released and cursor.packets[0]._lease_check is None
    expected = cursor.packets[0].release_at(.000031).copy()
    reused = bridge.submit(gpu_source(segment(),fake),segment_id=1)
    np.testing.assert_array_equal(cursor.packets[0].release_at(.000031),expected)
    bridge.release(reused)


def test_checkpoint_retains_pending_segment_packet_anchors_and_rejects_corruption(tmp_path):
    runner,state = tensor_fixture()
    state.validate(runner)
    receipt = save_complete(runner,state,tmp_path/'checkpoint')
    loaded = load_complete(runner,tmp_path/'checkpoint',expected_manifest_hash=receipt['manifest_hash'])
    np.testing.assert_array_equal(loaded.point.native.y,state.point.native.y)
    np.testing.assert_array_equal(loaded.cursor.packets[0].release_at(.00005),
                                  state.cursor.packets[0].release_at(.00005))
    np.testing.assert_array_equal(loaded.cursor.frontier.anchor.at(.000067)[1],
                                  state.cursor.frontier.anchor.at(.000067)[1])
    assert loaded.cursor.consumed==state.cursor.consumed
    with pytest.raises(ContractError,match='absent'):
        save_complete(runner,state,tmp_path/'checkpoint')
    path = tmp_path/'checkpoint'/'array_00000.npy'
    path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ContractError,match='receipt'):
        load_complete(runner,tmp_path/'checkpoint',expected_manifest_hash=receipt['manifest_hash'])


def test_checkpoint_refuses_old_schema_changed_ids_and_staging(tmp_path):
    runner,state = tensor_fixture()
    receipt = save_complete(runner,state,tmp_path/'checkpoint')
    runner.binding_hash = 'other-input'
    with pytest.raises(ContractError,match='differs'):
        load_complete(runner,tmp_path/'checkpoint',expected_manifest_hash=receipt['manifest_hash'])
    with pytest.raises(ContractError,match='Uncommitted'):
        load_complete(runner,tmp_path/'checkpoint.writing-123',expected_manifest_hash='old')
    p = tmp_path/'old';p.mkdir()
    (p/'manifest.json').write_text(json.dumps({'schema':'old','manifest_hash':'old'}))
    with pytest.raises(ContractError,match='schema'):
        load_complete(runner,p,expected_manifest_hash='old')


def test_zero_budget_failure_is_strict_json_and_never_relaxed():
    result = compare_vector([1.],[2.],[0.],gate='zero')
    assert result['status']=='FAIL' and result['zero_allowance_violation']
    assert result['max_normalized'] is None
    json.dumps(result,allow_nan=False)
    assert compare_vector([1.],[1.],[0.],gate='zero')['status']=='PASS'


def test_selected_delay_projects_on_device_before_CPU_conversion():
    from tools.validate_cone_hc_reduced import _selected_delay
    operations = []
    class Selected:
        def detach(self):return self
        def cpu(self):
            operations.append('cpu_selected')
            return self
        def numpy(self):return np.array([3.,7.])
    class Full:
        def index_select(self,axis,rows):
            assert axis==0 and rows==[3,7]
            operations.append('select_device')
            return Selected()
        def cpu(self):raise AssertionError('Unselected HC transfer')
    result = _selected_delay(NS(at=lambda t:Full()),.01,[3,7])
    np.testing.assert_array_equal(result,[3.,7.])
    assert operations==['select_device','cpu_selected']


def test_reference_recording_uses_real_SimulationResult_fields_and_events():
    from retina_sim.contracts import SimulationResult,RunMode,Quantity
    from tools.validate_cone_hc_reduced import _collect_reference_recording
    result = SimulationResult(RunMode.NUMERIC_FIXTURE,'fixture','source',{},
        {'native.observed_Vh':Quantity([[1.,2.]],'vH_native_voltage','native_relative_voltage','synthetic_fixture')},
        None,{},'fixture',{'spike_events':{'rgc':[dict(cell_index=0,time_seconds=.00005)]}})
    delays,events = {},[]
    _collect_reference_recording(result,'native',[0.,.0001],delays,events)
    np.testing.assert_array_equal(delays[.0001],[2.])
    assert events==[dict(component='rgc',cell_index=0,time_seconds=.00005)]


def test_inner_evaluator_body_keeps_original_gather_gap_and_rhs_order():
    # The reference file is hash-bound by old checkpoints and stays byte exact.
    import inspect
    from retina_sim.resident_evaluation import ResidentEvaluator
    from retina_sim.cone_hc_reduced_evaluation import ReducedEvaluator
    original = inspect.getsource(ResidentEvaluator.__call__)
    reduced = inspect.getsource(ReducedEvaluator.evaluate_inner)
    marker = '        # This private Quantity'
    assert original[original.index(marker):]==reduced[reduced.index(marker):]


def test_added_native_endpoints_do_not_move_original_event_ticks_by_one_ULP():
    import torch
    from retina_sim.cone_hc_reduced_native import AnalyticREAnchor,QuadraticNativeSegment
    from retina_sim.cone_hc_reduced_runner import ReducedConeHCRunner
    runner,state = tensor_fixture()
    engine = ReducedConeHCRunner.__new__(ReducedConeHCRunner)
    engine.__dict__.update(vars(runner))
    engine.model = NS(evidence_tag='synthetic_fixture')
    engine.partition.external = []
    engine.partition.work = engine.partition.program = engine.partition.weights = np.array([])
    engine.partition.indices = engine.partition.pointers = engine.partition.derivatives = np.array([])
    engine.partition.fn = lambda y,*args:np.zeros_like(y)
    engine.periods,engine.observation_dt,engine.channels = {'inner':.0001},.0001,{}
    engine.compiled.spec = NS(external_inputs=())
    engine.metrics = dict(native_seconds=0.,inner_seconds=0.,monitor_seconds=0.,advance_seconds=0.,
                          inner_nfev=0,native_segments=0)
    old_anchor = state.point.native.anchor
    a = AnalyticREAnchor(0.,old_anchor.R,old_anchor.E,old_anchor.light,1.)
    initial = ReducedNativeState(0.,state.point.native.y,a,engine.native.source_version,False)
    state.point,state.cursor = ReducedPoint(initial,state.point.inner),ReducedCursor(
        initial,consumed=dict.fromkeys(CONSUMERS,0.))
    state.component_tick_indices,state.sample_index = {'inner':0},0
    engine.compiled.components['inner'].native_model.sample_spikes = True
    state.spike_states = {'inner':dict(history=np.zeros((1,3)),last_bin_counts=np.zeros(1),
        rng_state=np.random.default_rng(1).bit_generator.state,conditional_rate_Hz=np.zeros(1),
        pending_events=[dict(cell_index=0,time_seconds=.00005),dict(cell_index=0,time_seconds=.0003)])}
    visited = []
    def sample(st,stimulus):
        visited.append(st.time_seconds)
        st.component_tick_indices['inner'] += 1
        st.sample_index += 1
    engine._sample = sample
    def ensure(st):
        if st.cursor.frontier.time_seconds>st.time_seconds:
            return
        n = st.cursor.frontier
        end = np.nextafter(n.time_seconds+.0001,-np.inf)
        seg = QuadraticNativeSegment(n.time_seconds,end,n.y,torch.zeros_like(n.y),n.y,
                                     0,a,n.source_version,False)
        st.cursor.frontier = ReducedNativeState(end,n.y,a,n.source_version,False)
        st.cursor.segments.append(seg)
        st.cursor.packets.append(engine.selection.project(seg,segment_id=st.cursor.next_id))
        st.cursor.next_id += 1
    engine._ensure_segment = ensure
    events = []
    ending = engine.advance(state,{},.0002,sample_times=np.array([0.,.0001,.0002]),
                            state_observer=lambda *args:None,event_sink=lambda cid,e:events.append((cid,e)))
    assert visited==[0.,.0001,.0002]
    assert ending.component_tick_indices=={'inner':3}
    assert state.time_seconds==0. and state.component_tick_indices=={'inner':0}
    assert events==[('inner',dict(cell_index=0,time_seconds=.00005))]
    assert ending.spike_states['inner']['pending_events']==[dict(cell_index=0,time_seconds=.0003)]
    assert len(state.spike_states['inner']['pending_events'])==2


def test_monitor_covers_actual_states_analytic_filters_and_equation_RHS():
    import torch
    from retina_sim.cone_hc_reduced_evaluation import ReducedStateSpec
    from retina_sim.cone_hc_reduced_monitor import ReducedWindowMonitor
    from retina_sim.background import BackgroundPolicy
    runner,state = tensor_fixture()
    model,_,old,_ = fixture()
    names = ('R','E','X','C','Vis','g_i','Vbar','z1','z2','Vh')
    units = [s.unit for s in model.state_specs]
    units[7] = 'vH_native_voltage'
    shapes = [3]*8+[2]*2
    specs = tuple(ReducedStateSpec(name,tuple('cell%d'%i for i in range(n)),(n,),unit,
        'test_retained_state','fixture') for name,n,unit in zip(names,shapes,units))
    runner.native.model.state_specs = specs
    runner.compiled.components = {
        'native':NS(native_model=runner.native.model,state_specs=specs,time_contract=NS(kind='continuous')),
        'inner':NS(native_model=NS(),time_contract=NS(kind='continuous'))}
    inner = ReducedStateSpec('inner.V',('i0',),(1,),'V','physical_membrane_voltage','fixture')
    runner.model = NS(state_specs=tuple(replace(s,state_name='native.'+s.state_name) for s in specs)+(inner,),
                      slices={'inner':slice(28,29)})
    runner.partition.offset = 28
    cuts = np.cumsum([0]+shapes)
    reference = NS(ordered_state_arrays=tuple(old[cuts[i]:cuts[i+1]] for i in range(10))+(np.array([.5]),))
    calls = []
    def evaluate(t,point,bundle,**kwargs):
        calls.append((t,bundle.left,kwargs['require_rhs']))
        return NS(native_rhs=torch.full_like(point.native.y,7.),inner_y=point.inner,
            inner_rhs=np.array([-.5]),outputs={},native_outputs={
                'R_rhs':torch.full((3,),13.,dtype=torch.float64),
                'E_rhs':torch.full((3,),17.,dtype=torch.float64)})
    runner.evaluate = evaluate
    monitor = ReducedWindowMonitor(runner,reference,BackgroundPolicy())
    monitor.observe(state.time_seconds,state.point,NS(left=True))
    report = monitor.report(.0001)
    assert calls==[(state.time_seconds,True,True)]
    assert 'native.w' in report['state_coverage'] and 'native.z1' not in report['state_coverage']
    assert len(report['state_coverage'])==12 and not report['full_z1_available']
    by_name = {r['name']:r for r in monitor.rows if r['native']}
    assert torch.equal(by_name['R']['rhs'],torch.full((3,),13.,dtype=torch.float64))
    assert torch.equal(by_name['E']['rhs'],torch.full((3,),17.,dtype=torch.float64))
    assert torch.equal(by_name['Vis']['rhs'],torch.full((3,),7.,dtype=torch.float64))


def test_original_CPU_gather_observe_and_RHS_are_actually_callable():
    from scipy import sparse
    from retina_sim.assembly import PortSpec
    from retina_sim.contracts import Quantity
    from retina_sim.cone_hc_reduced_evaluation import ReducedEvaluator
    tag = 'synthetic_fixture'
    release = PortSpec('release','dimensionless','effective_synaptic_drive',('c0','c2'),tag)
    drive = PortSpec('drive','dimensionless','effective_synaptic_drive',('i0',),tag)
    voltage = PortSpec('voltage','V','physical_membrane_voltage',('i0',),tag)
    inner = NS(inputs={'drive':drive},outputs={'voltage':voltage},output_dependencies={'voltage':()},
        native_model=NS(),observe=lambda name,t,local,deps:Quantity(local[0],'V','physical_membrane_voltage',tag))
    link = NS(specification=NS(source_component='native',source_port='release'),
              delayed_operator=None,matrix=sparse.csr_matrix([[.6,.4]]))
    runner = NS(partition=NS(cid='native',inner_size=1,used=np.array([0,2]),offset=28,external=[],
        fn=lambda y,values,*args:np.array([.6*values[0]+.4*values[1]-y[0]]),
        work=None,program=None,weights=None,indices=None,pointers=None,derivatives=None),
        compiled=NS(components={'native':NS(outputs={'release':release}),'inner':inner},
            spec=NS(external_inputs=()),output_order=(NS(cyclic=False,nodes=(('inner','voltage'),)),)),
        model=NS(_context_providers={'native':None},_local_slices={'inner':(slice(28,29),)},_gap_plans=[]))
    evaluator = ReducedEvaluator(runner,{('inner','drive'):[link]})
    evidence = evaluator.evaluate_inner(0.,np.array([.5]),NS(channels={}),np.array([.2,.8]),require_rhs=True)
    np.testing.assert_allclose(evidence.inputs['inner']['drive'].values,[.44],rtol=0,atol=1e-15)
    np.testing.assert_allclose(evidence.inner_rhs,[-.06],rtol=0,atol=1e-15)
    assert not evidence.inner_y.flags.writeable
