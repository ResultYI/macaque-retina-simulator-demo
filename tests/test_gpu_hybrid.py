"""Optional CUDA execution parity, event/history and checkpoint boundaries."""
from pathlib import Path
import json
import numpy as np
import pytest
torch=pytest.importorskip('torch')
if not torch.cuda.is_available():pytest.skip('Explicit optional CUDA execution requires CUDA',allow_module_level=True)
from scipy.integrate import solve_ivp
from retina_sim.contracts import ContractError,SolverConfig
from retina_sim.hybrid import HybridRunner
from retina_sim.gpu_hybrid import CudaHybridRunner
from retina_sim.gpu_dop853 import integrate
from retina_sim.network import InputBundle
from examples.full_network_fixture import full_fixture

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def fixture():
    torch.set_num_threads(1)
    compiled,channels,_=full_fixture(ROOT)
    return compiled,channels


def test_dop853_dense_nonautonomous_times_and_evaluations():
    rates=np.array([-300.,-2500.,-20.]);initial=np.array([1.,.1,10.])
    sampled=np.array([0.,.00003,.0001,.00027,.0005,.001])
    expected=solve_ivp(lambda t,y:rates*y+np.sin(t*12345.),(0.,.001),initial,method='DOP853',
        rtol=1e-9,atol=1e-12,max_step=.0001,t_eval=sampled)
    device_rates=torch.as_tensor(rates,device='cuda')
    actual=integrate(lambda t,y:device_rates*y+np.sin(t*12345.),0.,torch.as_tensor(initial,device='cuda'),.001,
        rtol=1e-9,atol=1e-12,max_step=.0001,t_eval=sampled)
    np.testing.assert_array_equal(actual['t'],expected.t)
    np.testing.assert_allclose(actual['y'],expected.y,rtol=1e-12,atol=1e-12)
    assert actual['nfev']==expected.nfev


def test_partition_rhs_retains_full_inner_plan_and_isolation(fixture):
    compiled,channels=fixture;runner=CudaHybridRunner(compiled)
    model=runner.model;state=runner.initialize();initial=runner._flat(state)
    bundle=InputBundle({key:channel.at(.00037) for key,channel in channels.items()})
    rng=np.random.default_rng(2172);perturbed=initial.copy()
    perturbed[:runner.partition.offset]*=rng.uniform(.995,1.005,runner.partition.offset)
    first=None
    for values in (initial,perturbed,initial):
        expected=model.rhs_flat(.00037,values,bundle,model.parameters,model.graph)
        actual=runner.partition(.00037,torch.as_tensor(values.copy(),device='cuda'),bundle).cpu().numpy()
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=2e-9)
        if first is None:first=actual.copy()
    np.testing.assert_array_equal(actual,first)
    assert len(actual)==sum(s.shape[0] for s in model.state_specs)


@pytest.mark.parametrize('kind',['conductance','adaptation','negative_OS','nonfinite'])
def test_native_domains_reject_without_clipping(fixture,kind):
    compiled,channels=fixture;runner=CudaHybridRunner(compiled);values=runner._flat(runner.initialize()).copy()
    blocks=runner.model._local_slices[runner.partition.cid]
    index={'conductance':5,'adaptation':6,'negative_OS':2,'nonfinite':0}[kind]
    values[blocks[index].start]=np.nan if kind=='nonfinite' else -1.
    bundle=InputBundle({key:channel.at(0.) for key,channel in channels.items()})
    with pytest.raises(ContractError,match='domain'):
        runner.partition(0.,torch.as_tensor(values,device='cuda'),bundle)


def test_joint_events_recording_rng_and_checkpoint_continuation(fixture):
    compiled,channels=fixture;cpu=HybridRunner(compiled,compute_backend='numba');gpu=CudaHybridRunner(compiled)
    config=SolverConfig(method='DOP853',rtol=1e-8,atol=1e-11,max_step=.0001)
    original_cpu=cpu.initialize(config);original_gpu=gpu.initialize(config)
    with pytest.raises(ContractError):gpu.restore(cpu.checkpoint(original_cpu))
    with pytest.raises(ContractError):cpu.restore(gpu.checkpoint(original_gpu))
    times=np.array([0.,.00031,.00075,.00123,.0021])
    left,cpu_result=cpu.advance(original_cpu,channels,.0021,sample_times=times)
    right,gpu_result=gpu.advance(original_gpu,channels,.0021,sample_times=times)
    np.testing.assert_allclose(cpu._flat(left),gpu._flat(right),rtol=2e-10,atol=2e-10)
    np.testing.assert_array_equal(cpu_result.outputs_with_units['time'].values,gpu_result.outputs_with_units['time'].values)
    assert left.component_tick_indices==right.component_tick_indices
    assert left.sample_index==right.sample_index
    for cid,cpu_aux in left.spike_states.items():
        gpu_aux=right.spike_states[cid]
        assert cpu_aux.keys()==gpu_aux.keys()
        for key,value in cpu_aux.items():
            if key=='conditional_rate_Hz':
                np.testing.assert_allclose(value,gpu_aux[key],rtol=2e-10,atol=2e-10)
            elif key=='pending_events':
                assert len(value)==len(gpu_aux[key])
                for expected_event,actual_event in zip(value,gpu_aux[key]):
                    assert expected_event.keys()==actual_event.keys()
                    for field in expected_event:
                        if field=='time_seconds':
                            assert abs(expected_event[field]-actual_event[field])<1e-12
                        else:assert expected_event[field]==actual_event[field]
            else:
                assert json.dumps(value,sort_keys=True)==json.dumps(gpu_aux[key],sort_keys=True)
    for key,value in cpu_result.outputs_with_units.items():
        np.testing.assert_allclose(value.values,gpu_result.outputs_with_units[key].values,rtol=2e-10,atol=2e-10)
    text=gpu.checkpoint(right);restored=gpu.restore(text)
    assert gpu.checkpoint(restored)==text
    a,_=gpu.advance(right,channels,.0026);b,_=gpu.advance(restored,channels,.0026)
    assert gpu.checkpoint(a)==gpu.checkpoint(b)


def test_full_target_inner_mechanisms_and_gap_plan():
    from examples.full_target_network import full_target_network
    compiled,channels,_=full_target_network(ROOT,effective_cones=64,effective_rods=96,duration=.003,
        cone_route='native_shared_H1_approximation',bounds_mm=(.004,4.,-.04,.04))
    native=compiled.components['outer_native'].native_model
    before=[tuple(array.copy() for array in (m.data,m.indices,m.indptr)) for m in (native.A,native.B)]
    cpu=HybridRunner(compiled,compute_backend='numba');gpu=CudaHybridRunner(compiled)
    for matrix,saved in zip((native.A,native.B),before):
        for actual,original in zip((matrix.data,matrix.indices,matrix.indptr),saved):
            np.testing.assert_array_equal(actual,original)
    config=SolverConfig(method='DOP853',rtol=1e-8,atol=1e-11,max_step=.0001)
    left=cpu.initialize(config);right=gpu.initialize(config)
    values=cpu._flat(left);bundle=InputBundle({key:channel.at(.0001) for key,channel in channels.items()})
    expected=cpu.model.rhs_flat(.0001,values,bundle,cpu.model.parameters,cpu.model.graph)
    actual=gpu.partition(.0001,torch.as_tensor(values.copy(),device='cuda'),bundle).cpu().numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=2e-9)
    left,_=cpu.advance(left,channels,.00035);right,_=gpu.advance(right,channels,.00035)
    np.testing.assert_allclose(cpu._flat(left),gpu._flat(right),rtol=2e-10,atol=2e-10)
    assert left.component_tick_indices==right.component_tick_indices
    assert any(c.mechanism=='HistoryBipolarTerminal' for c in compiled.components.values())
    assert any(c.mechanism=='RodContactTRPM1' for c in compiled.components.values())


def test_complete_background_checkpoint_with_cuda_identity(fixture,tmp_path):
    from dataclasses import replace
    from retina_sim.assembly import RetinaBuilder
    from retina_sim.contracts import RunMode
    from retina_sim.background import BackgroundPolicy
    from retina_sim.background_checkpoint import prepare_with_checkpoint,MemoryGuard
    from retina_sim.runtime_profiles import LightContext
    _,_,audit=full_fixture(ROOT);spec=audit['spec']
    spec=replace(spec,populations=spec.populations[:2],components=spec.components[:1],links=())
    compiled=RetinaBuilder(audit['registry']).build(spec,RunMode.NUMERIC_FIXTURE)
    runner=CudaHybridRunner(compiled);reference=runner.initialize(SolverConfig(input_interpolation='linear'))
    policy=BackgroundPolicy(window_seconds=.0001,stable_windows=2,minimum_seconds=.0002,
        known_tau_multiplier=0.,delay_multiplier=0.,max_sim_seconds=.001,max_wall_seconds=20.,
        monitor_step_seconds=.0001,require_converged=True)
    path=tmp_path/'background.json'
    state,report,store=prepare_with_checkpoint(ROOT,runner,reference,LightContext(),policy,path,
        guard=MemoryGuard(12*1024**3,1024**3))
    assert report['converged']
    restored,value=store.load(path,require_converged=True)
    assert runner.checkpoint(state)==runner.checkpoint(restored)
    assert value['binding']['model_hash']==runner.model.model_hash


def test_named_factory_and_full_registered_verification(fixture):
    from retina_sim.execution_backends import make_runner,verify_cuda_execution,CUDA,CPU
    from retina_sim.runtime_profiles import LightContext
    compiled,_=fixture
    assert isinstance(make_runner(compiled,CPU),HybridRunner)
    with pytest.raises(ContractError):make_runner(compiled,'silent_unknown')
    runner=make_runner(compiled,CUDA)
    result=verify_cuda_execution(runner,SolverConfig(rtol=1e-6,atol=1e-9,max_step=.0001,input_interpolation='linear'),LightContext())
    assert result['event_clocks_equal'] and result['rng_counts_history_equal']
    assert result['maximum_difference_over_original_solver_scale']<.01
    assert runner.integration_metrics['solver_calls']==0
