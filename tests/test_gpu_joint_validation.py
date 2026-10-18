from pathlib import Path
import json
import numpy as np
import pytest
from retina_sim.contracts import ContractError,SolverConfig,canonical_hash
from retina_sim.gpu_hybrid import CudaHybridRunner
from retina_sim.gpu_joint_validation import CPUFiniteJointCudaRunner,CPUFiniteJointPartition
from retina_sim.network import InputBundle
from retina_sim.recording import RecordingPlan
from retina_sim.recording_indices import IndexedRecordingPlan
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def gpu():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('Existing CUDA device required')
    from examples.full_network_fixture import full_fixture
    compiled,channels,_=full_fixture(ROOT)
    original=CudaHybridRunner(compiled);candidate=CPUFiniteJointCudaRunner.from_runner(original)
    return torch,compiled,channels,original,candidate


def test_every_rhs_value_and_repeated_stage_changes(gpu):
    torch,compiled,channels,original,candidate=gpu
    assert candidate.model is not original.model
    assert candidate.model.parameters is original.model.parameters
    assert candidate.model.model_hash!=original.model.model_hash
    assert candidate.partition.native is original.partition.native
    assert candidate.partition.work is original.partition.work
    initial=original._flat(original.initialize());rng=np.random.default_rng(72)
    for values in (initial,initial*rng.uniform(.999,1.001,len(initial)),initial):
        device=torch.as_tensor(values.copy(),dtype=torch.float64,device='cuda')
        bundle=InputBundle({name:channel.at(.00037) for name,channel in channels.items()})
        a=original.partition(.00037,device,bundle).cpu().numpy()
        b=candidate.partition(.00037,device,bundle).cpu().numpy()
        np.testing.assert_array_equal(a.view(np.uint64),b.view(np.uint64))
    assert candidate.partition.statistics['redundant_full_GPU_checks_avoided']>=3


@pytest.mark.parametrize('failure',['native_negative','native_nonfinite','inner_nan','inner_inf','channels'])
def test_invalid_domains_still_reject(gpu,monkeypatch,failure):
    torch,_,channels,original,candidate=gpu;p=candidate.partition
    values=original._flat(original.initialize())
    if failure=='native_negative':values[2*p.native.nc]=-1.  # X is an explicitly checked nonnegative source state.
    if failure=='native_nonfinite':values[0]=np.nan
    if failure.startswith('inner_'):
        invalid=np.zeros(p.inner_size);invalid[0]=np.nan if failure=='inner_nan' else np.inf
        monkeypatch.setattr(p,'fn',lambda *args:invalid)
    bundle=InputBundle({name:channel.at(.00037) for name,channel in channels.items()})
    if failure=='channels':bundle=InputBundle({})
    with pytest.raises(ContractError):
        p(.00037,torch.as_tensor(values,device='cuda'),bundle)


def test_complete_window_monitor_outputs_and_checkpoint_identity(gpu):
    torch,_,channels,original,candidate=gpu
    from retina_sim.background import _WindowMonitor,BackgroundPolicy
    config=SolverConfig(method='DOP853',rtol=1e-7,atol=1e-9,max_step=.0001)
    initial=[r.initialize(config) for r in (original,candidate)]
    states=[];results=[];reports=[];points=[]
    plan=IndexedRecordingPlan(RecordingPlan())
    for runner,state in zip((original,candidate),initial):
        monitor=_WindowMonitor(runner,state,BackgroundPolicy());observed=[]
        def observe(t,y,bundle):monitor.observe(t,y,bundle);observed.append(t)
        end,result=runner.advance(state,channels,.0004,sample_times=np.array([0.,.0002,.0004]),
            recording_plan=plan,state_observer=observe)
        states.append(end);results.append(result);reports.append(monitor.report(.0004));points.append(observed)
    for a,b in zip(states[0].ordered_state_arrays,states[1].ordered_state_arrays):
        np.testing.assert_array_equal(a.view(np.uint64),b.view(np.uint64))
    for field in ('time_seconds','grid_origin_seconds','sample_index','component_tick_indices','spike_states',
                  'delay_states','link_delay_states','input_position'):
        assert canonical_hash(getattr(states[0],field))==canonical_hash(getattr(states[1],field))
    assert canonical_hash(reports[0])==canonical_hash(reports[1]) and points[0]==points[1]
    for key,q in results[0].outputs_with_units.items():
        other=results[1].outputs_with_units[key]
        assert (q.unit,q.meaning,q.evidence_tag)==(other.unit,other.meaning,other.evidence_tag)
        np.testing.assert_array_equal(q.values.view(np.uint64),other.values.view(np.uint64))
    metadata=[dict(result.metadata) for result in results]
    for item in metadata:item.pop('compute_backend',None)
    assert canonical_hash(metadata[0])==canonical_hash(metadata[1])
    with pytest.raises(ContractError,match='identity|model_hash'):
        candidate.restore(original.checkpoint(states[0]))
    restored=candidate.restore(candidate.checkpoint(states[1]))
    for a,b in zip(states[1].ordered_state_arrays,restored.ordered_state_arrays):
        np.testing.assert_array_equal(a.view(np.uint64),b.view(np.uint64))


def test_unreviewed_native_source_rejected(gpu,monkeypatch):
    import retina_sim.gpu_joint_validation as module
    monkeypatch.setattr(module,'PINS',dict(module.PINS,**{'gpu_native.py':'f'*64}))
    with pytest.raises(ContractError,match='source change'):
        CPUFiniteJointPartition(gpu[3].partition)


def test_explicit_backend_factory_keeps_GPU_and_private_identity(gpu):
    from retina_sim.execution_backends import make_runner,CUDA_CPU_FINITE
    runner=make_runner(gpu[1],CUDA_CPU_FINITE)
    assert isinstance(runner,CPUFiniteJointCudaRunner)
    assert runner.model.model_hash==gpu[4].model.model_hash
    assert runner.model.compute_metadata['integration_backend']['backend']==CUDA_CPU_FINITE
    assert runner.partition.native.nc==gpu[3].partition.native.nc
