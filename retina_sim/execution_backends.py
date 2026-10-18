"""Explicit runner selection and bounded execution-only parity verification."""
import json,time
import numpy as np
from .contracts import ContractError

CPU='scipy_cpu_v1'
CUDA='cuda_native_joint_dop853_float64_v1'
CUDA_COMPACT='cuda_compact_native_joint_dop853_float64_v2'
CUDA_CPU_FINITE='cuda_native_joint_cpu_finite_validation_float64_v1'
CUDA_BACKENDS=(CUDA,CUDA_COMPACT,CUDA_CPU_FINITE)
BACKENDS=(CPU,)+CUDA_BACKENDS


def make_runner(compiled,backend=CPU):
    if backend==CPU:
        from .hybrid import HybridRunner
        return HybridRunner(compiled)
    if backend==CUDA:
        from .gpu_hybrid import CudaHybridRunner
        return CudaHybridRunner(compiled)
    if backend==CUDA_COMPACT:
        from .gpu_compact import CompactCudaHybridRunner
        return CompactCudaHybridRunner(compiled)
    if backend==CUDA_CPU_FINITE:
        from .gpu_joint_validation import CPUFiniteJointCudaRunner
        return CPUFiniteJointCudaRunner(compiled)
    raise ContractError('Unknown explicit numerical integration backend: '+str(backend))


def verify_cuda_execution(runner,solver,light):
    """One real full-network100us comparison, without checkpoint migration.

    Both states are independently initialized from the same unchanged assembly.
    This is numerical backend verification only. Actual required background
    preparation starts afresh after this comparison and is never bypassed.
    """
    from .hybrid import HybridRunner
    from .background import constant_background_channels
    from .recording import RecordingPlan
    cpu=HybridRunner(runner.compiled,compute_backend='numba')
    left=cpu.initialize(solver);right=runner.initialize(solver)
    np.testing.assert_array_equal(cpu._flat(left),runner._flat(right))
    channels=constant_background_channels(runner.compiled,light,.0001,interpolation=solver.input_interpolation)
    plan=RecordingPlan(states=(),cell_ids=frozenset(),max_bytes=1024**2)
    start=time.perf_counter();left,_=cpu.advance(left,channels,.0001,sample_times=np.array([0.,.0001]),recording_plan=plan)
    cpu_seconds=time.perf_counter()-start
    start=time.perf_counter();right,_=runner.advance(right,channels,.0001,sample_times=np.array([0.,.0001]),recording_plan=plan)
    gpu_seconds=time.perf_counter()-start
    a=cpu._flat(left);b=runner._flat(right)
    if a.shape!=b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ContractError('Actual CUDA comparison has nonfinite/different state layout')
    delta=np.abs(a-b);scale=solver.atol+solver.rtol*np.maximum(np.abs(a),np.abs(b))
    maximum=float(np.max(delta/scale))
    if maximum>.01:raise ContractError('Actual CUDA endpoint differs by more than1% of original solver scale')
    if (left.time_seconds!=right.time_seconds or left.grid_origin_seconds!=right.grid_origin_seconds or
            left.sample_index!=right.sample_index or left.component_tick_indices!=right.component_tick_indices):
        raise ContractError('Actual CUDA comparison changed event/observation clocks')
    for cid,aux in left.spike_states.items():
        other=right.spike_states[cid]
        for key in ('rng_state','history','last_bin_counts'):
            if json.dumps(aux[key],sort_keys=True)!=json.dumps(other[key],sort_keys=True):
                raise ContractError('Actual CUDA comparison changed RNG/spike counts/history')
    result={'status':'FULL_REGISTERED_STATE_CUDA_CPU_100US_COMPARISON_PASS',
        'state_scalars':len(a),'components':len(runner.compiled.components),'duration_seconds':.0001,
        'maximum_absolute_state_difference':float(delta.max()),'maximum_difference_over_original_solver_scale':maximum,
        'comparison_limit_fraction_of_original_solver_scale':.01,'cpu_seconds':cpu_seconds,'cuda_seconds':gpu_seconds,
        'event_clocks_equal':True,'rng_counts_history_equal':True,'checkpoint_migration':False,
        'original_solver':{'method':solver.method,'rtol':solver.rtol,'atol':solver.atol,'max_step':solver.max_step},
        'numerical_execution':runner.model.compute_metadata,'physiological_validation':False,
        'background_preparation':'NOT_PERFORMED_BY_THIS_CHECK; actual required policy must still run from fresh initial state'}
    # Diagnostic verification contributes no simulated time to the real run.
    runner.integration_metrics={'solver_calls':0,'nfev':0,'njev':0,'nlu':0,'integration_wall_seconds':0.,
        'minimum_interval_seconds':None,'maximum_nfev_per_interval':0}
    return result
