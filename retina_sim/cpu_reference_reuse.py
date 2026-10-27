"""QA-only whole-CPU reference reusing the original compiled RHS and lock.

The original CUDA runner already owns a complete CPU CompiledRHS, in addition
to its native-CUDA/inner-CPU stage partition. Reuse that full CPU plan instead
of constructing a second model-size scratch/operator allocation. Production
stages never call this helper. References still use original CPU SciPy DOP853.
"""
from copy import copy
import hashlib
from pathlib import Path
from .contracts import ContractError,canonical_hash
from .hybrid import HybridRunner


def shared_cpu_reference(original):
    if not isinstance(original,HybridRunner) or original.discrete is not None:
        raise ContractError('Whole CPU reference requires the original continuous HybridRunner')
    kernel=getattr(original.model,'_compiled_rhs',None)
    if kernel is None or kernel.network is not original.model:
        raise ContractError('Reuse the original full CPU compiled RHS owner, never a stage fallback')
    reference=HybridRunner.__new__(HybridRunner)
    reference.__dict__.update(original.__dict__)
    reference.__dict__.pop('partition',None)
    reference.model=copy(original.model)
    identity=dict(procedure='shared_original_whole_CPU_RHS_QA_v1',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_model_hash=original.model.model_hash,CPU_plan_hash=kernel.metadata['plan_hash'],
        integration='Original CPU SciPy DOP853; references alone use their declared tighter solver settings',
        immutable_parameters_graph_and_layout='Same original objects',mutable_scratch='Same existing per-call lock; sequential QA arms',
        candidate_stage_fallback=False,checkpoint_migration=False)
    reference.model.model_hash=canonical_hash(identity)
    reference.model.compute_metadata=dict(kernel.metadata,requested='numba',QA_reference=identity)
    reference.integration_metrics=dict(solver_calls=0,nfev=0,njev=0,nlu=0,integration_wall_seconds=0.,
        minimum_interval_seconds=None,maximum_nfev_per_interval=0)
    return reference
