"""Cache invariant BC kernel validation; leave equations, solver and events intact."""
import hashlib,inspect,math,textwrap
from types import MethodType
import numpy as np
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.spatial_group_model import BC
from tools.cell6_compiled_inner_v1 import CompiledCell6Runner


def dominant_nonpositive(values):
    """Same first-maximum/NaN behavior as values[argmax(abs(values))]<=0."""
    if len(values)==0:raise ValueError('attempt to get argmax of an empty sequence')
    best=0;magnitude=abs(float(values[0]))
    if not math.isnan(magnitude):
        for i in range(1,len(values)):
            current=abs(float(values[i]))
            if math.isnan(current):best=i;break
            if current>magnitude:best=i;magnitude=current
    return bool(float(values[best])<=0.)


def install_guard(model):
    original=model._commit_ticks.__func__;source=textwrap.dedent(inspect.getsource(original))
    assignments="kernel = np.asarray(self.p[pop]['kernel'], dtype=np.float64)"
    reduction='if kernel[np.argmax(np.abs(kernel))] <= 0:'
    if source.count(assignments)!=1 or source.count(reduction)!=1:raise ContractError('Exact BC invariant-validation source changed')
    kernels={pop:np.array(model.p[pop]['kernel'],dtype=np.float64,copy=True) for pop in BC}
    decisions={pop:dominant_nonpositive(values) for pop,values in kernels.items()}
    for values in kernels.values():values.flags.writeable=False
    model._fixed_BC_kernels=kernels;model._fixed_BC_kernel_nonpositive=decisions
    replacement=source.replace(assignments,'kernel = self._fixed_BC_kernels[pop]').replace(reduction,'if self._fixed_BC_kernel_nonpositive[pop]:')
    namespace=dict(original.__globals__);exec(compile(replacement,__file__+':constant_BC_validation','exec'),namespace)
    model._commit_ticks=MethodType(namespace['_commit_ticks'],model)
    return dict(schema='frozen-BC-kernel-validation-cache-v2',kernel_hash=canonical_hash(kernels),
        source_method_sha256=hashlib.sha256(source.encode()).hexdigest(),replacement_sha256=hashlib.sha256(replacement.encode()).hexdigest(),
        cached_kernels={k:v.tolist() for k,v in kernels.items()},dominant_nonpositive=decisions,
        original_kernel_math_and_event_method_preserved=True,only_invariant_allocation_and_validation_cached=True,
        frozen_parameter_requirement=True,numerical_backend_identity_unchanged=True,
        cache_execution_code_identity_must_record_this_module=True,native_corruption_root_cause_proven=False)


class GuardedCompiledCell6Runner(CompiledCell6Runner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.kernel_guard_evidence=install_guard(self.inner)
        self.compiled_build['constant_BC_kernel_validation_guard']=self.kernel_guard_evidence

    def checkpoint_payload(self,**kwargs):
        result=super().checkpoint_payload(**kwargs)
        result['constant_BC_kernel_validation_guard']=self.kernel_guard_evidence
        return result
