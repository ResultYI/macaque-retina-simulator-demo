"""Opt-in equivalent final joint-domain check, retaining the full GPU v1 route.

The pinned native call already checks every native derivative/release on CUDA.
Check the independently computed FP64 inner derivative on CPU before its exact
FP64 upload, instead of scanning the entire concatenated GPU derivative again.
All invalid domains still reject; no stage/state/dynamic-result cache is added.
"""
from pathlib import Path
from copy import copy
import hashlib
import numpy as np
from .contracts import ContractError,canonical_hash
from .gpu_hybrid import CudaHybridRunner,NativeInnerPartition
from .gpu_native import CudaNative

BACKEND='cuda_native_joint_cpu_finite_validation_float64_v1'
PINS={
    'gpu_native.py':'16a7081587c5a1b7a77410d9374c8547141062e6e5b8d43ec65df336e4c774da',
    'gpu_hybrid.py':'f4447c356b540e630b43d9a1d3eb33a4e74c950581cdb40f21f4838dee19c82d',
    'compiled_kernel.py':'759d64b81fd2b25123d89256b0a0d44e4e020bcb7a932bda8058d15a234891ce'}


class CPUFiniteJointPartition:
    def __init__(self,reference):
        if type(reference) is not NativeInnerPartition or type(reference.native) is not CudaNative:
            raise ContractError('Equivalent joint check requires the original verified GPU v1 partition/native')
        for name,expected in PINS.items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=expected:
                raise ContractError('Joint finite-check proof requires review after source change: '+name)
        self.__dict__.update(reference.__dict__)
        self.inner_inputs=tuple(self.external)
        self.expected_channels=frozenset([self.native_external.channel]+[e.channel for e,_ in self.inner_inputs])
        self.statistics={'RHS_calls':0,'redundant_full_GPU_checks_avoided':0,'inner_CPU_checks':0}

    def __call__(self,t,y,bundle):
        torch=self.torch
        if set(bundle.channels)!=self.expected_channels:raise ContractError('CUDA joint external channel set differs')
        q=bundle.channels[self.native_external.channel];self.native_port.validate(q)
        light=torch.as_tensor(np.asarray(q.values).copy(),dtype=torch.float64,device='cuda')
        # The unchanged pinned native call validates its input domain, then the
        # complete native derivative AND release. It retains both native syncs.
        outer,release=self.native(y[:self.offset],light)
        values=[]
        for external,port in self.inner_inputs:
            value=bundle.channels[external.channel];port.validate(value);values.append(value.values)
        values.append(release.cpu().numpy())
        inner=self.fn(y[self.offset:].cpu().numpy(),np.concatenate(values),self.work,self.program,
            self.weights,self.indices,self.pointers,self.derivatives)
        if (not isinstance(inner,np.ndarray) or inner.dtype!=np.float64 or
                inner.shape!=(self.inner_size,) or not np.all(np.isfinite(inner))):
            raise ContractError('Nonfinite or invalid CUDA joint inner derivative')
        # FP64 CPU -> FP64 CUDA conversion preserves finiteness, signed zeros
        # and every value. Native validity above + inner validity here covers
        # exactly the complete vector checked by the original final GPU scan.
        result=torch.cat((outer,torch.as_tensor(inner,dtype=torch.float64,device='cuda')))
        self.statistics['RHS_calls']+=1;self.statistics['inner_CPU_checks']+=1
        self.statistics['redundant_full_GPU_checks_avoided']+=1
        return result


class CPUFiniteJointCudaRunner(CudaHybridRunner):
    @classmethod
    def from_runner(cls,reference):
        """Sequential execution may reuse the immutable graph/device operators.

        Both runners retain private runtime metadata and metrics. Like the
        original partition, the shared scratch plan is not reentrant; callers
        must execute sequentially. No persisted state/checkpoint is migrated.
        """
        if type(reference) is not CudaHybridRunner:
            raise ContractError('Joint validation candidate starts from the original GPU v1 runner')
        self=cls.__new__(cls);self.__dict__.update(reference.__dict__)
        self.model=copy(reference.model)
        self.integration_metrics=dict(reference.integration_metrics)
        self.partition=CPUFiniteJointPartition(reference.partition)
        identity=dict(reference.model.compute_metadata['integration_backend'],backend=BACKEND,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_pins=PINS,
            final_joint_check='Unchanged native CUDA derivative/release check plus complete FP64 inner CPU check before exact FP64 upload',
            native_syncs_and_full_release_transfer='unchanged; compact bridge disabled',
            input_state_clock_integrator_RNG_history_recording='unchanged',cross_stage_dynamic_cache=False)
        self.model.model_hash=canonical_hash({'original_GPU_v1':reference.model.model_hash,'equivalent_joint_validation':identity})
        self.model.compute_metadata=dict(reference.model.compute_metadata,integration_backend=identity)
        return self

    def __init__(self,compiled,*,compute_backend='numba'):
        original=CudaHybridRunner(compiled,compute_backend=compute_backend)
        self.__dict__.update(type(self).from_runner(original).__dict__)
