"""Opt-in current compact device-inner candidate; no numerical waiver.

Reuse the existing generated DeviceInnerPlan, original native CUDA equations
and original host event/DOP853 loop. Every dynamic expression runs every stage.
CPU author/history/RNG/delay updates enter through each original interval upload.
"""
from copy import copy
from pathlib import Path
import hashlib
import numpy as np
from .contracts import ContractError,canonical_hash
from .compiled_kernel import IMPLEMENTATIONS
from .gpu_compact import CompactCudaHybridRunner,CompactNativeInnerPartition
from .gpu_device_inner import DeviceInnerPlan

BACKEND='cuda_compact_device_inner_dop853_float64_candidate_v1'
PINS={
    'gpu_compact.py':'ef0dd65664888d1babe3dde8e8df046e1fcb81ccad23226b370396f08588b01b',
    'gpu_native.py':'16a7081587c5a1b7a77410d9374c8547141062e6e5b8d43ec65df336e4c774da',
    'gpu_hybrid.py':'f4447c356b540e630b43d9a1d3eb33a4e74c950581cdb40f21f4838dee19c82d',
    'gpu_dop853.py':'94d03925a324074338ff24a17e4d0e8f7cc3bf6aec88daf6527ea23d722b5f7e',
    'compiled_kernel.py':'e7e9554ab7ba6824c8c3e10e69f76e36061966a41982321f87472143503e25e5',
}


class CompactDeviceInnerPartition(CompactNativeInnerPartition):
    def __init__(self,reference,*,fused_inner=False):
        if type(reference) is not CompactNativeInnerPartition:
            raise ContractError('Device-inner candidate requires the current explicit compact reference')
        for name,expected in dict(IMPLEMENTATIONS,**PINS).items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=expected:
                raise ContractError('Device-inner source review required after change: '+name)
        self.__dict__.update(reference.__dict__)
        self.inner=DeviceInnerPlan(reference)
        self.fusion_plan=None
        if fused_inner:
            from .gpu_fused_inner import FusedDeviceInnerPlan
            self.inner=FusedDeviceInnerPlan(self.inner,reference.program)
            self.fusion_plan=self.inner
        self.statistics=dict(RHS_calls=0,release_D2H_bytes=0,inner_state_D2H_bytes=0,inner_derivative_H2D_bytes=0,
            external_H2D_bytes=0,light_uploads=0,light_H2D_bytes=0,identical_light_hits=0,
            CPU_inner_calls=0,scalar_validity_syncs_retained=True,dynamic_cross_stage_cache=False)

    def inputs_and_native(self,y,bundle):
        torch=self.torch
        if y.dtype!=torch.float64 or y.device.type!='cuda' or y.shape!=(self.offset+self.inner_size,):
            raise ContractError('Device-inner complete FP64 state layout differs')
        expected={self.native_external.channel}|{external.channel for external,_ in self.external}
        if set(bundle.channels)!=expected:raise ContractError('CUDA joint external channel set differs')
        q=bundle.channels[self.native_external.channel];self.native_port.validate(q)
        outer,release=self.native(y[:self.offset],self._light(q))
        values=[]
        for external,port in self.external:
            value=bundle.channels[external.channel];port.validate(value)
            # Actual current input values, including changes after author ticks;
            # no timestamp/object-identity cache is applied to these channels.
            snapshot=np.asarray(value.values).copy()
            values.append(torch.as_tensor(snapshot,dtype=torch.float64,device='cuda'))
            self.statistics['external_H2D_bytes']+=snapshot.nbytes
        values.append(torch.index_select(release,0,self.gather))
        return outer,values

    def __call__(self,t,y,bundle):
        if self.fusion_plan is not None:self.fusion_plan.validate_constants()
        outer,values=self.inputs_and_native(y,bundle)
        inner=self.inner(y[self.offset:],values)
        result=self.torch.cat((outer,inner))
        if not self.torch.isfinite(result).all().item():raise ContractError('Nonfinite CUDA joint derivative')
        self.statistics['RHS_calls']+=1
        return result


class CompactDeviceInnerCudaRunner(CompactCudaHybridRunner):
    @classmethod
    def from_runner(cls,reference,*,graph_inner=False,fused_inner=False,numerical_contract=None):
        if type(reference) is not CompactCudaHybridRunner:
            raise ContractError('Device-inner candidate starts from the unchanged compact runner')
        self=cls.__new__(cls);self.__dict__.update(reference.__dict__)
        self.model=copy(reference.model);self.integration_metrics=dict(reference.integration_metrics)
        self.partition=CompactDeviceInnerPartition(reference.partition,fused_inner=fused_inner)
        if graph_inner:
            from .gpu_device_variants import CudaGraphInner
            self.partition.inner=CudaGraphInner(self.partition.inner)
        identity=dict(reference.model.compute_metadata['integration_backend'],backend=BACKEND,
            source_pins=dict(IMPLEMENTATIONS,**PINS),
            implementation_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('gpu_compact_device.py','gpu_device_inner.py')},
            inner_generated_program_sha256=self.partition.inner.source_sha256,
            dynamic_cross_stage_cache=False,CPU_author_history_clock_RNG_delays='Original host event loop and per-interval state upload',
            downstream='Existing complete current inner Plan lowered to CUDA FP64; includes continuous BC/AC/rod membrane/contact/RBC/AII',
            numerical_gate='Strict single-RHS first; no new tolerance accepted',default_enabled=False)
        identity.update(graph_inner=graph_inner,graph_scope='Pure inner execution only; original adaptive solver and CPU events outside capture')
        if fused_inner:
            identity.update(fused_inner=True,fusion_coverage=self.partition.fusion_plan.coverage,
                fusion_generator_sha256=hashlib.sha256(Path(__file__).with_name('gpu_fused_inner.py').read_bytes()).hexdigest())
            identity['backend']+='_fused'
        if graph_inner:
            identity['CUDA_graph_helper_sha256']=hashlib.sha256(Path(__file__).with_name('gpu_device_variants.py').read_bytes()).hexdigest()
            identity['backend']+='_graph'
        if numerical_contract is not None:
            from .compact_device_equivalence import CompactDeviceEquivalence
            if not isinstance(numerical_contract,CompactDeviceEquivalence):raise ContractError('Explicit frozen compact device contract required')
            identity.update(numerical_gate=numerical_contract.identity,contract_hash=numerical_contract.contract_hash,
                backend=identity['backend']+'_error_equivalence')
            self.numerical_contract=numerical_contract
        self.model.model_hash=canonical_hash(dict(compact_reference=reference.model.model_hash,device_inner_candidate=identity))
        self.model.compute_metadata=dict(reference.model.compute_metadata,integration_backend=identity)
        return self

    def __init__(self,compiled,*,compute_backend='numba',graph_inner=False,fused_inner=False,numerical_contract=None):
        reference=CompactCudaHybridRunner(compiled,compute_backend=compute_backend)
        self.__dict__.update(type(self).from_runner(reference,graph_inner=graph_inner,fused_inner=fused_inner,numerical_contract=numerical_contract).__dict__)
