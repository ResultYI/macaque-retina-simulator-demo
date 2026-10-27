"""Opt-in prototype keeping the complete continuous joint RHS on CUDA.

The original FP64 DOP853, native operators, causal ticks, clocks, histories,
event generators and observations are retained. This is an experimental
execution identity with no accepted cross-device numerical error budget yet.
"""
from copy import copy
from pathlib import Path
import hashlib
import numpy as np
from .contracts import ContractError, canonical_hash
from .gpu_hybrid import CudaHybridRunner, NativeInnerPartition
from .gpu_native import CudaNative
from .gpu_joint_validation import PINS
from .gpu_device_inner import DeviceInnerPlan

BACKEND = 'cuda_device_joint_dop853_float64_prototype_v1'


class DeviceJointPartition:
    def __init__(self, reference):
        if type(reference) is not NativeInnerPartition or type(reference.native) is not CudaNative:
            raise ContractError('Device joint prototype requires the original GPU v1 partition')
        for name, expected in PINS.items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
                raise ContractError('Device joint lowering requires review after source change: '+name)
        self.__dict__.update(reference.__dict__)
        self.inner = DeviceInnerPlan(reference)
        self.expected_channels = frozenset([self.native_external.channel]+[e.channel for e, _ in self.external])
        self.statistics = {'RHS_calls': 0, 'H2D_bytes': 0, 'D2H_bytes': 0,
            'native_release_D2H_bytes': 0, 'downstream_state_D2H_bytes': 0,
            'downstream_derivative_H2D_bytes': 0}

    def __call__(self, t, y, bundle):
        torch = self.torch
        if (y.dtype != torch.float64 or y.device.type != 'cuda' or
                y.shape != (self.offset+self.inner_size,)):
            raise ContractError('Device joint complete state layout differs')
        if set(bundle.channels) != self.expected_channels:
            raise ContractError('CUDA joint external channel set differs')
        q = bundle.channels[self.native_external.channel]
        self.native_port.validate(q)
        light = torch.as_tensor(np.asarray(q.values).copy(), dtype=torch.float64, device='cuda')
        outer, release = self.native(y[:self.offset], light)
        values = []
        transferred = q.values.nbytes
        for external, port in self.external:
            value = bundle.channels[external.channel]
            port.validate(value)
            values.append(torch.as_tensor(np.asarray(value.values).copy(), dtype=torch.float64, device='cuda'))
            transferred += value.values.nbytes
        values.append(release)
        inner = self.inner(y[self.offset:], values)
        result = torch.cat((outer, inner))
        if not torch.isfinite(result).all().item():
            raise ContractError('Nonfinite CUDA joint derivative')
        self.statistics['RHS_calls'] += 1
        self.statistics['H2D_bytes'] += transferred
        return result


class DeviceJointCudaRunner(CudaHybridRunner):
    @classmethod
    def from_runner(cls, reference):
        if type(reference) is not CudaHybridRunner:
            raise ContractError('Device joint candidate starts from the original GPU v1 runner')
        self = cls.__new__(cls)
        self.__dict__.update(reference.__dict__)
        self.model = copy(reference.model)
        self.integration_metrics = dict(reference.integration_metrics)
        self.partition = DeviceJointPartition(reference.partition)
        identity = dict(reference.model.compute_metadata['integration_backend'], backend=BACKEND,
            implementation_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('gpu_device_joint.py', 'gpu_device_inner.py')},
            source_pins=PINS, inner_generated_program_sha256=self.partition.inner.source_sha256,
            native_operators='original resident A/B reused', downstream='complete CUDA continuous RHS and CSR',
            input_state_clock_integrator_RNG_history_recording='unchanged',
            cross_stage_dynamic_cache=False, numerical_equivalence_budget='not yet accepted',
            bitwise_cpu_parity=False)
        self.model.model_hash = canonical_hash({'original_GPU_v1': reference.model.model_hash, 'device_joint_prototype': identity})
        self.model.compute_metadata = dict(reference.model.compute_metadata, integration_backend=identity)
        return self

    def __init__(self, compiled, *, compute_backend='numba'):
        original = CudaHybridRunner(compiled, compute_backend=compute_backend)
        self.__dict__.update(type(self).from_runner(original).__dict__)
