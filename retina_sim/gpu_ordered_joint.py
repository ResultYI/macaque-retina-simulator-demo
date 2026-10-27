"""Opt-in CUDA joint RHS with the original ordered inner reductions.

This execution candidate retains the original native CUDA RHS and the original
adaptive solver and event loop. Inner CSR products and gap additions follow the
CPU plan's stored edge order, with separate FP64 multiply and add operations.
It has a separate checkpoint identity; numerical acceptance is measured, never
inferred from this lowering. Historical SpMM/fusion prototypes are not selected.
"""
from copy import copy
from pathlib import Path
import hashlib
import re

import numpy as np

from .compiled_kernel import CSR, SCATTER
from .contracts import ContractError, canonical_hash
from .gpu_device_inner import DeviceInnerPlan
from .gpu_device_joint import DeviceJointPartition
from .gpu_device_variants import BatchedValidationInner, CudaGraphInner
from .gpu_hybrid import CudaHybridRunner
from .gpu_joint_validation import PINS

BACKEND = 'cuda_ordered_joint_dop853_float64_candidate_v1'


class OrderedInnerPlan(DeviceInnerPlan):
    def __init__(self, reference):
        super().__init__(reference)
        torch = self.torch
        lines = self.source.splitlines()
        replacements = {}
        self.ordered_operations = []

        def tensor_index(values):
            index = len(self.indices)
            self.indices.append(torch.as_tensor(values, dtype=torch.int64, device='cuda'))
            return f'indices[{index}]'

        def tensor_constant(values):
            index = len(self.constants)
            self.constants.append(torch.as_tensor(values, dtype=torch.float64, device='cuda'))
            return f'constants[{index}]'

        def ordered(name, source, rows, indices, weights=None, initial=None):
            counts = np.asarray([len(row) for row in indices], dtype=np.int64)
            ranks = int(counts.max(initial=0))
            result = [f'    {name} = ' + (initial if initial is not None else
                      f'torch.zeros([{rows}], dtype=y.dtype, device=y.device)')]
            if not ranks:
                return result
            padded = np.zeros((ranks, rows), dtype=np.int64)
            mask = np.zeros((ranks, rows), dtype=np.float64)
            factors = np.zeros((ranks, rows), dtype=np.float64)
            for row, positions in enumerate(indices):
                padded[:len(positions), row] = positions
                mask[:len(positions), row] = 1.0
                if weights is not None:
                    factors[:len(positions), row] = weights[row]
            gather = tensor_index(padded.ravel())
            expression = f'torch.index_select({source}, 0, {gather})'
            if weights is not None:
                expression += ' * ' + tensor_constant(factors.ravel())
            result.append(f'    {name}_edges = ({expression}).reshape([{ranks}, {rows}])')
            for rank in range(ranks):
                addition = f'{name} + {name}_edges[{rank}]'
                if np.all(mask[rank]):
                    result.append(f'    {name} = {addition}')
                else:
                    valid = tensor_constant(mask[rank])
                    result.append(f'    {name} = torch.where({valid} > 0.0, {addition}, {name})')
            return result

        for k, instruction in enumerate(reference.program):
            op, dest, n, a, an, b, bn = map(int, instruction)
            if op not in (CSR, SCATTER):
                continue
            name = f'v{k}'
            line = next(line for line in lines if line.startswith(f'    {name} = '))
            if op == CSR:
                match = re.fullmatch(r'    v\d+ = torch\.mv\(matrices\[\d+\], (.+)\)', line)
                if match is None:
                    raise ContractError('Original CSR source expression changed')
                pointers = reference.pointers[b:b+n+1]
                columns, weights = [], []
                for lo, hi in zip(pointers[:-1], pointers[1:]):
                    columns.append(reference.indices[bn+int(lo):bn+int(hi)].copy())
                    weights.append(reference.weights[bn+int(lo):bn+int(hi)].copy())
                replacements[line] = ordered(name, match[1], n, columns, weights)
                self.ordered_operations.append(dict(instruction=k, kind='CSR', rows=n,
                    edges=sum(map(len, columns)), max_row_edges=max(map(len, columns), default=0)))
            else:
                match = re.fullmatch(r'    v\d+ = (.+) \+ torch\.mv\(matrices\[(\d+)\], (.+)\)', line)
                if match is None:
                    raise ContractError('Original gap scatter source expression changed')
                rows = self.matrices[int(match[2])].shape[0]
                target_indices = reference.indices[b:b+n]
                positions = [np.flatnonzero(target_indices == row) for row in range(rows)]
                replacements[line] = ordered(name, match[3], rows, positions, initial=match[1])
                self.ordered_operations.append(dict(instruction=k, kind='SCATTER', rows=rows,
                    edges=n, max_row_edges=max(map(len, positions), default=0)))
        self.source = '\n'.join(part for line in lines for part in replacements.get(line, [line]))+'\n'
        if 'torch.mv(' in self.source:
            raise ContractError('An inner sparse reduction escaped ordered lowering')
        # No sparse matrix is used by the new executable. Native A/B remain intact.
        self.matrices = []
        self.source_sha256 = hashlib.sha256(self.source.encode()).hexdigest()
        self.compilation = torch.jit.CompilationUnit(self.source)


class OrderedJointPartition(DeviceJointPartition):
    def __init__(self, reference, *, graph=True):
        super().__init__(reference)
        ordered = OrderedInnerPlan(reference)
        batched = BatchedValidationInner(ordered)
        self.inner = CudaGraphInner(batched) if graph else batched
        self.ordered_operations = ordered.ordered_operations
        self.validation_regions = batched.validation_regions


class OrderedJointCudaRunner(CudaHybridRunner):
    @classmethod
    def from_runner(cls, reference, *, graph=True, spmm_native=False, fused_native=False):
        if spmm_native or fused_native:
            raise ContractError('SpMM/fused native candidates failed strict parity and are not selectable')
        if type(reference) is not CudaHybridRunner:
            raise ContractError('Ordered joint candidate requires the original GPU runner')
        self = cls.__new__(cls)
        self.__dict__.update(reference.__dict__)
        self.model = copy(reference.model)
        self.integration_metrics = dict(reference.integration_metrics)
        self.partition = OrderedJointPartition(reference.partition, graph=graph)
        identity = dict(reference.model.compute_metadata['integration_backend'], backend=BACKEND,
            implementation_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('gpu_ordered_joint.py', 'gpu_device_joint.py', 'gpu_device_inner.py', 'gpu_device_variants.py')},
            source_pins=PINS, inner_generated_program_sha256=self.partition.inner.source_sha256,
            native_operators='original CUDA native A/B and pointwise operations',
            inner_reductions='original stored row/edge order; separate FP64 multiply/add',
            graph_inner=graph, validation_regions=self.partition.validation_regions,
            spmm_native=False, fused_native=False, cross_stage_dynamic_cache=False,
            numerical_equivalence_budget='strict parity required; acceptance evidence is separate')
        self.model.model_hash = canonical_hash({'original_GPU_v1': reference.model.model_hash, 'ordered_joint': identity})
        self.model.compute_metadata = dict(reference.model.compute_metadata, integration_backend=identity)
        return self

    def __init__(self, compiled, *, compute_backend='numba', graph=True):
        original = CudaHybridRunner(compiled, compute_backend=compute_backend)
        self.__dict__.update(type(self).from_runner(original, graph=graph).__dict__)
