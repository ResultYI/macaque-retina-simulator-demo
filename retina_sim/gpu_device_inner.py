"""Experimental FP64 device lowering of an existing, pinned inner RHS plan.

Only immutable instructions/operators are cached. Every dynamic expression is
evaluated on every call. CPU causal ticks, events and RNG remain with the runner.
CUDA reductions/transcendentals may round differently from the Numba reference;
this prototype does not declare an accepted numerical equivalence budget.
"""
import hashlib
import numpy as np
from .contracts import ContractError
from .compiled_kernel import (ADD, SUB, MUL, DIV, POW, SIGMOID, LOG, EXP,
    MINIMUM, MAXIMUM, CSR, TAKE, SCATTER, ZERO, FINITE, NONNEG, POSITIVE,
    POSITIVE_MASK)


class DeviceInnerPlan:
    def __init__(self, reference, device='cuda'):
        import torch
        if str(device) != 'cuda' or not torch.cuda.is_available():
            raise ContractError('Device inner prototype requires explicit CUDA')
        self.torch = torch
        self.nstate = reference.inner_size
        self.constants = []
        self.matrices = []
        self.indices = []
        # The original v1 plan consumes the complete release vector; the compact
        # plan remaps those exact columns. Its input region must match that plan.
        release_size = len(reference.used) if hasattr(reference, 'used') else reference.native.nc
        self.external_sizes = [len(p.cell_ids) for _, p in reference.external] + [release_size]
        self.statistics = {'calls': 0, 'dynamic_cross_stage_cache': False}
        regions = [(0, self.nstate, 'y')]
        cursor = self.nstate
        for i, n in enumerate(self.external_sizes):
            regions.append((cursor, n, f'external[{i}]'))
            cursor += n
        # All writes are to fresh Plan allocations except explicit gap scatters.
        writes = [(int(d), int(n)) for op, d, n, *_ in reference.program
                  if op not in (FINITE, NONNEG, POSITIVE, SCATTER)]
        lines = ['def execute(y: Tensor, external: List[Tensor], constants: List[Tensor], matrices: List[Tensor], indices: List[Tensor]) -> Tuple[Tensor, Tensor]:',
                 '    invalid = torch.logical_not(torch.isfinite(y)).any()']
        for i in range(len(self.external_sizes)):
            lines.append(f'    invalid = invalid | torch.logical_not(torch.isfinite(external[{i}])).any()')

        def resolve(start, n):
            start, n = int(start), int(n)
            for offset, size, expr in reversed(regions):
                if offset <= start and start+n <= offset+size:
                    return expr if start == offset and n == size else f'{expr}.narrow(0, {start-offset}, {n})'
            if start < cursor or start < 0 or n < 1 or start+n > len(reference.work):
                raise ContractError('Device plan references an invalid/unwritten input region')
            if any(max(start, d) < min(start+n, d+size) for d, size in writes):
                raise ContractError('Device plan reads an intermediate before definition')
            values = reference.work[start:start+n]
            if not np.all(np.isfinite(values)):
                raise ContractError('Device plan has nonfinite immutable constants')
            index = len(self.constants)
            self.constants.append(torch.as_tensor(values.copy(), dtype=torch.float64, device=device))
            expr = f'constants[{index}]'
            regions.append((start, n, expr))
            return expr

        def matrix(indptr, ix, values, shape):
            # Execution-only canonical copy; no graph operator is rebuilt or pruned.
            from scipy.sparse import csr_matrix
            stored = csr_matrix((values.copy(), ix.copy(), indptr.copy()), shape=shape)
            stored.sort_indices()
            if not stored.has_canonical_format:
                raise ContractError('Device CSR requires distinct columns in each row')
            self.matrices.append(torch.sparse_csr_tensor(
                torch.as_tensor(stored.indptr.astype(np.int64), device=device),
                torch.as_tensor(stored.indices.astype(np.int64), device=device),
                torch.as_tensor(stored.data, dtype=torch.float64, device=device),
                size=shape, dtype=torch.float64, device=device, check_invariants=True))
            return f'matrices[{len(self.matrices)-1}]'

        binary = {ADD: '+', SUB: '-', MUL: '*', DIV: '/', POW: '**'}
        for k, row in enumerate(reference.program):
            op, dest, n, a, an, b, bn = map(int, row)
            if op in (FINITE, NONNEG, POSITIVE):
                av = resolve(a, n)
                lines.append(f'    invalid = invalid | torch.logical_not(torch.isfinite({av})).any()')
                if op != FINITE:
                    lines.append(f'    invalid = invalid | ({av} {"<" if op == NONNEG else "<="} 0.0).any()')
                continue
            name = f'v{k}'
            if op == CSR:
                ip = reference.pointers[b:b+n+1]
                lo, hi = int(ip[0]), int(ip[-1])
                mat = matrix(ip-lo, reference.indices[bn+lo:bn+hi], reference.weights[bn+lo:bn+hi], (n, an))
                expression = f'torch.mv({mat}, {resolve(a, an)})'
            elif op == TAKE:
                ix = reference.indices[b:b+n]
                if len(ix) != n or np.any(ix < 0) or np.any(ix >= an):
                    raise ContractError('Device gather indices outside original source')
                self.indices.append(torch.as_tensor(ix.copy(), dtype=torch.int64, device=device))
                expression = f'torch.index_select({resolve(a, an)}, 0, indices[{len(self.indices)-1}])'
            elif op == SCATTER:
                targets = [(offset, size, value) for offset, size, value in regions if offset == dest]
                if not targets or an != n:
                    raise ContractError('Invalid device gap scatter target/source')
                _, size, old = targets[-1]
                ix = reference.indices[b:b+n]
                if len(ix) != n or np.any(ix < 0) or np.any(ix >= size):
                    raise ContractError('Device scatter index outside original target')
                # Separate columns retain every edge, including repeated targets.
                # Stable within-row order follows the original scatter edge order.
                order = np.argsort(ix, kind='stable')
                ip = np.concatenate(([0], np.cumsum(np.bincount(ix, minlength=size))))
                mat = matrix(ip, order, np.ones(n, dtype=np.float64), (size, n))
                expression = f'{old} + torch.mv({mat}, {resolve(a, n)})'
                n = size
            elif op == ZERO:
                expression = f'torch.zeros_like(y.narrow(0, 0, {n}))'
            else:
                av = resolve(a, an)
                if op in binary:
                    expression = f'{av} {binary[op]} {resolve(b, bn)}'
                elif op == SIGMOID:
                    expression = f'1.0 / (1.0 + torch.exp(-{av}))'
                elif op in (LOG, EXP):
                    expression = f'torch.{"log" if op == LOG else "exp"}({av})'
                elif op in (MINIMUM, MAXIMUM):
                    expression = f'torch.{"minimum" if op == MINIMUM else "maximum"}({av}, {resolve(b, bn)})'
                elif op == POSITIVE_MASK:
                    expression = f'({av} > 0.0).to(dtype=y.dtype)'
                else:
                    raise ContractError(f'Unsupported device inner opcode {op}')
            lines.append(f'    {name} = {expression}')
            regions.append((dest, n, name))
        parts = []
        expected = 0
        for dest, source, n in reference.derivatives:
            dest, source, n = int(dest), int(source), int(n)
            if dest != expected:
                raise ContractError('Device derivative state order differs')
            parts.append(f'torch.zeros_like(y.narrow(0, {dest}, {n}))' if source < 0 else resolve(source, n))
            expected += n
        if expected != self.nstate:
            raise ContractError('Device derivative layout incomplete')
        lines += ['    result = torch.cat([' + ', '.join(parts) + '], 0)',
                  '    invalid = invalid | torch.logical_not(torch.isfinite(result)).any()',
                  '    return result, invalid']
        self.source = '\n'.join(lines) + '\n'
        self.source_sha256 = hashlib.sha256(self.source.encode()).hexdigest()
        self.compilation = torch.jit.CompilationUnit(self.source)

    def __call__(self, y, external):
        torch = self.torch
        if y.dtype != torch.float64 or y.device.type != 'cuda' or y.shape != (self.nstate,):
            raise ContractError('Device inner state must retain its complete CUDA float64 layout')
        if len(external) != len(self.external_sizes):
            raise ContractError('Device inner external count differs')
        for value, size in zip(external, self.external_sizes):
            if value.dtype != torch.float64 or value.device.type != 'cuda' or value.shape != (size,):
                raise ContractError('Device inner external shape/dtype/device differs')
        # Keep the literal operation graph; speculative TorchScript profiling
        # and fusion are a separate optimization, outside this first prototype.
        with torch.jit.optimized_execution(False):
            result, invalid = self.compilation.execute(y, external, self.constants, self.matrices, self.indices)
        if invalid.item():
            raise ContractError('Device inner finite/nonnegative/positive domain violated; no clipping')
        self.statistics['calls'] += 1
        return result
