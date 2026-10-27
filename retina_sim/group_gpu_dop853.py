"""Full-layout FP64 DOP853 transport using the existing group GPU equations.

This adapter changes storage and execution only. It retains every held row in
initial-step selection, all error norms, and all accepted dense polynomials.
Creation is explicit; importing this module never initializes CUDA.
"""
from bisect import bisect_left
from hashlib import sha256
from pathlib import Path
from threading import Lock
from time import perf_counter
import math

import numpy as np

from .contracts import ContractError, canonical_hash
from .group_gpu_rhs import GroupGpuRHS
from .group_inner_dynamics import InnerAdvance
from .resident_dop853 import Dop853Workspace, integrate_device


class GroupGpuDop853:
    """Reusable storage; no step controller or derivative crosses a commit."""

    def __init__(self, inner, compiled, torch, device='cuda:0', guard=None, *, gpu_rhs=None):
        self.inner, self.compiled, self.torch, self.guard = inner, compiled, torch, guard
        self.device = torch.device(device)
        self._lock, self._closed = Lock(), False
        if (self.device.type != 'cuda' or compiled.model is not inner
                or compiled.nstate != inner.nstate or inner.nstate < 1):
            raise ContractError('Group GPU DOP853 requires the full bound inner layout on CUDA')
        self._check_guard()
        self._owns_rhs = gpu_rhs is None
        with torch.cuda.device(self.device):
            selected_device = torch.device('cuda', torch.cuda.current_device())
            if gpu_rhs is not None and (gpu_rhs.compiled is not compiled or gpu_rhs.torch is not torch
                                        or gpu_rhs.device != selected_device):
                raise ContractError('Borrowed GPU RHS must have the same model and device binding')
            self.rhs = GroupGpuRHS(compiled, torch) if gpu_rhs is None else gpu_rhs
            try:
                self.device = self.rhs.device
                self._state = torch.empty((inner.nstate,), dtype=torch.float64, device=self.device)
                self.workspace = Dop853Workspace(self._state, 1e-9)
            except Exception:
                if self._owns_rhs:
                    self.rhs.close()
                raise
        import scipy
        from scipy.integrate._ivp.rk import DOP853
        from . import resident_dop853
        implementation = {Path(path).name: sha256(Path(path).read_bytes()).hexdigest()
                          for path in (__file__, resident_dop853.__file__)}
        coefficients = canonical_hash({name: getattr(DOP853, name).tolist() for name in
                                       ('A', 'B', 'C', 'E3', 'E5', 'A_EXTRA', 'C_EXTRA', 'D')})
        self.identity = canonical_hash(dict(
            implementation_sha256=implementation, scipy_version=scipy.__version__,
            coefficients_sha256=coefficients,
            schema='retipath-group-gpu-dop853-1', rhs=self.rhs.identity,
            method='DOP853', rtol=1e-6, atol=1e-9, max_step_seconds=1e-4,
            error_norm_dimension=inner.nstate, initial_step='original_full_state',
            controller_restart='every_original_event_free_interval',
            dense='original_seven_coefficient_DOP853_every_accepted_step'))

    def _check_guard(self):
        if self.guard is not None:
            result = self.guard.require() if hasattr(self.guard, 'require') else self.guard()
            if result is False:
                raise ContractError('Group GPU DOP853 execution guard stopped the attempt')

    def advance(self, state, t0, t1, *, rtol, atol):
        """Accept one event-free interval, preserving caller ownership on failure."""
        if (not math.isfinite(t0) or not math.isfinite(t1) or not t0 < t1
                or t1-t0 > 1.00000001e-4 or not np.isscalar(rtol)
                or not np.isscalar(atol) or rtol != 1e-6 or atol != 1e-9):
            raise ContractError('Group GPU DOP853 requires original tolerances and an interval at most 100us')
        if not self._lock.acquire(blocking=False):
            raise ContractError('Group GPU DOP853 workspace is already in use')
        try:
            if self._closed:
                raise ContractError('Group GPU DOP853 is closed')
            self._check_guard()
            started = perf_counter()
            packed = self.inner.pack(state)
            metrics = dict(
                method='full_group_resident_gpu_DOP853', precision='float64',
                original_full_dimension=self.inner.nstate,
                error_norm_dimension=self.inner.nstate,
                held_in_initial_step_and_error_norm=True,
                controller_restarted=True, max_step_seconds=1e-4,
                nfev=0, rhs_wall_seconds=0., transfer_wall_seconds=0.,
                full_state_H2D=0, full_state_D2H=0,
                H2D_bytes=0, D2H_bytes=0, state_D2H_bytes=0, dense_queries=0,
                blocking_state_transfer_calls=0,
                dense_query_wall_seconds=0., full_state_D2H_per_stage=0,
                original_rhs_for_monitor=True,
                device=str(self.device), dense_export='owned_device_packets',
                uniform_error_certificate='not_established',
                kernel_launch_accounting='generated_RHS_only_torch_kernels_not_instrumented',
                timing_scope='nested_host_wall_times_do_not_add_to_inner_wall')

            def download(value):
                before = perf_counter()
                output = value.detach().cpu().numpy().copy()
                metrics['transfer_wall_seconds'] += perf_counter()-before
                metrics['full_state_D2H'] += 1
                metrics['D2H_bytes'] += output.nbytes
                metrics['state_D2H_bytes'] += output.nbytes
                metrics['blocking_state_transfer_calls'] += 1
                return output

            def derivative(t, y, output):
                self._check_guard()
                before = perf_counter()
                metrics['nfev'] += 1
                try:
                    return self.rhs.evaluate_into(t, y, output)
                finally:
                    metrics['rhs_wall_seconds'] += perf_counter()-before

            with self.torch.cuda.device(self.device):
                before = perf_counter()
                # Every committed state, including author-updated held rows, is fresh.
                self._state.copy_(self.torch.from_numpy(packed))
                metrics['transfer_wall_seconds'] += perf_counter()-before
                metrics['full_state_H2D'] = 1
                metrics['blocking_state_transfer_calls'] += 1
                metrics['H2D_bytes'] = packed.nbytes
                result = integrate_device(None, t0, self._state, t1,
                    rtol=rtol, atol=atol, max_step=1e-4, workspace=self.workspace,
                    dense_output=True, fun_into=derivative)
                final = self.inner.unpack(download(result['final']), copy=False)
                self._check_guard()
            segments = result['dense_segments']
            if (not segments or segments[0]['t_old'] != t0 or segments[-1]['t'] != t1
                    or any(a['t'] != b['t_old'] for a, b in zip(segments, segments[1:]))):
                raise ContractError('Group GPU DOP853 accepted dense coverage is incomplete')
            if metrics['nfev'] != result['nfev']:
                raise ContractError('Group GPU DOP853 RHS accounting differs')
            stops = tuple(s['t'] for s in segments)
            metrics.update({name: result[name] for name in
                            ('steps', 'attempts', 'rejected_attempts', 'dense_polynomials')})
            scalar_bytes = metrics['nfev']*4+(3+2*result['attempts'])*8+1
            metrics['D2H_bytes'] += scalar_bytes
            work = self.workspace
            storage = (self._state, work.K, work.F, *work.rhs_buffers,
                       work.A, work.B, work.E3, work.E5, work.AE, work.DD, work.at)
            metrics.update(
                scalar_D2H_bytes=scalar_bytes,
                persistent_device_bytes=((self.rhs.metrics['resident_bytes'] if self._owns_rhs else 0)+
                                         sum(v.numel()*v.element_size() for v in storage)),
                shared_rhs_device_bytes=0 if self._owns_rhs else self.rhs.metrics['resident_bytes'],
                accepted_dense_device_bytes=sum(
                    s[k].numel()*s[k].element_size() for s in segments for k in ('y_old', 'F')),
                initial_rhs_calls=2, main_stage_rhs_calls=12*result['attempts'],
                dense_stage_rhs_calls=3*result['dense_polynomials'],
                rhs_kernel_launches=metrics['nfev']*self.rhs.metrics['kernel_launches_per_rhs'],
                rhs_scalar_D2H=metrics['nfev'], solver_scalar_D2H=result['solver_scalar_D2H'],
                scalar_readback_sync_points=metrics['nfev']+result['solver_scalar_D2H'],
                rhs_output_clones=0, wall_seconds=perf_counter()-started)

            def evaluate(t, side):
                self._check_guard()
                before = perf_counter()
                # Match the reference's first inclusive segment at shared boundaries.
                segment = segments[min(bisect_left(stops, t), len(segments)-1)]
                x = (t-segment['t_old'])/(segment['t']-segment['t_old'])
                with self.torch.cuda.device(self.device):
                    value = self.torch.zeros_like(segment['y_old'])
                    for j, row in enumerate(reversed(segment['F'].unbind(0))):
                        value += row
                        value *= x if j % 2 == 0 else 1-x
                    value += segment['y_old']
                    output = self.inner.unpack(download(value), copy=False)
                    self._check_guard()
                metrics['dense_queries'] += 1
                metrics['dense_query_wall_seconds'] += perf_counter()-before
                return output

            advance = InnerAdvance(self.inner, float(t0), float(t1), final, evaluate, {},
                                   metrics, (float(t0), *map(float, stops)))
            advance.dense_segments = segments
            return advance
        finally:
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise ContractError('Cannot close a Group GPU DOP853 workspace in use')
        try:
            if not self._closed:
                if self._owns_rhs:
                    self.rhs.close()
                self._closed = True
        finally:
            self._lock.release()
