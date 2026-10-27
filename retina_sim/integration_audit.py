"""Validation-only, interval-scoped accepted-step and dense-output capture.

No production integrator file, tolerance, requested sample, or endpoint selection
is changed. Additional dense stages call the interval's existing read-only RHS;
they never call a scheduler, publish a history, or draw an event RNG. Captures are
chunk-owned: inspect them and call ``clear`` before advancing the next chunk.
This context patches module call sites temporarily and is deliberately exclusive
and single-threaded. Its overhead is QA cost, never a production timing claim.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import ast
import hashlib
import inspect
import threading
import time

import numpy as np

from .contracts import ContractError

_PATCH_LOCK = threading.Lock()


def _array(value):
    result = np.asarray(value, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _readonly_view(value):
    """Own an array header, sharing immutable dense coefficients without copy.

    The view's write flag is private: the original SciPy array and solver are
    untouched. DOP853 creates a fresh F array for each dense interpolant.
    """
    original = np.asarray(value)
    if original.dtype != np.float64:
        raise ContractError('Audit dense coefficients must retain original FP64 dtype')
    result = original.view()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AcceptedStep:
    t_start: float
    t_end: float
    y_start: np.ndarray
    y_end: np.ndarray
    F7: np.ndarray
    dense_source: str
    qa_extra_rhs_calls: int = 0

    def evaluate(self, t):
        """Original DOP853 polynomial, including its own endpoint rounding."""
        t = float(t)
        if not np.isfinite(t) or not self.t_start <= t <= self.t_end:
            raise ContractError('Audit dense evaluation must remain inside its accepted step')
        x = (t-self.t_start)/(self.t_end-self.t_start)
        value = np.zeros_like(self.y_start)
        for j, row in enumerate(reversed(self.F7)):
            value += row
            value *= x if j % 2 == 0 else 1-x
        value += self.y_start
        return value


@dataclass
class AuditInterval:
    arm: str
    index: int
    backend: str
    start: float
    stop: float
    original_t_eval: np.ndarray | None
    accepted_steps: list = field(default_factory=list)
    # Optional installed SciPy interpolants, retained only on the CPU QA arm.
    interpolants: list = field(default_factory=list)
    original_nfev: int = 0
    original_steps: int = 0
    qa_extra_rhs_calls: int = 0
    qa_extra_rhs_wall_seconds: float = 0.
    capture_wall_seconds: float = 0.
    total_audited_wall_seconds: float = 0.
    status: str = 'RUNNING'
    streamed: bool = False
    streamed_steps_consumed: int = 0
    stream_consumer_wall_seconds: float = 0.
    _stream_end_times: list = field(default_factory=list, repr=False)

    @property
    def accepted_time_stamps(self):
        ends = self._stream_end_times if self.streamed else [step.t_end for step in self.accepted_steps]
        return (self.start, *ends)

    @property
    def ts(self):
        return _array(self.accepted_time_stamps)

    @property
    def accepted_endpoints(self):
        if self.streamed:
            raise ContractError('Streamed accepted states were released; use accepted_time_stamps and the step consumer')
        if not self.accepted_steps:
            return ()
        return ((self.start, self.accepted_steps[0].y_start),)+tuple(
            (step.t_end, step.y_end) for step in self.accepted_steps)

    def evaluate(self, t):
        if self.streamed:
            raise ContractError('Streamed dense coefficients were released; evaluate the step inside its consumer')
        t = float(t)
        if not np.isfinite(t) or not self.start <= t <= self.stop:
            raise ContractError('Audit cannot interpolate across an interval/tick/breakpoint boundary')
        if not self.accepted_steps:
            raise ContractError('Audit interval has no accepted dense steps')
        # At an interior step boundary choose the preceding polynomial, like
        # the installed default SciPy OdeSolution. True accepted endpoints are
        # separately exposed above; evaluate never substitutes their values.
        ends = [step.t_end for step in self.accepted_steps]
        index = min(int(np.searchsorted(ends, t, side='left')), len(ends)-1)
        return self.accepted_steps[index].evaluate(t)


def _gpu_clone(original, hook):
    """Instrument only two verified locations in a private AST function copy."""
    source = inspect.getsource(original)
    tree = ast.parse(source)
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or function.name != 'integrate':
        raise ContractError('Audit requires the original GPU integrate function')
    function.name = '_audited_integrate'
    found = dict(dense=0, accepted=0)

    class Inject(ast.NodeTransformer):
        def visit_If(self, node):
            node = self.generic_visit(node)
            if ast.dump(node.test, include_attributes=False) == ast.dump(
                    ast.parse('position < end', mode='eval').body, include_attributes=False):
                found['dense'] += 1
                flag = ast.parse('__audit_dense_used = position < end').body[0]
                return [ast.copy_location(flag, node), node]
            return node

        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 't' and isinstance(node.value, ast.Name)
                    and node.value.id == 't_new'):
                found['accepted'] += 1
                call = ast.parse('_audit_accept(locals())').body[0]
                return [ast.copy_location(call, node), node]
            return node

    tree = Inject().visit(tree)
    if found != dict(dense=1, accepted=1):
        raise ContractError('GPU integrate structure changed; audit hook needs explicit review')
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__, _audit_accept=hook)
    exec(compile(tree, '<validation-only GPU DOP853 accepted-step audit>', 'exec'), namespace)
    return namespace['_audited_integrate'], hashlib.sha256(source.encode()).hexdigest()


class AuditSession:
    """Temporarily audit existing HybridRunner/CudaHybridRunner call sites.

    Example::

        with AuditSession() as audit:
            with audit.arm('original'):
                state, result = runner.advance(...)
            for interval in audit.intervals['original']:
                value = audit.evaluate('original', interval.index, t)
            audit.clear('original')

    ``evaluate`` always uses the original dense polynomial; ``accepted_endpoints``
    contains actual accepted y values, including any different endpoint rounding.
    Original nfev excludes QA-only stages. Caller metrics' wall time necessarily
    includes this validation instrumentation; it must not be used as a speed test.
    With ``retain_cpu_interpolants=False``, CPU steps keep read-only views of
    SciPy's fresh F7 arrays and no interpolant objects. Start/end states remain
    independent copies. GPU capture and the default CPU capture are unchanged.

    ``cpu_stream_consumers={arm: callback}`` instead streams each accepted CPU
    step to ``callback(interval, step)`` and retains only its timestamps/counts.
    This mode requires the original endpoint-only call (t_eval=None). It calls
    solve_ivp with dense_output=False, so no OdeSolution retains all F7 arrays.
    The consumer must finish every required query using step.evaluate before
    returning, and must not retain the arrays if bounded storage is required.
    """
    def __init__(self, active_label=None, *, retain_cpu_interpolants=True, cpu_stream_consumers=None):
        if type(retain_cpu_interpolants) is not bool:
            raise ContractError('CPU interpolant retention must be an explicit boolean')
        self.active_label = active_label
        self.retain_cpu_interpolants = retain_cpu_interpolants
        self.cpu_stream_consumers = dict(cpu_stream_consumers or {})
        if any(not isinstance(label, str) or not label or not callable(callback)
                for label, callback in self.cpu_stream_consumers.items()):
            raise ContractError('CPU streaming consumers require nonempty arm labels and callbacks')
        self.intervals = {}
        self._entered = False
        self._current_interval = None
        self.source_hashes = {}

    def __enter__(self):
        if self._entered or not _PATCH_LOCK.acquire(blocking=False):
            raise ContractError('Integration audit contexts cannot overlap')
        try:
            from . import gpu_hybrid, hybrid
            from .gpu_dop853 import integrate
            self._owner_thread = threading.get_ident()
            self._gpu_module, self._cpu_module = gpu_hybrid, hybrid
            self._old_gpu, self._old_cpu = gpu_hybrid.integrate, hybrid.solve_ivp
            if self._old_gpu is not integrate:
                raise ContractError('GPU integration call site is already replaced')
            self._gpu_function, self.source_hashes['GPU_integrate'] = _gpu_clone(integrate, self._capture_gpu)
            self.source_hashes['CPU_solve_ivp'] = hashlib.sha256(inspect.getsource(self._old_cpu).encode()).hexdigest()
            gpu_hybrid.integrate, hybrid.solve_ivp = self._gpu_integrate, self._cpu_solve_ivp
            self._entered = True
            return self
        except BaseException:
            _PATCH_LOCK.release()
            raise

    def __exit__(self, exc_type, exc, tb):
        try:
            self._gpu_module.integrate = self._old_gpu
            self._cpu_module.solve_ivp = self._old_cpu
        finally:
            self._entered = False
            self._current_interval = None
            _PATCH_LOCK.release()

    @contextmanager
    def arm(self, label):
        if not isinstance(label, str) or not label:
            raise ContractError('Audit arm requires a nonempty label')
        previous = self.active_label
        self.active_label = label
        try:
            yield self
        finally:
            self.active_label = previous

    @property
    def accepted_endpoints(self):
        return {arm: tuple((interval.index, t, y) for interval in intervals
            for t, y in interval.accepted_endpoints) for arm, intervals in self.intervals.items()}

    def evaluate(self, arm, interval_index, t):
        return self.intervals[arm][interval_index].evaluate(t)

    def clear(self, arm=None):
        if self._current_interval is not None:
            raise ContractError('Cannot clear an integrating audit interval')
        if arm is None:
            self.intervals.clear()
        else:
            self.intervals.pop(arm, None)

    @contextmanager
    def _interval(self, backend, start, stop, t_eval):
        if (not self._entered or threading.get_ident() != self._owner_thread
                or not isinstance(self.active_label, str) or not self.active_label):
            raise ContractError('Active integration audit arm and owning thread required')
        if self._current_interval is not None:
            raise ContractError('Nested interval integration is not supported by audit')
        rows = self.intervals.setdefault(self.active_label, [])
        interval = AuditInterval(self.active_label, len(rows), backend, float(start), float(stop),
            None if t_eval is None else _array(t_eval))
        rows.append(interval)
        self._current_interval = interval
        tick = time.perf_counter()
        try:
            yield interval
            interval.status = 'COMPLETE'
        except BaseException:
            interval.status = 'FAILED'
            raise
        finally:
            interval.total_audited_wall_seconds = time.perf_counter()-tick
            self._current_interval = None

    def _capture_gpu(self, local):
        import torch
        from scipy.integrate._ivp.rk import DOP853
        interval = self._current_interval
        tick = time.perf_counter()
        extra = 0
        if local['__audit_dense_used']:
            F = local['F']
            dense_source = 'original_requested_dense'
        else:
            # Never write the production K buffer or increment its rhs counter.
            K = local['K'].clone()
            stage_tick = time.perf_counter()
            for stage in range(13, 16):
                delta = torch.mv(K[:stage].T, local['AE'][stage-13, :stage])*local['h']
                value = local['fun'](float(local['t']+DOP853.C_EXTRA[stage-13]*local['h']), local['y']+delta)
                if value.shape != local['y'].shape or value.dtype != torch.float64:
                    raise ContractError('QA dense RHS must retain original complete FP64 layout')
                K[stage].copy_(value)
                extra += 1
            # Finish these QA kernels before assigning their wall cost.
            torch.cuda.synchronize(local['y'].device)
            interval.qa_extra_rhs_wall_seconds += time.perf_counter()-stage_tick
            delta = local['y_new']-local['y']
            F = torch.empty((7, local['n']), dtype=torch.float64, device=local['device'])
            F[0].copy_(delta)
            F[1].copy_(local['h']*K[0]-delta)
            F[2].copy_(2*delta-local['h']*(local['f_new']+K[0]))
            F[3:].copy_(local['h']*torch.mm(local['DD'], K))
            dense_source = 'QA_extra_dense_raw_RHS'
        step = AcceptedStep(float(local['t']), float(local['t_new']),
            _array(local['y'].cpu().numpy()), _array(local['y_new'].cpu().numpy()),
            _array(F.cpu().numpy()), dense_source, extra)
        interval.accepted_steps.append(step)
        interval.qa_extra_rhs_calls += extra
        interval.capture_wall_seconds += time.perf_counter()-tick

    def _gpu_integrate(self, fun, t0, y0, t_bound, **kwargs):
        if self.active_label in self.cpu_stream_consumers:
            raise ContractError('CPU streaming audit cannot be selected for CUDA integration')
        with self._interval('original_CUDA_DOP853', t0, t_bound, kwargs.get('t_eval')) as interval:
            result = self._gpu_function(fun, t0, y0, t_bound, **kwargs)
            interval.original_nfev, interval.original_steps = result['nfev'], result['steps']
            if len(interval.accepted_steps) != result['steps']:
                raise ContractError('Audit missed an accepted CUDA step')
            return result

    def _cpu_solve_ivp(self, fun, t_span, y0, **kwargs):
        from scipy.integrate._ivp.rk import DOP853
        if kwargs.get('method', 'RK45') != 'DOP853' or kwargs.get('dense_output', False):
            raise ContractError('CPU audit supports the original DOP853 dense_output=False call only')
        consumer = self.cpu_stream_consumers.get(self.active_label)
        if consumer is not None and kwargs.get('t_eval') is not None:
            raise ContractError('CPU streaming audit requires original endpoint_only t_eval=None')
        with self._interval('original_CPU_DOP853', *t_span, kwargs.get('t_eval')) as interval:
            interval.streamed = consumer is not None
            requested = interval.original_t_eval

            class AuditedDOP853(DOP853):
                def _step_impl(solver):
                    if consumer is None:
                        return super()._step_impl()
                    start = solver.t
                    accepted, message = super()._step_impl()
                    if accepted:
                        # OdeSolver.step updates t_old only AFTER this method
                        # returns. The accepted RK state/y_old/h_previous/f are
                        # already final here. Supply the current start only for
                        # the extra dense calculation, then restore t_old even
                        # when the consumer or an extra RHS raises.
                        previous_t_old = solver.t_old
                        solver.t_old = start
                        try:
                            solver._dense_output_impl()
                        finally:
                            solver.t_old = previous_t_old
                    return accepted, message

                def _dense_output_impl(solver):
                    # Original solve_ivp computes dense stages only when this
                    # accepted step contains an original requested sample.
                    needed = requested is not None and np.any(
                        ((requested > solver.t_old) | ((not interval.accepted_steps)
                         & (requested == solver.t_old))) & (requested <= solver.t))
                    before = solver.nfev
                    tick = time.perf_counter()
                    interpolant = super()._dense_output_impl()
                    elapsed = time.perf_counter()-tick
                    extra = 0 if needed else solver.nfev-before
                    if not needed:
                        solver.nfev = before
                        interval.qa_extra_rhs_wall_seconds += elapsed
                    capture_tick = time.perf_counter()
                    if self.retain_cpu_interpolants and consumer is None:
                        interval.interpolants.append(interpolant)
                        coefficients = _array(interpolant.F)
                    else:
                        # solve_ivp retains the original interpolant until its
                        # result leaves HybridRunner._integrate. Sharing F7
                        # prevents a second full coefficient allocation then;
                        # afterward this view alone keeps that F array alive.
                        coefficients = _readonly_view(interpolant.F)
                    step = AcceptedStep(float(solver.t_old), float(solver.t),
                        _array(solver.y_old), _array(solver.y), coefficients,
                        'original_requested_dense' if needed else 'QA_extra_dense_raw_RHS', extra)
                    interval.qa_extra_rhs_calls += extra
                    interval.capture_wall_seconds += time.perf_counter()-capture_tick
                    if consumer is None:
                        interval.accepted_steps.append(step)
                    else:
                        interval._stream_end_times.append(step.t_end)
                        interval.original_steps += 1
                        interval.original_nfev = solver.nfev
                        consumer_tick = time.perf_counter()
                        try:
                            consumer(interval, step)
                            interval.streamed_steps_consumed += 1
                        finally:
                            interval.stream_consumer_wall_seconds += time.perf_counter()-consumer_tick
                    return interpolant

            options = dict(kwargs, method=AuditedDOP853, dense_output=consumer is None)
            result = self._old_cpu(fun, t_span, y0, **options)
            interval.original_nfev = result.nfev
            if consumer is None:
                interval.original_steps = len(interval.accepted_steps)
            return result
