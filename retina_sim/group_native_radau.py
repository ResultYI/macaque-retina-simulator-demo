"""Structured FP64 Radau IIA5 for the unchanged reduced Cone/HC equations.

Actual parent collocation stages drive every downstream SCC.  This candidate
provides bounded numerical solves and an operational error ledger, not an
interval error certificate or qualification of the biological model.
"""
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np
from scipy import sparse

from .contracts import ContractError, canonical_hash
from .cone_hc_reduced_native import STATE_NAMES, _owned, _signature
from .group_radau_math import tableau, affine_stages, batched_newton, gmres, GMRESWorkspace
from .group_scc_schedule import topological_order


METHOD = 'group_native_scc_radau_iia5_two_halfsteps_fp64_v1'
MAX_STEP = 1e-4
MIN_STEP = MAX_STEP / 64
NATIVE_PROVENANCE = 'source:cone_hc_reduced_native.ReducedNative._coefficients; not inner typed IR'
_NATIVE_FAMILIES = {
    'native_RE': (('R', 'E'), ()),
    'native_XC': (('X', 'C'), ('native_RE',)),
    'native_Vis_gi': (('Vis', 'g_i'), ('native_XC',)),
    'native_Vbar': (('Vbar',), ('native_Vis_gi',)),
    'native_HC': (('w', 'z2', 'Vh'), ('native_Vis_gi', 'native_Vbar')),
    'native_z1_aux': (('z1_aux',), ('native_Vis_gi', 'native_Vbar', 'native_HC')),
}


def _native_key(name):
    if name in ('R', 'E'):
        return ('cones', 'analytic_RE', name)
    if name in ('w', 'z2', 'Vh'):
        return ('HC', 'reduced_native', name)
    if name == 'z1_aux':
        return ('cones', 'selected_original_z1_auxiliary', name)
    return ('cones', 'reduced_native', name)


@dataclass(frozen=True)
class NativeRadauHalfStep:
    start: float
    dt: float
    stages: object
    rhs: object
    error: object


@dataclass(frozen=True)
class NativeRadauSegment:
    start: float
    stop: float
    y0: object
    f0: object
    ym: object
    fm: object
    y1: object
    f1: object
    anchor: object
    physics: object

    def __post_init__(self):
        if not math.isfinite(self.start) or not math.isfinite(self.stop) or not self.start < self.stop:
            raise ContractError('Invalid native Radau dense interval')
        object.__setattr__(self, 'start', float(self.start))
        object.__setattr__(self, 'stop', float(self.stop))
        for name in ('y0', 'f0', 'ym', 'fm', 'y1', 'f1'):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        object.__setattr__(self, '_versions', tuple(_signature(getattr(self, n))
            for n in ('y0', 'f0', 'ym', 'fm', 'y1', 'f1')))

    def _theta(self, t):
        self.anchor.check_owned()
        if self._versions != tuple(_signature(getattr(self, n))
                for n in ('y0', 'f0', 'ym', 'fm', 'y1', 'f1')):
            raise ContractError('Native Radau dense anchors were mutated')
        if not math.isfinite(float(t)) or not self.start <= t <= self.stop:
            raise ContractError('Native Radau dense query outside accepted interval')
        return (t - self.start) / (self.stop - self.start)

    def at(self, t):
        x = self._theta(t)
        if t == self.start:
            return _owned(self.y0)
        if t == self.stop:
            return _owned(self.y1)
        if t == self.start + (self.stop-self.start)/2:
            return _owned(self.ym)
        l0, lm, l1 = 2*x*x-3*x+1, 4*x*(1-x), 2*x*x-x
        h = self.stop-self.start
        return ((1+6*x)*l0*l0*self.y0 + lm*lm*self.ym + (7-6*x)*l1*l1*self.y1
                + h*(x*l0*l0*self.f0 + (x-.5)*lm*lm*self.fm + (x-1)*l1*l1*self.f1))

    __call__ = at

    def representation_derivative(self, t):
        x = self._theta(t)
        if t == self.start:
            return _owned(self.f0)
        if t == self.stop:
            return _owned(self.f1)
        if t == self.start + (self.stop-self.start)/2:
            return _owned(self.fm)
        l0, lm, l1 = 2*x*x-3*x+1, 4*x*(1-x), 2*x*x-x
        d0, dm, d1 = 4*x-3, 4-8*x, 4*x-1
        h = self.stop-self.start
        return ((6*l0*l0+2*(1+6*x)*l0*d0)*self.y0
                + 2*lm*dm*self.ym + (-6*l1*l1+2*(7-6*x)*l1*d1)*self.y1)/h + (
                (l0*l0+2*x*l0*d0)*self.f0 + (lm*lm+2*(x-.5)*lm*dm)*self.fm
                + (l1*l1+2*(x-1)*l1*d1)*self.f1)

    def rhs_at(self, t):
        if self.physics is None:
            raise ContractError('Projected native dense rows require the complete state for original RHS')
        return self.physics.rhs(self.at(t), t, self.anchor)

    def project_rows(self, indices):
        return NativeRadauSegment(self.start, self.stop,
            *(getattr(self, n)[indices] for n in ('y0', 'f0', 'ym', 'fm', 'y1', 'f1')),
            self.anchor, None)


@dataclass(frozen=True)
class PiecewiseNativeRadauDense:
    segments: tuple

    def _segment(self, t):
        # The right interval owns a shared endpoint; native continuous states agree.
        for segment in reversed(self.segments):
            if segment.start <= t <= segment.stop:
                return segment
        raise ContractError('Native Radau dense query outside accepted intervals')

    def at(self, t):
        return self._segment(t).at(t)

    __call__ = at

    def representation_derivative(self, t):
        return self._segment(t).representation_derivative(t)

    def rhs_at(self, t):
        return self._segment(t).rhs_at(t)


@dataclass(frozen=True)
class NativeRadauResult:
    y: object
    dense: PiecewiseNativeRadauDense
    accepted_times: tuple
    metrics: dict
    stage_cache: tuple
    error: object


class GroupNativeRadau:
    """GPU-capable SCC adapter; constructing the NumPy adapter never starts CUDA.

    error_weights and budgets, when supplied, are frozen original coordinate
    scales. budgets only reports an estimate-to-budget diagnostic. The embedded
    third-order difference is not a bound on the fifth-order method's error;
    cumulative candidate-reference error is qualified by the independent QA.
    """

    def __init__(self, physics, *, rtol=1e-6, atol=1e-9,
                 algebraic_rtol=1e-13, algebraic_atol=1e-13,
                 linear_rtol=1e-10, linear_atol=1e-14,
                 error_weights=None, budgets=None, temporal_fraction=.45, guard=None, native_plan=None):
        self.physics, self.p = physics, physics.p
        self.torch, self.device = physics.torch, physics.device
        self.rtol, self.atol = float(rtol), float(atol)
        self.algebraic_rtol, self.algebraic_atol = float(algebraic_rtol), float(algebraic_atol)
        self.linear_rtol, self.linear_atol = float(linear_rtol), float(linear_atol)
        tolerances = (self.rtol, self.atol, self.algebraic_rtol, self.algebraic_atol,
                      self.linear_rtol, self.linear_atol)
        if any(not math.isfinite(x) or x < 0 for x in tolerances) or self.atol <= 0 or self.algebraic_atol <= 0:
            raise ContractError('Native Radau tolerances must be finite and nonnegative with positive absolute tolerances')
        if not math.isfinite(temporal_fraction) or not 0 < temporal_fraction <= 1:
            raise ContractError('Invalid native temporal budget allocation')
        self.temporal_fraction, self.guard = float(temporal_fraction), guard
        self.statistics = {}
        self._bind_native_schedule(native_plan)
        self.error_weights = self._coordinate_scale(error_weights)
        self.budgets = self._coordinate_scale(budgets)
        self.error_ledger = None
        self.next_step = MAX_STEP
        self._window_end = None
        self._window_attempts = self._window_accepts = 0
        self._tableau = None
        self._gmres_workspaces = {stages: GMRESWorkspace(self._zeros((stages, 3, physics.nh)), restart=32)
                                  for stages in (3, 1)}
        # diag(A diag(gain) B) = (A elementwise B.T) gain; never form AB.
        diagonal = sparse.csr_matrix(physics.model.A.multiply(physics.model.B.T))
        diagonal.sort_indices()
        if self.torch is None:
            self._diagonal_plan = diagonal
        else:
            from .resident_cusparse import NativeCSRPlan
            torch = self.torch
            matrix = torch.sparse_csr_tensor(
                torch.as_tensor(diagonal.indptr.astype(np.int64), dtype=torch.int64, device=self.device),
                torch.as_tensor(diagonal.indices.astype(np.int64), dtype=torch.int64, device=self.device),
                torch.as_tensor(diagonal.data, dtype=torch.float64, device=self.device),
                size=diagonal.shape, dtype=torch.float64, device=self.device)
            self._diagonal_plan = NativeCSRPlan(matrix, physics.A.api, torch)
        self.identity = dict(method=METHOD, physics_source_version=physics.source_version,
            native_layout_identity=self.native_layout_identity, native_schedule_identity=self.schedule_identity,
            native_schedule_provenance=self.schedule_provenance,
            gmres_workspace_policy='two_persistent_fixed_shape_banks_stage_and_filtered',
            max_step=MAX_STEP, min_nonboundary_step=MIN_STEP, max_newton=8,
            gmres_restart=32, gmres_maxiter=96, preconditioner='fixed_local_stage_block_Jacobi_9x9',
            dense='quintic_Hermite_original_RHS_anchors', filtered_estimator='full_native_J0',
            original_auxiliary_ids=list(physics.aux_cell_ids),
            algebraic_rtol=self.algebraic_rtol, algebraic_atol=self.algebraic_atol,
            linear_rtol=self.linear_rtol, linear_atol=self.linear_atol,
            operational_rtol=self.rtol, operational_atol=self.atol, temporal_fraction=self.temporal_fraction,
            error_ledger='operational_absolute_filtered_estimates_not_certificate')

    def _bind_native_schedule(self, native_plan):
        """Bind source-traced native families without claiming inner typed IR."""
        counts = dict(zip(STATE_NAMES, self.physics.sizes))
        counts.update(R=self.physics.nc, E=self.physics.nc)
        registry = tuple(_native_key(n) for n in ('X', 'C', 'Vis', 'g_i', 'Vbar', 'R', 'E', 'w', 'z2', 'Vh', 'z1_aux'))
        expected = {name: (states, parents) for name, (states, parents) in _NATIVE_FAMILIES.items()
                    if sum(counts[state] for state in states)}
        records = {}
        if native_plan is None:
            if getattr(self.device, 'type', None) == 'cuda':
                raise ContractError('CUDA native Radau requires the actual source-derived native plan')
            self.native_layout_identity = None
            self.schedule_provenance = NATIVE_PROVENANCE+'; CPU fixture fallback registry'
            for name, (states, parents) in expected.items():
                records[name] = dict(states=states, keys=tuple(_native_key(n) for n in states),
                    parents=tuple(sorted(parents)), scalar_count=sum(counts[n] for n in states),
                    components=None, provenance=self.schedule_provenance)
            registry_sizes = tuple(counts[key[2]] for key in registry)
        else:
            from .group_scc_plan import NativeLayout
            if not isinstance(native_plan, NativeLayout) or native_plan.provenance != NATIVE_PROVENANCE:
                raise ContractError('Native Radau plan must use the separate source-derived native provenance')
            if not isinstance(native_plan.identity, str) or not native_plan.identity:
                raise ContractError('Native Radau plan source identity is missing')
            allowed_registry = (registry, registry[:-1]) if counts['z1_aux'] == 0 else (registry,)
            keys = tuple(native_plan.full_keys)
            if keys not in allowed_registry or set(native_plan.full_slices) != set(keys):
                raise ContractError('Native Radau source registry keys/order differ')
            numerical, analytic, position = [], [], 0
            for key in keys:
                size = counts[key[2]]
                if native_plan.full_slices[key] != slice(position, position+size):
                    raise ContractError('Native Radau source registry slices/counts differ')
                (analytic if key[1] == 'analytic_RE' else numerical).extend(range(position, position+size))
                position += size
            for values, wanted in ((native_plan.numerical_indices, numerical),
                                   (native_plan.analytic_indices, analytic)):
                values = np.asarray(values)
                if values.ndim != 1 or values.dtype.kind not in 'iu' or not np.array_equal(values, wanted):
                    raise ContractError('Native Radau numerical/analytic coverage differs')
            claimed = set()
            for block in native_plan.blocks:
                name = block.name
                if name not in expected or name in records:
                    raise ContractError('Native Radau has an unknown, duplicate or unsupported family')
                states, required_parents = expected[name]
                wanted_keys = tuple(_native_key(n) for n in states)
                parents = tuple(block.parent_names)
                if (tuple(block.state_keys) != wanted_keys or len(set(parents)) != len(parents)
                        or set(parents) != set(required_parents)):
                    raise ContractError('Native Radau family keys/parents or allowed coupling differ')
                size = sum(counts[n] for n in states)
                components = np.asarray(block.component_sizes)
                if (isinstance(block.scalar_count, bool) or not isinstance(block.scalar_count, (int, np.integer))
                        or block.scalar_count != size or isinstance(block.component_count, bool)
                        or not isinstance(block.component_count, (int, np.integer))
                        or components.ndim != 1 or components.dtype.kind not in 'iu'
                        or block.component_count != len(components) or not len(components)
                        or np.any(components <= 0) or sum(int(n) for n in components) != size):
                    raise ContractError('Native Radau family scalar/component coverage differs')
                if name == 'native_HC':
                    if not np.array_equal(components, self._native_hc_components()):
                        raise ContractError('Native Radau HC SCC coupling differs from the source operators')
                else:
                    width = 2 if name in ('native_XC', 'native_Vis_gi') else 1
                    if not np.array_equal(components, np.full(size//width, width)):
                        raise ContractError('Native Radau local family SCC coupling differs')
                if block.provenance != NATIVE_PROVENANCE or claimed.intersection(wanted_keys):
                    raise ContractError('Native Radau family provenance/unique ownership differs')
                claimed.update(wanted_keys)
                records[name] = dict(states=states, keys=wanted_keys, parents=tuple(sorted(parents)),
                    scalar_count=size, components=tuple(int(n) for n in components), provenance=block.provenance)
            required_keys = {_native_key(n) for states, _ in expected.values() for n in states}
            if set(records) != set(expected) or claimed != required_keys:
                raise ContractError('Native Radau family coverage is incomplete')
            self.native_layout_identity = native_plan.identity
            self.schedule_provenance = native_plan.provenance
            registry, registry_sizes = keys, tuple(counts[key[2]] for key in keys)
        edges = tuple((parent, name) for name, record in records.items() for parent in record['parents'])
        self.execution_order = topological_order(tuple(sorted(records)), edges)
        self._family_records = MappingProxyType({name: MappingProxyType(records[name])
                                                  for name in self.execution_order})
        self.schedule_identity = canonical_hash(dict(schema='source_native_family_schedule_1',
            physics=self.physics.source_version, native_layout=self.native_layout_identity,
            provenance=self.schedule_provenance, registry=registry, sizes=registry_sizes,
            families=tuple((name, records[name]) for name in self.execution_order), order=self.execution_order))

    def _native_hc_components(self):
        # Validate the source descriptor with w->z2->h->release->w support,
        # preserving the release vertices instead of forming a dense A@B.
        from scipy.sparse.csgraph import connected_components
        nc, nh = self.physics.nc, self.physics.nh
        A, B = self.physics.model.A.astype(bool), self.physics.model.B.astype(bool)
        eye = sparse.eye(nh, dtype=bool, format='csr')
        hh = sparse.csr_matrix((nh, nh), dtype=bool)
        cc = sparse.csr_matrix((nc, nc), dtype=bool)
        hc = sparse.csr_matrix((nh, nc), dtype=bool)
        ch = sparse.csr_matrix((nc, nh), dtype=bool)
        graph = sparse.bmat([[hh, hh, hh, A], [eye, hh, hh, hc],
                            [hh, eye, hh, hc], [ch, ch, B, cc]], format='csr')
        _, labels = connected_components(graph, directed=True, connection='strong')
        return np.unique(labels[:3*nh], return_counts=True)[1]

    @property
    def gmres_workspaces(self):
        return MappingProxyType(self._gmres_workspaces)

    def workspace_metrics(self):
        return dict(count=len(self._gmres_workspaces),
            owned_bytes=sum(w.owned_bytes for w in self._gmres_workspaces.values()),
            device_bytes=sum(w.device_bytes for w in self._gmres_workspaces.values()),
            host_bytes=sum(w.host_bytes for w in self._gmres_workspaces.values()))

    def _workspace_metric_fields(self):
        return {'gmres_workspace_'+key: value for key, value in self.workspace_metrics().items()}

    def _count(self, name, amount=1):
        self.statistics[name] = self.statistics.get(name, 0) + amount

    def _check_guard(self):
        if self.guard is not None:
            status = self.guard()
            if status is not None and not status:
                raise ContractError('Native Radau external execution guard stopped work')

    def _scalar(self, value):
        if self.torch is not None:
            self._count('host_scalar_reads')
            return value.item()
        return float(value)

    def _coordinate_scale(self, value):
        if value is None:
            return None
        value = self.physics._array(value)
        if tuple(value.shape) != (self.physics.nstate,):
            raise ContractError('Native Radau frozen scale layout differs')
        finite = np.isfinite(value).all() if self.torch is None else self.torch.isfinite(value).all()
        if not bool(self._scalar(finite & (value > 0).all())):
            raise ContractError('Native Radau frozen scales must be finite and positive')
        return _owned(value)

    def _zeros(self, shape):
        return np.zeros(shape, dtype=np.float64) if self.torch is None else self.torch.zeros(
            shape, dtype=self.torch.float64, device=self.device)

    def _eye(self, n):
        return np.eye(n, dtype=np.float64) if self.torch is None else self.torch.eye(
            n, dtype=self.torch.float64, device=self.device)

    def _stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis) if self.torch is None else self.torch.stack(tuple(arrays), dim=axis)

    def _cat(self, arrays, axis=0):
        return np.concatenate(arrays, axis=axis) if self.torch is None else self.torch.cat(tuple(arrays), dim=axis)

    def _transpose(self, value, axes):
        return value.transpose(axes) if self.torch is None else value.permute(*axes)

    def _einsum(self, equation, *values):
        return np.einsum(equation, *values) if self.torch is None else self.torch.einsum(equation, *values)

    def _abs(self, value):
        return np.abs(value) if self.torch is None else self.torch.abs(value)

    def _finite(self, value):
        return np.isfinite(value) if self.torch is None else self.torch.isfinite(value)

    def _sigmoid(self, value):
        if self.torch is not None:
            return self.torch.sigmoid(value)
        from scipy.special import expit
        return expit(value)

    def _solve(self, matrix, rhs):
        self._check_guard()
        self._count('local_direct_solves')
        if not bool(self._scalar(self._finite(matrix).all() & self._finite(rhs).all())):
            raise ContractError('Native Radau direct solve has nonfinite inputs')
        try:
            if self.torch is None:
                result = np.linalg.solve(matrix, rhs[..., None])[..., 0]
            else:
                result, info = self.torch.linalg.solve_ex(matrix, rhs.unsqueeze(-1), check_errors=False)
                if not bool(self._scalar((info == 0).all())):
                    raise ContractError('Native Radau direct linear solve is singular')
                result = result.squeeze(-1)
        except (np.linalg.LinAlgError, RuntimeError) as exc:
            raise ContractError('Native Radau direct linear solve failed') from exc
        if not bool(self._scalar(self._finite(result).all())):
            raise ContractError('Native Radau direct solve returned nonfinite values')
        return result

    def _factor(self, matrix):
        self._count('preconditioner_factorizations')
        if not bool(self._scalar(self._finite(matrix).all())):
            raise ContractError('Native Radau preconditioner has nonfinite coefficients')
        if self.torch is not None:
            lu, pivots, info = self.torch.linalg.lu_factor_ex(matrix, check_errors=False)
            if not bool(self._scalar((info == 0).all())):
                raise ContractError('Native Radau block preconditioner is singular')
            return lambda rhs: self.torch.linalg.lu_solve(lu, pivots, rhs.unsqueeze(-1)).squeeze(-1)
        # SciPy MKL LU aborts after Torch loads a second Windows OpenMP runtime.
        # One NumPy solve builds the same fixed block-Jacobi map for every RHS.
        try:
            inverse = np.linalg.solve(matrix, np.broadcast_to(np.eye(matrix.shape[-1]), matrix.shape))
        except np.linalg.LinAlgError as exc:
            raise ContractError('Native Radau block preconditioner is singular') from exc
        if not bool(self._scalar(self._finite(inverse).all())):
            raise ContractError('Native Radau block preconditioner inverse is nonfinite')
        return lambda rhs: np.einsum('nij,nj->ni', inverse, rhs)

    def _mv(self, matrix, vector):
        self._check_guard()
        self._count('sparse_matvecs')
        self._count('A_matvecs' if matrix is self.physics.A else
                    'B_matvecs' if matrix is self.physics.B else 'preconditioner_diagonal_matvecs')
        return self.physics._mv(matrix, vector)

    def _validate(self, y, anchor, t):
        self._count('provider_domain_checks')
        if self.torch is not None:
            self._count('host_scalar_reads')
        return self.physics._validate(y, anchor, t)

    def _require_finite(self, *values):
        self._count('provider_finite_checks')
        if self.torch is not None:
            self._count('host_scalar_reads')
        return self.physics._require_finite(*values)

    def _rhs(self, y, t, anchor):
        self._check_guard()
        self._count('fresh_full_rhs_evaluations')
        self._count('sparse_matvecs', 3)
        self._count('A_matvecs', 2)
        self._count('B_matvecs')
        self._count('provider_domain_checks')
        self._count('provider_finite_checks')
        if self.torch is not None:
            self._count('host_scalar_reads', 2)
        return self.physics.rhs(y, t, anchor)

    def _parameters(self, like):
        if self._tableau is None:
            self._tableau = tableau(like)
        return self._tableau

    def _mix(self, matrix, stages):
        return (matrix @ stages.reshape(stages.shape[0], -1)).reshape(stages.shape)

    def _local(self, y0, dt, rates_jacobian, domain):
        n, width = y0.shape
        a = self._parameters(y0)[0]
        initial = self._stack((y0, y0, y0), axis=1).reshape(n, 3*width)

        def residual(flat):
            stages = flat.reshape(n, 3, width)
            rates, _ = rates_jacobian(stages)
            return (stages-y0[:, None, :]-dt*self._einsum('ij,njk->nik', a, rates)).reshape(n, -1)

        def jacobian(flat):
            _, jac = rates_jacobian(flat.reshape(n, 3, width))
            return self._eye(3*width)[None, :, :] - dt*self._einsum('ij,njab->niajb', a, jac).reshape(
                n, 3*width, 3*width)

        result, info = batched_newton(initial, residual, jacobian,
            atol=self.algebraic_atol, rtol=self.algebraic_rtol, maxiter=8,
            domain=lambda flat: domain(flat.reshape(n, 3, width)),
            guard=self._check_guard, counters=self.statistics)
        if not info['converged']:
            raise ContractError('Native local Radau Newton failed: '+str(info))
        values = result.reshape(n, 3, width)
        rates, _ = rates_jacobian(values)
        return self._transpose(values, (1, 2, 0)), self._transpose(rates, (1, 2, 0))

    def _xc_rates(self, values, E):
        p = self.p
        x, c = values[..., 0], values[..., 1]
        beta = p['c_beta'] + p['k_beta']*E
        base = p['a_C']*c
        alpha = 1/(1+base**p['n_C'])
        rates = self._stack((1000*(alpha-beta*x), 1000*(x-c)/p['tau_C']), axis=-1)
        jac = self._zeros(values.shape+(2,))
        jac[..., 0, 0] = -1000*beta
        jac[..., 0, 1] = -1000*p['n_C']*p['a_C']*base**(p['n_C']-1)*alpha**2
        jac[..., 1, 0] = 1000/p['tau_C']
        jac[..., 1, 1] = -1000/p['tau_C']
        return rates, jac

    def _vg_rates(self, values, X):
        p = self.p
        v, g = values[..., 0], values[..., 1]
        rates = self._stack((1000*(X/g-v)/p['tau_m'],
            1000*(p['a_is']*v**p['gamma']-g)/p['tau_is']), axis=-1)
        jac = self._zeros(values.shape+(2,))
        jac[..., 0, 0] = -1000/p['tau_m']
        jac[..., 0, 1] = -1000*X/(p['tau_m']*g*g)
        jac[..., 1, 0] = 1000*p['a_is']*p['gamma']*v**(p['gamma']-1)/p['tau_is']
        jac[..., 1, 1] = -1000/p['tau_is']
        return rates, jac

    def _hc_context(self, V, u):
        p = self.p
        pooled = self._stack(tuple(self._mv(self.physics.A, row) for row in u))
        if not bool(self._scalar(self._finite(pooled).all() & (pooled > 0).all())):
            raise ContractError('Native HC pooled adaptation outside positive domain')
        power = (pooled/p['V_I'])**p['mu']
        return V, (u/p['V_I'])**p['mu'], 1000/(p['tau_2']*power), 1000/(p['tau_h']*power), pooled

    def _hc_rates(self, values, context):
        V, d, d2, dh, _ = context
        p, physics = self.p, self.physics
        bh = self._stack(tuple(self._mv(physics.B, row) for row in values[:, 2]))
        q = self._sigmoid((V-bh-p['V_k'])/p['V_n'])
        release = q/d
        pooled = self._stack(tuple(self._mv(physics.A, row) for row in release))
        a = 1000/p['tau_1']
        rates = self._stack((a*(p['g_t']*pooled-values[:, 0]),
            d2*(values[:, 0]-values[:, 1]), dh*(values[:, 1]-values[:, 2])), axis=1)
        gain = q*(1-q)/(p['V_n']*d)
        return rates, gain, release

    def _hc_jv(self, values, gain, d2, dh):
        a, gt = 1000/self.p['tau_1'], self.p['g_t']
        feedback = self._stack(tuple(self._mv(self.physics.A,
            gain[i]*self._mv(self.physics.B, values[i, 2])) for i in range(values.shape[0])))
        return self._stack((-a*(values[:, 0]+gt*feedback),
            d2*(values[:, 0]-values[:, 1]), dh*(values[:, 1]-values[:, 2])), axis=1)

    def _hc_linear_solver(self, gain, d2, dh, coupling):
        stages, nh = d2.shape
        a, gt = 1000/self.p['tau_1'], self.p['g_t']
        kdiag = gt*self._stack(tuple(self._mv(self._diagonal_plan, row) for row in gain))
        local = self._zeros((nh, stages, 3, 3))
        local[..., 0, 0] = -a
        local[..., 0, 2] = -a*self._transpose(kdiag, (1, 0))
        local[..., 1, 0] = self._transpose(d2, (1, 0))
        local[..., 1, 1] = -self._transpose(d2, (1, 0))
        local[..., 2, 1] = self._transpose(dh, (1, 0))
        local[..., 2, 2] = -self._transpose(dh, (1, 0))
        matrix = self._eye(3*stages)[None, :, :] - self._einsum(
            'ij,njab->niajb', coupling, local).reshape(nh, 3*stages, 3*stages)
        solve_blocks = self._factor(matrix)

        def precondition(rhs):
            self._count('preconditioner_applications')
            packed = self._transpose(rhs, (2, 0, 1)).reshape(nh, 3*stages)
            return self._transpose(solve_blocks(packed).reshape(nh, stages, 3), (1, 2, 0))

        def action(value):
            self._count('hc_jacobian_actions')
            return value-self._mix(coupling, self._hc_jv(value, gain, d2, dh))

        def solve(rhs):
            result, info = gmres(action, rhs, precondition,
                rtol=self.linear_rtol, atol=self.linear_atol, restart=32, maxiter=96,
                guard=self._check_guard, counters=self.statistics,
                workspace=self._gmres_workspaces[stages])
            if not info['converged']:
                raise ContractError('Native HC Radau GMRES failed: '+str(info))
            return result
        return solve

    def _hc_stages(self, y0, dt, context):
        a = self._parameters(y0)[0]
        current = self._stack((y0, y0, y0))
        scale = self.algebraic_atol+self.algebraic_rtol*self._abs(current)
        for iteration in range(9):
            self._check_guard()
            rates, gain, release = self._hc_rates(current, context)
            residual = current-y0[None, :, :]-dt*self._mix(a, rates)
            norm = self._scalar((self._abs(residual)/scale).max())
            self._count('hc_residual_evaluations')
            if not math.isfinite(norm):
                raise ContractError('Native HC Newton produced a nonfinite residual')
            if norm <= 1:
                return current, rates, release
            if iteration == 8:
                raise ContractError('Native HC Radau exceeded eight Newton updates')
            solve = self._hc_linear_solver(gain, context[2], context[3], dt*a)
            update = solve(-residual)
            self._count('hc_newton_updates')
            for factor in (1., .5, .25, .125):
                self._count('hc_line_search_trials')
                candidate = current+factor*update
                if not bool(self._scalar(self._finite(candidate).all())):
                    continue
                trial_rates, _, _ = self._hc_rates(candidate, context)
                trial = candidate-y0[None, :, :]-dt*self._mix(a, trial_rates)
                trial_norm = self._scalar((self._abs(trial)/scale).max())
                if math.isfinite(trial_norm) and (trial_norm < norm or trial_norm <= 1):
                    current = candidate
                    break
            else:
                raise ContractError('Native HC Radau bounded line search failed')
        raise AssertionError('unreachable')

    def _jacobian_data(self, y, t, anchor):
        self._validate(y, anchor, t)
        x, c, v, g, u, w, z, h, aux = self.physics.split(y)
        if not bool(self._scalar((v > 0).all())):
            raise ContractError('Native voltage zero has no finite Newton Jacobian; no clipping')
        _, E = anchor.at(t)
        _, xc = self._xc_rates(self._stack((x, c), axis=-1), E)
        _, vg = self._vg_rates(self._stack((v, g), axis=-1), x)
        context = self._hc_context(v[None, :], u[None, :])
        hc_rates, gain, release = self._hc_rates(self._stack((w, z, h))[None, :, :], context)
        return xc, vg, context, hc_rates, gain, release

    def jacobian_action(self, y, t, anchor, vector):
        """Full numeric-state original Jacobian, including every feedforward edge."""
        vector = self.physics._array(vector)
        dx, dc, dv, dg, du, dw, dz, dh, daux = self.physics.split(vector)
        x, c, v, g, u, *_ = self.physics.split(y)
        xc, vg, context, rates, gain, release = self._jacobian_data(y, t, anchor)
        rxc = self._einsum('nij,nj->ni', xc, self._stack((dx, dc), axis=-1))
        rvg = self._einsum('nij,nj->ni', vg, self._stack((dv, dg), axis=-1))
        rvg[:, 0] += 1000*dx/(self.p['tau_m']*g)
        ru = 1000*(dv-du)/self.p['tau_a']
        feedback = self._mv(self.physics.B, dh)
        dr = gain[0]*(dv-feedback)-self.p['mu']*release[0]*du/u
        a = 1000/self.p['tau_1']
        rwh = a*(self.p['g_t']*self._mv(self.physics.A, dr)-dw)
        pooled_relative = self.p['mu']*self._mv(self.physics.A, du)/context[4][0]
        rz = context[2][0]*(dw-dz)-rates[0, 1]*pooled_relative
        rh = context[3][0]*(dz-dh)-rates[0, 2]*pooled_relative
        ra = a*(self.p['g_t']*dr[self.physics._aux]-daux)
        return self._cat((rxc[:, 0], rxc[:, 1], rvg[:, 0], rvg[:, 1], ru, rwh, rz, rh, ra))

    def _filtered_error(self, y, t, dt, anchor, f0, stage_rhs):
        _, b, _, gamma, bhat = self._parameters(y)
        defect = dt*(gamma*f0+((bhat-b) @ stage_rhs))
        shift = gamma*dt
        xc, vg, context, rates, gain, release = self._jacobian_data(y, t, anchor)
        x, c, v, g, u, *_ = self.physics.split(y)
        dx, dc, dv, dg, du, dw, dz, dh, daux = self.physics.split(defect)
        ex = self._solve(self._eye(2)[None, :, :]-shift*xc, self._stack((dx, dc), axis=-1))
        rhs_vg = self._stack((dv+shift*1000*ex[:, 0]/(self.p['tau_m']*g), dg), axis=-1)
        ev = self._solve(self._eye(2)[None, :, :]-shift*vg, rhs_vg)
        ku = 1000/self.p['tau_a']
        eu = (du+shift*ku*ev[:, 0])/(1+shift*ku)
        upstream_release = gain[0]*ev[:, 0]-self.p['mu']*release[0]*eu/u
        a = 1000/self.p['tau_1']
        upstream_w = a*self.p['g_t']*self._mv(self.physics.A, upstream_release)
        pooled_relative = self.p['mu']*self._mv(self.physics.A, eu)/context[4][0]
        rhs_hc = self._stack((dw+shift*upstream_w,
            dz-shift*rates[0, 1]*pooled_relative, dh-shift*rates[0, 2]*pooled_relative))[None, :, :]
        solve = self._hc_linear_solver(gain, context[2], context[3], shift*self._eye(1))
        eh = solve(rhs_hc)[0]
        dr = upstream_release-gain[0]*self._mv(self.physics.B, eh[2])
        ea = (daux+shift*a*self.p['g_t']*dr[self.physics._aux])/(1+shift*a)
        self._count('filtered_estimator_solves')
        error = self._cat((ex[:, 0], ex[:, 1], ev[:, 0], ev[:, 1], eu, eh[0], eh[1], eh[2], ea))
        self._require_finite(error)
        return error

    def _stage_family(self, name, initial, parents, parent_ports, start, dt, anchor):
        if name == 'native_RE':
            nodes = ((4-math.sqrt(6))/10, (4+math.sqrt(6))/10, 1.)
            values = tuple(anchor.at(start+node*dt) for node in nodes)
            return {'R': self._stack(tuple(v[0] for v in values)),
                    'E': self._stack(tuple(v[1] for v in values))}, {}, {}
        if name == 'native_XC':
            stages, rates = self._local(self._stack((initial['X'], initial['C']), axis=-1), dt,
                lambda values: self._xc_rates(values, self._transpose(parents['E'], (1, 0))),
                lambda values: (values >= 0).all(axis=(1, 2)) if self.torch is None
                else (values >= 0).all(dim=2).all(dim=1))
            return dict(X=stages[:, 0], C=stages[:, 1]), dict(X=rates[:, 0], C=rates[:, 1]), {}
        if name == 'native_Vis_gi':
            stages, rates = self._local(self._stack((initial['Vis'], initial['g_i']), axis=-1), dt,
                lambda values: self._vg_rates(values, self._transpose(parents['X'], (1, 0))),
                lambda values: (values > 0).all(axis=(1, 2)) if self.torch is None
                else (values > 0).all(dim=2).all(dim=1))
            return {'Vis': stages[:, 0], 'g_i': stages[:, 1]}, {'Vis': rates[:, 0], 'g_i': rates[:, 1]}, {}
        if name == 'native_Vbar':
            rate = 1000/self.p['tau_a']
            stages = affine_stages(initial['Vbar'], rate*parents['Vis'], rate, dt, counters=self.statistics)
            if not bool(self._scalar(self._finite(stages).all() & (stages > 0).all())):
                raise ContractError('Native Vbar collocation stages outside positive domain')
            return {'Vbar': stages}, {'Vbar': rate*(parents['Vis']-stages)}, {}
        if name == 'native_HC':
            context = self._hc_context(parents['Vis'], parents['Vbar'])
            stages, rates, release = self._hc_stages(
                self._stack((initial['w'], initial['z2'], initial['Vh'])), dt, context)
            return ({n: stages[:, i] for i, n in enumerate(('w', 'z2', 'Vh'))},
                    {n: rates[:, i] for i, n in enumerate(('w', 'z2', 'Vh'))}, {'release': release})
        if name == 'native_z1_aux':
            rate = 1000/self.p['tau_1']
            forcing = rate*self.p['g_t']*parent_ports['native_HC']['release'][:, self.physics._aux]
            stages = affine_stages(initial['z1_aux'], forcing, rate, dt, counters=self.statistics)
            return {'z1_aux': stages}, {'z1_aux': forcing-rate*stages}, {}
        raise ContractError('Native Radau has no handler for the scheduled family')

    def _halfstep(self, y, start, dt, anchor, f0):
        initial = self.physics.blocks(y)
        stages, rates, ports, completed = {}, {}, {}, set()
        for name in self.execution_order:
            record = self._family_records[name]
            if not set(record['parents']).issubset(completed):
                raise ContractError('Native Radau family ran before its declared actual parent stages')
            own_initial = {n: initial[n] for n in record['states'] if n in initial}
            parents = {n: stages[n] for parent in record['parents'] for n in self._family_records[parent]['states']}
            parent_ports = {parent: ports[parent] for parent in record['parents']}
            values, derivatives, readouts = self._stage_family(name, own_initial, parents, parent_ports, start, dt, anchor)
            expected_rates = set(record['states']) if name != 'native_RE' else set()
            if (set(values) != set(record['states']) or set(derivatives) != expected_rates
                    or set(values).intersection(stages) or set(derivatives).intersection(rates)):
                raise ContractError('Native Radau handler changed its planned state ownership')
            for state, value in values.items():
                count = self.physics.nc if state in ('R', 'E') else initial[state].shape[0]
                if tuple(value.shape) != (3, count):
                    raise ContractError('Native Radau handler stage coverage differs')
            for state, derivative in derivatives.items():
                if tuple(derivative.shape) != tuple(values[state].shape):
                    raise ContractError('Native Radau handler RHS coverage differs')
            stages.update(values)
            rates.update(derivatives)
            ports[name] = readouts
            completed.add(name)
            self._count('family_executions:'+name)
        if not len(initial['z1_aux']):
            stages['z1_aux'], rates['z1_aux'] = self._zeros((3, 0)), self._zeros((3, 0))
        if set(rates) != set(STATE_NAMES):
            raise ContractError('Native Radau scheduled families left numerical states unpublished')
        packed_stages = self._cat(tuple(stages[n] for n in STATE_NAMES), axis=1)
        rhs = self._cat(tuple(rates[n] for n in STATE_NAMES), axis=1)
        self._require_finite(packed_stages, rhs)
        error = self._filtered_error(y, start, dt, anchor, f0, rhs)
        return NativeRadauHalfStep(start, dt, packed_stages, rhs, error)

    def step(self, y, start, dt, anchor, *, retain_stages=True):
        """One tentative macro step; does not mutate acceptance or error ledgers."""
        self._check_guard()
        y = self.physics._array(y)
        if not all(math.isfinite(float(x)) for x in (start, dt)) or not 0 < dt <= MAX_STEP:
            raise ContractError('Native Radau step outside (0,100us]')
        stop = start+dt
        if not anchor.time_seconds <= start < stop <= anchor.input_stop_seconds:
            raise ContractError('Native Radau step crosses analytic input boundary')
        start, dt, stop = float(start), float(dt), float(stop)
        self._validate(y, anchor, start)
        f0 = self._rhs(y, start, anchor)
        first = self._halfstep(y, start, dt/2, anchor, f0)
        mid = start+dt/2
        ym = first.stages[-1]
        fm = self._rhs(ym, mid, anchor)
        second = self._halfstep(ym, mid, stop-mid, anchor, fm)
        y1 = second.stages[-1]
        f1 = self._rhs(y1, stop, anchor)
        segment = NativeRadauSegment(start, stop, y, f0, ym, fm, y1, f1, anchor, self.physics)
        error = self._abs(first.error)+self._abs(second.error)
        self._count('macro_attempts')
        return NativeRadauResult(_owned(y1), PiecewiseNativeRadauDense((segment,)),
            (start, stop), dict(self.statistics, certified=False, **self._workspace_metric_fields()),
            (first, second) if retain_stages else (), error)

    def controller_state(self):
        """Independent controller storage; arrays retain their current device."""
        return dict(identity=dict(self.identity, original_auxiliary_ids=list(self.identity['original_auxiliary_ids'])),
            next_step=self.next_step,
            window_end=self._window_end, window_attempts=self._window_attempts,
            window_accepts=self._window_accepts, statistics=dict(self.statistics),
            error_weights=None if self.error_weights is None else _owned(self.error_weights),
            budgets=None if self.budgets is None else _owned(self.budgets),
            error_ledger=None if self.error_ledger is None else _owned(self.error_ledger))

    def restore_controller(self, state):
        """Restore the identical bounded policy without changing scientific scales."""
        fields = {'identity', 'next_step', 'window_end', 'window_attempts', 'window_accepts',
                  'statistics', 'error_weights', 'budgets', 'error_ledger'}
        if not isinstance(state, dict) or set(state) != fields:
            raise ContractError('Native Radau controller schema differs')
        if state['identity'] != self.identity:
            raise ContractError('Native Radau controller identity differs')
        step = state['next_step']
        if type(step) not in (int, float) or step not in tuple(MAX_STEP/(2**i) for i in range(7)):
            raise ContractError('Native Radau restored step is outside the frozen ladder')
        end = state['window_end']
        if end is not None and (type(end) not in (int, float) or not math.isfinite(end)):
            raise ContractError('Native Radau restored window is nonfinite')
        for name, maximum in (('window_attempts', 127), ('window_accepts', 64)):
            if type(state[name]) is not int or not 0 <= state[name] <= maximum:
                raise ContractError('Native Radau restored iteration counters differ')
        replacements = {}
        for name in ('error_weights', 'budgets'):
            value = self._coordinate_scale(state[name])
            current = getattr(self, name)
            if current is not None and (value is None or not bool(self._scalar((current == value).all()))):
                raise ContractError('Native Radau restored frozen coordinate scale differs')
            replacements[name] = value
        ledger = state['error_ledger']
        if ledger is not None:
            ledger = self.physics._array(ledger)
            if tuple(ledger.shape) != (self.physics.nstate,) or not bool(self._scalar(
                    self._finite(ledger).all() & (ledger >= 0).all())):
                raise ContractError('Native Radau restored estimate ledger differs')
            ledger = _owned(ledger)
        statistics = state['statistics']
        if not isinstance(statistics, dict) or any(type(value) is not int or value < 0 for value in statistics.values()):
            raise ContractError('Native Radau restored statistics differ')
        self.next_step, self._window_end = step, end
        self._window_attempts, self._window_accepts = state['window_attempts'], state['window_accepts']
        self.error_weights, self.budgets = replacements['error_weights'], replacements['budgets']
        self.error_ledger, self.statistics = ledger, dict(statistics)

    def _error_measure(self, error, ledger):
        scaled = error/self.error_weights
        mean = self._scalar((scaled*scaled).mean())
        if not math.isfinite(mean) or mean < 0:
            raise ContractError('Native Radau embedded estimate is nonfinite')
        rms = math.sqrt(mean)
        coordinate = 0. if self.budgets is None else self._scalar(
            (ledger/(self.temporal_fraction*self.budgets)).max())
        # The O(h^4) embedded difference controls steps, not the O(h^6) local
        # error budget of the fifth-order solution or the independent pair QA.
        return rms, rms, coordinate

    @staticmethod
    def _ladder(requested):
        for i in range(7):
            value = MAX_STEP/(2**i)
            if value <= requested:
                return value
        return MIN_STEP

    def advance(self, y, start, stop, anchor, *, retain_stages=False):
        """Advance within one analytic input epoch using the frozen bounded ladder."""
        before = dict(self.statistics)
        y = self.physics._array(y)
        if not all(math.isfinite(float(x)) for x in (start, stop)) or not anchor.time_seconds <= start < stop <= anchor.input_stop_seconds:
            raise ContractError('Native Radau advance crosses input boundary or has invalid interval')
        start, stop = float(start), float(stop)
        self._validate(y, anchor, start)
        if self.error_weights is None:
            self.error_weights = _owned(self.atol+self.rtol*self._abs(y))
        if self.error_ledger is None:
            self.error_ledger = self._zeros(y.shape)
        if self._window_end is None:
            self._window_end = start+MAX_STEP
        t, current = start, y
        segments, caches, times = [], [], [start]
        total_error = self._zeros(y.shape)
        max_rms = max_coordinate = 0.
        while t < stop:
            self._check_guard()
            if t >= self._window_end:
                self._window_end += MAX_STEP
                self._window_attempts = self._window_accepts = 0
            end = min(stop, self._window_end)
            requested = min(self.next_step, end-t)
            # A true boundary remainder may be smaller than the nonboundary floor.
            dt = requested
            boundary = dt == end-t
            if self._window_attempts >= 127 or self._window_accepts >= 64:
                raise ContractError('Native Radau exceeded 127 attempts or 64 accepts per 100us')
            self._window_attempts += 1
            trial = self.step(current, t, dt, anchor, retain_stages=retain_stages)
            candidate_ledger = self.error_ledger+trial.error
            eta, rms, coordinate = self._error_measure(trial.error, candidate_ledger)
            if not math.isfinite(eta):
                raise ContractError('Native Radau embedded estimate is nonfinite')
            factor = 2. if eta == 0 else max(.2, min(2., .8*eta**(-.25)))
            if eta <= 1:
                t = end if boundary else t+dt
                current = trial.y
                self.error_ledger = candidate_ledger
                total_error += trial.error
                times.append(t)
                segments.extend(trial.dense.segments)
                caches.extend(trial.stage_cache)
                self._window_accepts += 1
                self._count('accepted_macros')
                max_rms, max_coordinate = max(max_rms, rms), max(max_coordinate, coordinate)
                self.next_step = self._ladder(min(MAX_STEP, dt*factor))
            else:
                self._count('rejected_macros')
                if dt <= MIN_STEP:
                    raise ContractError('Native Radau error budget failed at minimum permitted step')
                self.next_step = self._ladder(min(dt/2, dt*factor))
        metrics = {name: value-before.get(name, 0) for name, value in self.statistics.items()}
        metrics.update(self._workspace_metric_fields())
        metrics.update(error_rms=max_rms, coordinate_error_ratio=max_coordinate,
                       certified=False, error_ledger_kind='absolute_filtered_estimates',
                       coordinate_error_ratio_is_diagnostic=True)
        return NativeRadauResult(_owned(current), PiecewiseNativeRadauDense(tuple(segments)),
            tuple(times), metrics, tuple(caches), _owned(total_error))
