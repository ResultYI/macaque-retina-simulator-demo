"""Experimental HC-only Anderson initialization; source solvers stay intact.

No learned coefficients, runtime RHS changes, graph changes or adapted-state
cache. Dense arrays below have at most ``memory`` history columns; A/B remain
sparse. Accepted local roots still require real history/RNG/background wash-in.
"""
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
import hashlib
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from .contracts import ContractError, Quantity
from .vh2005 import NativeConeH1, SOURCE_VALUES
from .finite_feedback_native import FiniteContactConeH1
from .background_seed import seed_background_state


HC_RESIDUAL_LIMIT = 1e-8  # Unchanged source acceptance, native voltage units.
SOURCE_HASHES = {
    'vh2005.py': '60c1091971527216961df8a49b40e3dad8125f44de1d82d4d9374afcd4462858',
    'finite_feedback_native.py': 'c07e1b77576ad820e5c0d578bdf35010fc4bd30930898f19598162eb3c1de756',
}


@dataclass(frozen=True)
class AndersonPolicy:
    """Numerical work budgets; not biological parameters or relaxed criteria."""
    memory: int = 5
    max_iterations: int = 200
    max_function_evaluations: int = 2000
    max_backtracks: int = 20
    max_wall_seconds: float = 60.
    damping: float = 1.
    regularization_relative: float = 64 * np.finfo(np.float64).eps
    working_residual_target: float = 1e-13

    def __post_init__(self):
        for name in ('memory', 'max_iterations', 'max_function_evaluations', 'max_backtracks'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError('Nonnegative integer Anderson budget required: ' + name)
        if not 1 <= self.memory <= 10 or self.max_function_evaluations < 1:
            raise ContractError('Small bounded Anderson history and positive map budget required')
        if (not np.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0 or
                not np.isfinite(self.damping) or not 0 < self.damping <= 1 or
                not np.isfinite(self.regularization_relative) or self.regularization_relative <= 0 or
                not np.isfinite(self.working_residual_target) or
                not 0 < self.working_residual_target <= HC_RESIDUAL_LIMIT):
            raise ContractError('Finite positive Anderson safeguards required')


class AndersonInterrupted(ContractError):
    """Explicit cancellation/protection: never start a heavier fallback."""


class _NumericalFailure(Exception):
    pass


def _in_box(h, upper):
    return h.shape == upper.shape and np.all(np.isfinite(h)) and np.all(h >= 0) and np.all(h <= upper)


def _direction(hs, gs, damping, regularization):
    """Type-II: beta*g - (delta_h + beta*delta_g)*gamma, ridge LS."""
    dh = np.column_stack([b - a for a, b in zip(hs[:-1], hs[1:])])
    dg = np.column_stack([b - a for a, b in zip(gs[:-1], gs[1:])])
    scale = float(np.sum(dg * dg))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).tiny:
        raise _NumericalFailure('degenerate_Anderson_history')
    lam = max(regularization * scale, np.finfo(np.float64).tiny)
    matrix = np.vstack((dg, np.sqrt(lam) * np.eye(dg.shape[1])))
    rhs = np.concatenate((gs[-1], np.zeros(dg.shape[1])))
    gamma = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    direction = damping * gs[-1] - (dh + damping * dg) @ gamma
    if not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(direction)):
        raise _NumericalFailure('nonfinite_Anderson_step')
    return direction, lam


def _checked_result(native, background, values):
    if len(values) != len(native.state_specs):
        raise ContractError('HC candidate returned an incomplete source state registry')
    for q, spec in zip(values, native.state_specs):
        if not isinstance(q, Quantity) or q.evidence_tag != native.tag:
            raise ContractError('HC candidate source-state evidence mismatch')
        q.require(spec.unit, spec.meaning, spec.shape, check_finite=True)
    blocks = tuple(q.values for q in values)
    R, E, X, C, Vis, gi, Vbar, z1, z2, h = blocks
    if (any(np.any(a < 0) for a in (R, E, X, C, Vis)) or
            np.any(gi <= 0) or np.any(Vbar <= 0)):
        raise ContractError('HC candidate violated original native state domain')
    p = native.p
    aI = (Vbar / p['V_I']) ** p['mu']
    upper = np.asarray(native.A @ (p['g_t'] / aI)).ravel()
    if not np.all(np.isfinite(aI)) or np.any(aI <= 0) or not _in_box(h, upper):
        raise ContractError('HC candidate violated the original nonnegative HC bounds')
    drive = p['g_t'] / aI * expit((Vis - native.B @ h - p['V_k']) / p['V_n'])
    residual = float(np.max(np.abs(h - native.A @ drive)))
    if not np.isfinite(residual) or residual > HC_RESIDUAL_LIMIT:
        raise ContractError('HC candidate failed the unchanged1e-8 source residual')
    rhs = native.derivative(0., blocks, {'light': background})
    return residual, {s.state_name: float(np.max(np.abs(q.values)))
                     for s, q in zip(native.state_specs, rhs)}


def anderson_equilibrium(native, background, policy=None, *, on_check=None):
    """Return ``(source-state Quantities, audit)`` using a named HC candidate.

    Invalid input is rejected. Numerical failure calls the original model's
    real equilibrium method on an isolated shallow copy. Cancellation and
    MemoryError propagate immediately; neither is converted into fallback.
    Source scalar cone equations and their Brent tolerance are transcribed
    exactly from the pinned source, with cooperative checks between cells.
    """
    if type(native) not in (NativeConeH1, FiniteContactConeH1):
        raise ContractError('HC Anderson candidate supports only reviewed native/finite source types')
    if dict(native.p) != SOURCE_VALUES:
        raise ContractError('HC Anderson candidate requires unchanged complete source parameters')
    light = background.require('td', 'native_retinal_illuminance', (native.nc,), check_finite=True)
    if background.evidence_tag != native.tag or np.any(light < 0):
        raise ContractError('HC Anderson requires matching evidence and nonnegative total light')
    policy = policy or AndersonPolicy()
    if not isinstance(policy, AndersonPolicy):
        raise ContractError('Explicit AndersonPolicy required')
    started = perf_counter()
    audit = {'schema_version': 'hc_safeguarded_anderson_v1', 'policy': asdict(policy),
        'HC_residual_limit_native': HC_RESIDUAL_LIMIT,
        'source_acceptance_limit': HC_RESIDUAL_LIMIT,
        'working_residual_target': policy.working_residual_target,
        'working_target_is_independent_numerical_policy_not_source_LS_gtol': True,
        'candidate_success': False, 'candidate_working_target_reached': False,
        'fallback_used': False, 'function_evaluations': 0, 'iterations': 0,
        'trace': [], 'history_matrix_max_columns': policy.memory,
        'source_equations_reference_parameters_changed': False,
        'whole_network_steady_state_claim': False, 'wash_in_completed': False,
        'physiological_validation': 'NOT_RUN', 'training': False,
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def protection():
        if on_check is not None and on_check() is False:
            raise AndersonInterrupted('HC Anderson cancelled at a cooperative boundary')

    def budget():
        protection()
        if perf_counter() - started >= policy.max_wall_seconds:
            raise _NumericalFailure('candidate_wall_budget')

    def fallback(reason):
        # Real dispatch is intentional: FiniteContact must keep its own empty-
        # feedback handling. Do not substitute a source-scale nominal state.
        protection()
        audit.update(fallback_used=True, fallback_reason=str(reason),
                     candidate_wall_seconds=perf_counter() - started)
        tick = perf_counter()
        isolated = copy(native)
        result = isolated.equilibrium(background)
        protection()
        residual, rhs = _checked_result(native, background, result)
        audit.update(fallback_wall_seconds=perf_counter() - tick,
            maximum_HC_residual_native=residual, original_rhs_max_abs_per_second=rhs,
            returned_state_residual_meets_working_target=residual <= policy.working_residual_target,
            original_equilibrium_called=True,
            fallback_source_audit=deepcopy(getattr(isolated, 'background_equilibrium_audit', None)),
            wall_seconds=perf_counter() - started)
        return result, audit

    protection()
    actual_sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in SOURCE_HASHES}
    audit['source_sha256'] = actual_sources
    if actual_sources != SOURCE_HASHES:
        return fallback('source_transcription_identity_changed')
    p = native.p
    try:
        budget()
        beta = p['c_beta'] + p['k_beta'] * light
        current = np.empty(native.nc, dtype=np.float64)
        for i, b in enumerate(beta):
            if i % 256 == 0:
                budget()
            current[i] = brentq(lambda c: c - (1 / (b * (1 + (p['a_C'] * c) ** p['n_C']))) ** p['n_X'],
                0, (1 / b) ** p['n_X'], xtol=1e-12)
        X = current ** (1 / p['n_X'])
        Vis = (current / p['a_is']) ** (1 / (1 + p['gamma']))
        gi = p['a_is'] * Vis ** p['gamma']
        aI = (Vis / p['V_I']) ** p['mu']
        if (np.any(~np.isfinite(aI)) or np.any(aI <= 0) or
                np.any(gi <= 0) or np.any(Vis <= 0)):
            raise _NumericalFailure('invalid_source_outer_equilibrium_domain')
        upper = np.asarray(native.A @ (p['g_t'] / aI)).ravel()
        if upper.shape != (native.nh,) or np.any(~np.isfinite(upper)) or np.any(upper <= 0):
            raise _NumericalFailure('invalid_source_HC_bounds')

        def fixed_point(h):
            budget()
            if audit['function_evaluations'] >= policy.max_function_evaluations:
                raise _NumericalFailure('candidate_map_budget')
            audit['function_evaluations'] += 1
            return np.asarray(native.A @ (p['g_t'] / aI *
                expit((Vis - native.B @ h - p['V_k']) / p['V_n']))).ravel()

        # Identical original sparse-solver initial guess, not a new background.
        h = np.minimum(upper * .5, np.max(Vis))
        g = fixed_point(h) - h
        hs, gs = [h], [g]
        for iteration in range(policy.max_iterations + 1):
            budget()
            norm = float(np.max(np.abs(g)))
            if not np.isfinite(norm):
                raise _NumericalFailure('nonfinite_HC_residual')
            if norm <= policy.working_residual_target:
                break
            if iteration == policy.max_iterations:
                raise _NumericalFailure('candidate_iteration_budget')
            directions = []
            if len(hs) > 1:
                direction, lam = _direction(hs, gs, policy.damping, policy.regularization_relative)
                directions.append(('anderson', direction, lam))
            directions.append(('damped_fixed_point', policy.damping * g, None))
            accepted = None
            for method, direction, lam in directions:
                for backtrack in range(policy.max_backtracks + 1):
                    budget()
                    alpha = .5 ** backtrack
                    trial = h + alpha * direction
                    row = {'iteration': iteration, 'method': method, 'alpha': alpha,
                        'regularization': lam, 'previous_residual_native': norm}
                    if not _in_box(trial, upper):
                        row.update(accepted=False, reason='outside_HC_domain')
                    else:
                        trial_g = fixed_point(trial) - trial
                        trial_norm = float(np.max(np.abs(trial_g)))
                        row['residual_native'] = trial_norm if np.isfinite(trial_norm) else None
                        if np.isfinite(trial_norm) and (trial_norm < norm or trial_norm <= policy.working_residual_target):
                            row.update(accepted=True, reason='residual_decreased_or_working_target')
                            accepted = trial, trial_g
                        else:
                            row.update(accepted=False, reason='residual_not_decreased')
                    audit['trace'].append(row)
                    if accepted is not None:
                        break
                if accepted is not None:
                    if method == 'damped_fixed_point' and len(directions) > 1:
                        hs, gs = [h], [g]
                    break
            if accepted is None:
                raise _NumericalFailure('safeguarded_step_stagnation')
            h, g = accepted
            hs.append(h); gs.append(g)
            hs, gs = hs[-policy.memory-1:], gs[-policy.memory-1:]
            audit['iterations'] = iteration + 1
        z1 = p['g_t'] / aI * expit((Vis - native.B @ h - p['V_k']) / p['V_n'])
        z2 = native.A @ z1
        result = tuple(Quantity(a, s.unit, s.meaning, native.tag) for a, s in zip(
            (light, light, X, current, Vis, gi, Vis, z1, z2, h), native.state_specs))
        residual, rhs = _checked_result(native, background, result)
        if residual > policy.working_residual_target:
            raise _NumericalFailure('working_target_failed_original_map_recheck')
        protection()
        audit.update(candidate_success=True, candidate_working_target_reached=True,
            returned_state_residual_meets_working_target=True, maximum_HC_residual_native=residual,
            original_rhs_max_abs_per_second=rhs, original_equilibrium_called=False,
            wall_seconds=perf_counter() - started, candidate_wall_seconds=perf_counter() - started)
        return result, audit
    except (AndersonInterrupted, MemoryError):
        raise
    except (_NumericalFailure, np.linalg.LinAlgError, FloatingPointError, OverflowError,
            ValueError) as exc:
        return fallback(str(exc))


def seed_hc_anderson_state(runner, state, channels, policy=None, *, on_check=None):
    """Independent seed route using only temporary local initializer adapters.

    Runtime objects remain untouched. The existing seed helper performs all
    schema/unit/evidence checks, isolation, rod fixed points and reference setup.
    This function never changes global methods or replaces a runtime model.
    """
    policy = policy or AndersonPolicy()
    local_runner = copy(runner)
    local_compiled = copy(runner.compiled)
    local_compiled.components = dict(runner.compiled.components)
    local_runner.compiled = local_compiled
    audits = {}
    for cid, component in runner.compiled.components.items():
        if type(component.native_model) not in (NativeConeH1, FiniteContactConeH1):
            continue
        original = component.native_model
        local_model = copy(original)
        local_audit = {}
        local_model.background_equilibrium_audit = local_audit
        def initialize(background, original=original, cid=cid, local_audit=local_audit):
            values, audit = anderson_equilibrium(original, background, policy, on_check=on_check)
            audits[cid] = audit
            local_audit.update(method='independent_HC_safeguarded_Anderson_seed_initializer',
                maximum_residual_native=audit['maximum_HC_residual_native'],
                candidate_success=audit['candidate_success'], fallback_used=audit['fallback_used'],
                whole_inner_network_steady_claim=False)
            return values
        local_model.equilibrium = initialize
        local_compiled.components[cid] = replace(component, native_model=local_model)
    if not audits and not any(type(c.native_model) in (NativeConeH1, FiniteContactConeH1)
                             for c in runner.compiled.components.values()):
        raise ContractError('HC Anderson seed requires a reviewed native component')
    seed = seed_background_state(local_runner, state, channels)
    audit = deepcopy(seed.audit)
    audit.update(schema_version='background_seed_HC_anderson_v1',
        independent_initialization_procedure='HC-only safeguarded Anderson; original runtime unchanged',
        HC_anderson=audits, numerical_difference_automatically_accepted=False)
    return replace(seed, audit=audit)
