"""Independent full-state, whole-interval comparison of published waveforms.

This QA consumes accepted packets only. It does not advance a model, evaluate
its RHS, alter a frozen budget, or assert either arm's absolute ODE error.
"""
from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np

from .contracts import ContractError
from .continuous_state import AcceptedSeries
from .group_dense_certificate import (dop853_power_coefficient_enclosure,
    quintic_power_coefficient_enclosure, polynomial_pair_enclosures,
    polynomial_range_enclosure, _add, _multiply, _subtract)
from .resident_monitor import UNIT_INTERVAL_MEANINGS, NONNEGATIVE_MEANINGS
from .spatial_group_model import BC
from .vh2005 import SOURCE_VALUES


SCHEMA = 'retipath-scc-radau-dense-pair-certificate-1'
_MAX_PIECES = 64
_EPS = np.finfo(np.float64).eps
_NATIVE = {name: ('cones', 'reduced_native', name) for name in ('X', 'C', 'Vis', 'g_i', 'Vbar')}
_NATIVE.update({name: ('HC', 'reduced_native', name) for name in ('w', 'z2', 'Vh')})
_NATIVE['z1_aux'] = ('cones', 'selected_original_z1_auxiliary', 'z1_aux')
_ANALYTIC = (('cones', 'analytic_RE', 'R'), ('cones', 'analytic_RE', 'E'))


@dataclass
class _Piece:
    t0: float
    t1: float
    source: object
    kind: str
    rows: object = None
    prepared: object = None


@dataclass
class _Prepared:
    coefficients: tuple
    remainder: np.ndarray
    roundoff: np.ndarray
    unsupported: str | None


def _state_name(key):
    return ':'.join(key)


def _cpu(value, counters, rows=None, *, matrix=False):
    if rows is not None:
        value = value[:, rows] if matrix else value[rows]
    if hasattr(value, 'detach'):
        counters['host_readback_bytes'] += int(value.numel()*value.element_size())
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype != np.float64 or not np.isfinite(array).all():
        raise ContractError('Accepted dense arrays must be finite original FP64')
    return np.array(array, copy=True)


def _source_times(source):
    if isinstance(source, dict):
        if set(source) != {'t_old', 't', 'y_old', 'F'}:
            raise ContractError('Explicit original GPU DOP853 dense packet fields required')
        return float(source['t_old']), float(source['t'])
    if hasattr(source, 't_old'):
        return float(source.t_old), float(source.t)
    if hasattr(source, 't0'):
        return float(source.t0), float(source.t1)
    return float(source.start), float(source.stop)


def _piece(source, kind, rows=None):
    t0, t1 = _source_times(source)
    if not math.isfinite(t0) or not math.isfinite(t1) or not t0 < t1:
        raise ContractError('Finite increasing accepted dense segment required')
    return _Piece(t0, t1, source, kind, rows)


def _cover(pieces, t0, t1):
    result = sorted((p for p in pieces if p.t0 < t1 and p.t1 > t0), key=lambda p: p.t0)
    cursor = t0
    for piece in result:
        begin, end = max(t0, piece.t0), min(t1, piece.t1)
        if begin != cursor:
            raise ContractError('Accepted dense coverage has a gap or overlap')
        cursor = end
    if cursor != t1:
        raise ContractError('Accepted dense coverage does not reach the requested endpoint')
    return result


def _extract(arm, advance, key, count, t0, t1):
    inner = arm.inner
    if key in inner.slices:
        if getattr(advance, 't0', None) != t0 or getattr(advance, 't1', None) != t1:
            raise ContractError('Inner accepted interval identity differs')
        section = inner.slices[key]
        if section.stop-section.start != count:
            raise ContractError('Original packed inner row count differs')
        dense = getattr(advance, 'dense_segments', ())
        if dense:
            return _cover([_piece(segment, 'dop853_dense', section) for segment in dense], t0, t1)
        entries = getattr(advance, 'representations', {}).get(key)
        if not entries or len(entries) != 1 or not np.array_equal(entries[0][0], np.arange(count)):
            raise ContractError('Complete ordered per-state accepted representations required')
        representation = entries[0][1]
        segments = representation.representations if isinstance(representation, AcceptedSeries) else (representation,)
        pieces = []
        expected = np.arange(section.start, section.stop, dtype=np.int64)
        for segment in segments:
            if not segment.accepted or not np.array_equal(segment.state_ids, expected):
                raise ContractError('Accepted packet original packed state IDs differ')
            pieces.append(_piece(segment, segment.kind))
        return _cover(pieces, t0, t1)
    name = key[-1]
    if _NATIVE.get(name) != key or name not in arm.native_physics.slices:
        raise ContractError('State is absent from the original inner/native registry')
    section = arm.native_physics.slices[name]
    if section.stop-section.start != count:
        raise ContractError('Original packed native row count differs')
    pieces = []
    for interval in arm.native_intervals:
        if interval.start >= t1 or interval.stop <= t0:
            continue
        if hasattr(interval, 'dense'):
            pieces.extend(_piece(segment, 'quintic_hermite', section) for segment in interval.dense.segments)
        else:
            if name not in interval.curves:
                raise ContractError('Native accepted state curve is absent')
            pieces.extend(_piece(segment, segment.kind) for segment in interval.curves[name].segments)
    return _cover(pieces, t0, t1)


def _sum_upper(values):
    total = np.zeros_like(values[0])
    for value in values:
        total = _add((total, total), (value, value))[1]
    return total


def _roundoff(majorants, operations):
    """Conservative scalar-expression gamma bound, with gradual-underflow floor.

    DOP853 has about 15 scalar operations and normalized derivative at most
    7*sum(abs(F)). Hermite basis derivative sensitivity is bounded by
    48*sum(abs(y))+7*h*sum(abs(f)); the 1024eps count covers its scalar
    operations plus normalized-time/map rounding. These deliberately loose
    counts do not cover transcendental libraries
    or large native reductions; 8192eps remains a separate engineering screen.
    """
    magnitude = _sum_upper(majorants)
    gamma = np.nextafter(operations*_EPS/(1-operations*_EPS), np.inf)
    charge = _multiply((gamma, gamma), (magnitude, magnitude))[1]
    floor = 4096*np.nextafter(0., 1.)
    return np.where(magnitude == 0, 0., _add((charge, charge), (floor, floor))[1])


def _prepare(piece, count, counters):
    if piece.prepared is not None:
        return piece.prepared
    source, rows = piece.source, piece.rows
    if hasattr(source, '_check'):
        source._check()
    if hasattr(source, '_theta'):
        source._theta(piece.t0)
    h = piece.t1-piece.t0
    if not math.isfinite(h) or h < np.finfo(float).tiny:
        raise ContractError('Normal positive FP64 duration required for dense arithmetic enclosure')
    field = lambda name: source[name] if isinstance(source, dict) else getattr(source, name)
    has = lambda name: name in source if isinstance(source, dict) else hasattr(source, name)
    get = lambda name, matrix=False: _cpu(field(name), counters, rows, matrix=matrix)
    zero = np.zeros(count)
    unsupported = None
    if piece.kind == 'dop853_dense':
        y0 = get('y_old' if has('y_old') else 'y0')
        f = get('F' if has('F') else 'coefficients', True)
        if y0.shape != (count,) or f.shape != (7, count):
            raise ContractError('Original DOP853 accepted state rows differ')
        constant = np.all(f == 0, axis=0)
        if np.all(constant):
            _add((zero, zero), (y0, y0))
            coefficients = (np.zeros((8, count)), np.zeros((8, count)))
            roundoff = zero.copy()
        else:
            coefficients = dop853_power_coefficient_enclosure(y0, f)
            roundoff = _roundoff([abs(y0), *abs(f)], 64)
        roundoff[constant] = 0.
        coefficients[0][:, constant] = 0.; coefficients[1][:, constant] = 0.
        coefficients[0][0, constant] = y0[constant]; coefficients[1][0, constant] = y0[constant]
        remainder = zero
    elif piece.kind == 'quintic_hermite':
        arrays = [get(name) for name in ('y0', 'f0', 'ym', 'fm', 'y1', 'f1')]
        if any(value.shape != (count,) for value in arrays):
            raise ContractError('Original Hermite accepted state rows differ')
        coefficients = quintic_power_coefficient_enclosure(*arrays, duration=h)
        majorants = [abs(value) if i % 2 == 0 else
            _multiply((h, h), (abs(value), abs(value)))[1] for i, value in enumerate(arrays)]
        roundoff = _roundoff(majorants, 1024)
        if hasattr(source, '_theta'):
            # Native packets return ym at the rounded physical midpoint.
            midpoint = piece.t0+h/2
            displacement = np.nextafter(abs((midpoint-piece.t0)/h-.5)+4*_EPS, np.inf)
            magnitude = np.maximum(abs(coefficients[0]), abs(coefficients[1]))
            derivative_bound = _sum_upper([_multiply((float(k), float(k)),
                (magnitude[k], magnitude[k]))[1] for k in range(1, 6)])
            snap = _multiply((displacement, displacement), (derivative_bound, derivative_bound))[1]
            roundoff = _sum_upper([roundoff, snap])
        remainder = zero
    elif piece.kind == 'discrete_hold':
        y0 = get('y0')
        if y0.shape != (count,):
            raise ContractError('Original held accepted state rows differ')
        coefficients, roundoff, remainder = (y0[None].copy(), y0[None].copy()), zero, zero
    elif piece.kind == 'exponential_forced':
        try:
            from .group_dense_analytic import constant_forcing_power_enclosure
        except ImportError as exc:
            raise ContractError('Optional constant-forcing enclosure is unavailable; analytic arithmetic unsupported') from exc
        y0, forcing = get('y0'), get('coefficients', True)
        decay = _cpu(source.decay, counters)
        if rows is not None and decay.ndim:
            decay = decay[rows]
        if y0.shape != (count,) or forcing.shape != (8, count) or np.any(forcing[1:] != 0):
            raise ContractError('Only original constant-forcing analytic rows have a Taylor certificate')
        coefficients, remainder = constant_forcing_power_enclosure(y0, forcing[0], decay, duration=h)
        roundoff = zero
        unsupported = 'Published CPU exp/phi evaluation has no verified transcendental roundoff bound'
    else:
        raise ContractError('Unsupported accepted dense kind: '+str(piece.kind))
    counters['polynomial_conversions'] += 1
    piece.prepared = _Prepared(coefficients, remainder, roundoff, unsupported)
    return piece.prepared


def _normalized(piece, begin, end):
    h = piece.t1-piece.t0
    lo = 0. if begin == piece.t0 else (begin-piece.t0)/h
    hi = 1. if end == piece.t1 else (end-piece.t0)/h
    if not 0 <= lo < hi <= 1:
        raise ContractError('Common refinement cannot be normalized without crossing an accepted boundary')
    return lo, hi


def _expand(bounds, error):
    return _add(bounds, (-error, error))


def _range(prepared, lo, hi, result=None):
    if result is None:
        result = polynomial_range_enclosure(prepared.coefficients, lo=lo, hi=hi)
    lower, upper = prepared.coefficients
    constant = np.all(lower[1:] == 0, axis=0) & np.all(upper[1:] == 0, axis=0) & (lower[0] == upper[0])
    result[0][constant] = lower[0, constant]; result[1][constant] = upper[0, constant]
    return _expand(result, _sum_upper([prepared.remainder, prepared.roundoff]))


def _domains(key, row, arm):
    spec = row['spec']; lower = upper = None; strict = False
    if spec.unit == 'dimensionless' and (spec.meaning in UNIT_INTERVAL_MEANINGS or key[1] == 'RGC_expected_history'):
        lower, upper = -1e-10, 1+1e-10
    if spec.meaning in NONNEGATIVE_MEANINGS:
        lower = -1e-10 if lower is None else max(lower, -1e-10)
    if key[:2] == ('cones', 'reduced_native'):
        if key[2] in ('X', 'C', 'Vis'):
            lower = 0.
        elif key[2] in ('g_i', 'Vbar'):
            lower, strict = 0., True
    if key[0] in BC and key[2] == 'A':
        parameters = arm.inner.parameters['populations'][key[0]]
        scale = np.asarray(parameters['adaptation_scale'])
        if not np.isfinite(scale).all() or np.any(scale < 0):
            raise ContractError('Original BC adaptation scale is unavailable')
        upper = np.broadcast_to(scale, row['initial'].shape)+1e-10
    return lower, upper, strict


def _domain_masks(bounds, domain):
    lo, hi = bounds; lower, upper, strict = domain
    unresolved = np.zeros(lo.shape, bool); failed = np.zeros(lo.shape, bool)
    if lower is not None:
        unresolved |= lo <= lower if strict else lo < lower
        failed |= hi <= lower if strict else hi < lower
    if upper is not None:
        unresolved |= hi > upper
        failed |= lo > upper
    return unresolved, failed



def _packed_count(arm, key):
    if key in arm.inner.slices:
        section = arm.inner.slices[key]
    elif _NATIVE.get(key[-1]) == key and key[-1] in arm.native_physics.slices:
        section = arm.native_physics.slices[key[-1]]
    else:
        raise ContractError('Frozen state is missing from the original packed layout')
    return section.stop-section.start


def _expected_keys(arm):
    result = set(arm.inner.state_keys)
    result.update(_NATIVE[name] for name in arm.native_physics.slices)
    if arm.native_physics.slices:
        result.update(_ANALYTIC)
    return result


def _analytic_pair(ref_arm, cand_arm, t0, t1, rows, counters, issue):
    ref = _cover([_Piece(float(i.start), float(i.stop), i.anchor, 'analytic_RE')
                  for i in ref_arm.native_intervals], t0, t1)
    cand = _cover([_Piece(float(i.start), float(i.stop), i.anchor, 'analytic_RE')
                   for i in cand_arm.native_intervals], t0, t1)
    cuts = sorted({t0, t1, *(max(t0, p.t0) for p in ref+cand), *(min(t1, p.t1) for p in ref+cand)})
    count = len(rows[_ANALYTIC[0]]['initial'])
    if len(rows[_ANALYTIC[1]]['initial']) != count or any(getattr(arm.native_physics, 'nc', count) != count for arm in (ref_arm, cand_arm)):
        raise ContractError('Both original native analytic R/E row counts must match the cone coordinates')
    for begin, end in zip(cuts, cuts[1:]):
        a = next(p.source for p in ref if p.t0 <= begin < p.t1)
        b = next(p.source for p in cand if p.t0 <= begin < p.t1)
        for anchor in (a, b):
            anchor.check_owned()
        if type(a) is not type(b) or any(getattr(a, n) != getattr(b, n) for n in
                ('time_seconds', 'input_stop_seconds', 'input_epoch')):
            raise ContractError('Native analytic R/E exact function/anchor identity differs')
        values = []
        for name in ('R', 'E', 'light'):
            av, bv = getattr(a, name), getattr(b, name)
            if type(av) is not type(bv) or str(getattr(av, 'device', 'cpu')) != str(getattr(bv, 'device', 'cpu')):
                raise ContractError('Native analytic R/E execution backend differs')
            av, bv = _cpu(av, counters), _cpu(bv, counters)
            if av.shape != (count,) or not np.array_equal(av, bv):
                raise ContractError('Native analytic R/E initial/input arrays differ')
            values.append(av)
        if a.time_seconds > begin or a.input_stop_seconds < end:
            raise ContractError('Native analytic R/E query crosses its known input')
        if np.any(values[2] < 0):
            raise ContractError('Native analytic light violates its original nonnegative domain')
        # Only the expm1 range [-1,0] for nonpositive input is used here;
        # independent intervals include cancellation in the published E formula.
        r, e, light = ((value, value) for value in values)
        q = (np.array(0.), np.array(1.))
        factor = SOURCE_VALUES['tau_R']/(SOURCE_VALUES['tau_R']-SOURCE_VALUES['tau_E'])
        rb = _add(r, _multiply(_subtract(light, r), q))
        eb = _add(_add(e, _multiply(_subtract(light, e), q)),
            _multiply(_multiply(_subtract(r, light), (factor, factor)), (np.array(-1.), np.array(1.))))
        for key, bounds in zip(_ANALYTIC, (rb, eb)):
            possible = bounds[0] < -1e-10
            if np.any(possible):
                index = int(np.flatnonzero(possible)[0])
                issue('INCONCLUSIVE', 'Published analytic R/E domain is not enclosed by finite interval algebra', key,
                      index=index, lower=float(bounds[0][index]), upper=float(bounds[1][index]))
        counters['analytic_identity_pieces'] += 1
    return count*2


def compare_dense_interval(ref_arm, cand_arm, ref_advance, cand_advance, t0, t1, budgets):
    """Certify complete candidate/reference published state waveforms, or stop.

    ``budgets`` is the original frozen GroupTimeMonitor.rows mapping. A PASS
    covers relative represented states plus the stated elementary-operation
    charges. Original RHS/derived-port/event gates remain separate. The old
    8192eps screen is checked separately and never added to B.
    """
    started = perf_counter()
    counters = dict(polynomial_conversions=0, common_pieces=0, subdivisions=0,
        host_readback_bytes=0, analytic_identity_pieces=0, scalar_coordinates=0,
        state_rows=0, maximum_subdivision_depth=0)
    issues, summaries = [], []
    issue_counts = dict(DOMAIN_FAIL=0, NUMERICAL_FAIL=0, INCONCLUSIVE=0)
    covered, incomplete = set(), set()
    arithmetic_complete = True
    def issue(status, reason, key=None, **details):
        issue_counts[status] += 1
        if len(issues) < 64:
            issues.append(dict(status=status, reason=reason,
                **({} if key is None else dict(state=_state_name(key))), **details))
    try:
        if not math.isfinite(t0) or not math.isfinite(t1) or not t0 < t1:
            raise ContractError('Finite increasing dense QA interval required')
        if not isinstance(budgets, dict) or set(budgets) != _expected_keys(ref_arm) or set(budgets) != _expected_keys(cand_arm):
            raise ContractError('Frozen full-state registry differs from either arm')
        for key, row in budgets.items():
            for field in ('initial', 'scale', 'budget'):
                value = np.asarray(row[field])
                if value.dtype != np.float64 or value.shape != (row['spec'].count,) or not np.isfinite(value).all():
                    raise ContractError('Missing original FP64 frozen dense QA '+field)
            if np.any(row['scale'] <= 0) or np.any(row['budget'] <= 0):
                raise ContractError('Positive original frozen scale/budget required')
        if any(key in budgets for key in _ANALYTIC):
            if not all(key in budgets for key in _ANALYTIC):
                raise ContractError('Both original analytic R/E state rows are required')
            try:
                counters['scalar_coordinates'] += _analytic_pair(ref_arm, cand_arm, t0, t1, budgets, counters, issue)
                covered.update(_ANALYTIC)
                summaries.extend(dict(state=_state_name(key), maximum_bound_over_budget=0.,
                    pieces=counters['analytic_identity_pieces'], method='same_function_same_backend_same_anchor') for key in _ANALYTIC)
            except ContractError as exc:
                incomplete.update(_ANALYTIC); issue('INCONCLUSIVE', str(exc))
        for key, row in budgets.items():
            if key in _ANALYTIC:
                continue
            count = row['spec'].count
            try:
                if any(_packed_count(arm, key) != count for arm in (ref_arm, cand_arm)):
                    raise ContractError('Frozen state row count differs from the original packed layout')
                if count == 0:
                    covered.add(key)
                    continue
                reference = _extract(ref_arm, ref_advance, key, count, t0, t1)
                candidate = _extract(cand_arm, cand_advance, key, count, t0, t1)
                ref_domain, cand_domain = _domains(key, row, ref_arm), _domains(key, row, cand_arm)
                if any(not np.array_equal(a, b) for a, b in zip(ref_domain, cand_domain)):
                    raise ContractError('Original scalar domain bindings differ between arms')
                covered.add(key); counters['scalar_coordinates'] += count
                cuts = sorted({t0, t1, *(max(t0, p.t0) for p in reference+candidate),
                               *(min(t1, p.t1) for p in reference+candidate)})
                piece_counts = {}
                for begin in cuts[:-1]:
                    owner = next(p for p in candidate if p.t0 <= begin < p.t1)
                    piece_counts[id(owner)] = piece_counts.get(id(owner), 0)+1
                if any(value > _MAX_PIECES for value in piece_counts.values()):
                    raise ContractError('Common refinement already exceeds the frozen 64-piece limit')
                max_ratio = 0.; worst = None; leaves = 0
                for begin, end in zip(cuts, cuts[1:]):
                    ref_piece = next(p for p in reference if p.t0 <= begin < p.t1)
                    cand_piece = next(p for p in candidate if p.t0 <= begin < p.t1)
                    a, b = _prepare(ref_piece, count, counters), _prepare(cand_piece, count, counters)
                    unsupported = a.unsupported or b.unsupported
                    if unsupported:
                        arithmetic_complete = False
                        issue('INCONCLUSIVE', unsupported, key)
                    arithmetic = _sum_upper([a.roundoff, b.roundoff])
                    screen = np.minimum(8192*_EPS*row['scale'], row['budget'])
                    arithmetic_failed = arithmetic > screen
                    if np.any(arithmetic_failed):
                        arithmetic_complete = False
                        index = int(np.flatnonzero(arithmetic_failed)[0])
                        issue('INCONCLUSIVE', 'Basic-operation charge exceeds unchanged 8192eps engineering screen', key,
                              index=index, charge=float(arithmetic[index]), screen=float(screen[index]))
                    stack = [(begin, end, 0)]
                    while stack:
                        lo_time, hi_time, depth = stack.pop()
                        counters['common_pieces'] += 1
                        counters['maximum_subdivision_depth'] = max(counters['maximum_subdivision_depth'], depth)
                        alo, ahi = _normalized(ref_piece, lo_time, hi_time)
                        blo, bhi = _normalized(cand_piece, lo_time, hi_time)
                        difference, candidate_range, reference_range = polynomial_pair_enclosures(
                            b.coefficients, a.coefficients, lo=blo, hi=bhi, reference_lo=alo, reference_hi=ahi)
                        difference = _expand(difference, _sum_upper([a.remainder, b.remainder, arithmetic]))
                        absolute = np.maximum(abs(difference[0]), abs(difference[1]))
                        with np.errstate(over='ignore', divide='ignore'):
                            ratio = absolute/row['budget']
                        unresolved = absolute > row['budget']
                        numeric_failure = (difference[0] > row['budget']) | (difference[1] < -row['budget'])
                        domain_undecided = np.zeros(count, bool); domain_failure = np.zeros(count, bool)
                        for values, domain in ((_range(a, alo, ahi, reference_range), ref_domain), (_range(b, blo, bhi, candidate_range), cand_domain)):
                            uncertain, failed = _domain_masks(values, domain)
                            domain_undecided |= uncertain; domain_failure |= failed
                        if np.any(numeric_failure | domain_failure):
                            mask = domain_failure if np.any(domain_failure) else numeric_failure
                            index = int(np.flatnonzero(mask)[0])
                            issue('DOMAIN_FAIL' if np.any(domain_failure) else 'NUMERICAL_FAIL',
                                'A whole subinterval is outside the original domain/budget', key,
                                index=index, start=lo_time, stop=hi_time,
                                difference_lower=float(difference[0][index]), difference_upper=float(difference[1][index]),
                                budget=float(row['budget'][index]))
                        elif np.any(unresolved | domain_undecided):
                            midpoint = lo_time+(hi_time-lo_time)/2
                            if piece_counts[id(cand_piece)] < _MAX_PIECES and lo_time < midpoint < hi_time:
                                piece_counts[id(cand_piece)] += 1
                                counters['subdivisions'] += 1
                                stack.extend(((midpoint, hi_time, depth+1), (lo_time, midpoint, depth+1)))
                                continue
                            index = int(np.flatnonzero(unresolved | domain_undecided)[0])
                            issue('INCONCLUSIVE', 'Frozen 64-piece interval enclosure limit reached without closure', key,
                                index=index, start=lo_time, stop=hi_time, bound=float(absolute[index]),
                                budget=float(row['budget'][index]), domain_unresolved=bool(domain_undecided[index]))
                        leaves += 1
                        index = int(np.argmax(ratio))
                        if float(ratio[index]) > max_ratio:
                            max_ratio = float(ratio[index]); worst = index
                summaries.append(dict(state=_state_name(key), maximum_bound_over_budget=max_ratio if math.isfinite(max_ratio) else 'infinity',
                                      worst_index=worst, pieces=leaves))
            except (ContractError, AttributeError, KeyError, TypeError, ValueError) as exc:
                arithmetic_complete = False
                incomplete.add(key); covered.discard(key); issue('INCONCLUSIVE', str(exc), key)
    except (ContractError, AttributeError, KeyError, TypeError, ValueError) as exc:
        arithmetic_complete = False
        incomplete.update(budgets if isinstance(budgets, dict) else ()); issue('INCONCLUSIVE', str(exc))
    counters['state_rows'] = len(covered)
    status = next((name for name in ('DOMAIN_FAIL', 'NUMERICAL_FAIL', 'INCONCLUSIVE')
                   if issue_counts[name]), 'PASS')
    complete = isinstance(budgets, dict) and set(budgets) == covered and not incomplete
    if not complete and status == 'PASS':
        status = 'INCONCLUSIVE'; issue(status, 'Full accepted state coverage is incomplete')
    return dict(schema=SCHEMA, status=status, complete_coverage=complete,
        scope='relative_published_state_waveforms_not_absolute_ODE_error',
        absolute_ODE_error_proved=False, original_B_unchanged=True,
        elementary_arithmetic_complete=arithmetic_complete,
        domain_scope='original_scalar_state_predicates_and_native_denominator_domains',
        remaining_gates='original_RHS_derived_ports_gap_history_events_and_full_monitor',
        arithmetic_assumptions='IEEE754_binary64_round_to_nearest_with_gradual_underflow; finite expm1(nonpositive) in [-1,0]',
        roundoff_screen='original_8192eps_engineering_screen_not_a_global_error_proof',
        interval=dict(start=float(t0) if math.isfinite(t0) else None, stop=float(t1) if math.isfinite(t1) else None),
        issues=issues, issue_counts=issue_counts, rows=summaries,
        counters=counters, wall_seconds=perf_counter()-started)
