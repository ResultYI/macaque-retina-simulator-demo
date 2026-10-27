"""Conditional functional fitting on frozen native conductance trajectories.

This is an analysis layer, not a new retinal mechanism.  Voltage uses SI units,
the retained passive equation, and the retained binary three-tap spike sampler.
Observed spikes are never synthesized and never feed back into voltage.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from numba import njit

BIN_SECONDS = .001
CACHE_STEP_SECONDS = .0001
STATES = np.asarray(list(product((0, 1), repeat=3)), dtype=np.int64)
NEXT0 = np.asarray([list(map(tuple, STATES)).index((0, int(s[0]), int(s[1]))) for s in STATES])
NEXT1 = np.asarray([list(map(tuple, STATES)).index((1, int(s[0]), int(s[1]))) for s in STATES])


@dataclass(frozen=True)
class MembraneParameters:
    C: float
    gL: float
    EL: float
    EE: float
    EI: float
    alpha: float
    mu: float
    beta: float
    history_kernel_V: tuple
    dt: float = BIN_SECONDS

    @classmethod
    def from_dict(cls, value):
        names = ('C', 'gL', 'EL', 'EE', 'EI', 'alpha', 'mu', 'beta')
        if any(k not in value for k in names):
            raise ValueError('All frozen SI membrane/output parameters are required')
        kernel = tuple(value['history_kernel_V'])
        result = cls(**{k: float(value[k]) for k in names}, history_kernel_V=kernel,
                     dt=float(value.get('dt', BIN_SECONDS)))
        if (not all(np.isfinite(getattr(result, k)) for k in names) or
                min(result.C, result.gL, result.alpha, result.beta) <= 0 or
                result.dt != BIN_SECONDS or len(kernel) != 3 or not np.isfinite(kernel).all()):
            raise ValueError('Frozen native SI parameters or binary-history contract invalid')
        return result


def binary_history(counts, raw_context_ticks_100us, kernel_V):
    """Use observed bins -3,-2,-1 and strictly past live bins; never clip counts."""
    y = np.asarray(counts)
    ticks = np.asarray(raw_context_ticks_100us)
    kernel = np.asarray(kernel_V, dtype=float)
    if y.ndim != 1 or not np.isin(y, (0, 1)).all() or ticks.ndim != 1 or kernel.shape != (3,):
        raise ValueError('Binary one-dimensional observations and three history taps required')
    prior = np.bincount((ticks[(ticks >= -30) & (ticks < 0)] // 10 + 3).astype(int), minlength=3)
    if not np.isin(prior, (0, 1)).all():
        raise ValueError('Source prehistory has >1 event in a native binary bin; no clipping')
    return binary_history_from_past(y, prior[::-1], kernel)


def binary_history_from_past(counts, past_1_2_3, kernel_V):
    """Prepared source prehistory is ordered newest to oldest, as native taps."""
    y, prior, kernel = np.asarray(counts), np.asarray(past_1_2_3), np.asarray(kernel_V, dtype=float)
    if (y.ndim != 1 or prior.shape != (3,) or kernel.shape != (3,) or
            not np.isin(y, (0, 1)).all() or not np.isin(prior, (0, 1)).all()):
        raise ValueError('Binary source counts and actual newest-first three-bin history required')
    padded = np.concatenate((prior[::-1], y))
    history = np.zeros(len(y), dtype=float)
    for lag, weight in enumerate(kernel, 1):
        history += weight * padded[3-lag:3-lag+len(y)]
    return history


def background_voltage(parameters, background_gE_S, background_gI_S, excitation=1., inhibition=1.):
    p = parameters
    ge, gi = excitation * float(background_gE_S), inhibition * float(background_gI_S)
    if not np.isfinite([ge, gi]).all() or min(ge, gi) < 0:
        raise ValueError('Identified nonnegative native background conductances are required')
    return (p.gL*p.EL + ge*p.EE + gi*p.EI)/(p.gL+ge+gi)


@njit(nogil=True)
def _passive_rk4(ge, gi, h, C, gL, EL, EE, EI, V0):
    out = np.empty(len(ge), dtype=np.float64)
    out[0] = V0
    for k in range(len(ge)-1):
        gm_e, gm_i = .5*(ge[k]+ge[k+1]), .5*(gi[k]+gi[k+1])
        def f(v, e, i):
            return (gL*(EL-v)+e*(EE-v)+i*(EI-v))/C
        v = out[k]
        a = f(v, ge[k], gi[k])
        b = f(v+.5*h*a, gm_e, gm_i)
        c = f(v+.5*h*b, gm_e, gm_i)
        d = f(v+h*c, ge[k+1], gi[k+1])
        out[k+1] = v+h*(a+2*b+2*c+d)/6
    return out


def candidate_voltage(relative_time, gE_S, gI_S, parameters, *, excitation=1., inhibition=1.,
                      background_gE_S, background_gI_S):
    """RK4 on the declared linear100us conductance nodes, including endpoint."""
    t, ge, gi = (np.asarray(x, dtype=np.float64) for x in (relative_time, gE_S, gI_S))
    if (t.ndim != 1 or len(t) < 2 or ge.shape != t.shape or gi.shape != t.shape or
            not all(np.isfinite(x).all() for x in (t, ge, gi)) or min(ge.min(), gi.min()) < 0 or
            abs(t[0]) > 1e-12 or not np.allclose(np.diff(t), CACHE_STEP_SECONDS, rtol=0, atol=1e-12)):
        raise ValueError('Finite native100us conductance nodes from0 including endpoint required')
    if not .5 <= excitation <= 2 or not .5 <= inhibition <= 2:
        raise ValueError('Multiplier outside the predeclared engineering domain')
    p = parameters
    v0 = background_voltage(p, background_gE_S, background_gI_S, excitation, inhibition)
    voltage = _passive_rk4(ge*excitation, gi*inhibition, CACHE_STEP_SECONDS,
                           p.C, p.gL, p.EL, p.EE, p.EI, v0)
    if not np.isfinite(voltage).all():
        raise ValueError('Nonfinite candidate membrane trajectory')
    return voltage


def prepare_event_conductance(relative_time, gE_S, gI_S, event_time, event_gE_S, event_gI_S):
    """Preserve native AC jumps; duplicate event rows are ordered pre then post.

    BC terminal release is continuous and is interpolated from its actual nodes.
    The residual AC activation is a held history state.  A separate event mesh
    therefore keeps its recorded right limit until the next actual transition.
    This is an optional cache representation, never an inferred timing offset.
    """
    t, ge, gi, et, ee, ei = (np.asarray(x, dtype=np.float64) for x in
                            (relative_time, gE_S, gI_S, event_time, event_gE_S, event_gI_S))
    if (t.ndim != 1 or len(t) < 2 or ge.shape != t.shape or gi.shape != t.shape or
            et.ndim != 1 or ee.shape != et.shape or ei.shape != et.shape or not len(et) or
            not all(np.isfinite(x).all() for x in (t, ge, gi, et, ee, ei)) or
            np.any(np.diff(t) <= 0) or np.any(np.diff(et) < 0) or
            min(ge.min(), gi.min(), ee.min(), ei.min()) < 0 or
            et[0] < t[0]-1e-12 or et[-1] > t[-1]+1e-12):
        raise ValueError('Ordered finite native conductance events in the actual cache interval required')
    # The last row at exactly the same event stamp is the post-transition value.
    last = np.r_[np.diff(et) != 0, True]
    et, ee, ei = et[last], ee[last], ei[last]
    inside = (et >= t[0]) & (et <= t[-1])
    et, ee, ei = et[inside], ee[inside], ei[inside]
    if not len(et):
        raise ValueError('No actual AC event lies inside the cached interval')
    mesh = np.unique(np.r_[t, et])
    excitation = np.interp(mesh, t, ge)
    excitation[np.searchsorted(mesh, et)] = ee
    event_index = np.searchsorted(et, mesh, side='right')-1
    inhibition = np.full(len(mesh), gi[0])
    present = event_index >= 0
    inhibition[present] = ei[event_index[present]]
    return mesh, excitation, inhibition, np.searchsorted(mesh, t)


@njit(nogil=True)
def _passive_event_rk4(t, ge, gi, C, gL, EL, EE, EI, V0):
    out = np.empty(len(t), dtype=np.float64)
    out[0] = V0
    for k in range(len(t)-1):
        h = t[k+1]-t[k]
        gm_e = .5*(ge[k]+ge[k+1])
        held_i = gi[k]
        def f(v, e):
            return (gL*(EL-v)+e*(EE-v)+held_i*(EI-v))/C
        v = out[k]
        a = f(v, ge[k])
        b = f(v+.5*h*a, gm_e)
        c = f(v+.5*h*b, gm_e)
        d = f(v+h*c, ge[k+1])
        out[k+1] = v+h*(a+2*b+2*c+d)/6
    return out


def candidate_voltage_events(prepared_events, parameters, *, excitation=1., inhibition=1.,
                             background_gE_S, background_gI_S):
    """The same passive equation with recorded AC ZOH event boundaries."""
    if not .5 <= excitation <= 2 or not .5 <= inhibition <= 2:
        raise ValueError('Multiplier outside the predeclared engineering domain')
    t, ge, gi, observed_indices = prepared_events
    p = parameters
    v0 = background_voltage(p, background_gE_S, background_gI_S, excitation, inhibition)
    voltage = _passive_event_rk4(t, ge*excitation, gi*inhibition,
                               p.C, p.gL, p.EL, p.EE, p.EI, v0)[observed_indices]
    if not np.isfinite(voltage).all():
        raise ValueError('Nonfinite event-aware candidate membrane trajectory')
    return voltage


def native_voltage_equivalence(candidate, native, *, rtol=1e-6, atol=1e-9):
    a, b = np.asarray(candidate), np.asarray(native)
    if a.shape != b.shape or not np.isfinite(b).all():
        raise ValueError('Same-grid finite native voltage required for equivalence')
    delta = abs(a-b)
    ratio = float(np.max(delta/(atol+rtol*abs(b))))
    return dict(passed=ratio <= 1, maximum_absolute_difference_V=float(delta.max()),
                maximum_difference_over_native_solver_scale=ratio, rtol=rtol, atol=atol)


@njit(nogil=True)
def _softplus(z):
    return z+np.log1p(np.exp(-z)) if z > 0 else np.log1p(np.exp(z))


@njit(nogil=True)
def _teacher_nll(voltage_bin_start, observed, history, indices, alpha, mu, beta):
    loss = 0.
    for trial in range(observed.shape[0]):
        for j in indices:
            z = (voltage_bin_start[j]+history[trial, j]-mu)/beta
            sp = _softplus(z)
            a = alpha*BIN_SECONDS*sp
            if observed[trial, j]:
                # log softplus remains representable for extremely negative z.
                log_a = np.log(alpha*BIN_SECONDS)+(z if z < -35 else np.log(sp))
                log_p = log_a-.5*np.exp(log_a)+np.exp(2*log_a)/24 if log_a < -20 else np.log(-np.expm1(-a))
                loss -= log_p
            else:
                loss += a
    return loss


def teacher_forced_nll(voltage_bin_start, observed, history, indices, parameters):
    v, y, h = np.asarray(voltage_bin_start), np.asarray(observed), np.asarray(history)
    idx = np.asarray(indices, dtype=np.int64)
    if (v.ndim != 1 or y.ndim != 2 or y.shape != h.shape or y.shape[1] != len(v) or
            not np.isin(y, (0, 1)).all() or not np.isfinite(v).all() or not np.isfinite(h).all() or
            idx.ndim != 1 or not len(idx) or np.any(idx < 0) or np.any(idx >= len(v))):
        raise ValueError('Native-bin voltage, actual binary repeats, strict history and scored indices required')
    return float(_teacher_nll(v, y, h, idx, parameters.alpha, parameters.mu, parameters.beta))


def teacher_probability(voltage_bin_start, history, parameters):
    p = parameters
    a = p.alpha*BIN_SECONDS*np.logaddexp(0., (np.asarray(voltage_bin_start)[None, :]+history-p.mu)/p.beta)
    return -np.expm1(-a)


def stationary_history(voltage, parameters):
    p = parameters
    h = STATES @ np.asarray(p.history_kernel_V)
    probability = -np.expm1(-p.alpha*BIN_SECONDS*np.logaddexp(0., (voltage+h-p.mu)/p.beta))
    matrix = np.zeros((8, 8))
    matrix[np.arange(8), NEXT0] += 1-probability
    matrix[np.arange(8), NEXT1] += probability
    left = matrix.T-np.eye(8)
    left[-1] = 1
    rhs = np.zeros(8)
    rhs[-1] = 1
    law = np.linalg.solve(left, rhs)
    if law.min() < -1e-13 or np.max(abs(law @ matrix-law)) > 1e-12 or abs(law.sum()-1) > 1e-12:
        raise ValueError('Constant-background history stationary law failed')
    return law


@njit(nogil=True)
def _free_probability(v, alpha, mu, beta, kernel, initial, states, next0, next1):
    law = initial.copy()
    out = np.empty(len(v))
    mass_error = 0.
    for j in range(len(v)):
        nxt = np.zeros(8)
        mean = 0.
        for k in range(8):
            h = states[k, 0]*kernel[0]+states[k, 1]*kernel[1]+states[k, 2]*kernel[2]
            prob = -np.expm1(-alpha*BIN_SECONDS*_softplus((v[j]+h-mu)/beta))
            mean += law[k]*prob
            nxt[next0[k]] += law[k]*(1-prob)
            nxt[next1[k]] += law[k]*prob
        out[j] = mean
        law = nxt
        mass_error = max(mass_error, abs(law.sum()-1))
    return out, mass_error


def free_probability(voltage_bin_start, parameters, *, background_V):
    p = parameters
    v = np.asarray(voltage_bin_start, dtype=float)
    if v.ndim != 1 or not np.isfinite(v).all():
        raise ValueError('Finite native-bin voltage required')
    law = stationary_history(background_V, p)
    result, error = _free_probability(v, p.alpha, p.mu, p.beta, np.asarray(p.history_kernel_V), law, STATES, NEXT0, NEXT1)
    if error > 1e-12 or not np.isfinite(result).all() or np.any(result < -1e-13) or np.any(result > 1+1e-13):
        raise ValueError('Free binary-history probability conservation failed')
    return result


def constant_nll(probability, observed):
    y = np.asarray(observed)
    if not 0 <= probability <= 1 or not np.isin(y, (0, 1)).all():
        raise ValueError('A training-estimated Bernoulli probability and binary observations are required')
    spikes, size = int(y.sum()), y.size
    if probability == 0:
        return 0. if spikes == 0 else float('inf')
    if probability == 1:
        return 0. if spikes == size else float('inf')
    return float(-spikes*np.log(probability)-(size-spikes)*np.log1p(-probability))


def psth_metrics(observed, expected_probability, indices, *, bin_seconds=.020):
    """No smoothing/lag search/rescaling; a short last bin keeps its exposure."""
    y = np.asarray(observed)
    expected = np.asarray(expected_probability)
    idx = np.asarray(indices, dtype=int)
    width = int(round(bin_seconds/BIN_SECONDS))
    if (y.ndim != 2 or expected.shape != (y.shape[1],) or len(idx) < 2 or
            not np.all(np.diff(idx) == 1) or width < 1 or abs(width*BIN_SECONDS-bin_seconds) > 1e-12):
        raise ValueError('Contiguous source scoring bins and declared rectangular PSTH width required')
    chunks = [idx[j:j+width] for j in range(0, len(idx), width)]
    exposure = np.asarray([len(c)*BIN_SECONDS for c in chunks])
    o = np.asarray([y[:, c].sum()/(len(y)*e) for c, e in zip(chunks, exposure)])
    prediction = np.asarray([expected[c].sum()/e for c, e in zip(chunks, exposure)])
    times = np.asarray([(c[0]+len(c)/2)*BIN_SECONDS for c in chunks])
    observed_rate = float(y[:, idx].mean()/BIN_SECONDS)
    predicted_rate = float(expected[idx].mean()/BIN_SECONDS)
    observed_std = float(np.sqrt(np.average((o-observed_rate)**2, weights=exposure)))
    predicted_std = float(np.sqrt(np.average((prediction-predicted_rate)**2, weights=exposure)))
    corr = float(np.corrcoef(o, prediction)[0, 1]) if np.std(o) and np.std(prediction) else None
    result = dict(observed_rate_Hz=observed_rate, predicted_rate_Hz=predicted_rate,
                  signed_rate_error_Hz=predicted_rate-observed_rate,
                  absolute_rate_error_Hz=abs(predicted_rate-observed_rate),
                  relative_rate_error=(abs(predicted_rate-observed_rate)/observed_rate if observed_rate else None),
                  PSTH_RMSE_Hz=float(np.sqrt(np.average((prediction-o)**2, weights=exposure))),
                  PSTH_Pearson=corr, observed_modulation_std_Hz=observed_std,
                  predicted_modulation_std_Hz=predicted_std,
                  predicted_to_observed_modulation_std_ratio=(predicted_std/observed_std if observed_std else None),
                  modulation_std_absolute_error_Hz=abs(predicted_std-observed_std),
                  observed_peak_to_peak_Hz=float(np.ptp(o)), predicted_peak_to_peak_Hz=float(np.ptp(prediction)),
                  peak_amplitude_absolute_error_Hz=None, peak_time_absolute_error_seconds=None,
                  peak_status='NOT_APPLICABLE: no predeclared event in the natural movie',
                  bin_seconds=bin_seconds, smoothing='none', lag_search=False, rescaling=False,
                  ON_OFF_direction='NOT_INFERRED_FROM_NATURAL_MOVIE_CORRELATION; requires frozen controlled-polarity check')
    # Diagnostic lag scanning does not shift the PSTH, objective, or scores.
    lag_steps = int(round(.100/bin_seconds))
    lag_correlations = []
    for lag in range(-lag_steps, lag_steps+1):
        a, b = (o[-lag:], prediction[:lag]) if lag < 0 else (o[:-lag], prediction[lag:]) if lag > 0 else (o, prediction)
        lag_correlations.append(float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) and np.std(b) else None)
    result['diagnostic_lag_seconds'] = [k*bin_seconds for k in range(-lag_steps, lag_steps+1)]
    result['diagnostic_lag_Pearson'] = lag_correlations
    result['diagnostic_lag_not_used_to_shift_or_select'] = True
    result['diagnostic_lag_definition'] = 'Positive lag compares observed(t) with prediction(t+lag); descriptive only'
    curves = dict(time_seconds=times, observed_PSTH_Hz=o, expected_PSTH_Hz=prediction,
                  bin_exposure_seconds=exposure)
    return result, curves


def stimulus_covariance_metrics(observed, expected_probability, stimulus_projection_1ms, indices, *, bin_seconds=.020):
    """Fixed projection supplied by the input contract, never estimated from spikes."""
    y, expected = np.asarray(observed), np.asarray(expected_probability)
    projection, idx = np.asarray(stimulus_projection_1ms), np.asarray(indices, dtype=int)
    if projection.shape != (y.shape[1],) or not np.isfinite(projection).all():
        raise ValueError('Finite fixed source-stimulus projection on every native bin required')
    width = int(round(bin_seconds/BIN_SECONDS))
    chunks = [idx[j:j+width] for j in range(0, len(idx), width)]
    exposure = np.asarray([len(c)*BIN_SECONDS for c in chunks])
    stimulus = np.asarray([projection[c].mean() for c in chunks])
    actual = np.asarray([y[:, c].sum()/(len(y)*e) for c, e in zip(chunks, exposure)])
    model = np.asarray([expected[c].sum()/e for c, e in zip(chunks, exposure)])
    center = stimulus-np.average(stimulus, weights=exposure)
    actual_cov = float(np.average(center*(actual-np.average(actual, weights=exposure)), weights=exposure))
    model_cov = float(np.average(center*(model-np.average(model, weights=exposure)), weights=exposure))
    return dict(observed_stimulus_covariance=actual_cov, predicted_stimulus_covariance=model_cov,
                stimulus_covariance_absolute_error=abs(model_cov-actual_cov),
                status='FIXED_INPUT_PROJECTION_DESCRIPTIVE_COVARIANCE', bin_seconds=bin_seconds,
                physiological_ON_OFF_classification=False, gray_subtraction=False, lag_search=False)
