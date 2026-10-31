"""Causal stimulus LN and own-history softplus point-process GLM baselines."""
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from retina_sim.temporal_readout_v1 import stable_terms

LO, HI, ALPHA = 1000, 7000, 90.
GRID = (1e-5, 1e-4, 1e-3, 1e-2)
OPTIONS = dict(maxiter=500, maxfun=2000, maxls=40, ftol=1e-12, gtol=1e-7)


def raised_cosine(lags_ms, count, offset_ms=1.):
    time = np.log(np.asarray(lags_ms)+offset_ms)
    centers = np.linspace(time[0], time[-1], count)
    phase = (time[:, None]-centers)/(centers[1]-centers[0])*np.pi
    basis = np.where(abs(phase) < np.pi, .5*(1+np.cos(phase)), 0.)
    return basis/np.sqrt((basis*basis).sum(axis=0))


def stimulus_basis():
    return raised_cosine(np.arange(5000)*.1, 16)


def history_basis():
    result = np.zeros((200, 13))
    result[:3, :3] = np.eye(3)
    result[3:, 3:] = raised_cosine(np.arange(4, 201), 10)
    return result


def stimulus_features(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 1 or len(raw) % 10 or not np.isfinite(raw).all():
        raise ValueError('Finite100us stimulus; complete1ms bins required')
    basis = stimulus_basis()
    # Known constant gray before acquisition; no reset at the scoring-window start.
    return np.column_stack([lfilter(b, [1.], raw-raw[0])[::10]+raw[0]*b.sum() for b in basis.T])


def history_features(counts):
    return np.stack([lfilter(np.r_[0., b], [1.], counts, axis=1) for b in history_basis().T], axis=-1)


def unscaled_design(case, family):
    stim = np.broadcast_to(case['stimulus_features'], (*case['counts'].shape, 16))
    return stim if family == 'LN' else np.concatenate((stim, case['history_features']), axis=2)


def objective(theta, x, y, ridge):
    loss, gradient = stable_terms(x@theta, ALPHA, y)
    return (float(loss.mean()+.5*ridge*(theta[1:]@theta[1:])),
        x.T@gradient/len(y)+ridge*np.r_[0., theta[1:]])


def fit(cases, groups, family, ridge, guard=lambda: None):
    start = perf_counter()
    raw_x = np.concatenate([unscaled_design(cases[g], family)[:, LO:HI].reshape(-1, 16 if family == 'LN' else 29) for g in groups])
    y = np.concatenate([cases[g]['counts'][:, LO:HI].ravel() for g in groups])
    mean, scale = raw_x.mean(axis=0), raw_x.std(axis=0)
    constant = scale < 1e-12
    scale[constant] = 1.
    x = np.column_stack((np.ones(len(y)), (raw_x-mean)/scale))
    p = float(y.mean())
    if not 0 < p < 1:
        raise ValueError('Nondegenerate training observations required')
    theta = np.zeros(x.shape[1])
    theta[0] = np.log(np.expm1(-np.log1p(-p)/.09))
    def evaluate(t):
        guard()
        return objective(t, x, y, ridge)
    opt = minimize(evaluate, theta, jac=True, method='L-BFGS-B', options=OPTIONS)
    return dict(family=family, theta=opt.x.tolist(), mean=mean.tolist(), scale=scale.tolist(),
        constant_columns=np.flatnonzero(constant).tolist(), ridge=ridge,
        parameters=len(theta), converged=bool(opt.success), message=str(opt.message),
        iterations=int(opt.nit), function_evaluations=int(opt.nfev),
        penalized_objective=float(opt.fun), raw_NLL=float(objective(opt.x, x, y, 0.)[0]),
        wall_seconds=perf_counter()-start, train_groups=list(groups), starts=1)


def conditional(case, model):
    theta, mean, scale = (np.asarray(model[k]) for k in ('theta', 'mean', 'scale'))
    x = (unscaled_design(case, model['family'])-mean)/scale
    signal = theta[0]+x@theta[1:]
    loss, _ = stable_terms(signal, ALPHA, case['counts'])
    return loss


def choose_and_fit(cases, family, guard=lambda: None):
    records = []
    for ridge in GRID:
        folds = []
        for holdout in (0, 1, 2):
            model = fit(cases, [g for g in (0, 1, 2) if g != holdout], family, ridge, guard)
            score = float(conditional(cases[holdout], model)[:, LO:HI].mean()) if model['converged'] else None
            folds.append(dict(held_out_stimulus=holdout, NLL=score, fit=model))
        scores = [f['NLL'] for f in folds]
        records.append(dict(ridge=ridge, folds=folds,
            mean_heldout_NLL=None if any(s is None for s in scores) else float(np.mean(scores))))
    valid = [r for r in records if r['mean_heldout_NLL'] is not None]
    if not valid:
        raise RuntimeError('No declared regularization candidate converged; no expanded search')
    best = min(r['mean_heldout_NLL'] for r in valid)
    selected = max(r['ridge'] for r in valid if r['mean_heldout_NLL'] <= best+1e-8)
    model = fit(cases, (0, 1, 2), family, selected, guard)
    if not model['converged']:
        raise RuntimeError('Final fixed training failed; no restart or expanded search')
    return model, records


def free_glm(signal, kernel, *, seed, paths=4096, batches=16, guard=lambda: None):
    """Rao-Blackwell mean hazard over self-generated histories; never accepts data spikes."""
    if paths % batches:
        raise ValueError('Equal independent Monte Carlo batches required')
    size = len(kernel)
    history = np.zeros((paths, size), dtype=np.float64)
    weights = np.asarray([kernel[(p-1-np.arange(size)) % size] for p in range(size)])
    rng = np.random.Generator(np.random.PCG64(seed))
    curves = np.empty((batches, len(signal)))
    for t, drive in enumerate(signal):
        if t % 100 == 0:
            guard()
        pos = t % size
        z = drive+history@weights[pos]
        q = -np.expm1(-.09*np.logaddexp(0., z))
        if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError('Generated probability domain failed')
        curves[:, t] = q.reshape(batches, -1).mean(axis=1)
        history[:, pos] = rng.random(paths) < q
    return curves.mean(axis=0), curves


def predict(case, model, *, seed, guard=lambda: None):
    theta, mean, scale = (np.asarray(model[k]) for k in ('theta', 'mean', 'scale'))
    signal = theta[0]+((case['stimulus_features']-mean[:16])/scale[:16])@theta[1:17]
    if model['family'] == 'LN':
        return -np.expm1(-.09*np.logaddexp(0., signal)), None
    signal -= (mean[16:]/scale[16:])@theta[17:]
    kernel = history_basis()@(theta[17:]/scale[16:])
    return free_glm(signal, kernel, seed=seed, guard=guard)
