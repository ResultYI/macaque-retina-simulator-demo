"""Matched multichannel optical input baselines; no fitted spatial surrogate RF."""
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from retina_sim import stimulus_baselines_v1 as base
from retina_sim.spatial_group_graph import digest
from tools.durable_research_runtime_v1 import atomic_json
from tools.gin_hc2_bc1_context_support_v2 import ROOT, OUT, read, background_directory
from tools.gin69_voltage_fit_support_v2 import counts_prefix, metrics, SEED


def hold_features(drive, frame_times, background, rows, destination, guard):
    """Exact causal FIR weights on a 100us held signal, sampled at 1ms."""
    drive = np.asarray(drive, dtype=np.float64)
    frame_times = np.asarray(frame_times, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    if drive.ndim != 2 or drive.shape[0] != len(frame_times) or background.shape != drive.shape[1:]:
        raise ValueError('Registered input shape differs')
    if frame_times[0] != 0 or np.any(np.diff(frame_times) <= 0):
        raise ValueError('Strict increasing actual frame times from zero required')
    if not np.isfinite(drive).all() or not np.isfinite(background).all():
        raise ValueError('Finite actual inputs required')
    sample_times = np.arange(rows * 10) * .0001
    starts = np.searchsorted(sample_times, frame_times, side='left')
    stops = np.r_[starts[1:], rows * 10]
    basis = base.stimulus_basis()
    integral = np.vstack((np.zeros((1, 16)), np.cumsum(basis, axis=0)))
    row_parts, col_parts, lower_parts, upper_parts = [], [], [], []
    for first in range(0, rows, 1000):
        guard()
        for j in range(first, min(first + 1000, rows)):
            n = j * 10
            lower_sample = max(0, n - 4999)
            left = np.searchsorted(stops, lower_sample, side='right')
            right = np.searchsorted(starts, n, side='right')
            columns = np.arange(left, right)
            # In particular, no sample after n contributes to row j.
            lower_lag = n - np.minimum(n + 1, stops[columns]) + 1
            upper_lag = n - np.maximum(lower_sample, starts[columns]) + 1
            if np.any(lower_lag < 0) or np.any(upper_lag > 5000):
                raise ValueError('Causal FIR support violated')
            row_parts.append(np.full(len(columns), j, dtype=np.int32))
            col_parts.append(columns.astype(np.int32))
            lower_parts.append(lower_lag.astype(np.int32))
            upper_parts.append(upper_lag.astype(np.int32))
    rr, cc, ll, uu = (np.concatenate(parts) for parts in (row_parts, col_parts, lower_parts, upper_parts))
    missing = np.minimum(np.arange(rows) * 10 + 1, 5000)
    channels = drive.shape[1]
    for k in range(16):
        guard()
        weights = integral[uu, k] - integral[ll, k]
        operator = sparse.csr_matrix((weights, (rr, cc)), shape=(rows, len(frame_times)))
        destination[:, k * channels:(k + 1) * channels] = operator @ drive
        destination[:, k * channels:(k + 1) * channels] += (integral[5000, k] - integral[missing, k])[:, None] * background
    if hasattr(destination, 'flush'):
        destination.flush()


def ensure_features(folder, guard):
    location = read(OUT / 'FORMAL_INPUT_LOCATION.json')
    source = Path(location['path'])
    receipt = read(source / 'RESULT.json')
    payload = source / 'NATIVE_INPUTS.npz'
    background = background_directory() / 'MOVIE_NATIVE_INPUTS.npz'
    if digest(source / 'RESULT.json') != location['result_sha256'] or digest(payload) != receipt['array_sha256']:
        raise ValueError('Exact network input identity changed')
    identity = dict(source_receipt_sha256=location['result_sha256'], array_sha256=receipt['array_sha256'],
        background_sha256=digest(background), code_sha256=digest(Path(__file__)),
        basis_code_sha256=digest(ROOT / 'retina_sim/stimulus_baselines_v1.py'),
        sampling='100us right-side held input; output bin starts every1ms; actual frame times',
        prehistory='actual equilibrium native_background, not first live frame',
        columns='basis-major then unchanged local_cone_rows; all spatial channels; no fitted RF/PCA',
        sealed_time_features_generated=False,
        inherited_scale='Approved fixed first-minute brightness calibration uses stimulus covariates; no response-derived or per-window normalization')
    path = folder / 'MATCHED_FEATURES.npy'
    stamp = folder / 'FEATURES_IDENTITY.json'
    if stamp.exists():
        saved = read(stamp)
        if saved['identity'] != identity or digest(path) != saved['feature_sha256']:
            raise ValueError('Existing matched feature identity differs')
        return np.load(path, mmap_mode='r', allow_pickle=False)
    if path.exists() or path.with_suffix('.partial.npy').exists():
        raise FileExistsError('Uncommitted features preserved; inspect before recovery')
    began = time.perf_counter()
    with np.load(payload, allow_pickle=False) as z:
        drive, clock = z['drive'], z['frame_start_seconds']
    with np.load(background, allow_pickle=False) as z:
        bg = z['native_background']
    if drive.shape != (6750, 493) or not np.array_equal(clock, np.arange(6750) / 150):
        raise ValueError('Frozen 45s,493-channel actual input expected')
    temporary = path.with_suffix('.partial.npy')
    dest = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float64, shape=(45000, 493 * 16))
    hold_features(drive, clock, bg, 45000, dest, guard)
    del dest
    temporary.replace(path)
    atomic_json(stamp, dict(identity=identity, feature_sha256=digest(path), shape=[45000, 7888],
        wall_seconds=time.perf_counter() - began))
    return np.load(path, mmap_mode='r', allow_pickle=False)


def shared_objective(theta, x, history, y, ridge):
    """Same expanded-repeat likelihood without copying the spatial design 6 times."""
    width = x.shape[1]
    signal = theta[0] + x @ theta[1:width + 1]
    if history is None:
        z = np.broadcast_to(signal, y.shape)
    else:
        z = signal[None, :] + np.einsum('rti,i->rt', history, theta[width + 1:])
    loss, derivative = base.stable_terms(z, base.ALPHA, y)
    grad = np.empty_like(theta)
    grad[0] = derivative.mean()
    grad[1:width + 1] = x.T @ derivative.sum(axis=0) / y.size
    if history is not None:
        grad[width + 1:] = np.einsum('rti,rt->i', history, derivative) / y.size
    grad[1:] += ridge * theta[1:]
    return float(loss.mean() + .5 * ridge * (theta[1:] @ theta[1:])), grad


def fit(raw, counts, history, mask, family, ridge, guard):
    began = time.perf_counter()
    x = np.asarray(raw[mask]).copy()
    mean, scale = x.mean(axis=0), x.std(axis=0)
    constant = scale < 1e-12
    scale[constant] = 1.
    x -= mean
    x /= scale
    h = None
    if family == 'S_GLM':
        h = history[:, mask].copy()
        hm, hs = h.mean(axis=(0, 1)), h.std(axis=(0, 1))
        hc = hs < 1e-12
        hs[hc] = 1.
        h -= hm
        h /= hs
        mean, scale, constant = np.r_[mean, hm], np.r_[scale, hs], np.r_[constant, hc]
    y = counts[:, mask]
    p = float(y.mean())
    if not 0 < p < 1:
        raise ValueError('Nondegenerate original counts required')
    theta = np.zeros(len(mean) + 1)
    theta[0] = np.log(np.expm1(-np.log1p(-p) / .09))
    def objective(t):
        guard()
        return shared_objective(t, x, h, y, ridge)
    result = minimize(objective, theta, jac=True, method='L-BFGS-B', options=base.OPTIONS)
    return dict(family=family, theta=result.x.tolist(), mean=mean.tolist(), scale=scale.tolist(),
        constant_columns=np.flatnonzero(constant).tolist(), ridge=ridge, parameters=len(theta),
        converged=bool(result.success), message=str(result.message), iterations=int(result.nit),
        function_evaluations=int(result.nfev), raw_NLL=shared_objective(result.x, x, h, y, 0)[0],
        penalized_objective=float(result.fun), starts=1, wall_seconds=time.perf_counter() - began)


def linear_signal(raw, fitted):
    theta, mean, scale = (np.asarray(fitted[k]) for k in ('theta', 'mean', 'scale'))
    width = raw.shape[1]
    weights = theta[1:width + 1] / scale[:width]
    # No repeated large normalized design for evaluation.
    signal = raw @ weights + theta[0] - mean[:width] @ weights
    if fitted['family'] == 'S_GLM':
        signal -= (mean[width:] / scale[width:]) @ theta[width + 1:]
        kernel_weights = theta[width + 1:] / scale[width:]
    else:
        kernel_weights = None
    return signal, kernel_weights


def conditional(raw, counts, history, fitted):
    signal, hw = linear_signal(raw, fitted)
    z = np.broadcast_to(signal, counts.shape) if hw is None else signal[None, :] + np.einsum('rti,i->rt', history, hw)
    return base.stable_terms(z, base.ALPHA, counts)[0]


def paired(counts, probabilities, losses):
    """Inherited1000 paired repeat/200ms block draws, dimensions adapted to6x15s."""
    # A spawned stream avoids reusing the free-prediction random numbers.
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(SEED).spawn(1)[0]))
    trials = rng.integers(0, 6, (1000, 6))
    blocks = rng.integers(0, 75, (1000, 75))
    observed = counts[:, 30000:45000].reshape(6, 75, 10, 20).mean(axis=3) * 1000
    rates = {name: p[30000:45000].reshape(75, 10, 20).mean(axis=2) * 1000 for name, p in probabilities.items()}
    nll = {name: a[:, 30000:45000].reshape(6, 75, 200).mean(axis=2) for name, a in losses.items()}
    values = {name: [] for name in ('S_LN', 'S_GLM')}
    for tr, bl in zip(trials, blocks):
        obs = observed[tr][:, bl].mean(axis=0).ravel()
        network = rates['V_GLM'][bl].ravel()
        nr = np.corrcoef(obs, network)[0, 1] if min(obs.std(), network.std()) > 1e-12 else np.nan
        for name in values:
            other = rates[name][bl].ravel()
            sr = np.corrcoef(obs, other)[0, 1] if min(obs.std(), other.std()) > 1e-12 else np.nan
            values[name].append([(nll[name] - nll['V_GLM'])[tr][:, bl].mean(),
                np.mean((other - obs) ** 2) - np.mean((network - obs) ** 2), nr - sr])
    result = {}
    for name, array in values.items():
        intervals = {}
        for k, key in enumerate(('delta_NLL', 'delta_MSE', 'delta_r')):
            a = np.asarray(array)[:, k]
            a = a[np.isfinite(a)]
            intervals[key + '_95'] = np.quantile(a, [.025, .5, .975]).tolist() if len(a) else None
        result[name] = dict(intervals=intervals, positive='favors V_GLM',
            scope='Descriptive paired repeat/200ms block resampling of one fixed15s source; not independent stimulus generalization',
            draws=1000, seed_entropy=SEED, seed_spawn_key=[0], MC_random_stream_shared=False,
            MC_prediction_noise_in_bootstrap=False)
    return result, trials, blocks


def execute(output, guard, mode):
    began = time.perf_counter()
    folder = OUT / 'matched_spatial_baselines'
    folder.mkdir(exist_ok=True)
    protocol = dict(schema='gin69_actual_multichannel_input_LN_GLM_v1',
        code_sha256=digest(Path(__file__)), base_code_sha256=digest(ROOT / 'retina_sim/stimulus_baselines_v1.py'),
        split_sha256=digest(OUT / 'SPLIT_CONTRACT.json'), source_cell='69#6',
        inputs='All493 actual registered native Cone drive channels, same150Hz holds and100us sampling as network; no response-derived spatial RF',
        input_unit='native reference-equivalent td, engineering LMS bridge; not independently calibrated absorption',
        temporal_bases=16, temporal_support_ms=[0, 499.9], history_bases=13, history_lags_ms=[1, 200],
        parameters=dict(S_LN=7889, S_GLM=7902, V_GLM=30),
        link='lambda=90softplus(z), p=1-exp(-.001lambda); Bernoulli point process',
        prehistory='actual native gray equilibrium input; zero spikes before0; no split reset',
        train_seconds=[0, 30], development_seconds=[30, 45], sealed_seconds=[45, 60], sealed_response_access=False,
        prior_access='Source metadata/repeat counts and fixed first-minute brightness covariates were previously inspected; not retrospectively blind-sealed',
        CV='three10s leave-block-out folds, all6 repeats same mask, purge200ms after held block in fit',
        ridge=list(base.GRID), optimizer=base.OPTIONS, starts=1,
        selection='smallest mean held-block conditionalNLL; strongest ridge within1e-8; no expanded search',
        normalization='training rows only in each fold/final; each channel/basis and history column independently',
        free=dict(paths=4096, batches=16, seed=SEED, observed_evaluation_history=False),
        resampling=dict(draws=1000, seed_entropy=SEED, seed_spawn_key=[0], MC_random_stream_shared=False,
            repeats=6, blocks=75, block_ms=200, PSTH_ms=20),
        comparison='Matched input information, different predictor capacities; cannot isolate physiological structural contribution',
        FIT_READY=False, background_qualified=False, production_adopted=False)
    path = folder / 'PROTOCOL_FROZEN.json'
    if path.exists():
        if read(path) != protocol:
            raise ValueError('Frozen baseline protocol changed')
    else:
        atomic_json(path, protocol)
    raw = ensure_features(folder, guard)
    n = 30000 if mode == 'train' else 45000
    counts, evidence = counts_prefix(n)
    history = base.history_features(counts)
    if mode == 'train':
        frozen_path = folder / 'MODELS_FROZEN.json'
        if frozen_path.exists():
            raise FileExistsError('Original baseline fits retained')
        models = {}
        for family in ('S_LN', 'S_GLM'):
            selection = []
            for ridge in base.GRID:
                folds = []
                for hold in range(3):
                    lo, hi = hold * 10000, (hold + 1) * 10000
                    mask = np.ones(30000, dtype=bool)
                    mask[lo:min(30000, hi + 200)] = False
                    fitted = fit(raw[:30000], counts, history, mask, family, ridge, guard)
                    loss = conditional(raw[lo:hi], counts[:, lo:hi], history[:, lo:hi], fitted)
                    folds.append(dict(block=hold, model=fitted, NLL=float(loss.mean()) if fitted['converged'] else None))
                    atomic_json(folder / 'TRAINING_PROGRESS.json', dict(family=family, ridge=ridge, hold=hold,
                        wall_seconds=time.perf_counter() - began, converged=fitted['converged']))
                scores = [f['NLL'] for f in folds]
                selection.append(dict(ridge=ridge, folds=folds, score=None if any(s is None for s in scores) else float(np.mean(scores))))
            atomic_json(folder / (family + '_CROSS_VALIDATION.json'), selection)
            valid = [row for row in selection if row['score'] is not None]
            if not valid:
                raise ValueError('Declared baseline ridge candidates failed convergence; no new search')
            best = min(row['score'] for row in valid)
            ridge = max(row['ridge'] for row in valid if row['score'] <= best + 1e-8)
            fitted = fit(raw[:30000], counts, history, np.ones(30000, dtype=bool), family, ridge, guard)
            if not fitted['converged']:
                raise ValueError('Fixed final baseline optimizer failed; preserve evidence')
            models[family] = fitted
            atomic_json(folder / (family + '_MODEL_FROZEN.json'), fitted)
        atomic_json(frozen_path, dict(models=models, protocol_sha256=digest(path),
            features_identity_sha256=digest(folder / 'FEATURES_IDENTITY.json'), observations=evidence,
            development_outcomes_used=False, training_wall_seconds=time.perf_counter() - began))
        atomic_json(output / 'STIMULUS_BASELINE_RESULT.json', dict(status='MATCHED_BASELINES_FROZEN', wall_seconds=time.perf_counter() - began))
    elif mode == 'evaluate':
        frozen_path = folder / 'MODELS_FROZEN.json'
        freeze = digest(frozen_path)
        frozen = read(frozen_path)
        if frozen['protocol_sha256'] != digest(path) or frozen['features_identity_sha256'] != digest(folder / 'FEATURES_IDENTITY.json'):
            raise ValueError('Frozen inputs/model mismatch')
        result_path = folder / 'FIXED_DEVELOPMENT_RESULT.json'
        if result_path.exists():
            raise FileExistsError('Fixed baseline evaluation already completed')
        with np.load(OUT / 'readout/FIXED_DEVELOPMENT_PREDICTIONS.npz', allow_pickle=False) as z:
            if not np.array_equal(z['observed_counts'], counts):
                raise ValueError('Matched evaluation observations differ')
            probabilities = dict(V_GLM=z['probability'], constant=z['constant_probability'])
            losses = dict(V_GLM=z['conditional_loss'], constant=z['constant_conditional_loss'])
        mc, saved = {}, {}
        for family, fitted in frozen['models'].items():
            signal, hw = linear_signal(raw, fitted)
            if hw is None:
                p = -np.expm1(-.09 * np.logaddexp(0., signal))
            else:
                p, batches = base.free_glm(signal, base.history_basis() @ hw, seed=SEED, guard=guard)
                saved[family + '_MC_batch_probability'] = batches
                rates = batches[:, 30000:45000].reshape(16, 750, 20).mean(axis=2) * 1000
                se = rates.std(axis=0, ddof=1) / 4
                mc[family] = dict(paths=4096, batches=16, mean20ms_rate_SE_Hz=float(se.mean()), max20ms_rate_SE_Hz=float(se.max()))
            probabilities[family] = p
            losses[family] = conditional(raw, counts, history, fitted)
        intervals, trials, blocks = paired(counts, probabilities, losses)
        for name in probabilities:
            saved[name + '_probability'] = probabilities[name]
            saved[name + '_conditional_loss'] = losses[name]
        if digest(frozen_path) != freeze:
            raise ValueError('Baseline models changed during fixed evaluation')
        np.savez_compressed(folder / 'FIXED_DEVELOPMENT_PREDICTIONS.npz', **saved)
        np.savez_compressed(folder / 'PAIRED_BOOTSTRAP_INDICES.npz', repeats=trials, blocks=blocks)
        result = dict(status='MATCHED_SPATIAL_BASELINES_FIXED_DEVELOPMENT_COMPLETE',
            metrics={name: metrics(counts, p, losses[name], 30000, 45000) for name, p in probabilities.items()},
            paired_V_GLM_vs_baselines=intervals, MC=mc, frozen_models_sha256=freeze,
            protocol_sha256=digest(path), observations=evidence, wall_seconds=time.perf_counter() - began,
            sealed_values_used=False, FIT_READY=False, background_qualified=False, production_adopted=False)
        atomic_json(result_path, result)
        atomic_json(output / 'STIMULUS_BASELINE_RESULT.json', result)
    else:
        raise ValueError('Only fixed train/evaluate phases')
