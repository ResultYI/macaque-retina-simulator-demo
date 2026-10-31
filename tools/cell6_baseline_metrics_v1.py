"""Existing Cell6 scoring and paired resampling, plus separate MC precision."""
import numpy as np
from retina_sim.stimulus_baselines_v1 import LO, HI


def free_metrics(counts, probability):
    observed = counts[:, LO:HI].reshape(3, 300, 20).mean(axis=(0, 2))*1000
    predicted = probability[LO:HI].reshape(300, 20).mean(axis=1)*1000
    correlation = None if min(predicted.std(), observed.std()) < 1e-12 else float(np.corrcoef(predicted, observed)[0, 1])
    return dict(free_r=correlation, free_MSE_Hz2=float(np.mean((predicted-observed)**2)),
        mean_Hz=float(predicted.mean()), SD_Hz=float(predicted.std()),
        observed_mean_Hz=float(observed.mean()), observed_SD_Hz=float(observed.std()))


def metrics(counts, probability, loss):
    return dict(conditional_NLL_nat_per1ms=float(loss[:, LO:HI].mean()), **free_metrics(counts, probability))


def mc_precision(counts, batch_probability):
    batches = len(batch_probability)
    values = batch_probability[:, LO:HI].reshape(batches, 300, 20).mean(axis=2)*1000
    rate_se = values.std(axis=0, ddof=1)/np.sqrt(batches)
    total = batch_probability.sum(axis=0)
    deleted = [free_metrics(counts, (total-q)/(batches-1)) for q in batch_probability]
    report = dict(method='16 independent256-path batches; delete-one-batch jackknife for metrics',
        trajectories=4096, mean_20ms_rate_SE_Hz=float(rate_se.mean()), maximum_20ms_rate_SE_Hz=float(rate_se.max()))
    for key in ('free_r', 'free_MSE_Hz2', 'mean_Hz'):
        v = [row[key] for row in deleted]
        report[key+'_MC_SE'] = None if any(x is None for x in v) else float(np.sqrt((batches-1)/batches*np.sum((v-np.mean(v))**2)))
    return report, rate_se


def paired(counts, probabilities, losses, trials, blocks):
    observed = counts[:, LO:HI].reshape(3, 30, 10, 20).mean(axis=3)*1000
    rates = {name: p[LO:HI].reshape(30, 10, 20).mean(axis=2)*1000 for name, p in probabilities.items()}
    mean_loss = {name: loss[:, LO:HI].reshape(3, 30, 200).mean(axis=2) for name, loss in losses.items()}
    values = {name: dict(delta_NLL=[], delta_MSE=[], delta_r=[]) for name in ('LN', 'GLM')}
    for tr, bl in zip(trials, blocks):
        obs = observed[tr][:, bl].mean(axis=0).ravel()
        t = rates['T'][bl].ravel()
        rt = np.corrcoef(obs, t)[0, 1] if min(obs.std(), t.std()) > 1e-12 else np.nan
        for name, record in values.items():
            other = rates[name][bl].ravel()
            ro = np.corrcoef(obs, other)[0, 1] if min(obs.std(), other.std()) > 1e-12 else np.nan
            record['delta_NLL'].append(float((mean_loss[name]-mean_loss['T'])[tr][:, bl].mean()))
            record['delta_MSE'].append(float(np.mean((other-obs)**2)-np.mean((t-obs)**2)))
            record['delta_r'].append(float(rt-ro))
    result = {}
    for name, record in values.items():
        intervals = {}
        for key, value in record.items():
            value = np.asarray(value);value = value[np.isfinite(value)]
            intervals[key+'_95'] = None if not len(value) else np.quantile(value, [.025, .5, .975]).tolist()
        signs = {k: 'T_better' if v is not None and v[0] > 0 else 'baseline_better' if v is not None and v[2] < 0 else 'uncertain' for k, v in intervals.items()}
        result[name] = dict(**intervals, direction='positive favors frozen network T', metric_conclusions=signs,
            scope='Descriptive original block/trial resampling within one fixed stimulus; repeats are not independent stimuli',
            MC_prediction_noise_in_bootstrap=False)
    return result
