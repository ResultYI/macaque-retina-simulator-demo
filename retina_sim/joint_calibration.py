"""Observation-scale joint level/waveform scores and explicit data partitions.

No optimizer, source reclassification, or invented firing-rate target. A fixed
scale is numerical weighting, not an inferred experimental noise variance.
"""
import numpy as np
from .contracts import ContractError


def observation_partition(record_ids,*,training,validation):
    """Partition whole source records; never split digitized columns as trials."""
    records=tuple(record_ids);train=tuple(training);valid=tuple(validation)
    if (len(records)!=len(set(records)) or not train or not valid or
            len(train)!=len(set(train)) or len(valid)!=len(set(valid)) or
            set(train)&set(valid) or set(train)|set(valid)!=set(records)):
        raise ContractError('Nonoverlapping complete whole-record training/validation partition required')
    return dict(training=list(train),validation=list(valid),split_unit='source_array_record',
        biological_independence_verified=False,external_independent_validation=False)


def joint_level_waveform_score(observed,predicted,*,scale):
    """Orthogonal mean-level and centered-dynamics residuals in one window.

Their sum equals MSE/scale^2. The window mean is not labelled resting potential
or adapted baseline without an experiment-specific baseline protocol.
"""
    a=np.asarray(observed,float);b=np.asarray(predicted,float)
    if (a.ndim!=1 or len(a)<2 or a.shape!=b.shape or not np.isfinite(a).all() or
            not np.isfinite(b).all() or isinstance(scale,bool) or not np.isfinite(scale) or scale<=0):
        raise ContractError('Finite equal single-trace arrays and a fixed positive observation-unit scale required')
    residual=(b-a)/scale;level=float(residual.mean());wave=residual-level
    return dict(mean_level_squared=level**2,centered_waveform_MSE=float(wave@wave/len(wave)),
        joint_MSE=float(residual@residual/len(residual)),samples=len(a),fixed_scale=float(scale),
        sample_mean_observed=float(a.mean()),sample_mean_predicted=float(b.mean()),
        scale_is_experimental_variance=False,adapted_baseline_claim=False)


def waveform_observation_summary(times,values):
    t=np.asarray(times,float);a=np.asarray(values,float)
    if t.ndim!=1 or a.shape!=t.shape or len(t)<2 or not np.isfinite(t).all() or not np.isfinite(a).all() or np.any(np.diff(t)<=0):
        raise ContractError('Explicit increasing seconds and a finite source waveform required')
    return dict(samples=len(a),relative_start_seconds=float(t[0]),relative_last_sample_seconds=float(t[-1]),
        mean_level=float(a.mean()),centered_RMS=float(np.sqrt(np.mean((a-a.mean())**2))),
        minimum=float(a.min()),maximum=float(a.max()),mean_level_is_steady_baseline=False)
