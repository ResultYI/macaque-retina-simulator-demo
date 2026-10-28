"""Opt-in conditional voltage readout; never replaces retinal dynamics.

The extra state is an effective causal filter, without an ion-channel claim.
Native voltage is held for 100 us; probabilities refer to 1 ms binary bins.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from scipy.special import expit

from .rgc_expectation import STATES, NEXT0, NEXT1

NATIVE_DT = .0001
BIN_DT = .001
ALPHA = 90.


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf8')).hexdigest()


def stable_terms(z, alpha, counts):
    """Bernoulli negative log likelihood and its derivative wrt z; no clipping."""
    z = np.asarray(z, float)
    k = BIN_DT * alpha
    sp = np.logaddexp(0., z)
    exposure = k * sp
    logsp = np.empty_like(z)
    small = z < -35.
    logsp[small] = z[small]
    logsp[~small] = np.log(sp[~small])
    loga = np.log(k) + logsp
    logp = np.empty_like(z)
    small = loga < -20.
    v = np.exp(loga[small])
    logp[small] = loga[small] - .5 * v + v * v / 24.
    logp[~small] = np.log(-np.expm1(-exposure[~small]))
    loss = np.where(counts, -logp, exposure)
    grad = np.where(counts, -np.exp(np.log(k) - np.logaddexp(0., -z) - exposure - logp),
                    k * expit(z))
    if not np.isfinite(loss).all() or not np.isfinite(grad).all():
        raise ValueError('Nonfinite readout likelihood; no clipping/retry')
    return loss, grad


def causal_state(x, tau_s, initial_a, derivative=False):
    """a[j] precedes x[j]; exact solution for a ZOH native input."""
    x = np.asarray(x, float)
    if x.ndim != 1 or not np.isfinite(x).all() or not 0 < tau_s:
        raise ValueError('Finite scalar native input and positive tau required')
    rho = np.exp(-NATIVE_DT / tau_s)
    delta = -np.expm1(-NATIVE_DT / tau_s)
    after, _ = lfilter([delta], [1., -rho], x, zi=[rho * initial_a])
    a = np.r_[initial_a, after]
    if not derivative:
        return a
    # derivative wrt log(tau); the shared gray initial_a is tau independent.
    forcing = rho * NATIVE_DT / tau_s * (a[:-1] - x)
    after_d, _ = lfilter([1.], [1., -rho], forcing, zi=[0.])
    return a, np.r_[0., after_d]


def free_probability(signal, eta, initial_pi, alpha=ALPHA):
    """Exact expectation over eight strictly-past three-bin histories."""
    pi = np.asarray(initial_pi, float).copy()
    if pi.shape != (8,) or np.min(pi) < 0 or abs(pi.sum() - 1) > 1e-12:
        raise ValueError('Invalid initial history law')
    h = STATES @ np.asarray(eta)
    out = np.empty(len(signal))
    for i, drive in enumerate(signal):
        q = -np.expm1(-BIN_DT * alpha * np.logaddexp(0., drive + h))
        if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError('Invalid readout probability')
        out[i] = pi @ q
        pi = (np.bincount(NEXT0, weights=pi * (1 - q), minlength=8)
              + np.bincount(NEXT1, weights=pi * q, minlength=8))
    if abs(pi.sum() - 1) > 2e-12 or np.min(pi) < 0:
        raise ValueError('History probability conservation failed')
    return out, pi


def load_model(path, *, allow_unqualified=False):
    model = json.loads(Path(path).read_text(encoding='utf-8'))
    if digest(model['core']) != model['model_id']:
        raise ValueError('Model identity mismatch')
    own_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if model['core']['inference_code_sha256'] != own_sha:
        raise ValueError('Inference implementation changed')
    if not allow_unqualified and model['qualification'] != 'RETAIN_CONDITIONAL':
        raise ValueError('Experimental model has not passed its frozen functional gates')
    return model


class TemporalReadout:
    """Chunkable expectation inference with readout-only state restoration.

    No observed spikes are accepted. This state is not a retina checkpoint.
    """
    def __init__(self, model, state=None):
        self.model = model
        c = model['core']
        self.a = float(c['initial_a'])
        self.pi = np.asarray(c['initial_history_probability'], float).copy()
        self.native_index = 0
        if state is not None:
            if state['model_id'] != model['model_id'] or state['scope'] != 'READOUT_ONLY':
                raise ValueError('Readout restore identity mismatch')
            self.a = float(state['a'])
            self.pi = np.asarray(state['history_probability'], float)
            self.native_index = int(state['native_index'])
            if (not np.isfinite(self.a) or self.native_index < 0
                    or self.pi.shape != (8,) or not np.isfinite(self.pi).all()
                    or np.min(self.pi) < 0 or abs(self.pi.sum() - 1) > 2e-12):
                raise ValueError('Invalid readout restore state')

    def predict(self, voltage_V):
        v = np.asarray(voltage_V, float)
        if (v.ndim != 1 or not np.isfinite(v).all()
                or np.any((v < -.08) | (v > 0))):
            raise ValueError('Expected native scalar RGC voltage in [-.08,0] V')
        if not len(v):
            return dict(bin_index=np.array([], int), probability=np.array([]),
                        expected_count_rate_Hz=np.array([]))
        c = self.model['core']
        x = (v - c['mean_train_V']) / c['SD_train_V']
        a = causal_state(x, c['tau_s'], self.a)
        take = (self.native_index + np.arange(len(v))) % 10 == 0
        signal = c['b'] + c['w0'] * x[take] + c['w1'] * a[:-1][take]
        p, self.pi = free_probability(signal, c['eta'], self.pi)
        indices = (self.native_index + np.flatnonzero(take)) // 10
        self.a = float(a[-1])
        self.native_index += len(v)
        return dict(bin_index=indices, probability=p, expected_count_rate_Hz=p / BIN_DT)

    def state(self):
        return dict(scope='READOUT_ONLY', model_id=self.model['model_id'], a=self.a,
                    history_probability=self.pi.tolist(), native_index=self.native_index)
