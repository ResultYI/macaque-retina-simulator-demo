"""Small algebra/phase checks only; no neural replay or fitting of observations."""
from pathlib import Path
import sys
import time
import numpy as np
from scipy.signal import lfilter
from retina_sim import stimulus_baselines_v1 as base
from tools.gin69_spatial_baselines_support_v2 import hold_features, shared_objective
from tools.gin_hc2_bc1_context_support_v2 import background_directory
from tools.durable_research_runtime_v1 import atomic_json


def main(output):
    began = time.perf_counter()
    with np.load(background_directory() / 'MOVIE_NATIVE_INPUTS.npz', allow_pickle=False) as z:
        drive = z['live_native_drive'][:, :5].copy()
        bg = z['native_background'][:5].copy()
    times = np.arange(len(drive)) / 150
    rows = 50
    actual = np.empty((rows, 80))
    hold_features(drive, times, bg, rows, actual, lambda: None)
    frame = np.searchsorted(times, np.arange(rows * 10) * .0001, side='right') - 1
    reference = np.column_stack([lfilter(b, [1.], drive[frame, channel] - bg[channel])[::10] + bg[channel] * b.sum()
        for b in base.stimulus_basis().T for channel in range(5)])
    np.testing.assert_allclose(actual, reference, rtol=5e-14, atol=2e-13)
    changed = drive.copy()
    changed[4:] *= 2
    future = np.empty_like(actual)
    hold_features(changed, times, bg, rows, future, lambda: None)
    unaffected = np.arange(rows) * .001 < times[4]
    np.testing.assert_array_equal(actual[unaffected], future[unaffected])
    rng = np.random.default_rng(717)
    x = rng.normal(size=(17, 5))
    h = rng.normal(size=(6, 17, 3))
    y = (rng.random((6, 17)) < .1).astype(float)
    worst = 0.
    for hist in (None, h):
        width = 6 if hist is None else 9
        theta = rng.normal(size=width) * .2
        stimulus = np.broadcast_to(x, (6, 17, 5))
        raw = stimulus if hist is None else np.concatenate((stimulus, hist), axis=2)
        expanded = np.column_stack((np.ones(102), raw.reshape(102, width - 1)))
        value, gradient = shared_objective(theta, x, hist, y, 1e-3)
        ref_value, ref_grad = base.objective(theta, expanded, y.ravel(), 1e-3)
        np.testing.assert_allclose(value, ref_value, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(gradient, ref_grad, rtol=1e-13, atol=1e-14)
        worst = max(worst, float(np.max(abs(gradient - ref_grad))))
        for j in range(width):
            step = np.zeros(width)
            step[j] = 1e-6
            finite = (shared_objective(theta + step, x, hist, y, 1e-3)[0] - shared_objective(theta - step, x, hist, y, 1e-3)[0]) / 2e-6
            np.testing.assert_allclose(gradient[j], finite, rtol=2e-5, atol=2e-8)
    atomic_json(output / 'RESULT.json', dict(status='PASS', actual_source_prefix_frames=8,
        checked_bins=50, channels=5, all16_temporal_bases=True, max_feature_abs_error=float(np.max(abs(actual-reference))),
        max_repeat_shared_gradient_error=worst, future_input_perturbation_unchanged=True,
        objective_fixture='synthetic numbers only, not research outcomes',
        no_neural_run=True, no_observed_response_fitting=True, wall_seconds=time.perf_counter()-began))


if __name__ == '__main__':
    main(Path(sys.argv[2]))
