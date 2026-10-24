"""Exact expected native first-event counts on the supplied teacher edges.

This integrates the timestamp law in retina_sim.inner.HistoryRGC.sample_spikes;
it does not interpolate native rates, change the sampler, fit a lag, or draw RNG.
"""
import numpy as np
from retina_sim.rgc_expectation import STATES, NEXT0, NEXT1, INDEX


def expected_counts_on_edges(native, voltage, native_starts, initial_history, edges):
    voltage = np.asarray(voltage, dtype=float)
    starts = np.asarray(native_starts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    history = np.asarray(initial_history)
    dt = float(native.dt)
    n = int(native.n)
    if (not native.deterministic_state_independent_of_spikes
            or np.shape(native.history_kernel) != (3,) or dt != .001
            or starts.ndim != 1 or not len(starts)
            or voltage.shape != (n, len(starts)) or history.shape != (3, n)
            or not np.isin(history, (0, 1)).all()
            or not np.isfinite(voltage).all() or not np.isfinite(starts).all()
            or edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)
            or (len(starts) > 1 and not np.allclose(np.diff(starts), dt, rtol=0, atol=1e-12))):
        raise ValueError("Exact original 1ms three-tap sampler, native starts and fixed teacher edges required")
    if edges[0] < starts[0] - 1e-12 or edges[-1] > starts[-1] + dt + 1e-12:
        raise ValueError("Teacher window exceeds complete new-draw coverage; recapture history, never crop loss")
    alpha, mu, beta = (np.asarray(getattr(native, k), dtype=float) for k in ("alpha", "mu", "beta"))
    if any(v.shape != (n,) or not np.isfinite(v).all() for v in (alpha, mu, beta)) or np.any(alpha <= 0) or np.any(beta <= 0):
        raise ValueError("Invalid original per-cell parameters")
    offsets = STATES @ np.asarray(native.history_kernel)
    law = np.zeros((n, 8))
    for cell in range(n):
        law[cell, INDEX[tuple(int(x) for x in history[:, cell])]] = 1.
    counts = np.zeros((len(edges) - 1, n))
    mass_error = 0.
    for j, start in enumerate(starts):
        hazard = alpha[:, None] * np.logaddexp(0., (voltage[:, j, None] + offsets[None, :] - mu[:, None]) / beta[:, None])
        probability = -np.expm1(-hazard * dt)
        stop = start + dt
        k = max(0, int(np.searchsorted(edges, start, side="right") - 1))
        while k < len(counts) and edges[k] < stop:
            left = max(0., float(edges[k] - start))
            right = min(dt, float(edges[k + 1] - start))
            if right > left:
                # The native sampler uses u~U[0,1), event time -log(1-u)/hazard.
                # P(first event in [a,b)) = exp(-hazard*a)*(1-exp(-hazard*(b-a))).
                counts[k] += (law * np.exp(-hazard * left) * -np.expm1(-hazard * (right - left))).sum(axis=1)
            k += 1
        following = np.zeros_like(law)
        for state in range(8):
            following[:, NEXT0[state]] += law[:, state] * (1 - probability[:, state])
            following[:, NEXT1[state]] += law[:, state] * probability[:, state]
        law = following
        mass_error = max(mass_error, float(np.max(abs(law.sum(axis=1) - 1))))
    if mass_error > 1e-12 or not np.isfinite(counts).all():
        raise ValueError("Original history-law probability conservation failed")
    return counts, mass_error


def replay_rates_on_teacher_edges(replay, result, absolute_edges, ordered_cell_ids):
    """Adapt original BoundaryReplay voltage to exact teacher bins, retaining history."""
    times = np.asarray(result.outputs_with_units["time"].values)
    initial = replay.cache.metadata["initial"]
    columns = {}
    mass_errors = {}
    for cid, component in replay.runner.compiled.components.items():
        if component.mechanism != "HistoryConductanceRGC":
            continue
        native = component.native_model
        first = initial["first_new_draw_bin_seconds"][cid]
        grid = initial["origin"]
        first_tick = int(round((first - grid) / native.dt))
        stop_tick = int(round((times[-1] - grid) / native.dt))
        if abs(grid + first_tick * native.dt - first) > 1e-12:
            raise ValueError("Recorded first draw is not on the original native grid")
        starts = grid + np.arange(first_tick, stop_tick, dtype=float) * native.dt
        if not len(starts) or abs(starts[-1] + native.dt - times[-1]) > 1e-12:
            raise ValueError("Complete native final bin required")
        positions = np.searchsorted(times, starts)
        positions = np.minimum(positions, len(times) - 1)
        previous = np.maximum(positions - 1, 0)
        positions = np.where(abs(times[previous] - starts) < abs(times[positions] - starts), previous, positions)
        if np.any(abs(times[positions] - starts) > 1e-12):
            raise ValueError("Exact native bin-start voltage missing; no interpolation")
        aux = initial["spikes"][cid]
        history = np.asarray(aux["history"])
        if aux["last_bin_counts"] is not None:
            history = np.vstack((np.asarray(aux["last_bin_counts"])[None, :], history[:-1]))
        voltage = result.outputs_with_units[cid + ".voltage"].values[:, positions]
        counts, error = expected_counts_on_edges(native, voltage, starts, history, absolute_edges)
        for i, cell_id in enumerate(component.state_specs[0].cell_ids):
            columns[cell_id] = counts[:, i] / np.diff(absolute_edges)
        mass_errors[cid] = error
    if any(cell not in columns for cell in ordered_cell_ids):
        raise ValueError("Explicit teacher/student cell mapping absent from replay")
    return np.column_stack([columns[cell] for cell in ordered_cell_ids]), mass_errors
