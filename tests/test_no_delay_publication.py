"""Equivalent guarded publication, including non-grid delayed history updates."""
import cProfile
import json
from pathlib import Path
import pstats
import unittest
import numpy as np
from examples.full_network_fixture import full_fixture
from examples.plugin_boundary_fixture import plugin_fixture
from retina_sim.contracts import SolverConfig
from retina_sim.delays import EdgeDelayBuffer
from retina_sim.hybrid import HybridRunner
from retina_sim.no_delay_publication import no_empty_link_advance

ROOT = Path(__file__).resolve().parents[1]


class PublicationTests(unittest.TestCase):
    def compare(self, compiled, channels, delay):
        runner = HybridRunner(compiled, compute_backend='python')
        patched, identity = no_empty_link_advance(runner)
        initial = runner.initialize(SolverConfig(method='DOP853', rtol=1e-7, atol=1e-10, max_step=.0001))
        serialized = runner.checkpoint(initial)
        end_states, traces, results, profiles = [], [], [], []
        original_append = EdgeDelayBuffer.append
        try:
            for advance in (runner.advance, patched):
                state = runner.restore(serialized)
                calls, observed, windows = [], [], []

                def append(buffer, time, quantity):
                    calls.append((time, quantity.values.tobytes(), quantity.unit, quantity.meaning))
                    return original_append(buffer, time, quantity)

                def observer(time, values, bundle):
                    observed.append((time, values.tobytes()))

                EdgeDelayBuffer.append = append
                profile = cProfile.Profile()
                profile.enable()
                for times in (np.array([0., .00031, .00123, .00235]), np.array([.00235, .00317, .0047])):
                    state, result = advance(state, channels, float(times[-1]), sample_times=times, state_observer=observer)
                    windows.append(result)
                profile.disable()
                end_states.append(runner.checkpoint(state))
                traces.append((calls, observed))
                results.append(windows)
                stats = pstats.Stats(profile).stats
                profiles.append(sum(value[1] for key, value in stats.items()
                                    if key[0].endswith('hybrid.py') and key[2] == '_flat'))
        finally:
            EdgeDelayBuffer.append = original_append
        self.assertEqual(end_states[0], end_states[1])
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(runner.advance.__func__, HybridRunner.advance)
        for left, right in zip(*results):
            self.assertEqual(json.dumps(left.metadata, sort_keys=True), json.dumps(right.metadata, sort_keys=True))
            self.assertEqual(left.outputs_with_units.keys(), right.outputs_with_units.keys())
            for key, q in left.outputs_with_units.items():
                actual = right.outputs_with_units[key]
                self.assertEqual((q.unit, q.meaning, q.evidence_tag, q.values.dtype, q.values.shape),
                                 (actual.unit, actual.meaning, actual.evidence_tag, actual.values.dtype, actual.values.shape))
                self.assertEqual(q.values.tobytes(), actual.values.tobytes())
        if delay:
            self.assertGreater(len(traces[0][0]), 0)
            self.assertEqual(profiles[0], profiles[1])
        else:
            self.assertEqual(len(traces[0][0]), 0)
            self.assertLess(profiles[1], profiles[0])
        print(json.dumps({'delay': delay, 'publication_append_calls_each': len(traces[0][0]),
                          'flat_calls_original_patched': profiles, 'complete_state_history_RNG_equal': True,
                          'observer_trace_equal': True, 'identity': identity}), flush=True)

    def test_no_delay_with_spikes_clocks_and_observer(self):
        compiled, channels, _ = full_fixture(ROOT)
        self.compare(compiled, channels, False)

    def test_delayed_publication_arrivals_and_non_grid_chunking(self):
        compiled, channels, _, _ = plugin_fixture(ROOT)
        self.compare(compiled, channels, True)


if __name__ == '__main__':
    unittest.main()
