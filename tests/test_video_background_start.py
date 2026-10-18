"""Video initialization contracts; small fixture, no neural time integration."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import tempfile
import unittest

from examples.full_network_fixture import full_fixture
from retina_sim.background import BackgroundPolicy, NUMERICAL_UNIT_SCALES
from retina_sim.background_checkpoint import BackgroundCheckpointStore, checkpoint_hash_diagnostics
from retina_sim.contracts import ContractError
from retina_sim.runtime_profiles import LightContext
from tools.video_background_start import (
    add_background_arguments, select_background_source, restore_background_start, begin_media_stimulus,
)

ROOT = Path(__file__).resolve().parents[1]


class BackgroundModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiled, _, _ = full_fixture(ROOT)
        cls.runner = compiled.runner()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'background_checkpoint.json'
        self.state = self.runner.initialize()
        self.store = BackgroundCheckpointStore(ROOT, self.runner, self.state, LightContext(), BackgroundPolicy())
        self.report = {
            'status': 'NOT_REACHED', 'converged': False, 'policy': asdict(self.store.policy),
            'numerical_unit_registry': NUMERICAL_UNIT_SCALES, 'uncovered_states': [],
            'checked_states': self.store.coverage()['states'], 'windows': [],
        }
        self.store.save(self.path, self.state, self.report)

    def test_default_is_strict_and_transient_needs_explicit_source(self):
        parser = argparse.ArgumentParser()
        add_background_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual(args.background_mode, 'strict')
        _, checkpoint, identity = select_background_source(ROOT, ROOT/'original', {}, 'strict', None)
        self.assertEqual(checkpoint, ROOT/'original/matched_response/background_checkpoint.json')
        self.assertIsNone(identity)
        with self.assertRaisesRegex(ValueError, 'explicit --background-source'):
            select_background_source(ROOT, ROOT/'original', {}, 'transient', None)

    def test_strict_rejects_same_unqualified_checkpoint(self):
        with self.assertRaises(ContractError):
            restore_background_start(self.store, self.path, 'strict')

    def test_transient_retains_full_checkpoint_and_never_certifies_convergence(self):
        before = self.runner.checkpoint(self.state)
        restored, envelope, annotation = restore_background_start(self.store, self.path, 'transient')
        self.assertEqual(self.runner.checkpoint(restored), before)
        self.assertEqual(annotation['status'], 'NOT_REACHED')
        self.assertFalse(annotation['converged'])
        self.assertFalse(envelope['report']['converged'])

    def test_background_identity_mismatch_still_rejected(self):
        other = BackgroundCheckpointStore(
            ROOT, self.runner, self.runner.initialize(), LightContext(native_background_td=101.), BackgroundPolicy())
        with self.assertRaisesRegex(ContractError, 'identity mismatch'):
            restore_background_start(other, self.path, 'transient')

    def test_tampered_causal_state_still_rejected(self):
        value = json.loads(self.path.read_bytes())
        ticks = value['checkpoint']['payload']['component_tick_indices']
        ticks[next(iter(ticks))] += 1
        self.path.write_text(json.dumps(value), encoding='utf8')
        with self.assertRaisesRegex(ContractError, 'identity changed'):
            restore_background_start(self.store, self.path, 'transient')

    def test_report_criteria_cannot_be_relaxed(self):
        self.report['policy'] = dict(self.report['policy'], relative_tolerance=.1)
        self.store.save(self.path, self.state, self.report)
        with self.assertRaisesRegex(ContractError, 'criteria or unit scales'):
            restore_background_start(self.store, self.path, 'transient')

    def test_domain_failure_is_not_a_permitted_transient(self):
        self.report['windows'] = [{'domain_failures': ['invalid_gate']}]
        self.store.save(self.path, self.state, self.report)
        with self.assertRaisesRegex(ContractError, 'invariant/history failures'):
            restore_background_start(self.store, self.path, 'transient')

    def test_stimulus_switch_keeps_absolute_clock_rng_and_histories(self):
        state = SimpleNamespace(time_seconds=1.012, grid_origin_seconds=0., component_tick_indices={'cell': 10120},
                                spike_states={'rng': [4, 8], 'pending': [1.013]}, delay_states={'queue': [1.011]},
                                input_position={'phase': 'uniform_background', 'background_context_hash': 'original'})
        before = (state.time_seconds, state.grid_origin_seconds, dict(state.component_tick_indices),
                  dict(state.spike_states), dict(state.delay_states))
        begin_media_stimulus(state, 'new-video-sha', 1.012, {'mode': 'transient', 'status': 'NOT_REACHED'})
        after = (state.time_seconds, state.grid_origin_seconds, state.component_tick_indices,
                 state.spike_states, state.delay_states)
        self.assertEqual(before, after)
        self.assertEqual(state.input_position['phase'], 'media_stimulus')
        self.assertEqual(state.input_position['background_context_hash'], 'original')
        self.assertEqual(state.input_position['background_initialization_status'], 'NOT_REACHED')

    def test_in_loader_diagnostics_are_read_only(self):
        original = self.path.read_bytes()
        before = checkpoint_hash_diagnostics(json.loads(original), 'fixture-file-sha')
        diagnostic = self.path.with_name('load_diagnostics.json')
        restore_background_start(self.store, self.path, 'transient', diagnostic_path=diagnostic)
        inside = json.loads(diagnostic.read_bytes())
        self.assertEqual(before['payload_fields'], inside['payload_fields'])
        self.assertEqual(before['complete'], inside['complete'])
        self.assertTrue(inside['after_decode']['source_and_decoded_equal'])
        self.assertEqual(self.path.read_bytes(), original)

    def test_diagnostics_identify_clock_tampering_without_accepting_it(self):
        value = json.loads(self.path.read_bytes())
        before = checkpoint_hash_diagnostics(value, 'fixture-file-sha')
        ticks = value['checkpoint']['payload']['component_tick_indices']
        ticks[next(iter(ticks))] += 1
        self.path.write_text(json.dumps(value), encoding='utf8')
        diagnostic = self.path.with_name('rejected_diagnostics.json')
        with self.assertRaisesRegex(ContractError, 'identity changed'):
            restore_background_start(self.store, self.path, 'transient', diagnostic_path=diagnostic)
        inside = json.loads(diagnostic.read_bytes())
        self.assertNotEqual(inside['complete']['expected'], inside['complete']['actual'])
        self.assertNotEqual(before['payload_fields']['component_tick_indices'],
                            inside['payload_fields']['component_tick_indices'])
        self.assertEqual(before['payload_fields']['spike_states'], inside['payload_fields']['spike_states'])


if __name__ == '__main__':
    unittest.main()
