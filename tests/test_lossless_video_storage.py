"""Protocol and recovery checks for equivalent video storage only."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from examples.full_network_fixture import full_fixture
from retina_sim.contracts import ContractError, Quantity, SolverConfig
from retina_sim.hybrid import HybridRunner
from retina_sim.recording import NPZWindowRecorder
from retina_sim.lossless_video_storage import checkpoint_file_spooled, StoredNPZWindowRecorder
from tools.run_video_05s import checkpoint_file


class LosslessStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiled, _, _ = full_fixture(ROOT)
        cls.runner = HybridRunner(compiled, compute_backend='python')

    def test_original_bytes_and_original_strict_loader(self):
        state = self.runner.initialize(SolverConfig())
        values = (np.arange(17005, dtype=float) * .001).tolist()
        values[8191], values[8192] = -0., 1e-307
        state.input_position['format_probe'] = [values, '\u4e2d\u6587', True, None, 1e300]
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / 'old.json', Path(directory) / 'new.json'
            checkpoint_file(self.runner, state, old)
            receipt = checkpoint_file_spooled(self.runner, state, new)
            self.assertEqual(old.read_bytes(), new.read_bytes())
            restored = self.runner.restore(new.read_text(encoding='utf8'))
            self.assertEqual(self.runner.checkpoint(state), self.runner.checkpoint(restored))
            self.assertEqual(receipt['canonical_subtrees'], 2)
            corrupted = json.loads(new.read_bytes())
            corrupted['payload']['dynamic']['state']['ordered_state_arrays'][0][0] += 1
            with self.assertRaises(ContractError):
                self.runner.restore(json.dumps(corrupted))

    def test_rejected_payload_preserves_previous_complete_file(self):
        state = self.runner.initialize(SolverConfig())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.json'
            checkpoint_file_spooled(self.runner, state, path)
            previous = path.read_bytes()
            state.input_position['invalid_probe'] = float('nan')
            with self.assertRaises(ValueError):
                checkpoint_file_spooled(self.runner, state, path)
            self.assertEqual(previous, path.read_bytes())

    def test_record_arrays_metadata_and_signed_zero_unchanged(self):
        result = SimpleNamespace(outputs_with_units={
            'state:probe': Quantity([[0., -0., 1e-307], [1., 2., 3.]], 'mV', 'voltage', 'synthetic_fixture'),
            'time': Quantity([0., .001, .002], 's', 'absolute_time', 'synthetic_fixture')},
            metadata={'ordered_ids': ['a', '\u4e2d\u6587'], 'events': [], 'tuple': (1, 2)})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = NPZWindowRecorder(root / 'old').consume(result)
            new = StoredNPZWindowRecorder(root / 'new').consume(result)
            with np.load(old, allow_pickle=False) as a, np.load(new, allow_pickle=False) as b:
                self.assertEqual(a.files, b.files)
                for key in a.files:
                    self.assertEqual(a[key].dtype, b[key].dtype)
                    self.assertEqual(a[key].shape, b[key].shape)
                    self.assertEqual(a[key].tobytes(), b[key].tobytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
