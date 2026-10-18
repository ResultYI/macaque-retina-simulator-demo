"""Keep the pre-existing owned-worker cleanup regression alongside long paths."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from retina_platform import bounded_iset_batch as batch


class BatchCleanupTests(unittest.TestCase):
    def test_boundary_failure_stops_only_owned_process(self):
        for mode in ('raise', 'false'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)

                class Process:
                    returncode = None
                    pid = 123

                    def poll(self):
                        return self.returncode

                process, stopped = Process(), []

                def stop(value):
                    self.assertIs(value, process)
                    stopped.append(value.pid)
                    value.returncode = 1

                def boundary():
                    if mode == 'raise':
                        raise MemoryError('original reserve')
                    return False

                with patch.object(batch.subprocess, 'Popen', return_value=process), \
                     patch.object(batch, '_terminate_owned', side_effect=stop):
                    with self.assertRaises(MemoryError if mode == 'raise' else batch.ContractError):
                        batch.run_batch(root, {'matlab_executable': 'unused', 'maximum_export_wall_seconds': 300},
                            [{'output': str(root / 'capture.mat')}], root, None, boundary)
                self.assertEqual(stopped, [123])
