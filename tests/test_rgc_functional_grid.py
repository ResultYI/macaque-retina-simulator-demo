"""Permit FP64 construction roundoff while rejecting any physical time shift."""
import unittest

import numpy as np

from tools.analyze_rgc_functional_v1 import native_grid


class NativeGridChecks(unittest.TestCase):
    def test_equivalent_100us_and_1ms_lattice_constructions(self):
        a = np.arange(901) * .001
        b = (np.arange(901) * 10) * .0001
        self.assertFalse(np.array_equal(a, b))
        self.assertTrue(native_grid(a)['integer_native_bin_indices_exact'])
        self.assertLessEqual(native_grid(b)['max_roundoff_seconds'], 4 * np.spacing(1.0))

    def test_physical_shift_is_rejected_without_interpolation(self):
        with self.assertRaises(ValueError):
            native_grid(np.arange(901) * .001 + 1e-7)

    def test_missing_or_duplicate_bin_is_rejected(self):
        with self.assertRaises(ValueError):
            native_grid(np.arange(900) * .001)
        a = np.arange(901) * .001
        a[10] = a[9]
        with self.assertRaises(ValueError):
            native_grid(a)


if __name__ == '__main__':
    unittest.main()
