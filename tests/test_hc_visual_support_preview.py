"""The visual observation disk must have complete anatomical contact support."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'semi_reports/2026-11-02_HC'))
import refresh_hc_current as preview
from retina_sim.contracts import ContractError


class VisualSupportContract(unittest.TestCase):
    def test_extension_is_explicit_and_preserves_original_functions(self):
        base = preview.JointV2HCDensityProfile(ROOT)
        with self.assertRaises(ContractError):
            base.radial(1.75)
        preview.select_visual_support_repair()
        extended = preview.DiskProfile(ROOT)
        radius = np.linspace(0, 1.338, 1001)
        np.testing.assert_array_equal(base.radial(radius), extended.radial(radius))
        for kind in ('H1', 'H2'):
            np.testing.assert_array_equal(base.contact_radius_mm(radius, kind),
                                          extended.contact_radius_mm(radius, kind))
        self.assertGreater(extended.radial(1.75), 0)

    def test_actual_optical_cones_have_complete_contact_support(self):
        preview.select_visual_support_repair()
        profile = preview.DiskProfile(ROOT)
        pedicles = np.load(ROOT / 'data/atlas/human_disk3mm_typed_tables_v1/cones/opl_pedicle.npy')
        with np.load(preview.VIDEO / 'geometry.npz') as geometry:
            rows = geometry['source_table_rows']
        maximum = np.linalg.norm(pedicles[rows], axis=1).max()
        radius = max(float(profile.contact_radius_mm(1.75, t)) for t in ('H1', 'H2'))
        self.assertGreater(maximum, 1.338)
        self.assertLess(maximum + radius, 1.65)
        self.assertLess(maximum + radius, 1.75)

    def test_curved_domain_bounds_do_not_query_unused_corners(self):
        preview.select_visual_support_repair()
        profile = preview.DiskProfile(ROOT)
        bounds = profile.bounds_for_rectangle((-1.75, 1.75, -1.75, 1.75))
        self.assertGreater(bounds['H2_min'], 0)
        self.assertGreaterEqual(bounds['total_max'], profile.radial(1.75))


if __name__ == '__main__':
    unittest.main()
