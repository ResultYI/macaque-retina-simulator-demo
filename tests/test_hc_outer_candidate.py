import unittest
from pathlib import Path
import numpy as np
from retina_sim.hc_outer_candidate import OuterHCCandidate
from retina_sim.contracts import ContractError


class OuterCandidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = OuterHCCandidate(Path(__file__).resolve().parents[1])

    def test_central_unchanged_and_join_C1(self):
        p = self.p
        r = np.linspace(0, p.join, 1001)
        for kind in ('H1', 'H2'):
            np.testing.assert_array_equal(p.contact_radius_mm(r, kind), p.central.contact_radius_mm(r, kind))
            xy = np.c_[r, np.zeros_like(r)]
            np.testing.assert_array_equal(p.type_density(xy, kind), p.central.type_density(xy, kind))
            x = p.join; h = 1e-6
            f = lambda r: float(p.type_density([r, 0], kind))
            self.assertAlmostEqual((f(x+h)-f(x))/h, (f(x)-f(x-h))/h, delta=.02)
            f = lambda r: float(p.contact_radius_mm(r, kind))
            self.assertAlmostEqual((f(x+h)-f(x))/h, (f(x)-f(x-h))/h, delta=1e-7)
        np.testing.assert_array_equal(p.radial(r), p.central.radial(r))
        np.testing.assert_array_equal(p.radial_fraction(r), p.central.radial_fraction(r))

    def test_source_nodes_and_contact_anchor(self):
        p = self.p
        for row in p.source_rows:
            r = np.mean(row['eccentricity_interval_mm'])
            if r > 6: continue
            for kind in ('H1', 'H2'):
                self.assertAlmostEqual(float(p.type_density([r, 0], kind)), row[kind], places=8)
        for kind, count in [('H1', 18), ('H2', 19)]:
            np.testing.assert_allclose(p.contact_radius_mm([5, 5.5, 6], kind), np.sqrt(count/(np.pi*6000)), rtol=1e-14)
        with self.assertRaises(ContractError): p.radial(6.001)

    def test_rejection_bounds_enclose_dense_samples(self):
        p = self.p
        for bounds in [(1.2, 1.5, -.1, .1), (2.1, 2.6, -.1, .2), (4.3, 4.6, -.2, .2)]:
            limits = p.bounds_for_rectangle(bounds)
            x, y = np.meshgrid(np.linspace(*bounds[:2], 100), np.linspace(*bounds[2:], 100))
            xy = np.c_[x.ravel(), y.ravel()]
            for kind in ('H1', 'H2'):
                rho = p.type_density(xy, kind)
                self.assertTrue(np.all(rho >= limits[kind+'_min']-1e-10))
                self.assertTrue(np.all(rho <= limits[kind+'_max']+1e-10))
