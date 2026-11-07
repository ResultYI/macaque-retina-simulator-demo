"""Regression checks for the fixed Table 2 protocol and acceptance gate."""
import csv
import unittest
import numpy as np
from retina_sim.contracts import canonical_hash
from retina_sim.hc_teacher_calibration import make_homogeneous_source, DerivedHomogeneousTeacher
from tools.calibrate_h1_temporal import ROOT, TARGET, PROTOCOL, accept, amplitude_ratios


class TemporalContract(unittest.TestCase):
    def test_fixed_targets(self):
        with TARGET.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([(int(r['background_td']), int(r['pulse_ms'])) for r in rows],
                         [(100,10),(10,10),(1,10),(100,100),(10,100),(1,100)])
        self.assertEqual([float(r['peak_latency_ms']) for r in rows],
                         [36.35,39.55,42.70,45.80,58.33,101.67])
        self.assertTrue(all(r['weber_contrast']=='19' and r['source_table']=='Table 2' for r in rows))

    def test_holdout_and_polarity_can_veto_training_improvement(self):
        def records(train, held, polarity=True):
            return [dict(split=s, standardized_error=e, polarity_hyperpolarizing=polarity)
                    for s,e in [('train',train),('heldout',held)]]
        original = records(2., 1.)
        self.assertTrue(accept(original, records(1., 1.)))
        self.assertFalse(accept(original, records(1., 1.01)))
        self.assertFalse(accept(original, records(2., .5)))
        self.assertFalse(accept(original, records(1., .5, False)))

    def test_native_feedback_equilibrium_and_only_tau_h_changes(self):
        source = make_homogeneous_source(ROOT, 10.)
        identity = canonical_hash(PROTOCOL)
        a,b = [DerivedHomogeneousTeacher.create(source,tau,identity,source.model_hash) for tau in (20.,40.)]
        np.testing.assert_array_equal(a.equilibrium,b.equilibrium)
        self.assertLess(float(np.max(abs(a.rhs(0,a.equilibrium,10.)))),1e-8)
        y = a.equilibrium.copy(); y[-1] -= 1.
        da,db = a.rhs(0,y,10.),b.rhs(0,y,10.)
        np.testing.assert_array_equal(da[:-1],db[:-1])
        self.assertAlmostEqual(db[-1]*2, da[-1])
        self.assertEqual(source.p['tau_h'],20.)
        np.testing.assert_array_equal(source.A.toarray(),[[1.]])
        np.testing.assert_array_equal(source.B.toarray(),[[1.]])

    def test_amplitude_ratios_use_same_pulse_reference(self):
        rows = [dict(background_td=bg,pulse_ms=p,amplitude_native=a,observed_amplitude_mV=b)
                for bg,p,a,b in [(100,10,2.,13.97),(10,10,1.,4.31),(100,100,4.,19.38),(1,100,1.,1.85)]]
        amplitude_ratios(rows)
        self.assertEqual(rows[1]['predicted_amplitude_ratio_to_100td'],.5)
        self.assertEqual(rows[3]['predicted_amplitude_ratio_to_100td'],.25)
        self.assertAlmostEqual(rows[1]['observed_amplitude_ratio_to_100td'],4.31/13.97)


if __name__ == '__main__':
    unittest.main()
