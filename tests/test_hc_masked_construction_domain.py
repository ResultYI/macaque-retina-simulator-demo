"""Software regression: unused bounding-box corners are not density extrapolation."""
import numpy as np
import pytest
from retina_sim.hc_mosaic import HCDensityProfile, RadialMaskedHCDomain, _Quadrature, _domain_density_bounds
from retina_sim.contracts import ContractError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_masked_quadrature_evaluates_only_supported_points():
    profile = HCDensityProfile(ROOT)
    bounds = (1.32, 1.34, -.04, .04)
    with pytest.raises(ContractError):
        profile.bounds_for_rectangle(bounds)
    domain = RadialMaskedHCDomain(bounds, bounds, 1.338, 'synthetic-boundary-regression', 'synthetic')
    envelope = _domain_density_bounds(profile, domain, bounds)
    assert envelope['H1_min'] > 0
    points = np.array([[1.337, 0.], [1.339, 0.]])
    assert domain.mask(points).tolist() == [True, False]
    original = profile.type_density
    def guarded(points, kind):
        assert np.all(np.linalg.norm(points, axis=1) <= 1.338)
        return original(points, kind)
    profile.type_density = guarded
    result = _Quadrature(profile, domain, 8, 1e-4, 1e-5, 24, 100000).rule(bounds, 8)
    assert np.all(result > 0) and np.isfinite(result).all()
    assert result[0] / result[1] == pytest.approx(4.)

def test_previous_mask_boundary_failure_is_integrated_at_original_tolerance():
    profile = HCDensityProfile(ROOT)
    bounds = (1.3115744451943538, 1.311574478258768, -.2646062054576865, -.26460617407997133)
    domain = RadialMaskedHCDomain(bounds,bounds,1.338,'synthetic-boundary-regression','synthetic')
    q = _Quadrature(profile,domain,12,1e-4,1e-5,24,100000)
    mass,error = q.integrate(bounds)
    assert np.all(mass>0) and np.all(error<=1e-5+1e-4*np.abs(mass))
    assert mass[0]/mass[1] == pytest.approx(4.)
