from copy import deepcopy
import numpy as np
import pytest
from retina_sim.contracts import ContractError
from retina_sim.cone_aperture_eccentricity import resolve_apertures


def profile():
    return dict(schema='macaque-cone-aperture-eccentricity-1',profile_id='unit_fixture',
        coordinate_binding=dict(microns_per_degree=200.),supported_eccentricity_degrees=[0.,6.],
        meridians={k:dict(knots=[dict(eccentricity_degrees=0.,reference_diameter_um=2.,evidence_class='source_derived',source_id='fixture'),
            dict(eccentricity_degrees=6.,reference_diameter_um=5.,evidence_class='engineering',source_id='fixture')]) for k in ['nasal','temporal']},
        kernel=dict(shape='Gaussian',sigma_per_reference_diameter=.204,zone_diameter_width_um=.05))


def test_gaussian_collecting_area_matches_integral_of_peak_one_capture_profile():
    r=resolve_apertures(['c'],['L'],np.array([[0.,0.]]),profile())
    sigma=r['sigma_um'][0]*1e-6
    # Independent polar integral of exp(-radius^2/(2sigma^2)), no second disk area.
    radial=np.linspace(0.,8*sigma,10001)
    area=np.trapezoid(2*np.pi*radial*np.exp(-radial**2/(2*sigma**2)),radial)
    np.testing.assert_allclose(r['effective_area_m2'][0],area,rtol=1e-7)
    np.testing.assert_allclose(r['characteristic_radius_um'],np.sqrt(2)*r['sigma_um'])
    assert r['area_applied_once']


def test_real_coordinates_and_identity_are_preserved_and_zones_are_bounded():
    xy=np.array([[.1,0.],[-1.,.2]]);before=xy.copy();p=profile();old=deepcopy(p)
    r=resolve_apertures(['a','b'],['L','S'],xy,p)
    np.testing.assert_array_equal(r['positions_mm'],before);np.testing.assert_array_equal(xy,before)
    assert list(r['cone_ids'])==['a','b'] and p==old
    assert np.all(abs(r['reference_diameter_um']-r['zone_diameter_um'])<=.025+1e-14)
    assert np.all(r['evidence_class']=='engineering')


def test_does_not_extrapolate_or_lose_cone_identity():
    with pytest.raises(ContractError):resolve_apertures(['a'],['M'],[[1.21,0]],profile())
    with pytest.raises(ContractError):resolve_apertures(['a','a'],['L','M'],[[0,0],[.1,0]],profile())
