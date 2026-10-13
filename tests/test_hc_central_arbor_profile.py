"""Small fixture checks of the optional W2000 OPL calculation support.

No network dynamics, statistical generation or source-author code is run here.
"""
from pathlib import Path
import json
import math

import numpy as np
import pytest

from retina_sim.contracts import ContractError, Quantity
from retina_sim.hc_geometry import read_hc_geometry
from retina_sim.hc_arbor import build_hc_arbor_contacts


ROOT = Path(__file__).resolve().parents[1]


def profile():
    return read_hc_geometry(ROOT, 'hc_joint_synthetic_v1',
                            arbor_profile_id='central_OPL_equal_area_W2000')


def q(points, meaning, frame):
    return Quantity(np.asarray(points, dtype=float), 'mm', meaning,
                    'synthetic_fixture', frame)


def fixture_contacts(*, input_points=((0., 0.),),
                     pedicle_points=((.75, 0.),),
                     hc_points=((.75, 0.), (.756, 0.)),
                     cone_types=('L',), hc_types=('H1', 'H2')):
    data = profile()
    return build_hc_arbor_contacts(tuple(f'cone:{i}' for i in range(len(input_points))),
        q(input_points, 'functional_position', 'functional_visual_field'),
        q(pedicle_points, 'pedicle_position', 'OPL'),
        tuple(f'HC:{i}' for i in range(len(hc_points))), hc_types,
        q(hc_points, 'anatomical_position', 'anatomical_retina'),
        q(hc_points, 'arbor_anchor_position', 'OPL'),
        np.ones(len(input_points), bool), data['footprints'], cone_types=cone_types,
        source_radial_support_mm=data['contact_source_radial_support_mm'],
        cone_pedicle_generation_bounds_mm=(.48, 1.02, -.05, .05),
        hc_generation_bounds_mm=(.48, 1.02, -.05, .05))


def test_optional_named_arbor_keeps_original_defaults_and_no_subcentral_extension():
    original = read_hc_geometry(ROOT)
    current = read_hc_geometry(ROOT, 'hc_joint_synthetic_v1')
    assert original['profile_id'] == 'source_patch_v1'
    assert original['arbor_profile_id'] == current['arbor_profile_id'] == 'explicit_footprints'
    assert current['footprints'] == {'H1': None, 'H2': None}
    central = profile()
    assert central['contact_source_radial_support_mm'] == [.5, 1.]
    assert central['arbor_source_profile']['subcentral_extension_enabled'] is False
    assert central['arbor_source_profile']['default_selected'] is False
    assert central['divergence_profile'] is None
    assert central['require_owner_quota'] is False
    assert central['arbor_source_profile']['dynamics_changes'] is False


def test_profile_radii_units_types_formula_and_source_identity_are_explicit():
    data = profile()
    for subtype, radius, count in (('H1', .00804, 6.5), ('H2', .01222, 15.)):
        row = data['footprints'][subtype]
        assert row['kind'] == 'disk'
        assert row['unit'] == 'mm' and row['coordinate_frame'] == 'OPL'
        assert row['radius_mm'] == radius
        assert abs(radius - math.sqrt(count / (math.pi * 32000))) < .000005
        assert row['radial_support_mm'] == [.5, 1.]
        assert row['evidence_class'] == 'project_approximation'
        assert row['approximation_id'] == 'HC_central_OPL_equal_area_W2000_v1'
        assert row['source_profile_id'].endswith(subtype + '_v1')
    source = data['arbor_source_profile']
    assert source['derivation']['single_cell_contact_quota_imposed'] is False
    assert source['derivation']['divergence_quota_imposed'] is False
    assert 'not independent validation' in source['source_records']['PackerDacey2005']['comparison_role']
    assert len(source['source_configuration']['sha256']) == 64


def test_legacy_source_patch_cannot_silently_activate_named_arbor():
    with pytest.raises(ContractError, match='source_patch_v1 is incompatible'):
        read_hc_geometry(ROOT, arbor_profile_id='central_OPL_equal_area_W2000')
    with pytest.raises(ContractError, match='Unknown named HC arbor'):
        read_hc_geometry(ROOT, 'hc_joint_synthetic_v1', arbor_profile_id='invented')


@pytest.mark.parametrize('radii', ({'h1_radius_mm': .00804}, {'h2_radius_mm': .01222}))
def test_named_source_support_rejects_manual_radius_even_when_equal(radii):
    with pytest.raises(ContractError, match='conflicts with manual radii'):
        read_hc_geometry(ROOT, 'hc_joint_synthetic_v1',
            arbor_profile_id='central_OPL_equal_area_W2000', **radii)


def test_external_json_can_select_named_arbor_but_not_override_its_footprints(tmp_path):
    path = tmp_path / 'geometry.json'
    path.write_text(json.dumps({'arbor_profile_id': 'central_OPL_equal_area_W2000',
        'generator': {'sweeps': 1}}), encoding='utf-8')
    resolved = read_hc_geometry(ROOT, 'hc_joint_synthetic_v1', config_path=str(path))
    assert resolved['arbor_profile_id'] == 'central_OPL_equal_area_W2000'
    assert resolved['footprints']['H1']['radius_mm'] == .00804
    assert resolved['generator']['sweeps'] == 1
    path.write_text(json.dumps({'arbor_profile_id': 'central_OPL_equal_area_W2000',
                               'footprints': {'H1': None, 'H2': None}}), encoding='utf-8')
    with pytest.raises(ContractError, match='conflicts with manual radii or JSON footprints'):
        read_hc_geometry(ROOT, 'hc_joint_synthetic_v1', config_path=str(path))
    path.write_text(json.dumps({'arbor_profile_id': 'explicit_footprints'}), encoding='utf-8')
    with pytest.raises(ContractError, match='conflicts with geometry JSON'):
        read_hc_geometry(ROOT, 'hc_joint_synthetic_v1', config_path=str(path),
                         arbor_profile_id='central_OPL_equal_area_W2000')


def test_band_is_actual_opl_and_does_not_reject_shifted_rf_input():
    A, B, audit = fixture_contacts(input_points=((0., 0.),))
    assert A.nnz == B.nnz == 2
    assert audit['paired_cones'][0]['input_xy_mm'] == [0., 0.]
    assert audit['paired_cones'][0]['pedicle_xy_mm'] == [.75, 0.]
    assert audit['source_radial_support_mm'] == [.5, 1.]
    assert audit['divergence']['status'] == 'DIVERGENCE_UNSELECTED_CANDIDATE_GRAPH'
    assert audit['geometry_is_conductance'] is False


@pytest.mark.parametrize('pedicle,anchor', ((.49, .75), (.75, .49),
                                          (1.01, .75), (.75, 1.01)))
def test_real_pedicle_and_anchor_must_both_be_in_source_band(pedicle, anchor):
    with pytest.raises(ContractError, match='CONTACT_DOMAIN_UNSUPPORTED'):
        fixture_contacts(pedicle_points=((pedicle, 0.),),
                         hc_points=((anchor, 0.),), hc_types=('H1',))


@pytest.mark.parametrize('anchor,subtype', ((.505, 'H1'), (.995, 'H2')))
def test_entire_arbor_must_fit_band_even_when_anchor_and_pedicle_are_inside(anchor, subtype):
    with pytest.raises(ContractError, match='finite arbor.*extends beyond source'):
        fixture_contacts(pedicle_points=((anchor, 0.),),
                         hc_points=((anchor, 0.),), hc_types=(subtype,))


def test_no_contact_does_not_enlarge_named_support_or_insert_nearest():
    data = profile()
    A, B, audit = build_hc_arbor_contacts(('cone:0',),
        q([[.1, 0.]], 'functional_position', 'functional_visual_field'),
        q([[.75, 0.]], 'pedicle_position', 'OPL'), ('HC:0',), ('H1',),
        q([[.77, 0.]], 'anatomical_position', 'anatomical_retina'),
        q([[.77, 0.]], 'arbor_anchor_position', 'OPL'), np.ones(1, bool),
        data['footprints'], cone_types=('L',), source_radial_support_mm=(.5, 1.),
        cone_pedicle_generation_bounds_mm=(.7, .8, -.05, .05))
    assert A.nnz == B.nnz == 0
    assert audit['hc_rows'][0]['status'] == 'ZERO_LEGAL_TERMINALS_IN_FINITE_SUPPORT'
    assert audit['hc_rows'][0]['arbor']['radius_mm'] == .00804
    assert audit['outside_support_nearest_fallback'] is False


def test_existing_h1_lm_h2_lms_type_policy_is_retained():
    A, B, audit = fixture_contacts(input_points=((0., 0.),), cone_types=('S',))
    assert A.getrow(0).nnz == 0
    assert A.getrow(1).nnz == B.nnz == 1
    assert audit['edges'][0]['hc_type'] == 'H2'
    assert audit['feedback']['unit'] == 'dimensionless'
