"""HC configuration/runtime integration regressions, deliberately NOT RUN.

Fixture spans below test software contracts; they are not proposed physiological
arbor defaults. This file does not authorize simulation or statistical validation.
"""
from pathlib import Path
import json

import numpy as np
import pytest
from scipy import sparse

from retina_sim.contracts import CellTable, ContractError, Quantity
from retina_sim.geometry import RetinalDomain
from retina_sim.spatial import Population
from retina_sim.hc_geometry import (read_hc_geometry, joint_hc_geometry,
                                    hc_contact_runtime_view)
from retina_sim.hc_mosaic import JointHCMosaic
from retina_platform.config import RunConfig, input_assumptions

ROOT = Path(__file__).resolve().parents[1]


def q(values, meaning, frame, unit='mm'):
    return Quantity(np.asarray(values, dtype=float), unit, meaning,
                    'synthetic_fixture', frame)


def hc_population():
    points = np.array([[.7, 0.], [.71, 0.], [.9, 0.]])
    ids = ('canonical:H1:0', 'canonical:H2:0', 'canonical:H1:1')
    soma = q(points, 'anatomical_position', 'anatomical_retina')
    cells = CellTable(cell_id=ids, cell_type=('HC',)*3,
        subtype=('H1', 'H2', 'H1'), species_profile_id=('fixture',)*3,
        hemiretina=('test',)*3, meridian=('test',)*3,
        soma_position=soma, rf_position=None, position_unit='mm',
        coordinate_frame='anatomical_retina',
        eccentricity=q(np.linalg.norm(points, axis=1), 'global_eccentricity', 'anatomical_retina'),
        parameter_profile_id=('fixture',)*3, evidence_tags=('synthetic_fixture',)*3)
    return Population('HC', cells, ('center',)*3, anatomical_positions=soma,
        dendritic_positions=q(points, 'dendritic_position', 'OPL'))


def test_original_configs_keep_source_patch_and_never_get_arbor_defaults():
    original = RunConfig.from_dict({'execution_profile':'source_constrained_approximation'})
    assert original.hc_geometry_profile == 'source_patch_v1'
    assert original.hc_h1_radius_mm is None and original.hc_h2_radius_mm is None
    current = read_hc_geometry(ROOT, 'hc_joint_synthetic_v1')
    assert current['footprints'] == {'H1':None, 'H2':None}
    with pytest.raises(ContractError, match='joint synthetic'):
        read_hc_geometry(ROOT, 'source_patch_v1', h1_radius_mm=.01)


def test_gui_config_roundtrip_preserves_explicit_uncalibrated_geometry():
    config = RunConfig(hc_geometry_profile='hc_joint_synthetic_v1',
        hc_h1_radius_mm=.01, hc_h2_radius_mm=.02)
    restored = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config
    resolved = read_hc_geometry(ROOT, restored.hc_geometry_profile,
        h1_radius_mm=restored.hc_h1_radius_mm, h2_radius_mm=restored.hc_h2_radius_mm)
    assert resolved['footprints']['H1']['evidence_class'] == 'project_approximation'
    assert resolved['footprints']['H2']['radius_mm'] == .02
    declared = input_assumptions(restored)['geometry_request']['HC_geometry']
    assert declared['profile_id'] == restored.hc_geometry_profile
    assert declared['H1_explicit_radius_mm'] == .01


def test_external_geometry_cannot_rewrite_density_sources(tmp_path):
    path = tmp_path/'geometry.json'
    path.write_text(json.dumps({'source_records':{}}), encoding='utf-8')
    with pytest.raises(ContractError, match='may configure only'):
        read_hc_geometry(ROOT, 'hc_joint_synthetic_v1', config_path=str(path))


def test_runtime_view_preserves_all_geometry_and_only_selects_contact_participants():
    original = hc_population()
    A = sparse.csr_matrix([[1., 0.], [0., 1.], [0., 0.]])
    B = sparse.csr_matrix([[1., 0., 0.], [0., 1., 0.]])
    view, a, b, selected, audit = hc_contact_runtime_view(original, A, B)
    assert original.cells.cell_id == ('canonical:H1:0', 'canonical:H2:0', 'canonical:H1:1')
    assert view.cells.cell_id == original.cells.cell_id[:2]
    assert selected.tolist() == [True, True, False]
    np.testing.assert_array_equal(view.anatomical_positions.values, original.anatomical_positions.values[:2])
    np.testing.assert_array_equal(view.dendritic_positions.values, original.dendritic_positions.values[:2])
    assert a.shape == (2, 2) and b.shape == (2, 2)
    assert audit['geometry_only_cell_ids'] == ['canonical:H1:1']
    assert audit['density_instance_thinned'] is False
    assert audit['native_kinetics_changed'] is False
    assert audit['implicit_background_feedback'] is False


def test_native_runtime_reports_uncovered_cones_and_does_not_fill_feedback():
    A = sparse.csr_matrix([[1., 0.], [0., 1.], [0., 0.]])
    B = sparse.csr_matrix([[1., 0., 0.], [0., 0., 0.]])
    with pytest.raises(ContractError, match='UNCOVERED_CONES'):
        hc_contact_runtime_view(hc_population(), A, B)
    assert B.getrow(1).nnz == 0


def test_joint_geometry_uses_physical_canonical_context_and_shared_crop_ids(monkeypatch):
    import retina_sim.hc_mosaic as module
    seen = {}
    def capture(profile, domain, seed, max_cells, options):
        seen.update(domain=domain, profile=profile, seed=seed)
        points = np.array([[.7, 0.], [.71, 0.]])
        return JointHCMosaic(q(points, 'anatomical_position', 'anatomical_retina'),
            np.array(['H1','H2']), ('master:H1:0','master:H2:0'),
            domain.contains(points, owner=True),
            {'canonical_bounds_mm':list(domain.bounds_mm),
             'mask_source_id':domain.mask_source_id,'fixture':True})
    monkeypatch.setattr(module, 'generate_joint_hc', capture)
    mapped = RetinalDomain((.69,.72,-.02,.02),(.7,.71,-.005,.005),
        lambda xy:np.ones(len(xy),bool), 'mapped_fixture_mask','mapped_OPL_fixture')
    configuration = read_hc_geometry(ROOT,'hc_joint_synthetic_v1',
        h1_radius_mm=.01,h2_radius_mm=.02)
    mosaics, audit = joint_hc_geometry(ROOT,mapped,123,8192,configuration)
    assert audit['HC_is_displaced_by_cone_map'] is False
    assert audit['canonical_context_buffer_mm'] >= audit['mosaic_pair_support_halo_mm']
    assert seen['domain'].bounds_mm[0] < audit['active_bounds_mm'][0] < mapped.bounds_mm[0]
    assert [cell_id for _, mosaic in mosaics for cell_id in mosaic.cell_ids] == ['master:H1:0','master:H2:0']
    assert audit['canonical_instance']['cell_ids'] == ['master:H1:0','master:H2:0']
    assert all(mosaic.positions.coordinate_frame == 'anatomical_retina' for _, mosaic in mosaics)


def test_primary_geometry_label_cannot_bypass_material_identity(tmp_path):
    footprint = {'kind':'disk','radius_mm':.01,'unit':'mm','coordinate_frame':'OPL',
        'source_profile_id':'claimed_primary','evidence_class':'measurement',
        'radial_support_mm':[.6,1.],
        'primary_verification':{'verified':True,'material_path':'missing-material.bin',
            'material_sha256':'0'*64,'geometry_sha256':'0'*64,
            'locator':'fixture quotation','condition':'fixture only'}}
    path=tmp_path/'geometry.json'
    path.write_text(json.dumps({'footprints':{'H1':footprint,'H2':footprint}}),encoding='utf-8')
    with pytest.raises(ContractError,match='material identity mismatch'):
        read_hc_geometry(ROOT,'hc_joint_synthetic_v1',config_path=str(path))
