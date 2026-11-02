"""Synthetic software fixtures; not research observations. Not executed yet."""
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pytest
from retina_sim.bc_current import PROFILE, current_bc, require_current_bc_asset
from retina_sim.bc_pedicle_density import RATIOS, RegisteredPedicleField, anchor_rows, cone_bc_densities, crop_anchor_rows
from retina_sim.contracts import ContractError

def test_exact_ratios_and_observed_vs_closed_anchor():
    values = cone_bc_densities(25060., 1452.)
    assert values['midget_BC_ON'] == 25060
    assert values['midget_BC_OFF'] == 26512
    assert values['diffuse_BC_ON'] == pytest.approx(10514.)
    assert values['diffuse_BC_OFF'] == pytest.approx(23327.)
    assert values['S_ON_BC'] == pytest.approx(2702.)
    assert sum(values.values()) == pytest.approx(88115.)
    assert sum(values.values()) + 1726 == pytest.approx(89841.)
    for value in cone_bc_densities(0., 0.).values():
        assert value == 0
    with pytest.raises(ContractError):
        cone_bc_densities(-1., 0.)

def test_private_parent_ids_off_includes_S_and_finite_quotas():
    ids = tuple('synthetic-cone-' + str(i) for i in range(101))
    labels = np.array(['L'] * 50 + ['M'] * 40 + ['S'] * 11)
    groups = anchor_rows(ids, labels, np.arange(101))
    assert len(groups['midget_BC_ON'][0]) == 90
    assert len(groups['midget_BC_OFF'][0]) == 101
    assert np.all(labels[groups['midget_BC_ON'][0]] != 'S')
    assert np.any(labels[groups['midget_BC_OFF'][0]] == 'S')
    for name, ratio in RATIOS.items():
        source = 11 if name == 'S_ON_BC' else 101
        expected = (2 * source * ratio.numerator + ratio.denominator) // (2 * ratio.denominator)
        indices, slots = groups[name]
        assert len(indices) == expected
        assert len(set(zip(indices.tolist(), slots.tolist()))) == expected
    again = anchor_rows(ids, labels, np.arange(101)[::-1])
    for name in groups:
        np.testing.assert_array_equal(groups[name], again[name])

def field_fixture(labels=('L', 'M', 'S')):
    xy = np.array([[.8, 0], [.801, 0], [.802, 0]])[:len(labels)]
    return SimpleNamespace(record={'coordinates': {'opl_pedicle': {'unit': 'mm', 'meaning': 'pedicle_position', 'frame': 'OPL'}},
        'source_bindings': {'Schein': {'identity': 'synthetic-outward-fixture'}, 'fixed_RF_query_radius_mm': 1.5}},
        coordinates={'opl_pedicle': xy, 'rf': xy}, extras={'subtype': np.array(labels, dtype='S1')})

def test_local_crop_keeps_full_parent_quota_and_unique_identity():
    ids=tuple('synthetic-'+str(i) for i in range(100))
    xy=np.column_stack((np.linspace(.8,1.,100),np.zeros(100)))
    groups=anchor_rows(ids,np.array(['L']*90+['S']*10),np.arange(100))
    query={'unit':'mm','frame':'OPL','bounds_mm':[.85,.95,-.01,.01]}
    crop=crop_anchor_rows(groups,xy,query)
    for name,(indices,slots) in crop.items():
        original=set(zip(groups[name][0].tolist(),groups[name][1].tolist()))
        assert set(zip(indices.tolist(),slots.tolist())) <= original
        assert np.all((xy[indices,0]>=.85)&(xy[indices,0]<.95))
    with pytest.raises(ContractError):
        crop_anchor_rows(groups,xy,dict(query,frame='functional_visual_field'))

CONFIG = {'local_density': {'radius_mm': .04}, 'geometry': {'S_ON_max_contact_mm': .04},
          'application_anchor_radius_mm': [0, 1.338]}

def test_disk_selection_eligibility_no_nearest_fill_and_S_cap():
    field = RegisteredPedicleField(field_fixture(), CONFIG)
    found, audit = field.contacts([.8, 0], 'diffuse_BC_ON')
    assert set(found) == {0, 1}
    assert audit['kernel'] == 'uniform_disk' and audit['actual_degree'] == 2
    found, audit = field.contacts([.802, 0], 'S_ON_BC')
    assert found.tolist() == [2] and audit['cap_applied']
    assert audit['radius_mm'] == .04
    absent = RegisteredPedicleField(field_fixture(('L', 'M')), CONFIG)
    found, audit = absent.contacts([.8, 0], 'S_ON_BC')
    assert not len(found) and audit['status'] == 'INCOMPLETE_NO_ELIGIBLE_PEDICLE'
    with pytest.raises(ContractError):
        field.contacts([1.339, 0], 'S_ON_BC')

def test_registered_pedicle_contract_and_query_completeness():
    cone = field_fixture()
    cone.record['coordinates']['opl_pedicle']['frame'] = 'functional_visual_field'
    with pytest.raises(ContractError):
        RegisteredPedicleField(cone, CONFIG)
    field = RegisteredPedicleField(field_fixture(), CONFIG)
    with pytest.raises(ContractError, match='INCOMPLETE'):
        field.density([1.48, 0], 'LM')

def test_new_identity_rejects_old_assets_without_loading_them(tmp_path):
    (tmp_path / 'configurations').mkdir()
    parameter = tmp_path / 'parameters.json'
    parameter.write_text('{}')
    selected = {'selected_profile_id': PROFILE, 'bindings': {'parameters': {'path': 'parameters.json',
        'sha256': hashlib.sha256(parameter.read_bytes()).hexdigest()}}}
    (tmp_path / 'configurations/bc_density_geometry_current.json').write_text(json.dumps(selected))
    identity = current_bc(tmp_path)['identity']
    with pytest.raises(ContractError, match='STALE'):
        require_current_bc_asset(tmp_path, {})
    require_current_bc_asset(tmp_path, {'bc_density_geometry_identity': identity})
    parameter.write_text('{"changed":true}')
    with pytest.raises(ContractError, match='IDENTITY_CHANGED'):
        current_bc(tmp_path)
