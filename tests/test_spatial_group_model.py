"""CPU-only structural checks; no retinal dynamics or qualified initial states."""
import hashlib
import json

import numpy as np
import pytest
from scipy import sparse

from retina_sim.contracts import ContractError
from retina_sim.spatial_group_graph import digest, save_csr
from retina_sim.spatial_group_model import (AC, BC, CONTACT_STATES, POPULATIONS, RGC,
    AdvanceRequest, GroupModelAssembly, REQUIRED_ROUTE_PAIRS, SCHEMA, recording_contract)


def seal(root, manifest):
    manifest = dict(manifest)
    manifest.pop('manifest_sha256', None)
    manifest['files'] = {p.relative_to(root).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)}
                         for p in root.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    manifest['manifest_sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf8')


@pytest.fixture
def artifact(tmp_path):
    populations = {}
    for name in POPULATIONS:
        folder = tmp_path / 'populations' / name
        folder.mkdir(parents=True)
        arrays = dict(group_ids=np.array(['group:' + name]), counts=np.array([2], dtype=np.int64),
            member_offsets=np.array([0, 2], dtype=np.int64), member_global_rows=np.array([0, 2], dtype=np.int64),
            fine_to_group=np.array([0, -1, 0], dtype=np.int32), centroid_mm=np.zeros((1, 2), dtype=np.float64))
        arrays.update(keys=np.array([[0, -1, -1, -1]], dtype=np.int64),
            recording_auxiliary_fine_rows=np.array([0, 2], dtype=np.int64),
            recording_auxiliary_group_rows=np.array([0, 0], dtype=np.int64),
            leaf_level=np.array([0], dtype=np.int64), leaf_width_deg=np.array([.4]),
            polygon_deg=np.array([[[0., 0.], [.4, 0.], [.4, .4], [0., .4]]]),
            covariance_deg2=np.zeros((1, 2, 2)), lambda_max_deg2=np.zeros(1), centroid_deg=np.zeros((1, 2)),
            leaf_keys=np.array([[0, 0, 0]], dtype=np.int64), precision_status=np.array(['BACKGROUND_APPROXIMATION']),
            requested_fmax_cpd=np.zeros(1), covariance_linear_bound=np.zeros(1),
            admissible_linear_average_fmax_cpd=np.full(1, np.inf), original_singleton_row=np.array([-1], dtype=np.int64))
        for key, value in arrays.items():
            np.save(folder / (key + '.npy'), value, allow_pickle=False)
        populations[name] = dict(group_count=1, fine_neurons=2, full_asset_count=3,
            coordinate_role='soma' if name == 'HC' else 'rf', source_table={'path': 'original_table'},
            source_keys={'path': 'original_keys'}, parameter_class_source='effective_' + name,
            derived_parameter_fields=['V0'])
    contact_pairs = {('rods', 'rod_BC'): 'individual_rod_RBC_contacts', ('rod_BC', 'AII_AC'): 'RBC_AII_AMPA',
                    ('AII_AC', 'midget_BC_OFF'): 'AII_FMB_glycine', ('AII_AC', 'diffuse_BC_OFF'): 'AII_DB1_glycine'}
    pairs = [(pre, post, contact_pairs.get((pre, post))) for pre, post in REQUIRED_ROUTE_PAIRS]
    routes = []
    for pre, post, contact in pairs:
        name = pre + '_to_' + post
        folder = tmp_path / 'routes' / name
        folder.mkdir(parents=True)
        save_csr(folder / 'sum_weights.npz', sparse.csr_matrix([[2.]]))
        save_csr(folder / 'mean_weights.npz', sparse.csr_matrix([[1.]]))
        row = dict(name=name, pre=pre, post=post, kind='primary_chemical' if contact else 'native',
                   group_pair_entries=1, sum_semantics='raw summed weights', mean_semantics='divide by post count')
        if contact:
            row['contact'] = dict(component=contact, groups=1, fine_contacts=2,
                                  coefficient_unit='S', coefficient_sum=1e-9)
            for key, value in dict(contact_post_group=[0], contact_pre_group=[0],
                                   contact_member_offsets=[0, 2], fine_contact_global_rows=[1, 3],
                                   fine_contact_to_group=[0, 0], contact_member_global_rows=[1, 3]).items():
                np.save(folder / (key + '.npy'), np.array(value, dtype=np.int64))
            save_csr(folder / 'contact_coefficient_S.npz', sparse.csr_matrix([[1e-9]]))
        routes.append(row)
    (tmp_path / 'gap').mkdir()
    save_csr(tmp_path / 'gap' / 'laplacian_S.npz', sparse.csr_matrix((1, 1)))
    contact_states = sum(len(v) for v in CONTACT_STATES.values())
    cell_states = 7 + 3 + 7 + 3 + 3 + 7 * len(BC) + len(AC) + len(RGC)
    manifest = dict(schema=SCHEMA, runtime_ready=False, model_forward_started=False,
        populations=populations, routes=routes, gap=dict(population_order=['AII_AC'], laplacian_entries=0),
        budget_audit=dict(core_state_scalars=cell_states + contact_states,
                          extra_RGC_history_probability_scalars=8 * len(RGC), cell_slots=len(POPULATIONS),
                          extra_selected_z1_auxiliary_scalars=2,
                          contact_groups={k: 1 for k in CONTACT_STATES}),
        recording_contract=recording_contract())
    seal(tmp_path, manifest)
    return tmp_path, manifest


def test_complete_uninitialized_layout_and_observation_contract(artifact):
    root, manifest = artifact
    model = GroupModelAssembly(root)
    assert set(model.populations) == set(POPULATIONS)
    assert model.state_scalar_count == manifest['budget_audit']['core_state_scalars']
    assert model.probability_scalar_count == 40
    assert model.runtime_ready is False
    assert not hasattr(model, 'initial_state')
    assert model.populations['HC'].coordinate_role == 'soma'
    assert model.populations['cones'].group_for_fine_row(2) == 0
    assert model.populations['cones'].source_keys['path'] == 'original_keys'
    with pytest.raises(ContractError, match='outside this experiment'):
        model.populations['cones'].group_for_fine_row(1)
    assert {f.name for f in model.recording_fields if f.population == 'cones'} == set(recording_contract()['fields']['cones'])
    assert all(len(f.rows) == 1 for f in model.recording_fields if f.primary)
    assert all(f.identity == 'representative_group' for f in model.recording_fields if f.primary)
    assert all(f.identity == 'group_prediction_at_original_selected_cell' for f in model.recording_fields if not f.primary)
    assert len(model.unresolved_bindings) == 18
    assert [(b.name, b.unit) for b in model.state_blocks if b.population == 'HC'] == [
        ('w', 'vH_native_voltage'), ('z2', 'vH_native_voltage'), ('Vh', 'vH_native_voltage')]
    assert {b.name for b in model.state_blocks if b.role == 'analytic_anchor'} == {'R', 'E'}
    assert not any(b.name == 'z1' for b in model.state_blocks)
    description = model.describe()
    assert json.loads(json.dumps(description, allow_nan=False))['time_backend']['integration_step_seconds'] is None
    assert description['numerical_dynamics_implemented'] is False
    assert description['populations']['cones']['fine_members'] == 2
    assert len(model.populations['cones'].mesh_geometry['polygon_deg']) == 1
    auxiliary = next(f for f in model.recording_fields if f.population == 'cones' and f.name == 'X')
    np.testing.assert_array_equal(auxiliary.rows, [0, 0])
    np.testing.assert_array_equal(auxiliary.original_fine_rows, [0, 2])
    assert model.auxiliary_z1_scalar_count == 2
    assert model.auxiliary_z1_blocks[0].count == 2
    assert model.auxiliary_z1_blocks[0].role == 'uninitialized_selected_original_auxiliary'
    assert description['extra_selected_z1_auxiliary_scalars'] == 2


def test_no_backend_or_pending_bindings_cannot_advance(artifact):
    model = GroupModelAssembly(artifact[0])
    request = AdvanceRequest(0., .001, (0., .0005, .001), (.0007,))
    with pytest.raises(ContractError, match='backend is not provided'):
        model.advance(None, None, request)
    class FixtureBackend:
        def advance(self, *args):
            raise AssertionError('No model advance is allowed in the structural stage')
    with pytest.raises(ContractError, match='derived parameter bindings'):
        model.advance(None, None, request, backend=FixtureBackend())
    assert not hasattr(request, 'step_seconds')
    for observation in ((.0005, .0005), (.002,), (float('nan'),)):
        with pytest.raises(ContractError):
            AdvanceRequest(0., .001, observation, ())


def test_hash_schema_and_membership_fail_closed(artifact):
    root, manifest = artifact
    np.save(root / 'populations' / 'cones' / 'fine_to_group.npy', np.array([0, 0, 0], dtype=np.int32))
    with pytest.raises(ContractError, match='array identity'):
        GroupModelAssembly(root)
    seal(root, manifest)
    with pytest.raises(ContractError, match='complete selected upstream'):
        GroupModelAssembly(root)
    manifest['schema'] = 'retipath-static-spatial-group-graph-1'
    seal(root, manifest)
    with pytest.raises(ContractError, match='static v2'):
        GroupModelAssembly(root)


def test_route_and_contact_totals_are_not_untyped_weights(artifact):
    root, manifest = artifact
    model = GroupModelAssembly(root)
    route = model.routes['rods_to_rod_BC']
    assert route.contact_groups == 1 and route.fine_contacts == 2
    assert route.coefficient_sum_S == 1e-9
    mean = model.matrix('routes/rods_to_rod_BC/mean_weights.npz')
    assert not mean.data.flags.writeable
    save_csr(root / 'routes' / 'rods_to_rod_BC' / 'contact_coefficient_S.npz', sparse.csr_matrix([[2e-9]]))
    seal(root, manifest)
    with pytest.raises(ContractError, match='contact layout/totals'):
        GroupModelAssembly(root)


def test_recording_reuses_full_primary_and_spatial_auxiliary_selection(artifact):
    root, manifest = artifact
    # The selector contract is checked directly at realistic size; no model values are synthesized.
    from tools.cone_hc_demo_recording import selected_rows
    xy = np.column_stack((np.linspace(-1., 1., 1200), np.zeros(1200)))
    assert len(selected_rows(xy)) < 900
    assert recording_contract()['auxiliary_selection'].endswith('selected_rows')
    assert recording_contract()['primary_selection'] == 'all computed groups'
    manifest['recording_contract']['fields']['cones'].remove('R')
    seal(root, manifest)
    with pytest.raises(ContractError, match='field/selection contract'):
        GroupModelAssembly(root)


def test_missing_feedback_route_and_contact_member_mapping_are_rejected(artifact):
    root, manifest = artifact
    kept = manifest['routes']
    manifest['routes'] = [r for r in kept if (r['pre'], r['post']) != ('S_ON_BC', 'AII_AC')]
    seal(root, manifest)
    with pytest.raises(ContractError, match='reciprocal feedback routes'):
        GroupModelAssembly(root)
    manifest['routes'] = kept
    np.save(root / 'routes' / 'rods_to_rod_BC' / 'contact_member_global_rows.npy', np.array([3, 1], dtype=np.int64))
    seal(root, manifest)
    with pytest.raises(ContractError, match='contact member mapping'):
        GroupModelAssembly(root)


def test_mesh_singleton_identity_is_not_invented(artifact):
    root, manifest = artifact
    np.save(root / 'populations' / 'cones' / 'original_singleton_row.npy', np.array([0], dtype=np.int64))
    seal(root, manifest)
    with pytest.raises(ContractError, match='leaf/precision/member identity'):
        GroupModelAssembly(root)


def test_auxiliary_fine_selection_and_selected_z1_ledger_are_bound(artifact):
    root, manifest = artifact
    manifest['budget_audit']['extra_selected_z1_auxiliary_scalars'] = 1
    seal(root, manifest)
    with pytest.raises(ContractError, match='capacity ledger'):
        GroupModelAssembly(root)
    manifest['budget_audit']['extra_selected_z1_auxiliary_scalars'] = 2
    np.save(root / 'populations' / 'cones' / 'recording_auxiliary_fine_rows.npy', np.array([0, 1], dtype=np.int64))
    seal(root, manifest)
    with pytest.raises(ContractError, match='original auxiliary recording selection'):
        GroupModelAssembly(root)
