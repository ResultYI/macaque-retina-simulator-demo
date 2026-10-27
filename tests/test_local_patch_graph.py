"""Identity, retained operator and nested-selection checks on frozen assets."""
from pathlib import Path
import json

import numpy as np
import pytest

from retina_sim.contracts import ContractError
from retina_sim.local_patch_graph import LocalPatchAssembly, build_local_patch_graph
from retina_sim.local_patch import load_patch_config

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/local_patch_milestone_v1'


@pytest.fixture(scope='session')
def patches(tmp_path_factory):
    directory = tmp_path_factory.mktemp('local_patch_graphs')
    return tuple(build_local_patch_graph(ROOT,
        load_patch_config(ROOT / 'configurations/local_patch_v1' / (name+'.json')),
        directory / name) for name in ('aggregate_small', 'aggregate_large'))


def test_nested_source_cells_contacts_and_target(patches):
    small, large = patches
    assert small.target_parent_row == large.target_parent_row == 8163
    assert small.populations[small.target_population].counts[small.target_local_row] == 1
    assert small.artifact_sha256 != large.artifact_sha256
    for name, rows in small.parent_population_rows.items():
        assert np.isin(rows, large.parent_population_rows[name]).all()
        assert small.populations[name].group_count > 0
    for name, rows in small.parent_contact_rows.items():
        assert np.isin(rows, large.parent_contact_rows[name]).all()
        assert len(rows) > 0


def test_retained_equations_receive_all_parent_inputs(patches):
    small, _ = patches
    rng = np.random.Generator(np.random.PCG64(82026))
    for route in small.routes.values():
        original = small.base.matrix('routes/'+route.name+'/sum_weights.npz')
        current = small.matrix('routes/'+route.name+'/sum_weights.npz')
        signal = rng.standard_normal(original.shape[1])
        expected = (original @ signal)[small.parent_population_rows[route.post]]
        actual = current @ signal[small.parent_population_rows[route.pre]]
        np.testing.assert_array_equal(actual, expected)
    lap = small.matrix('gap/laplacian_S.npz')
    assert np.max(np.abs(np.asarray(lap.sum(axis=1)).ravel()), initial=0.) < 1e-20


def test_input_is_owned_readonly_without_reprojection(patches):
    small, _ = patches
    name = 'input/group_background_native_td.npy'
    value, parent = small.array(name), small.base.array(name)
    np.testing.assert_array_equal(value, parent)
    assert not np.shares_memory(value, parent)
    assert not value.flags.writeable


def test_fine_request_cannot_silently_use_aggregate(tmp_path):
    config = load_patch_config(ROOT / 'configurations/local_patch_v1/aggregate_small.json')
    config['representation'] = 'original_fine_required_v1'
    with pytest.raises(ContractError, match='explicit representative-group mode'):
        build_local_patch_graph(ROOT, config, REPORT / 'graphs/fine_forbidden')
    assert not (REPORT / 'graphs/fine_forbidden').exists()


def test_tampered_local_manifest_is_rejected(tmp_path):
    raw = json.loads((REPORT / 'graphs/aggregate_small_v1/manifest.json').read_text())
    raw['configuration']['eccentricity_deg'] = 4.5
    (tmp_path / 'manifest.json').write_text(json.dumps(raw))
    with pytest.raises(ContractError, match='manifest identity differs'):
        LocalPatchAssembly(tmp_path, ROOT)


def test_requested_rf_eccentricity_cannot_exceed_parent_scope():
    config = load_patch_config(ROOT / 'configurations/local_patch_v1/aggregate_small.json')
    config['eccentricity_deg'] = 6.01
    with pytest.raises(ContractError, match='eccentricity exceeds'):
        build_local_patch_graph(ROOT, config, REPORT / 'graphs/out_of_scope_forbidden')
    assert not (REPORT / 'graphs/out_of_scope_forbidden').exists()


def test_original_recorder_preserves_parent_derived_subtypes(patches, tmp_path):
    from retina_sim.group_time_recording import GroupRecording

    def original_subtypes(base, population, cache):
        if population in cache:
            return cache[population]
        metadata = base.manifest['populations'][population]
        keys = base.array('populations/'+population+'/keys.npy')
        labels = metadata.get('subtype_labels', [])
        parent = metadata.get('parent_population')
        if labels:
            value = np.asarray(labels)[keys[:, 1]]
        elif parent:
            value = original_subtypes(base, parent, cache)[keys[:, 2]]
        else:
            value = np.full(base.populations[population].group_count, population)
        cache[population] = value
        return value

    for index, patch in enumerate(patches):
        recorder = GroupRecording(patch, tmp_path / ('recording_'+str(index)), [0., .0005, .001])
        fields = {field.key: field for field in patch.recording_fields}
        cache = {}
        assert sum(len(component['ports']) for component in recorder.components) == 45
        for component in recorder.components:
            for port in component['ports'].values():
                field = fields[port['key']]
                parent_rows = patch.parent_population_rows[field.population][field.rows]
                expected = original_subtypes(patch.base, field.population, cache)[parent_rows]
                np.testing.assert_array_equal(np.asarray(port['subtypes']), expected)
        for population, metadata in patch.manifest['populations'].items():
            parent = metadata.get('parent_population')
            if parent:
                column = metadata['key_columns'].index('parent_group')
                local = patch.array('populations/'+population+'/keys.npy')[:, column]
                np.testing.assert_array_equal(patch.parent_population_rows[parent][local],
                    patch.parent_population_keys[population][:, column])
