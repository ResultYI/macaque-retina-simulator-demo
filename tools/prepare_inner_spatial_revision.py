"""Extract unchanged, dependency-complete local BC/RGC structural views.

This does not initialize or run a neural model, infer RFs, or calibrate rules.
Original global row IDs, member counts, source statuses and weights are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.disk_graph_storage import FrozenSparseRoute
from retina_sim.disk_population import FixedDiskPopulation
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.background_checkpoint import MemoryGuard

MEMORY = MemoryGuard(24 * 1024**3, 8 * 1024**3)

PARENT = Path('reports/spatial_group_graph_v02_20261025_v1/artifact_r2')
SELECTION = Path('reports/position_S0_far_originaltarget_20261029_resume_v2/prepared/local_static/selection.npz')
NAMES = (
    'cones_to_midget_BC_ON', 'cones_to_midget_BC_OFF', 'cones_to_S_ON_BC',
    'cones_to_diffuse_BC_ON', 'cones_to_diffuse_BC_OFF',
    'midget_BC_ON_to_midget_RGC_ON', 'midget_BC_OFF_to_midget_RGC_OFF',
    'diffuse_BC_ON_to_parasol_RGC_ON', 'diffuse_BC_OFF_to_parasol_RGC_OFF',
    'S_ON_BC_to_small_bistratified_S_ON', 'diffuse_BC_OFF_to_small_bistratified_S_ON',
    'rods_to_rod_BC', 'rod_BC_to_AII_AC',
)


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def guard() -> tuple[int, int]:
    if not MEMORY.check():
        raise MemoryError(str((MEMORY.stop_reason, MEMORY.last)))
    return MEMORY.peak_rss_bytes, MEMORY.last['available_physical_bytes']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/retinotopic_rf_revision/inner_spatial_v1'))
    parser.add_argument('--resume', action='store_true', help='Reuse only routes with completed SOURCE.json receipts')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / 'reports/retinotopic_rf_revision'):
        raise ValueError('Output must be in the independent revision directory')
    output.mkdir(parents=True, exist_ok=args.resume)
    if (output / 'MANIFEST.json').exists():
        raise FileExistsError('Completed structural artifact already exists; no overwrite')
    started = time.perf_counter()
    start_record = output / ('RESUME_STARTED.json' if args.resume else 'STARTED.json')
    with start_record.open('x', encoding='utf8') as stream:
        json.dump({'utc': datetime.now(timezone.utc).isoformat(), 'model_started': False}, stream)
    manifest = json.loads((ROOT / PARENT / 'manifest.json').read_text(encoding='utf8'))
    group_routes = json.loads((ROOT / PARENT / 'group_routes.json').read_text(encoding='utf8'))
    tables: dict[str, FrozenDiskTable | FixedDiskPopulation] = {}
    groups: dict[str, dict[str, np.ndarray]] = {}
    with np.load(ROOT / SELECTION, allow_pickle=False) as archive:
        selection = {key: archive[key] for key in archive.files}
    peak, lowest_free = guard()

    def population(name: str) -> tuple[FrozenDiskTable | FixedDiskPopulation, dict[str, np.ndarray]]:
        if name not in tables:
            binding = manifest['populations'][name]['source_table']
            fixed = json.loads((ROOT / binding['path']).read_text(encoding='utf8'))['schema'] == 'fixed_disk_population_arrays_v1'
            tables[name] = FixedDiskPopulation(ROOT, binding) if fixed else FrozenDiskTable(ROOT, binding)
            directory = ROOT / PARENT / 'populations' / name
            fields = ['fine_to_group', 'member_global_rows', 'member_offsets', 'counts', 'group_ids']
            for key in fields:
                relative = f'populations/{name}/{key}.npy'
                assert digest(ROOT / PARENT / relative) == manifest['files'][relative]['sha256']
            groups[name] = {key: np.load(directory / (key + '.npy'), mmap_mode='r', allow_pickle=False) for key in fields}
            assert groups[name]['fine_to_group'].shape == (tables[name].record['count'],)
        return tables[name], groups[name]

    summaries = []
    reused = []
    for name in NAMES:
        record = group_routes[name]
        receipt = output / name / 'SOURCE.json'
        if args.resume and receipt.exists():
            row = json.loads(receipt.read_text(encoding='utf8'))
            assert row['source'] == record['source'] and row['crop_renormalized'] is False
            for filename in ['fine_weights.npz', 'group_mean_weights.npz', 'identities.npz']:
                assert (receipt.parent / filename).is_file()
            summaries.append(row); reused.append(name)
            continue
        pre, post = record['pre'], record['post']
        pre_table, pg = population(pre)
        post_table, qg = population(post)
        selected_groups = selection['population__' + post]
        assert np.array_equal(selected_groups, np.unique(selected_groups)), 'Local group rows must be sorted and unique'
        fine_post = np.unique(np.concatenate([qg['member_global_rows'][qg['member_offsets'][g]:qg['member_offsets'][g + 1]] for g in selected_groups]))
        route = FrozenSparseRoute(ROOT, record['source'])
        recipe = route.record['recipe']
        assert recipe.get('pre_table', recipe.get('pre_master')) == pre_table.binding
        assert recipe.get('post_table', recipe.get('post_master')) == post_table.binding
        fine_post, weights, audits, statuses = route.selected_rows(fine_post)
        assert np.all(np.isfinite(weights.data)) and np.all(weights.data >= 0)
        pre_rows = np.unique(weights.indices)
        assert np.all(pg['fine_to_group'][pre_rows] >= 0), 'Active input missing from parent representation'
        post_group = qg['fine_to_group'][fine_post]
        assert np.all(np.isin(post_group, selected_groups))
        assert np.all(np.bincount(post_group, minlength=len(qg['counts']))[selected_groups] == qg['counts'][selected_groups])
        parent_check = None
        post_extras = getattr(post_table, 'extras', {})
        if name in {'cones_to_midget_BC_ON', 'cones_to_midget_BC_OFF', 'cones_to_S_ON_BC'}:
            expected = post_extras['cone_row'][fine_post]
            assert np.all(np.diff(weights.indptr) == 1) and np.array_equal(weights.indices, expected)
            parent_check = 'EXACT_ORIGINAL_CONE_PARENT'
        elif 'parent_bc_row' in post_extras and pre.startswith('midget_BC'):
            expected = post_extras['parent_bc_row'][fine_post]
            assert all(expected[i] in weights.indices[weights.indptr[i]:weights.indptr[i + 1]] for i in range(len(fine_post)))
            parent_check = 'ORIGINAL_BC_PARENT_INCLUDED'
        # Fine rows keep their complete incoming dependency set; no coordinate re-query.
        dependency_group = np.unique(pg['fine_to_group'][pre_rows])
        pre_local_groups = selection['population__' + pre]
        pre_is_local = np.isin(pg['fine_to_group'][pre_rows], pre_local_groups)
        local_col = np.isin(pg['fine_to_group'][weights.indices], pre_local_groups)
        coo = weights.tocoo()
        group_matrix = sparse.csr_matrix((coo.data / qg['counts'][post_group[coo.row]],
            (np.searchsorted(selected_groups, post_group[coo.row]), pg['fine_to_group'][coo.col])),
            shape=(len(selected_groups), len(pg['counts'])))
        relative = record['directory'] + '/mean_weights.npz'
        assert digest(ROOT / PARENT / relative) == manifest['files'][relative]['sha256']
        with np.load(ROOT / PARENT / relative, allow_pickle=False) as z:
            assert set(z.files) == {'data', 'indices', 'indptr', 'shape'}
            existing = sparse.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))[selected_groups]
        difference = group_matrix - existing
        max_error = float(np.max(np.abs(difference.data), initial=0))
        assert max_error < 1e-12, f'Fine/group projection mismatch: {name}: {max_error}'
        fine_dependency = weights[:, pre_rows]
        directory = output / name
        directory.mkdir()
        sparse.save_npz(directory / 'fine_weights.npz', fine_dependency)
        sparse.save_npz(directory / 'group_mean_weights.npz', group_matrix[:, dependency_group])
        runtime_ready = np.array([s.startswith('READY') for s in statuses])
        target_unmet = np.array(['UNMET' in s or s.startswith('INCOMPLETE') for s in statuses])
        core_groups = selection['core__' + post]
        core_rows = np.isin(post_group, core_groups)
        outside_mass = np.asarray(sparse.csr_matrix((coo.data * (~local_col), (coo.row, coo.col)), shape=weights.shape).sum(axis=1)).ravel()
        np.savez_compressed(directory / 'identities.npz', post_fine_rows=fine_post, pre_fine_rows=pre_rows,
            post_cell_ids=np.asarray(post_table.rows(fine_post).global_cell_ids, dtype='U'),
            pre_cell_ids=np.asarray(pre_table.rows(pre_rows).global_cell_ids, dtype='U'),
            post_group_rows=selected_groups, pre_group_rows=dependency_group,
            post_group_ids=qg['group_ids'][selected_groups], pre_group_ids=pg['group_ids'][dependency_group],
            post_member_counts=qg['counts'][selected_groups], pre_member_counts=pg['counts'][dependency_group],
            post_rf_mm=(post_table.positions if isinstance(post_table, FixedDiskPopulation) else post_table.coordinates['rf'])[fine_post],
            pre_rf_mm=(pre_table.positions if isinstance(pre_table, FixedDiskPopulation) else pre_table.coordinates['rf'])[pre_rows],
            source_status=np.asarray(statuses, dtype='U'),
            runtime_ready=runtime_ready, source_target_unmet=target_unmet,
            source_target_met=runtime_ready & ~target_unmet,
            physiological_calibrated=np.zeros(len(fine_post), dtype=bool),
            post_is_core=core_rows, pre_in_existing_local=pre_is_local,
            outside_existing_local_mass=outside_mass)
        row = {'name': name, 'pre': pre, 'post': post, 'fine_post_rows': len(fine_post),
            'core_fine_post_rows': int(core_rows.sum()), 'target_groups': len(selected_groups),
            'core_target_groups': len(core_groups), 'source_edges': weights.nnz,
            'pre_dependencies': len(pre_rows), 'dependencies_outside_existing_local': int((~pre_is_local).sum()),
            'rows_with_outside_existing_local': int(np.count_nonzero(outside_mass)),
            'core_rows_with_outside_existing_local': int(np.count_nonzero(outside_mass[core_rows])),
            'statuses': dict(Counter(statuses)), 'source_target_unmet_rows': int(target_unmet.sum()),
            'original_parent_check': parent_check, 'group_projection_max_abs_error': max_error,
            'degree_range': [int(np.diff(weights.indptr).min()), int(np.diff(weights.indptr).max())],
            'row_weight_sum_range': [float(np.asarray(weights.sum(axis=1)).min()), float(np.asarray(weights.sum(axis=1)).max())],
            'source': record['source'], 'recipe': recipe,
            'crop_renormalized': False, 'dynamics_or_boundary_values_created': False}
        (directory / 'SOURCE.json').write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding='utf8')
        summaries.append(row)
        rss, free = guard(); peak = max(peak, rss); lowest_free = min(lowest_free, free)
        print(json.dumps({k: row[k] for k in ['name', 'fine_post_rows', 'source_edges', 'source_target_unmet_rows', 'core_rows_with_outside_existing_local']}, ensure_ascii=False), flush=True)
    result = {'status': 'STRUCTURAL_VIEWS_COMPLETE_DYNAMICS_NOT_RUN', 'utc': datetime.now(timezone.utc).isoformat(),
        'parent': {'path': str(PARENT / 'manifest.json'), 'sha256': digest(ROOT / PARENT / 'manifest.json')},
        'selection': {'path': str(SELECTION), 'sha256': digest(ROOT / SELECTION)},
        'implementation': {'path': 'tools/prepare_inner_spatial_revision.py', 'sha256': digest(Path(__file__))},
        'routes': summaries, 'reused_completed_routes': reused,
        'wall_seconds_this_attempt': time.perf_counter() - started, 'peak_rss_observed_bytes': peak,
        'minimum_system_available_observed_bytes': lowest_free, 'resource_guard_bytes': 24 * 1024**3,
        'new_HC_binding': 'reports/retinotopic_rf_revision/native_validation_v1/CANDIDATE_MODEL.json',
        'new_HC_downstream_initialized': False, 'boundary_dependencies_preserved': True,
        'FIT_READY': False, 'background_qualified': False, 'production_adopted': False,
        'functional_RF_measured': False, 'physiological_subtype_convergence_calibrated': False}
    (output / 'MANIFEST.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    print('STRUCTURAL_VIEWS_COMPLETE', result['wall_seconds_this_attempt'], flush=True)


if __name__ == '__main__':
    main()
