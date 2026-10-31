"""Bind separate, unchanged engineering routes to source scope and two positions.

No topology, physiological parameter, initial state, dynamics, or full view rebuild.
Far statistics reuse the completed inner_spatial_v2 views; near reads four sources.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.disk_graph_storage import FrozenSparseRoute
from retina_sim.disk_tables import FrozenDiskTable

PARENT = Path('reports/spatial_group_graph_v02_20261025_v1/artifact_r2')
FAR = Path('reports/retinotopic_rf_revision/inner_spatial_v2')
VIDEO = Path('reports/external_video_two_positions_4s_20261029_v1/PREPARATION.json')
PATHS = {
    'Cone_to_diffuse_BC': ['cones_to_diffuse_BC_ON', 'cones_to_diffuse_BC_OFF'],
    'diffuse_BC_to_parasol': ['diffuse_BC_ON_to_parasol_RGC_ON', 'diffuse_BC_OFF_to_parasol_RGC_OFF'],
}


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path: Path, record: object) -> None:
    with path.open('x', encoding='utf8') as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)


def binding(path: Path) -> dict:
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}


def describe(values: np.ndarray) -> dict:
    return dict(min=float(values.min()), mean=float(values.mean()), max=float(values.max()))


def run(output: Path) -> None:
    started = time.perf_counter()
    memory = MemoryGuard(24 * 1024**3, 8 * 1024**3)
    minimum_free = 2**64

    def guard() -> None:
        nonlocal minimum_free
        if not memory.check():
            raise MemoryError(str((memory.stop_reason, memory.last)))
        if time.perf_counter() - started > 180:
            raise TimeoutError('Bounded structural statistics exceeded 180 seconds')
        minimum_free = min(minimum_free, memory.last['available_physical_bytes'])

    guard()
    manifest = json.loads((ROOT / PARENT / 'manifest.json').read_text(encoding='utf8'))
    routes = json.loads((ROOT / PARENT / 'group_routes.json').read_text(encoding='utf8'))
    video = json.loads((ROOT / VIDEO).read_text(encoding='utf8'))
    far_manifest = json.loads((ROOT / FAR / 'MANIFEST.json').read_text(encoding='utf8'))
    tables: dict[str, FrozenDiskTable] = {}
    populations: dict[str, dict[str, np.ndarray]] = {}
    rows, group_rows, stats, locations = [], [], [], {}
    route_bindings = {}

    def population(name: str):
        if name not in tables:
            tables[name] = FrozenDiskTable(ROOT, manifest['populations'][name]['source_table'])
            fields = ['fine_to_group', 'counts', 'member_offsets', 'member_global_rows']
            data = {}
            for field in fields:
                relative = f'populations/{name}/{field}.npy'
                path = ROOT / PARENT / relative
                assert sha(path) == manifest['files'][relative]['sha256']
                data[field] = np.load(path, mmap_mode='r', allow_pickle=False)
            populations[name] = data
        return tables[name], populations[name]

    for location in ['near', 'far']:
        prepared = Path(video['bindings'][location]['prepared'])
        assert prepared.is_relative_to(ROOT)
        selection_path = prepared / 'local_static/selection.npz'
        identity_path = prepared / 'local_static/selection_identity.json'
        identity = json.loads(identity_path.read_text(encoding='utf8'))
        assert identity['configuration']['source_artifact'] == binding(ROOT / PARENT / 'manifest.json')
        with np.load(selection_path, allow_pickle=False) as archive:
            selection = {name: archive[name] for name in archive.files}
        center = identity['configuration']['rf_center_deg']
        locations[location] = dict(selection=binding(selection_path), selection_identity=binding(identity_path),
            patch_center_deg=center, patch_center_eccentricity_deg=float(np.linalg.norm(center)),
            patch_center_is_not_each_cell_position=True)
        for pathway, names in PATHS.items():
            for name in names:
                guard()
                record = routes[name]
                pre, post = record['pre'], record['post']
                pt, pg = population(pre); qt, qg = population(post)
                selected = selection['population__' + post]
                core = selection['core__' + post]
                assert np.array_equal(selected, np.unique(selected))
                if location == 'far':
                    directory = ROOT / FAR / name
                    receipt = json.loads((directory / 'SOURCE.json').read_text(encoding='utf8'))
                    assert receipt['source'] == record['source']
                    assert sha(ROOT / record['source']['path']) == record['source']['sha256']
                    weights = sparse.load_npz(directory / 'fine_weights.npz')
                    with np.load(directory / 'identities.npz', allow_pickle=False) as z:
                        fine = z['post_fine_rows']; inputs = z['pre_fine_rows']; statuses = z['source_status'].tolist()
                        assert np.array_equal(z['post_group_rows'], selected)
                    recipe = receipt['recipe']
                    reused = binding(directory / 'SOURCE.json')
                else:
                    fine = np.unique(np.concatenate([qg['member_global_rows'][qg['member_offsets'][g]:qg['member_offsets'][g + 1]] for g in selected]))
                    route = FrozenSparseRoute(ROOT, record['source'])
                    fine, global_weights, _, statuses = route.selected_rows(fine)
                    inputs = np.unique(global_weights.indices)
                    weights = global_weights[:, inputs]
                    recipe = route.record['recipe']; reused = None
                assert recipe['pre_table'] == pt.binding and recipe['post_table'] == qt.binding
                assert recipe['selection_budget'] == 8 and recipe['hard_support_mm'] == .04
                assert np.isfinite(weights.data).all() and np.all(weights.data >= 0)
                assert np.all(pg['fine_to_group'][inputs] >= 0)
                groups = qg['fine_to_group'][fine]
                assert np.all(np.bincount(groups, minlength=len(qg['counts']))[selected] == qg['counts'][selected])
                coo = weights.tocoo()
                pre_rows = inputs[coo.col]
                pre_xy = pt.coordinates['opl_pedicle'] if pathway == 'Cone_to_diffuse_BC' else pt.coordinates['ipl_preterminal']
                post_role = 'ipl_preterminal' if pathway == 'Cone_to_diffuse_BC' else 'ipl_post_arbor'
                post_xy = qt.coordinates[post_role]
                distances = np.linalg.norm(pre_xy[pre_rows] - post_xy[fine[coo.row]], axis=1)
                assert np.all(distances <= .04 + 1e-12)
                if pathway == 'Cone_to_diffuse_BC':
                    assert not np.any(pt.extras['subtype'][inputs] == b'S')
                mass = np.asarray(weights.sum(axis=1)).ravel()
                degree = np.diff(weights.indptr)
                effective = mass**2 / np.asarray(weights.multiply(weights).sum(axis=1)).ravel()
                row_ids = qt.rows(fine).global_cell_ids
                rf_degrees = qt.coordinates['rf'][fine] / .223
                for j, source_row in enumerate(fine):
                    a, b = weights.indptr[j:j + 2]
                    rows.append(dict(location=location, pathway=pathway, route=name, fine_post_row=int(source_row),
                        cell_id=row_ids[j], parent_group_row=int(groups[j]), core=bool(groups[j] in core),
                        RF_x_deg=float(rf_degrees[j, 0]), RF_y_deg=float(rf_degrees[j, 1]),
                        RF_eccentricity_deg=float(np.linalg.norm(rf_degrees[j])),
                        contact_anchor_x_mm=float(post_xy[source_row, 0]), contact_anchor_y_mm=float(post_xy[source_row, 1]),
                        source_status=statuses[j], actual_fine_input_count=int(degree[j]), weight_sum=float(mass[j]),
                        effective_weighted_input_count=float(effective[j]), minimum_weight=float(weights.data[a:b].min()),
                        maximum_weight=float(weights.data[a:b].max()), maximum_selected_distance_um=float(distances[a:b].max() * 1000),
                        anatomical_contact_radius_validated=False, new_weight_or_gain_change=False))
                gm = sparse.csr_matrix((coo.data / qg['counts'][groups[coo.row]],
                    (np.searchsorted(selected, groups[coo.row]), pg['fine_to_group'][pre_rows])),
                    shape=(len(selected), len(pg['counts'])))
                relative = record['directory'] + '/mean_weights.npz'
                assert sha(ROOT / PARENT / relative) == manifest['files'][relative]['sha256']
                with np.load(ROOT / PARENT / relative, allow_pickle=False) as z:
                    old = sparse.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))[selected]
                error = float(np.max(np.abs((gm - old).data), initial=0))
                assert error < 1e-12
                for j, group in enumerate(selected):
                    mask = groups == group
                    group_rows.append(dict(location=location, pathway=pathway, route=name, parent_group_row=int(group),
                        member_count=int(qg['counts'][group]), core=bool(group in core),
                        minimum_fine_inputs=int(degree[mask].min()), mean_fine_inputs=float(degree[mask].mean()), maximum_fine_inputs=int(degree[mask].max()),
                        distinct_presynaptic_groups=int(gm[j].nnz), group_mean_weight_sum=float(gm[j].sum()),
                        group_projection_error_bound=error, group_is_not_single_cell=int(qg['counts'][group]) > 1))
                for label, mask in [('all_selected', np.ones(len(fine), bool)), ('core', np.isin(groups, core))]:
                    assert mask.any()
                    stats.append(dict(location=location, pathway=pathway, route=name, scope=label,
                        target_fine_rows=int(mask.sum()), target_groups=int(np.unique(groups[mask]).size),
                        selected_edges=int(degree[mask].sum()), fine_input_count=describe(degree[mask]), weight_sum=describe(mass[mask]),
                        effective_weighted_input_count=describe(effective[mask]), target_RF_eccentricity_deg=describe(np.linalg.norm(rf_degrees[mask], axis=1)),
                        source_status_counts=dict(Counter(np.asarray(statuses)[mask])), group_projection_max_abs_error=error,
                        fixed8_control_same_source_weights=True, actual_new_edge_changes=0, actual_new_weight_changes=0,
                        reused_existing_far_view=reused, physiological_validated=False))
                route_bindings[name] = dict(source=record['source'], pre_table=pt.binding, post_table=qt.binding,
                    recipe=recipe, pre_contact_coordinate=pt.record['coordinates']['opl_pedicle' if pathway == 'Cone_to_diffuse_BC' else 'ipl_preterminal'],
                    post_contact_coordinate=qt.record['coordinates'][post_role])
    for filename, data in [('FINE_ROW_STATISTICS.csv', rows), ('GROUP_STATISTICS.csv', group_rows)]:
        with (output / filename).open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    for pathway, names in PATHS.items():
        write(output / (pathway + '_CANDIDATE.json'), dict(schema='spatial_path_source_scope_candidate_v1', pathway=pathway,
            status='RETAIN_EXISTING_FIXED8_ENGINEERING_UNSUPPORTED_FOR_PHYSIOLOGICAL_REPLACEMENT',
            new_geometry_candidate_generated=False, edge_changes=0, weight_changes=0,
            formal_builder_adopted=False, dynamics_run=False, source_reading='VERIFIED_SOURCE_SCOPE.json',
            reason='Existing40um point proxy, effective subtype scaffold and weight scales lack a matched measured contact support law; published counts alone do not determine a radius',
            routes={name: route_bindings[name] for name in names}, locations=locations,
            gain_policy='Original normalized activation pooling, original total gain, all declared member mean; no crop normalization',
            FIT_READY=False, background_qualified=False, production_adopted=False))
    guard()
    write(output / 'STATISTICS.json', stats)
    write(output / 'RESULT.json', dict(status='SOURCE_BOUND_FIXED8_RETENTION_COMPLETE_NO_NEW_PHYSIOLOGICAL_RULE',
        utc=datetime.now(timezone.utc).isoformat(), locations=locations, parent=binding(ROOT / PARENT / 'manifest.json'),
        reused_13_route_manifest=binding(ROOT / FAR / 'MANIFEST.json'), reused_far_routes=4, newly_recreated_full_views=0,
        fine_statistic_rows=len(rows), group_statistic_rows=len(group_rows), summaries=len(stats),
        max_group_projection_error=max(s['group_projection_max_abs_error'] for s in stats),
        FIT_READY=False, background_qualified=False, production_adopted=False, formal_builder_adopted=False,
        new_model_simulations=0, new_scientific_parameters=0, new_edges=0, new_weights=0,
        wall_seconds=time.perf_counter() - started, peak_rss_bytes=memory.peak_rss_bytes,
        minimum_system_available_observed_bytes=minimum_free, historical_cost_reset=False,
        implementation=binding(Path(__file__))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/retinotopic_rf_revision/two_path_candidates_v1'))
    args = parser.parse_args()
    directory = (ROOT / args.output).resolve()
    if not directory.is_relative_to(ROOT / 'reports/retinotopic_rf_revision'):
        raise ValueError('Independent revision output required')
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / 'STARTED.json', dict(utc=datetime.now(timezone.utc).isoformat(), model_started=False))
    try:
        run(directory)
    except Exception as error:
        write(directory / 'FAILED.json', dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc(), model_started=False))
        raise
