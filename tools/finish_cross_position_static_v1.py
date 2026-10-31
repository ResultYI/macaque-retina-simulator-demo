"""Independent static coverage after a source-evidence guard blocked new state."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.ac_vs_model import ACVSAssembly, load_config
from retina_sim.contracts import canonical_hash
from retina_sim.h1_h2_effective import H1H2Assembly, verify_typed_source, save_artifact
from retina_sim.local_patch_graph import LocalPatchAssembly
from retina_sim.local_patch_boundary import BoundedInnerDynamics, BoundedNativePhysics
from retina_sim.spatial_group_graph import digest
from retina_sim.spatial_group_model import GroupModelAssembly, RGC
from tools.prepare_cross_position_coverage_v1 import selection


def write(path, value):
    from retina_sim.parameter_system import data
    with path.open('x', encoding='utf8') as f:
        json.dump(data(value), f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attempt', default='static_only')
    args = parser.parse_args()
    out = args.output.resolve()
    if Path(args.attempt).name != args.attempt:
        raise ValueError('Single directory name required for a preserved attempt')
    start = time.perf_counter()
    if not (out / 'PREPARATION_FAILURE.json').exists():
        raise ValueError('Preserved source-construction failure required; no hidden substitute state')
    if (out / 'STATIC_COVERAGE_RESULT.json').exists():
        raise FileExistsError('Preserve completed static coverage')
    geometry = json.loads((out / 'GEOMETRY_SELECTION.json').read_bytes())
    center = np.asarray(geometry['center_deg'])
    if abs(np.linalg.norm(center) - geometry['actual_eccentricity_deg']) > 1e-12:
        raise ValueError('Frozen coordinate units disagree')
    main_cfg = json.loads((ROOT / 'configurations/retina_platform_v1.json').read_bytes())
    dependencies = {}
    for pop, binding in main_cfg['rgc_output_defaults']['accepted_types'].items():
        accepted = json.loads((ROOT / binding['acceptance']['path']).read_bytes())
        for item in accepted.get('evidence_files', []):
            p = ROOT / item['path']
            row = dependencies.setdefault(item['path'], dict(path=item['path'],
                expected_sha256=item['sha256'], bound_types=[], present=p.is_file(),
                actual_sha256=digest(p) if p.is_file() else None))
            row['bound_types'].append(pop)
            row['passed'] = row['actual_sha256'] == row['expected_sha256']
    missing = [r for r in dependencies.values() if not r['passed']]
    evidence_file = out / 'REQUIRED_SOURCE_EVIDENCE.json'
    if evidence_file.exists():
        if json.loads(evidence_file.read_bytes()) != list(dependencies.values()):
            raise ValueError('Source evidence changed; use a new research output directory')
    else:
        write(evidence_file, list(dependencies.values()))
    ledger = json.loads((out / 'RESOURCE_LEDGER.json').read_bytes())
    stage = 'static_graph_selection'
    try:
        base = GroupModelAssembly(ROOT / 'reports/spatial_group_graph_v02_20261025_v1/artifact_r2')
        cfg = json.loads((ROOT / 'configurations/local_patch_v2/halo_030.json').read_bytes())
        cfg.pop('azimuth_deg')
        cfg.update(eccentricity_deg=geometry['actual_eccentricity_deg'], rf_center_deg=center.tolist(),
            target_rgc_type=geometry['selected_type'], target_rgc_id=None,
            position_provenance='Geometry-only near-center candidate from centroid_mm/0.223; no response-based location selection',
            target_provenance='Actual frozen group identity; original input seed binding unresolved, no new background or runtime qualification')
        present = [n for n in RGC if geometry['geometric_presence'][n]['core_centroid_groups']]
        def check():
            if time.perf_counter() - start > 300:
                raise TimeoutError('Static coverage guard')
        small = selection(base, cfg, present, check)
        bigcfg = deepcopy(cfg); bigcfg['patch_geometry']['support_radius_deg'] = .6
        large = selection(base, bigcfg, present, check)
        rows = dict(small[0])
        for p in ('cones', 'HC'):
            if not np.isin(rows[p], large[0][p]).all():
                raise ValueError('Native support is not nested')
            rows[p] = large[0][p]
        folder = out / args.attempt; folder.mkdir()
        arrays = {'population__' + k: v for k, v in rows.items()}
        arrays.update({'contact__' + k: v for k, v in small[1].items()})
        arrays.update({'core__' + k: v for k, v in small[2].items()})
        np.savez_compressed(folder / 'selection.npz', **arrays)
        descriptor = dict(schema='static-only-cross-position-native060-inner030-1',
            configuration=cfg, parent_artifact=base.artifact_sha256,
            selection_sha256=canonical_hash(arrays), valid_new_seed_binding=None,
            static_only=True, runtime_ready=False, background_solved=False)
        write(folder / 'STATIC_GRAPH_DESCRIPTOR.json', descriptor)
        patch = object.__new__(LocalPatchAssembly)
        patch._assemble(ROOT, folder, base, cfg, rows, small[1], small[2], small[3],
                        canonical_hash(descriptor), descriptor)
        patch.runtime_factories = dict(inner=BoundedInnerDynamics, native=BoundedNativePhysics)
        ac = ACVSAssembly(patch, load_config(ROOT / 'configurations/ac_vs_functional_v1.json'))
        stage = 'static_H1_H2_projection'
        typed, evidence = verify_typed_source(base, ROOT)
        h = H1H2Assembly(ac, base, typed, evidence, engineering=True)
        save_artifact(h, typed, folder / 'typed_artifact')
        gap = patch.matrix('gap/laplacian_S.npz')
        outside = patch.external_matrix('gap/laplacian_S.npz')
        counts = {p: pop.group_count for p, pop in h.populations.items()}
        members = {p: int(pop.counts.sum()) for p, pop in h.populations.items()}
        cores = {p: len(h.core_population_rows.get(p, [])) for p in h.populations}
        record = dict(status='STATIC_GRAPH_READY_DYNAMIC_BACKGROUND_BLOCKED',
            actual_eccentricity_deg=geometry['actual_eccentricity_deg'], center_deg=center,
            selection_rule=geometry['selection_rule'], counts=counts, represented_members=members,
            core_counts=cores, H1_groups=len(h.h1_rows), H2_groups=len(h.h2_rows),
            internal_gap_nnz=gap.nnz, exterior_gap_nnz=outside.nnz,
            gap_weights_changed=False, exterior_gap_ports_must_not_be_zeroed=True,
            source_artifact=base.artifact_sha256, selected_artifact=h.artifact_sha256,
            unresolved_evidence=missing, local_initial_state_created=False,
            local_background_solved=False, forward_seconds=0., scientific_parameters_changed=False,
            FIT_READY=False, background_qualified=False, production_adopted=False)
        ref = ROOT / 'reports/layerwise_physiology/RGC_functional_v1_laptop_20261029'
        old = {r['type']: r for r in json.loads((ref / 'source_coupled_v1/analysis/TYPE_COMPLETION.json').read_bytes())}
        rows_csv = []
        for pop in RGC:
            rows_csv.append(dict(type=pop, near_eccentricity_deg=geometry['actual_eccentricity_deg'],
                near_core_groups=cores[pop], near_retained_groups=counts[pop], near_members=members[pop],
                near_function='NOT_RUN_SOURCE_EVIDENCE_BLOCKED', near_V_mV='', near_rate_Hz='',
                reference_eccentricity_deg=4.73, reference_groups=old[pop]['groups'],
                reference_function=old[pop]['status'], reference_V_mV=old[pop]['baseline_V_mV'],
                reference_rate_Hz=old[pop]['baseline_source_rate_Hz']))
        with (out / 'TWO_POSITION_COVERAGE.csv').open('x', encoding='utf8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0]));w.writeheader();w.writerows(rows_csv)
        write(out / 'REFERENCE_473_REUSE.json', dict(
            final_result_path=str((ref / 'FINAL_RESULT.json').relative_to(ROOT)),
            final_result_sha256=digest(ref / 'FINAL_RESULT.json'),
            curves_path=str((ref / 'source_coupled_v1/analysis/ALL_TYPE_MEAN_TIME_COURSES.csv').relative_to(ROOT)),
            recomputed=False, normalized_to_near_background=False))
        write(out / 'STATIC_COVERAGE_RESULT.json', record)
        print(json.dumps(record, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v), flush=True)
    except BaseException as exc:
        import traceback
        write(out / (args.attempt + '_FAILURE.json'), dict(stage=stage, error=str(exc),
            traceback=traceback.format_exc(), dynamic_model_started=False,
            original_source_evidence_block=missing, no_guard_bypassed=True))
        raise
    finally:
        ledger['independent_static_work_wall_seconds'] = (
            ledger.get('independent_static_work_wall_seconds', 0.) + time.perf_counter() - start)
        ledger['total_preparation_wall_seconds'] = (ledger['this_session_wall_seconds']
                                                  + ledger['independent_static_work_wall_seconds'])
        ledger['historical_plus_this_session_seconds'] = (ledger['historical_model_wall_seconds']
                                                       + ledger['total_preparation_wall_seconds'])
        ledger['source_forward_seconds'] = 0.
        ledger['new_output_bytes'] = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        ledger['status'] = 'BLOCKED_SOURCE_EVIDENCE_STATIC_WORK_PRESERVED'
        (out / 'RESOURCE_LEDGER.json').write_text(json.dumps(ledger, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
