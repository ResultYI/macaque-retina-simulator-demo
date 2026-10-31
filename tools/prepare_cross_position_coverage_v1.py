"""Geometric near-center choice and original source construction, no time advance."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.group_parameter_binding import build_group_construction_seed
from retina_sim.group_time_storage import save_bundle
from retina_sim.local_patch_graph import _select_bounded, LocalPatchAssembly
from retina_sim.spatial_group_graph import digest
from retina_sim.spatial_group_model import GroupModelAssembly, RGC
from retina_platform.config import RunConfig


def write(path, value):
    from retina_sim.parameter_system import data
    with path.open('x', encoding='utf8') as stream:
        json.dump(data(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def choose_geometry(artifact, center_deg=None):
    m = json.loads((artifact / 'manifest.json').read_bytes())
    scale = m['configuration']['mm_per_degree']
    summaries, candidates = [], []
    for pop in m['populations']:
        xy = np.load(artifact / 'populations' / pop / 'centroid_mm.npy') / scale
        r = np.linalg.norm(xy, axis=1)
        nearest = int(np.argmin(r))
        summaries.append(dict(population=pop, coordinate_role=m['populations'][pop]['coordinate_role'],
            groups=len(r), nearest_radius_deg=float(r[nearest]), farthest_radius_deg=float(r.max()),
            nearest_xy_deg=xy[nearest], nearest_parent_group_row=nearest))
        if pop in RGC:
            candidates.append((float(r[nearest]), RGC.index(pop), pop, nearest, xy[nearest]))
    radius, _, pop, row, center = min(candidates, key=lambda x: (x[0], x[1], x[3]))
    if center_deg is not None:
        center=np.asarray(center_deg,dtype=float)
        if center.shape!=(2,) or not np.isfinite(center).all():
            raise ContractError('Explicit finite x/y center in visual degrees required')
        candidates=[]
        for name in RGC:
            xy=np.load(artifact/'populations'/name/'centroid_mm.npy')/scale
            distance=np.linalg.norm(xy-center,axis=1);nearest=int(np.argmin(distance))
            if distance[nearest]<=.15:
                candidates.append((float(distance[nearest]),RGC.index(name),name,nearest))
        if not candidates:
            raise ContractError('Requested center has no existing RGC centroid in the frozen .15deg core')
        _,_,pop,row=min(candidates)
        radius=float(np.linalg.norm(center))
    presence = {}
    for name in m['populations']:
        xy = np.load(artifact / 'populations' / name / 'centroid_mm.npy') / scale
        d = np.linalg.norm(xy - center, axis=1)
        presence[name] = dict(coordinate_role=m['populations'][name]['coordinate_role'],
            core_centroid_groups=int((d <= .15).sum()),
            inner_geometric_groups=int((d <= .3).sum()),
            native_geometric_groups=int((d <= .6).sum()))
    return dict(selection_rule=('Minimum radius among existing frozen RGC group centroids' if center_deg is None else 'Explicit requested center; nearest existing core RGC target')+'; deterministic type/row tie break; no response-based selection',
        selected_type=pop, selected_parent_group_row=row, center_deg=center,
        actual_eccentricity_deg=radius, exact_zero_claimed=False,
        parent_declared_output_radius_deg=m['configuration']['output_radius_deg'],
        parent_has_broader_ancestor_rows=True, population_coordinate_bounds=summaries,
        geometric_presence=presence, radii_deg=dict(core=.15, inner=.3, native=.6))


def selection(base, cfg, present, check):
    packets = []
    for pop in present:
        request = dict(cfg, target_rgc_type=pop, target_rgc_id=None)
        packets.append(_select_bounded(base, request, check))
    rows = {p: np.unique(np.concatenate([s[0][p] for s in packets])) for p in packets[0][0]}
    contacts = {p: np.unique(np.concatenate([s[1][p] for s in packets])) for p in packets[0][1]}
    core = {p: np.unique(np.concatenate([s[2][p] for s in packets])) for p in packets[0][2]}
    target = packets[present.index(cfg['target_rgc_type'])][3]
    return rows, contacts, core, target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--historical-seconds', type=float, default=4184.440768200063)
    p.add_argument('--fixed-geometry', type=Path)
    p.add_argument('--center-deg',type=float,nargs=2)
    p.add_argument('--parent-construction',type=Path)
    p.add_argument('--selection-template',type=Path)
    p.add_argument('--max-rss-gib',type=int,choices=[8,24],default=8)
    args = p.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'reports'):
        raise ContractError('Fresh research output under reports required')
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    stage = 'geometry'
    guard = MemoryGuard(args.max_rss_gib * 1024**3, 8 * 1024**3)
    def check():
        if time.perf_counter() - start > 820 or (out / 'REQUEST_STOP').exists():
            raise TimeoutError('Bounded source preparation stopped')
        if not guard.check():
            raise MemoryError('Preparation memory guard: ' + str(guard.stop_reason))
    ledger = dict(start_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User autonomous continuation; parent cross-position functional coverage delegation after RGC summary review',
        historical_model_wall_seconds=args.historical_seconds, this_session_wall_seconds=0.,
        per_process_max_seconds=900, deadline_from_previous_task_is_archival=True,
        source_forward_seconds=0., FIT_READY=False, background_qualified=False, production_adopted=False)
    try:
        artifact = ROOT / 'reports/spatial_group_graph_v02_20261025_v1/artifact_r2'
        geometry = choose_geometry(artifact,args.center_deg)
        if args.fixed_geometry:
            frozen = json.loads(args.fixed_geometry.read_bytes())
            for key in ('selected_type', 'selected_parent_group_row', 'actual_eccentricity_deg'):
                if geometry[key] != frozen[key]:
                    raise ContractError('Previously fixed geometry changed: ' + key)
            if not np.array_equal(geometry['center_deg'], frozen['center_deg']):
                raise ContractError('Previously fixed center changed')
        write(out / 'GEOMETRY_SELECTION.json', geometry)
        present = [n for n in RGC if geometry['geometric_presence'][n]['core_centroid_groups']]
        write(out / 'PROTOCOL.json', dict(schema='cross-position-functional-coverage-1',
            positions=[dict(name='requested_position', eccentricity_deg=geometry['actual_eccentricity_deg']),
                       dict(name='reference', eccentricity_deg=4.73,
                            reuse='reports/layerwise_physiology/RGC_functional_v1_laptop_20261029/source_coupled_v1')],
            present_core_RGC_types=present, duration_seconds=.9,
            probe_epochs=[['native_cone_release_x0.9', .05, .2], ['native_cone_release_x1.1', .35, .5], ['rod_light_x2', .6, .75]],
            no_per_position_dynamic_normalization=True, new_eccentricity_parameter_function=False,
            source_construction='Existing construction t=0 reused if supplied, otherwise original algebra on unchanged parent graph; no lattice rebuild or global forward integration',
            source_background_must_be_newly_solved=True, old473_seed_used_as_new_equilibrium=False,
            boundary='Existing near-center construction t=0 exterior preparation rule; not the historical far-point equilibrated-annulus source',
            missing_core_types_not_filled_from_halo=True, new_optical_capture_not_assumed=True))
        stage = 'parent_source_construction'
        base = GroupModelAssembly(artifact)
        policy = ROOT / base.manifest['configuration']['source_run'] / 'input_policy.json'
        binding = dict(path=policy.relative_to(ROOT).as_posix(), sha256=digest(policy))
        if args.parent_construction:
            source_seed=args.parent_construction.resolve()
            if not source_seed.is_relative_to(ROOT/'reports'):
                raise ContractError('Existing research parent construction must remain under reports')
            parent=json.loads((source_seed/'manifest.json').read_bytes())
            source=parent['identity']['seed']
            if (parent['kind']!='source_construction_seed' or parent['identity']['artifact_sha256']!=base.artifact_sha256
                or source.get('classification')!='SOURCE_CONSTRUCTION_IVP_NOT_PREPARED_BACKGROUND'
                or source.get('neural_model_advanced') is not False or source.get('time_seconds')!=0.):
                raise ContractError('Reusable parent must be this graph unadvanced construction t=0')
        else:
            check()
            seed = build_group_construction_seed(base, ROOT, input_policy_binding=binding,
                rng_seed=RunConfig.project_default(ROOT).seed, time_seconds=0., guard=check)
            bg = np.asarray(seed.input_reference['native_background_td'])
            payload = dict(states=seed.states, spike_states=seed.spike_states,
                inner_parameters=seed.inner_parameters, observation_prehistory=seed.observation_prehistory,
                input_reference=seed.input_reference, time_seconds=seed.time_seconds,
                native_inputs={0: bg.copy()}, rod_inputs={0: np.full(base.populations['rods'].group_count,
                    seed.input_reference['rod_background_rstar_s'])})
            save_bundle(out / 'parent_construction_seed', kind='source_construction_seed',
                identity=dict(artifact_sha256=base.artifact_sha256, seed=seed.identity, input_policy=binding),
                payload=payload, qualification=dict(default_adopted=False, fit_eligible=False, qualified_background=False))
            source_seed=out/'parent_construction_seed'
        stage = 'new_local_selection'
        cfg = (deepcopy(json.loads(args.selection_template.read_bytes())['configuration']) if args.selection_template
               else json.loads((ROOT/'configurations/local_patch_v2/halo_030.json').read_bytes()))
        cfg.pop('azimuth_deg',None)
        cfg.update(eccentricity_deg=geometry['actual_eccentricity_deg'],
            rf_center_deg=np.asarray(geometry['center_deg']).tolist(),
            target_rgc_type=geometry['selected_type'], target_rgc_id=None,
            input_identity=dict(path=str((source_seed/'manifest.json').relative_to(ROOT)),
                                sha256=digest(source_seed/'manifest.json')),
            position_provenance=geometry['selection_rule']+'; actual requested-center radius recorded; no single fine-cell claim',
            target_provenance='Geometry-selected model group only; no electrophysiological target mapping or output-based location search')
        small = selection(base, cfg, present, check)
        bigcfg = deepcopy(cfg); bigcfg['patch_geometry']['support_radius_deg'] = .6
        large = selection(base, bigcfg, present, check)
        rows = dict(small[0])
        for name in ('cones', 'HC'):
            if not np.isin(rows[name], large[0][name]).all():
                raise ContractError('Native0.6 does not contain original0.3 ancestors')
            rows[name] = large[0][name]
        arrays = {'population__' + n: v for n, v in rows.items()}
        arrays.update({'contact__' + n: v for n, v in small[1].items()})
        arrays.update({'core__' + n: v for n, v in small[2].items()})
        folder = out / 'local_static'; folder.mkdir()
        np.savez_compressed(folder / 'selection.npz', **arrays)
        descriptor = dict(schema='geometry-selected-cross-position-native060-inner030-1',
            configuration=cfg, parent_artifact=base.artifact_sha256,
            selection_sha256=canonical_hash(arrays), native_radius=.6, inner_radius=.3,
            present_core_types=present, no_new_geometry=True)
        write(folder / 'selection_identity.json', descriptor)
        patch = object.__new__(LocalPatchAssembly)
        patch._assemble(ROOT, folder, base, cfg, rows, small[1], small[2], small[3],
                        canonical_hash(descriptor), descriptor)
        write(out / 'STATIC_COUNTS.json', dict(
            counts={n: p.group_count for n, p in patch.populations.items()},
            members={n: int(p.counts.sum()) for n, p in patch.populations.items()},
            core={n: len(patch.core_population_rows[n]) for n in patch.populations},
            source_parent_construction_new=not bool(args.parent_construction), old_local_seed_reused=False,
            parent_construction_source=str(source_seed.relative_to(ROOT)),
            source_forward_seconds=0., qualified_background=False))
        ledger['status'] = 'STATIC_SELECTION_AND_FRESH_PARENT_CONSTRUCTION_READY'
    except BaseException as exc:
        import traceback
        ledger['status'] = 'BLOCKED_OR_FAILED_PREPARATION'
        write(out / 'PREPARATION_FAILURE.json', dict(stage=stage, error=type(exc).__name__,
            message=str(exc), traceback=traceback.format_exc(), source_forward_seconds=0.,
            old_local_seed_substituted=False, hash_guard_bypassed=False))
        raise
    finally:
        ledger['stage'] = stage
        ledger['this_session_wall_seconds'] = time.perf_counter() - start
        ledger['historical_plus_this_session_seconds'] = (
            ledger['historical_model_wall_seconds'] + ledger['this_session_wall_seconds'])
        ledger['new_output_bytes'] = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        write(out / 'RESOURCE_LEDGER.json', ledger)
        print(json.dumps(ledger), flush=True)


if __name__ == '__main__':
    main()
