"""Freeze a finite-halo protocol and reject irreparable fixed-core contact gaps.

No neural integration or new HC sampling is attempted after structural failure.
The exact target A row is built on the approved source for a discrete mass audit.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
import time
import traceback

import numpy as np
from scipy import sparse
from scipy.io import loadmat
from scipy.spatial import cKDTree

from retina_sim.contracts import canonical_hash
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.hc_current import current_hc
from retina_sim.hc_outer_candidate import OuterHCCandidate
from retina_sim.hc_biexponential import build_spatial_operators, support_radius
from retina_sim.source_atlas import SourceDensityAtlas
from retina_sim.atlas_s_partition import EffectiveMulattaSPartition


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    began = datetime.now(timezone.utc).isoformat(); clock = time.perf_counter()
    protocol_path = root/'configurations/research/HC_finite_dynamic_halo_v1.json'
    protocol = json.loads(protocol_path.read_bytes())
    frozen = dict(frozen_utc=began, protocol_sha256=sha(protocol_path), protocol=protocol,
                  no_neural_results_examined=True)
    write(out/'FROZEN_PROTOCOL.json', frozen)
    result = dict(started_utc=began, protocol_sha256=frozen['protocol_sha256'], status='RUNNING',
                  production_adopted=False, physiological_validation=False)
    sources = [protocol_path, Path(__file__), root/'retina_sim/hc_biexponential.py',
               root/'retina_sim/hc_support_closure.py', root/'retina_sim/hc_outer_candidate.py',
               root/'retina_sim/h1_h2_effective.py', root/'retina_sim/vh2005.py']
    try:
        base = root/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/executions/assemble_native_01'
        assembly_path = base/'NATIVE_ASSEMBLY_RESULT.json'
        assembly = json.loads(assembly_path.read_bytes())
        if assembly['current_HC_identity'] != current_hc(root)['identity']:
            raise RuntimeError('Frozen central HC selection changed')
        cones, hcs = (FrozenDiskTable(root, assembly[name]) for name in ('cone_table', 'hc_table'))
        cx, hx = cones.coordinates['opl_pedicle'], hcs.coordinates['soma']
        ct, ht = cones.extras['subtype'].astype('U1'), hcs.extras['subtype'].astype('U2')
        raw = sparse.load_npz(base/'RAW_HC_BY_CONE.npz').tocsr()
        j = protocol['target_H1_parent_row']; target = hx[j]
        if ht[j] != 'H1': raise RuntimeError('Frozen target is not H1')
        np.testing.assert_array_equal(target, protocol['stimulus']['center_mm'])
        master_path = root/'data/atlas/GIN_69_6_HC2_BC1_SUPPORT_V2/HC_master.json'
        master = json.loads(master_path.read_bytes())
        bounds = np.asarray(master['boundary']['canonical_bounds_mm'])
        profile = OuterHCCandidate(root)
        radii = np.array([profile.contact_radius_mm(r, k) for r, k in zip(np.linalg.norm(hx, axis=1), ht)])
        roots = np.concatenate([v.derivative().roots(extrapolate=False) for v in profile.contacts.values()])
        nodes = np.r_[np.linspace(0, profile.join, 10001), profile.join, 5., 6.,
                      roots[np.isfinite(roots) & (roots >= profile.join) & (roots <= 6)]]
        max_radius = max(float(profile.contact_radius_mm(nodes, kind).max()) for kind in ('H1', 'H2'))
        # The inner Hermite contact join can have extrema between endpoints.
        # A conservative micron pad makes the exclusion certificate independent of grid extremum location.
        contact_envelope = max_radius + .001
        edge_distance = np.minimum.reduce([cx[:, 0]-bounds[0], bounds[1]-cx[:, 0],
                                           cx[:, 1]-bounds[2], bounds[3]-cx[:, 1]])
        sealed = edge_distance > contact_envelope
        neighbors = cKDTree(cx).query_ball_point(hx, radii)
        rr, cc = [], []
        for row, ids in enumerate(neighbors):
            ids = np.asarray(ids, int); ids = ids[(ct[ids] != 'S') | (ht[row] == 'H2')]
            rr.extend([row]*len(ids)); cc.extend(ids)
        geometry = sparse.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=raw.shape)
        # Compare sealed rows only: no new outside-fixed-region HC can contact these cones.
        if (geometry[:, sealed] != raw[:, sealed]).nnz:
            raise RuntimeError('Fixed-region raw contacts disagree with the preserved profile')
        B_count = np.asarray(geometry[ht == 'H1'].sum(0)).ravel()
        gaps = np.flatnonzero(sealed & (ct != 'S') & (B_count == 0))
        dtarget = np.linalg.norm(cx-target, axis=1)
        gaps = gaps[np.argsort(dtarget[gaps])]
        core = raw.indices[raw.indptr[j]:raw.indptr[j+1]]
        if not len(gaps):
            raise RuntimeError('No fixed-core obstruction: this preflight must be extended to continuous halo construction')
        result['fixed_region'] = dict(bounds_mm=bounds.tolist(), HC_count=hcs.count,
            conservative_max_contact_mm=contact_envelope, contacts_recomputed_exact_on_sealed_columns=True,
            sealed_LM_cones=int(np.sum(sealed & (ct != 'S'))), sealed_LM_B_gaps=len(gaps),
            core_cone_parent_rows=core.tolist(), core_all_have_H1_B=bool(np.all(B_count[core] > 0)))
        print('Fixed-region structural gaps:', len(gaps), flush=True)

        source = root/'reports/CONE_HC_OUTER_SOURCE_EXTENSION_20261102_v3/cones_01'
        record = json.loads((source/'RESULT.json').read_bytes())
        for name, digest in record['files'].items():
            if sha(source/name) != digest: raise RuntimeError('Outer source asset changed: '+name)
        xy = np.load(source/'cone_inner_segment_mm.npy')
        ped = np.load(source/'cone_pedicle_mm.npy')
        official_rows = np.load(source/'official_source_rows.npy')
        config_path = root/'configurations/research/HC_biexponential_vh2007_reference_v1.json'
        config = json.loads(config_path.read_bytes())
        cutoff = support_radius(config['H1']['lambda_l_um']/1000, config['tail_mass_budget'])
        if np.linalg.norm(target)+cutoff > record['IS_radius_mm']:
            raise RuntimeError('Target support not covered by approved source')
        selected = np.flatnonzero(np.linalg.norm(ped-target, axis=1) <= cutoff)
        keys = tuple('sdr_human_54eb0a9ef905_row:'+str(r) for r in official_rows[selected])

        # Label complete source macros, retaining the existing deterministic subtype rule.
        atlas = SourceDensityAtlas(root, profile_id='retina_atlas_human_borrowed_disk3mm_v1')
        expanded = copy.copy(atlas.cone_master)
        official_path = root/'vendor_pinned/isetbio/data/sdr/isetbio-mosaics/cone_lattices/right/nominal-fov-58deg.mat'
        if sha(official_path) != record['official_source_sha256']: raise RuntimeError('Official source changed')
        official = loadmat(official_path, simplify_cells=True)
        if float(official['params']['radius'])/1000 < 6.5: raise RuntimeError('Incomplete label context')
        all_xy = -np.asarray(official['rfPositions'], float)/1000
        label_rows = np.flatnonzero(np.linalg.norm(all_xy, axis=1) <= 6.5)
        expanded.positions = all_xy[label_rows]
        expanded.source_keys = tuple('sdr_human_54eb0a9ef905_row:'+str(r) for r in label_rows)
        expanded.bounds = (-6.5, 6.5, -6.5, 6.5)
        expanded._tiles = {}; expanded._tile_indices = {}
        original_tile = expanded.tile
        def supported_tile(tx, ty, size):
            corners = np.array([[tx, ty], [tx+1, ty], [tx, ty+1], [tx+1, ty+1]])*size
            if np.linalg.norm(corners, axis=1).max() > 6.5:
                raise RuntimeError('Label macro outside complete source context')
            return original_tile(tx, ty, size)
        expanded.tile = supported_tile; atlas.cone_master = expanded
        print('Assign exact target-row source types:', len(selected), flush=True)
        labels, audit = EffectiveMulattaSPartition(atlas).assign(atlas._table('cones', keys, xy[selected]))
        labels = np.asarray(labels)
        old_rows = np.array([int(k.decode().split(':')[-1]) for k in cones.keys])
        common, new_i, old_i = np.intersect1d(official_rows[selected], old_rows, return_indices=True)
        np.testing.assert_array_equal(labels[new_i], ct[old_i])
        np.testing.assert_array_equal(ped[selected[new_i]], cx[old_i])
        local = np.flatnonzero(np.isin(official_rows[selected], old_rows[core]))
        raw_target = sparse.csr_matrix((np.ones(len(local)), (np.zeros(len(local), int), local)),
                                       shape=(1, len(selected)))
        ops = build_spatial_operators(raw_target, ped[selected], target[None, :], labels, np.array(['H1']),
                                      configuration=config, parent_disk_radius_mm=6.)
        target_A = ops['A']; sparse.save_npz(out/'TARGET_A_ONLY.npz', target_A)
        # This single-row B is deliberately not used or saved: it is not the actual feedback graph.
        np.savez_compressed(out/'TARGET_A_MEMBERS.npz', cone_candidate_rows=selected,
            official_source_rows=official_rows[selected], cone_types=labels,
            cone_IS_mm=xy[selected], cone_pedicle_mm=ped[selected], target_hc_parent_row=j,
            fixed_gap_cone_parent_rows=gaps, core_cone_parent_rows=core)
        write(out/'SUBTYPE_PROVENANCE.json', audit)
        weights = target_A.toarray()[0]
        distance = np.linalg.norm(ped[selected]-target, axis=1)
        gap_local = np.array([np.flatnonzero(official_rows[selected] == old_rows[i])[0] for i in gaps])
        trials = []
        for radius in protocol['dynamic_radii_mm']:
            dynamic = (distance <= radius) & (labels != 'S')
            trials.append(dict(radius_mm=radius, dynamic_LM_count=int(dynamic.sum()),
                target_A_dynamic_mass=float(weights[dynamic].sum()),
                target_A_held_background_mass=float(weights[~dynamic].sum()),
                target_A_missing_source_mass=0.,
                known_dynamic_B_gaps_in_fixed_region=int(np.sum(dtarget[gaps] <= radius)),
                target_A_mass_on_fixed_B_gaps=float(weights[gap_local[dtarget[gaps] <= radius]].sum()),
                admission='FAIL_DYNAMIC_LOCAL_B_INCOMPLETE', neural_run=False))
        witnesses = []
        h1 = np.flatnonzero(ht == 'H1')
        for i, ai in zip(gaps, gap_local):
            d = np.linalg.norm(hx-cx[i], axis=1)
            neighbor = h1[np.argmin(d[h1]-radii[h1])]
            witnesses.append(dict(cone_parent_row=int(i), cone_key=cones.keys[i].decode(),
                cone_type=str(ct[i]), cone_IS_mm=cones.coordinates['rf'][i].tolist(),
                cone_pedicle_mm=cx[i].tolist(), distance_to_target_mm=float(dtarget[i]),
                target_A_weight=float(weights[ai]), nearest_H1_parent_row=int(neighbor),
                nearest_H1_distance_mm=float(d[neighbor]), contact_radius_mm=float(radii[neighbor]),
                shortfall_um=float((d[neighbor]-radii[neighbor])*1000),
                distance_to_fixed_region_edge_mm=float(edge_distance[i]),
                outside_fixed_region_cannot_repair=bool(edge_distance[i] > contact_envelope)))
        # Directed type quotient: no path here implies no path in any compliant fine graph.
        e = geometry.tocoo()
        edges = set(zip(ct[e.col].tolist(), ht[e.row].tolist()))
        edges |= {(str(ht[row]), str(ct[col])) for row, col in zip(e.row, e.col)
                  if (ht[row] == 'H1' and ct[col] != 'S') or (ht[row] == 'H2' and ct[col] == 'S')}
        edges |= {('L', 'H1'), ('M', 'H1')}
        ancestors = {'H1'}
        while True:
            updated = ancestors | {source_type for source_type, dest in edges if dest in ancestors}
            if updated == ancestors: break
            ancestors = updated
        if ancestors != {'H1', 'L', 'M'}: raise RuntimeError('Unexpected H1 causal paths')
        h2_evidence = root/'reports/H2_CONTACT_EVIDENCE_20261102_v4/NEIGHBORHOOD.json'
        h2 = json.loads(h2_evidence.read_bytes())['patches'][0]['missing_S'][0]
        result.update(status='BLOCKED_FIXED_REGION_LM_FEEDBACK_GAPS',
            target_H1_parent_row=j, target_H1_key=hcs.keys[j].decode(), target_OPL_mm=target.tolist(),
            target_A_support_radius_mm=cutoff, target_A_source_count=len(selected),
            target_A_LM_count=int(np.sum(labels != 'S')), overlap_labels_checked=len(common),
            target_A_scope='one complete target row only; not a continuous dynamic HC parent',
            trials=trials, witnesses=witnesses,
            H2_S_causal_audit=dict(original_missing_S_official_row=h2['official_source_row'],
                directed_type_edges=sorted(edges), H1_ancestor_types=sorted(ancestors),
                original_H2_S_gap_is_H1_ancestor=False, S_to_H1_target_A_nnz=int(np.count_nonzero(weights[labels == 'S'])),
                scope='observed local type edges plus declared long A rule; no H2-to-LM feedback or explicit lateral coupling'),
            continuous_HC_generation='NOT_STARTED: fixed-region obstruction cannot be repaired by added outside HC',
            background='NOT_RUN_ON_REAL_GRAPH', HC_120ms='NOT_RUN', cone_surround='NOT_RUN', RGC='NOT_RUN',
            halo_response_convergence='NOT_EVALUATED',
            structure_gate=False, boundary_gate=None, mechanism_gate=None,
            stop_reason='Frozen dynamic sets include actual LM ancestors with zero B, deep inside unchanged HC region',
            alternative_not_applied='No missing LM clamping, nearest-neighbor addition, radius expansion or contact-rule retuning')
        sources += [assembly_path, base/'RAW_HC_BY_CONE.npz', master_path, source/'RESULT.json',
                    config_path, h2_evidence, root/'retina_sim/atlas_s_partition.py']
        write(out/'SOURCE_IDENTITIES.json', {str(p.relative_to(root)): sha(p) for p in sources})
    except Exception as error:
        result.update(status='FAILED_RETAINED', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(ended_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-clock)
        write(out/'RESULT.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'wall_seconds', 'trials')}, indent=2), flush=True)


if __name__ == '__main__': main()
