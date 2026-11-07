"""Materialize source-based outer HC patches and local B on official Cone rows."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.io import loadmat
from retina_sim.hc_outer_candidate import OuterHCCandidate
from retina_sim.hc_mosaic import generate_joint_hc
from retina_sim.geometry import RetinalDomain
from retina_sim.source_atlas import SourceDensityAtlas
from retina_sim.atlas_s_partition import EffectiveMulattaSPartition
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.hc_biexponential import support_radius


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat(); clock = time.perf_counter()
    receipt = dict(started_utc=started, status='RUNNING', production_selected=False, neural_run=False)
    try:
        source = root/'reports/CONE_HC_OUTER_SOURCE_EXTENSION_20261102_v3/cones_01'
        record = json.loads((source/'RESULT.json').read_bytes())
        for name, digest in record['files'].items():
            if sha(source/name) != digest: raise RuntimeError('Cone candidate bytes changed')
        xy = np.load(source/'cone_inner_segment_mm.npy')
        ped = np.load(source/'cone_pedicle_mm.npy')
        rows = np.load(source/'official_source_rows.npy')
        cone_tree = cKDTree(ped)
        atlas = SourceDensityAtlas(root, profile_id='retina_atlas_human_borrowed_disk3mm_v1')
        original_master = atlas.cone_master
        expanded = copy.copy(original_master)
        official = root/'vendor_pinned/isetbio/data/sdr/isetbio-mosaics/cone_lattices/right/nominal-fov-58deg.mat'
        if sha(official) != record['official_source_sha256']:
            raise RuntimeError('Official source bytes changed')
        data = loadmat(official,simplify_cells=True)
        all_xy = -np.asarray(data['rfPositions'],dtype=float)/1000
        if float(data['params']['radius'])/1000 < 6.5:
            raise RuntimeError('Official source domain does not enclose label context')
        label_rows = np.flatnonzero(np.linalg.norm(all_xy,axis=1)<=6.5)
        expanded.positions = all_xy[label_rows]
        expanded.source_keys = tuple('sdr_human_54eb0a9ef905_row:'+str(r) for r in label_rows)
        expanded.bounds = (-6.5, 6.5, -6.5, 6.5)
        expanded._tiles = {}; expanded._tile_indices = {}
        atlas.cone_master = expanded
        # These patches and their complete label macros lie strictly inside the disk.
        # A disk asset must never advertise its empty corners as complete tiles.
        original_tile = expanded.tile
        def supported_tile(tx, ty, size):
            corners = np.array([[tx,ty],[tx+1,ty],[tx,ty+1],[tx+1,ty+1]])*size
            if np.linalg.norm(corners,axis=1).max() > 6.5:
                raise RuntimeError('Label macro exceeds complete extended Cone disk')
            return original_tile(tx,ty,size)
        expanded.tile = supported_tile
        partition = EffectiveMulattaSPartition(atlas)
        assembly = root/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/executions/assemble_native_01/NATIVE_ASSEMBLY_RESULT.json'
        assembly_record = json.loads(assembly.read_bytes())
        old_cones = FrozenDiskTable(root, assembly_record['cone_table'])
        old_hc = FrozenDiskTable(root, assembly_record['hc_table'])
        selected = np.flatnonzero((old_cones.coordinates['rf'][:,0] > 1.02)
                                 & (old_cones.coordinates['rf'][:,0] < 1.08)
                                 & (np.abs(old_cones.coordinates['rf'][:,1]) < .03))
        keys = tuple(k.decode('ascii') for k in old_cones.keys[selected])
        labels, _ = partition.assign(atlas._table('cones', keys, old_cones.coordinates['rf'][selected]))
        if not np.array_equal(labels, old_cones.extras['subtype'][selected].astype('U1')):
            raise RuntimeError('Expanded source changed existing overlap subtype assignments')
        profile = OuterHCCandidate(root)
        (out/'PROFILE.json').write_text(json.dumps(profile.metadata,indent=2),encoding='utf-8')
        contact_nodes = np.r_[0., .58, .6, profile.join, 5., 6.]
        contact_roots = np.concatenate([c.derivative().roots(extrapolate=False) for c in profile.contacts.values()])
        contact_nodes = np.r_[contact_nodes, contact_roots[np.isfinite(contact_roots) & (contact_roots >= profile.join) & (contact_roots <= 6)]]
        max_contact = max(float(profile.contact_radius_mm(contact_nodes,k).max()) for k in ('H1','H2'))
        spatial_cfg = json.loads((root/'configurations/research/HC_biexponential_vh2007_reference_v1.json').read_bytes())
        long_radius = support_radius(spatial_cfg['H1']['lambda_l_um']/1000, spatial_cfg['tail_mass_budget'])
        witness = json.loads((root/'reports/CONE_HC_SUPPORT_CLOSURE_20261102_v2/audit_01/RESULT.json').read_bytes())
        target = np.array(witness['hc_anchor_mm'])
        patches = []
        receipt['patches'] = patches
        receipt['label_context'] = dict(radius_mm=6.5, role='complete subtype macros only; no pedicle mapping or neural owners beyond6mm')
        merged_xy = [np.array(old_hc.coordinates['soma'])]
        merged_types = [old_hc.extras['subtype'].astype('U2')]
        merged_keys = [old_hc.keys.astype('U512')]
        for name, center in [('transition',[2.4,0.]), ('outer_first_hop',[4.4,0.]), ('source_anchor',[5.5,0.])]:
            center = np.array(center); half = .14
            bounds = [center[0]-half,center[0]+half,center[1]-half,center[1]+half]
            domain = RetinalDomain(bounds,bounds,lambda p:np.ones(len(p),bool),
                                  'HC_outer_local_validation_'+name,'source-based-local-patch')
            options = json.loads((root/'configurations/hc_joint_v2_generator.json').read_bytes())['generator']
            print('generate',name,flush=True)
            hc = generate_joint_hc(profile,domain,20261102,5000,options=options)
            hx = hc.positions.values; ht = hc.types
            r = np.linalg.norm(hx,axis=1)
            radii = np.array([profile.contact_radius_mm(v,k) for v,k in zip(r,ht)])
            neighbors = cone_tree.query_ball_point(hx,radii)
            candidate_rows = np.unique(np.concatenate([np.asarray(v,dtype=int) for v in neighbors]))
            # Include owner cones with no contacts so they cannot disappear from the audit.
            margin = float(max_contact+hc.diagnostics['maximum_interaction_support_mm'])
            owners = np.flatnonzero(np.all(np.abs(ped-center) < half-margin,axis=1))
            candidate_rows = np.union1d(candidate_rows, owners)
            keys = tuple('sdr_human_54eb0a9ef905_row:'+str(rows[i]) for i in candidate_rows)
            labels, label_audit = partition.assign(atlas._table('cones',keys,xy[candidate_rows]))
            ct = np.asarray(labels)
            rr, cc = [], []
            for j, neighbors_j in enumerate(neighbors):
                local = np.searchsorted(candidate_rows, neighbors_j)
                local = local[(ct[local]!='S') | (ht[j]=='H2')]
                rr.extend([j]*len(local)); cc.extend(local.tolist())
            rr = np.asarray(rr,dtype=int); cc = np.asarray(cc,dtype=int)
            raw = sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(hx),len(ct)))
            allowed = ((ht[rr]=='H1') & (ct[cc]!='S')) | ((ht[rr]=='H2') & (ct[cc]=='S'))
            raw_b = sparse.csr_matrix((np.ones(allowed.sum()),(cc[allowed],rr[allowed])),shape=raw.shape[::-1])
            mass = np.asarray(raw_b.sum(axis=1)).ravel()
            owner_local = np.searchsorted(candidate_rows,owners)
            complete = np.zeros(len(ct),bool); complete[owner_local] = True
            # Rim rows lack an exterior HC parent: retain raw contacts, never renormalize a crop.
            b = sparse.diags(np.divide(1.,mass,out=np.zeros_like(mass),where=(mass>0)&complete))@raw_b
            missing = owners[mass[owner_local]==0]
            if len(rr) and np.any(np.linalg.norm(ped[candidate_rows[cc]]-hx[rr],axis=1)>radii[rr]+1e-12):
                raise RuntimeError('Nonlocal feedback edge')
            np.testing.assert_allclose(np.asarray(b.sum(axis=1)).ravel()[(mass>0)&complete],1,rtol=0,atol=1e-12)
            patch_dir = out/name; patch_dir.mkdir()
            sparse.save_npz(patch_dir/'RAW_LOCAL_CONTACTS.npz',raw)
            sparse.save_npz(patch_dir/'RAW_TYPED_FEEDBACK.npz',raw_b)
            sparse.save_npz(patch_dir/'B_LOCAL.npz',b)
            np.savez_compressed(patch_dir/'MEMBERS.npz',cone_candidate_rows=candidate_rows,
                cone_official_rows=rows[candidate_rows],cone_types=ct,cone_is_mm=xy[candidate_rows],
                cone_pedicle_mm=ped[candidate_rows],hc_positions_mm=hx,hc_types=ht,
                hc_source_keys=hc.cell_ids,hc_contact_radius_mm=radii,owner_cone_candidate_rows=owners,
                missing_feedback_candidate_rows=missing,B_row_complete_local_geometry=complete)
            (patch_dir/'GENERATION.json').write_text(json.dumps(hc.diagnostics,indent=2),encoding='utf-8')
            (patch_dir/'SUBTYPE_PROVENANCE.json').write_text(json.dumps(label_audit,indent=2),encoding='utf-8')
            first = np.linalg.norm(ped[candidate_rows]-target,axis=1)<=long_radius
            first &= ct!='S'
            item = dict(name=name,hc_count=len(hx),cone_count=len(ct),raw_contact_count=raw.nnz,
                typed_feedback_edges=raw_b.nnz,complete_owner_B_edges=b.nnz,
                rim_B_rows_disabled=True,owner_cones=len(owners),owner_cones_without_B=len(missing),
                missing_by_type={k:int(np.sum(ct[owner_local][mass[owner_local]==0]==k)) for k in 'LMS'},
                margin_mm=margin,bounds_mm=bounds,first_hop_cones_here=int(first.sum()),
                first_hop_cones_here_with_B=int(np.sum(first&(mass>0))),
                H1_wide_support_beyond_6mm=int(np.sum((ht=='H1')&(r+long_radius>6))),
                required_H1_outer_envelope_mm=float(np.max(r[ht=='H1']+long_radius)),
                owner_missing_is_not_patch_edge=True,complete_recurrent_patch=False)
            paths = np.flatnonzero(first & complete & (mass>0))
            if len(paths):
                path_cone = int(paths[-1]); path_hc = int(raw_b.indices[raw_b.indptr[path_cone]])
                item['dependency_witness'] = dict(target_H1_id=witness['hc_id'],
                    target_H1_OPL_mm=target.tolist(),cone_official_row=int(rows[candidate_rows[path_cone]]),
                    cone_pedicle_mm=ped[candidate_rows[path_cone]].tolist(),cone_type=str(ct[path_cone]),
                    feedback_H1_key=hc.cell_ids[path_hc],feedback_H1_OPL_mm=hx[path_hc].tolist(),
                    target_A_distance_mm=float(np.linalg.norm(ped[candidate_rows[path_cone]]-target)),
                    B_distance_mm=float(np.linalg.norm(ped[candidate_rows[path_cone]]-hx[path_hc])),
                    B_contact_radius_mm=float(radii[path_hc]),
                    next_A_required_outer_radius_mm=float(np.linalg.norm(hx[path_hc])+long_radius))
            patches.append(item); print(json.dumps(item),flush=True)
            (patch_dir/'RESULT.json').write_text(json.dumps(item,indent=2),encoding='utf-8')
            merged_xy.append(hx);merged_types.append(ht);merged_keys.append(np.asarray(hc.cell_ids))
        np.savez_compressed(out/'PRESERVED_CENTRAL_PLUS_PATCHES.npz',hc_positions_mm=np.concatenate(merged_xy),
            hc_types=np.concatenate(merged_types),hc_source_keys=np.concatenate(merged_keys),old_hc_count=old_hc.count)
        with np.load(out/'PRESERVED_CENTRAL_PLUS_PATCHES.npz',allow_pickle=False) as merged:
            np.testing.assert_array_equal(merged['hc_positions_mm'][:old_hc.count], old_hc.coordinates['soma'])
            np.testing.assert_array_equal(merged['hc_types'][:old_hc.count], old_hc.extras['subtype'].astype('U2'))
            np.testing.assert_array_equal(merged['hc_source_keys'][:old_hc.count], old_hc.keys.astype('U512'))
        receipt.update(status='LOCAL_GEOMETRY_AND_B_MATERIALIZED_FULL_PARENT_NOT_QUALIFIED',
            patches=patches,old_hc_count_preserved=old_hc.count,
            old_cone_labels_checked=len(selected),old_cone_labels_exact=True,
            old_HC_positions_types_keys_exact=True,complete_owner_B_only=True,
            old_HC_binding=assembly_record['hc_table'],profile_sha256=profile.profile_hash,
            full_parent_assembled=False,closed_background_qualified=False,
            stop_reasons=['INTERIOR_TYPED_B_HOLES','OUTER_H1_A_SUPPORT_EXCEEDS_APPROVED_6MM'],
            production_defaults_changed=False,HC_RF='NOT_RUN',Cone_surround='NOT_RUN',RGC='NOT_RUN')
        files = [Path(__file__),root/'retina_sim/hc_outer_candidate.py',assembly,
                 root/'configurations/hc_joint_v2_generator.json',source/'RESULT.json']
        (out/'IDENTITIES.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in files},indent=2),encoding='utf-8')
        receipt['artifact_sha256'] = {p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    except Exception as error:
        receipt.update(status='FAILED_RETAINED',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-clock)
        (out/'RESULT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')


if __name__=='__main__': main()
