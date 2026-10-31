"""Build the approved two-anchor engineering candidate, preserving fine identity."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv, hashlib, json, sys, time, traceback
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.geometry_anchored_pooling import anchored_radius,complete_d8,select_row
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.disk_graph_storage import FrozenSparseRoute,array_identity
from retina_sim.background_checkpoint import MemoryGuard

OUT=ROOT/'reports/retinotopic_rf_revision/SPATIAL_POOLING_CANDIDATE_V1'
PARENT=ROOT/'reports/spatial_group_graph_v02_20261025_v1/artifact_r2'
PREVIOUS=ROOT/'reports/retinotopic_rf_revision/two_path_candidates_v1'
NAMES=('cones_to_diffuse_BC_ON','cones_to_diffuse_BC_OFF','diffuse_BC_ON_to_parasol_RGC_ON','diffuse_BC_OFF_to_parasol_RGC_OFF')


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def bound(path):return dict(path=Path(path).relative_to(ROOT).as_posix(),sha256=sha(path))
def write(path,value):
    with Path(path).open('x',encoding='utf8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
def parent_matrix(relative):
    with np.load(PARENT/relative,allow_pickle=False) as z:return sparse.csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(z['shape']))


def main():
    start=time.perf_counter();memory=MemoryGuard(24*1024**3,8*1024**3)
    def guard():
        if not memory.check():raise MemoryError(str((memory.stop_reason,memory.last)))
        if time.perf_counter()-start>180:raise TimeoutError('Structural construction180s guard')
    guard();manifest=read(PARENT/'manifest.json');routes=read(PARENT/'group_routes.json');previous=read(PREVIOUS/'RESULT.json')
    support=read(ROOT/'semi_reports/2026-11-02_HC/hc_connectivity_validation.json')['cone_support']['conservative_OPL_outer_radius_mm']
    locations={};selections={}
    for location,meta in previous['locations'].items():
        for b in [meta['selection'],meta['selection_identity']]:assert sha(ROOT/b['path'])==b['sha256']
        with np.load(ROOT/meta['selection']['path'],allow_pickle=False) as z:selections[location]={k:z[k] for k in z.files}
        locations[location]=meta
    tables={};groups={}
    def pop(name):
        if name not in tables:
            tables[name]=FrozenDiskTable(ROOT,manifest['populations'][name]['source_table']);g={}
            for key in ['counts','fine_to_group','member_global_rows','member_offsets']:
                relative=f'populations/{name}/{key}.npy';assert sha(PARENT/relative)==manifest['files'][relative]['sha256']
                g[key]=np.load(PARENT/relative,mmap_mode='r',allow_pickle=False)
            groups[name]=g
        return tables[name],groups[name]
    summaries=[];row_records=[]
    for name in NAMES:
        guard();record=routes[name];pre,post=record['pre'],record['post'];pt,pg=pop(pre);qt,qg=pop(post)
        route=FrozenSparseRoute(ROOT,record['source']);recipe=route.record['recipe']
        if pre=='cones':
            source_xy=pt.coordinates['opl_pedicle'];eligible=np.flatnonzero(pt.extras['subtype'].astype('U1')!='S')
            target_xy=source_xy[qt.extras['cone_row']]
            assert np.array_equal(target_xy,qt.coordinates['ipl_preterminal'])
            support_radius=float(support)
            coordinate=dict(pre='Cone opl_pedicle OPL/mm',post='BC OPL input proxy at original seed Cone pedicle; equal to stored IPL preterminal XY',post_is_measured_OPL_dendrite=False)
        else:
            source_xy=pt.coordinates['ipl_preterminal'];eligible=np.flatnonzero(pt.extras['eligible_drive'])
            target_xy=qt.coordinates['ipl_post_arbor'];support_radius=float(pt.record['source_bindings']['physical_support_radius_mm'])
            coordinate=dict(pre='BC ipl_preterminal IPL/mm',post='RGC ipl_post_arbor IPL/mm; frozen point proxy',post_is_measured_dendritic_arbor=False)
        assert array_identity(eligible)==recipe['resolved_numerics']['eligible_pre_rows']
        tree=cKDTree(source_xy[eligible]);windows={};anchors={}
        for location in ['near','far']:
            selection=selections[location];selected=selection['population__'+post];core=selection['core__'+post]
            fine=np.unique(np.concatenate([qg['member_global_rows'][qg['member_offsets'][g]:qg['member_offsets'][g+1]] for g in selected]))
            fine,old,_,statuses=route.selected_rows(fine);post_groups=qg['fine_to_group'][fine];is_core=np.isin(post_groups,core)
            effective=np.array([s.startswith('READY') for s in statuses]);core_rows=fine[is_core & effective]
            d8,valid=complete_d8(tree,target_xy[core_rows],support_radius)
            if not valid.any():raise RuntimeError('BLOCKED_GEOMETRY_SUPPORT: no reliable core d8 anchor '+name+' '+location)
            anchors[location]=dict(radius_mm=float(np.median(d8[valid])),patch_center_eccentricity_deg=locations[location]['patch_center_eccentricity_deg'],
                core_effective_rows=len(core_rows),complete_anchor_rows=int(valid.sum()),invalid_anchor_rows=int((~valid).sum()),
                d8_min_mm=float(d8[valid].min()),d8_max_mm=float(d8[valid].max()))
            windows[location]=(selected,fine,old,post_groups,is_core,effective)
        for location,(selected,fine,old,post_groups,is_core,effective) in windows.items():
            guard();selection=selections[location];ecc=np.linalg.norm(qt.coordinates['rf'][fine],axis=1)/.223
            radii=anchored_radius(ecc,anchors['near']['patch_center_eccentricity_deg'],anchors['far']['patch_center_eccentricity_deg'],anchors['near']['radius_mm'],anchors['far']['radius_mm'])
            data=[];indices=[];indptr=[0];statuses=[];changed=[];old_mass=np.asarray(old.sum(axis=1)).ravel()
            ids=qt.rows(fine).global_cell_ids
            for j,row in enumerate(fine):
                found,weights,status=select_row(tree,target_xy[row],radii[j],old_mass[j],support_radius)
                if not effective[j]:found=None;status='LEGACY_FALLBACK_UNSUPPORTED_INEFFECTIVE_TARGET'
                a,b=old.indptr[j:j+2];old_cols=old.indices[a:b];old_values=old.data[a:b]
                if found is None:cols=old_cols;values=old_values
                else:cols=eligible[found];order=np.argsort(cols);cols=cols[order];values=weights[order]
                if np.any(pg['fine_to_group'][cols]<0):
                    cols=old_cols;values=old_values;status='LEGACY_FALLBACK_UNSUPPORTED_PARENT_REPRESENTATION'
                is_changed=not(np.array_equal(cols,old_cols) and np.array_equal(values,old_values));changed.append(is_changed)
                indices.extend(cols);data.extend(values);indptr.append(len(data));statuses.append(status)
                row_records.append(dict(location=location,route=name,post_fine_row=int(row),cell_id=ids[j],post_parent_group=int(post_groups[j]),core=bool(is_core[j]),
                    target_eccentricity_deg=float(ecc[j]),radius_mm=float(radii[j]),nearest_anchor_extension=bool(ecc[j]<anchors['near']['patch_center_eccentricity_deg'] or ecc[j]>anchors['far']['patch_center_eccentricity_deg']),
                    old_inputs=len(old_cols),candidate_inputs=len(cols),old_Neff=float(old_values.sum()**2/(old_values@old_values)),candidate_Neff=float(values.sum()**2/(values@values)),
                    old_row_mass=float(old_values.sum()),candidate_row_mass=float(values.sum()),changed=is_changed,status=status))
            candidate=sparse.csr_matrix((data,indices,indptr),shape=old.shape);coo=candidate.tocoo()
            gm=sparse.csr_matrix((coo.data/qg['counts'][post_groups[coo.row]],(np.searchsorted(selected,post_groups[coo.row]),pg['fine_to_group'][coo.col])),shape=(len(selected),len(pg['counts'])))
            old_group=parent_matrix(record['directory']+'/mean_weights.npz')[selected]
            assert np.allclose(np.asarray(candidate.sum(1)).ravel(),old_mass,rtol=0,atol=1e-12)
            expected=np.bincount(np.searchsorted(selected,post_groups),weights=old_mass/qg['counts'][post_groups],minlength=len(selected))
            assert np.allclose(np.asarray(gm.sum(1)).ravel(),expected,rtol=0,atol=1e-12)
            pre_local=selection['population__'+pre];other=np.setdiff1d(np.arange(len(pg['counts'])),pre_local)
            exterior=gm[:,other];directory=OUT/'structure'/location/name;directory.mkdir(parents=True)
            for filename,matrix in [('fine_old.npz',old),('fine_candidate.npz',candidate),('group_old_full.npz',old_group),('group_candidate_full.npz',gm),('group_candidate_local.npz',gm[:,pre_local]),('group_candidate_exterior.npz',exterior)]:sparse.save_npz(directory/filename,matrix)
            np.savez_compressed(directory/'IDENTITIES.npz',fine_post_rows=fine,post_parent_groups=selected,pre_parent_groups=pre_local,exterior_parent_groups=other,
                source_eligible_rows=eligible,target_eccentricity_deg=ecc,radius_mm=radii,status=np.asarray(statuses),is_core=is_core,core_parent_groups=selection['core__'+post])
            difference=candidate-old;gdiff=gm-old_group
            summary=dict(location=location,route=name,pre=pre,post=post,source=record['source'],pre_table=pt.binding,post_table=qt.binding,
                coordinate_contract=coordinate,complete_source_support_radius_mm=support_radius,eligible_source_rows=len(eligible),anchors=anchors,
                selected_fine_rows=len(fine),core_fine_rows=int(is_core.sum()),fine_changed_rows=int(sum(changed)),
                candidate_statuses=dict(Counter(statuses)),core_statuses=dict(Counter(np.asarray(statuses)[is_core])),
                old_edges=old.nnz,candidate_edges=candidate.nnz,weight_difference_nnz=difference.nnz,max_weight_difference=float(np.max(np.abs(difference.data),initial=0)),
                group_changed_entries=gdiff.nnz,group_max_difference=float(np.max(np.abs(gdiff.data),initial=0)),
                external_group_entries=exterior.nnz,dynamic_local_closure=exterior.nnz==0,
                row_mass_max_abs_error=float(np.max(np.abs(np.asarray(candidate.sum(1)).ravel()-old_mass))),
                member_projection_max_abs_error=float(np.max(np.abs(np.asarray(gm.sum(1)).ravel()-expected))),
                outside_anchor_policy='Nearest anchor only within this explicitly saved original core/halo ROI; no whole0..6deg extrapolation',
                normalization='Complete fine parent source row once; arithmetic all-member group mean; no local renormalization',
                status='ENGINEERING_SPATIAL_POOLING_CANDIDATE',physiological_RF_calibration=False)
            write(directory/'SOURCE.json',summary);summaries.append(summary)
        print(json.dumps(dict(route=name,anchors=anchors),ensure_ascii=True),flush=True)
    with (OUT/'ROW_COMPARISON.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row_records[0]));writer.writeheader();writer.writerows(row_records)
    write(OUT/'STRUCTURE_RESULT.json',dict(candidate_id='RETINOTOPIC_GEOMETRY_ANCHORED_POOLING_V1',classification='ENGINEERING_SPATIAL_POOLING_CANDIDATE',
        routes=summaries,locations=locations,parent=bound(PARENT/'manifest.json'),old_two_path=bound(PREVIOUS/'RESULT.json'),
        formal_builder_adopted=False,FIT_READY=False,background_qualified=False,production_adopted=False,
        wall_seconds=time.perf_counter()-start,peak_rss_bytes=memory.peak_rss_bytes,minimum_available_bytes=memory.last['available_physical_bytes'],
        implementation=bound(Path(__file__)),kernel=bound(ROOT/'retina_sim/geometry_anchored_pooling.py')))


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False);write(OUT/'STARTED.json',dict(utc=datetime.now(timezone.utc).isoformat(),model_started=False))
    try:main()
    except Exception as e:
        write(OUT/'STRUCTURE_FAILURE.json',dict(error=str(e),traceback=traceback.format_exc(),model_started=False));raise
