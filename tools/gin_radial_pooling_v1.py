"""Project the frozen two-anchor candidate onto one new radial selection."""
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from tools.build_spatial_pooling_candidate_v1 import ROOT,PARENT,NAMES,read,parent_matrix,sha
from tools.cross_position_context_v1 import write
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.disk_graph_storage import FrozenSparseRoute,array_identity
from retina_sim.geometry_anchored_pooling import anchored_radius,select_row

def build(out,selection,guard):
    frozen=ROOT/'reports/retinotopic_rf_revision/SPATIAL_POOLING_CANDIDATE_V1/STRUCTURE_RESULT.json'
    prior=read(frozen);manifest=read(PARENT/'manifest.json');routes=read(PARENT/'group_routes.json');summary=[]
    tables={};groups={}
    def pop(n):
        if n not in tables:
            tables[n]=FrozenDiskTable(ROOT,manifest['populations'][n]['source_table'])
            groups[n]={k:np.load(PARENT/f'populations/{n}/{k}.npy',mmap_mode='r',allow_pickle=False) for k in ('counts','fine_to_group','member_global_rows','member_offsets')}
        return tables[n],groups[n]
    for name in NAMES:
        guard();r=routes[name];pt,pg=pop(r['pre']);qt,qg=pop(r['post']);saved=next(x for x in prior['routes'] if x['route']==name)
        route=FrozenSparseRoute(ROOT,r['source']);anchors=saved['anchors'];support=saved['complete_source_support_radius_mm']
        if r['pre']=='cones':
            xy=pt.coordinates['opl_pedicle'];eligible=np.flatnonzero(pt.extras['subtype'].astype('U1')!='S');target=xy[qt.extras['cone_row']]
            if not np.array_equal(target,qt.coordinates['ipl_preterminal']):raise ValueError('Existing OPL proxy contract differs')
        else:xy=pt.coordinates['ipl_preterminal'];eligible=np.flatnonzero(pt.extras['eligible_drive']);target=qt.coordinates['ipl_post_arbor']
        if array_identity(eligible)!=route.record['recipe']['resolved_numerics']['eligible_pre_rows']:raise ValueError('Eligible source identity differs')
        selected=selection['population__'+r['post']];localpre=selection['population__'+r['pre']]
        fine=np.unique(np.concatenate([qg['member_global_rows'][qg['member_offsets'][g]:qg['member_offsets'][g+1]] for g in selected]))
        fine,old,_,status=route.selected_rows(fine);postgroups=qg['fine_to_group'][fine]
        ecc=np.linalg.norm(qt.coordinates['rf'][fine],axis=1)/.223
        radii=anchored_radius(ecc,anchors['near']['patch_center_eccentricity_deg'],anchors['far']['patch_center_eccentricity_deg'],anchors['near']['radius_mm'],anchors['far']['radius_mm'])
        tree=cKDTree(xy[eligible]);data=[];indices=[];indptr=[0];statuses=[];changed=0;mass=np.asarray(old.sum(1)).ravel()
        for j,row in enumerate(fine):
            found,weights,s=select_row(tree,target[row],radii[j],mass[j],support)
            a,b=old.indptr[j:j+2];oc=old.indices[a:b];ov=old.data[a:b]
            if not status[j].startswith('READY'):found=None;s='LEGACY_FALLBACK_UNSUPPORTED_INEFFECTIVE_TARGET'
            if found is None:cols=oc;values=ov
            else:cols=eligible[found];order=np.argsort(cols);cols=cols[order];values=weights[order]
            if np.any(pg['fine_to_group'][cols]<0):cols=oc;values=ov;s='LEGACY_FALLBACK_UNSUPPORTED_PARENT_REPRESENTATION'
            changed+=int(not(np.array_equal(cols,oc) and np.array_equal(values,ov)));indices.extend(cols);data.extend(values);indptr.append(len(data));statuses.append(s)
        candidate=sparse.csr_matrix((data,indices,indptr),shape=old.shape);coo=candidate.tocoo()
        gm=sparse.csr_matrix((coo.data/qg['counts'][postgroups[coo.row]],(np.searchsorted(selected,postgroups[coo.row]),pg['fine_to_group'][coo.col])),shape=(len(selected),len(pg['counts'])))
        other=np.setdiff1d(np.arange(len(pg['counts'])),localpre);exterior=gm[:,other]
        err=float(abs(np.asarray(candidate.sum(1)).ravel()-mass).max(initial=0))
        expected=np.bincount(np.searchsorted(selected,postgroups),weights=mass/qg['counts'][postgroups],minlength=len(selected))
        projection_error=float(abs(np.asarray(gm.sum(1)).ravel()-expected).max(initial=0))
        if max(err,projection_error)>1e-12:raise ValueError('Original row/member budget exceeded')
        dest=out/'pooling'/name;dest.mkdir(parents=True)
        for filename,matrix in [('fine_candidate.npz',candidate),('group_candidate_full.npz',gm),('group_candidate_local.npz',gm[:,localpre]),('group_candidate_exterior.npz',exterior)]:sparse.save_npz(dest/filename,matrix)
        np.savez_compressed(dest/'IDENTITIES.npz',fine_post_rows=fine,post_parent_groups=selected,pre_parent_groups=localpre,radii_mm=radii,status=np.asarray(statuses))
        record=dict(route=name,pre=r['pre'],post=r['post'],anchors=anchors,frozen_source_sha256=sha(frozen),coordinate_contract=saved['coordinate_contract'],
            selected_fine_rows=len(fine),changed_fine_rows=changed,status_counts=dict(Counter(statuses)),edges=candidate.nnz,
            radius_min_mm=float(radii.min()),radius_max_mm=float(radii.max()),row_mass_max_abs_error=err,member_projection_max_abs_error=projection_error,
            exterior_entries=exterior.nnz,dynamic_local_closure=exterior.nnz==0,local_matrix_sha256=sha(dest/'group_candidate_local.npz'))
        write(dest/'SOURCE.json',record);summary.append(record)
    write(out/'POOLING_RESULT.json',dict(routes=summary,frozen_anchor_identity=sha(frozen),new_anchor_fit=False,local_renormalization=False,all_dynamic_closed=all(r['dynamic_local_closure'] for r in summary)))
    if not all(r['dynamic_local_closure'] for r in summary):raise ValueError('New geometry pooling dynamic support not closed; no silent crop/renormalization')
