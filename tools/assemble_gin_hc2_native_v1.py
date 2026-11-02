"""Complete fine HC disks and all-member projection on the new bounded master."""
from pathlib import Path
import json
import time
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from retina_sim.contracts import ContractError, Quantity
from retina_sim.hc_current import current_hc
from retina_sim.hc_mosaic import HCDensityProfile
from retina_sim.disk_tables import FrozenDiskTable, write_table
from retina_sim.spatial_group_model import GroupModelAssembly
from retina_sim.spatial_group_graph import group_members, digest
from retina_sim.spatial_multiresolution import build_quadtree, group_geometry
from retina_sim.hc_group_projection import project_hc_contacts
from tools.durable_research_runtime_v1 import atomic_json

ROOT=Path(r'D:\macaque retinal simulator')
OUT=ROOT/'reports/GIN_69_6_HC2_BC1_FIT_V1'

def build(output,guard):
    start=time.perf_counter();selected=current_hc(ROOT)
    cfg=json.loads((ROOT/selected['master_configuration']).read_bytes())
    path=ROOT/cfg['asset_path']
    if digest(path)!=cfg['expected_asset_sha256']:raise ContractError('New HC master source changed')
    master=json.loads(path.read_bytes());record=next(t for t in master['tables'] if t['population']=='HC')
    xy=np.asarray(record['coordinates']['soma']['values'],dtype=np.float64)
    types=np.asarray(master['source']['types'],dtype='U2')
    if master['source']['hc_density_geometry_identity']!=selected['identity']:
        raise ContractError('New HC master differs from current geometry code identity')
    hc=write_table(ROOT,'data/atlas/GIN_69_6_HC2_BC1_V1/HC_complete_BC_dependency_table',
        {k:record[k] for k in ('atlas_id','atlas_version','source_version','population','reference_id')},
        len(xy),lambda a,b:tuple(record['source_keys'][a:b]),
        {'soma':Quantity(xy,'mm','anatomical_position','joint_HC_v2_engineering_density','anatomical_retina')},
        extras={'subtype':types.astype('S2')},source_bindings=dict(master_path=cfg['asset_path'],
            master_sha256=cfg['expected_asset_sha256'],hc_density_geometry_identity=selected['identity'],
            OPL_anchor_transfer='HC_soma_to_OPL_colocation_project_v1',
            fixed_cohort_OPL_radius_mm=1.338,pair_context_in_master=True,
            construction='complete explicitly bounded joint master, new namespace; not a whole retina population'))
    base=GroupModelAssembly(ROOT/'reports/spatial_group_graph_v02_20261025_v1/artifact_r2')
    cone=FrozenDiskTable(ROOT,base.manifest['populations']['cones']['source_table'])
    cxy=np.asarray(cone.coordinates['opl_pedicle']);ct=cone.extras['subtype'].astype('U1')
    profile=HCDensityProfile(ROOT);radii=np.zeros(len(xy))
    for kind in ('H1','H2'):
        rows=np.flatnonzero(types==kind)
        radii[rows]=profile.contact_radius_mm(np.linalg.norm(xy[rows],axis=1),kind)
    tree=cKDTree(cxy);indptr=[0];indices=[]
    for first in range(0,len(xy),512):
        guard()
        neighbors=tree.query_ball_point(xy[first:first+512],radii[first:first+512]+1e-12,workers=1,return_sorted=True)
        for j,found in enumerate(neighbors,first):
            rows=np.asarray(found,np.int64)
            if types[j]=='H1':rows=rows[ct[rows]!='S']
            indices.extend(rows);indptr.append(len(indices))
    raw=sparse.csr_matrix((np.ones(len(indices)),indices,indptr),shape=(len(xy),cone.count))
    mesh=build_quadtree(xy/.223,np.full(len(xy),2,np.int64),base_deg=.4)
    labels,inverse=np.unique(types,return_inverse=True)
    hg=group_members(np.arange(len(xy)),mesh['cells'],subtype=inverse)
    geometry=group_geometry(xy/.223,hg['fine_to_group'],hg['counts'],mesh['leaf_keys'],hg['keys'][:,0],base_deg=.4)
    hg.update(centroid_deg=geometry['centroid_deg'],leaf_keys=mesh['leaf_keys'])
    with np.load(OUT/'executions/assemble_fine_01/SELECTION.npz',allow_pickle=False) as z:
        local_c=z['groups__cones'].copy()
    original=base.populations['cones']
    connected=original.fine_to_group[np.unique(raw.indices)]
    if np.any(connected<0):raise ContractError('HC input Cone outside original full Cone representation')
    full_c=np.union1d(local_c,connected)
    old_to_full=np.full(original.group_count,-1,np.int64);old_to_full[full_c]=np.arange(len(full_c))
    cm=np.full(cone.count,-1,np.int64)
    represented=original.fine_to_group>=0
    cm[represented]=old_to_full[original.fine_to_group[represented]]
    cc=original.counts[full_c]
    projected=project_hc_contacts(raw,ct,types,cm,hg['fine_to_group'],cc,hg['counts'])
    local_native_rows=old_to_full[local_c]
    local_h=np.unique(projected['typed_B'][local_native_rows].indices)
    if not len(local_h):raise ContractError('No typed HC feedback for selected Cone population')
    # The master was extended from all fine members, not group centroids.
    fine=np.concatenate([original.member_global_rows[original.member_offsets[g]:original.member_offsets[g+1]] for g in local_c])
    b=cfg['canonical_bounds_mm'];p=cxy[fine];margin=float(radii.max())
    clearance=np.minimum.reduce((p[:,0]-b[0],b[1]-p[:,0],p[:,1]-b[2],b[3]-p[:,1]))
    if np.any(clearance<margin-1e-12) or np.any(np.linalg.norm(p,axis=1)+margin>1.338+1e-12):
        raise ContractError('Actual owner Cone disk lacks complete HC source coverage')
    sparse.save_npz(output/'RAW_HC_BY_CONE.npz',raw)
    for name in ('A','typed_B'):sparse.save_npz(output/(name+'.npz'),projected[name])
    np.savez_compressed(output/'MEMBERSHIP.npz',**hg,**{'geometry_'+k:v for k,v in geometry.items()},
        global_cone_parent_group_rows=full_c,local_cone_rows=local_native_rows,local_HC_rows=local_h,
        global_cone_fine_to_group=cm,cone_counts=cc,HC_types=labels[hg['keys'][:,1]],
        cone_types=base.array('populations/cones/keys.npy')[full_c,1],
        cone_supported_member_fraction=projected['cone_supported_member_fraction'],
        HC_supported_member_fraction=projected['HC_supported_member_fraction'])
    result=dict(schema='GIN69-new-HC-complete-fine-projection-1',status='STATIC_NATIVE_OPERATORS_READY',
        hc_table=hc.binding,cone_table=cone.binding,current_HC_identity=selected['identity'],
        global_HC_members=len(xy),global_HC_groups=len(hg['counts']),global_Cone_groups=len(full_c),
        global_Cone_members=int(cc.sum()),local_HC_groups=len(local_h),local_HC_members=int(hg['counts'][local_h].sum()),
        local_Cone_groups=len(local_c),local_Cone_members=len(fine),raw_direct_contacts=raw.nnz,
        complete_owner_contact_envelope=True,minimum_HC_master_margin_mm=float(clearance.min()),max_contact_radius_mm=margin,
        HC_full_zero_input_members=int(np.sum(np.diff(raw.indptr)==0)),
        local_Cone_unsupported_member_mass=float(np.sum(cc[local_native_rows]*(1-projected['cone_supported_member_fraction'][local_native_rows]))),
        normalization=projected['normalization'],H1_excludes_S=True,H2_input='LMS',H2_feedback='S_only',
        boundary='finite master complete for retained Cone feedback; exterior Cone input remains explicit held background',
        wall_seconds=time.perf_counter()-start,neural_forward_started=False,FIT_READY=False,background_qualified=False,production_adopted=False)
    atomic_json(output/'NATIVE_ASSEMBLY_RESULT.json',result)
    print(json.dumps(result),flush=True)
