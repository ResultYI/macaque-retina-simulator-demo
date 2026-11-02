"""Rebind changed BC routes to original downstream cells and frozen group policy.

This stage freezes structural operators and membership only. It never reads
recorded spikes, initializes a dynamical model, or advances a neural state.
"""
from pathlib import Path
import json
import time
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.disk_population import FixedDiskPopulation
from retina_sim.disk_graph_storage import FrozenSparseRoute
from retina_sim.disk_pooling import bounded_matrix
from retina_sim.geometry_anchored_pooling import anchored_radius, select_row
from retina_sim.spatial_group_graph import group_members, digest
from retina_sim.spatial_multiresolution import build_quadtree, group_geometry
from retina_sim.spatial_group_model import GroupModelAssembly, BC, AC, RGC
from tools.durable_research_runtime_v1 import atomic_json

ROOT = Path(r'D:\macaque retinal simulator')
BASE = ROOT/'reports/spatial_group_graph_v02_20261025_v1/artifact_r2'
OLD = ROOT/'reports/GIN_69_6_RADIAL_CONDITIONAL_V1/candidate_v2'
OUT = ROOT/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2'

def read(path):
    return json.loads(Path(path).read_bytes())

def fine_rows(pop, selected):
    return np.unique(np.concatenate([pop.member_global_rows[pop.member_offsets[r]:pop.member_offsets[r+1]]
        for r in selected])).astype(np.int64) if len(selected) else np.empty(0,np.int64)

def build(output, guard):
    started = time.perf_counter()
    base = GroupModelAssembly(BASE)
    catalog = read(OUT/'BC_catalog.json')
    old_routes = read(BASE/'group_routes.json')
    pools = read(ROOT/'reports/retinotopic_rf_revision/SPATIAL_POOLING_CANDIDATE_V1/STRUCTURE_RESULT.json')
    with np.load(OLD/'prepared/local_static/selection.npz',allow_pickle=False) as z:
        chosen = {n:z['population__'+n].copy() for n in base.populations}
        original_core = {n:z['core__'+n].copy() for n in base.populations}
    old_tables = {}
    def old_table(name):
        if name not in old_tables:
            old_tables[name] = FrozenDiskTable(ROOT,base.manifest['populations'][name]['source_table'])
        return old_tables[name]
    tables = {n:FrozenDiskTable(ROOT,catalog['tables'][n]) for n in BC}
    cone = old_table('cones')
    center = np.array([3.88,0.])
    groups = {}
    for name,t in tables.items():
        guard()
        xy = np.asarray(t.coordinates['rf'])/.223
        mesh = build_quadtree(xy,np.full(t.count,2,np.int64),base_deg=.4)
        parent = None
        if name.startswith('midget') or name=='S_ON_BC':
            parent = base.populations['cones'].fine_to_group[t.extras['cone_row']]
            if np.any(parent<0):
                raise ContractError('New BC categorical parent absent from original Cone representation')
        g = group_members(np.arange(t.count),mesh['cells'],parent=parent,
            eligibility=t.extras['eligible_drive'].astype(np.int64))
        geometry = group_geometry(xy,g['fine_to_group'],g['counts'],mesh['leaf_keys'],g['keys'][:,0],base_deg=.4)
        g.update(centroid_deg=geometry['centroid_deg'],leaf_keys=mesh['leaf_keys'])
        groups[name]=g
        chosen[name]=np.flatnonzero(np.linalg.norm(g['centroid_deg']-center,axis=1)<=.3)
        d=output/'populations'/name;d.mkdir(parents=True)
        np.savez_compressed(d/'MEMBERSHIP.npz',**g,**{'geometry_'+k:v for k,v in geometry.items()})
    aii = FixedDiskPopulation(ROOT,base.manifest['populations']['AII_AC']['source_table'])
    aii_xy=np.asarray(aii.positions)
    aii_tree=cKDTree(aii_xy)
    changed={};facts={}
    # Matrices have full new-BC row spaces and original full unchanged row spaces.
    # Unchanged downstream chemical targets are the already selected real cells.
    for name,r in base.routes.items():
        if r.pre not in BC or r.post in BC or r.kind=='gap':
            continue
        guard();post=fine_rows(base.populations[r.post],chosen[r.post]);pt=old_table(r.post)
        target=pt.coordinates['ipl_post_arbor'][post]
        source=FrozenSparseRoute(ROOT,old_routes[name]['source'])
        _,previous,_,statuses=source.selected_rows(post)
        mass=np.asarray(previous.sum(1)).ravel()
        pre=tables[r.pre]
        if r.post.startswith('midget_RGC'):
            # Preserve each original convergence row's Cone ancestry, not old BC row numbers.
            prior_bc=old_table(r.pre)
            lookup={int(c):i for i,c in enumerate(pre.extras['cone_row'])}
            coo=previous.tocoo()
            parents=prior_bc.extras['cone_row'][coo.col]
            if any(int(c) not in lookup for c in parents):
                raise ContractError('Original midget private/convergence BC parent outside new source view')
            m=sparse.csr_matrix((coo.data,(coo.row,[lookup[int(c)] for c in parents])),shape=(len(post),pre.count))
            rule='Original midget input Cone IDs and row weights preserved; BC rows rebound by ancestry'
        elif r.post.startswith('parasol_RGC'):
            frozen=next(v for v in pools['routes'] if v['route']==name)
            anchors=frozen['anchors'];near=anchors['near'];far=anchors['far']
            radius=anchored_radius(np.linalg.norm(pt.coordinates['rf'][post],axis=1)/.223,
                near['patch_center_eccentricity_deg'],far['patch_center_eccentricity_deg'],near['radius_mm'],far['radius_mm'])
            tree=cKDTree(pre.coordinates['ipl_preterminal']);rr=[];cc=[];vv=[]
            bounds=catalog['anchor_query']['bounds_mm']
            for j in range(len(post)):
                point=target[j]
                clearance=min(point[0]-bounds[0],bounds[1]-point[0],point[1]-bounds[2],bounds[3]-point[1])
                if clearance<radius[j]:
                    raise ContractError('Parasol finite query lacks full candidate radius')
                idx,w,status=select_row(tree,point,radius[j],mass[j],read(ROOT/'configurations/research/GIN69_OWNER_SUPPORT_V2.json')['BC_anchor_radius_mm'])
                if idx is None:
                    raise ContractError('Frozen parasol candidate has unsupported new source row: '+status)
                rr.extend([j]*len(idx));cc.extend(idx);vv.extend(w)
            m=sparse.csr_matrix((vv,(rr,cc)),shape=(len(post),pre.count))
            rule='Frozen geometry-anchored radius and exponent2, old per-target total gain, all new eligible BC within radius'
        else:
            k=4 if r.post in ('local_feedforward_AC','local_crossover_AC') else 16 if r.post=='widefield_AC' else 8
            sigma=.02+.06*np.linalg.norm(pt.coordinates['rf'][post],axis=1) if r.post=='small_bistratified_S_ON' and r.pre=='diffuse_BC_OFF' else None
            m,audit=bounded_matrix(pre.coordinates['ipl_preterminal'],target,k,support_mm=.04,gaussian_scale=sigma)
            if r.post=='small_bistratified_S_ON' and r.pre=='diffuse_BC_OFF':
                if any(a['missing_degree'] for a in audit):
                    raise ContractError('Existing exact Sbi diffuse input degree unavailable in new BC source')
            # Old unresolved target rows remain unresolved, never promoted by density replacement.
            m=(sparse.diags(mass)@m).tocsr()
            rule='Original finite40um/k rule and eligibility, new BC population; old row total retained'
        err=float(np.max(np.abs(np.asarray(m.sum(1)).ravel()-mass),initial=0))
        if err>1e-12:raise ContractError('Changed BC downstream row gain not preserved')
        changed[name]=(r.pre,r.post,post,m)
        facts[name]=dict(post_rows=len(post),edges=m.nnz,row_gain_max_abs_error=err,rule=rule,source_route=old_routes[name]['source'])
        chosen[r.pre]=np.union1d(chosen[r.pre],groups[r.pre]['fine_to_group'][np.unique(m.indices)])
    # Reciprocal gap law is unchanged; frozen original AII master is the full source.
    for name in ('midget_BC_ON','diffuse_BC_ON','S_ON_BC'):
        guard();xy=tables[name].coordinates['ipl_preterminal'];_,nearest=aii_tree.query(xy,k=1)
        distance=np.linalg.norm(aii_xy[nearest]-xy,axis=1);valid=distance<=.04
        rows=np.flatnonzero(valid);g=.15e-9*np.exp(-(distance[valid]/.03)**2)
        forward=sparse.csr_matrix((g,(rows,nearest[valid])),shape=(len(xy),len(aii_xy)))
        reverse=forward.T.tocsr()
        core_aii=fine_rows(base.populations['AII_AC'],original_core['AII_AC'])
        adjacent=np.unique(reverse[core_aii].indices)
        chosen[name]=np.union1d(chosen[name],groups[name]['fine_to_group'][adjacent])
        changed['AII_AC_to_'+name]=('AII_AC',name,np.arange(len(xy)),forward)
        changed[name+'_to_AII_AC']=(name,'AII_AC',np.arange(len(aii_xy)),reverse)
    # Rebuild chemical AII->BC rows; the parent AII/RBC/rod topology is unchanged.
    for name in ('midget_BC_OFF','diffuse_BC_OFF'):
        guard();xy=tables[name].coordinates['ipl_preterminal']
        m,audit=bounded_matrix(aii_xy,xy,3,support_mm=.04)
        changed['AII_AC_to_'+name]=('AII_AC',name,np.arange(len(xy)),m)
    selected_bc={n:fine_rows(type('P',(),groups[n])(),chosen[n]) for n in BC}
    for name in ('midget_BC_OFF','diffuse_BC_OFF'):
        m=changed['AII_AC_to_'+name][3][selected_bc[name]]
        source=np.unique(m.indices)
        mapped=base.populations['AII_AC'].fine_to_group[source]
        if np.any(mapped<0):raise ContractError('New chemical AII dependency outside existing full parent representation')
        chosen['AII_AC']=np.union1d(chosen['AII_AC'],mapped)
    for post,pre in (('AII_AC','rod_BC'),('rod_BC','rods')):
        name=pre+'_to_'+post
        source=FrozenSparseRoute(ROOT,old_routes[name]['source'])
        dependency=source.dependency_rows(fine_rows(base.populations[post],chosen[post]))
        mapped=base.populations[pre].fine_to_group[dependency]
        if np.any(mapped<0):raise ContractError('Original rod branch dependency outside original representative parent')
        chosen[pre]=np.union1d(chosen[pre],mapped)
    cone_dependencies=[]
    for name in BC:
        route=FrozenSparseRoute(ROOT,catalog['routes']['cones->'+name])
        post,m,_,status=route.selected_rows(selected_bc[name])
        cone_dependencies.extend(m.indices)
        # Categorical ancestral Cone identity is also retained for S-ON groups.
        if name.startswith('midget') or name=='S_ON_BC':cone_dependencies.extend(tables[name].extras['cone_row'][post])
        changed['cones_to_'+name]=('cones',name,post,m)
    cone_dependencies=np.unique(cone_dependencies)
    mapped=base.populations['cones'].fine_to_group[cone_dependencies]
    if np.any(mapped<0):raise ContractError('New BC required Cone outside old unchanged full Cone grouping')
    chosen['cones']=np.union1d(chosen['cones'],mapped)
    cones=fine_rows(base.populations['cones'],chosen['cones'])
    if np.any(np.linalg.norm(cone.coordinates['rf'][cones],axis=1)>1.338):
        raise ContractError('Dynamic Cone owner expansion is not authorized; preserve outside owners as held dependencies')
    pedicle=cone.coordinates['opl_pedicle'][cones]
    master_cfg=read(ROOT/'configurations/research/HC69_owner_support_v2_master.json')
    b=master_cfg['canonical_bounds_mm']
    clearance=np.minimum.reduce((pedicle[:,0]-b[0],b[1]-pedicle[:,0],pedicle[:,1]-b[2],b[3]-pedicle[:,1]))
    radius_guard=master_cfg['owner_support']['HC_outer_contact_margin_mm']
    required=np.maximum(0.,radius_guard-clearance)
    folder=output/'fine_routes';folder.mkdir()
    for name,(pre,post,rows,m) in changed.items():
        guard();d=folder/name;d.mkdir()
        sparse.save_npz(d/'weights.npz',m);np.save(d/'post_source_rows.npy',rows,allow_pickle=False)
        facts.setdefault(name,{}).update(pre=pre,post=post,shape=list(m.shape),edges=m.nnz,
            weights_sha256=digest(d/'weights.npz'),post_rows_sha256=digest(d/'post_source_rows.npy'),
            row_semantics='full source rows normalized before any runtime restriction; gaps are physical S, unnormalized')
    np.savez_compressed(output/'SELECTION.npz',**{'groups__'+n:v for n,v in chosen.items()},
        **{'fine__'+n:v for n,v in selected_bc.items()},fine__cones=cones)
    result=dict(schema='GIN69-HC2-BC1-structural-rebinding-1',status='FINE_BC_ROUTES_AND_GROUP_SELECTION_SAVED',
        wall_seconds=time.perf_counter()-started,BC_identity=catalog['bc_density_geometry_identity'],
        unchanged_parent_artifact=base.artifact_sha256,old_geometry_selection_sha256=digest(OLD/'prepared/local_static/selection.npz'),
        counts={n:len(v) for n,v in chosen.items()},BC_fine_counts={n:len(v) for n,v in selected_bc.items()},
        native_cone_fine_count=len(cones),new_native_cone_group_count=len(chosen['cones']),
        native_cone_pedicle_bounds_mm=[float(pedicle[:,0].min()),float(pedicle[:,0].max()),float(pedicle[:,1].min()),float(pedicle[:,1].max())],
        native_cone_radial_max_mm=float(np.linalg.norm(pedicle,axis=1).max()),
        HC_master_coverage=dict(cones_outside_rectangle=int(np.sum(clearance<0)),
            cones_lacking_conservative_contact_margin=int(np.sum(required>1e-12)),
            minimum_clearance_mm=float(clearance.min()),max_contact_radius_mm=radius_guard,
            complete=False,status='PLANNED_BOUNDS_ONLY_MASTER_BUILT_NEXT',planned_margin_sufficient=bool(np.all(required<=1e-12))),
        routes=facts,neural_forward_started=False,background_constructed=False,
        source_scale_unchanged=True,no_recorded_spikes_read=True,FIT_READY=False,background_qualified=False,production_adopted=False)
    atomic_json(output/'FINE_ASSEMBLY_RESULT.json',result)
    atomic_json(OUT/'CURRENT_ASSEMBLY_STAGE.json',dict(output=output.relative_to(ROOT).as_posix(),**{k:v for k,v in result.items() if k!='routes'}))
    print(json.dumps({k:v for k,v in result.items() if k!='routes'}),flush=True)
