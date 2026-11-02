"""Project complete fine rows into the selected actual member groups.

No dynamical state, spike observation, or fitted parameter is read here.
"""
from pathlib import Path
from copy import deepcopy
import json
import time
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.spatial_group_graph import digest, save_csr
from retina_sim.spatial_group_model import GroupModelAssembly, BC, POPULATIONS
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.disk_population import FixedDiskPopulation
from retina_sim.disk_graph_storage import FrozenSparseRoute
from retina_sim.ac_vs_model import full_input_kernel, load_config
from tools.durable_research_runtime_v1 import atomic_json

ROOT=Path(r'D:\macaque retinal simulator')
OUT=ROOT/'reports/GIN_69_6_HC2_BC1_FIT_V1'
FINE=OUT/'executions/assemble_fine_01'
NATIVE=OUT/'executions/assemble_native_01'
OLD=ROOT/'reports/GIN_69_6_RADIAL_CONDITIONAL_V1/candidate_v2'

def read(path):
    return json.loads(Path(path).read_bytes())

def bundle(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def select_members(g,keep):
    counts=g['counts'][keep]
    members=np.concatenate([g['member_global_rows'][g['member_offsets'][i]:g['member_offsets'][i+1]] for i in keep])
    mapping=np.full(len(g['fine_to_group']),-1,np.int64)
    mapping[members]=np.repeat(np.arange(len(keep)),counts)
    return counts,members,np.r_[0,np.cumsum(counts)],mapping

def build(output,guard):
    start=time.perf_counter();base=GroupModelAssembly(ROOT/'reports/spatial_group_graph_v02_20261025_v1/artifact_r2')
    selected=bundle(FINE/'SELECTION.npz');native=bundle(NATIVE/'MEMBERSHIP.npz')
    nr=read(NATIVE/'NATIVE_ASSEMBLY_RESULT.json');catalog=read(OUT/'BC_catalog.json')
    fine_receipt=read(FINE/'FINE_ASSEMBLY_RESULT.json')
    old_routes=read(base.root/'group_routes.json')
    groups={};local={};meta=deepcopy(base.manifest);parent_geometry={};parent_rows={};allkeys={}
    for name in POPULATIONS:
        guard();p=base.populations[name]
        if name in BC or name=='HC':
            g=native if name=='HC' else bundle(FINE/'populations'/name/'MEMBERSHIP.npz')
            table=FrozenDiskTable(ROOT,nr['hc_table'] if name=='HC' else catalog['tables'][name])
            keep=native['local_HC_rows'] if name=='HC' else selected['groups__'+name]
            geom={k.removeprefix('geometry_'):v for k,v in g.items() if k.startswith('geometry_')}
            centroid=g['centroid_deg']*.223
            ids=np.array([name+':new:'+canonical_hash(dict(table=table.binding,keys=row))[:24] for row in g['keys']])
            popmeta=deepcopy(meta['populations'][name])
            popmeta.update(source_table=table.binding,source_keys=table.record['source_keys'],
                full_asset_count=table.count,coordinate_role='soma' if name=='HC' else 'rf')
            if name=='HC':popmeta.update(subtype_labels=['H1','H2'],key_columns=['leaf','subtype'])
            else:
                popmeta['key_columns']=['leaf']+(['parent_group'] if name.startswith('midget') or name=='S_ON_BC' else [])+['eligibility']
                popmeta['parent_population']='cones' if 'parent_group' in popmeta['key_columns'] else None
        else:
            keep=(native['global_cone_parent_group_rows'][native['local_cone_rows']] if name=='cones' else selected['groups__'+name])
            g={k:getattr(p,k) for k in ('counts','member_offsets','member_global_rows','fine_to_group')}
            g['keys']=base.array('populations/'+name+'/keys.npy')
            geom=p.mesh_geometry;centroid=p.centroid_mm;ids=p.group_ids
            popmeta=deepcopy(meta['populations'][name])
        parent_rows[name]=np.asarray(keep,np.int64)
        # Native parent Cone numbering is its own explicitly retained source view.
        if name=='cones':parent_rows[name]=native['local_cone_rows']
        groups[name]=g;allkeys[name]=g['keys']
        parent_geometry[name]=SimpleNamespace(group_count=len(centroid),counts=g['counts'],centroid_mm=centroid)
        counts,members,offsets,mapping=select_members(g,keep)
        aux=np.sort(members[offsets[:-1]])[:900]
        if name not in BC and name!='HC':
            previous=p.recording_auxiliary_fine_rows
            present=previous[mapping[previous]>=0]
            if len(present):aux=present
        arrays=dict(group_ids=ids[keep],counts=counts,member_offsets=offsets,member_global_rows=members,
            fine_to_group=mapping,centroid_mm=centroid[keep],keys=g['keys'][keep],
            recording_auxiliary_fine_rows=aux,recording_auxiliary_group_rows=mapping[aux],
            **{'geometry_'+k:(v if k=='leaf_keys' else v[keep]) for k,v in geom.items()})
        folder=output/'populations'/name;folder.mkdir(parents=True)
        np.savez_compressed(folder/'MEMBERSHIP.npz',**arrays)
        local[name]=arrays
        popmeta.update(group_count=len(keep),fine_neurons=len(members),singleton_groups=int(np.sum(counts==1)))
        meta['populations'][name]=popmeta
    # The four already-approved functional AC pools must see their full kernels.
    ac=load_config(ROOT/'configurations/ac_vs_functional_v1.json');ac_closure=[]
    for channel in ac['channels']:
        pre,post=channel['pre'],channel['target']
        matrix,empty=full_input_kernel(parent_geometry[pre],parent_geometry[post],ac['geometry'])
        rows=parent_rows[post];cols=parent_rows[pre]
        missing=matrix[rows].nnz-matrix[rows][:,cols].nnz
        if missing or len(np.intersect1d(rows,empty)):
            raise ContractError(f'Existing functional AC ancestor closure incomplete: {channel["pool"]}, {missing}')
        ac_closure.append(dict(pool=channel['pool'],missing_edges=0,full_kernel=canonical_hash(dict(data=matrix.data,indices=matrix.indices,indptr=matrix.indptr,shape=matrix.shape))))
    matrices={};route_meta=[];gap_records=[];contact_counts={};tables={}
    def table(name):
        if name not in tables:
            binding=meta['populations'][name]['source_table']
            if name=='AII_AC':
                fixed=FixedDiskPopulation(ROOT,binding)
                # Existing AII fixed-chart position is also its declared IPL proxy.
                tables[name]=SimpleNamespace(coordinates={'ipl_preterminal':fixed.positions})
            else:tables[name]=FrozenDiskTable(ROOT,binding)
        return tables[name]
    aii_xy=table('AII_AC').coordinates['ipl_preterminal'][local['AII_AC']['member_global_rows']]
    bounds=catalog['anchor_query']['bounds_mm']
    clearance=np.minimum.reduce((aii_xy[:,0]-bounds[0],bounds[1]-aii_xy[:,0],aii_xy[:,1]-bounds[2],bounds[3]-aii_xy[:,1]))
    if np.any(clearance<.04-1e-12):
        atomic_json(output/'BC_EXTERIOR_SCOPE.json',dict(minimum_margin_mm=float(clearance.min()),required_mm=.04,
            missing_members=int(np.sum(clearance<.04-1e-12)),current_query_bounds_mm=bounds,
            needed_bounds_mm=[float(aii_xy[:,0].min()-.04),float(aii_xy[:,0].max()+.04),float(aii_xy[:,1].min()-.04),float(aii_xy[:,1].max()+.04)]))
        raise ContractError('Full AII incident gap degree needs a larger exterior-only BC query; no cropped degree accepted')
    def save_matrix(relative,m):
        path=output/relative;path.parent.mkdir(parents=True,exist_ok=True);save_csr(path,m);matrices[relative]=m
    native_A=sparse.load_npz(NATIVE/'A.npz');native_B=sparse.load_npz(NATIVE/'typed_B.npz')
    for name,r in base.routes.items():
        guard();a,b=local[r.pre],local[r.post];folder=output/'routes'/name;folder.mkdir(parents=True)
        shape=(len(b['counts']),len(a['counts']));record=deepcopy(next(v for v in base.manifest['routes'] if v['name']==name))
        changed=FINE/'fine_routes'/name
        if name in ('cones_to_HC','HC_to_cones'):
            complete=native_A if name=='cones_to_HC' else native_B
            mean=complete[parent_rows[r.post]][:,parent_rows[r.pre]].tocsr()
            total=(sparse.diags(b['counts'])@mean).tocsr()
            external=complete[parent_rows[r.post]].copy().tolil();external[:,parent_rows[r.pre]]=0
            external=external.tocsr();external.eliminate_zeros()
            save_matrix('boundary/'+name+'.npz',external)
            if name=='HC_to_cones' and external.nnz:raise ContractError('Retained Cone lacks actual HC feedback ancestors')
            selected_edges=None;post=None;W=None
        else:
            post=np.sort(b['member_global_rows'])
            if changed.exists():
                full=sparse.load_npz(changed/'weights.npz');declared=np.load(changed/'post_source_rows.npy',allow_pickle=False)
                idx=np.searchsorted(declared,post)
                if np.any(idx>=len(declared)) or not np.array_equal(declared[idx],post):raise ContractError('Fine route target identity incomplete: '+name)
                W=full[idx].tocsr()
            else:
                post,W,_,_=FrozenSparseRoute(ROOT,old_routes[name]['source']).selected_rows(post)
            coo=W.tocoo();pg=b['fine_to_group'][post[coo.row]];cg=a['fine_to_group'][coo.col]
            included=cg>=0
            if r.kind!='gap' and not np.all(included):raise ContractError('Missing actual chemical ancestor: '+name)
            total=sparse.csr_matrix((coo.data[included],(pg[included],cg[included])),shape=shape)
            mean=(sparse.diags(1/b['counts'])@total).tocsr()
            expected=np.bincount(pg,weights=coo.data,minlength=shape[0])/b['counts']
            if r.kind!='gap' and not np.allclose(np.asarray(mean.sum(1)).ravel(),expected,rtol=0,atol=1e-12):
                raise ContractError('Complete fine row/member projection mismatch: '+name)
            selected_edges=int(np.sum(included))
            if r.kind=='gap':
                degree=np.bincount(pg,weights=coo.data,minlength=shape[0])
                exterior=np.bincount(pg[~included],weights=coo.data[~included],minlength=shape[0])
                gap_records.append((r.pre,r.post,total,degree,exterior))
        save_matrix('routes/'+name+'/sum_weights.npz',total)
        save_matrix('routes/'+name+'/mean_weights.npz',mean)
        record.update(group_pair_entries=total.nnz,fine_edge_entries=selected_edges,
            sum_semantics='Complete fine rows projected by member incidence; no crop normalization',
            mean_semantics='Sum divided by ALL postsynaptic members exactly once')
        if r.contact_component:
            prefix='routes/'+name+'/'
            if changed.exists():
                distance=np.linalg.norm(table(r.post).coordinates['ipl_preterminal'][post[coo.row]]-table(r.pre).coordinates['ipl_preterminal'][coo.col],axis=1)
                p=read(ROOT/'configurations/effective_dynamics_v1.json')['profiles']['effective_'+r.contact_component]
                gain=p['g_peak']['value']*np.exp(-(distance/.03)**2)
                coefficient=sparse.csr_matrix((gain,(pg,cg)),shape=shape)
                pairs=pg*shape[1]+cg;keys=np.unique(pairs);mapping=np.searchsorted(keys,pairs)
                fine_ids=np.arange(len(pairs),dtype=np.int64)
                values=dict(contact_post_group=keys//shape[1],contact_pre_group=keys%shape[1],
                    fine_contact_global_rows=fine_ids,fine_contact_to_group=mapping,
                    contact_member_offsets=np.r_[0,np.cumsum(np.bincount(mapping,minlength=len(keys)))],
                    contact_member_global_rows=fine_ids[np.argsort(mapping,kind='stable')],
                    fine_contact_pre_source_rows=coo.col,fine_contact_post_source_rows=post[coo.row])
                record['contact']['source_geometry']=dict(rule='Original exp(-(distance/.03mm)^2), new actual IPL endpoints',
                    source_prior=record['contact']['source_geometry'],namespace=canonical_hash(dict(BC=catalog['bc_density_geometry_identity'],route=name)))
            else:
                old_post=base.array(prefix+'contact_post_group.npy');old_pre=base.array(prefix+'contact_pre_group.npy')
                mp=np.full(base.populations[r.post].group_count,-1,np.int64);mp[parent_rows[r.post]]=np.arange(shape[0])
                mc=np.full(base.populations[r.pre].group_count,-1,np.int64);mc[parent_rows[r.pre]]=np.arange(shape[1])
                take=np.flatnonzero(mp[old_post]>=0)
                if np.any(mc[old_pre[take]]<0):raise ContractError('Physical contact closure incomplete: '+name)
                c_map=np.full(len(old_post),-1,np.int64);c_map[take]=np.arange(len(take))
                old_map=base.array(prefix+'fine_contact_to_group.npy');keep=c_map[old_map]>=0
                mapping=c_map[old_map[keep]];fine_ids=base.array(prefix+'fine_contact_global_rows.npy')[keep]
                values=dict(contact_post_group=mp[old_post[take]],contact_pre_group=mc[old_pre[take]],
                    fine_contact_global_rows=fine_ids,fine_contact_to_group=mapping,
                    contact_member_offsets=np.r_[0,np.cumsum(np.bincount(mapping,minlength=len(take)))],
                    contact_member_global_rows=fine_ids[np.argsort(mapping,kind='stable')])
                coefficient=base.matrix(prefix+'contact_coefficient_S.npz')[parent_rows[r.post]][:,parent_rows[r.pre]].tocsr()
            for key,value in values.items():np.save(folder/(key+'.npy'),value,allow_pickle=False)
            save_matrix(prefix+'contact_coefficient_S.npz',coefficient)
            record['contact'].update(groups=len(values['contact_post_group']),fine_contacts=len(values['fine_contact_global_rows']),coefficient_sum=float(coefficient.sum()))
            contact_counts[r.contact_component]=len(values['contact_post_group'])
        route_meta.append(record)
    # Full fine incident degree is retained; missing neighbors are a held reservoir.
    order=base.manifest['gap']['population_order'];sizes=[len(local[n]['counts']) for n in order];offsets=np.r_[0,np.cumsum(sizes)]
    blocks=[[sparse.csr_matrix((i,j)) for j in sizes] for i in sizes];degree=np.zeros(offsets[-1]);missing=np.zeros_like(degree);product=np.zeros_like(degree)
    profiles=read(ROOT/'configurations/effective_dynamics_v1.json')['profiles']
    for pre,post,m,d,e in gap_records:
        i,j=order.index(post),order.index(pre);blocks[i][j]=blocks[i][j]+m
        section=slice(offsets[i],offsets[i+1]);degree[section]+=d;missing[section]+=e
        product[section]-=e*float(profiles['effective_'+pre]['V0']['value'])
    adjacency=sparse.bmat(blocks,format='csr');lap=(sparse.diags(degree)-adjacency).tocsr();lap.eliminate_zeros()
    delta=lap-lap.T;scale=np.max(np.abs(lap.data),initial=0.)
    if delta.nnz and np.max(np.abs(delta.data))>1e-12*scale:raise ContractError('Reciprocal new BC/AII gap symmetry failed')
    if not np.allclose(np.asarray(lap.sum(1)).ravel(),missing,rtol=1e-12,atol=1e-24):raise ContractError('Gap exterior conservation failed')
    save_matrix('gap/laplacian_S.npz',lap)
    np.savez_compressed(output/'GAP_BOUNDARY.npz',gap_product=product,gap_missing_conductance_S=missing)
    config=read(OLD/'prepared/local_static/selection_identity.json')['configuration']
    meta.update(schema='GIN69-HC2-BC1-member-projected-local-1',routes=route_meta,
        local_patch=dict(configuration=config),hc_projection_contract=dict(schema='fine_typed_complete_rows_all_member_mean_v1'),
        source_artifacts=dict(unchanged_parent=base.artifact_sha256,new_BC=catalog['bc_density_geometry_identity'],new_HC=nr['current_HC_identity'],
            fine_receipt=digest(FINE/'FINE_ASSEMBLY_RESULT.json'),native_receipt=digest(NATIVE/'NATIVE_ASSEMBLY_RESULT.json')),
        boundary=dict(gap='Full incident degree; held per-type source-construction V0 at t=0',native='Complete typed parent at independently solved source background',
            missing_conductance_sum_S=float(missing.sum()),crop_normalization=False),
        gap=dict(population_order=order,group_counts=sizes,laplacian_entries=lap.nnz),
        functional_AC_closure=ac_closure,contact_groups=contact_counts,
        target=read(OLD/'prepared/GEOMETRY_SELECTION.json'),neural_forward_started=False,
        FIT_READY=False,background_qualified=False,production_adopted=False)
    meta.pop('manifest_sha256',None);meta.pop('files',None)
    np.savez_compressed(output/'PARENT_ROWS.npz',**parent_rows)
    for name,g in parent_geometry.items():
        if name in BC:np.savez_compressed(output/'populations'/name/'PARENT_GEOMETRY.npz',counts=g.counts,centroid_mm=g.centroid_mm)
    meta['files']={p.relative_to(output).as_posix():dict(sha256=digest(p),bytes=p.stat().st_size) for p in output.rglob('*') if p.is_file() and p.suffix in ('.npy','.npz')}
    meta['artifact_sha256']=canonical_hash(meta);atomic_json(output/'GROUP_ARTIFACT.json',meta)
    result=dict(status='GROUP_ROUTES_CONTACTS_AND_HELD_GAP_READY',artifact=meta['artifact_sha256'],
        groups={n:len(g['counts']) for n,g in local.items()},contact_groups=contact_counts,routes=len(route_meta),
        AC_missing_edges=0,gap_missing_conductance_sum_S=float(missing.sum()),wall_seconds=time.perf_counter()-start,
        background_constructed=False,neural_forward_started=False)
    atomic_json(output/'RESULT.json',result);print(json.dumps(result),flush=True)
