"""Freeze the existing typed inner rules against one global cone population.

Construction uses global coordinate trees and sparse row blocks. It does not
turn effective BC/AC sites into measured subtype densities or alter dynamics.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from .contracts import Quantity,ContractError
from .ipl_geometry import read_ipl_geometry,RESEARCH
from .atlas_inner import ADAPTER
from .disk_tables import FrozenDiskTable,write_table
from .disk_graph_storage import SparseRouteWriter,FrozenSparseRoute,materialize_bounded_route,save_json,array_identity
from .disk_source_geometry import wool_temporal_array
from .disk_pooling import bounded_matrix
from .wiring import wool_midget_profile

def point(values,meaning,frame):return Quantity(values,'mm',meaning,'project_approximation',frame)
def route_path(name):return name.replace('->','_to_')

def metadata(parent,name,source_version=None):
    return dict(parent.metadata,population=name,source_version=source_version or parent.metadata['source_version'])

def route_recipe(name,pre,post):
    return {'route_id':name,'pre_population':pre.metadata['population'],'post_population':post.metadata['population'],
        'pre_table':pre.binding,'post_table':post.binding,'pre_count':pre.count,'post_count':post.count,
        'hard_support_mm':.04,'evidence_class':'project_approximation','boundary':'finite_global_canonical_network',
        'weights':'original rule frozen once; no crop renormalization'}

def freeze_bc_tables(root,outer,support,directory):
    root=Path(root).resolve();directory=Path(directory);cone=FrozenDiskTable(root,outer['cones'])
    cfg=read_ipl_geometry(root,RESEARCH);version=cone.metadata['source_version']+'|'+ADAPTER+'|'+cfg['profile_sha256']+'|connected_cone_centroid_isetbio_v1'
    rf=cone.coordinates['rf'];opl=cone.coordinates['opl_pedicle'];labels=cone.extras['subtype']
    eligible=np.flatnonzero(np.sum(opl**2,axis=1)<=support['limits']['inner_OPL_support_radius_mm']**2)
    groups={'midget_BC_ON':eligible[labels[eligible]!=b'S'],'midget_BC_OFF':eligible[labels[eligible]!=b'S'],
        'S_ON_BC':eligible[labels[eligible]==b'S']}
    diffuse=[]
    for start in range(0,len(eligible),8192):
        indices=eligible[start:start+8192];ids=cone.rows(indices).global_cell_ids
        diffuse.extend(int(i) for i,gid in zip(indices,ids) if
            int(hashlib.sha256((gid+'|effective_LM_stride4_v1').encode()).hexdigest()[:16],16)%4==0)
    # Existing diffuse seed order is lexical parent-ID order, retained globally.
    diffuse=np.asarray(diffuse,dtype=np.int64)
    ids=cone.rows(diffuse).global_cell_ids
    diffuse=diffuse[np.argsort(np.asarray(ids),kind='stable')]
    groups.update(diffuse_BC_ON=diffuse,diffuse_BC_OFF=diffuse)
    result={'tables':{},'routes':{},'inner_source_version':version,'IPL_profile_sha256':cfg['profile_sha256'],
        'status':'GLOBAL_BC_TABLES_AND_CONE_INPUTS_PENDING','subtype_density_claim':False}
    for name,indices in groups.items():
        ready=labels[indices]!=b'S' if name.startswith('diffuse') else np.ones(len(indices),bool)
        table=write_table(root,directory/name,metadata(cone,name,version),len(indices),
            lambda a,b,idx=indices:cone.rows(idx[a:b]).global_cell_ids,
            {'rf':point(rf[indices],'functional_position','functional_visual_field'),
             'ipl_preterminal':point(opl[indices],'terminal_position','IPL')},
            extras={'cone_row':indices,'eligible_drive':ready},source_bindings={
                'cones':cone.binding,'IPL_profile_sha256':cfg['profile_sha256'],
                'sampling':'one parent per nonS midget/S seed, or existing one-in-four hash diffuse effective seeds',
                'physical_support_radius_mm':support['limits']['inner_OPL_support_radius_mm'],
                'known_S_diffuse_placeholders':'structural rows retained but have no functional cone drive'})
        result['tables'][name]=table.binding;name_route='cones->'+name;recipe=route_recipe(name_route,cone,table)
        target=directory/'routes'/route_path(name_route)
        if name.startswith('diffuse'):
            recipe.update(selection_budget=8,gaussian_scale='.08+.04*RF_radius_mm',status_gate='known S seed blocked; otherwise minimum one finite input')
            binding=materialize_bounded_route(root,target,recipe,opl,table.coordinates['ipl_preterminal'],8,
                gaussian_scale=.08+.04*np.linalg.norm(table.coordinates['rf'],axis=1),
                pre_eligible=np.flatnonzero(labels!=b'S'),post_enabled=ready)
        else:
            recipe.update(selection='explicit canonical cone parent; no nearest substitution',parent_cone_rows=array_identity(indices))
            writer=SparseRouteWriter(root,target,recipe)
            if not writer.manifest.exists():
                for start in range(writer.next_row,table.count,4096):
                    stop=min(table.count,start+4096);count=stop-start
                    matrix=sparse.csr_matrix((np.ones(count),(np.arange(count),indices[start:stop])),shape=(count,cone.count))
                    writer.append(start,matrix,[{'target_degree':1,'actual_degree':1,'missing_degree':0,'distance_mm':0.,
                        'source_rule':'explicit qualified parent'} for _ in range(count)],row_status=['READY_PROJECT_PRIVATE_PARENT']*count)
            binding=writer.freeze()
        result['routes'][name_route]=binding
        save_json(root/directory/'BC_catalog.json',result)
    result['status']='GLOBAL_BC_TABLES_AND_CONE_INPUTS_FROZEN';save_json(root/directory/'BC_catalog.json',result)
    return result

def freeze_midget_rgcs(root,bc_catalog,directory,*,owner_radius=3.,minimum_degree_rule='ceil_target_v1'):
    """Freeze Wool/private-line pools, then select by actual connected-cone RF."""
    from .geometry import PackerDensityField
    from .runtime_profiles import frozen_packer_table
    from .isetbio_geometry import connected_cone_rf_centers
    root=Path(root).resolve();directory=Path(directory)
    if minimum_degree_rule not in ('ceil_target_v1','floor_minimum_keep_continuous_deficit_v1'):
        raise ContractError('Explicit supported midget minimum-degree rule required')
    floor_minimum=minimum_degree_rule=='floor_minimum_keep_continuous_deficit_v1'
    field=PackerDensityField(frozen_packer_table(root),central_policy='central_constant_cap',angular_policy='log_NT_cosine')
    result={'tables':{},'routes':{},'candidate_routes':{},'blocked_candidates':{},'status':'CONSTRUCTING_GLOBAL_MIDGET_OUTPUTS'}
    for polarity in ('ON','OFF'):
        pre=FrozenDiskTable(root,bc_catalog['tables']['midget_BC_'+polarity]);name='midget_RGC_'+polarity
        positions=pre.coordinates['ipl_preterminal'];rf=pre.coordinates['rf'];equivalent=wool_temporal_array(field,rf)
        valid=np.isfinite(equivalent)&(equivalent>=0)&(equivalent<=10)
        convergence=wool_midget_profile(np.where(valid,equivalent,0.))
        target=convergence['effective_contact_target'];scale=np.where(convergence['private_line_branch'],.001,convergence['center_radius_mm'])
        radius=np.minimum(.04,scale*np.sqrt(np.log(100.)))
        recipe={'route_id':'candidate_midget_BC_'+polarity+'->'+name,'pre_population':pre.metadata['population'],
            'post_population':name+'_candidate_geometry','pre_table':pre.binding,'pre_count':pre.count,'post_count':pre.count,
            'candidate_identity':'canonical BC parent IDs in complete fixed BC row order',
            'temporal_equivalence':'existing Wool/Packer temporal iso-density equation; no retinal x/r substitution',
            'geometry_scope':'Retained project pooling rule on actual supplied cone parents, including explicitly borrowed human geometry; no Curcio-to-Packer density fit',
            'source_fit':'Wool .25..10mm temporal equivalent; existing central private/minimum1 convention retained',
            'support':'min(.04,Rc*sqrt(log100)); private Rc=.001mm; no support enlargement',
            'equivalence':array_identity(equivalent),'targets':array_identity(target),'radii':array_identity(radius)}
        if floor_minimum:
            if np.any(~np.isfinite(target)) or np.any(target<1):
                raise ContractError('Approved floor minimum requires finite original n>=1; no zero-input qualification')
            recipe.update(minimum_degree_rule=minimum_degree_rule,
                target_interpretation='continuous target retained; floor is execution minimum only, population average not certified',
                original_ceil_deficit_retained=True,pooling_formula_or_support_changed=False)
        work=directory/'candidate_routes'/name;writer=SparseRouteWriter(root,work,recipe);tree=cKDTree(positions)
        functional_rf=point(rf,'functional_position','functional_visual_field')
        if not writer.manifest.exists():
            for start in range(writer.next_row,pre.count,4096):
                stop=min(pre.count,start+4096);matrix,rows=bounded_matrix(positions,positions[start:stop],target[start:stop],
                    support_mm=radius[start:stop],gaussian_scale=scale[start:stop],pre_tree=tree)
                matrix=matrix.tolil();statuses=[]
                for j,row in enumerate(rows):
                    index=start+j
                    if not valid[index]:status='BLOCKED_WOOL_TEMPORAL_EQUIVALENCE_MISSING'
                    elif target[index]<=1:
                        # This exact qualified parent is at the same IPL point.
                        matrix.rows[j]=[index];matrix.data[j]=[1.];status='READY_PROJECT_PRIVATE_OR_MINIMUM_ONE'
                    elif floor_minimum and row['actual_degree']>=int(np.floor(target[index])):
                        status='READY_PROJECT_WOOL_FLOOR_MINIMUM_WITH_UNMET_TARGET' if row['missing_degree'] else 'READY_PROJECT_WOOL_CONVERGENCE'
                    elif row['missing_degree']:status='INCOMPLETE_MIDGET_CONVERGENCE_FINITE_SUPPORT'
                    else:status='READY_PROJECT_WOOL_CONVERGENCE'
                    if not status.startswith('READY'):matrix.rows[j]=[];matrix.data[j]=[]
                    row.update(temporal_equivalent_mm=float(equivalent[index]) if valid[index] else None,
                        qualified_parent_bc_row=index,private_or_minimum_one_convention=bool(target[index]<=1),
                        RF_center_rule='connected_cone_centroid_isetbio_v1')
                    if floor_minimum:
                        row.update(minimum_degree_rule=minimum_degree_rule,
                            minimum_required_degree=int(np.floor(target[index])),
                            minimum_degree_deficit=max(0,int(np.floor(target[index]))-row['actual_degree']),
                            unmet_continuous_target_count=max(0.,float(target[index])-row['actual_degree']),
                            population_average_target_satisfied='NOT_ESTABLISHED')
                    statuses.append(status)
                matrix=matrix.tocsr();centers=np.full((stop-start,2),np.nan);ready=np.array([s.startswith('READY') for s in statuses])
                if np.any(ready):centers[ready]=connected_cone_rf_centers(functional_rf,matrix[ready]).values
                writer.append(start,matrix,rows,row_status=statuses,extras={'RF_center_mm':centers})
        candidate_binding=writer.freeze();candidate=FrozenSparseRoute(root,candidate_binding)
        centers=np.full((pre.count,2),np.nan);ready=candidate.ready_rows()
        for block in candidate.record['blocks']:
            with np.load(root/block['path'],allow_pickle=False) as z:centers[block['start']:block['stop']]=z['RF_center_mm']
        selected=np.flatnonzero(ready&(np.sum(centers**2,axis=1)<=owner_radius**2))
        blocked=np.flatnonzero(~ready&(np.sum(rf**2,axis=1)<=owner_radius**2))
        table=write_table(root,directory/name,metadata(pre,name),len(selected),lambda a,b,idx=selected:pre.rows(idx[a:b]).global_cell_ids,
            {'rf':point(centers[selected],'functional_position','functional_visual_field'),
             'ipl_post_arbor':point(positions[selected],'dendritic_position','IPL')},
            extras={'parent_bc_row':selected},source_bindings={'BC_table':pre.binding,'candidate_pool':candidate_binding,
                'functional_output_disk_radius_mm':owner_radius,'selection':'actual connected-cone RF center; not soma or seed radius'})
        final_recipe=route_recipe(pre.metadata['population']+'->'+name,pre,table)
        final_recipe.update(frozen_candidate_pool=candidate_binding,selected_candidate_rows=array_identity(selected),
            source_rule='unchanged candidate row weights copied without normalization')
        final=SparseRouteWriter(root,directory/'routes'/route_path(pre.metadata['population']+'->'+name),final_recipe)
        if not final.manifest.exists():
            written=0
            for block in candidate.record['blocks']:
                rows=selected[(selected>=block['start'])&(selected<block['stop'])]
                if not len(rows):continue
                stop=written+len(rows)
                if stop<=final.next_row:written=stop;continue
                if written!=final.next_row:raise ContractError('Filtered midget route resume boundary differs')
                matrix,audits,statuses=candidate.block(block);local=rows-block['start']
                final.append(written,matrix[local],[audits[i] for i in local],row_status=[statuses[i] for i in local]);written=stop
        result['tables'][name]=table.binding;result['routes'][pre.metadata['population']+'->'+name]=final.freeze()
        result['candidate_routes'][name]=candidate_binding
        result['blocked_candidates'][name]={'count':len(blocked),'parent_BC_rows':blocked.tolist(),
            'meaning':'unresolved/deficient candidate; not biological absence or a synthetic replacement'}
        save_json(root/directory/'midget_catalog.json',result)
    result['status']='GLOBAL_MIDGET_OUTPUTS_FROZEN_WITH_EXPLICIT_BLOCKED_CANDIDATES';save_json(root/directory/'midget_catalog.json',result)
    return result


def freeze_remaining_inner(root,outer,bc_catalog,midget_catalog,populations,support,directory,*,atlas_profile='retina_atlas_disk3mm_v1'):
    """Parasol/S outputs, fixed AC sites and unchanged finite typed contacts."""
    from .disk_population import FixedDiskPopulation
    from .ipl_geometry import freeze_centroid_anchors,widefield_grid
    from .source_atlas import SourceDensityAtlas
    from .disk_source_geometry import schein_array
    root=Path(root).resolve();directory=Path(directory);cone=FrozenDiskTable(root,outer['cones'])
    atlas=SourceDensityAtlas(root,profile_id=atlas_profile);radius=support['limits']['inner_OPL_support_radius_mm']
    tables={name:FrozenDiskTable(root,binding) for name,binding in dict(bc_catalog['tables'],**midget_catalog['tables']).items()}
    routes=dict(bc_catalog['routes'],**midget_catalog['routes']);geometry_routes={}
    ready={name:FrozenSparseRoute(root,routes['cones->'+name]).ready_rows() for name in bc_catalog['tables']}
    version=bc_catalog['inner_source_version'];parasol_sites=[]
    for polarity in ('ON','OFF'):
        name='parasol_RGC_'+polarity;pre=tables['diffuse_BC_'+polarity]
        master=FixedDiskPopulation(root,populations[name]);seed=master.rows(np.arange(len(master.positions)))
        if seed.reference_id!=cone.metadata['reference_id']:raise ContractError('Parasol master reference differs from cone atlas')
        rf=master.positions;pedicle=schein_array(atlas.henle,rf);eligible=np.flatnonzero(ready[pre.metadata['population']])
        pre_ids=tuple(g for start in range(0,pre.count,8192) for g in pre.ids(start,min(pre.count,start+8192)))
        eligible_ids=tuple(pre_ids[i] for i in eligible)
        anchors,records=freeze_centroid_anchors(pedicle,pre.coordinates['ipl_preterminal'][eligible],eligible_ids)
        table=write_table(root,directory/name,{key:getattr(seed,key) for key in ('atlas_id','atlas_version','source_version','population','reference_id')},
            len(rf),lambda a,b:seed.source_keys[a:b],
            {'rf':point(rf,'functional_position','functional_visual_field'),'ipl_post_arbor':point(anchors,'dendritic_position','IPL')},
            source_bindings={'parasol_master':master.binding,'BC_table':pre.binding,'anchor_rule':'existing freeze_centroid_anchors, all ready BCs within40um, equal mean, canonical ID order',
                'anchor_shift':'frozen before final contacts; never adjusted to fix deficits','Schein':atlas.henle.provenance})
        tables[name]=table;parasol_sites.extend((gid,r,p) for gid,r,p in zip(table.ids(0,table.count),rf,anchors))
        lookup={gid:i for i,gid in enumerate(pre_ids)};row=[];column=[];weight=[]
        for j,record in enumerate(records):
            selected=record['candidate_bc_ids'];row.extend([j]*len(selected));column.extend(lookup[gid] for gid in selected)
            weight.extend(record['candidate_weights'])
        matrix=sparse.csr_matrix((weight,(row,column)),shape=(table.count,pre.count))
        recipe=route_recipe('frozen_anchor_pool_'+name,pre,table)
        recipe.update(role='geometry construction ancestry only; not additional dynamical contact',selection='all ready BC within40um of fixed Schein seed; equal mean')
        writer=SparseRouteWriter(root,directory/'anchor_pools'/name,recipe)
        if not writer.manifest.exists():
            for start in range(writer.next_row,table.count,4096):
                stop=min(table.count,start+4096);block=matrix[start:stop]
                writer.append(start,block,[{'actual_degree':int(n),'missing_degree':0,'geometry_only':True} for n in np.diff(block.indptr)])
        geometry_routes[name]=writer.freeze()
    pre=tables['S_ON_BC'];selected=np.flatnonzero(np.sum(pre.coordinates['rf']**2,axis=1)<=3.**2)
    name='small_bistratified_S_ON'
    tables[name]=write_table(root,directory/name,metadata(pre,name),len(selected),lambda a,b:pre.rows(selected[a:b]).global_cell_ids,
        {'rf':point(pre.coordinates['rf'][selected],'functional_position','functional_visual_field'),
         'ipl_post_arbor':point(pre.coordinates['ipl_preterminal'][selected],'dendritic_position','IPL')},
        extras={'parent_bc_row':selected},source_bindings={'S_ON_BC':pre.binding,'functional_output_disk_radius_mm':3.,
            'rule':'existing S-seeded effective relay; no forced central S cell or measured RGC density claim'})
    # Same source/site keys as revise_local_graph, sorted before assigning types.
    selected=np.flatnonzero(np.sum(cone.coordinates['opl_pedicle']**2,axis=1)<=radius**2)
    sites=[]
    for start in range(0,len(selected),8192):
        indices=selected[start:start+8192];gids=cone.rows(indices).global_cell_ids
        sites.extend(('cone_site:'+gid,cone.coordinates['rf'][i],cone.coordinates['opl_pedicle'][i]) for gid,i in zip(gids,indices))
    sites.extend(parasol_sites);sites.sort(key=lambda row:row[0])
    keys=tuple(row[0] for row in sites);rf=np.asarray([row[1] for row in sites]);xy=np.asarray([row[2] for row in sites])
    for name in ('local_feedforward_AC','local_crossover_AC'):
        tables[name]=write_table(root,directory/name,metadata(cone,name,version),len(keys),lambda a,b:keys[a:b],
            {'rf':point(rf,'functional_position','functional_visual_field'),
             'ipl_preterminal':point(xy,'terminal_position','IPL'),'ipl_post_arbor':point(xy,'dendritic_position','IPL')},
            source_bindings={'cones':cone.binding,'parasol_tables':{n:tables[n].binding for n in ('parasol_RGC_ON','parasol_RGC_OFF')},
                'sampling':'existing one independent typed effective state per fixed cone/parasol site; not measured AC density',
                'finite_site_support_radius_mm':radius,'IPL_profile_sha256':bc_catalog['IPL_profile_sha256']})
    keys,grid,_=widefield_grid((-radius,radius,-radius,radius),max_cells=1000000)
    keep=np.flatnonzero(np.sum(grid**2,axis=1)<=radius**2);grid=grid[keep];keys=tuple(keys[i] for i in keep)
    tables['widefield_AC']=write_table(root,directory/'widefield_AC',metadata(cone,'widefield_AC',version),len(keys),lambda a,b:keys[a:b],
        {'ipl_preterminal':point(grid,'terminal_position','IPL'),'ipl_post_arbor':point(grid,'dendritic_position','IPL')},
        source_bindings={'grid_spacing_mm':.04/3,'grid_origin_mm':[0.,0.],'finite_IPL_disk_radius_mm':radius,
            'sampling':'existing fixed effective computational grid; not measured AC density',
            'RF_and_soma':'ABSENT; no inverse through Schein pedicle-free center and no coordinate fallback',
            'anatomical_eccentricity_view':'norm of registered physical IPL anchor; explicit project chart registration only'})
    def pool(pre_name,post_name,k,*,sigma=None,exact=False):
        pre,post=tables[pre_name],tables[post_name];name=pre_name+'->'+post_name
        recipe=route_recipe(name,pre,post);recipe.update(selection_budget=k,exact_required_degree=exact,
            pre_eligibility='ready upstream typed drive only',gaussian_rule='existing sigma by target RF' if sigma is not None else 'equal finite weights')
        routes[name]=materialize_bounded_route(root,directory/'routes'/route_path(name),recipe,
            pre.coordinates['ipl_preterminal'],post.coordinates['ipl_post_arbor'],k,gaussian_scale=sigma,
            pre_eligible=np.flatnonzero(ready[pre_name]),exact_degree=exact)
        return FrozenSparseRoute(root,routes[name]).ready_rows()
    for name,pre,budget in (('local_feedforward_AC','diffuse_BC_ON',4),('local_crossover_AC','diffuse_BC_OFF',4),('widefield_AC','diffuse_BC_ON',16)):
        ready[name]=pool(pre,name,budget)
    pool('S_ON_BC','small_bistratified_S_ON',8)
    for name,pre,inh in (('midget_RGC_ON','midget_BC_ON','local_crossover_AC'),
        ('midget_RGC_OFF','midget_BC_OFF','local_feedforward_AC'),('small_bistratified_S_ON','S_ON_BC','local_feedforward_AC'),
        ('parasol_RGC_ON','diffuse_BC_ON','widefield_AC'),('parasol_RGC_OFF','diffuse_BC_OFF','widefield_AC')):
        sigma=.02+.06*np.linalg.norm(tables[name].coordinates['rf'],axis=1)
        pool(inh,name,1 if name.startswith('midget') else 8,sigma=sigma,exact=True)
        if name.startswith('parasol'):pool(pre,name,8,sigma=sigma,exact=True)
        if name=='small_bistratified_S_ON':pool('diffuse_BC_OFF',name,8,sigma=sigma,exact=True)
    result={'tables':{name:table.binding for name,table in tables.items()},'routes':routes,'parasol_anchor_geometry_pools':geometry_routes,
        'midget_blocked_candidates':midget_catalog['blocked_candidates'],
        'status':'GLOBAL_INNER_TABLES_AND_TYPED_CONTACTS_FROZEN_PENDING_AII_AND_RUNTIME',
        'population_density_claims':'parasol source/model rule and S route retained; BC/AC/midget sites remain existing effective scaffold',
        'parameter_profile':'effective_dynamics_v1','parameter_changes':False,'regenerate_on_ROI':False,
        'physiological_validation':False,'old_default_changed':False}
    save_json(root/directory/'inner_catalog.json',result);return result


def freeze_aii_bc_contacts(root,inner,populations,directory):
    """Existing chemical/gap laws on all eligible fixed BC/AII rows."""
    from .disk_population import FixedDiskPopulation
    from .disk_outer import _write_matrix
    root=Path(root).resolve();directory=Path(directory);aii=FixedDiskPopulation(root,populations['AII_AC'])
    xy=aii.positions;routes={};gap_routes={};tree=cKDTree(xy)
    for name in ('midget_BC_OFF','diffuse_BC_OFF'):
        post=FrozenDiskTable(root,inner['tables'][name]);enabled=FrozenSparseRoute(root,inner['routes']['cones->'+name]).ready_rows()
        recipe={'route_id':'AII_'+name+'_glycine','pre_population':'AII_AC','post_population':name,
            'pre_master':aii.binding,'post_table':post.binding,'pre_count':len(xy),'post_count':post.count,
            'hard_support_mm':.04,'selection_budget':3,'overlap':'exp(-(distance/.03)^2), original physical synapse contact law',
            'weights':'unchanged bounded finite pool; deficits retained','source':'existing project primary pathway coefficients',
            'physiological_validation':False}
        routes[recipe['route_id']]=materialize_bounded_route(root,directory/recipe['route_id'],recipe,xy,
            post.coordinates['ipl_preterminal'],3,post_enabled=enabled)
    for name in ('midget_BC_ON','diffuse_BC_ON','S_ON_BC'):
        post=FrozenDiskTable(root,inner['tables'][name]);enabled=FrozenSparseRoute(root,inner['routes']['cones->'+name]).ready_rows()
        points=post.coordinates['ipl_preterminal'];_,nearest=tree.query(points,k=1)
        distance=np.linalg.norm(xy[nearest]-points,axis=1);selected=np.flatnonzero(enabled&(distance<=.04))
        conductance=.15e-9*np.exp(-(distance[selected]/.03)**2)
        matrix=sparse.csr_matrix((conductance,(selected,nearest[selected])),shape=(post.count,len(xy)))
        name_route='AII_'+name+'_bidirectional_gap'
        recipe={'route_id':name_route,'pre_population':'AII_AC','post_population':name,
            'pre_master':aii.binding,'post_table':post.binding,'pre_count':len(xy),'post_count':post.count,
            'selection':'unchanged nearest one AII per eligible BC then physical40um cutoff; no nearest fallback beyond support',
            'conductance_S':'.15e-9*exp(-(distance/.03)^2)','normalization':False,
            'reciprocal':True,'physiological_validation':False}
        forward=_write_matrix(root,directory/name_route,recipe,matrix,unit='S',empty_status='NO_GAP_PARTNER_WITHIN40UM')
        reverse_recipe=dict(recipe,route_id=name_route+'_reverse',pre_population=name,post_population='AII_AC',
            pre_count=post.count,post_count=len(xy),forward_gap=forward)
        reverse_recipe.pop('pre_master');reverse_recipe.pop('post_table')
        reverse_recipe.update(pre_table=post.binding,post_master=aii.binding)
        reverse=_write_matrix(root,directory/(name_route+'_reverse'),reverse_recipe,matrix.T.tocsr(),unit='S',empty_status='NO_RECIPROCAL_BC_GAP_PARTNERS')
        gap_routes[name_route]={'forward':forward,'reverse':reverse,'undirected_edges':matrix.nnz}
    result={'chemical_routes':routes,'gap_routes':gap_routes,'status':'GLOBAL_AII_BC_CONTACTS_FROZEN',
        'HC_gap_enabled':False,'new_conductance_or_receptor_fit':False,'physiological_validation':False}
    save_json(root/directory/'AII_BC_catalog.json',result);return result
