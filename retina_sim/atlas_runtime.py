"""Bind actual atlas IDs/typed edges to existing dynamics, without another sampler.

This named owner-midget cone circuit is an explicit usable partial scope. The
source-missing rod/AII and unbound parasol routes are reported, never made zero.
"""
from pathlib import Path
from dataclasses import replace
import json,hashlib
import numpy as np
from scipy import sparse

from .contracts import Quantity,CellTable,RunMode,ContractError,canonical_hash
from .spatial import Population,InputChannel
from .atlas import AtlasROI
from .atlas_connections import ConnectionAtlas
from .assembly import ModelSpec,ComponentSpec,ExternalInput,PortLink,RetinaBuilder
from .graph import ConnectionTable
from .core_sources import load_core_registry
from .runtime_profiles import RuntimeProfile,LightContext
from .provenance import SourceRecord
from .vh2005 import KEY,PROFILE
from examples.full_target_network import register_profile,PROJECT_SOURCE

TAG='project_approximation'
EXECUTION='atlas_midget_research_v1'
def q(v,u,m,frame=None):return Quantity(v,u,m,TAG,frame)


def _sampled_native_light_channel(duration,light,rf_x):
    last_index=int(round(duration/.0001));phase=rf_x[None,:]/.223
    cached_index=None;cached_rows=None
    def evaluate(t):
        nonlocal cached_index,cached_rows
        position=np.clip(t/.0001,0,last_index);i=min(int(np.floor(position)),last_index)
        fraction=position-i
        if i!=cached_index:
            # Preserve the original sampled grid and interpolation; keep only two rows.
            grid=np.array((i,min(i+1,last_index)),dtype=np.int64)*.0001
            cached_rows=light.native_background_td*(1+.1*np.sin(2*np.pi*80*grid[:,None]+phase))
            cached_index=i
        return q((1-fraction)*cached_rows[0]+fraction*cached_rows[1],'td','native_retinal_illuminance')
    return InputChannel(0,duration,evaluate,'td','native_retinal_illuminance','linear',TAG,
        {'source':'declared_short_synthetic_stimulus','background_context_hash':light.background_context_hash,'not_physically_calibrated':True})


def bind_runtime_profile(source_registry,name,n,derived=None):
    derived=derived or {};definitions=source_registry.runtime_profile.data['profiles'][name];entries={}
    if name in getattr(source_registry,'mechanism_profiles',{}):
        old=source_registry.mechanism_profiles[name]
        records=tuple(replace(r,value=Quantity(derived[r.parameter_id],r.value.unit,r.value.meaning,
            r.value.evidence_tag,r.value.coordinate_frame)) if r.parameter_id in derived else r for r in old.records)
        updated=replace(old,records=records);source_registry.mechanism_profiles[name]=updated;source_registry.profiles[name]=updated
        return updated
    kernel_len=len(definitions.get('kernel',definitions.get('spike_history_kernel',{'value':[]}))['value'])
    for key,v in definitions.items():
        if v['binding']=='derived':
            if key not in derived:raise ContractError('Actual atlas initialization missing derived parameter '+name+'.'+key)
            value=derived[key]
        elif v['binding']=='literal':value=v['value']
        else:value=np.full((kernel_len,n) if 'history' in key else (n,),v['value'])
        entries[key]=(value,v['unit'],v['meaning'])
    return register_profile(source_registry,name,n,entries)


def _subset(record,ids):
    if hasattr(record,'select'):return record.select(ids)
    wanted=set(ids);keep=[i for i,g in enumerate(record['global_cell_ids']) if g in wanted]
    return dict(record,global_cell_ids=[record['global_cell_ids'][i] for i in keep],
        source_keys=[record['source_keys'][i] for i in keep],coordinates={role:dict(v,values=[v['values'][i] for i in keep])
            for role,v in record['coordinates'].items()})


def _population(record,kind,profile,species,subtypes=None,opl_pre=None,opl_post=None):
    ids=tuple(record['global_cell_ids']);n=len(ids);coords=record['coordinates']
    def coordinate(role):
        v=coords.get(role)
        return None if v is None else q(v['values'],v['unit'],v['meaning'],v['frame'])
    rf=coordinate('rf');soma=coordinate('soma');pedicle=coordinate('opl_pedicle')
    if subtypes is None:subtypes=(record['population'],)*n
    if rf is None and soma is None:
        explicit=record.get('global_eccentricity')
        if explicit is None:raise ContractError('Explicit eccentricity required when RF and soma coordinates are unavailable')
        eccentricity=q(explicit['values'],explicit['unit'],'global_eccentricity',explicit['frame'])
        radial=eccentricity.values;cell_frame=explicit['frame']
    else:
        cell_frame=rf.coordinate_frame if rf is not None else soma.coordinate_frame
        radial=np.linalg.norm((rf or soma).values,axis=1)
        eccentricity=q(radial,'mm','global_eccentricity',cell_frame)
    regions=tuple('center' if r<1 else 'transition' if r<5 else 'periphery' for r in radial)
    cells=CellTable(ids,(kind,)*n,tuple(subtypes),(species,)*n,('composite_declared',)*n,('registered_axes',)*n,
        soma,rf,'mm',cell_frame,eccentricity,
        (profile,)*n,(TAG,)*n)
    terminal=coordinate('ipl_preterminal')
    dendrite=coordinate('ipl_post_arbor')
    if opl_pre is not None:terminal=q(opl_pre,'mm','terminal_position','OPL')
    if opl_post is not None:dendrite=q(opl_post,'mm','dendritic_position','OPL')
    return Population(record['population'],cells,regions,soma,rf,pedicle,terminal,dendrite)


def _matrix(edges,pre,post):
    from .compact_population_record import CompactContactEdges
    if isinstance(edges,CompactContactEdges):
        rows=edges.remap(pre.cells.cell_id,post.cells.cell_id)
        return sparse.csr_matrix((rows.weights,(rows.rows,rows.columns)),
            shape=(len(rows.post_ids),len(rows.pre_ids))),rows
    a={g:i for i,g in enumerate(pre.cells.cell_id)};b={g:i for i,g in enumerate(post.cells.cell_id)}
    rows=[e for e in edges if e['pre_id'] in a and e['post_id'] in b]
    return sparse.csr_matrix(([e['weight'] for e in rows],([b[e['post_id']] for e in rows],
        [a[e['pre_id']] for e in rows])),shape=(len(b),len(a))),rows

def _contact_edges(contact,pre,post):
    edges=contact['edges']
    from .compact_population_record import CompactContactEdges
    if isinstance(edges,CompactContactEdges):return edges.remap(pre.cells.cell_id,post.cells.cell_id)
    return edges.select(pre.cells.cell_id,post.cells.cell_id) if hasattr(edges,'select') else edges

def _owner_ids(table,bounds):
    if hasattr(table,'owner_ids'):return table.owner_ids(bounds)
    if 'rf' not in table['coordinates']:return ()
    xy=np.asarray(table['coordinates']['rf']['values']);b=bounds
    ids=tuple(table['global_cell_ids'])
    if not ids and xy.size==0:return ()
    if xy.shape!=(len(ids),2):raise ContractError('Owner RF coordinates must match the global IDs as an N-by-2 array')
    mask=(xy[:,0]>=b[0])&(xy[:,0]<b[1])&(xy[:,1]>=b[2])&(xy[:,1]<b[3])
    return tuple(g for g,k in zip(ids,mask) if k)

def _population_owner_mask(population,bounds):
    coordinate=population.cells.rf_position or population.anatomical_positions
    if coordinate is None:return [False]*len(population.cells.cell_id)
    b=bounds
    return [bool(b[0]<=xy[0]<b[1] and b[2]<=xy[1]<b[3]) for xy in coordinate.values]


def _link(pre_component,pre_port,post_component,post_port,pre,post,W,frame,edge_ids=None):
    W=W.tocoo();n=W.nnz
    table=ConnectionTable(tuple(edge_ids or (f'{pre_component}>{post_component}:{j}' for j in range(n))),
        tuple(pre.cells.cell_id[i] for i in W.col),tuple(post.cells.cell_id[i] for i in W.row),
        ('actual_atlas_effective_contact',)*n,tuple(float(x) for x in W.data),('dimensionless',)*n,
        ('nonnegative_contact_weight; receptor owns sign',)*n,(0.,)*n,(post.cells.parameter_profile_id[0],)*n,
        ('unchanged actual atlas contact selection',)*n,(TAG,)*n)
    return PortLink(pre_component,pre_port,post_component,post_port,table,frame)


def atlas_midget_network(root,*,duration=.0012,light_context=None,owner_bounds_mm=(.34,.36,-.01,.01),
        halo_mm=.04,max_spatial_cells=60000,seed=20261010,connection_record=None,typed_completion=False,atlas_profile_id=None,allow_empty_owner_outputs=False,spatial_asset=None,_atlas_context=None,max_compile_bytes=None):
    root=Path(root);light=light_context or LightContext()
    execution=json.loads((root/'configurations'/(EXECUTION+'.json')).read_bytes())
    if atlas_profile_id is not None:execution=dict(execution,atlas_profile=atlas_profile_id)
    atlas=ConnectionAtlas(root,atlas_profile_id=execution['atlas_profile'],s_partition_profile=execution['S_partition_profile']) if _atlas_context is None else _atlas_context
    if max_compile_bytes is not None:
        if not isinstance(max_compile_bytes,int) or max_compile_bytes<=0:raise ContractError('Explicit positive compile budget required')
        execution=dict(execution,budgets=dict(execution['budgets'],max_compile_bytes=max_compile_bytes))
    b=tuple(owner_bounds_mm);roi=AtlasROI(b,(b[0]-halo_mm,b[1]+halo_mm,b[2]-halo_mm,b[3]+halo_mm),atlas.atlas.config['reference_id'])
    frozen=execution['frozen_default_graph']
    if connection_record is not None:
        record=connection_record
        graph_binding={'kind':'supplied_actual_atlas_record','record_hash':record.semantic_hash if hasattr(record,'semantic_hash') else canonical_hash(record)}
    elif np.allclose(b,frozen['owner_bounds_mm'],rtol=0,atol=1e-12) and abs(halo_mm-frozen['halo_mm'])<1e-12:
        payload=(root/frozen['path']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=frozen['sha256']:
            raise ContractError('Frozen actual atlas graph identity mismatch')
        record=json.loads(payload)
        graph_binding={'kind':'frozen_actual_atlas_record',**frozen}
    else:
        record=atlas.preview(roi,max_cells=max_spatial_cells,
            phase='phase2' if max(np.hypot(x,y) for x in b[:2] for y in b[2:])>3 else 'phase1')
        graph_binding={'kind':'fresh_actual_atlas_query','record_hash':canonical_hash(record)}
    if not np.allclose(record['atlas']['owner_domain'],b,rtol=0,atol=1e-12):
        raise ContractError('Atlas owner domain differs from runtime request')
    if record['summary']['S_partition_profile']!=execution['S_partition_profile']:
        raise ContractError('Atlas S partition differs from selected runtime profile')
    if record['atlas']['profile_sha256']!=atlas.atlas.profile_hash or record['atlas']['mosaic_namespace']!=atlas.atlas.mosaic_namespace:
        raise ContractError('Atlas frozen source/profile identity differs; explicit graph refresh required')
    if record['HC']['metadata']['file_sha256']!=atlas.hc_master.config['expected_asset_sha256']:
        raise ContractError('Atlas graph HC master identity differs')
    if record['cone_type_provenance']['profile_sha256']!=atlas.s_partition.profile_sha256:
        raise ContractError('Atlas graph S source profile identity differs')
    population_sizes=([n for k,n in record.population_counts.items() if k=='cones' or k.startswith('inner_tables/')] if hasattr(record,'population_counts') else
        [len(record['cones']['global_cell_ids'])]+[len(t['global_cell_ids']) for t in record['inner_tables'].values()])
    if max(population_sizes)>max_spatial_cells:
        raise ContractError('Atlas graph exceeds requested population budget; no density reduction')
    if record['summary']['HC_query_status']!='supported':raise ContractError('Atlas runtime HC fixed master unsupported; no old-generator replacement')
    if record['summary']['zero_owner_candidate_rows']:
        raise ContractError('JOINT_PROFILE_UNSUPPORTED: owner cones have no HC partners; atlas runtime does not fill or zero-certify this central case')
    if any(t=='UNKNOWN' for t in record['cone_subtypes']):raise ContractError('Atlas runtime requires the explicitly selected effective S route; unknown subtype weights cannot run')
    revised='population_revision' in record
    if revised:
        execution=dict(execution,requested_outputs=['midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON'],
            profile_id=record['population_revision'].get('execution_profile_id','atlas_population_revision_v1'),
            scope=record['population_revision'].get('scope','Approved local .35 population revision with all typed owner pathways and unchanged dynamics'))
        typed_completion=True
        graph_binding=dict(graph_binding,population_revision=record['population_revision'])
    elif typed_completion:
        from .atlas_typed_completion import complete_typed_contacts
        record=complete_typed_contacts(record,atlas)
        execution=dict(execution,requested_outputs=['midget_RGC_ON','midget_RGC_OFF','small_bistratified_S_ON'],
            profile_id='atlas_typed_candidate_v1',
            scope='All available owner cone typed pathways and local/wide AC on the same actual finite atlas; full replacement blocked by explicit absent routes')
        graph_binding=dict(graph_binding,typed_completion=record['typed_completion'])
    canonical_targets=tuple(execution['requested_outputs'])
    required={};needed=set();empty_owner_outputs=[]
    for name in execution['requested_outputs']:
        t=record['inner_tables'][name];required[name]=_owner_ids(t,b)
        if not required[name]:
            if not allow_empty_owner_outputs:raise ContractError('EMPTY_TARGET_OWNER: '+name)
            empty_owner_outputs.append(name)
        needed.update(required[name])
    needed.update(record.get('required_global_dependency_ids',()))
    if empty_owner_outputs:
        execution=dict(execution,requested_outputs=[name for name in execution['requested_outputs'] if name not in empty_owner_outputs])
        if not execution['requested_outputs']:raise ContractError('EMPTY_TARGET_OWNER: all requested outputs are empty samples')
    if typed_completion:
        for name in ('local_feedforward_AC','local_crossover_AC','widefield_AC'):
            table=record['inner_tables'][name];required[name]=_owner_ids(table,b)
            needed.update(required[name])
    while True:
        before=len(needed)
        for contact in record['inner_contacts'].values():
            edges=contact['edges']
            needed.update(edges.parents_for(needed) if hasattr(edges,'parents_for') else (e['pre_id'] for e in edges if e['post_id'] in needed))
        if len(needed)==before:break
    cone_ids=set(record['cones']['global_cell_ids']);cone_needed=needed&cone_ids;hc_needed=set()
    hc_edges=record['HC_candidate_edges']
    if spatial_asset is not None and hasattr(spatial_asset,'native_dependency_ids'):
        cone_needed,hc_needed=spatial_asset.native_dependency_ids(cone_needed)
    else:
        while True:
            before=len(cone_needed)+len(hc_needed)
            pairs=zip(hc_edges.pre,hc_edges.post) if hasattr(hc_edges,'pre') else ((e['cone_id'],e['hc_id']) for e in hc_edges)
            for cone_id,hc_id in pairs:
                if cone_id in cone_needed:hc_needed.add(hc_id)
                if hc_id in hc_needed:cone_needed.add(cone_id)
            if len(cone_needed)+len(hc_needed)==before:break
    source_registry=load_core_registry(root);source_registry.runtime_profile=RuntimeProfile(root,execution['parameter_profile'])
    asset=root/'sources/primary_pathway_effective_profile.json'
    source_registry.register_source(SourceRecord(PROJECT_SOURCE,'Existing effective circuit parameters on actual atlas geometry',('Project implementation',),2026,
        'https://pmc.ncbi.nlm.nih.gov/articles/PMC6218188/','supplement',None,str(asset.relative_to(root)),
        'unchanged project parameter ledger; atlas binding does not imply measurement',source_registry.runtime_profile.source_identity(PROJECT_SOURCE),None,
        'Existing project implementation','identity_verified','Effective parameters; no new physiological calibration','2026-10-12',{}))
    def profile(name,n,derived=None):return bind_runtime_profile(source_registry,name,n,derived)
    pops={};specs=[];links=[];by={};binding_edges={}
    cr=_subset(record['cones'],cone_needed);cr['population']='cones'
    cone_type=dict(zip(record['cones']['global_cell_ids'],record['cone_subtypes']))
    native_parameters=source_registry.load_parameters(PROFILE,RunMode.REFERENCE)
    cones=_population(cr,'cone',PROFILE,native_parameters.profile.species,[cone_type[g] for g in cr['global_cell_ids']]);pops['cones']=cones
    hr=_subset(record['HC']['table'],hc_needed);hr['population']='HC'
    hc_type=dict(zip(record['HC']['table']['global_cell_ids'],record['HC']['types']))
    anchor=dict(zip(record['HC']['table']['global_cell_ids'],record['HC']['OPL_anchor']['values']))
    hc=_population(hr,'HC',PROFILE,native_parameters.profile.species,[hc_type[g] for g in hr['global_cell_ids']],opl_post=[anchor[g] for g in hr['global_cell_ids']]);pops['HC']=hc
    ci={g:i for i,g in enumerate(cones.cells.cell_id)};hi={g:i for i,g in enumerate(hc.cells.cell_id)}
    he=spatial_asset.native_contact_summary(ci,hi) if spatial_asset is not None and hasattr(spatial_asset,'native_contact_summary') else (
        hc_edges.select(ci,hi) if hasattr(hc_edges,'select') else [e for e in hc_edges if e['cone_id'] in ci and e['hc_id'] in hi])
    if spatial_asset is None:
        raw=sparse.csr_matrix((np.ones(len(he)),([hi[e['hc_id']] for e in he],[ci[e['cone_id']] for e in he])),shape=(len(hi),len(ci)))
        rsum=np.asarray(raw.sum(1)).ravel();csum=np.asarray(raw.sum(0)).ravel()
        if np.any(rsum==0) or np.any(csum==0):raise ContractError('Actual atlas ancestor cone-HC component has an empty required native row; no phantom unit-DC input')
        A=sparse.diags(1/rsum)@raw;B=sparse.diags(1/csum)@raw.T
    else:A,B=spatial_asset.native_matrices(cones,hc)
    outer=ComponentSpec('outer_native',KEY,{'cone':'cones','h1':'HC'},PROFILE,
        {'explicit_opt_in':True,'A':A,'B':B,'background_td':q(np.full(len(ci),light.native_background_td),'td','native_retinal_illuminance'),
         'pooling_policy':'pool_Vbar_before_aI','geometry_frame':'OPL','observation_interval_seconds':.0001,
         'unsupported_type_policy':'project_approximation_shared_H1_kinetics'},'cone_native_synapse')
    specs.append(outer);builder=RetinaBuilder(source_registry,max_compile_bytes=execution['budgets']['max_compile_bytes'])
    native=builder.registry.instantiate(outer,pops,source_registry,RunMode.REFERENCE)
    native_baseline=native.observe('release',0,tuple(v.values for v in native.initial()),{}).values
    by['outer_native']=cones;initial_activation={}
    selected={name:_subset(t,needed) for name,t in record['inner_tables'].items() if set(t['global_cell_ids'])&needed}
    for name in ('midget_BC_ON','midget_BC_OFF','diffuse_BC_ON','diffuse_BC_OFF')+(('S_ON_BC',) if typed_completion else ()):
        if allow_empty_owner_outputs and name not in selected:continue
        t=selected[name];pid='effective_'+name;n=len(t['global_cell_ids'])
        contact=record['inner_contacts']['cones->'+name]
        anchors=dict(zip(contact['post_ids'],contact['post_positions']['values']))
        pop=_population(t,'BC',pid,'Macaque_effective_project_profile',opl_post=[anchors[g] for g in t['global_cell_ids']]);pops[name]=pop;by[name]=pop
        W,edges=_matrix(_contact_edges(contact,cones,pop),cones,pop)
        if np.any(np.asarray(W.sum(1)).ravel()==0):raise ContractError('Required atlas BC row empty: '+name)
        profile(pid,n,{'baseline':np.asarray(W@native_baseline).ravel()})
        options={'polarity_representation':'canonical_dominant_positive','presynaptic_inhibition':False,
            'scope':'existing causal BC residual route; physical terminal and rod/AII glycine route explicitly unbound'}
        if name.startswith('midget'):
            original=record['inner_tables'][name]['global_cell_ids'];parents=dict(zip(original,record['inner_parents'][name]));options['parent_cone_ids']=tuple(parents[g] for g in pop.cells.cell_id)
        specs.append(ComponentSpec(name,'ResidualBC_ON' if name.endswith('ON') or name=='S_ON_BC' else 'ResidualBC_OFF',{'cells':name},pid,options,name))
        links.append(_link('outer_native','release',name,'drive',cones,pop,W,'OPL'));binding_edges['cones->'+name]=edges
        initial_activation[name]=np.logaddexp(0.,source_registry.mechanism_profiles[pid].quantity('bias','dimensionless','activation_bias').values)
    for name,pre in (('local_feedforward_AC','diffuse_BC_ON'),('local_crossover_AC','diffuse_BC_OFF'))+(
        (('widefield_AC','diffuse_BC_ON'),) if typed_completion and 'widefield_AC' in selected else ()):
        if allow_empty_owner_outputs and name not in selected:continue
        t=selected[name];pid='effective_'+name;pop=_population(t,'AC',pid,'Macaque_effective_project_profile');pops[name]=pop;by[name]=pop
        W,edges=_matrix(_contact_edges(record['inner_contacts'][pre+'->'+name],pops[pre],pop),pops[pre],pop)
        if np.any(np.asarray(W.sum(1)).ravel()==0):raise ContractError('Required atlas AC row empty: '+name)
        p=profile(pid,len(pop.cells.cell_id),{'baseline':np.asarray(W@initial_activation[pre]).ravel()})
        specs.append(ComponentSpec(name,'ResidualAC',{'cells':name},pid,{'polarity_representation':'stored_signed'},name))
        links.append(_link(pre,'activation',name,'drive',pops[pre],pop,W,'IPL'));binding_edges[pre+'->'+name]=edges
        initial_activation[name]=np.logaddexp(0.,p.quantity('bias','dimensionless','activation_bias').values)
    for ordinal,name in enumerate(execution['requested_outputs']):
        s_route=name=='small_bistratified_S_ON'
        parasol=name.startswith('parasol')
        pre='S_ON_BC' if s_route else ('diffuse_BC_' if parasol else 'midget_BC_')+name.rsplit('_',1)[-1]
        inh='widefield_AC' if parasol else 'local_feedforward_AC' if s_route or name.endswith('OFF') else 'local_crossover_AC'
        t=selected[name];pid='effective_'+name;pop=_population(t,'RGC',pid,'Macaque_effective_project_profile');pops[name]=pop;by[name]=pop;n=len(pop.cells.cell_id)
        totals={}
        for source,receptor in ((pre,'excitatory'),(inh,'inhibitory'))+((('diffuse_BC_OFF','excitatory'),) if s_route else ()):
            W,edges=_matrix(_contact_edges(record['inner_contacts'][source+'->'+name],pops[source],pop),pops[source],pop)
            if np.any(np.asarray(W.sum(1)).ravel()==0):raise ContractError('Required owner atlas RGC '+receptor+' input empty: '+name)
            cid=name+'_'+receptor+'_'+source;gainpid='effective_'+cid;gp=profile(gainpid,n)
            specs.append(ComponentSpec(cid,'SynapticConductance',{'cells':name},gainpid,{'receptor':receptor,'target_population_profile_id':pid}));by[cid]=pop
            links.extend((_link(source,'activation',cid,'activation',pops[source],pop,W,'IPL'),
                _link(cid,'conductance',name,'excitation' if receptor=='excitatory' else 'inhibition',pop,pop,sparse.eye(n,format='csr'),'IPL')))
            binding_edges[source+'->'+name]=edges
            totals[receptor]=totals.get(receptor,0)+gp.quantity('gain','S','synaptic_conductance_scale').values*np.asarray(W@initial_activation[source]).ravel()
        definitions=source_registry.runtime_profile.data['profiles'][pid]
        gL=definitions['gL']['value'];V0=(gL*definitions['EL']['value']+totals['excitatory']*definitions['EE']['value']+
            totals['inhibitory']*definitions['EI']['value'])/(gL+totals['excitatory']+totals['inhibitory'])
        parents=dict(zip(record['inner_tables'][name]['global_cell_ids'],record['inner_parents'][name])) if name.startswith('midget') else {}
        rgc_options={'rng_seed':seed+100+ordinal}
        if parents:rgc_options['parent_bc_ids']=tuple(parents[g] for g in pop.cells.cell_id)
        profile(pid,n,{'V0':V0});specs.append(ComponentSpec(name,'HistoryConductanceRGC',{'cells':name},pid,
            rgc_options,name))
    required_layers=('cone_outer_segment','cone_native_synapse','HC_H1','HC_H2',
        'midget_BC_ON','midget_BC_OFF','diffuse_BC_ON','diffuse_BC_OFF','local_feedforward_AC','local_crossover_AC','midget_RGC_ON','midget_RGC_OFF')+(
        ('S_ON_BC','small_bistratified_S_ON')+(('widefield_AC',) if 'widefield_AC' in pops else ()) if typed_completion else ())+(
        ('parasol_RGC_ON','parasol_RGC_OFF') if revised else ())
    if allow_empty_owner_outputs:required_layers=tuple(layer for layer in required_layers if layer not in record['inner_tables'] or layer in pops)
    spec=ModelSpec(execution['profile_id'],tuple(pops.values()),tuple(specs),tuple(links),(ExternalInput('native_light','outer_native','light'),),required_layers,
        scope=execution['scope'],clock_policy={'kind':'explicit_event_clock','base_step_seconds':.0001,'sampled_output_hold':'zero_order_hold'})
    compiled=builder.build(spec,RunMode.REFERENCE)
    channels={'native_light':_sampled_native_light_channel(duration,light,cones.cells.rf_position.values[:,0])}
    masks={name:_population_owner_mask(p,b) for name,p in pops.items()}
    unavailable={'rod_pathway':{'field':'PackerFourMeridianDensity(rod)','status':record['rod_inner']['metadata']['source_status'],
        'error':record['rod_inner']['metadata']['rod_source_error'],'old_profile':'analytic .020mm-free annulus to4mm peak is a separately labelled old project approximation; cannot pass as new raw Packer fine curve'},
        'AII':'independent geometry retained, source driver unavailable; no unapproved AII resting surrogate',
        'BC_terminal_and_OFF_glycine':'Existing causal ResidualBC ON/OFF is the named partial route; HistoryBipolarTerminal physical terminal and its mandatory AII glycine input are not instantiated. No zero glycine driver or alternative native voltage gain is supplied.',
        'parasol':record.get('typed_completion',{}).get('parasol_blocker','independent canonical seeds not supplied')}
    if not typed_completion:unavailable.update(S_terminal_outputs='not selected in historical midget-only short run; geometry retained',
        widefield_AC='not an ancestor of historical selected midget outputs; geometry retained')
    from .atlas_graph_storage import materialize
    audit={'registry':source_registry,'spec':spec,'light_context':light.manifest(0.),'runtime_profile':source_registry.runtime_profile.manifest(),
        'execution_profile':execution,'resolved_execution_profile':execution,'cone_route':'actual_atlas_native_shared_H1_project',
        'spatial_profile':{'atlas':record['atlas'],'HC_geometry':materialize(record['HC']),'actual_atlas_geometry':True},
        'connectivity_support':{'atlas_graph':record['dependency_query'],'runtime_binding_edges':binding_edges,'finite_graph_exterior_closure':'NOT_CERTIFIED'},
        'population_owner_mask':masks,'resource_estimate':builder.estimate(spec),'rod_inner_geometry':materialize(record['rod_inner']),
        'IPL_geometry':{'source':record['inner_metadata'],'requested_owner_counts':{k:len(v) for k,v in required.items()},'runtime_ancestor_view':True},
        'population_records':{k:p.record(compact=getattr(spatial_asset,'compact_metadata',False)) for k,p in pops.items()},'density_integrals':{'scale':1.,'population_instance':'unchanged atlas, exact dependency view'},
        'regional_support':{'partial_scope':execution['scope'],'unavailable_paths':unavailable},
        'physical_reference_applicability':{'Chen':'outside this declared native-cone route'},'glycine_source_condition':{'not_in_scope':'rod/AII glycine omitted explicitly due source driver gap'},
        'atlas_runtime':{'graph_binding':graph_binding,'source_atlas':record['atlas'],
            'ancestor_population_counts':{k:len(p.cells.cell_id) for k,p in pops.items()},
            'HC_edges':he,'HC_A_row_gain':'unit DC of actual eligible edges','HC_B_row_gain':'unit DC of same reciprocal edges',
            'global_tissue_thinned':False,'geometry_or_edge_resampled':False,'unsupported_paths':unavailable,
            'full_target_readiness':'BLOCKED','typed_completion_enabled':typed_completion,
            'original_finite_graph_counts':{'cones':len(record['cones']['global_cell_ids']),'HC':len(record['HC']['table']['global_cell_ids'])},
            'exterior_support_only_cones_not_dynamical':len(record['cones']['global_cell_ids'])-len(ci),
            'HC_empty_direct_input_geometry_only_at_gap0':len(record['HC']['table']['global_cell_ids'])-len(hi),
            'unexecuted_geometry_population_counts':{name:len(t['global_cell_ids'])-len(pops[name].cells.cell_id) if name in pops else len(t['global_cell_ids'])
                for name,t in record['inner_tables'].items()},
            'full_20_layer_network_claim':False,'background_steady_state_claim':False,'positive_gap':False,'passive_HC_instantiated':0}}
    audit['atlas_runtime']['empty_owner_outputs']={name:'EMPTY_SAMPLE; full geometry retained; not biological absence' for name in empty_owner_outputs}
    audit['atlas_runtime']['owner_target_contract']={
        'canonical_outputs':list(canonical_targets),'active_owner_outputs':list(execution['requested_outputs']),
        'status':{name:'EMPTY_SAMPLE' if name in empty_owner_outputs else 'INSTANTIATED_OWNER_PATH' for name in canonical_targets},
        'empty_is_biological_absence':False,'synthetic_cells_added':False,
        'empty_target_geometry_retained':True,'all_canonical_owner_paths_executed':not empty_owner_outputs}
    return compiled,channels,audit


def atlas_typed_candidate_network(root,**options):
    """All source-supported owner typed cone paths; full-target blockers exported."""
    return atlas_midget_network(root,typed_completion=True,**options)
