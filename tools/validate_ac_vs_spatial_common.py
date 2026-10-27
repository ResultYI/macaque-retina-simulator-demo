"""One0.6 equilibrium, exact ID projection and nested-boundary spatial trial."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import shutil
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.ac_vs_model import ACVSAssembly,functional_patch,load_config
from retina_sim.ac_vs_background import qualify_background
from retina_sim.ac_vs_monitor import DynamicDefectMonitor
from retina_sim.group_inner_dynamics import CONTACTS
from retina_sim.group_time_monitor import GroupTimeMonitor,frozen_budget,rhs_allowance
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.group_time_storage import load_bundle,save_bundle
from retina_sim.local_patch_boundary import BoundedNativePhysics,BoundedInnerDynamics,configure_boundary_runtime
from retina_sim.spatial_group_model import RGC
from retina_sim.background_checkpoint import MemoryGuard
from tools.validate_ac_vs import Worker,QUALIFICATION,write,dynamic_observe,causal_hash
from tools.validate_ac_vs_equilibrium_start import verify_sources,initial_match
from tools.validate_ac_vs_timescale import calendar,CoreRecorder,sparse_states

OUT=ROOT/'reports/ac_vs_spatial_common_v1';REG=OUT/'pre_registration_v1.json'
SCHEMA='ac-vs-common-seed-spatial-1'
CELL_PARAMETERS={'V0','baseline','reference_input'}
KERNEL_PARAMETERS={'kernel','spike_history_kernel'}
PRODUCTS=('A_Vbar','A_release','B_Vh','gap_product','gap_missing_conductance_S')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    from retina_sim.spatial_group_graph import digest
    return digest(path)


def by_ids(source,target,role):
    source=list(map(str,source));target=list(map(str,target))
    if len(set(source))!=len(source) or len(set(target))!=len(target):raise ContractError('Nonunique global IDs: '+role)
    lookup={v:i for i,v in enumerate(source)}
    missing=[v for v in target if v not in lookup]
    if missing:raise ContractError('Small IDs absent from0.6: '+role+' '+str(missing[:3]))
    return np.array([lookup[v] for v in target],dtype=np.int64)


def assembly(halo):
    directory=ROOT/('reports/ac_vs_redesign_v1/functional_static_halo_'+halo+'_v1')
    if not directory.exists():raise ContractError('Existing frozen geometry required')
    config=read(ROOT/('configurations/local_patch_v2/halo_'+halo+'.json'))
    patch=functional_patch(ROOT,config,directory,lambda:None)
    patch.runtime_factories=dict(inner=BoundedInnerDynamics,native=BoundedNativePhysics)
    return ACVSAssembly(patch,load_config(ROOT/'configurations/ac_vs_functional_v1.json'),max_step=.0001)


def contact_ids(a,component):
    return [component+':parent-contact-group:'+str(int(i)) for i in a.parent_contact_rows[component]]


def mappings(small,large):
    if small.base.artifact_sha256!=large.base.artifact_sha256:raise ContractError('Different global parent artifacts')
    pops={}
    for pop,p in small.populations.items():
        q=large.populations[pop];rows=by_ids(q.group_ids,p.group_ids,pop)
        if not np.array_equal(p.counts,q.counts[rows]) or not np.array_equal(p.centroid_mm,q.centroid_mm[rows]):
            raise ContractError('Shared group membership/coordinate differs: '+pop)
        pops[pop]=rows
    for pop in RGC:
        if not np.array_equal(small.populations[pop].group_ids,large.populations[pop].group_ids):
            raise ContractError('RGC global draw order differs; cannot reset/remap RNG: '+pop)
    contacts={c:by_ids(contact_ids(large,c),contact_ids(small,c),c) for c in CONTACTS}
    for c,rows in contacts.items():
        rs=next(r for r in small.routes.values() if r.contact_component==c)
        rl=next(r for r in large.routes.values() if r.contact_component==c)
        for label,pop in [('pre',rs.pre),('post',rs.post)]:
            si=small.array('routes/'+rs.name+'/contact_'+label+'_group.npy')
            li=large.array('routes/'+rl.name+'/contact_'+label+'_group.npy')[rows]
            if not np.array_equal(small.populations[pop].group_ids[si],large.populations[pop].group_ids[li]):
                raise ContractError('Global contact endpoint differs: '+c+':'+label)
        sp,ss=small.array('routes/'+rs.name+'/contact_pre_group.npy'),small.array('routes/'+rs.name+'/contact_post_group.npy')
        lp,ls=large.array('routes/'+rl.name+'/contact_pre_group.npy')[rows],large.array('routes/'+rl.name+'/contact_post_group.npy')[rows]
        sc=np.asarray(small.matrix('routes/'+rs.name+'/contact_coefficient_S.npz')[ss,sp]).ravel()
        lc=np.asarray(large.matrix('routes/'+rl.name+'/contact_coefficient_S.npz')[ls,lp]).ravel()
        if not np.array_equal(sc,lc):raise ContractError('Shared physical contact coefficient differs: '+c)
    aux=by_ids(large.populations['cones'].recording_auxiliary_fine_rows,small.populations['cones'].recording_auxiliary_fine_rows,'original auxiliary cone IDs')
    return dict(populations=pops,contacts=contacts,auxiliary=aux,
        identity=canonical_hash(dict(populations=pops,contacts=contacts,auxiliary=aux,parent=small.base.artifact_sha256)))


def state_rows(key,maps):
    return maps['contacts'][key[1]] if key[1] in CONTACTS else maps['auxiliary'] if key[1]=='selected_original_z1_auxiliary' else maps['populations'][key[0]]


def project_seed(source,small,large,maps):
    original=canonical_hash(source);result=deepcopy(source)
    expected={(b.population,b.mechanism,b.name):b for b in (*small.state_blocks,*small.probability_blocks,*small.auxiliary_z1_blocks)}
    if set(source['states'])!=set(expected):raise ContractError('Different source/small state registries')
    result['states']={k:np.asarray(v)[state_rows(k,maps)].copy() for k,v in source['states'].items()}
    for k,b in expected.items():
        if result['states'][k].shape!=(b.count,):raise ContractError('Projected state length differs: '+str(k))
    parameters=result['inner_parameters']
    for pop,fields in parameters['populations'].items():
        for name,value in list(fields.items()):
            if isinstance(value,np.ndarray):
                if name in CELL_PARAMETERS:
                    if value.shape!=(large.populations[pop].group_count,):raise ContractError('Unbound per-group parameter: '+pop+'.'+name)
                    fields[name]=value[maps['populations'][pop]].copy()
                elif name not in KERNEL_PARAMETERS:raise ContractError('Unregistered parameter array: '+pop+'.'+name)
    for c,fields in parameters['contacts'].items():
        for name,value in list(fields.items()):
            if isinstance(value,np.ndarray):
                if name!='tau_decay' or c not in ('AII_FMB_glycine','AII_DB1_glycine'):raise ContractError('Unregistered contact parameter: '+c+'.'+name)
                fields[name]=value[maps['contacts'][c]].copy()
    for route,value in parameters['route_gains'].items():
        if not np.isscalar(value):raise ContractError('Unregistered route gain vector; explicit projection required')
    parameters['identity']=canonical_hash(dict(schema=SCHEMA,parent_parameters=source['inner_parameters']['identity'],map=maps['identity']))
    parameters['provenance']=dict(source=deepcopy(source['inner_parameters'].get('provenance')),exact_projection_from_060=True,map_identity=maps['identity'])
    for pop,spike in result['spike_states'].items():
        rows=maps['populations'][pop];original_spike=source['spike_states'][pop]
        if original_spike['pending_events'] or original_spike['last_bin_counts'] is not None:raise ContractError('Uncommitted common equilibrium sampler required')
        spike['history']=original_spike['history'][:,rows].copy()
        spike['conditional_rate_Hz']=original_spike['conditional_rate_Hz'][rows].copy()
        if canonical_hash(spike['rng_state'])!=canonical_hash(original_spike['rng_state']):raise ContractError('Projection changed PCG64 state')
    h=result['observation_prehistory'];original_h=source['observation_prehistory'];rows=maps['populations']['HC']
    h['history_values']=original_h['history_values'][:,rows].copy();h['prehistory']=original_h['prehistory'][rows].copy()
    h['prehistory_kind']='exact restriction of the common0.6 conditional-equilibrium HC history'
    for category,pop in [('native_inputs','cones'),('rod_inputs','rods')]:
        result[category]={i:v[maps['populations'][pop]].copy() for i,v in source[category].items()}
    result['input_reference']['native_background_td']=source['input_reference']['native_background_td'][maps['populations']['cones']].copy()
    result['patch_initialization']=dict(schema=SCHEMA,source_sha256=deepcopy(source['patch_initialization']['source_sha256']),
        source_060_payload_sha256=original,map_identity=maps['identity'],population_rows=deepcopy(small.parent_population_rows),
        contact_rows=deepcopy(small.parent_contact_rows),cone_auxiliary_rows=deepcopy(small.parent_auxiliary_rows['cones']),
        patch_artifact_sha256=small.artifact_sha256,classification='EXACT_COMMON_EQUILIBRIUM_ID_RESTRICTION_NO_RNG_RESET')
    result['ac_vs_lineage']=dict(schema=SCHEMA,artifact=small.artifact_sha256,common_060_payload_sha256=original,
        saved_background_reused=True,RNG_copied_without_draw=True,small_equilibrium_solved=False,whole_retina_background_qualified=False)
    if canonical_hash(source)!=original:raise ContractError('Common source mutated by projection')
    return result


def nested(matrix_small,matrix_large,posts,pres,role):
    expected=matrix_large[posts][:,pres].tocsr();difference=matrix_small-expected;difference.eliminate_zeros()
    if difference.nnz:raise ContractError('Nested operator changed or renormalized: '+role)


def annulus_product(matrix,posts,pres,values):
    mask=np.ones(matrix.shape[1],dtype=bool);mask[pres]=False
    return np.asarray(matrix[posts][:,mask]@values[mask]).ravel()


def common_boundary(small,large,source,maps,release_large):
    c,h=maps['populations']['cones'],maps['populations']['HC'];old=large.boundary_runtime
    A=large.matrix('routes/cones_to_HC/mean_weights.npz');B=large.matrix('routes/HC_to_cones/mean_weights.npz')
    nested(small.matrix('routes/cones_to_HC/mean_weights.npz'),A,h,c,'native A')
    nested(small.matrix('routes/HC_to_cones/mean_weights.npz'),B,c,h,'native B')
    gapnames=large.manifest['gap']['population_order']
    if gapnames!=small.manifest['gap']['population_order']:raise ContractError('Gap population order differs')
    offsets={};at=0
    for pop in gapnames:offsets[pop]=at;at+=large.populations[pop].group_count
    gapmap=np.concatenate([maps['populations'][pop]+offsets[pop] for pop in gapnames])
    L=large.matrix('gap/laplacian_S.npz');nested(small.matrix('gap/laplacian_S.npz'),L,gapmap,gapmap,'gap Laplacian')
    V=np.concatenate([source['states'][pop,'ReceptorConductanceMembrane' if pop=='AII_AC' else 'HistoryBipolarTerminal','V'] for pop in gapnames])
    values=dict(A_Vbar=old['A_Vbar'][h]+annulus_product(A,h,c,source['states']['cones','reduced_native','Vbar']),
        A_release=old['A_release'][h]+annulus_product(A,h,c,release_large),
        B_Vh=old['B_Vh'][c]+annulus_product(B,c,h,source['states']['HC','reduced_native','Vh']),
        gap_product=old['gap_product'][gapmap]+annulus_product(L,gapmap,gapmap,V),
        gap_missing_conductance_S=old['gap_missing_conductance_S'][gapmap]-annulus_product(L,gapmap,gapmap,np.ones(at)))
    expected=-np.asarray(small.external_matrix('gap/laplacian_S.npz').sum(axis=1)).ravel()
    scale=float(np.max(np.abs(small.matrix('gap/laplacian_S.npz').data),initial=0.))
    if np.max(np.abs(values['gap_missing_conductance_S']-expected),initial=0.)>1e-12*scale:
        raise ContractError('Original gap row-conservation contract differs')
    descriptor=dict(schema=SCHEMA,classification='COMMON_060_EQUILIBRIUM_ANNULUS_PLUS_ORIGINAL_OUTSIDE_060_RESERVOIR',
        source_060_payload_sha256=canonical_hash(source),source_time_seconds=source['time_seconds'],map_identity=maps['identity'],
        outer_reservoir_identity=old['identity'],boundary_products_sha256=canonical_hash(values),
        background_qualified=False,arbitrary_dynamic_exterior_input_supported=False,weights_renormalized=False,
        native_pooling='linear exterior products inserted before the original adaptation power',
        annulus='large retained rows minus small retained rows; held at the common equilibrium')
    identity=canonical_hash(descriptor)
    return dict(values,identity=identity),dict(descriptor,identity=identity)


def setup_original_boundary(a):
    binding=a.patch.config['input_identity'];path=ROOT/binding['path']
    if sha(path)!=binding['sha256']:raise ContractError('Original outer-reservoir source changed')
    parent=load_bundle(path.parent,expected_kind='source_construction_seed')['payload']
    configure_boundary_runtime(a.patch,parent,{})


def preflight(output):
    small,large=assembly('030'),assembly('060');maps=mappings(small,large)
    raw=load_bundle((ROOT/read(REG)['construction_060']).parent,expected_kind='source_construction_seed')
    if raw['identity']['artifact']!=large.artifact_sha256 or raw['identity']['seed']!=canonical_hash(raw['payload']):raise ContractError('Frozen large source changed')
    verify_sources(raw['identity']['sources']);verify_sources(raw['payload']['patch_initialization']['source_sha256'])
    counts={label:sum(b.count for b in (*a.state_blocks,*a.probability_blocks,*a.auxiliary_z1_blocks)) for label,a in [('030',small),('060',large)]}
    memory=MemoryGuard(16*1024**3,4*1024**3);healthy=memory.check();disk=shutil.disk_usage(ROOT)
    prior=read(ROOT/'reports/ac_vs_timescale_v1/coarse_v2/resource_receipt.json')
    projected_peak=int(prior['peak_sampled_RSS_bytes']*counts['060']/counts['030']+512*1024**2)
    projected_disk=256*1024**2
    feasible=healthy and memory.last['available_physical_bytes']>4*1024**3+projected_peak and projected_peak<16*1024**3 and disk.free>2*projected_disk
    result=dict(passed=bool(feasible),status='PREFLIGHT_ONLY_NO_SOLVE_OR_ADVANCE',state_scalar_counts=counts,
        mapping_counts={p:len(rows) for p,rows in maps['populations'].items()},map_identity=maps['identity'],
        actual_memory=memory.last,actual_disk_free_bytes=disk.free,projected_peak_RSS_bytes=projected_peak,
        projected_disk_bytes=projected_disk,large_dynamic_wall_projection_seconds=prior['elapsed_seconds']*counts['060']/counts['030'],
        model_limits=read(ROOT/'configurations/ac_vs_acceptance_v1.json')['resources'],
        registration_sha256=sha(REG),adapter_sha256=sha(Path(__file__)),separate_large_equilibrium_solves_planned=1,small_equilibrium_solves_planned=0)
    write(output/'MEMORY_AND_MAPPING_PREFLIGHT.json',result)
    if not feasible:raise ContractError('Common-spatial memory/disk preflight failed; no solver started')
    return result


def background(output):
    with Worker(output) as worker:
        small,large=assembly('030'),assembly('060');maps=mappings(small,large)
        raw=load_bundle((ROOT/read(REG)['construction_060']).parent,expected_kind='source_construction_seed')
        if raw['identity']['artifact']!=large.artifact_sha256 or raw['identity']['seed']!=canonical_hash(raw['payload']):raise ContractError('Frozen0.6 source IVP differs')
        verify_sources(raw['identity']['sources']);verify_sources(raw['payload']['patch_initialization']['source_sha256'])
        setup_original_boundary(large)
        write(output/'solve_attempt.json',dict(large_equilibrium_solver_invocations=1,small_equilibrium_solver_invocations=0,
            global_mapping_identity=maps['identity'],sources=raw['identity']['sources'],registration_sha256=sha(REG)))
        result=qualify_background(large,raw['payload'],output,worker.check)
        save_bundle(output/'original_boundary_products',kind='validation_evidence',identity=dict(artifact=large.artifact_sha256,source=raw['manifest_hash']),
            payload=deepcopy(large.boundary_runtime),qualification=QUALIFICATION)
        return result


def spatial_stimulus(seed,a,registration):
    result=deepcopy(seed);spec=read(ROOT/registration['input']['global_cone_ids_from']);ids=spec['mask_group_ids']
    rows=by_ids(a.populations['cones'].group_ids,ids,'frozen global stimulus mask')
    if len(rows)!=15:raise ContractError('Different global stimulus mask count')
    mask=np.zeros(a.populations['cones'].group_count,dtype=bool);mask[rows]=True
    B=seed['input_reference']['native_background_td'];rod=np.full(a.populations['rods'].group_count,seed['input_reference']['rod_background_rstar_s'])
    frames=range(748,765)
    result['native_inputs']={i:B*(1+.05*mask) if 750<=i<759 else B.copy() for i in frames}
    result['rod_inputs']={i:rod.copy() for i in frames}
    return result,dict(global_mask_ids=ids,global_mask_sha256=canonical_hash(ids),local_rows_resolved_from_ids=rows.tolist(),
        native_frame_hashes={str(i):canonical_hash(v) for i,v in result['native_inputs'].items()},native_unit='td',rod_unit='R*/s')


def core_selection(a,seed,snapshot):
    contacts={r.contact_component:r for r in a.routes.values() if r.contact_component};core=a.core_population_rows
    crows={c:np.flatnonzero(np.isin(a.array('routes/'+r.name+'/contact_post_group.npy'),core[r.post])) for c,r in contacts.items()}
    fields={};units={':'.join((b.population,b.mechanism,b.name)):b.unit for b in (*a.state_blocks,*a.probability_blocks,*a.auxiliary_z1_blocks)}
    for key in seed['states']:
        pop,mech,name=key
        rows=crows[mech] if mech in CONTACTS else np.empty(0,dtype=int) if mech=='selected_original_z1_auxiliary' else core.get(pop,np.empty(0,dtype=int))
        ids=np.array(contact_ids(a,mech))[rows].tolist() if mech in CONTACTS else [] if mech=='selected_original_z1_auxiliary' else np.asarray(a.populations[pop].group_ids)[rows].tolist()
        fields['states__'+':'.join(key)]=dict(category='states',key=list(key),rows=np.asarray(rows,dtype=int).tolist(),ids=ids,unit=units[':'.join(key)],spatial_comparable=True)
    for key,values in snapshot['ports'].items():
        pop,name=key;comparable=True
        if pop=='cones' and name=='z1_aux':rows=np.empty(0,dtype=int);ids=[]
        elif pop in contacts:
            rows=crows[pop] if name=='contact_conductance' else core[contacts[pop].post]
            ids=np.array(contact_ids(a,pop))[rows].tolist() if name=='contact_conductance' else np.asarray(a.populations[contacts[pop].post].group_ids)[rows].tolist()
        elif pop=='local_patch_boundary':rows=np.array([0]);ids=['whole-boundary:'+a.artifact_sha256];comparable=False
        else:rows=core.get(pop,np.empty(0,dtype=int));ids=np.asarray(a.populations[pop].group_ids)[rows].tolist()
        if max(rows,default=-1)>=len(values):raise ContractError('Semantic port row layout differs: '+str(key))
        fields['ports__'+':'.join(key)]=dict(category='ports',key=list(key),rows=np.asarray(rows,dtype=int).tolist(),ids=ids,unit=snapshot['port_units'][key],spatial_comparable=comparable)
    return dict(fields=fields,selection_hash=canonical_hash(fields))


def point_check(small,large,ss,ls,maps,protocol,guard):
    arms={};snapshots={};monitors={};initial={}
    try:
        for label,a,seed in [('030',small,ss),('060',large,ls)]:
            arm=GroupTimeArm(a,seed,protocol,mode='reference',guard=guard);arms[label]=arm
            initial[label],_=initial_match(arm,seed)
            snap=arm.initial_snapshot();snapshots[label]=snap
            m=GroupTimeMonitor(a,seed['states'],parameters=seed['inner_parameters'])
            m.observe(seed['time_seconds'],'right',snap['states'],snap['rhs'],transitions=snap['transitions'],ports=snap['ports'],aux=snap['aux'])
            monitors[label]=m.report(.1)
        s,l=snapshots['030'],snapshots['060'];rows=[];exact=[]
        blocks={ (b.population,b.mechanism,b.name):b for b in (*small.state_blocks,*small.probability_blocks,*small.auxiliary_z1_blocks)}
        for key,b in blocks.items():
            select=state_rows(key,maps);reference=l['states'][key][select];value=s['states'][key]
            if not np.array_equal(value,reference):raise ContractError('Common initial state differs: '+str(key))
            exact.append(':'.join(key));scale,_,budget=frozen_budget(value,b.unit)
            rf=l['rhs'][key][select];sf=s['rhs'][key];allowed=rhs_allowance(scale,budget,rf,sf)
            norm=float(np.max(np.abs(sf-rf)/allowed,initial=0.))
            rows.append(dict(state=':'.join(key),RHS_maximum_norm=norm,passed=norm<=1))
        transitions=[]
        for key,value in s['transitions'].items():
            select=state_rows(key,maps);other=l['transitions'][key][select]
            budget=frozen_budget(s['states'][key],blocks[key].unit)[2]
            bound=budget+8192*np.finfo(float).eps*np.maximum(abs(value),abs(other))
            norm=float(np.max(abs(value-other)/bound,initial=0.));transitions.append(dict(state=':'.join(key),maximum_norm=norm,passed=norm<=1))
        selections={k:core_selection(small if k=='030' else large,ss if k=='030' else ls,v) for k,v in snapshots.items()}
        for name,f in selections['030']['fields'].items():
            g=selections['060']['fields'][name]
            if f['spatial_comparable'] and set(f['ids'])!=set(g['ids']):raise ContractError('Core global IDs differ before advance: '+name)
        history=dict(sampled_history_exact={p:canonical_hash(ss['spike_states'][p])==canonical_hash(ls['spike_states'][p]) for p in RGC},
            HC_times_exact=np.array_equal(ss['observation_prehistory']['history_times'],ls['observation_prehistory']['history_times']),
            HC_values_exact=np.array_equal(ss['observation_prehistory']['history_values'],ls['observation_prehistory']['history_values'][:,maps['populations']['HC']]))
        passed=all(r['passed'] for r in rows+transitions) and all(r['passed'] for r in monitors.values()) and all(history['sampled_history_exact'].values()) and history['HC_times_exact'] and history['HC_values_exact']
        result=dict(passed=passed,initial_matches=initial,common_exact_state_blocks=exact,RHS_comparisons=rows,transition_comparisons=transitions,
            equilibrium_and_domain=monitors,history_projection=history,RNG_redrawn_or_reset=False,small_equilibrium_solved=False)
        return result,selections
    finally:
        for arm in arms.values():arm.close()


def project(background_dir,output):
    reg=read(REG)
    if reg['spatial_budget']!=read(ROOT/'configurations/ac_vs_acceptance_v1.json')['spatial_budget']:raise ContractError('Original spatial thresholds changed')
    with Worker(output) as worker:
        small,large=assembly('030'),assembly('060');maps=mappings(small,large)
        bg=Path(background_dir).resolve();receipt=read(bg/'result.json')
        if receipt.get('passed') is not True:raise ContractError('The one0.6 equilibrium did not qualify')
        bundle=load_bundle(bg/'candidate_constant_background',expected_kind='source_construction_seed')
        source=bundle['payload'];verify_sources(bundle['identity']['sources']);verify_sources(source['patch_initialization']['source_sha256'])
        if bundle['identity']['artifact']!=large.artifact_sha256 or canonical_hash(source)!=receipt['seed_sha256']:raise ContractError('Common0.6 source identity differs')
        large.boundary_runtime=load_bundle(bg/'original_boundary_products',expected_kind='validation_evidence')['payload']
        if large.boundary_runtime['identity']!=source['patch_boundary']['identity']:raise ContractError('Original0.6 reservoir identity differs')
        native=BoundedNativePhysics(large)
        y,anchor=native.construction_point(SimpleNamespace(states=source['states'],time_seconds=source['time_seconds']),source['input_reference']['native_background_td'],5.,748)
        release=native.evaluate(y,source['time_seconds'],anchor).outputs['release']
        projected=project_seed(source,small,large,maps)
        products,descriptor=common_boundary(small,large,source,maps,release)
        small.boundary_runtime=products;projected['patch_boundary']=descriptor
        sseed,sinput=spatial_stimulus(projected,small,reg);lseed,linput=spatial_stimulus(source,large,reg)
        protocol=deepcopy(read(ROOT/'reports/ac_vs_timescale_v1/prepared_v2/protocol.json'))
        protocol['spatial_common_registration_sha256']=sha(REG);protocol['initialization']='one common0.6 conditional equilibrium; exact ID restriction and nested fixed boundary for0.3'
        protocol['equilibrium_start_stimulus_identity']=canonical_hash(dict(global_ids=sinput['global_mask_ids'],registration=reg['input']))
        check,selections=point_check(small,large,sseed,lseed,maps,protocol,worker.check)
        write(output/'COMMON_INITIAL_AND_BOUNDARY_CHECK.json',check)
        if not check['passed']:raise ContractError('Common initial state/boundary algebra or equilibrium failed before dynamic advance')
        pins=dict(bundle['identity']['sources']);pins.update(source['patch_initialization']['source_sha256']);pins.update({str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [Path(__file__),REG,ROOT/'tools/validate_ac_vs_timescale.py',ROOT/'tools/validate_ac_vs_equilibrium_start.py',ROOT/'retina_sim/local_patch_boundary.py']})
        common=dict(schema=SCHEMA,registration=reg,registration_sha256=sha(REG),sources=pins,common_background_manifest=bundle['manifest_hash'],
            common_background_payload=canonical_hash(source),map_identity=maps['identity'],protocol=protocol,
            local_equilibrium_solves=dict(large_060=1,small_030=0),projected_without_RNG_reset=True,FIT_READY=False)
        for label,a,seed,selection_,input_ in [('030',small,sseed,selections['030'],sinput),('060',large,lseed,selections['060'],linput)]:
            identity=dict(common=common,halo=label,artifact=a.artifact_sha256,seed_sha256=canonical_hash(seed),selection=selection_,input=input_,boundary=seed['patch_boundary'])
            save_bundle(output/('seed_'+label),kind='source_construction_seed',identity=identity,payload=seed,qualification=QUALIFICATION)
            save_bundle(output/('boundary_'+label),kind='validation_evidence',identity=identity,payload=deepcopy(a.boundary_runtime),qualification=QUALIFICATION)
        write(output/'preparation.json',dict(status='COMMON_SEED_BOUNDARY_QUALIFIED_BEFORE_ADVANCE',common=common,mapping_counts={p:len(v) for p,v in maps['populations'].items()},
            history_RNG_projected_exactly=True,all_120_state_blocks_projected=True,real_dynamic_advance_started=False))
        return dict(status='COMMON_SEED_BOUNDARY_QUALIFIED_BEFORE_ADVANCE',passed=True)


def load_arm(prepared,halo):
    p=Path(prepared).resolve();bundle=load_bundle(p/('seed_'+halo),expected_kind='source_construction_seed');identity=bundle['identity']
    verify_sources(identity['common']['sources']);a=assembly(halo)
    if a.artifact_sha256!=identity['artifact'] or canonical_hash(bundle['payload'])!=identity['seed_sha256']:raise ContractError('Frozen spatial arm changed')
    a.boundary_runtime=load_bundle(p/('boundary_'+halo),expected_kind='validation_evidence',expected_identity=identity)['payload']
    if a.boundary_runtime['identity']!=bundle['payload']['patch_boundary']['identity']:raise ContractError('Spatial arm boundary identity differs')
    return a,bundle['payload'],identity


def run(prepared,halo,output):
    a,seed,identity=load_arm(prepared,halo);protocol=identity['common']['protocol']
    with Worker(output) as worker:
        arm=GroupTimeArm(a,seed,protocol,mode='reference',guard=worker.check)
        monitor=GroupTimeMonitor(a,seed['states'],parameters=seed['inner_parameters']);dynamic=DynamicDefectMonitor(a,seed['states']);recorder=CoreRecorder(identity['selection']);edges=[]
        write(output/'model_identity.json',identity)
        def observe(t,side,record=False):
            s=arm.snapshot(t,side,queue_node=True)
            monitor.observe(t,side,s['states'],s['rhs'],transitions=s['transitions'],ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
            if monitor.domain_failures:raise ContractError('Spatial runtime domain/history/conservation failed')
            if record:recorder.record(s)
            return s
        try:
            init,_=initial_match(arm,seed);write(output/'initial_seed_match.json',init)
            observe(arm.start,'right');arm.initial_commit();observe(arm.start,'right',True);arm.ack(arm.start)
            for end,flags in calendar(protocol):
                if end<=arm.inner_time:continue
                begin=arm.inner_time;arm.ensure_native(end,'left');advance=arm.advance_inner(begin,end)
                for curve in advance.dense_segments:
                    mid=curve.t0+(curve.t1-curve.t0)/2;side='left' if mid==advance.t1 else 'right'
                    witness=dynamic_observe(arm,advance,mid,dynamic,side)
                    if max(r['maximum_norm'] for r in dynamic.rows.values())>1:
                        save_bundle(output/'failed_dynamic_witness',kind='validation_evidence',identity=identity,
                            payload=dict(states=witness['states'],rhs=witness['rhs'],time_seconds=mid,side=side,dynamic=dynamic.report()),qualification=QUALIFICATION)
                        raise ContractError('Original dynamic defect budget failed')
                before=observe(end,'left' if flags['frame'] else 'right',flags['frame'])
                if flags['rod'] or flags['bc']:arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],rod_sample_time=max(arm.start,end-.0001+1e-13) if flags['rod'] else None)
                last=observe(end,'right',flags['record'] or flags['frame'])
                if flags['frame']:
                    keys=[k for k in last['states'] if k[1] not in ('Chen2024PrimateRodCurrent','RGC_expected_history') and not(k[1]=='HistoryBipolarTerminal' and(k[2].startswith('history_') or k[2]=='activation'))]
                    edges.append(dict(time_seconds=end,continuous_state_max_jump=max((float(np.max(abs(last['states'][k]-before['states'][k]),initial=0.)) for k in keys),default=0.)))
                arm.ack(end)
                if end==protocol['checkpoint_seconds']:sparse_states(output/'full_state_mid.npz',last)
            sparse_states(output/'full_state_final.npz',last);recorder.save(output/'core_traces.npz')
            payload=arm.checkpoint_payload(monitor=monitor.payload(),recording=recorder.cursor)
            save_bundle(output/'final_checkpoint',kind='complete_model_checkpoint',identity=identity,payload=payload,qualification=QUALIFICATION)
            clock_ok=payload['clocks']['indices']==dict(rod=50930,bc=5093,rgc=5093)
            result=dict(status='COMMON_BACKGROUND_SPATIAL_ARM_COMPLETE',passed=dynamic.report()['passed'] and clock_ok and all(e['continuous_state_max_jump']==0 for e in edges),
                halo=halo,start_seconds=arm.start,end_seconds=arm.inner_time,actual_simulated_seconds=arm.inner_time-arm.start,
                initial_match=init,DYNAMIC_NUMERICAL_DEFECT=dynamic.report(),PHYSICAL_DOMAIN_AND_CONSERVATION=dict(passed=not monitor.domain_failures,failures=monitor.domain_failures),
                INPUT_EVENT_AND_HISTORY=dict(passed=clock_ok and all(e['continuous_state_max_jump']==0 for e in edges),edges=edges,clocks=payload['clocks']),
                raw_absolute_equilibrium_monitor=monitor.report(.1),metrics=arm.metrics,final_causal_hash=causal_hash(payload),core_records=recorder.cursor['records'],
                saved_common_060_background_used=True,old_030_run_reused=False,FIT_READY=False,whole_retina_or_arbitrary_exterior_background_qualified=False)
            write(output/'result.json',result);return result
        except Exception as exc:
            if recorder.stamps:recorder.save(output/'stopped_core_traces.npz')
            write(output/'failure_witness.json',dict(error=type(exc).__name__,message=str(exc),time_seconds=arm.inner_time,dynamic=dynamic.report(),metrics=arm.metrics))
            raise
        finally:arm.close()


def compare(small_dir,large_dir,output):
    s,l=Path(small_dir),Path(large_dir);si,li=read(s/'model_identity.json'),read(l/'model_identity.json')
    if si['common']!=li['common']:raise ContractError('Spatial arms have different common background/protocol')
    budget=si['common']['registration']['spatial_budget'];rows=[];excluded=[]
    with np.load(s/'core_traces.npz',allow_pickle=False) as a,np.load(l/'core_traces.npz',allow_pickle=False) as b:
        for name in ('time_seconds','side','generation'):np.testing.assert_array_equal(a[name],b[name])
        for name,f in si['selection']['fields'].items():
            g=li['selection']['fields'][name]
            if not f['spatial_comparable']:excluded.append(dict(field=name,reason='different whole-boundary domains; independent conservation gate retained'));continue
            mapping=by_ids(g['ids'],f['ids'],name)
            if len(mapping)!=len(g['ids']):raise ContractError('Unmatched core IDs: '+name)
            x,y=a[name],b[name][:,mapping];unit=f['unit']
            allowance=budget['voltage_V'] if unit=='V' else budget['conductance_S'] if unit=='S' else budget['expected_rate_Hz'] if unit=='Hz' else budget['general_atol']+budget['general_rtol']*np.maximum(abs(x),abs(y))
            norms=abs(x-y)/allowance;maximum=float(np.max(norms,initial=0.))
            row=dict(field=name,unit=unit,global_ids=f['ids'],samples=len(x),maximum_norm=maximum,max_abs=float(np.max(abs(x-y),initial=0.)),passed=maximum<=1)
            if norms.size:
                t,j=np.unravel_index(int(np.argmax(norms)),norms.shape);row.update(worst_time_seconds=float(a['time_seconds'][t]),worst_side=str(a['side'][t]),worst_global_id=f['ids'][j],small_value=float(x[t,j]),large_value=float(y[t,j]))
            rows.append(row)
    passed=all(r['passed'] for r in rows);worst=max(rows,key=lambda r:r['maximum_norm'])
    result=dict(status='COMMON_BACKGROUND_100MS_SPATIAL_PASS' if passed else 'COMMON_BACKGROUND_100MS_SPATIAL_FAIL',passed=passed,
        maximum_norm=worst['maximum_norm'],worst=worst,per_field=rows,pre_registered_excluded_noncore_flux=excluded,
        common_seed_and_boundary_checked_before_runs=True,thresholds=budget,thresholds_changed=False,
        old_030_run_reused=False,FIT_READY=False,whole_retina_or_arbitrary_exterior_background_qualified=False)
    write(output/'result.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('preflight','background','project','run','compare'));p.add_argument('--output',required=True)
    p.add_argument('--background');p.add_argument('--prepared');p.add_argument('--halo',choices=('030','060'));p.add_argument('--small');p.add_argument('--large')
    args=p.parse_args();output=Path(args.output).resolve()
    if not output.is_relative_to(OUT) or output.exists():raise ContractError('Fresh independent common-spatial output required')
    output.mkdir(parents=True)
    try:
        if args.mode=='preflight':result=preflight(output)
        elif args.mode=='background':result=background(output)
        elif args.mode=='project':result=project(args.background,output)
        elif args.mode=='run':result=run(args.prepared,args.halo,output)
        else:result=compare(args.small,args.large,output)
        print(json.dumps(dict(status=result.get('status',result.get('category')),passed=result.get('passed'),output=str(output))))
    except Exception as exc:
        write(output/'failure.json',dict(error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
