"""Independent conditional BC/AC adapter: real fresh upstream forward, no optimizer.

No production default, biological equation or old source package is overwritten.
The default entry is an identity/contract dry-run; execution needs a separately
frozen launch contract and native/rod input tables. E/I trajectory injection is
not accepted as a forward input.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import zipfile
import numpy as np
from .contracts import ContractError,canonical_hash
from .spatial_group_model import BC,AC,RGC
from .computation_resolution import resolve,require_available,grouped_members

SCHEMA='bc-ac-fullchain-conditional-adapter-1'
SELECTORS={'bc.tau_release_s':('BC','tau_release'),
           'bc.tau_adaptation_s':('BC','tau_a_seconds'),
           'ac.membrane_tau_s':('AC','C'),
           'ac.gate_tau_scale':('AC',('alpha','beta'))}
GAIN_KEYS={'gain','output_gain','bc_gain','ac_gain','G_BA_S','G_AR_S','ge_scale','gi_scale','output_scale'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    if config.get('schema')!=SCHEMA or config.get('mode')!='CONDITIONAL':raise ContractError('Explicit CONDITIONAL full-chain schema required')
    n=config['normalization'];p=config['parameters']
    if n['mode']!='separate_signed_self_rms' or not n['retain_DC'] or any(n[k] for k in ['subtract_baseline','clip','fit_time_shift','truncate_source']):raise ContractError('Signed/DC/time/source contract must not change')
    active=p['active_parameters']
    if len(set(active))!=len(active):raise ContractError('Duplicate free parameter')
    if set(active)&GAIN_KEYS:raise ContractError('Pure multiplicative gain is unidentifiable under own-RMS normalization')
    if set(active)-set(SELECTORS):raise ContractError('Parameter outside BC/AC time-only whitelist')
    for family in ['BC','AC']:
        if sum(SELECTORS[k][0]==family for k in active)>1:raise ContractError('At most one '+family+' effective time may be free')
    if p['maximum_free_BC_times']!=1 or p['maximum_free_AC_times']!=1 or p['all_other_parameters_fixed'] is not True:raise ContractError('Low-dimensional fixed-parameter policy changed')
    if set(p['bounds'])!=set(active):raise ContractError('Every and only active time parameter needs explicit bounds')
    for key in active:
        bound=p['bounds'][key]
        if len(bound)!=2 or not all(np.isfinite(bound)) or not 0<bound[0]<bound[1]:raise ContractError('Positive finite ordered bounds required: '+key)
        if not p[SELECTORS[key][0]+'_populations']:raise ContractError('Explicit population scope required: '+key)
    return config


def bind_parameters(seed,config,values,ordinary_ac_pools):
    """Copy a complete IVP and modify only declared existing dynamical fields."""
    validate_config(config);p=config['parameters'];active=p['active_parameters']
    if set(values)!=set(active):raise ContractError('Values must cover exactly active selectors; fixed fields cannot be released')
    result=deepcopy(seed);parameters=result['inner_parameters'];before=canonical_hash(seed);changes=[]
    for key in active:
        value=values[key]
        if isinstance(value,bool) or not np.isscalar(value) or not np.isfinite(value):raise ContractError('Finite scalar candidate required')
        lo,hi=p['bounds'][key]
        if not lo<=value<=hi:raise ContractError('Candidate outside frozen bounds: '+key)
        family,field=SELECTORS[key];scope=p[family+'_populations'];allowed=set(BC) if family=='BC' else set(ordinary_ac_pools)
        if len(set(scope))!=len(scope) or not set(scope)<=allowed:raise ContractError('Population outside actual BC/ordinary-AC whitelist')
        for pop in scope:
            old=seed['inner_parameters']['populations'][pop];target=parameters['populations'][pop]
            if key=='bc.tau_adaptation_s' and old.get('adaptation_enabled') is not True:raise ContractError('Cannot fit inactive BC adaptation time')
            if key=='ac.membrane_tau_s':
                target['C']=float(value)*old['gL'];changed={'C':target['C']}
            elif key=='ac.gate_tau_scale':
                target['alpha']=old['alpha']/float(value);target['beta']=old['beta']/float(value);changed={k:target[k] for k in ['alpha','beta']}
            else:target[field]=float(value);changed={field:float(value)}
            changes.append(dict(selector=key,population=pop,before={k:deepcopy(old[k]) for k in changed},after=changed))
    receipt=dict(schema=SCHEMA,parent_parameter_identity=parameters['identity'],active_parameters=active,candidate_values=dict(values),changes=changes,
        state_history_RNG_copied_without_reset=True,all_other_parameters_fixed=True,ordinary_AC_other_timescale_fixed=True)
    if changes:
        parameters['identity']=canonical_hash(receipt)
        parameters['provenance']=dict(parent=deepcopy(parameters.get('provenance')),BC_AC_time_binding=receipt)
    result['bc_ac_candidate_lineage']=dict(schema=SCHEMA,parameter_binding=receipt,source_seed_sha256=before,
        new_fresh_forward_required=True,old_EI_cache_used=False,conditional_only=True)
    if canonical_hash(seed)!=before:raise ContractError('Parameter binding mutated frozen source seed')
    return result,receipt


def signed_self_rms(values,active):
    """Positive scale only. Sign, DC and complete time axis are retained."""
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=1 or not np.isfinite(a).all():raise ContractError('Finite signed full-grid waveform required')
    lo,hi=active
    if not 0<=lo<hi<=len(a):raise ContractError('Declared normalization window outside unchanged source grid')
    scale=np.max(abs(a[lo:hi]))
    denominator=scale*np.sqrt(np.mean((a[lo:hi]/scale)**2)) if scale else 0.
    if not np.isfinite(denominator) or denominator<=0:raise ContractError('Zero/nonfinite own RMS; no substitute')
    return a/denominator,float(denominator)


def training_shape_objective(predictions,cases,channel):
    """Equal-weight training-group loss on frozen signed, DC-retaining shapes.

    Predictions must already use a fixed observation/polarity mapping. This
    routine neither converts currents to conductances nor selects validation.
    """
    if channel not in ('E','I') or len(cases)!=4 or any(c['split']!='train' for c in cases):
        raise ContractError('Exactly four original training groups and one E/I channel required')
    ids=[c['stimulus_sha256'] for c in cases]
    if len(set(ids))!=4 or set(predictions)!=set(ids):raise ContractError('Predictions must match the four training stimulus identities exactly')
    losses=[]
    for case in cases:
        predicted=np.asarray(predictions[case['stimulus_sha256']],dtype=np.float64)
        observed=np.asarray(case['targets'][channel],dtype=np.float64)
        if predicted.shape!=(71000,) or observed.shape!=(71000,):raise ContractError('Complete original source grid required')
        p,_=signed_self_rms(predicted,[10000,70000]);o,_=signed_self_rms(observed,[10000,70000])
        losses.append(float(np.mean((p[10000:70000]-o[10000:70000])**2)))
    return dict(channel=channel,mean_loss=float(np.mean(losses)),per_group=dict(zip(ids,losses)),
        equal_group_weight=True,DC_retained=True,validation_used=False,physical_current_units_inferred=False)


def load_target_contract(root,config,load_arrays=False):
    """Only the six already isolated admitted groups; never the8-group inputs."""
    validate_config(config);root=Path(root);spec=config['targets'];directory=root/spec['directory']
    for name,key in [('target_definition.json','definition_sha256'),('target_manifest.json','manifest_sha256'),('targets_native_100us.npz','arrays_sha256')]:
        if sha(directory/name)!=spec[key]:raise ContractError('Frozen target identity changed: '+name)
    definition=read(directory/'target_definition.json');manifest=read(directory/'target_manifest.json')
    groups=definition['groups']
    if [g['stimulus_sha256'] for g in groups]!=[g['stimulus_sha256'] for g in manifest['groups']]:raise ContractError('Target group identities changed')
    if [g['split'] for g in groups]!=['train']*4+['validation']*2 or len({g['stimulus_sha256'] for g in groups})!=6:raise ContractError('Original4train/2val group split required')
    if definition['excluded_sealed_groups']!=2 or definition['sealing']['test_waveforms_materialized'] or definition['sealing']['test_metrics_read']:raise ContractError('Six-group sealed-data isolation changed')
    if definition['arrays']['samples']!=71000 or definition['arrays']['sample_dt_s']!=.0001 or definition['arrays']['active_samples']!=[10000,70000]:raise ContractError('Original exported grid/window required')
    with zipfile.ZipFile(directory/'targets_native_100us.npz') as z:
        names=z.namelist();allowed={f'g{i}_' for i in range(6)}
        if any(n!='time_exported_grid_s.npy' and not any(n.startswith(prefix) for prefix in allowed) for n in names):raise ContractError('Unadmitted array present; refuse to open waveforms')
    cases=[];access=[]
    if load_arrays:
        with np.load(directory/'targets_native_100us.npz',allow_pickle=False) as arrays:
            t=arrays['time_exported_grid_s'];access.append('time_exported_grid_s')
            if not np.array_equal(t,np.arange(71000)*.0001):raise ContractError('Source sample origin/grid differs')
            for j,g in enumerate(groups):
                prefix='g'+str(j);stim=arrays[prefix+'_stimulus_raw'];access.append(prefix+'_stimulus_raw')
                if stim.shape!=(71000,) or not np.isfinite(stim).all() or hashlib.sha256(stim.tobytes()).hexdigest()!=g['stimulus_sha256']:raise ContractError('Original full stimulus hash/shape differs')
                case=dict(g,source_time_s=t.copy(),stimulus_raw=stim.copy(),targets={},normalization={})
                for channel in ['E','I']:
                    key=prefix+'_'+channel+'_mean';raw=arrays[key];access.append(key)
                    normalized,den=signed_self_rms(raw,[10000,70000]);frozen=arrays[prefix+'_'+channel+'_normalized_mean'];access.append(prefix+'_'+channel+'_normalized_mean')
                    if raw.shape!=(71000,) or not np.allclose(normalized,frozen,rtol=2e-14,atol=2e-14):raise ContractError('Signed DC-retaining target definition differs')
                    case['targets'][channel]=raw.copy();case['normalization'][channel]=dict(positive_RMS=den,DC_subtracted=False,sign_changed=False)
                cases.append(case)
    return dict(definition=definition,groups=groups,cases=cases,accessed_array_keys=access,historical_test_waveforms_read=0,historical_test_metrics_read=0,
        original_stimulus_unit=None,original_current_unit=None,source_export_origin_s=0.,physical_origin_verified=False,source_dt_s=.0001,model_input_fps=150.,
        fixed_resampling_and_photometric_mapping=config['launch']['fixed_input_driver'],fixed_clock_bridge=config['launch']['fixed_clock_bridge'],
        response_polarity=config['targets']['polarity'],DC_retained=True,qualification='CONDITIONAL')


def feedback_graph(root,config):
    """Mirror the actual ACVSAssembly route selection on pinned graph metadata."""
    spec=config['feedback_graph'];path=Path(root)/spec['parent_manifest']
    if sha(path)!=spec['parent_manifest_sha256']:raise ContractError('Actual parent graph identity differs')
    parent=read(path);ac_config=read(Path(root)/spec['AC_configuration']);legacy='local_feedforward_AC'
    routes=[dict(name=r['name'],pre=r['pre'],post=r['post'],kind=r['kind']) for r in parent['routes'] if
        (r['pre'] not in AC and r['post'] not in AC) or r['post']==legacy or (r['pre']==legacy and r['post']=='small_bistratified_S_ON')]
    pools=[c['pool'] for c in ac_config['channels']]
    for c in ac_config['channels']:
        for pre,post in [(c['pre'],c['pool']),(c['pool'],c['target'])]:routes.append(dict(name=pre+'_to_'+post,pre=pre,post=post,kind='inner_chemical'))
    bad=[r for r in routes if r['pre'] in pools and r['post'] in BC or r['pre']=='AII_AC' and r['post'] in RGC]
    if bad:raise ContractError('Unexpected ordinary-AC/BC or direct AII/RGC topology')
    pairs={(r['pre'],r['post']) for r in routes if r['kind'] in ['gap','native']}
    reciprocal=sorted({tuple(sorted((a,b))) for a,b in pairs if a!=b and (b,a) in pairs})
    return dict(routes=routes,reciprocal_feedback_pairs=reciprocal,ordinary_AC_to_BC=[],AII_to_RGC=[],
        parameters_fixed_not_state_trajectories=True,all_existing_feedback_evolved_each_forward=True,added_edges=0)


def startup_missing(config):
    fields=[k for k,v in config['launch'].items() if v is None]
    if config['launch']['approved_contract'] is not True:fields.insert(0,'approved_contract')
    return fields


def dry_run(root,config):
    validate_config(config);targets=load_target_contract(root,config,load_arrays=True);graph=feedback_graph(root,config)
    backend=config.get('execution_backend')
    if backend is not None:
        from .ac_vs_rhs_readouts import contract
        if backend!=contract():raise ContractError('Explicit RHS adapter implementation identity differs')
    choice=config['computation'];resolution=resolve(root,read(Path(root)/choice['config']),choice['spatial_scope'],choice['granularity'])
    prepared=Path(root)/config['prepared_model']/'initial_seed/manifest.json'
    if sha(prepared)!=config['prepared_seed_manifest_sha256']:raise ContractError('Frozen mixed-domain source changed')
    return dict(status='ENGINEERING_ADAPTER_READY_AWAITING_FINAL_LAUNCH_CONTRACT',mode='CONDITIONAL',active_parameters=config['parameters']['active_parameters'],
        eligible_selectors_are_not_final_parameter_choices=list(SELECTORS),groups=[dict(stimulus_sha256=g['stimulus_sha256'],split=g['split']) for g in targets['groups']],
        source_samples=71000,source_dt_s=.0001,model_input_fps=150.,source_origin_s=0.,physical_origin_verified=False,
        physical_illuminance=None,physical_current_unit=None,physical_DC_offset=None,input_driver=config['launch']['fixed_input_driver'],response_polarity=targets['response_polarity'],
        normalization=config['normalization'],data_access=dict(array_keys=targets['accessed_array_keys'],historical_test_waveforms_read=0,historical_test_metrics_read=0),
        feedback_graph=graph,computation_resolution=resolution,execution_backend=backend,missing_startup_fields=startup_missing(config),model_loads=0,model_runs=0,optimizer_steps=0,default_adopted=False,FIT_READY=False)


def validate_drive(drive,seed,protocol,targets,config=None):
    allowed={'stimulus_sha256','native_inputs','rod_inputs','model_origin_s','model_stop_s','source_origin_s','source_dt_s','source_samples','input_driver_identity','clock_bridge_identity','native_unit','rod_unit'}
    if set(drive)!=allowed:raise ContractError('Forward accepts only frozen native/rod drive, never cached E/I trajectories')
    if drive['stimulus_sha256'] not in {g['stimulus_sha256'] for g in targets['groups']}:raise ContractError('Stimulus outside original admitted4/2 groups')
    if drive['source_samples']!=71000 or drive['source_dt_s']!=.0001 or drive['source_origin_s']!=0. or drive['model_origin_s']!=seed['time_seconds']:raise ContractError('Fixed source/model origin and complete sample grid required; no fitted shift')
    sample_count=71000
    if config is not None and config.get('diagnostic_pilot') is not None:
        pilot=config['diagnostic_pilot']
        if pilot.get('purpose')!='TRAINING_INPUT_FORWARD_ONLY_NO_LOSS_OR_FIT' or pilot.get('source_start_index')!=0:
            raise ContractError('Only a preregistered non-fitting source-prefix diagnostic may use a shorter forward')
        sample_count=pilot.get('source_sample_count')
        if type(sample_count) is not int or not 10000<sample_count<=71000:raise ContractError('Pilot must preserve the full pre-stimulus history and some active input')
    if drive['model_stop_s']!=drive['model_origin_s']+sample_count*.0001:raise ContractError('Frozen source duration differs; no arbitrary truncation')
    if drive['native_unit']!='td' or drive['rod_unit']!='R*/s':raise ContractError('Native engineering and rod source input units required')
    if not drive['input_driver_identity'] or not drive['clock_bridge_identity']:raise ContractError('Fixed driver/clock evidence identities required')
    if config is not None:
        for key,actual in [('fixed_input_driver',drive['input_driver_identity']),('fixed_clock_bridge',drive['clock_bridge_identity'])]:
            declared=config['launch'][key]
            if not isinstance(declared,dict) or declared.get('identity')!=actual:
                raise ContractError('Drive does not match frozen '+key+' identity')
        hashes=config['launch']['fixed_input_driver'].get('stimulus_payload_sha256',{})
        if hashes.get(drive['stimulus_sha256'])!=canonical_hash(drive):
            raise ContractError('Native/rod tables differ from the frozen stimulus payload')
    if set(drive['native_inputs'])!=set(drive['rod_inputs']) or not drive['native_inputs']:raise ContractError('Both original input channels/frames required')
    if config is not None:
        origin=protocol['input_origin_seconds'];fps=protocol['input_fps']
        if fps!=150.:raise ContractError('Original150Hz input clock required')
        first=math.floor((drive['model_origin_s']-origin)*fps)
        last=math.floor((np.nextafter(drive['model_stop_s'],-np.inf)-origin)*fps)
        if set(drive['native_inputs'])!=set(range(first,last+1)):
            raise ContractError('Frozen native/rod frame coverage is incomplete or includes extraneous frames')
    for category,pop in [('native_inputs','cones'),('rod_inputs','rods')]:
        count=len(seed['states'][pop,'reduced_native' if pop=='cones' else 'rod_membrane','Vbar' if pop=='cones' else 'V'])
        for value in drive[category].values():
            a=np.asarray(value)
            if a.dtype!=np.float64 or a.shape!=(count,) or not np.isfinite(a).all() or np.any(a<0):raise ContractError('Native/rod input finite nonnegative FP64 shape differs')
    return dict(protocol,start_seconds=drive['model_origin_s'],stop_seconds=drive['model_stop_s'],input_frame_indices=sorted(drive['native_inputs']))


def forward(root,config,drive,values,output,*,loader=None,arm_factory=None,guard=None):
    """Each evaluation reloads the full mixed-domain IVP and advances all feedback.

    Injection arguments exist only for small synthetic wiring checks. Production
    CLI always uses the real mixed-domain loader and GroupTimeArm, no E/I cache.
    Raw model ports retain their own declared units; no MAT-current conversion.
    """
    validate_config(config)
    missing=startup_missing(config)
    if missing:raise ContractError('Launch contract not closed: '+', '.join(missing))
    choice=config['computation'];resolution=require_available(resolve(root,read(Path(root)/choice['config']),choice['spatial_scope'],choice['granularity']))
    if loader is None:
        from tools.validate_ac_vs_native_support import load
        loader=load
    if arm_factory is None:
        from .group_time_runner import GroupTimeArm
        arm_factory=GroupTimeArm
    from tools.validate_ac_vs_timescale import calendar
    from .group_time_monitor import GroupTimeMonitor
    from tools.validate_ac_vs_equilibrium_start import initial_match
    from tools.validate_ac_vs import dynamic_observe
    from .ac_vs_monitor import DynamicDefectMonitor
    from .group_time_storage import save_bundle
    root=Path(root);targets=load_target_contract(root,config,load_arrays=False)
    if sha(root/config['prepared_model']/'initial_seed/manifest.json')!=config['prepared_seed_manifest_sha256']:
        raise ContractError('Frozen mixed-domain source changed')
    a,source,identity=loader(root/config['prepared_model'])
    execution_backend=None
    if config.get('execution_backend') is not None:
        from .ac_vs_rhs_readouts import install
        execution_backend=install(a,config['execution_backend'])
    if a.artifact_sha256!=resolution['binding']['graph_artifact_sha256']:raise ContractError('Runtime graph differs from declared resolution identity')
    policy=config['launch']['initial_state_history_policy']
    if not isinstance(policy,dict) or policy.get('source_seed_sha256')!=canonical_hash(source) or policy.get('mode')!='fresh_verified_source':
        raise ContractError('Initial state/history does not match the frozen source policy')
    seed,binding=bind_parameters(source,config,values,[c['pool'] for c in a.ac_config['channels']])
    protocol=validate_drive(drive,seed,identity['protocol'],targets,config)
    seed['native_inputs']=deepcopy(drive['native_inputs']);seed['rod_inputs']=deepcopy(drive['rod_inputs'])
    input_identity=canonical_hash(drive);protocol['fullchain_fixed_input_identity']=input_identity;protocol['computation_resolution_identity']=resolution['identity']
    if execution_backend is not None:protocol['rhs_execution_backend']=execution_backend
    seed['bc_ac_candidate_lineage'].update(input_identity=input_identity,full_forward_restarted_from_source=True,previous_state_trajectory_reused=False)
    selected=config['launch']['fixed_output_selection']
    if not isinstance(selected,list) or not selected:raise ContractError('Explicit fixed target global IDs/ports/units required')
    sample_count=config.get('diagnostic_pilot',{}).get('source_sample_count',71000)
    times=drive['model_origin_s']+np.arange(sample_count)*.0001
    events=dict(calendar(protocol));indices={float(t):j for j,t in enumerate(times)}
    for t in times:
        if t>protocol['start_seconds']:events.setdefault(float(t),dict(rod=False,bc=False,frame=False,record=False))
    result={item['name']:[] for item in selected};units={item['name']:item['unit'] for item in selected};rows={};group_mapping={}
    for item in selected:
        pop=item['population'];lookup={str(v):j for j,v in enumerate(a.populations[pop].group_ids)}
        if not item['global_ids'] or any(v not in lookup for v in item['global_ids']):raise ContractError('Output group identity not bound')
        rows[item['name']]=[lookup[v] for v in item['global_ids']]
        group_mapping[item['name']]=grouped_members(a.populations[pop],rows[item['name']])
    arm=arm_factory(a,seed,protocol,mode='reference',guard=guard)
    def observe(t,side,store=False):
        s=arm.snapshot(t,side,queue_node=True)
        monitor.observe(t,side,s['states'],s['rhs'],transitions=s['transitions'],ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
        if monitor.domain_failures:raise ContractError('Forward physical/domain/history gate failed')
        if store:
            for item in selected:
                key=(item['population'],item['port']);name=item['name']
                if s['port_units'][key]!=item['unit']:raise ContractError('Model output unit differs; no implicit current conversion')
                value=np.asarray(s['ports'][key])[rows[name]]
                if not np.isfinite(value).all():raise ContractError('Nonfinite forward output')
                result[name].append(value.copy())
            count=len(result[selected[0]['name']])
            if count==1 or count%1000==0 or count==sample_count:
                pending=Path(output)/'progress.json.writing'
                pending.write_text(json.dumps(dict(completed_samples=count,total_samples=sample_count,model_time_s=t,
                    source_time_s=t-drive['model_origin_s'],initial_precommit_pass=initial_receipt['passed'])),encoding='utf8')
                pending.replace(Path(output)/'progress.json')
    try:
        monitor=GroupTimeMonitor(a,seed['states'],parameters=seed['inner_parameters'])
        dynamic=DynamicDefectMonitor(a,seed['states']) if config['launch'].get('dynamic_sampling_contract') else None
        initial_receipt,_=initial_match(arm,seed);observe(arm.start,'right');arm.initial_commit();observe(arm.start,'right',True);arm.ack(arm.start)
        for end,flags in sorted(events.items()):
            if end<=arm.inner_time:continue
            begin=arm.inner_time;arm.ensure_native(end,'left');advance=arm.advance_inner(begin,end)
            if dynamic is not None and (flags['bc'] or flags['frame']) and end-begin>1e-12:
                for curve in advance.dense_segments:
                    mid=curve.t0+(curve.t1-curve.t0)/2;witness=dynamic_observe(arm,advance,mid,dynamic)
                    if max((r['maximum_norm'] for r in dynamic.rows.values()),default=0.)>1:
                        save_bundle(Path(output)/'failed_dynamic_witness',kind='validation_evidence',identity=dict(input_identity=input_identity,parameter_binding=binding),
                            payload=dict(time_seconds=mid,states=witness['states'],rhs=witness['rhs'],ports=witness['ports'],
                                inner_derivative=advance.representation_derivative(mid,'right'),
                                inner_dense=dict(t0=curve.t0,t1=curve.t1,y0=curve.y0,coefficients=curve.coefficients),
                                native_interval=arm._interval_at(mid,'right').payload(),dynamic=dynamic.report()),
                            qualification=dict(default_adopted=False,fit_eligible=False))
                        raise ContractError('DYNAMIC_NUMERICAL_DEFECT exceeds original budget; witness preserved')
            if flags['frame']:observe(end,'left')
            if flags['rod'] or flags['bc']:arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],rod_sample_time=max(arm.start,end-.0001+1e-13) if flags['rod'] else None)
            observe(end,'right',end in indices);arm.ack(end)
        arrays={k:np.stack(v) for k,v in result.items()}
        if any(len(v)!=sample_count for v in arrays.values()):raise ContractError('Frozen forward source/output grid incomplete')
        np.savez_compressed(Path(output)/'forward_raw_model_ports.npz',source_time_s=np.arange(sample_count)*.0001,model_time_s=times,**arrays)
        return dict(status='CONDITIONAL_FULLCHAIN_FORWARD_COMPLETE',input_identity=input_identity,binding=binding,units=units,
            mode='CONDITIONAL',physical_current_unit=None,source_stimulus_sha256=drive['stimulus_sha256'],fresh_upstream_forward=True,cached_EI_used=False,
            diagnostic_pilot=config.get('diagnostic_pilot'),computation_resolution=resolution,group_memberships=group_mapping,initial_precommit=initial_receipt,states_frozen_during_forward=False,
            execution_backend=execution_backend,DYNAMIC_NUMERICAL_DEFECT=None if dynamic is None else dynamic.report(),dynamic_sampling_contract=config['launch'].get('dynamic_sampling_contract'),
            raw_absolute_monitor=monitor.report(protocol['stop_seconds']-protocol['start_seconds']),metrics=arm.metrics,optimizer_steps=0,FIT_READY=False)
    finally:arm.close()
