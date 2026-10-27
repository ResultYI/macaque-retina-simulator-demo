"""Stage1 endogenous RGC outputs; fixed upstream, resumable capture, no test reuse."""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json,math,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.group_time_storage import load_bundle,save_bundle
from retina_sim.rgc_stage1_cache import capture,read_cache,write
from retina_sim.rgc_stage1_readout import observed_past,objective,fit,decode,free_mean,NAMES,bounds
from tools.validate_ac_vs_native_support import load

OUT=ROOT/'reports/rgc_stage1_complete_v1'
SOURCE=ROOT/'reports/layer_calibration/rgc_cbem_conditional_v1/frozen_preparation_v3'
DRIVES=ROOT/'reports/bc_ac_fullchain_fit_sol_v1/full_trial_preparation_v1'
CODE=['retina_sim/rgc_stage1_cache.py','retina_sim/rgc_stage1_readout.py','tools/fit_rgc_stage1.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def prepare():
    OUT.mkdir(exist_ok=False);a,seed,identity=load(ROOT/'reports/ac_vs_native_support_v1/prepared_v1')
    targets=read(ROOT/'reports/layer_calibration/ac_ei_targets_v1/target_definition.json')['groups']
    if sha(SOURCE/'inputs.npz')!='b7984a99d2d0368f9568832155dbc121a0f28b15cf480eb9b248cc8a09bfb8d6':raise ContractError('Original isolated Cell6 preparation changed')
    recording={}
    for pop in ['midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON']:
        field=identity['selection']['fields']['ports__'+pop+':voltage'];rows=field['rows'];p=a.populations[pop]
        recording[pop]=dict(rows=rows,global_ids=field['ids'],members=np.asarray(p.counts)[rows].astype(int).tolist(),centroid_mm=np.asarray(p.centroid_mm)[rows].tolist())
    on=recording['parasol_RGC_ON'];center=np.array([4.73*.223,0.]);dist=np.linalg.norm(np.asarray(on['centroid_mm'])-center,axis=1);selected=int(np.argmin(dist));row=on['rows'][selected]
    p=seed['inner_parameters']['populations']['parasol_RGC_ON'];base={}
    for name in ['C','gL','EL','EE','EI','alpha','mu','beta']:
        v=np.asarray(p[name]);base[name]=float(v if v.ndim==0 else v[row])
    base['history_kernel']=np.asarray(p['spike_history_kernel']).tolist()
    pi=[float(seed['states']['parasol_RGC_ON','RGC_expected_history','history_probability_'+str(i)][row]) for i in range(8)]
    arrays={};cases=[];access=[]
    with np.load(SOURCE/'inputs.npz',allow_pickle=False) as data:
        for i,g in enumerate(targets):
            stim=data['g'+str(i)+'_stimulus_values'];counts=data['g'+str(i)+'_spike_counts'];access+=['g'+str(i)+'_stimulus_values','g'+str(i)+'_spike_counts']
            if sha_bytes(stim)!=g['stimulus_sha256'] or counts.shape!=(3,7100) or not np.isin(counts,[0,1]).all():raise ContractError('Actual stimulus/repeat/bin identity differs')
            drive=load_bundle(DRIVES/('drive_g'+str(i)),expected_kind='validation_evidence')['payload']
            if drive['stimulus_sha256']!=g['stimulus_sha256']:raise ContractError('Endogenous driver and observed spike stimulus differ')
            arrays['g'+str(i)+'_counts']=counts.copy()
            cases.append(dict(case_id='cell6_g'+str(i),population='parasol_RGC_ON',split=g['split'],stimulus_sha256=g['stimulus_sha256'],
                drive_bundle=str((DRIVES/('drive_g'+str(i))).relative_to(ROOT)),drive_sha256=canonical_hash(drive),counts_key='g'+str(i)+'_counts',
                score_bins=[1000,7000],source_export_origin_s=0.,samples_100us=71000,model_global_id=on['global_ids'][selected],model_column=selected,
                identity_scope='conditional same-type full-field functional representative; not anatomical Cell6 registration',physical_eccentricity_match_verified=False))
    np.savez_compressed(OUT/'actual_observations.npz',**arrays)
    spec=dict(schema='rgc-stage1-output-contract-1',prepared_model='reports/ac_vs_native_support_v1/prepared_v1',
        prepared_manifest_sha256=sha(ROOT/'reports/ac_vs_native_support_v1/prepared_v1/initial_seed/manifest.json'),
        source_seed_sha256=canonical_hash(seed),internal_parameter_identity=seed['inner_parameters']['identity'],recording=recording,
        target=dict(population='parasol_RGC_ON',global_id=on['global_ids'][selected],column=selected,selection='existing core RF nearest fixed4.73deg project center; frozen before any new likelihood',
            physical_identity_verified=False,actual_observed_cell='Latimer Cell6',assay_scope='conditional fixed-position full-field output calibration'),
        base=base,initial_probability=pi,cases=cases,counts_sha256=sha(OUT/'actual_observations.npz'),
        source_counts_manifest_sha256=sha(SOURCE/'snapshot_manifest.json'),input_mapping='unchanged frozen BC/AC engineering driver: Btd*(1+0.05*(raw[j]-raw[0])); rod unchanged;150Hz ZOH',
        physical_input_calibrated=False,absolute_acquisition_latency_s=None,index_origin=1,index_origin_status='retained source loader convention; unverified, no fitted shift',
        code_pins=dict(identity['source_pins'],**{n:sha(ROOT/n) for n in CODE}),
        eligible_output_parameters=list(NAMES),output_bounds=bounds(base),rank_threshold=.001,
        membrane_parameters_fixed=dict(reason='spike-only natural data and uncertain clamp physical units do not establish absolute membrane parameters',names=['C','gL','EL','EE','EI','V0']),
        all_internal_and_route_gains_fixed=True,observed_history='strictly past original binary observations; first1s propagated, score1..7s',
        training_split='original4 groups',validation_split='original2 groups, developmental and previously used in old clamp study',historical_test_reopened=False,
        optimizer=dict(method='L-BFGS-B',maxiter=100,maxfun=1000,maxls=30,ftol=1e-10,gtol=1e-6),
        validation_gate='each original group nondegrading within native-roundoff loss allowance; mean improvement greater than allowance; no test selection',
        single_worker_safety_seconds=9054,checkpoint_policy='complete original clocks/history/RNG/native/controller/readout; first50ms then500ms, same-identity resume only',
        numerical=dict(mode='reference',rtol=1e-6,atol=1e-9,max_step=.0001,clock_rod=.0001,clock_BC_RGC=.001,input_fps=150),
        consistency=dict(duration_seconds=.01,rate_roundoff_multiplier=8192,cache_voltage_exact=True),
        source_condition='fixed exterior conditional local equilibrium; no global background or0.6 spatial convergence upgrade',
        physical_qualification=False,FIT_READY=False)
    write(OUT/'CONTRACT.json',spec);write(OUT/'DATA_ACCESS.json',dict(accessed=access,measured_EI_arrays_loaded=False,historical_test_keys_loaded=False))
    write(OUT/'CURRENT_STATUS.json',dict(status='CONTRACT_PREPARED_NO_NEW_OPTIMIZATION',model_workers=0,eligible_outputs=list(NAMES),covered_data_family='Cell6 ON-parasol conditional',other_types_not_claimed_fitted=True))
    print(json.dumps(dict(status='PREPARED',target=spec['target'],base=base,eligible_output_parameters=list(NAMES))))


def sha_bytes(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def verify():
    spec=read(OUT/'CONTRACT.json')
    for name,digest in spec['code_pins'].items():
        if sha(ROOT/name)!=digest:raise ContractError('Frozen stage1 code/source changed: '+name)
    if sha(OUT/'actual_observations.npz')!=spec['counts_sha256']:raise ContractError('Actual spike target cache changed')
    if sha(ROOT/spec['prepared_model']/'initial_seed/manifest.json')!=spec['prepared_manifest_sha256']:raise ContractError('Initial state/history package changed')
    return spec


def consistency(spec):
    # Engineering transition is a source-labelled test, never substituted for data.
    a,source,identity=load(ROOT/spec['prepared_model']);directory=OUT/'consistency';directory.mkdir(exist_ok=False)
    end=source['time_seconds']+.01;frames=range(math.floor(source['time_seconds']*150),math.floor(np.nextafter(end,-np.inf)*150)+1)
    stim=canonical_hash(dict(kind='engineering15cone_step_existing_tables_NOT_Cell6',start=source['time_seconds'],stop=end))
    drive=dict(stimulus_sha256=stim,model_origin_s=source['time_seconds'],model_stop_s=end,
        native_inputs={k:source['native_inputs'][k] for k in frames},rod_inputs={k:source['rod_inputs'][k] for k in frames})
    save_bundle(directory/'drive',kind='validation_evidence',identity=dict(stimulus=stim),payload=drive,qualification=dict(default_adopted=False,fit_eligible=False))
    case=dict(case_id='engineering_consistency_only',drive_bundle=str((directory/'drive').relative_to(ROOT)),stimulus_sha256=stim,drive_sha256=canonical_hash(drive))
    small=deepcopy(spec);small['single_worker_safety_seconds']=120;small['additional_checkpoint_ticks']=[50]
    try:capture(ROOT,directory,small,case,stop_after_checkpoint=True)
    except ContractError as exc:
        if str(exc)!='Requested engineering checkpoint stop':raise
    else:raise ContractError('Required mid-run checkpoint was not reached')
    capture(ROOT,directory,small,case,resume=True)
    data,manifest=read_cache(directory/'cache'/case['case_id'],small,case)
    with data:
        target=spec['target'];v=data[target['population']+':V'][::10,target['column']][:-1]
        direct=data['expected_rate_base_Hz'][:,target['column']]
        replay=free_mean(v,spec['base'],spec['initial_probability'])/.001
        allowance=8192*np.finfo(float).eps*np.maximum(abs(direct),1.)
        maximum=float(np.max(abs(replay-direct)/allowance,initial=0.))
        result=dict(passed=maximum<=1,maximum_roundoff_norm=maximum,maximum_absolute_rate_difference_Hz=float(np.max(abs(replay-direct),initial=0.)),
            source='direct GroupTimeArm pre-event native8state expectation versus cached-V readout',bins=len(v),not_a_fit=True,
            complete_checkpoint_resume_exercised=True,no_initial_commit_repeated=True)
    write(directory/'RESULT.json',result)
    if not result['passed']:raise ContractError('Cached RGC history output differs from direct model')
    return result


def load_fit_cases(spec,split):
    cases=[]
    with np.load(OUT/'actual_observations.npz',allow_pickle=False) as observed:
        for c in spec['cases']:
            if c['split']!=split:continue
            cached,m=read_cache(OUT/'cache'/c['case_id'],spec,c)
            with cached:v=cached[c['population']+':V'][::10,c['model_column']][:-1].copy()
            y=observed[c['counts_key']].copy();cases.append(dict(c,voltage=v,counts=y,past=observed_past(y)))
    return cases


def optimize(spec):
    if not read(OUT/'consistency/RESULT.json')['passed']:raise ContractError('Native/cache consistency required')
    if (OUT/'TRAINING_SELECTION.json').exists():raise ContractError('Training selection already frozen; no implicit refit')
    train=load_fit_cases(spec,'train');started=time.perf_counter()
    def guard():
        if (ROOT/'REQUEST_STOP').exists() or (OUT/'REQUEST_STOP').exists():raise ContractError('Stage1 output optimizer STOP')
    def accepted(row):
        with (OUT/'optimizer_trace.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(row)+'\n')
        write(OUT/'optimizer_checkpoint.json',dict(row,contract_identity=canonical_hash(spec)))
        print(json.dumps(dict(stage='OPTIMIZER_ACCEPTED_ITERATION',iteration=row['iteration'],training_NLL=row['training_NLL_per_bin'])),flush=True)
    write(OUT/'OPTIMIZATION_STARTED.json',dict(utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),upstream_reruns_per_candidate=0))
    result=fit(train,spec['base'],callback=accepted,guard=guard);result['wall_seconds']=time.perf_counter()-started
    write(OUT/'TRAINING_SELECTION.json',result);return result


def evaluate(spec):
    selected=read(OUT/'TRAINING_SELECTION.json');theta=np.asarray(selected['theta']);validation=load_fit_cases(spec,'validation');train=load_fit_cases(spec,'train')
    from retina_sim.functional_fit_math import psth_metrics,constant_nll
    null=sum(int(c['counts'][:,c['score_bins'][0]:c['score_bins'][1]].sum()) for c in train)/sum(c['counts'][:,c['score_bins'][0]:c['score_bins'][1]].size for c in train)
    metrics=[]
    for c in validation:
        baseline,_=objective(np.zeros(6),[c],spec['base']);candidate,_=objective(theta,[c],spec['base']);lo,hi=c['score_bins'];idx=np.arange(lo,hi)
        p=decode(theta,spec['base']);free=free_mean(c['voltage'],p,spec['initial_probability'])
        metrics.append(dict(case_id=c['case_id'],baseline_NLL_per_bin=baseline,candidate_NLL_per_bin=candidate,
            constant_NLL_per_bin=constant_nll(null,c['counts'][:,lo:hi])/c['counts'][:,lo:hi].size,
            free_history_PSTH=psth_metrics(c['counts'],free,idx),model_scope=c['identity_scope']))
    tolerance=8192*np.finfo(float).eps
    deltas=[r['baseline_NLL_per_bin']-r['candidate_NLL_per_bin'] for r in metrics]
    accepted=bool(selected.get('success') and all(v>=-tolerance for v in deltas) and np.mean(deltas)>tolerance)
    result=dict(status='STAGE1_CONDITIONAL_CELL6_OUTPUT_RESULT',accepted=accepted,training=selected,validation=metrics,
        old_test_not_reused=True,production_writeback=False,internal_parameters_changed=False,other_RGC_types_not_claimed_fitted=True,FIT_READY=False)
    write(OUT/'CELL6_RESULT.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','consistency','capture','fit','evaluate','run']);p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.mode=='prepare':prepare();return
    spec=verify()
    if args.mode in ['consistency','run'] and not (OUT/'consistency/RESULT.json').exists():print(json.dumps(consistency(spec)),flush=True)
    if args.mode in ['capture','run']:
        for c in spec['cases']:
            print(json.dumps(dict(stage='FIXED_ENDOGENOUS_CAPTURE',case_id=c['case_id'],split=c['split'])),flush=True)
            capture(ROOT,OUT,spec,c,resume=args.resume)
    if args.mode in ['fit','run']:print(json.dumps(optimize(spec)),flush=True)
    if args.mode in ['evaluate','run']:print(json.dumps(evaluate(spec)),flush=True)

if __name__=='__main__':main()
