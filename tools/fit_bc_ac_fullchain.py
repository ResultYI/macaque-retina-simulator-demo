"""Explicit train-only sensitivity and finite BC/AC full-chain optimizer entry."""
from pathlib import Path
from copy import deepcopy
import argparse,datetime,json,math,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.bc_ac_fullchain_candidate import read,sha,forward,bind_parameters,load_target_contract
from retina_sim.bc_ac_fullchain_fit import sensitivity_report,shape_with_error,bounded_pattern_fit,admissible_parameters
from retina_sim.group_time_storage import load_bundle
from retina_sim.group_time_monitor import frozen_budget
from retina_sim.contracts import canonical_hash,ContractError
from retina_sim.background_checkpoint import MemoryGuard
from tools.validate_ac_vs import Worker,write

OUT=ROOT/'reports/bc_ac_fullchain_fit_sol_v1'
BASE=ROOT/'reports/bc_ac_fullchain_adapter_v1/terra_pilot_g0_v3'
PREP=ROOT/'reports/bc_ac_fullchain_terra_v1/pilot_preparation_v2'
KEYS=['bc.tau_release_s','ac.gate_tau_scale']
VALUES={KEYS[0]:.003,KEYS[1]:1.}
BOUNDS={KEYS[0]:[.0015,.006],KEYS[1]:[.5,2.]}


def candidate_config():
    c=read(PREP/'config.json')
    c['parameters'].update(active_parameters=KEYS,bounds=BOUNDS,BC_populations=['diffuse_BC_ON','diffuse_BC_OFF'],AC_populations=['cross_OFF_ON_parasol_AC_v1'])
    c['launch']['final_parameter_choice_and_bounds']=dict(values=VALUES,bounds=BOUNDS,
        rationale='factor-two engineering neighborhood of original uncalibrated 3ms release and gate time; no claim of measured macaque intervals',
        fixed='all membrane scales, gain, history, adaptation, AII, HC, topology and RGC parameters; no time shift')
    return c


def observables(folder):
    r=read(folder/'result.json');selected=r['group_memberships']['ports__parasol_RGC_ON:gE']
    total=sum(v['member_count'] for v in selected)
    with np.load(folder/'forward_raw_model_ports.npz',allow_pickle=False) as a:
        result={}
        for ch,sign in [('E',-1),('I',1)]:
            x=a['ports__parasol_RGC_ON:g'+ch];budget=frozen_budget(x[0],'S')[2]
            result[ch]=(sign*x.sum(axis=1)/total,float(budget.sum()/total))
    return result


def prepare():
    OUT.mkdir(exist_ok=False);config=candidate_config()
    source=load_bundle(ROOT/config['prepared_model']/'initial_seed',expected_kind='source_construction_seed')['payload']
    bound,_=bind_parameters(source,config,VALUES,['cross_OFF_ON_parasol_AC_v1'])
    # Baseline reuse changes metadata only, not any numerical parameter/state.
    for pop in source['inner_parameters']['populations']:
        if canonical_hash(bound['inner_parameters']['populations'][pop])!=canonical_hash(source['inner_parameters']['populations'][pop]):raise ContractError('Nominal candidate differs from saved baseline')
    baseline=read(BASE/'result.json');drive=load_bundle(PREP/'drive',expected_kind='validation_evidence')['payload']
    if baseline['input_identity']!=canonical_hash(drive) or not baseline['initial_precommit']['passed'] or baseline['raw_absolute_monitor']['domain_failures']:
        raise ContractError('Baseline input/initial/domain identity changed')
    contract=dict(schema='bc-ac-conditional-finite-training-search-1',recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parameter_values=VALUES,bounds=BOUNDS,perturbation_factor=1.25,sensitivity_window=[10000,11000],minimum_singular_ratio=.01,
        sensitivity_scope='g0 training source prefix only; insufficient signal cannot establish nonidentifiability of full6s or other groups',
        numerical_signal_gate='normalized response change must exceed sum of two original propagated numerical bounds',
        rank_gate='minimum/maximum singular value >=0.01; engineering condition number <=100, not a statistical confidence criterion',
        full_fit_window=[10000,70000],training_groups=[0,1,2,3],validation_groups=[4,5],historical_test_waveforms_read=0,
        normalization='signed own-RMS over active window; DC retained; no offset, gain, clipping or time shift',
        observation='fixed per-member mean core ON-parasol conductance proxy, E negative/I positive; not actual clamp current or measured Cell6 RF',
        source_clock='shared exported stimulus/current row j, dt100us; spike0/1 basis irrelevant and unused',absolute_acquisition_latency_s=None,
        timing_interpretation='effective time within fixed export-index/engineering driver convention; no intrinsic physiological latency identification',
        optimizer=dict(initial_log_step=math.log(1.25),minimum_log_step=math.log(1.1),maximum_sweeps=3,
            acceptance='training loss reduction exceeds both propagated numerical loss errors',validation='both separate groups non-degrading within numerical loss error; mean improvement exceeds it',
            stages=['BC:E only; AC fixed','AC:I only; selected BC fixed; E guard','joint:E+I equal weight; only admissible times']),
        baseline_reused=str(BASE.relative_to(ROOT)),baseline_hashes={n:sha(BASE/n) for n in ['result.json','forward_raw_model_ports.npz']},
        input_pins={n:sha(PREP/n) for n in ['config.json','drive/manifest.json']},
        code_pins={n:sha(ROOT/n) for n in ['retina_sim/bc_ac_fullchain_candidate.py','retina_sim/bc_ac_fullchain_fit.py','tools/fit_bc_ac_fullchain.py']},
        preserved_legacy_absolute_monitor=baseline['raw_absolute_monitor']['passed'],FIT_READY=False,physical_qualification=False,
        data_access_correction='Prior eight-group input_contract metadata was accidentally displayed during handoff review; included historical aggregate ranges, not waveforms; none used for decisions. All current target/optimization access is isolated six-group bundle.')
    write(OUT/'PRE_REGISTERED.json',contract);write(OUT/'candidate_config.json',config)
    print(json.dumps(dict(status='FROZEN',directory=str(OUT))))


def verify_registration():
    p=read(OUT/'PRE_REGISTERED.json')
    for name,digest in p['code_pins'].items():
        if sha(ROOT/name)!=digest:raise ContractError('Preregistered code changed: '+name)
    for name,digest in p['baseline_hashes'].items():
        if sha(BASE/name)!=digest:raise ContractError('Frozen source baseline changed: '+name)
    for name,digest in p['input_pins'].items():
        if sha(PREP/name)!=digest:raise ContractError('Frozen pilot inputs changed: '+name)
    return p


def execute(c,drive,values,directory):
    directory.mkdir(parents=True,exist_ok=False);worker=Worker(directory)
    limits=c['launch']['forward_resource_contract'];worker.resources=dict(worker.resources,**limits)
    worker.memory=MemoryGuard(worker.resources['process_RSS_limit_bytes'],worker.resources['available_RAM_floor_bytes'])
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        with worker:r=forward(ROOT,c,drive,values,directory,guard=worker.check)
        write(directory/'result.json',r)
        write(directory/'execution.json',dict(start_utc=started,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),values=values,configuration_identity=canonical_hash(c)))
        return r
    except Exception as e:
        write(directory/'failure.json',dict(error=type(e).__name__,reason=str(e),values=values,start_utc=started));raise


def run_sensitivity():
    p=verify_registration();c=read(OUT/'candidate_config.json')
    drive=load_bundle(PREP/'drive',expected_kind='validation_evidence')['payload'];changed={}
    for key in KEYS:
        folder=OUT/('perturb_'+key.split('.')[0]);values=dict(VALUES);values[key]*=p['perturbation_factor']
        if folder.exists():raise ContractError('Fresh perturbation output required; do not silently resume/reuse')
        print(json.dumps(dict(stage='START_PARAMETER_FORWARD',parameter=key,values=values)),flush=True)
        execute(c,drive,values,folder);changed[key]=observables(folder)
    report=sensitivity_report(observables(BASE),changed,{k:math.log(p['perturbation_factor']) for k in KEYS},p['sensitivity_window'],p['minimum_singular_ratio'])
    report.update(scope=p['sensitivity_scope'],status='PREFIX_SENSITIVITY_COMPLETE_NOT_FULL_TRAINING_QUALIFICATION',full_duration_training_sensitivity=False,
        physical_latency_identified=False,FIT_READY=False)
    write(OUT/'SENSITIVITY_RESULT.json',report);print(json.dumps(report),flush=True)


def run_fit(spec_path):
    """Executable full4/2 entry; requires independently frozen complete drives and full-training sensitivity."""
    spec=read(spec_path);p=read(OUT/'PRE_REGISTERED.json');c=read(spec['configuration'])
    if not spec.get('code_pins'):raise ContractError('Explicit frozen optimizer source pins required')
    for name,digest in spec['code_pins'].items():
        if sha(ROOT/name)!=digest:raise ContractError('Frozen optimizer implementation changed: '+name)
    registration=read(spec['full_registration'])
    for name,digest in registration['input_pins'].items():
        if sha(ROOT/name)!=digest:raise ContractError('Frozen complete fit input changed: '+name)
    if c.get('diagnostic_pilot') is not None:raise ContractError('Diagnostic prefix cannot be used for fitting')
    sensitivity=read(spec['full_training_sensitivity'])
    if sensitivity.get('code_pins')!=spec['code_pins'] or sensitivity.get('source_protocol_sha256')!=sha(spec['full_registration']):
        raise ContractError('Full sensitivity provenance differs from frozen complete trial')
    if sensitivity.get('scope')!='FOUR_TRAINING_GROUPS_FULL_ACTIVE_WINDOW':
        raise ContractError('Full four-group training sensitivity required; prefix alone is insufficient')
    selected=admissible_parameters(sensitivity,KEYS)
    targets=load_target_contract(ROOT,c,load_arrays=True);cases=targets['cases'];train=cases[:4];validation=cases[4:]
    if set(spec['drive_bundles'])!={g['stimulus_sha256'] for g in cases}:raise ContractError('Exactly six original full-source drives required')
    out=Path(spec['output']).resolve()
    if not out.is_relative_to(OUT) or out.exists():raise ContractError('Fresh fit output inside independent report directory required')
    out.mkdir();write(out/'FROZEN_RUN_SPEC.json',spec);cache={};forward_count=0
    def predictions(values,case):
        nonlocal forward_count
        stim=case['stimulus_sha256'];key=canonical_hash(dict(config=c,values=values,stimulus=stim))
        if key not in cache:
            for row in sensitivity.get('forward_cache',[]):
                if row['stimulus_sha256']!=stim or row['values']!=values:continue
                folder=Path(row['directory'])
                for name,digest in row['hashes'].items():
                    if sha(folder/name)!=digest:raise ContractError('Identical-identity forward cache changed')
                receipt=read(folder/'result.json')
                if (read(folder/'execution.json')['configuration_identity']!=canonical_hash(c)
                        or receipt['input_identity']!=c['launch']['fixed_input_driver']['stimulus_payload_sha256'][stim]
                        or receipt['binding']['candidate_values']!=values or not receipt['initial_precommit']['passed']
                        or receipt['raw_absolute_monitor']['domain_failures']):raise ContractError('Cached full forward identity/domain differs')
                cache[key]=observables(folder);break
        if key not in cache:
            drive=load_bundle(spec['drive_bundles'][stim],expected_kind='validation_evidence')['payload']
            folder=out/('evaluation_'+str(forward_count).zfill(4));forward_count+=1
            execute(c,drive,values,folder);cache[key]=observables(folder)
        return cache[key]
    def score(values,which):
        losses={};errors={}
        for ch in ['E','I']:
            ls=[];es=[]
            for case in which:
                x,b=predictions(values,case)[ch];y=case['targets'][ch]
                xn,en=shape_with_error(x,b,p['full_fit_window']);yn,_=shape_with_error(y,0.,p['full_fit_window']);lo,hi=p['full_fit_window']
                d=(xn-yn)[lo:hi];e=en[lo:hi];ls.append(float(np.mean(d*d)));es.append(float(np.mean(2*np.abs(d)*e+e*e)))
            losses[ch]=ls;errors[ch]=es
        return dict(per_group=losses,numerical_per_group=errors,mean={ch:float(np.mean(v)) for ch,v in losses.items()},error={ch:float(np.mean(v)) for ch,v in errors.items()})
    baseline=score(VALUES,train);current=dict(VALUES);stages=[]
    for label,keys,channels in [('BC',[KEYS[0]],['E']),('AC',[KEYS[1]],['I']),('joint',KEYS,['E','I'])]:
        keys=[k for k in keys if k in selected]
        if not keys or label=='joint' and len(keys)<2:continue
        frozen=dict(current)
        def evaluate(partial):
            values=dict(frozen,**partial);s=score(values,train)
            if label=='AC' and s['mean']['E']-baseline['mean']['E']>s['error']['E']+baseline['error']['E']:raise ContractError('AC candidate violates fixed E guard')
            return dict(loss=float(np.mean([s['mean'][ch] for ch in channels])),numerical_error=float(np.mean([s['error'][ch] for ch in channels])))
        setting=p['optimizer'];result=bounded_pattern_fit(evaluate,{k:current[k] for k in keys},{k:BOUNDS[k] for k in keys},
            initial_log_step=setting['initial_log_step'],minimum_log_step=setting['minimum_log_step'],maximum_sweeps=setting['maximum_sweeps'])
        current.update(result['values']);stages.append(dict(stage=label,result=result));write(out/('stage_'+label+'.json'),result)
    # Frozen training selection is recorded before either validation group is scored.
    write(out/'TRAINING_SELECTION.json',dict(values=current,free_time_parameters=selected,stages=stages,validation_opened_for_selection=False))
    before=score(VALUES,validation);after=score(current,validation);gates=[]
    for ch in ['E','I']:
        delta=np.asarray(before['per_group'][ch])-np.asarray(after['per_group'][ch]);error=np.asarray(before['numerical_per_group'][ch])+np.asarray(after['numerical_per_group'][ch])
        gates.append(dict(channel=ch,per_group_improvement=delta.tolist(),nondegrading=bool(np.all(delta>=-error)),mean_improvement_above_numerical=float(delta.mean())>float(error.mean())))
    result=dict(status='CONDITIONAL_FULLCHAIN_TRIAL_COMPLETED',values=current,stages=stages,validation_before=before,validation_after=after,validation_gates=gates,
        free_time_parameters=selected,actual_new_forward_runs=forward_count,cached_identical_forward_runs=len(cache)-forward_count,
        accepted_for_conditional_use=all(g['nondegrading'] and (g['mean_improvement_above_numerical'] if (g['channel']=='E' and KEYS[0] in selected or g['channel']=='I' and KEYS[1] in selected) else True) for g in gates) and all(s['result']['converged'] for s in stages),
        historical_test_waveforms_read=0,default_adopted=False,FIT_READY=False,physical_units_calibrated=False)
    write(out/'FINAL_RESULT.json',result)


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=['prepare','sensitivity','fit']);a.add_argument('--spec');args=a.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='sensitivity':run_sensitivity()
    elif not args.spec:raise ContractError('Explicit frozen complete full-fit run specification required')
    else:run_fit(args.spec)

if __name__=='__main__':main()
