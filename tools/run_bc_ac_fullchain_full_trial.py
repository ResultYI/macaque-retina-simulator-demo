"""Prepare or run the original full4/2 conditional trial, one model worker at a time."""
from pathlib import Path
import argparse,hashlib,json,math,sys,os,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.fit_bc_ac_fullchain import OUT,PREP,BASE,KEYS,VALUES,BOUNDS,candidate_config,execute,observables,run_fit
from retina_sim.bc_ac_fullchain_candidate import read,sha,load_target_contract,validate_drive
from retina_sim.group_time_storage import load_bundle,save_bundle
from retina_sim.bc_ac_fullchain_fit import sensitivity_report
from retina_sim.contracts import canonical_hash,ContractError
from tools.validate_ac_vs import write

FULL=OUT/'full_trial_preparation_v2'
CODE=['retina_sim/bc_ac_fullchain_candidate.py','retina_sim/bc_ac_fullchain_fit.py','tools/fit_bc_ac_fullchain.py','tools/run_bc_ac_fullchain_full_trial.py']


def prepare():
    FULL.mkdir(exist_ok=False);c=candidate_config();c.pop('diagnostic_pilot')
    c['launch']['dynamic_sampling_contract']=dict(mode='original dense derivative defect facility at midpoints of accepted inner segments departing1ms BC/RGC ticks or150Hz frame edges',
        thresholds='original frozen_state_budget/100us; no threshold change',not_a_full_time_convergence_claim=True)
    source_bundle=load_bundle(ROOT/c['prepared_model']/'initial_seed',expected_kind='source_construction_seed');source=source_bundle['payload']
    targets=load_target_contract(ROOT,c,load_arrays=False);drives={};payload_hashes={}
    t0=source['time_seconds'];stop=t0+7.1
    native_base=np.asarray(source['input_reference']['native_background_td'],dtype=np.float64)
    rod_base=np.broadcast_to(np.asarray(source['input_reference']['rod_background_rstar_s'],dtype=np.float64),source['states']['rods','rod_membrane','V'].shape).copy()
    first=math.floor(t0*150.);last=math.floor(np.nextafter(stop,-np.inf)*150.)
    with np.load(ROOT/c['targets']['directory']/'targets_native_100us.npz',allow_pickle=False) as a:
        for j,group in enumerate(targets['groups']):
            raw=a['g'+str(j)+'_stimulus_raw'];stim=group['stimulus_sha256']
            if hashlib.sha256(raw.tobytes()).hexdigest()!=stim:raise ContractError('Original isolated source hash differs')
            native={};rod={}
            for frame in range(first,last+1):
                index=max(0,math.floor((frame/150.-t0)/.0001))
                value=native_base*(1.+.05*(float(raw[index])-float(raw[0])))
                if not np.isfinite(value).all() or np.any(value<0):raise ContractError('Engineering native input outside domain; no clipping')
                native[frame]=value;rod[frame]=rod_base
            payload=dict(stimulus_sha256=stim,native_inputs=native,rod_inputs=rod,model_origin_s=t0,model_stop_s=stop,
                source_origin_s=0.,source_dt_s=.0001,source_samples=71000,input_driver_identity=c['launch']['fixed_input_driver']['identity'],
                clock_bridge_identity=c['launch']['fixed_clock_bridge']['identity'],native_unit='td',rod_unit='R*/s')
            digest=canonical_hash(payload);c['launch']['fixed_input_driver']['stimulus_payload_sha256'][stim]=digest
            validate_drive(payload,source,source_bundle['identity']['protocol'],targets,c)
            folder=OUT/'full_trial_preparation_v1'/('drive_g'+str(j))
            if not folder.exists():
                folder=FULL/('drive_g'+str(j));save_bundle(folder,kind='validation_evidence',identity=dict(stimulus=stim,source_samples=71000,split=group['split']),payload=payload,qualification=dict(default_adopted=False,fit_eligible=False))
            elif canonical_hash(load_bundle(folder,expected_kind='validation_evidence')['payload'])!=digest:
                raise ContractError('Reusable pure stimulus payload differs; no model trajectory reused')
            drives[stim]=str(folder);payload_hashes[stim]=digest
    c['launch']['fixed_input_driver']['stimulus_payload_sha256']=payload_hashes
    measured=read(BASE/'resource_receipt.json')['elapsed_seconds'];limit=math.ceil(3*measured*7.1/1.1)
    c['launch']['forward_resource_contract'].update(single_worker_wall_seconds=limit,rationale='3x linear projection from actual427.53s/1.1s, per-worker safety; no cumulative limit')
    write(FULL/'config.json',c)
    protocol=read(OUT/'PRE_REGISTERED.json')
    spec=dict(configuration=str(FULL/'config.json'),full_registration=str(FULL/'PRE_REGISTERED_FULL.json'),full_training_sensitivity=str(FULL/'FULL_TRAINING_SENSITIVITY.json'),
        drive_bundles=drives,output=str(OUT/'full_trial_fit_v1'),code_pins={n:sha(ROOT/n) for n in CODE})
    pins={str((FULL/'config.json').relative_to(ROOT)):sha(FULL/'config.json')}
    for folder in drives.values():pins[str((Path(folder)/'manifest.json').relative_to(ROOT))]=sha(Path(folder)/'manifest.json')
    write(FULL/'PRE_REGISTERED_FULL.json',dict(scope='FOUR_TRAINING_GROUPS_FULL_ACTIVE_WINDOW',full_source_samples=71000,active_window=[10000,70000],
        values=VALUES,bounds=BOUNDS,perturbation_factor=1.25,minimum_singular_ratio=.01,optimizer=protocol['optimizer'],
        code_pins=spec['code_pins'],input_pins=pins,source_stimulus_and_clock_unchanged=True,baseline_prefix_not_reused_as_full_run=True,
        no_validation_used_for_sensitivity=True,historical_test_waveforms_read=0,per_worker_limit_seconds=limit,default_adopted=False,FIT_READY=False))
    write(FULL/'fit_spec.json',spec);print(json.dumps(dict(status='FULL_TRIAL_PREPARED_NOT_RUN',directory=str(FULL),per_worker_limit_seconds=limit)))


def run_sensitivity():
    p=read(FULL/'PRE_REGISTERED_FULL.json');spec=read(FULL/'fit_spec.json');c=read(spec['configuration'])
    for name,digest in dict(p['code_pins'],**p['input_pins']).items():
        if sha(ROOT/name)!=digest:raise ContractError('Frozen full-trial identity changed: '+name)
    targets=load_target_contract(ROOT,c,load_arrays=False);base={};changed={key:{} for key in KEYS};cache=[]
    for j,group in enumerate(targets['groups'][:4]):
        stim=group['stimulus_sha256'];drive=load_bundle(spec['drive_bundles'][stim],expected_kind='validation_evidence')['payload']
        for label,values in [('baseline',VALUES),('BC',dict(VALUES,**{KEYS[0]:VALUES[KEYS[0]]*1.25})),('AC',dict(VALUES,**{KEYS[1]:1.25}))]:
            folder=FULL/('train_g'+str(j)+'_'+label);print(json.dumps(dict(stage='FULL_TRAINING_FORWARD',group=j,arm=label,values=values)),flush=True)
            execute(c,drive,values,folder);obs=observables(folder)
            cache.append(dict(stimulus_sha256=stim,values=values,directory=str(folder),hashes={n:sha(folder/n) for n in ['result.json','forward_raw_model_ports.npz','execution.json']}))
            destination=base if label=='baseline' else changed[KEYS[0] if label=='BC' else KEYS[1]]
            destination.update({str(j)+':'+ch:v for ch,v in obs.items()})
    result=sensitivity_report(base,changed,{k:math.log(1.25) for k in KEYS},[10000,70000],.01)
    result.update(scope=p['scope'],full_duration_training_sensitivity=True,source_protocol_sha256=sha(FULL/'PRE_REGISTERED_FULL.json'),
        code_pins=p['code_pins'],forward_cache=cache,validation_used=False,fit_steps=0,FIT_READY=False)
    write(FULL/'FULL_TRAINING_SENSITIVITY.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','sensitivity','fit','run']);args=p.parse_args()
    if args.mode=='prepare':prepare();return
    owner=dict(pid=os.getpid(),start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),mode=args.mode)
    lease=FULL/'BATCH_OWNER.json'
    with lease.open('x',encoding='utf8') as stream:json.dump(owner,stream)
    try:
        if args.mode in ['sensitivity','run']:r=run_sensitivity()
        if args.mode in ['fit','run']:
            if args.mode=='run' and not r['selected_parameters']:
                print('No time parameter resolved under the frozen numerical budget; optimizer not started');return
            run_fit(FULL/'fit_spec.json')
    except Exception as exc:
        write(FULL/'BATCH_FAILURE.json',dict(owner,error=type(exc).__name__,reason=str(exc)));raise
    finally:
        write(FULL/'BATCH_RELEASE.json',dict(owner,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),released=True))
        if read(lease)==owner:lease.unlink()

if __name__=='__main__':main()
