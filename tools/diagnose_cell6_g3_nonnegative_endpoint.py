"""One authorized affine-input endpoint diagnostic; no optimizer or source edits."""
from pathlib import Path
import argparse,copy,hashlib,json,math,os,sys,time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
THREADS={k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']}
os.environ.update(THREADS)
import numpy as np
from retina_sim.contracts import canonical_hash,ContractError
from retina_sim.group_time_storage import load_bundle,save_bundle
from retina_sim.rgc_stage1_cache import write
from retina_sim.rgc_cpu_case_isolation import resources
from tools.fit_rgc_stage1 import verify
OUT=ROOT/'reports/cell6_g3_nonnegative_endpoint_20261028_v1'
SOURCE=ROOT/'reports/layer_calibration/rgc_cbem_conditional_v1/frozen_preparation_v3/inputs.npz'
OLD=ROOT/'reports/rgc_stage1_cpu_parallel_full_v1/workers/cell6_g3/cache/cell6_g3'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def stat(a):
    a=np.asarray(a);return dict(minimum=float(a.min()),maximum=float(a.max()),mean=float(a.mean()),std=float(a.std()),peak_to_peak=float(np.ptp(a)))

def prepare():
    original=verify();case=original['cases'][3];assert case['case_id']=='cell6_g3' and case['split']=='train'
    drive=load_bundle(ROOT/case['drive_bundle'],expected_kind='validation_evidence')['payload']
    if canonical_hash(drive)!=case['drive_sha256']:raise ContractError('Original g3 drive differs')
    seed=load_bundle(ROOT/original['prepared_model']/'initial_seed',expected_kind='source_construction_seed')['payload']
    if canonical_hash(seed)!=original['source_seed_sha256']:raise ContractError('Original source seed differs')
    with np.load(SOURCE,allow_pickle=False) as z:raws=[z[f'g{i}_stimulus_values'].copy() for i in range(4)]
    for i,x in enumerate(raws):
        if sha_array(x)!=original['cases'][i]['stimulus_sha256'] or x.shape!=(71000,):raise ContractError('Training raw identity differs')
    denominator=max(float(x[0]-x.min()) for x in raws);kappa=1./denominator
    if kappa!=0.6458123207409828:raise ContractError('Previously declared training endpoint differs')
    raw_factors=[1.+kappa*(x-x[0]) for x in raws]
    if any(not np.isfinite(x).all() or np.any(x<0) for x in raw_factors):raise ContractError('Floating endpoint negative: no epsilon/clip permitted')
    B=np.asarray(seed['input_reference']['native_background_td']);native={};indexes={};allframes=[]
    for i,x in enumerate(raws):
        old=load_bundle(ROOT/original['cases'][i]['drive_bundle'],expected_kind='validation_evidence')['payload']
        indices=[max(0,math.floor((frame/150.-old['model_origin_s'])/.0001)) for frame in old['native_inputs']]
        if min(indices)<0 or max(indices)>=len(x):raise ContractError('Training frame/raw clock out of range')
        factors=1.+kappa*(x[indices]-x[0]);allframes.append(dict(case_id=f'cell6_g{i}',raw_factor_min=float(raw_factors[i].min()),actual_150Hz_factor=stat(factors)))
        if i==3:
            for frame,index in zip(old['native_inputs'],indices):
                if not np.array_equal(old['native_inputs'][frame],B*(1.+.05*(float(x[index])-float(x[0])))):raise ContractError('Original formula/clock mismatch')
                v=B*(1.+kappa*(float(x[index])-float(x[0])))
                if not np.isfinite(v).all() or np.any(v<=0):raise ContractError('Selected g3 actual light must be strictly positive')
                native[frame]=v;indexes[str(frame)]=index
    if not np.array_equal(native[min(native)],drive['native_inputs'][min(native)]):raise ContractError('Initial input changed; cannot reuse initial-state contract')
    # Input enters affine analytic R/E; the source divides by gi,Vbar and HC
    # pooled adaptation, never light. g3 actual frames are strictly positive.
    # This is input-domain admission, not a proof of future state invariance.
    newdrive=copy.deepcopy(drive);newdrive['native_inputs']=native
    newdrive['input_driver_identity']=canonical_hash(dict(original=drive['input_driver_identity'],formula='B*(1+k*(raw[j]-raw[0]))',k=kappa,training_domain_sha256=[sha_array(x) for x in raws]))
    OUT.mkdir(exist_ok=True)
    if not (OUT/'drive/manifest.json').exists():save_bundle(OUT/'drive',kind='validation_evidence',identity=dict(original_g3_drive_sha256=case['drive_sha256'],kappa=kappa),payload=newdrive,qualification=dict(default_adopted=False,fit_eligible=False))
    if canonical_hash(load_bundle(OUT/'drive',expected_kind='validation_evidence')['payload'])!=canonical_hash(newdrive):raise ContractError('Existing endpoint drive differs')
    spec=copy.deepcopy(original);spec.update(schema='single-g3-nonnegative-affine-endpoint-diagnostic-1',input_mapping='same B*(1+k*(raw[j]-raw[0])), k=training-domain endpoint, no clipping',diagnostic_only=True,parent_contract_sha256=sha(ROOT/'reports/rgc_stage1_complete_v1/CONTRACT.json'))
    candidate=copy.deepcopy(case);candidate.update(case_id='cell6_g3_endpoint',drive_bundle=str((OUT/'drive').relative_to(ROOT)),drive_sha256=canonical_hash(newdrive),source_case_id=case['case_id'])
    # Do not replace the frozen original spec or mislabel any old cache.
    spec['cases']=[candidate]
    selection=ROOT/'reports/rgc_stage1_cpu_parallel_full_v1/fit/TRAINING_SELECTION.json'
    contract=dict(status='PRE_REGISTERED_SINGLE_TRAINING_G3_ONLY',created_utc=utc(),authorization='User approved single g3 nonnegative endpoint diagnostic; no fitting or sweeps',
        kappa=kappa,kappa_hex=kappa.hex(),denominator=denominator,derivation='min over all 4 training full100us raw arrays of 1/(raw[0]-raw_min); not spike-, validation- or sampled-frame-selected',
        training_frames=allframes,full_raw_minimum=float(min(x.min() for x in raw_factors)),g3_light_min_Td=float(min(v.min() for v in native.values())),
        floating_safety='float64 full raw factors >=0 exactly, g0 raw minimum=0; all transmitted g3 150Hz frames strictly positive; no epsilon/clip/nextafter',
        zero_light_source_domain='light>=0 accepted in ReducedNative._validate; R/E analytic has no log(light) or division by light. Other positive state denominators remain original runtime guards; input admission does not prove their future validity.',
        source_input_domain_pins={n:sha(ROOT/n) for n in ['retina_sim/cone_hc_reduced_native.py','retina_sim/group_native_physics.py','retina_sim/local_patch_boundary.py','retina_sim/group_time_monitor.py']},
        old_cache_manifest_sha256=sha(OLD/'manifest.json'),original_contract_sha256=spec['parent_contract_sha256'],source_seed_sha256=original['source_seed_sha256'],
        output_selection_sha256=sha(selection),output_theta=read(selection)['theta'],model_output_parameters='unchanged frozen source; identical selected theta applied offline to old and new V, never refit',
        keep_fixed=['source gray equilibrium and all states/history/RNG','B and fixed rod input','HC/BC/AC/AII, all route gains and RGC membrane','original FP64 DOP853 tolerances/maxstep and event/observation clocks','score [1,7)s, original signed raw and observations, no lag or normalization changes'],
        new_cache_identity=canonical_hash(dict(spec=spec,case=candidate)),backend='original CPU reference',duration_seconds=7.1,
        observer='selected actual sparse-parent weighted scalar ports sampled at existing100us precommit snapshot; no extra RHS/advance, no full-layer export',
        physical_qualification=False,FIT_READY=False,optimizer_called=False,validation_used=False,historical_test_used=False,
        comparison=dict(original_NLL_unchanged=True,free_PSTH_bin_s=.020,lag_s=0.,smoothing=False,report=['V/gE/gI mean,std,peak-to-peak','actual-parent weighted cone/BC/AC inputs','fixed-theta conditional NLL','fixed-theta autonomous free PSTH','raw absolute monitor false retained and physical-domain gate'],no_new_success_threshold=True),
        cost='Old g3 interrupted record-heavy attempt reached6.9s at4777.63s wall; recovery tail additionally~538s including26GB prefix copy and overlap. Not a clean full-run timing. Plan roughly1–1.5h with strong uncertainty; compact observer avoids full-layer export. Nonlinearity/load may change cost.',
        resource_policy=dict(max_worker_RSS_bytes=17179869184,min_available_RAM_bytes=4294967296,one_worker=True,thread_environment=THREADS,single_worker_safety_seconds=original['single_worker_safety_seconds']),
        spec=spec,case=candidate,raw_frame_indexes=indexes)
    path=OUT/'CONTRACT.json'
    if path.exists():
        previous=read(path)
        for field in ['spec','case','kappa_hex','source_input_domain_pins','output_selection_sha256']:
            if previous[field]!=contract[field]:raise ContractError('Existing registered endpoint identity differs')
        contract=previous
    else:write(path,contract)
    return contract

def sha_array(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()

def observer_class(base):
    class ObservedArm(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            from tools.validate_ac_vs_timescale import calendar
            self._endpoint_clock={self.start:0}
            for end,flags in calendar(self.protocol):
                if flags['rod'] or end==self.stop:self._endpoint_clock[end]=int(round((end-self.start)/.0001))
            a=self.assembly;j=1
            self._endpoint_on=a.matrix('routes/diffuse_BC_ON_to_parasol_RGC_ON/mean_weights.npz').getrow(j)
            self._endpoint_off=a.matrix('routes/diffuse_BC_OFF_to_cross_OFF_ON_parasol_AC_v1/mean_weights.npz').getrow(j)
            self._endpoint_cone=np.asarray((self._endpoint_on@a.matrix('routes/cones_to_diffuse_BC_ON/mean_weights.npz')).toarray()).ravel()
            self._endpoint_hc=np.asarray(self._endpoint_cone@a.matrix('routes/HC_to_cones/mean_weights.npz')).ravel()
            self._endpoint_columns=[('cones',n) for n in ['E','Vis','Bh','Vbar','release']]+[('HC','Vh')]
            self._endpoint_columns += [(p,n) for p in ['diffuse_BC_ON','diffuse_BC_OFF'] for n in ['drive','generator_activation','voltage','unadapted_release','adaptation_activity','effective_release']]
            self._endpoint_columns += [('cross_OFF_ON_parasol_AC_v1',n) for n in ['gE','voltage','p','s']]+[('parasol_RGC_ON',n) for n in ['voltage','gE','gI']]
            self._endpoint_values=np.zeros((71001,len(self._endpoint_columns)));self._endpoint_recorded=0
        def snapshot(self,t,side,**kwargs):
            s=super().snapshot(t,side,**kwargs);j=self._endpoint_clock.get(t)
            if kwargs.get('queue_node') and j is not None and j==self._endpoint_recorded:
                row=[]
                for p,n in self._endpoint_columns:
                    x=np.asarray(s['ports'][p,n])
                    if p=='cones':v=x@self._endpoint_cone
                    elif p=='HC':v=x@self._endpoint_hc
                    elif p=='diffuse_BC_ON':v=x[self._endpoint_on.indices]@self._endpoint_on.data
                    elif p=='diffuse_BC_OFF':v=x[self._endpoint_off.indices]@self._endpoint_off.data
                    else:v=x[1]
                    row.append(float(v))
                if not np.isfinite(row).all():raise ContractError('Endpoint observer nonfinite, no clipping')
                self._endpoint_values[j]=row;self._endpoint_recorded+=1
            return s
        def checkpoint_payload(self,*,monitor=None,recording=None):
            rec={} if recording is None else dict(recording)
            rec['endpoint_observer']=dict(columns=self._endpoint_columns,recorded=self._endpoint_recorded,values=self._endpoint_values[:self._endpoint_recorded].copy(),semantics='first original author100us snapshot, before commit; existing dense numerical native sample, no sparse interpolation')
            return super().checkpoint_payload(monitor=monitor,recording=rec)
        @classmethod
        def from_checkpoint(cls,assembly,seed,protocol,payload,**kwargs):
            result=super().from_checkpoint(assembly,seed,protocol,payload,**kwargs);o=payload['recording']['endpoint_observer']
            if [list(x) for x in result._endpoint_columns]!=[list(x) for x in o['columns']]:raise ContractError('Selected observer mapping changed')
            result._endpoint_recorded=o['recorded'];result._endpoint_values[:o['recorded']]=o['values'];return result
    return ObservedArm

def run(resume):
    c=prepare();spec=c['spec'];case=c['case'];from retina_sim import rgc_stage1_cache as cache
    before=resources();started=time.perf_counter();cpu=time.process_time();oldarm=cache.GroupTimeArm
    if before['available_RAM_bytes']<9*1024**3 or before['commit_headroom_bytes']<5*1024**3:raise ContractError('Single-case admission requires9GiB available RAM/5GiB commit headroom, original4GiB runtime reserve retained')
    attempt=OUT/('attempt_'+utc().replace(':','').replace('.','_'));attempt.mkdir()
    write(attempt/'START.json',dict(pid=os.getpid(),utc=utc(),entry=__file__,entry_sha256=sha(__file__),contract_sha256=sha(OUT/'CONTRACT.json'),resources=before,no_other_case_or_optimizer=True,original_stops_retained=True))
    cache.GroupTimeArm=observer_class(oldarm)
    try:
        print(json.dumps(dict(event='ENDPOINT_CAPTURE_START',pid=os.getpid(),utc=utc(),kappa=c['kappa'],cache_identity=c['new_cache_identity'],output=str(OUT))),flush=True)
        manifest=cache.capture(ROOT,OUT,spec,case,resume=resume)
        point=read(OUT/'cache'/case['case_id']/'latest_checkpoint.json')
        final=load_bundle(OUT/'cache'/case['case_id']/point['directory'],expected_kind='complete_model_checkpoint')['payload']['recording']['endpoint_observer']
        if final['recorded']!=71001:raise ContractError('Selected observer missing original100us nodes')
        np.savez_compressed(OUT/'ACTUAL_WEIGHTED_PORTS.npz',time_source_s=np.arange(71001)*.0001,values=final['values'])
        write(OUT/'ACTUAL_WEIGHTED_PORTS.json',dict(columns=final['columns'],semantics=final['semantics'],source_case='g3',rows=final['recorded'],sha256=sha(OUT/'ACTUAL_WEIGHTED_PORTS.npz'),group_id=case['model_global_id'],members=2))
        write(OUT/'FORWARD_RESULT.json',dict(status='COMPLETE_SINGLE_G3_ENDPOINT_FORWARD',utc=utc(),pid=os.getpid(),wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,resources_after=resources(),manifest_sha256=sha(OUT/'cache'/case['case_id']/'manifest.json'),initial_match=manifest['initial_match'],raw_absolute_monitor=manifest['raw_absolute_monitor'],no_optimizer=True,FIT_READY=False))
        print(json.dumps(dict(event='ENDPOINT_COMPLETE',wall_seconds=time.perf_counter()-started)),flush=True)
    except BaseException as e:
        write(attempt/'FAILURE.json',dict(error=type(e).__name__,reason=str(e),utc=utc(),elapsed_seconds=time.perf_counter()-started));raise
    finally:
        cache.GroupTimeArm=oldarm
        lease=ROOT/'reports/ac_vs_redesign_v1/ACTIVE_WORKER.json'
        write(attempt/'RELEASE.json',dict(utc=utc(),pid=os.getpid(),global_lease_absent=not lease.exists(),wall_seconds=time.perf_counter()-started))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.mode=='prepare':c=prepare();print(json.dumps({k:c[k] for k in ['status','kappa','full_raw_minimum','g3_light_min_Td','new_cache_identity']}))
    else:run(args.resume)
