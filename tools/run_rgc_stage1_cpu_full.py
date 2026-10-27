"""Authorized two-slot original compact captures, then original RGC fit/validation."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,uuid
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
THREADS={k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS']}
os.environ.update(THREADS)
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.rgc_cpu_case_isolation import resources
from retina_sim.rgc_stage1_cache import write
from tools.validate_group_scc_radau import WindowsWorkerJob,process_identity
from tools.fit_rgc_stage1 import verify
from tools.merge_rgc_stage1_cpu_cache import inspect
from tools.run_rgc_stage1_cpu_full_case import checkpoint_identity
OUT=ROOT/'reports/rgc_stage1_cpu_parallel_full_v1'
WRAPPERS=['tools/run_rgc_stage1_cpu_full.py','tools/run_rgc_stage1_cpu_full_case.py','tools/fit_rgc_stage1_cpu_registry.py',
          'retina_sim/rgc_cpu_case_isolation.py','tools/merge_rgc_stage1_cpu_cache.py']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()

def prepare():
    spec=verify();short=read(ROOT/'reports/rgc_stage1_cpu_parallel_v1/validation_v1/FINAL_RESULT.json')
    if not short['numerical_and_causal_passed'] or not short['throughput_screen_passed']:raise ContractError('Required finite CPU checks not passed')
    reused=[inspect(ROOT/'reports/rgc_stage1_complete_v1/cache'/c['case_id'],spec,c) for c in spec['cases'][:2]]
    resume=None;reason=None
    try:
        case=spec['cases'][2];old=ROOT/'reports/rgc_stage1_complete_v1/cache'/case['case_id'];point=read(old/'latest_checkpoint.json')
        resume=checkpoint_identity(old/point['directory'],spec,case)
        if resume['source_sample']!=500:raise ContractError('Expected oldg2 complete50ms pointer differs')
    except Exception as e:reason=type(e).__name__+': '+str(e);resume=None
    contract=dict(schema='originalCPU-two-slot-full-capture-and-stage1-1',authorization='User explicitly approved official entry, remaining four7.1s groups and originalRGC fitting/validation',
        scientific_contract_sha256=sha(ROOT/'reports/rgc_stage1_complete_v1/CONTRACT.json'),scientific_source_pins=spec['code_pins'],
        case_identities={c['case_id']:canonical_hash(dict(spec=spec,case=c)) for c in spec['cases']},reused_readonly_caches=reused,
        cases_to_capture=[c['case_id'] for c in spec['cases'][2:]],g2_resume=resume,g2_fresh_start_reason=reason,
        maximum_workers=2,fresh_process_per_case=True,thread_policy=THREADS,source_ROOT=str(ROOT),no_science_ROOT_relocation=True,
        old_CPU_and_GPU_STOPs_retained=True,compact_original_capture=True,original_spec_not_modified=True,original_training_validation_unchanged=True,
        shared_write_audit='each child writes only within its own case workspace',per_process_RAM_limit_bytes=17179869184,
        system_available_reserve_bytes=4294967296,estimated_case_admission_bytes=5368709120,
        original_single_worker_safety_seconds=spec['single_worker_safety_seconds'],history_test_reopened=False,GPU_used=False,FIT_READY=False)
    OUT.mkdir(exist_ok=True)
    if (OUT/'RUN_CONTRACT.json').exists():
        if read(OUT/'RUN_CONTRACT.json')!=contract:raise ContractError('Existing formal run contract differs; preserve old identity')
    else:write(OUT/'RUN_CONTRACT.json',contract)
    return spec,contract

class Coordinator:
    def __init__(self,attempt,spec,contract):
        self.attempt=attempt;self.spec=spec;self.contract=contract;self.session=str(uuid.uuid4());self.supervisor=process_identity(os.getpid())
        self.lease=ROOT/'reports/ac_vs_redesign_v1/ACTIVE_WORKER.json';self.samples=[];self.completed=[]
        self.owner=dict(pid=os.getpid(),creation_ticks=self.supervisor['creation_ticks'],session_token=self.session,
            output=str(attempt),scope='authorized original compactCPU remaining four groups and RGC output fit',maximum_workers=2)
        self.identity=canonical_hash(dict(contract_sha256=sha(OUT/'RUN_CONTRACT.json'),wrapper_pins={x:sha(ROOT/x) for x in WRAPPERS}))
    def check(self):
        if (ROOT/'REQUEST_STOP').exists() or (OUT/'REQUEST_STOP').exists() or (self.attempt/'REQUEST_STOP').exists():raise ContractError('Authorized formal-run STOP')
        if read(self.lease)!=self.owner:raise ContractError('Formal global coordinator ownership changed')
        r=resources()
        if r['available_RAM_bytes']<4294967296:raise ContractError('Original4GiB system RAM reserve reached')
        return r
    def registry(self):
        rows=[];missing=[]
        for c in self.spec['cases']:
            path=ROOT/'reports/rgc_stage1_complete_v1/cache'/c['case_id'] if c in self.spec['cases'][:2] else OUT/'workers'/c['case_id']/'cache'/c['case_id']
            if not (path/'manifest.json').exists():missing.append(c['case_id']);continue
            rows.append(inspect(path,self.spec,c))
        registry=dict(schema='same-scientific-contract-cpu-cache-registry-1',caches=rows,missing=missing,all_six_complete=not missing,
            original_spec_sha256=self.contract['scientific_contract_sha256'],short_prefixes_accepted=False,GPU_caches_accepted=False,FIT_READY=False)
        write(OUT/'CACHE_REGISTRY.json',registry);return registry
    def pair(self,case_ids):
        if not case_ids:return
        admission=self.check();n=len(case_ids)
        if n>2 or admission['available_RAM_bytes']<4294967296+n*5368709120 or admission['commit_headroom_bytes']<n*5368709120:
            raise ContractError('Two-slot5GiB/case RAM and commit admission failed')
        live=[];pair_start=time.perf_counter();pair_receipts=[]
        try:
            for case_id in case_ids:
                private=OUT/'workers'/case_id;private.mkdir(parents=True,exist_ok=True);directory=private/self.attempt.name;directory.mkdir()
                (directory/'tmp').mkdir();(directory/'numba').mkdir()
                case=next(c for c in self.spec['cases'] if c['case_id']==case_id)
                job=dict(case_id=case_id,directory=str(directory),private_root=str(private),phase=str(self.attempt),
                    token=str(uuid.uuid4()),session_token=self.session,supervisor_process=self.supervisor,global_owner=self.owner,
                    computation_identity=self.contract['case_identities'][case_id],orchestration_identity=self.identity,thread_policy=THREADS,
                    resume_source=self.contract['g2_resume']['directory'] if case_id=='cell6_g2' and self.contract['g2_resume'] else None)
                write(directory/'job.json',job);env=os.environ.copy();env.update(THREADS)
                env.update(TEMP=str(directory/'tmp'),TMP=str(directory/'tmp'),NUMBA_CACHE_DIR=str(directory/'numba'),PYTHONDONTWRITEBYTECODE='1')
                log=(directory/'worker.log').open('w',encoding='utf8');owned=WindowsWorkerJob();started=time.perf_counter()
                try:
                    process=subprocess.Popen([sys.executable,'-I','-X','utf8','-B',str(ROOT/'tools/run_rgc_stage1_cpu_full_case.py'),'--job',str(directory/'job.json')],
                        cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    owned.bind_launcher(process)
                except BaseException:owned.close();log.close();raise
                live.append(dict(case_id=case_id,directory=directory,private=private,owned=owned,process=process,log=log,started=started,accepted=False,peak_RSS_bytes=0))
                print(json.dumps(dict(event='FULL_CAPTURE_LAUNCHED',utc=utc(),case_id=case_id,launcher_pid=process.pid,job=str(directory/'job.json'))),flush=True)
            last_sample=0.
            while live:
                r=self.check();now=time.perf_counter();active=[]
                for item in live[:]:
                    d=item['directory'];p=item['process'];owned=item['owned'];handshake=d/'case_worker_handshake.json'
                    if not item['accepted'] and handshake.exists():
                        receipt=owned.accept(read(handshake),token=read(d/'job.json')['token'],label='case',supervisor=self.supervisor)
                        write(d/'case_worker_ownership.json',receipt);item['accepted']=True
                    measured=owned.resources(p._handle)
                    if measured:item['peak_RSS_bytes']=max(item['peak_RSS_bytes'],measured['rss_bytes'])
                    cache=item['private']/'cache'/item['case_id'];progress=read(cache/'progress.json') if (cache/'progress.json').exists() else None
                    active.append(dict(case_id=item['case_id'],launcher_pid=p.pid,actual_worker=owned.worker,elapsed_seconds=now-item['started'],progress=progress,resources=measured))
                    if now-item['started']>self.spec['single_worker_safety_seconds']:raise ContractError('Original single-worker runaway guard reached')
                    code=p.poll()
                    if code is None:continue
                    item['log'].close();owned.close();live.remove(item)
                    receipt=dict(case_id=item['case_id'],exit_code=code,launcher_pid=p.pid,actual_worker=owned.worker,wall_seconds=time.perf_counter()-item['started'],
                        peak_RSS_bytes=item['peak_RSS_bytes'],ended_utc=utc())
                    write(d/'PROCESS_RECEIPT.json',receipt);pair_receipts.append(receipt);self.completed.append(receipt)
                    if code or not (d/'RESULT.json').exists():raise ContractError('Owned full capture failed; see '+str(d/'worker.log'))
                    self.registry();print(json.dumps(dict(event='FULL_CAPTURE_COMPLETE',**receipt)),flush=True)
                status=dict(utc=utc(),coordinator=self.owner,orchestration_identity=self.identity,completed_this_attempt=self.completed,
                    active=active,system_resources=r,remaining_cases=self.registry_missing_fast(),phase='CAPTURING_FIXED_INTERNAL_ENDOGENOUS_TRAJECTORIES')
                write(OUT/'LIVE_STATUS.json',status)
                if now-last_sample>=5:
                    self.samples.append(status);last_sample=now
                    with (self.attempt/'RESOURCE_SAMPLES.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(status)+'\n')
                if live:time.sleep(1)
            write(self.attempt/('PAIR_'+case_ids[0]+'_RESULT.json'),dict(wall_seconds=time.perf_counter()-pair_start,processes=pair_receipts,admission=admission))
        except BaseException:
            (self.attempt/'REQUEST_STOP').write_text('Owned formal attempt stopped; preserve checkpoints and evidence\n',encoding='utf8')
            deadline=time.monotonic()+20
            while any(x['process'].poll() is None for x in live) and time.monotonic()<deadline:time.sleep(.1)
            for item in live:
                if item['process'].poll() is None:item['owned'].terminate()
                item['process'].wait(timeout=10);item['owned'].close();item['log'].close()
            raise
    def registry_missing_fast(self):
        return [c['case_id'] for c in self.spec['cases'][2:] if not (OUT/'workers'/c['case_id']/'cache'/c['case_id']/'manifest.json').exists()]

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--resume',action='store_true');args=p.parse_args()
    spec,contract=prepare()
    if args.mode=='prepare':print(json.dumps(dict(status='FORMAL_ENTRY_PREPARED',g2_resume=contract['g2_resume'],reused=[r['case_id'] for r in contract['reused_readonly_caches']])));return
    if (ROOT/'REQUEST_STOP').exists() or (OUT/'REQUEST_STOP').exists():raise ContractError('Formal run STOP retained')
    attempts=OUT/'attempts';attempts.mkdir(exist_ok=True);existing=sorted(attempts.glob('attempt_*'))
    if existing and not args.resume:raise ContractError('Prior formal attempt exists; explicit --resume required')
    attempt=attempts/('attempt_'+str(len(existing)+1).zfill(4));attempt.mkdir();c=Coordinator(attempt,spec,contract)
    started=time.perf_counter();acquired=False
    write(attempt/'EXECUTION_IDENTITY.json',dict(orchestration_identity=c.identity,wrapper_pins={x:sha(ROOT/x) for x in WRAPPERS},contract_sha256=sha(OUT/'RUN_CONTRACT.json'),owner=c.owner,started_utc=utc()))
    try:
        with c.lease.open('x',encoding='utf8') as f:json.dump(c.owner,f)
        acquired=True;c.check();reg=c.registry();remaining=[x for x in contract['cases_to_capture'] if x in reg['missing']]
        while remaining:
            admission=c.check()
            dual=(len(remaining)>1 and admission['available_RAM_bytes']>=15032385536 and admission['commit_headroom_bytes']>=10737418240)
            slots=2 if dual else 1
            write(attempt/('SLOT_ADMISSION_'+remaining[0]+'.json'),dict(resources=admission,selected_slots=slots,
                maximum_authorized_slots=2,reason='dual admission met' if dual else 'retain original limits; single fresh case admission only'))
            print(json.dumps(dict(event='SLOT_ADMISSION',selected_slots=slots,available_RAM_bytes=admission['available_RAM_bytes'],
                commit_headroom_bytes=admission['commit_headroom_bytes'])),flush=True)
            c.pair(remaining[:slots]);remaining=remaining[slots:]
        reg=c.registry()
        if not reg['all_six_complete']:raise ContractError('Incomplete full endogenous capture registry')
        verify();write(OUT/'LIVE_STATUS.json',dict(utc=utc(),coordinator=c.owner,phase='RGC_TRAINING_SELECTION_AND_VALIDATION',active=[],completed_registry=str(OUT/'CACHE_REGISTRY.json')))
        from tools.fit_rgc_stage1_cpu_registry import run
        result=run(OUT/'CACHE_REGISTRY.json',OUT/'fit',c.check)
        write(OUT/'FINAL_RESULT.json',dict(status='CONDITIONAL_CELL6_STAGE1_COMPLETE',accepted=result['accepted'],all_six_caches_complete=True,
            fit_result=str(OUT/'fit/CELL6_RESULT.json'),old_test_not_reused=True,internal_parameters_changed=False,
            other_RGC_types_not_claimed_fitted=True,wall_this_attempt_seconds=time.perf_counter()-started,ended_utc=utc(),FIT_READY=False))
        print(json.dumps(dict(event='STAGE1_RESULT',accepted=result['accepted'])),flush=True)
    except BaseException as e:
        write(attempt/'FAILURE.json',dict(utc=utc(),error=type(e).__name__,reason=str(e),wall_seconds=time.perf_counter()-started));raise
    finally:
        if acquired and c.lease.exists() and read(c.lease)==c.owner:c.lease.unlink()
        write(attempt/'RELEASE_RECEIPT.json',dict(utc=utc(),owner=c.owner,global_lease_released=not c.lease.exists(),owned_children_released=True))

if __name__=='__main__':main()
