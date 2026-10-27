"""Authorized finite CPU isolation checks, never a full-capture or fit launcher."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,uuid
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.rgc_cpu_case_isolation import resources
from retina_sim.rgc_stage1_cache import write
from tools.validate_group_scc_radau import WindowsWorkerJob,process_identity

PHASE=ROOT/'reports/rgc_stage1_cpu_parallel_v1'
WRAPPERS=['retina_sim/rgc_cpu_case_isolation.py','tools/run_rgc_stage1_cpu_case.py',
          'tools/run_rgc_stage1_cpu_parallel.py','tools/merge_rgc_stage1_cpu_cache.py']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()


def compare(left,right,*,require_exact,oracle=False):
    """Use the original unit budgets and independently compare causal fields."""
    import numpy as np
    from retina_sim.group_time_monitor import frozen_budget
    from retina_sim.group_time_storage import load_bundle
    from tools.validate_ac_vs import causal_hash
    l=read(left/'trace_metadata.json');r=read(right/'trace_metadata.json')
    if l['stamps']!=r['stamps'] or l['state_units']!=r['state_units'] or l['port_units']!=r['port_units']:
        raise ContractError('Source observations, field mapping or units differ')
    spec=read(ROOT/'reports/rgc_stage1_complete_v1/CONTRACT.json')
    seed=load_bundle(ROOT/spec['prepared_model']/'initial_seed',expected_kind='source_construction_seed')['payload']
    rows=[];exact=True
    with np.load(left/'traces.npz',allow_pickle=False) as a,np.load(right/'traces.npz',allow_pickle=False) as b:
        if set(a.files)!=set(b.files):raise ContractError('State/port trace coverage differs')
        for kind,units in [('state',l['state_units']),('port',l['port_units'])]:
            for key,unit in units.items():
                name=kind+'__'+key;x=a[name];y=b[name]
                if x.shape!=y.shape:raise ContractError('Trace shape differs: '+name)
                initial=seed['states'][tuple(key.split('|'))] if kind=='state' else x[0]
                budget=frozen_budget(initial,unit)[2];delta=np.abs(x-y)
                norm=float(np.max(delta/budget,initial=0.));same=np.array_equal(x,y);exact &= same
                rows.append(dict(field=name,unit=unit,max_absolute_difference=float(np.max(delta,initial=0.)),original_budget_norm=norm,bitwise_equal=bool(same),passed=bool(norm<=1 and np.isfinite(y).all())))
    lp=load_bundle(left/('checkpoint' if oracle else 'final_checkpoint'),expected_kind='complete_model_checkpoint')['payload']
    rp=load_bundle(right/'final_checkpoint',expected_kind='complete_model_checkpoint')['payload']
    # Recording and measured wall/GC metrics differ in the old reference, but
    # every actual causal field, including the pending events, must agree.
    keys=['spike_states','clocks','input_position','observation_delay','readout_cursor']
    causal={k:causal_hash(lp[k])==causal_hash(rp[k]) for k in keys}
    controller_keys=[k for k in lp['controller'] if k!='metrics']
    causal['controller']=causal_hash({k:lp['controller'][k] for k in controller_keys})==causal_hash({k:rp['controller'][k] for k in controller_keys})
    full_exact=causal_hash(lp)==causal_hash(rp)
    passed=all(v['passed'] for v in rows) and all(causal.values()) and (not require_exact or (exact and full_exact))
    return dict(passed=bool(passed),state_fields=len(l['state_units']),port_fields=len(l['port_units']),all_trajectories_bitwise_equal=bool(exact),
        complete_causal_exact=full_exact,causal_fields_exact=causal,maximum_original_budget_norm=max(x['original_budget_norm'] for x in rows),fields=rows)


class Coordinator:
    def __init__(self,out,registration):
        self.out=out;self.registration=registration;self.session=str(uuid.uuid4());self.supervisor=process_identity(os.getpid())
        self.lease=ROOT/'reports/ac_vs_redesign_v1/ACTIVE_WORKER.json'
        self.owner=dict(pid=os.getpid(),creation_ticks=self.supervisor['creation_ticks'],session_token=self.session,
            output=str(out),scope='authorized finite CPU two-slot case checks',maximum_workers=2)
        self.samples=[];self.jobs=[]
    def check(self):
        if (ROOT/'REQUEST_STOP').exists() or (PHASE/'REQUEST_STOP').exists():raise ContractError('Authorized coordinator STOP')
        if read(self.lease)!=self.owner:raise ContractError('Global coordinator ownership changed')
        r=resources()
        if r['available_RAM_bytes']<4294967296:raise ContractError('Original4GiB available RAM reserve reached')
        return r
    def run(self,descriptors):
        if not 1<=len(descriptors)<=2:raise ContractError('Maximum two independent case slots')
        admission=self.check()
        required=4294967296+len(descriptors)*5368709120
        if admission['available_RAM_bytes']<required or admission['commit_headroom_bytes']<len(descriptors)*5368709120:
            raise ContractError('Fresh-case5GiB admission estimate plus original4GiB reserve unavailable')
        pair_started=time.perf_counter();live=[];results=[]
        try:
            for d in descriptors:
                directory=self.out/d['name'];directory.mkdir();(directory/'tmp').mkdir();(directory/'numba').mkdir()
                case=next(c for c in self.spec['cases'] if c['case_id']==d['case_id'])
                job=dict(d,directory=str(directory),phase=str(PHASE),token=str(uuid.uuid4()),session_token=self.session,
                    global_owner=self.owner,supervisor_process=self.supervisor,thread_policy=self.registration['thread_policy'],
                    computation_identity=canonical_hash(dict(spec=self.spec,case=case)),orchestration_identity=self.identity)
                write(directory/'job.json',job)
                env=os.environ.copy();env.update(job['thread_policy']);env.update(TEMP=str(directory/'tmp'),TMP=str(directory/'tmp'),
                    NUMBA_CACHE_DIR=str(directory/'numba'),PYTHONDONTWRITEBYTECODE='1')
                log=(directory/'worker.log').open('w',encoding='utf8');owned=WindowsWorkerJob();started=time.perf_counter()
                try:
                    process=subprocess.Popen([sys.executable,'-I','-X','utf8','-B',str(ROOT/'tools/run_rgc_stage1_cpu_case.py'),
                        '--job',str(directory/'job.json')],cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    owned.bind_launcher(process)
                except BaseException:
                    owned.close();log.close();raise
                item=dict(name=d['name'],directory=directory,process=process,owned=owned,log=log,started=started,accepted=False,peak_RSS_bytes=0)
                live.append(item);print(json.dumps(dict(event='launched',utc=utc(),name=d['name'],launcher_pid=process.pid)),flush=True)
            while live:
                r=self.check();active=[]
                for item in live[:]:
                    directory=item['directory'];process=item['process'];owned=item['owned']
                    handshake=directory/'case_worker_handshake.json'
                    if not item['accepted'] and handshake.exists():
                        receipt=owned.accept(read(handshake),token=read(directory/'job.json')['token'],label='case',supervisor=self.supervisor)
                        write(directory/'case_worker_ownership.json',receipt);item['accepted']=True
                    measured=owned.resources(process._handle)
                    if measured:
                        item['peak_RSS_bytes']=max(item['peak_RSS_bytes'],measured['rss_bytes'])
                        active.append(dict(name=item['name'],**measured))
                    if time.perf_counter()-item['started']>9054:raise ContractError('Original single-worker9054s safety limit reached')
                    code=process.poll()
                    if code is None:continue
                    item['log'].close();owned.close();live.remove(item)
                    receipt=dict(name=item['name'],exit_code=code,launcher_pid=process.pid,wall_seconds=time.perf_counter()-item['started'],
                        peak_RSS_bytes=item['peak_RSS_bytes'],accepted_actual_worker=item['accepted'],completed_utc=utc())
                    write(directory/'PROCESS_RECEIPT.json',receipt);self.jobs.append(receipt);results.append(receipt)
                    print(json.dumps(dict(event='completed',**receipt)),flush=True)
                    if code or not (directory/'evidence/RESULT.json').exists():raise ContractError('Owned case failed; see '+str(directory/'worker.log'))
                self.samples.append(dict(utc=utc(),**r,workers=active))
                write(self.out/'LIVE_STATUS.json',dict(utc=utc(),coordinator=self.owner,completed=self.jobs,active=[x['name'] for x in live]))
                if live:time.sleep(.25)
            return dict(wall_seconds=time.perf_counter()-pair_started,processes=results,admission=admission)
        except BaseException:
            # Cooperative stop first. Never address an unrelated process.
            (PHASE/'REQUEST_STOP').write_text('Owned short-check failure; evidence retained\n',encoding='utf8')
            deadline=time.monotonic()+15
            while any(x['process'].poll() is None for x in live) and time.monotonic()<deadline:time.sleep(.1)
            for item in live:
                if item['process'].poll() is None:item['owned'].terminate()
                item['process'].wait(timeout=10);item['owned'].close();item['log'].close()
            raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);args=p.parse_args()
    if not args.attempt.replace('_','').isalnum():raise ContractError('Fresh simple attempt name required')
    if (ROOT/'REQUEST_STOP').exists() or (PHASE/'REQUEST_STOP').exists():raise ContractError('Preserved STOP blocks this attempt')
    from tools.fit_rgc_stage1 import verify
    spec=verify();registration=read(PHASE/'PRE_REGISTERED.json')
    if sha(ROOT/'reports/rgc_stage1_complete_v1/CONTRACT.json')!=registration['CPU_contract_sha256']:raise ContractError('Original full CPU contract changed')
    if any(sha(ROOT/k)!=v for k,v in registration['source_pins'].items()):raise ContractError('Frozen CPU source pin changed')
    out=PHASE/args.attempt;out.mkdir();c=Coordinator(out,registration);c.spec=spec
    identity=dict(preregistration_sha256=sha(PHASE/'PRE_REGISTERED.json'),wrapper_pins={x:sha(ROOT/x) for x in WRAPPERS},
        scientific_spec_sha256=sha(ROOT/'reports/rgc_stage1_complete_v1/CONTRACT.json'),shared_scientific_ROOT=str(ROOT),
        science_path_relocation=False,authorized_scope='finite checks only, no full cache or fitting')
    c.identity=canonical_hash(identity);write(out/'EXECUTION_IDENTITY.json',dict(identity=identity,orchestration_identity=c.identity,started_utc=utc()))
    started=time.perf_counter();acquired=False
    try:
        with c.lease.open('x',encoding='utf8') as f:json.dump(c.owner,f)
        acquired=True;c.check()
        identity_run=c.run([dict(name='identity_g0',case_id='cell6_g0',duration_seconds=.0101)])
        equality=compare(ROOT/'reports/rgc_stage1_gpu_adaptation_v1/validation_v2/cpu',out/'identity_g0/evidence',require_exact=False,oracle=True)
        write(out/'IDENTITY_COMPARISON.json',equality)
        if not equality['passed']:raise ContractError('Original CPU isolation identity gate failed')
        def descriptor(name,g):return dict(name=name,case_id='cell6_g'+str(g),duration_seconds=.05,mid_checkpoint_seconds=.025)
        s2=c.run([descriptor('serial_g2',2)]);s3=c.run([descriptor('serial_g3',3)])
        parallel=c.run([descriptor('parallel_g2',2),descriptor('parallel_g3',3)])
        pair={str(g):compare(out/('serial_g'+str(g))/'evidence',out/('parallel_g'+str(g))/'evidence',require_exact=True) for g in [2,3]}
        write(out/'PARALLEL_COMPARISON.json',pair)
        if not all(x['passed'] for x in pair.values()):raise ContractError('Serial/parallel exact causal or numerical gate failed')
        resumed=c.run([dict(name='resume_g2',case_id='cell6_g2',duration_seconds=.05,resume_checkpoint=str(out/'serial_g2/evidence/mid_checkpoint'))])
        recovery=compare(out/'serial_g2/evidence',out/'resume_g2/evidence',require_exact=True)
        write(out/'RESTORE_COMPARISON.json',recovery)
        if not recovery['passed']:raise ContractError('New-process exact restore gate failed')
        ratio=(s2['wall_seconds']+s3['wall_seconds'])/parallel['wall_seconds']
        original_pins_unchanged=all(sha(ROOT/k)==v for k,v in registration['source_pins'].items())
        final=dict(status='FINITE_CHECKS_COMPLETE',numerical_and_causal_passed=True,throughput_screen_passed=ratio>=1.3,
            measured_single_pair_throughput_ratio=ratio,required_engineering_ratio=1.3,identity_run=identity_run,serial_g2=s2,serial_g3=s3,
            parallel_pair=parallel,resume_run=resumed,original_51_scientific_pins_unchanged=original_pins_unchanged,
            original_CPU_STOP_preserved=(ROOT/'reports/rgc_stage1_complete_v1/REQUEST_STOP').exists(),
            original_GPU_STOP_preserved=(ROOT/'reports/rgc_stage1_gpu_v1/REQUEST_STOP').exists(),
            wall_seconds=time.perf_counter()-started,ended_utc=utc(),model_workers_released=True,
            long_capture_started=False,optimizer_iterations=0,FIT_READY=False,
            scope='single finite pair and exact restore only; no full background, long trajectory or physiological upgrade')
        write(out/'FINAL_RESULT.json',final);print(json.dumps(final),flush=True)
    except BaseException as e:
        write(out/'FAILURE.json',dict(utc=utc(),error=type(e).__name__,reason=str(e),elapsed_seconds=time.perf_counter()-started,completed_jobs=c.jobs));raise
    finally:
        write(out/'RESOURCE_SAMPLES.json',c.samples)
        if acquired and c.lease.exists() and read(c.lease)==c.owner:c.lease.unlink()
        write(out/'RELEASE_RECEIPT.json',dict(utc=utc(),owner=c.owner,global_lease_released=not c.lease.exists(),no_long_work_started=True))

if __name__=='__main__':main()
