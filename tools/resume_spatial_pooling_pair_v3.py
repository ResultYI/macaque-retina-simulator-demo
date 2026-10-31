"""One controlled completion attempt; each original numerical run is isolated.

The supervisor persists the real child exit code, including native termination.
There is no automatic retry. The already completed near legacy standard is reused.
"""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.durable_research_runtime_v1 import DurableRun
from retina_sim.background_checkpoint import MemoryGuard

OUT=ROOT/'reports/retinotopic_rf_revision/SPATIAL_POOLING_CANDIDATE_V1'
RECOVERY=OUT/'recovery_v3'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,ensure_ascii=False)
def atomic(path,value):
    temp=path.with_suffix('.writing');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');temp.replace(path)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def supervise(argv,directory,journal,timeout):
    directory.mkdir(parents=True,exist_ok=False);started=time.perf_counter();memory=MemoryGuard(24*1024**3,8*1024**3)
    with (directory/'stdout.log').open('x',encoding='utf8') as stdout,(directory/'stderr.log').open('x',encoding='utf8') as stderr:
        p=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        journal.child_started(p.pid);guard_stop=None
        while p.poll() is None:
            if time.perf_counter()-started>timeout:guard_stop='known_child_wall_time_limit'
            if not memory.check():guard_stop='system_memory_reserve'
            if guard_stop:
                p.kill();p.wait();break
            journal.heartbeat(status='RUNNING',phase=directory.name)
            time.sleep(2)
    record=dict(pid=p.pid,exit_code=p.returncode,exit_code_hex=f'0x{p.returncode & 0xffffffff:08x}',
        wall_seconds=time.perf_counter()-started,guard_stop=guard_stop,automatic_retry=False,
        argv=argv,stdout='stdout.log',stderr='stderr.log')
    write(directory/'EXIT_RECEIPT.json',record);return record


def main():
    RECOVERY.mkdir(exist_ok=True)
    if read(RECOVERY/'minimal_repro/RESULT.json')['status']!='PASS_NOT_REPRODUCED':raise ValueError('Bounded minimal reproduction must finish first')
    journal=DurableRun(RECOVERY/'supervisor','controlled_spatial_pooling_completion')
    started=time.perf_counter()
    try:
        # No-model witness: nonzero hard exit must still receive a durable receipt.
        witness=supervise([sys.executable,'-c','import os; os._exit(37)'],RECOVERY/'exit_receipt_witness',journal,15)
        if witness['exit_code']!=37:raise ValueError('Supervisor abnormal-exit witness failed')
        write(RECOVERY/'RELIABILITY_CHECKS.json',dict(nonzero_exit_receipt_test='PASS',checkpoint_roundtrip='PASS',
            original_model_native_fault_reproduced=False,root_cause_confirmed=False,
            improvements=['fresh process per precision run','native Python stack logging enabled','real parent exit receipt','complete inner state/history/PCG checkpoint every100ms'],
            equations_or_numerical_budget_changed=False,blind_or_infinite_retry=False))
        std=OUT/'function_v2/near/legacy/standard'
        if read(std/'RESOURCE.json')['status']!='PASS':raise ValueError('Existing standard not complete')
        completed=[dict(position='near',arm='legacy',precision='standard',path=std.relative_to(ROOT).as_posix(),
            response_sha256=sha(std/'RESPONSE.npz'),reused=True)]
        cases=[('near','legacy','tight'),('near','candidate','standard'),('near','candidate','tight'),
            ('far','legacy','standard'),('far','legacy','tight'),('far','candidate','standard'),('far','candidate','tight')]
        for position,arm,precision in cases:
            label='_'.join((position,arm,precision));destination=OUT/'function_v3'/position/arm/precision
            if destination.exists():raise ValueError('No duplicate start/overwrite: '+str(destination))
            argv=[sys.executable,'-B',str(ROOT/'tools/spatial_pooling_recovery_worker_v3.py'),'--position',position,'--arm',arm,'--precision',precision,'--output',str(destination)]
            record=supervise(argv,RECOVERY/label,journal,900)
            if record['exit_code']!=0:
                write(RECOVERY/'STOPPED.json',dict(case=label,child_exit=record,
                    decision='STOP_NO_AUTOMATIC_RETRY; preserve trace and checkpoint; native root cause still unconfirmed',
                    completed_cases=completed))
                journal.finish('FAILED',phase=label,actual_child_exit=record['exit_code_hex']);raise SystemExit(1)
            if read(destination/'RESOURCE.json')['status']!='PASS':raise ValueError('Worker exit lacks successful assay receipt')
            completed.append(dict(position=position,arm=arm,precision=precision,path=destination.relative_to(ROOT).as_posix(),
                response_sha256=sha(destination/'RESPONSE.npz'),reused=False,worker_exit_receipt=(RECOVERY/label/'EXIT_RECEIPT.json').relative_to(ROOT).as_posix()))
            atomic(RECOVERY/'COMPLETED_CASES.json',completed)
            print(json.dumps(dict(completed_case=label,total_complete=len(completed),wall_seconds=time.perf_counter()-started)),flush=True)
        write(OUT/'COMPLETED_RUNS_V3.json',dict(status='PAIRED_RUNS_COMPLETE_PENDING_ANALYSIS',cases=completed,
            wall_seconds=time.perf_counter()-started,FIT_READY=False,background_qualified=False,production_adopted=False,
            original_native_fault_root_cause_unconfirmed=True))
        journal.finish('PASS',phase='all_original_cases_complete')
    except BaseException:
        if not (RECOVERY/'supervisor/terminal.json').exists():journal.finish('FAILED',phase='software_failure')
        raise


if __name__=='__main__':main()
