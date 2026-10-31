"""Durable serial continuation: existingg0 → g1..g5 → one fit → saved-data delivery."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.cell6_halo_context_v1 import BASE
from tools.durable_research_runtime_v1 import DurableRun,atomic_json,inspect_identity

WORK=Path(r'C:\Users\win11-pc\Documents\Codex\2026-10-29\task\hc_revision')
SHELL=Path(r'C:\Users\win11-pc\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe')


def read(p):return json.loads(p.read_text(encoding='utf8'))


def main():
    out=BASE/'pipeline';out.mkdir(exist_ok=False);journal=DurableRun(out/'runtime','cell6_raw_serial_pipeline')
    began=time.perf_counter();stage='g0';receipts=[];status='FAILED';error=None
    atomic_json(out/'PIPELINE_CONTRACT.json',dict(order=['existing_g0','g1','g2','g3','g4','g5','one_training_and_fixed_evaluation','delivery'],
        single_model_worker=True,memory_limit_GiB=24,system_reserve_GiB=8,no_artificial_wall_deadline=True,
        stopped_native_fault_never_auto_retried=True,source_or_model_parameters_not_tuned=True))
    try:
        # Existing g0 owns the actual worker. Never relaunch it from a missing progress file.
        capture=BASE/'capture/g0';missing_since=None
        while not (capture/'TERMINAL.json').exists():
            if (BASE/'REQUEST_STOP').exists():raise RuntimeError('Explicit pipeline STOP')
            ready=capture/'WORKER_READY.json'
            if ready.exists():
                state=inspect_identity(read(ready)['identity'])
                if state['state'] in ('NOT_FOUND','PID_REUSED'):
                    missing_since=missing_since or time.monotonic()
                    if time.monotonic()-missing_since>20:raise RuntimeError('Existingg0 ended without durable terminal receipt; no blind retry')
                else:missing_since=None
            progress=read(BASE/'cache/g0/PROGRESS.json') if (BASE/'cache/g0/PROGRESS.json').exists() else None
            journal.heartbeat(stage='waiting_existing_g0',progress=progress);time.sleep(5)
        require_complete('g0');receipts.append(dict(stage='g0',existing_run_reused=True,terminal=read(capture/'TERMINAL.json')))
        for stage in ['g1','g2','g3','g4','g5','fit']:
            if (BASE/'REQUEST_STOP').exists():raise RuntimeError('Explicit pipeline STOP')
            log=(out/(stage+'_controller.log')).open('x',encoding='utf8');started=time.perf_counter()
            argv=[str(SHELL),'-NoProfile','-File',str(WORK/'capture_cell6_raw_v1.ps1'),'-Stage',stage]
            process=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            journal.child_started(process.pid)
            while process.poll() is None:
                if (BASE/'REQUEST_STOP').exists():raise RuntimeError('Explicit STOP while owned stage runs; worker guard will retain state')
                progress_path=BASE/'cache'/stage/'PROGRESS.json'
                progress=read(progress_path) if progress_path.exists() else None
                journal.heartbeat(stage=stage,progress=progress,controller_pid=process.pid);time.sleep(5)
            log.close();receipt=dict(stage=stage,exit_code=process.returncode,controller_wall_seconds=time.perf_counter()-started,
                terminal=read(BASE/'capture'/stage/'TERMINAL.json') if (BASE/'capture'/stage/'TERMINAL.json').exists() else None)
            receipts.append(receipt);atomic_json(out/'STAGE_RECEIPTS.json',receipts)
            if process.returncode:raise RuntimeError('Stage controller failed: '+stage)
            require_complete(stage)
        status='COMPLETE';journal.finish(status,phase='all_fixed_evaluation_complete')
    except BaseException as exc:
        error=str(exc);atomic_json(out/'FAILURE.json',dict(stage=stage,error=error,automatic_retry=False));journal.finish('FAILED',phase=stage)
    finally:
        atomic_json(out/'TERMINAL.json',dict(status=status,stage=stage,error=error,wall_seconds=time.perf_counter()-began,receipts=receipts,automatic_retry=False))
        delivery=ROOT/'tools/deliver_cell6_raw_results_v1.py'
        if delivery.exists():
            done=subprocess.run([r'D:\anaconda\python.exe',str(delivery)],cwd=ROOT,capture_output=True,text=True)
            atomic_json(out/'DELIVERY_RECEIPT.json',dict(exit_code=done.returncode,stdout=done.stdout,stderr=done.stderr))
        print(json.dumps(dict(status=status,stage=stage,error=error,output=str(BASE))),flush=True)
    if status!='COMPLETE':raise SystemExit(1)


def require_complete(stage):
    terminal=read(BASE/'capture'/stage/'TERMINAL.json')
    if terminal['status']!='INTEGRATION_STAGE_COMPLETED_UNDER_NATIVE_DEBUGGER':raise RuntimeError('Owned Native-monitored stage did not complete: '+stage)
    folder=BASE/'readout' if stage=='fit' else BASE/'cache'/stage
    if read(folder/'RESOURCE.json')['status']!='PASS':raise RuntimeError('Owned worker failed: '+stage)
    if stage=='fit':
        if read(folder/'COMPLETE.json')['groups_evaluated']!=6:raise RuntimeError('Fixed evaluation incomplete')
    elif read(folder/'manifest.json')['samples']!=71001:raise RuntimeError('Full new7.1s neural cache incomplete')


if __name__=='__main__':main()
