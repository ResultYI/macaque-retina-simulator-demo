"""One trajectory, with clean process handoffs at existing full checkpoints."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(r'D:\macaque retinal simulator')
sys.path.insert(0,str(ROOT))
from tools.durable_research_runtime_v1 import atomic_json,process_identity

WORK=Path(r'C:\Users\win11-pc\Documents\Codex\2026-10-29\task\hc_revision')
SHELL=Path(r'C:\Users\win11-pc\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe')
STOPS=tuple(i*.5 for i in range(1,15))+(7.1,)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def controller_identity(current):
    try:return process_identity(current.pid)
    except OSError as error:
        try:current.wait(timeout=2.)
        except subprocess.TimeoutExpired:raise error
        return None


def check_receipt(directory):
    terminal=read(directory/'TERMINAL.json');debug=read(directory/'DEBUG_TERMINAL.json')
    if terminal['status']!='EXITED_0' or not debug['worker_exited'] or debug['native_failure_captured'] or debug['exit_code']!=0:
        raise RuntimeError('Native child failed; no retry or failed-child checkpoint reuse: '+str(directory))
    resource=read(directory/'RESOURCE.json')
    if resource['status']!='PASS' or resource['guard_stop'] is not None:raise RuntimeError('Child resource guard failed')
    return resource


def main(group,output):
    experiment=output.parent.parent;cache=experiment/'cache'/f'g{group}'
    plan_path=experiment/f'RESUME_SEGMENTS_g{group}.json'
    plan=read(plan_path) if plan_path.exists() else None
    if cache.exists() and not plan:raise FileExistsError('Fresh cache or explicit clean-checkpoint resume required')
    if plan and Path(plan['parent_capture_directory']).resolve()!=output.resolve():raise RuntimeError('Resume plan ownership differs')
    output.mkdir(exist_ok=False)
    atomic_json(output/'CONTROLLER.json',dict(pid=os.getpid(),identity=process_identity(os.getpid()),
        mode='one_trajectory_clean_process_handoffs',automatic_retry=False,root_cause_proven=False))
    began=time.perf_counter();status='FAILED';receipts=[];owned=[];current=None;child_dir=None
    request_path=experiment/f'SEGMENT_REQUEST_g{group}.json'
    try:
        start=0
        if plan:
            start=plan['start_index'];receipts=read(plan['prior_receipts'])
            if not 0<start<len(STOPS) or len(receipts)!=start:raise RuntimeError('Clean prefix coverage differs')
            for index,receipt in enumerate(receipts):
                if receipt['index']!=index or receipt['source_stop']!=STOPS[index]:raise RuntimeError('Prefix is not contiguous')
                check_receipt(Path(receipt['capture']))
            prefix=read(cache/f'SEGMENT_COMPLETE_{start-1:02d}.json')
            if prefix['checkpoint_manifest_sha256']!=plan['checkpoint_manifest_sha256']:raise RuntimeError('Authorized clean prefix changed')
            atomic_json(output/'RESUME_PLAN.json',plan)
        for index in range(start,len(STOPS)):
            stop=STOPS[index];suffix=plan['capture_suffix'] if plan else 'v1'
            child_dir=experiment/'capture'/f'g{group}_part{index:02d}_{suffix}'
            if child_dir.exists():raise FileExistsError('Existing native child retained: '+str(child_dir))
            request=dict(index=index,group=group,capture_directory=str(child_dir),stop_seconds=stop,
                source_start_seconds=0. if index==0 else STOPS[index-1],
                same_trajectory=True,model_clock_reset=False,original_precision_and_gates_retained=True,
                root_cause_proven=False,automatic_retry=False)
            if index:
                prior=receipts[-1];check_receipt(Path(prior['capture']))
                prefix=read(cache/f'SEGMENT_COMPLETE_{index-1:02d}.json')
                if prefix['source_seconds']!=STOPS[index-1] or sha(cache/prefix['checkpoint']/'manifest.json')!=prefix['checkpoint_manifest_sha256']:
                    raise RuntimeError('Clean handoff checkpoint identity changed')
                authorization=dict(checkpoint=prefix['checkpoint'],checkpoint_manifest_sha256=prefix['checkpoint_manifest_sha256'],
                    prefix_cache_identity=prefix['cache_identity'],capture_directory=str(child_dir),
                    predecessor_capture=prior['capture'],reason='clean normal process handoff; no failed-native prefix reuse')
                atomic_json(cache/'RESUME_AUTHORIZATION.json',authorization)
                request['authorization']=authorization
            atomic_json(request_path,request)
            args=[str(SHELL),'-NoProfile','-File',str(WORK/'wp1_capture_compiled_cache_v1.ps1'),
                  '-Group',str(group),'-OutputDirectory',str(child_dir)]
            owned.append(child_dir.name)
            with (output/f'part{index:02d}_controller.log').open('x',encoding='utf8') as log:
                current=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                while current.poll() is None:
                    if any(p.exists() for p in (experiment/'REQUEST_STOP',experiment.parent/'REQUEST_STOP',ROOT/'REQUEST_STOP',output/'STOP')):
                        atomic_json(child_dir/'STOP',dict(reason='Explicit owning stage STOP'))
                        raise RuntimeError('Explicit STOP; owned child fail-stop requested')
                    identity=controller_identity(current)
                    if identity is None:break
                    atomic_json(output/'STATE.json',dict(status='RUNNING',group=group,segment=index,
                        source_start=request['source_start_seconds'],source_stop=stop,controller=identity,
                        utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False))
                    time.sleep(5)
            if current.returncode:raise RuntimeError('Native controller failed: '+str(current.returncode))
            resource=check_receipt(child_dir)
            atomic_json(child_dir/'SEGMENT_EXECUTION_CONTRACT.json',request)
            receipts.append(dict(index=index,capture=str(child_dir),source_stop=stop,resource=resource))
            atomic_json(output/'SEGMENT_RECEIPTS.json',receipts)
        manifest=read(cache/'manifest.json')
        if manifest['status']!='COMPLETE_COMPILED_BACKEND_CELL6_CACHE' or manifest['samples']!=71001 or manifest['bins']!=7100:
            raise RuntimeError('Full original coverage required after last handoff')
        if not manifest['physical_gate_passed'] or not manifest['complete_research_restore']:raise RuntimeError('Original full cache gates failed')
        status='PASS'
    except BaseException as error:
        import traceback
        atomic_json(output/'FAILURE.json',dict(error=str(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
    finally:
        if current is not None and current.poll() is None and child_dir is not None and child_dir.exists():
            atomic_json(child_dir/'STOP',dict(reason='Owning group controller failed; no automatic retry'))
        atomic_json(output/'RESOURCE.json',dict(status=status,wall_seconds=time.perf_counter()-began,
            peak_rss_bytes=max((v['resource']['peak_rss_bytes'] for v in receipts),default=None),
            scope='maximum of serial native child peaks, not parent process RSS',guard_stop=None if status=='PASS' else 'stage_failed',
            child_receipts='SEGMENT_RECEIPTS.json',memory_limit_GiB=24,system_reserve_GiB=8,automatic_retry=False))
        atomic_json(output/'DEBUG_TERMINAL.json',dict(schema='clean-native-children-aggregate-v1',
            worker_exited=status=='PASS',native_failure_captured=any((Path(v['capture'])/'EXACT_NATIVE_EXCEPTION.json').exists() for v in receipts)
                or bool(child_dir and (child_dir/'EXACT_NATIVE_EXCEPTION.json').exists()),
            exit_code=0 if status=='PASS' else 1,automatic_retry=False))
        atomic_json(output/'TERMINAL.json',dict(status='EXITED_0' if status=='PASS' else 'WORKER_FAILED',
            wall_seconds=time.perf_counter()-began,aggregates_native_child_stages=True,
            owned_child_stages=owned,automatic_retry=False,
            science_or_precision_changed=False,root_cause_proven=False))


if __name__=='__main__':main(int(sys.argv[1]),Path(sys.argv[2]))
