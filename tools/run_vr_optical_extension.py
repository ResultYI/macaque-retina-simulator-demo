"""Run only missing exact scenes with normal MATLAB and original resource caps."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def finish(directory,extension):
    import numpy as np
    from scipy.io import loadmat
    from retina_sim.contracts import ContractError,canonical_hash
    from retina_sim.split_chunk_store import write_json
    from retina_sim.useful_runtime_budget import runtime_limits
    from retina_sim.split_execution_guard import SplitExecutionGuard
    from tools.run_vr_useful_split import require_frontend
    budget,frontend=require_frontend(directory)
    plan=json.loads((extension/'plan.json').read_bytes())
    if plan['prefix_frontend_sha256']!=canonical_hash(frontend):raise ContractError('Original actual prefix changed')
    guard=SplitExecutionGuard(runtime_limits(directory,budget),output_directory=extension)
    prefix=np.load(directory/'linear_inputs/cone_expected_counts.npy',allow_pickle=False,mmap_mode='r')
    target=extension/'unique_cone_expected_counts.npy'
    if target.exists():raise FileExistsError('Preserve a prior/partial extension basis; explicit recovery required')
    basis=np.lib.format.open_memmap(target,mode='w+',dtype=np.float64,
        shape=(len(plan['unique_frame_representatives']),prefix.shape[1]))
    covered=np.zeros(prefix.shape[1],dtype=np.uint8)
    ref=json.loads((directory/'actual_global_reference.json').read_bytes())
    for tile in plan['tiles']:
        guard.require()
        record=tile['global_rows'];path=ROOT/record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ContractError('Frozen tile ownership changed')
        rows=np.load(path,allow_pickle=False,mmap_mode='r')
        if np.any(covered[rows]):raise ContractError('Duplicate optical tile ownership')
        captured=None
        if 'capture_output' in tile:
            capture=Path(tile['capture_output']).resolve()
            if not capture.is_relative_to(extension) or not Path(str(capture)+'.ready').is_file():
                raise ContractError('Extension tile is not committed in its owned root')
            raw=loadmat(capture);meta=json.loads(raw['manifest_json'].item())
            captured=raw['expected_counts']
            if (meta['schema']!='approved_vr_linear_capture_v1' or meta['request_signature']!=tile['request_signature']
                    or meta['input_policy_sha256']!=plan['input_policy_sha256']
                    or not np.array_equal(raw['q_ref_LMS'].reshape(-1),ref['q_ref_LMS_rstar_s'])
                    or captured.shape!=(tile['new_unique_scenes'],len(rows))
                    or not np.isfinite(captured).all() or np.any(captured<0)):
                raise ContractError('Actual extension tile identity/reference/layout invalid')
        for i,source in enumerate(tile['sources']):
            if source['kind']=='prefix_global_counts':values=prefix[source['column'],rows]
            elif source['kind']=='new_tile_capture' and captured is not None:values=captured[source['column']]
            else:raise ContractError('Missing exact scene capture; no approximation allowed')
            basis[i,rows]=values
        covered[rows]=1
    if not np.all(covered==1):raise ContractError('Actual extension cone coverage incomplete')
    basis.flush();del basis
    lookup=extension/'frame_to_unique.npy';np.save(lookup,np.asarray(plan['frame_to_unique'],dtype=np.int32),allow_pickle=False)
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (target,lookup)}
    receipt=dict(schema='actual-vr-optical-exact-frame-basis-1',raw_frame_range=plan['raw_frame_range'],
        input_policy_sha256=plan['input_policy_sha256'],raw_index_sha256=plan['raw_index_sha256'],
        prefix_frontend_sha256=plan['prefix_frontend_sha256'],plan_sha256=hashlib.sha256((extension/'plan.json').read_bytes()).hexdigest(),
        files=files,unique_frame_representatives=plan['unique_frame_representatives'],frame_sha256=plan['frame_sha256'],
        reference_global_sha256=plan['reference_global_sha256'],exact_scene_reuse_only=True,
        actual_capture_window_seconds=1/150.,original_750_background_frames_preserved=True,
        complete_original_background_input=plan['raw_frame_range']==[0,750],
        physical_calibration_claim=False,fit_eligible=False,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(extension/'frontend_receipt.json',receipt)
    return receipt


def main(directory,extension, *, peer_root_pid=None):
    from retina_sim.contracts import ContractError
    from retina_sim.split_chunk_store import write_json
    from retina_sim.useful_runtime_budget import runtime_limits,OVERNIGHT_SCOPE
    from retina_sim.split_execution_guard import SplitExecutionGuard
    directory=Path(directory).resolve();extension=Path(extension).resolve()
    if not extension.is_relative_to(directory/'optical_extensions'):raise ContractError('Current owned extension required')
    budget=json.loads((directory/'round_budget.json').read_bytes());authorization=runtime_limits(directory,budget)
    if authorization.get('authorized_goal')!=OVERNIGHT_SCOPE:raise ContractError('Explicit overnight goal required')
    guard=SplitExecutionGuard(authorization,stop_file=directory/'REQUEST_STOP',output_directory=extension)
    jobs=json.loads((extension/'batch.json').read_bytes());pending=[]
    for job in jobs:
        output=Path(job['output']).resolve();configuration=Path(job['configuration']).resolve()
        if not output.is_relative_to(extension) or configuration.parent!=output.parent:raise ContractError('Job escaped extension')
        if not Path(str(output)+'.ready').exists():
            if output.exists():
                preserved=output.with_name('capture_interrupted_uncommitted-'+str(time.time_ns())+'.mat')
                if preserved.parent!=output.parent or preserved.exists():raise ContractError('Unsafe partial capture preservation')
                output.rename(preserved)
            pending.append(job)
    if (extension/'frontend_receipt.json').exists():raise ContractError('Existing completed input must be reused, not rerun')
    attempt=extension/f'attempt-{time.time_ns()}';attempt.mkdir();write_json(attempt/'pending_batch.json',pending)
    status=dict(stage='RUNNING_ONLY_MISSING_ACTUAL_OPTICAL_SCENES',pid=os.getpid(),active_matlab_pid=None,
        total_jobs=len(jobs),completed_jobs=len(jobs)-len(pending),time_deadline_enforced=False)
    status_file=attempt/'status.json';write_json(status_file,status)
    with (attempt/'watchdog.log').open('xb') as log:
        watch_args=[sys.executable,str(ROOT/'tools/watch_useful_capture_process.py'),'--run-dir',str(directory),
            '--root-pid',str(os.getpid()),'--status-file',str(status_file)]
        if peer_root_pid is not None:watch_args+=['--peer-root-pid',str(peer_root_pid)]
        watcher_started=time.time()
        watcher=subprocess.Popen(watch_args,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    child=None
    try:
        guard.require(force=True)
        watch_path=directory/f'resource_watchdog_{os.getpid()}.json'
        while True:
            if watcher.poll() is not None:raise ContractError('Owned process watchdog failed before MATLAB start')
            if watch_path.exists():
                watch=json.loads(watch_path.read_bytes())
                if (watch.get('updated_unix',0)>=watcher_started and watch.get('stage')=='WATCHING_OWNED_PROCESS_TREE'
                        and watch.get('concurrent_source_root_pid')==peer_root_pid):break
            if time.time()-watcher_started>10:raise ContractError('No live resource watchdog receipt; MATLAB not started')
            time.sleep(.1)
        if pending:
            profile=json.loads((ROOT/'configurations/iset_lcd_apple_mw_v1.json').read_bytes())
            quote=lambda p:str(Path(p).resolve()).replace("'","''")
            expression="addpath('"+quote(ROOT/'tools')+"');export_vr_linear_disk_batch('"+quote(attempt/'pending_batch.json')+"');"
            with (attempt/'matlab.log').open('xb') as log:
                child=subprocess.Popen([profile['matlab_executable'],'-batch',expression],stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                status['active_matlab_pid']=child.pid;last=time.monotonic();completed=status['completed_jobs']
                while child.poll() is None:
                    guard.require()
                    count=sum(Path(j['output']+'.ready').is_file() for j in jobs)
                    if count>completed:completed=count;last=time.monotonic()
                    status.update(completed_jobs=count,updated_unix=time.time(),resources=guard.last)
                    write_json(status_file,status,replace=True)
                    if time.monotonic()-last>profile['maximum_export_wall_seconds']:
                        raise ContractError('Original per-export no-progress guard; completed input preserved')
                    time.sleep(2.)
                if child.returncode:raise ContractError('Actual missing-scene MATLAB capture failed')
        receipt=finish(directory,extension)
        status.update(stage='ACTUAL_OPTICAL_EXTENSION_COMMITTED',completed_jobs=len(jobs),receipt_files=receipt['files'])
    except Exception as exc:
        status.update(stage='OPTICAL_EXTENSION_STOPPED',exception_type=type(exc).__name__,message=str(exc));raise
    finally:
        if child is not None and child.poll() is None:child.kill();child.wait(timeout=15)
        status.update(active_matlab_pid=None,finished_unix=time.time());write_json(status_file,status,replace=True)
        write_json(attempt/'outcome.json',status)
    print(json.dumps(status),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--extension',required=True)
    parser.add_argument('--peer-root-pid',type=int)
    args=parser.parse_args();main(args.run_dir,args.extension,peer_root_pid=args.peer_root_pid)
