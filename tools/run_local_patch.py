"""Run the bounded aggregate Local Patch demo with the unchanged CPU solver.

One new stage owns preparation and three sequential workers. Historical leases
and STOP files are read-only. This is a functional experiment, not a benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.validate_group_scc_radau import (process_identity, process_alive, verified_ancestry,
    WindowsWorkerJob, worker_handshake as original_worker_handshake)
from tools.supervise_resident_validation import memory as process_memory
PATCH_FILES = ('retina_sim/local_patch.py', 'retina_sim/local_patch_graph.py',
               'retina_sim/local_patch_runner.py', 'tools/run_local_patch.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def replace_ledger(path, value):
    from retina_sim.atomic_io import replace_with_retry
    temporary = path.with_suffix('.pending.json')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    replace_with_retry(temporary, path)


def gpu_memory():
    command = ['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total', '--format=csv,noheader,nounits']
    proc = subprocess.run(command, capture_output=True, text=True, timeout=8,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if proc.returncode:
        raise RuntimeError('Whole-card resource measurement failed: ' + proc.stderr)
    rows = [x.split(',') for x in proc.stdout.strip().splitlines()]
    if len(rows) != 1:
        raise RuntimeError('An explicitly identified single GPU is required')
    return dict(uuid=rows[0][0].strip(), used_bytes=int(rows[0][1])*1024**2,
                total_bytes=int(rows[0][2])*1024**2)


class Guard:
    def __init__(self, stage, auth, *, parent=None, worker_started=None, extra_stop=None):
        from retina_sim.background_checkpoint import MemoryGuard
        self.stage, self.auth = stage, auth
        self.parent, self.worker_started, self.extra_stop = parent, worker_started, extra_stop
        self.memory = MemoryGuard(16*1024**3, 4*1024**3)
        self.last = 0.
        self.observations = []
        self.preparation_deadline = None

    def __call__(self):
        from retina_sim.group_scc_protocol import resource_check
        now = time.time()
        if self.preparation_deadline is not None and time.monotonic() >= self.preparation_deadline:
            raise RuntimeError('Preparation 300s budget reached')
        deadline=self.auth['deadline_unix']
        if deadline is None and self.auth.get('task_wall_limit_removed_by_user') is not True:
            raise RuntimeError('Missing explicit authorization for removal of task wall limit')
        if (deadline is not None and now >= deadline) or (self.stage/'REQUEST_STOP').exists():
            raise RuntimeError('Local Patch stage deadline or STOP reached')
        if self.extra_stop is not None and (self.extra_stop/'REQUEST_STOP').exists():
            raise RuntimeError('Local Patch worker STOP reached')
        if self.worker_started is not None and time.monotonic()-self.worker_started >= 300.:
            raise RuntimeError('Local Patch single-worker 300s limit reached')
        if self.parent is not None:
            if not process_alive(self.parent['pid'], self.parent['creation_ticks']):
                raise RuntimeError('Supervisor process identity changed')
        if time.monotonic()-self.last < .5:
            return
        if not self.memory.check():
            raise RuntimeError('Original RAM gate: ' + str(self.memory.stop_reason))
        gpu = gpu_memory()
        resource_check(rss=self.memory.last['rss_bytes'], available=self.memory.last['available_physical_bytes'],
                       gpu_used=gpu['used_bytes'], gpu_total=gpu['total_bytes'])
        for relative, expected in self.auth['source_pins'].items():
            if sha(ROOT/relative) != expected:
                raise RuntimeError('Source identity changed: ' + relative)
        self.observations.append(dict(unix=now, **self.memory.last, gpu=gpu))
        self.last = time.monotonic()


def worker(job_path):
    from retina_sim.local_patch import load_patch_config
    from retina_sim.local_patch_graph import load_local_patch_graph
    from retina_sim.local_patch_runner import prepare_seed, make_protocol, run_patch
    job_path = Path(job_path).resolve()
    job = read(job_path)
    stage = Path(job['stage']).resolve()
    auth_path = stage/job.get('authorization_file', 'AUTHORIZATION.json')
    if auth_path.parent != stage:
        raise RuntimeError('Authorization must belong to this stage')
    if sha(auth_path) != job['authorization_sha256']:
        raise RuntimeError('New-stage authorization identity differs')
    auth = read(auth_path)
    if 'base_authorization_sha256' in auth:
        if tuple(auth.get('repairable_new_files', ())) != PATCH_FILES:
            raise RuntimeError('Repair may change only the new patch implementation files')
        original = read(stage/'AUTHORIZATION.json')
        if (sha(stage/'AUTHORIZATION.json') != auth['base_authorization_sha256']
                or any(auth[k] != original[k] for k in ('stage_token', 'created_unix', 'deadline_unix',
                    'preparation_limit_seconds', 'total_worker_limit_seconds', 'single_worker_limit_seconds', 'resource_limits'))
                or any(auth['source_pins'].get(k) != v for k,v in original['source_pins'].items()
                       if k not in auth['repairable_new_files'])):
            raise RuntimeError('Repair amendment changes original stage protection')
    if job['stage_token'] != auth['stage_token']:
        raise RuntimeError('Wrong stage token')
    if sha(ROOT/job['config']) != job['config_sha256']:
        raise RuntimeError('Configuration identity differs')
    logdir = Path(job['worker_log_directory'])
    ownership_handshake = original_worker_handshake(logdir, job, 'patch')
    own = ownership_handshake['worker']
    ancestry = ownership_handshake['ancestry']
    write(logdir/'handshake.json', dict(**own, ancestry=ancestry,
        worker_token=job['worker_token'], job_sha256=sha(job_path), stage_token=job['stage_token']))
    started = time.monotonic()
    guard = Guard(stage, auth, parent=job['supervisor'], worker_started=started,
                  extra_stop=Path(job['run_directory']))
    status = 'FAILED'
    try:
        guard()
        config = load_patch_config(ROOT/job['config'])
        assembly = load_local_patch_graph(ROOT, job['graph_directory'], guard=guard)
        seed = prepare_seed(ROOT, assembly, config, guard=guard)
        protocol = make_protocol(seed, duration_seconds=.0011)
        result = run_patch(assembly, seed, protocol, Path(job['run_directory']), guard,
                           resume=job['resume'], stop_at_checkpoint=job['stop_at_checkpoint'])
        status = 'COMPLETE'
        return result
    finally:
        completed = status == 'COMPLETE'
        measurement_error = None
        try:
            if not guard.memory.check():
                measurement_error = str(guard.memory.stop_reason)
        except Exception as error:
            measurement_error = repr(error)
        if measurement_error is not None:
            status = 'RESOURCE_CHECK_FAILED'
        write(logdir/'worker_receipt.json', dict(status=status, wall_seconds=time.monotonic()-started,
            resource_peak=guard.memory.last, resource_observations=guard.observations,
            resource_final_measurement_error=measurement_error,
            model_mode='CPU_FP64_ORIGINAL_DOP853', source_pins=auth['source_pins']))
        if completed and measurement_error is not None:
            raise RuntimeError('Final resource check failed: ' + measurement_error)


def demo(args):
    from retina_sim.local_patch import load_patch_config
    from retina_sim.local_patch_graph import build_local_patch_graph
    repairable = list(PATCH_FILES)
    continuing = getattr(args, 'continue_stage', None)
    if continuing:
        stage = Path(continuing).resolve()
        auth = read(stage/'AUTHORIZATION.json')
        ledger_path = stage/'ledger.json'
        ledger = read(ledger_path)
        if (ledger['status'] != 'FAILED_PRESERVED' or ledger.get('active_worker') is not None
                or time.time() >= auth['deadline_unix'] or ledger['used_worker_seconds'] >= 900.
                or (stage/'REQUEST_STOP').exists()):
            raise RuntimeError('Closed, stopped, exhausted or active stage cannot be continued')
        last_worker = ledger['workers'][-1]
        last_job = read(last_worker.get('job_path', stage/last_worker['name']/'job.json'))
        failed_path = Path(last_job['run_directory'])/('resume_failure.json' if last_job['resume'] else 'failure.json')
        prior_receipt = read(Path(last_job['worker_log_directory'])/'worker_receipt.json')
        ownership_bookkeeping_error = (
            ledger.get('error') == "RuntimeError('Worker identity or completion receipt differs')"
            and last_worker.get('exit_code') == 0 and not failed_path.exists()
            and prior_receipt.get('status') == 'COMPLETE'
            and prior_receipt.get('resource_final_measurement_error') is None)
        if ownership_bookkeeping_error:
            prior_handshake = read(Path(last_job['worker_log_directory'])/'handshake.json')
            ancestry = prior_handshake.get('ancestry', [])
            if (prior_handshake.get('worker_token') != last_job['worker_token']
                    or prior_handshake.get('stage_token') != last_job['stage_token']
                    or prior_handshake.get('job_sha256') != sha(last_worker.get('job_path', stage/last_worker['name']/'job.json'))
                    or len(ancestry) < 3 or ancestry[-1] != last_job['supervisor']
                    or ancestry[0] != {k: prior_handshake[k] for k in ('pid', 'creation_ticks')}):
                raise RuntimeError('Preserved worker tokens/source job/ancestry differ; no bookkeeping repair')
            repair_basis = 'IDENTITY_UNQUALIFIED_WRAPPER_BOOKKEEPING; old results/checkpoint are not adopted or reused'
            write(stage/('IDENTITY_UNQUALIFIED_'+str(len(list(stage.glob('AUTHORIZATION_REPAIR_*.json'))))+'.json'),
                  dict(run_directory=last_job['run_directory'], checkpoint_reuse_permitted=False,
                       status='IDENTITY_UNQUALIFIED', reason=ledger['error']))
        else:
            failed = read(failed_path)
            if failed.get('model_forward_started') is not False or failed.get('error') not in ('IndexError', 'KeyError', 'TypeError', 'AttributeError', 'NameError', 'ModuleNotFoundError'):
                raise RuntimeError('Only a preserved pre-forward implementation error permits this bounded repair continuation')
            if prior_receipt.get('status') != 'FAILED' or prior_receipt.get('resource_final_measurement_error') is not None:
                raise RuntimeError('A resource failure cannot be repaired through this implementation-only continuation')
            repair_basis = 'PRE_FORWARD_IMPLEMENTATION_ERROR'
        reason = getattr(args, 'repair_reason', None)
        if not reason:
            raise RuntimeError('Explicit implementation repair reason required')
        version = len(list(stage.glob('AUTHORIZATION_REPAIR_*.json'))) + 1
        work = stage/('repair_'+str(version))
        work.mkdir(exist_ok=False)
        amended = dict(auth, base_authorization_sha256=sha(stage/'AUTHORIZATION.json'),
                       repair_reason=reason, repair_basis=repair_basis, repairable_new_files=repairable,
                       previous_ledger_sha256=sha(ledger_path), source_pins=dict(auth['source_pins']))
        for relative in repairable:
            amended['source_pins'][relative] = sha(ROOT/relative)
        authorization_file = 'AUTHORIZATION_REPAIR_'+str(version)+'.json'
        write(stage/authorization_file, amended)
        auth = amended
        started = auth['created_unix']
        ledger.update(status='PREPARING_REPAIR', current_repair=version)
        replace_ledger(ledger_path, ledger)
    else:
        stage = args.output.resolve()
        stage.mkdir(parents=True, exist_ok=False)
        work = stage
        started = time.time()
        inventory = read(ROOT/'reports/acceleration_closeout_20261026_v1/CHANGE_INVENTORY.json')
        pins = dict(inventory['current_formal_implementation_pins'])
        if len(pins) != 239:
            raise RuntimeError('Complete original 239-source identity required')
        for relative in repairable:
            pins[relative] = sha(ROOT/relative)
        auth = dict(schema='local-patch-bounded-demo-authorization-1', stage_token=uuid.uuid4().hex,
            authorized_by='User Local Retinal Patch implementation milestone request, 2026-10-26',
            new_stage_only=True, historical_lease_reused=False, created_unix=started,
            deadline_unix=started+1200., preparation_limit_seconds=300.,
            total_worker_limit_seconds=900., single_worker_limit_seconds=300.,
            source_pins=pins, resource_limits=dict(max_rss_bytes=16*1024**3,
                minimum_available_bytes=4*1024**3,max_whole_GPU_bytes=12*1024**3,
                minimum_GPU_free_bytes=512*1024**2),
            plan=['small to checkpoint', 'small new-process continuation', 'large complete short interval'],
            prohibited=['optimization', 'fitting', 'full8ms', 'background production', 'teacher'])
        authorization_file = 'AUTHORIZATION.json'
        write(stage/authorization_file, auth)
        ledger_path = stage/'ledger.json'
        ledger = dict(status='PREPARING', used_worker_seconds=0., workers=[], active_worker=None)
        write(ledger_path, ledger)
    guard = Guard(stage, auth)
    preparation_started = time.monotonic()
    previous_preparation = ledger.get('preparation_seconds', 0.)
    guard.preparation_deadline = preparation_started + 300. - previous_preparation
    try:
        for name, config in [('small', args.small), ('large', args.large)]:
            guard()
            build_local_patch_graph(ROOT, load_patch_config(config), work/('graph_'+name), guard=guard)
            if time.monotonic() >= guard.preparation_deadline:
                raise RuntimeError('Preparation 300s budget reached')
        ledger['preparation_seconds'] = previous_preparation + time.monotonic()-preparation_started
        guard.preparation_deadline = None
        supervisor = process_identity(os.getpid())
        configurations = dict(small=args.small.resolve(), large=args.large.resolve())
        for name, model, resume, stop_cp in [('small_prefix','small',False,True),
                                           ('small_restore','small',True,False),
                                           ('large_complete','large',False,False)]:
            guard()
            if ledger['used_worker_seconds'] >= 900.:
                raise RuntimeError('Cumulative worker budget reached')
            logdir = work/name
            logdir.mkdir()
            config = configurations[model]
            job = dict(stage=str(stage), stage_token=auth['stage_token'],
                authorization_file=authorization_file, authorization_sha256=sha(stage/authorization_file), worker_token=uuid.uuid4().hex,
                supervisor=supervisor, supervisor_process=supervisor, config=config.relative_to(ROOT).as_posix(),
                config_sha256=sha(config), graph_directory=str(work/('graph_'+model)),
                run_directory=str(work/('run_'+model)), worker_log_directory=str(logdir),
                resume=resume, stop_at_checkpoint=stop_cp)
            job['token'] = job['worker_token']
            write(logdir/'job.json',job)
            command=[sys.executable,'-I','-X','utf8','-B',str(Path(__file__).resolve()),
                     '--worker',str(logdir/'job.json')]
            begin=time.monotonic()
            owned = WindowsWorkerJob()
            with (logdir/'stdout.log').open('w',encoding='utf8') as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                birth = None
                try:
                    owned.bind_launcher(process)
                    birth=owned.launcher['creation_ticks']
                    ledger.update(status='RUNNING',active_worker=dict(name=name,pid=process.pid,creation_ticks=birth))
                    replace_ledger(ledger_path,ledger)
                    while process.poll() is None:
                        guard()
                        elapsed=time.monotonic()-begin
                        if elapsed>=300. or elapsed+ledger['used_worker_seconds']>=900.:
                            raise RuntimeError('Worker wall budget reached')
                        handshake_path = logdir/'patch_worker_handshake.json'
                        if owned.worker is None and handshake_path.exists():
                            acknowledgement = owned.accept(read(handshake_path), token=job['token'],
                                label='patch', supervisor=supervisor)
                            write(logdir/'patch_worker_ownership.json', acknowledgement)
                        resources = owned.resources(process._handle)
                        if resources is not None:
                            from retina_sim.group_scc_protocol import resource_check
                            resource_check(rss=max(resources['rss_bytes'], resources['peak_rss_bytes']),
                                           available=resources['available_physical_bytes'])
                        time.sleep(.25)
                except BaseException:
                    if process.poll() is None:
                        owned.terminate()
                    process.wait(timeout=15)
                    raise
                finally:
                    elapsed=time.monotonic()-begin
                    owned.close()
                    ledger['used_worker_seconds']+=elapsed
                    ledger['workers'].append(dict(name=name,wall_seconds=elapsed,exit_code=process.poll(),
                        job_path=str(logdir/'job.json'),run_directory=job['run_directory']))
                    ledger['active_worker']=None
                    replace_ledger(ledger_path,ledger)
            if process.returncode:
                raise RuntimeError('Worker failed, preserved: '+str(logdir/'stdout.log'))
            handshake = read(logdir/'handshake.json')
            receipt = read(logdir/'worker_receipt.json')
            if (owned.worker is None or handshake['pid'] != owned.worker['pid']
                    or handshake['creation_ticks'] != owned.worker['creation_ticks']
                    or handshake['worker_token'] != job['worker_token']
                    or handshake['stage_token'] != job['stage_token']
                    or handshake['job_sha256'] != sha(logdir/'job.json')
                    or receipt['status'] != 'COMPLETE'):
                raise RuntimeError('Worker identity or completion receipt differs')
            print(name+' complete',flush=True)
        guard()
        ledger['status']='COMPLETE'
    except BaseException as error:
        ledger.update(status='FAILED_PRESERVED',error=repr(error))
        raise
    finally:
        if guard.preparation_deadline is not None:
            ledger['preparation_seconds'] = previous_preparation + time.monotonic()-preparation_started
        ledger['total_wall_seconds']=time.time()-started
        ledger['final_resource']=guard.memory.last
        ledger['resource_observations']=guard.observations
        replace_ledger(ledger_path,ledger)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--small',type=Path)
    parser.add_argument('--large',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--continue-stage',type=Path,help='Append a source-pinned repair to a current, unexpired pre-forward failure; never reset budgets')
    parser.add_argument('--repair-reason')
    args=parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.small and args.large and (args.output or args.continue_stage):
        demo(args)
    else:
        parser.error('--small, --large and a new --output directory are required')


if __name__=='__main__':
    main()
