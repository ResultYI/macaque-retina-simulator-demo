"""Finite Local Patch v2 acceptance, one immutable plan and shared worker budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_local_patch import (Guard as OriginalGuard,read,sha,write,replace_ledger,
    process_identity,process_alive,WindowsWorkerJob,original_worker_handshake)

class Guard(OriginalGuard):
    def __init__(self,*args,role_name=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.role_name=role_name
        caps=self.auth.get('role_caps_seconds')
        if caps is not None and (role_name not in caps or not 0.<float(caps[role_name])<=180.):
            raise RuntimeError('Missing or invalid authorized worker role cap')

    def __call__(self):
        if self.worker_started is not None and time.monotonic()-self.worker_started>=180.:
            raise RuntimeError('Single worker 180s budget reached')
        if self.worker_started is not None:
            elapsed=time.monotonic()-self.worker_started
            caps=self.auth.get('role_caps_seconds')
            if caps is not None and elapsed>=caps[self.role_name]:
                raise RuntimeError('Authorized worker role wall cap reached')
            limit=self.auth.get('total_worker_limit_seconds')
            if limit is not None and elapsed+read(self.stage/'ledger.json')['used_worker_seconds']>=limit:
                raise RuntimeError('Cumulative worker wall cap reached')
        before=len(self.observations)
        super().__call__()
        if len(self.observations)!=before:
            for path,expected in self.auth.get('prior_evidence_pins',{}).items():
                if sha(Path(path))!=expected:
                    raise RuntimeError('Prior immutable evidence changed: '+path)


def validate_role(job,auth):
    caps=auth.get('role_caps_seconds')
    if caps is not None and (job.get('role_name') not in caps or
            job.get('role_wall_cap_seconds')!=caps[job['role_name']]):
        raise RuntimeError('Worker job differs from authorized role cap')
    return job.get('role_name')

def worker(path):
    from retina_sim.local_patch import load_patch_config
    from retina_sim.local_patch_graph import load_local_patch_graph
    from retina_sim.local_patch_runner import prepare_seed,make_protocol,run_patch
    job=read(path);stage=Path(job['stage']);auth=read(stage/'AUTHORIZATION.json')
    if sha(stage/'AUTHORIZATION.json')!=job['authorization_sha256'] or job['stage_token']!=auth['stage_token']:
        raise RuntimeError('Stage authorization identity differs')
    if sha(ROOT/job['config'])!=job['config_sha256']:
        raise RuntimeError('Frozen configuration differs')
    role=validate_role(job,auth)
    log=Path(job['worker_log_directory'])
    handshake=original_worker_handshake(log,job,'patch')
    write(log/'handshake.json',dict(**handshake['worker'],ancestry=handshake['ancestry'],
        worker_token=job['token'],stage_token=job['stage_token'],job_sha256=sha(path)))
    begin=time.monotonic();status='FAILED';timing={}
    guard=Guard(stage,auth,parent=job['supervisor'],worker_started=begin,extra_stop=Path(job['run_directory']),role_name=role)
    try:
        guard();t=time.monotonic()
        config=load_patch_config(ROOT/job['config'])
        assembly=load_local_patch_graph(ROOT,job['graph_directory'],guard=guard)
        timing['graph_load_seconds']=time.monotonic()-t;t=time.monotonic()
        seed=prepare_seed(ROOT,assembly,config,guard=guard)
        timing['seed_and_boundary_preparation_seconds']=time.monotonic()-t
        if job.get('verify_boundary'):
            from retina_sim.local_patch_boundary import verify_initial_boundary
            guard();t=time.monotonic()
            write(log/'initial_boundary_algebra.json',verify_initial_boundary(ROOT,assembly,seed))
            timing['initial_boundary_algebra_seconds']=time.monotonic()-t;guard()
        protocol=make_protocol(seed,duration_seconds=auth['plan']['duration_seconds'],bounded_v2=True)
        t=time.monotonic()
        run_patch(assembly,seed,protocol,Path(job['run_directory']),guard,
            resume=job['resume'],stop_at_checkpoint=job['stop_at_checkpoint'])
        timing['run_including_recording_checkpoint_seconds']=time.monotonic()-t
        status='COMPLETE'
    finally:
        write(log/'worker_receipt.json',dict(status=status,wall_seconds=time.monotonic()-begin,
            timing=timing,resource_peak=guard.memory.last,resource_observations=guard.observations,
            source_pins=auth['source_pins'],model='CPU_FP64_ORIGINAL_DOP853',
            role_name=role,role_wall_cap_seconds=auth.get('role_caps_seconds',{}).get(role),
            supervisor_identity=job['supervisor']))

def launch(stage,auth,ledger,name,model,*,resume=False,stop_cp=False):
    log=stage/name;log.mkdir();config=ROOT/auth['configs'][model]
    supervisor=process_identity(os.getpid())
    job=dict(stage=str(stage),stage_token=auth['stage_token'],authorization_sha256=sha(stage/'AUTHORIZATION.json'),
        supervisor=supervisor,supervisor_process=supervisor,token=uuid.uuid4().hex,
        config=config.relative_to(ROOT).as_posix(),config_sha256=sha(config),
        graph_directory=str(stage/('graph_'+model)),run_directory=str(stage/('run_'+model)),
        worker_log_directory=str(log),resume=resume,stop_at_checkpoint=stop_cp,
        verify_boundary=(name=='halo_030_prefix'),role_name=name,
        role_wall_cap_seconds=auth.get('role_caps_seconds',{}).get(name))
    write(log/'job.json',job)
    owned=WindowsWorkerJob();begin=time.monotonic();process=None
    guard=Guard(stage,auth,worker_started=begin,role_name=name)
    try:
        guard()
        with (log/'stdout.log').open('w',encoding='utf8') as stream:
            process=subprocess.Popen([sys.executable,'-I','-X','utf8','-B',str(Path(__file__).resolve()),'--worker',str(log/'job.json')],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            owned.bind_launcher(process)
            ledger.update(status='RUNNING',active_worker=dict(name=name,**owned.launcher));replace_ledger(stage/'ledger.json',ledger)
            while process.poll() is None:
                guard();elapsed=time.monotonic()-begin
                limit=auth.get('total_worker_limit_seconds',600.)
                if elapsed>=180. or (limit is not None and elapsed+ledger['used_worker_seconds']>=limit):
                    raise RuntimeError('Shared worker wall budget reached')
                handshake=log/'patch_worker_handshake.json'
                if owned.worker is None and handshake.exists():
                    write(log/'patch_worker_ownership.json',owned.accept(read(handshake),token=job['token'],label='patch',supervisor=supervisor))
                resources=owned.resources(process._handle)
                if resources:
                    from retina_sim.group_scc_protocol import resource_check
                    resource_check(rss=max(resources['rss_bytes'],resources['peak_rss_bytes']),available=resources['available_physical_bytes'])
                time.sleep(.25)
        if process.returncode:
            raise RuntimeError('Worker failed: '+str(log/'stdout.log'))
        h=read(log/'handshake.json');receipt=read(log/'worker_receipt.json')
        if (owned.worker is None or h['pid']!=owned.worker['pid'] or h['creation_ticks']!=owned.worker['creation_ticks']
            or h['worker_token']!=job['token'] or h['stage_token']!=job['stage_token']
            or h['job_sha256']!=sha(log/'job.json') or receipt['status']!='COMPLETE'):
            raise RuntimeError('Worker ownership/completion failed')
    finally:
        if process is not None and process.poll() is None:
            owned.terminate();process.wait(timeout=15)
        identity=owned.worker;owned.close()
        ledger['used_worker_seconds']+=time.monotonic()-begin
        ledger['workers'].append(dict(name=name,model=model,wall_seconds=time.monotonic()-begin,
            exit_code=None if process is None else process.poll(),identity=identity))
        ledger['active_worker']=None;replace_ledger(stage/'ledger.json',ledger)
    print(name+' complete',flush=True)

def demo(args):
    from retina_sim.local_patch import load_patch_config
    from retina_sim.local_patch_graph import build_local_patch_graph
    stage=args.output.resolve();stage.mkdir(parents=True,exist_ok=False)
    plan=read(args.plan);pins=read(ROOT/'reports/acceleration_closeout_20261026_v1/CHANGE_INVENTORY.json')['current_formal_implementation_pins']
    oldpins=dict(pins);changes={}
    allowed={'retina_sim/group_time_runner.py','retina_sim/group_inner_compiled.py'}
    for p,expected in oldpins.items():
        actual=sha(ROOT/p)
        if actual!=expected:
            if p not in allowed:raise RuntimeError('Unexpected original source identity change: '+p)
            changes[p]=dict(original=expected,current=actual,reason='Explicit optional assembly factories; original default/reference unchanged')
            pins[p]=actual
    for pattern in ('retina_sim/local_patch*.py','tools/run_local_patch_v2.py','tools/run_local_patch.py'):
        for p in ROOT.glob(pattern):pins[p.relative_to(ROOT).as_posix()]=sha(p)
    configs={k:Path(v).as_posix() for k,v in plan['configs'].items()}
    begin=time.time();auth=dict(schema='local-patch-v2-bounded-authorization-1',stage_token=uuid.uuid4().hex,
        authorized_by='User Local Patch v2 implementation and bounded acceptance, 2026-10-26',
        created_unix=begin,deadline_unix=begin+1200.,preparation_limit_seconds=300.,
        total_worker_limit_seconds=600.,single_worker_limit_seconds=180.,source_pins=pins,
        original_implementation_pin_changes=changes,plan=plan,configs=configs,
        resource_limits=dict(max_rss_bytes=16*1024**3,minimum_available_bytes=4*1024**3,
            max_whole_GPU_bytes=12*1024**3,minimum_GPU_free_bytes=512*1024**2),historical_lease_reused=False)
    write(stage/'AUTHORIZATION.json',auth);write(stage/'acceptance_plan.json',plan)
    ledger=dict(status='PREPARING',used_worker_seconds=0.,workers=[],active_worker=None)
    write(stage/'ledger.json',ledger);guard=Guard(stage,auth);preparation=time.monotonic()
    guard.preparation_deadline=preparation+300.
    try:
        timings={}
        for model,p in configs.items():
            guard();t=time.monotonic();assembly=build_local_patch_graph(ROOT,load_patch_config(ROOT/p),stage/('graph_'+model),guard=guard)
            timings[model]=dict(graph_build_seconds=time.monotonic()-t,summary=assembly.manifest)
        write(stage/'graph_build_receipt.json',timings)
        ledger['preparation_seconds']=time.monotonic()-preparation;guard.preparation_deadline=None
        launch(stage,auth,ledger,'reference_complete','reference')
        launch(stage,auth,ledger,'halo_030_prefix','halo_030',stop_cp=True)
        launch(stage,auth,ledger,'halo_030_restore','halo_030',resume=True)
        launch(stage,auth,ledger,'halo_060_complete','halo_060')
        guard();ledger['status']='COMPLETE'
    except BaseException as error:
        ledger.update(status='FAILED_PRESERVED',error=repr(error));raise
    finally:
        ledger['total_wall_seconds']=time.time()-begin;ledger['resource_observations']=guard.observations
        replace_ledger(stage/'ledger.json',ledger)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.worker:worker(a.worker)
    elif a.plan and a.output:demo(a)
    else:p.error('--plan and new --output are required')
if __name__=='__main__':main()
