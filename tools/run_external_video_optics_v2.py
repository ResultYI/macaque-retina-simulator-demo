"""One original MATLAB batch,24GiB guard, durable heartbeat and exit observations."""
import argparse,json,os
from pathlib import Path
import subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_cross_position_optical_v1 import tree_rss
from tools.durable_research_runtime_v1 import DurableRun
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.spatial_group_graph import digest


def run(out):
    start=time.perf_counter();journal=DurableRun(out/'runtime_optics','optical-supervisor')
    p=None;peak=0;minimum=None;status='FAILED';reason=None;completed=0;phase='validate';error_trace=None
    try:
        plan=json.loads((out/'OPTICAL_PLAN.json').read_bytes());jobs=json.loads((out/'jobs.json').read_bytes())
        for relative,expected in plan['source_pins'].items():
            if digest(ROOT/relative)!=expected:raise RuntimeError('Bound source changed: '+relative)
        if not jobs:raise RuntimeError('No pending jobs; do not launch MATLAB')
        if any(Path(j['output']).exists() or Path(j['output']+'.ready').exists() for j in jobs):
            raise RuntimeError('A pending target already exists; refuse overwrite or duplicate capture')
        profile=json.loads(Path(jobs[0]['configuration']).read_bytes())['profile'];env=os.environ.copy()
        for key,folder in [('MATLAB_PREFDIR','matlab_preferences'),('TEMP','matlab_temp')]:
            directory=out/folder;directory.mkdir(exist_ok=False);env[key]=str(directory)
        env['TMP']=env['TEMP'];quote=lambda path:str(path).replace("'","''")
        command=[profile['matlab_executable'],'-batch',"maxNumCompThreads(4);addpath('"+quote(ROOT/'tools')+"');export_cone_video_aperture_batch('"+quote(out/'jobs.json')+"');"]
        memory=MemoryGuard(1024**3,8*1024**3)
        if not memory.check():raise MemoryError('Insufficient system reserve before MATLAB')
        journal.heartbeat(force=True,phase='launch',guard=dict(tree_limit_bytes=24*1024**3,minimum_available_bytes=8*1024**3))
        with (out/'optical_stdout.log').open('xb') as so,(out/'optical_stderr.log').open('xb') as se:
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
            journal.child_started(p.pid);phase='capture';progress=time.perf_counter();notice=progress
            while p.poll() is None:
                rss,_=tree_rss(p.pid);peak=max(peak,rss);healthy=memory.check()
                available=memory.last['available_physical_bytes'];minimum=available if minimum is None else min(minimum,available)
                count=sum(Path(j['output']+'.ready').exists() for j in jobs)
                if count>completed:completed=count;progress=time.perf_counter()
                journal.heartbeat(status='RUNNING',phase=phase,completed_jobs=completed,total_jobs=len(jobs),
                    peak_owned_process_RSS_bytes=peak,available_physical_bytes=available)
                if peak>24*1024**3 or not healthy:raise MemoryError('24GiB tree or8GiB available-memory guard')
                if time.perf_counter()-progress>profile['maximum_export_wall_seconds']:raise TimeoutError('Original300s without completed tile batch')
                if time.perf_counter()-start>3600:raise TimeoutError('Bounded one-hour optical stage')
                if time.perf_counter()-notice>30:
                    print(json.dumps(dict(completed_jobs=completed,total_jobs=len(jobs),wall_seconds=time.perf_counter()-start,tree_peak_bytes=peak)),flush=True);notice=time.perf_counter()
                time.sleep(.5)
            if p.returncode!=0:raise RuntimeError('MATLAB optical batch failed; raw logs retained')
            completed=sum(Path(j['output']+'.ready').exists() for j in jobs)
            if completed!=len(jobs):raise RuntimeError('Incomplete optical outputs')
            status='PASS';phase='complete'
    except BaseException as exc:
        reason=str(exc);error_trace=traceback.format_exc()
        if p is not None and p.poll() is None:
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,timeout=15);p.wait(timeout=15)
        raise
    finally:
        receipt=dict(status=status,reason=reason,traceback=error_trace,phase=phase,
            wall_seconds=time.perf_counter()-start,completed_jobs=completed,
            supervisor_identity=journal.state['self_process'],child_identity=journal.state.get('child'),
            child_returncode=None if p is None else p.poll(),peak_owned_process_RSS_bytes=peak,
            minimum_available_memory_bytes=minimum,process_tree_limit_bytes=24*1024**3,
            single_MATLAB_process_tree=True,automatic_retry=False)
        with (out/'OPTICAL_RESOURCE.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
        journal.finish(status,phase=phase,exit_reason=reason,child_returncode=receipt['child_returncode'])
        print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    run(p.parse_args().experiment.resolve())
