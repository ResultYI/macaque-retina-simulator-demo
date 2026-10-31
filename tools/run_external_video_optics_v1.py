"""One existing MATLAB batch at a time, with original progress timeout and24GiB guard."""
import argparse,json,os
from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_cross_position_optical_v1 import tree_rss
from retina_sim.background_checkpoint import MemoryGuard


def run(out):
    plan=json.loads((out/'OPTICAL_PLAN.json').read_bytes());jobs=json.loads((out/'jobs.json').read_bytes())
    for relative,expected in plan['source_pins'].items():
        from retina_sim.spatial_group_graph import digest
        if digest(ROOT/relative)!=expected:raise RuntimeError('Bound source changed: '+relative)
    profile=json.loads(Path(jobs[0]['configuration']).read_bytes())['profile']
    env=os.environ.copy()
    for key,folder in [('MATLAB_PREFDIR','matlab_preferences'),('TEMP','matlab_temp')]:
        p=out/folder;p.mkdir(exist_ok=False);env[key]=str(p)
    env['TMP']=env['TEMP'];quote=lambda p:str(p).replace("'","''")
    command=[profile['matlab_executable'],'-batch',"maxNumCompThreads(4);addpath('"+quote(ROOT/'tools')+"');export_cone_video_aperture_batch('"+quote(out/'jobs.json')+"');"]
    memory=MemoryGuard(1024**3,8*1024**3);start=time.perf_counter();peak=0;minimum=2**63;status='FAILED';reason=None;completed=0
    if not memory.check():raise MemoryError('Insufficient system reserve before MATLAB')
    with (out/'optical_stdout.log').open('xb') as so,(out/'optical_stderr.log').open('xb') as se:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
        progress=time.perf_counter();notice=progress
        try:
            while p.poll() is None:
                rss,_=tree_rss(p.pid);peak=max(peak,rss);healthy=memory.check()
                minimum=min(minimum,memory.last['available_physical_bytes'])
                count=sum(Path(j['output']+'.ready').exists() for j in jobs)
                if count>completed:completed=count;progress=time.perf_counter()
                if peak>24*1024**3 or not healthy:raise MemoryError('24GiB tree or8GiB available-memory guard')
                if time.perf_counter()-progress>profile['maximum_export_wall_seconds']:raise TimeoutError('Original300s without completed tile batch')
                if time.perf_counter()-start>3600:raise TimeoutError('Bounded one-hour optical stage')
                if time.perf_counter()-notice>30:
                    print(json.dumps(dict(completed_jobs=completed,total_jobs=len(jobs),wall_seconds=time.perf_counter()-start,tree_peak_bytes=peak)),flush=True);notice=time.perf_counter()
                time.sleep(.5)
            if p.returncode!=0:raise RuntimeError('MATLAB optical batch failed; raw logs retained')
            completed=sum(Path(j['output']+'.ready').exists() for j in jobs)
            if completed!=len(jobs):raise RuntimeError('Incomplete optical outputs')
            status='PASS'
        except BaseException as exc:
            reason=str(exc)
            if p.poll() is None:subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,timeout=15)
            raise
        finally:
            receipt=dict(status=status,reason=reason,wall_seconds=time.perf_counter()-start,completed_jobs=completed,
                peak_owned_process_RSS_bytes=peak,minimum_available_memory_bytes=minimum,
                process_tree_limit_bytes=24*1024**3,single_MATLAB_process_tree=True,automatic_retry=False)
            (out/'OPTICAL_RESOURCE.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    run(p.parse_args().experiment.resolve())
