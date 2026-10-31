"""One authorized recovery, remaining15 captures then serial frozen4s local models."""
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.durable_research_runtime_v1 import DurableRun,atomic_json
BASE=ROOT/'reports/external_video_two_positions_4s_20261029_v1'
PYTHON=ROOT/'.venv/Scripts/python.exe'


def read(path):return json.loads(path.read_bytes())


def main():
    out=BASE/'recovery_v2';proof=read(out/'PREFLIGHT.json')
    if proof['status']!='PASS':raise RuntimeError('Recovery preflight absent')
    with (out/'STARTED.json').open('x',encoding='utf8') as f:json.dump(dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),authorization=proof['authorization']),f)
    journal=DurableRun(out/'runtime_controller','recovery-controller');child=None;status='FAILED';stage='entry'
    def execute(name,command,limit):
        nonlocal child,stage
        stage=name;started=time.perf_counter();journal.event('stage_start',stage=name,command=list(map(str,command)))
        journal.heartbeat(force=True,status='RUNNING',phase=name)
        with (out/(name+'_stdout.log')).open('xb') as so,(out/(name+'_stderr.log')).open('xb') as se:
            child=subprocess.Popen(list(map(str,command)),cwd=ROOT,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
            journal.child_started(child.pid)
            while child.poll() is None:
                journal.heartbeat(phase=name)
                if time.perf_counter()-started>limit:
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,timeout=15)
                    child.wait(timeout=15);raise TimeoutError('Stage wall guard: '+name)
                time.sleep(.5)
            receipt=dict(stage=name,returncode=child.returncode,wall_seconds=time.perf_counter()-started)
            atomic_json(out/(name+'_EXIT.json'),receipt);journal.event('stage_exit',**receipt)
            if child.returncode:raise RuntimeError(name+' failed; no automatic retry')
        print(json.dumps(receipt),flush=True)
    try:
        execute('optics_remaining15',[PYTHON,ROOT/'tools/run_external_video_optics_v2.py','--experiment',BASE/'pending_optics_v1'],3700)
        new=read(BASE/'pending_optics_v1/OPTICAL_RESOURCE.json');old=read(BASE/'INTERRUPTION.json')
        jobs=read(BASE/'jobs.json')
        if new['status']!='PASS' or not all(Path(j['output']+'.ready').exists() for j in jobs):raise RuntimeError('Combined capture is not complete')
        aggregate=dict(new,status='PASS',completed_jobs=64,newly_computed_jobs=15,reused_verified_jobs=49,
            measured_recovery_seconds=new['wall_seconds'],prior_interruption=old,
            wall_seconds=new['wall_seconds']+old['optical_elapsed_conservative_upper_seconds'],
            contains_conservative_interruption_cost=True,previous_terminal_peak_unknown=True)
        with (BASE/'OPTICAL_RESOURCE.json').open('x',encoding='utf8') as f:json.dump(aggregate,f,indent=2)
        execute('collect',[PYTHON,ROOT/'tools/prepare_external_video_v1.py','--output',BASE,'--collect'],180)
        names=['run_external_first_frame_v1.py','run_external_video_v1.py','run_external_video_optics_v2.py','durable_research_runtime_v1.py','resume_external_video_v2.py']
        atomic_json(BASE/'EXECUTION_SOURCES.json',{'tools/'+n:hashlib.sha256((ROOT/'tools'/n).read_bytes()).hexdigest() for n in names})
        for position in ['near','far']:
            execute(position,[PYTHON,ROOT/'tools/run_external_video_v1.py','--preparation',BASE,
                '--position',position,'--output',BASE/(position+'_run_v1')],6100)
        execute('analysis',[PYTHON,ROOT/'tools/analyze_external_video_v1.py','--experiment',BASE],240)
        execute('plot',[Path('D:/anaconda/python.exe'),ROOT/'tools/analyze_external_video_v1.py','--experiment',BASE,'--plot-only'],180)
        execute('viewer',[PYTHON,ROOT/'tools/build_external_video_viewer_v1.py','--experiment',BASE],180)
        status='PASS';stage='complete'
    except BaseException as exc:
        if child is not None and child.poll() is None:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,timeout=15);child.wait(timeout=15)
        atomic_json(out/'FAILURE.json',dict(stage=stage,error=str(exc),traceback=traceback.format_exc(),automatic_retry=False))
        raise
    finally:
        journal.finish(status,phase=stage)
        atomic_json(out/'TERMINAL.json',dict(status=status,phase=stage,automatic_retry=False,
            child_returncode=None if child is None else child.poll(),all_stage_children_joined=True))


if __name__=='__main__':
    out=BASE/'recovery_v2'
    sys.stdout=(out/'controller_stdout.log').open('x',encoding='utf8',buffering=1)
    sys.stderr=(out/'controller_stderr.log').open('x',encoding='utf8',buffering=1)
    main()
