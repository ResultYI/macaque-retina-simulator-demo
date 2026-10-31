"""One-command serial preparation, capture, continuous two-position video and delivery."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.durable_research_runtime_v1 import DurableRun
PYTHON=ROOT/'.venv/Scripts/python.exe'


def read(path):return json.loads(path.read_bytes())
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')


def execute(out,name,command,limit,journal):
    start=time.perf_counter();receipt=dict(stage=name,status='RUNNING',command=list(map(str,command)))
    write(out/'PIPELINE_STATUS.json',receipt);print(json.dumps(receipt),flush=True)
    journal.event('stage_start',stage=name);journal.heartbeat(force=True,status='RUNNING',phase=name)
    with (out/(name+'_stdout.log')).open('xb') as so,(out/(name+'_stderr.log')).open('xb') as se:
        p=subprocess.Popen(list(map(str,command)),cwd=ROOT,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
        journal.child_started(p.pid)
        try:
            while p.poll() is None:
                journal.heartbeat(phase=name)
                if time.perf_counter()-start>limit:
                    subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,timeout=15)
                    raise TimeoutError('Independent finite stage wall guard: '+name)
                time.sleep(1)
            receipt.update(returncode=p.returncode,wall_seconds=time.perf_counter()-start,status='PASS' if p.returncode==0 else 'FAILED')
            write(out/(name+'_CONTROLLER.json'),receipt)
            journal.event('stage_exit',stage=name,returncode=p.returncode)
            if p.returncode:raise RuntimeError(name+' failed; exact stdout/stderr retained')
        except BaseException:
            if p.poll() is None:subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,timeout=15)
            raise


def pipeline(out,continue_capture):
    start=time.perf_counter()
    if not continue_capture:
        if out.exists():raise FileExistsError('Use a fresh output; no implicit overwrite or replay')
        subprocess.run([str(PYTHON),str(ROOT/'tools/prepare_external_video_v1.py'),'--output',str(out)],cwd=ROOT,check=True)
    if not out.is_dir() or (out/'PIPELINE_STARTED.json').exists():raise RuntimeError('Absent plan or controller already started')
    write(out/'PIPELINE_STARTED.json',dict(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        continue_only_already_running_optical=continue_capture,serial=True,no_automatic_retry=True))
    journal=DurableRun(out/'runtime_pipeline','new-video-pipeline');status='FAILED'
    try:
        if continue_capture:
            while not (out/'OPTICAL_RESOURCE.json').exists():
                journal.heartbeat(status='RUNNING',phase='awaiting_existing_optics')
                if time.perf_counter()-start>3700:raise TimeoutError('Previously running optical stage has no terminal receipt')
                time.sleep(1)
            if read(out/'OPTICAL_RESOURCE.json')['status']!='PASS':raise RuntimeError('Existing optical stage failed; no automatic restart')
        else:execute(out,'optics',[PYTHON,ROOT/'tools/run_external_video_optics_v2.py','--experiment',out],3700,journal)
        execute(out,'collect',[PYTHON,ROOT/'tools/prepare_external_video_v1.py','--output',out,'--collect'],180,journal)
        sources=['tools/run_external_first_frame_v1.py','tools/run_external_video_v1.py','tools/prepare_external_video_v1.py',
                 'tools/run_external_video_optics_v2.py','tools/durable_research_runtime_v1.py','tools/run_external_video_pipeline_v1.py']
        write(out/'EXECUTION_SOURCES.json',{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
        for position in ['near','far']:
            execute(out,position,[PYTHON,ROOT/'tools/run_external_video_v1.py','--preparation',out,
                '--position',position,'--output',out/(position+'_run_v1')],6100,journal)
        execute(out,'analysis',[PYTHON,ROOT/'tools/analyze_external_video_v1.py','--experiment',out],240,journal)
        execute(out,'plot',[Path('D:/anaconda/python.exe'),ROOT/'tools/analyze_external_video_v1.py','--experiment',out,'--plot-only'],180,journal)
        execute(out,'viewer',[PYTHON,ROOT/'tools/build_external_video_viewer_v1.py','--experiment',out],180,journal)
        execute(out,'package',[PYTHON,ROOT/'tools/export_response_bundle.py','--experiment',out,
            '--output',out/'packages'/('retina_'+out.name+'.response.zip')],900,journal)
        write(out/'PIPELINE_STATUS.json',dict(status='COMPLETE',wall_seconds=time.perf_counter()-start,
            index=str(out/'index.html'),no_scientific_qualification_upgrade=True))
        print(json.dumps(dict(status='COMPLETE',index=str(out/'index.html'))),flush=True)
        status='PASS'
    except BaseException as exc:
        write(out/'PARTIAL_RESPONSE_EXPORT.json',dict(status='partial_or_export_failed',complete_package=False,
            response_summary_exists=(out/'FINAL_RESULT.json').exists(),original_outputs_retained=True))
        write(out/'PIPELINE_FAILED.json' ,dict(error=str(exc),traceback=traceback.format_exc(),
            wall_seconds=time.perf_counter()-start,automatic_retry=False,prior_outputs_retained=True))
        raise
    finally:journal.finish(status)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'reports'/('external_video_two_positions_repeat_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    p.add_argument('--continue-existing-capture',action='store_true');a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'reports'):raise ValueError('Only an owned reports directory is allowed')
    pipeline(out,a.continue_existing_capture)
