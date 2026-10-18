"""Original sequential ISET jobs with guaranteed cleanup of the owned process tree.

Only process lifetime/error handling differs from the pinned legacy runner.
MATLAB expression, exporter, input arrays, time grid and optics are unchanged.
"""
from pathlib import Path
import os,subprocess,time,uuid
from retina_sim.contracts import ContractError
from retina_sim.disk_graph_storage import save_json


def _terminate_owned(process):
    if process.poll() is not None:return
    if os.name=='nt':
        # Popen still owns the live launcher; /T includes only its descendants.
        result=subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=15)
        if result.returncode and process.poll() is None:
            raise ContractError('Cannot stop the owned MATLAB process tree; original outputs retained')
    else:process.terminate()
    process.wait(timeout=15)


def run_batch(root,profile,jobs,directory,on_status,on_boundary):
    if not jobs:return
    directory=Path(directory);batch=directory/('batch_'+uuid.uuid4().hex+'.json');save_json(batch,jobs)
    quote=lambda path:str(Path(path).resolve()).replace("'","''")
    expression="addpath('"+quote(root/'tools')+"');export_disk_mixed_batch('"+quote(batch)+"');"
    pref=directory/'matlab_preferences';temporary=directory/'matlab_temp';pref.mkdir(exist_ok=True);temporary.mkdir(exist_ok=True)
    environment=dict(os.environ,MATLAB_PREFDIR=str(pref),TEMP=str(temporary),TMP=str(temporary))
    with (directory/('stdout_'+batch.stem+'.log')).open('wb') as out,(directory/('stderr_'+batch.stem+'.log')).open('wb') as err:
        process=subprocess.Popen([profile['matlab_executable'],'-batch',expression],env=environment,stdout=out,stderr=err,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            last_progress=time.monotonic();last_status=last_progress;done=0
            while process.poll() is None:
                if on_boundary is not None and on_boundary() is False:
                    raise ContractError('Global ISET capture stopped at requested boundary; completed tile files retained')
                completed=sum(Path(job['output']+'.ready').is_file() for job in jobs)
                if completed>done:done=completed;last_progress=time.monotonic()
                if time.monotonic()-last_progress>profile['maximum_export_wall_seconds']:
                    raise ContractError('Global ISET tile exceeded original per-export wall budget; completed tiles retained')
                if time.monotonic()-last_status>=15:
                    if on_status:on_status('ISET fixed global tiles %d/%d'%(done,len(jobs)))
                    last_status=time.monotonic()
                time.sleep(.2)
            if process.returncode:raise ContractError('Pinned global MATLAB/ISET capture failed; no RGB cone fallback: '+str(directory))
        finally:
            _terminate_owned(process)
