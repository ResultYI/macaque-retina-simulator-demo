"""Validate existing data and current memory without optical or neural execution."""
import argparse,hashlib,json,os
from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.background_checkpoint import MemoryGuard
from tools.durable_research_runtime_v1 import atomic_json


def check(base):
    start=time.perf_counter();recovery=base/'recovery_v2';recovery.mkdir(exist_ok=True)
    if (recovery/'STARTED.json').exists():raise RuntimeError('Controlled recovery already started; no automatic second attempt')
    done=json.loads((base/'VERIFIED_COMPLETED_CAPTURES.json').read_bytes());pending=json.loads((base/'PENDING_CAPTURE_JOBS.json').read_bytes())
    if len(done)!=49 or len(pending)!=15:raise RuntimeError('Frozen recovery scope changed')
    for entry in done:
        path=(ROOT/entry['path']).resolve()
        if not path.is_relative_to(base.resolve()) or not Path(str(path)+'.ready').is_file():raise RuntimeError('Completed capture marker absent')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('Previously verified capture changed')
    if pending!=json.loads((base/'pending_optics_v1/jobs.json').read_bytes()):raise RuntimeError('Pending list differs')
    for item in pending:
        if Path(item['output']).exists() or Path(item['output']+'.ready').exists():raise RuntimeError('Pending target exists; no overwrite')
    memory=MemoryGuard(24*1024**3,8*1024**3)
    if not memory.check():raise MemoryError(str(memory.stop_reason))
    shell=Path(r'C:\Users\win11-pc\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe')
    script=r'''
$since=(Get-Date).AddMinutes(-15)
$checks=@()
foreach($name in @('System','Microsoft-Windows-Resource-Exhaustion-Detector/Operational')){
 try {
  $q=@{LogName=$name;StartTime=$since}
  if($name -eq 'System'){$q.Id=@(41,6008,2004)}
  $events=@(Get-WinEvent -FilterHashtable $q -ErrorAction Stop)
  $checks+=@{log=$name;count=$events.Count;status='READ'}
 }catch{
  if($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*'){$checks+=@{log=$name;count=0;status='NO_MATCH'}}
  else{throw}
 }
}
@{events=$checks;processes=@(Get-Process -Name python,pythonw,matlab -ErrorAction SilentlyContinue | Select-Object Id,ProcessName)}|ConvertTo-Json -Depth 5
'''
    r=subprocess.run([str(shell),'-NoProfile','-Command',script],capture_output=True,check=True)
    state=json.loads(r.stdout.decode('utf-8-sig'))
    others=[p for p in state['processes'] if p['Id'] not in (os.getpid(),os.getppid())]
    if others:raise RuntimeError('Another Python/MATLAB process is present: '+json.dumps(others))
    if any(e['count'] for e in state['events']):raise RuntimeError('Current danger event; stop recovery')
    proof=dict(status='PASS',completed_verified=49,pending_absent_targets=15,memory=memory.last,
        recent_resource_checks=state['events'],no_duplicate_compute_process=True,preflight_seconds=time.perf_counter()-start,
        authorization=dict(utc='2026-10-29T23:29:13Z',user='或者继续恢复计算 接着研究',
            scope='One controlled resume; preserve guards; another hard interruption stops automatic retries'),
        historical_cause_still_inconclusive=True,no_model_or_matlab_run=True)
    atomic_json(recovery/'PREFLIGHT.json',proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);check(p.parse_args().experiment.resolve())
