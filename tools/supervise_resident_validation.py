"""Single-process resident validation supervisor with live Windows resource caps."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def memory(handle):
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    class Status(ctypes.Structure):
        _fields_=[('dwLength',wintypes.DWORD),('dwMemoryLoad',wintypes.DWORD)]+[(k,ctypes.c_ulonglong) for k in
            ('ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual')]
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    status=Status();status.dwLength=ctypes.sizeof(status)
    if not psapi.GetProcessMemoryInfo(wintypes.HANDLE(int(handle)),ctypes.byref(counters),counters.cb):
        raise OSError(ctypes.get_last_error(),'Cannot query owned child process memory')
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(),'Cannot query available physical memory')
    return dict(rss_bytes=int(counters.WorkingSetSize),peak_rss_bytes=int(counters.PeakWorkingSetSize),
                available_physical_bytes=int(status.ullAvailPhys))


def main(kind,name):
    if os.environ.get('RETINA_RESIDENT_VALIDATION_AUTHORIZED')!='user_finished_game_20261022':
        raise PermissionError('Explicit released gaming restriction required')
    if not re.fullmatch('[a-z0-9_]{1,48}',name):raise ValueError('Simple new run name required')
    script=ROOT/'tools'/dict(full='validate_resident_full_bounded.py',small='validate_resident_small_bounded.py',hot='benchmark_resident_from_receipt.py',native='benchmark_resident_native_pointwise.py',continuation='benchmark_resident_continuation.py')[kind]
    base=ROOT/'reports/resident_runtime_validation_20261022_v1'
    output=base/name;supervisor=base/(name+'_supervisor')
    if output.exists() or supervisor.exists():raise FileExistsError('Keep all previous run evidence')
    supervisor.mkdir(parents=True);temp=supervisor/'temp';temp.mkdir()
    snapshots=supervisor/'sources';snapshots.mkdir()
    sources={}
    paths=[script,Path(__file__),ROOT/'retina_sim/canonical_stream.py']+list((ROOT/'retina_sim').glob('resident_*.py'))+[
        ROOT/'configurations/numerical_equivalence/resident_native_fp64_v1.json',
        ROOT/'configurations/numerical_equivalence/resident_ribbon_event_rate_addendum_v1.json']
    if kind in ('hot','native','continuation'):paths.append(ROOT/'tools/validate_resident_full_bounded.py')
    if kind in ('native','continuation'):paths.append(ROOT/'tools/benchmark_resident_from_receipt.py')
    for path in paths:
        raw=path.read_bytes();relative=path.relative_to(ROOT).as_posix()
        sources[relative]=hashlib.sha256(raw).hexdigest()
        (snapshots/path.name).write_bytes(raw)
    environment=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',TEMP=str(temp),TMP=str(temp))
    flag=dict(full='--execute-full-qa',small='--execute-small-diagnostic',hot='--execute-hot-from-receipt',native='--execute-native-pointwise',continuation='--execute-continuation')[kind]
    command=[sys.executable,'-B',str(script),flag,'--output',str(output)]
    start=time.monotonic();record=dict(kind=kind,command=command,source_sha256=sources,wall_limit_seconds=900,
        max_rss_bytes=(12 if kind in ('full','hot','native','continuation') else 8)*1024**3,minimum_available_bytes=(8 if kind in ('full','hot','native','continuation') else 2)*1024**3,
        started_UTC=datetime.now(timezone.utc).isoformat(),status='STARTING')
    def save():
        target=supervisor/'supervisor.json';tmp=supervisor/'supervisor.json.tmp'
        tmp.write_text(json.dumps(record,indent=2)+'\n');os.replace(tmp,target)
    with (supervisor/'worker.log').open('w',encoding='utf8') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
        record.update(pid=child.pid,status='RUNNING');save();print(json.dumps(dict(pid=child.pid,output=str(output))),flush=True)
        while child.poll() is None:
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if child.poll() is not None:break
                record['elapsed_seconds']=time.monotonic()-start
                try:
                    record['memory']=memory(child._handle)
                    if record['memory']['peak_rss_bytes']>record['max_rss_bytes']:
                        record['stop_reason']='process_peak_rss_limit'
                    elif record['memory']['available_physical_bytes']<record['minimum_available_bytes']:
                        record['stop_reason']='available_physical_memory_reserve'
                except OSError as exc:
                    if child.poll() is not None:break
                    record['stop_reason']='resource_query_failed: '+str(exc)
                if record['elapsed_seconds']>=900:record['stop_reason']='outer_wall_limit'
                if record.get('stop_reason'):
                    child.kill();child.wait();record['status']='SUPERVISOR_STOP'
                save()
        record.update(returncode=child.returncode,elapsed_seconds=time.monotonic()-start,
                      status=record['status'] if record.get('stop_reason') else 'EXITED',
                      finished_UTC=datetime.now(timezone.utc).isoformat())
        save()
    print(json.dumps(record),flush=True)
    print((supervisor/'worker.log').read_text(encoding='utf8',errors='replace')[-14000:],flush=True)
    return child.returncode


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['full','small','hot','native','continuation'],required=True)
    parser.add_argument('--name',required=True);args=parser.parse_args()
    raise SystemExit(main(args.kind,args.name))
