"""Authorized orchestration only; never changes model ROOT, state or parameters."""
from pathlib import Path
import ctypes,json,os,sys
from ctypes import wintypes as W

from .contracts import ContractError
from .rgc_stage1_cache import StageWorker


def resources():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=W.HANDLE
    class Perf(ctypes.Structure):
        _fields_=[('cb',W.DWORD)]+[(k,ctypes.c_size_t) for k in ['CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize']]+[(k,W.DWORD) for k in ['HandleCount','ProcessCount','ThreadCount']]
    class IO(ctypes.Structure):
        _fields_=[(k,ctypes.c_ulonglong) for k in ['read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes']]
    p=Perf();p.cb=ctypes.sizeof(p);io=IO()
    psapi.GetPerformanceInfo.argtypes=[ctypes.POINTER(Perf),W.DWORD]
    kernel.GetProcessIoCounters.argtypes=[W.HANDLE,ctypes.POINTER(IO)]
    if not psapi.GetPerformanceInfo(ctypes.byref(p),p.cb) or not kernel.GetProcessIoCounters(kernel.GetCurrentProcess(),ctypes.byref(io)):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(available_RAM_bytes=p.PhysicalAvailable*p.PageSize,commit_bytes=p.CommitTotal*p.PageSize,
        commit_limit_bytes=p.CommitLimit*p.PageSize,commit_headroom_bytes=(p.CommitLimit-p.CommitTotal)*p.PageSize,
        process_IO={k:int(getattr(io,k)) for k,_ in io._fields_})


def enforce_private_writes(directory):
    directory=Path(directory).resolve();blocked=[]
    def require(path,event):
        if isinstance(path,int) or path is None:return
        p=Path(os.fsdecode(path)).resolve()
        if not p.is_relative_to(directory):
            blocked.append(dict(event=event,path=str(p)))
            raise ContractError('Unauthorized shared-path write rejected: '+str(p))
    def audit(event,args):
        if event=='open':
            path,mode,flags=args
            if (isinstance(mode,str) and any(x in mode for x in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):require(path,event)
        elif event in ('os.mkdir','os.remove','os.rmdir','os.chmod','os.truncate'):require(args[0],event)
        elif event=='os.rename':require(args[0],event);require(args[1],event)
        elif event in ('os.symlink','os.link'):raise ContractError('Worker cannot create shared links')
    sys.addaudithook(audit)
    return blocked


class ScopedStageWorker(StageWorker):
    def __init__(self,root,out,phase,wall_limit,job):
        super().__init__(root,out,phase,wall_limit)
        self.global_lease=self.lease;self.lease=Path(out)/'ACTIVE_WORKER.json';self.job=job
        self.owner.update(scope='authorized CPU short case under two-slot global coordinator',session=job['session_token'],case_id=job['case_id'])

    def check(self):
        from tools.validate_group_scc_radau import process_alive
        super().check()
        if (self.out/'REQUEST_STOP').exists():raise ContractError('Owned case STOP')
        parent=json.loads(self.global_lease.read_text(encoding='utf-8'))
        if parent!=self.job['global_owner'] or not process_alive(self.job['supervisor_process']['pid'],self.job['supervisor_process']['creation_ticks']):
            raise ContractError('Case lacks its live authorized global coordinator')
