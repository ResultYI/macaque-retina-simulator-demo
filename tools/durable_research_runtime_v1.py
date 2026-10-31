"""Durable execution observations, without detached jobs or automatic retries."""
from datetime import datetime,timezone
import ctypes
from ctypes import wintypes as w
import json,os
from pathlib import Path
import time,uuid


def utc() -> str:return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path,value: object) -> None:
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    deadline=time.monotonic()+2.
    while True:
        try:
            os.replace(temporary,path)
            return
        except PermissionError as error:
            if os.name!='nt' or getattr(error,'winerror',None) not in (5,32,33) or time.monotonic()>=deadline:
                raise
            time.sleep(.02)


def process_identity(pid: int) -> dict:
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenProcess.restype=w.HANDLE
    kernel.GetProcessTimes.argtypes=[w.HANDLE,*([ctypes.POINTER(w.FILETIME)]*4)]
    kernel.GetExitCodeProcess.argtypes=[w.HANDLE,ctypes.POINTER(w.DWORD)]
    kernel.QueryFullProcessImageNameW.argtypes=[w.HANDLE,w.DWORD,w.LPWSTR,ctypes.POINTER(w.DWORD)]
    kernel.IsProcessInJob.argtypes=[w.HANDLE,w.HANDLE,ctypes.POINTER(w.BOOL)]
    kernel.CloseHandle.argtypes=[w.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        error=ctypes.get_last_error()
        return dict(pid=pid,state='NOT_FOUND' if error==87 else 'UNKNOWN',winerror=error)
    try:
        creation,exit_time,kernel_time,user_time=(w.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle,ctypes.byref(creation),ctypes.byref(exit_time),ctypes.byref(kernel_time),ctypes.byref(user_time)):
            raise ctypes.WinError(ctypes.get_last_error())
        code=w.DWORD()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
        image=ctypes.create_unicode_buffer(32768);length=w.DWORD(len(image))
        if not kernel.QueryFullProcessImageNameW(handle,0,image,ctypes.byref(length)):raise ctypes.WinError(ctypes.get_last_error())
        in_job=w.BOOL()
        if not kernel.IsProcessInJob(handle,None,ctypes.byref(in_job)):raise ctypes.WinError(ctypes.get_last_error())
        return dict(pid=pid,state='ALIVE' if code.value==259 else 'EXITED',exit_code=None if code.value==259 else code.value,
            creation_100ns=str((creation.dwHighDateTime<<32)|creation.dwLowDateTime),image=image.value,in_job=bool(in_job.value))
    finally:kernel.CloseHandle(handle)


def inspect_identity(expected: dict) -> dict:
    current=process_identity(expected['pid'])
    if current['state'] in ('NOT_FOUND','UNKNOWN'):return current
    if current['creation_100ns']!=expected.get('creation_100ns') or current['image'].casefold()!=expected.get('image','').casefold():
        return dict(pid=expected['pid'],state='PID_REUSED',observed=current)
    return current


class DurableRun:
    def __init__(self,directory: Path,role: str,*,period: float=5.):
        directory.mkdir(exist_ok=False);self.directory=directory;self.period=period;self.start=time.perf_counter();self.last=-1e100
        self.state=dict(role=role,nonce=uuid.uuid4().hex,started_utc=utc(),status='STARTING',phase='entry',
            self_process=process_identity(os.getpid()),parent_pid=os.getppid(),child=None,
            automatic_retry=False,forced_exit_code_cannot_be_reconstructed=True)
        self.event('started');self.heartbeat(force=True)

    def event(self,name: str,**values) -> None:
        item=dict(event=name,utc=utc(),elapsed_seconds=time.perf_counter()-self.start,**values)
        with (self.directory/'events.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(item,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

    def child_started(self,pid: int) -> None:
        self.state['child']=process_identity(pid);self.event('child_started',identity=self.state['child']);self.heartbeat(force=True)

    def heartbeat(self,*,force: bool=False,**values) -> None:
        self.state.update(values)
        now=time.perf_counter()
        if force or now-self.last>=self.period:
            self.state.update(heartbeat_utc=utc(),elapsed_seconds=now-self.start)
            atomic_json(self.directory/'state.json',self.state);self.last=now

    def finish(self,status: str,**values) -> None:
        self.state.update(status=status,ended_utc=utc(),**values);self.heartbeat(force=True)
        atomic_json(self.directory/'terminal.json',self.state);self.event('terminal',status=status)


def inspect_run(directory: Path) -> dict:
    state=json.loads((directory/'state.json').read_bytes());terminal=directory/'terminal.json'
    members=[state['self_process']]+([state['child']] if state.get('child') else [])
    observations=[inspect_identity(p) for p in members]
    if any(p['state'] in ('ALIVE','UNKNOWN') for p in observations):decision='BLOCK_ACTIVE_OR_UNCERTAIN_PROCESS'
    elif terminal.exists():decision='TERMINAL_RECORDED'
    else:decision='INTERRUPTED_UNKNOWN_EXIT_NO_AUTOMATIC_RETRY'
    return dict(decision=decision,observations=observations,last_state=state,
        original_exit_cause_confirmed=False,observation_utc=utc())
