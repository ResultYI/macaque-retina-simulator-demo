"""Owned short radial candidate worker with existing native/resource guards."""
from pathlib import Path
import ctypes,faulthandler,json,os,sys,threading,time,traceback

def main(nonce,folder,stage):
    from tools.durable_research_runtime_v1 import atomic_json,process_identity,DurableRun
    output=Path(folder);ctypes.pythonapi.PyThreadState_Get.argtypes=[];ctypes.pythonapi.PyThreadState_Get.restype=ctypes.c_void_p
    (output/'PYTHON_TSTATE.txt').write_text(str(os.getpid())+'\t'+str(threading.get_native_id())+'\t'+format(ctypes.pythonapi.PyThreadState_Get(),'x'),encoding='ascii')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.IsDebuggerPresent.restype=ctypes.c_int
    atomic_json(output/'WORKER_READY.json',dict(nonce=nonce,identity=process_identity(os.getpid()),argv=sys.argv,phase='WAIT_BEFORE_MODEL'))
    deadline=time.monotonic()+120
    while not (output/'GO.json').exists():
        if (output/'STOP').exists() or time.monotonic()>deadline:raise RuntimeError('Native attachment gate stopped')
        time.sleep(.1)
    go=json.loads((output/'GO.json').read_text(encoding='utf-8-sig'))
    if go['nonce']!=nonce or go['pid']!=os.getpid() or not kernel.IsDebuggerPresent():raise RuntimeError('Owned native gate mismatch')
    from retina_sim.background_checkpoint import MemoryGuard
    from tools.checkpoint_json_hash_runtime_v1 import install
    install()
    from tools.gin_radial_context_v1 import BASE
    memory=MemoryGuard(24*1024**3,8*1024**3);journal=DurableRun(output/'runtime','gin69_'+stage);start=time.perf_counter();status='FAILED'
    def guard():
        if (BASE/'REQUEST_STOP').exists() or (output/'STOP').exists():raise RuntimeError('Explicit STOP')
        if not memory.check():raise MemoryError(str(memory.last))
        journal.heartbeat(phase=stage,memory=memory.last)
    try:
        guard()
        if stage=='optical_setup':
            from tools.gin_radial_context_v1 import optical_setup
            optical_setup()
        elif stage=='initialize':
            from tools.gin_radial_initialize_v1 import initialize
            initialize(guard)
        elif stage=='forward':
            from tools.gin_radial_forward_v1 import run
            run(guard,journal)
        else:raise ValueError(stage)
        status='PASS';journal.finish(status,phase=stage)
    except BaseException as error:
        atomic_json(output/'FAILURE.json',dict(error=str(error),traceback=traceback.format_exc()));journal.finish('FAILED',phase=stage);raise
    finally:atomic_json(output/'RESOURCE.json',dict(status=status,wall_seconds=time.perf_counter()-start,peak_rss_bytes=memory.peak_rss_bytes,memory=memory.last,guard_stop=memory.stop_reason,automatic_retry=False,wall_deadline_imposed=False))

if __name__=='__main__':
    nonce,folder,stage=sys.argv[1:]
    with (Path(folder)/'NATIVE_FAULTHANDLER.log').open('x',encoding='utf8') as f:
        faulthandler.enable(file=f,all_threads=True);main(nonce,folder,stage)
