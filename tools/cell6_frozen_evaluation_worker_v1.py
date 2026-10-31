"""Resume only the existing frozen evaluations under the native debugger."""
from pathlib import Path
import ctypes
import faulthandler
import hashlib
import json
import os
import sys
import threading
import time
import traceback

ROOT=Path(r'D:\macaque retinal simulator')
sys.path.insert(0,str(ROOT))


def main(nonce: str, output: Path) -> None:
    from tools.durable_research_runtime_v1 import atomic_json,process_identity,DurableRun
    ctypes.pythonapi.PyThreadState_Get.argtypes=[]
    ctypes.pythonapi.PyThreadState_Get.restype=ctypes.c_void_p
    (output/'PYTHON_TSTATE.txt').write_text(str(os.getpid())+'\t'+str(threading.get_native_id())+'\t'+format(ctypes.pythonapi.PyThreadState_Get(),'x'),encoding='ascii')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.IsDebuggerPresent.restype=ctypes.c_int
    atomic_json(output/'WORKER_READY.json',dict(nonce=nonce,identity=process_identity(os.getpid()),argv=sys.argv,phase='WAIT_BEFORE_MODEL_OR_FIT'))
    deadline=time.monotonic()+120
    while not (output/'GO.json').exists():
        if (output/'STOP').exists():raise RuntimeError('Explicit STOP')
        if time.monotonic()>deadline:raise TimeoutError('Native attachment gate120s')
        time.sleep(.1)
    go=json.loads((output/'GO.json').read_text(encoding='utf-8-sig'))
    if go['nonce']!=nonce or go['pid']!=os.getpid() or not kernel.IsDebuggerPresent():
        raise RuntimeError('Native attachment identity mismatch')
    from retina_sim.background_checkpoint import MemoryGuard
    from tools.cell6_halo_context_v1 import BASE
    from tools.cell6_frozen_evaluation_resume_v1 import resume as fit
    experiment=output.parent.parent
    memory=MemoryGuard(24*1024**3,8*1024**3)
    journal=DurableRun(output/'runtime','cell6_frozen_evaluation_resume')
    started=time.perf_counter();status='FAILED'
    def guard():
        if any(p.exists() for p in (output/'STOP',experiment/'REQUEST_STOP',BASE/'REQUEST_STOP',ROOT/'REQUEST_STOP')):
            raise RuntimeError('Explicit research STOP')
        if not memory.check():raise MemoryError(str((memory.stop_reason,memory.last)))
        journal.heartbeat(phase='resume_frozen_fixed_evaluation',memory=memory.last)
    atomic_json(output/'EXECUTION_CONTRACT.json',dict(original_preparation=str(BASE),isolated_output=str(experiment),
        train_groups=[0,1,2],touched_internal_group=3,historical_development_groups=[4,5],historical_test_accessed=False,
        structure_and_optimizer_changed=False,one_temporal_fit_already_frozen=True,optimizer_started_in_this_process=False,automatic_restart=False,
        fit_code_sha256=hashlib.sha256((ROOT/'tools/fit_cell6_raw_readout_v1.py').read_bytes()).hexdigest(),
        allocator=os.environ.get('PYTHONMALLOC'),FIT_READY=False,background_qualified=False,production_adopted=False))
    try:
        guard();fit(guard=guard,output_root=experiment)
        status='PASS';journal.finish(status,phase='complete')
    except BaseException as error:
        atomic_json(output/'FAILURE.json',dict(error=str(error),traceback=traceback.format_exc()))
        journal.finish('FAILED',phase='readout');raise
    finally:
        atomic_json(output/'RESOURCE.json',dict(status=status,wall_seconds=time.perf_counter()-started,
            memory=memory.last,peak_rss_bytes=memory.peak_rss_bytes,guard_stop=memory.stop_reason,
            automatic_retry=False,wall_deadline_imposed=False,native_root_cause_fixed_claimed=False))


if __name__=='__main__':
    nonce,folder=sys.argv[1:]
    with (Path(folder)/'NATIVE_FAULTHANDLER.log').open('x',encoding='utf8') as stream:
        faulthandler.enable(file=stream,all_threads=True);main(nonce,Path(folder))
