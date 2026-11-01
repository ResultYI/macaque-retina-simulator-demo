"""One full original Cell6 cache under passive native capture, with isolated output."""
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

ROOT = Path(r'D:\macaque retinal simulator')
sys.path.insert(0, str(ROOT))


def main(nonce: str, output: Path, group: int) -> None:
    from tools.durable_research_runtime_v1 import atomic_json, process_identity, DurableRun
    ctypes.pythonapi.PyThreadState_Get.argtypes = []
    ctypes.pythonapi.PyThreadState_Get.restype = ctypes.c_void_p
    (output/'PYTHON_TSTATE.txt').write_text(str(os.getpid())+'\t'+str(threading.get_native_id())+'\t'+format(ctypes.pythonapi.PyThreadState_Get(),'x'), encoding='ascii')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.IsDebuggerPresent.restype = ctypes.c_int
    atomic_json(output/'WORKER_READY.json', dict(nonce=nonce,
        identity=process_identity(os.getpid()), argv=sys.argv, phase='WAIT_BEFORE_MODEL'))
    deadline = time.monotonic()+120
    while not (output/'GO.json').exists():
        if (output/'STOP').exists():
            raise RuntimeError('Stopped before model')
        if time.monotonic()>deadline:
            raise TimeoutError('Native attachment gate120s')
        time.sleep(.1)
    go = json.loads((output/'GO.json').read_text(encoding='utf-8-sig'))
    if go['nonce']!=nonce or go['pid']!=os.getpid() or not kernel.IsDebuggerPresent():
        raise RuntimeError('Native attachment identity mismatch')
    from retina_sim.background_checkpoint import MemoryGuard
    from tools.checkpoint_json_hash_runtime_v1 import install as install_hash_runtime,evidence as hash_evidence
    install_hash_runtime()
    from tools.cell6_halo_context_v1 import BASE
    from tools.wp1_latimer_compiled_cache_v1 import capture
    from tools.cell6_compiled_inner_v1 import CLOSED_INSTANCES
    from tools.cell6_runtime_monitor_v3 import evidence as monitor_evidence
    experiment = output.parent.parent
    if not 1<=group<=17:raise RuntimeError('Only frozen Cell2-4 input registry admitted')
    gate=json.loads((experiment/'PAIR_GATE.json').read_text(encoding='utf8'))
    actual=hashlib.sha256((ROOT/'tools/cell6_compiled_inner_v1.py').read_bytes()).hexdigest()
    if not gate['passed'] or actual!=gate['adapter_sha256']:raise RuntimeError('Accepted pair adapter changed')
    guard_hash=hashlib.sha256((ROOT/'tools/cell6_compiled_kernel_guard_v2.py').read_bytes()).hexdigest()
    if guard_hash!=gate['kernel_guard_sha256']:raise RuntimeError('Accepted constant validation guard changed')
    memory = MemoryGuard(24*1024**3, 8*1024**3)
    journal = DurableRun(output/'runtime', 'cell6_monitor_repair_g'+str(group))
    started = time.perf_counter()
    status = 'FAILED'
    def guard():
        if any(p.exists() for p in (output/'STOP', experiment/'REQUEST_STOP', BASE/'REQUEST_STOP', ROOT/'REQUEST_STOP')):
            raise RuntimeError('Explicit research STOP')
        if not memory.check():
            raise MemoryError(str((memory.stop_reason, memory.last)))
        journal.heartbeat(phase='full_g'+str(group), memory=memory.last)
    atomic_json(output/'EXECUTION_CONTRACT.json', dict(group=group,
        original_preparation=str(BASE), isolated_output=str(experiment),
        original_start_seconds=0, full_input_duration_seconds=7.1,
        fresh_from_original_compatible_seed=json.loads((experiment/('SEGMENT_REQUEST_g'+str(group)+'.json')).read_text(encoding='utf8'))['index']==0, old_partial_checkpoint_reused=False,
        compiled_adapter_sha256=actual,kernel_guard_sha256=guard_hash, allocator=os.environ.get('PYTHONMALLOC'),
        Python_trace_or_profile_hook_enabled=(sys.gettrace() is not None or sys.getprofile() is not None),
        root_cause_proven=False, science_or_numerical_contract_changed=False,
        checkpoint_hash_runtime=hash_evidence(),
        FIT_READY=False, background_qualified=False, production_adopted=False))
    try:
        guard()
        capture(group, guard, journal, memory, output_root=experiment)
        status = 'PASS'
        journal.finish(status, phase='complete')
    except BaseException as error:
        atomic_json(output/'FAILURE.json', dict(error=str(error), traceback=traceback.format_exc()))
        journal.finish('FAILED', phase='full_g'+str(group))
        raise
    finally:
        atomic_json(output/'COMPILED_EXECUTION.json',dict(instances=CLOSED_INSTANCES,
            checkpoint_hash_runtime=hash_evidence(),
            monitor_statistics=dict(monitor_evidence(),call_scope='current serial worker; not cumulative across process handoffs'),
            compiled_rhs_calls=sum(v['compiled_rhs_calls'] for v in CLOSED_INSTANCES),
            construction_and_first_rhs_seconds=sum(v['build']['construction_plus_first_call_seconds'] for v in CLOSED_INSTANCES),
            python_direct_continuous_rhs_fallback_calls=0,
            thread_environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
        atomic_json(output/'RESOURCE.json', dict(status=status,
            wall_seconds=time.perf_counter()-started, memory=memory.last,
            peak_rss_bytes=memory.peak_rss_bytes, guard_stop=memory.stop_reason,
            automatic_retry=False, wall_deadline_imposed=False,
            native_root_cause_fixed_claimed=False))


if __name__=='__main__':
    nonce, folder, number = sys.argv[1:]
    with (Path(folder)/'NATIVE_FAULTHANDLER.log').open('x',encoding='utf8') as stream:
        faulthandler.enable(file=stream, all_threads=True)
        main(nonce, Path(folder), int(number))
