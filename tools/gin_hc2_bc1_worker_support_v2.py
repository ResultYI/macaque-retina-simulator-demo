"""Owned guarded worker; native attachment precedes any new geometry/model work."""
from pathlib import Path
import ctypes
import faulthandler
import json
import os
import sys
import threading
import time
import traceback

def main(nonce, folder, stage):
    from tools.durable_research_runtime_v1 import atomic_json, process_identity
    output = Path(folder)
    ctypes.pythonapi.PyThreadState_Get.argtypes = []
    ctypes.pythonapi.PyThreadState_Get.restype = ctypes.c_void_p
    (output/'PYTHON_TSTATE.txt').write_text(str(os.getpid())+'\t'+str(threading.get_native_id())+'\t'+format(ctypes.pythonapi.PyThreadState_Get(), 'x'), encoding='ascii')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.IsDebuggerPresent.restype = ctypes.c_int
    atomic_json(output/'WORKER_READY.json', dict(nonce=nonce, identity=process_identity(os.getpid()), argv=sys.argv, phase='WAIT_BEFORE_MODEL'))
    deadline = time.monotonic()+120
    while not (output/'GO.json').exists():
        if (output/'STOP').exists() or time.monotonic() > deadline:
            raise RuntimeError('Owned attachment gate stopped')
        time.sleep(.1)
    go = json.loads((output/'GO.json').read_text(encoding='utf-8-sig'))
    if go['nonce'] != nonce or go['pid'] != os.getpid() or not kernel.IsDebuggerPresent():
        raise RuntimeError('Owned native gate mismatch')
    from retina_sim.background_checkpoint import MemoryGuard
    memory = MemoryGuard(24*1024**3, 8*1024**3)
    stopped = threading.Event()
    def monitor():
        while not stopped.is_set():
            okay = memory.check()
            atomic_json(output/'RESOURCE_PROGRESS.json', dict(stage=stage, memory=memory.last, peak_rss_bytes=memory.peak_rss_bytes, wall_seconds=time.perf_counter()-start))
            if not okay or (output/'STOP').exists():
                atomic_json(output/'RESOURCE_STOP.json', dict(reason=memory.stop_reason, memory=memory.last, explicit_stop=(output/'STOP').exists()))
                os._exit(137)
            stopped.wait(1.)
    def guard():
        if stopped.is_set() or (output/'STOP').exists():
            raise RuntimeError('Owned stop')
    start = time.perf_counter()
    thread = threading.Thread(target=monitor, name='owned-resource-guard', daemon=True)
    thread.start()
    status = 'FAILED'
    try:
        if stage == 'install_owner_support':
            from tools.install_gin69_owner_support_v2 import build
            build(output, guard)
        elif stage == 'boundary_regression':
            from tools.check_gin69_owner_support_v2 import build
            build(output, guard)
        elif stage == 'build_master':
            from tools.build_gin69_support_master_v2 import build
            build(output, guard)
        elif stage == 'build_bc':
            from tools.build_gin_bc1_local_support_v2 import build
            build(output, guard)
        elif stage == 'assemble_fine':
            from tools.assemble_gin_hc2_bc1_fine_support_v2 import build
            build(output, guard)
        elif stage == 'extend_master':
            from tools.extend_gin_hc2_master_v1 import build
            build(output, guard)
        elif stage == 'assemble_native':
            from tools.assemble_gin_hc2_native_support_v2 import build
            build(output, guard)
        elif stage == 'prepare_optical':
            from tools.prepare_gin_hc2_bc1_optical_v1 import build
            from tools.assemble_gin_hc2_bc1_groups_support_v2 import OUT
            build(output, guard, source_out=OUT)
        elif stage == 'assemble_groups':
            from tools.assemble_gin_hc2_bc1_groups_support_v2 import build
            build(output, guard)
        elif stage == 'initialize':
            from tools.initialize_gin_hc2_bc1_support_v2 import build
            build(output, guard)
        elif stage == 'prepare_formal_optical':
            from tools.prepare_gin69_formal_optics_v1 import build
            from tools.assemble_gin_hc2_bc1_groups_support_v2 import OUT
            build(output, guard, source_out=OUT)
        elif stage == 'prepare_native_prefix':
            from tools.prepare_gin69_native_prefix_support_v2 import build
            build(output, guard)
        elif stage == 'formal_segment':
            from tools.checkpoint_json_hash_runtime_v1 import install
            install()
            from tools.gin69_formal_cache_support_v2 import run
            run(output, guard)
        elif stage in ('train','evaluate'):
            from tools.gin69_voltage_fit_support_v2 import execute
            execute(output, guard, stage)
            from tools.gin69_spatial_baselines_support_v2 import execute as stimulus_baselines
            stimulus_baselines(output, guard, stage)
            if stage == 'evaluate':
                from tools.report_gin69_matched_support_v2 import build
                delivery=output/'delivery';delivery.mkdir()
                build(delivery, guard)
        elif stage == 'short_forward':
            from tools.checkpoint_json_hash_runtime_v1 import install
            install()
            from tools.durable_research_runtime_v1 import DurableRun
            from tools.gin_hc2_bc1_forward_support_v2 import run
            journal = DurableRun(output/'runtime', 'gin69_new_HCBC_short')
            run(guard, journal, output)
            journal.finish('PASS', phase=stage)
        else:
            raise ValueError(stage)
        status = 'PASS'
    except BaseException as error:
        atomic_json(output/'FAILURE.json', dict(error=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        stopped.set()
        thread.join(timeout=3.)
        atomic_json(output/'RESOURCE.json', dict(status=status, wall_seconds=time.perf_counter()-start,
            peak_rss_bytes=memory.peak_rss_bytes, memory=memory.last, automatic_retry=False, wall_deadline_imposed=False))

if __name__ == '__main__':
    nonce, folder, stage = sys.argv[1:]
    with (Path(folder)/'NATIVE_FAULTHANDLER.log').open('x', encoding='utf8') as log:
        faulthandler.enable(file=log, all_threads=True)
        main(nonce, folder, stage)
