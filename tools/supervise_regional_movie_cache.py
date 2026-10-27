"""Durable local launcher for the unchanged regional native cache producer.

This wrapper records a child exit code even if native code terminates Python.
It adds no model, data, solver, resource-budget or acceptance override.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.atomic_io import replace_with_retry,execution_name


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    replace_with_retry(temporary, path)


def _resources(handle):
    """Read the exact child process handle; no process enumeration is needed."""
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (key, ctypes.c_size_t) for key in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]

    class FileTime(ctypes.Structure):
        _fields_ = [('low', wintypes.DWORD), ('high', wintypes.DWORD)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(FileTime)] * 4
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    creation, exit_time, kernel_time, user_time = (FileTime() for _ in range(4))
    result = {}
    if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        result.update(RSS_bytes=int(counters.WorkingSetSize), peak_RSS_bytes=int(counters.PeakWorkingSetSize))
    else:
        result['memory_measurement_error'] = ctypes.get_last_error()
    if kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                              ctypes.byref(kernel_time), ctypes.byref(user_time)):
        ticks = lambda value: (int(value.high) << 32) | int(value.low)
        result['CPU_seconds'] = (ticks(kernel_time) + ticks(user_time)) / 1e7
    else:
        result['CPU_measurement_error'] = ctypes.get_last_error()
    return result


def main(args):
    if os.name != 'nt':
        raise RuntimeError('This local supervisor is for the configured Windows host')
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('The attempt must stay within this existing project')
    directory = output / execution_name('_supervisor',args.execution_label)
    directory.mkdir(parents=True, exist_ok=False)
    native_args = list(args.native_args)
    if native_args[:1] == ['--']:
        native_args.pop(0)
    if any(value == '--output-dir' or value.startswith('--output-dir=') for value in native_args):
        raise ValueError('Provide the attempt output directory once, to the supervisor')
    if any(value=='--execution-label' or value.startswith('--execution-label=') for value in native_args):
        raise ValueError('Provide the execution label once, to the supervisor')
    command = [sys.executable, '-B', '-X', 'utf8', '-X', 'faulthandler',
               str(ROOT / 'tools/run_regional_movie_cache.py'),
               '--output-dir', output.relative_to(ROOT).as_posix(), *native_args]
    if args.execution_label:command.extend(['--execution-label',args.execution_label])
    result_path = directory / 'result.json'
    report = dict(schema='regional-native-process-supervisor-1', status='STARTING',
                  created_UTC=datetime.now(timezone.utc).isoformat(),
                  supervisor_pid=os.getpid(), command=command,
                  execution_label=args.execution_label,
                  model_or_budget_overrides_added=False,
                  native_log=(directory / 'native.log').relative_to(ROOT).as_posix())
    _write(result_path, report)
    start = time.monotonic()
    with (directory / 'native.log').open('wb', buffering=0) as log:
        child = subprocess.Popen(command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        report.update(status='RUNNING', child_pid=child.pid)
        try:
            while True:
                code = child.poll()
                report.update(_resources(int(child._handle)))
                report.update(wall_seconds=time.monotonic() - start,
                              last_checked_UTC=datetime.now(timezone.utc).isoformat())
                if code is not None:
                    report.update(status='CHILD_EXITED', returncode=code,
                                  unsigned_returncode_hex=f'0x{code & 0xffffffff:08X}')
                    _write(result_path, report)
                    return 0 if code == 0 else 1
                _write(result_path, report)
                time.sleep(2)
        except BaseException as exc:
            report.update(status='SUPERVISOR_INTERRUPTED', error_type=type(exc).__name__, error=str(exc))
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            report.update(returncode=child.returncode,
                          wall_seconds=time.monotonic() - start)
            _write(result_path, report)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--execution-label',default='')
    parser.add_argument('native_args', nargs=argparse.REMAINDER)
    sys.exit(main(parser.parse_args()))
