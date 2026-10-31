"""Single MATLAB capture with Windows process-tree memory and time protection."""
import argparse
import ctypes
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.background_checkpoint import MemoryGuard


class Entry(ctypes.Structure):
    _fields_ = [('dwSize', w.DWORD), ('cntUsage', w.DWORD), ('pid', w.DWORD),
        ('heap', ctypes.c_size_t), ('module', w.DWORD), ('threads', w.DWORD),
        ('parent', w.DWORD), ('priority', w.LONG), ('flags', w.DWORD), ('exe', w.WCHAR * 260)]


class Counters(ctypes.Structure):
    _fields_ = [('cb', w.DWORD), ('faults', w.DWORD)] + [(n, ctypes.c_size_t) for n in
        ('peak', 'working', 'peakPaged', 'paged', 'peakNonpaged', 'nonpaged', 'pagefile', 'peakPagefile')]


def tree_rss(pid):
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.CreateToolhelp32Snapshot.restype = w.HANDLE
    k.Process32FirstW.argtypes = k.Process32NextW.argtypes = [w.HANDLE, ctypes.POINTER(Entry)]
    k.OpenProcess.restype = w.HANDLE
    k.CloseHandle.argtypes = [w.HANDLE]
    snap = k.CreateToolhelp32Snapshot(2, 0)
    if snap == ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
    e = Entry(); e.dwSize = ctypes.sizeof(e); parents = {}
    try:
        ok = k.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            parents[int(e.pid)] = int(e.parent)
            ok = k.Process32NextW(snap, ctypes.byref(e))
    finally: k.CloseHandle(snap)
    owned = {pid}
    while True:
        enlarged = owned | {p for p, parent in parents.items() if parent in owned}
        if enlarged == owned: break
        owned = enlarged
    ps = ctypes.WinDLL('psapi', use_last_error=True)
    ps.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(Counters), w.DWORD]
    rss = 0
    for p in owned:
        handle = k.OpenProcess(0x410, False, p)
        if not handle: continue
        try:
            c = Counters(); c.cb = ctypes.sizeof(c)
            if ps.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb): rss += int(c.working)
        finally: k.CloseHandle(handle)
    return rss, owned


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--max-rss-gib',type=int,choices=[8,24],default=8)
    parser.add_argument('--historical-seconds',type=float,default=4263.204762000069)
    args=parser.parse_args()
    out=args.output.resolve()
    env = os.environ.copy()
    for name, folder in [('MATLAB_PREFDIR', 'matlab_preferences'), ('TEMP', 'matlab_temp')]:
        p = out / folder; p.mkdir(exist_ok=False); env[name] = str(p)
    env['TMP'] = env['TEMP']
    expression = (f"addpath('{(ROOT / 'tools').as_posix()}');capture_cross_position_background_v1("
                  f"'{(out / 'OPTICAL_CONFIG.json').as_posix()}','{(out / 'capture.mat').as_posix()}')")
    command = [r'C:\Program Files\MATLAB\R2025b\bin\matlab.exe', '-batch', expression]
    start = time.perf_counter(); peak = 0; minimum = 2**63; code = None; reason = None
    status = 'FAILED'; owned = set(); memory = MemoryGuard(1024**3, 8 * 1024**3)
    if not memory.check(): raise RuntimeError('Insufficient memory before launch')
    with (out / 'optical_stdout.log').open('xb') as so, (out / 'optical_stderr.log').open('xb') as se:
        p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=so, stderr=se,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while p.poll() is None:
                rss, owned = tree_rss(p.pid); peak = max(peak, rss)
                healthy = memory.check(); minimum = min(minimum, memory.last['available_physical_bytes'])
                if time.perf_counter() - start > 180 or peak > args.max_rss_gib * 1024**3 or not healthy:
                    raise RuntimeError('Bounded optical resource/time protection')
                time.sleep(.5)
            code = p.returncode
            status = 'PASS' if code == 0 and (out / 'capture.mat').is_file() else 'FAILED'
        except BaseException as exc:
            reason = str(exc)
            subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], capture_output=True, timeout=15)
            raise
        finally:
            receipt = dict(status=status, returncode=code, reason=reason,
                wall_seconds=time.perf_counter() - start, peak_owned_process_RSS_bytes=peak,
                minimum_available_memory_bytes=minimum, single_optical_process_tree=True,process_tree_limit_bytes=args.max_rss_gib*1024**3,
                historical_seconds=args.historical_seconds,
                neural_forward_seconds=0., hardware_health_claim=False)
            with (out / 'optical_capture_RESOURCE.json').open('x', encoding='utf8') as f:
                json.dump(receipt, f, indent=2)
            print(json.dumps(receipt), flush=True)
    if status != 'PASS': raise SystemExit(1)


if __name__ == '__main__': main()
