"""Run one protected MATLAB equivalence check with the approved notebook limits."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from run_cross_position_optical_v1 import MemoryGuard, tree_rss

OUT = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT / 'reports/isetbio_native_aperture_convergence_20261029_v1'
attempt = OUT / 'validation_run_1'
attempt.mkdir(exist_ok=False)
env = os.environ.copy()
for name, suffix in (('MATLAB_PREFDIR', 'preferences'), ('TEMP', 'temp')):
    folder = attempt / suffix
    folder.mkdir()
    env[name] = str(folder)
env['TMP'] = env['TEMP']
expression = f"addpath('{(ROOT / 'tools').as_posix()}');validate_native_aperture_v1('{OUT.as_posix()}')"
command = [r'C:\Program Files\MATLAB\R2025b\bin\matlab.exe', '-batch', expression]
memory = MemoryGuard(1024**3, 8*1024**3)
if not memory.check():
    raise RuntimeError('Insufficient system memory before launch')
start = time.perf_counter()
peak = 0
minimum = 2**63
reason = None
status = 'FAILED'
code = None
with (attempt / 'stdout.log').open('xb') as stdout, (attempt / 'stderr.log').open('xb') as stderr:
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
        creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        while process.poll() is None:
            rss, _ = tree_rss(process.pid)
            peak = max(peak, rss)
            healthy = memory.check()
            minimum = min(minimum, memory.last['available_physical_bytes'])
            if time.perf_counter()-start > 600 or rss > 24*1024**3 or not healthy:
                raise RuntimeError('Time, 24GiB process-tree, or 8GiB system-free guard triggered')
            time.sleep(.5)
        code = process.returncode
        status = 'PASS' if code == 0 and (OUT / 'VALIDATION.json').is_file() else 'FAILED'
    except BaseException as exc:
        reason = str(exc)
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True, timeout=15)
        raise
    finally:
        receipt = dict(status=status, returncode=code, reason=reason, wall_seconds=time.perf_counter()-start,
            peak_owned_process_RSS_bytes=peak, minimum_available_memory_bytes=minimum,
            process_tree_limit_bytes=24*1024**3, minimum_system_free_bytes=8*1024**3,
            single_MATLAB_process_tree=True, neural_forward_seconds=0., historical_cumulative_seconds=5509.143878800069)
        (attempt / 'RESOURCE.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(json.dumps(receipt), flush=True)
if status != 'PASS':
    raise SystemExit(1)
