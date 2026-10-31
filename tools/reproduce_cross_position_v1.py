"""Sequential current-seed replay with an owned-process-tree hard timeout."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--historical-seconds', type=float, required=True)
    args = p.parse_args()
    if not math.isfinite(args.historical_seconds) or args.historical_seconds < 0:
        raise ValueError('Finite nonnegative historical counter required')
    prepared, out = args.prepared.resolve(), args.output.resolve()
    if not prepared.is_relative_to(ROOT) or not out.is_relative_to(ROOT / 'reports'):
        raise ValueError('Existing project preparation and fresh reports output required')
    out.mkdir(parents=True, exist_ok=False)
    stages = []; started = time.perf_counter(); status = 'FAILED'
    python = str(ROOT / '.venv/Scripts/python.exe')
    def execute(label, argv, seconds):
        start = time.perf_counter()
        row = dict(stage=label, start_utc=datetime.now(timezone.utc).isoformat(), timeout_seconds=seconds)
        with (out / (label + '.stdout.log')).open('xb') as so, (out / (label + '.stderr.log')).open('xb') as se:
            child = subprocess.Popen(argv, cwd=ROOT, stdout=so, stderr=se, creationflags=subprocess.CREATE_NO_WINDOW)
            row['owned_pid'] = child.pid
            try:
                code = child.wait(timeout=seconds)
                row['returncode'] = code
                if code: raise RuntimeError('Stage failed; preserved output: ' + label)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True, timeout=20)
                row['owned_tree_stopped_on_timeout'] = True
                raise
            except BaseException:
                if child.poll() is None:
                    subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True, timeout=20)
                    row['owned_tree_stopped_on_interruption'] = True
                raise
            finally:
                row['wall_seconds'] = time.perf_counter() - start
                stages.append(row)
    try:
        for duration, precision, label in [(6, 'standard', 'short6'), (900, 'standard', 'standard'), (900, 'tight', 'tight')]:
            execute(label, [python, '-B', str(ROOT / 'tools/run_cross_position_rgc_v1.py'),
                '--prepared', str(prepared), '--background', str(prepared / 'background_24g_v3'),
                '--output', str(out / label), '--precision', precision, '--milliseconds', str(duration)], 900)
        execute('source_port_checks', [python, '-B', str(ROOT / 'tools/check_cross_position_ports_v1.py'),
            '--prepared', str(prepared), '--background', str(prepared / 'background_24g_v3'),
            '--output', str(out / 'source_port_checks')], 180)
        execute('analysis', [python, '-B', str(ROOT / 'tools/analyze_cross_position_v1.py'),
            '--prepared', str(prepared), '--runs', str(out), '--output', str(out / 'analysis')], 180)
        status = 'PASS'
    finally:
        receipt = dict(status=status, inherited_recorded_seconds=args.historical_seconds, stages=stages,
            cumulative_including_this_launcher_wall_seconds=args.historical_seconds + time.perf_counter() - started,
            single_process_sequence=True, automatic_retry=False, original_scientific_parameters_unchanged=True,
            FIT_READY=False, background_qualified=False, production_adopted=False)
        with (out / 'REPLAY_SEQUENCE_RECEIPT.json').open('x', encoding='utf8') as f:
            json.dump(receipt, f, indent=2)


if __name__ == '__main__': main()
