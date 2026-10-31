"""Bounded source-coupled qualification; preserve the prior replay result."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    out = p.parse_args().output.resolve()
    if not out.is_relative_to(ROOT / 'reports'):
        raise ValueError('Existing experiment under reports required')
    ledger_path = out / 'RESOURCE_LEDGER.json'
    ledger = json.loads(ledger_path.read_bytes())
    qual = out / 'source_coupled_v1'
    qual.mkdir(exist_ok=False)
    protocol = dict(start_utc=datetime.now(timezone.utc).isoformat(),
        reason='Replay endpoint forcing error exceeds strict source allocation; preserve that limitation',
        cases=['standard', 'tight'], duration_seconds=.9,
        equations='All6305 original inner states, actual reciprocal gap and original BC protocol',
        input='Frozen Cone/HC source background plus original cone release/rod stimuli; no optical recomputation',
        sampler='Original joint event commit, PCG64 histories and all five types;900 bins only',
        standard=dict(rtol=1e-6, atol=1e-9, max_step=.0001),
        tight=dict(rtol=1e-10, atol=1e-13, max_step=.000025),
        per_process_timeout_seconds=900, no_new_scientific_parameters=True)
    (qual / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf8')
    ledger['source_coupled_runs'] = []
    for precision in protocol['cases']:
        started = time.perf_counter()
        record = dict(precision=precision, timeout_seconds=900)
        try:
            with (qual / (precision + '.log')).open('x', encoding='utf8') as log:
                r = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/rgc_functional_coupled.py'),
                    '--output', str(qual / precision), '--precision', precision],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900, check=False)
            record.update(status='COMPLETED' if r.returncode == 0 else 'FAILED', exit_code=r.returncode)
        except subprocess.TimeoutExpired:
            record.update(status='TIMEOUT', exit_code=None)
        record['wall_seconds'] = time.perf_counter() - started
        ledger['source_coupled_runs'].append(record)
        ledger['model_wall_seconds_this_session'] += record['wall_seconds']
        ledger['historical_plus_this_session_seconds'] = (
            ledger['historical_model_wall_seconds'] + ledger['model_wall_seconds_this_session'])
        ledger['status'] = record['status']
        ledger['end_utc'] = datetime.now(timezone.utc).isoformat()
        ledger['historical_ledger_unchanged'] = hashlib.sha256(
            (ROOT / ledger['historical_ledger']).read_bytes()).hexdigest() == ledger['historical_ledger_sha256']
        ledger['new_output_bytes'] = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        ledger_path.write_text(json.dumps(ledger, indent=2) + '\n', encoding='utf8')
        print(json.dumps(record), flush=True)
        if record['status'] != 'COMPLETED':
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
