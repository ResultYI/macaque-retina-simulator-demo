"""One predeclared type-selective condition, without changing model parameters."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if not out.is_relative_to(ROOT / 'reports'):
        raise ValueError('Existing experiment directory under reports required')
    ledger_path = out / 'RESOURCE_LEDGER.json'
    ledger = json.loads(ledger_path.read_bytes())
    if ledger['status'] != 'COMPLETED' or ledger.get('supplemental_runs'):
        raise ValueError('Complete initial phase and unused selective condition required')
    protocol = dict(case='s_channel_only_standard', duration_seconds=.9,
        input='Replay actual cached S_ON_BC terminal release; hold other BC terminal drives at their initial values',
        purpose='Distinguish S pathway activation from common-input mixed net polarity in existing S-bistratified type',
        interpretation='Selective BC-terminal assay, not an optical S-cone-isolating or color-opponency validation',
        parameters_changed=False, connections_changed=False,
        expected='S_ON_BC increased effective release should increase S-bistratified gE, V and expected count rate',
        start_utc=datetime.now(timezone.utc).isoformat())
    with (out / 'S_TYPE_PROBE_PROTOCOL.json').open('x', encoding='utf8') as stream:
        json.dump(protocol, stream, indent=2)
    start = time.perf_counter()
    record = dict(case=protocol['case'], timeout_seconds=900)
    try:
        with (out / 's_probe.log').open('x', encoding='utf8') as log:
            r = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/rgc_functional_worker.py'),
                '--output', str(out), '--s-probe'], cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, timeout=900, check=False)
        record.update(status='COMPLETED' if r.returncode == 0 else 'FAILED', exit_code=r.returncode)
    except subprocess.TimeoutExpired:
        record.update(status='TIMEOUT', exit_code=None)
    finally:
        record['wall_seconds'] = time.perf_counter() - start
        ledger['model_wall_seconds_initial_phase'] = ledger['model_wall_seconds_this_session']
        ledger['model_wall_seconds_this_session'] += record['wall_seconds']
        ledger['historical_plus_this_session_seconds'] = (
            ledger['historical_model_wall_seconds'] + ledger['model_wall_seconds_this_session'])
        ledger['supplemental_runs'] = [record]
        ledger['status'] = record['status']
        ledger['new_output_bytes'] = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        ledger['end_utc'] = datetime.now(timezone.utc).isoformat()
        old = ROOT / ledger['historical_ledger']
        ledger['historical_ledger_unchanged'] = (
            hashlib.sha256(old.read_bytes()).hexdigest() == ledger['historical_ledger_sha256'])
        ledger_path.write_text(json.dumps(ledger, indent=2) + '\n', encoding='utf8')
    print(json.dumps(record))
    return 0 if record['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
