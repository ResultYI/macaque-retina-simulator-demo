"""Fresh laptop session: bounded original-equation conditional RGC assays."""
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
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'reports'):
        raise ValueError('Fresh experiment output must remain under project reports')
    out.mkdir(parents=True, exist_ok=False)
    receipt = ROOT / 'reports/BC_AC_asset_restore_20261029_2b995b23/migration_summary.json'
    migration = json.loads(receipt.read_bytes())
    if migration['status'] != 'MIGRATION_AND_CONSTRUCTION_READY':
        raise ValueError('Completed migration receipt required')
    old = ROOT / 'reports/layerwise_physiology/RESOURCE_LEDGER.json'
    historical = json.loads(old.read_bytes())
    ledger = dict(start_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User2026-10-29T11:02:15UTC: continue autonomous laptop research',
        historical_ledger=str(old.relative_to(ROOT)),
        historical_ledger_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
        historical_model_wall_seconds=historical['model_wall_seconds'],
        old_deadline_is_archival=True, migration_receipt=str(receipt.relative_to(ROOT)),
        migration_package_sha256=migration['package_sha256'],
        per_process_timeout_seconds=900, new_data_limit_bytes=2_000_000_000,
        model_wall_seconds_this_session=0., production_adopted=False,
        background_qualified=False, FIT_READY=False, status='RUNNING')
    protocol = dict(schema='conditional-RGC-functional-assay-1', duration_seconds=.9,
        cases=['background_standard', 'assay_standard', 'assay_tight', 'no_inhibition_standard'],
        assay='Frozen actual BC group releases at1ms from previous fully coupled BC protocol',
        forcing='Groupwise linear interpolation; AII indirect contribution remains inside BC cache',
        cone_epochs=[['native_release_0.9', .05, .2], ['native_release_1.1', .35, .5]],
        rod_epoch=['original_Rstar_s_x2', .6, .75],
        inhibition_control='Only remove existing RGC gI in one counterfactual readout arm; no saved parameter/route edits',
        background='Constant initial actual BC release; same original conditional initial state',
        main_node_readouts='After original probability-history tick; preserve original exported convention',
        binary_counts='Separate pre-tick law for each of900 half-open1ms bins; endpoint has no sampled future bin',
        spike_sampler='Original strict-past HistoryRGC.sample_spikes and saved PCG64 states',
        reference='Same forcing tighter integration plus exact original full-state algebra and saved900ms endpoint',
        numerical_allocation='Original frozen_budget on initial units; V/s uses original V budget divided by100us',
        no_new_parameters=True, no_new_connections=True, no_target_PSTH_used=True,
        no_full_network_trajectory_equivalence_assumed=True, FIT_READY=False)
    (out / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf8')
    path = out / 'RESOURCE_LEDGER.json'
    path.write_text(json.dumps(ledger, indent=2) + '\n', encoding='utf8')
    start = time.perf_counter()
    try:
        with (out / 'worker.log').open('x', encoding='utf8') as log:
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/rgc_functional_worker.py'),
                '--output', str(out)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                timeout=900, check=False)
        ledger.update(status='COMPLETED' if result.returncode == 0 else 'FAILED',
                      exit_code=result.returncode)
    except subprocess.TimeoutExpired:
        ledger.update(status='TIMEOUT', exit_code=None)
    finally:
        ledger['model_wall_seconds_this_session'] = time.perf_counter() - start
        ledger['historical_plus_this_session_seconds'] = (
            ledger['historical_model_wall_seconds'] + ledger['model_wall_seconds_this_session'])
        ledger['new_output_bytes'] = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        ledger['historical_ledger_unchanged'] = (
            hashlib.sha256(old.read_bytes()).hexdigest() == ledger['historical_ledger_sha256'])
        ledger['end_utc'] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(ledger, indent=2) + '\n', encoding='utf8')
    print(json.dumps(ledger))
    return 0 if ledger['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
