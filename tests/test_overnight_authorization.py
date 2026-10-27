"""The new compute stage preserves every scientific setting and old charge."""
import json
import os
from pathlib import Path
import subprocess
import sys

from retina_sim.group_scc_protocol import LEGACY_REPORT_PARENT, load_protocol


ROOT = Path(__file__).resolve().parents[1]


def test_overnight_profile_only_changes_compute_stage_and_keeps_archived_history():
    code = "from retina_sim.group_scc_protocol import *; import json; print(json.dumps(dict(protocol=load_protocol('.', 'gpu_dop853'),parent=REPORT_PARENT)))"
    env = dict(os.environ, RETIPATH_ACCEPTANCE_BATCH='overnight_20261026_v1')
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                            capture_output=True, text=True, check=True)
    new = json.loads(result.stdout)
    old = load_protocol(ROOT, 'gpu_dop853')
    changed = {key for key in old if old[key] != new['protocol'][key]}
    assert changed == {'batch_id', 'budget_limits_seconds'}
    assert new['parent'] != LEGACY_REPORT_PARENT
    history = new['protocol']['historical_batches'][0]
    assert history['used'] == dict(cold=181.80423092842102,
                                  feasibility=174.97222447395325, paired=674.6077086925507)
    assert history['ledger_sha256'] == '9ea19ecab62abf3fa6cb1c3ebd9b699871c2f6a0de54fc36f477458d4f0da02f'


def test_unknown_profile_fails_closed():
    result = subprocess.run([sys.executable, '-c', 'import retina_sim.group_scc_protocol'],
        cwd=ROOT, env=dict(os.environ, RETIPATH_ACCEPTANCE_BATCH='unrecognized'), capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Unknown explicit acceptance authorization profile' in result.stderr


def test_short_diagnostic_keeps_the_same_ledger_and_cannot_be_full_qualification():
    code = "from retina_sim.group_scc_protocol import *; import json; print(json.dumps(dict(protocol=load_protocol('.', 'gpu_dop853'),parent=REPORT_PARENT)))"
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
        env=dict(os.environ, RETIPATH_ACCEPTANCE_BATCH='overnight_short_20261026_v1'),
        capture_output=True, text=True, check=True)
    value = json.loads(result.stdout)
    p = value['protocol']
    assert value['parent'] == 'reports/solver_gpu_overnight_20261026_v1'
    assert p['batch_id'] == 'gpu-overnight-20261026-authorization-v1'
    assert p['diagnostic_scope_only'] is True
    assert p['diagnostic_wall_cap_seconds'] == 600.
    assert abs(p['stop_seconds']-p['start_seconds']-.0002) < 1e-12
    assert abs(p['checkpoint_seconds']-p['start_seconds']-.0001) < 1e-12
    old = load_protocol(ROOT, 'gpu_dop853')
    for key in ('reference', 'candidate', 'numerical_budget', 'resource_limits', 'execution_contract',
                'rod_step_seconds', 'BC_RGC_period_seconds', 'recording_step_seconds'):
        assert p[key] == old[key]


def test_snapshot_stage_is_additional_accounting_with_both_old_charges_bound():
    code = "from retina_sim.group_scc_protocol import *; import json; print(json.dumps(dict(protocol=load_protocol('.', 'gpu_dop853'),parent=REPORT_PARENT)))"
    result = subprocess.run([sys.executable,'-c',code],cwd=ROOT,
        env=dict(os.environ,RETIPATH_ACCEPTANCE_BATCH='overnight_snapshot_20261026_v1'),
        capture_output=True,text=True,check=True)
    value=json.loads(result.stdout);p=value['protocol']
    assert value['parent']=='reports/solver_gpu_snapshot_20261026_v1'
    assert p['batch_id']=='gpu-snapshot-20261026-authorization-v1'
    assert len(p['historical_batches'])==2
    assert p['historical_batches'][1]['ledger_sha256']=='fc4639c6529ad6d89b80c0d5c1a630a2fc11ad7af95ff4203bb577c4e2000377'
    assert p['historical_batches'][1]['used']==dict(cold=112.67233920097351,feasibility=0.,paired=5007.933922767639)
    old=load_protocol(ROOT,'gpu_dop853')
    changed={key for key in old if old[key]!=p[key]}
    assert changed=={'batch_id','budget_limits_seconds'}
    assert set(p)-set(old)=={'historical_batches','compute_authorization'}
    assert (p['start_seconds'],p['checkpoint_seconds'],p['stop_seconds'])==(4.993,4.997,5.001)
