"""Additional stage retains every prior bill and unchanged scientific protocol."""
from pathlib import Path
import os,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]


def test_event_stage_preserves_full_science_historical_charges_and_old_failure(tmp_path):
    code='''
from pathlib import Path
import json,sys
from retina_sim import group_scc_protocol as g
r=Path.cwd();p=g.load_protocol(r,'gpu_dop853')
old=g.read_json(r/'reports/solver_gpu_snapshot_20261026_v1/paired_combined_r5_r1/protocol.json')
omit={'batch_id','historical_batches','budget_limits_seconds','compute_authorization'}
assert {k:v for k,v in p.items() if k not in omit}=={k:v for k,v in old.items() if k not in omit}
receipts=p['historical_batches'];assert len(receipts)==3
assert receipts[-1]['used']==dict(cold=94.70864367485046,feasibility=0.,paired=4783.335707187653)
assert len(receipts[-1]['evidence_pins'])==5
assert (p['start_seconds'],p['checkpoint_seconds'],p['stop_seconds'])==(4.993,4.997,5.001)
assert g.REPORT_PARENT=='reports/solver_gpu_event_20261026_v1'
# A test-only root gets an independent ledger; its historical receipts retain
# all real prior totals, without modifying any real-stage accounting.
g.historical_batch_receipts=lambda _:receipts
root=Path(sys.argv[1])
with g.BudgetLedger(root) as ledger:
 assert ledger.data['historical_batches']==receipts
 token=ledger.begin(root/g.REPORT_PARENT/'fixture','paired',{'test':'new-stage'})
 ledger.charge('paired',.125);ledger.finish('INCONCLUSIVE')
assert g.read_json(root/g.REPORT_PARENT/'budget_ledger.json')['used']['paired']==.125
assert g.sha256(r/'reports/solver_gpu_snapshot_20261026_v1/budget_ledger.json')==receipts[-1]['ledger_sha256']
'''
    env=dict(os.environ,RETIPATH_ACCEPTANCE_BATCH='overnight_event_20261026_v1')
    subprocess.run([sys.executable,'-X','utf8','-B','-c',code,str(tmp_path)],cwd=ROOT,env=env,check=True,capture_output=True,text=True)


def test_original_restore_failure_is_an_adoption_hold_not_a_rewritten_scientific_result():
    from tools.validate_group_scc_radau import preserve_unresolved_restore_hold
    outcome=dict(status='PERFORMANCE_BELOW_10X',scientific_status='PASS',restore_status='RESTORE_PASS',
        adoptable=True,performance=dict(status='PERFORMANCE_BELOW_10X',speedup=2.,adoptable=True))
    held=preserve_unresolved_restore_hold(outcome,{'historical_batches':[{'evidence_pins':{'failure.json':'frozen'}}]})
    assert held['scientific_status']=='PASS' and held['restore_status']=='RESTORE_PASS'
    assert held['performance']['speedup']==2. and held['status']==outcome['status']
    assert held['adoptable'] is False and held['performance']['adoptable'] is False
    assert outcome['adoptable'] is True
