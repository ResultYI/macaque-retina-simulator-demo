"""Authorize only the last clean6.5s prefix after the proven I/O byte-parity fix."""
from pathlib import Path
import hashlib,json,shutil,sys
ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.durable_research_runtime_v1 import atomic_json,inspect_identity
OUT=ROOT/'reports/WP1_LATIMER_CELL2_4_V1'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    regression=read(OUT/'CHECKPOINT_HASH_REPAIR_REGRESSION.json')
    if not regression['passed']:raise RuntimeError('Hash equivalence not passed')
    old=OUT/'capture/g4_process_handoffs_v1';cache=OUT/'cache/g4';failed=OUT/'capture/g4_part13_v1'
    debug=read(failed/'DEBUG_TERMINAL.json')
    if not debug['worker_exited'] or not debug['native_failure_captured']:raise RuntimeError('Failed worker outcome uncertain')
    ready=read(failed/'WORKER_READY.json')['identity'];observed=inspect_identity(ready)
    if observed['state'] in ('ALIVE','UNKNOWN'):raise RuntimeError('Failed worker may still be active')
    receipts=read(old/'SEGMENT_RECEIPTS.json')
    if len(receipts)!=13 or any(x['index']!=i or x['source_stop']!=(i+1)*.5 for i,x in enumerate(receipts)):raise RuntimeError('Clean prefix not contiguous')
    for row in receipts:
        folder=Path(row['capture'])
        if read(folder/'TERMINAL.json')['status']!='EXITED_0' or read(folder/'DEBUG_TERMINAL.json')['native_failure_captured'] or read(folder/'RESOURCE.json')['status']!='PASS':raise RuntimeError('Prefix not clean')
    marker=read(cache/'SEGMENT_COMPLETE_12.json');checkpoint=cache/marker['checkpoint']
    if marker['source_seconds']!=6.5 or sha(checkpoint/'manifest.json')!=marker['checkpoint_manifest_sha256']:raise RuntimeError('Clean prefix identity changed')
    repair=OUT/'repairs/checkpoint_hash_v1';repair.mkdir(parents=True,exist_ok=False)
    for name in ('RESUME_AUTHORIZATION.json','INPUT_AND_MODEL_IDENTITY.json','LATEST_CHECKPOINT.json','PROGRESS.json'):
        if (cache/name).exists():shutil.copy2(cache/name,repair/name)
    target=OUT/'capture/g4_process_handoffs_hash_io_v2'
    atomic_json(OUT/'RESUME_SEGMENTS_g4.json',dict(start_index=13,prior_receipts=str(old/'SEGMENT_RECEIPTS.json'),
        parent_capture_directory=str(target),checkpoint_manifest_sha256=marker['checkpoint_manifest_sha256'],capture_suffix='hash_io_v2',
        reason='Native failure was manifest hash reading before runner construction; replace storage plainJSON hash traversal with byte-identical standard encoding',
        failed_process_state_used=False,scientific_or_hash_schema_changed=False))
    atomic_json(repair/'REPAIR.json',dict(regression_sha256=sha(OUT/'CHECKPOINT_HASH_REPAIR_REGRESSION.json'),
        hash_runtime_sha256=sha(ROOT/'tools/checkpoint_json_hash_runtime_v1.py'),
        actual_worker_sha256=sha(ROOT/'tools/wp1_latimer_cache_worker_v1.py'),actual_queue_sha256=sha(ROOT/'tools/wp1_latimer234_queue_v1.py'),
        original_frozen_code_pins_preserved=True,original_normal_prefix_until=6.5,remaining_source_seconds=.6,
        no_native_failed_checkpoint_used=True,new_capture=str(target),native_root_cause_proven=False))
    print(json.dumps(dict(resume_group=4,source_seconds=6.5,remaining_seconds=.6,worker_outcome=observed['state'],prefix_stages=13)))


if __name__=='__main__':main()
