"""Resume only the last normally completed prefix with a qualified installed runtime."""
from pathlib import Path
import hashlib,json,shutil,sys

ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.durable_research_runtime_v1 import atomic_json,inspect_identity
OUT=ROOT/'reports/WP1_LATIMER_CELL2_4_V1'
WORK=Path(r'C:\Users\win11-pc\Documents\Codex\2026-10-29\task\hc_revision')


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    pair=OUT/'runtime_isolation_pair_v2'
    if not read(pair/'PAIR_RESULT.json')['passed']:raise RuntimeError('Original numerical qualification failed')
    runtime=read(pair/'PYTHON_RUNTIME_IDENTITY.json')
    if runtime['scientific_versions']!=dict(numpy='2.4.6',scipy='1.17.1',numba='0.67.0',llvmlite='0.49.0'):
        raise RuntimeError('Scientific packages differ')
    dll=Path(r'D:\anaconda\envs\snn_env\python311.dll')
    if sha(dll)!=runtime['interpreter_dll_sha256']:raise RuntimeError('Qualified interpreter changed')
    old=OUT/'capture/g9_process_handoffs_v1';cache=OUT/'cache/g9';failed=OUT/'capture/g9_part13_v1'
    debug=read(failed/'DEBUG_TERMINAL.json')
    if not debug['worker_exited'] or not debug['native_failure_captured']:raise RuntimeError('Failure outcome uncertain')
    observed=inspect_identity(read(failed/'WORKER_READY.json')['identity'])
    if observed['state'] in ('ALIVE','UNKNOWN'):raise RuntimeError('Failed worker might still be active')
    for name in ('queue_resume_v3_hash_io/CONTROLLER.json','capture/g9_process_handoffs_v1/CONTROLLER.json'):
        recorded=read(OUT/name)['identity'];state=inspect_identity(recorded)
        if state['state'] in ('ALIVE','UNKNOWN'):raise RuntimeError('Prior controller might still be active: '+name)
    receipts=read(old/'SEGMENT_RECEIPTS.json')
    if len(receipts)!=13 or any(r['index']!=i or r['source_stop']!=(i+1)*.5 for i,r in enumerate(receipts)):
        raise RuntimeError('Normal prefix is not contiguous')
    for row in receipts:
        folder=Path(row['capture'])
        if read(folder/'TERMINAL.json')['status']!='EXITED_0' or read(folder/'DEBUG_TERMINAL.json')['native_failure_captured'] or read(folder/'RESOURCE.json')['status']!='PASS':
            raise RuntimeError('Prefix has an unsuccessful segment')
    marker=read(cache/'SEGMENT_COMPLETE_12.json');checkpoint=cache/marker['checkpoint']
    if marker['source_seconds']!=6.5 or sha(checkpoint/'manifest.json')!=marker['checkpoint_manifest_sha256']:
        raise RuntimeError('Clean checkpoint identity changed')
    repair=OUT/'repairs/existing_python_v1';repair.mkdir(parents=True,exist_ok=False)
    for name in ('RESUME_AUTHORIZATION.json','INPUT_AND_MODEL_IDENTITY.json','LATEST_CHECKPOINT.json','PROGRESS.json'):
        if (cache/name).exists():shutil.copy2(cache/name,repair/name)
    target=OUT/'capture/g9_process_handoffs_existing_python_v2'
    atomic_json(OUT/'RESUME_SEGMENTS_g9.json',dict(start_index=13,prior_receipts=str(old/'SEGMENT_RECEIPTS.json'),
        parent_capture_directory=str(target),checkpoint_manifest_sha256=marker['checkpoint_manifest_sha256'],capture_suffix='existing_python_v2',
        reason='Qualified existing CPython3.11.15 build with the exact original scientific package files; failed process state discarded',
        failed_process_state_used=False,scientific_equations_or_numerical_gates_changed=False))
    sources=['wp1_existing_python_runtime_v1.py','wp1_latimer_segmented_cache_v1.py','wp1_latimer234_queue_v1.py',
        'wp1_latimer234_fit_worker_v1.py','wp1_latimer234_fit_v1.py']
    atomic_json(repair/'REPAIR.json',dict(pair_result_sha256=sha(pair/'PAIR_RESULT.json'),
        qualified_runtime=runtime,actual_source_sha256={n:sha(ROOT/'tools'/n) for n in sources},
        wrapper_sha256={n:sha(WORK/n) for n in ('wp1_capture_latimer_cache_existing_python_v2.ps1','wp1_capture_latimer_fit_existing_python_v2.ps1')},
        original_frozen_protocol_preserved=True,normal_prefix_until=6.5,remaining_source_seconds=.6,
        qualification_scope='501 nodes over original 1.0-1.05s; original gates, events, PCG, history and restore',
        post_qualification_bootstrap_change='Only fit-worker thread environment before scientific imports and provenance metadata; cache branch unchanged',
        interpreter_difference_is_execution_identity=True,failed_native_prefix_not_used=True,
        root_cause_proven=False,installation_or_system_changes=False,automatic_native_retry=False))
    print(json.dumps(dict(resume_group=9,source_seconds=6.5,remaining_seconds=.6,old_worker=observed['state'],normal_prefix_segments=13)))


if __name__=='__main__':main()
