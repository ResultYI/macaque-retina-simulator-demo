"""Qualified execution-only transfer from the last normally completed g14 checkpoint."""
from pathlib import Path
import hashlib,json,shutil,sys
ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.durable_research_runtime_v1 import atomic_json,inspect_identity
OUT=ROOT/'reports/WP1_LATIMER_CELL2_4_V1'
WORK=Path(r'C:\Users\win11-pc\Documents\Codex\2026-10-29\task\hc_revision')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    pair=OUT/'runtime_portable_codegen_pair_v1';result=read(pair/'PAIR_RESULT.json');codegen=read(pair/'ACTUAL_COMPILED_CODEGEN.json')
    if not result['passed'] or codegen['magic_tuple'][1:]!=['generic',''] or not codegen['SVML_disabled']:
        raise RuntimeError('Portable actual code generation or original numerical gate differs')
    failed=OUT/'capture/g14_part06_v1';debug=read(failed/'DEBUG_TERMINAL.json')
    if not debug['worker_exited'] or debug['exit_code']!=1 or debug['native_failure_captured']:raise RuntimeError('Failure exit differs')
    if 'setobject.c:2342: bad argument to internal function' not in read(failed/'FAILURE.json')['error']:raise RuntimeError('Failure evidence changed')
    observed=inspect_identity(read(failed/'WORKER_READY.json')['identity'])
    if observed['state'] in ('ALIVE','UNKNOWN'):raise RuntimeError('Failed worker outcome uncertain')
    for name in ('queue_resume_v6_numpy_baseline/CONTROLLER.json','capture/g14_process_handoffs_v1/CONTROLLER.json'):
        if inspect_identity(read(OUT/name)['identity'])['state'] in ('ALIVE','UNKNOWN'):raise RuntimeError('Prior owner outcome uncertain')
    old=OUT/'capture/g14_process_handoffs_v1';receipts=read(old/'SEGMENT_RECEIPTS.json');cache=OUT/'cache/g14'
    if len(receipts)!=6 or any(r['index']!=i or r['source_stop']!=(i+1)*.5 for i,r in enumerate(receipts)):
        raise RuntimeError('Normally completed prefix is not contiguous')
    for row in receipts:
        folder=Path(row['capture'])
        if read(folder/'TERMINAL.json')['status']!='EXITED_0' or read(folder/'DEBUG_TERMINAL.json')['native_failure_captured'] or read(folder/'RESOURCE.json')['status']!='PASS':
            raise RuntimeError('Prefix not normally completed')
    marker=read(cache/'SEGMENT_COMPLETE_05.json')
    if marker['source_seconds']!=3. or sha(cache/marker['checkpoint']/'manifest.json')!=marker['checkpoint_manifest_sha256']:
        raise RuntimeError('Complete3s checkpoint identity changed')
    repair=OUT/'repairs/portable_codegen_v1';repair.mkdir(exist_ok=False)
    for name in ('RESUME_AUTHORIZATION.json','INPUT_AND_MODEL_IDENTITY.json','LATEST_CHECKPOINT.json','PROGRESS.json'):
        if (cache/name).exists():shutil.copy2(cache/name,repair/name)
    target=OUT/'capture/g14_process_handoffs_portable_v1'
    atomic_json(OUT/'RESUME_SEGMENTS_g14.json',dict(start_index=6,prior_receipts=str(old/'SEGMENT_RECEIPTS.json'),
        parent_capture_directory=str(target),capture_suffix='portable_v1',checkpoint_manifest_sha256=marker['checkpoint_manifest_sha256'],
        reason='Qualified portable CPU codegen/SVML-free execution; actual Python/runtime identity changes are explicit, original scientific state and clocks retained',
        failed_process_state_used=False,failed3_1s_trace_not_checkpoint=True,execution_identity_changes=True,
        source_runtime=read(failed/'PYTHON_RUNTIME_IDENTITY.json'),destination_runtime=read(pair/'PYTHON_RUNTIME_IDENTITY.json'),
        scientific_equations_or_numerical_gates_changed=False))
    atomic_json(repair/'REPAIR.json',dict(failure=read(failed/'FAILURE.json'),failed_capture=str(failed),
        hypotheses=['Unchecked native write or invalid buffer lifetime','Host-specific native code generation/ABI or math runtime','Other process or hardware corruption not established'],
        observed_bounds_checks_already_enabled=True,out_of_bounds_exception_observed=False,NumPy_CPU_baseline_alone_not_sufficient=True,
        root_cause_proven=False,hardware_cause_claimed=False,scope='Explicitly isolated worker-level compatibility candidate; not a claimed identified root fix',
        pair_result_sha256=sha(pair/'PAIR_RESULT.json'),actual_codegen=codegen,maximum_state_budget_ratio=result['maximum_budget_ratio'],
        actual_source_sha256={n:sha(ROOT/'tools'/n) for n in ('wp1_existing_python_runtime_v3.py','wp1_portable_codegen_pair_v1.py','wp1_latimer_segmented_cache_v1.py')},
        wrapper_sha256=sha(WORK/'wp1_capture_latimer_cache_portable_v4.ps1'),
        scientific_prefix_until=3.,remaining_source_seconds=4.1,first13_complete_caches_retained=True,
        frozen_model_sha256={str(c):sha(OUT/'targets'/f'Latimer_Cell_{c}'/'MODELS_FROZEN.json') for c in (2,3,4)},
        installation_or_system_changes=False,automatic_native_retry=False))
    print(json.dumps(dict(ready=True,resume_group=14,normal_checkpoint_seconds=3.,preserved_complete_caches=13,frozen_cells=[2,3,4])))


if __name__=='__main__':main()
