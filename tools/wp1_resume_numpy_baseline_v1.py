"""Preserve the failed initial observation and start only g10 afresh after qualification."""
from pathlib import Path
import hashlib,json,shutil,sys
ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.durable_research_runtime_v1 import atomic_json,inspect_identity
OUT=ROOT/'reports/WP1_LATIMER_CELL2_4_V1'
WORK=Path(r'C:\Users\win11-pc\Documents\Codex\2026-10-29\task\hc_revision')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    pair=OUT/'runtime_baseline_simd_pair_v1';result=read(pair/'PAIR_RESULT.json')
    if not result['passed']:raise RuntimeError('Original numerical qualification failed')
    runtime=read(pair/'PYTHON_RUNTIME_IDENTITY.json')
    if runtime['numpy_minimum_dispatch']['minimum']['ddd']['current']!='baseline(X86_V2)':raise RuntimeError('Qualified CPU dispatch differs')
    failed=OUT/'capture/g10_part00_v1';debug=read(failed/'DEBUG_TERMINAL.json')
    if not debug['worker_exited'] or not debug['native_failure_captured']:raise RuntimeError('Native worker exit uncertain')
    observed=inspect_identity(read(failed/'WORKER_READY.json')['identity'])
    if observed['state'] in ('ALIVE','UNKNOWN'):raise RuntimeError('Failed worker could still be active')
    for name in ('queue_resume_v5_input_witness/CONTROLLER.json','capture/g10_process_handoffs_v1/CONTROLLER.json'):
        if inspect_identity(read(OUT/name)['identity'])['state'] in ('ALIVE','UNKNOWN'):raise RuntimeError('Prior owner could be active')
    cache=OUT/'cache/g10'
    if any(cache.glob('CHECKPOINT_*')) or (cache/'manifest.json').exists() or (cache/'endogenous.npz').exists():
        raise RuntimeError('A saved trajectory exists; do not discard or treat it as a fresh start')
    repair=OUT/'repairs/numpy_baseline_v1';repair.mkdir(exist_ok=False)
    if not cache.resolve().is_relative_to(OUT.resolve()):raise RuntimeError('Owned path escaped')
    shutil.move(str(cache),str(repair/'failed_initial_g10'))
    atomic_json(repair/'EMPTY_PRIOR_RECEIPTS.json',[])
    target=OUT/'capture/g10_process_handoffs_numpy_baseline_v1'
    atomic_json(OUT/'RESUME_SEGMENTS_g10.json',dict(start_index=0,prior_receipts=str(repair/'EMPTY_PRIOR_RECEIPTS.json'),
        parent_capture_directory=str(target),capture_suffix='numpy_baseline_v1',checkpoint_manifest_sha256=None,
        reason='Native failure at initial monitor observation before any integration; no checkpoint used; process-only NumPy baseline CPU dispatch qualified',
        failed_process_state_used=False,scientific_equations_or_numerical_gates_changed=False))
    atomic_json(repair/'REPAIR.json',dict(failed_capture=str(failed),exception=read(failed/'EXACT_NATIVE_EXCEPTION.json'),
        observed_python_frame='Monitor.observe, minimum(row.lo,a), offset2078; before first integration',
        native_module='Original NumPy _multiarray_umath.cp311-win_amd64.pyd',native_module_offset='0x2a3b66',
        module_sha256=sha(ROOT/'.venv/Lib/site-packages/numpy/_core/_multiarray_umath.cp311-win_amd64.pyd'),
        isolation_changes_alone_not_sufficient=True,root_cause_proven=False,hardware_cause_claimed=False,
        targeted_change='NPY_DISABLE_CPU_FEATURES=X86_V3 only for these cache worker processes, before NumPy import',
        qualified_runtime=runtime,pair_result_sha256=sha(pair/'PAIR_RESULT.json'),maximum_state_budget_ratio=result['maximum_budget_ratio'],
        original_fit_parameters_preserved={str(c):sha(OUT/'targets'/f'Latimer_Cell_{c}'/'MODELS_FROZEN.json') for c in (2,3,4)},
        actual_source_sha256={n:sha(ROOT/'tools'/n) for n in ('wp1_existing_python_runtime_v2.py','wp1_latimer_segmented_cache_v1.py')},
        wrapper_sha256=sha(WORK/'wp1_capture_latimer_cache_numpy_baseline_v3.ps1'),
        first9_completed_caches_not_recomputed=True,installation_or_system_changes=False,automatic_native_retry=False))
    print(json.dumps(dict(ready=True,fresh_group=10,preserved_complete_caches=9,frozen_cells=[2,3,4],no_failed_process_state_used=True)))


if __name__=='__main__':main()
