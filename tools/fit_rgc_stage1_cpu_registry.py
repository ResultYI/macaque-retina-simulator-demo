"""Original RGC fit/readout over validated read-only CPU cache locations."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from retina_sim.contracts import ContractError
from retina_sim.rgc_stage1_cache import write,read_cache
from tools.merge_rgc_stage1_cpu_cache import inspect

ROOT=Path(__file__).resolve().parents[1]

def report_plain(value):
    """Lossless JSON report representation; no change to evaluated values."""
    if isinstance(value,np.ndarray):return report_plain(value.tolist())
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:report_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [report_plain(v) for v in value]
    return value


def write_report(path,value):
    write(path,report_plain(value))


def run(registry,output,guard):
    from tools import fit_rgc_stage1 as original
    spec=original.verify();output=Path(output);output.mkdir(exist_ok=True)
    registry_path=Path(registry);reg=json.loads(registry_path.read_text())
    if not reg.get('all_six_complete') or len(reg['caches'])!=6:raise ContractError('All six complete endogenous CPU caches required')
    paths={c['case_id']:Path(c['directory']) for c in reg['caches']}
    for case in spec['cases']:inspect(paths[case['case_id']],spec,case)
    old_output=original.OUT;old_loader=original.load_fit_cases;old_fit=original.fit;old_write=original.write
    consistency_path=old_output/'consistency/RESULT.json';consistency=json.loads(consistency_path.read_text())
    if not consistency['passed']:raise ContractError('Existing same-source native/cache readout consistency failed')
    write(output/'REUSED_CONSISTENCY_PROVENANCE.json',dict(source=str(consistency_path),source_sha256=hashlib.sha256(consistency_path.read_bytes()).hexdigest(),
        reused_same_source_and_numerical_CPU_contract=True,new_model_check_performed=False))
    if not (output/'consistency').exists():(output/'consistency').mkdir();write(output/'consistency/RESULT.json',consistency)
    def load_cases(frozen,split):
        cases=[]
        with np.load(old_output/'actual_observations.npz',allow_pickle=False) as observed:
            for case in frozen['cases']:
                if case['split']!=split:continue
                guard();data,_=read_cache(paths[case['case_id']],frozen,case)
                with data:voltage=data[case['population']+':V'][::10,case['model_column']][:-1].copy()
                counts=observed[case['counts_key']].copy()
                cases.append(dict(case,voltage=voltage,counts=counts,past=original.observed_past(counts)))
        return cases
    coordinator_guard=guard
    def guarded_fit(*args,guard=None,**kwargs):
        def combined_guard():
            coordinator_guard()
            if guard is not None:guard()
        return old_fit(*args,guard=combined_guard,**kwargs)
    original.OUT=output;original.load_fit_cases=load_cases;original.fit=guarded_fit;original.write=write_report
    try:
        guard();began=time.perf_counter()
        if not (output/'TRAINING_SELECTION.json').exists():original.optimize(spec)
        guard();selection_finished=time.perf_counter()
        if not (output/'CELL6_RESULT.json').exists():original.evaluate(spec)
        result=json.loads((output/'CELL6_RESULT.json').read_text())
        write(output/'FIT_EXECUTION_RECEIPT.json',dict(registry=str(registry_path),registry_sha256=hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            fitting_wall_this_invocation_seconds=selection_finished-began,validation_wall_this_invocation_seconds=time.perf_counter()-selection_finished,
            original_scientific_contract_unchanged=True,original_fit_math_unchanged=True,only_cache_paths_and_report_destination_adapted=True,
            old_test_not_reopened=True,accepted=result['accepted'],other_RGC_types_not_claimed_fitted=True,FIT_READY=False))
        return result
    finally:original.OUT=old_output;original.load_fit_cases=old_loader;original.fit=old_fit;original.write=old_write
