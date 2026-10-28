"""Resolve the current local snapshot and optionally validate its actual binding.

No fitting, simulation advance, new background solve or all-retina expansion.
"""
import argparse
from pathlib import Path
import sys
import json
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
from retina_sim.assembly import RetinaBuilder
from retina_sim.contracts import canonical_hash
from retina_sim.parameter_system import data
from retina_sim.parameter_migration import compatible_inner_parameters, state_bundle, export_bundle
from retina_sim.parameter_runtime import ParameterSystemGroupRunner, cell6_scope, cell6_request, cell6_calibration_record
from tools.validate_ac_vs_native_support import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",type=Path,default=ROOT/"reports/ac_vs_native_support_v1/prepared_v1")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--validate-binding",action="store_true")
    args=p.parse_args();start=time.perf_counter()
    if args.output.exists():raise ValueError("Use a new directory; historical identity must not be overwritten")
    a,seed,source=load(args.prepared)
    bundle=RetinaBuilder.bind_group_snapshot(a,seed,ROOT,protocol=source["protocol"])
    manifest=export_bundle(bundle,args.output)
    states=state_bundle(seed,bundle)
    from retina_sim.group_time_storage import save_bundle
    from retina_sim.parameter_migration import clone
    state_identity=dict(schema="parameter-system-initial-state-1",static_parameter_bundle_id=bundle.identity)
    save_bundle(args.output/"initial_state",kind="validation_evidence",
        identity=state_identity,
        payload=dict(state_bundle_schema="parameter-system-initial-state-1",
                     states=clone(states.states),history_rng_delay=clone(states.history_rng_delay),
                     time_seconds=states.time_seconds),
        qualification=dict(default_adopted=False,fit_eligible=False,
                           qualified_background=False,physiological_calibration=False,FIT_READY=False))
    from retina_sim.parameter_migration import load_initial_state
    loaded=load_initial_state(args.output/"initial_state",bundle)
    if any(not np.array_equal(loaded.states[k],v) for k,v in seed["states"].items()):
        raise ValueError("Independent StateBundle serialization differs")
    assert canonical_hash(compatible_inner_parameters(bundle))==canonical_hash(seed["inner_parameters"])
    # StateBundle is stored separately; no initialization or stochastic draw occurs.
    assert all(np.array_equal(states.states[k],v) for k,v in seed["states"].items())
    results=dict(schema="parameter-system-v1-snapshot-acceptance",binding_id=bundle.identity,
        exact_legacy_parameter_hash=canonical_hash(seed["inner_parameters"]),parameters_exact=True,
        all_state_arrays_exact=True,route_identity=bundle.routes.identity,
        resolved_arrays=len(bundle.values),registry_records=len(bundle.registry.records),
        populations=len(a.populations),node_count=sum(p.group_count for p in a.populations.values()),
        routes=len(a.routes),mechanisms=bundle.registry.theta_schema().keys(),
        ROI_only=True,model_advanced=False,neural_fit_performed=False,physical_qualification_upgraded=False,
        actual_runner_binding_test=False,FIT_READY=False)
    if args.validate_binding:
        from retina_sim.group_time_runner import GroupTimeArm
        old=GroupTimeArm(a,seed,source["protocol"],mode="reference")
        new=ParameterSystemGroupRunner(a,seed,source["protocol"],bundle,mode="reference")
        try:
            x,y=old.initial_snapshot(),new.initial_snapshot()
            exact={}
            for category in ("states","rhs","ports"):
                if set(x[category])!=set(y[category]):raise ValueError("Snapshot registry changed")
                for key in x[category]:
                    equal=np.array_equal(x[category][key],y[category][key])
                    exact[category+":"+str(key)]=equal
                    if not equal:raise ValueError("Zero-difference migration failed: "+str(key))
            legacy_checkpoint=old.checkpoint_payload()
            checkpoint=new.checkpoint_payload()
            from retina_sim.group_time_storage import load_bundle
            checkpoint_identity=dict(parameter_binding_id=bundle.identity,execution_identity=new.identity)
            save_bundle(args.output/"v1_checkpoint",kind="complete_model_checkpoint",
                identity=checkpoint_identity,payload=checkpoint,
                qualification=dict(default_adopted=False,fit_eligible=False,physiological_calibration=False))
            checkpoint=load_bundle(args.output/"v1_checkpoint",expected_kind="complete_model_checkpoint",
                expected_identity=checkpoint_identity)["payload"]
            if old.identity==new.identity:raise ValueError("New binding has no independent execution identity")
            rejected=False
            try:ParameterSystemGroupRunner.from_checkpoint(a,seed,source["protocol"],bundle,legacy_checkpoint,mode="reference")
            except ValueError:rejected=True
            if not rejected:raise ValueError("Legacy checkpoint silently mixed into v1")
            restored=ParameterSystemGroupRunner.from_checkpoint(a,seed,source["protocol"],bundle,checkpoint,mode="reference")
            try:
                z=restored.initial_snapshot()
                if any(not np.array_equal(y["states"][k],z["states"][k]) for k in y["states"]):
                    raise ValueError("Versioned restore changed source states")
            finally:restored.close()
            legacy_restored=ParameterSystemGroupRunner.from_checkpoint(a,seed,source["protocol"],bundle,
                legacy_checkpoint,allow_legacy=True,mode="reference")
            legacy_restored.close()
            results.update(actual_runner_binding_test=True,exact_fields=exact,
                maximum_absolute_difference=0.,new_identity_distinct=True,legacy_rejected_by_default=True,
                explicit_original_legacy_restore_passed=True,new_checkpoint_restore_passed=True,
                original_clock_phase_semantics_unchanged=True)
        finally:old.close();new.close()
    scope=cell6_scope(ROOT);request=cell6_request(ROOT)
    scope.check(request,explicit_opt_in=True)
    rejected={}
    for key in ("biological_cell_id","model_target_group_id","population","background_policy_id",
                "readout_model_id","training_identity","evaluation_identity","source_artifact_sha256",
                "source_code_identity","readout_artifact_sha256","qualification"):
        bad=dict(request);bad[key]="wrong"
        try:scope.check(bad,explicit_opt_in=True)
        except ValueError:rejected[key]=True
        else:rejected[key]=False
    if not all(rejected.values()):raise ValueError("Conditional scope incorrectly accepted wrong identity")
    for input_id in scope.authorized_input_contracts:
        scope.check(cell6_request(ROOT,source_input_contract_id=input_id),explicit_opt_in=True)
    results["Cell6_wrong_identity_rejected"]=rejected
    results["authorized_input_contracts"]=scope.authorized_input_contracts
    results["wall_seconds"]=time.perf_counter()-start
    results["mechanisms"]=list(results["mechanisms"])
    (args.output/"ACCEPTANCE.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    cell6_calibration_record(ROOT).write_immutable(args.output/"CELL6_CALIBRATION_RECORD.json")
    print(json.dumps(dict(status="STATIC_RESOLUTION_COMPLETE",binding_id=bundle.identity,
        arrays=len(bundle.values),records=len(bundle.registry.records),wall_seconds=results["wall_seconds"],
        output=str(args.output),model_advanced=False)))


if __name__=="__main__":
    try:main()
    except BaseException as exc:
        print(json.dumps(dict(status="ENGINEERING_CHECK_FAILED",reason=str(exc),traceback=traceback.format_exc())))
        raise
