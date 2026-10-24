"""Small manifest handoff; no array/model import, source capture, or fitting.

The main producer owns original graph/background qualification and resource
limits. This module checks preparation references and can supply an identity seed
only after an externally recorded human policy approval and frontend receipt.
An identity seed is never a qualified cache or permission to start computation.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PLAN = "tools/vr_teacher_handoff_v1/fit_integration_v1/capture_preparation_v2/handoff.json"
GROUPS = ("midget_RGC_ON", "midget_RGC_OFF", "parasol_RGC_ON", "parasol_RGC_OFF")
CLOCK = {
    "input_hz": 150,
    "raw_source_frames": [0, 9751],
    "recorded_background_frames": [0, 750],
    "active_frames": [750, 9750],
    "guard_frames": [9750, 9751],
    "raw_active_seconds": [5, 65],
    "effective_active_seconds": [0, 60],
    "raw_to_effective_offset_seconds": 5,
    "guard_contributes_target_or_loss": False,
    "reset_at_split": False,
}
SPLITS = {"train": [0, 35], "validation": [40, 50], "preliminary_test": [55, 60],
          "no_loss_state_continuation_gaps": [[35, 40], [50, 55]],
          "final_holdout": [480, 600], "final_holdout_used": False}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_bound(root, reference):
    root = Path(root).resolve()
    path = (root / reference["path"]).resolve()
    if not path.is_relative_to(root) or path.stat().st_size > 1024 * 1024:
        raise ValueError("Preparation reference outside root or small-file budget")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("Preparation reference identity changed: " + reference["path"])
    return json.loads(raw)


def load_preparation(root, *, plan_path=PLAN):
    root = Path(root).resolve()
    path = (root / plan_path).resolve()
    if not path.is_relative_to(root) or path.stat().st_size > 1024 * 1024:
        raise ValueError("Invalid preparation plan path/budget")
    plan = json.loads(path.read_bytes())
    if plan.get("status") != "NOT_READY" or plan.get("fit_eligible") is not False:
        raise ValueError("A preparation manifest cannot declare itself fit eligible")
    if plan["clock"] != CLOCK or plan["splits"] != SPLITS:
        raise ValueError("Original continuous time/split contract changed")
    refs = plan["references"]
    objects = {key: read_bound(root, value) for key, value in refs.items()}
    cells = objects["cells"]["cells"]
    config = objects["capture_configuration"]
    contract = objects["pilot_contract"]
    if (tuple(row["student_type_group"] for row in cells) != GROUPS
            or [row["teacher_column"] for row in cells] != list(range(4))
            or len({row["student_cell_id"] for row in cells}) != 4):
        raise ValueError("Four fixed functional target identities/order required")
    expected = deepcopy(plan["retained_eight_owner_rows"])
    if sum(map(len, expected.values())) != 8:
        raise ValueError("Original eight output rows must be preserved")
    for row in cells:
        expected.setdefault(row["student_component"], []).append(row["student_global_row"])
    expected = {key: sorted(set(value)) for key, value in expected.items()}
    if config["owner_global_rows"] != expected or sum(map(len, expected.values())) != 12:
        raise ValueError("Capture must retain the exact 8+4 comparison cohort")
    if (config.get("fit_eligible") is not False or config["crop_recurrent_dependencies"] is not False
            or contract.get("fit_eligible") is not False or contract["splits"] != SPLITS
            or contract["cells"] != cells):
        raise ValueError("Prepared graph/contract lost frozen targets or qualification gate")
    policy = objects["input_policy"]
    read_bound(root, policy["provenance"]["approved_display_profile"])
    read_bound(root, policy["provenance"]["approved_global_input_plan"])
    conversion = contract["input"]["native_light_conversion"]
    if (policy["approval_status"] != "PENDING_USER_APPROVAL"
            or policy["fit_eligible"] is not False
            or config["source_selection_binding"] != refs["cells"]
            or contract["received_binding_sha256"] != refs["received_assets"]["sha256"]
            or config["input_plan"] != policy["provenance"]["approved_global_input_plan"]
            or conversion["policy_proposal"] != refs["input_policy"]
            or contract["source_capture_configuration"] != refs["capture_configuration"]):
        raise ValueError("Proposed input is not an approved calibration")
    return plan, objects


def preflight(root, *, plan_path=PLAN):
    plan, objects = load_preparation(root, plan_path=plan_path)
    return {"status": "NOT_READY", "fit_eligible": False,
            "blockers": list(plan["blockers"]),
            "cell_count": len(objects["cells"]["cells"]), "capture_output_count": 12,
            "policy_proposal_sha256": plan["references"]["input_policy"]["sha256"],
            "counts_shape": [9000, 4], "clock": plan["clock"], "splits": plan["splits"],
            "model_initialized": False, "background_qualification_started": False,
            "capture_started": False, "fit_started": False}


def approved_policy(root, approval_receipt, *, plan_path=PLAN):
    """Read proposal after an externally recorded, exact human decision."""
    plan, objects = load_preparation(root, plan_path=plan_path)
    policy_sha = plan["references"]["input_policy"]["sha256"]
    if (not isinstance(approval_receipt, dict)
            or approval_receipt.get("approved_by") != "human_user"
            or approval_receipt.get("decision") != "ACCEPT_EXACT_ENGINEERING_INPUT_POLICY"
            or approval_receipt.get("policy_sha256") != policy_sha
            or not approval_receipt.get("trusted_conversation_evidence_id")):
        raise ValueError("Exact human input-policy approval is absent/mismatched")
    return plan, objects


def bind_input_contract(root, *, approval_receipt, frontend_receipt, plan_path=PLAN):
    """Derive an input-bound, still NOT_READY contract in memory. Never save/run."""
    plan, objects = approved_policy(root, approval_receipt, plan_path=plan_path)
    policy_sha = plan["references"]["input_policy"]["sha256"]
    required = ("implementation_sha256", "global_reference_numeric_sha256", "spectral_basis_sha256",
                "capture_geometry_sha256", "actual_background_input_sha256")
    if (not isinstance(frontend_receipt, dict)
            or frontend_receipt.get("policy_sha256") != policy_sha
            or any(not isinstance(frontend_receipt.get(k), str)
                   or len(frontend_receipt[k]) != 64
                   or any(c not in "0123456789abcdef" for c in frontend_receipt[k]) for k in required)
            or frontend_receipt.get("same_global_reference_for_all_rois") is not True
            or frontend_receipt.get("per_frame_or_cell_normalization") is not False
            or frontend_receipt.get("physical_calibration_claim") is not False):
        raise ValueError("Explicit same-policy fixed frontend receipt required")
    contract = deepcopy(objects["pilot_contract"])
    contract["status"] = "NOT_READY_ORIGINAL_BACKGROUND_AND_CACHE_PENDING"
    contract["input"]["native_light_conversion"].update(
        status="EXPLICITLY_BOUND_UNCHANGED_POLICY",
        policy_id=objects["input_policy"]["policy_id"], policy_sha256=policy_sha,
        approval_receipt={"sha256": canonical(approval_receipt), "receipt": deepcopy(approval_receipt)},
        frontend_receipt={"sha256": canonical(frontend_receipt), "receipt": deepcopy(frontend_receipt)})
    return contract


def source_identity_seed(root, *, approval_receipt, frontend_receipt, plan_path=PLAN):
    """Describe fixed source identity after approval; never starts/qualifies it.

    Approval must be recorded by the main thread from the human's actual answer,
    not inferred from agreeing to chunked execution or generated by this module.
    Original native producers/loaders still enforce background and cache content.
    """
    contract = bind_input_contract(root, approval_receipt=approval_receipt,
                                   frontend_receipt=frontend_receipt, plan_path=plan_path)
    plan, objects = load_preparation(root, plan_path=plan_path)
    policy_sha = plan["references"]["input_policy"]["sha256"]
    condition = {"input": contract["input"],
                 "policy_sha256": policy_sha, "frontend": frontend_receipt,
                 "clock": plan["clock"], "splits": plan["splits"]}
    return {"status": "IDENTITY_SEED_ONLY_NOT_QUALIFIED", "fit_eligible": False,
            "boundary_mode": "rgc_only", "boundary_schema": "ac-rgc-boundary-replay-1.2",
            "input_condition_hash": canonical(contract["input"]),
            "frontend_condition_hash": canonical(condition),
            "calibration_hash": canonical(condition),
            "stimulus_hash": objects["received_assets"]["stimulus_index"]["sha256"],
            "source_counts_hash": objects["received_assets"]["assets"]["counts"]["sha256"],
            "input_policy_sha256": policy_sha,
            "frontend_receipt_sha256": canonical(frontend_receipt),
            "policy_approval_receipt_sha256": canonical(approval_receipt),
            "source_configuration_sha256": plan["references"]["capture_configuration"]["sha256"],
            "target_binding_sha256": plan["references"]["received_assets"]["sha256"],
            "student_cell_binding_sha256": plan["references"]["cells"]["sha256"],
            "split_policy_hash": canonical(plan["splits"]),
            "remaining_native_bindings": list(plan["cache_contract"]["native_bindings_required"])}


def main():
    parser = argparse.ArgumentParser(description="Read small preparation manifests only; never run a model")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.project_root), indent=2))
    return 2  # Preparation is deliberately not a production/fit-ready assertion.


if __name__ == "__main__":
    raise SystemExit(main())
