"""Real pilot target/input binding and one existing-model replay evaluation.

The CLI only preflights. evaluate_candidate is callable after a qualified source
cache is supplied; it never assembles the upstream graph or starts a search.
"""
import argparse
from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np

from .pilot_temporal import replay_rates_on_teacher_edges

GROUPS = ("midget_RGC_ON", "midget_RGC_OFF", "parasol_RGC_ON", "parasol_RGC_OFF")
KINDS = ("rgc_excitation_gain", "rgc_inhibition_gain")
RANGES = {"train": (0, 5250), "validation": (6000, 7500), "preliminary_test": (8250, 9000)}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def local(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Pilot path escapes its declared root")
    return path


class Gray8Chunks:
    """One immutable stored uint8 chunk at a time, never reconstruct Y_norm."""
    def __init__(self, root, index, *, frame_shape=(256, 256), max_chunk_bytes=16 * 1024 * 1024):
        self.root = Path(root)
        self.entries = tuple(index["chunks"])
        self.starts = tuple(e["start"] for e in self.entries)
        self.frame_shape = tuple(frame_shape)
        self.max_chunk_bytes = max_chunk_bytes
        self._cached_index = None
        self._cached = None
        if not self.entries or self.starts[0] != 0 or any(a["stop"] != b["start"] for a, b in zip(self.entries, self.entries[1:])):
            raise ValueError("Contiguous source frame inventory required")

    def frame(self, source_index):
        if type(source_index) is not int or not 0 <= source_index < 9751:
            raise ValueError("Only original background, active movie and one genuine guard frame are bound")
        j = bisect_right(self.starts, source_index) - 1
        entry = self.entries[j]
        if source_index >= entry["stop"]:
            raise ValueError("Frame is outside the supplied chunk")
        if self._cached_index != j:
            path = local(self.root, entry["path"])
            if path.stat().st_size != entry["bytes"]:
                raise ValueError("Stored chunk size changed")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while True:
                    block = stream.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError("Stored gray8 chunk identity changed")
            with np.load(path, allow_pickle=False) as data:
                # No access to Y_norm, even if it is present in the same NPZ.
                value = data["gray8"]
                indices = data["source_index"]
            expected = (entry["stop"] - entry["start"],) + self.frame_shape
            if value.dtype != np.uint8 or value.shape != expected or value.nbytes > self.max_chunk_bytes or not np.array_equal(indices, np.arange(entry["start"], entry["stop"])):
                raise ValueError("Stored gray8 layout/source indices differ")
            value.setflags(write=False)
            self._cached, self._cached_index = value, j
        # Bytes-backed view prevents callers from changing future cached input.
        return np.frombuffer(self._cached[source_index - entry["start"]].tobytes(), dtype=np.uint8).reshape(self.frame_shape)

    def frame_at_raw_seconds(self, relative_seconds):
        if not np.isfinite(relative_seconds) or not 0 <= relative_seconds < 9751 / 150:
            raise ValueError("Input time outside the complete recorded run")
        # Explicit boundaries preserve [left,right) sides without epsilon shifts.
        index = int(np.searchsorted(np.arange(9752, dtype=float) / 150, relative_seconds, side="right") - 1)
        return self.frame(index)


@dataclass(frozen=True)
class PilotTarget:
    counts: np.ndarray
    binding_sha256: str
    cell_ids: tuple
    target_identity: str

    def loss(self, predicted_hz, *, role="train", frozen_selection=None, candidate_hash=None, contract_hash=None):
        prediction = np.asarray(predicted_hz, dtype=float)
        if prediction.shape != (9000, 4) or not np.isfinite(prediction).all() or np.any(prediction < 0):
            raise ValueError("Complete ordered finite expected rate required")
        if role not in RANGES:
            raise ValueError("Gaps/final holdout are not scoreable roles")
        if role != "train":
            receipt = frozen_selection or {}
            if (receipt.get("selected_using") != "train_only" or not candidate_hash or not contract_hash
                    or receipt.get("candidate_hash") != candidate_hash or receipt.get("contract_hash") != contract_hash
                    or receipt.get("target_identity") != self.target_identity):
                raise ValueError("Heldout scoring requires an identity-bound frozen training selection")
        left, right = RANGES[role]
        error = prediction[left:right] - self.counts[left:right].astype(float) * 150
        return float(np.mean(error * error))


def load_target(contract, project_root):
    root = local(Path(project_root), contract["intake_root"])
    binding_path = local(root, "received_asset_binding.json")
    raw = binding_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != contract["received_binding_sha256"]:
        raise ValueError("Received binding identity changed")
    binding = json.loads(raw)
    spec = binding["assets"]["counts"]
    path = local(root, spec["path"])
    if path.stat().st_size > 1024 * 1024 or hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError("Actual small target identity differs")
    with np.load(path, allow_pickle=False) as data:
        value = data["counts"]
        if value.shape != (9000, 4) or value.dtype != np.uint32 or not np.array_equal(data["vr_unit_id"], np.arange(4)):
            raise ValueError("Actual counts shape/dtype/order differs")
        if not np.array_equal(data["source_index"], np.arange(750, 9750)):
            raise ValueError("Target source frame origin differs")
        for name, (start, stop) in {"train_eligible": (0, 5250), "validation_eligible": (6000, 7500), "pilot_test_eligible": (8250, 9000)}.items():
            expected = np.zeros(9000, dtype=bool); expected[start:stop] = True
            if not np.array_equal(data[name], expected):
                raise ValueError("Target fixed temporal split differs")
        if np.any(data["final_holdout_eligible"]):
            raise ValueError("Final holdout must remain unused")
        immutable = np.frombuffer(value.tobytes(), dtype=np.uint32).reshape(9000, 4)
    cells = tuple(row["student_cell_id"] for row in contract["cells"])
    return PilotTarget(immutable, contract["received_binding_sha256"], cells,
                       canonical({"received_binding": contract["received_binding_sha256"], "counts": spec["sha256"], "cells": cells}))


def candidate_from_values(contract, values):
    parameters = contract["parameters"]
    if len(parameters) != 2 or {p["kind"] for p in parameters} != set(KINDS):
        raise ValueError("Contract cannot open another parameter stage")
    if any(p["bounds"] != [.5, 2.] or tuple(p["groups"]) != GROUPS or p.get("initial", 1.) != 1.
           for p in parameters):
        raise ValueError("Contract changed the approved bounds/shared type bindings")
    if set(values) != set(KINDS):
        raise ValueError("Exactly the two approved shared RGC E/I multipliers are required")
    candidate = {}
    for parameter in contract["parameters"]:
        value = values[parameter["kind"]]
        lo, hi = parameter["bounds"]
        if type(value) not in (float, int) or not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError("Candidate exceeds the approved engineering domain")
        candidate[parameter["kind"]] = {name: float(value) for name in parameter["groups"]}
    return candidate


def preflight(contract, project_root):
    candidate_from_values(contract, {kind: 1. for kind in KINDS})
    if contract["loss"]["kind"] != "uniform_mean_squared_rate_error_Hz2" or contract["loss"]["teacher_bin_seconds"] != 1 / 150:
        raise ValueError("Pilot loss/time grid is not the fixed approved objective")
    target = load_target(contract, project_root)
    blockers = []
    source = contract["source_cache"]
    if contract["input"]["native_light_conversion"]["status"] != "EXPLICITLY_BOUND_UNCHANGED_POLICY":
        blockers.append("NATIVE_LIGHT_CALIBRATION_NOT_BOUND")
    if contract["qualified_background"]["status"] != "CURRENT_ORIGINAL_POLICY_QUALIFIED":
        blockers.append("NO_CURRENT_INPUT_AND_GRAPH_QUALIFIED_BACKGROUND")
    if not source.get("path") or not source.get("expected_identity_hash") or not source.get("expected_content_hash"):
        blockers.append("NO_SAME_INPUT_RGC_ONLY_BOUNDARY_CACHE")
    else:
        path = local(Path(project_root), source["path"])
        metadata = json.loads((path / "manifest.json").read_bytes())
        if metadata.get("schema_version") != "ac-rgc-boundary-replay-1.2" or metadata.get("mode") != "rgc_only":
            blockers.append("SOURCE_CACHE_SCHEMA_OR_MODE_DIFFERS")
        if metadata.get("identity_hash") != source["expected_identity_hash"] or metadata.get("content_hash") != source["expected_content_hash"]:
            blockers.append("SOURCE_CACHE_IDENTITY_DIFFERS")
        if metadata.get("identity", {}).get("input_condition_hash") != canonical(contract["input"]):
            blockers.append("SOURCE_CACHE_INPUT_CONDITION_DIFFERS")
        if (metadata.get("initial", {}).get("preparation") or {}).get("status") != "verified_original_background_contract":
            blockers.append("SOURCE_CACHE_PREPARATION_NOT_ORIGINALLY_QUALIFIED")
    return {"schema_version": "vr-pilot-execution-preflight-1.0", "target_shape": [9000, 4],
            "target_identity": target.target_identity, "student_cells": list(target.cell_ids),
            "blockers": blockers, "ready_for_parent_to_authorize_bounded_evaluation": not blockers,
            "model_initialized": False, "upstream_graph_assembled": False, "fit_started": False}


def evaluate_candidate(contract, project_root, values):
    """Future one-candidate CPU replay through the existing runner; no optimizer."""
    deadline = monotonic() + contract["budget"]["max_evaluation_seconds"]
    def guard(*args):
        if monotonic() > deadline:
            raise TimeoutError("Bounded pilot evaluation wall cap reached")
    status = preflight(contract, project_root)
    if status["blockers"]:
        raise ValueError("Pilot evaluation blocked: " + ", ".join(status["blockers"]))
    from retina_sim.boundary_replay import BoundaryCache, BoundaryReplay
    from retina_sim.recording import RecordingPlan
    root = Path(project_root)
    spec = contract["source_cache"]
    cache = BoundaryCache.load(local(root, spec["path"]), expected_identity_hash=spec["expected_identity_hash"])
    guard()
    if cache.content_hash != spec["expected_content_hash"]:
        raise ValueError("Numerical boundary content identity differs")
    candidate = candidate_from_values(contract, values)
    replay = BoundaryReplay(cache, root, candidate=candidate, compute_backend="python")
    state = replay.initialize()
    origin = spec["input_raw_zero_absolute_seconds"]
    # Finish on a complete native bin, with the genuine guard providing coverage.
    stop = state.grid_origin_seconds + np.ceil((origin + 65 - state.grid_origin_seconds) / .001) * .001
    if cache.times[0] > origin or cache.times[-1] < origin + 9751 / 150:
        raise ValueError("Cache must include actual background, entire movie and genuine guard")
    if state.time_seconds != origin:
        raise ValueError("No intermediate restart or time rebasing")
    ticks = np.arange(int(round((stop - state.grid_origin_seconds) / .001)) + 1)
    times = state.grid_origin_seconds + ticks * .001
    times = times[(times >= origin) & (times <= stop)]
    times = np.unique(np.r_[origin, times, stop])
    ports = tuple(cid + ".voltage" for cid in state.spike_states)
    _, result = replay.runner.advance(state, replay.channels, stop, replay.config, sample_times=times,
        recording_plan=RecordingPlan(states=(), ports=ports), state_observer=guard)
    edges = origin + 5 + np.arange(9001, dtype=float) / 150
    target = load_target(contract, root)
    rates, errors = replay_rates_on_teacher_edges(replay, result, edges, target.cell_ids)
    guard()
    return {"candidate": candidate, "objective_Hz2": target.loss(rates), "role": "train",
            "target_identity": target.target_identity, "candidate_hash": canonical(candidate),
            "cache_identity": cache.identity_hash, "cache_content_hash": cache.content_hash,
            "expected_rate_hz": rates, "history_probability_mass_errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Preflight only; no model/optimizer starts")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    value = preflight(json.loads(args.contract.read_text(encoding="utf-8")), args.project_root)
    print(json.dumps(value, indent=2))
    return 0 if not value["blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
