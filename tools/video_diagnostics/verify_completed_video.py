"""Read-only completion checks for the actual authorized video output."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from retina_sim.contracts import canonical_hash


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_checkpoint(path: Path, expected_time: float, manifest: dict) -> dict:
    document = read(path)
    require(document["version"] == "retina-architecture-1.0", "Checkpoint version differs")
    payload = document["payload"]
    require(document["payload_hash"] == canonical_hash(payload), "Checkpoint payload hash differs")
    dynamic = payload["dynamic"]
    require(dynamic["payload_hash"] == canonical_hash(dynamic["state"]), "Dynamic hash differs")
    state = dynamic["state"]
    require(abs(state["time_seconds"] - expected_time) <= 1e-12, "Checkpoint time differs")
    require(payload["assembly_hash"] == manifest["assembly_hash"], "Assembly differs")
    require(state["model_hash"] == manifest["runtime"]["model_hash"], "Model differs")
    count = 0
    for row in state["ordered_state_arrays"]:
        array = np.asarray(row)
        require(np.all(np.isfinite(array)), "Nonfinite checkpoint state")
        count += array.size
    require(count == 4967023, "Actual v7 full-state coverage differs")
    result = {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size,
              "payload_hash": document["payload_hash"], "dynamic_hash": dynamic["payload_hash"],
              "state_scalars": count, "state_blocks": len(state["ordered_state_arrays"]),
              "absolute_time_seconds": state["time_seconds"], "sample_index": payload["sample_index"],
              "recorded_window_count": payload["input_position"].get("recorded_window_count"),
              "nested_hashes_and_finite_states_passed": True}
    del document, payload, dynamic, state
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run, report = args.run.resolve(), args.report.resolve()
    require(run.is_relative_to(ROOT) and report.is_relative_to(ROOT), "Project paths required")
    require(not report.exists(), "Verification report must be new")
    started = time.perf_counter()
    trial, manifest = read(run / "trial.json"), read(run / "manifest.json")
    require(trial["status"] == "COMPLETED", "Actual video is not completed")
    require(abs(trial["stimulus_seconds"] - .5) <= 1e-12, "Video duration differs")
    require(trial["windows"] == 250, "Window count differs")
    require(trial["identity"] == manifest["identity"], "Trial/manifest identity differs")
    require(manifest["recording"]["samples"] == 501, "Declared recording samples differ")
    require(read(run / "execution_backend_check.json")["status"] ==
            "FULL_REGISTERED_STATE_CUDA_CPU_100US_COMPARISON_PASS", "Backend qualification failed")
    paths = sorted((run / "windows").glob("window_*.npz"))
    require(len(paths) == 250, "Actual window file count differs")
    stamps, total_bytes, state_keys = [], 0, None
    origin = manifest["runtime"]["stimulus_origin_seconds"]
    for index, path in enumerate(paths):
        require(path.name == f"window_{index:08d}.npz", "Window sequence incomplete")
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            require("time" in keys and "manifest_json" in keys, "Window fields incomplete")
            selected = keys - {"time", "manifest_json"}
            require(all(key.startswith("state:") for key in selected), "Recording scope differs")
            if state_keys is None:
                state_keys = selected
            require(selected == state_keys, "Window state layout changed")
            t = archive["time"]
            require(t.shape == ((3,) if index == 0 else (2,)), "Window sample shape differs")
            require(np.all(np.isfinite(t)), "Nonfinite timestamps")
            rows = 0
            for key in selected:
                value = archive[key]
                require(value.ndim == 2 and value.shape[1] == len(t), "State/time shape differs")
                require(value.dtype == np.float64 and np.all(np.isfinite(value)), "Invalid recorded state")
                rows += value.shape[0]
            require(rows == manifest["recording"]["scalar_count"], "Cohort scalar coverage differs")
            metadata = json.loads(str(archive["manifest_json"]))
            require(set(metadata["ports"]) == selected | {"time"}, "Units/metadata coverage differs")
            stamps.extend(t.tolist())
        total_bytes += path.stat().st_size
        if index % 25 == 0:
            print(f"WINDOWS_VERIFIED {index + 1}/250", flush=True)
    t = np.asarray(stamps)
    require(len(t) == 501 and np.all(np.diff(t) > 0), "Timestamp coverage incomplete")
    require(np.allclose(t, origin + np.arange(501) * .001, rtol=0, atol=1e-12), "1ms grid differs")
    start = check_checkpoint(run / "start_checkpoint.json", origin, manifest)
    end = check_checkpoint(run / "end_checkpoint.json", origin + .5, manifest)
    latest_sha = digest(run / "latest_checkpoint.json")
    require(latest_sha == end["sha256"], "Latest/end checkpoint bytes differ")
    require(end["recorded_window_count"] == 250, "End checkpoint recording commit differs")
    result = {"status": "ACTUAL_VIDEO_05S_OUTPUT_VERIFIED", "run": str(run),
              "duration_seconds": .5, "recorded_samples": len(t), "windows": len(paths),
              "state_keys": len(state_keys), "recording": manifest["recording"],
              "record_files_bytes": total_bytes, "start_checkpoint": start, "end_checkpoint": end,
              "latest_checkpoint_byte_identical_to_end": True,
              "phase_seconds": trial["phase_seconds"], "wall_seconds": trial["wall_seconds"],
              "integration_metrics": trial["integration_metrics"],
              "memory_statistics": trial["memory_statistics"],
              "identity": manifest["identity"], "background_initialization": manifest["background_initialization"],
              "scope": "Real original 0.5s clip; transient NOT_REACHED background; observation cohort, full-state checkpoints",
              "verification_wall_seconds": time.perf_counter() - started}
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "duration_seconds", "windows", "recorded_samples")}), flush=True)


if __name__ == "__main__":
    main()
