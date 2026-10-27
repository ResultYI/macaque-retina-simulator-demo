"""Optional post-fit visualization export. Never imports/builds the forward model.

Run only by the authorized fitting owner after all six caches and the final result
exist. Reuses the original cached-voltage output readout, with no optimization.
"""
import argparse
from pathlib import Path
import json
import os

import numpy as np

from .replay_data import ReplayData, require, sha


def probability_to_rate(probability):
    q=np.asarray(probability,dtype=float)
    require(np.isfinite(q).all() and (q>=0).all() and (q<=1).all(),"Original readout must return bin probabilities")
    return q/.001


def export(project_root):
    data = ReplayData(project_root)
    require(data.registry.get("all_six_complete") and set(data.caches) == set(data.rows), "Six complete caches required")
    data.load_case(sorted(data.caches)[0])
    require(data.fit is not None, "Verified final fit and execution receipt required: " + data.fit_warning)
    # This is the existing output-only readout, not a new implementation or a fit.
    from retina_sim.rgc_stage1_readout import decode, free_mean
    parameters = decode(np.asarray(data.fit["training"]["theta"]), data.spec["base"])
    result_path = data.phase / "fit/CELL6_RESULT.json"
    result_sha, registry_sha = sha(result_path), sha(data.registry_path)
    arrays = {"time_source_s": np.arange(7100)*.001}
    for case in sorted(data.caches):
        data.load_case(case)
        require(data.fit is not None and sha(result_path) == result_sha, "Fit changed during export")
        voltage = data.arrays[data.case["population"]+":V"][::10,data.case["model_column"]][:-1]
        probability=free_mean(voltage, parameters, data.spec["initial_probability"])
        rate=probability_to_rate(probability)
        require(rate.shape == (7100,) and np.isfinite(rate).all() and (rate >= 0).all(), "Invalid original readout export")
        arrays[case+"_rate_Hz"] = rate
    require(sha(data.registry_path) == registry_sha and sha(result_path) == result_sha, "Registry or fit changed during export")
    output = data.root / "reports/cell6_replay/predictions"
    output.mkdir(parents=True, exist_ok=True)
    payload = output / "predictions.npz"
    temporary = output / "predictions.npz.writing"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, payload)
    manifest = dict(schema="cell6-replay-predictions-1", scientific_contract_sha256=data.spec_sha,
                    cache_registry_sha256=registry_sha, fit_result_sha256=result_sha,
                    target_global_id=data.spec["target"]["global_id"], rate_kind="free_history_expected_rate_Hz",
                    payload=payload.name, payload_sha256=sha(payload),
                    accepted=data.fit["accepted"], source="unchanged rgc_stage1_readout.decode/free_mean on original cached V; probability / 0.001 s",
                    bin_seconds=.001,
                    upstream_advanced=False, optimizer_called=False, source_assets_modified=False,
                    fitted_scope="Conditional Cell6 ON-parasol output only; no anatomical or physical qualification")
    temporary = output / "manifest.json.writing"
    temporary.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    os.replace(temporary,output / "manifest.json")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(export(parser.parse_args().project_root)))
