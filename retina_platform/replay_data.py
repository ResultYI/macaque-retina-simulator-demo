"""Read-only Cell6 cache adapter. Bundle IO only; never builds/advances a model."""
from pathlib import Path
import hashlib
import json

import numpy as np

from retina_sim.contracts import canonical_hash
from retina_sim.group_time_storage import load_bundle


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class ReplayData:
    def __init__(self, root, registry=None):
        self.root = Path(root).resolve()
        self.phase = self.root / "reports/rgc_stage1_cpu_parallel_full_v1"
        self.source = self.root / "reports/rgc_stage1_complete_v1"
        self.registry_path = Path(registry).resolve() if registry else self.phase / "CACHE_REGISTRY.json"
        self.spec_path = self.source / "CONTRACT.json"
        self.spec = read_json(self.spec_path)
        self.spec_sha = sha(self.spec_path)
        self.registry = read_json(self.registry_path)
        require(self.registry.get("schema")=="same-scientific-contract-cpu-cache-registry-1" and
                self.registry.get("short_prefixes_accepted") is False and self.registry.get("GPU_caches_accepted") is False,
                "Only the complete original CPU cache registry is supported")
        require(self.registry.get("original_spec_sha256") == self.spec_sha, "Registry / scientific contract SHA256 differs")
        require(self.spec.get("schema") == "rgc-stage1-output-contract-1", "Unsupported scientific contract")
        for name, digest in self.spec["code_pins"].items():
            require(sha(self.root / name) == digest, "Frozen source pin differs: " + name)
        require(sha(self.source / "actual_observations.npz") == self.spec["counts_sha256"], "Observation payload SHA256 differs")
        self.rows = {c["case_id"]: c for c in self.spec["cases"]}
        self.caches = {c["case_id"]: c for c in self.registry["caches"]}
        require(len(self.caches) == len(self.registry["caches"]), "Duplicate cache case IDs")
        require(set(self.caches) <= set(self.rows), "Unknown cache case ID")
        self.nodes = []
        for pop, group in self.spec["recording"].items():
            for col, gid in enumerate(group["global_ids"]):
                self.nodes.append(dict(population=pop, column=col, id=gid,
                                       members=group["members"][col], xy=group["centroid_mm"][col]))
        self.target = next(i for i, n in enumerate(self.nodes) if n["id"] == self.spec["target"]["global_id"])
        self.case_id = None
        self.fit = None
        self.fit_warning = ""
        self.prediction = None

    def load_case(self, case_id):
        require(case_id in self.caches, "Case has no complete registered cache: " + case_id)
        case, entry = self.rows[case_id], self.caches[case_id]
        require(entry.get("backend")=="original reference CPU" and entry.get("readonly") is True,"Cache backend / readonly identity differs")
        directory = Path(entry["directory"])
        if not directory.is_absolute():
            directory = self.root / directory
        manifest = read_json(directory / "manifest.json")
        identity = canonical_hash(dict(spec=self.spec, case=case))
        require(manifest.get("schema") == "rgc-stage1-fixed-endogenous-cache-1" and
                manifest.get("status") == "COMPLETE_FIXED_ENDOGENOUS_CACHE", "Incomplete or unsupported cache")
        require(identity == entry["cache_identity"] == manifest["cache_identity"], "Cache identity differs")
        require(manifest["case"] == case, "Cache case / target / stimulus / split differs")
        require(sha(directory / "endogenous.npz") == manifest["arrays_sha256"] == entry["payload_sha256"], "Cache payload SHA256 differs")
        require(manifest["units"] == dict(V="V", gE="S", gI="S") and manifest["initial_match"]["passed"], "Cache units / initial match differs")
        require(manifest["recorded"] == 71001 and manifest["bins"] == 7100, "Cache is a short prefix")
        with np.load(directory / "endogenous.npz", allow_pickle=False) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        time = arrays["time_source_s"]
        require(np.array_equal(time, np.arange(71001) * .0001), "Source time clock differs")
        for pop, group in self.spec["recording"].items():
            for field in ("V", "gE", "gI"):
                a = arrays[pop + ":" + field]
                require(a.shape == (71001, len(group["rows"])) and a.dtype==np.float64 and np.isfinite(a).all(), "Invalid cached array: " + pop + ":" + field)
        require(arrays["expected_rate_base_Hz"].shape == (7100, 4) and
                np.isfinite(arrays["expected_rate_base_Hz"]).all() and
                (arrays["expected_rate_base_Hz"] >= 0).all(), "Invalid cached baseline rate")
        with np.load(self.source / "actual_observations.npz", allow_pickle=False) as saved:
            counts = saved[case["counts_key"]].copy()
        require(counts.shape == (3, 7100) and np.isin(counts, [0, 1]).all(), "Observation repeat / bin contract differs")
        inputs = self.root / "reports/layer_calibration/rgc_cbem_conditional_v1/frozen_preparation_v3"
        require(sha(inputs / "snapshot_manifest.json") == self.spec["source_counts_manifest_sha256"], "Source observation manifest differs")
        snapshot = read_json(inputs / "snapshot_manifest.json")
        require(sha(inputs / "inputs.npz") == snapshot["input_npz_sha256"], "Source input payload differs")
        # Only the selected g0..g5 stimulus is accessed; historical test arrays stay unopened.
        with np.load(inputs / "inputs.npz", allow_pickle=False) as saved:
            raw = saved[case["counts_key"].replace("_counts", "_stimulus_values")].copy()
        require(raw.shape == (71000,) and hashlib.sha256(raw.tobytes()).hexdigest() == case["stimulus_sha256"], "Selected actual stimulus differs")
        drive = load_bundle(self.root / case["drive_bundle"], expected_kind="validation_evidence")["payload"]
        require(canonical_hash(drive) == case["drive_sha256"] and drive["stimulus_sha256"] == case["stimulus_sha256"], "Saved native input drive differs")
        self.case_id, self.case, self.cache_path = case_id, case, directory
        self.arrays, self.time, self.counts, self.raw, self.drive = arrays, time, counts, raw, drive
        self.stop = round(float(time[-1]), 10)
        self.frames = np.array(sorted(drive["native_inputs"]), dtype=int)
        self.frame_time = self.frames / self.spec["numerical"]["input_fps"] - drive["model_origin_s"]
        self.native = np.array([[np.min(drive["native_inputs"][f]), np.mean(drive["native_inputs"][f]),
                                 np.max(drive["native_inputs"][f])] for f in self.frames])
        self.load_fit()
        return self

    def load_fit(self):
        self.fit, self.prediction, self.fit_warning = None, None, ""
        self.goal_assessment=None;self.goal_assessment_sha=None
        fit_dir = self.phase / "fit"
        result_path = fit_dir / "CELL6_RESULT.json"
        if not result_path.exists():
            return
        try:
            require(self.registry.get("all_six_complete") is True and set(self.caches)==set(self.rows), "Final fit requires the complete six-case registry")
            result = read_json(result_path)
            require(result.get("status") == "STAGE1_CONDITIONAL_CELL6_OUTPUT_RESULT" and
                    type(result.get("accepted")) is bool, "Unrecognized fit result")
            receipt_path = fit_dir / "FIT_EXECUTION_RECEIPT.json"
            require(receipt_path.exists(), "Fit receipt not yet available; result identity is unverified")
            receipt = read_json(receipt_path)
            require(receipt.get("registry_sha256") == sha(self.registry_path) and
                    receipt.get("original_scientific_contract_unchanged") is True and
                    receipt.get("accepted") == result["accepted"], "Fit receipt / cache registry differs")
            require({v["case_id"] for v in result["validation"]} ==
                    {k for k,v in self.rows.items() if v["split"] == "validation"}, "Validation groups differ")
            self.fit = result
            assessment_path=self.root/'reports/rgc_stage1_aligned_v1/OUTPUT_GOAL_ASSESSMENT.json'
            if assessment_path.exists():
                try:
                    assessment=read_json(assessment_path)
                    require(assessment.get('frozen_NLL_acceptance')==result['accepted'] and
                            assessment.get('original_acceptance_unchanged') is True and
                            assessment.get('validation_not_used_for_refitting') is True,"Output-goal assessment / original acceptance differs")
                    reported={row['case_id']:row for row in result['validation']}
                    require({row['case_id'] for row in assessment['validation']}==set(reported),"Output-goal validation cases differ")
                    for row in assessment['validation']:
                        original=reported[row['case_id']];block=original['free_history_PSTH']
                        metrics=block[0] if isinstance(block,list) else block
                        require(row['baseline_NLL_per_bin']==original['baseline_NLL_per_bin'] and
                                row['candidate_NLL_per_bin']==original['candidate_NLL_per_bin'] and
                                all(value==metrics[key] for key,value in row['PSTH'].items()),"Output-goal assessment metrics differ")
                    self.goal_assessment=assessment;self.goal_assessment_sha=sha(assessment_path)
                except (OSError,ValueError,KeyError,TypeError) as exc:
                    self.fit_warning='动态目标评价未核验：'+str(exc)
            prediction_path = self.root / "reports/cell6_replay/predictions/manifest.json"
            if not prediction_path.exists():
                self.fit_warning = "拟合结果已读取；拟合后期望率尚未导出，曲线仍为固定基线。"
                return
            p = read_json(prediction_path)
            require(p.get("schema") == "cell6-replay-predictions-1" and
                    p["scientific_contract_sha256"] == self.spec_sha and
                    p["cache_registry_sha256"] == sha(self.registry_path) and
                    p["fit_result_sha256"] == sha(result_path) and
                    p["target_global_id"] == self.spec["target"]["global_id"] and
                    p.get('accepted')==result['accepted'] and p.get('bin_seconds')==.001 and
                    p["rate_kind"] == "free_history_expected_rate_Hz", "Fitted prediction provenance differs")
            payload = (prediction_path.parent / p["payload"]).resolve()
            require(payload.parent == prediction_path.parent.resolve(), "Prediction must be a local sibling file")
            require(sha(payload) == p["payload_sha256"], "Fitted prediction payload differs")
            with np.load(payload, allow_pickle=False) as saved:
                t, rate = saved["time_source_s"], saved[self.case_id + "_rate_Hz"]
                require(np.array_equal(t, np.arange(7100)*.001) and rate.shape == (7100,) and
                        np.isfinite(rate).all() and (rate >= 0).all() and (rate <= 1000).all(), "Fitted prediction clock / shape / values differ")
                self.prediction = rate.copy()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.fit_warning = "拟合资产未核验：" + str(exc)

    @property
    def fit_label(self):
        if self.fit is None:
            return "未取得已核验拟合结果 · 当前曲线为 pre-fit 固定基线"
        if self.dynamic_output_insufficient:
            return "拟合已运行，动态输出仍不足 · 原 NLL 门通过，不能视为完整动态拟合成功"
        return "拟合已运行：" + ("仅原 NLL 门通过；动态目标待核验" if self.fit["accepted"] else "原 NLL 门未通过 / 拒绝采用")

    @property
    def dynamic_output_insufficient(self):
        assessment=getattr(self,'goal_assessment',None)
        return bool(assessment and assessment.get('status')=='FROZEN_NLL_GATE_PASSED_DYNAMIC_OUTPUT_GOAL_UNRESOLVED' and
                    assessment.get('all_RGC_outputs_fitted') is False)

    def validation_metrics(self):
        rows=[]
        if self.fit is None:return rows
        for row in self.fit['validation']:
            block=row.get('free_history_PSTH',{})
            metrics=block[0] if isinstance(block,list) else block
            rows.append(dict(case_id=row['case_id'],baseline_NLL_per_bin=row.get('baseline_NLL_per_bin'),
                             candidate_NLL_per_bin=row.get('candidate_NLL_per_bin'),PSTH=metrics))
        return rows

    @property
    def evaluation_definition(self):
        return "原4训练(g0–g3)/2验证(g4/g5)，评分[1,7)s。NLL/bin：1ms条件Bernoulli负对数似然，使用严格过去真实历史，按重复×bin平均；PSTH：20ms，3次重复平均实测 vs 自由历史期望，无平滑、时移或幅度重标定。历史test未重开。"

    @property
    def evaluation_summary(self):
        if self.fit is None:return ""
        lines=[]
        for row in self.validation_metrics():
            p=row['PSTH']
            if not p:continue
            old,new=row['baseline_NLL_per_bin'],row['candidate_NLL_per_bin']
            correlation='未定义' if p.get('PSTH_Pearson') is None else f"{p['PSTH_Pearson']:.5f}"
            lines.append(f"{row['case_id'].replace('cell6_','')}：NLL {old:.6f}→{new:.6f}；PSTH r={correlation}；均值预测/实测 {p['predicted_rate_Hz']:.2f}/{p['observed_rate_Hz']:.2f} Hz；调制SD {p['predicted_modulation_std_Hz']:.5f}/{p['observed_modulation_std_Hz']:.2f} Hz")
        return "验证 "+" | ".join(lines) if lines else ""

    def stimulus_at(self, seconds, *, frame=None):
        if frame is None:
            i = int(np.clip(np.searchsorted(self.frame_time, seconds, side="right") - 1, 0, len(self.frames)-1))
        else:
            i=int(np.searchsorted(self.frames,frame))
            require(i<len(self.frames) and self.frames[i]==frame,"Aligned input frame is absent from the frozen drive")
        j = max(0, int(np.floor((self.frames[i]/150. - self.drive["model_origin_s"])/.0001)))
        j = min(j, len(self.raw)-1)
        multiplier = 1 + .05 * (self.raw[j] - self.raw[0])
        return dict(frame=int(self.frames[i]), source_index=j, raw=float(self.raw[j]),
                    multiplier=float(multiplier), native_td=self.native[i])

    def node_values(self, field, seconds):
        i = int(np.clip(np.searchsorted(self.time, seconds+1e-12, side="right")-1, 0, len(self.time)-1))
        factor = 1000 if field == "V" else 1e9
        return np.array([self.arrays[n["population"]+":"+field][i,n["column"]]*factor for n in self.nodes])

    def trace(self, node, field="V"):
        n = self.nodes[node]
        return self.arrays[n["population"]+":"+field][:,n["column"]] * (1000 if field == "V" else 1e9)

    def evidence(self):
        return dict(case=self.case, cache_directory=str(self.cache_path), cache=self.caches[self.case_id],
                    source_contract_sha256=self.spec_sha, target=self.spec["target"],
                    input_mapping=self.spec["input_mapping"], source_condition=self.spec["source_condition"],
                    fit_status=self.fit_label, fit_warning=self.fit_warning, fit=self.fit,
                    output_goal_assessment=getattr(self,'goal_assessment',None),
                    output_goal_assessment_sha256=getattr(self,'goal_assessment_sha',None),evaluation_definition=self.evaluation_definition,
                    limits=["RGC group/pool centroids in model coordinates; not biological Cell6 registration",
                            "Only RGC V/gE/gI and ON-parasol baseline expectation are continuous cache signals",
                            "No continuous cone/HC/BC/AC trajectories or connectivity are inferred",
                            "Source latency / sample origin and physical photometry remain unverified",
                            "Only this conditional Cell6 branch is eligible; other types are not fitted"])
