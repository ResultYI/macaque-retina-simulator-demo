"""Explicit opt-in binding of the frozen, inconclusive Cell6 output candidate.

No training, retinal state updates, observed spike inputs, or default selection.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from retina_sim.contracts import canonical_hash
from retina_sim.temporal_readout_v1 import TemporalReadout, digest, load_model

PRODUCTION = "原生产 / 既有读出（默认）"
EXPERIMENTAL = "实验 temporal_readout_v1（INCONCLUSIVE）"
MODEL_ID = "1fb6f5909d24d92f3c742ff7d60808d59f1cd2eeb041b56de4cae06204d65845"
TARGET = "sg2:61468d3761f46cda:parasol_RGC_ON:8166"
PHASE = "source_cache_before_discrete_commit"
MODEL_PATH = "reports/cell6_temporal_readout_v1_20261028/MODEL.json"
NOTICE = ("INCONCLUSIVE · Cell6 开发验证 · 非生理标定；τ为有效编码尺度。"
          "自由期望率 p/1ms，未使用真实 spike history；原生产默认不变。")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding_for_replay(data, node):
    """Use the source identities already verified by the existing replay loader."""
    spec = data.spec
    pop = spec["recording"][node["population"]]
    column = node["column"]
    if (node["id"] != TARGET or node["population"] != "parasol_RGC_ON"
            or column != 1 or node["members"] != 2
            or pop["global_ids"][column] != TARGET):
        raise ValueError("实验输出不适用：仅限 Cell6 条件性 ON-parasol group8166/column1/两成员")
    return dict(case_id=data.case_id, cache_identity=data.caches[data.case_id]["cache_identity"],
        payload_sha256=data.caches[data.case_id]["payload_sha256"],
        stimulus_sha256=data.case["stimulus_sha256"], source_contract_sha256=data.spec_sha,
        population=node["population"], global_id=node["id"], column=column,
        members=node["members"], representation="coarse_group_state", unit="V",
        native_dt_s=.0001, bin_dt_s=.001, phase=PHASE)


class ConditionalTemporalOutput:
    """Online voltage observer for exactly the approved frozen case binding.

    A push consumes past-to-present 100 us voltage nodes. No true spike/history
    argument exists. Restored states cover this readout only, not a retina.
    """
    def __init__(self, root, binding, *, mode, state=None, output_kind="free_expectation"):
        if mode != EXPERIMENTAL:
            raise ValueError("Experimental output requires explicit opt-in mode")
        if output_kind != "free_expectation":
            raise ValueError("Only frozen free-history expectation is implemented; sampled spikes unavailable")
        self.root = Path(root).resolve()
        self.binding = dict(binding)
        spec_path = self.root / "reports/rgc_stage1_complete_v1/CONTRACT.json"
        spec = read(spec_path)
        source = read(self.root / "reports/cell6_temporal_readout_v1_20261028/CONTRACT.json")
        registry_path = self.root / "reports/rgc_stage1_cpu_parallel_full_v1/CACHE_REGISTRY.json"
        if (sha(spec_path) != source["pins"][str(spec_path.relative_to(self.root))]
                or sha(registry_path) != source["pins"][str(registry_path.relative_to(self.root))]):
            raise ValueError("Frozen source/registry identity differs")
        cases = {c["case_id"]: c for c in spec["cases"][:6]}
        if binding.get("case_id") not in cases:
            raise ValueError("Experimental output limited to the six original Cell6 development cases")
        case = cases[binding["case_id"]]
        entry = next(x for x in read(registry_path)["caches"] if x["case_id"] == case["case_id"])
        expected = dict(case_id=case["case_id"], cache_identity=canonical_hash(dict(spec=spec, case=case)),
            payload_sha256=entry["payload_sha256"],
            stimulus_sha256=case["stimulus_sha256"], source_contract_sha256=sha(spec_path),
            population="parasol_RGC_ON", global_id=TARGET, column=1, members=2,
            representation="coarse_group_state", unit="V",
            native_dt_s=.0001, bin_dt_s=.001, phase=PHASE)
        if self.binding != expected:
            raise ValueError("Experimental output target/model/input/clock/phase binding does not match")
        # This explicit mode is authorized for the retained INCONCLUSIVE candidate.
        # Preserve its qualification; do not change the frozen loader's default guard.
        self.model = load_model(self.root / MODEL_PATH, allow_unqualified=True)
        if self.model["model_id"] != MODEL_ID or self.model["qualification"] != "INCONCLUSIVE":
            raise ValueError("This opt-in version binds only the frozen INCONCLUSIVE candidate")
        from retina_sim.parameter_runtime import cell6_scope,cell6_request
        self.parameter_scope=cell6_scope(self.root)
        self.parameter_scope_request=cell6_request(self.root)
        self.parameter_scope.check(self.parameter_scope_request,explicit_opt_in=True)
        self.binding_id = digest(expected)
        self.adapter_sha256 = sha(Path(__file__))
        saved = None
        if state is not None:
            if (state.get("schema") != "conditional-temporal-output-state-1"
                    or state.get("binding_id") != self.binding_id
                    or state.get("adapter_sha256") != self.adapter_sha256):
                raise ValueError("Online output restore binding/adapter identity differs")
            saved = state["readout"]
        self.readout = TemporalReadout(self.model, saved)
        if self.readout.native_index > 71000:
            raise ValueError("Restore cursor exceeds the frozen source case")

    def push(self, voltage_V, time_source_s, *, phase):
        v = np.asarray(voltage_V)
        t = np.asarray(time_source_s)
        start = self.readout.native_index
        if (phase != PHASE or v.dtype != np.float64 or t.dtype != np.float64
                or v.ndim != 1 or t.shape != v.shape
                or start + len(v) > 71000
                or not np.array_equal(t, np.arange(start, start + len(v)) * .0001)):
            raise ValueError("Native source nodes/phase/cursor differ; terminal 7.1s has no following bin")
        if start == 0 and len(v) and v[0] != self.model["core"]["initial_gray_V"]:
            raise ValueError("Source gray initial voltage differs; do not re-estimate initial state")
        result = self.readout.predict(v)
        result["time_source_s"] = result["bin_index"] * .001
        return result

    def state(self):
        return dict(schema="conditional-temporal-output-state-1", binding_id=self.binding_id,
                    adapter_sha256=self.adapter_sha256, readout=self.readout.state())

    def metadata(self):
        return dict(mode=EXPERIMENTAL, model_id=MODEL_ID, qualification="INCONCLUSIVE",
            binding=self.binding, binding_id=self.binding_id,
            parameter_system_scope=self.parameter_scope_request,
            parameter_system_status="conditional / INCONCLUSIVE, never validated by migration",
            tau_s=self.model["core"]["tau_s"], statistics="frozen train g0-g2, not recalculated",
            output_kind="free_history_expected_binary_count_rate_Hz", sampled_spikes=False,
            phase=PHASE, bin_semantics="1ms bin start; a precedes current100us input; three strictly past spike bins marginalized in8 states",
            initial="shared frozen source-gray a0/pi; first1s original recorded context, not re-estimated per case",
            representation="one group state/two model members, not two independent cell states",
            notice=NOTICE, physical_qualification=False, FIT_READY=False,
            absolute_monitor_passed=False, production_default_changed=False)


def replay_prediction(data, node):
    """Read voltage only, adding a separate output curve; never modifies data."""
    adapter = ConditionalTemporalOutput(data.root, binding_for_replay(data, node), mode=EXPERIMENTAL)
    v = data.arrays[node["population"] + ":V"][:-1, node["column"]]
    pieces = []
    for start in range(0, len(v), 1000):
        stop = min(start + 1000, len(v))
        pieces.append(adapter.push(v[start:stop], data.time[start:stop], phase=PHASE)["expected_count_rate_Hz"])
    return np.concatenate(pieces), adapter.metadata()
