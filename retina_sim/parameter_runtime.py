"""Versioned execution/checkpoint and Cell6 conditional binding adapters."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib

from .contracts import ContractError, canonical_hash
from .parameter_system import SCHEMA, CalibrationRecord, identity, data, frozen
from .parameter_migration import compatible_inner_parameters


class ParameterSystemGroupRunner:
    """Actual unchanged GroupTimeArm consumes the resolved compatibility view."""
    def __init__(self, assembly, seed, protocol, bundle, **kwargs):
        from .group_time_runner import GroupTimeArm
        self.bundle = bundle; self.assembly = assembly
        self.seed = deepcopy(seed); self.protocol = dict(protocol)
        self.seed["inner_parameters"] = compatible_inner_parameters(bundle)
        if canonical_hash(self.seed["inner_parameters"]) != canonical_hash(seed["inner_parameters"]):
            raise ContractError("v1 migration requires zero-change current dynamics/initialization")
        self.protocol["parameter_binding_v1"] = dict(schema=SCHEMA, binding_id=bundle.identity,
            representation="coarse_group", route_id=bundle.routes.identity)
        self.arm = GroupTimeArm(assembly, self.seed, self.protocol, **kwargs)
        self.execution_schema = SCHEMA

    def __getattr__(self, name):
        return getattr(self.arm, name)

    def checkpoint_payload(self, **kwargs):
        raw = self.arm.checkpoint_payload(**kwargs)
        raw["controller"]["parameter_binding"] = dict(schema=SCHEMA, binding_id=self.bundle.identity,
            route_id=self.bundle.routes.identity, execution_schema=self.execution_schema,
            scientific_qualification="unchanged")
        return raw

    @classmethod
    def from_checkpoint(cls, assembly, seed, protocol, bundle, payload, *, allow_legacy=False, **kwargs):
        from .group_time_runner import GroupTimeArm
        raw = deepcopy(payload); binding = raw["controller"].pop("parameter_binding", None)
        if binding is None and not allow_legacy:
            raise ContractError("Legacy checkpoint requires explicit legacy-compatible restore")
        if binding is not None and (binding.get("schema") != SCHEMA
                or binding.get("binding_id") != bundle.identity
                or binding.get("route_id") != bundle.routes.identity):
            raise ContractError("Checkpoint static parameter/graph binding mismatch")
        result = object.__new__(cls)
        result.bundle, result.assembly = bundle, assembly
        result.seed = deepcopy(seed)
        result.seed["inner_parameters"] = compatible_inner_parameters(bundle)
        result.protocol = dict(protocol)
        execution = binding.get("execution_schema") if binding else "legacy"
        if execution == SCHEMA:
            result.protocol["parameter_binding_v1"] = dict(schema=SCHEMA,binding_id=bundle.identity,
                representation="coarse_group",route_id=bundle.routes.identity)
        elif not allow_legacy:
            raise ContractError("Legacy execution semantics require explicit opt-in")
        # The original restore still verifies its entire controller identity,
        # clocks/history/RNG/frontiers. No old source hash is waived.
        result.arm = GroupTimeArm.from_checkpoint(assembly,result.seed,result.protocol,raw,**kwargs)
        result.execution_schema = execution
        return result


CELL6_MODEL_ID = "1fb6f5909d24d92f3c742ff7d60808d59f1cd2eeb041b56de4cae06204d65845"
CELL6_GROUP_ID = "sg2:61468d3761f46cda:parasol_RGC_ON:8166"
CELL6_MODEL_PATH = "reports/cell6_temporal_readout_v1_20261028/MODEL.json"
CELL6_SPEC_PATH = "reports/rgc_stage1_complete_v1/CONTRACT.json"


def _read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ConditionalOverrideScope:
    exact: object
    authorized_input_contracts: tuple
    biological_cell_id: str = "Latimer2019:Cell6"
    status: str = "conditional"
    performance_qualification: str = "INCONCLUSIVE"

    def __post_init__(self):
        object.__setattr__(self,"exact",frozen(self.exact))

    def check(self, request, *, explicit_opt_in=False):
        if not explicit_opt_in: raise ContractError("Conditional override is opt-in only")
        for key, value in self.exact.items():
            if data(request.get(key)) != data(value):
                raise ContractError("Cell6 conditional scope mismatch: "+key)
        contract = request.get("source_input_contract_id")
        if contract not in self.authorized_input_contracts:
            raise ContractError("New input contract requires explicit scope authorization")
        if request.get("biological_cell_id") != self.biological_cell_id:
            raise ContractError("Cell6 biological recording identity mismatch")
        return dict(status=self.status,performance_qualification=self.performance_qualification,
                    physical_qualification=False,FIT_READY=False)


def cell6_scope(root):
    """Existing frozen identity guards plus explicit input-contract authorization.

    An authorized input contract is not the hash of just one training waveform.
    The separate completed cone-step protocol is listed explicitly; arbitrary
    videos/new backgrounds remain unauthorized.
    """
    root=Path(root);spec=_read(root/CELL6_SPEC_PATH);model=_read(root/CELL6_MODEL_PATH)
    if model["model_id"]!=CELL6_MODEL_ID or model["qualification"]!="INCONCLUSIVE":
        raise ContractError("Only current frozen inconclusive Cell6 candidate is registered")
    manifest=_read(root/spec["prepared_model"]/"initial_seed/manifest.json")
    artifact=manifest["identity"]["artifact"]
    training_id=identity(dict(groups=[0,1,2],counts_sha256=spec["counts_sha256"],
                              statistics=model["core"]["mean_train_V"],
                              SD=model["core"]["SD_train_V"]))
    evaluation_id=identity(dict(groups=[3,4,5],development_touched=True,
                                result_sha256=_sha(root/"reports/cell6_temporal_readout_v1_20261028/RESULT.json")))
    exact=dict(model_target_group_id=CELL6_GROUP_ID,population="parasol_RGC_ON",column=1,
        member_count=2,representation="coarse_group",
        background_policy_id="frozen_source_seed:"+spec["source_seed_sha256"],
        readout_model_id=model["model_id"],readout_artifact_sha256=_sha(root/CELL6_MODEL_PATH),
        source_artifact_sha256=artifact,source_code_identity=identity(spec["code_pins"]),
        training_identity=training_id,evaluation_identity=evaluation_id,
        qualification="INCONCLUSIVE")
    allowed=["cell6_original_engineering_input:"+identity(spec["input_mapping"])]
    controlled=root/"reports/cell6_controlled_function_v1_20261028/attempt_0002/CONTRACT.json"
    if controlled.exists():
        p=_read(controlled)
        if p["model_id"]!=CELL6_MODEL_ID or p["source_seed_sha256"]!=spec["source_seed_sha256"]:
            raise ContractError("Controlled protocol is not the same authorized source")
        allowed.append("authorized_cone_steps627ms:"+_sha(controlled))
    return ConditionalOverrideScope(exact,tuple(allowed))


def cell6_request(root, *, source_input_contract_id=None):
    scope=cell6_scope(root)
    request=dict(data(scope.exact),biological_cell_id="Latimer2019:Cell6",
                 source_input_contract_id=source_input_contract_id or scope.authorized_input_contracts[0])
    return request


def cell6_calibration_record(root):
    root=Path(root);scope=cell6_scope(root);model=_read(root/CELL6_MODEL_PATH)
    fit=_read(root/"reports/cell6_temporal_readout_v1_20261028/FIT.json")
    contract=_read(root/"reports/cell6_temporal_readout_v1_20261028/CONTRACT.json")
    return CalibrationRecord("cell6_temporal_readout_v1",("b","w0","w1","tau_s","eta"),
        dict(training=[0,1,2],internal_development=[3],development_validation=[4,5],
            historical_test="sealed",contract_sha256=model["core"]["contract_sha256"]),
        data(scope.exact),dict(optimizer=contract["optimizer"],bounds=contract["bounds"],
                              regularization=contract["regularization"]),
        {k:model["core"][k] for k in ("b","w0","w1","tau_s","eta","mean_train_V","SD_train_V")},
        dict(source="frozen paired descriptive intervals",physiological_identifiability=False),
        ("frozen Cell6 conditional functional fit; no physiological parameter adoption",),
        dict(converged=fit["converged"],status="INCONCLUSIVE",result_sha256=_sha(
            root/"reports/cell6_temporal_readout_v1_20261028/RESULT.json")),
        "conditional",("g3/g4/g5 previously development touched","g5 MSE interval crossed zero",
                       "no physical current/voltage-scale calibration","no cross-type/eccentricity adoption"),
        "User-authorized explicit experimental Cell6 output; original production remains default")
