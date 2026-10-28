"""Versioned static parameter contracts. No dynamics, fitting, or unsafe eval."""
from dataclasses import dataclass, fields, is_dataclass, field
from collections.abc import Mapping
from types import MappingProxyType
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy import sparse
from .contracts import ContractError

SCHEMA = "retina-parameter-system-1.0"
RULES = frozenset(("shared_constant", "type_constant", "eccentricity_function",
                  "retinotopic_function", "background_rule", "derived", "conditional_override"))
STATUSES = frozenset(("proposed", "diagnostic", "conditional", "validated_for_scope",
                      "engineering_default", "unsupported", "not_implemented"))


def data(value):
    if is_dataclass(value):
        return {f.name: data(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): data(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [data(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def identity(value):
    return hashlib.sha256(json.dumps(data(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf8")).hexdigest()


class FrozenList(tuple):
    """Immutable list preserving its legacy serialization type for adapters."""


def frozen(value):
    if isinstance(value, np.ndarray):
        a = value.copy(); a.flags.writeable = False; return a
    if isinstance(value, Mapping):
        return MappingProxyType({k: frozen(v) for k, v in value.items()})
    if isinstance(value, list):
        return FrozenList(frozen(v) for v in value)
    if isinstance(value, tuple):
        return tuple(frozen(v) for v in value)
    return value


def fp64(value):
    a = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(a).all():
        raise ContractError("Nonfinite resolved parameter")
    a.flags.writeable = False
    return a


def unit_dimension(unit):
    # No hidden numeric unit conversion. Opaque native units stay distinct.
    return {"dimensionless": {}, "1": {}, "s": {"t": 1}, "Hz": {"t": -1},
            "s^-1": {"t": -1}, "V": {"v": 1}, "A": {"a": 1},
            "S": {"a": 1, "v": -1}, "F": {"a": 1, "t": 1, "v": -1},
            "m": {"length": 1}}.get(unit, {unit: 1})


@dataclass(frozen=True)
class RunContext:
    species: str
    eye: str
    optics: Mapping
    background: Mapping
    units: Mapping
    input_contract: Mapping
    clocks: Mapping
    roi: Mapping
    seed: int | None
    engineering_preview: bool = False
    conditional: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not self.species or not self.eye or self.seed is not None and type(self.seed) is not int:
            raise ContractError("Explicit species/eye (unknown allowed) and RNG seed required")
        for name in ("optics", "background", "units", "input_contract", "clocks", "roi", "conditional"):
            object.__setattr__(self, name, frozen(getattr(self, name)))
        if not self.units or not self.input_contract or not self.clocks or not self.roi:
            raise ContractError("Run units/input/clocks/ROI must be explicit")


@dataclass(frozen=True)
class PopulationIndex:
    population_id: str
    ids: tuple
    cell_types: tuple
    member_count: np.ndarray
    representation: str
    xy: np.ndarray | None
    xy_unit: str
    eccentricity_deg: np.ndarray | None
    azimuth_rad: np.ndarray | None
    cone_owner: tuple
    compartment: tuple
    source_identity: str
    coordinate_frame: str = "source_declared_unknown"

    def __post_init__(self):
        n = len(self.ids)
        if (not n or len(set(self.ids)) != n or len(self.cell_types) != n
                or self.representation not in ("per_cell", "coarse_group")
                or len(self.cone_owner) != n or len(self.compartment) != n):
            raise ContractError("Population ID/type/owner/compartment layout mismatch")
        counts = fp64(self.member_count)
        if counts.shape != (n,) or np.any(counts < 1) or np.any(counts != np.floor(counts)):
            raise ContractError("Explicit integer member counts required")
        if self.representation == "per_cell" and np.any(counts != 1):
            raise ContractError("Per-cell index cannot conceal grouped members")
        object.__setattr__(self, "member_count", counts)
        for name, shape in (("xy", (n, 2)), ("eccentricity_deg", (n,)), ("azimuth_rad", (n,))):
            raw = getattr(self, name)
            if raw is not None:
                a = fp64(raw)
                if a.shape != shape: raise ContractError("Population geometry shape: " + name)
                object.__setattr__(self, name, a)
        if self.xy is not None and self.xy_unit not in ("mm", "m", "deg"):
            raise ContractError("Explicit geometry unit required")
        if self.eccentricity_deg is not None and np.any(self.eccentricity_deg < 0):
            raise ContractError("Negative eccentricity")


@dataclass(frozen=True)
class Evidence:
    source_class: str
    source_id: str
    locator: str
    source_sha256: str | None = None
    evidence_validated_eccentricity_deg: tuple | None = None
    limitations: tuple = ()

    def __post_init__(self):
        if not self.source_class or not self.source_id or not self.locator:
            raise ContractError("Concrete evidence class/source/locator required")
        object.__setattr__(self,"limitations",tuple(self.limitations))
        if self.evidence_validated_eccentricity_deg is not None:
            object.__setattr__(self,"evidence_validated_eccentricity_deg",
                               tuple(self.evidence_validated_eccentricity_deg))


@dataclass(frozen=True)
class Applicability:
    runtime_eccentricity_deg: tuple | None = None
    evidence_validated_eccentricity_deg: tuple | None = None
    species: tuple = ()
    background_policy_id: str | None = None
    conditional: Mapping = field(default_factory=dict)
    extrapolation_policy: str = "error"

    def __post_init__(self):
        if self.extrapolation_policy != "error": raise ContractError("Extrapolation must fail closed")
        for v in (self.runtime_eccentricity_deg, self.evidence_validated_eccentricity_deg):
            if v is not None and (len(v) != 2 or v[0] > v[1] or not np.isfinite(v).all()):
                raise ContractError("Invalid eccentricity interval")
        object.__setattr__(self, "conditional", frozen(self.conditional))
        object.__setattr__(self,"species",tuple(self.species))
        for k in ("runtime_eccentricity_deg","evidence_validated_eccentricity_deg"):
            if getattr(self,k) is not None:object.__setattr__(self,k,tuple(getattr(self,k)))

    def check(self, context, population):
        if self.species and context.species not in self.species:
            raise ContractError("Parameter species mismatch")
        if self.background_policy_id and context.background.get("policy_id") != self.background_policy_id:
            raise ContractError("Parameter background mismatch")
        if self.runtime_eccentricity_deg is not None:
            e = population.eccentricity_deg
            lo, hi = self.runtime_eccentricity_deg
            if e is None or np.any((e < lo) | (e > hi)):
                raise ContractError("Outside runtime eccentricity domain")
        for k, v in self.conditional.items():
            if data(context.conditional.get(k)) != data(v):
                raise ContractError("Conditional parameter identity mismatch: " + k)


@dataclass(frozen=True)
class AggregationRule:
    kind: str = "compatible"
    subject: str = "per_member"
    assumptions: tuple = ()

    def __post_init__(self):
        object.__setattr__(self,"assumptions",tuple(self.assumptions))
        if self.kind not in ("compatible", "extensive", "leak_weighted_reversal",
                             "already_aggregated_contact", "gap_laplacian", "none"):
            raise ContractError("Unknown aggregation")

    def aggregate(self, values, offsets, *, cell_types=None, backgrounds=None, leak=None):
        x = fp64(values); offsets = np.asarray(offsets, int)
        if cell_types is None or backgrounds is None or len(cell_types)!=len(x) or len(backgrounds)!=len(x):
            raise ContractError("Aggregation requires explicit member type/background identities")
        if (offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0
                or offsets[-1] != len(x) or np.any(np.diff(offsets) <= 0)):
            raise ContractError("Explicit group member offsets required")
        if self.kind in ("already_aggregated_contact", "gap_laplacian", "none"):
            raise ContractError("Already aggregated/route values must not be member-aggregated")
        result = []
        for lo, hi in zip(offsets[:-1], offsets[1:]):
            for labels in (cell_types, backgrounds):
                if labels is not None and len(set(labels[lo:hi])) != 1:
                    raise ContractError("Mixed type/background aggregation forbidden")
            part = x[lo:hi]
            equal = np.array_equal(part, np.broadcast_to(part[0], part.shape))
            if self.kind == "extensive":
                # Preserve exact historical homogeneous N*value operation.
                value = (hi-lo)*part[0] if equal else np.sum(part, axis=0)
            elif self.kind == "leak_weighted_reversal":
                if "equipotential" not in self.assumptions or leak is None:
                    raise ContractError("Heterogeneous reversal needs gL weights/equipotential assumption")
                g = np.asarray(leak)[lo:hi]
                if np.any(g <= 0): raise ContractError("Positive leak weights required")
                value = part[0] if equal else np.sum(g*part, axis=0)/np.sum(g, axis=0)
            else:
                if not equal:
                    raise ContractError("Nonlinear/kernel/history heterogeneity requires explicit effective reduction")
                value = part[0]
            result.append(value)
        return fp64(result)


@dataclass(frozen=True)
class ParameterRecord:
    parameter_id: str
    mechanism: str
    cell_types: tuple
    quantity: str
    value_spec: Mapping
    unit: str
    applicability: Applicability
    evidence: Evidence
    binding_subject: str
    aggregation: AggregationRule = field(default_factory=AggregationRule)
    calibration_permissions: tuple = ()
    domain: Mapping = field(default_factory=dict)
    status: str = "diagnostic"
    schema_version: str = SCHEMA

    def __post_init__(self):
        if (not self.parameter_id or not self.mechanism or not self.quantity or not self.unit
                or not self.binding_subject or self.status not in STATUSES
                or self.value_spec.get("rule") not in RULES or self.schema_version != SCHEMA):
            raise ContractError("Incomplete versioned parameter record/unit/rule")
        if self.value_spec["rule"] == "conditional_override" and not self.applicability.conditional:
            raise ContractError("Conditional override requires exact scoped identities")
        object.__setattr__(self, "value_spec", frozen(self.value_spec))
        object.__setattr__(self, "domain", frozen(self.domain))
        object.__setattr__(self,"cell_types",tuple(self.cell_types))
        object.__setattr__(self,"calibration_permissions",tuple(self.calibration_permissions))


class IntrinsicParameterRegistry:
    def __init__(self, records=()):
        self._records = {}; self._sealed = False
        for r in records: self.register(r)

    def register(self, record):
        if self._sealed: raise ContractError("Resolved parameter registry is immutable")
        if record.parameter_id in self._records:
            raise ContractError("Duplicate parameter ID")
        self._records[record.parameter_id] = record

    @property
    def records(self): return tuple(self._records.values())

    def freeze(self): self._sealed = True

    @property
    def identity(self): return identity(self.records)

    def theta_schema(self):
        result = {}
        for r in self.records:
            result.setdefault(r.mechanism, {}).setdefault(r.binding_subject, []).append(dict(
                parameter_id=r.parameter_id, cell_types=r.cell_types, quantity=r.quantity, unit=r.unit,
                shape=np.shape(r.value_spec.get("value")), status=r.status,
                calibration_permissions=r.calibration_permissions))
        return result

    def write_json(self, path):
        payload = dict(schema_version=SCHEMA, registry_id=self.identity, records=data(self.records))
        with Path(path).open("x",encoding="utf8") as stream:
            json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False)

    @classmethod
    def read_json(cls, path):
        payload=json.loads(Path(path).read_text(encoding="utf8"))
        if payload.get("schema_version")!=SCHEMA:raise ContractError("Registry version mismatch")
        result=cls()
        for raw in payload["records"]:
            row=dict(raw)
            try:
                row["cell_types"]=tuple(row["cell_types"])
                row["calibration_permissions"]=tuple(row.get("calibration_permissions",()))
                row["applicability"]=Applicability(**row["applicability"])
                row["evidence"]=Evidence(**row["evidence"])
                row["aggregation"]=AggregationRule(**row["aggregation"])
                result.register(ParameterRecord(**row))
            except (TypeError,KeyError) as exc:raise ContractError("Malformed versioned record") from exc
        if result.identity!=payload["registry_id"]:raise ContractError("Registry content identity mismatch")
        return result


class ParameterResolver:
    """Pure ordered resolution; no eval, interpolation extrapolation, or hidden fallback."""
    PRIORITY = {"shared_constant": 0, "type_constant": 1, "eccentricity_function": 2,
                "retinotopic_function": 2, "background_rule": 2, "derived": 3, "conditional_override": 4}

    def __init__(self, registry, context, populations):
        self.registry, self.context, self.populations = registry, context, populations

    def resolve(self, subject, quantity, expected_unit, *, mechanism=None, stack=()):
        key = (subject, quantity)
        if key in stack: raise ContractError("Derived parameter dependency cycle")
        pop = self.populations[subject]
        candidates = [r for r in self.registry.records if r.binding_subject == subject
                      and r.quantity == quantity and (mechanism is None or r.mechanism == mechanism)
                      and (not r.cell_types or set(pop.cell_types) <= set(r.cell_types))
                      and r.status not in ("unsupported", "not_implemented")]
        if not candidates: raise ContractError("Missing parameter: " + str(key))
        # A conditional record is not an automatic fallback on a mismatched request.
        for r in candidates:
            if r.status == "proposed":
                raise ContractError("Proposed catalog record is not a runnable binding")
            if r.status == "engineering_default" and not self.context.engineering_preview:
                raise ContractError("Engineering fallback requires explicit preview")
            if r.unit != expected_unit: raise ContractError("Parameter unit mismatch: " + r.parameter_id)
        priority = max(self.PRIORITY[r.value_spec["rule"]] for r in candidates)
        chosen = [r for r in candidates if self.PRIORITY[r.value_spec["rule"]] == priority]
        if len(chosen) != 1: raise ContractError("Ambiguous/conflicting same-level physical sources")
        r = chosen[0]; r.applicability.check(self.context, pop); v = r.value_spec
        rule = v["rule"]
        if rule in ("shared_constant", "type_constant", "conditional_override"):
            value = fp64(v["value"])
        elif rule in ("eccentricity_function", "retinotopic_function", "background_rule"):
            required = ("anchors", "values", "axis", "axis_unit", "interpolation", "coordinate_frame")
            if any(k not in v for k in required) or v["interpolation"] != "linear":
                raise ContractError("Sourced coordinate/units/anchors/linear interpolation required")
            if not r.evidence.locator: raise ContractError("Function source missing")
            if (rule != "background_rule" and
                    (pop.coordinate_frame == "source_declared_unknown"
                     or v["coordinate_frame"] != pop.coordinate_frame)):
                raise ContractError("Position function coordinate-frame mismatch/unknown")
            axis = v["axis"]
            if axis == "eccentricity_deg" and v["axis_unit"] == "deg":
                points = pop.eccentricity_deg
            elif axis in ("x", "y") and v["axis_unit"] == pop.xy_unit and pop.xy is not None:
                points = pop.xy[:, 0 if axis == "x" else 1]
            elif rule == "background_rule":
                points = np.asarray(self.context.background.get(axis), float)
                if self.context.background.get(axis+"_unit") != v["axis_unit"]:
                    raise ContractError("Background axis units missing")
            else: raise ContractError("Unsupported explicit function coordinate")
            a, b = fp64(v["anchors"]), fp64(v["values"])
            if (points is None or a.ndim != 1 or b.shape != a.shape or len(a) < 2
                    or np.any(np.diff(a) <= 0) or np.any(points < a[0]) or np.any(points > a[-1])):
                raise ContractError("Function shape/domain/extrapolation error")
            value = fp64(np.interp(points, a, b))
        else:
            declared_deps = v.get("dependencies", ())
            deps = [self.resolve(d["subject"], d["quantity"], d["unit"],
                                 stack=stack+(key,))[0] for d in declared_deps]
            op = v.get("operation")
            if not deps: raise ContractError("Derived parameter dependencies missing")
            expected = unit_dimension(r.unit)
            if op == "sum":
                if any(unit_dimension(d["unit"]) != expected for d in declared_deps):
                    raise ContractError("Derived sum dimensional conflict")
            else:
                dimensions = {}
                for i, d in enumerate(declared_deps):
                    sign = -1 if op == "ratio" and i else 1
                    for axis, power in unit_dimension(d["unit"]).items():
                        dimensions[axis] = dimensions.get(axis, 0)+sign*power
                dimensions = {k: v for k, v in dimensions.items() if v}
                if dimensions != expected: raise ContractError("Derived physical unit conflict")
            if op == "sum": value = fp64(sum(deps))
            elif op == "product": value = fp64(np.prod(np.stack(np.broadcast_arrays(*deps)), axis=0))
            elif op == "ratio" and len(deps) == 2 and np.all(deps[1] != 0): value = fp64(deps[0]/deps[1])
            else: raise ContractError("Unsupported/invalid safe derived operation")
        if v.get("layout") == "broadcast":
            value = fp64(np.broadcast_to(value, (len(pop.ids),)+value.shape))
        elif v.get("layout") == "per_group" and value.shape[:1] != (len(pop.ids),):
            raise ContractError("Population parameter shape mismatch")
        if "minimum" in r.domain and np.any(value < r.domain["minimum"]):
            raise ContractError("Parameter below legal domain")
        if "maximum" in r.domain and np.any(value > r.domain["maximum"]):
            raise ContractError("Parameter above legal domain")
        return value, r.parameter_id


@dataclass(frozen=True)
class RouteRecord:
    route_id: str
    pre: str
    post: str
    operators: Mapping
    source_identity: str
    semantics: str
    units: Mapping

    def __post_init__(self):
        object.__setattr__(self,"operators",frozen(self.operators))
        object.__setattr__(self,"units",frozen(self.units))
        if not self.route_id or not self.pre or not self.post or not self.source_identity:
            raise ContractError("Route requires explicit source and target identity")


class RouteRegistry:
    def __init__(self, records=(), matrices=None):
        self.records = tuple(records)
        self._matrices = {} if matrices is None else dict(matrices)
        if len({r.route_id for r in self.records}) != len(self.records):
            raise ContractError("Duplicate route identities")
        for k, m in self._matrices.items():
            if not sparse.isspmatrix_csr(m): raise ContractError("Route operator must remain CSR")
            if not np.isfinite(m.data).all(): raise ContractError("Nonfinite route")

    @property
    def identity(self): return identity(self.records)

    def matrix(self, route_id, kind):
        return self._matrices[(route_id, kind)]


@dataclass(frozen=True)
class StateBundle:
    states: Mapping
    history_rng_delay: Mapping
    time_seconds: float
    static_parameter_bundle_id: str
    qualified_background: bool = False

    def __post_init__(self):
        if not np.isfinite(self.time_seconds) or not self.static_parameter_bundle_id:
            raise ContractError("State requires finite time and independent static parameter identity")
        object.__setattr__(self, "states", frozen(self.states))
        object.__setattr__(self, "history_rng_delay", frozen(self.history_rng_delay))


@dataclass(frozen=True)
class CalibrationRecord:
    record_id: str
    parameter_keys: tuple
    data_split_sha: Mapping
    input_model_identity: Mapping
    estimator: Mapping
    parameters: Mapping
    uncertainty: Mapping
    evidence: tuple
    performance: Mapping
    qualification: str
    limitations: tuple
    authorization: str

    def __post_init__(self):
        if self.qualification not in STATUSES or not self.authorization or not self.record_id:
            raise ContractError("Immutable calibration scope/authorization/status required")
        for k in ("data_split_sha", "input_model_identity", "estimator", "parameters", "uncertainty", "performance"):
            object.__setattr__(self, k, frozen(getattr(self, k)))

    @property
    def identity(self): return identity(self)

    def write_immutable(self,path):
        with Path(path).open("x",encoding="utf8") as stream:
            json.dump(dict(record_id=self.record_id,content_sha256=self.identity,record=data(self)),
                      stream,ensure_ascii=False,indent=2,allow_nan=False)

    @classmethod
    def read_immutable(cls,path):
        payload=json.loads(Path(path).read_text(encoding="utf8"))
        row=dict(payload["record"])
        for k in ("parameter_keys","evidence","limitations"):row[k]=tuple(row[k])
        result=cls(**row)
        if result.identity!=payload["content_sha256"]:raise ContractError("Calibration record altered")
        return result


@dataclass(frozen=True)
class ResolvedParameterBundle:
    context: RunContext
    populations: Mapping
    values: Mapping
    provenance_refs: Mapping
    registry: IntrinsicParameterRegistry
    routes: RouteRegistry
    legacy_parameters: Mapping
    coverage: Mapping
    schema_version: str = SCHEMA

    def __post_init__(self):
        self.registry.freeze()
        object.__setattr__(self, "values", MappingProxyType({k: fp64(v) for k, v in self.values.items()}))
        object.__setattr__(self, "provenance_refs", frozen(self.provenance_refs))
        object.__setattr__(self, "coverage", frozen(self.coverage))
        object.__setattr__(self, "legacy_parameters", frozen(self.legacy_parameters))
        object.__setattr__(self, "populations", MappingProxyType(dict(self.populations)))

    @property
    def identity(self):
        return identity(dict(schema=self.schema_version, context=self.context,
            populations=self.populations, values=self.values, provenance=self.provenance_refs,
            routes=self.routes.identity, coverage=self.coverage))

    def manifest(self):
        return data(dict(schema=self.schema_version, binding_id=self.identity, context=self.context,
            registry_id=self.registry.identity, route_id=self.routes.identity,
            population_indices=self.populations, provenance_refs=self.provenance_refs,
            theta_schema=self.registry.theta_schema(), coverage=self.coverage,
            scientific_qualification_unchanged=True))
