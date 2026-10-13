from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import numpy as np

class ContractError(ValueError): pass
class MissingCalibrationError(ContractError): pass
class UnsupportedReferenceBranch(ContractError): pass

class RunMode(str, Enum):
    FULL_PHYSIOLOGY = "full_physiology"
    REFERENCE = "component_reference"
    NUMERIC_FIXTURE = "numeric_fixture"

def canonical_hash(value):
    def convert(v):
        if isinstance(v, np.ndarray):
            return {"dtype": str(v.dtype), "shape": list(v.shape), "data": v.tolist()}
        if isinstance(v, Enum): return v.value
        if hasattr(v, "__dataclass_fields__"):
            return {k: convert(getattr(v, k)) for k in v.__dataclass_fields__}
        if isinstance(v, dict): return {str(k): convert(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [convert(x) for x in v]
        return v
    raw = json.dumps(convert(value), sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()

def finite_array(value, shape=None):
    a = np.array(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(a)): raise ContractError("Nonfinite array")
    if shape is not None and a.shape != tuple(shape): raise ContractError(f"Shape {a.shape} != {shape}")
    a.setflags(write=False)
    return a

@dataclass(frozen=True)
class Quantity:
    values: np.ndarray
    unit: str
    meaning: str
    evidence_tag: str
    coordinate_frame: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "values", finite_array(self.values))
        if not self.unit or not self.meaning or not self.evidence_tag:
            raise ContractError("Quantity requires unit, semantic meaning and evidence tag")

    def require(self, unit, meaning, shape=None, frame=None):
        if (self.unit, self.meaning) != (unit, meaning):
            raise ContractError(f"Expected {unit}/{meaning}, received {self.unit}/{self.meaning}")
        if frame is not None and self.coordinate_frame != frame:
            raise ContractError(f"Coordinate frame {self.coordinate_frame} != {frame}")
        if shape is not None and self.values.shape != tuple(shape):
            raise ContractError(f"Shape {self.values.shape} != {shape}")
        return self.values

@dataclass(frozen=True)
class CellTable:
    cell_id: tuple
    cell_type: tuple
    subtype: tuple
    species_profile_id: tuple
    hemiretina: tuple
    meridian: tuple
    soma_position: Quantity | None
    rf_position: Quantity | None
    position_unit: str
    coordinate_frame: str
    eccentricity: Quantity
    parameter_profile_id: tuple
    evidence_tags: tuple

    def __post_init__(self):
        n = len(self.cell_id)
        if n == 0 or len(set(self.cell_id)) != n: raise ContractError("Cell IDs must be unique and nonempty")
        for name in ("cell_type", "subtype", "species_profile_id", "hemiretina", "meridian", "parameter_profile_id", "evidence_tags"):
            if len(getattr(self, name)) != n: raise ContractError(f"CellTable length mismatch: {name}")
        if any(not x for x in self.evidence_tags): raise ContractError("Every cell requires evidence tag")
        if self.soma_position is not None:
            self.soma_position.require(self.position_unit, "anatomical_position", (n, 2), self.coordinate_frame)
        if self.rf_position is not None:
            self.rf_position.require(self.position_unit, "functional_position", (n, 2), self.coordinate_frame)
        if self.eccentricity.meaning != "global_eccentricity" or self.eccentricity.values.shape != (n,):
            raise ContractError("Global eccentricity must be explicit and separate from patch coordinates")
        if self.eccentricity.coordinate_frame not in ("anatomical_retina", "functional_visual_field"):
            raise ContractError("Eccentricity frame is required")
        if self.eccentricity.unit not in ("mm", "deg") or np.any(self.eccentricity.values < 0):
            raise ContractError("Invalid eccentricity units/domain")

@dataclass(frozen=True)
class StateSpec:
    state_name: str
    cell_ids: tuple
    shape: tuple
    unit: str
    meaning: str
    initial_condition_source: str

    def __post_init__(self):
        if self.shape != (len(self.cell_ids),) or len(set(self.cell_ids)) != len(self.cell_ids):
            raise ContractError("State shape/IDs mismatch")
        voltage_meanings = ("absolute_voltage", "generator_deviation", "generator_with_output_offset")
        explicit_units={"activity":"dimensionless","effective_opsin_activity":"s^-2","effective_pde_activity":"s^-1","cgmp_concentration":"uM","outer_segment_calcium":"uM","cgmp_synthesis_cache":"uM/s"}
        if (self.meaning in voltage_meanings and self.unit != "mV") or (self.meaning in explicit_units and self.unit != explicit_units[self.meaning]) or self.meaning not in voltage_meanings + tuple(explicit_units):
            raise ContractError("Unsupported state semantic/unit")
        if not self.initial_condition_source: raise ContractError("Initial condition source required")

@dataclass(frozen=True)
class StateDerivativeVector:
    """Ordered derivative blocks retain each state unit and semantic identity."""
    ordered_derivatives: tuple
    state_spec_hash: str

    def flatten(self, state_specs, evidence_tag):
        if self.state_spec_hash != canonical_hash(state_specs) or len(self.ordered_derivatives) != len(state_specs):
            raise ContractError("Derivative/state registry mismatch")
        arrays = []
        for derivative, spec in zip(self.ordered_derivatives, state_specs):
            derivative.require(spec.unit + "/s", spec.meaning + "_derivative", spec.shape)
            if derivative.evidence_tag != evidence_tag:
                raise ContractError("Derivative lost evidence tag")
            arrays.append(derivative.values)
        return np.concatenate(arrays)

@dataclass(frozen=True)
class SolverConfig:
    method: str = "DOP853"
    rtol: float = 1e-10
    atol: float = 1e-12
    max_step: float = 0.002
    dtype: str = "float64"
    input_interpolation: str = "analytic"
    fixed_step_seconds: float | None = None

    def __post_init__(self):
        if self.method not in ("RK45", "DOP853", "Radau", "BDF", "Chen2024AuthorDiscrete") or self.dtype != "float64":
            raise ContractError("Unsupported solver/dtype")
        if any(not np.isfinite(x) or x <= 0 for x in (self.rtol, self.atol, self.max_step)):
            raise ContractError("Solver tolerances and max_step must be finite positive")
        if self.input_interpolation not in ("analytic", "zero_order_hold", "linear"):
            raise ContractError("Explicit input interpolation required")
        if self.method=="Chen2024AuthorDiscrete" and (self.fixed_step_seconds!=0.0001 or self.input_interpolation!="zero_order_hold"):
            raise ContractError("Chen2024 requires source dt=0.0001s and explicit interval ZOH")
        if self.method!="Chen2024AuthorDiscrete" and self.fixed_step_seconds is not None:
            raise ContractError("Adaptive solver cannot silently consume a fixed author step")

@dataclass(frozen=True)
class StimulusSegment:
    start_seconds: float
    stop_seconds: float
    evaluator: object
    unit: str
    meaning: str
    interpolation: str
    evidence_tag: str
    input_contract: dict
    breakpoints_seconds: tuple = ()

    def __post_init__(self):
        if not np.isfinite(self.start_seconds+self.stop_seconds) or self.stop_seconds <= self.start_seconds:
            raise ContractError("Invalid stimulus interval")
        if (self.unit,self.meaning) not in (("dimensionless","cone_contrast"),("R*/s","photoisomerization_rate")):
            raise ContractError("Stimulus port requires explicit contrast or photoisomerization rate")
        if not callable(self.evaluator) or not self.evidence_tag:
            raise ContractError("Stimulus needs evaluator and per-object evidence tag")
        if any(not self.start_seconds < t < self.stop_seconds for t in self.breakpoints_seconds):
            raise ContractError("Breakpoints must lie strictly within segment")
        if tuple(sorted(set(self.breakpoints_seconds))) != self.breakpoints_seconds:
            raise ContractError("Breakpoints must be unique and sorted")

    def at(self, t):
        if t < self.start_seconds-1e-12 or t > self.stop_seconds+1e-12:
            raise ContractError("Stimulus evaluated outside segment")
        q = self.evaluator(t)
        q.require(self.unit, self.meaning)
        if q.evidence_tag != self.evidence_tag: raise ContractError("Stimulus lost evidence tag")
        return q

@dataclass(frozen=True)
class SimulationState:
    time_seconds: float
    ordered_state_arrays: tuple
    state_spec_hash: str
    graph_hash: str
    parameter_hash: str
    model_id: str
    model_version: str
    model_hash: str
    input_interpolation: str
    solver_config: SolverConfig
    evidence_tag: str
    optional_rng_state: object = None

    def __post_init__(self):
        if not np.isfinite(self.time_seconds): raise ContractError("Invalid absolute time")
        object.__setattr__(self, "ordered_state_arrays", tuple(finite_array(x) for x in self.ordered_state_arrays))
        if self.optional_rng_state is not None: raise ContractError("No stochastic mechanism authorized")

@dataclass(frozen=True)
class SimulationResult:
    run_mode: RunMode
    model_id: str
    source_manifest_hash: str
    input_contract: dict
    outputs_with_units: dict
    coverage_report: object
    numeric_checks: dict
    claim_level: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.run_mode == RunMode.FULL_PHYSIOLOGY:
            raise ContractError("Full physiological result creation is blocked in A/B")
        if self.run_mode == RunMode.NUMERIC_FIXTURE and any(q.evidence_tag != "synthetic_fixture" for q in self.outputs_with_units.values()):
            raise ContractError("Every synthetic output must inherit its evidence tag")
