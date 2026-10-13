"""Supplied HC electrical contacts in SI units; no native calibration or g default.

The existing vH code uses native_relative_voltage, not an identified physical
membrane voltage/C. The paper's Vh label in mV and fixed gh do not identify an
affine conversion for this project state. This module does not change its RHS.
"""

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np

from .contracts import ContractError, Quantity, canonical_hash


def _ids(values):
    ids = tuple(values)
    if not ids or any(not isinstance(value, str) or not value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise ContractError("HC gap cell IDs must be unique, nonempty stable strings")
    return ids


def _quantity(value, unit, meaning, shape):
    if not isinstance(value, Quantity):
        raise ContractError(f"HC gap requires supplied {unit}/{meaning} Quantity; no default calibration")
    return value.require(unit, meaning, shape)


def _evidence(*quantities):
    return "|".join(dict.fromkeys(q.evidence_tag for q in quantities))


def _contact_indices(cell_ids, edges):
    lookup = {cell_id: i for i, cell_id in enumerate(cell_ids)}
    contacts, seen = [], set()
    for edge in edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            raise ContractError("HC gap edges require pairs of stable cell IDs")
        left, right = edge
        if not isinstance(left, str) or not isinstance(right, str):
            raise ContractError("HC gap edge endpoints must be stable string IDs")
        if left not in lookup or right not in lookup:
            raise ContractError(f"Unknown HC gap cell ID in {edge!r}")
        if left == right:
            raise ContractError(f"HC gap self-edge is invalid: {left}")
        pair = frozenset((left, right))
        if pair in seen:
            raise ContractError(f"Duplicate undirected HC gap edge: {edge!r}")
        seen.add(pair)
        contacts.append((left, right))
    left = np.array([lookup[a] for a, _ in contacts], dtype=int)
    right = np.array([lookup[b] for _, b in contacts], dtype=int)
    left.setflags(write=False)
    right.setflags(write=False)
    return tuple(contacts), left, right


class NativeHCVoltageCalibration:
    """An explicit project affine map V=Vref+gain*(native-native_ref).

    Positive or negative gain must be supplied with its units. A named map is
    an application assumption; this interface does not make it a measurement.
    """

    def __init__(self, cell_ids, *, profile_name, native_reference, voltage_reference,
                 gain, project_evidence_tag):
        self.cell_ids = _ids(cell_ids)
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise ContractError("Native HC calibration requires a supplied profile name")
        if not isinstance(project_evidence_tag, str) or not project_evidence_tag.startswith("project_approximation"):
            raise ContractError("Native HC calibration requires named project_approximation evidence")
        if not isinstance(native_reference, Quantity) or (native_reference.unit, native_reference.meaning) not in (
                ("vH_native_voltage", "native_relative_voltage"),
                ("dimensionless", "effective_HC_activation")):
            raise ContractError("Native HC reference needs explicit project state unit/meaning")
        n = len(self.cell_ids)
        self.native_unit = native_reference.unit
        self.native_meaning = native_reference.meaning
        self.native_reference = _quantity(native_reference, self.native_unit, self.native_meaning, (n,))
        self.voltage_reference = _quantity(voltage_reference, "V", "physical_membrane_voltage", (n,))
        self.gain = _quantity(gain, "V/" + self.native_unit, "native_to_voltage_gain", (n,))
        if np.any(self.gain == 0):
            raise ContractError("Native HC gain must be nonzero for invertible voltage/derivative conversion")
        self.profile_name = profile_name
        self.evidence_tag = project_evidence_tag
        self._reference_evidence = _evidence(native_reference, voltage_reference, gain)

    @property
    def metadata(self):
        return {
            "profile_name": self.profile_name,
            "evidence_status": "project_approximation",
            "project_evidence_tag": self.evidence_tag,
            "supplied_reference_evidence": self._reference_evidence,
            "cell_ids": list(self.cell_ids),
            "native_unit": self.native_unit,
            "native_meaning": self.native_meaning,
            "voltage_unit": "V",
            "gain_unit": "V/" + self.native_unit,
            "physical_calibration_claim": False,
            "source_label_limit": "vH paper Vh=mV and gh fixed to one do not identify project native-state voltage gain or membrane capacitance",
            "native_runtime_time_unit": "s",
            "native_time_conversion": "existing vH RHS converts source per-ms rates to per-second by multiplying by 1000; gap I/C is already per-second",
            "missing_native_hooks": ["membrane_capacitance", "physical voltage output", "gap current/derivative RHS input"],
        }

    def voltage(self, native_state):
        values = _quantity(native_state, self.native_unit, self.native_meaning, (len(self.cell_ids),))
        return Quantity(self.voltage_reference + self.gain * (values - self.native_reference),
                        "V", "physical_membrane_voltage", self.evidence_tag + "|" + native_state.evidence_tag)

    def native_derivative(self, physical_derivative, *, time_unit="s"):
        values = _quantity(physical_derivative, "V/s", "physical_membrane_voltage_derivative", (len(self.cell_ids),))
        if time_unit not in ("s", "ms"):
            raise ContractError("Native HC derivative time unit must be explicitly s or ms")
        # A derivative per millisecond is 1/1000 of the derivative per second.
        rate = values / self.gain * (1.0 if time_unit == "s" else .001)
        return Quantity(rate, self.native_unit + "/" + time_unit,
                        self.native_meaning + "_derivative", self.evidence_tag + "|" + physical_derivative.evidence_tag)


class HCGapOperator:
    """Reciprocal undirected finite contacts, disabled unless explicitly enabled.

    Every supplied member is retained, including HC cells with no cone input.
    Contacts outside this finite instance remain unknown and are not normalized
    into the included edges. Conductances are never guessed or row-scaled.
    """

    def __init__(self, cell_ids, edges, conductance, *, enabled=False, boundary_mode="finite_open"):
        self.cell_ids = _ids(cell_ids)
        if type(enabled) is not bool:
            raise ContractError("HC gap enabled flag must be an explicit boolean")
        if boundary_mode != "finite_open":
            raise ContractError("HC gap only supports explicit finite_open boundary; exterior closure is unknown")
        self.edges, self._left, self._right = _contact_indices(self.cell_ids, edges)
        self.conductance = _quantity(conductance, "S", "gap_junction_conductance", (len(self.edges),))
        if np.any(self.conductance <= 0):
            raise ContractError("Supplied HC gap edge conductances must be positive; use no edge for zero conductance")
        self._conductance_quantity = conductance
        self.enabled = enabled
        self.boundary_mode = boundary_mode

    @property
    def metadata(self):
        return {
            "enabled": self.enabled,
            "default_enabled": False,
            "boundary_mode": self.boundary_mode,
            "outside_domain_dependencies": "UNKNOWN",
            "external_contacts": "not included; absence is not biological zero",
            "recurrent_exterior_closure_certified": False,
            "cell_ids": list(self.cell_ids),
            "edge_count": len(self.edges),
            "edge_conductance_unit": "S",
            "edge_conductance_evidence": self._conductance_quantity.evidence_tag,
            "conductance_default": None,
            "conductance_normalization": "none",
            "reciprocal_conservative": True,
            "zero_cone_input_members_retained": True,
            "component_role": "supplied topology; coupling is inactive when disabled",
            "physiological_validation": False,
        }

    def connected_components(self):
        """Topology components include isolated IDs; cone input is not a filter."""
        neighbors = {cell_id: set() for cell_id in self.cell_ids}
        for left, right in self.edges:
            neighbors[left].add(right)
            neighbors[right].add(left)
        unseen = set(self.cell_ids)
        groups = []
        for first in self.cell_ids:
            if first not in unseen:
                continue
            pending = [first]
            members = set()
            while pending:
                cell_id = pending.pop()
                if cell_id in members:
                    continue
                members.add(cell_id)
                pending.extend(neighbors[cell_id] - members)
            unseen -= members
            groups.append(tuple(cell_id for cell_id in self.cell_ids if cell_id in members))
        return tuple(groups)

    def currents(self, voltage):
        values = _quantity(voltage, "V", "physical_membrane_voltage", (len(self.cell_ids),))
        currents = np.zeros(len(self.cell_ids))
        if self.enabled:
            edge_current = self.conductance * (values[self._right] - values[self._left])
            np.add.at(currents, self._left, edge_current)
            np.add.at(currents, self._right, -edge_current)
        return Quantity(currents, "A", "electrical_gap_current", _evidence(self._conductance_quantity, voltage))

    def voltage_derivative(self, voltage, capacitance):
        C = _quantity(capacitance, "F", "membrane_capacitance", (len(self.cell_ids),))
        if np.any(C <= 0):
            raise ContractError("HC gap needs supplied positive membrane capacitance per cell")
        current = self.currents(voltage)
        return Quantity(current.values / C, "V/s", "physical_membrane_voltage_derivative", _evidence(current, capacitance))

    def native_derivative(self, native_state, calibration, capacitance, *, time_unit="s"):
        if not isinstance(calibration, NativeHCVoltageCalibration):
            raise ContractError("Native HC gap requires explicit named voltage calibration adapter")
        if calibration.cell_ids != self.cell_ids:
            raise ContractError("Native HC calibration cell IDs/order must match the gap operator")
        voltage = calibration.voltage(native_state)
        return calibration.native_derivative(self.voltage_derivative(voltage, capacitance), time_unit=time_unit)


class EffectiveHCGapOperator:
    """Native effective rates, requiring equal capacity and a common state scale.

    k is supplied per millisecond. The added term is sum(k*(h_j-h_i)), with a
    single conversion to seconds and no division by the chemical aI*tau_h.
    """

    connected_components = HCGapOperator.connected_components

    def __init__(self, cell_ids, edges, rate, *, assumption_profile_name,
                 project_evidence_tag, equal_effective_capacity,
                 common_state_scale_reference, enabled=False, boundary_mode="finite_open"):
        self.cell_ids = _ids(cell_ids)
        if not isinstance(assumption_profile_name, str) or not assumption_profile_name.strip():
            raise ContractError("Effective HC rate requires a named assumption profile")
        if not isinstance(project_evidence_tag, str) or not project_evidence_tag.startswith("project_approximation"):
            raise ContractError("Effective HC rate requires named project_approximation evidence")
        if equal_effective_capacity is not True or common_state_scale_reference is not True:
            raise ContractError("Symmetric native rates require explicit equal effective capacity and common scale/reference assumptions")
        if type(enabled) is not bool or boundary_mode != "finite_open":
            raise ContractError("Effective HC rate requires boolean enabled and finite_open boundary")
        self.edges, self._left, self._right = _contact_indices(self.cell_ids, edges)
        self.rate = _quantity(rate, "ms^-1", "effective_HC_gap_rate", (len(self.edges),))
        if np.any(self.rate < 0):
            raise ContractError("Effective HC gap rates must be nonnegative")
        self.rate_quantity = rate
        self.enabled = enabled
        self.boundary_mode = boundary_mode
        self.assumption_profile_name = assumption_profile_name
        self.evidence_tag = project_evidence_tag

    @property
    def metadata(self):
        return {
            "enabled": self.enabled, "default_enabled": False,
            "route": "symmetric_effective_native_rate",
            "evidence_status": "project_approximation",
            "assumption_profile_name": self.assumption_profile_name,
            "project_evidence_tag": self.evidence_tag,
            "supplied_rate_evidence": self.rate_quantity.evidence_tag,
            "equal_effective_capacity": True,
            "common_state_scale_reference": True,
            "rate_unit": "ms^-1", "rate_default": None,
            "physical_conductance_claim": False,
            "boundary_mode": self.boundary_mode,
            "outside_domain_dependencies": "UNKNOWN",
            "recurrent_exterior_closure_certified": False,
            "cell_ids": list(self.cell_ids), "edge_count": len(self.edges),
            "component_role": "supplied topology; zero-rate edges or disabled coupling transmit no current",
            "zero_cone_input_members_retained": True,
            "rate_normalization": "none", "chemical_aI_tau_division": False,
            "time_conversion_to_seconds": 1000,
            "source_label_limit": "native-state rate is not physical nS; no conversion from the disputed Packer/Dacey Table 1 resistance value",
            "physiological_validation": False,
        }

    def native_derivative(self, native_state):
        values = _quantity(native_state, "vH_native_voltage", "native_relative_voltage", (len(self.cell_ids),))
        derivative = np.zeros(len(self.cell_ids))
        if self.enabled:
            pair_rate = 1000 * self.rate * (values[self._right] - values[self._left])
            np.add.at(derivative, self._left, pair_rate)
            np.add.at(derivative, self._right, -pair_rate)
        return Quantity(derivative, "vH_native_voltage/s", "native_relative_voltage_derivative",
                        self.evidence_tag + "|" + _evidence(self.rate_quantity, native_state))


class NativeHCGapExtension:
    """Delegate source dynamics and add a supplied gap term to the existing Vh block."""

    def __init__(self, native_model, gap, *, calibration=None, capacitance=None):
        if not isinstance(gap, (EffectiveHCGapOperator, HCGapOperator)):
            raise ContractError("Native HC extension requires a supplied effective-rate or SI gap operator")
        self.native_model = native_model
        self.gap = gap
        self.calibration = calibration
        self.capacitance = capacitance
        matches = [i for i, spec in enumerate(native_model.state_specs) if spec.state_name == "Vh"]
        if len(matches) != 1:
            raise ContractError("Native HC gap extension requires one existing Vh state block")
        self.hc_state_index = matches[0]
        spec = native_model.state_specs[self.hc_state_index]
        if (spec.unit, spec.meaning) != ("vH_native_voltage", "native_relative_voltage") or tuple(spec.cell_ids) != gap.cell_ids:
            raise ContractError("Native HC gap operator IDs/order must match all existing Vh states; omitted HC cells cannot be silently frozen or invented")
        if isinstance(gap, HCGapOperator):
            if not isinstance(calibration, NativeHCVoltageCalibration) or calibration.cell_ids != gap.cell_ids:
                raise ContractError("Physical native HC gap requires a matching explicit calibration")
            if (calibration.native_unit, calibration.native_meaning) != (spec.unit, spec.meaning):
                raise ContractError("Physical native HC calibration must match the existing Vh state unit/meaning")
            C = _quantity(capacitance, "F", "membrane_capacitance", spec.shape)
            if np.any(C <= 0):
                raise ContractError("Physical native HC gap requires positive supplied C")
        elif calibration is not None or capacitance is not None:
            raise ContractError("Effective-rate route must not imply physical calibration or capacitance")
        self.approximations = list(getattr(native_model, "approximations", ())) + [{
            "id": "explicit_native_HC_gap_extension", "gap_contract": gap.metadata,
            "initial_condition": "unchanged source initializer; coupled background convergence must be checked when nonzero gap is enabled",
            "no_cone_HC_limit": "operator retains all supplied states; this wrapper does not construct missing HC states or their chemical adaptation factor",
        }]

    def __getattr__(self, name):
        return getattr(self.native_model, name)

    @property
    def model_hash(self):
        return canonical_hash({
            "native_model": self.native_model.model_hash,
            "gap": self.gap.metadata, "edges": self.gap.edges,
            "weights": self.gap.rate if isinstance(self.gap, EffectiveHCGapOperator) else self.gap.conductance,
            "calibration": None if self.calibration is None else {
                "metadata": self.calibration.metadata, "gain": self.calibration.gain,
                "native_reference": self.calibration.native_reference,
                "voltage_reference": self.calibration.voltage_reference},
            "capacitance": self.capacitance,
            "extension_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        })

    def derivative(self, t, blocks, incoming):
        original = self.native_model.derivative(t, blocks, incoming)
        h = Quantity(blocks[self.hc_state_index], "vH_native_voltage", "native_relative_voltage", self.native_model.tag)
        extra = (self.gap.native_derivative(h) if isinstance(self.gap, EffectiveHCGapOperator)
                 else self.gap.native_derivative(h, self.calibration, self.capacitance, time_unit="s"))
        if not np.any(extra.values):
            return original
        changed = list(original)
        q = original[self.hc_state_index]
        changed[self.hc_state_index] = Quantity(q.values + extra.values, q.unit, q.meaning, q.evidence_tag)
        return tuple(changed)


def with_native_hc_gap(component, gap, *, calibration=None, capacitance=None):
    """Opt-in RuntimeComponent wrapper; source file, initializer and readout stay intact."""
    native = NativeHCGapExtension(component.native_model, gap, calibration=calibration, capacitance=capacitance)
    layer = "HC_gap_effective_rate" if isinstance(gap, EffectiveHCGapOperator) else "HC_gap_physical_adapter"
    return replace(component, native_model=native, derivative=native.derivative,
                   layers=component.layers + ((layer,) if gap.enabled else ()),
                   implementation_hash=canonical_hash({"original": component.implementation_hash,
                       "HC_gap_extension_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))
