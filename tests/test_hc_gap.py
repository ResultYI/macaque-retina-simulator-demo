from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from examples.full_network_fixture import population
from retina_sim.assembly import ComponentSpec
from retina_sim.contracts import ContractError, Quantity, RunMode
from retina_sim.core_sources import load_core_registry
from retina_sim.hc_gap import (EffectiveHCGapOperator, HCGapOperator,
                               NativeHCVoltageCalibration, with_native_hc_gap)
from retina_sim.vh2005 import KEY, PROFILE, SOURCE_VALUES, vh_component


ROOT = Path(__file__).resolve().parents[1]
IDS = ("HC:a", "HC:b", "HC:zero_cone_input", "HC:isolated")
PROJECT = "project_approximation:synthetic_gap_test_v1"


def q(values, unit, meaning):
    return Quantity(values, unit, meaning, "synthetic_fixture")


def rate_gap(ids=IDS, values=(.01, .02), *, enabled=True):
    return EffectiveHCGapOperator(ids, [(ids[0], ids[1]), (ids[1], ids[2])],
        q(values, "ms^-1", "effective_HC_gap_rate"),
        assumption_profile_name="equal_capacity_common_scale_synthetic_test",
        project_evidence_tag=PROJECT, equal_effective_capacity=True,
        common_state_scale_reference=True, enabled=enabled)


def test_si_currents_are_conservative_dissipative_and_not_normalized():
    g = q([2e-9, 3e-9], "S", "gap_junction_conductance")
    operator = HCGapOperator(IDS, [(IDS[0], IDS[1]), (IDS[1], IDS[2])], g, enabled=True)
    voltage = q([.01, 0, -.02, .2], "V", "physical_membrane_voltage")
    current = operator.currents(voltage)
    np.testing.assert_allclose(current.values, [-2e-11, -4e-11, 6e-11, 0], rtol=1e-15, atol=1e-25)
    assert np.sum(current.values) == pytest.approx(0, abs=1e-25)
    assert voltage.values @ current.values == pytest.approx(-(2e-9 * .01**2 + 3e-9 * .02**2))
    assert voltage.values @ current.values <= 0
    np.testing.assert_array_equal(operator.currents(q(np.full(4, -.06), "V", "physical_membrane_voltage")).values, 0)
    assert operator.connected_components() == (IDS[:3], IDS[3:])
    assert operator.metadata["outside_domain_dependencies"] == "UNKNOWN"
    assert operator.metadata["conductance_normalization"] == "none"


def test_si_capacitance_and_charge_scale_in_seconds():
    operator = HCGapOperator(IDS, [(IDS[0], IDS[1]), (IDS[1], IDS[2])],
                            q([2e-9, 3e-9], "S", "gap_junction_conductance"), enabled=True)
    voltage = q([.01, 0, -.02, .2], "V", "physical_membrane_voltage")
    C = q([10e-12, 20e-12, 30e-12, 40e-12], "F", "membrane_capacitance")
    derivative = operator.voltage_derivative(voltage, C)
    derivative.require("V/s", "physical_membrane_voltage_derivative", (4,))
    np.testing.assert_allclose(derivative.values, [-2, -2, 2, 0], rtol=1e-15, atol=1e-14)
    dt_seconds = .001
    charge = C.values * derivative.values * dt_seconds
    np.testing.assert_allclose(charge, operator.currents(voltage).values * dt_seconds, rtol=1e-15, atol=1e-28)
    assert np.sum(charge) == pytest.approx(0, abs=1e-28)
    assert HCGapOperator(IDS, [], q([], "S", "gap_junction_conductance")).enabled is False
    disabled = HCGapOperator(IDS, [(IDS[0], IDS[1])], q([1e-9], "S", "gap_junction_conductance"))
    np.testing.assert_array_equal(disabled.currents(voltage).values, 0)


@pytest.mark.parametrize("edges,values,message", [
    ([(IDS[0], IDS[0])], [1e-9], "self-edge"),
    ([(IDS[0], IDS[1]), (IDS[1], IDS[0])], [1e-9, 1e-9], "Duplicate undirected"),
    ([(IDS[0], "missing")], [1e-9], "Unknown HC"),
    ([(IDS[0], [])], [1e-9], "stable string IDs"),
    ([(IDS[0], IDS[1])], [0], "must be positive"),
    ([(IDS[0], IDS[1])], [-1e-9], "must be positive"),
])
def test_invalid_si_contacts_are_rejected(edges, values, message):
    with pytest.raises(ContractError, match=message):
        HCGapOperator(IDS, edges, q(values, "S", "gap_junction_conductance"), enabled=True)


def test_si_units_and_unknown_parameters_are_rejected():
    edge = [(IDS[0], IDS[1])]
    with pytest.raises(ContractError, match="Expected S"):
        HCGapOperator(IDS, edge, q([1], "nS", "gap_junction_conductance"))
    with pytest.raises(ContractError, match="supplied S"):
        HCGapOperator(IDS, edge, None)
    operator = HCGapOperator(IDS, edge, q([1e-9], "S", "gap_junction_conductance"), enabled=True)
    for state in (q([1] * 4, "dimensionless", "effective_HC_activation"),
                  q([1] * 4, "vH_native_voltage", "native_relative_voltage"),
                  q([1] * 4, "mV", "physical_membrane_voltage")):
        with pytest.raises(ContractError, match="Expected V"):
            operator.currents(state)
    voltage = q([0] * 4, "V", "physical_membrane_voltage")
    with pytest.raises(ContractError, match="supplied F"):
        operator.voltage_derivative(voltage, None)
    with pytest.raises(ContractError, match="positive membrane"):
        operator.voltage_derivative(voltage, q([0] * 4, "F", "membrane_capacitance"))
    with pytest.raises(ContractError, match="explicit named voltage"):
        operator.native_derivative(q([1] * 4, "vH_native_voltage", "native_relative_voltage"), None, None)


def test_physical_native_adapter_handles_heterogeneous_C_and_gains():
    ids = IDS[:2]
    calibration = NativeHCVoltageCalibration(ids, profile_name="synthetic_heterogeneous_affine",
        native_reference=q([0, 0], "vH_native_voltage", "native_relative_voltage"),
        voltage_reference=q([-.06, -.07], "V", "physical_membrane_voltage"),
        gain=q([.002, .004], "V/vH_native_voltage", "native_to_voltage_gain"),
        project_evidence_tag=PROJECT)
    operator = HCGapOperator(ids, [(ids[0], ids[1])], q([1e-9], "S", "gap_junction_conductance"), enabled=True)
    h = q([10, 10], "vH_native_voltage", "native_relative_voltage")
    C = q([10e-12, 20e-12], "F", "membrane_capacitance")
    np.testing.assert_allclose(calibration.voltage(h).values, [-.04, -.03], atol=1e-15)
    dh_seconds = operator.native_derivative(h, calibration, C)
    np.testing.assert_allclose(dh_seconds.values, [500, -125], rtol=1e-14)
    current = operator.currents(calibration.voltage(h)).values
    np.testing.assert_allclose(C.values * calibration.gain * dh_seconds.values, current, rtol=1e-14, atol=1e-25)
    assert np.sum(C.values * calibration.gain * dh_seconds.values) == pytest.approx(0, abs=1e-25)
    assert np.sum(dh_seconds.values) != 0  # h sum is not conserved for heterogeneous C/gain.
    dh_ms = operator.native_derivative(h, calibration, C, time_unit="ms")
    dh_ms.require("vH_native_voltage/ms", "native_relative_voltage_derivative", (2,))
    np.testing.assert_allclose(dh_ms.values, dh_seconds.values / 1000)
    with pytest.raises(ContractError, match="explicitly s or ms"):
        operator.native_derivative(h, calibration, C, time_unit="ticks")


def test_effective_rate_is_symmetric_and_converts_ms_once():
    operator = rate_gap()
    h = q([1, 3, 6, 7], "vH_native_voltage", "native_relative_voltage")
    derivative = operator.native_derivative(h)
    np.testing.assert_allclose(derivative.values, [20, 40, -60, 0])
    assert np.sum(derivative.values) == 0
    assert h.values @ derivative.values == -220
    np.testing.assert_array_equal(operator.native_derivative(q([1] * 4, "vH_native_voltage", "native_relative_voltage")).values, 0)
    assert operator.connected_components() == (IDS[:3], IDS[3:])
    assert not operator.metadata["chemical_aI_tau_division"]
    assert not operator.metadata["physical_conductance_claim"]
    with pytest.raises(ContractError, match="Expected ms"):
        EffectiveHCGapOperator(IDS, [], q([], "s^-1", "effective_HC_gap_rate"),
            assumption_profile_name="test", project_evidence_tag=PROJECT,
            equal_effective_capacity=True, common_state_scale_reference=True)
    with pytest.raises(ContractError, match="equal effective capacity"):
        EffectiveHCGapOperator(IDS, [], q([], "ms^-1", "effective_HC_gap_rate"),
            assumption_profile_name="test", project_evidence_tag=PROJECT,
            equal_effective_capacity=False, common_state_scale_reference=True)


def native_component():
    registry = load_core_registry(ROOT)
    p = registry.load_parameters(PROFILE, RunMode.NUMERIC_FIXTURE)
    xy = np.array([[.7, 0], [.71, 0], [.72, 0]])
    cones = population("c", "cone", ("L", "M", "L"), xy, ("center",) * 3, PROFILE, p.profile.species)
    hc = population("h", "HC", ("H1",) * 3, xy, ("center",) * 3, PROFILE, p.profile.species)
    light = q([100, 100, 100], "td", "native_retinal_illuminance")
    spec = ComponentSpec("native", KEY, {"cone": "c", "h1": "h"}, PROFILE, {
        "explicit_opt_in": True, "A": sparse.eye(3, format="csr"), "B": sparse.eye(3, format="csr"),
        "background_td": light, "pooling_policy": "pool_Vbar_before_aI", "geometry_frame": "OPL",
        "observation_interval_seconds": .001})
    return vh_component(spec, {"c": cones, "h": hc}, registry, RunMode.NUMERIC_FIXTURE), light


def test_zero_gap_wrapper_is_exact_original_rhs_and_nonzero_changes_only_Vh():
    component, light = native_component()
    ids = component.state_specs[-1].cell_ids
    blocks = [item.values.copy() for item in component.initial()]
    blocks[-1] += [1, 3, 6]
    original = component.derivative(.1, blocks, {"light": light})
    zero = rate_gap(ids, (0, 0), enabled=True)
    zero_component = with_native_hc_gap(component, zero)
    unchanged = zero_component.derivative(.1, blocks, {"light": light})
    for before, after in zip(original, unchanged):
        np.testing.assert_array_equal(before.values, after.values)
        assert (before.unit, before.meaning, before.evidence_tag) == (after.unit, after.meaning, after.evidence_tag)
    disabled_component = with_native_hc_gap(component, rate_gap(ids, enabled=False))
    for before, after in zip(original, disabled_component.derivative(.1, blocks, {"light": light})):
        np.testing.assert_array_equal(before.values, after.values)
    active = with_native_hc_gap(component, rate_gap(ids))
    changed = active.derivative(.1, blocks, {"light": light})
    for before, after in zip(original[:-1], changed[:-1]):
        np.testing.assert_array_equal(before.values, after.values)
    np.testing.assert_allclose(changed[-1].values - original[-1].values, [20, 40, -60], atol=1e-12)
    assert active.native_model.p == SOURCE_VALUES
    assert SOURCE_VALUES["tau_h"] == 20  # Source ms constant unchanged.
    assert active.native_model.model_hash != component.native_model.model_hash
    assert active.initial == component.initial and active.observe == component.observe
    assert "HC_gap_effective_rate" in active.layers
    with pytest.raises(ContractError, match="omitted HC cells"):
        with_native_hc_gap(component, rate_gap(IDS))


def test_physical_native_wrapper_uses_only_explicit_calibration_and_C():
    component, light = native_component()
    ids = component.state_specs[-1].cell_ids
    calibration = NativeHCVoltageCalibration(ids, profile_name="supplied_SI_fixture_only",
        native_reference=q([0] * 3, "vH_native_voltage", "native_relative_voltage"),
        voltage_reference=q([-.06] * 3, "V", "physical_membrane_voltage"),
        gain=q([.002, .003, .004], "V/vH_native_voltage", "native_to_voltage_gain"),
        project_evidence_tag=PROJECT)
    gap = HCGapOperator(ids, [(ids[0], ids[1])], q([1e-9], "S", "gap_junction_conductance"), enabled=True)
    C = q([10e-12, 20e-12, 30e-12], "F", "membrane_capacitance")
    with pytest.raises(ContractError, match="supplied F"):
        with_native_hc_gap(component, gap, calibration=calibration)
    wrapped = with_native_hc_gap(component, gap, calibration=calibration, capacitance=C)
    blocks = [item.values.copy() for item in component.initial()]
    blocks[-1] += [1, 3, 6]
    before = component.derivative(0, blocks, {"light": light})
    after = wrapped.derivative(0, blocks, {"light": light})
    expected = gap.native_derivative(q(blocks[-1], "vH_native_voltage", "native_relative_voltage"), calibration, C)
    np.testing.assert_allclose(after[-1].values - before[-1].values, expected.values, rtol=1e-14, atol=1e-12)
    for original, changed in zip(before[:-1], after[:-1]):
        np.testing.assert_array_equal(original.values, changed.values)


def test_zero_gap_short_DOP853_trajectory_matches_original_native_component():
    from scipy.integrate import solve_ivp

    component, _ = native_component()
    source_parameters = dict(component.native_model.p)
    ids = component.state_specs[-1].cell_ids
    wrapped = with_native_hc_gap(component, rate_gap(ids, (0, 0), enabled=True))
    initial = component.initial()
    offsets = np.r_[0, np.cumsum([item.values.size for item in initial])]
    y0 = np.concatenate([item.values for item in initial])
    light = q([110, 80, 120], "td", "native_retinal_illuminance")

    def rhs(model):
        def evaluate(t, y):
            blocks = tuple(y[left:right] for left, right in zip(offsets[:-1], offsets[1:]))
            return np.concatenate([item.values for item in model.derivative(t, blocks, {"light": light})])
        return evaluate

    options = dict(method="DOP853", rtol=1e-10, atol=1e-12, t_eval=[0, .0001, .0002])
    original = solve_ivp(rhs(component), (0, .0002), y0.copy(), **options)
    zero_gap = solve_ivp(rhs(wrapped), (0, .0002), y0.copy(), **options)
    assert original.success and zero_gap.success
    assert original.nfev == zero_gap.nfev
    assert np.all(np.isfinite(original.y)) and np.all(np.isfinite(zero_gap.y))
    np.testing.assert_array_equal(original.t, zero_gap.t)
    np.testing.assert_array_equal(original.y, zero_gap.y)
    assert not np.array_equal(original.y[:, 0], original.y[:, -1])
    assert component.native_model.p == wrapped.native_model.p == source_parameters == SOURCE_VALUES
