"""Small synthetic-only checks; no source fit, GPU, or full network run."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import math

import numpy as np
import pytest

from retina_sim import rgc_input_clamp_fit as fit
from retina_sim.functional_fit_math import (MembraneParameters, binary_history_from_past,
    teacher_forced_nll, teacher_probability)
from retina_sim.latimer2019 import (observed_history_bias, conditional_score,
    voltage_zoh, parameterized_native, reference_rgc, exact_mean_events)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native():
    reference, _ = reference_rgc(ROOT)
    return reference


def test_real_binding_only_changes_two_existing_output_parameters(native):
    _, pop = reference_rgc(ROOT)
    candidate = parameterized_native(native, pop, 170., -.066)
    assert candidate.alpha[0] == 170. and candidate.mu[0] == -.066
    for key in ("C", "gL", "EL", "EE", "EI", "V0", "beta", "history_kernel", "initial_spike_history"):
        np.testing.assert_array_equal(getattr(candidate, key), getattr(native, key))
    assert candidate.dt == native.dt == .001


def test_history_has_no_current_or_future_leakage(native):
    y = np.zeros((1, 15), dtype=int)
    baseline = observed_history_bias(native, y)
    y[0, 5] = 1
    changed = observed_history_bias(native, y)
    np.testing.assert_array_equal(changed[:, :6], baseline[:, :6])
    np.testing.assert_array_equal(changed[0, 6:9], native.history_kernel)
    expected = binary_history_from_past(y[0], [0, 0, 0], native.history_kernel)
    np.testing.assert_array_equal(changed[0], expected)


def test_native_bernoulli_nll_matches_retained_law(native):
    p = fit.parameters(native)
    y = np.array([[0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0]])
    v = np.linspace(-.072, -.061, 6)
    h = observed_history_bias(native, y)
    q = teacher_probability(v, h, p)
    idx = np.arange(6)
    explicit = -(y*np.log(q)+(1-y)*np.log1p(-q)).sum()
    stable = teacher_forced_nll(v, y, h, idx, p)
    retained = conditional_score(native, np.broadcast_to(v, y.shape), y, np.ones(6, dtype=bool))["nll"]
    assert stable == pytest.approx(explicit, rel=1e-13)
    assert stable == pytest.approx(retained, rel=1e-13)


def test_counts_above_one_are_rejected_without_clipping(native):
    with pytest.raises(ValueError, match="binary"):
        teacher_forced_nll(np.zeros(3), np.array([[0, 2, 0]]), np.zeros((1, 3)), np.arange(3), fit.parameters(native))


def test_zoh_records_bin_start_and_preserves_native_membrane(native):
    ge = np.full((1, 20), 1e-9)
    gi = np.full((1, 20), .5e-9)
    v, final = voltage_zoh(native, ge, gi)
    G = native.gL[0]+ge[0, 0]+gi[0, 0]
    equilibrium = (native.gL[0]*native.EL[0]+ge[0, 0]*native.EE[0]+gi[0, 0]*native.EI[0])/G
    expected = equilibrium+(native.V0[0]-equilibrium)*np.exp(-np.array([0., .001])*G/native.C[0])
    np.testing.assert_allclose(v[0], expected, rtol=0, atol=1e-16)
    assert final[0] == pytest.approx(equilibrium+(native.V0[0]-equilibrium)*np.exp(-.002*G/native.C[0]), abs=1e-16)
    with pytest.raises(Exception, match="nonnegative"):
        voltage_zoh(native, -ge, gi)


def test_normalized_analytic_gradient_and_extreme_bias(native, monkeypatch):
    monkeypatch.setattr(fit, "SCORE", np.arange(10, 80))
    rng = np.random.default_rng(619)
    y = (rng.random((3, 90)) < .09).astype(int)
    cases = [dict(y=y, voltage=np.linspace(-.075, -.055, 90), history=observed_history_bias(native, y))]
    base = fit.parameters(native)
    theta = np.array([.2, -.4])
    value, gradient = fit.value_gradient(theta, cases, base)
    finite_difference = []
    for k in range(2):
        step = np.zeros(2)
        step[k] = 1e-5
        finite_difference.append((fit.value_gradient(theta+step, cases, base)[0]-fit.value_gradient(theta-step, cases, base)[0])/2e-5)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-8, atol=1e-10)
    assert np.isfinite(value)
    extreme = replace(base, mu=1.)
    assert np.isfinite(teacher_forced_nll(cases[0]["voltage"], y, cases[0]["history"], fit.SCORE, extreme))


def test_free_history_is_not_observed_history(native):
    v = np.full(12, -.065)
    free = exact_mean_events(native, v)
    zero_observed = teacher_probability(v, np.zeros((1, 12)), fit.parameters(native))[0]
    assert free[0] == pytest.approx(zero_observed[0])
    assert free[1] < zero_observed[1]
    assert np.all((free >= 0) & (free <= 1))


def test_parameter_domain_is_native_anchored_and_unregularized(native):
    protocol = fit.read(ROOT/"configurations/calibration/rgc_cbem_conditional_v1.json")
    ledger = fit.parameter_ledger(native, protocol)
    assert ledger["free"][0] == dict(name="alpha", unit="Hz", initial=150., bounds=[1.5, 15000.])
    assert ledger["free"][1]["initial"] == -.07
    assert ledger["free"][1]["bounds"] == pytest.approx([-.124, .02])
    assert ledger["regularization"] == 0
    assert ledger["fixed"]["EI_multipliers"] == [1., 1.]


def test_release_is_bound_to_exact_identity(tmp_path):
    snapshot = dict(identity_sha256="frozen-identity")
    with pytest.raises(ValueError, match="release"):
        fit.check_release(None, snapshot)
    path = tmp_path/"release.json"
    fit.save(path, dict(identity_sha256="other", allow_real_optimization=True, coordinator_thread_id="parent"))
    with pytest.raises(ValueError, match="authorize"):
        fit.check_release(path, snapshot)
    fit.save(path, dict(identity_sha256="frozen-identity", allow_real_optimization=True, coordinator_thread_id="parent"))
    fit.check_release(path, snapshot)


def test_stop_guard_does_not_remove_stop(tmp_path):
    stop = tmp_path/"STOP"
    stop.write_text("keep", encoding="utf8")
    protocol = dict(resources=dict(stop_files=["STOP"]))
    with pytest.raises(RuntimeError, match="STOP"):
        fit.guard(tmp_path, tmp_path/"output", protocol)
    assert stop.read_text() == "keep"


def test_sealed_test_requires_accepted_frozen_candidate_and_is_once(tmp_path, native, monkeypatch):
    protocol, snapshot = dict(), dict(identity_sha256="identity")
    monkeypatch.setattr(fit, "verify", lambda *a: (protocol, snapshot))
    monkeypatch.setattr(fit, "check_release", lambda *a: None)
    monkeypatch.setattr(fit, "guard", lambda *a: {})
    frozen = dict(identity_sha256="identity", accepted=False,
                  candidate=dict(alpha_Hz=150., mu_V=-.07), training_constant_probability=.1)
    frozen["candidate_sha256"] = fit.digest(frozen)
    fit.save(tmp_path/"candidate_frozen.json", frozen)
    with pytest.raises(ValueError, match="not accept"):
        fit.evaluate(ROOT, Path("unused"), tmp_path, None, test=True)
    assert not (tmp_path/"test_once.json").exists()
    frozen.pop("candidate_sha256")
    frozen["accepted"] = True
    frozen["candidate_sha256"] = fit.digest(frozen)
    fit.save(tmp_path/"candidate_frozen.json", frozen)
    calls = []
    def synthetic_evaluation(*args, **kwargs):
        calls.append(args[6])
        return {"sealed_test": {"synthetic_only": True}}
    monkeypatch.setattr(fit, "evaluate_splits", synthetic_evaluation)
    assert fit.evaluate(ROOT, Path("unused"), tmp_path, None, test=True)["status"] == "TEST_RECORDED_ONCE"
    with pytest.raises(FileExistsError):
        fit.evaluate(ROOT, Path("unused"), tmp_path, None, test=True)
    assert calls == [["sealed_test"]]


def test_failed_test_attempt_remains_consumed(tmp_path, monkeypatch):
    snapshot = dict(identity_sha256="identity")
    monkeypatch.setattr(fit, "verify", lambda *a: ({}, snapshot))
    monkeypatch.setattr(fit, "check_release", lambda *a: None)
    monkeypatch.setattr(fit, "guard", lambda *a: {})
    frozen = dict(identity_sha256="identity", accepted=True,
                  candidate=dict(alpha_Hz=150., mu_V=-.07), training_constant_probability=.1)
    frozen["candidate_sha256"] = fit.digest(frozen)
    fit.save(tmp_path/"candidate_frozen.json", frozen)
    def failure(*a, **kw):
        raise ValueError("synthetic domain failure")
    monkeypatch.setattr(fit, "evaluate_splits", failure)
    with pytest.raises(ValueError, match="synthetic domain"):
        fit.evaluate(ROOT, Path("unused"), tmp_path, None, test=True)
    assert fit.read(tmp_path/"test_once.json")["status"] == "STARTED_ONCE_NO_RETRY"
    with pytest.raises(FileExistsError):
        fit.evaluate(ROOT, Path("unused"), tmp_path, None, test=True)


def test_synthetic_fit_checkpoint_freeze_and_terminal_resume_guard(tmp_path, native, monkeypatch):
    # A small artificial40-bin objective exercises optimizer plumbing only.
    protocol = fit.read(ROOT/"configurations/calibration/rgc_cbem_conditional_v1.json")
    snapshot = dict(identity_sha256="synthetic-only")
    monkeypatch.setattr(fit, "verify", lambda *a: (protocol, snapshot))
    monkeypatch.setattr(fit, "check_release", lambda *a: None)
    monkeypatch.setattr(fit, "guard", lambda *a: {})
    monkeypatch.setattr(fit, "SCORE", np.arange(5, 40))
    voltage = np.tile(np.linspace(-.075, -.052, 10), 4)
    y = np.zeros((4, 40), dtype=int)
    y[:, [7, 18, 29, 38]] = 1
    case = dict(y=y, voltage=voltage, history=observed_history_bias(native, y))
    monkeypatch.setattr(fit, "load_cases", lambda *a, **kw: [case])
    fit.save(tmp_path/"parameter_ledger.json", fit.parameter_ledger(native, protocol))
    calls = []
    def synthetic_validation(*args, **kwargs):
        calls.append((args[6], kwargs["origin"]))
        row = {"candidate": {"NLL_nats": 1.}, "native": {"NLL_nats": 2.},
               "training_constant": {"NLL_nats": 3.}, "groups": []}
        return {"train": row, "validation": row}
    monkeypatch.setattr(fit, "evaluate_splits", synthetic_validation)
    assert fit.fit(ROOT, Path("unused"), tmp_path, None)["status"] == "VALIDATION_ACCEPTED_TEST_PENDING"
    checkpoint = fit.read(tmp_path/"optimizer_checkpoint.json")
    assert checkpoint["validation_scored"] is True
    assert 2 < checkpoint["evaluations"] <= 1000
    assert checkpoint["iterations"] <= 100
    frozen = fit.read(tmp_path/"candidate_frozen.json")
    assert frozen["accepted"] is True and frozen["nonzero_update"] is True
    assert calls == [(["train", "validation"], 1), (["train", "validation"], 0)]
    assert not (tmp_path/"test_once.json").exists()
    assert fit.read(tmp_path/"evaluation_cost.json")["warm_complete_training_value_gradient_seconds"] > 0
    with pytest.raises(ValueError, match="frozen"):
        fit.fit(ROOT, Path("unused"), tmp_path, None, resume=True)
