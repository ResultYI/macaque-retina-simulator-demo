"""Independent conditional current-proxy assay; never writes production parameters."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import csv
import ctypes
import hashlib
import json
import math
import os
import shutil
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .latimer2019 import (read_cell6, paired_cases, reference_rgc, parameterized_native,
                         voltage_zoh, observed_history_bias, conditional_score,
                         exact_mean_events, MAT_SHA256)
from .functional_fit_math import (MembraneParameters, teacher_forced_nll,
                                  teacher_probability, constant_nll)

SCHEMA = "rgc_cbem_conditional_conductance_proxy_v1"
SCORE = np.arange(1000, 7000)
DEPENDENCIES = ("retina_sim/latimer2019.py", "retina_sim/functional_fit_math.py",
                "retina_sim/inner.py", "retina_sim/contracts.py",
                "retina_sim/mechanism_contracts.py", "retina_sim/rgc_input_clamp_fit.py",
                "tools/fit_rgc_input_clamp.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def parameters(native):
    return MembraneParameters.from_dict(dict(
        **{k: float(getattr(native, k)[0]) for k in ("C", "gL", "EL", "EE", "EI", "alpha", "mu", "beta")},
        history_kernel_V=native.history_kernel.tolist(), dt=native.dt))


def parameter_ledger(native, protocol):
    p = parameters(native)
    af = protocol["parameters"]["alpha_bound_native_factors"]
    margin = protocol["parameters"]["mu_bound_beta_margins"] * p.beta
    mu_bounds = [min(p.EL, p.EE, p.EI) + sum(min(x, 0.) for x in p.history_kernel_V) - margin,
                 max(p.EL, p.EE, p.EI) + sum(max(x, 0.) for x in p.history_kernel_V) + margin]
    return dict(free=[dict(name="alpha", unit="Hz", initial=p.alpha, bounds=[p.alpha*x for x in af]),
                      dict(name="mu", unit="V", initial=p.mu, bounds=mu_bounds)],
                fixed=dict(C_F=p.C, gL_S=p.gL, EL_V=p.EL, EE_V=p.EE, EI_V=p.EI,
                           V0_V=float(native.V0[0]), beta_V=p.beta, history_kernel_V=list(p.history_kernel_V),
                           initial_spike_history=native.initial_spike_history.tolist(), dt_s=p.dt,
                           EI_multipliers=[1., 1.], upstream="not instantiated"),
                initial_source="configurations/effective_dynamics_v1.json:effective_parasol_RGC_ON",
                source_class="project_approximation", bound_rationale=protocol["parameters"]["bound_rationale"],
                regularization=0., prior_status=protocol["parameters"]["prior_status"])


def model_identity(root):
    profile = read(root / "configurations/effective_dynamics_v1.json")["profiles"]["effective_parasol_RGC_ON"]
    return dict(source_sha256={p: sha(root/p) for p in DEPENDENCIES},
                selected_native_profile_sha256=digest(profile), selected_native_profile=profile,
                coupling="RGC module only; no AC/upstream graph or GPU instantiated")


def validate_protocol(protocol):
    if (protocol["protocol_id"] != SCHEMA or protocol["physical_qualification"] != "DATA_BLOCKED" or
            protocol["parameters"]["free"] != ["alpha", "mu"] or
            protocol["assumptions"]["index_origin"]["value"] != 1 or
            protocol["parameters"]["regularization_weight"] != 0 or
            protocol["split"]["sizes"] != [4, 2, 2] or
            protocol["input"]["score_bins_half_open"] != [1000, 7000] or
            protocol["input"]["sample_dt_s"] != .0001 or protocol["input"]["spike_dt_s"] != .001 or
            protocol["input"]["samples"] != 71000 or protocol["assumptions"]["baseline_offset_A"] != 0 or
            protocol["assumptions"]["current_clipping"] is not False or
            protocol["assumptions"]["exc_current_divisor_V"]["value"] != -.07 or
            protocol["assumptions"]["inh_current_divisor_V"]["value"] != .07 or
            protocol["assumptions"]["export_current_unit"]["value"] != "pA" or
            protocol["resources"]["gpu"] is not False or protocol["resources"]["threads"] != 1 or
            protocol["optimizer"]["method"] != "L-BFGS-B"):
        raise ValueError("Unsupported frozen conditional contract")


def restore_minimum(root, protocol):
    destination = root / protocol["source_directory"]
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name in protocol["restore_files"]:
        source, target = Path(protocol["backup_directory"]) / name, destination / name
        if not source.is_file():
            raise ValueError(f"Required frozen backup source missing: {source}")
        expected = sha(source)
        if target.exists():
            if sha(target) != expected:
                raise ValueError(f"Existing source differs; refusing overwrite: {target}")
            action = "REUSED_IDENTICAL"
        else:
            shutil.copyfile(source, target)
            action = "RESTORED_MINIMUM"
        if sha(target) != expected:
            raise ValueError("Restored source hash differs")
        records.append(dict(source=str(source), target=str(target), bytes=target.stat().st_size,
                            sha256=expected, action=action))
    return records


def guard(root, output, protocol):
    for path in [root / p for p in protocol["resources"]["stop_files"]] + [output / "STOP"]:
        if path.exists():
            raise RuntimeError(f"STOP present: {path}")
    resource = dict(pid=os.getpid(), parent_pid=os.getppid(), threads=1, GPU_used=False, timestamp=now())
    if os.name == "nt":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (x, ctypes.c_ulonglong) for x in ("total", "available", "page", "available_page", "virtual", "available_virtual", "extended")]
        m = Memory()
        m.length = ctypes.sizeof(m)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            raise RuntimeError("RAM protection query failed")
        resource.update(available_RAM_GiB=m.available / 2**30, system_memory_load_percent=m.load)
        if (resource["available_RAM_GiB"] < protocol["resources"]["minimum_available_RAM_GiB"] or
                m.load > protocol["resources"]["maximum_system_memory_load_percent"]):
            raise RuntimeError("RAM protection triggered")
    return resource


def prepare(root, protocol_path, output):
    """Domain checks only, including sealed records; never compute a model score."""
    root, output = Path(root), Path(output)
    protocol = read(protocol_path)
    validate_protocol(protocol)
    if (output / "snapshot_manifest.json").exists():
        verify(root, protocol_path, output)
        return read(output / "readiness.json")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite a partial preparation; use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    before = model_identity(root)
    resources = guard(root, output, protocol)
    restoration = restore_minimum(root, protocol)
    data = read_cell6(root)
    cases, groups = paired_cases(data, acknowledge_conditional_units=True, index_origin=1)
    sensitivity, _ = paired_cases(data, acknowledge_conditional_units=True, index_origin=0)
    reference, _ = reference_rgc(root)
    ledger = parameter_ledger(reference, protocol)
    missing_i = sorted((set(groups["ExcStm"]) & set(groups["SpikeStm"])) - set(groups["InhStm"]))
    if len(missing_i) != 2 or len(cases) != 8:
        raise ValueError("Expected all8 common groups and exactly2 missing I groups")
    split = {k: [c["stimulus_sha256"] for c in cases[a:b]] for k, a, b in
             (("train", 0, 4), ("validation", 4, 6), ("sealed_test", 6, 8))}
    arrays, descriptions = {}, []
    for j, (case, alternate) in enumerate(zip(cases, sensitivity)):
        if case["stimulus_sha256"] != alternate["stimulus_sha256"]:
            raise ValueError("Origin sensitivity changed group identities")
        for key in ("ge_S", "gi_S", "spike_counts", "stimulus_values"):
            value = np.asarray(case[key])
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite prepared array")
            arrays[f"g{j}_{key}"] = value
        arrays[f"g{j}_origin0_counts"] = alternate["spike_counts"]
        descriptions.append(dict(stimulus_sha256=case["stimulus_sha256"], array_prefix=f"g{j}",
            **{k: case[k] for k in ("exc_trial_indices_zero_based", "inh_trial_indices_zero_based", "spike_trial_indices_zero_based")},
            ge_range_S=[float(case["ge_S"].min()), float(case["ge_S"].max())],
            gi_range_S=[float(case["gi_S"].min()), float(case["gi_S"].max())],
            counts_shape=list(case["spike_counts"].shape), max_count_per_1ms=int(case["spike_counts"].max()),
            origin0_domain_pass=True, origin0_changed_bins=int(np.count_nonzero(case["spike_counts"] != alternate["spike_counts"])),
            finite=True, full_stimulus_exact_match=True))
    np.savez_compressed(output / "inputs.npz", **arrays)
    after = model_identity(root)
    if before != after:
        raise ValueError("Production source changed during preflight; coordinate and prepare fresh")
    source = dict(restoration=restoration, MAT_sha256=MAT_SHA256, original_current_arrays_retained_in_MAT=True,
                  original_MAT_never_modified=True, group_rows=groups, missing_inhibition_hashes=missing_i,
                  missing_I_exclusion_reason="complete cross-mode intersection, not loss or conductance selection",
                  plan=dict(library_file_id=protocol["source_plan_library_id"], access="read full529 lines successfully; no cloud path assumed"))
    contract = dict(qualification="CONDITIONAL_ENGINEERING_ONLY", physical_qualification="DATA_BLOCKED",
                    assumptions=protocol["assumptions"], hard_gaps=protocol["hard_gaps"],
                    input=protocol["input"], cases=descriptions, raw_current_units="ASSUMED pA; not verified",
                    matched_measurements="independent repeats of same stimulus; no simultaneous noise covariance",
                    input_npz_sha256=sha(output / "inputs.npz"), counts_checks_are_not_test_metrics=True)
    files = {"source_manifest.json": source, "model_identity.json": before, "input_contract.json": contract,
             "split_manifest.json": dict(splits=split, **protocol["split"]), "parameter_ledger.json": ledger,
             "protocol_frozen.json": protocol}
    for name, value in files.items():
        save(output / name, value)
    snapshot = dict(schema=SCHEMA, protocol_sha256=sha(protocol_path),
                    artifacts={name: sha(output/name) for name in files}, input_npz_sha256=sha(output/"inputs.npz"))
    snapshot["identity_sha256"] = digest(snapshot)
    save(output / "snapshot_manifest.json", snapshot)
    for name in ("validation_metrics.json", "test_metrics.json"):
        save(output/name, dict(status="SEALED_NOT_SCORED", metrics_computed=False))
    save(output/"baseline_comparison.json", dict(status="NOT_SCORED", training_constant_probability=None))
    save(output/"execution_receipt.json", dict(mode="dry-run", timestamp=now(), resources=resources,
        optimizer_evaluations=0, model_metrics_evaluations=0, full_model_runs=0, source_restore=restoration))
    readiness = dict(status="READY", execution="WAITING_FOR_PARENT_RELEASE", identity_sha256=snapshot["identity_sha256"],
                     qualification="CONDITIONAL_ENGINEERING_ONLY", physical_qualification="DATA_BLOCKED", groups=8,
                     split_sizes=[4, 2, 2], sealed_test_metrics_computed=False)
    save(output/"readiness.json", readiness)
    return readiness


def verify(root, protocol_path, output):
    protocol = read(protocol_path)
    validate_protocol(protocol)
    snapshot = read(output/"snapshot_manifest.json")
    payload = {k: v for k, v in snapshot.items() if k != "identity_sha256"}
    if digest(payload) != snapshot["identity_sha256"] or sha(protocol_path) != snapshot["protocol_sha256"]:
        raise ValueError("Frozen protocol/snapshot identity differs")
    for name, expected in snapshot["artifacts"].items():
        if sha(output/name) != expected:
            raise ValueError(f"Frozen artifact identity differs: {name}")
    if sha(output/"inputs.npz") != snapshot["input_npz_sha256"]:
        raise ValueError("Frozen input identity differs")
    if model_identity(root) != read(output/"model_identity.json"):
        raise ValueError("Current native/model code differs; coordinator must prepare a fresh contract")
    # Validate the raw active MAT as well as the frozen transformed arrays.
    source_path = root/protocol["source_directory"]/"Cell_6.mat"
    if sha(source_path) != MAT_SHA256:
        raise ValueError("Active raw MAT identity differs")
    for restored in read(output/"source_manifest.json")["restoration"]:
        if sha(restored["target"]) != restored["sha256"]:
            raise ValueError("Restored source/metadata identity differs")
    return protocol, snapshot


def check_release(path, snapshot):
    if not path:
        raise ValueError("Parent coordinator release is required before real fitting/evaluation")
    release = read(path)
    if (release.get("identity_sha256") != snapshot["identity_sha256"] or
            release.get("allow_real_optimization") is not True or not release.get("coordinator_thread_id")):
        raise ValueError("Coordinator release does not authorize this frozen identity")


def load_cases(output, native, split_names, origin=1):
    splits = read(output/"split_manifest.json")["splits"]
    admitted = {s: set(splits[s]) for s in split_names}
    result = []
    with np.load(output/"inputs.npz", allow_pickle=False) as data:
        for row in read(output/"input_contract.json")["cases"]:
            group = row["stimulus_sha256"]
            split = next((s for s in split_names if group in admitted[s]), None)
            if split is None:
                continue
            prefix = row["array_prefix"]
            y = data[prefix + ("_spike_counts" if origin == 1 else "_origin0_counts")].copy()
            ge, gi = (data[prefix+"_"+k].copy() for k in ("ge_S", "gi_S"))
            if (y.ndim != 2 or not np.isin(y, (0, 1)).all() or ge.shape != (71000,) or
                    gi.shape != ge.shape or not np.isfinite([ge, gi]).all() or min(ge.min(), gi.min()) < 0):
                raise ValueError("Prepared finite/nonnegative/binary input domain differs")
            v, _ = voltage_zoh(native, ge[None, :], gi[None, :])
            if v.shape != (1, 7100) or not np.isfinite(v).all():
                raise ValueError("Invalid native ZOH trajectory")
            result.append(dict(group=group, prefix=prefix, split=split, y=y, voltage=v[0],
                               history=observed_history_bias(native, y)))
    return result


def value_gradient(theta, cases, base):
    """Exact retained Bernoulli NLL and gradient in frozen normalized coordinates."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2,) or not np.isfinite(theta).all():
        raise ValueError("Finite two-parameter vector required")
    p = replace(base, alpha=base.alpha*math.exp(float(theta[0])), mu=base.mu+base.beta*float(theta[1]))
    total, gradient, bins = 0., np.zeros(2), 0
    for case in cases:
        y = case["y"][:, SCORE]
        h = case["history"][:, SCORE]
        z = (case["voltage"][SCORE][None, :]+h-p.mu)/p.beta
        a = p.alpha*p.dt*np.logaddexp(0., z)
        probability = -np.expm1(-a)
        if (not np.isfinite(a).all() or np.any(a <= 0) or
                not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))):
            raise ValueError("Candidate hazard/probability domain error; no clipping")
        total += teacher_forced_nll(case["voltage"], case["y"], case["history"], SCORE, p)
        derivative = np.ones_like(a)
        event = y == 1
        derivative[event] = -np.exp(-a[event])/probability[event]
        gradient += [np.sum(derivative*a), np.sum(derivative*(-p.alpha*p.dt*expit(z)))]
        bins += y.size
    if not bins or not np.isfinite(total) or not np.isfinite(gradient).all():
        raise ValueError("Empty/nonfinite objective; candidate stopped")
    return total/bins, gradient/bins


def fit(root, protocol_path, output, release_path, *, resume=False):
    protocol, snapshot = verify(root, protocol_path, output)
    check_release(release_path, snapshot)
    if (output/"test_once.json").exists() or (output/"candidate_frozen.json").exists():
        raise ValueError("Candidate/test is frozen; no further optimization in this run")
    if (output/"optimizer_result.json").exists() or (output/"optimizer_blocked.json").exists():
        raise ValueError("Candidate already reached a terminal result/domain failure; no implicit retry")
    reference, pop = reference_rgc(root)
    base = parameters(reference)
    checkpoint_path = output/"optimizer_checkpoint.json"
    if resume and not checkpoint_path.exists():
        raise ValueError("No accepted-iterate checkpoint to resume")
    if not resume and checkpoint_path.exists():
        raise ValueError("Existing optimizer checkpoint; use resume")
    checkpoint = read(checkpoint_path) if resume else dict(theta=[0., 0.], iterations=0, evaluations=0,
        identity_sha256=snapshot["identity_sha256"], validation_scored=False)
    if checkpoint["identity_sha256"] != snapshot["identity_sha256"] or checkpoint.get("validation_scored"):
        raise ValueError("Checkpoint identity differs or validation already read; no selection retry")
    guard(root, output, protocol)
    started = time.perf_counter()
    cases = load_cases(output, reference, ["train"])
    prep_seconds = time.perf_counter()-started
    p_constant = sum(int(c["y"][:, SCORE].sum()) for c in cases)/sum(c["y"][:, SCORE].size for c in cases)
    ledger = read(output/"parameter_ledger.json")["free"]
    bounds = [(math.log(x/base.alpha)) for x in ledger[0]["bounds"]], [(x-base.mu)/base.beta for x in ledger[1]["bounds"]]
    options = protocol["optimizer"]
    counts = dict(iterations=checkpoint["iterations"], evaluations=checkpoint["evaluations"])
    if counts["iterations"] >= options["maxiter"] or counts["evaluations"] >= options["maxfun"]:
        raise ValueError("Frozen global optimization budget exhausted")
    if counts["evaluations"]+2 >= options["maxfun"]:
        raise ValueError("Insufficient remaining evaluation budget for deterministic cost probes")
    trace_path = output/"optimizer_trace.csv"
    warm_started = time.perf_counter()
    initial_value, initial_grad = value_gradient(checkpoint["theta"], cases, base)
    cold_seconds = time.perf_counter()-warm_started
    warm_started = time.perf_counter()
    repeat_value, repeat_grad = value_gradient(checkpoint["theta"], cases, base)
    warm_seconds = time.perf_counter()-warm_started
    if initial_value != repeat_value or not np.array_equal(initial_grad, repeat_grad):
        save(output/"optimizer_blocked.json", dict(status="DETERMINISM_ERROR_STOPPED", timestamp=now()))
        raise ValueError("Repeated deterministic objective differs")
    counts["evaluations"] += 2
    save(output/"evaluation_cost.json", dict(voltage_and_training_load_seconds=prep_seconds,
        cold_complete_training_value_gradient_seconds=cold_seconds,
        warm_complete_training_value_gradient_seconds=warm_seconds,
        maxfun=options["maxfun"], objective_upper_projection_seconds=warm_seconds*options["maxfun"],
        optimizer_wall_limit="none; finite evaluations/iterations and resource guard", timestamp=now()))
    def persist(theta):
        save(checkpoint_path, dict(theta=np.asarray(theta).tolist(), **counts,
             identity_sha256=snapshot["identity_sha256"], validation_scored=False,
             resume_semantics=options["resume"]))
    persist(checkpoint["theta"])
    def objective(theta):
        guard(root, output, protocol)
        if counts["evaluations"] >= options["maxfun"]:
            raise ValueError("Frozen global evaluation budget exhausted")
        start = time.perf_counter()
        counts["evaluations"] += 1
        # Preserve the last accepted iterate while charging every attempted evaluation.
        saved = read(checkpoint_path)
        saved.update(counts)
        save(checkpoint_path, saved)
        try:
            value, gradient = value_gradient(theta, cases, base)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            save(output/"optimizer_blocked.json", dict(status="DOMAIN_ERROR_STOPPED", reason=str(exc),
                 theta=np.asarray(theta).tolist(), **counts))
            raise
        new = not trace_path.exists()
        with trace_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if new:
                writer.writerow(["evaluation", "alpha_Hz", "mu_V", "training_NLL_per_bin", "gradient_logalpha", "gradient_mu_over_beta", "seconds"])
            writer.writerow([counts["evaluations"], base.alpha*math.exp(theta[0]), base.mu+base.beta*theta[1],
                             value, *gradient, time.perf_counter()-start])
        return value, gradient
    def callback(theta):
        counts["iterations"] += 1
        persist(theta)
    result = minimize(objective, checkpoint["theta"], method="L-BFGS-B", jac=True, bounds=bounds,
        callback=callback, options={k: options[k] for k in ("maxls", "ftol", "gtol")} |
        dict(maxiter=options["maxiter"]-counts["iterations"], maxfun=options["maxfun"]-counts["evaluations"]))
    persist(result.x)
    candidate = dict(alpha_Hz=base.alpha*math.exp(result.x[0]), mu_V=base.mu+base.beta*result.x[1])
    changed = candidate["alpha_Hz"] != base.alpha or candidate["mu_V"] != base.mu
    save(output/"optimizer_result.json", dict(success=bool(result.success), message=str(result.message),
         **counts, theta=result.x.tolist(), candidate=candidate, nonzero_update=changed,
         training_constant_probability=p_constant, objective=float(result.fun)))
    if not result.success:
        save(output/"failure_or_completion.json", dict(status="NOT_CONVERGED", message=str(result.message)))
        return dict(status="NOT_CONVERGED")
    native = parameterized_native(reference, pop, candidate["alpha_Hz"], candidate["mu_V"])
    # Mark the one final validation attempt before reading any validation score.
    checkpoint = read(checkpoint_path)
    checkpoint["validation_scored"] = True
    save(checkpoint_path, checkpoint)
    metrics = evaluate_splits(root, output, protocol, native, reference, p_constant, ["train", "validation"], origin=1)
    validation = metrics["validation"]
    accepted = bool(changed and validation["candidate"]["NLL_nats"] <= validation["native"]["NLL_nats"] and
                    validation["candidate"]["NLL_nats"] < validation["training_constant"]["NLL_nats"])
    save(output/"validation_metrics.json", metrics)
    save(output/"baseline_comparison.json", dict(training_constant_probability=p_constant, origin=1,
          metrics={s: {k: v for k, v in m.items() if k != "groups"} for s, m in metrics.items()}))
    frozen = dict(identity_sha256=snapshot["identity_sha256"], candidate=candidate,
                  training_constant_probability=p_constant, accepted=accepted, nonzero_update=changed,
                  qualification="CONDITIONAL_ENGINEERING_ONLY", physical_qualification="DATA_BLOCKED", frozen_at=now())
    frozen["candidate_sha256"] = digest(frozen)
    save(output/"candidate_frozen.json", frozen)
    sensitivity = evaluate_splits(root, output, protocol, native, reference, p_constant, ["train", "validation"], origin=0)
    save(output/"origin0_sensitivity.json", dict(policy="fixed candidate; no refit, no origin selection, no test",
                                               metrics=sensitivity, candidate_sha256=frozen["candidate_sha256"]))
    save(output/"failure_or_completion.json", dict(status="VALIDATION_ACCEPTED_TEST_PENDING" if accepted else
         "VALIDATION_NOT_ACCEPTED", changed=changed, production_writeback=False, timestamp=now()))
    verify(root, protocol_path, output)
    return dict(status="VALIDATION_ACCEPTED_TEST_PENDING" if accepted else "VALIDATION_NOT_ACCEPTED")


def evaluate_splits(root, output, protocol, native, reference, p_constant, split_names, *, origin):
    p = parameters(native)
    cases = load_cases(output, native, split_names, origin)
    result, predictions = {}, {}
    for split in split_names:
        rows, totals = [], {k: 0. for k in ("candidate", "native", "training_constant")}
        bins = events = 0
        for case in (c for c in cases if c["split"] == split):
            guard(root, output, protocol)
            y, v, h = case["y"], case["voltage"], case["history"]
            probability = teacher_probability(v, h, p)
            if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
                raise ValueError("Nonfinite/out-of-domain prediction")
            mask = np.zeros(len(v), dtype=bool)
            mask[SCORE] = True
            score = conditional_score(native, np.broadcast_to(v, y.shape), y, mask)
            nll = teacher_forced_nll(v, y, h, SCORE, p)
            if not np.isfinite(score["nll"]) or not np.isclose(score["nll"], nll, rtol=1e-12, atol=1e-8):
                raise ValueError("Retained conditional_score and stable native NLL differ")
            null_nll = constant_nll(p_constant, y[:, SCORE])
            native_nll = teacher_forced_nll(v, y, h, SCORE, parameters(reference))
            if not np.isfinite([null_nll, native_nll, nll]).all():
                raise ValueError("Nonfinite candidate or baseline score")
            free = exact_mean_events(native, v)
            if not np.isfinite(free).all() or np.any((free < 0) | (free > 1)):
                raise ValueError("Free-history probability domain error; no clipping")
            tfmean = probability.mean(axis=0)
            width = int(round(protocol["evaluation"]["psth_bin_s"]/native.dt))
            indices = SCORE.reshape(-1, width)
            observed_psth = y[:, indices].mean(axis=(0, 2))/native.dt
            tfpsth, freepsth = (x[indices].mean(axis=1)/native.dt for x in (tfmean, free))
            native_probability = teacher_probability(v, h, parameters(reference))
            native_free = exact_mean_events(reference, v)
            native_tf_psth = native_probability.mean(axis=0)[indices].mean(axis=1)/native.dt
            native_free_psth = native_free[indices].mean(axis=1)/native.dt
            reliability = []
            q, observed = probability[:, SCORE].ravel(), y[:, SCORE].ravel()
            nbins = protocol["evaluation"]["probability_bins"]
            for k in range(nbins):
                selected = (q >= k/nbins) & (q < (k+1)/nbins if k < nbins-1 else q <= 1)
                reliability.append(dict(lo=k/nbins, hi=(k+1)/nbins, bins=int(selected.sum()),
                    mean_probability=float(q[selected].mean()) if selected.any() else None,
                    observed_frequency=float(observed[selected].mean()) if selected.any() else None))
            rows.append(dict(stimulus_sha256=case["group"], NLL_nats=nll, events=int(y[:, SCORE].sum()),
                bins=int(y[:, SCORE].size), Brier=float(np.mean((q-observed)**2)), reliability=reliability,
                teacher_forced_PSTH_RMSE_Hz=float(np.sqrt(np.mean((tfpsth-observed_psth)**2))),
                free_history_PSTH_RMSE_Hz=float(np.sqrt(np.mean((freepsth-observed_psth)**2))),
                native_teacher_forced_PSTH_RMSE_Hz=float(np.sqrt(np.mean((native_tf_psth-observed_psth)**2))),
                native_free_history_PSTH_RMSE_Hz=float(np.sqrt(np.mean((native_free_psth-observed_psth)**2))),
                training_constant_PSTH_RMSE_Hz=float(np.sqrt(np.mean((p_constant/native.dt-observed_psth)**2))),
                native_Brier=float(np.mean((native_probability[:, SCORE]-y[:, SCORE])**2)),
                training_constant_Brier=float(np.mean((p_constant-y[:, SCORE])**2)),
                observed_rate_Hz=float(y[:, SCORE].mean()/native.dt),
                free_history_initialization="native zero history at source t0; full prefix propagated; exact8 states",
                predicted_teacher_forced_rate_Hz=float(tfmean[SCORE].mean()/native.dt),
                predicted_free_history_rate_Hz=float(free[SCORE].mean()/native.dt)))
            prefix = case["prefix"]
            for key, array in dict(voltage_V=v, teacher_forced_probability=probability,
                    free_history_probability=free, observed_PSTH_Hz=observed_psth,
                    teacher_forced_PSTH_Hz=tfpsth, free_history_PSTH_Hz=freepsth,
                    native_teacher_forced_probability=native_probability, native_free_history_probability=native_free,
                    native_teacher_forced_PSTH_Hz=native_tf_psth, native_free_history_PSTH_Hz=native_free_psth,
                    training_constant_probability=np.array(p_constant),
                    PSTH_time_s=(indices[:, 0]+width/2)*native.dt).items():
                predictions[prefix+"_"+key] = array
            for k, value in (("candidate", nll), ("native", native_nll), ("training_constant", null_nll)):
                totals[k] += value
            bins += y[:, SCORE].size
            events += int(y[:, SCORE].sum())
        result[split] = {k: dict(NLL_nats=v, NLL_nats_per_bin=v/bins,
             bits_per_spike_improvement_over_training_constant=(totals["training_constant"]-v)/(events*math.log(2)) if events else None)
                         for k, v in totals.items()}
        result[split].update(groups=rows, bins=bins, events=events, origin=origin)
    np.savez_compressed(output/f"predictions_origin{origin}_{'_'.join(split_names)}.npz", **predictions)
    return result


def evaluate(root, protocol_path, output, release_path, *, test=False):
    protocol, snapshot = verify(root, protocol_path, output)
    check_release(release_path, snapshot)
    frozen = read(output/"candidate_frozen.json")
    if digest({k: v for k, v in frozen.items() if k != "candidate_sha256"}) != frozen["candidate_sha256"]:
        raise ValueError("Frozen candidate identity differs")
    if frozen["identity_sha256"] != snapshot["identity_sha256"]:
        raise ValueError("Frozen candidate belongs to another source/model/split")
    guard(root, output, protocol)
    if test:
        if not frozen["accepted"]:
            raise ValueError("Validation did not accept this candidate; test remains sealed")
        # Exclusive once marker precedes test reads. A crash consumes this attempt.
        with (output/"test_once.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(status="STARTED_ONCE_NO_RETRY", candidate_sha256=frozen["candidate_sha256"], timestamp=now()), stream)
    reference, pop = reference_rgc(root)
    candidate = frozen["candidate"]
    native = parameterized_native(reference, pop, candidate["alpha_Hz"], candidate["mu_V"])
    splits = ["sealed_test"] if test else ["train", "validation"]
    metrics = evaluate_splits(root, output, protocol, native, reference,
                              frozen["training_constant_probability"], splits, origin=1)
    verify(root, protocol_path, output)
    save(output/("test_metrics.json" if test else "reevaluation_metrics.json"),
         dict(candidate_sha256=frozen["candidate_sha256"], metrics=metrics, timestamp=now()))
    if test:
        save(output/"test_once.json", dict(status="COMPLETED_ONCE", candidate_sha256=frozen["candidate_sha256"], timestamp=now()))
        save(output/"failure_or_completion.json", dict(status="CONDITIONAL_EXPERIMENT_TEST_RECORDED",
            physical_qualification="DATA_BLOCKED", production_writeback=False))
    return dict(status="TEST_RECORDED_ONCE" if test else "FROZEN_CANDIDATE_EVALUATED")
