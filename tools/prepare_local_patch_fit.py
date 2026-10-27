"""One engineering-only Local Patch forward and retained NLL interface probe.

No scientific split, parameter family, regularizer or optimizer is invented.
The synthetic loader fixture is explicitly unrelated to recording 67#33.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRIOR = ROOT/'reports/local_patch_v2/repair_once_v1/retry_execution_v1'

from tools import run_local_patch_v2 as launch
from tools.run_local_patch import read, write, sha, replace_ledger


def fixture(directory):
    """Exercise the existing prepared-source loader using named test data."""
    import numpy as np
    from retina_sim.natural_movie_supervision import bin_live_trial, load_prepared_real_trial
    directory.mkdir()
    ticks = np.array([-20, 10], dtype=np.int64)
    counts = bin_live_trial(ticks, 60)
    np.savez(directory/'fixture_arrays.npz', counts=counts, ticks=ticks)
    record_id = 'engineering_fixture_NOT_67_33'
    write(directory/'source_spike_index.json', dict(
        engineering_only=True, synthetic_test_fixture=True,
        arrays=dict(path='fixture_arrays.npz', sha256=sha(directory/'fixture_arrays.npz')),
        records=[dict(record_id=record_id, exact_duplicate_of_record=None,
            source_cell_key=record_id, model_family_candidate='parasol_RGC_OFF',
            project_RF_binding=None, RF_source='NONE_SYNTHETIC_TEST_FIXTURE',
            underlying_movie_id='NO_REAL_MOVIE_PAIRED', trials=[dict(
                is_live_movie_repeat=True, column_zero_based=0, count_key='counts',
                raw_ticks_key='ticks', live_events=1)])]))
    loaded = load_prepared_real_trial(directory, record_id, 0)
    assert loaded['counts_1ms'].shape == (60000,)
    np.testing.assert_array_equal(loaded['counts_1ms'][:3], [0, 1, 0])
    np.testing.assert_array_equal(loaded['bin_start_source_ticks_100us'][:3], [0, 10, 20])
    np.testing.assert_array_equal(loaded['movie_frame_zero_based'][:3], [751, 751, 751])
    return loaded


def target_voltage(case):
    """Select exact native bin starts by original target row, never interpolate."""
    import numpy as np
    graph = case['graph']
    pop, local = graph.target_population, graph.target_local_row
    assert pop == 'parasol_RGC_OFF'
    assert graph.populations[pop].counts[local] == 1
    start = case['seed']['time_seconds']
    times = start + np.arange(3)*.001
    values = []
    for stamp in times:
        matches = [(f, full) for t, (f, full) in case['frames'].items() if abs(t-stamp) < 1e-12]
        if len(matches) != 1:
            raise ValueError('Exact native bin-start observation missing or ambiguous')
        frame, full = matches[0]
        if frame['side'] != 'right':
            raise ValueError('Original post-commit snapshot required')
        rows = graph.parent_population_rows[pop] if full else frame['identities'][pop]['parent_group_rows']
        chosen = np.flatnonzero(rows == graph.target_parent_row)
        if len(chosen) != 1:
            raise ValueError('Unique original singleton target absent')
        values.append(float(frame['states'][pop, 'RGC_spike_history', 'V'][chosen[0]]))
    if times[-1]+.001 > case['result']['final_time_seconds']+1e-12:
        raise ValueError('Scoring exceeds complete original native-bin support')
    return times, np.asarray(values)


def loss_probe(stage):
    import numpy as np
    from retina_sim.contracts import canonical_hash
    from retina_sim.functional_fit_math import MembraneParameters, binary_history, teacher_forced_nll
    from tools.compare_local_patch_v2 import load_case
    loaded = fixture(stage/'synthetic_loader_fixture')
    case = load_case(stage, 'halo_030')
    times, voltage = target_voltage(case)
    p = case['seed']['inner_parameters']['populations']['parasol_RGC_OFF']
    parameters = MembraneParameters.from_dict(dict(p, history_kernel_V=p['spike_history_kernel']))
    y = loaded['counts_1ms'][:3][None, :]
    history = binary_history(y[0], loaded['raw_context_ticks_100us'], parameters.history_kernel_V)[None, :]
    indices = np.arange(3, dtype=np.int64)
    np.testing.assert_array_equal(history[0], [parameters.history_kernel_V[1],
        parameters.history_kernel_V[2], parameters.history_kernel_V[0]])
    loss = teacher_forced_nll(voltage, y, history, indices, parameters)
    assert np.isfinite(loss)
    # An independent direct Bernoulli calculation checks units/bin exposure.
    hazard = parameters.alpha*np.logaddexp(0., (voltage[None, :]+history-parameters.mu)/parameters.beta)
    exposure = .001*hazard
    direct = float(np.sum(np.where(y, -np.log(-np.expm1(-exposure)), exposure)))
    np.testing.assert_allclose(loss, direct, rtol=1e-13, atol=1e-13)
    assert loss == teacher_forced_nll(voltage, y, history, indices, parameters)
    invalid = y.copy(); invalid[0, 0] = 2
    try:
        teacher_forced_nll(voltage, invalid, history, indices, parameters)
    except ValueError:
        rejects_multicount = True
    else:
        raise AssertionError('Counts greater than one must never be clipped')
    prior = load_case(PRIOR, 'halo_030')
    old_times, old_voltage = target_voltage(prior)
    assert np.array_equal(times, old_times)
    reproducible = np.array_equal(voltage, old_voltage)
    # A failed short reproducibility check cannot authorize a longer run.
    if not reproducible:
        raise AssertionError('Native target voltage differs from saved same-IVP short run')
    np.savez(stage/'loss_inputs.npz', absolute_bin_start_seconds=times,
        voltage_V=voltage, synthetic_counts=y, strict_past_history_V=history, scored_indices=indices)
    result = dict(engineering_only=True, status='PASS_RETAINED_NLL_PRIMITIVE_ONLY',
        loader='natural_movie_supervision.load_prepared_real_trial',
        loss='functional_fit_math.teacher_forced_nll', loss_value=loss,
        independent_Bernoulli_value=direct, repeated_loss_exact=True,
        independent_process_prior_voltage_exact=reproducible,
        prior_voltage_sha256=canonical_hash(old_voltage), voltage_sha256=canonical_hash(voltage),
        shapes=dict(voltage=list(voltage.shape), observations=list(y.shape), history=list(history.shape)),
        units=dict(raw_ticks='100us', bin_width_seconds=.001, voltage='V', conductance='S', loss='dimensionless NLL sum'),
        synthetic_fixture=True, synthetic_record_id=loaded['record_id'],
        actual_67_33_data_used=False, optical_or_physical_alignment_claimed=False,
        loader_frame_origin=751, forward_frames=[748,749],
        fixture_clock_offset_seconds=float(times[0]),
        clock_offset_scope='Explicit test-only relative-bin placement; never a real movie/spike offset',
        loss_support_absolute_seconds=[float(times[0]),float(times[-1]+.001)],
        loss_indices_are_test_fixture_only=True, scientific_train_split=None,
        rejects_multicount=rejects_multicount, optimizer_steps=0,
        optimizer_status='BLOCKED_MISSING_APPROVED_REAL_DATA_PARAMETER_SPLIT_REGULARIZER_OPTIMIZER_PROTOCOL',
        parameter_updates=0, full_original_objective_evaluated=False,
        source_parameter_sha256=canonical_hash(case['seed']['inner_parameters']), fit_eligible=False)
    write(stage/'ENGINEERING_PIPELINE.json', result)
    return result


def monitor_diagnosis(stage):
    import numpy as np
    from retina_sim.group_time_storage import load_bundle
    from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
    from retina_sim.group_time_monitor import frozen_budget
    old = read(PRIOR/'run_halo_030/result.json')['absolute_monitor']
    directory = PRIOR/'run_halo_030'
    seed = load_bundle(directory/'construction_seed', expected_kind='source_construction_seed')['payload']
    snap = load_bundle(directory/'final_snapshot', expected_kind='validation_evidence')['payload']
    initial = seed['states']['cones','analytic_RE','R']
    _, denominator, _ = frozen_budget(initial, 'td')
    t0, edge, end = seed['time_seconds'], 749/150, snap['time_seconds']
    anchor = AnalyticREAnchor(t0, initial, seed['states']['cones','analytic_RE','E'],
        seed['native_inputs'][748], edge, 748)
    r_edge, e_edge = anchor.at(edge)
    next_anchor = AnalyticREAnchor(edge,r_edge,e_edge,seed['native_inputs'][749],750/150,749)
    expected_R, expected_E = next_anchor.at(end)
    np.testing.assert_array_equal(expected_R, snap['states']['cones','analytic_RE','R'])
    np.testing.assert_array_equal(expected_E, snap['states']['cones','analytic_RE','E'])
    assert np.array_equal(initial, seed['input_reference']['native_background_td'])
    initial_norms = .0031*np.abs(anchor.rhs(t0)['R'])/denominator
    edge_norms = .0031*np.abs(next_anchor.rhs(edge)['R'])/denominator
    norms = np.maximum(initial_norms, edge_norms)
    row = old['dominant']['cell_index']
    np.testing.assert_allclose(norms[row],old['maximum_norm'],rtol=1e-12)
    failed = [s for s in old['states'] if max(s['maximum_observed_excursion_norm'],
        s['original_rhs_norm'],s['discrete_defect_norm']) > 1]
    evidence = dict(engineering_only=True, status='EXPLAINED_NO_CONFIRMED_IMPLEMENTATION_BUG',
        meaning='Fixed-point excursion, width*original RHS and width*pure-map defect, each divided by frozen unit denominator; not an integration-error estimate',
        formulas=dict(denominator='unit_atol + 1e-4*max(abs(initial), unit_floor)',
            continuous='1000*(light_td-R_td)/source_tau_R_ms'),
        passed=old['passed'],maximum_norm=old['maximum_norm'],dominant=old['dominant'],
        domain_failures=old['domain_failures'],missing_scales=old['missing_scales'],failed_states=failed,
        dominant_witness=dict(local_cone_row=row, frozen_R_td=float(initial[row]),
            frozen_denominator_td=float(denominator[row]),frame748_td=float(seed['native_inputs'][748][row]),
            frame749_td=float(seed['native_inputs'][749][row]),edge_seconds=edge,
            initial_RHS_td_per_s=float(anchor.rhs(t0)['R'][row]),
            initial_norm=float(initial_norms[row]),edge_norm=float(edge_norms[row]),
            edge_R_td=float(r_edge[row]),edge_RHS_td_per_s=float(next_anchor.rhs(edge)['R'][row]),
            reproduced_norm=float(norms[row]),analytic_final_RE_exact=True),
        reasons=['Initial R equals declared background, whereas frozen engineering frame748 already differs; frame749 changes it again during fixed-point monitoring',
            'Other construction states are locally referenced, not coupled-equilibrium states',
            'Legal original zero-history delta law evolves under nonzero OFF-parasol rate'],
        initializer_compatibility=dict(background_py='prepare_background requires compiled JointNetwork/HybridRunner, complete timing/history monitor and constant channels; no GroupTimeArm adapter',
            initialization_py='Local release/event reference balances only',
            retained_background_py='Retained-circuit closure; no qualified grouped-patch state/context source'),
        background_status='BLOCKED_NO_SOURCE_COMPATIBLE_QUALIFIED_GROUP_BACKGROUND',
        warmup_runs=0,scales_changed=False,reference_changed=False,populations_excluded=False,
        integration_error_claim='Dominant analytic RE discrepancy is exact physical input response. Other time-discretization errors remain unestimated; prior spatial pair is separate.')
    write(stage/'MONITOR_DIAGNOSIS.json', evidence)


def execute(stage):
    started = time.time()
    if stage.exists() or not stage.parent.is_dir() or not stage.is_relative_to(ROOT/'reports'):
        raise ValueError('Fresh report stage under reports required')
    old = read(PRIOR/'ledger.json')
    assert old['active_worker'] is None and (PRIOR/'REQUEST_STOP').exists()
    stage.mkdir()
    plan = dict(engineering_only=True, one_forward_seconds=.0031, graph='halo_030',
        initial_condition='Unchanged legal frozen construction IVP',
        window_qualification='Only previously spatial-tested 4.993..4.9961, 3 complete native bins; insufficient for physiological fit',
        input='Existing frozen native cone td / rod R*/s engineering input; no optical contrast claim',
        target='original-row:25496 singleton, not biological67#33',
        completion='One forward plus retained NLL fixture shape/unit/time/reproducibility checks; or preserved hard failure',
        failure='No long run after short failure; no automatic retry',
        full_original_objective=False, optimizer_steps=0,
        missing=['real67#33 data package','qualified grouped background','2D RF/optics and movie timing binding',
            'approved real-data parameter family/shared rules/split/regularizer/optimizer protocol'],
        task_time_quota=None,role_time_quota=None,historical_stages_read_only=True)
    write(stage/'PLAN.json',plan)
    monitor_diagnosis(stage)
    shutil.copytree(PRIOR/'graph_halo_030',stage/'graph_halo_030')
    auth = deepcopy(read(PRIOR/'AUTHORIZATION.json'))
    auth.update(schema='local-patch-fit-preparation-engineering-1',stage_token=uuid.uuid4().hex,
        authorized_by='Current delegated user-approved fitting preparation; one engineering-only forward/NLL probe',
        created_unix=started,deadline_unix=None,total_worker_limit_seconds=None,
        preparation_limit_seconds=None,task_wall_limit_removed_by_user=True,
        task_cumulative_limit_removed_by_user=True,previous_stage=str(PRIOR),
        plan=dict(duration_seconds=.0031),configs=dict(halo_030='configurations/local_patch_v2/halo_030.json'))
    auth.pop('role_caps_seconds',None)
    auth['source_pins'][Path(__file__).relative_to(ROOT).as_posix()] = sha(__file__)
    auth['source_pins']['retina_sim/functional_fit_math.py'] = sha(ROOT/'retina_sim/functional_fit_math.py')
    auth['source_pins']['retina_sim/natural_movie_supervision.py'] = sha(ROOT/'retina_sim/natural_movie_supervision.py')
    write(stage/'AUTHORIZATION.json',auth)
    ledger = dict(status='PREPARED',used_worker_seconds=0.,workers=[],active_worker=None,
        historical_lease_reused=False,prior_stage_ledger_sha256=sha(PRIOR/'ledger.json'))
    write(stage/'ledger.json',ledger)
    try:
        launch.launch(stage,auth,ledger,'engineering_forward','halo_030')
        launch.Guard(stage,auth)()
        loss_probe(stage)
        ledger['status']='ENGINEERING_NLL_PRIMITIVE_PASS_SCIENTIFIC_FIT_BLOCKED'
    except BaseException as exc:
        ledger.update(status='FAILED_PRESERVED',error=repr(exc))
        raise
    finally:
        ledger['task_wall_seconds']=time.time()-started
        ledger['model_worker_launch_count']=len(ledger['workers'])
        replace_ledger(stage/'ledger.json',ledger)
        write(stage/'REQUEST_STOP',dict(stage_closed=True,status=ledger['status'],further_model_runs_allowed=False))
    print(json.dumps(dict(status=ledger['status'],stage=str(stage),worker_seconds=ledger['used_worker_seconds'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,required=True)
    stage = parser.parse_args().stage.resolve()
    new_stage_requested = not stage.exists()
    try:
        execute(stage)
    except BaseException as exc:
        if new_stage_requested and stage.is_relative_to(ROOT/'reports') and stage.exists() and not (stage/'REQUEST_STOP').exists():
            write(stage/'PRELAUNCH_FAILURE.json',dict(status='FAILED_PRESERVED',error=repr(exc),
                model_worker_launch_count=0,model_advanced=False))
            write(stage/'REQUEST_STOP',dict(stage_closed=True,further_model_runs_allowed=False))
        raise
