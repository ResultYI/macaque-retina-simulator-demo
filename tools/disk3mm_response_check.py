"""Matched 20 ms check after the existing numerical background preparation."""
from dataclasses import replace
from pathlib import Path
import json
import time
import numpy as np
from retina_sim.contracts import ContractError, SolverConfig, canonical_hash
from retina_sim.background import constant_background_channels
from retina_sim.background_checkpoint import MemoryGuard, prepare_with_checkpoint
from retina_sim.recording import RecordingPlan, NPZWindowRecorder
from retina_sim.execution_backends import make_runner, verify_cuda_execution, CUDA_BACKENDS
from retina_sim.background_advance import choose_background_runner
from retina_sim.recording_indices import indexed_recording_plan
from retina_sim.disk_graph_storage import save_json


def aligned_response_times(runner, grid_origin, origin, duration, sample_seconds=.0001):
    """Keep prepared endpoints and choose the existing absolute clock stamps.

    Adding an accumulated preparation origin to a relative grid can put a
    readout about1e-13s before the event whose history update already occurred.
    At a delay-pruning boundary that older bracket is correctly unavailable.
    Match the background/GUI clock representation rather than clipping a delay
    query or substituting a background value. State, clocks and RNG are untouched.
    """
    intervals=int(round(duration/sample_seconds))
    if intervals<1 or abs(intervals*sample_seconds-duration)>1e-12:
        raise ContractError('Matched response duration must contain complete sampling intervals')
    nominal=origin+np.arange(intervals+1,dtype=float)*sample_seconds
    times=nominal.copy()
    periods=list(runner.periods.values())+[runner.observation_dt,runner.delayed_step]
    for index,t in enumerate(nominal[1:-1],start=1):
        candidates=[grid_origin+round((t-grid_origin)/p)*p for p in periods if p
                    and abs(grid_origin+round((t-grid_origin)/p)*p-t)<1e-12]
        if candidates:times[index]=min(candidates)
    times[0]=origin;times[-1]=origin+duration
    if np.any(np.diff(times)<=0):raise ContractError('Aligned response recording times must increase')
    return times,{'maximum_adjustment_seconds':float(np.max(np.abs(times-nominal))),
        'clock_coincidence_tolerance_seconds':1e-12,'prepared_origin_preserved':True,
        'exact_requested_stop_preserved':True,'state_time_or_RNG_rebased':False}


def observation_path(compiled, audit):
    """Record actual upstream cells, without reducing the executed network."""
    for target in ('midget_RGC_ON', 'midget_RGC_OFF', 'parasol_RGC_ON', 'parasol_RGC_OFF', 'small_bistratified_S_ON'):
        owner = np.flatnonzero(audit['population_owner_mask'].get(target, []))
        if target in compiled.components and len(owner):
            break
    else:
        raise ContractError('No actual owner RGC available for the one functional check')
    pending = [(target, int(owner[0]))]; seen = set(); chosen = set(); bc = set(); edges = []
    while pending:
        component, row = pending.pop()
        if (component, row) in seen:
            continue
        seen.add((component, row))
        if len(seen) > 256:
            raise ContractError('Actual observation ancestry exceeds recording budget; network is not thinned')
        c = compiled.components[component]
        if c.mechanism == 'HistoryBipolarTerminal':
            bc.add(component)
        if component == 'outer_native':
            chosen.add(c.outputs['release'].cell_ids[row])
        else:
            for port in c.outputs.values():
                if row < len(port.cell_ids):
                    chosen.add(port.cell_ids[row])
        for link in compiled.links:
            spec = link.specification
            if spec.target_component != component:
                continue
            incoming = link.matrix.getrow(row)
            if not incoming.nnz:
                continue
            parent = int(incoming.indices.min())
            source = compiled.components[spec.source_component]
            chosen.add(source.outputs[spec.source_port].cell_ids[parent])
            edges.append({'producer': spec.source_component, 'port': spec.source_port,
                          'consumer': component, 'input': spec.target_port,
                          'source_local_row': parent, 'target_local_row': row})
            pending.append((spec.source_component, parent))
    native = compiled.components['outer_native'].native_model
    cone_ids = compiled.components['outer_native'].outputs['release'].cell_ids
    for row, gid in enumerate(cone_ids):
        if gid in chosen:
            hc = native.B.getrow(row).indices
            if len(hc):
                chosen.add(compiled.components['outer_native'].outputs['Vh'].cell_ids[int(hc.min())])
    return target, tuple(sorted(chosen)), sorted(bc), edges


def paired_response(config, compiled, audit, frontend, media, directory):
    """Reuse one real constant-image ISET capture and one prepared full state.

    ISET supplies expected counts, with no upstream photon noise or temporal
    phototransduction. For this existing static image and fixed gaze, its rate
    is constant. Repeating that rate does not recalibrate it or claim a second
    ISET export. The existing declared background supplies the matched control.
    """
    if media.info['kind'] != 'image' or config.gaze_csv:
        raise ContractError('This one reused functional case requires the existing static image/fixed gaze')
    directory = Path(directory); directory.mkdir()
    duration = .02  # Reuse the existing 20 ms engineering window, not a biological threshold.
    target, ids, bc, edges = observation_path(compiled, audit)
    bc_names = ('midget_BC_ON','midget_BC_OFF','diffuse_BC_ON','diffuse_BC_OFF','S_ON_BC')
    bc_mechanisms = {name:compiled.components[name].mechanism for name in bc_names if name in compiled.components}
    diagnostic_consumers = [q.specification.target_component for q in compiled.links
                            if q.specification.source_port == 'generator_activation']
    if any(value != 'HistoryBipolarTerminal' for value in bc_mechanisms.values()) or diagnostic_consumers:
        raise ContractError('Actual assembled cone-BC terminal ownership/diagnostic-only port contract changed')
    captured = frontend.window(0., .0001)
    policy = config.background_policy()
    if config.background_initialization_mode != 'bounded_convergence' or not policy.require_converged:
        raise ContractError('Default acceptance retains the existing required numerical background convergence')
    runner = make_runner(compiled,config.neural_integration_backend)
    print('ACTUAL_COMPUTE_BACKEND',json.dumps(runner.model.compute_metadata,default=str),flush=True)
    solver = SolverConfig(**audit['resolved_execution_profile']['solver'])
    execution_check = None
    if config.neural_integration_backend in CUDA_BACKENDS:
        execution_check = verify_cuda_execution(runner,solver,config.light_context())
        save_json(directory/'execution_backend_check.json',execution_check)
        print('ACTUAL_FULL_NETWORK_CUDA_CPU_COMPARISON_PASS',execution_check['state_scalars'],
              execution_check['maximum_difference_over_original_solver_scale'],flush=True)
    reference = runner.initialize(solver)
    reference.input_position = {'background_context_hash': config.light_context().background_context_hash}
    checkpoint = directory/'background_checkpoint.json'
    def preparation_progress(current, row):
        save_json(directory/'background_progress.json', dict(row, absolute_time_seconds=current.time_seconds,
            integration_metrics=runner.integration_metrics))
        print('BACKGROUND_PREPARATION', directory.parent.name, current.time_seconds, row['phase'],
              'integration_seconds='+str(runner.integration_metrics['integration_wall_seconds']), flush=True)
    background_runner,background_procedure=choose_background_runner(runner)
    preparation_started=time.perf_counter()
    prepared, preparation, store = prepare_with_checkpoint(
        Path(__file__).resolve().parents[1], background_runner, reference, config.light_context(), policy, checkpoint,
        guard=MemoryGuard(int(config.background_memory_limit_mb*1024**2),
                          int(config.background_min_available_mb*1024**2)), on_progress=preparation_progress,
        resume_path=config.background_checkpoint_path or None,
        checkpoint_interval_seconds=config.background_checkpoint_interval_seconds)
    preparation_this_run_seconds=time.perf_counter()-preparation_started
    save_json(directory/'background_initialization.json', preparation)
    if not preparation['converged']:
        raise ContractError('Original background convergence not reached: '+preparation['status']+
                            '; budget_exit='+str(preparation.get('budget_exit')))
    origin = prepared.time_seconds
    grid_origin = prepared.grid_origin_seconds
    del prepared, reference
    stimulus = {}
    for name, channel in captured.items():
        value = channel.at(.00005)
        stimulus[name] = replace(channel, start_seconds=origin, stop_seconds=origin+duration,
            evaluator=lambda t, value=value: value, breakpoints_seconds=(),
            input_contract=dict(channel.input_contract, static_image_expected_rate_repeated=True,
                                original_capture_window_seconds=.0001, stimulus_origin_seconds=origin,
                                observation_stop_seconds=origin+duration))
    background = constant_background_channels(compiled, config.light_context(), origin+duration, start_seconds=origin)
    input_difference = {name: float(np.max(np.abs(stimulus[name].at(origin).values - background[name].at(origin).values)))
                        for name in stimulus}
    if not any(value > 0 for value in input_difference.values()):
        raise ContractError('Existing controlled image does not differ from the declared background')
    plan = indexed_recording_plan(RecordingPlan(states=tuple(name+'.terminal_release' for name in bc),
                         cell_ids=frozenset(ids), max_bytes=64*1024**2))
    times,recording_alignment = aligned_response_times(runner,grid_origin,origin,duration)
    relative_times = times-origin
    outputs = {}; restores = {}; terminal_checks = {};branch_timings={}
    for name, channels in (('background', background), ('stimulus', stimulus)):
        restore_started=time.perf_counter()
        state, saved = store.load(checkpoint, require_converged=True)
        restore_seconds=time.perf_counter()-restore_started;hash_started=time.perf_counter()
        restored_hash = canonical_hash(json.loads(runner.checkpoint(state)))
        restore_hash_seconds=time.perf_counter()-hash_started
        if restored_hash != saved['complete_checkpoint_hash']:
            raise ContractError('Matched branch full-state restore differs before execution')
        restores[name] = {'complete_checkpoint_hash': restored_hash,
                         'absolute_time_seconds': state.time_seconds, 'exact_restore': True}
        state.input_position.update(phase='media_stimulus', stimulus_origin_seconds=origin)
        guard = MemoryGuard(int(config.background_memory_limit_mb*1024**2),
                            int(config.background_min_available_mb*1024**2)); last = [time.monotonic()]
        def observe(t, y, bundle):
            if not guard.check():
                raise ContractError('Functional-check memory budget: ' + str(guard.last))
            if time.monotonic()-last[0] > 15:
                print('MATCHED_PATHWAY_RESPONSE', name, t, flush=True); last[0] = time.monotonic()
        advance_started=time.perf_counter()
        state, result = runner.advance(state, channels, origin+duration, sample_times=times,
                                       recording_plan=plan, state_observer=observe)
        branch_timings[name]={'strict_restore_seconds':restore_seconds,'restored_full_state_hash_seconds':restore_hash_seconds,
            'advance_and_record_seconds':time.perf_counter()-advance_started,'simulation_seconds':duration}
        if not all(np.all(np.isfinite(q.values)) for q in result.outputs_with_units.values()):
            raise ContractError('Nonfinite actual matched-response output')
        terminal_checks[name] = {}
        for component in bc:
            port = component+'.activation'; state_key = 'state:'+component+'.terminal_release'
            recorded = result.metadata['recorded_cell_ids']
            if (port not in result.outputs_with_units or state_key not in result.outputs_with_units
                    or recorded[port] != recorded[state_key]
                    or not np.array_equal(result.outputs_with_units[port].values,
                                          result.outputs_with_units[state_key].values)):
                raise ContractError('Actual BC activation differs from its recorded physical terminal release: '+component)
            terminal_checks[name][component] = {'activation_equals_terminal_release': True,
                                                 'recorded_cell_ids': recorded[port]}
        NPZWindowRecorder(directory/name).consume(result)
        outputs[name] = result
        del state
    differences = {}; traces = {}
    for key, value in outputs['stimulus'].outputs_with_units.items():
        if key == 'time' or value.values.ndim != 2:
            continue
        control = outputs['background'].outputs_with_units[key]
        delta = value.values-control.values
        changed = np.flatnonzero(np.any(delta != 0., axis=0))
        differences[key] = {'max_abs_difference': float(np.max(np.abs(delta))), 'unit': value.unit,
            'first_floating_nonzero_seconds': None if not len(changed) else float(relative_times[changed[0]])}
        if key == target+'.voltage':
            traces[key] = (relative_times, delta, value.unit)
    required = ['outer_native.release', target+'.voltage']
    if any(differences.get(key, {}).get('max_abs_difference', 0.) == 0 for key in required) or not any(
            differences.get(name+'.activation', {}).get('max_abs_difference', 0.) > 0 for name in bc):
        raise ContractError('No demonstrated actual native-release -> BC-terminal -> owner-RGC voltage difference')
    report = {'status': 'MATCHED_STATIC_INPUT_PATHWAY_RESPONSE_PASS_NUMERICALLY_PREPARED', 'duration_seconds': duration,
        'execution_procedures':{'background':background_procedure,'recording_indices':plan.procedure_identity},
        'stage_timings_seconds':{'background_prepare_or_restore_this_run':preparation_this_run_seconds,
            'background_stage_kind':'converged_checkpoint_reuse' if preparation.get('cache_reused') else 'fresh_or_partial_resume_preparation',
            'historical_background_report_wall_seconds':preparation['wall_seconds'],'branches':branch_timings},
        'neural_windows': 2, 'ISET_exports_added': 0, 'target_RGC': target, 'recorded_cell_ids': ids,
        'neural_integration_backend': config.neural_integration_backend, 'execution_backend_check': execution_check,
        'recording_clock_alignment': recording_alignment,
        'actual_observation_edges': edges, 'input_difference': input_difference, 'port_differences': differences,
        'initial_conditions': 'Identical complete numerically converged background checkpoint, including absolute clocks/RNG/history/delays',
        'stimulus_origin_seconds': origin, 'background_preparation': preparation, 'branch_restores': restores,
        'background_binding_hash': store.binding_hash, 'background_checkpoint': str(checkpoint),
        'BC_producer_port_consumer_audit': {'actual_mechanisms':bc_mechanisms,
            'generator_activation_consumers':diagnostic_consumers,'recorded_terminal_equality':terminal_checks,
            'activation_edges':[{'producer':q.specification.source_component,'port':q.specification.source_port,
                'consumer':q.specification.target_component,'input':q.specification.target_port}
                for q in compiled.links if q.specification.source_component in bc_mechanisms
                and q.specification.source_port == 'activation']},
        'capture_reuse': 'Existing real100us expected-rate capture, repeated for unchanged static image and fixed gaze',
        'delay_binding': '2.82ms applies to observed_Vh readout; BC consumes native release without that added edge delay',
        'latency_claim': 'Floating nonzero times are numerical diagnostics, not measured biological latencies',
        'physiological_validation': False, 'steady_state_claim': False,
        'stochastic_statistics': 'INSUFFICIENT; numerical background convergence is not spike stationarity'}
    save_json(directory/'response_check.json', report)
    return report, traces
