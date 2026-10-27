"""CPU reference execution of an exact dependency-closed representative-group patch.

This path restricts an existing, source-bound construction IVP. It never claims
fine-cell reconstruction, a prepared background, or parent RNG equivalence.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from .contracts import ContractError, canonical_hash
from .group_inner_dynamics import CONTACTS
from .group_time_monitor import GroupTimeMonitor
from .group_time_recording import GroupRecording
from .group_time_runner import GroupTimeArm
from .group_time_storage import save_bundle, load_bundle
from .local_patch import validate_patch_config
from .spatial_group_graph import digest
from .spatial_group_model import POPULATIONS, RGC

SCHEMA = 'local-retinal-patch-cpu-run-1'
QUALIFICATION = dict(default_adopted=False, fit_eligible=False,
                     coupled_background_qualified=False, physiological_calibration=False)
CELL_PARAMETERS = frozenset(('V0', 'baseline', 'reference_input'))
TIME_KERNELS = frozenset(('kernel', 'spike_history_kernel'))
V2_WRAPPER_PREDECESSORS = {
    'retina_sim/group_time_runner.py': '08e2b8691be5c74ef7c92a9e3a4a95cc7683f42e4ed19bae5317cbd03ce3bad1',
    'retina_sim/group_inner_compiled.py': '4b8c517fcf8072f98b4b22492cd9d71a05deae981ee5ea6950965e6d9bb3c8fd',
}


def _indices(value, size, role, *, allow_empty=False):
    rows = np.asarray(value)
    if (rows.dtype.kind not in 'iu' or rows.ndim != 1 or (not len(rows) and not allow_empty)
            or np.any(rows < 0) or np.any(rows >= size) or np.any(rows[1:] <= rows[:-1])):
        raise ContractError('Sorted nonempty original rows required: ' + role)
    return rows.astype(np.int64, copy=True)


def restrict_construction_payload(parent, population_rows, contact_rows, auxiliary_rows, *, rng_seed, allow_empty=False):
    """Pure row restriction; no dynamics, spatial averaging or parameter changes."""
    if type(rng_seed) is not int or rng_seed < 0 or set(population_rows) != set(POPULATIONS):
        raise ContractError('Complete patch population rows and explicit PCG64 seed required')
    if set(contact_rows) != set(CONTACTS):
        raise ContractError('Complete original contact identities required')
    sizes = {}
    for key, value in parent['states'].items():
        if key[1] not in CONTACTS and key[1] != 'selected_original_z1_auxiliary':
            sizes.setdefault(key[0], len(value))
    rows = {pop: _indices(population_rows[pop], sizes[pop], pop, allow_empty=allow_empty) for pop in POPULATIONS}
    contacts = {}
    for component in CONTACTS:
        keys = [key for key in parent['states'] if key[1] == component]
        if not keys:
            raise ContractError('Parent construction omitted contact states: ' + component)
        contacts[component] = _indices(contact_rows[component], len(parent['states'][keys[0]]), component)
    aux_key = ('cones', 'selected_original_z1_auxiliary', 'z1_aux')
    aux = _indices(auxiliary_rows['cones'], len(parent['states'][aux_key]), 'cone auxiliary', allow_empty=allow_empty)
    result = deepcopy(parent)
    result['states'] = {}
    for key, value in parent['states'].items():
        selected = contacts[key[1]] if key[1] in contacts else aux if key == aux_key else rows[key[0]]
        result['states'][key] = np.array(value[selected], copy=True)
    parameters = result['inner_parameters']
    for pop, fields in parent['inner_parameters']['populations'].items():
        for name, value in fields.items():
            if isinstance(value, np.ndarray):
                if name in CELL_PARAMETERS:
                    if value.shape != (sizes[pop],):
                        raise ContractError('Parent per-cell parameter shape differs: ' + pop + '.' + name)
                    parameters['populations'][pop][name] = value[rows[pop]].copy()
                elif name not in TIME_KERNELS:
                    raise ContractError('Unregistered array parameter; no shape-based guess: ' + pop + '.' + name)
    for component, fields in parent['inner_parameters']['contacts'].items():
        for name, value in fields.items():
            if isinstance(value, np.ndarray):
                if name != 'tau_decay' or component not in ('AII_FMB_glycine', 'AII_DB1_glycine'):
                    raise ContractError('Unregistered per-contact parameter: ' + component + '.' + name)
                parameters['contacts'][component][name] = value[contacts[component]].copy()
    result['spike_states'] = {}
    for ordinal, pop in enumerate(RGC):
        original = parent['spike_states'][pop]
        if original['pending_events'] or original['last_bin_counts'] is not None:
            raise ContractError('Only an unadvanced construction sampler may be restricted')
        spike = deepcopy(original)
        spike['history'] = np.asarray(original['history'])[:, rows[pop]].copy()
        spike['conditional_rate_Hz'] = np.asarray(original['conditional_rate_Hz'])[rows[pop]].copy()
        spike['rng_state'] = deepcopy(np.random.Generator(np.random.PCG64(rng_seed + 100 + ordinal)).bit_generator.state)
        result['spike_states'][pop] = spike
    for category, pop in (('native_inputs', 'cones'), ('rod_inputs', 'rods')):
        result[category] = {key: np.asarray(value)[rows[pop]].copy() for key, value in parent[category].items()}
    observation = result['observation_prehistory']
    observation['history_values'] = np.asarray(parent['observation_prehistory']['history_values'])[:, rows['HC']].copy()
    observation['prehistory'] = np.asarray(parent['observation_prehistory']['prehistory'])[rows['HC']].copy()
    if 'native_background_td' in result['input_reference']:
        result['input_reference']['native_background_td'] = np.asarray(parent['input_reference']['native_background_td'])[rows['cones']].copy()
    restriction = dict(classification='AGGREGATE_SUBMODEL_IVP_RESTRICTION',
        parent_state_sha256=canonical_hash(parent['states']),
        parent_parameters_sha256=canonical_hash(parent['inner_parameters']),
        population_rows=rows, contact_rows=contacts, cone_auxiliary_rows=aux,
        rng_seed=rng_seed, parent_rng_equivalence_claimed=False,
        original_equations_and_parameter_values_retained=True,
        source_construction_legal=True, coupled_background_qualified=False)
    parameters['identity'] = canonical_hash(restriction)
    parameters['provenance'] = dict(classification=restriction['classification'],
        parent_parameters_sha256=restriction['parent_parameters_sha256'],
        rule='Exact selected group/contact entries; temporal kernels unchanged; no new physiological parameters')
    result['patch_initialization'] = restriction
    return result


def prepare_seed(repo_root, assembly, config, *, guard=None):
    """Validate the bound parent bundle, then construct a new patch IVP identity."""
    check = guard or (lambda: None)
    check()
    root = Path(repo_root).resolve()
    config = validate_patch_config(config)
    if config.get('representation') != 'frozen_representative_groups_v1':
        raise ContractError('This executable path requires explicitly declared representative groups')
    binding = config['input_identity']
    manifest = (root / binding['path']).resolve()
    if (not manifest.is_relative_to(root) or manifest.name != 'manifest.json'
            or not manifest.is_file() or digest(manifest) != binding['sha256']):
        raise ContractError('Exact parent construction input/seed manifest binding required')
    parent_bundle = load_bundle(manifest.parent, expected_kind='source_construction_seed')
    source = parent_bundle['identity']['seed']
    parent_artifact = getattr(assembly, 'parent_artifact_sha256', None)
    if parent_artifact is None:
        from .spatial_group_model import _verify_artifact
        parent_artifact = _verify_artifact(assembly.base_artifact_root)[2]
    if (parent_bundle['identity']['artifact_sha256'] != parent_artifact
            or source.get('artifact_sha256') != parent_artifact
            or source.get('classification') != 'SOURCE_CONSTRUCTION_IVP_NOT_PREPARED_BACKGROUND'
            or source.get('source_construction_legal') is not True
            or source.get('neural_model_advanced') is not False
            or source.get('coupled_background_qualified') is not False):
        raise ContractError('Bound parent is not this graph\'s unadvanced source construction IVP')
    wrapper_upgrades = {}
    for relative, expected in source['source_sha256'].items():
        check()
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ContractError('Parent scientific/source parameter identity changed: ' + relative)
        current = digest(path)
        if current != expected:
            if V2_WRAPPER_PREDECESSORS.get(relative) != expected:
                raise ContractError('Parent scientific/source parameter identity changed: ' + relative)
            wrapper_upgrades[relative] = dict(parent_sha256=expected, current_sha256=current,
                scope='Explicit v2 factory/affine-gap lowering; unchanged original equation sources and defaults')
    parent = parent_bundle['payload']
    if (canonical_hash({':'.join(key): value for key, value in parent['states'].items()}) != source['initial_state_sha256']
            or canonical_hash(parent['inner_parameters']) != source['parameter_sha256']):
        raise ContractError('Parent construction state/parameter identity differs')
    for pop in (() if config['boundary_policy'] in ('bounded_background_halo_v2', 'full_reference_v2') else ('cones', 'HC')):
        key = next(key for key in parent['states'] if key[0] == pop and key[1] != 'selected_original_z1_auxiliary')
        if not np.array_equal(assembly.parent_population_rows[pop], np.arange(len(parent['states'][key]))):
            raise ContractError('This first patch implementation retains complete native feedback support')
    payload = restrict_construction_payload(parent, assembly.parent_population_rows,
        assembly.parent_contact_rows, assembly.parent_auxiliary_rows, rng_seed=config['seed'],
        allow_empty=config['boundary_policy']=='bounded_background_halo_v2')
    specs = (*assembly.state_blocks, *assembly.probability_blocks, *assembly.auxiliary_z1_blocks)
    if (set(payload['states']) != {(block.population, block.mechanism, block.name) for block in specs}
            or any(payload['states'][block.population, block.mechanism, block.name].shape != (block.count,) for block in specs)):
        raise ContractError('Restricted source IVP does not cover the exact patch state registry')
    payload['patch_initialization'].update(patch_artifact_sha256=assembly.artifact_sha256,
        parent_bundle_manifest_hash=parent_bundle['manifest_hash'], parent_manifest_file_sha256=binding['sha256'],
        request_sha256=canonical_hash(config), source_sha256=deepcopy(source['source_sha256']))
    if wrapper_upgrades:
        payload['patch_initialization']['wrapper_upgrades'] = wrapper_upgrades
    if config['boundary_policy']=='bounded_background_halo_v2':
        from .local_patch_boundary import configure_boundary_runtime
        configure_boundary_runtime(assembly, parent, payload)
    check()
    return payload


def make_protocol(seed, *, duration_seconds=.0011, checkpoint_seconds=None, bounded_v2=False):
    start = float(seed['time_seconds'])
    if type(duration_seconds) not in (int, float) or not math.isfinite(duration_seconds) or not 0 < duration_seconds <= (.004 if bounded_v2 else .002):
        raise ContractError('Explicit positive patch demonstration within the selected 2/4 ms protocol cap required')
    stop = start + duration_seconds
    checkpoint = start + .0005 if checkpoint_seconds is None else checkpoint_seconds
    if not start < checkpoint < stop:
        raise ContractError('Checkpoint must be inside the approved short segment')
    return dict(schema='local-retinal-patch-reference-protocol-1', start_seconds=start,
        stop_seconds=stop, checkpoint_seconds=float(checkpoint), input_fps=150.,
        input_origin_seconds=0., clock_origin_seconds=0., rod_step_seconds=.0001,
        BC_RGC_period_seconds=.001, recording_step_seconds=.0005, monitor_step_seconds=.0001,
        native_observation_delay_seconds=.00282, input_frame_indices=sorted(seed['native_inputs']),
        reference=dict(method='joint_DOP853', rtol=1e-6, atol=1e-9, max_step=.0001),
        candidate=dict(rtol=1e-6, atol=1e-9, fast_max_step=.0001),
        initialization='Bound aggregate dependency-closed restriction of an unadvanced construction IVP',
        qualification=deepcopy(QUALIFICATION))


def _write_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _selected_snapshot(assembly, snapshot):
    """Save actual core states and ports with explicit original group identities."""
    core = assembly.core_population_rows
    contacts = {route.contact_component: route for route in assembly.routes.values() if route.contact_component}
    selected = dict(time_seconds=snapshot['time_seconds'], side=snapshot['side'],
        event_generation=snapshot['event_generation'], states={}, rhs={}, ports={}, identities={}, port_rows={})
    for pop in POPULATIONS:
        rows = np.asarray(core[pop], dtype=np.int64)
        selected['identities'][pop] = dict(local_rows=rows.copy(),
            parent_group_rows=np.asarray(assembly.parent_population_rows[pop])[rows].copy(),
            group_ids=np.asarray(assembly.populations[pop].group_ids)[rows].copy(),
            member_counts=np.asarray(assembly.populations[pop].counts)[rows].copy(),
            representation='frozen_representative_group; not registered individual cell')
    for key, value in snapshot['states'].items():
        if key[1] in contacts:
            route = contacts[key[1]]
            post = assembly.array('routes/' + route.name + '/contact_post_group.npy')
            rows = np.flatnonzero(np.isin(post, core[route.post]))
        elif key[1] == 'selected_original_z1_auxiliary':
            rows = np.flatnonzero(np.isin(assembly.populations['cones'].recording_auxiliary_group_rows, core['cones']))
        else:
            rows = core[key[0]]
        selected['states'][key] = np.asarray(value)[rows].copy()
        selected['rhs'][key] = np.asarray(snapshot['rhs'][key])[rows].copy()
    for key, value in snapshot['ports'].items():
        if key == ('cones', 'z1_aux'):
            auxiliary = assembly.populations['cones'].recording_auxiliary_group_rows
            rows = np.flatnonzero(np.isin(auxiliary, core['cones']))
            selected['port_rows'][key] = dict(local_auxiliary_rows=rows.copy(),
                local_cone_group_rows=np.asarray(auxiliary)[rows].copy(),
                parent_cone_group_rows=np.asarray(assembly.parent_population_rows['cones'])[np.asarray(auxiliary)[rows]].copy())
        elif key == ('local_patch_boundary', 'gap_current'):
            rows = np.arange(len(value), dtype=np.int64)
            selected['port_rows'][key] = dict(identity='omitted reciprocal gap reservoir; equal opposite current', unit='A')
        elif key[0] in core:
            rows = core[key[0]]
        elif key[0] in contacts:
            route = contacts[key[0]]
            if key[-1] == 'contact_conductance':
                post = assembly.array('routes/' + route.name + '/contact_post_group.npy')
                rows = np.flatnonzero(np.isin(post, core[route.post]))
            else:
                rows = core[route.post]
        else:
            raise ContractError('Unregistered output identity in patch observation: ' + str(key))
        selected['ports'][key] = np.asarray(value)[rows].copy()
    return selected


def state_witness_hash(payload):
    """Exclude measured build/wall counters while retaining all causal state."""
    copy = deepcopy(payload)
    copy['controller'].pop('metrics', None)
    copy['recording'] = {key: value for key, value in copy['recording'].items()
        if key not in ('write_seconds', 'cursor_hash')}
    return canonical_hash(copy)


def run_patch(assembly, seed, protocol, outdir, guard, *, resume=False, stop_at_checkpoint=True):
    """Run one bounded CPU segment; caller supplies independent authorization/guard."""
    from tools.validate_group_time_candidate import clock_events, sample_grid, event_snapshot_sides
    output = Path(outdir).resolve()
    if guard is None or not callable(guard):
        raise ContractError('Explicit independent run resource/source/STOP guard required')
    if output.exists() != resume:
        raise ContractError('Fresh run output or explicit matching checkpoint continuation required')
    guard()
    if not resume:
        output.mkdir(parents=False)
    identity = dict(schema=SCHEMA, patch_artifact_sha256=assembly.artifact_sha256,
        seed_sha256=canonical_hash(seed), protocol_sha256=canonical_hash(protocol),
        runner_sha256=digest(Path(__file__)), backend='scipy_cpu_v1')
    events = clock_events(protocol)
    start, stop = protocol['start_seconds'], protocol['stop_seconds']
    records = set(sample_grid(start, stop, .0005))
    if len(records) < 2:
        raise ContractError('Short demonstration must include at least two original recording points')
    input_edges = {protocol['input_origin_seconds'] + int(index)/150.
        for index in seed['native_inputs'] if start < protocol['input_origin_seconds'] + int(index)/150. <= stop}
    queue_nodes = set(sample_grid(start, stop, .0001)) | records | input_edges | {start, stop}
    if stop_at_checkpoint and protocol['checkpoint_seconds'] not in {row[0] for row in events}:
        raise ContractError('Checkpoint must coincide with an original event boundary')
    recording_type = GroupRecording
    if getattr(assembly, 'bounded_local', False):
        from .local_patch_boundary import BoundedRecording
        recording_type = BoundedRecording
    arm = monitor = recorder = None
    last_snapshot = None
    selected_observations = []
    wall_started = perf_counter()
    restore = None
    try:
        if resume:
            session = json.loads((output/'session_checkpoint.json').read_text(encoding='utf8'))
            session_hash = session.pop('session_hash', None)
            if session_hash != canonical_hash(session) or session['identity'] != identity:
                raise ContractError('Patch continuation session identity differs')
            saved = load_bundle(output/'checkpoint', expected_kind='complete_model_checkpoint', expected_identity=identity)
            arm = GroupTimeArm.from_checkpoint(assembly, seed, protocol, saved['payload'], mode='reference', guard=guard)
            monitor = GroupTimeMonitor.from_payload(assembly, seed['states'], saved['payload']['monitor'], parameters=seed['inner_parameters'])
            recorder = recording_type.restore(assembly, output/'recording', saved['payload']['recording'], artifactidentity=identity)
            rebuilt = arm.checkpoint_payload(monitor=monitor.payload(), recording=recorder.checkpoint_payload())
            if canonical_hash(rebuilt) != canonical_hash(saved['payload']):
                raise ContractError('Complete fresh-process checkpoint reconstruction differs')
            recorded = set(session['recorded_times'])
            next_event = session['next_event']
            checkpoint_snapshot = load_bundle(output/'checkpoint_snapshot', expected_kind='validation_evidence', expected_identity=identity)['payload']
            restored_snapshot = arm.snapshot(arm.time_seconds, 'right', queue_node=False)
            if canonical_hash(checkpoint_snapshot) != canonical_hash(restored_snapshot):
                raise ContractError('Fresh-process complete state/RHS/aux snapshot differs')
            save_bundle(output/'restored_snapshot', kind='validation_evidence', identity=identity,
                payload=restored_snapshot, qualification=QUALIFICATION)
            restore = dict(status='EXACT_COMPLETE_PAYLOAD_AND_SNAPSHOT_RECONSTRUCTION',
                saved_snapshot_sha256=canonical_hash(checkpoint_snapshot), restored_snapshot_sha256=canonical_hash(restored_snapshot),
                saved_payload_sha256=canonical_hash(saved['payload']), rebuilt_payload_sha256=canonical_hash(rebuilt),
                public_time_seconds=arm.time_seconds, native_frontier_seconds=arm.native_state.time_seconds)
            _write_new(output/'restore_receipt.json', restore)
        else:
            _write_new(output/'run_identity.json', identity)
            _write_new(output/'protocol.json', protocol)
            save_bundle(output/'construction_seed', kind='source_construction_seed', identity=identity,
                payload=seed, qualification=QUALIFICATION)
            arm = GroupTimeArm(assembly, seed, protocol, mode='reference', guard=guard)
            monitor = GroupTimeMonitor(assembly, seed['states'], parameters=seed['inner_parameters'])
            recorder = recording_type(assembly, output/'recording', np.array(sorted(records), dtype=np.float64), artifactidentity=identity)
            recorded, next_event = set(), 0
        def observe(t, side, *, state=None, do_record=False):
            nonlocal last_snapshot
            guard()
            options = dict(queue_node=t in queue_nodes)
            if state is not None: options['inner_state'] = state
            snapshot = arm.snapshot(t, side, **options)
            monitor.observe(t, side, snapshot['states'], snapshot['rhs'], transitions=snapshot['transitions'],
                ports=snapshot['ports'], event_generation=snapshot['event_generation'],
                aux=snapshot['aux'], rhs_kind=snapshot['rhs_kind'])
            if monitor.domain_failures:
                raise ContractError('Original state/history/gap domain monitor failed: ' + str(monitor.domain_failures[:3]))
            last_snapshot = snapshot
            if do_record and t in records and t not in recorded:
                recorder.record(t, arm.record_native_values(t, side, snapshot=snapshot), snapshot['inner_outputs'],
                    delayedHC=np.asarray(snapshot['delayed_HC']), eventlogs=arm.event_log)
                arm.event_log = {pop: [] for pop in arm.event_log}
                recorded.add(t)
                selected_observations.append(_selected_snapshot(assembly, snapshot))
            return snapshot
        if not resume:
            observe(start, 'right')
            arm.initial_commit()
            observe(start, 'right', do_record=True)
            arm.ack(start)
        for event_index in range(next_event, len(events)):
            guard()
            end, rod, bc = events[event_index]
            begin = arm.inner_time
            arm.ensure_native(end, 'left')
            advance = arm.advance_inner(begin, end)
            points = {t for t in queue_nodes if begin < t < end}
            points.update(t for t in advance.accepted_times if begin < t < end)
            for interval in arm.native_intervals:
                for curve in interval.curves.values():
                    points.update(segment.t1 for segment in curve.segments if begin < segment.t1 < end)
            for t in sorted(points):
                state = advance.evaluate(t, 'right')
                if t in input_edges: observe(t, 'left', state=state)
                observe(t, 'right', state=state, do_record=t in records)
            pre_side, post_side = event_snapshot_sides(end, input_edges)
            observe(end, pre_side)
            if rod or bc:
                arm.commit(end, rod=rod, bc=bc, rgc=bc,
                    rod_sample_time=max(start, end-.0001+1e-13) if rod else None)
                observe(end, post_side, do_record=True)
            elif end in records:
                observe(end, 'right', do_record=True)
            arm.ack(end)
            if not resume and stop_at_checkpoint and end == protocol['checkpoint_seconds']:
                checkpoint_snapshot = arm.snapshot(end, 'right', queue_node=False)
                save_bundle(output/'checkpoint_snapshot', kind='validation_evidence', identity=identity,
                    payload=checkpoint_snapshot, qualification=QUALIFICATION)
                payload = arm.checkpoint_payload(monitor=monitor.payload(), recording=recorder.checkpoint_payload())
                checkpoint = save_bundle(output/'checkpoint', kind='complete_model_checkpoint', identity=identity,
                    payload=payload, qualification=QUALIFICATION)
                session = dict(identity=identity, public_time_seconds=end, next_event=event_index+1,
                    recorded_times=sorted(recorded))
                _write_new(output/'session_checkpoint.json', dict(session, session_hash=canonical_hash(session)))
                save_bundle(output/'observations_prefix', kind='validation_evidence', identity=identity,
                    payload=dict(observations=selected_observations), qualification=QUALIFICATION)
                result = dict(status='CHECKPOINT_SAVED_REQUIRES_NEW_PROCESS_CONTINUATION', identity=identity,
                    time_seconds=end, checkpoint=checkpoint, whole_worker_wall_seconds=perf_counter()-wall_started,
                    absolute_monitor=monitor.report(end-start), neural_metrics=arm.metrics,
                    fit_eligible=False, physiological_calibration=False)
                _write_new(output/'checkpoint_receipt.json', result)
                return result
        if recorded != records:
            raise ContractError('Original ordinary recording coverage incomplete')
        if last_snapshot is None:
            raise ContractError('No final source state observed')
        final_snapshot = arm.snapshot(stop, 'right', queue_node=False)
        selected_observations.append(_selected_snapshot(assembly, final_snapshot))
        full_snapshot = save_bundle(output/'final_snapshot', kind='validation_evidence', identity=identity,
            payload=final_snapshot, qualification=QUALIFICATION)
        payload = arm.checkpoint_payload(monitor=monitor.payload(), recording=recorder.checkpoint_payload())
        final_checkpoint = save_bundle(output/'final_checkpoint', kind='complete_model_checkpoint', identity=identity,
            payload=payload, qualification=QUALIFICATION)
        observations = save_bundle(output/('observations_suffix' if resume else 'observations_complete'),
            kind='validation_evidence', identity=identity, payload=dict(observations=selected_observations), qualification=QUALIFICATION)
        result = dict(status='SHORT_AGGREGATE_PATCH_FUNCTIONAL_RUN_COMPLETE', identity=identity,
            final_time_seconds=arm.time_seconds, simulated_seconds=stop-start,
            final_checkpoint=final_checkpoint, final_snapshot=full_snapshot, observations=observations,
            state_witness_sha256=state_witness_hash(payload), restore=restore,
            absolute_monitor=monitor.report(stop-start), neural_metrics=arm.metrics,
            recorded_times=sorted(recorded), recording_cursor_sha256=payload['recording']['cursor_hash'],
            whole_worker_wall_seconds=perf_counter()-wall_started,
            numerical_error_absolute_truth=None, spatial_convergence_qualified=False,
            qualified_background=False, fit_eligible=False, physiological_calibration=False,
            recording_status='All original scheduled fields/windows saved; group replay UI finalization is not used for patch geometry')
        _write_new(output/'result.json', result)
        return result
    except Exception as exc:
        failure = dict(status='STOPPED_OR_FAILED', error=type(exc).__name__, message=str(exc),
            identity=identity, whole_worker_wall_seconds=perf_counter()-wall_started,
            public_time_seconds=None if arm is None else arm.time_seconds,
            inner_time_seconds=None if arm is None else arm.inner_time,
            model_forward_started=bool(arm is not None and arm.metrics['inner_advances']),
            absolute_monitor=None if monitor is None or not monitor.count else monitor.report(max(0., arm.inner_time-start)))
        _write_new(output/('resume_failure.json' if resume else 'failure.json'), failure)
        raise
    finally:
        if arm is not None: arm.close()
