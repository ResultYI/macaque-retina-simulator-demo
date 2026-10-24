"""Opt-in resident background preparation and complete qualification envelopes.

UNVALIDATED implementation, never imported by the default backend. Calling these
APIs is an explicit execution request: binding may initialize the existing CPU
background and upload it; preparation integrates; save/load exports/uploads a
complete state. Merely adding this module authorizes none of those operations.

The original numerical policy and event/grid observations are retained. A plain
runtime checkpoint or a progress envelope without a complete converged report
cannot be used as a qualified background cache. Background convergence remains a
sampled fixed-point claim, not resident-backend QA acceptance or fit eligibility.
"""
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
import hashlib
import math
import numpy as np

from .background import (
    BackgroundPolicy, NUMERICAL_UNIT_SCALES, constant_background_channels,
    timing_requirements,
)
from .background_checkpoint import BackgroundCheckpointStore, MemoryGuard
from .contracts import ContractError, canonical_hash
from .recording import RecordingPlan
from .resident_monitor import resident_history_consistency
from .resident_runner import ResidentNativeRunner
from .resident_state import ResidentStateHandle


def _scope(runner, policy, *, bound=True):
    # Opt-in source split reuses exactly the original qualification predicates.
    from .cone_hc_split_runner import ConeHCSplitRunner
    from .cone_hc_split_contract import USEFUL_SCOPE
    split = type(runner) is ConeHCSplitRunner
    if split:
        runner.contract.require_execution_approval()
        if (runner.contract.runtime_authorization or {}).get('scope') != USEFUL_SCOPE:
            raise ContractError('Split background requires its explicit useful-source permit')
    if (type(runner) is not ResidentNativeRunner and not split) or type(policy) is not BackgroundPolicy:
        raise ContractError('Explicit resident candidate runner and original BackgroundPolicy required')
    if bound:
        if runner.binding is None:
            raise ContractError('Bind original state, monitor reference, policy and channels first')
        if canonical_hash(asdict(policy)) != runner.binding['monitor_policy_hash']:
            raise ContractError('Resident background policy differs from the frozen binding')
    requirements = timing_requirements(runner, policy)
    # The current full graph has the original 1.25 s minimum and 3-window rule.
    # Reject a weakened policy instead of silently repairing or relaxing it.
    if policy.stable_windows < 3 or requirements['minimum_required_seconds'] < 1.25 - 1e-12:
        raise ContractError('Resident scope cannot weaken 1.25 s / 3 complete-window qualification')
    return requirements


def _execution_record(runner):
    return {
        'execution': ('cone_hc_split_background_candidate_v1' if
                      type(runner).__name__ == 'ConeHCSplitRunner' else 'resident_native_background_candidate_v1'),
        'resident_binding_hash': runner.binding['binding_hash'],
        'resident_model_hash': runner.model.model_hash,
        'numerical_contract_hash': runner.contract.sha256,
        'preparation_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'backend_validation': 'UNVALIDATED',
        'background_qualification_is_backend_acceptance': False,
        'production_or_fit_eligibility': False,
    }


def _constant_channels(runner, channels):
    expected = {x.channel for x in runner.compiled.spec.external_inputs}
    if set(channels) != expected:
        raise ContractError('Background channel coverage differs from the compiled external ports')
    if any(ch.input_contract.get('declared_constant_background') is not True or
           ch.breakpoints_seconds for ch in channels.values()):
        raise ContractError('Resident preparation requires explicitly constant background channels')


@dataclass(frozen=True)
class ResidentBackgroundSession:
    """Prepared API inputs; construction is explicit, not a qualification result."""
    state: ResidentStateHandle
    reference_state: object
    channels: dict
    preparation_origin_seconds: float
    deterministic_initialization: object


def _origin(runner, state, requested=None):
    """Only the imported session or a strictly restored envelope owns elapsed time."""
    origin = (runner.binding or {}).get('background_preparation_origin_seconds')
    if (origin is None or not math.isfinite(origin) or origin > state.time_seconds or
            state.input_position.get('resident_background_origin_seconds') != origin or
            state.input_position.get('phase') != 'uniform_background'):
        raise ContractError('Background elapsed time has no bound uniform-background origin')
    if requested is not None and float(requested) != origin:
        raise ContractError('Caller cannot backdate background preparation origin')
    return origin


def bind_resident_background(runner, snapshot, light_context, policy=None, *,
                             reference_state=None, preparation_origin_seconds=None):
    """Perform the existing initialization BEFORE the explicit resident upload.

    This helper owns the constant channel evaluators and hashes their actual
    typed values. Retain the returned channels for every advance/restore; the
    runner intentionally rejects replacement evaluator identities. The frozen
    CPU monitor reference is never modified. Importing a diagnostic background
    starts elapsed time at that snapshot. Earlier elapsed time is recovered only
    by the strict envelope load into the same original session binding.
    """
    policy = policy or BackgroundPolicy()
    _scope(runner, policy, bound=False)
    if runner.binding is not None:
        raise ContractError('Create a separate resident runner for a different background binding')
    origin = (snapshot.time_seconds if preparation_origin_seconds is None
              else float(preparation_origin_seconds))
    if not math.isfinite(origin) or origin != snapshot.time_seconds:
        raise ContractError('A new background binding starts at the actual supplied state time')
    if (snapshot.time_seconds != snapshot.grid_origin_seconds or snapshot.sample_index != 0):
        if (snapshot.input_position.get('phase') != 'uniform_background' or
                snapshot.input_position.get('background_context_hash') != light_context.background_context_hash):
            raise ContractError('Advanced movie/runtime state cannot be relabelled as prior uniform background')
    reference = snapshot if reference_state is None else reference_state
    channels = constant_background_channels(
        runner.compiled, light_context, origin + policy.max_sim_seconds + 1.)
    _constant_channels(runner, channels)
    # Keep the complete frozen reference, including input_position, unchanged.
    initial = copy(snapshot)
    initial.input_position = deepcopy(snapshot.input_position)
    initialization = None
    if (getattr(runner.compiled, 'retained_background_initializer', None) is not None and
            initial.time_seconds == initial.grid_origin_seconds and initial.sample_index == 0):
        from .retained_background import close_retained_background
        initial, initialization = close_retained_background(runner.original, initial, channels)
        initialization['policy'] = runner.compiled.retained_background_initializer
    initial.input_position.update(
        background_context_hash=light_context.background_context_hash,
        phase='uniform_background', resident_background_origin_seconds=origin)
    if initialization is not None:
        initial.input_position['resident_background_initializer'] = {
            'policy': runner.compiled.retained_background_initializer,
            'performed_before_resident_upload': True,
            'source_sha256': hashlib.sha256(
                Path(__file__).with_name('retained_background.py').read_bytes()).hexdigest(),
        }
    # These channels were constructed here as constants, not merely accepted
    # because of caller metadata. Hash the actual values, units and evidence.
    content = {name: canonical_hash(channel.at(initial.time_seconds))
               for name, channel in channels.items()}
    state = runner.import_reference_state(
        initial, reference_state=reference, channels=channels,
        channel_data_hashes=content, monitor_policy=policy)
    return ResidentBackgroundSession(state, reference, channels, origin, initialization)


def prepare_resident_background(runner, state, channels, policy=None, *,
                                reference_state, unit_scales=None, explicit_scales=None,
                                absolute_fraction=None, on_chunk=None, on_window=None,
                                preparation_origin_seconds=None,
                                deterministic_initialization=None):
    """Original bounded preparation loop using full resident monitor predicates.

    No complete state/history export occurs in this loop. on_chunk/on_window
    receive a read-only ResidentStateHandle; explicit checkpoint callbacks may
    export it. Resumption deliberately restarts the consecutive-window streak.
    """
    policy = policy or BackgroundPolicy()
    requirements = _scope(runner, policy)
    if not isinstance(state, ResidentStateHandle):
        raise ContractError('Resident background preparation requires a resident state handle')
    if (getattr(runner.compiled, 'retained_background_initializer', None) is not None and
            state.time_seconds == state.grid_origin_seconds and state.sample_index == 0):
        expected_initialization = {
            'policy': runner.compiled.retained_background_initializer,
            'performed_before_resident_upload': True,
            'source_sha256': hashlib.sha256(
                Path(__file__).with_name('retained_background.py').read_bytes()).hexdigest(),
        }
        if state.input_position.get('resident_background_initializer') != expected_initialization:
            raise ContractError('Fresh background requires bind_resident_background before device preparation')
    _constant_channels(runner, channels)
    runner.contract.validate_channels(channels, runner.binding)
    author_periods = [c.time_contract.fixed_step_seconds
                      for c in runner.compiled.components.values()
                      if c.time_contract.kind == 'author_discrete']
    for period in author_periods:
        for name in ('window_seconds', 'monitor_step_seconds', 'max_sim_seconds'):
            ticks = getattr(policy, name) / period
            if abs(ticks - round(ticks)) > 1e-7:
                raise ContractError('Background ' + name + ' must preserve the author input/grid phase')
    wall = perf_counter()
    from .useful_runtime_budget import goal_only_authorization
    goal_only=goal_only_authorization(getattr(runner.contract,'runtime_authorization',None) or {})
    monitor = runner.make_monitor(reference_state, policy, unit_scales=unit_scales,
                                  explicit_scales=explicit_scales,
                                  absolute_fraction=absolute_fraction)
    segment_start = state.time_seconds
    start = _origin(runner, state, preparation_origin_seconds)
    windows = []
    stable = observations = 0
    reason, status = 'max_sim_seconds', 'NOT_REACHED'
    end = start + policy.max_sim_seconds
    while state.time_seconds < end - 1e-12:
        left = state.time_seconds
        right = min(left + policy.window_seconds, end)
        monitor.reset()
        window_observations = 0
        while state.time_seconds < right - 1e-12:
            if not goal_only and perf_counter() - wall >= policy.max_wall_seconds:
                reason = 'max_wall_seconds'
                break
            if observations >= policy.max_observations:
                reason = 'max_observations'
                break
            if on_chunk is not None and on_chunk(state) is False:
                reason = 'cancelled'
                break
            remaining = policy.max_observations - observations
            stop = min(right, state.time_seconds + policy.monitor_step_seconds * min(10, remaining))
            first = int(np.floor((state.time_seconds - start) / policy.monitor_step_seconds + 1e-8)) + 1
            last = int(np.floor((stop - start) / policy.monitor_step_seconds + 1e-8))
            grid = start + np.arange(first, last + 1, dtype=np.int64) * policy.monitor_step_seconds

            def aligned(t):
                periods = list(runner.periods.values()) + [runner.observation_dt, runner.delayed_step]
                values = [state.grid_origin_seconds + round((t - state.grid_origin_seconds) / p) * p
                          for p in periods if p and abs(state.grid_origin_seconds +
                          round((t - state.grid_origin_seconds) / p) * p - t) < 1e-12]
                return min(values) if values else float(t)

            times = np.array(sorted(set([state.time_seconds, stop] + [aligned(t) for t in grid
                             if state.time_seconds + 1e-12 < t < stop - 1e-12])))

            def observe_budgeted(t, y, bundle):
                if observations + monitor.count < policy.max_observations:
                    monitor.observe(t, y, bundle)
                else:
                    monitor.coverage_limited = True

            state, _ = runner.advance(
                state, channels, stop, sample_times=times,
                recording_plan=RecordingPlan(ports=(), states=(), cell_ids=()),
                state_observer=observe_budgeted)
            observations += monitor.count
            window_observations += monitor.count
            monitor.count = 0
        width = state.time_seconds - left
        if width <= 0:
            break
        row = monitor.report(width)
        row.update(observations=window_observations, start_seconds=left,
                   stop_seconds=state.time_seconds,
                   complete_window=abs(width - policy.window_seconds) < 1e-10,
                   history_consistency_failures=resident_history_consistency(
                       state, runner, policy, unit_scales))
        row['monitoring_budget_limited'] = monitor.coverage_limited
        row['passed'] = (row['passed'] and row['complete_window'] and
                         not row['history_consistency_failures'] and not monitor.coverage_limited)
        stable = stable + 1 if row['passed'] else 0
        row['consecutive_passed_windows'] = stable
        windows.append(row)
        if on_window is not None:
            on_window(state, row)
        if reason in ('max_wall_seconds', 'max_observations', 'cancelled'):
            break
        if (stable >= policy.stable_windows and
                state.time_seconds - start >= requirements['minimum_required_seconds'] - 1e-12):
            status, reason = 'CONVERGED', 'criteria_satisfied'
            break
    unknown_random = [x['component'] for x in requirements['stochastic_history']
                      if not x['deterministic_state_independent_of_spikes']]
    unknown_scales = sorted({failure for row in windows
                             for failure in row['history_consistency_failures']
                             if 'unregistered_queue_scale' in failure})
    missing = (monitor.missing + ([{'stochastic_state_dependency_unknown': unknown_random}]
               if unknown_random else []) + ([{'unregistered_queue_scales': unknown_scales}]
               if unknown_scales else []))
    if missing:
        status = 'INCONCLUSIVE'
    elapsed = state.time_seconds - start
    report = {
        'schema_version': '1.0', 'policy': asdict(policy), 'status': status,
        'converged': status == 'CONVERGED', 'budget_exit': reason,
        'elapsed_seconds': elapsed, 'absolute_preparation_interval_seconds': [start, state.time_seconds],
        'segment_start_seconds': segment_start, 'monitor_reference': 'frozen initial reference',
        'consecutive_window_streak_reset_on_resume': segment_start > start,
        'grid_origin_seconds': state.grid_origin_seconds, 'stimulus_origin_seconds': state.time_seconds,
        'wall_seconds': perf_counter() - wall, 'observations': observations,
        'application_wall_budget_enforced': not goal_only,
        'timing_requirements': requirements,
        'numerical_unit_registry': unit_scales if unit_scales is not None else NUMERICAL_UNIT_SCALES,
        'uncovered_states': missing, 'checked_states': [r['spec'].state_name for r in monitor.rows],
        'windows': windows, 'dominant': windows[-1]['dominant'] if windows else None,
        'deterministic_status': status,
        'stochastic_status': 'STATISTICS_INSUFFICIENT' if state.spike_states else 'NOT_APPLICABLE',
        'stochastic_burnin_completed': elapsed >= requirements['minimum_stochastic_burnin_seconds'] - 1e-12,
        'stochastic_history_is_separate_from_fixed_point_test': True,
        'whole_network_steady_state_claim': False,
        'domain_and_conservation_scope': 'explicit registered positive chemical states, unit-interval release/resource states, reciprocal gap-current sum, queue timestamps/finite values and constant-queue residual; no undeclared invariant invented',
        'rng_states_preserved_and_advanced': {cid: canonical_hash(aux['rng_state'])
                                             for cid, aux in state.spike_states.items()},
        'stationarity_statistics': {'enabled': False,
            'required_explicit_policy': ['window_seconds_at_least_1', 'at_least_3_windows',
                'mean_variance_rate_autocorrelation_tolerances', 'minimum_effective_sample_size'],
            'claim': 'No universal statistical acceptance tolerance or noise-free spike criterion supplied'},
        'maximum_scope': 'all observed accepted event endpoints and declared numerical monitoring grid; sub-grid extrema not certified',
        'adaptation_cache_enabled': False, 'physiological_validation': 'NOT_RUN',
        'deterministic_initialization': deterministic_initialization,
        'claim': 'Sampled deterministic numerical fixed-point candidate only; model calibration and stochastic stationarity not established',
        'resident_execution': _execution_record(runner),
    }
    state.input_position.update(
        stimulus_origin_seconds=state.time_seconds, background_initialization_status=status,
        background_preparation_policy_hash=canonical_hash(asdict(policy)),
        background_input_hash=canonical_hash({k: v.input_contract for k, v in channels.items()}))
    return state, report


class ResidentBackgroundCheckpointStore(BackgroundCheckpointStore):
    """Original strict envelope, isolated resident identity and resident restore.

    Save is an explicit complete export. Load uses the original envelope/hash/
    geometry checks and the resident public restore hook; the CPU parsed shortcut
    cannot be selected. No original CPU _WindowMonitor is constructed here.
    """

    def __init__(self, root, runner, reference_state, light_context, policy):
        _scope(runner, policy)
        if canonical_hash(reference_state) != runner.binding['monitor_reference_hash']:
            raise ContractError('Background cache monitor reference differs from the frozen QA binding')
        # Base construction writes phase metadata. Isolate it from the frozen
        # complete monitor reference, whose hash includes its input_position.
        private = copy(reference_state)
        private.input_position = deepcopy(reference_state.input_position)
        super().__init__(root, runner, private, light_context, policy)
        self.reference = reference_state
        self.expected_context['resident_qa_binding_hash'] = runner.binding['binding_hash']
        self.binding.update(
            resident_qa_binding=deepcopy(runner.binding),
            resident_execution=_execution_record(runner),
            resident_background_origin_seconds=runner.binding.get('background_preparation_origin_seconds'),
            resident_outer_sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('background_checkpoint.py', 'checkpoint_loading.py')})
        self.binding_hash = canonical_hash(self.binding)

    def coverage(self):
        # Registry-only inspection; make_monitor performs actual denominator
        # validation once for preparation. Never allocate a second GPU monitor.
        specs = self.runner.model.state_specs
        missing = [{'state': s.state_name, 'unit': s.unit}
                   for s in specs if s.unit not in NUMERICAL_UNIT_SCALES]
        states = [s.state_name for s in specs if s.unit in NUMERICAL_UNIT_SCALES]
        return {'state_blocks': len(specs), 'covered_blocks': len(states),
                'state_scalars': sum(s.shape[0] for s in specs), 'missing': missing,
                'states': states, 'timing_requirements': timing_requirements(self.runner, self.policy)}

    def save(self, path, state, report=None, *, origin_seconds=None, safety=None):
        if not isinstance(state, ResidentStateHandle):
            raise ContractError('Resident cache requires a resident state, not a CPU/default checkpoint')
        origin_seconds = _origin(self.runner, state, origin_seconds)
        if report is not None:
            if report.get('resident_execution') != _execution_record(self.runner):
                raise ContractError('Background report has no matching resident execution identity')
            if report.get('converged'):
                self.require_convergence({'report': report, 'absolute_time_seconds': state.time_seconds,
                    'preparation_origin_seconds': float(origin_seconds), 'binding': self.binding,
                    'binding_hash': self.binding_hash})
        return super().save(path, state, report, origin_seconds=origin_seconds, safety=safety)

    def load(self, path, *, require_converged=False, decode_mode=None, diagnostic_path=None):
        # Explicitly bypass environment selection of a CPU parsed decoder, never
        # bypass its validations. Original loader auto-detects custom restore and
        # invokes runner.restore, which checks resident model + QA binding.
        if decode_mode not in (None, 'auto', 'legacy'):
            raise ContractError('Resident cache must use its identity-checking public restore hook')
        state, value = super().load(path, require_converged=require_converged,
                                    decode_mode='legacy', diagnostic_path=diagnostic_path)
        _origin(self.runner, state, value['preparation_origin_seconds'])
        return state, value

    def require_convergence(self, value):
        if value.get('binding') != self.binding or value.get('binding_hash') != self.binding_hash:
            raise ContractError('Qualified background cache belongs to another resident execution')
        super().require_convergence(value)
        report = value['report']
        if report.get('resident_execution') != _execution_record(self.runner):
            raise ContractError('Runtime export cannot substitute for resident qualification evidence')
        origin, stop = value['preparation_origin_seconds'], value['absolute_time_seconds']
        if origin != self.binding.get('resident_background_origin_seconds'):
            raise ContractError('Background qualification origin differs from the bound session')
        interval = report.get('absolute_preparation_interval_seconds', ())
        if (not math.isfinite(origin) or not math.isfinite(stop) or len(interval) != 2 or
                abs(interval[0] - origin) > 1e-10 or abs(interval[1] - stop) > 1e-10 or
                abs(report.get('stimulus_origin_seconds', float('inf')) - stop) > 1e-10 or
                report.get('stochastic_burnin_completed') is not True):
            raise ContractError('Background qualification interval/burn-in differs from complete state')
        recent = report['windows'][-self.policy.stable_windows:]
        expected_states = self.coverage()['states']
        previous = None
        for row in recent:
            left, right = row['start_seconds'], row['stop_seconds']
            if (not math.isfinite(left) or not math.isfinite(right) or
                    abs(right - left - self.policy.window_seconds) > 1e-10 or
                    (previous is not None and abs(left - previous) > 1e-10) or
                    row.get('monitoring_budget_limited') is not False or row.get('observations', 0) <= 0 or
                    [r.get('state') for r in row.get('states', [])] != expected_states):
                raise ContractError('Background cache lacks complete contiguous monitored windows')
            norms = [r[key] for r in row['states'] for key in (
                'maximum_observed_excursion_norm', 'original_rhs_norm', 'discrete_defect_norm')]
            if (any(not math.isfinite(n) or n < 0 for n in norms) or
                    row['maximum_norm'] != max(norms, default=0.)):
                raise ContractError('Background per-state report and full reduction disagree')
            previous = right
        if previous is None or abs(previous - stop) > 1e-10:
            raise ContractError('Background report is stale relative to the checkpoint state')


def prepare_resident_with_checkpoint(root, runner, reference_state, light_context, policy,
                                     destination, *, state, channels, resume_path=None,
                                     preparation_origin_seconds=None,
                                     deterministic_initialization=None, guard=None,
                                     on_progress=None, on_boundary=None,
                                     checkpoint_interval_seconds=45.):
    """Explicit candidate outer workflow; no default CLI/backend registration.

    Checkpoints occur at the original wall interval and completed window
    boundaries, never at each 100 us monitor observation. Every save exports a
    complete state/history/RNG snapshot. A partial save has report=None and
    cannot pass require_convergence. An existing converged cache is still subject
    to all original and resident identity/coverage/qualification checks.
    """
    _scope(runner, policy)
    _constant_channels(runner, channels)
    if not math.isfinite(checkpoint_interval_seconds) or checkpoint_interval_seconds <= 0:
        raise ContractError('Positive execution checkpoint interval required')
    store = ResidentBackgroundCheckpointStore(root, runner, reference_state, light_context, policy)
    coverage = store.coverage()
    if coverage['missing'] or coverage['covered_blocks'] != coverage['state_blocks']:
        raise ContractError('Incomplete registered background state coverage: ' + str(coverage['missing']))
    destination = Path(destination)
    guard = guard or MemoryGuard()
    state.validate(runner)
    state = state.fork()
    origin = _origin(runner, state, preparation_origin_seconds)
    if resume_path:
        state, cached = store.load(resume_path)
        origin = cached['preparation_origin_seconds']
        if (cached.get('report') or {}).get('converged'):
            store.require_convergence(cached)
            if not guard.check():
                raise ContractError('Memory safety budget reached before reusing prepared stimulus state')
            report = dict(cached['report'], cache_reused=True, prepared_checkpoint_source=str(resume_path))
            store.save(destination, state, report, origin_seconds=origin, safety=guard.last)
            return state, report, store
    origin = _origin(runner, state, origin)
    # No relabelling of arbitrary movie/runtime state as a background phase.
    if any(state.input_position.get(k) != v for k, v in store.expected_context.items()):
        raise ContractError('Resident state is not bound to this declared uniform-background phase')
    if not guard.check():
        raise ContractError('Memory safety budget reached before background checkpoint export')
    store.save(destination, state, origin_seconds=origin, safety=guard.last)
    last_save = perf_counter()
    wall_start = last_save
    last_window = None

    def progress(current):
        nonlocal last_save
        if not guard.check():
            return False
        if on_boundary is not None and on_boundary() is False:
            return False
        if perf_counter() - last_save >= checkpoint_interval_seconds:
            store.save(destination, current, origin_seconds=origin, safety=guard.last)
            last_save = perf_counter()
            if on_progress is not None:
                on_progress(current, {'phase': 'background', 'wall_seconds': last_save - wall_start,
                    'last_window': last_window, 'memory': guard.last})
        return True

    def window(current, row):
        nonlocal last_save, last_window
        last_window = row
        store.save(destination, current, origin_seconds=origin, safety=guard.last)
        last_save = perf_counter()
        if on_progress is not None:
            on_progress(current, {'phase': 'background_window', 'wall_seconds': last_save - wall_start,
                'last_window': row, 'memory': guard.last})

    state, report = prepare_resident_background(
        runner, state, channels, policy, reference_state=reference_state,
        on_chunk=progress, on_window=window, preparation_origin_seconds=origin,
        deterministic_initialization=deterministic_initialization)
    if guard.stop_reason:
        report['budget_exit'] = guard.stop_reason
    report.update(coverage=coverage, safety=dict(guard.last, cooperative_boundary_checks=True),
                  complete_state_checkpoint=str(destination), cache_reused=False)
    value = store.save(destination, state, report, origin_seconds=origin, safety=guard.last)
    if report['converged']:
        store.require_convergence(value)
    return state, report, store
