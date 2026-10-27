"""Optional disabled Cone--HC block runner, independent of both old backends.

Implementation only, runtime NOT RUN. No backend registration, source-cache
selection, CUDA initialization or forward execution occurs at module import.
The candidate factory fails while the separate B numerical/input contract is
pending. CPU inner equations, event patches, PCG64 and history use the existing
resident mechanisms; only the continuous numerical partition is new.
"""
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
import hashlib
import json
import math
import time
import numpy as np

from .contracts import ContractError, canonical_hash
from .resident_runner import ResidentNativeRunner
from .resident_state import ResidentStateHandle
from .cone_hc_split_contract import BACKEND, ConeHCSplitContract, guard_block_triangular


CHECKPOINT_VERSION = 'cone-hc-split-resident-checkpoint-1'


@dataclass
class ConeHCSplitState(ResidentStateHandle):
    native_cursor: object = None

    @classmethod
    def from_handle(cls, handle):
        return cls(**vars(handle))

    def fork(self):
        other = super().fork()
        other.native_cursor = None if self.native_cursor is None else self.native_cursor.fork()
        return other


def native_input_at(channel, t, *, left_limit=False):
    """Same repaired per-channel endpoint rule as NetworkStimulus.at."""
    query = t
    if channel.interpolation == 'zero_order_hold':
        if left_limit and t > channel.start_seconds:
            query = np.nextafter(t, -np.inf)
        query = min(query, np.nextafter(channel.stop_seconds, channel.start_seconds))
    return channel.at(query)


def next_native_input_bound(channel, t):
    """Only this light channel constrains native, never rod/BC/observer ticks."""
    future = [float(x) for x in channel.breakpoints_seconds if x > t]
    bound = min(future+[float(channel.stop_seconds)])
    if bound <= t:
        raise ContractError('Native input is exhausted; no extrapolation or implicit new phase')
    return bound, bound in channel.breakpoints_seconds


class ConeHCSplitRunner(ResidentNativeRunner):
    """Persistent native ahead; inner consumes only its accepted dense horizon.

    This reuses complete original evaluation for monitoring and sampling. Native
    stage RHS stays native-only. Inner stage RHS uses original compact CPU plan
    plus the nonlinear readout of selected dense native state, without invoking
    native RHS. Neither callbacks nor input chunk ends restart native.
    """

    @classmethod
    def from_resident(cls, retained, *, purpose, numerical_contract=None):
        # Keep this BEFORE any tensor operation/allocation or sparse-plan build.
        contract = numerical_contract or ConeHCSplitContract()
        if type(contract) is not ConeHCSplitContract:
            raise ContractError('Split runner requires its own explicit B contract')
        contract.require_execution_approval()
        from .cone_hc_split_contract import USEFUL_SCOPE
        useful = (purpose == 'bounded_source' and
                  (contract.runtime_authorization or {}).get('scope') == USEFUL_SCOPE)
        if (purpose != 'numerical_qa' and not useful) or type(retained) is not ResidentNativeRunner:
            raise ContractError('Split QA wraps the retained resident runner only')
        if not hasattr(retained, 'native_sparse_plan_policy'):
            raise ContractError('Use the retained CSR ALG2 resident baseline, not the unadopted pointwise trial')
        if contract.runtime_authorization is not None and (
                retained.model.model_hash != contract.runtime_authorization['source_resident_model_hash']
                or retained.binding['binding_hash'] != contract.runtime_authorization['baseline_binding_hash']):
            raise ContractError('One-off split source resident model/binding differs')
        contract.validate_reference(retained.original)
        from .gpu_compact import compact_release_layout
        from .resident_evaluation import ResidentEvaluator
        from .cone_hc_split_readout import NativeDenseRelease

        obj = cls.__new__(cls)
        obj.__dict__ = dict(retained.__dict__)
        # Arm-specific QA observer/integrator hooks belong to the resident arm,
        # not to a fresh split instance (notably a cross-permit restore runner).
        obj.__dict__.pop('_integrate', None)
        obj.__dict__.pop('_update_observation_delays', None)
        from .compiled_kernel import fork_partition
        obj.model, obj.partition = copy(retained.model), fork_partition(retained.partition)
        obj.partition.statistics = dict.fromkeys(retained.partition.statistics, 0)
        obj.partition._light_cpu = obj.partition._light_gpu = None
        obj.contract, obj._binding, obj._lock = contract, None, RLock()
        obj._retained_model_hash = retained.model.model_hash
        obj._channel_evaluators = {}
        obj.workspace = None
        obj.native_workspace = None
        obj.runtime_guard = None
        obj.split_graph_guard = guard_block_triangular(obj)
        native_model = obj.compiled.components[obj.partition.cid].native_model
        if (obj.observation_dt != contract.candidate['observation']['common_grid_seconds']
                or native_model.observation_interval_seconds != obj.observation_dt
                or native_model.p['observation_delay']*.001 != contract.candidate['observation']['native_readout_delay_seconds']):
            raise ContractError('Split proposal does not authorize a different native observation grid/delay')
        _, incoming, _ = compact_release_layout(obj.model, obj.partition.cid, obj.partition.native.nc)
        obj.evaluator = ResidentEvaluator(obj, incoming)
        obj.dense_release = NativeDenseRelease(obj)
        names = ('cone_hc_split_contract.py', 'cone_hc_split_solver.py',
                 'cone_hc_split_readout.py', 'cone_hc_split_runner.py', 'cone_hc_split_checkpoint.py')
        identity = dict(backend=BACKEND, retained_model_hash=retained.model.model_hash,
                        split_contract_sha256=contract.sha256,
                        candidate_manifest_sha256=contract.candidate_sha256,
                        implementation_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                               for name in names},
                        graph_guard=obj.split_graph_guard, readout_sha256=obj.dense_release.sha256,
                        qualification='NOT_RUN', classification='B', dtype='float64',
                        native_solver=dict(contract.candidate['native']),
                        inner_solver=dict(contract.candidate['inner']),
                        observation_contract=dict(contract.candidate['observation']),
                        inner_error_norm=contract.candidate['inner']['controller'],
                        native_error_norm=contract.candidate['native']['controller'])
        obj.model.model_hash = canonical_hash(identity)
        obj.model.compute_metadata = dict(retained.model.compute_metadata, integration_backend=identity)
        obj.integration_metrics = dict.fromkeys(retained.integration_metrics, 0)
        obj.split_metrics = dict(native_nfev=0, native_accepts=0, inner_nfev=0,
                                 inner_accepts=0, compact_release_D2H_bytes=0,
                                 native_input_restarts=0)
        return obj

    def import_reference_state(self, *args, **kwargs):
        handle = super().import_reference_state(*args, **kwargs)
        return ConeHCSplitState.from_handle(handle)

    def import_resident_state(self, state, *, retained, reference_state, channels,
                              channel_data_hashes, monitor_policy):
        """Explicit QA bridge from a VALIDATED retained state, without relabelling
        its old serialized checkpoint or inventing a compact-reference identity.
        The new binding retains the actual resident source snapshot hash.
        """
        from .engine import validate_state
        self.contract.require_execution_approval()
        with self._lock, self.torch.no_grad():
            if type(retained) is not ResidentNativeRunner or retained.model.model_hash != self._retained_model_hash:
                raise ContractError('Split QA import source is not the retained resident identity')
            state.validate(retained)
            validate_state(reference_state.dynamic, self.original.model)
            snapshot = retained.export_state(state)
            binding = self.contract.bind_inputs(self.original, snapshot, reference_state, channels,
                                                channel_data_hashes, monitor_policy)
            binding.pop('binding_hash')
            binding['actual_import_source_model_hash'] = retained.model.model_hash
            binding['binding_hash'] = canonical_hash(binding)
            if self._binding is not None and binding != self._binding:
                raise ContractError('Split runner is already bound to another import')
            handle = ConeHCSplitState.from_handle(ResidentStateHandle.upload(snapshot, self))
            handle.input_position['resident_qa_binding_hash'] = binding['binding_hash']
            handle.validate(self)
            self._binding = binding
            self._channel_evaluators = {n: c.evaluator for n, c in channels.items()}
            return handle

    def _cursor(self, state):
        from .cone_hc_split_solver import PersistentNativeDop853
        from .cone_hc_split_readout import AcceptedNativeCursor
        if state.native_cursor is None:
            p = self.contract.candidate['native']
            solver = PersistentNativeDop853(state.time_seconds, state.y[:self.partition.offset],
                                           **{k: p[k] for k in ('rtol', 'atol', 'max_step')})
            state.native_cursor = AcceptedNativeCursor(solver, self.dense_release, self._binding['binding_hash'])
        if state.native_cursor.source_binding != self._binding['binding_hash']:
            raise ContractError('Native lookahead belongs to another input phase')
        return state.native_cursor

    def _check_runtime(self):
        self.contract.require_execution_approval()
        if self.runtime_guard is not None and self.runtime_guard() is False:
            raise ContractError('Split runtime resource/wall guard requested a safe stop')

    def _accept_native_until(self, state, stimulus, target):
        from .resident_dop853 import Dop853Workspace
        cursor = self._cursor(state)
        if self.native_workspace is None:
            self.native_workspace = Dop853Workspace(cursor.solver.y, cursor.solver.atol)
        channel = stimulus.channels[self.partition.native_external.channel]

        def rhs(t, y, *, left_limit):
            light = native_input_at(channel, t, left_limit=left_limit)
            self.partition.native_port.validate(light)
            derivative, _ = self.partition.native(y, self.partition._light(light))
            return derivative

        before = dict(cursor.solver.statistics)
        while cursor.solver.t < target:
            self._check_runtime()
            bound, discontinuity = next_native_input_bound(channel, cursor.solver.t)
            segment = cursor.solver.accept_next(rhs, bound, self.native_workspace,
                                                discontinuity=discontinuity)
            cursor.append(segment)
        after = cursor.solver.statistics
        self.split_metrics['native_nfev'] += after['nfev']-before['nfev']
        self.split_metrics['native_accepts'] += after['accepted_steps']-before['accepted_steps']
        self.split_metrics['native_input_restarts'] += after['input_restarts']-before['input_restarts']
        return cursor

    def _inner_rhs(self, cursor, stimulus, interval_stop, t, inner):
        part = self.partition
        if not np.all(np.isfinite(inner)):
            raise ContractError('Nonfinite split inner state')
        left = bool(t >= interval_stop)
        bundle = stimulus.at(t, left_limit=left)
        values = []
        for external, port in part.external:
            value = bundle.channels[external.channel]
            port.validate(value)
            values.append(value.values)
        release = cursor.release_at(t, left_limit=left).detach().cpu().numpy().copy()
        values.append(release)
        self.split_metrics['compact_release_D2H_bytes'] += release.nbytes
        result = np.asarray(part.fn(np.asarray(inner, dtype=np.float64), np.concatenate(values),
                                    part.work, part.program, part.weights, part.indices,
                                    part.pointers, part.derivatives), dtype=np.float64).copy()
        if result.shape != inner.shape or not np.all(np.isfinite(result)):
            raise ContractError('Nonfinite or malformed original compact CPU inner derivative')
        return result

    def _integrate(self, state, stimulus, stop, config, samples):
        """Inner interval keeps its old event limit; native never uses that bound."""
        from scipy.integrate import DOP853
        from .solver_lifetime import collection_epoch, release_solver_cycles
        start = state.time_seconds
        tick = time.perf_counter()
        cursor = self._accept_native_until(state, stimulus, stop)
        inner = state.y[self.partition.offset:].detach().cpu().numpy().copy()
        rhs = lambda t, y: self._inner_rhs(cursor, stimulus, stop, t, y)
        epoch = collection_epoch()
        solver = DOP853(rhs, start, inner, stop, rtol=config.rtol, atol=config.atol, max_step=config.max_step)
        requested = {float(t) for t in samples if start < t <= stop}
        requested.update(s.stop for s in cursor.segments if start < s.stop <= stop)
        rows = {}
        accepts = 0
        while solver.status == 'running':
            self._check_runtime()
            previous = solver.t
            message = solver.step()
            if solver.status == 'failed':
                raise ContractError('Split CPU inner DOP853 failed: '+str(message))
            accepts += 1
            targets = sorted({solver.t} | {t for t in requested if previous < t <= solver.t})
            dense = solver.dense_output() if any(t < solver.t for t in targets) else None
            for t in targets:
                local = solver.y.copy() if t == solver.t else np.asarray(dense(t), dtype=np.float64).copy()
                native = cursor.state_at(t)
                rows[float(t)] = self.torch.cat((native, self.torch.as_tensor(
                    local, dtype=self.torch.float64, device=state.y.device)))
        state.commit(rows[float(stop)], stop, self)
        cursor.acknowledge('inner', stop)
        self.split_metrics['inner_nfev'] += solver.nfev
        self.split_metrics['inner_accepts'] += accepts
        metrics = self.integration_metrics
        metrics['solver_calls'] += 1
        metrics['nfev'] += solver.nfev  # native calls are separately, explicitly accounted
        metrics['integration_wall_seconds'] += time.perf_counter()-tick
        metrics['accepted_steps'] += accepts
        metrics['minimum_interval_seconds'] = min(metrics['minimum_interval_seconds'] or stop-start, stop-start)
        metrics['maximum_nfev_per_interval'] = max(metrics['maximum_nfev_per_interval'], solver.nfev)
        del solver, dense
        cleanup = release_solver_cycles(epoch)
        metrics['solver_lifetime_gc_wall_seconds'] = metrics.get('solver_lifetime_gc_wall_seconds', 0.)+cleanup['wall_seconds']
        metrics['integration_wall_seconds'] += cleanup['wall_seconds']
        return rows

    def _update_observation_delays(self, state, t, y):
        # New block accepted endpoints are monitoring times, not extra delay
        # samples. Queue linear interpolation/prehistory remains on the old grid.
        if t > state.time_seconds:
            raise ContractError('Native lookahead cannot populate the public observation delay queue')
        original_dense_sample = t in getattr(self, '_queue_sample_times', ())
        original_interval_stop = t == getattr(self, '_queue_interval_stop', None)
        if original_dense_sample or original_interval_stop:
            super()._update_observation_delays(state, t, y)
        elif self.observation_dt:
            index = round((t-state.grid_origin_seconds)/self.observation_dt)
            grid = state.grid_origin_seconds+index*self.observation_dt
            if abs(t-grid) <= 1e-12:
                super()._update_observation_delays(state, t, y)

    @staticmethod
    def _is_input_break(stimulus, t):
        return any(t in channel.breakpoints_seconds for channel in stimulus.channels.values())

    def _observe(self, observer, stimulus, t, y):
        if self._is_input_break(stimulus, t):
            observer(t, y, stimulus.at(t, left_limit=True))
        observer(t, y, stimulus.at(t))

    def _acknowledge_complete(self, state):
        if state.native_cursor is not None:
            for consumer in state.native_cursor.CONSUMERS:
                state.native_cursor.acknowledge(consumer, state.time_seconds)

    def advance(self, state, channels, stop, solver_config=None, sample_times=None,
                recording_plan=None, state_observer=None):
        """Original clocks plus complete accepted-endpoint/input-side monitoring.

        Monitor callbacks are synchronous and finish before buffer pruning.
        Observation snapshots are appended only when inner reaches their old
        common grid, never when native happens to accept a future state.
        """
        from .network import NetworkStimulus
        from .recording import RecordingPlan
        from .resident_recording import ResidentRecording
        self.contract.require_execution_approval()
        if (type(state) is not ConeHCSplitState or self._binding is None
                or type(recording_plan) is not RecordingPlan or state_observer is None):
            raise ContractError('Split advance requires explicit split state, recording and complete observer')
        with self._lock, self.torch.no_grad():
            state.validate(self)
            if state.native_cursor is not None:
                visible = state.native_cursor.state_at(state.time_seconds)
                if not self.torch.equal(visible.view(self.torch.int64), state.y[:self.partition.offset].view(self.torch.int64)):
                    raise ContractError('Public native state differs from its owned accepted dense state')
            config = solver_config or state.solver_config
            if config != state.solver_config:
                raise ContractError('Split numerical policy cannot change between chunks')
            self.validate_solver(config)
            self.contract.validate_channels(channels, self._binding)
            if (set(channels) != set(self._channel_evaluators) or any(
                    channels[n].evaluator is not self._channel_evaluators[n] for n in channels)):
                raise ContractError('Split input evaluator replaced without a new explicit phase')
            if state.input_position.get('resident_qa_binding_hash') != self._binding['binding_hash']:
                raise ContractError('Split state/input QA identity differs')
            start = state.time_seconds
            if not math.isfinite(stop) or stop <= start:
                raise ContractError('Split advance must increase time')
            stimulus = NetworkStimulus(self.compiled, channels, state, stop,
                                       config.input_interpolation, self.model.evidence_tag)
            times = np.asarray([start, stop] if sample_times is None else sample_times, dtype=float)
            if (times.ndim != 1 or not len(times) or not np.all(np.isfinite(times))
                    or np.any(np.diff(times) <= 0) or times[0] < start or times[-1] > stop):
                raise ContractError('Split recording times must be increasing within advance')
            # The old resident also records Vh at explicitly requested dense
            # sample times and interval ends. Preserve those queue knots; only
            # NEW block-acceptance monitor points are excluded from the queue.
            self._queue_sample_times = frozenset(float(t) for t in times)
            state = state.fork()
            recorder = ResidentRecording(self, state, stimulus, recording_plan, times, stop)
            breaks = iter(stimulus.breakpoints_seconds)
            next_break = next(breaks, stop)
            while state.time_seconds < stop:
                self._check_runtime()
                current = state.time_seconds
                # Subsequent interval rows already observed this pre-event
                # state. Do not repeat an identical full RHS/monitor call.
                if current == start:
                    self._observe(state_observer, stimulus, current, state.y)
                grid = min(state.grid_origin_seconds+state.component_tick_indices[cid]*p
                           for cid, p in self.periods.items())
                if grid < current-1e-10:
                    raise ContractError('Split history sampling phase lost')
                if abs(grid-current) < 1e-10:
                    self._sample(state, stimulus)  # unchanged rod/BC/PCG64 ordering
                    self._observe(state_observer, stimulus, current, state.y)
                recorder.record_requested(current, state.y)
                self._acknowledge_complete(state)
                while next_break <= current:
                    next_break = next(breaks, stop)
                next_grid = min(state.grid_origin_seconds+state.component_tick_indices[cid]*p
                                for cid, p in self.periods.items())
                next_obs = (state.grid_origin_seconds+
                    (int(np.floor((current-state.grid_origin_seconds)/self.observation_dt+1e-8))+1)*self.observation_dt
                    if self.observation_dt else stop)
                target = min(float(stop), float(next_break), float(next_grid), float(next_obs))
                self._queue_interval_stop = target
                rows = self._integrate(state, stimulus, target, config, times)
                recorder.flush_events(target)
                for t, y in sorted(rows.items()):
                    self._update_observation_delays(state, t, y)
                    self._observe(state_observer, stimulus, t, y)
                    if abs(t-next_grid) > 1e-10:
                        recorder.record_requested(t, y)
                self._acknowledge_complete(state)
            final_tick = min(state.grid_origin_seconds+state.component_tick_indices[cid]*p
                             for cid, p in self.periods.items())
            if abs(final_tick-stop) < 1e-12:
                self._sample(state, stimulus)
            self._observe(state_observer, stimulus, stop, state.y)
            if not recorder.recorded or recorder.recorded[-1] < times[-1]-1e-12:
                recorder.record(float(times[-1]), state.y)
            recorder.flush_events(stop)
            self._acknowledge_complete(state)
            state.input_position.update(last_time_seconds=float(stop), contracts=stimulus.input_contract,
                                        cone_hc_split_candidate_sha256=self.contract.candidate_sha256)
            return state, recorder.finish(config)

    def replace_useful_runtime_permit(self, contract):
        """Freeze a now-known movie origin without changing science or budget.

        Fresh cold sessions may use a newly authorized overall round. Within an
        active process, replacing a phase permit cannot extend this round.
        """
        from .cone_hc_split_contract import USEFUL_SCOPE
        old = self.contract.runtime_authorization or {}
        if type(contract) is not ConeHCSplitContract:
            raise ContractError('Split phase permit requires the exact contract class')
        contract.require_execution_approval()
        new = contract.runtime_authorization or {}
        from .useful_runtime_budget import goal_only_authorization
        old_goal,new_goal=goal_only_authorization(old),goal_only_authorization(new)
        keys = ('source_resident_model_hash','baseline_binding_hash','input_policy_receipt_sha256',
                'target_graph_receipt_sha256','execution_protocol_sha256','round_started_unix',
                'max_wall_seconds','max_rss_bytes','max_vram_bytes')
        if (old.get('scope') != USEFUL_SCOPE or new.get('scope') != USEFUL_SCOPE
                or self.contract.sha256 != contract.sha256
                or self.contract.candidate_sha256 != contract.candidate_sha256
                or any(old.get(k) != new.get(k) for k in keys)
                or old_goal!=new_goal
                or (old_goal and old.get('user_walltime_override_sha256')!=new.get('user_walltime_override_sha256'))
                or (not old_goal and (new['max_wall_seconds'] > old['max_wall_seconds']
                                     or new['expires_unix'] > old['expires_unix']))
                or any(new[k] > old[k] for k in ('max_rss_bytes','max_vram_bytes'))):
            raise ContractError('Phase permit changed science/inputs/targets or extended the original budget')
        self.contract = contract

    def bind_continuation_inputs(self, state, channels, channel_data_hashes, *, phase):
        # A phase change may discard unconsumed future native trajectory. It must
        # be an explicit reinitialization at public time, not a relabelled old
        # checkpoint or an implicit change in an ordinary stream chunk.
        self.contract.require_execution_approval()
        self.contract.require_input_policy(channels, phase=phase)
        with self._lock, self.torch.no_grad():
            continued = super().bind_continuation_inputs(state, channels, channel_data_hashes, phase=phase)
            continued.native_cursor = None
            return continued

    def checkpoint(self, state):
        """Full ordinary aux/history plus native lookahead AND controller state."""
        from .hybrid import HybridRunner
        if type(state) is not ConeHCSplitState:
            raise ContractError('Split checkpoint cannot migrate an old resident checkpoint')
        with self._lock, self.torch.no_grad():
            state.validate(self)
            cursor = state.native_cursor
            if cursor is not None and any(t != state.time_seconds for t in cursor.consumed.values()):
                raise ContractError('Checkpoint requires all synchronous split consumers to finish')
            payload = dict(model_hash=self.model.model_hash, binding_hash=self._binding['binding_hash'],
                           split_contract_sha256=self.contract.sha256,
                           public_state=json.loads(HybridRunner.checkpoint(self, state.export(self))),
                           native_cursor=None if cursor is None else cursor.payload())
            return json.dumps(dict(version=CHECKPOINT_VERSION, payload=payload,
                                   payload_hash=canonical_hash(payload)), sort_keys=True, allow_nan=False)

    def restore(self, serialized, *, expected_input_context=None):
        from .hybrid import HybridRunner
        from .cone_hc_split_readout import AcceptedNativeCursor
        self.contract.require_execution_approval()
        raw = json.loads(serialized)
        payload = raw.get('payload', {})
        if (self._binding is None or raw.get('version') != CHECKPOINT_VERSION
                or raw.get('payload_hash') != canonical_hash(payload)
                or payload.get('model_hash') != self.model.model_hash
                or payload.get('split_contract_sha256') != self.contract.sha256
                or payload.get('binding_hash') != self._binding['binding_hash']):
            raise ContractError('Split checkpoint version/content/model/contract/input identity differs')
        return self._restore_components(json.dumps(payload['public_state']), payload['native_cursor'],
                                        expected_input_context=expected_input_context)

    def _restore_components(self, public_serialized, native_payload, *, expected_input_context=None):
        """Shared strict restore for verified JSON or atomic binary manifests.

        Callers must verify their outer content envelope. Public model/solver/
        binding and native controller/dense ownership guards remain mandatory.
        A fresh runtime permit is required even when scientific identity matches.
        """
        from .hybrid import HybridRunner
        from .cone_hc_split_readout import AcceptedNativeCursor
        self._check_runtime()
        with self._lock, self.torch.no_grad():
            snapshot = HybridRunner.restore(self, public_serialized,
                                            expected_input_context=expected_input_context)
            if snapshot.input_position.get('resident_qa_binding_hash') != self._binding['binding_hash']:
                raise ContractError('Split checkpoint public input binding differs')
            state = ConeHCSplitState.from_handle(ResidentStateHandle.upload(snapshot, self))
            if native_payload is not None:
                state.native_cursor = AcceptedNativeCursor.from_payload(
                    native_payload, self, state.y[:self.partition.offset],
                    self.contract.candidate['native'], state.time_seconds)
                visible = state.native_cursor.state_at(state.time_seconds)
                if not self.torch.equal(visible.view(self.torch.int64), state.y[:self.partition.offset].view(self.torch.int64)):
                    raise ContractError('Split checkpoint public native state disagrees with accepted dense cursor')
            state.validate(self)
            return state
