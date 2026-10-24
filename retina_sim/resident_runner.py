"""Independent resident-native QA runner; never registered as a default backend.

UNVALIDATED: implementation only. No automatic run, warmup, compilation or
background preparation is launched by this file. Explicit construction wraps an
already prepared compact runner; it does not assemble a graph or replace it.
"""
from copy import copy, deepcopy
from dataclasses import asdict
from pathlib import Path
from threading import RLock
import hashlib
import time
import numpy as np

from .contracts import ContractError, canonical_hash
from .resident_evaluation import ResidentEvaluator
from .resident_state import ResidentStateHandle


BACKEND = "cuda_resident_native_cpu_inner_joint_dop853_float64_v1"


class ResidentNativeRunner:
    """One joint solver, existing CPU inner, device-native monitor and history.

    Selected ordinary SimulationResult recording exports only requested rows.
    Explicit device observers retain the complete monitoring interface; legacy
    NumPy observers are not silently emulated by downloading the full state.
    """

    @classmethod
    def from_runner(cls, reference, *, purpose, numerical_contract=None):
        from .gpu_compact import CompactCudaHybridRunner, compact_release_layout
        from .resident_contract import ResidentNativeContract

        from .source_graph_contract import SourceGraphContract
        source_binding = purpose == 'approved_source_binding' and type(numerical_contract) is SourceGraphContract
        if purpose != "numerical_qa" and not source_binding:
            raise ContractError("Resident backend is unvalidated; only explicit numerical_qa binding exists")
        if type(reference) is not CompactCudaHybridRunner:
            raise ContractError("Resident runner requires the unchanged compact reference, not GPU-inner")
        contract = numerical_contract or ResidentNativeContract()
        contract.validate_reference(reference)
        self = cls.__new__(cls)
        self.original = reference
        self.compiled = reference.compiled
        self.model = copy(reference.model)
        self.torch = reference.partition.torch
        # Immutable CSR/constants may be shared; CPU work and input caches may not.
        self.partition = copy(reference.partition)
        self.partition.work = reference.partition.work.copy()
        self.partition.statistics = dict.fromkeys(reference.partition.statistics, 0)
        self.partition._light_cpu = None
        self.partition._light_gpu = None
        self.nstate = sum(spec.shape[0] for spec in self.model.state_specs)
        self.contract = contract
        self.discrete = None
        self._lock = RLock()
        self.workspace = None
        self._binding = None
        for name in ("history_dt", "strides", "event_clock", "periods",
                     "delayed_step", "observation_dt"):
            setattr(self, name, getattr(reference, name))
        if not self.event_clock:
            raise ContractError("Resident initial scope requires the existing explicit event clock")
        native_cid = self.partition.cid
        if self.compiled.components[native_cid].time_contract.kind != "continuous":
            raise ContractError("Resident event patches require continuous native states")
        if native_cid in self.periods:
            raise ContractError("Native state cannot be in the discrete event write set")
        if any(native_cid in (left, right) for left, right, *_ in self.model._gap_plans):
            raise ContractError("Resident partition cannot omit native gap feedback")
        for link in self.compiled.links:
            s = link.specification
            if (link.delayed_operator is not None or s.target_component == native_cid or
                    (s.source_component == native_cid and s.source_port != "release")):
                raise ContractError("Unreviewed resident native/delayed link")
        used, incoming, _ = compact_release_layout(
            self.model, native_cid, self.partition.native.nc)
        if not np.array_equal(used, self.partition.used):
            raise ContractError("Resident compact release layout differs")
        self.evaluator = ResidentEvaluator(self, incoming)
        names = (
            "resident_runner.py", "resident_state.py", "resident_evaluation.py",
            "resident_dop853.py", "resident_monitor.py", "resident_delay.py",
            "resident_contract.py", "resident_recording.py", "resident_background.py",
        )
        identity = {
            "backend": BACKEND, "reference_backend_hash": reference.model.model_hash,
            "contract_hash": contract.sha256, "source_pins_hash": contract.source_pins_hash,
            "implementation_sha256": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in names
            },
            "dtype": "float64", "solver_controller": "original_restart_each_interval",
            "inner": "unchanged_compact_CPU_NumBa", "qualification": "UNVALIDATED",
            "recording": "selected_SimulationResult_with_explicit_device_observer",
        }
        self.model.model_hash = canonical_hash(identity)
        self.model.compute_metadata = dict(reference.model.compute_metadata,
                                           integration_backend=identity)
        self.integration_metrics = dict(reference.integration_metrics)
        for key in self.integration_metrics:
            self.integration_metrics[key] = 0
        self.integration_metrics.update(accepted_steps=0, rejected_attempts=0,
                                        dense_polynomials=0)
        return self

    def validate_solver(self, config):
        self.contract.validate_solver(config)

    def import_reference_state(self, snapshot, *, reference_state, channels,
                               channel_data_hashes, monitor_policy):
        """Explicit QA import; never weakens original restore/hash rejection.

        Both snapshots are checked against the original model. The reference
        state fixes monitor scales and is included in the prospective binding.
        Importing diagnostic progress does not create background qualification.
        """
        from .engine import validate_state

        with self._lock:
            validate_state(snapshot.dynamic, self.original.model)
            validate_state(reference_state.dynamic, self.original.model)
            self.validate_solver(snapshot.solver_config)
            if reference_state.solver_config != snapshot.solver_config:
                raise ContractError("Resident QA monitor reference solver differs")
            binding = self.contract.bind_inputs(
                self.original, snapshot, reference_state, channels,
                channel_data_hashes, monitor_policy)
            if self._binding is not None and binding != self._binding:
                raise ContractError("Resident runner is already bound to another QA scenario")
            handle = ResidentStateHandle.upload(snapshot, self)
            handle.input_position["resident_qa_binding_hash"] = binding["binding_hash"]
            handle.validate(self)
            self._binding = binding
            self._channel_evaluators = {
                name: channel.evaluator for name, channel in channels.items()
            }
            return handle

    @property
    def binding(self):
        return None if self._binding is None else deepcopy(self._binding)

    def bind_continuation_inputs(self, state, channels, channel_data_hashes, *, phase):
        """Explicit background-to-movie boundary, retaining the complete state.

        A full export here binds the continuation starting state once. It is not
        an interval-end transfer, a new monitor scale, or fitting approval.
        Restoring its later checkpoint requires recreating this same phase chain.
        Prepared-content digests must be computed from actual input content.
        """
        import re
        from .resident_contract import _channel_manifest

        with self._lock, self.torch.no_grad():
            state.validate(self)
            if self._binding is None or state.input_position.get('resident_qa_binding_hash') != self._binding['binding_hash']:
                raise ContractError('Continuation state is not bound to this runner')
            if phase != 'movie':
                raise ContractError('Only the explicit background-to-movie continuation is supported')
            expected = {e.channel for e in self.compiled.spec.external_inputs}
            if (set(channels) != expected or set(channel_data_hashes) != expected or
                    any(not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v)
                        for v in channel_data_hashes.values())):
                raise ContractError('Continuation requires complete actual prepared-content SHA256 bindings')
            snapshot_hash = canonical_hash(self.export_state(state))
            binding = deepcopy(self._binding)
            binding.update(parent_binding_hash=self._binding['binding_hash'],
                           continuation_state_and_aux_hash=snapshot_hash,
                           continuation_time_seconds=state.time_seconds, continuation_phase=phase,
                           channel_metadata_hash=canonical_hash(_channel_manifest(channels)),
                           channel_data_hashes=dict(channel_data_hashes))
            binding.pop('binding_hash')
            binding['binding_hash'] = canonical_hash(binding)
            continued = state.fork()
            continued.input_position.update(phase=phase, resident_qa_binding_hash=binding['binding_hash'])
            self._binding = binding
            self._channel_evaluators = {name: channel.evaluator for name, channel in channels.items()}
            return continued

    def evaluate(self, t, y, bundle, *, sampling=False, sampled_components=None,
                 require_rhs=False):
        with self._lock, self.torch.no_grad():
            return self.evaluator(
                t, y, bundle, sampling=sampling,
                sampled_components=sampled_components, require_rhs=require_rhs)

    def make_monitor(self, reference_state, policy, **kwargs):
        from .resident_monitor import ResidentWindowMonitor
        from .engine import validate_state
        from .background import NUMERICAL_UNIT_SCALES

        validate_state(reference_state.dynamic, self.original.model)
        if self._binding is None:
            raise ContractError("Bind the complete QA initial/reference state before monitoring")
        if canonical_hash(reference_state) != self._binding["monitor_reference_hash"]:
            raise ContractError("Resident monitor reference differs from frozen binding")
        if canonical_hash(asdict(policy)) != self._binding["monitor_policy_hash"]:
            raise ContractError("Resident monitor policy differs from frozen binding")
        if (kwargs.get("explicit_scales") is not None or
                kwargs.get("absolute_fraction") is not None or
                (kwargs.get("unit_scales") is not None and
                 canonical_hash(kwargs["unit_scales"]) != canonical_hash(NUMERICAL_UNIT_SCALES))):
            raise ContractError("Resident QA requires the complete original numerical unit registry")
        monitor = ResidentWindowMonitor(self, reference_state, policy, **kwargs)
        if monitor.missing or len(monitor.rows) != len(self.model.state_specs):
            raise ContractError("Resident QA requires complete monitor state coverage")
        return monitor

    def _sample(self, state, stimulus):
        t = state.time_seconds
        due = {
            cid for cid, period in self.periods.items()
            if abs(state.grid_origin_seconds + state.component_tick_indices[cid] * period - t) < 1e-12
        }
        evidence = self.evaluate(t, state.y, stimulus.at(t), sampling=True,
                                 sampled_components=due)
        changed = evidence.inner_y.copy()
        for cid, component in self.compiled.components.items():
            if cid == self.partition.cid:
                continue
            local = None
            if component.time_contract.kind == "causal_history" and cid in due:
                local = component.native_model.transition(
                    evidence.local_blocks(cid), evidence.inputs[cid],
                    {port: evidence.outputs[(cid, port)] for port in component.outputs})
            if (component.time_contract.kind == "author_discrete" and cid in due and
                    t > state.grid_origin_seconds + 1e-12):
                previous = t - component.time_contract.fixed_step_seconds
                external = next((e for e in self.compiled.spec.external_inputs
                                 if e.target_component == cid and e.target_port == "light"), None)
                if external is None:
                    raise ContractError("Author light requires explicit external total-rate input")
                channel = stimulus.channels[external.channel]
                if (channel.interpolation != "zero_order_hold" or
                        channel.input_contract.get("grid_step_seconds") != component.time_contract.fixed_step_seconds or
                        channel.input_contract.get("includes_background") is not True):
                    raise ContractError("Mixed author input requires exact total-rate ZOH grid")
                if previous < channel.start_seconds - 1e-12:
                    raise ContractError("Author interval-start input unavailable; preserve input across chunk boundary")
                incoming = {"light": channel.at(max(channel.start_seconds, previous + 1e-13))}
                local = component.native_model.transition(evidence.local_blocks(cid), incoming)
            if local is not None:
                sl = self.model.slices[cid]
                changed[sl.start-self.partition.offset:sl.stop-self.partition.offset] = np.concatenate(local)
        # Copy-on-write protects the input handle and earlier observer snapshots.
        patched = state.y.clone()
        for cid in due:
            component = self.compiled.components[cid]
            if component.time_contract.kind not in ("causal_history", "author_discrete"):
                continue
            sl = self.model.slices[cid]
            if sl.start < self.partition.offset:
                raise ContractError("Discrete patch would modify continuous native state")
            values = changed[sl.start-self.partition.offset:sl.stop-self.partition.offset]
            patched[sl].copy_(self.torch.as_tensor(
                values, dtype=self.torch.float64, device=patched.device))
        state.commit(patched, t, self)
        for cid in due:
            state.component_tick_indices[cid] += 1
        for cid, aux in list(state.spike_states.items()):
            if cid not in due:
                continue
            sl = self.model._local_slices[cid][0]
            voltage = changed[sl.start-self.partition.offset:sl.stop-self.partition.offset]
            updated, _ = self.compiled.components[cid].native_model.sample_spikes(t, voltage, aux)
            state.spike_states[cid] = updated
        state.sample_index += 1

    def _integrate(self, state, stimulus, stop, config, samples):
        from .resident_dop853 import Dop853Workspace, integrate_device

        start = state.time_seconds
        if self.workspace is None:
            self.workspace = Dop853Workspace(state.y, config.atol)
        def rhs(t, values):
            # Shared repaired reference semantics: only actual ZOH channels
            # query the left side; linear/analytic channels keep the endpoint.
            return self.partition(t, values, stimulus.at(t, left_limit=bool(t >= stop)))
        requested = np.array(sorted(set(float(x) for x in samples if start < x <= stop)))
        endpoint_only = not np.any(requested < stop)
        targets = None if endpoint_only else np.array(sorted(set(requested.tolist() + [float(stop)])))
        tick = time.perf_counter()
        result = integrate_device(
            rhs, start, state.y, stop, rtol=config.rtol, atol=config.atol,
            max_step=config.max_step, t_eval=targets, workspace=self.workspace)
        metrics = self.integration_metrics
        metrics["solver_calls"] += 1
        metrics["nfev"] += result["nfev"]
        metrics["integration_wall_seconds"] += time.perf_counter() - tick
        metrics["minimum_interval_seconds"] = min(
            metrics["minimum_interval_seconds"] or stop-start, stop-start)
        metrics["maximum_nfev_per_interval"] = max(
            metrics["maximum_nfev_per_interval"], result["nfev"])
        metrics["accepted_steps"] += result["steps"]
        metrics["rejected_attempts"] += result["rejected_attempts"]
        metrics["dense_polynomials"] += result["dense_polynomials"]
        rows = {float(stop): result["final"]} if endpoint_only else {
            float(t): value for t, value in zip(result["t"], result["samples"])
        }
        persisted = result["final"] if endpoint_only else result["samples"][-1]
        state.commit(persisted, stop, self)
        return rows

    def _update_observation_delays(self, state, t, y):
        for cid, delay in state.delay_states.items():
            if t > delay.history_times[-1]:
                component = self.compiled.components[cid]
                names = [spec.state_name for spec in component.state_specs]
                sl = self.model._local_slices[cid][names.index("Vh")]
                port = component.outputs["Vh"]
                delay.append(t, y[sl], port.unit, port.meaning)

    def advance(self, state, channels, stop, solver_config=None, sample_times=None,
                recording_plan=None, state_observer=None):
        """Original event schedule with selected recording; observers receive device y.

        No full-state CPU conversion, no automatic background qualification and
        no implicit support for a legacy NumPy observer.
        """
        from .network import NetworkStimulus
        from .recording import RecordingPlan
        from .resident_recording import ResidentRecording

        if not isinstance(state, ResidentStateHandle) or self._binding is None:
            raise ContractError("Explicitly imported resident state required")
        if type(recording_plan) is not RecordingPlan:
            raise ContractError("Resident QA advance requires an explicit RecordingPlan")
        if state_observer is None:
            raise ContractError("Resident QA advance requires an explicit full device observer")
        with self._lock, self.torch.no_grad():
            state.validate(self)
            config = solver_config or state.solver_config
            if config != state.solver_config:
                raise ContractError("Solver changes across chunks require explicit reinitialization")
            self.contract.validate_channels(channels, self._binding)
            if (set(channels) != set(self._channel_evaluators) or any(
                    channels[name].evaluator is not evaluator
                    for name, evaluator in self._channel_evaluators.items())):
                raise ContractError("Resident QA input evaluator was replaced")
            if state.input_position.get("resident_qa_binding_hash") != self._binding["binding_hash"]:
                raise ContractError("Resident state belongs to a different frozen QA binding")
            start = state.time_seconds
            if not np.isfinite(stop) or stop <= start:
                raise ContractError("Advance must increase absolute time")
            stimulus = NetworkStimulus(
                self.compiled, channels, state, stop, config.input_interpolation,
                self.model.evidence_tag)
            if stimulus.delay_buffers:
                raise ContractError("Resident initial scope excludes explicit delayed links")
            times = np.array([start, stop] if sample_times is None else sample_times, dtype=float)
            if (times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)) or
                    np.any(np.diff(times) <= 0) or times[0] < start or times[-1] > stop):
                raise ContractError("Record times must be increasing within advance")
            state = state.fork()
            recorder = ResidentRecording(self, state, stimulus, recording_plan, times, stop)
            breaks = iter(stimulus.breakpoints_seconds)
            next_break = next(breaks, stop)
            while state.time_seconds < stop-1e-14:
                current = state.time_seconds
                if self.history_dt is not None:
                    grid = min(state.grid_origin_seconds + state.component_tick_indices[cid]*period
                               for cid, period in self.periods.items())
                    if grid < current-1e-10:
                        raise ContractError("History sampling phase lost")
                    if abs(grid-current) < 1e-10:
                        self._sample(state, stimulus)
                recorder.record_requested(current, state.y)
                state_observer(current, state.y, stimulus.at(current))
                while next_break <= current+1e-14:
                    next_break = next(breaks, stop)
                next_grid = (min(state.grid_origin_seconds + state.component_tick_indices[cid]*period
                                 for cid, period in self.periods.items())
                             if self.history_dt is not None else stop)
                next_observation = (state.grid_origin_seconds +
                    (int(np.floor((current-state.grid_origin_seconds)/self.observation_dt+1e-8))+1)*self.observation_dt
                    if self.observation_dt else stop)
                next_delay = (state.grid_origin_seconds +
                    (int(np.floor((current-state.grid_origin_seconds)/self.delayed_step+1e-8))+1)*self.delayed_step
                    if self.delayed_step else stop)
                target = min(float(stop), float(next_break), float(next_grid),
                             float(next_observation), float(next_delay))
                rows = self._integrate(state, stimulus, target, config, times)
                recorder.flush_events(target)
                for t, y in rows.items():
                    self._update_observation_delays(state, t, y)
                    state_observer(t, y, stimulus.at(t))
                    if abs(t-next_grid) > 1e-10:
                        recorder.record_requested(t, y)
            final_tick = (min(state.grid_origin_seconds + state.component_tick_indices[cid]*period
                              for cid, period in self.periods.items())
                          if self.history_dt is not None else None)
            if final_tick is not None and abs(final_tick-stop) < 1e-12:
                self._sample(state, stimulus)
            state_observer(stop, state.y, stimulus.at(stop))
            if not recorder.recorded or recorder.recorded[-1] < times[-1]-1e-12:
                recorder.record(float(times[-1]), state.y)
            recorder.flush_events(stop)
            state.input_position.update(last_time_seconds=float(stop), contracts=stimulus.input_contract)
            return state, recorder.finish(config)

    def stream(self, state, channels, stop, chunk_seconds, recorder, *,
               recording_plan, state_observer, sample_times=None, on_boundary=None):
        """Explicit bounded movie windows; no automatic backend selection.

        The caller supplies the original absolute recording timestamps. An
        optional cooperative guard can stop before any new integration window.
        This method does not create fitting eligibility or background evidence.
        """
        if not np.isfinite(chunk_seconds) or chunk_seconds <= 0:
            raise ContractError("Positive stream window required")
        samples = None if sample_times is None else np.asarray(sample_times, dtype=float)
        if samples is not None and (samples.ndim != 1 or not np.all(np.isfinite(samples)) or
                                    np.any(np.diff(samples) <= 0) or
                                    np.any(samples < state.time_seconds) or np.any(samples > stop)):
            raise ContractError("Movie recording timestamps must be increasing in the stream interval")
        while state.time_seconds < stop:
            if on_boundary is not None and on_boundary() is False:
                return state
            end = min(float(stop), state.time_seconds + chunk_seconds)
            selected = None if samples is None else samples[(samples >= state.time_seconds) & (samples <= end)]
            plan = recording_plan
            if selected is not None and not len(selected):
                raise ContractError('Each recorded stream window needs an original sample timestamp; align chunks instead of dropping spike events')
            state, result = self.advance(state, channels, end, sample_times=selected,
                recording_plan=plan, state_observer=state_observer)
            if result is not None:
                recorder.consume(result)
            if samples is not None:
                samples = samples[samples > end]
        return state

    def export_state(self, state):
        with self._lock, self.torch.no_grad():
            return state.export(self)

    def checkpoint(self, state):
        """Use the unchanged complete RetinaState codec after explicit export."""
        from .hybrid import HybridRunner

        return HybridRunner.checkpoint(self, self.export_state(state))

    def restore(self, serialized, *, expected_input_context=None):
        from .hybrid import HybridRunner

        if self._binding is None:
            raise ContractError("Bind the original QA reference/inputs before resident checkpoint restore")
        with self._lock, self.torch.no_grad():
            snapshot = HybridRunner.restore(
                self, serialized, expected_input_context=expected_input_context)
            if snapshot.input_position.get("resident_qa_binding_hash") != self._binding["binding_hash"]:
                raise ContractError("Resident checkpoint belongs to another QA scenario")
            self.validate_solver(snapshot.solver_config)
            state = ResidentStateHandle.upload(snapshot, self)
            state.validate(self)
            return state
