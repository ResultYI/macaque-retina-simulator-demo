"""Explicit reduced candidate; GPU native and the unchanged CPU inner clocks.

There is no conversion back to a full old native vector: full cone z1 is absent.
All four consumers finish synchronously before a segment or pinned slot is freed.
"""
from copy import copy, deepcopy
from dataclasses import dataclass, field
from threading import RLock
from time import perf_counter
import hashlib
import math
from pathlib import Path

import numpy as np

from .contracts import ContractError, canonical_hash
from .cone_hc_reduced_native import ReducedNative, ReducedNativeState, METHOD
from .cone_hc_reduced_bridge import BridgeSelection, PinnedSegmentBridge
from .cone_hc_split_runner import native_input_at, next_native_input_bound
from .cone_hc_observation_queue import is_queue_node, queue_contract


CONSUMERS = ('inner', 'monitor', 'recording', 'observation')


@dataclass(frozen=True)
class ReducedPoint:
    native: ReducedNativeState
    inner: np.ndarray
    compact_release: object = None

    def __post_init__(self):
        value = np.array(self.inner, dtype=np.float64, copy=True)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ContractError('Reduced CPU inner state must be finite FP64')
        value.setflags(write=False)
        object.__setattr__(self, 'inner', value)
        if self.compact_release is not None:
            release = np.array(self.compact_release,dtype=np.float64,copy=True)
            if release.ndim!=1 or not np.isfinite(release).all():
                raise ContractError('Reduced endpoint release cache must be finite FP64')
            release.setflags(write=False)
            object.__setattr__(self,'compact_release',release)


@dataclass
class ReducedCursor:
    frontier: ReducedNativeState
    segments: list = field(default_factory=list)
    packets: list = field(default_factory=list)
    consumed: dict = field(default_factory=dict)
    next_id: int = 0

    def fork(self):
        return type(self)(self.frontier.fork(), list(self.segments),
            [p.selection.from_payload(p.payload()) for p in self.packets],
            dict(self.consumed), self.next_id)

    def at(self, t):
        if t == self.frontier.time_seconds:
            return self.frontier.fork()
        for segment in self.segments:
            if segment.start <= t <= segment.stop:
                return ReducedNativeState(t, segment.at(t), segment.anchor,
                                          segment.source_version, False)
        raise ContractError('Reduced query outside accepted native history')

    def packet_at(self, t, *, left=False):
        candidates = [p for p in self.packets if p.start <= t <= p.stop]
        if not candidates:
            raise ContractError('CPU query outside accepted bridge history')
        return candidates[0] if left else candidates[-1]

    def own_packets(self,bridge):
        # A returned/forked state owns immutable copies; it cannot strand a live
        # bridge lease or alias a slot reused by a later advance.
        for i,packet in enumerate(self.packets):
            if packet._lease_check is not None:
                owned = packet.selection.from_payload(packet.payload())
                bridge.release(packet)
                self.packets[i] = owned

    def acknowledge(self, consumer, t, bridge):
        if consumer not in CONSUMERS or not self.consumed[consumer] <= t <= self.frontier.time_seconds:
            raise ContractError('Invalid reduced consumer watermark')
        self.consumed[consumer] = t
        floor = min(self.consumed.values())
        while self.segments and self.segments[0].stop <= floor:
            self.segments.pop(0)
            packet = self.packets.pop(0)
            if packet._lease_check is not None:
                bridge.release(packet)


@dataclass
class ReducedState:
    point: ReducedPoint
    cursor: ReducedCursor
    identity: str
    input_binding: str
    solver_config: object
    grid_origin_seconds: float
    sample_index: int
    spike_states: dict
    delay_states: dict
    input_position: dict
    component_tick_indices: dict
    link_delay_states: dict = field(default_factory=dict)
    generation: int = 0

    @property
    def time_seconds(self):
        return self.point.native.time_seconds

    @property
    def y(self):
        return self.point

    def fork(self):
        return type(self)(self.point, self.cursor.fork(), self.identity, self.input_binding,
            self.solver_config, self.grid_origin_seconds, self.sample_index,
            deepcopy(self.spike_states), {k:v.fork() for k,v in self.delay_states.items()},
            deepcopy(self.input_position), dict(self.component_tick_indices),
            deepcopy(self.link_delay_states), self.generation)

    def validate(self, runner):
        from .resident_state import ResidentStateHandle
        ResidentStateHandle._validate_aux(self, runner)
        runner.validate_solver(self.solver_config)
        if (self.identity != runner.identity or self.input_binding != runner.binding_hash
                or self.point.inner.shape != (runner.partition.inner_size,)
                or self.link_delay_states or not math.isfinite(self.time_seconds)):
            raise ContractError('Reduced state identity/input/layout differs')
        if (self.point.native.y.shape!=(runner.native.nstate,)
                or self.cursor.frontier.y.shape!=(runner.native.nstate,)
                or (self.point.compact_release is not None and
                    self.point.compact_release.shape!=(runner.selection.count,))):
            raise ContractError('Reduced native/endpoint cache layout differs')
        runner.native._state_guard(self.point.native)
        runner.native._state_guard(self.cursor.frontier)
        if (set(self.cursor.consumed) != set(CONSUMERS)
                or any(not math.isfinite(t) or t > self.time_seconds for t in self.cursor.consumed.values())
                or self.cursor.frontier.time_seconds < self.time_seconds):
            raise ContractError('Reduced cursor ownership/time differs')
        if self.cursor.frontier.time_seconds > self.time_seconds and (
                not self.cursor.segments or self.cursor.segments[0].start > self.time_seconds
                or self.cursor.segments[-1].stop != self.cursor.frontier.time_seconds):
            raise ContractError('Reduced cursor lost unconsumed lookahead')
        if len(self.cursor.segments) != len(self.cursor.packets):
            raise ContractError('Reduced cursor lost a CPU packet')
        previous = None
        for segment,packet in zip(self.cursor.segments,self.cursor.packets):
            segment.check_owned()
            if (segment.source_version!=runner.native.source_version
                    or (previous is not None and segment.start!=previous)
                    or (packet.start,packet.stop,packet.source_version)!=(
                        segment.start,segment.stop,segment.source_version)):
                raise ContractError('Reduced accepted segment/packet ownership differs')
            previous = segment.stop
        visible = self.cursor.at(self.time_seconds)
        if not runner.torch.equal(visible.y.view(runner.torch.int64),
                                  self.point.native.y.view(runner.torch.int64)):
            raise ContractError('Reduced public native state differs from accepted history')


class ReducedConeHCRunner:
    @classmethod
    def from_resident(cls, retained, *, aux_indices=(), runtime_guard=None):
        from .resident_runner import ResidentNativeRunner
        from .cone_hc_split_contract import guard_block_triangular
        from .gpu_compact import compact_release_layout
        from .cone_hc_reduced_evaluation import ReducedEvaluator
        if type(retained) is not ResidentNativeRunner:
            raise ContractError('Reduced candidate requires the retained CSR resident runner')
        if not hasattr(retained, 'native_sparse_plan_policy'):
            raise ContractError('Reduced candidate requires retained CSR ALG2 plans')
        obj = cls()
        obj.torch, obj.compiled, obj.model = retained.torch, retained.compiled, retained.model
        obj.partition = copy(retained.partition)
        obj.partition.work = retained.partition.work.copy()
        obj.partition._light_cpu = obj.partition._light_gpu = None
        obj.periods = dict(retained.periods)
        obj.observation_dt, obj.history_dt = retained.observation_dt, retained.history_dt
        obj.nstate = retained.nstate  # Only the original CPU plan uses these offsets.
        obj.graph_guard = guard_block_triangular(obj)
        obj.reference, obj.runtime_guard, obj._lock = retained, runtime_guard, RLock()
        original = obj.compiled.components[obj.partition.cid].native_model
        obj.native = ReducedNative(original, torch=obj.torch,
            A=obj.partition.native.A, B=obj.partition.native.B, aux_indices=aux_indices)
        used, incoming, _ = compact_release_layout(obj.model, obj.partition.cid, original.nc)
        obj.selection = BridgeSelection(original.B, used, cone_ids=original.cones.cells.cell_id,
                                        hc_ids=original.hcs.cells.cell_id)
        obj.bridge = PinnedSegmentBridge(obj.selection, obj.torch, obj.native.device)
        obj.evaluator = ReducedEvaluator(obj, incoming)
        obj.identity = canonical_hash(dict(method=METHOD, native=obj.native.identity,
            observation_queue=queue_contract(),
            retained_model=retained.model.model_hash, graph=obj.graph_guard,
            selected_ids=obj.selection.sha256, inner_plan=obj.partition.plan_hash,
            implementation={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                for n in ('cone_hc_reduced_runner.py','cone_hc_reduced_evaluation.py',
                          'cone_hc_reduced_monitor.py','cone_hc_reduced_checkpoint.py',
                          'cone_hc_reduced_bridge.py')}))
        obj.binding_hash = None
        obj.channels = None
        obj._active_cursor = None
        obj.metrics = dict(native_seconds=0., inner_seconds=0., monitor_seconds=0.,
                           advance_seconds=0., inner_nfev=0, native_segments=0)
        return obj

    def validate_solver(self, config):
        self.reference.validate_solver(config)
        if (config.method,config.rtol,config.atol,config.max_step) != ('DOP853',1e-6,1e-9,.0001):
            raise ContractError('Reduced CPU inner requires the unchanged frozen solver')

    def _check_runtime(self):
        if self.runtime_guard is not None and self.runtime_guard() is False:
            raise ContractError('Reduced candidate runtime guard requested stop')

    def migrate(self, old_state, *, source_runner, channels):
        old_state.validate(source_runner)
        self.validate_solver(old_state.solver_config)
        if source_runner.compiled is not self.compiled:
            raise ContractError('Migration requires the identical actual graph')
        self.channels = dict(channels)
        self.binding_hash = canonical_hash(dict(source_binding=source_runner.binding,
                                               reduced_identity=self.identity))
        channel = channels[self.partition.native_external.channel]
        t = old_state.time_seconds
        bound,_ = next_native_input_bound(channel,t)
        q = native_input_at(channel,t)
        self.partition.native_port.validate(q)
        native = self.native.migrate(old_state.y[:self.partition.offset], time_seconds=t,
            light=self.partition._light(q), input_stop_seconds=bound)
        initial_release = self.native.evaluate(native.y,t,native.anchor).outputs['release']
        compact = initial_release.index_select(0,self.partition.gather).cpu().numpy()
        point = ReducedPoint(native, old_state.y[self.partition.offset:].detach().cpu().numpy(),compact)
        state = ReducedState(point, ReducedCursor(native, consumed=dict.fromkeys(CONSUMERS,t)),
            self.identity,self.binding_hash,old_state.solver_config,old_state.grid_origin_seconds,
            old_state.sample_index,deepcopy(old_state.spike_states),
            {k:v.fork() for k,v in old_state.delay_states.items()},deepcopy(old_state.input_position),
            dict(old_state.component_tick_indices))
        state.input_position['observation_queue'] = queue_contract()
        state.validate(self)
        return state

    def evaluate(self, t, point, bundle, **kwargs):
        return self.evaluator(t,point,bundle,**kwargs)

    def _ensure_segment(self, state):
        cursor = state.cursor
        if cursor.frontier.time_seconds > state.time_seconds:
            return
        native = cursor.frontier
        if native.time_seconds == native.anchor.input_stop_seconds:
            channel = self.channels[self.partition.native_external.channel]
            bound,_ = next_native_input_bound(channel,native.time_seconds)
            q = native_input_at(channel,native.time_seconds)
            self.partition.native_port.validate(q)
            native = self.native.refresh_input(native,self.partition._light(q),bound,
                                                native.anchor.input_epoch+1)
        tick = perf_counter()
        cursor.frontier,segment = self.native.step(native)
        self.torch.cuda.synchronize(self.native.device)
        self.metrics['native_seconds'] += perf_counter()-tick
        packet = self.bridge.submit(segment,segment_id=cursor.next_id)
        cursor.segments.append(segment)
        cursor.packets.append(packet)
        cursor.next_id += 1
        self.metrics['native_segments'] += 1

    def _inner_rhs(self, cursor, stimulus, stop, t, inner):
        part = self.partition
        bundle = stimulus.at(t,left_limit=bool(t>=stop))
        values = []
        for external,port in part.external:
            value = bundle.channels[external.channel]
            port.validate(value)
            values.append(value.values)
        values.append(cursor.packet_at(t,left=t>=stop).release_at(t))
        rhs = np.asarray(part.fn(inner,np.concatenate(values),part.work,part.program,
            part.weights,part.indices,part.pointers,part.derivatives),dtype=np.float64).copy()
        if rhs.shape != inner.shape or not np.isfinite(rhs).all():
            raise ContractError('Original CPU inner RHS failed')
        return rhs

    def _sample(self, state, stimulus):
        # Same component traversal, transition, tick and PCG64 order as ResidentNativeRunner._sample.
        t = state.time_seconds
        due = {cid for cid,p in self.periods.items()
               if abs(state.grid_origin_seconds+state.component_tick_indices[cid]*p-t)<1e-12}
        evidence = self.evaluate(t,state.point,stimulus.at(t),sampling=True,sampled_components=due)
        changed = evidence.inner_y.copy()
        for cid,component in self.compiled.components.items():
            if cid == self.partition.cid:
                continue
            local = None
            if component.time_contract.kind == 'causal_history' and cid in due:
                local = component.native_model.transition(evidence.local_blocks(cid),evidence.inputs[cid],
                    {p:evidence.outputs[(cid,p)] for p in component.outputs})
            if component.time_contract.kind == 'author_discrete' and cid in due and t>state.grid_origin_seconds+1e-12:
                previous = t-component.time_contract.fixed_step_seconds
                external = next(e for e in self.compiled.spec.external_inputs
                                if e.target_component==cid and e.target_port=='light')
                channel = stimulus.channels[external.channel]
                if (channel.interpolation!='zero_order_hold'
                        or channel.input_contract.get('grid_step_seconds')!=component.time_contract.fixed_step_seconds
                        or channel.input_contract.get('includes_background') is not True
                        or previous<channel.start_seconds-1e-12):
                    raise ContractError('Original rod total-rate ZOH contract differs')
                local = component.native_model.transition(evidence.local_blocks(cid),
                    {'light':channel.at(max(channel.start_seconds,previous+1e-13))})
            if local is not None:
                sl = self.model.slices[cid]
                changed[sl.start-self.partition.offset:sl.stop-self.partition.offset] = np.concatenate(local)
        state.point = ReducedPoint(state.point.native,changed,state.point.compact_release)
        state.generation += 1
        for cid in due:
            state.component_tick_indices[cid] += 1
        for cid,aux in list(state.spike_states.items()):
            if cid in due:
                sl = self.model._local_slices[cid][0]
                voltage = changed[sl.start-self.partition.offset:sl.stop-self.partition.offset]
                state.spike_states[cid],_ = self.compiled.components[cid].native_model.sample_spikes(t,voltage,aux)
        state.sample_index += 1

    def _observe(self, observer, stimulus, t, point):
        tick = perf_counter()
        if any(t in c.breakpoints_seconds for c in self.channels.values()):
            observer(t,point,stimulus.at(t,left_limit=True))
        observer(t,point,stimulus.at(t))
        self.metrics['monitor_seconds'] += perf_counter()-tick

    def advance(self, state, channels, stop, *, sample_times=None, state_observer=None,
                record=None, event_sink=None):
        """Synchronous complete observer and optional selected-record callback.

        record(t,point,evidence,state) runs only at exact requested times. The
        caller selects native/inner/auxiliary fields explicitly from evidence.
        event_sink(cid,event) consumes original (start,stop] events; without a
        sink pending events stay owned by the state, including future-bin events.
        """
        from scipy.integrate import DOP853
        from .network import NetworkStimulus
        if state_observer is None or self.channels is None or set(channels)!=set(self.channels) or any(
                channels[k].evaluator is not self.channels[k].evaluator for k in channels):
            raise ContractError('Reduced advance needs complete monitoring and frozen bound inputs')
        if not math.isfinite(stop) or stop<=state.time_seconds:
            raise ContractError('Reduced advance must increase time')
        times = np.asarray([state.time_seconds,stop] if sample_times is None else sample_times,dtype=float)
        if (times.ndim!=1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
                or times[0]<state.time_seconds or times[-1]>stop):
            raise ContractError('Reduced recording times must increase inside advance')
        with self._lock,self.torch.no_grad():
            state.validate(self)
            state = state.fork()
            self._active_cursor = state.cursor
            stimulus = NetworkStimulus(self.compiled,channels,state,stop,
                state.solver_config.input_interpolation,self.model.evidence_tag)
            breaks = iter(stimulus.breakpoints_seconds)
            next_break = next(breaks,stop)
            start = state.time_seconds
            tick = perf_counter()
            recorded = set()
            original_boundary = True
            def consume(t,point,*,queue=False,do_record=True):
                if queue:
                    h = self.native.blocks(point.native.y)['Vh']
                    for delay in state.delay_states.values():
                        if t>delay.history_times[-1]:
                            delay.append(t,h,delay.unit,delay.meaning)
                self._observe(state_observer,stimulus,t,point)
                if do_record and t in times and t not in recorded:
                    if record is not None:
                        record(t,point,self.evaluate(t,point,stimulus.at(t)),state)
                    recorded.add(t)
            def acknowledge():
                for consumer in CONSUMERS:
                    state.cursor.acknowledge(consumer,state.time_seconds,self.bridge)
            def flush_events(t):
                if event_sink is not None:
                    for cid,aux in state.spike_states.items():
                        for event in aux['pending_events']:
                            if start<event['time_seconds']<=t:
                                event_sink(cid,dict(event))
                        aux['pending_events'] = [e for e in aux['pending_events'] if e['time_seconds']>t]
            try:
                while state.time_seconds<stop:
                    self._check_runtime()
                    current = state.time_seconds
                    if current==start:
                        consume(current,state.point,do_record=False)
                    grid = min(state.grid_origin_seconds+state.component_tick_indices[cid]*p
                               for cid,p in self.periods.items())
                    if grid<current-1e-10:
                        raise ContractError('Original history sampling phase lost')
                    if original_boundary and abs(grid-current)<1e-10:
                        self._sample(state,stimulus)
                        self._observe(state_observer,stimulus,current,state.point)
                    if current in times and current not in recorded:
                        if record is not None:
                            record(current,state.point,self.evaluate(current,state.point,stimulus.at(current)),state)
                        recorded.add(current)
                    acknowledge()
                    self._ensure_segment(state)
                    while next_break<=current:
                        next_break = next(breaks,stop)
                    next_grid = min(state.grid_origin_seconds+state.component_tick_indices[cid]*p
                                    for cid,p in self.periods.items())
                    next_obs = (state.grid_origin_seconds+(int(np.floor(
                        (current-state.grid_origin_seconds)/self.observation_dt+1e-8))+1)*self.observation_dt)
                    outer_target = min(float(stop),float(next_break),float(next_grid),float(next_obs))
                    target = min(outer_target,state.cursor.frontier.time_seconds)
                    original_boundary = target==outer_target
                    packet = state.cursor.packet_at(current)
                    packet.wait()  # Exactly once per accepted packet, outside inner compute timing.
                    begun = perf_counter()
                    solver = DOP853(lambda t,y:self._inner_rhs(state.cursor,stimulus,target,t,y),
                        current,state.point.inner.copy(),target,rtol=state.solver_config.rtol,
                        atol=state.solver_config.atol,max_step=state.solver_config.max_step)
                    rows = {}
                    while solver.status=='running':
                        self._check_runtime()
                        previous = solver.t
                        message = solver.step()
                        if solver.status=='failed':
                            raise ContractError('Reduced CPU DOP853 failed: '+str(message))
                        targets = sorted({solver.t}|{float(t) for t in times if previous<t<=solver.t})
                        dense = solver.dense_output() if any(t<solver.t for t in targets) else None
                        for t in targets:
                            inner = solver.y if t==solver.t else dense(t)
                            rows[t] = ReducedPoint(state.cursor.at(t),inner,
                                                   state.cursor.packet_at(t,left=t>=target).release_at(t))
                    self.metrics['inner_seconds'] += perf_counter()-begun
                    self.metrics['inner_nfev'] += solver.nfev
                    state.point = rows[target]
                    state.generation += 1
                    flush_events(target)
                    for t,point in sorted(rows.items()):
                        queue = is_queue_node(t,times,outer_target)
                        # Original event-grid requests are recorded after that event.
                        defer_record = original_boundary and t==target and abs(t-next_grid)<=1e-10
                        consume(t,point,queue=queue,do_record=not defer_record)
                    acknowledge()
                final_tick = min(state.grid_origin_seconds+state.component_tick_indices[cid]*p
                                 for cid,p in self.periods.items())
                if abs(final_tick-stop)<1e-12:
                    self._sample(state,stimulus)
                self._observe(state_observer,stimulus,stop,state.point)
                flush_events(stop)
                if stop in times and stop not in recorded:
                    if record is not None:
                        record(stop,state.point,self.evaluate(stop,state.point,stimulus.at(stop)),state)
                    recorded.add(stop)
                acknowledge()
                if set(times)!=recorded:
                    raise ContractError('Reduced selected recording coverage lost')
                state.input_position.update(last_time_seconds=stop,reduced_identity=self.identity)
                state.cursor.own_packets(self.bridge)
                state.validate(self)
                return state
            finally:
                state.cursor.own_packets(self.bridge)
                self.metrics['advance_seconds'] += perf_counter()-tick
                self._active_cursor = None
