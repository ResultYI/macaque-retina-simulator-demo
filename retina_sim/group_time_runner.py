"""One approved group-model numerical arm, preserving original event clocks.

Native lookahead is independent of rod ticks. Published readouts own their CPU
coefficients, while the current/previous accepted native intervals remain on the
selected device for full-equation monitoring and exact checkpoint continuation.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import hashlib
import math

import numpy as np

from .contracts import ContractError, Quantity, canonical_hash
from .cone_hc_reduced_native import AnalyticREAnchor, STATE_NAMES
from .native_multirate import NativeMultirateBackend, NativeMultirateState, AcceptedNativeInterval
from .group_native_physics import GroupNativePhysics
from .group_inner_dynamics import GroupInnerDynamics, RODS, BC_MECH, RGC_MECH, MEMBRANE
from .group_inner_compiled import CompiledGroupInnerRHS
from .group_time_transport import PinnedContinuousTransport, OwnedReadoutWindow
from .continuous_bridge import ContinuousBridgeSelection, ContinuousReleasePacket, CONSUMERS
from .continuous_state import Dop853Dense, ExponentialForced, QuinticHermite
from .spatial_group_model import BC, RGC
from .recording import TimestampedDelay
from .resident_delay import ResidentTimestampedDelay


SCHEMA = 'retipath-group-time-arm-1'
ROOT = Path(__file__).resolve().parents[1]


def _cpu(value):
    return value.detach().cpu().numpy().copy() if hasattr(value, 'detach') else np.array(value, copy=True)


def _snapshot_copy(value):
    """Preserve deepcopy ownership/aliases without visiting immutable list scalars."""
    memo = {}
    atomic = frozenset((type(None), bool, int, float, complex, str, bytes))
    def owned(item):
        if type(item) in atomic:
            return item
        identity = id(item)
        if identity in memo:
            return memo[identity]
        if type(item) is list:
            if atomic.issuperset(map(type, item)):
                result = item.copy()
                memo[identity] = result
                return result
            result = []
            memo[identity] = result
            result.extend(owned(entry) for entry in item)
            return result
        if type(item) is dict:
            result = {}
            memo[identity] = result
            for key, entry in item.items():
                copied = owned(entry)
                result[key if type(key) in atomic else deepcopy(key, memo)] = copied
            return result
        return deepcopy(item, memo)
    return owned(value)


def _native_state(backend, raw):
    fields = {'time_seconds','y','anchor','backend_id','next_window_seconds'}
    if not isinstance(raw,dict) or set(raw) != fields:
        raise ContractError('Complete native frontier and controller proposal required')
    a = raw['anchor']
    if not isinstance(a,dict) or set(a) != {'time_seconds','R','E','light','input_stop_seconds','input_epoch'}:
        raise ContractError('Complete native analytic anchor required')
    for name in ('y',):
        if np.asarray(raw[name]).dtype != np.float64:
            raise ContractError('Native checkpoint state must retain FP64')
    for name in ('R','E','light'):
        if np.asarray(a[name]).dtype != np.float64:
            raise ContractError('Native checkpoint anchor must retain FP64')
    anchor = AnalyticREAnchor(a['time_seconds'],backend.native._array(a['R']),backend.native._array(a['E']),
        backend.native._array(a['light']),a['input_stop_seconds'],a['input_epoch'])
    state = NativeMultirateState(raw['time_seconds'],backend.native._array(raw['y']),anchor,
        raw['backend_id'],raw['next_window_seconds'])
    backend._guard(state)
    return state


class GroupTimeArm:
    def __init__(self, assembly, construction_bundle_payload, protocol, *, mode='candidate',
                 torch=None, device=None, guard=None, shared_plans=None):
        self.assembly, self.seed, self.protocol = assembly, construction_bundle_payload, dict(protocol)
        self.mode, self.torch, self.device = mode, torch, None if torch is None else torch.device(device)
        self.guard = guard or (lambda:None)
        if (mode not in ('reference','candidate','radau','gpu_dop853') or protocol['input_fps'] != 150.
                or protocol['rod_step_seconds'] != .0001 or protocol['BC_RGC_period_seconds'] != .001
                or protocol['recording_step_seconds'] != .0005 or protocol['native_observation_delay_seconds'] != .00282
                or protocol['reference']['rtol'] != 1e-6 or protocol['reference']['atol'] != 1e-9
                or protocol['reference']['max_step'] != .0001 or protocol['candidate']['rtol'] != 1e-6
                or protocol['candidate']['atol'] != 1e-9 or protocol['candidate']['fast_max_step'] != .0001):
            raise ContractError('Original fixed clocks/tolerances and explicitly selected arm required')
        if torch is not None and self.device.type != 'cuda':
            raise ContractError('Production native execution requires explicit CUDA; use torch=None only for CPU fixtures')
        self.start = float(protocol['start_seconds'])
        self.stop = float(protocol['stop_seconds'])
        if (not math.isfinite(self.start) or not math.isfinite(self.stop) or self.start >= self.stop
                or construction_bundle_payload['time_seconds'] != self.start):
            raise ContractError('Construction IVP time and protocol scope differ')
        self.guard()
        started = perf_counter()
        factories = getattr(assembly, 'runtime_factories', {})
        self.inner = factories.get('inner', GroupInnerDynamics)(assembly, construction_bundle_payload['inner_parameters'])
        self.compiled = factories.get('compiled', CompiledGroupInnerRHS)(self.inner,repo_root=ROOT)
        self.inner.compiled_rhs = self.compiled
        self.inner_state = {k:np.array(construction_bundle_payload['states'][k],copy=True) for k in self.inner.state_keys}
        self.inner.validate(self.inner_state)
        self.gpu_rhs = self.inner_backend = self.scc_plan = None
        self._snapshot_gpu_state = self._snapshot_gpu_rhs = None
        if mode in ('radau','gpu_dop853'):
            if torch is None:
                if mode == 'gpu_dop853':
                    raise ContractError('GPU DOP853 requires the explicitly selected CUDA runtime')
            else:
                from .group_gpu_rhs import GroupGpuRHS
                self.gpu_rhs = GroupGpuRHS(self.compiled,torch)
                self._snapshot_gpu_state = torch.empty(self.inner.nstate,dtype=torch.float64,device=self.device)
                self._snapshot_gpu_rhs = torch.empty_like(self._snapshot_gpu_state)
            if mode == 'radau':
                from .group_inner_radau import GroupInnerRadau
                from .group_scc_plan import build_scc_plan
                from .group_time_monitor import frozen_budget
                self.scc_plan = build_scc_plan(self.inner,self.compiled)
                packed = self.inner.pack(self.inner_state)
                budgets = np.concatenate([frozen_budget(self.inner_state[k],b.unit)[2]
                    for k,b in zip(self.inner.state_keys,self.inner.blocks)])
                self.inner_backend = GroupInnerRadau(self.inner,torch=torch,device=self.device or 'cpu',
                    guard=self.guard,compiled=self.compiled,scc_plan=self.scc_plan,original_rhs=self.gpu_rhs,
                    error_weights=1e-9+1e-6*np.abs(packed),budgets=budgets)
            else:
                from .group_gpu_dop853 import GroupGpuDop853
                self.inner_backend = GroupGpuDop853(self.inner,self.compiled,torch,
                    device=self.device,guard=self.guard,gpu_rhs=self.gpu_rhs)
        expected_frames=set(int(i) for i in protocol['input_frame_indices'])
        if set(construction_bundle_payload['native_inputs']) != expected_frames or set(construction_bundle_payload['rod_inputs']) != expected_frames:
            raise ContractError('All and only the approved cached input frames must be bound')
        for frame in expected_frames:
            for category,count in (('native_inputs',self.inner.counts['cones']),('rod_inputs',self.inner.counts['rods'])):
                a=np.asarray(construction_bundle_payload[category][frame])
                if a.dtype != np.float64 or a.shape != (count,) or not np.isfinite(a).all() or np.any(a<0):
                    raise ContractError('Cached total-rate input dtype/shape/domain differs: '+category)
        self.spike_states = deepcopy(construction_bundle_payload['spike_states'])
        self.native_physics = factories.get('native', GroupNativePhysics)(assembly,torch=torch,device=self.device,plans=shared_plans)
        self.shared_plans = (self.native_physics.A,self.native_physics.B)
        self._owns_plans = shared_plans is None
        if mode == 'radau':
            from .group_native_radau_adapter import GroupNativeRadauBackend
            self.native_backend = GroupNativeRadauBackend(self.native_physics,rtol=1e-6,atol=1e-9,guard=self.guard,
                native_plan=self.scc_plan.native)
        else:
            self.native_backend = NativeMultirateBackend(self.native_physics,rtol=1e-6,atol=1e-9)
        initial_frame = self.frame_at(self.start,'right')
        seed = SimpleNamespace(states=construction_bundle_payload['states'],time_seconds=self.start)
        y, anchor = self.native_physics.construction_point(seed,self.seed['native_inputs'][initial_frame],
            self.frame_end(initial_frame),initial_frame)
        self.native_state = self.native_backend.initial_state(y,anchor)
        if mode in ('reference','gpu_dop853'):
            self.native_state = NativeMultirateState(self.start,self.native_state.y,anchor,
                self.native_backend.reference_identity,self.native_state.next_window_seconds)
        cone_columns = []
        for route in self.inner.routes:
            if route.pre == 'cones' and route.kind == 'inner_chemical':
                cone_columns.append(self.inner.mean[route.name].indices)
        used = np.unique(np.concatenate(cone_columns)) if cone_columns else np.empty(0,np.int64)
        if not len(used):
            raise ContractError('Complete group inner graph must retain actual Cone dependencies')
        self.selection = ContinuousBridgeSelection(assembly.matrix('routes/HC_to_cones/mean_weights.npz'),used)
        self._selection_device_rows = None if torch is None else torch.as_tensor(self.selection.cone_rows.copy(),dtype=torch.int64,device=self.device)
        source = self.native_backend.reference_identity if mode in ('reference','gpu_dop853') else self.native_backend.identity
        self.transport = None if torch is None else PinnedContinuousTransport(self.selection,torch,self.device,source_version=source)
        self.native_intervals, self.readout_windows = [], []
        self._readout_ids = set()
        self.last_inner_advance = None
        self.time_seconds = self.inner_time = self.start
        self.event_generation = 0
        self.last_draws, self.last_events = {}, {}
        self.event_log = {pop:[] for pop in RGC}
        self.emitted_events_unrecorded = {pop:[] for pop in RGC}
        self._emitted_until = self.start
        self.initial_commit_done = False
        self.clock_indices = dict(rod=int(round((self.start-protocol['clock_origin_seconds'])/.0001)),
            bc=int(round((self.start-protocol['clock_origin_seconds'])/.001))-1,
            rgc=int(round((self.start-protocol['clock_origin_seconds'])/.001))-1)
        self.last_commit_times = dict(rod=None,bc=None,rgc=None)
        self.native_consumed = dict.fromkeys(CONSUMERS,self.start)
        self._release_cache_time, self._release_cache = None, None
        self._make_delay(construction_bundle_payload['observation_prehistory'])
        self.identity = canonical_hash(dict(schema=SCHEMA,artifact=assembly.artifact_sha256,mode=mode,
            protocol=protocol,source_seed=construction_bundle_payload,inner=self.inner.identity,
            native=source,compiled=self.compiled.identity,
            **(dict(inner_backend=self.inner_backend.identity,
                gpu_rhs=None if self.gpu_rhs is None else self.gpu_rhs.identity,
                scc_plan=None if self.scc_plan is None else self.scc_plan.identity) if self.inner_backend is not None else {}),
            implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        self.metrics = dict(build_seconds=perf_counter()-started,native_wall_seconds=0.,inner_wall_seconds=0.,
            native_advances=0,inner_advances=0,event_commits=0,full_state_D2H_per_stage=0,
            native_counters=[],inner_counters=[],solver_GC_seconds=0.,solver_GC_collected_objects=0,
            snapshot_count=0,snapshot_wall_seconds=0.,snapshot_native_seconds=0.,
            snapshot_rhs_seconds=0.,snapshot_bridge_seconds=0.,snapshot_outputs_seconds=0.,
            snapshot_transition_seconds=0.,snapshot_other_seconds=0.,
            scc_plan=None if self.scc_plan is None else dict(identity=self.scc_plan.identity,
                typed_ir_identity=self.scc_plan.typed_ir_identity,counts=dict(self.scc_plan.counts),
                memory_bytes=dict(self.scc_plan.memory_bytes)))
        self.guard()

    def frame_at(self,t,side='right'):
        if not math.isfinite(t) or side not in ('left','right'):
            raise ContractError('Finite stimulus time and exact side required')
        frames = sorted(int(i) for i in self.seed['native_inputs'])
        origin = self.protocol['input_origin_seconds']
        candidates = [i for i in frames if (origin+i/150 <= t < origin+(i+1)/150) or
            (side=='left' and t == origin+(i+1)/150)]
        if side == 'left':
            candidates = [i for i in candidates if t != origin+i/150 or t == self.start and not any(j < i for j in frames)]
        else:
            candidates = [i for i in candidates if t != origin+(i+1)/150]
        if not candidates:
            raise ContractError('Cached frame does not cover the requested exact stimulus side')
        return min(candidates) if side=='left' else max(candidates)

    def frame_end(self,index):
        return self.protocol['input_origin_seconds']+(int(index)+1)/150

    def rod_sample_at(self,t,side='right'):
        """Original author-discrete channel floor; native frames are separate."""
        if not math.isfinite(t) or side not in ('left','right'):
            raise ContractError('Finite original rod input timestamp and side required')
        origin = self.protocol['clock_origin_seconds']
        tick = int(round((t-origin)/.0001))
        query = np.nextafter(t,-np.inf) if side=='left' and t == origin+tick*.0001 else t
        index = int(np.floor((query-origin)/.0001))
        sampled = origin+index*.0001
        return sampled,self.frame_at(sampled,'right')

    def rod_rate_at(self,t,side='right'):
        sampled,frame = self.rod_sample_at(t,side)
        return self.seed['rod_inputs'][frame]

    def _make_delay(self,observation):
        if observation.get('delay_seconds') != .00282 or observation.get('interpolation') != 'linear':
            raise ContractError('Original 2.82ms linear observation delay required')
        self._prehistory_aux_values=np.asarray(observation['prehistory'],dtype=np.float64).copy()
        raw = dict(delay_seconds=.00282,history_times=[float(t) for t in observation['history_times']],
            history_values=[np.array(v,copy=True) for v in observation['history_values']],
            prehistory=dict(values=np.asarray(observation['prehistory']),unit=observation['unit'],
                meaning=observation['meaning'],evidence_tag='project_approximation'),
            interpolation='linear',prehistory_stop_seconds=float(observation['origin_seconds']))
        self.delay = TimestampedDelay.from_payload(raw) if self.torch is None else ResidentTimestampedDelay.from_payload(raw,self.torch,self.device)

    def _interval_at(self,t,side):
        for interval in self.native_intervals:
            if (interval.start < t < interval.stop or t == interval.start and side=='right'
                    or t == interval.stop and side=='left'):
                return interval
        if t == self.stop and self.native_intervals and t == self.native_intervals[-1].stop:
            return self.native_intervals[-1]
        return None

    def ensure_native(self,t,side='left'):
        if not math.isfinite(t) or not self.start <= t <= self.stop or side not in ('left','right'):
            raise ContractError('Native availability request outside approved segment')
        if t == self.start and not self.native_intervals:
            return None
        while self._interval_at(t,side) is None:
            self.guard()
            current = self.native_state
            if current.time_seconds == current.anchor.input_stop_seconds:
                frame = self.frame_at(current.time_seconds,'right')
                if frame != current.anchor.input_epoch+1:
                    raise ContractError('Native input epoch did not advance exactly once')
                current = self.native_backend.refresh_input(current,self.seed['native_inputs'][frame],self.frame_end(frame),frame)
                self.native_state = current
            if current.time_seconds >= self.stop:
                if self.native_intervals and t == self.stop:
                    return self.native_intervals[-1]
                raise ContractError('Native interval side is unavailable at the accepted frontier')
            if len(self.native_intervals) >= 2:
                old = self.native_intervals[0]
                if min(self.native_consumed.values()) >= old.stop:
                    self.native_intervals.pop(0)
                else:
                    raise ContractError('Native lookahead needs an unconsumed third interval')
            started = perf_counter()
            limit = min(self.stop,current.anchor.input_stop_seconds)
            if self.mode in ('candidate','radau'):
                result = self.native_backend.advance_one(current,limit)
            else:
                result = self.native_backend.advance_joint_reference(current,min(current.time_seconds+.0005,limit))
            self.metrics['native_wall_seconds'] += perf_counter()-started
            self.metrics['native_advances'] += 1
            self.metrics['native_counters'].append(result.metrics)
            self.native_state = result.state
            self.native_intervals.extend(result.intervals)
        return self._interval_at(t,side)

    def _native_evaluation(self,t,side):
        if t == self.native_state.time_seconds and not self.native_intervals:
            anchor=self.native_state.anchor
            frame=self.frame_at(t,side)
            if frame != anchor.input_epoch:
                R,E=anchor.at(t)
                anchor=AnalyticREAnchor(t,R,E,self.native_physics._array(self.seed['native_inputs'][frame]),self.frame_end(frame),frame)
            return self.native_physics.evaluate(self.native_state.y,t,anchor),self.native_state.y,anchor
        interval = self.ensure_native(t,side)
        local_side = 'left' if t == interval.stop else 'right' if t == interval.start else side
        y = interval.evaluate(t,local_side)
        # At an artificial terminal limit the continuous state remains valid
        # on the public right side; a true input edge uses its incoming anchor.
        anchor = interval.anchor
        expected_frame = self.frame_at(t,side)
        if anchor.input_epoch != expected_frame:
            R,E = anchor.at(t)
            anchor = AnalyticREAnchor(t,R,E,self.native_physics._array(self.seed['native_inputs'][expected_frame]),
                self.frame_end(expected_frame),expected_frame)
        return self.native_physics.evaluate(y,t,anchor),y,anchor

    def _window_readout(self,interval):
        if interval.representation_id not in self._readout_ids:
            chunks = interval.readout_chunks(self.selection.cone_rows,self.selection.hc_rows)
            if self.transport is not None:
                packets = self.transport.transfer(chunks)
            else:
                packets = []
                for chunk in chunks:
                    representations = {}
                    for name,rep in chunk.items():
                        if rep is None:
                            representations[name] = None
                            continue
                        args = dict(t0=rep.t0,t1=rep.t1,state_ids=rep.state_ids,epoch=rep.input_epoch)
                        if rep.kind == 'quintic_hermite':
                            args.update({n:getattr(rep,n) for n in ('y0','f0','ym','fm','y1','f1')})
                            representation_class = QuinticHermite
                        else:
                            args.update(y0=rep.y0,coefficients=rep.coefficients)
                            if rep.kind == 'exponential_forced':args['decay']=np.asarray(rep.decay,dtype=np.float64)
                            representation_class = ExponentialForced if rep.kind=='exponential_forced' else Dop853Dense
                        representations[name] = representation_class(**args)
                    packets.append(ContinuousReleasePacket(self.selection,representations,source_version=interval.source_version,
                        segment_id=len(packets)).wait())
            self.readout_windows.append((interval.representation_id,OwnedReadoutWindow(packets)))
            self._readout_ids.add(interval.representation_id)
        return next(w for ident,w in self.readout_windows if ident == interval.representation_id)

    def _cone_release(self,t,side,native_outputs=None):
        if self.mode in ('radau','gpu_dop853'):
            if native_outputs is None:
                native_outputs = self._native_evaluation(t,side)[0].outputs
            return _cpu(native_outputs['release'])
        if t == self._release_cache_time:
            compact = self._release_cache
        elif t == self.native_state.time_seconds and not self.native_intervals:
            if native_outputs is None:native_outputs=self._native_evaluation(t,side)[0].outputs
            compact = (_cpu(native_outputs['release'][self.selection.cone_rows]) if self.torch is None else
                _cpu(native_outputs['release'].index_select(0,self._selection_device_rows)))
        else:
            interval = self.ensure_native(t,side)
            window = self._window_readout(interval)
            local_side = 'left' if t == window.stop else 'right' if t == window.start else side
            compact = window.at(t,local_side)
        value = np.zeros(self.native_physics.nc,dtype=np.float64)
        value[self.selection.cone_rows] = compact
        return value

    def advance_inner(self,t0,t1):
        if t0 != self.inner_time or not t0 < t1 <= self.stop:
            raise ContractError('Inner continuous interval must start at its accepted public state')
        self.guard()
        started = perf_counter()
        fn = (self.inner_backend.advance if self.inner_backend is not None else
              self.inner.advance_candidate if self.mode=='candidate' else self.inner.advance_reference)
        from .solver_lifetime import collection_epoch,release_solver_cycles
        epoch=collection_epoch()
        try:
            result = fn(self.inner_state,t0,t1,rtol=1e-6,atol=1e-9)
        finally:
            cleanup=release_solver_cycles(epoch)
            self.metrics['solver_GC_seconds'] += cleanup['wall_seconds']
            self.metrics['solver_GC_collected_objects'] += cleanup['collected_objects']
        self.inner_state,self.inner_time,self.last_inner_advance = result.state,t1,result
        self.metrics['inner_wall_seconds'] += perf_counter()-started
        self.metrics['inner_advances'] += 1
        self.metrics['inner_counters'].append(result.metrics)
        return result

    def _draw_witness(self,t,before,after,events,state):
        result = {}
        index = int(round((t-self.protocol['clock_origin_seconds'])/.001))
        for pop in RGC:
            a,b = before[pop],after[pop]
            h = np.asarray(a['history']) if a['last_bin_counts'] is None else np.vstack((np.asarray(a['last_bin_counts'])[None,:],np.asarray(a['history'])[:-1]))
            rng = np.random.Generator(np.random.PCG64(0));rng.bit_generator.state=deepcopy(a['rng_state'])
            u = rng.random(self.inner.counts[pop])
            if canonical_hash(rng.bit_generator.state) != canonical_hash(b['rng_state']):
                raise ContractError('Original PCG64 sampler draw count/order differs')
            rate = np.asarray(b['conditional_rate_Hz'],dtype=np.float64)
            probability = -np.expm1(-.001*rate)
            decision = np.asarray(b['last_bin_counts'],dtype=bool)
            if not np.array_equal(decision,u<probability):
                raise ContractError('Original sampled RGC probability law differs')
            times = [None]*len(u)
            for event in events[pop]:times[event['cell_index']]=float(event['time_seconds'])
            result[pop] = dict(u=u,rate_Hz=rate,probability=probability,event_decision=decision,
                event_time_seconds=times,bin_start_seconds=float(t),history_hash=canonical_hash(h),
                rng_before_hash=canonical_hash(a['rng_state']),rng_after_hash=canonical_hash(b['rng_state']),
                draw_index=index*len(u)+np.arange(len(u),dtype=np.int64))
        return result

    def commit(self,t,*,rod=False,bc=False,rgc=False,rod_sample_time=None):
        if t != self.inner_time or any(type(x) is not bool for x in (rod,bc,rgc)):
            raise ContractError('Original events require the accepted common pre-event state')
        before, auxiliary = self.inner_state,self.spike_states
        planned = {}
        for name,due,period in (('rod',rod,.0001),('bc',bc,.001),('rgc',rgc,.001)):
            if due:
                index = int(round((t-self.protocol['clock_origin_seconds'])/period))
                if index != self.clock_indices[name]+1:
                    raise ContractError('Original event index skipped or repeated: '+name)
                planned[name]=index
        light = None
        if rod:
            if rod_sample_time is None:raise ContractError('Original departing rod sample timestamp required')
            light = self.rod_rate_at(rod_sample_time,'right')
        release = self._cone_release(t,'left' if t != self.start else 'right') if bc else None
        commit=self.inner._commit_ticks if self.mode=='gpu_dop853' else self.inner.commit_ticks
        copy_options=dict(_auxiliary_copy=_snapshot_copy) if self.mode=='gpu_dop853' else {}
        state, spike, events = commit(t,before,cone_release=release,rod_light=light,
            rod_tick=rod,bc_tick=bc,rgc_tick=rgc,spike_states=auxiliary,**copy_options)
        draws=self._draw_witness(t,auxiliary,spike,events,before) if rgc else self.last_draws
        self.inner_state = state
        if spike is not None:self.spike_states=spike
        self.last_draws=draws
        self.last_events = events
        for name,index in planned.items():
            self.clock_indices[name]=index;self.last_commit_times[name]=float(t)
        self.event_generation += 1
        self.metrics['event_commits'] += 1
        return events

    def initial_commit(self):
        if self.initial_commit_done or self.inner_time != self.start:
            raise ContractError('Initial BC/RGC commit may occur once at source IVP start')
        events = self.commit(self.start,bc=True,rgc=True)
        self.initial_commit_done = True
        return events

    def initial_snapshot(self):
        if self.initial_commit_done or self.native_intervals or self.inner_time != self.start:
            raise ContractError('Initial frozen ports must be queried before integration or source sampling')
        return self.snapshot(self.start,'right')

    def _delay_append(self,t,h):
        if self.delay.history_times and t == self.delay.history_times[-1]:return
        if self.torch is None:self.delay.append(t,Quantity(h,'vH_native_voltage','native_relative_voltage','project_approximation'))
        else:self.delay.append(t,h,'vH_native_voltage','native_relative_voltage')

    def delay_payload(self):
        return self.delay.payload() if self.torch is None else self.delay.payload(arrays=True)

    def flush_events(self,t):
        """Deliver already sampled events once; keep only genuinely future ones."""
        if not math.isfinite(t) or not self._emitted_until <= t <= self.inner_time:
            raise ContractError('Original event delivery time cannot move backwards or pass accepted inner state')
        for pop,a in self.spike_states.items():
            due=[e for e in a.get('pending_events',[]) if e['time_seconds'] <= t]
            self.event_log[pop].extend(deepcopy(due))
            self.emitted_events_unrecorded[pop].extend(deepcopy(due))
            a['pending_events']=[e for e in a.get('pending_events',[]) if e['time_seconds'] > t]
        self._emitted_until=float(t)

    def _exact_aux(self,t,side):
        if self.torch is None:
            delay=self.delay.payload()
            delay['history_values']=np.asarray(delay['history_values'],dtype=np.float64)
        else:
            delay=self.delay.payload(arrays=True)
            delay['history_values']=self.torch.stack(delay['history_values'])
        delay['prehistory']=dict(delay['prehistory'],values=self._prehistory_aux_values.copy())
        sampled,rod_frame = self.rod_sample_at(t,side)
        copy = _snapshot_copy if self.mode == 'gpu_dop853' else deepcopy
        return dict(spike_states=copy(self.spike_states),clocks=dict(origin=self.protocol['clock_origin_seconds'],
            indices=dict(self.clock_indices),last_commit_times=dict(self.last_commit_times)),
            input_position=dict(public_native_frame=self.frame_at(t,side),rod_sample_frame=rod_frame,
                rod_sample_time=sampled,input_origin=self.protocol['input_origin_seconds']),observation_delay=delay)

    def snapshot(self,t,side='right',*,inner_state=None,queue_node=False,event_generation=None,generation=None):
        snapshot_started=perf_counter()
        self.guard()
        if inner_state is None:
            if t != self.inner_time:raise ContractError('An interior snapshot requires its accepted dense inner state')
            inner_state = self.inner_state
        self.inner.validate(inner_state)
        self.flush_events(t)
        started=perf_counter()
        evaluation,y,anchor = self._native_evaluation(t,side)
        output = dict(evaluation.outputs)
        if self.mode == 'gpu_dop853':
            R,E = output['R'],output['E']
        else:
            R,E = anchor.at(t);output['R'],output['E']=R,E
        native_seconds=perf_counter()-started
        states = dict(inner_state)
        started=perf_counter()
        if self.gpu_rhs is None:
            rhs = self.inner.unpack(self.compiled.rhs_flat(t,self.inner.pack(inner_state)),copy=False)
        else:
            packed = self.inner.pack(inner_state)
            self._snapshot_gpu_state.copy_(self.torch.from_numpy(packed))
            self.gpu_rhs.evaluate_into(t,self._snapshot_gpu_state,self._snapshot_gpu_rhs)
            rhs = self.inner.unpack(_cpu(self._snapshot_gpu_rhs),copy=False)
        rhs_seconds=perf_counter()-started
        for name,value in self.native_physics.blocks(y).items():
            k = ('cones','selected_original_z1_auxiliary','z1_aux') if name=='z1_aux' else ('HC','reduced_native',name) if name in ('w','z2','Vh') else ('cones','reduced_native',name)
            states[k] = value;rhs[k] = evaluation.rhs[self.native_physics.slices[name]]
        for name,value in (('R',R),('E',E)):
            k=('cones','analytic_RE',name);states[k]=value;rhs[k]=evaluation.analytic_rhs[name]
        if queue_node:self._delay_append(t,output['Vh'])
        delayed = self.delay.at(t)
        if self.torch is None:delayed=delayed.values
        started=perf_counter()
        release = self._cone_release(t,side,output)
        bridge_seconds=perf_counter()-started
        inner_outputs,pure,observation_timing = self.inner.observation_witness(t,inner_state,
            release,lambda:self.rod_rate_at(t,side))
        transition_keys = {k for k in inner_state if k[0] in BC or k[1] in (RODS,'RGC_expected_history')}
        transitions = {k:pure[k] for k in transition_keys}
        ports,units = {},{}
        native_units = dict(X='vH_native_precursor',C='vH_native_current',Vis='vH_native_voltage',g_i='vH_native_conductance',
            Vbar='vH_native_voltage',w='vH_native_voltage',z2='vH_native_voltage',Vh='vH_native_voltage',z1_aux='vH_native_voltage',
            R='td',E='td',q='dimensionless',release='dimensionless',Ios='vH_native_current',Bh='vH_native_voltage',aI='dimensionless',pooled_aI='dimensionless')
        for name,value in output.items():
            pop='HC' if name in ('w','z2','Vh','pooled_aI') else 'cones'
            ports[pop,name]=value;units[pop,name]=native_units[name]
        for k,value in inner_outputs.items():
            unit = 'V' if k[-1]=='voltage' else 'S' if k[-1] in ('gE','gI','contact_conductance','conductance') else 'A' if k[-1]=='gap_current' else 'events/s' if k[-1]=='event_rate' else 'dimensionless'
            unit = getattr(self.inner, 'port_unit_overrides', {}).get(k[-1], unit)
            ports[k]=value;units[k]=unit
        ports['HC','delayed_Vh']=delayed;units['HC','delayed_Vh']='vH_native_voltage'
        event_generation=self.event_generation if event_generation is None and generation is None else event_generation if event_generation is not None else generation
        result = dict(time_seconds=float(t),side=side,event_generation=int(event_generation),states=states,rhs=rhs,
            transitions=transitions,ports=ports,port_units=units,native_outputs=output,inner_outputs=inner_outputs,
            delayed_HC=delayed,aux=self._exact_aux(t,side),
            draws=(_snapshot_copy(self.last_draws) if self.mode=='gpu_dop853' else deepcopy(self.last_draws)),
            event_witness_required=bool(self.last_draws),rhs_kind='original_equation',domain_failures=[])
        elapsed=perf_counter()-snapshot_started
        parts=dict(native=native_seconds,rhs=rhs_seconds,bridge=bridge_seconds,
            outputs=observation_timing['outputs_seconds'],transition=observation_timing['transition_seconds'])
        self.metrics['snapshot_count']+=1
        self.metrics['snapshot_wall_seconds']+=elapsed
        for name,seconds in parts.items():self.metrics['snapshot_'+name+'_seconds']+=seconds
        self.metrics['snapshot_other_seconds']+=elapsed-sum(parts.values())
        return result

    def record_native_values(self,t,side='right',snapshot=None):
        output = self._native_evaluation(t,side)[0].outputs if snapshot is None else snapshot['native_outputs']
        result = {}
        for f in self.assembly.recording_fields:
            if f.population not in ('cones','HC'):continue
            value = output[f.name]
            selected = self.native_physics.torch.as_tensor(f.rows,dtype=self.torch.int64,device=self.device) if self.torch is not None else f.rows
            result[f.name] = dict(values=_cpu(value[selected]),group_rows=np.array(f.rows,copy=True))
        return result

    def ack(self,t):
        if not self.time_seconds <= t <= self.inner_time:
            raise ContractError('Public watermark cannot pass accepted inner state or move backwards')
        if t > self.start and self.mode not in ('radau','gpu_dop853'):
            release=self._cone_release(t,'left')
            self._release_cache_time=t;self._release_cache=release[self.selection.cone_rows].copy()
        for _,window in self.readout_windows:
            for consumer in CONSUMERS:
                if t >= window.start:window.acknowledge(consumer,min(t,window.stop))
        self.readout_windows=[(ident,w) for ident,w in self.readout_windows if min(w.consumed.values()) < w.stop]
        self._readout_ids={ident for ident,_ in self.readout_windows}
        self.native_consumed=dict.fromkeys(CONSUMERS,float(t))
        while len(self.native_intervals)>1 and self.native_intervals[0].stop < t:
            self.native_intervals.pop(0)
        self.flush_events(t)
        self.time_seconds=float(t)

    def checkpoint_payload(self,*,monitor=None,recording=None):
        if self.time_seconds != self.inner_time:
            raise ContractError('Complete checkpoint requires public inner state at the common consumer watermark')
        self.flush_events(self.time_seconds)
        state=self.native_state;anchor=state.anchor
        native=dict(point=dict(time_seconds=state.time_seconds,y=state.y,backend_id=state.backend_id,
            next_window_seconds=state.next_window_seconds,anchor=dict(time_seconds=anchor.time_seconds,R=anchor.R,E=anchor.E,
                light=anchor.light,input_stop_seconds=anchor.input_stop_seconds,input_epoch=anchor.input_epoch)),
            intervals=[i.payload() for i in self.native_intervals])
        return dict(native=native,inner={k:v.copy() for k,v in self.inner_state.items()},spike_states=deepcopy(self.spike_states),
            clocks=dict(indices=dict(self.clock_indices),last_commit_times=dict(self.last_commit_times),origin=self.protocol['clock_origin_seconds']),
            input_position=dict(public_frame=self.frame_at(self.time_seconds,'right'),native_frame=anchor.input_epoch),
            observation_delay=self.delay_payload(),
            controller=dict(schema=SCHEMA,identity=self.identity,mode=self.mode,time_seconds=self.time_seconds,
                inner_time=self.inner_time,event_generation=self.event_generation,initial_commit_done=self.initial_commit_done,
                last_draws=deepcopy(self.last_draws),last_events=deepcopy(self.last_events),event_log=deepcopy(self.event_log),
                emitted_events_unrecorded=deepcopy(self.emitted_events_unrecorded),emitted_until=self._emitted_until,metrics=deepcopy(self.metrics),
                **(dict(numerical_backend=dict(
                    native=self.native_backend.core.controller_state() if self.mode=='radau' else None,
                    inner=self.inner_backend.controller_state() if hasattr(self.inner_backend,'controller_state') else None))
                    if self.inner_backend is not None else {})),
            readout_cursor=dict(consumed=dict(self.native_consumed),release_cache_time=self._release_cache_time,
                release_cache=None if self._release_cache is None else self._release_cache.copy(),
                transport_next_id=None if self.transport is None else self.transport.next_id,
                transport_metrics=None if self.transport is None else dict(self.transport.metrics),
                windows=[dict(native_representation_id=ident,consumed=dict(w.consumed),packets=[p.payload() for p in w.packets]) for ident,w in self.readout_windows]),
            monitor={} if monitor is None else monitor,recording={} if recording is None else recording)

    @classmethod
    def from_checkpoint(cls,assembly,construction_bundle_payload,protocol,payload,**kwargs):
        result=cls(assembly,construction_bundle_payload,protocol,**kwargs)
        c=payload['controller']
        if c.get('schema') != SCHEMA or c.get('identity') != result.identity or c.get('mode') != result.mode:
            raise ContractError('Group arm checkpoint source/protocol/mode differs')
        result.inner.validate(payload['inner'])
        result.inner_state={k:np.array(v,copy=True) for k,v in payload['inner'].items()}
        result.native_state=_native_state(result.native_backend,payload['native']['point'])
        expected=result.native_backend.reference_identity if result.mode in ('reference','gpu_dop853') else result.native_backend.identity
        if result.native_state.backend_id != expected:raise ContractError('Checkpoint numerical arm changed')
        result.native_intervals=[(result.native_backend.interval_from_payload(v) if result.mode=='radau'
            else AcceptedNativeInterval.from_payload(result.native_backend,v)) for v in payload['native']['intervals']]
        if result.inner_backend is not None:
            saved = c.get('numerical_backend')
            if not isinstance(saved,dict) or set(saved)!={'native','inner'}:
                raise ContractError('Complete numerical backend controller state required')
            if result.mode=='radau':
                result.native_backend.core.restore_controller(saved['native'])
            if hasattr(result.inner_backend,'restore_controller'):
                result.inner_backend.restore_controller(saved['inner'])
        result.time_seconds,result.inner_time=float(c['time_seconds']),float(c['inner_time'])
        if (not result.start <= result.time_seconds == result.inner_time <= result.native_state.time_seconds <= result.stop
                or len(result.native_intervals)>2 or any(a.stop!=b.start for a,b in zip(result.native_intervals,result.native_intervals[1:]))
                or result.native_intervals and result.native_intervals[-1].stop != result.native_state.time_seconds
                or result.native_state.time_seconds > result.time_seconds and (not result.native_intervals or result.native_intervals[0].start > result.time_seconds)):
            raise ContractError('Checkpoint lost the native future frontier or an unconsumed interval')
        result.spike_states=deepcopy(payload['spike_states'])
        result.event_generation=int(c['event_generation']);result.initial_commit_done=c['initial_commit_done']
        result.last_draws=deepcopy(c['last_draws']);result.last_events=deepcopy(c['last_events']);result.event_log=deepcopy(c['event_log'])
        result.emitted_events_unrecorded=deepcopy(c['emitted_events_unrecorded']);result._emitted_until=float(c['emitted_until'])
        if result._emitted_until != result.time_seconds:
            raise ContractError('Checkpoint retains events past the public delivery watermark')
        result.metrics=deepcopy(c['metrics'])
        if payload['clocks']['origin'] != protocol['clock_origin_seconds']:
            raise ContractError('Checkpoint original clock origin changed')
        result.clock_indices=dict(payload['clocks']['indices']);result.last_commit_times=dict(payload['clocks']['last_commit_times'])
        if set(result.clock_indices) != {'rod','bc','rgc'} or set(result.last_commit_times) != set(result.clock_indices):
            raise ContractError('Checkpoint original clock coverage differs')
        d=payload['observation_delay']
        result.delay=TimestampedDelay.from_payload(d) if result.torch is None else ResidentTimestampedDelay.from_payload(d,result.torch,result.device)
        if (result.delay.delay_seconds != .00282 or result.delay.interpolation != 'linear'
                or not all(math.isfinite(t) for t in result.delay.history_times)
                or np.any(np.diff(result.delay.history_times)<=0)
                or any(t>result.time_seconds for t in result.delay.history_times)
                or result.delay.prehistory_stop_seconds != result.start
                or not np.array_equal(np.asarray(d['prehistory']['values'],dtype=np.float64),result._prehistory_aux_values)):
            raise ContractError('Checkpoint observation queue contains native future samples')
        cursor=payload['readout_cursor']
        if set(cursor['consumed']) != set(CONSUMERS) or any(t!=result.time_seconds for t in cursor['consumed'].values()):
            raise ContractError('Checkpoint public consumers differ')
        result.native_consumed=dict(cursor['consumed']);result._release_cache_time=cursor['release_cache_time']
        result._release_cache=None if cursor['release_cache'] is None else np.array(cursor['release_cache'],copy=True)
        result.readout_windows=[]
        owners={i.representation_id:i for i in result.native_intervals}
        for w in cursor['windows']:
            if w['native_representation_id'] not in owners:raise ContractError('Readout checkpoint has no accepted native owner')
            owner=owners[w['native_representation_id']]
            packets=[ContinuousReleasePacket.from_payload(result.selection,p) for p in w['packets']]
            window=OwnedReadoutWindow(packets)
            if (window.start!=owner.start or window.stop!=owner.stop
                    or any(p.source_version!=owner.source_version or p.input_epoch!=owner.input_epoch for p in packets)):
                raise ContractError('Readout checkpoint source/time/epoch differs from its accepted native owner')
            if set(w['consumed'])!=set(CONSUMERS) or any(not window.start<=t<=window.stop for t in w['consumed'].values()):
                raise ContractError('Readout checkpoint consumer watermark invalid')
            window.consumed=dict(w['consumed']);result.readout_windows.append((w['native_representation_id'],window))
        result._readout_ids={i for i,_ in result.readout_windows}
        if result.transport is not None:
            if type(cursor['transport_next_id']) is not int or cursor['transport_next_id'] < 0:
                raise ContractError('Checkpoint transfer producer index missing')
            ids=[p.segment_id for _,w in result.readout_windows for p in w.packets]
            if ids and cursor['transport_next_id'] <= max(ids):
                raise ContractError('Checkpoint transfer producer index may reuse a pending packet')
            result.transport.next_id=cursor['transport_next_id'];result.transport.metrics=dict(cursor['transport_metrics'])
        if payload['input_position'] != dict(public_frame=result.frame_at(result.time_seconds,'right'),native_frame=result.native_state.anchor.input_epoch):
            raise ContractError('Checkpoint original input position differs')
        return result

    def close(self):
        if self.inner_backend is not None and hasattr(self.inner_backend,'close'):
            self.inner_backend.close()
        if self.gpu_rhs is not None:
            self.gpu_rhs.close()
        self.native_backend.close()
        if self.torch is not None and self._owns_plans:
            self.native_physics.A.close();self.native_physics.B.close()
