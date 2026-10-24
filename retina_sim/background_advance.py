"""State-only background interface for the reviewed frozen joint event loop.

No integrator, event/history/delay/RNG or observation point is removed. The
original model/solver identity and checkpoint codec remain intact. Empty trace
evaluation and result/metadata assembly are omitted only for source-pinned pure
readouts and a mandatory background observer. No runtime code generation.
"""
from pathlib import Path
from dataclasses import replace
import hashlib,inspect,json,textwrap
import numpy as np
from .contracts import ContractError
from .engine import validate_state
from .hybrid import HybridRunner,RetinaState
from .network import NetworkStimulus
from .recording import RecordingPlan,TimestampedDelay,fork_timestamped_delay_payloads

LEGACY_ADVANCE_SHA256='64194d0ce68ecfdfd1eab43fde372ce257e600f469ed8ee3f66d8e526013708e'
PURE_FACTORY_SOURCES={'vh2005.py': '60c1091971527216961df8a49b40e3dad8125f44de1d82d4d9374afcd4462858', 'inner.py': '32c469e6a1de2c75a28b1bf6c6d9fcac8c6942f0391386d60608b95688bac0e3', 'rod_pathway.py': '5601dd757e22177fb02ebd72a4305ea145f8ddd05aa130822fe43f95b27c862c', 'bc_gain_control.py': 'db45db6c6f107b1655ab6c26bcd96643ae0afec3d40fdbd9434de30755888e01', 'physical.py': '6bd9ae8413388086ce28f7b55489944964b283dc66bea8ec5159f9d4838c0813', 'ac_inhibition.py': '1e170f21f1617eb953116905d2fefdfdeaa9dd3b697e630eab08cf9d20301873'}
PURE_MECHANISMS=frozenset(['Chen2024PrimateRodCurrent', 'DoubleExponentialChemicalSynapse', 'ExplicitPhysicalUnitConversion', 'HistoryBipolarTerminal', 'HistoryConductanceRGC', 'ReceptorConductanceMembrane', 'ResidualAC', 'ResidualBC_OFF', 'ResidualBC_ON', 'RodContactTRPM1', 'SynapticConductance', 'VanHateren2005NativeConeH1', 'EffectiveInhibitoryAC'])


class _EmptyRecordingPlan:
    ports=();states=();cell_ids=()
    def __init__(self,max_bytes):self.max_bytes=max_bytes
    def indices(self,ids):return np.empty(0,dtype=int)


class StateOnlyBackgroundRunner:
    """Adapter used only by prepare_background; responses retain the real runner.

    Audited observe functions construct outputs without mutating model state.
    Causal-history transitions and PCG64 draws occur in the unchanged _sample.
    The original mandatory observer still evaluates the full model, all state
    domains and transitions at every old numerical/event observation point.
    """
    def __init__(self,runner):
        if runner.discrete is not None:raise ContractError('State-only background requires a compiled joint model')
        actual=hashlib.sha256(textwrap.dedent(inspect.getsource(HybridRunner.advance)).encode()).hexdigest()
        if actual!=LEGACY_ADVANCE_SHA256:raise ContractError('Unreviewed legacy advance change; state-only interface unavailable')
        for name,expected in PURE_FACTORY_SOURCES.items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=expected:
                raise ContractError('Unreviewed observe implementation; state-only interface unavailable: '+name)
        if any(c.mechanism not in PURE_MECHANISMS or c.implementation_hash not in PURE_FACTORY_SOURCES.values()
               for c in runner.compiled.components.values()):
            raise ContractError('State-only interface requires reviewed pure component readouts')
        self.original=runner
        self.procedure_identity={'procedure':'state_only_background_v1',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'legacy_advance_sha256':LEGACY_ADVANCE_SHA256,'pure_factory_sources':dict(PURE_FACTORY_SOURCES),
            'neural_model_solver_graph_criteria_changed':False,
            'output_scope':'No SimulationResult, empty trace readout, population metadata or empty-ID scan',
            'observer_scope':'All original background monitor, event, dense and endpoint observations retained'}
    def __getattr__(self,name):return getattr(self.original,name)
    def advance(self,state,channels,stop,solver_config=None,sample_times=None,recording_plan=None,state_observer=None):
        p=recording_plan
        if (type(p) is not RecordingPlan or p.ports!=() or p.states!=() or p.cell_ids is None or len(p.cell_ids)):
            raise ContractError('State-only background requires an explicit empty RecordingPlan')
        if state_observer is None:raise ContractError('State-only background requires the original numerical observer')
        return _advance_state_only(self.original,state,channels,stop,solver_config,sample_times,p,state_observer),None


def choose_background_runner(runner):
    """Conservative output optimization; unknown readouts keep the legacy path."""
    try:
        selected=StateOnlyBackgroundRunner(runner)
        return selected,selected.procedure_identity
    except ContractError as exc:
        return runner,{'procedure':'legacy_trace_background_v1','reason':str(exc),
            'neural_model_solver_graph_criteria_changed':False}


class StateOnlyObservedRunner(StateOnlyBackgroundRunner):
    """Explicit empty-trace transport for an observed time-varying stimulus.

    The existing state-only event loop has no constant-input assumption: it
    constructs the original NetworkStimulus, keeps its breakpoints and calls
    the original runner's integrator, sampling and mandatory observer. This
    separate identity permits only empty traces, not omitted observation points
    or a background-convergence claim for a movie response.
    """
    def __init__(self,runner):
        super().__init__(runner)
        self.procedure_identity=dict(self.procedure_identity,
            procedure='state_only_observed_timevarying_v1',
            observer_scope='Mandatory caller state observer at every unchanged event, dense and endpoint observation; no background-convergence claim',
            input_scope='Original NetworkStimulus validation, analytic/ZOH values, breakpoints and left limits retained')


def choose_observed_runner(runner):
    """Use the same reviewed empty-record event loop or conservatively fall back."""
    try:
        selected=StateOnlyObservedRunner(runner)
        return selected,selected.procedure_identity
    except ContractError as exc:
        return runner,{'procedure':'legacy_trace_observed_v1','reason':str(exc),
            'neural_model_solver_graph_criteria_changed':False}


def _advance_state_only(self,state,channels,stop,solver_config=None,sample_times=None,recording_plan=None,state_observer=None):
    if not isinstance(state,RetinaState) or state.assembly_hash!=self.compiled.assembly_hash: raise ContractError('State does not belong to compiled retina')
    validate_state(state.dynamic,self.model)
    config=solver_config or state.solver_config
    if config!=state.solver_config: raise ContractError('Solver changes across chunks require explicit reinitialization')
    start=state.time_seconds
    if not np.isfinite(stop) or stop<=start: raise ContractError('Advance must increase absolute time')
    stimulus=NetworkStimulus(self.compiled,channels,state,stop,config.input_interpolation,self.model.evidence_tag)
    if self.discrete is not None:raise ContractError('Background state-only requires the compiled joint model')

    plan=_EmptyRecordingPlan(recording_plan.max_bytes)
    times=np.array([start,stop] if sample_times is None else sample_times,dtype=float)
    if times.ndim!=1 or not len(times) or np.any(~np.isfinite(times)) or np.any(np.diff(times)<=0) or times[0]<start or times[-1]>stop: raise ContractError('Record times must be increasing within advance')
    scalar_count=sum(len(plan.indices(s.cell_ids)) for s in self.model.state_specs if plan.states is None or s.state_name in plan.states)
    scalar_count+=sum(len(plan.indices(p.cell_ids)) for cid,c in self.compiled.components.items() for name,p in c.outputs.items() if plan.ports is None or cid+'.'+name in plan.ports)
    extra_scalars=sum(len(plan.indices(c.outputs['Vh'].cell_ids)) for c in self.compiled.components.values() if hasattr(c.native_model,'observation_interval_seconds') and (plan.ports is None or c.component_id+'.observed_Vh' in plan.ports))
    spike_cells=sum(len(plan.indices(c.state_specs[0].cell_ids)) for c in self.compiled.components.values() if hasattr(c.native_model,'sample_spikes'))
    extra_scalars+=spike_cells
    event_bound=spike_cells*(int(np.ceil((stop-start)/self.history_dt))+2) if self.history_dt else 0
    metadata_bound=1024*(len(self.compiled.components)+sum(len(plan.indices(p.cells.cell_id)) for p in self.compiled.populations.values()))
    estimate=8*len(times)*(scalar_count+extra_scalars+1)+128*event_bound+16*spike_cells+metadata_bound
    if estimate>plan.max_bytes: raise ContractError(f'Recording requires at least {estimate} bytes; select cells/windows or stream, never thin anatomy')
    # Functional API: fork mutable histories/RNG rather than mutate the input state.
    state=RetinaState(state.dynamic,state.assembly_hash,state.grid_origin_seconds,state.sample_index,json.loads(json.dumps(state.spike_states)),fork_timestamped_delay_payloads(state.delay_states),dict(state.input_position),json.loads(json.dumps(state.link_delay_states)),dict(state.component_tick_indices))
    rows={}; declarations={}; recorded=[]; record_ids={}; events={cid:[] for cid in state.spike_states}
    def flush_events(t):
        for cid,aux in state.spike_states.items():
            ids=self.compiled.components[cid].state_specs[0].cell_ids
            for e in aux['pending_events']:
                if start<e['time_seconds']<=t and (plan.cell_ids is None or ids[e['cell_index']] in plan.cell_ids):
                    events[cid].append(dict(e,cell_id=ids[e['cell_index']]))
            aux['pending_events']=[e for e in aux['pending_events'] if e['time_seconds']>t]
    def record(t,y):
        recorded.append(float(t))
    def record_requested(t,y=None):
        # Event endpoints and requested dense times can differ by an ULP.
        # Retain the requested stamp after the shared history transition,
        # using the existing 1e-12 recording coverage tolerance. This does
        # not move an integration/event endpoint or consume any RNG draw.
        index=len(recorded)
        if index<len(times) and abs(t-times[index])<=1e-12:
            record(float(times[index]),None)
    def update_delays(t,y):
        for cid,data in state.delay_states.items():
            delay=TimestampedDelay.from_payload(data)
            if t>delay.history_times[-1]:
                c=self.compiled.components[cid];q=c.observe('Vh',t,self.model.local_blocks(cid,y),{});delay.append(t,q);state.delay_states[cid]=delay.payload()
    def publish_links(t,y,left_limit=False):
        if not stimulus.delay_buffers:return
        tin=np.nextafter(t,-np.inf) if left_limit and t>start else t
        bundle=replace(stimulus.at(tin),delay_left_limit=left_limit)
        _,signals=self.model.evaluate(t,y,bundle)
        for link in self.compiled.links:
            if link.delayed_operator is not None:
                key=link.delayed_operator.key;source=(link.specification.source_component,link.specification.source_port)
                stimulus.delay_buffers[key].append(t,signals[source])
    breaks=iter(stimulus.breakpoints_seconds); next_break=next(breaks,stop)
    while state.time_seconds<stop-1e-14:
        current=state.time_seconds
        publish_links(current,self._flat(state))
        if self.history_dt is not None:
            grid=min(state.grid_origin_seconds+state.component_tick_indices[cid]*period for cid,period in self.periods.items()) if self.event_clock else state.grid_origin_seconds+state.sample_index*self.history_dt
            if grid<current-1e-10: raise ContractError('History sampling phase lost')
            if abs(grid-current)<1e-10:
                self._sample(state,stimulus);publish_links(current,self._flat(state))
        record_requested(current)
        if state_observer is not None:state_observer(current,self._flat(state),stimulus.at(current))
        while next_break<=current+1e-14: next_break=next(breaks,stop)
        next_grid=(min(state.grid_origin_seconds+state.component_tick_indices[cid]*period for cid,period in self.periods.items()) if self.event_clock else state.grid_origin_seconds+state.sample_index*self.history_dt) if self.history_dt is not None else stop
        next_observation=state.grid_origin_seconds+(int(np.floor((current-state.grid_origin_seconds)/self.observation_dt+1e-8))+1)*self.observation_dt if self.observation_dt else stop
        next_delay=state.grid_origin_seconds+(int(np.floor((current-state.grid_origin_seconds)/self.delayed_step+1e-8))+1)*self.delayed_step if self.delayed_step else stop
        arrival=min((stimulus.delay_buffers[link.delayed_operator.key].next_arrival(current,link.delayed_operator.delays_seconds[link.delayed_operator.delays_seconds>0]) for link in self.compiled.links if link.delayed_operator is not None),default=stop)
        target=min(float(stop),float(next_break),float(next_grid),float(next_observation),float(next_delay),float(arrival))
        rows_state=self._integrate(state,stimulus,target,config,times)
        flush_events(target)
        for t,y in rows_state.items():
            update_delays(t,y)
            if state_observer is not None:state_observer(t,y,stimulus.at(t))
            if abs(t-next_grid)>1e-10:record_requested(t,y)
        publish_links(target,self._flat(state),left_limit=True)
        for buffer in stimulus.delay_buffers.values():buffer.prune(target)
    final_tick=(min(state.grid_origin_seconds+state.component_tick_indices[cid]*period for cid,period in self.periods.items()) if self.event_clock else state.grid_origin_seconds+state.sample_index*self.history_dt) if self.history_dt is not None else None
    if final_tick is not None and abs(final_tick-stop)<1e-12:
        self._sample(state,stimulus)
    publish_links(stop,self._flat(state))
    if state_observer is not None:state_observer(stop,self._flat(state),stimulus.at(stop))
    state.link_delay_states={key:buffer.payload() for key,buffer in stimulus.delay_buffers.items()}
    if not recorded or recorded[-1]<times[-1]-1e-12: record(float(times[-1]),None)
    flush_events(stop)
    if len(recorded)!=len(times) or not np.allclose(recorded,times,rtol=0,atol=1e-12):raise ContractError('Recording timestamp coverage lost')
    state.input_position.update(last_time_seconds=float(stop),contracts=stimulus.input_contract)
    return state
