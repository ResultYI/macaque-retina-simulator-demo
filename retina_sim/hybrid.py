"""Continuous integration with timestamped causal-history transitions and spike bins."""
from dataclasses import dataclass,field,asdict,replace
import json
import numpy as np
from scipy.integrate import solve_ivp
from .contracts import ContractError,Quantity,SimulationState,SimulationResult,SolverConfig,canonical_hash
from .engine import initialize_state,validate_state,checkpoint,restore,advance as native_advance
from .chen2024 import author_discrete_config
from .network import JointNetwork,NetworkStimulus,execution_metadata
from .recording import RecordingPlan,TimestampedDelay,fork_timestamped_delay_payloads
from .provenance import CoverageReport

@dataclass(frozen=True)
class NetworkSolverConfig:
    method: str
    rtol: float = 1e-8
    atol: float = 1e-10
    max_step: float = .0001
    dtype: str = 'float64'
    input_interpolation: str = 'analytic'
    fixed_step_seconds: float | None = None
    def __post_init__(self):
        if self.method!='RK2' or self.fixed_step_seconds is None or self.fixed_step_seconds<=0 or self.dtype!='float64' or self.input_interpolation not in ('analytic','linear','zero_order_hold'): raise ContractError('Explicit RK2 integration contract required')

@dataclass
class RetinaState:
    dynamic: SimulationState
    assembly_hash: str
    grid_origin_seconds: float
    sample_index: int
    spike_states: dict = field(default_factory=dict)
    delay_states: dict = field(default_factory=dict)
    input_position: dict = field(default_factory=dict)
    link_delay_states: dict = field(default_factory=dict)
    component_tick_indices: dict = field(default_factory=dict)
    def __getattr__(self,key):
        # copy/deepcopy probes special methods before reconstructed fields exist.
        dynamic=object.__getattribute__(self,'dynamic')
        return getattr(dynamic,key)

class HybridRunner:
    def __init__(self,compiled,*,compute_backend='auto'):
        self.compiled=compiled
        first=next(iter(compiled.components.values()))
        self.discrete=first if len(compiled.components)==1 and first.time_contract.kind=='author_discrete' and hasattr(first.native_model,'initialize') else None
        self.model=self.discrete.native_model if self.discrete else JointNetwork(compiled,compute_backend=compute_backend)
        self.history_dt=compiled.clock['history_base_seconds']
        self.strides=compiled.clock['component_tick_strides']
        self.event_clock=compiled.clock.get('scheduler_kind')=='explicit_event_clock'
        self.periods=compiled.clock['component_periods_seconds']
        self.integration_metrics={'solver_calls':0,'nfev':0,'njev':0,'nlu':0,'integration_wall_seconds':0.,'minimum_interval_seconds':None,'maximum_nfev_per_interval':0}
        self.delayed_step=compiled.clock['delayed_source_step_seconds']
        observations=[c.native_model.observation_interval_seconds for c in compiled.components.values() if hasattr(c.native_model,'observation_interval_seconds')]
        self.observation_dt=min(observations) if observations else None

    def initialize(self,solver_config=None,time_seconds=0):
        config=solver_config or (author_discrete_config() if self.discrete else SolverConfig())
        if self.discrete:
            if config!=author_discrete_config(): raise ContractError('Native author step contract mismatch')
            dynamic=self.model.initialize(time_seconds)
        else: dynamic=initialize_state(self.model,self.model.parameters,self.model.graph,self.model.initial_conditions(),config,time_seconds)
        spikes={}; delays={}
        for cid,c in self.compiled.components.items():
            native=c.native_model
            if hasattr(native,'sample_spikes'):
                rng=np.random.Generator(np.random.PCG64(native.rng_seed))
                spikes[cid]={'history':native.initial_spike_history.tolist(),'last_bin_counts':None,'rng_state':rng.bit_generator.state,'pending_events':[],'conditional_rate_Hz':native.intensity(native.V0,native.initial_spike_history).tolist()}
            if hasattr(native,'observation_interval_seconds'):
                blocks=tuple(q.values for q in native.initial_conditions()); q=native.observe('Vh',time_seconds,blocks,{})
                delay=TimestampedDelay(native.p['observation_delay']*.001,[float(time_seconds)],[q.values.tolist()],q,'linear')
                delays[cid]=delay.payload()
        link_states={link.delayed_operator.key:link.delayed_operator.initialize(time_seconds).payload() for link in self.compiled.links if link.delayed_operator is not None}
        return RetinaState(dynamic,self.compiled.assembly_hash,float(time_seconds),0,spikes,delays,{},link_states,{cid:(1 if compiled_c.time_contract.kind=='author_discrete' else 0) for cid,compiled_c in self.compiled.components.items() if cid in self.periods})

    def _flat(self,state):return np.concatenate(state.ordered_state_arrays) if state.ordered_state_arrays else np.empty(0)

    def _replace(self,state,y,time):
        d=state.dynamic
        state.dynamic=SimulationState(float(time),self.model.unpack(y),d.state_spec_hash,d.graph_hash,d.parameter_hash,d.model_id,d.model_version,d.model_hash,d.input_interpolation,d.solver_config,d.evidence_tag)

    def _sample(self,state,stimulus):
        t=state.time_seconds; y=self._flat(state)
        due={cid for cid,period in self.periods.items() if abs(state.grid_origin_seconds+state.component_tick_indices[cid]*period-t)<1e-12} if self.event_clock else {cid for cid,stride in self.strides.items() if state.sample_index%stride==0}
        inputs,outputs=self.model.evaluate(t,y,stimulus.at(t),sampling=True,sampled_components=due)
        # Commit all causal histories from one pre-transition snapshot and joint closure.
        changed=y.copy()
        for cid,c in self.compiled.components.items():
            if c.time_contract.kind=='causal_history' and cid in due:
                local=c.native_model.transition(self.model.local_blocks(cid,y),inputs[cid],{port:outputs[(cid,port)] for port in c.outputs})
                changed[self.model.slices[cid]]=np.concatenate(local)
            if c.time_contract.kind=='author_discrete' and cid in due and t>state.grid_origin_seconds+1e-12:
                # Author update consumes the interval-start rate once; output changes only at its end.
                previous=t-c.time_contract.fixed_step_seconds
                external=next((e for e in self.compiled.spec.external_inputs if e.target_component==cid and e.target_port=='light'),None)
                if external is None:raise ContractError('Author light requires explicit external total-rate input')
                channel=stimulus.channels[external.channel]
                if channel.interpolation!='zero_order_hold' or channel.input_contract.get('grid_step_seconds')!=c.time_contract.fixed_step_seconds or channel.input_contract.get('includes_background') is not True:raise ContractError('Mixed author input requires exact total-rate ZOH grid')
                if previous<channel.start_seconds-1e-12:raise ContractError('Author interval-start input unavailable; preserve input across chunk boundary')
                incoming={'light':channel.at(max(channel.start_seconds,previous+1e-13))}
                local=c.native_model.transition(self.model.local_blocks(cid,y),incoming)
                changed[self.model.slices[cid]]=np.concatenate(local)
        self._replace(state,changed,t)
        if self.event_clock:
            for cid in due:state.component_tick_indices[cid]+=1
        for cid,aux in list(state.spike_states.items()):
            if cid not in due:continue
            c=self.compiled.components[cid]; V=self.model.local_blocks(cid,changed)[0]
            updated,_=c.native_model.sample_spikes(t,V,aux); state.spike_states[cid]=updated
        state.sample_index+=1

    def _integrate(self,state,stimulus,stop,config,samples):
        start=state.time_seconds; y=self._flat(state)
        def rhs(t,a):
            return self.model.rhs_flat(t,a,stimulus.at(t,left_limit=bool(t>=stop)),self.model.parameters,self.model.graph)
        requested=np.array(sorted(set(float(x) for x in samples if start<x<=stop)))
        if config.method=='RK2':
            targets=sorted(set(requested.tolist()+[float(stop)])); rows={}; current=start
            for target in targets:
                while current<target-1e-15:
                    dt=min(config.fixed_step_seconds,target-current); a=rhs(current,y); predictor=y+dt*a
                    y=y+dt*.5*(a+rhs(current+dt,predictor)); current+=dt
                    if not np.all(np.isfinite(y)): raise ContractError('RK2 nonfinite state; no clipping')
                rows[target]=y.copy()
            self._replace(state,y,stop); return rows
        targets=np.array(sorted(set(requested.tolist()+[float(stop)])))
        if len(y):
            from time import perf_counter
            from .solver_lifetime import collection_epoch,release_solver_cycles
            endpoint_only=not np.any(requested<stop)
            epoch=collection_epoch() if config.method=='DOP853' else None
            metrics=self.integration_metrics;copy_seconds=0.;solution=None
            try:
                tick=perf_counter()
                solution=solve_ivp(rhs,(start,stop),y,method=config.method,rtol=config.rtol,atol=config.atol,max_step=config.max_step,t_eval=None if endpoint_only else targets,dense_output=False)
                metrics['solver_calls']+=1
                for key in ('nfev','njev','nlu'):metrics[key]+=getattr(solution,key,0)
                metrics['integration_wall_seconds']+=perf_counter()-tick
                metrics['minimum_interval_seconds']=min(metrics['minimum_interval_seconds'] or (stop-start),stop-start)
                metrics['maximum_nfev_per_interval']=max(metrics['maximum_nfev_per_interval'],solution.nfev)
                if not solution.success: raise ContractError('Joint integration failed: '+solution.message)
                # A successful terminal event is not arrival at the requested stop.
                # Never label an earlier accepted state with a later timestamp.
                if solution.status!=0 or not len(solution.t) or solution.t[-1]!=stop:
                    raise ContractError('Joint integration did not reach the requested endpoint')
                tick=perf_counter()
                final=solution.y[:,-1].copy()  # Do not retain all accepted-step columns through a final-state view.
                copy_seconds=perf_counter()-tick
                rows={float(stop):final} if endpoint_only else {float(t):a for t,a in zip(solution.t,solution.y.T)}
                self._replace(state,final,stop)
            finally:
                solution=None
                metrics['solver_result_copy_wall_seconds']=metrics.get('solver_result_copy_wall_seconds',0.)+copy_seconds
                metrics['integration_wall_seconds']+=copy_seconds
                if epoch is not None:
                    cleanup=release_solver_cycles(epoch)
                    for label in ('wall_seconds','CPU_seconds','collected_objects'):
                        key='solver_lifetime_gc_'+label
                        metrics[key]=metrics.get(key,0)+cleanup[label]
                    key='solver_lifetime_gc_generation_'+str(cleanup['generation'])+'_calls'
                    metrics[key]=metrics.get(key,0)+1
                    metrics['integration_wall_seconds']+=cleanup['wall_seconds']
        else:
            rows={float(t):y for t in targets}; self._replace(state,y,stop)
        return rows

    def advance(self,state,channels,stop,solver_config=None,sample_times=None,recording_plan=None,state_observer=None):
        if not isinstance(state,RetinaState) or state.assembly_hash!=self.compiled.assembly_hash: raise ContractError('State does not belong to compiled retina')
        validate_state(state.dynamic,self.model)
        config=solver_config or state.solver_config
        if config!=state.solver_config: raise ContractError('Solver changes across chunks require explicit reinitialization')
        start=state.time_seconds
        if not np.isfinite(stop) or stop<=start: raise ContractError('Advance must increase absolute time')
        stimulus=NetworkStimulus(self.compiled,channels,state,stop,config.input_interpolation,self.model.evidence_tag)
        if self.discrete:
            plan=recording_plan or RecordingPlan()
            native_samples=int(np.ceil((stop-start)/self.discrete.time_contract.fixed_step_seconds))+2
            allocation_bound=native_samples*(16*sum(s.shape[0] for s in self.discrete.state_specs)+8)
            if allocation_bound>plan.max_bytes:raise ContractError('Native source allocation exceeds window budget; stream shorter windows; anatomy is never thinned')
            external=self.compiled.spec.external_inputs[0]
            new,result=native_advance(self.model,state.dynamic,channels[external.channel],stop,config,sample_times)
            state=RetinaState(new,state.assembly_hash,state.grid_origin_seconds,state.sample_index,state.spike_states,state.delay_states,dict(state.input_position,last_time_seconds=float(stop),contracts=stimulus.input_contract),state.link_delay_states,state.component_tick_indices)
            metadata=dict(result.metadata,**execution_metadata(self.compiled,plan.cell_ids))
            outputs={self.discrete.component_id+'.'+k:q for k,q in result.outputs_with_units.items() if k!='time'};outputs['time']=result.outputs_with_units['time']
            ids=self.discrete.populations[0].cells.cell_id;indices=plan.indices(ids)
            outputs={k:(q if k=='time' or q.values.ndim==0 else Quantity(q.values[indices],q.unit,q.meaning,q.evidence_tag)) for k,q in outputs.items() if k=='time' or plan.ports is None or k in plan.ports}
            metadata.update({'recorded_cell_ids':{k:[ids[i] for i in indices] for k,q in outputs.items() if k!='time' and q.values.ndim>0},'recording_allocation_bound_bytes':allocation_bound})
            return state,SimulationResult(result.run_mode,self.compiled.spec.model_id,result.source_manifest_hash,result.input_contract,outputs,result.coverage_report,result.numeric_checks,result.claim_level,metadata)
        plan=recording_plan or RecordingPlan()
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
            _,signals=self.model.evaluate(t,y,stimulus.at(min(t,np.nextafter(stop,start)) if stimulus.interpolation=='zero_order_hold' else t))
            local=self.model.unpack(y)
            values={}
            for s,a in zip(self.model.state_specs,local):
                if plan.states is None or s.state_name in plan.states:
                    indices=plan.indices(s.cell_ids); key='state:'+s.state_name
                    if len(indices): values[key]=Quantity(a[indices],s.unit,s.meaning,self.model.evidence_tag);record_ids[key]=[s.cell_ids[i] for i in indices]
            for (cid,port),q in signals.items():
                key=cid+'.'+port;p=self.compiled.components[cid].outputs[port]
                if plan.ports is None or key in plan.ports:
                    indices=plan.indices(p.cell_ids)
                    if len(indices): values[key]=Quantity(q.values[indices],q.unit,q.meaning,q.evidence_tag);record_ids[key]=[p.cell_ids[i] for i in indices]
            for cid,data in state.delay_states.items():
                key=cid+'.observed_Vh'
                if plan.ports is None or key in plan.ports:
                    delay=TimestampedDelay.from_payload(data); q=delay.at(t);p=self.compiled.components[cid].outputs['Vh'];indices=plan.indices(p.cell_ids)
                    if len(indices): values[key]=Quantity(q.values[indices],q.unit,q.meaning,q.evidence_tag);record_ids[key]=[p.cell_ids[i] for i in indices]
            for cid,aux in state.spike_states.items():
                key=cid+'.conditional_rate';c=self.compiled.components[cid];indices=plan.indices(c.state_specs[0].cell_ids)
                if (plan.ports is None or key in plan.ports) and len(indices):
                    values[key]=Quantity(np.array(aux['conditional_rate_Hz'])[indices],'Hz','conditional_spike_intensity',c.evidence_tag);record_ids[key]=[c.state_specs[0].cell_ids[i] for i in indices]
            for key,q in values.items(): rows.setdefault(key,[]).append(q.values);declarations[key]=(q.unit,q.meaning,q.evidence_tag)
            recorded.append(float(t))
        def record_requested(t,y=None):
            # Event endpoints and requested dense times can differ by an ULP.
            # Retain the requested stamp after the shared history transition,
            # using the existing 1e-12 recording coverage tolerance. This does
            # not move an integration/event endpoint or consume any RNG draw.
            index=len(recorded)
            if index<len(times) and abs(t-times[index])<=1e-12:
                record(float(times[index]),self._flat(state) if y is None else y)
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
        if not recorded or recorded[-1]<times[-1]-1e-12: record(float(times[-1]),self._flat(state))
        flush_events(stop)
        outputs={key:Quantity(np.stack(arrays,axis=1),*declarations[key]) for key,arrays in rows.items()}
        for cid,aux in state.spike_states.items():
            c=self.compiled.components[cid];ids=c.state_specs[0].cell_ids;indices=plan.indices(ids)
            counts=np.bincount([e['cell_index'] for e in events[cid]],minlength=len(ids))[indices]
            for suffix,values,unit,meaning in [('spike_count',counts,'dimensionless','observed_window_spike_count'),('window_rate',counts/(stop-start),'Hz','observed_window_firing_rate')]:
                key=cid+'.'+suffix
                if len(indices) and (plan.ports is None or key in plan.ports):outputs[key]=Quantity(values,unit,meaning,c.evidence_tag);record_ids[key]=[ids[i] for i in indices]
        outputs['time']=Quantity(recorded,'s','absolute_time',self.model.evidence_tag)
        metadata=execution_metadata(self.compiled,plan.cell_ids); metadata.update({'recorded_cell_ids':record_ids,'spike_events':events,'spike_state_metadata':{cid:{'conditional_rate_Hz':np.array(aux['conditional_rate_Hz'])[plan.indices(self.compiled.components[cid].state_specs[0].cell_ids)].tolist(),'observation_scheme':aux.get('observation_scheme')} for cid,aux in state.spike_states.items()},'event_count':{cid:len(e) for cid,e in events.items()},'event_window_seconds':[start,stop],'conditional_rate_is_not_observed_window_rate':True,'clock_contract':self.compiled.clock,'link_delay_buffer_count':len(state.link_delay_states),'history_interval_seconds':self.history_dt,'observation_interval_seconds':self.observation_dt,'sampling_index':state.sample_index,'recording_estimate_bytes':estimate,'observation_delay':'Source final delay is timestamped readout only; interpolation explicitly linear; not inserted in feedback','global_state_layout':[{'name':s.state_name,'size':s.shape[0],'unit':s.unit,'meaning':s.meaning} for s in self.model.state_specs]})
        metadata['compute_backend']=self.model.compute_metadata
        if len(recorded)!=len(times) or not np.allclose(recorded,times,rtol=0,atol=1e-12):raise ContractError('Recording timestamp coverage lost')
        state.input_position.update(last_time_seconds=float(stop),contracts=stimulus.input_contract)
        result=SimulationResult(self.compiled.run_mode,self.compiled.spec.model_id,canonical_hash({k:c.parameters.source_manifest_hash for k,c in self.compiled.components.items()}),stimulus.input_contract,outputs,CoverageReport(True,(),claim_limit='Executable architecture/reference or explicit software fixture; no physiological completion'),{'solver':asdict(config),'history_grid_held_fixed':True},'source_native_or_software_integration',metadata)
        return state,result

    def checkpoint(self,state):
        validate_state(state.dynamic,self.model)
        data={'assembly_hash':self.compiled.assembly_hash,'dynamic':json.loads(checkpoint(state.dynamic)),'grid_origin_seconds':state.grid_origin_seconds,'sample_index':state.sample_index,'spike_states':state.spike_states,'delay_states':state.delay_states,'input_position':state.input_position,'link_delay_states':state.link_delay_states,'component_tick_indices':state.component_tick_indices}
        return json.dumps({'version':'retina-architecture-1.0','payload':data,'payload_hash':canonical_hash(data)},sort_keys=True,allow_nan=False)
    def restore(self,serialized,*,expected_input_context=None):
        p=json.loads(serialized);d=p['payload']
        if p['version']!='retina-architecture-1.0' or p['payload_hash']!=canonical_hash(d) or d['assembly_hash']!=self.compiled.assembly_hash: raise ContractError('Checkpoint payload/assembly identity mismatch')
        if expected_input_context is not None and any(d['input_position'].get(k)!=v for k,v in expected_input_context.items()):
            raise ContractError('Checkpoint input/background context mismatch')
        expected={'state_spec_hash':canonical_hash(self.model.state_specs),'graph_hash':self.model.graph.graph_hash,'parameter_hash':self.model.parameters.parameter_hash,'model_hash':self.model.model_hash,'model_id':self.model.model_id,'model_version':self.model.version}
        if d['dynamic']['state']['solver_config']['method']=='RK2':
            raw=d['dynamic']['state']
            if canonical_hash(raw)!=d['dynamic']['payload_hash'] or any(raw[k]!=v for k,v in expected.items()): raise ContractError('Checkpoint dynamics identity mismatch')
            raw['solver_config']=NetworkSolverConfig(**raw['solver_config']);raw['ordered_state_arrays']=tuple(np.array(x) for x in raw['ordered_state_arrays']);dynamic=SimulationState(**raw)
        else: dynamic=restore(json.dumps(d['dynamic']),expected)
        validate_state(dynamic,self.model)
        return RetinaState(dynamic,d['assembly_hash'],d['grid_origin_seconds'],d['sample_index'],d['spike_states'],d['delay_states'],d['input_position'],d['link_delay_states'],d.get('component_tick_indices',{}))

    def stream(self,state,channels,stop,chunk_seconds,recorder,solver_config=None,recording_plan=None):
        if chunk_seconds<=0: raise ContractError('Positive stream window required')
        while state.time_seconds<stop:
            state,result=self.advance(state,channels,min(stop,state.time_seconds+chunk_seconds),solver_config,recording_plan=recording_plan)
            recorder.consume(result)
        return state

    def pre_adapt(self,state,channels,*,window_seconds,max_duration_seconds,scales,absolute_tolerance,relative_tolerance):
        """Compatibility check for explicitly selected states, not whole-retina adaptation."""
        if window_seconds<=0 or max_duration_seconds<window_seconds or absolute_tolerance<=0 or relative_tolerance<0 or not scales:raise ContractError('Explicit bounded warm-up policy/scales required')
        names={s.state_name for s in self.model.state_specs}
        if set(scales)-names:raise ContractError('Unknown warm-up state: '+next(iter(set(scales)-names)))
        if any(np.any(q.values<=0) for q in scales.values()):raise ContractError('Positive explicit warm-up numerical scale required')
        from .background import prepare_background,BackgroundPolicy
        policy=BackgroundPolicy(profile_id='legacy_selected_state_numerical_check',window_seconds=window_seconds,
            max_sim_seconds=max_duration_seconds,stable_windows=1,minimum_seconds=0.,known_tau_multiplier=0.,
            delay_multiplier=0.,monitor_step_seconds=min(window_seconds,.0001),relative_tolerance=relative_tolerance)
        state,report=prepare_background(self,state,channels,policy,explicit_scales=scales,absolute_fraction=absolute_tolerance)
        rows=report.pop('windows');report['window_diagnostics']=rows;report['windows']=len(rows)
        report['scaled_change_norm']=max((r['maximum_norm'] for r in rows[-1:]),default=None)
        report['claim']='Numerical check of selected states only; no whole-network adaptation or stochastic stationarity claim'
        return state,report
