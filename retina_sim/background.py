"""Bounded uniform-input preparation, without altering the model equations.

All tolerances/scales are numerical engineering choices.  A sampled fixed-point
candidate is not physiological validation or a stochastic stationarity proof.
"""
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from .contracts import ContractError, Quantity, canonical_hash
from .recording import RecordingPlan
from .spatial import InputChannel


@dataclass(frozen=True)
class BackgroundPolicy:
    profile_id: str = 'bounded_uniform_background_v1'
    window_seconds: float = .1
    stable_windows: int = 3
    minimum_seconds: float = 1.
    known_tau_multiplier: float = 5.
    delay_multiplier: float = 2.
    max_sim_seconds: float = 10.
    max_wall_seconds: float = 60.
    max_observations: int = 100000
    monitor_step_seconds: float = .0001
    relative_tolerance: float = 1e-4
    stochastic_burnin_seconds: float = 0.
    require_converged: bool = False
    def __post_init__(self):
        positive=('window_seconds','max_sim_seconds','max_wall_seconds','max_observations','monitor_step_seconds')
        nonnegative=('minimum_seconds','known_tau_multiplier','delay_multiplier','relative_tolerance','stochastic_burnin_seconds')
        if any(not np.isfinite(getattr(self,k)) or getattr(self,k)<=0 for k in positive):
            raise ContractError('Positive finite background budgets/window/monitor step required')
        if any(not np.isfinite(getattr(self,k)) or getattr(self,k)<0 for k in nonnegative):
            raise ContractError('Nonnegative finite background criteria required')
        if isinstance(self.stable_windows,bool) or int(self.stable_windows)!=self.stable_windows or self.stable_windows<1:
            raise ContractError('Positive integer stable-window count required')
        if int(self.max_observations)!=self.max_observations:raise ContractError('Observation budget must be an integer')
        if self.monitor_step_seconds>self.window_seconds:raise ContractError('Monitor step exceeds background window')


# Unit-specific numerical floors, not measured biological noise/variation.  The
# scale of each cell is frozen at max(abs(initial state), floor), not a moving
# denominator which can hide growth. Unknown units must be explicitly registered.
NUMERICAL_UNIT_SCALES = {
    'V': {'scale_floor': .01, 'atol': 1e-5},
    'mV': {'scale_floor': 10., 'atol': .01},
    'dimensionless': {'scale_floor': 1., 'atol': 1e-4},
    'td': {'scale_floor': 1., 'atol': 1e-4},
    'vH_native_precursor': {'scale_floor': 1., 'atol': 1e-4},
    'vH_native_current': {'scale_floor': 1., 'atol': 1e-4},
    'vH_native_voltage': {'scale_floor': 1., 'atol': 1e-4},
    'vH_native_conductance': {'scale_floor': 1., 'atol': 1e-4},
    'pA': {'scale_floor': 1., 'atol': 1e-4},
    'A': {'scale_floor': 1e-12, 'atol': 1e-16},
    'S': {'scale_floor': 1e-9, 'atol': 1e-13},
    'uM': {'scale_floor': 1., 'atol': 1e-4},
    'uM/s': {'scale_floor': 1., 'atol': 1e-4},
    'Chen_scaled_concentration': {'scale_floor': 1., 'atol': 1e-4},
    'Chen_scaled_concentration/s': {'scale_floor': 1., 'atol': 1e-4},
    's^-1': {'scale_floor': 1., 'atol': 1e-4},
    's^-2': {'scale_floor': 1., 'atol': 1e-4},
    'R*/s': {'scale_floor': 1., 'atol': 1e-4},
}


def constant_background_channels(compiled, light_context, stop_seconds, *, start_seconds=0., interpolation='linear'):
    """Match actual external ports, preserving source units and author ZOH ticks."""
    rates={'native_retinal_illuminance':light_context.native_background_td,
           'photoisomerization_rate':None}
    result={}
    for external in compiled.spec.external_inputs:
        c=compiled.components[external.target_component]; port=c.inputs[external.target_port]
        if port.meaning=='native_retinal_illuminance' and port.unit=='td':value=rates[port.meaning]
        elif port.meaning=='photoisomerization_rate' and port.unit=='R*/s':
            population_types={p.cells.cell_type[0] for p in c.populations}
            if population_types=={'rod'}:value=light_context.rod_background_rstar_s
            elif population_types=={'cone'}:value=light_context.cone_background_rstar_s
            else:raise ContractError('Background photoreceptor identity is ambiguous: '+external.channel)
        else:raise ContractError('No explicit background value for external input: '+external.channel)
        values=np.full(len(port.cell_ids),value)
        if port.meaning=='native_retinal_illuminance' and light_context.native_background_values is not None:
            values=np.asarray(light_context.native_background_values,dtype=float)
            if values.shape!=(len(port.cell_ids),):raise ContractError('Reference background cone IDs/count differ from consumer')
        q=Quantity(values,port.unit,port.meaning,port.evidence_tag)
        discrete=c.time_contract.kind=='author_discrete'
        result[external.channel]=InputChannel(float(start_seconds),float(stop_seconds),lambda t,q=q:q,
            port.unit,port.meaning,'zero_order_hold' if discrete else interpolation,port.evidence_tag,
            {'declared_constant_background':True,'includes_background':True,
             'background_context_hash':light_context.background_context_hash,
             'background_source':light_context.background_source,
             'grid_step_seconds':c.time_contract.fixed_step_seconds if discrete else .0001,
             'value':value,'unit':port.unit,'input_class':'uniform_declared_background'},())
    return result


def timing_requirements(runner, policy):
    """Only explicit time constants are used; latency/TTP is never reinterpreted."""
    taus=[];unknown=[];delays=[];memory=[]
    for cid,c in runner.compiled.components.items():
        known=[]
        control=getattr(c.native_model,'bc_output_gain_control',None)
        if control is not None and control.enabled:
            known.append(control.tau_a_seconds)
            taus.append({'component':cid,'parameter':'BC_release_adaptation_tau_a','seconds':control.tau_a_seconds})
        if c.mechanism=='EffectiveInhibitoryAC':
            seconds=float(c.native_model.tau_A_seconds)
            if not np.isfinite(seconds) or seconds<=0:
                raise ContractError('Effective AC requires a positive explicit continuous time constant')
            known.append(seconds)
            taus.append({'component':cid,'parameter':'effective_AC_tau_A_seconds','seconds':seconds})
        for record in c.parameters.records:
            value=record.value
            unit=getattr(value,'unit',getattr(record,'unit',None))
            meaning=getattr(value,'meaning',getattr(record,'meaning',''))
            name=record.parameter_id
            # vH's author registry has scalar ParameterRecord values/units.
            is_tau=('time_constant' in meaning or
                    (c.mechanism in ('VanHateren2005NativeConeH1','VanHateren2005FiniteContactZeroFeedbackApproximation') and name.removeprefix('VH2005_').startswith('tau_')))
            if is_tau and unit in ('s','ms'):
                raw=np.asarray(getattr(value,'values',value),dtype=float)
                seconds=float(np.max(raw))*(.001 if unit=='ms' else 1.)
                if np.isfinite(seconds) and seconds>0:known.append(seconds);taus.append({'component':cid,'parameter':name,'seconds':seconds})
        if not known:unknown.append({'component':cid,'scope':'no explicit time-constant registration; slower coupled modes not bounded by this list'})
        native=c.native_model
        if hasattr(native,'observation_interval_seconds') and hasattr(native,'p'):
            delay=native.p.get('observation_delay',0.)*.001
            delays.append(float(delay))
        if hasattr(native,'sample_spikes'):
            horizon=(len(native.initial_spike_history)+1)*float(native.dt)
            memory.append({'component':cid,'history_washout_seconds':horizon,'rng_seed':int(native.rng_seed),
                           'deterministic_state_independent_of_spikes':bool(getattr(native,'deterministic_state_independent_of_spikes',False))})
    for link in runner.compiled.links:
        if link.delayed_operator is not None:delays.extend(np.asarray(link.delayed_operator.delays_seconds).tolist())
    tau=max((x['seconds'] for x in taus),default=0.);delay=max(delays,default=0.)
    burn=max([policy.stochastic_burnin_seconds]+[x['history_washout_seconds'] for x in memory])
    return {'known_time_constants':taus,'unknown_time_constant_scope':unknown,
            'maximum_known_tau_seconds':tau,'maximum_delay_seconds':delay,
            'minimum_required_seconds':max(policy.minimum_seconds,policy.known_tau_multiplier*tau,policy.delay_multiplier*delay,burn),
            'stochastic_history':memory,'minimum_stochastic_burnin_seconds':burn,
            'time_constant_list_is_not_a_bound_on_all_coupled_relaxation_modes':True}


def rebase_stimulus_channels(channels,background,*,original_origin_seconds,stimulus_origin_seconds,stop_seconds):
    """Keep absolute integration/event time, map only the external stimulus phase."""
    result={}
    for name,channel in channels.items():
        def evaluator(t,channel=channel,name=name):
            if t<stimulus_origin_seconds-1e-12:return background[name].at(t)
            return channel.at(t-stimulus_origin_seconds+original_origin_seconds)
        shifted=tuple(t-original_origin_seconds+stimulus_origin_seconds for t in channel.breakpoints_seconds
            if original_origin_seconds<=t and t-original_origin_seconds+stimulus_origin_seconds<=stop_seconds)
        result[name]=InputChannel(0.,stop_seconds,evaluator,channel.unit,channel.meaning,channel.interpolation,
            channel.evidence_tag,dict(channel.input_contract,stimulus_origin_seconds=stimulus_origin_seconds,
                original_stimulus_origin_seconds=original_origin_seconds,absolute_time_preserved=True),shifted)
    return result


class _WindowMonitor:
    def __init__(self,runner,state,policy,unit_scales=None,explicit_scales=None,absolute_fraction=None):
        self.runner=runner;self.policy=policy;self.rows=[];self.missing=[];self.count=0;self.domain_failures=[]
        registry=NUMERICAL_UNIT_SCALES if unit_scales is None else unit_scales
        offset=0
        for spec,initial in zip(runner.model.state_specs,state.ordered_state_arrays):
            sl=slice(offset,offset+spec.shape[0]);offset+=spec.shape[0]
            if explicit_scales is not None:
                if spec.state_name not in explicit_scales:continue
                scale=explicit_scales[spec.state_name];scale.require(spec.unit,spec.meaning,spec.shape)
                values=scale.values;atol=absolute_fraction*values
            elif spec.unit in registry:
                entry=registry[spec.unit];values=np.maximum(np.abs(initial),entry['scale_floor']);atol=entry['atol']
            else:self.missing.append({'state':spec.state_name,'unit':spec.unit});continue
            den=np.asarray(atol+policy.relative_tolerance*values)
            if np.any(~np.isfinite(den)) or np.any(den<=0):raise ContractError('Positive numerical state scales required')
            self.rows.append({'spec':spec,'slice':sl,'den':den})
        self.reset()
    def reset(self):
        for row in self.rows:row.update(lo=None,hi=None,rhs=np.zeros(row['spec'].shape),defect=np.zeros(row['spec'].shape))
        self.count=0;self.domain_failures=[];self.coverage_limited=False
    def observe(self,t,y,bundle):
        self.count+=1
        if np.any(~np.isfinite(y)):raise ContractError('Nonfinite background state; no clipping')
        model=self.runner.model
        rhs=model.rhs_flat(t,y,bundle,model.parameters,model.graph)
        if np.any(~np.isfinite(rhs)):raise ContractError('Nonfinite original background RHS')
        # Pure native one-step transitions detect a held/history state with zero RHS.
        context_reuse=getattr(self.runner,'background_monitor_context_reuse',None)
        inputs,outputs=(model.evaluate(t,y,bundle,sampling=True) if context_reuse is None
                        else context_reuse.evaluate(t,y,bundle,sampling=True))
        gap=[q.values for (cid,port),q in outputs.items() if port=='gap_current']
        if gap:
            total=sum(float(np.sum(a)) for a in gap);magnitude=sum(float(np.sum(np.abs(a))) for a in gap)
            if abs(total)>1e-16+1e-10*magnitude:self.domain_failures.append('reciprocal_gap_current_conservation')
        transitions={}
        for cid,c in self.runner.compiled.components.items():
            if c.time_contract.kind in ('causal_history','author_discrete'):
                local=model.local_blocks(cid,y)
                if c.time_contract.kind=='causal_history':new=c.native_model.transition(local,inputs[cid],{p:outputs[(cid,p)] for p in c.outputs})
                else:new=c.native_model.transition(local,inputs[cid])
                transitions[cid]=np.concatenate(new)
        for row in self.rows:
            spec=row['spec'];a=y[row['slice']]
            row['lo']=a.copy() if row['lo'] is None else np.minimum(row['lo'],a)
            row['hi']=a.copy() if row['hi'] is None else np.maximum(row['hi'],a)
            row['rhs']=np.maximum(row['rhs'],np.abs(rhs[row['slice']]))
            # Only explicitly named unit-interval states are given a gate domain.
            if spec.unit=='dimensionless' and spec.meaning in ('excitatory_receptor_gate','inhibitory_receptor_gate','excitatory_synaptic_gate','release_probability','sodium_activation_gate','potassium_activation_gate','synaptic_resource_fraction','effective_release_activation','effective_inhibitory_activity','adaptation_gate'):
                if np.any((a < -1e-10)|(a > 1+1e-10)):self.domain_failures.append(spec.state_name)
            if spec.meaning in ('native_illuminance','native_precursor','native_os_current','native_conductance',
                'effective_opsin_activity','effective_pde_activity','cgmp_concentration','outer_segment_calcium',
                'cgmp_synthesis_cache','source_scaled_cgmp','source_scaled_calcium','source_scaled_cgmp_synthesis','receptor_gate','release_adaptation_activity'):
                if np.any(a < -1e-10):self.domain_failures.append(spec.state_name+':negative_domain')
            if spec.meaning=='release_adaptation_activity':
                control=self.runner.compiled.components[spec.state_name.split('.',1)[0]].native_model.bc_output_gain_control
                if np.any(a>control.scale+1e-10):self.domain_failures.append(spec.state_name+':above_scale')
            cid=spec.state_name.split('.',1)[0]
            if cid in transitions:
                c=self.runner.compiled.components[cid];local_slice=slice(row['slice'].start-model.slices[cid].start,row['slice'].stop-model.slices[cid].start)
                period=self.runner.periods[cid]
                row['defect']=np.maximum(row['defect'],np.abs(transitions[cid][local_slice]-a)/period)
    def report(self,width):
        result=[];dominant=None;worst=0.
        for row in self.rows:
            spec=row['spec'];den=row['den']
            norms=[(row['hi']-row['lo'])/den,width*row['rhs']/den,width*row['defect']/den]
            labels=('excursion','original_rhs','discrete_one_step_defect')
            maxima=[float(np.max(a)) if len(a) else 0. for a in norms]
            result.append({'state':spec.state_name,'unit':spec.unit,'maximum_observed_excursion_norm':maxima[0],
                           'original_rhs_norm':maxima[1],'discrete_defect_norm':maxima[2]})
            for label,array,value in zip(labels,norms,maxima):
                if value>worst:
                    index=int(np.argmax(array));worst=value
                    dominant={'state':spec.state_name,'component':spec.state_name.split('.',1)[0],
                              'cell_id':spec.cell_ids[index],'criterion':label,'norm':value}
        return {'states':result,'maximum_norm':worst,'dominant':dominant,'observations':self.count,
                'domain_failures':sorted(set(self.domain_failures)),'passed':worst<=1 and not self.domain_failures}


def _history_consistency(state,runner,policy,unit_scales=None):
    failures=[]
    registry=NUMERICAL_UNIT_SCALES if unit_scales is None else unit_scales
    for cid,aux in state.spike_states.items():
        h=np.asarray(aux['history']);counts=aux.get('last_bin_counts')
        if np.any(~np.isfinite(h)) or np.any(h<0) or np.any(h!=np.floor(h)):failures.append(cid+':invalid_spike_history')
        if counts is not None and (np.any(np.asarray(counts)<0) or np.any(~np.isfinite(counts))):failures.append(cid+':invalid_bin_counts')
        if any(e['time_seconds']<state.time_seconds-1e-12 for e in aux['pending_events']):failures.append(cid+':past_pending_event')
        if not aux.get('rng_state'):failures.append(cid+':missing_rng')
    for kind,queues in [('observation',state.delay_states),('link',state.link_delay_states)]:
        for key,payload in queues.items():
            times=payload.get('history_times',payload.get('times',[]))
            if len(times) and (np.any(np.diff(times)<0) or times[-1]>state.time_seconds+1e-12):failures.append(kind+':'+key+':invalid_timestamp_order')
            values=np.asarray(payload.get('history_values',payload.get('values',[])))
            if np.any(~np.isfinite(values)):failures.append(kind+':'+key+':nonfinite_values')
            unit=payload['prehistory']['unit']
            if unit not in registry:failures.append(kind+':'+key+':unregistered_queue_scale')
            elif len(values)>1:
                entry=registry[unit];scale=np.maximum(np.abs(payload['prehistory']['values']),entry['scale_floor'])
                if np.any(np.ptp(values,axis=0)>entry['atol']+policy.relative_tolerance*scale):
                    failures.append(kind+':'+key+':queue_not_constant_within_tolerance')
    for cid,period in runner.periods.items():
        if state.grid_origin_seconds+state.component_tick_indices[cid]*period < state.time_seconds-1e-12:
            failures.append(cid+':lost_tick_phase')
    return failures


def prepare_background(runner,state,channels,policy=None,*,unit_scales=None,explicit_scales=None,
                       absolute_fraction=None,on_chunk=None,on_window=None,reference_state=None,
                       preparation_origin_seconds=None):
    """Return the last valid state and a finite budget/convergence report.

    Monitoring is event-aligned plus the declared dense numerical grid; maxima
    are observed numerical maxima, not a proof of sub-grid continuous extrema.
    No state-time reset and no implicit adaptation-result cache are performed.
    """
    policy=policy or BackgroundPolicy()
    author_periods=[c.time_contract.fixed_step_seconds for c in runner.compiled.components.values() if c.time_contract.kind=='author_discrete']
    for period in author_periods:
        for name in ('window_seconds','monitor_step_seconds','max_sim_seconds'):
            ticks=getattr(policy,name)/period
            if abs(ticks-round(ticks))>1e-7:
                raise ContractError('Background '+name+' must preserve the author input/grid phase')
    if any(ch.input_contract.get('declared_constant_background') is not True for ch in channels.values()):
        raise ContractError('Preparation requires explicitly constant background channels')
    if runner.discrete is not None:raise ContractError('Background monitor currently requires a compiled JointNetwork; isolated author kernel uses its own source initialization')
    wall=perf_counter()
    deterministic_initialization=None
    if (getattr(runner.compiled,'retained_background_initializer',None) is not None and
            state.time_seconds==state.grid_origin_seconds and state.sample_index==0):
        from .retained_background import close_retained_background
        # Freeze the existing monitor denominator before replacing any state.
        # The original timing, residual and stochastic criteria still apply.
        if reference_state is None:reference_state=state
        state,deterministic_initialization=close_retained_background(runner,state,channels)
        deterministic_initialization['policy']=runner.compiled.retained_background_initializer
    requirements=timing_requirements(runner,policy)
    monitor=_WindowMonitor(runner,reference_state or state,policy,unit_scales,explicit_scales,absolute_fraction)
    segment_start=state.time_seconds
    start=segment_start if preparation_origin_seconds is None else float(preparation_origin_seconds)
    if start>segment_start:raise ContractError('Preparation origin cannot follow restored state time')
    windows=[];stable=0;observations=0;reason='max_sim_seconds';status='NOT_REACHED'
    end=start+policy.max_sim_seconds
    while state.time_seconds<end-1e-12:
        left=state.time_seconds;right=min(left+policy.window_seconds,end);monitor.reset();window_observations=0
        while state.time_seconds<right-1e-12:
            if perf_counter()-wall>=policy.max_wall_seconds:reason='max_wall_seconds';break
            if observations>=policy.max_observations:reason='max_observations';break
            if on_chunk is not None and on_chunk(state) is False:reason='cancelled';break
            # At most ten monitored grid intervals per allocation/chunk. Each
            # accepted event endpoint is also observed by HybridRunner.
            remaining=policy.max_observations-observations
            stop=min(right,state.time_seconds+policy.monitor_step_seconds*min(10,remaining))
            # Dense samples retain the declared spacing, but share the exact
            # event-clock representation where clocks coincide. Linspace can
            # differ by one ULP and lose a sample deferred to a history tick.
            first=int(np.floor((state.time_seconds-start)/policy.monitor_step_seconds+1e-8))+1
            last=int(np.floor((stop-start)/policy.monitor_step_seconds+1e-8))
            grid=start+np.arange(first,last+1,dtype=np.int64)*policy.monitor_step_seconds
            def aligned(t):
                periods=list(runner.periods.values())+[runner.observation_dt,runner.delayed_step]
                values=[state.grid_origin_seconds+round((t-state.grid_origin_seconds)/p)*p for p in periods if p
                    and abs(state.grid_origin_seconds+round((t-state.grid_origin_seconds)/p)*p-t)<1e-12]
                return min(values) if values else float(t)
            times=np.array(sorted(set([state.time_seconds,stop]+[aligned(t) for t in grid
                if state.time_seconds+1e-12<t<stop-1e-12])))
            def observe_budgeted(t,y,bundle):
                if observations+monitor.count<policy.max_observations:monitor.observe(t,y,bundle)
                else:monitor.coverage_limited=True
            state,_=runner.advance(state,channels,stop,sample_times=times,
                recording_plan=RecordingPlan(ports=(),states=(),cell_ids=()),state_observer=observe_budgeted)
            observations+=monitor.count;window_observations+=monitor.count;monitor.count=0
        width=state.time_seconds-left
        if width<=0:break
        row=monitor.report(width);row.update(observations=window_observations,start_seconds=left,stop_seconds=state.time_seconds,
            complete_window=abs(width-policy.window_seconds)<1e-10,history_consistency_failures=_history_consistency(state,runner,policy,unit_scales))
        row['monitoring_budget_limited']=monitor.coverage_limited
        row['passed']=row['passed'] and row['complete_window'] and not row['history_consistency_failures'] and not monitor.coverage_limited
        stable=stable+1 if row['passed'] else 0;row['consecutive_passed_windows']=stable;windows.append(row)
        if on_window is not None:on_window(state,row)
        if reason in ('max_wall_seconds','max_observations','cancelled'):break
        if stable>=policy.stable_windows and state.time_seconds-start>=requirements['minimum_required_seconds']-1e-12:
            status='CONVERGED';reason='criteria_satisfied';break
    unknown_random=[x['component'] for x in requirements['stochastic_history'] if not x['deterministic_state_independent_of_spikes']]
    unknown_queue_scales=sorted({failure for row in windows for failure in row['history_consistency_failures'] if 'unregistered_queue_scale' in failure})
    coverage_missing=monitor.missing+([{'stochastic_state_dependency_unknown':unknown_random}] if unknown_random else [])+([{'unregistered_queue_scales':unknown_queue_scales}] if unknown_queue_scales else [])
    if coverage_missing:status='INCONCLUSIVE'
    elapsed=state.time_seconds-start
    report={'schema_version':'1.0','policy':asdict(policy),'status':status,'converged':status=='CONVERGED',
            'budget_exit':reason,'elapsed_seconds':elapsed,'absolute_preparation_interval_seconds':[start,state.time_seconds],
            'segment_start_seconds':segment_start,'monitor_reference':'frozen initial reference' if reference_state is not None else 'segment initial state',
            'consecutive_window_streak_reset_on_resume':segment_start>start,
            'grid_origin_seconds':state.grid_origin_seconds,'stimulus_origin_seconds':state.time_seconds,
            'wall_seconds':perf_counter()-wall,'observations':observations,'timing_requirements':requirements,
            'numerical_unit_registry':unit_scales if unit_scales is not None else NUMERICAL_UNIT_SCALES,
            'uncovered_states':coverage_missing,'checked_states':[r['spec'].state_name for r in monitor.rows],
            'windows':windows,'dominant':windows[-1]['dominant'] if windows else None,
            'deterministic_status':status,'stochastic_status':'STATISTICS_INSUFFICIENT' if state.spike_states else 'NOT_APPLICABLE',
            'stochastic_burnin_completed':elapsed>=requirements['minimum_stochastic_burnin_seconds']-1e-12,
            'stochastic_history_is_separate_from_fixed_point_test':True,
            'whole_network_steady_state_claim':False,
            'domain_and_conservation_scope':'explicit registered positive chemical states, unit-interval release/resource states, reciprocal gap-current sum, queue timestamps/finite values and constant-queue residual; no undeclared invariant invented',
            'rng_states_preserved_and_advanced':{cid:canonical_hash(aux['rng_state']) for cid,aux in state.spike_states.items()},
            'stationarity_statistics':{'enabled':False,'required_explicit_policy':['window_seconds_at_least_1','at_least_3_windows','mean_variance_rate_autocorrelation_tolerances','minimum_effective_sample_size'],
                                     'claim':'No universal statistical acceptance tolerance or noise-free spike criterion supplied'},
            'maximum_scope':'all observed accepted event endpoints and declared numerical monitoring grid; sub-grid extrema not certified',
            'adaptation_cache_enabled':False,'physiological_validation':'NOT_RUN',
            'deterministic_initialization':deterministic_initialization,
            'claim':'Sampled deterministic numerical fixed-point candidate only; model calibration and stochastic stationarity not established'}
    state.input_position.update(stimulus_origin_seconds=state.time_seconds,
        background_initialization_status=status,background_preparation_policy_hash=canonical_hash(asdict(policy)),
        background_input_hash=canonical_hash({k:v.input_contract for k,v in channels.items()}))
    return state,report
