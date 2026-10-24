"""Explicit, nondefault effective baseline; no physiological fit is performed."""
from copy import copy,deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.special import logit
from .contracts import ContractError, Quantity, canonical_hash
from .provenance import SourceRecord
from .network import JointNetwork, InputBundle
from .spatial import InputChannel
from .contact_first_receptor import aggregate_occupancy
from .bc_ac_receptor_components import BC_TYPES


def register_hypothesis_source(registry,candidate):
    source=candidate.data['hypothesis_source']
    if not candidate.contact_first or candidate.test_only:
        raise ContractError('Hypothesis baseline requires the approved contact-first non-test route')
    record=SourceRecord(source['source_id'],'BC/AC explicit effective hypothesis baseline',
        ('Project hypothesis; not animal data',),2026,source['reference_url'],'data_table',None,
        str(candidate.path),'families: explicit assumptions, derivations, sensitivities',candidate.sha256,None,
        'Project-authored parameter description','identity_verified',
        'Author example provides equation/scale context only. No synthetic test coefficients are physiological sources.',
        '2026-10-19',{'classification':'effective hypothesis, not primary measurement'})
    registry.register_source(record)


def constant_background_channels(compiled,light,*,stop):
    """Typed independent consumer backgrounds; never inferred from a scene frame."""
    channels={}
    for ext in compiled.spec.external_inputs:
        c=compiled.components[ext.target_component];port=c.inputs[ext.target_port]
        n=len(port.cell_ids)
        if port.unit=='td':
            values=np.asarray(light.native_background_values if light.native_background_values is not None
                              else np.full(n,light.native_background_td),float)
        elif port.unit=='R*/s':
            rod=any(set(p.cells.cell_type)=={'rod'} for p in c.populations)
            values=np.full(n,light.rod_background_rstar_s if rod else light.cone_background_rstar_s)
        else:raise ContractError('Background helper requires actual light ports; supply other clamps explicitly')
        q=Quantity(values,port.unit,port.meaning,port.evidence_tag);port.validate(q)
        metadata=dict(declared_constant_background=True,
                source=light.background_source,light_context_hash=light.background_context_hash,
                independent_consumer_backgrounds=True,common_physical_scene_claim=False)
        if c.time_contract.kind=='author_discrete':metadata.update(grid_step_seconds=c.time_contract.fixed_step_seconds,includes_background=True)
        channels[ext.channel]=InputChannel(0.,float(stop),lambda t,q=q:q,port.unit,port.meaning,
            'zero_order_hold',port.evidence_tag,metadata)
    return channels


def _outer_background(model,y,inputs):
    from .vh2005 import NativeConeH1
    from .rod_pathway import ChenPrimateRod,phototransduction_equilibrium
    fixed=[]
    for cid,c in model.compiled.components.items():
        native=c.native_model
        if not isinstance(native,(NativeConeH1,ChenPrimateRod)):continue
        ext=[e for e in model.compiled.spec.external_inputs if e.target_component==cid and e.target_port=='light']
        if len(ext)!=1:raise ContractError('Outer background requires one actual light input')
        q=inputs[ext[0].channel]
        if isinstance(native,NativeConeH1):initial=copy(native).equilibrium(q)
        else:initial=phototransduction_equilibrium(native,q.require('R*/s','photoisomerization_rate'))
        for sl,value in zip(model._local_slices[cid],initial):y[sl]=value.values
        fixed.append(cid)
    return fixed


LEGACY_REFERENCE_RECIPE_SHA256 = 'c07671ccc885dbad9d69f59163ea4407ce55c45a3a5252d6cf9bd9cfda3706ce'


def _legacy_reference_bindings(parent,candidate,targets,background_inputs):
    if background_inputs is None:raise ContractError('Hypothesis binding requires explicit constant background inputs')
    if set(background_inputs)!={e.channel for e in parent.spec.external_inputs}:
        raise ContractError('Background inputs must cover every actual external port')
    for ext in parent.spec.external_inputs:
        q=background_inputs[ext.channel];c=parent.components[ext.target_component]
        c.inputs[ext.target_port].validate(q)
        if q.unit in ('td','R*/s'):
            rod=any(set(p.cells.cell_type)=={'rod'} for p in c.populations)
            key='native_background_td' if q.unit=='td' else 'rod_background_rstar_s' if rod else 'cone_background_rstar_s'
            if not np.all(q.values==candidate.data['background_spec'][key]):
                raise ContractError('Hypothesis reference background differs; new background needs explicit profile revision')
    model=JointNetwork(parent,compute_backend='python')
    y=np.concatenate([v.values for v in model.initial_conditions()])
    _outer_background(model,y,background_inputs)
    incoming,outputs=model.evaluate(0.,y,InputBundle(background_inputs))
    result={}
    for cid in targets:
        old=parent.components[cid];n=len(old.inputs['drive'].cell_ids)
        links=[l for l in parent.links if l.specification.target_component==cid and l.specification.target_port=='drive']
        if links:
            u=np.concatenate([outputs[(l.specification.source_component,l.specification.source_port)].values for l in links])
            W=sparse.hstack([l.matrix for l in links],format='csr')
        else:u=incoming[cid]['drive'].values;W=sparse.eye(n,format='csr')
        positive=u[u>0]
        if len(positive)==0:raise ContractError('Background-derived K needs a positive effective input reference')
        K=float(np.median(positive));rows=candidate.data['families'][cid]['parameters']
        a=lambda k:float(rows[k]['value'])
        occupied=aggregate_occupancy(W,u,K,a('hill'));totals=np.asarray(W.sum(axis=1)).ravel()
        e0=totals-occupied if BC_TYPES.get(cid)=='ON' else occupied
        gI=incoming[cid].get('inhibition');gi=np.zeros(n) if gI is None else gI.values
        gap=incoming[cid].get('__gap_current__');igap=np.zeros(n) if gap is None else gap.values
        numerator=a('gL')*a('EL')+a('gE')*e0*a('EE')
        if cid in BC_TYPES:numerator=numerator+igap
        if gI is not None:numerator=numerator+gi*a('EI')
        V0=numerator/(a('gL')+a('gE')*e0+gi)
        reference=float(np.median(outputs[(cid,'activation')].values))
        if not 0<reference<1:raise ContractError('Bounded output anchor incompatible with retained interface; no clipping')
        half_key='release_half' if cid in BC_TYPES else 'activation_half'
        slope_key='release_slope' if cid in BC_TYPES else 'activation_slope'
        result[cid]=dict(K=K,V0=V0,e0=e0,**{half_key:float(np.median(V0))-a(slope_key)*logit(reference)},
            _context=dict(parent_assembly_hash=parent.assembly_hash,background_hash=canonical_hash(background_inputs),
                upstream_reference='Actual declared outer equilibria plus retained conditional inner factory references; not measured data',
                output_anchor=reference,K_reference='median positive raw presynaptic effective input; not concentration',
                initial_status='background-derived conditional seed; coupled feedback equilibrium still required'))
    return result


def freeze_reference_coefficients(parent,candidate,targets,background_inputs):
    """Materialize the historical recipe once, before changing model coefficients.

    Only the frozen distributed recipe can derive new K/half anchors. Revised
    recipes must supply this explicit reference rather than re-anchor outputs.
    This is a project working-point convention, not a physiological calibration.
    """
    if candidate.sha256 != LEGACY_REFERENCE_RECIPE_SHA256:
        raise ContractError('Revised hypothesis requires frozen model coefficient reference; no automatic output reanchoring')
    values=_legacy_reference_bindings(parent,candidate,targets,background_inputs)
    payload=dict(schema_version='bc-ac-fixed-coefficient-reference-1.0',
        parent_assembly_hash=parent.assembly_hash,recipe_sha256=candidate.sha256,
        reference_background_hash=canonical_hash(background_inputs),
        coefficients={cid:{k:float(v[k]) for k in ('K','release_half' if cid in BC_TYPES else 'activation_half')}
                      for cid,v in values.items()},
        claim='Frozen effective reference coefficients; not animal measurements')
    return dict(payload,content_hash=canonical_hash(payload))


def resolve_background_bindings(parent,candidate,targets,background_inputs,coefficient_reference=None):
    """Keep K/half fixed; derive states using the actually resolved coefficients."""
    reference=(freeze_reference_coefficients(parent,candidate,targets,background_inputs)
               if coefficient_reference is None else deepcopy(coefficient_reference))
    payload={k:v for k,v in reference.items() if k!='content_hash'}
    if (reference.get('schema_version')!='bc-ac-fixed-coefficient-reference-1.0' or
            reference.get('content_hash')!=canonical_hash(payload) or
            reference.get('parent_assembly_hash')!=parent.assembly_hash or
            reference.get('recipe_sha256')!=LEGACY_REFERENCE_RECIPE_SHA256 or
            reference.get('reference_background_hash')!=canonical_hash(background_inputs) or
            not set(targets)<=set(reference.get('coefficients',{}))):
        raise ContractError('Frozen coefficient reference identity/background mismatch')
    # Retain the explicit background/domain checks without using revised gE to
    # derive another output nonlinearity. Outer equilibria are initial seeds.
    if background_inputs is None or set(background_inputs)!={e.channel for e in parent.spec.external_inputs}:
        raise ContractError('Hypothesis binding requires explicit constant background inputs')
    for ext in parent.spec.external_inputs:
        c=parent.components[ext.target_component];q=background_inputs[ext.channel];c.inputs[ext.target_port].validate(q)
        if q.unit in ('td','R*/s'):
            rod=any(set(p.cells.cell_type)=={'rod'} for p in c.populations)
            key='native_background_td' if q.unit=='td' else 'rod_background_rstar_s' if rod else 'cone_background_rstar_s'
            if not np.all(q.values==candidate.data['background_spec'][key]):
                raise ContractError('Hypothesis reference background differs; new background needs explicit profile revision')
    model=JointNetwork(parent,compute_backend='python');y=np.concatenate([v.values for v in model.initial_conditions()])
    _outer_background(model,y,background_inputs)
    incoming,outputs=model.evaluate(0.,y,InputBundle(background_inputs));result={}
    for cid in targets:
        rows=candidate.data['families'][cid]['parameters'];fixed=reference['coefficients'][cid]
        half='release_half' if cid in BC_TYPES else 'activation_half'
        if set(fixed)!={'K',half} or not all(np.isfinite(v) for v in fixed.values()) or fixed['K']<=0:
            raise ContractError('Invalid frozen model coefficient reference')
        def a(k):
            row=rows[k]
            if row['binding']=='derived_background':
                if k not in fixed:raise ContractError('Model coefficient missing frozen reference: '+k)
                return fixed[k]
            return float(row['value'])
        n=len(parent.components[cid].inputs['drive'].cell_ids)
        links=[l for l in parent.links if l.specification.target_component==cid and l.specification.target_port=='drive']
        if links:
            u=np.concatenate([outputs[(l.specification.source_component,l.specification.source_port)].values for l in links])
            W=sparse.hstack([l.matrix for l in links],format='csr')
        else:u=incoming[cid]['drive'].values;W=sparse.eye(n,format='csr')
        occupied=aggregate_occupancy(W,u,a('K'),a('hill'));totals=np.asarray(W.sum(axis=1)).ravel()
        e0=totals-occupied if BC_TYPES.get(cid)=='ON' else occupied
        gi=incoming[cid].get('inhibition');gi=np.zeros(n) if gi is None else gi.values
        gap=incoming[cid].get('__gap_current__');gap=np.zeros(n) if gap is None else gap.values
        numerator=a('gL')*a('EL')+a('gE')*e0*a('EE')
        if cid in BC_TYPES:numerator=numerator+gap
        if np.any(gi):numerator=numerator+gi*a('EI')
        V0=numerator/(a('gL')+a('gE')*e0+gi)
        result[cid]=dict(K=a('K'),V0=V0,e0=e0,**{half:a(half)},_context=dict(
            parent_assembly_hash=parent.assembly_hash,background_hash=canonical_hash(background_inputs),
            coefficient_reference=reference,model_coefficients_reanchored=False,
            upstream_reference='Actual outer equilibria and retained conditional references; not measured data',
            initial_status='State seed from resolved coefficients; full feedback equilibrium still required'))
    return result


def balance_rgc_initial_voltages(runner,state,channels):
    """Set only deterministic V/cache at an unadvanced background start.

    All original history, RNG, pending events and clocks stay unchanged. This
    neither draws a first bin nor claims a stationary spike process.
    """
    if state.time_seconds!=state.grid_origin_seconds or state.sample_index!=0:
        raise ContractError('RGC voltage initialization requires an unadvanced state')
    if state.assembly_hash!=runner.compiled.assembly_hash:
        raise ContractError('RGC initialization state/assembly mismatch')
    if not channels or any(ch.input_contract.get('declared_constant_background') is not True or ch.breakpoints_seconds for ch in channels.values()):
        raise ContractError('RGC initialization requires declared constant background inputs')
    y=np.concatenate(state.ordered_state_arrays);bundle=InputBundle({k:ch.at(state.time_seconds) for k,ch in channels.items()})
    incoming,_=runner.model.evaluate(state.time_seconds,y,bundle)
    spikes=deepcopy(state.spike_states)
    signature=lambda aux:canonical_hash({cid:{k:v for k,v in row.items() if k!='conditional_rate_Hz'} for cid,row in aux.items()})
    before=signature(spikes);rows={}
    for cid,c in runner.compiled.components.items():
        if c.mechanism!='HistoryConductanceRGC':continue
        if (not c.native_model.deterministic_state_independent_of_spikes or len(c.state_specs)!=1 or
                any(cid in (g.left_component,g.right_component) for g in runner.compiled.spec.gap_junctions)):
            raise ContractError('RGC passive initialization needs the retained spike-independent single membrane, without electrical contacts')
        if spikes[cid]['last_bin_counts'] is not None or spikes[cid]['pending_events']:
            raise ContractError('Cannot replace V after a spike bin was already drawn')
        native=c.native_model;n=len(c.outputs['voltage'].cell_ids)
        ge=incoming[cid]['excitation'].require('S','excitatory_conductance',(n,))
        gi=incoming[cid]['inhibition'].require('S','inhibitory_conductance',(n,))
        total=native.gL+ge+gi
        V=(native.gL*native.EL+ge*native.EE+gi*native.EI)/total
        sl=runner.model._local_slices[cid][0];old=y[sl].copy();y[sl]=V
        # This is a readout cache of the new V and the same history, not a draw.
        spikes[cid]['conditional_rate_Hz']=native.intensity(V,np.asarray(spikes[cid]['history'])).tolist()
        residual=(native.gL*(native.EL-V)+ge*(native.EE-V)+gi*(native.EI-V))/native.C
        rows[cid]=dict(old_V_range_V=[float(old.min()),float(old.max())],
            balanced_V_range_V=[float(V.min()),float(V.max())],mean_voltage_change_V=float(np.mean(V-old)),
            maximum_membrane_derivative_V_per_s=float(np.max(abs(residual))),
            cached_conditional_rate_recomputed_without_sampling=True)
    if signature(spikes)!=before:raise ContractError('RGC initialization changed history/RNG/events')
    result=replace(state,dynamic=replace(state.dynamic,ordered_state_arrays=runner.model.unpack(y)),spike_states=spikes)
    return result,dict(status='DETERMINISTIC_RGC_VOLTAGE_BALANCED_NOT_SPIKE_STATIONARITY',components=rows,
        history_RNG_events_and_clock_preserved=True,parameters_changed=False,spike_stationarity_verified=False,
        maximum_membrane_derivative_V_per_s=max((r['maximum_membrane_derivative_V_per_s'] for r in rows.values()),default=0.))


def _retained_history_projector(model,selected):
    """Eliminate only constant-input FIR prehistory at its original fixed point."""
    from .inner import CausalHistory
    histories={};specs={s.component_id:s for s in model.compiled.spec.components}
    for cid in selected:
        c=model.compiled.components[cid]
        if c.mechanism=='ResidualAC':h=c.native_model
        elif c.mechanism=='HistoryBipolarTerminal':
            pop=c.populations[0]
            h=CausalHistory(c.parameters,pop,c.evidence_tag,'ResidualBC_'+specs[cid].options['polarity'],
                {'polarity_representation':'canonical_dominant_positive','presynaptic_inhibition':False})
        else:continue
        if [s.state_name for s in c.state_specs[:len(h.state_specs)]]!=[s.state_name for s in h.state_specs]:
            raise ContractError('Retained sampled-state layout differs from original history factory')
        histories[cid]=h
    projected={(sl.start,sl.stop) for cid,h in histories.items() for sl in model._local_slices[cid][:len(h.state_specs)]}
    projected_outputs={(cid,'activation' if model.compiled.components[cid].mechanism=='ResidualAC' else 'generator_activation') for cid in histories}
    input_links={}
    for link in model.compiled.spec.links:input_links.setdefault((link.target_component,link.target_port),[]).append((link.source_component,link.source_port))
    def depends(output,seen):
        if output in projected_outputs:return True
        if output in seen:raise ContractError('Retained sampled algebraic cycle needs an explicit closure')
        cid,port=output;c=model.compiled.components[cid]
        return any(depends(src,seen|{output}) for name in c.output_dependencies[port] for src in input_links.get((cid,name),()))
    for cid in histories:
        if any(depends(src,set()) for src in input_links.get((cid,'drive'),())):
            raise ContractError('Sampled drive directly depends on another projected history; additional closure required')
    def project(y,bundle):
        incoming,_=model.evaluate(0.,y,bundle)
        for cid,h in histories.items():
            delta=incoming[cid]['drive'].values-h.baseline
            prehistory=np.broadcast_to(delta,(len(h.kernel),len(delta)))
            values=tuple(prehistory)+(h.activation(prehistory,incoming[cid]),)
            for sl,v in zip(model._local_slices[cid],values):y[sl]=v
        return y
    return histories,projected,project


def _effective_inhibitory_ac_projector(model,selected):
    """Eliminate continuous AC h at its actual conditional input target.

    BC release/adaptation and membrane states remain part of the surrounding
    root problem. No sampled history is constructed for the continuous AC.
    """
    components={cid:model.compiled.components[cid] for cid in sorted(selected)
                if model.compiled.components[cid].mechanism=='EffectiveInhibitoryAC'}
    projected=set()
    for cid,c in components.items():
        if (c.time_contract.kind!='continuous' or len(c.state_specs)!=1 or
                c.state_specs[0].state_name!='h' or c.state_specs[0].unit!='dimensionless' or
                c.state_specs[0].meaning!='effective_inhibitory_activity'):
            raise ContractError('Effective AC background requires its single continuous inhibitory state')
        sl=model._local_slices[cid][0];projected.add((sl.start,sl.stop))
    projected_outputs={(cid,name) for cid in components for name in ('activation','inhibition_state')}
    input_links={}
    for link in model.compiled.spec.links:
        input_links.setdefault((link.target_component,link.target_port),[]).append((link.source_component,link.source_port))
    def depends(output,seen):
        if output in projected_outputs:return True
        if output in seen:raise ContractError('Effective AC conditional input has an unresolved algebraic cycle')
        cid,port=output;c=model.compiled.components[cid]
        return any(depends(src,seen|{output}) for name in c.output_dependencies[port]
                   for src in input_links.get((cid,name),()))
    for cid in components:
        if any(depends(src,set()) for src in input_links.get((cid,'drive'),())):
            raise ContractError('Effective AC drive directly depends on projected inhibition; explicit algebraic closure required')
    def project(y,bundle):
        if not components:return y
        incoming,_=model.evaluate(0.,y,bundle)
        for cid,c in components.items():
            target=np.asarray(c.native_model.target(incoming[cid]['drive'].values),dtype=float)
            if (target.shape!=c.state_specs[0].shape or np.any(~np.isfinite(target)) or
                    np.any((target<0)|(target>1))):
                raise ContractError('Effective AC conditional background target outside its finite unit interval')
            y[model._local_slices[cid][0]]=target
        return y
    return components,projected,project


def prepare_feedback_initial_state(compiled,channels,*,max_unknown_scalars=1024,max_nfev=35,max_wall_seconds=60.,retained_reference=False):
    """Solve deterministic upstream feedback; retain all links/gaps and equations.

    This is an initial-state solve, not a fit. RGC spike/history stationarity is
    not claimed. Oversized systems fail explicitly instead of dropping feedback.
    """
    from scipy.optimize import least_squares
    from time import perf_counter
    from .hybrid import HybridRunner
    from .contracts import SolverConfig
    from .bc_ac_contact_components import BC_KEY,AC_KEY
    runner=HybridRunner(compiled,compute_backend='python');model=runner.model
    solver=SolverConfig(method='DOP853',rtol=1e-7,atol=1e-10,max_step=.0001,input_interpolation='zero_order_hold')
    state=runner.initialize(solver)
    if not channels or any(c.input_contract.get('declared_constant_background') is not True or c.breakpoints_seconds for c in channels.values()):
        raise ContractError('Feedback initialization requires explicitly constant background channels')
    inputs={name:c.at(0.) for name,c in channels.items()}
    for c in channels.values():
        if not np.array_equal(c.at(0.).values,c.at(np.nextafter(c.stop_seconds,0.)).values):
            raise ContractError('Background changed within its declared interval')
    y=np.concatenate([v for v in state.ordered_state_arrays]);fixed=_outer_background(model,y,inputs)
    keys=(('HistoryBipolarTerminal','ResidualAC') if retained_reference else (BC_KEY,AC_KEY))+('EffectiveInhibitoryAC',)
    selected={cid for cid,c in compiled.components.items() if c.mechanism in keys}
    edges=[(l.source_component,l.target_component) for l in compiled.spec.links]
    edges += [(a,b) for g in compiled.spec.gap_junctions for a,b in ((g.left_component,g.right_component),(g.right_component,g.left_component))]
    while True:
        ancestors=selected|{a for a,b in edges if b in selected}
        if ancestors==selected:break
        selected=ancestors
    bundle=InputBundle(inputs);histories={};projected=set();history_project=lambda y,bundle:y
    if retained_reference:
        histories,projected,history_project=_retained_history_projector(model,selected)
    effective_acs,ac_projected,ac_project=_effective_inhibitory_ac_projector(model,selected)
    projected|=ac_projected
    def project(y,bundle):
        # The second history projection also handles an AC state feeding a BC
        # sampled drive; direct algebraic h-to-h dependencies are rejected above.
        return history_project(ac_project(history_project(y,bundle),bundle),bundle)
    y=project(y,bundle)
    solved=[cid for cid in selected if cid not in fixed and compiled.components[cid].state_specs and
            compiled.components[cid].time_contract.kind in ('continuous','causal_history') and
            not hasattr(compiled.components[cid].native_model,'sample_spikes') and
            any((sl.start,sl.stop) not in projected for sl in model._local_slices[cid])]
    if not retained_reference and any(compiled.components[cid].time_contract.kind=='causal_history' for cid in selected):
        raise ContractError('Unresolved sampled feedback boundary; no history driver silently frozen')
    indices=[];scales=[];lower=[];upper=[]
    for cid in sorted(solved):
        for sl,spec in zip(model._local_slices[cid],compiled.components[cid].state_specs):
            if (sl.start,sl.stop) in projected:continue
            indices.extend(range(sl.start,sl.stop));scale=.01 if spec.unit=='V' else 1.
            scales.extend([scale]*spec.shape[0])
            lower.extend(([-.15] if spec.unit=='V' else [0.])*spec.shape[0])
            hi=np.full(spec.shape[0],.10 if spec.unit=='V' else np.inf)
            if spec.meaning in ('effective_release_activation','adaptation_gate'):hi[:]=1.
            if spec.meaning=='release_adaptation_activity':
                hi[:]=compiled.components[cid].native_model.bc_output_gain_control.scale
            if spec.meaning=='excitatory_synaptic_gate' and compiled.components[cid].mechanism in (BC_KEY,AC_KEY):
                cs=next(s for s in compiled.spec.components if s.component_id==cid)
                hi=np.asarray(cs.options['candidate_dynamics']['row_sum_W'])
            upper.extend(hi/scale)
            lower[-spec.shape[0]:]=[v/scale for v in lower[-spec.shape[0]:]]
    indices=np.asarray(indices,int);scales=np.asarray(scales,float)
    if not len(indices) or len(indices)>max_unknown_scalars:
        raise ContractError('Feedback initial-state solve exceeds explicit small-network budget; no contacts removed')
    started=perf_counter();calls=0
    def residual(x):
        nonlocal calls
        calls+=1
        if perf_counter()-started>max_wall_seconds:
            raise ContractError('Explicit feedback-initialization wall budget reached; no partial state accepted')
        trial=y.copy();trial[indices]=x*scales
        trial=project(trial,bundle)
        return model.rhs_flat_reference(0.,trial,bundle,model.parameters,model.graph)[indices]/scales
    # Component dependency sparsity preserves all feedback and reduces finite
    # differences; it does not approximate the actual residual.
    dependency=sparse.lil_matrix((len(indices),len(indices)),dtype=int)
    local={cid:np.flatnonzero(np.isin(indices,np.arange(model.slices[cid].start,model.slices[cid].stop))) for cid in solved}
    for target in solved:
        ancestors={target};front={target}
        while front:
            nxt={a for a,b in edges if b in front}-ancestors
            ancestors|=nxt;front=nxt
        for source in ancestors & set(solved):dependency[np.ix_(local[target],local[source])]=1
    initial=residual(y[indices]/scales)
    answer=least_squares(residual,y[indices]/scales,jac_sparsity=dependency.tocsr(),
        bounds=(np.asarray(lower),np.asarray(upper)),max_nfev=max_nfev,ftol=1e-11,xtol=1e-11,gtol=1e-11)
    final=residual(answer.x);norm=float(np.max(np.abs(final)))
    if not answer.success or norm>1e-5:
        raise ContractError(f'Feedback initial-state solve not converged: scaled residual {norm:.6g}/s, nfev={answer.nfev}')
    y[indices]=answer.x*scales
    y=project(y,bundle)
    full_rhs=model.rhs_flat_reference(0.,y,bundle,model.parameters,model.graph)
    ac_errors={cid:float(np.max(abs(full_rhs[model._local_slices[cid][0]]))) for cid in effective_acs}
    if max(ac_errors.values(),default=0.)>1e-5:
        raise ContractError('Effective AC conditional background did not close its original continuous RHS')
    sampled_errors={}
    if histories:
        incoming,out=model.evaluate(0.,y,bundle,sampling=True,sampled_components=set(histories))
        for cid in histories:
            c=compiled.components[cid];blocks=model.local_blocks(cid,y)
            transitioned=c.native_model.transition(blocks,incoming[cid],{name:out[(cid,name)] for name in c.outputs})
            error=max(float(np.max(abs(a-b))) for a,b in zip(transitioned,blocks))
            if error>1e-10:raise ContractError('Retained sampled background is not an actual transition fixed point: '+cid)
            sampled_errors[cid]=error
    arrays=model.unpack(y)
    state=replace(state,dynamic=replace(state.dynamic,ordered_state_arrays=arrays))
    state,rgc_balance=balance_rgc_initial_voltages(runner,state,channels)
    report=dict(status='DETERMINISTIC_FEEDBACK_INITIAL_STATE_SOLVED_NOT_WHOLE_NETWORK_ADAPTED',
        background_hash=canonical_hash(inputs),assembly_hash=compiled.assembly_hash,unknown_scalars=len(indices),
        solved_components=sorted(solved),outer_equilibrium_components=fixed,nfev=answer.nfev,
        initial_scaled_residual_per_s=float(np.max(np.abs(initial))),final_scaled_residual_per_s=norm,
        rhs_evaluations=calls,solve_wall_seconds=perf_counter()-started,
        retained_reference=retained_reference,constant_FIR_history_projection_components=sorted(histories),
        conditional_effective_AC_projection_components=sorted(effective_acs),effective_AC_RHS_max_abs_per_s=ac_errors,
        original_sample_transition_errors=sampled_errors,RGC_voltage_balance=rgc_balance,
        all_links_retained=True,all_gap_contacts_retained=True,parameter_fit_performed=False,
        RGC_history_and_spike_stationarity_verified=False,physiology_calibrated=False)
    return state,report,runner
