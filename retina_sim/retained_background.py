"""Deterministic background closure for the retained physical-terminal circuit.

Only states are solved. All coefficients, references, contacts, clocks and
random histories are retained. This is not a replacement for the existing
background preparation or a claim of stochastic stationarity.
"""
from copy import deepcopy
from dataclasses import replace
from time import perf_counter
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse.linalg import spsolve
from scipy.special import expit

from .contracts import ContractError, canonical_hash
from .network import InputBundle
from .bc_ac_hypothesis import (
    _outer_background, _retained_history_projector,
    _effective_inhibitory_ac_projector, balance_rgc_initial_voltages,
)


def close_retained_background(runner, state, channels):
    """Return a fresh exact deterministic equilibrium at actual typed inputs.

The existing feed-forward rod resource/chemical branch is eliminated
algebraically. The reciprocal AII/ON-BC voltages use one sparse linear solve;
OFF-BC voltage-dependent glycine uses a small diagonal nonlinear solve.
No neural time is advanced and no parameter is estimated from responses.
"""
    started=perf_counter()
    c = runner.compiled
    model = runner.model
    if state.time_seconds != state.grid_origin_seconds or state.sample_index != 0:
        raise ContractError('Retained background closure requires an unadvanced state')
    if state.assembly_hash != c.assembly_hash or state.link_delay_states:
        raise ContractError('Retained closure requires its matching circuit without transmission delays')
    required = {'rod', 'rod_membrane', 'rod_BC', 'AII_AC', 'individual_rod_RBC_contacts',
                'RBC_AII_AMPA'}
    if not required <= set(c.components):
        raise ContractError('Retained complete rod/BC circuit required')
    for post, synapse in (('midget_BC_OFF', 'AII_FMB_glycine'),
                          ('diffuse_BC_OFF', 'AII_DB1_glycine')):
        if post in c.components and synapse not in c.components:
            raise ContractError('Retained background omitted an actual OFF chemical dependency: '+synapse)
    if any(x.input_contract.get('declared_constant_background') is not True or x.breakpoints_seconds
           for x in channels.values()):
        raise ContractError('Actual explicitly constant typed background required')
    inputs = {name: ch.at(state.time_seconds) for name, ch in channels.items()}
    for name, ch in channels.items():
        for t in ((state.time_seconds+ch.stop_seconds)*.5,np.nextafter(ch.stop_seconds,ch.start_seconds)):
            if not np.array_equal(inputs[name].values,ch.at(t).values):
                raise ContractError('Background channel changes within declared interval')
    bundle = InputBundle(inputs)
    y = np.concatenate(state.ordered_state_arrays).copy()
    params = {cid: {r.parameter_id:r.value.values for r in x.parameters.records}
              for cid,x in c.components.items() if hasattr(x.parameters, 'records')
              and all(hasattr(r.value, 'values') for r in x.parameters.records)}
    index = {cid:{s.state_name:sl for s,sl in zip(x.state_specs,model._local_slices[cid])}
             for cid,x in c.components.items()}
    get = lambda cid, key: y[index[cid][key]]
    def put(cid, key, value):
        y[index[cid][key]] = value
    def evaluate():
        return model.evaluate(state.time_seconds, y, bundle)
    fixed = _outer_background(model, y, inputs)
    histories, _, history_project = _retained_history_projector(model,
        {cid for cid,x in c.components.items() if x.mechanism in ('HistoryBipolarTerminal', 'ResidualAC')})
    effective_acs, _, ac_project = _effective_inhibitory_ac_projector(model, set(c.components))
    # This analytic closure retains the existing BC/AII electrical feedback.
    # Generic AC feedback into that circuit requires the explicit coupled root.
    edges=[(x.source_component,x.target_component) for x in c.spec.links]
    edges += [(a,b) for g in c.spec.gap_junctions
              for a,b in ((g.left_component,g.right_component),(g.right_component,g.left_component))]
    reached=set(effective_acs)
    while True:
        descendants=reached|{b for a,b in edges if a in reached}
        if descendants==reached:break
        reached=descendants
    coupled_targets={'rod_BC','AII_AC'}|{cid for cid in histories
        if c.components[cid].mechanism=='HistoryBipolarTerminal'}
    if reached & coupled_targets:
        raise ContractError('Generic AC feedback into BC/AII exceeds retained analytic closure; use explicit coupled feedback initialization')
    def project(y,bundle):
        return history_project(ac_project(history_project(y,bundle),bundle),bundle)
    y = project(y,bundle)
    def release(cid):
        p = params[cid]
        key = 'terminal_release' if c.components[cid].mechanism == 'HistoryBipolarTerminal' else 'release'
        put(cid,key,expit((get(cid,'V')-p['release_half'])/p['release_slope']))
        control=getattr(c.components[cid].native_model,'bc_output_gain_control',None)
        if control is not None and control.enabled:
            put(cid,'A',control.scale*get(cid,key))
        if 'x' in index[cid]:
            attempt = p['rmax']*expit((get(cid,'V')-p['r_half'])/p['r_slope'])
            put(cid,'x',1/(1+p['tau_recovery']*p['U']*attempt))
    incoming, _ = evaluate()
    p = params['rod_membrane']
    put('rod_membrane','V',p['EL']-(incoming['rod_membrane']['current'].values-p['current_anchor'])/p['gL'])
    release('rod_membrane')
    incoming, _ = evaluate()
    p = params['individual_rod_RBC_contacts']
    put('individual_rod_RBC_contacts','m',expit(-p['hill']*np.log(
        np.maximum(incoming['individual_rod_RBC_contacts']['release'].values,np.finfo(float).tiny)/p['K'])))
    incoming, _ = evaluate()
    p = params['rod_BC']; ge = incoming['rod_BC']['excitation'].values
    put('rod_BC','V',(p['gL']*p['EL']+ge*p['EE'])/(p['gL']+ge)); release('rod_BC')
    chemicals = [cid for cid,x in c.components.items() if x.mechanism == 'DoubleExponentialChemicalSynapse']
    def chemical_equilibrium(cid):
        if not index[cid]:
            return
        incoming,_ = evaluate(); p = params[cid]; rate = incoming[cid]['event_rate'].values
        tau = p['tau_decay']
        if 'postsynaptic_voltage' in incoming[cid]:
            tau = p['decay_minus100']+(p['decay_0']-p['decay_minus100'])*np.clip(
                (incoming[cid]['postsynaptic_voltage'].values+.1)/.1,0,1)
        put(cid,'a',rate*p['tau_rise']);put(cid,'b',rate*tau)
    chemical_equilibrium('RBC_AII_AMPA')
    on = [cid for cid in histories if c.components[cid].mechanism == 'HistoryBipolarTerminal'
          and 'inhibition' not in c.components[cid].inputs]
    coupled = ['AII_AC']+sorted(on)
    offsets = {}; size=0
    for cid in coupled:
        offsets[cid]=size;size+=get(cid,'V').size
    diag=np.zeros(size);rhs=np.zeros(size);incoming,_=evaluate()
    for cid in coupled:
        p=params[cid];sl=slice(offsets[cid],offsets[cid]+get(cid,'V').size)
        if cid == 'AII_AC':
            ge=incoming[cid]['excitation'].values
            diag[sl]=p['gL']+ge;rhs[sl]=p['gL']*p['EL']+ge*p['EE']
        else:
            target=p['Vref']+p['native_voltage_gain']*(get(cid,'activation')-p['native_reference'])
            diag[sl]=p['gL'];rhs[sl]=p['gL']*target
    ri=[];ci=[];weights=[]
    ids={cid:{g:j for j,g in enumerate(c.components[cid].state_specs[0].cell_ids)} for cid in coupled}
    for gap in c.spec.gap_junctions:
        if gap.left_component not in offsets or gap.right_component not in offsets:
            raise ContractError('Additional gap endpoint requires an explicit closure equation')
        a=np.array([ids[gap.left_component][g]+offsets[gap.left_component] for g in gap.left_cell_ids])
        b=np.array([ids[gap.right_component][g]+offsets[gap.right_component] for g in gap.right_cell_ids])
        g=gap.conductance_evidence.value.values
        np.add.at(diag,a,g);np.add.at(diag,b,g)
        ri.extend(a);ci.extend(b);weights.extend(-g)
        ri.extend(b);ci.extend(a);weights.extend(-g)
    matrix=sparse.diags(diag)+sparse.csr_matrix((weights,(ri,ci)),shape=(size,size))
    voltage=spsolve(matrix,rhs)
    for cid in coupled:
        put(cid,'V',voltage[offsets[cid]:offsets[cid]+get(cid,'V').size]);release(cid)
    off=[cid for cid in histories if c.components[cid].mechanism == 'HistoryBipolarTerminal'
         and 'inhibition' in c.components[cid].inputs]
    off_indices=(np.concatenate([np.arange(index[cid]['V'].start,index[cid]['V'].stop) for cid in off])
                 if off else np.empty(0,dtype=np.int64))
    def off_residual(x):
        y[off_indices]=x*.01
        # Conditional release and A must follow this voltage trial before the
        # actual adapted BC output is used as the continuous AC drive.
        for cid in off:release(cid)
        project(y,bundle)
        for cid in chemicals:
            if cid != 'RBC_AII_AMPA':chemical_equilibrium(cid)
        incoming,_=evaluate();values=[]
        for cid in off:
            p=params[cid];v=get(cid,'V');gi=incoming[cid]['inhibition'].values
            target=p['Vref']+p['native_voltage_gain']*(get(cid,'activation')-p['native_reference'])
            values.extend((p['gL']*(target-v)+gi*(p['EI']-v))/p['C']/.01)
        return np.asarray(values)
    off_nfev=0
    if len(off_indices):
        solution=least_squares(off_residual,y[off_indices]/.01,jac_sparsity=sparse.eye(len(off_indices)),
            bounds=(-15.,10.),ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=50)
        final_off=off_residual(solution.x);off_nfev=solution.nfev
        if not solution.success or np.max(abs(final_off))>1e-7:
            raise ContractError('Retained OFF glycine background failed exact residual check')
    for cid in off:release(cid)
    y=project(y,bundle)
    result=replace(state,dynamic=replace(state.dynamic,ordered_state_arrays=model.unpack(y)))
    result,balance=balance_rgc_initial_voltages(runner,result,channels)
    y=np.concatenate(result.ordered_state_arrays)
    derivative=model.rhs_flat_reference(state.time_seconds,y,bundle,model.parameters,model.graph)
    residuals={s.state_name:float(np.max(abs(derivative[sl])))
               for s,sl in zip(model.state_specs,model._state_slices)}
    if max(residuals.values())>1e-6:
        raise ContractError('Retained background did not close all continuous RHS: '+str(residuals))
    incoming,outputs=evaluate();transition_errors={}
    for cid in histories:
        component=c.components[cid];blocks=model.local_blocks(cid,y)
        transitioned=component.native_model.transition(blocks,incoming[cid],
            {name:outputs[(cid,name)] for name in component.outputs})
        error=max(float(np.max(abs(a-b))) for a,b in zip(transitioned,blocks))
        if error>1e-12:raise ContractError('History transition did not close: '+cid)
        transition_errors[cid]=error
    delays=deepcopy(result.delay_states)
    for cid,data in delays.items():
        q=c.components[cid].observe('Vh',state.time_seconds,model.local_blocks(cid,y),{})
        data['history_values']=[q.values.tolist() for _ in data['history_times']]
        data['prehistory']['values']=q.values.tolist()
    result=replace(result,delay_states=delays)
    return result,dict(status='DETERMINISTIC_BACKGROUND_CLOSED_NOT_STOCHASTIC_STATIONARITY',
        assembly_hash=c.assembly_hash,input_hash=canonical_hash(inputs),outer_components=fixed,
        continuous_RHS_max_abs_per_s=residuals,sampled_transition_max_abs=transition_errors,
        conditional_effective_AC_projection_components=sorted(effective_acs),
        effective_AC_RHS_max_abs_per_s={cid:residuals[cid+'.h'] for cid in effective_acs},
        AII_ON_BC_unknown_voltages=size,OFF_glycine_unknown_voltages=len(off_indices),
        OFF_glycine_solver_nfev=off_nfev,RGC_voltage_balance=balance,
        closure_wall_seconds=perf_counter()-started,
        parameter_fit_performed=False,parameters_or_references_changed=False,neural_seconds=0.,
        original_background_preparation_required=True,stochastic_stationarity_verified=False)
