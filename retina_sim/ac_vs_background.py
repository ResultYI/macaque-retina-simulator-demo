"""Finite joint constant-background construction, checked against true maps.

Sparse electrical balance and bounded scalar glycine balances replace a blind
warmup. The result is conditional on the declared held exterior reservoirs.
"""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import brentq
from scipy.special import expit

from .ac_vs_model import ACVSInner,MECHANISM,LEGACY_S,equilibrium
from .contracts import ContractError,canonical_hash
from .group_inner_dynamics import BC_MECH,MEMBRANE,AC_MECH,RGC_MECH,key
from .group_time_monitor import GroupTimeMonitor
from .group_time_storage import save_bundle
from .local_patch_boundary import BoundedNativePhysics
from .rgc_expectation import stationary_history_rates,STATES
from .spatial_group_model import BC,RGC


def qualify_background(assembly,original_seed,output,guard):
    from tools.validate_ac_vs import write,QUALIFICATION,identity
    seed=deepcopy(original_seed);state=seed['states'];t=seed['time_seconds']
    model=ACVSInner(assembly,seed['inner_parameters']);native=BoundedNativePhysics(assembly)
    background=seed['input_reference']['native_background_td']
    y,anchor=native.construction_point(SimpleNamespace(states=state,time_seconds=t),background,t+1.,0)
    evaluation=native.evaluate(y,t,anchor);release=evaluation.outputs['release']
    count=0
    def put(pop,mech,name,value):state[pop,mech,name]=np.broadcast_to(value,(model.counts[pop],)).astype(np.float64).copy()
    def membrane_release(pop):
        mech='rod_membrane' if pop=='rods' else MEMBRANE
        V=state[pop,mech,'V']
        put(pop,mech,'release',expit((V-model.par(pop,'release_half'))/model.par(pop,'release_slope',positive=True)))
        if pop!='rods':
            attempts=model.par(pop,'rmax',positive=True)*expit((V-model.par(pop,'r_half'))/model.par(pop,'r_slope',positive=True))
            put(pop,mech,'x',1/(1+model.par(pop,'tau_recovery',positive=True)*model.par(pop,'U',positive=True)*attempts))
    def event_rate(pop):
        V=state[pop,MEMBRANE,'V'];x=state[pop,MEMBRANE,'x']
        return model.par(pop,'Npool',positive=True)*model.par(pop,'U',positive=True)*model.par(pop,'rmax',positive=True)*expit((V-model.par(pop,'r_half'))/model.par(pop,'r_slope',positive=True))*x
    def contacts(component):
        c=model.contacts[component];rate=event_rate(c['route'].pre)[c['pre']]
        rise,decay,norm=model._kinetics(state,component)
        state[c['route'].pre,component,'a']=rate*rise
        state[c['route'].pre,component,'b']=rate*decay
        return np.bincount(c['post'],weights=c['coefficient']*rate*(decay-rise)/norm,minlength=model.counts[c['route'].post])
    rodV=model.par('rods','EL')-(model.rod_current(state)-model.par('rods','current_anchor'))/model.par('rods','gL',positive=True)
    put('rods','rod_membrane','V',rodV);membrane_release('rods')
    c=model.contacts['individual_rod_RBC_contacts'];drive=state['rods','rod_membrane','release'][c['pre']]
    m=expit(-model.cp('individual_rod_RBC_contacts','hill',positive=True)*np.log(drive/model.cp('individual_rod_RBC_contacts','K',positive=True)))
    state['rods','individual_rod_RBC_contacts','m']=m
    ge=np.bincount(c['post'],weights=c['coefficient']*m,minlength=model.counts['rod_BC'])
    put('rod_BC',MEMBRANE,'V',(model.gL['rod_BC']*model.par('rod_BC','EL')+ge*model.par('rod_BC','EE'))/(model.gL['rod_BC']+ge));membrane_release('rod_BC')
    aii_ge=contacts('RBC_AII_AMPA')
    target={}
    for pop in BC:
        route=next(r for r in model.routes if r.pre=='cones' and r.post==pop)
        drive=model.mean[route.name]@release
        h=drive-model.par(pop,'baseline')
        for i in range(3):put(pop,BC_MECH,'history_'+str(i),h)
        sign=-1 if pop.endswith('_ON') or pop=='S_ON_BC' else 1
        activation=np.logaddexp(0.,model.par(pop,'bias')+sign*sum(model.p[pop]['kernel'])*h)
        put(pop,BC_MECH,'activation',activation)
        target[pop]=model.par(pop,'Vref')+model.par(pop,'native_voltage_gain')*(activation-model.par(pop,'native_reference'))
    diagonal=[];rhs=[]
    for pop in model.gap_names:
        g=model.gL[pop];vtarget=model.par(pop,'EL') if pop=='AII_AC' else target[pop]
        e=aii_ge if pop=='AII_AC' else np.zeros(model.counts[pop])
        diagonal.append(g+e);rhs.append(g*vtarget+e*(model.par(pop,'EE') if pop=='AII_AC' else 0.))
    matrix=model.gap.matrix+sparse.diags(np.concatenate(diagonal))
    voltage=spsolve(matrix,np.concatenate(rhs)-model.gap_boundary_product)
    if not np.isfinite(voltage).all():raise ContractError('Nonfinite joint electrical background')
    for pop,section in model.gap_slices.items():put(pop,MEMBRANE if pop=='AII_AC' else BC_MECH,'V',voltage[section])
    membrane_release('AII_AC')
    scalar_iterations=[]
    for pop,component in (('midget_BC_OFF','AII_FMB_glycine'),('diffuse_BC_OFF','AII_DB1_glycine')):
        c=model.contacts[component];rate=event_rate('AII_AC')[c['pre']]
        rise=model.cp(component,'tau_rise',positive=True)
        lo=model.cp(component,'decay_minus100',positive=True);hi=model.cp(component,'decay_0',positive=True)
        V=target[pop].copy();EI=model.par(pop,'EI')
        for row in range(model.counts[pop]):
            guard();indices=np.flatnonzero(c['post']==row)
            if not len(indices):continue
            def balance(v):
                nonlocal count
                count+=1
                # Retained voltage-dependent source interpolation only.
                decay=lo[indices]+(hi[indices]-lo[indices])*np.clip((v+.1)/.1,0,1)
                rise_here=rise[indices]
                tp=np.log(decay/rise_here)/(1/rise_here-1/decay)
                norm=np.exp(-tp/decay)-np.exp(-tp/rise_here)
                gi=np.sum(c['coefficient'][indices]*rate[indices]*(decay-rise_here)/norm)
                return (model.gL[pop][row]*(target[pop][row]-v)+gi*(EI[row]-v))/model.gL[pop][row]
            lower,upper=sorted((float(EI[row]),float(target[pop][row])))
            if lower==upper:V[row]=lower
            else:
                V[row],info=brentq(balance,lower,upper,xtol=1e-14,rtol=1e-14,maxiter=64,full_output=True)
                scalar_iterations.append(info.iterations)
        put(pop,BC_MECH,'V',V)
        contacts(component)
    for pop in BC:
        V=state[pop,BC_MECH,'V']
        r=expit((V-model.par(pop,'release_half'))/model.par(pop,'release_slope',positive=True))
        put(pop,BC_MECH,'terminal_release',r)
        put(pop,BC_MECH,'A',model.par(pop,'adaptation_scale')*r if model.p[pop]['adaptation_enabled'] else 0.)
    # Legacy S-ON h is solved on its own unchanged input; no new state sharing.
    out=model.outputs(t,{k:state[k] for k in model.state_keys},release)
    put(LEGACY_S,AC_MECH,'h',out[LEGACY_S,'target_inhibition'])
    for channel in model.channels:
        pop=channel['pool'];V,_,s=equilibrium(out[pop,'gE'],model.p[pop])
        put(pop,MECHANISM,'V',V);put(pop,MECHANISM,'s',s)
    out=model.outputs(t,{k:state[k] for k in model.state_keys},release)
    stationary=[]
    for pop in RGC:
        ge,gi=out[pop,'gE'],out[pop,'gI'];g=model.gL[pop]
        V=(g*model.par(pop,'EL')+ge*model.par(pop,'EE')+gi*model.par(pop,'EI'))/(g+ge+gi)
        put(pop,RGC_MECH,'V',V)
        if not len(V):continue
        law=stationary_history_rates(SimpleNamespace(history_kernel=np.asarray(model.p[pop]['spike_history_kernel']),dt=.001,
            deterministic_state_independent_of_spikes=True,**{n:model.par(pop,n) for n in ('alpha','mu','beta')}),V,cold_start_bins=1)
        pi=law['history_probabilities']
        for i in range(8):put(pop,'RGC_expected_history','history_probability_'+str(i),pi[:,i])
        rng=np.random.Generator(np.random.PCG64());rng.bit_generator.state=deepcopy(seed['spike_states'][pop]['rng_state'])
        picks=np.sum(rng.random(len(V))[:,None]>np.cumsum(pi,axis=1),axis=1)
        if np.any(picks>=8):raise ContractError('Stationary history categorical mass invalid; no clipping')
        history=STATES[picks].T.astype(np.float64)
        seed['spike_states'][pop].update(history=history,last_bin_counts=None,rng_state=deepcopy(rng.bit_generator.state),pending_events=[],
            conditional_rate_Hz=model.par(pop,'alpha')*np.logaddexp(0.,(V+np.asarray(model.p[pop]['spike_history_kernel'])@history-model.par(pop,'mu'))/model.par(pop,'beta')))
        stationary.append(dict(population=pop,maximum_probability_residual=law['maximum_stationary_residual'],sampled_from_stationary_law=True))
    inner={k:state[k] for k in model.state_keys}
    rhs=model.rhs(t,inner,release)
    outputs=model.outputs(t,inner,release)
    transitions=model.pure_transition_witness(inner,release,seed['input_reference']['rod_background_rstar_s'],time_seconds=t)
    for name,value in native.blocks(y).items():
        k=('HC','reduced_native',name) if name in ('w','z2','Vh') else ('cones','selected_original_z1_auxiliary','z1_aux') if name=='z1_aux' else ('cones','reduced_native',name)
        state[k]=value;rhs[k]=evaluation.rhs[native.slices[name]]
    for name,value in zip(('R','E'),anchor.at(t)):
        state['cones','analytic_RE',name]=value;rhs['cones','analytic_RE',name]=evaluation.analytic_rhs[name]
    observation=seed['observation_prehistory'];observation.update(history_times=np.array([t]),history_values=state['HC','reduced_native','Vh'][None,:],prehistory=state['HC','reduced_native','Vh'])
    seed['native_inputs']={f:background.copy() for f in seed['native_inputs']}
    seed['rod_inputs']={f:np.full_like(v,seed['input_reference']['rod_background_rstar_s']) for f,v in seed['rod_inputs'].items()}
    monitor=GroupTimeMonitor(assembly,state,parameters=seed['inner_parameters'])
    transition_keys=set(monitor.transition_keys)
    monitor.observe(t,'right',state,rhs,transitions={k:transitions[k] for k in transition_keys},ports=outputs,aux=dict(spike_states=seed['spike_states']))
    report=monitor.report(.1)
    result=dict(category='BACKGROUND_EQUILIBRIUM',status='FINITE_JOINT_BACKGROUND_CHECKED',passed=report['passed'],
        retained_local_background_equilibrium=report['passed'],whole_retina_or_natural_video_background_qualified=False,
        method='sparse linear AII/ONBC/gap solve; bracketed OFFBC voltage-dependent glycine; analytic release/resource/adaptation/AC; stationary expected history; categorical sampled prehistory; constant HC delay',
        absolute_fixed_point_report=report,scalar_function_evaluations=count,maximum_scalar_iterations=max(scalar_iterations,default=0),
        stationary_history=stationary,explicit_exterior=seed['patch_boundary'],
        native_constant_input_hash=canonical_hash(background),seed_sha256=canonical_hash(seed),
        finite_record_spike_stationarity_tested=False,automatically_used_by_dynamic_run=False,
        all_retained_state_history_delay_resource_gap_checked=True,**QUALIFICATION)
    save_bundle(output/'candidate_constant_background',kind='source_construction_seed',identity=identity(assembly,seed),payload=seed,qualification=QUALIFICATION)
    write(output/'result.json',result)
    return result
