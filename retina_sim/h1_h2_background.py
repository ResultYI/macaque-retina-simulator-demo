"""Independent finite held-background initialization; physiological values stay fixed.

The inner algebra follows ac_vs_background.qualify_background. The saved H1/H2
fixed point is reused, and every inner state is independently reconstructed.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve
from scipy.special import expit

from .contracts import ContractError,canonical_hash
from .spatial_group_graph import digest
from .spatial_group_model import BC,RGC
from .group_inner_dynamics import BC_MECH,MEMBRANE,AC_MECH,RGC_MECH,RODS,key
from .group_parameter_binding import source_rod_equilibrium
from .ac_vs_model import MECHANISM as AC_VS,LEGACY_S,equilibrium
from .rgc_expectation import stationary_history_rates,STATES
from .h1_h2_effective import H1H2Native,MODEL,bind_parameters

INITIALIZATION='h1_h2_fixed_reference_joint_background_v1'


def initialize_joint_background(assembly,outer_seed,repo_root,protocol,*,guard=lambda:None):
    """Solve the current finite local graph with immutable baseline and reservoirs."""
    if outer_seed.get('h1_h2_model',{}).get('artifact')!=assembly.artifact_sha256:
        raise ContractError('Exact saved H1/H2 outer seed required')
    parameters_hash=canonical_hash(outer_seed['inner_parameters'])
    seed=deepcopy(outer_seed);state=seed['states'];t=seed['time_seconds']
    outer_keys={k for k in state if k[0] in ('cones','HC')}
    outer_hash=canonical_hash({k:state[k] for k in outer_keys})
    native=H1H2Native(assembly);native.boundary=seed['h1_h2_model']['boundary']
    background=np.array(state['cones','analytic_RE','R'],copy=True)
    if (not np.array_equal(background,state['cones','analytic_RE','E']) or
            not np.array_equal(background,seed['input_reference']['native_background_td'])):
        raise ContractError('Saved outer fixed point and declared native Td background differ')
    y,anchor=native.construction_point(SimpleNamespace(states=state,time_seconds=t),background,t+1.,0)
    release=native.evaluate(y,t,anchor).outputs['release']
    model=assembly.runtime_factories['inner'](assembly,seed['inner_parameters'])
    written=set();scalar_iterations=[];stationary=[]
    def store(k,value):
        a=np.broadcast_to(value,(model.sizes[k],)).astype(np.float64).copy()
        if not np.isfinite(a).all():raise ContractError('Nonfinite background state: '+str(k))
        state[k]=a;written.add(k)
    def put(pop,mech,name,value):store(key(pop,mech,name),value)
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
        store((c['route'].pre,component,'a'),rate*rise)
        store((c['route'].pre,component,'b'),rate*decay)
        return np.bincount(c['post'],weights=c['coefficient']*rate*(decay-rise)/norm,minlength=model.counts[c['route'].post])
    rod_rate=float(seed['input_reference']['rod_background_rstar_s'])
    source_p,rod=source_rod_equilibrium(repo_root,rod_rate,model.counts['rods'])
    for n,v in source_p.items():
        if n in model.rod and not np.array_equal(np.asarray(v),np.asarray(model.rod[n])):
            raise ContractError('Rod equilibrium would use different frozen parameters: '+n)
    for n,v in rod.items():put('rods',RODS,n,v)
    rodV=model.par('rods','EL')-(model.rod_current(state)-model.par('rods','current_anchor'))/model.par('rods','gL',positive=True)
    put('rods','rod_membrane','V',rodV);membrane_release('rods')
    c=model.contacts['individual_rod_RBC_contacts'];drive=state['rods','rod_membrane','release'][c['pre']]
    m=expit(-model.cp('individual_rod_RBC_contacts','hill',positive=True)*np.log(drive/model.cp('individual_rod_RBC_contacts','K',positive=True)))
    store(('rods','individual_rod_RBC_contacts','m'),m)
    ge=np.bincount(c['post'],weights=c['coefficient']*m,minlength=model.counts['rod_BC'])
    put('rod_BC',MEMBRANE,'V',(model.gL['rod_BC']*model.par('rod_BC','EL')+ge*model.par('rod_BC','EE'))/(model.gL['rod_BC']+ge))
    membrane_release('rod_BC');aii_ge=contacts('RBC_AII_AMPA')
    target={}
    for pop in BC:
        route=next(r for r in model.routes if r.pre=='cones' and r.post==pop)
        h=model.mean[route.name]@release-model.par(pop,'baseline')
        history=np.vstack((h,h,h))
        for i in range(3):put(pop,BC_MECH,'history_'+str(i),history[i])
        sign=-1 if pop.endswith('_ON') or pop=='S_ON_BC' else 1
        activation=np.logaddexp(0.,model.par(pop,'bias')+sign*(np.asarray(model.p[pop]['kernel'])@history))
        put(pop,BC_MECH,'activation',activation)
        target[pop]=model.par(pop,'Vref')+model.par(pop,'native_voltage_gain')*(activation-model.par(pop,'native_reference'))
    diagonal=[];right=[]
    for pop in model.gap_names:
        g=model.gL[pop];vtarget=model.par(pop,'EL') if pop=='AII_AC' else target[pop]
        e=aii_ge if pop=='AII_AC' else np.zeros(model.counts[pop])
        diagonal.append(g+e);right.append(g*vtarget+e*(model.par(pop,'EE') if pop=='AII_AC' else 0.))
    voltage=spsolve(model.gap.matrix+sparse.diags(np.concatenate(diagonal)),np.concatenate(right)-model.gap_boundary_product)
    if not np.isfinite(voltage).all():raise ContractError('Nonfinite joint AII/ONBC/gap solve')
    for pop,section in model.gap_slices.items():put(pop,MEMBRANE if pop=='AII_AC' else BC_MECH,'V',voltage[section])
    membrane_release('AII_AC')
    for pop,component in (('midget_BC_OFF','AII_FMB_glycine'),('diffuse_BC_OFF','AII_DB1_glycine')):
        c=model.contacts[component];rate=event_rate('AII_AC')[c['pre']]
        rise=model.cp(component,'tau_rise',positive=True)
        lo=model.cp(component,'decay_minus100',positive=True);hi=model.cp(component,'decay_0',positive=True)
        V=target[pop].copy();EI=model.par(pop,'EI')
        for row in range(model.counts[pop]):
            guard();indices=np.flatnonzero(c['post']==row)
            if not len(indices):continue
            def balance(v):
                # Retained source interpolation saturation is part of the equation.
                decay=lo[indices]+(hi[indices]-lo[indices])*np.minimum(1.,np.maximum(0.,(v+.1)/.1))
                rt=rise[indices];tp=np.log(decay/rt)/(1/rt-1/decay)
                norm=np.exp(-tp/decay)-np.exp(-tp/rt)
                gi=np.sum(c['coefficient'][indices]*rate[indices]*(decay-rt)/norm)
                return (model.gL[pop][row]*(target[pop][row]-v)+gi*(EI[row]-v))/model.gL[pop][row]
            lower,upper=sorted((float(EI[row]),float(target[pop][row])))
            if lower==upper:V[row]=lower
            else:
                V[row],info=brentq(balance,lower,upper,xtol=1e-14,rtol=1e-14,maxiter=64,full_output=True)
                scalar_iterations.append(info.iterations)
        put(pop,BC_MECH,'V',V);contacts(component)
    for pop in BC:
        V=state[pop,BC_MECH,'V']
        r=expit((V-model.par(pop,'release_half'))/model.par(pop,'release_slope',positive=True))
        put(pop,BC_MECH,'terminal_release',r)
        put(pop,BC_MECH,'A',model.par(pop,'adaptation_scale')*r if model.p[pop]['adaptation_enabled'] else 0.)
    inner=lambda:{k:state[k] for k in model.state_keys}
    out=model.outputs(t,inner(),release)
    put(LEGACY_S,AC_MECH,'h',out[LEGACY_S,'target_inhibition'])
    for channel in model.channels:
        pop=channel['pool'];V,_,s=equilibrium(out[pop,'gE'],model.p[pop])
        put(pop,AC_VS,'V',V);put(pop,AC_VS,'s',s)
    out=model.outputs(t,inner(),release)
    for pop in RGC:
        ge,gi=out[pop,'gE'],out[pop,'gI'];g=model.gL[pop]
        V=(g*model.par(pop,'EL')+ge*model.par(pop,'EE')+gi*model.par(pop,'EI'))/(g+ge+gi)
        put(pop,RGC_MECH,'V',V)
        if not len(V):continue
        law=stationary_history_rates(SimpleNamespace(history_kernel=np.asarray(model.p[pop]['spike_history_kernel']),dt=.001,
            deterministic_state_independent_of_spikes=True,**{n:model.par(pop,n) for n in ('alpha','mu','beta')}),V,cold_start_bins=1)
        pi=law['history_probabilities']
        for i in range(8):put(pop,'RGC_expected_history','history_probability_'+str(i),pi[:,i])
        old_rng=deepcopy(seed['spike_states'][pop]['rng_state'])
        if seed['spike_states'][pop]['pending_events']:raise ContractError('Background initialization cannot discard pending events')
        rng=np.random.Generator(np.random.PCG64());rng.bit_generator.state=old_rng
        picks=np.sum(rng.random(len(V))[:,None]>np.cumsum(pi,axis=1),axis=1)
        if np.any(picks>=8):raise ContractError('Stationary categorical mass invalid; no clipping')
        history=STATES[picks].T.astype(np.float64)
        seed['spike_states'][pop].update(history=history,last_bin_counts=None,rng_state=deepcopy(rng.bit_generator.state),
            conditional_rate_Hz=model.par(pop,'alpha')*np.logaddexp(0.,(V+np.asarray(model.p[pop]['spike_history_kernel'])@history-model.par(pop,'mu'))/model.par(pop,'beta')))
        stationary.append(dict(population=pop,maximum_probability_residual=law['maximum_stationary_residual'],
            categorical_draws=len(V),rng_before_sha256=canonical_hash(old_rng),
            rng_after_sha256=canonical_hash(rng.bit_generator.state),
            semantics='sampled model prehistory from exact stationary law; no observed spike events fabricated'))
    if written!=set(model.state_keys):raise ContractError('Background solver omitted inner states: '+str(set(model.state_keys)-written))
    model.validate(inner())
    if canonical_hash(seed['inner_parameters'])!=parameters_hash:raise ContractError('Initialization changed parameters')
    if canonical_hash({k:state[k] for k in outer_keys})!=outer_hash:raise ContractError('Saved outer fixed point changed')
    seed['native_inputs']={i:background.copy() for i in seed['native_inputs']}
    seed['rod_inputs']={i:np.full_like(v,rod_rate) for i,v in seed['rod_inputs'].items()}
    meta=dict(schema=INITIALIZATION,initialized_model=INITIALIZATION,parent_dynamics_model=MODEL,
        parent_artifact=assembly.artifact_sha256,parent_seed_sha256=canonical_hash(outer_seed),
        parameters_sha256=parameters_hash,initial_state_sha256=canonical_hash(state),
        outer_state_sha256=outer_hash,held_reservoir_sha256=canonical_hash(dict(
            native=seed['h1_h2_model']['boundary'],inner=assembly.boundary_runtime)),
        baseline_policy='frozen_reference_unchanged; H0=H1=H2=current_background_drive-baseline',
        inner_states_reconstructed=len(written),scalar_max_iterations=max(scalar_iterations,default=0),
        sampled_prehistory=stationary,finite_record_spike_stationarity_verified=False,
        source_sha256=digest(Path(__file__)),outer_joint_solves=0,
        local_background_qualified=False,space_closure_qualified=False,physiological_calibration=False,FIT_READY=False)
    seed['joint_background_initialization']=meta
    bundle=bind_parameters(assembly,seed,repo_root,protocol)
    meta['parameter_binding_id']=bundle.identity
    meta['initialized_model_id']=canonical_hash(meta)
    return seed,bundle
