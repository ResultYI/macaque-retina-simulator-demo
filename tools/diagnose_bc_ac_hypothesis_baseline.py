"""Read-only diagnosis from recorded traces and exact retained model algebra.

Constructs the old engineering graph for its factory port/parameter references;
never advances it. A binary three-bin-history Markov chain is an algebraic
representation of the existing Bernoulli sampler, not a new model mechanism.
"""
from pathlib import Path
import argparse,json,csv,sys,hashlib,itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retina_sim.network import JointNetwork,InputBundle
from retina_sim.runtime_profiles import LightContext
from retina_sim.bc_ac_hypothesis import constant_background_channels
from examples.full_target_network import full_target_network

STATES=np.asarray(list(itertools.product((0,1),repeat=3)),int)
INDEX={tuple(h):i for i,h in enumerate(STATES)}
NEXT0=np.array([INDEX[(0,h[0],h[1])] for h in STATES])
NEXT1=np.array([INDEX[(1,h[0],h[1])] for h in STATES])


def hazard(V,history,alpha,mu,beta):
    return alpha*np.logaddexp(0.,(V+history-mu)/beta)


def stationary(V,alpha,mu,beta,kernel,dt):
    lam=hazard(V,STATES@kernel,alpha,mu,beta);p=-np.expm1(-dt*lam)
    P=np.zeros((8,8))
    for j in range(8):P[j,NEXT0[j]]+=1-p[j];P[j,NEXT1[j]]+=p[j]
    A=P.T-np.eye(8);A[-1]=1.;b=np.zeros(8);b[-1]=1.
    pi=np.linalg.solve(A,b)
    if np.min(pi)<-1e-12 or np.max(abs(pi@P-pi))>1e-10:raise ValueError('Stationary algebra failed')
    return dict(observed_Bernoulli_rate_Hz=float(pi@p/dt),mean_conditional_hazard_Hz=float(pi@lam),
        zero_history_hazard_Hz=float(hazard(V,0.,alpha,mu,beta)),
        zero_history_bin_rate_Hz=float(-np.expm1(-dt*hazard(V,0.,alpha,mu,beta))/dt))


def expected_on_recorded_V(V,aux,alpha,mu,beta,kernel,dt):
    """Exact future-noise expectation conditional on the saved prepared history.

    V is spike-independent in the retained RGC. The first bin was already drawn
    in the saved checkpoint, so its count is held fixed rather than redrawn.
    """
    n,nt=V.shape
    h=np.asarray(aux['history'],int);first=np.asarray(aux['last_bin_counts'],int)
    if h.shape!=(3,n) or np.any((h<0)|(h>1)) or np.any((first<0)|(first>1)):
        raise ValueError('This exact algebra requires the retained binary three-bin sampler')
    pi=np.zeros((n,8))
    for c in range(n):pi[c,INDEX[(int(first[c]),int(h[0,c]),int(h[1,c]))]]=1.
    expected_counts=first.astype(float);expected_hazard=np.empty((n,nt))
    expected_hazard[:,0]=hazard(V[:,0],kernel@h,alpha,mu,beta)
    for j in range(1,nt):
        lam=hazard(V[:,j,None],(STATES@kernel)[None,:],alpha,mu,beta)
        q=-np.expm1(-dt*lam);expected_hazard[:,j]=(pi*lam).sum(axis=1)
        if j==nt-1:break  # final endpoint's next-bin event is outside observation
        expected_counts+=(pi*q).sum(axis=1)
        nxt=np.zeros_like(pi)
        for k in range(8):nxt[:,NEXT0[k]]+=pi[:,k]*(1-q[:,k]);nxt[:,NEXT1[k]]+=pi[:,k]*q[:,k]
        pi=nxt
    return expected_counts,expected_hazard


def summarize(x):
    a=np.asarray(x);return dict(mean=float(a.mean()),minimum=float(a.min()),maximum=float(a.max()))


def rc_from_saved_traces(diagnosis,folder):
    """Closed-form membrane audit using the saved (nearly constant) currents."""
    original=json.loads((folder/'result.json').read_bytes())
    with np.load(folder/'short_wash_in/traces.npz',allow_pickle=False) as z:warm={k:z[k] for k in z.files}
    with np.load(folder/original['branch_folders']['background']/'traces.npz',allow_pickle=False) as z:bg={k:z[k] for k in z.files}
    rows={}
    for cid,row in diagnosis['baseline_decomposition'].items():
        p=row['RGC_constants'];ge=warm[cid+'.gE'][:,0];gi=warm[cid+'.gI'][:,0]
        G=p['gL_S']+ge+gi;tau=p['C_F']/G
        V=(p['gL_S']*p['EL_V']+ge*p['EE_V']+gi*p['EI_V'])/G
        V0=warm[cid+'.voltage'][:,0]
        predicted=lambda t:V[:,None]+(V0-V)[:,None]*np.exp(-np.asarray(t)[None,:]/tau[:,None])
        rows[cid]=dict(effective_tau_ms=summarize(tau*1000),
            maximum_wash_in_closed_form_error_V=float(np.max(abs(predicted(warm['time'])-warm[cid+'.voltage']))),
            maximum_control_closed_form_error_V=float(np.max(abs(predicted(bg['time'])-bg[cid+'.voltage']))),
            maximum_conductance_variation_S=max(float(np.max(abs(warm[cid+'.gE']-ge[:,None]))),
                float(np.max(abs(warm[cid+'.gI']-gi[:,None]))),float(np.max(abs(bg[cid+'.gE']-ge[:,None]))),
                float(np.max(abs(bg[cid+'.gI']-gi[:,None])))),
            formula='V(t)=Vinf+(V0-Vinf)*exp(-t*(gL+gE+gI)/C); no RNG or history enters V',
            implications='Large initial voltage drift is the declared unmatched V0 relaxing under held synaptic conductances; fixing initial state would not fix the high steady hazard')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--traces',type=Path,default=ROOT/'reports/modeling_scope_20261019/hypothesis_pulses_v1')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    p=args.traces;r=json.loads((p/'result.json').read_bytes());layout=json.loads((p/'recorded_layout.json').read_bytes())
    checkpoints={name:json.loads((p/name).read_bytes())['payload'] for name in ('root_state.json','prepared_state.json')}
    rootaux=checkpoints['root_state.json']['spike_states'];prepared=checkpoints['prepared_state.json']['spike_states']
    traces={}
    for name,folder in r['branch_folders'].items():
        with np.load(p/folder/'traces.npz',allow_pickle=False) as z:traces[name]={k:z[k].copy() for k in z.files}
    consumers=r['input_consumer_contracts']
    light=LightContext(native_background_td=float(consumers['native_light']['base_values'][0]),
        cone_background_rstar_s=float(consumers['physical_cone_light']['base_values'][0]),
        rod_background_rstar_s=float(consumers['rod_light']['base_values'][0]),
        background_source='read_only_same_consumer_values_as_recorded_pulse_test')
    old,_,audit=full_target_network(ROOT,effective_cones=64,effective_rods=64,duration=.0001,
        runtime_profile='effective_dynamics_v1',light_context=light)
    model=JointNetwork(old,compute_backend='python');y=np.concatenate([q.values for q in model.initial_conditions()])
    bg=constant_background_channels(old,light,stop=.0001)
    old_inputs,old_outputs=model.evaluate(0.,y,InputBundle({k:v.at(0.) for k,v in bg.items()}))
    rgcs=list(r['RGC_initialization_drift']);baseline={};expected={};arrays={}
    for cid in rgcs:
        c=old.components[cid];native=c.native_model
        if list(c.outputs['voltage'].cell_ids)!=layout['ports'][cid+'.voltage']['cell_ids']:raise ValueError('Old/new cell order differs')
        n=len(c.outputs['voltage'].cell_ids);a=float(native.alpha[0]);mu=float(native.mu[0]);beta=float(native.beta[0]);dt=native.dt;k=native.history_kernel
        if not (np.all(native.alpha==a) and np.all(native.mu==mu) and np.all(native.beta==beta)):
            raise ValueError('This diagnostic requires the actual broadcast RGC transfer parameters')
        if len(k)!=3:raise ValueError('Stationary analysis is for the actual three-bin history')
        gL=native.gL;EL=native.EL;EE=native.EE;EI=native.EI
        olde=old_inputs[cid]['excitation'].values;oldi=old_inputs[cid]['inhibition'].values
        newe=traces['background'][cid+'.gE'][:,-1];newi=traces['background'][cid+'.gI'][:,-1]
        balance=lambda e,i:(gL*EL+e*EE+i*EI)/(gL+e+i)
        vold=balance(olde,oldi);vnew=balance(newe,newi)
        ve=balance(newe,oldi);vi=balance(olde,newi)
        ex=.5*((ve-vold)+(vnew-vi));inh=.5*((vi-vold)+(vnew-ve))
        oldstats=[stationary(v,a,mu,beta,k,dt) for v in vold]
        newstats=[stationary(v,a,mu,beta,k,dt) for v in vnew]
        baseline[cid]=dict(cells=n,RGC_constants=dict(C_F=float(native.C[0]),gL_S=float(gL[0]),EL_V=float(EL[0]),EE_V=float(EE[0]),EI_V=float(EI[0]),
                alpha_Hz=a,mu_V=mu,beta_V=beta,history_kernel_V=k.tolist(),bin_seconds=dt),
            intrinsic_at_EL=stationary(float(EL[0]),a,mu,beta,k,dt),
            old_factory_V0_V=summarize(native.V0),
            old_factory_conditional_hazard_Hz=summarize(rootaux[cid]['conditional_rate_Hz']),
            old_factory_input_reference=dict(gE_S=summarize(olde),gI_S=summarize(oldi),
                held_input_voltage_balance_V=summarize(vold),
                input_current_at_EL_A=summarize(olde*(EE-EL)+oldi*(EI-EL)),
                stationary_single_RGC_algebra={key:float(np.mean([s[key] for s in oldstats])) for key in oldstats[0]},
                limitation='Held old factory outputs, not a solved or simulated old full-feedback steady state'),
            new_recorded_background=dict(gE_S=summarize(newe),gI_S=summarize(newi),voltage_end_V=summarize(traces['background'][cid+'.voltage'][:,-1]),
                algebraic_voltage_balance_V=summarize(vnew),
                max_balance_minus_recorded_end_V=float(np.max(abs(vnew-traces['background'][cid+'.voltage'][:,-1]))),
                gE_temporal_range_max_S=float(np.max(np.ptp(traces['background'][cid+'.gE'],axis=1))),
                gI_temporal_range_max_S=float(np.max(np.ptp(traces['background'][cid+'.gI'],axis=1))),
                stationary_single_RGC_algebra={key:float(np.mean([s[key] for s in newstats])) for key in newstats[0]},
                observed_mean_rate_Hz=float(traces['background'][cid+'.spike_count'].sum()/(n*.2))),
            new_minus_old_factory_voltage_balance_mV=dict(total=summarize((vnew-vold)*1000),
                excitatory_input_contribution=summarize(ex*1000),inhibitory_input_contribution=summarize(inh*1000),
                method='Symmetric two-order algebraic decomposition, fixed intrinsic RGC constants; not circuit ablation or causal old/new steady-state attribution'),
            voltage_is_independent_of_spike_RNG=native.deterministic_state_independent_of_spikes)
        expected[cid]={}
        for branch in ('background','plus10','minus10'):
            counts,lam=expected_on_recorded_V(traces[branch][cid+'.voltage'],prepared[cid],a,mu,beta,k,dt)
            arrays[branch+'.'+cid+'.expected_hazard_Hz']=lam
            expected[cid][branch]=dict(expected_count_given_prepared_history=float(counts.sum()),
                observed_count=int(traces[branch][cid+'.spike_count'].sum()),mean_expected_hazard_at50_Hz=float(lam[:,50].mean()),
                mean_observed_conditional_hazard_at50_Hz=float(traces[branch][cid+'.conditional_rate'][:,50].mean()))
        for branch in ('plus10','minus10'):
            expected[cid][branch]['minus_control_expected_count']=expected[cid][branch]['expected_count_given_prepared_history']-expected[cid]['background']['expected_count_given_prepared_history']
            expected[cid][branch]['minus_control_expected_hazard_at50_Hz']=expected[cid][branch]['mean_expected_hazard_at50_Hz']-expected[cid]['background']['mean_expected_hazard_at50_Hz']
            expected[cid][branch]['minus_control_observed_hazard_at50_Hz']=expected[cid][branch]['mean_observed_conditional_hazard_at50_Hz']-expected[cid]['background']['mean_observed_conditional_hazard_at50_Hz']
    interface={}
    for cid in r['matched_background_subtracted_layers']['plus10']:
        interface[cid]=dict(old_factory_output=summarize(old_outputs[(cid,'activation')].values),
            new_after_initialization_output_reference=r['AC_after_initialization'].get(cid),
            new_recorded_background_output=summarize(traces['background'][cid+'.activation'][:,0]),
            old_mechanism=old.components[cid].mechanism,new_mechanism='contact-first receptor/passive membrane candidate',
            old_range='unbounded softplus effective activity' if cid.endswith('_AC') else 'bounded sigmoid effective terminal release',
            new_range='bounded sigmoid effective output; no concentration or vesicle calibration')
    # Reapply unchanged, instantaneous positive conductance adapters to the saved
    # per-cell activities. Verify individual contributions sum to recorded totals.
    routes={};s_on='small_bistratified_S_ON'
    for final in old.links:
        l=final.specification
        if l.target_component!=s_on or l.target_port not in ('excitation','inhibition'):continue
        syn=old.components[l.source_component]
        gain=syn.parameters.quantity('gain','S','synaptic_conductance_scale').values
        upstream=[link for link in old.links if link.specification.target_component==l.source_component and link.specification.target_port=='activation']
        driver=upstream[0].specification.source_component
        entries={}
        for branch in ('background','plus10','minus10'):
            activation=sum((link.matrix@traces[branch][link.specification.source_component+'.activation'] for link in upstream))
            conductance=final.matrix@(gain[:,None]*activation)
            entries[branch]=conductance
        routes[l.source_component]=dict(driver=driver,receptor=l.target_port,all_weights_nonnegative=bool(np.all(final.matrix.data>=0) and all(np.all(x.matrix.data>=0) for x in upstream)),
            background_nS=summarize(entries['background'][:,0]*1e9),
            plus_at50_delta_nS=summarize((entries['plus10'][:,50]-entries['background'][:,50])*1e9),
            minus_at50_delta_nS=summarize((entries['minus10'][:,50]-entries['background'][:,50])*1e9))
        for branch,a in entries.items():arrays[branch+'.'+l.source_component+'.conductance_S']=a
    for branch in traces:
        for receptor,key in (('excitation','gE'),('inhibition','gI')):
            total=sum(arrays[branch+'.'+name+'.conductance_S'] for name,row in routes.items() if row['receptor']==receptor)
            if not np.allclose(total,traces[branch][s_on+'.'+key],rtol=0,atol=1e-22):raise ValueError('S-ON route sum differs from original runtime')
    subtype_by_id={gid:st for pop in old.populations.values() for gid,st in zip(pop.cells.cell_id,pop.cells.subtype)}
    drive_subtypes={}
    for cid in ('S_ON_BC','diffuse_BC_OFF'):
        target=old.components[cid].inputs['drive'];po={gid:i for i,gid in enumerate(target.cell_ids)}
        total=np.zeros(len(po));masses={st:np.zeros(len(po)) for st in ('L','M','S')}
        for l in old.spec.links:
            if l.target_component!=cid or l.target_port!='drive':continue
            for pre,post,w in zip(l.connections.pre_cell_id,l.connections.post_cell_id,l.connections.weight):
                total[po[post]]+=w;st=subtype_by_id.get(pre,'unknown')
                if st in masses:masses[st][po[post]]+=w
        drive_subtypes[cid]=dict(source_subtypes=[st for st,a in masses.items() if np.any(a>0)],
            target_cell_count=len(po),normalized_weight_fractions={st:summarize(a/total) for st,a in masses.items()},
            target_cells_with_S_input=int(np.count_nonzero(masses['S']>0)),
            preexisting_engineering_wiring_not_changed_by_candidate=True)
    retry={}
    for f in sorted((p/'background').glob('chunk_*.npz')):
        with np.load(f,allow_pickle=False) as a,np.load(p/r['branch_folders']['background']/f.name,allow_pickle=False) as b:
            retry[f.name]=dict(all_arrays_equal=all(np.array_equal(a[k],b[k]) for k in a.files),
                maximum_non_time_difference=max(float(np.max(abs(a[k]-b[k]))) for k in a.files if k!='time'))
    metrics=[]
    for branch in ('plus10','minus10'):
        for cid in rgcs:
            t=traces[branch]['time']-traces[branch]['time'][0]
            dv=(traces[branch][cid+'.voltage']-traces['background'][cid+'.voltage']).mean(axis=0)
            peak=int(np.argmax(abs(dv)));reported=r['matched_background_subtracted_RGC'][branch][cid]['voltage']
            if not np.isclose(reported['peak_mean'],dv[peak],atol=1e-15,rtol=0):raise ValueError('Reported peak sign/definition mismatch')
            if not np.isclose(reported['end_to_peak_ratio'],abs(dv[-1])/abs(dv[peak]),atol=1e-12,rtol=0):raise ValueError('Recovery definition mismatch')
            metrics.append(dict(branch=branch,rgc=cid,peak_uV=float(dv[peak]*1e6),peak_time_ms=float(t[peak]*1e3),
                end_uV=float(dv[-1]*1e6),residual_to_peak=abs(float(dv[-1]/dv[peak]))))
            per_cell=traces[branch][cid+'.voltage']-traces['background'][cid+'.voltage']
            peaks=np.max(abs(per_cell),axis=1);eligible=peaks>1e-7
            metrics[-1].update(per_cell_peak_over_0_1uV=int(eligible.sum()),
                per_cell_end_to_own_peak_ratio=summarize(abs(per_cell[eligible,-1])/peaks[eligible]),
                mean_absolute_end_uV=float(np.mean(abs(per_cell[:,-1]))*1e6),
                mean_absolute_end_over_peak_mean_absolute=float(np.mean(abs(per_cell[:,-1]))/np.max(np.mean(abs(per_cell),axis=0))))
    result=dict(status='READ_ONLY_FORMULA_AND_TRACE_DIAGNOSIS_NO_NEURAL_ADVANCE',
        baseline_decomposition=baseline,exact_noise_expectation_on_recorded_voltage=expected,
        interface_comparison=interface,S_ON_unchanged_conductance_routes=routes,BC_input_subtypes=drive_subtypes,
        old_incomplete_vs_retried_control_arrays=retry,peak_and_recovery_recomputed=metrics,
        input_pulse_scope='Native td, physical L/M cone R*/s and rod R*/s were independently scaled together; no spectrum or matched S/LM cone contrast',
        intrinsic_formula='lambda=alpha*softplus((V + history_kernel @ strict_past_counts - mu)/beta)',
        count_observation='Per 1ms bin Bernoulli 1-exp(-lambda*dt), at most one spike; count rate differs from hazard',
        Markov_method='Exact eight-state representation of the existing binary three-bin history at held V; finite-time expectations use recorded spike-independent V and the saved first-bin draw',
        limits=['Old reference is factory conductance, not old adapted full-network output',
            'No biological baseline or noise statistics validated','No RGC constants, mechanisms or gains changed',
            'Engineering V0 is -65mV; actual frozen-geometry assembly has a separate local-balance initialization path'])
    result['RGC_initialization_closed_form']=rc_from_saved_traces(result,p)
    np.savez_compressed(args.output/'algebraic_expectations_and_routes.npz',**arrays)
    (args.output/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(status=result['status'],baseline={cid:dict(
        old_reference_V_mV=row['old_factory_input_reference']['held_input_voltage_balance_V']['mean']*1000,
        old_reference_stationary_rate_Hz=row['old_factory_input_reference']['stationary_single_RGC_algebra']['observed_Bernoulli_rate_Hz'],
        new_V_mV=row['new_recorded_background']['algebraic_voltage_balance_V']['mean']*1000,
        new_stationary_rate_Hz=row['new_recorded_background']['stationary_single_RGC_algebra']['observed_Bernoulli_rate_Hz'],
        new_observed_rate_Hz=row['new_recorded_background']['observed_mean_rate_Hz']) for cid,row in baseline.items()},
        expected_hazard_effect_at50={cid:row['plus10']['minus_control_expected_hazard_at50_Hz'] for cid,row in expected.items()}),ensure_ascii=False))


if __name__=='__main__':main()
