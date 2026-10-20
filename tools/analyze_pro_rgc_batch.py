"""B0-B2 saved-network analysis; no simulation, fit, optical launch or adoption."""
from pathlib import Path
from types import SimpleNamespace
import argparse,csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_rgc_type_candidate import BASE,REFERENCE,RGCS,load_arrays,save,sha
from tools.run_rgc_spatial_probe import PROTOCOLS
from retina_sim.rgc_expectation import exact_history_rates

PHASES=('gray_before','increment','gray_middle','decrement','gray_after')


def saved_model(snapshot):
    n=SimpleNamespace(**{k:np.asarray(v,float) if k!='dt' else float(v) for k,v in snapshot.items()})
    n.deterministic_state_independent_of_spikes=True
    n.intensity=lambda V,h:n.alpha*np.logaddexp(0.,(V+n.history_kernel@h-n.mu)/n.beta)
    return n


def membrane_decomposition(stim,gray,cid,snapshot):
    """Exact current identity; approximate 1ms-grid linear-drive reconstruction.

    Interpolation is confined to this diagnostic decomposition. Count
    expectations separately consume the unmodified native bin-start voltages.
    """
    V=gray[cid+'.voltage'];de=stim[cid+'.gE']-gray[cid+'.gE'];di=stim[cid+'.gI']-gray[cid+'.gI']
    p={k:np.asarray(snapshot[k])[:,None] for k in ('C','gL','EE','EI')}
    fe=(p['EE']-V)*de;fi=(p['EI']-V)*di
    G=p['gL']+stim[cid+'.gE']+stim[cid+'.gI'];dv=stim[cid+'.voltage']-V
    dt=np.diff(stim['time']);parts=[]
    for force in (fe,fi):
        out=np.zeros_like(V)
        for j,h in enumerate(dt):
            g=(G[:,j]+G[:,j+1])/2.;f=(force[:,j]+force[:,j+1])/2.
            e=np.exp(-g*h/p['C'][:,0]);out[:,j+1]=out[:,j]*e+(f/g)*(-np.expm1(-g*h/p['C'][:,0]))
        parts.append(out)
    reconstruction=parts[0]+parts[1]
    phases={}
    for j,phase in enumerate(PHASES):
        sl=slice(j*100,(j+1)*100);e=float(fe[:,sl].mean());i=float(fi[:,sl].mean())
        phases[phase]=dict(delta_gE_pS=float(de[:,sl].mean()*1e12),delta_gI_pS=float(di[:,sl].mean()*1e12),
            excitatory_drive_delta_pA=e*1e12,inhibitory_drive_delta_pA=i*1e12,
            signed_current_sum_over_absolute_sum=abs(e+i)/(abs(e)+abs(i)) if abs(e)+abs(i)>1e-30 else None,
            mean_voltage_delta_mV=float(dv[:,sl].mean()*1000),
            reconstructed_exc_voltage_mV=float(parts[0][:,sl].mean()*1000),
            reconstructed_inh_voltage_mV=float(parts[1][:,sl].mean()*1000),
            mean_membrane_tau_ms=float((p['C']/G)[:,sl].mean()*1000))
    return dict(phases=phases,reconstruction_max_cell_error_mV=float(np.max(abs(reconstruction-dv))*1000),
        reconstruction_scope='1ms-grid midpoint-held g and force approximation; not a new solver or count-voltage interpolation')


def chain_metrics(stim,gray,units,active_ids=None):
    metrics={}
    for key in stim:
        if key not in gray or key=='time' or key.endswith(('.spike_count','.window_rate','.conditional_rate')):continue
        a=stim[key];b=gray[key]
        if a.ndim!=2 or a.shape!=b.shape:continue
        delta=a-b;mean=delta.mean(axis=0)
        row=dict(unit=units.get(key,{}).get('unit','V' if key.endswith('.voltage') else 'declared_in_segment'),
            mean_bright_delta=float(delta[:,100:200].mean()),mean_dark_delta=float(delta[:,300:400].mean()),
            peak_abs_population_mean_delta=float(abs(mean).max()),maximum_abs_cell_delta=float(abs(delta).max()),
            mean_gray_level=float(b[:,:500].mean()),final_50ms_mean_delta=float(delta[:,450:500].mean()))
        if key=='outer_native.release' and active_ids:
            ids=units[key]['cell_ids'];mask=np.isin(ids,active_ids)
            if mask.any():row['active_cone_mean_bright_delta']=float(delta[mask,100:200].mean())
        metrics[key]=row
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,default=BASE/'pro_b0_b2_v1')
    args=p.parse_args();d=args.case.resolve();rows=[];expectation_checks={};decisions={}
    old_snap=json.loads((REFERENCE/'actual_RGC_parameter_snapshots.json').read_bytes())
    old_traces={name:load_arrays(REFERENCE/'old'/name/'traces.npz') for name in ('stimulus','matched_gray')}
    old_cp=json.loads((REFERENCE/'old/prepared_checkpoint.json').read_bytes())['payload']['spike_states']
    old_rates={name:{} for name in old_traces}
    for kind,trace in old_traces.items():
        for cid in RGCS:
            rates=exact_history_rates(saved_model(old_snap['old'][cid]),trace[cid+'.voltage'],trace['time'],old_cp[cid])
            old_rates[kind][cid]=rates
    for cid in RGCS:
        folder=REFERENCE/'candidate' if cid=='parasol_RGC_ON' else d/cid
        status=json.loads(((REFERENCE/'result.json') if cid=='parasol_RGC_ON' else (folder/'result.json')).read_bytes())
        if not status['status'].startswith('COMPLETE_'):raise ValueError('Completed per-type experiment required: '+cid)
        snapshots=old_snap['candidate'] if cid=='parasol_RGC_ON' else json.loads((folder/'actual_RGC_parameters.json').read_bytes())
        own_cp=json.loads((folder/'prepared_checkpoint.json').read_bytes())['payload']['spike_states']
        traces={kind:load_arrays(folder/kind/'traces.npz') for kind in ('stimulus','matched_gray')}
        rates={kind:exact_history_rates(saved_model(snapshots[cid]),trace[cid+'.voltage'],trace['time'],own_cp[cid])
               for kind,trace in traces.items()}
        # Retained ON-parasol count evidence must agree with this generalized
        # per-cell implementation; do not silently rewrite historical arrays.
        if cid=='parasol_RGC_ON':
            saved={'stimulus':folder/'exact_expected_count_rates.npz',
                   'matched_gray':folder/'matched_gray/exact_expected_count_rates.npz'}
            for kind,path in saved.items():
                previous=load_arrays(path)[cid]
                np.testing.assert_allclose(rates[kind]['expected_binary_count_rate_Hz'],previous,rtol=0,atol=1e-12)
        expectation_checks[cid]=dict(maximum_probability_mass_error=max(r['maximum_probability_mass_error'] for r in rates.values()),
            native_bin_voltage_no_interpolation=True,first_bin_fixed=True,stationary_history_claim=False)
        delta=rates['stimulus']['expected_binary_count_rate_Hz']-rates['matched_gray']['expected_binary_count_rate_Hz']
        old_delta=old_rates['stimulus'][cid]['expected_binary_count_rate_Hz']-old_rates['matched_gray'][cid]['expected_binary_count_rate_Hz']
        for variant,trace_set,rate_set in [('old',old_traces,{k:v[cid] for k,v in old_rates.items()}),('candidate',traces,rates)]:
            for kind,trace in trace_set.items():
                meta=json.loads(((REFERENCE/'old') if variant=='old' else folder).joinpath(kind,'segment.json').read_bytes())
                for j,phase in enumerate(PHASES):
                    sl=slice(j*100,(j+1)*100);left=round(.1+j*.1,10);right=round(left+.1,10)
                    actual=sum(left<=e['time_seconds']<right for e in meta['spike_events'][cid])/(trace[cid+'.voltage'].shape[0]*.1)
                    rows.append(dict(component=cid,variant=variant,branch=kind,phase=phase,
                        zero_history_hazard_Hz=float(rate_set[kind]['zero_history_hazard_Hz'][:,sl].mean()),
                        mean_history_hazard_Hz=float(rate_set[kind]['mean_history_hazard_Hz'][:,sl].mean()),
                        expected_binary_count_rate_Hz=float(rate_set[kind]['expected_binary_count_rate_Hz'][:,sl].mean()),
                        actual_one_seed_count_rate_Hz=actual,
                        recorded_one_history_conditional_hazard_Hz=float(trace[cid+'.conditional_rate'][:,sl].mean())))
        decisions[cid]=dict(mapping_pass=True,production_adoption=False,decision='RETAIN_CANDIDATE_ENTER_B2',
            source_parameter_grade='B: original ON-parasol fit',
            target_transfer_grade='B source / D retained project conditions' if cid=='parasol_RGC_ON' else 'D cross-type borrowing; not a target-type V-spike fit',
            old_gray_expected_count_rate_Hz=float(old_rates['matched_gray'][cid]['expected_binary_count_rate_Hz'].mean()),
            candidate_gray_expected_count_rate_Hz=float(rates['matched_gray']['expected_binary_count_rate_Hz'].mean()),
            bright_expected_delta_Hz=float(delta[:,100:200].mean()),dark_expected_delta_Hz=float(delta[:,300:400].mean()),
            old_bright_expected_delta_Hz=float(old_delta[:,100:200].mean()),
            peak_abs_expected_response_Hz=float(abs(delta.mean(axis=0)).max()),
            recovery_final_50ms_mean_delta_Hz=float(delta[:,450:500].mean()),
            matched_physiological_amplitude_criterion_available=False)
    with (d/'four_rate_phase_metrics.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Complete per-type runs supply the extra upstream gray recording. No
    # checkpoint is transferred: prove its continuous prepared state matches
    # the older ON-parasol preparation before using graded reference arrays.
    chain_root=d/'parasol_RGC_OFF';gray=load_arrays(chain_root/'matched_gray/traces.npz')
    units=json.loads((chain_root/'recorded_ports.json').read_bytes())
    reference_prepared=json.loads((chain_root/'prepared_checkpoint.json').read_bytes())['payload']['dynamic']['state']['ordered_state_arrays']
    weak={};probe_arrays={}
    for protocol in PROTOCOLS:
        folder=d/'spatial'/protocol;run=json.loads((folder/'result.json').read_bytes())
        if not run['status'].startswith('COMPLETE_'):raise ValueError('Completed spatial probe required: '+protocol)
        cid=run['mapping_component'];own=REFERENCE/'candidate' if cid=='parasol_RGC_ON' else d/cid
        own_prepared=json.loads((own/'prepared_checkpoint.json').read_bytes())['payload']['dynamic']['state']['ordered_state_arrays']
        if len(own_prepared)!=len(reference_prepared) or not all(np.array_equal(a,b) for a,b in zip(own_prepared,reference_prepared)):
            raise ValueError('Extra upstream gray recording does not have identical continuous prepared state')
        trace=load_arrays(folder/'stimulus/traces.npz');own_gray=load_arrays(own/'matched_gray/traces.npz')
        for key in (c+'.'+k for c in RGCS for k in ('voltage','gE','gI')):
            if not np.array_equal(gray[key],own_gray[key]):raise ValueError('Upstream gray reuse must preserve actual whole graded trace')
        snapshots=old_snap['candidate'] if cid=='parasol_RGC_ON' else json.loads((own/'actual_RGC_parameters.json').read_bytes())
        aux=json.loads((own/'prepared_checkpoint.json').read_bytes())['payload']['spike_states']
        stim_rates=load_arrays(folder/'stimulus/four_rate_expectations.npz')
        gray_rates=exact_history_rates(saved_model(snapshots[cid]),own_gray[cid+'.voltage'],own_gray['time'],aux[cid])
        delta=stim_rates[cid+'.expected_binary_count_rate_Hz']-gray_rates['expected_binary_count_rate_Hz']
        contracts=json.loads((folder/'input_contracts.json').read_bytes());contract=next(c for c in contracts.values() if c.get('probe')==protocol)
        stats=chain_metrics(trace,gray,units,contract['actual_active_cone_ids'])
        weak[protocol]=dict(mapping_component=cid,active_cones=len(contract['actual_active_cone_ids']),
            input_contract=contract,chain=stats,membrane=membrane_decomposition(trace,gray,cid,snapshots[cid]),
            bright_expected_delta_Hz=float(delta[:,100:200].mean()),dark_expected_delta_Hz=float(delta[:,300:400].mean()),
            peak_abs_expected_delta_Hz=float(abs(delta.mean(axis=0)).max()),
            source_parameter_adjustment_performed=False,upstream_gray_reuse_continuous_identity_verified=True)
        probe_arrays[protocol+'.expected_delta_Hz']=delta
        probe_arrays[protocol+'.voltage_delta_mV']=(trace[cid+'.voltage']-gray[cid+'.voltage'])*1000
    np.savez_compressed(d/'paired_probe_response_arrays.npz',**probe_arrays)
    polarity=dict(center_bright_ON=weak['center20']['bright_expected_delta_Hz']>0,
        center_dark_ON=weak['center20']['dark_expected_delta_Hz']<0,
        center_contrast_ordering=abs(weak['center20']['bright_expected_delta_Hz'])>abs(weak['center05']['bright_expected_delta_Hz']),
        S_bright_positive=weak['S20']['bright_expected_delta_Hz']>0,
        LM_bright_negative=weak['LM20']['bright_expected_delta_Hz']<0)
    for cid,row in decisions.items():
        if cid=='small_bistratified_S_ON':
            passed=polarity['S_bright_positive'] and polarity['LM_bright_negative']
        else:
            sign=1 if cid.endswith('_ON') else -1
            passed=sign*row['bright_expected_delta_Hz']>0 and sign*row['dark_expected_delta_Hz']<0
        row['tested_expected_polarity_pass']=bool(passed)
        if not passed:row['decision']='REJECT_TESTED_FUNCTIONAL_POLARITY'
        row['functional_acceptance_scope']='Relative expected response on tested engineering stimuli only; amplitude not calibrated'
    result=dict(status='COMPLETE_B0_B1_AND_BOUNDED_B2',production_default_changed=False,
        physiology_calibrated=False,matlab_launches=0,git_push_attempts=0,new_neural_seconds=7.4,
        prior_ON_parasol_experiment_reused=True,decisions=decisions,expectation_contract=expectation_checks,
        spatial_function_checks=polarity,weak_response_probes=weak,
        source_supported_gain_adjustment='NOT_PERFORMED: no matching physiological amplitude/gain interval established for current contact-first reference',
        B0=dict(current_route='ContactFirstConeBipolarReceptorMembraneV2 / ContactFirstGenericAmacrineReceptorMembraneV2',
            historical_FIR_recommendations_applied=False,baseline='No FIR baseline field in this route; frozen coefficient reference K/release_half retained',
            downstream_BC_port='activation aliases dynamic release state; not historical FIR softplus a',
            downstream_AC_port='effective voltage sigmoid activation',
            outer_release='q/aI dimensionless native drive; not bounded q and not physical glutamate',
            RGC_history='strict past 1/2/3ms binary taps -12/-8/-4mV; bin-start frozen hazard; spike-independent voltage',
            workbook_materialization='Blocked: current Library helper requires os.setxattr unavailable on Windows; no usable local workbook or field read claimed',
            plan_material='Complete Pro text supplied in delegation; attachment not read'),
        remaining=['B2 source-supported amplitude change and physiological functional acceptance',
            'B2 sufficient recovery and single-cell center-surround acceptance not established',
            'B3 rod/outer/chemical-resource/gap closure checks not newly executed',
            'B4 matched-data local fits not executed; no new A-grade evidence',
            'B5 production default consolidation not performed'])
    save(d/'result.json',result)
    print(json.dumps(dict(status=result['status'],decisions=decisions,spatial_function_checks=polarity),ensure_ascii=False))


if __name__=='__main__':main()
