"""Analyze saved recovery and actual one-cell probes; no neural work or fit."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_rgc_type_candidate import BASE,REFERENCE,RGCS,load_arrays,save
from tools.analyze_pro_rgc_batch import saved_model
from retina_sim.rgc_expectation import exact_history_rates


def join_native(left,right):
    if abs(float(left['time'][-1])-float(right['time'][0]))>1e-12:raise ValueError('Continuation has a gap or overlap')
    joined={'time':np.concatenate((left['time'],right['time'][1:]))}
    for k,v in left.items():
        if k=='time' or k not in right or v.ndim!=2:continue
        if not np.array_equal(v[:,-1],right[k][:,0]):raise ValueError('Returned native endpoint does not roundtrip: '+k)
        joined[k]=np.concatenate((v,right[k][:,1:]),axis=1)
    if not np.allclose(np.diff(joined['time']),.001,rtol=0,atol=1e-12):raise ValueError('Native 1ms coverage lost')
    return joined


def analyze_rod_pair(d):
    for branch in ('rod_flash','rod_gray'):
        if json.loads((d/branch/'result.json').read_bytes())['status']!='COMPLETE_UNCHANGED_ROD_CHAIN_PROBE':
            raise ValueError('Completed rod pair required')
    stim=load_arrays(d/'rod_flash/trace/traces.npz');gray=load_arrays(d/'rod_gray/trace/traces.npz')
    t=stim['time']-stim['time'][0];rows={};delta_arrays={'rod_time_seconds':t}
    expected_signs={'rod.current':1,'rod_membrane.voltage':-1,'rod_membrane.release':-1,
        'individual_rod_RBC_contacts.gate':1,'rod_BC.voltage':1,'rod_BC.event_rate':1,
        'AII_AC.voltage':1,'AII_AC.event_rate':1,'RBC_AII_AMPA.conductance':1,
        'AII_FMB_glycine.conductance':1,'AII_DB1_glycine.conductance':1,
        'midget_BC_ON.voltage':1,'midget_BC_OFF.voltage':-1}
    # A fixed engineering response window, not a fitted latency or amplitude.
    response=(t>=.05-1e-12)&(t<.25-1e-12)
    for key,sign in expected_signs.items():
        delta=stim[key]-gray[key];np.testing.assert_allclose(delta[:,t<.05-1e-12],0.,rtol=0,atol=1e-12)
        mean=delta.mean(axis=0);j=int(np.argmax(abs(mean)));value=float(mean[response].mean())
        rows[key]=dict(mean_response_delta_native_unit=value,peak_signed_population_delta=float(mean[j]),
            peak_relative_seconds=float(t[j]),expected_direction=sign,mean_direction_pass=bool(sign*value>0),
            peak_resolved_inside_observation=bool(j<len(t)-2),
            unit=json.loads((d/'rod_flash/trace/segment.json').read_bytes())['output_units'][key]['unit'])
        delta_arrays['rod.'+key+'.delta']=delta
    snapshots=json.loads((REFERENCE/'actual_RGC_parameter_snapshots.json').read_bytes())['candidate']
    aux=json.loads((d/'recover_gray/final_checkpoint.json').read_bytes())['payload']['spike_states']
    rgcs={}
    for cid in RGCS:
        n=saved_model(snapshots[cid]);rs=exact_history_rates(n,stim[cid+'.voltage'],stim['time'],aux[cid]);rg=exact_history_rates(n,gray[cid+'.voltage'],gray['time'],aux[cid])
        delta=rs['expected_binary_count_rate_Hz']-rg['expected_binary_count_rate_Hz']
        rgcs[cid]=dict(mean_response_expected_delta_Hz=float(delta[:,response[:-1]].mean()),
            peak_abs_population_expected_delta_Hz=float(abs(delta.mean(axis=0)).max()),
            output_mapping='ON-parasol candidate' if cid=='parasol_RGC_ON' else 'retained old type mapping')
        delta_arrays['rod.'+cid+'.expected_delta_Hz']=delta
    return dict(status='COMPLETE_ACTUAL_PAIRED_ROD_POLARITY_PROBE',neural_seconds=.6,
        input='5 to6R*/s for100ms after50ms gray; total300ms; cones held; original rod reference retained',
        channels=rows,all_declared_mean_directions_pass=all(r['mean_direction_pass'] for r in rows.values()),
        peak_and_recovery_scope='300ms bounded polarity check; boundary maxima are not complete response-peak/latency estimates',
        RGC=rgcs,claim='Actual coupled numerical relative response, no physiological amplitude or spike-count significance claim'),delta_arrays


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,default=BASE/'pro_b2_b3_v1');a=p.parse_args();d=a.case
    for branch in ('recover_stimulus','recover_gray','cell_center','cell_annulus'):
        if json.loads((d/branch/'result.json').read_bytes())['status']!='COMPLETE_UNCHANGED_MODEL_FOLLOWUP':raise ValueError('Completed branch required: '+branch)
    aux=json.loads((REFERENCE/'candidate/prepared_checkpoint.json').read_bytes())['payload']['spike_states']
    snapshots=json.loads((REFERENCE/'actual_RGC_parameter_snapshots.json').read_bytes())['candidate']
    prior_stim=load_arrays(BASE/'pro_b0_b2_v1/spatial/full20/stimulus/traces.npz')
    prior_gray=load_arrays(REFERENCE/'candidate/matched_gray/traces.npz')
    stim=join_native(prior_stim,load_arrays(d/'recover_stimulus/trace/traces.npz'))
    gray=join_native(prior_gray,load_arrays(d/'recover_gray/trace/traces.npz'))
    recovery={};arrays={}
    for cid in RGCS:
        n=saved_model(snapshots[cid]);rs=exact_history_rates(n,stim[cid+'.voltage'],stim['time'],aux[cid])
        rg=exact_history_rates(n,gray[cid+'.voltage'],gray['time'],aux[cid])
        delta=rs['expected_binary_count_rate_Hz']-rg['expected_binary_count_rate_Hz']
        dv=1000*(stim[cid+'.voltage']-gray[cid+'.voltage'])[:,:-1]
        peak=float(abs(delta).max());vpeak=float(abs(dv).max());late=float(abs(delta[:,-100:]).max());vlate=float(abs(dv[:,-100:]).max())
        # Declared engineering diagnostic only: <=5% of peak OR the existing
        # 0.01Hz numerical-resolution example, whichever is larger. Report raw
        # values and do not infer physiological recovery from this threshold.
        limit=max(.01,.05*peak);vlimit=max(1e-5,.05*vpeak)
        recovery[cid]=dict(peak_abs_cell_expected_delta_Hz=peak,last100ms_max_abs_cell_expected_delta_Hz=late,
            engineering_rate_limit_Hz=limit,engineering_rate_recovery_pass=bool(late<=limit),
            peak_abs_cell_delta_V_mV=vpeak,last100ms_max_abs_cell_delta_V_mV=vlate,
            engineering_voltage_limit_mV=vlimit,engineering_voltage_recovery_pass=bool(vlate<=vlimit),
            last100ms_population_mean_delta_Hz=float(delta[:,-100:].mean()),
            actually_advanced_output_mapping='ON-parasol candidate' if cid=='parasol_RGC_ON' else 'retained old type mapping',
            scope='Exact expectation conditional only on original prepared first draw; Markov history not reset at continuation')
        arrays[cid+'.recovery_expected_delta_Hz']=delta;arrays[cid+'.recovery_voltage_delta_mV']=dv
    cells={}
    gray_expected=load_arrays(REFERENCE/'candidate/matched_gray/exact_expected_count_rates.npz')
    for branch in ('cell_center','cell_annulus'):
        geom=json.loads((d/branch/'single_cell_geometry.json').read_bytes());idx=geom['focus_cell_index'];cid=geom['focus_component']
        trace=load_arrays(d/branch/'trace/traces.npz');rates=load_arrays(d/branch/'trace/four_rate_expectations.npz')
        delta=rates[cid+'.expected_binary_count_rate_Hz']-gray_expected[cid]
        dv=(trace[cid+'.voltage']-prior_gray[cid+'.voltage'])*1000
        cells[branch]=dict(geometry=geom,focus_bright_expected_delta_Hz=float(delta[idx,100:200].mean()),
            focus_dark_expected_delta_Hz=float(delta[idx,300:400].mean()),
            focus_peak_abs_expected_delta_Hz=float(abs(delta[idx]).max()),
            focus_bright_delta_V_mV=float(dv[idx,100:200].mean()),
            all_cells_bright_delta_Hz=delta[:,100:200].mean(axis=1).tolist())
        arrays[branch+'.expected_delta_Hz']=delta;arrays[branch+'.voltage_delta_mV']=dv
    assert cells['cell_center']['geometry']['focus_cell_id']==cells['cell_annulus']['geometry']['focus_cell_id']
    ant=bool(cells['cell_annulus']['focus_bright_expected_delta_Hz']<0 and cells['cell_annulus']['focus_dark_expected_delta_Hz']>0)
    result=dict(status='COMPLETE_B2_RECOVERY_SINGLE_CELL_AND_B3_NUMERICAL_CHECKS',new_neural_seconds=2.,matlab_launches=0,
        model_parameters_changed=False,production_defaults_changed=False,physiological_calibration=False,
        recovery_after_dark_end_seconds=.6,recovery=recovery,single_cell=cells,
        engineering_center_polarity_pass=bool(cells['cell_center']['focus_bright_expected_delta_Hz']>0 and cells['cell_center']['focus_dark_expected_delta_Hz']<0),
        antagonistic_surround_demonstrated=ant,
        surround_decision='Relative antagonism seen on tested mask only' if ant else 'No antagonistic ring response in this exact cell/footprint; do not change wiring or support without a separate proposal',
        B3=json.loads((d/'background_closure/result.json').read_bytes()),
        small_bistratified_workpoint=json.loads((d/'diagnostics/small_bistratified_workpoint.json').read_bytes()),
        B4=json.loads((d/'diagnostics/B4_data_readiness.json').read_bytes()),
        remaining_decisions=[
            'Rod fixed dark anchor vs current per-background -40mV reference: explicit DC/reference change; proposal saved but not applied',
            'Single-cell surround physiology/geometry and amplitude acceptance; no spatial or wiring changes authorized'],
        ON_parasol_engineering_functional_readiness=dict(mapping=True,bright_dark=True,center_contrast_ordering=True,
            bounded_recovery=True,single_cell_center_surround=True,physiological_amplitude_calibrated=False,
            scope='Eligible for engineering-adoption review within tested input/background/bin/history domain'),
        B5='Not executed this batch: tested ON-parasol relative function passes; default integration and other candidate types need their own declared acceptance')
    if (d/'rod_flash/trace/traces.npz').exists() and (d/'rod_gray/trace/traces.npz').exists():
        result['rod_functional_probe'],rod_arrays=analyze_rod_pair(d);arrays.update(rod_arrays);result['new_neural_seconds']=2.6
    np.savez_compressed(d/'followup_response_arrays.npz',**arrays)
    save(d/'result.json',result)
    print(json.dumps(dict(status=result['status'],ON_parasol_recovery=recovery['parasol_RGC_ON'],
        single_cell={k:{q:v for q,v in row.items() if q!='geometry'} for k,row in cells.items()},
        antagonistic_surround_demonstrated=ant,B3=result['B3']['status'],B4=result['B4']['status'],
        rod_probe=result.get('rod_functional_probe',{}))),flush=True)


if __name__=='__main__':main()
