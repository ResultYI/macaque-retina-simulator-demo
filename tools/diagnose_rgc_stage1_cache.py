"""Read-only training-cache diagnostics; never build or optimize a retina."""
from pathlib import Path
import hashlib,json,math,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']})
import numpy as np
from tools.fit_rgc_stage1 import verify
from retina_sim.rgc_stage1_readout import objective,decode,free_mean,observed_past,bounds,NAMES
from retina_sim.functional_fit_math import MembraneParameters,teacher_forced_nll,psth_metrics,_free_probability,STATES,NEXT0,NEXT1
from retina_sim.group_time_storage import load_bundle
from tools.fit_rgc_stage1_cpu_registry import report_plain

OUT=ROOT/'reports/rgc_stage1_diagnosis_v1';PHASE=ROOT/'reports/rgc_stage1_cpu_parallel_full_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):
    a=np.asarray(a);return dict(min=float(a.min()),max=float(a.max()),mean=float(a.mean()),std=float(a.std()),peak_to_peak=float(np.ptp(a)))

def main():
    started=time.perf_counter();registration=read(OUT/'PRE_REGISTERED.json');spec=verify();selection=read(PHASE/'fit/TRAINING_SELECTION.json')
    assert sha(PHASE/'fit/TRAINING_SELECTION.json')==registration['frozen_selection_sha256']
    theta=np.array(selection['theta']);parameters=decode(theta,spec['base']);registry=read(PHASE/'CACHE_REGISTRY.json');paths={r['case_id']:Path(r['directory']) for r in registry['caches']}
    params=MembraneParameters.from_dict(dict(parameters,history_kernel_V=parameters['history_kernel']))
    predictions=ROOT/'reports/cell6_replay/predictions';export=read(predictions/'manifest.json');assert sha(predictions/export['payload'])==export['payload_sha256']
    cases=[];rows=[]
    with np.load(ROOT/'reports/rgc_stage1_complete_v1/actual_observations.npz',allow_pickle=False) as targets,np.load(ROOT/'reports/layer_calibration/rgc_cbem_conditional_v1/frozen_preparation_v3/inputs.npz',allow_pickle=False) as source,np.load(predictions/export['payload'],allow_pickle=False) as exported:
        for case in spec['cases']:
            if case['split']!='train':continue
            index=int(case['case_id'].split('_g')[1]);raw=source['g'+str(index)+'_stimulus_values'];counts=targets[case['counts_key']]
            assert hashlib.sha256(raw.tobytes()).hexdigest()==case['stimulus_sha256'] and np.array_equal(counts,source['g'+str(index)+'_spike_counts'])
            with np.load(paths[case['case_id']]/'endogenous.npz',allow_pickle=False) as data:
                assert np.array_equal(data['time_source_s'],np.arange(71001)*.0001)
                voltage=data[case['population']+':V'][::10,case['model_column']][:-1].copy();direct=data['expected_rate_base_Hz'][:,case['model_column']].copy()
            c=dict(case,voltage=voltage,counts=counts,past=observed_past(counts));cases.append(c);lo,hi=case['score_bins'];idx=np.arange(lo,hi)
            baseline=free_mean(voltage,spec['base'],spec['initial_probability'])/.001
            original_norm=float(np.max(abs(baseline-direct)/(8192*np.finfo(float).eps*np.maximum(abs(direct),1))))
            fitted=free_mean(voltage,parameters,spec['initial_probability'])
            independent,mass=_free_probability(voltage,parameters['alpha'],parameters['mu'],parameters['beta'],np.asarray(parameters['history_kernel']),np.asarray(spec['initial_probability']),STATES,NEXT0,NEXT1)
            history=np.einsum('rti,i->rt',c['past'],parameters['history_kernel'])
            ref=teacher_forced_nll(voltage,counts,history,idx,params)/counts[:,lo:hi].size
            loss,_=objective(theta,[c],spec['base']);constant=dict(c,voltage=np.full_like(voltage,voltage[lo:hi].mean()));constant_loss,_=objective(theta,[constant],spec['base'])
            pm,_=psth_metrics(counts,fitted,idx)
            driver=load_bundle(ROOT/case['drive_bundle'],expected_kind='validation_evidence')['payload'];first=next(iter(driver['native_inputs']));base=np.asarray(driver['native_inputs'][first]);errors=[];factors=[]
            for frame,value in driver['native_inputs'].items():
                j=max(0,math.floor((frame/150.-driver['model_origin_s'])/.0001));factor=1+.05*(float(raw[j])-float(raw[0]));factors.append(factor)
                if not np.array_equal(value,base*factor):errors.append(frame)
            rows.append(dict(case_id=case['case_id'],target_stimulus_and_counts_exact=True,source_dt_s=driver['source_dt_s'],spike_bin_s=.001,clock_origin_s=driver['model_origin_s'],input_fps=150,
                raw_stimulus=stats(raw),input_relative_factor=stats(factors),saved_input_frame_mismatch_count=len(errors),first_bad_frames=errors[:3],
                original_native_base_replay_max_roundoff_norm=original_norm,original_native_base_replay_passed=original_norm<=1,
                independent_teacher_NLL_abs_difference=abs(ref-loss),independent_free_probability_max_abs_difference=float(np.max(abs(independent-fitted))),independent_probability_mass_error=float(mass),
                fitted_export_bitwise_equal=np.array_equal(fitted/.001,exported[case['case_id']+'_rate_Hz']),selected_NLL=loss,constant_V_same_theta_NLL=constant_loss,
                NLL_loss_change_removing_all_voltage_modulation=constant_loss-loss,voltage_V=stats(voltage[lo:hi]),free_expected_Hz=stats(fitted[lo:hi]/.001),
                original20ms_PSTH_metrics={k:pm[k] for k in ['PSTH_Pearson','observed_rate_Hz','predicted_rate_Hz','observed_modulation_std_Hz','predicted_modulation_std_Hz','PSTH_RMSE_Hz']},
                fixed_history_hazard_max_relative_change_bound=float(np.expm1(np.ptp(voltage[lo:hi])/parameters['beta']))))
    gradient=[];zero=np.zeros(6);loss,analytic=objective(zero,cases,spec['base']);step=registration['gradient_check']['central_difference_step']
    for j,name in enumerate(NAMES):
        x=zero.copy();x[j]+=step;right=objective(x,cases,spec['base'])[0];x[j]-=2*step;left=objective(x,cases,spec['base'])[0];finite=(right-left)/(2*step)
        gradient.append(dict(parameter=name,analytic=float(analytic[j]),central_difference=float(finite),absolute_error=abs(float(analytic[j])-finite)))
    manifest=read(ROOT/'reports/rgc_stage1_aligned_v1/CASE_G3_COMPLETION.json')['manifest'];product=Path(manifest).parent;m=read(manifest)
    target=spec['target']['global_id'];vf=next(f for f in m['signals'] if f['key']==['parasol_RGC_ON','voltage']);center=np.asarray(vf['centroid_mm'])[vf['group_ids'].index(target)]
    wanted={'cones':['E','Vis','Vbar','release'],'HC':['Vh'],'diffuse_BC_ON':['drive','voltage','effective_release'],'diffuse_BC_OFF':['drive','voltage','effective_release'],'cross_OFF_ON_parasol_AC_v1':['gE','voltage','s'],'parasol_RGC_ON':['gE','gI','voltage']}
    layers=[]
    for pop,names in wanted.items():
        for name in names:
            field=next(f for f in m['signals'] if f['key']==[pop,name] and f['kind']=='ports');centers=np.asarray(field['centroid_mm']);col=int(np.argmin(np.linalg.norm(centers-center,axis=1)))
            if pop=='parasol_RGC_ON':col=field['group_ids'].index(target)
            values=np.load(product/field['values'],mmap_mode='r');series=values[10000:70000:10,col].copy();del values
            layers.append(dict(population=pop,field=name,unit=field['unit'],source_group_id=field['group_ids'][col],column=col,selection='nearest target RF centroid from fixed geometry; target exact ID',sampled_clock_s=.001,score_window=[1.,7.],stats=stats(series)))
    result=dict(status='TRAINING_CACHE_DIAGNOSIS_COMPLETE_NO_FIT',registration=str(OUT/'PRE_REGISTERED.json'),scientific_contract_sha256=sha(ROOT/'reports/rgc_stage1_complete_v1/CONTRACT.json'),training_selection_sha256=sha(PHASE/'fit/TRAINING_SELECTION.json'),source51_pins_verified=True,
        trained_parameters=parameters,original_coordinate_bounds=bounds(spec['base']),selected_coordinates=theta,training_diagnostics=rows,gradient_check_at_original_baseline=gradient,train_g3_aligned_layers=layers,
        physical_input_unit=None,physical_illuminance_mapping=None,absolute_acquisition_latency_s=None,spike_index_origin=1,spike_index_origin_verified=False,free_time_shift_fitted=False,
        production_source_modified=False,model_or_ODE_run=False,optimizer_run=False,validation_used_to_select_parameters=False,historical_test_waveform_loaded=False,wall_seconds=time.perf_counter()-started)
    path=OUT/'DIAGNOSTICS.json';assert not path.exists();path.write_text(json.dumps(report_plain(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(path=str(path),wall_seconds=result['wall_seconds'],max_gradient_abs_error=max(v['absolute_error'] for v in gradient),native_replay_all_passed=all(r['original_native_base_replay_passed'] for r in rows),input_frames_all_exact=all(r['saved_input_frame_mismatch_count']==0 for r in rows),exports_all_exact=all(r['fitted_export_bitwise_equal'] for r in rows))))

if __name__=='__main__':main()
