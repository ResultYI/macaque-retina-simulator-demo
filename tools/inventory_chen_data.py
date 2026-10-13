"""Inventory verified real MAT; no invented model inputs or error metrics."""
from pathlib import Path
import csv,datetime,hashlib,json
import numpy as np
from scipy.io import whosmat,loadmat
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'data/source/chen2024_dryad_v3_325663'
MAT_SHA='6724e2edd8e8708456f32900a14efae88192a5ba597365046d0031eb543d1331'
README_SHA='f25cb72716327f5fc08cfe26cc423bc3364f8f331900dd1972eeb664c1911d03'
def file_sha(p): return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def inspect(v):
    if hasattr(v,'_fieldnames'): return {'field_names':v._fieldnames,'fields':{k:inspect(getattr(v,k)) for k in v._fieldnames}}
    if not isinstance(v,np.ndarray): raise TypeError('Unexpected MATLAB type; no semantic guess')
    out={'shape':list(v.shape),'dtype':str(v.dtype),'unit_field_present':False}
    if v.dtype==object: out['elements']=[inspect(x) for x in v.flat]
    elif np.issubdtype(v.dtype,np.number): out.update(finite=bool(np.isfinite(v).all()),minimum=float(v.min()),maximum=float(v.max()),numeric_bytes_sha256=hashlib.sha256(v.tobytes(order='C')).hexdigest())
    return out
def intervals(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return [{'start_matlab_sample_1based':int(a+1),'end_matlab_sample_1based_inclusive':int(b),'start_relative_index_seconds':float(a*.0001),'end_relative_interval_seconds':float(b*.0001)} for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
def build():
    p=FOLDER/'Figure2-SourceData1.mat';readme=FOLDER/'README.md'
    if p.stat().st_size!=41490462 or file_sha(p)!=MAT_SHA or file_sha(readme)!=README_SHA: raise ValueError('Real source identity mismatch; no data consumption')
    text=readme.read_text(encoding='utf-8')
    assert 'Stimulus units are R*/sec' in text and 'The time step is 0.1 ms' in text
    archive=json.loads((ROOT/'reports/chen2024_archive_inventory.json').read_text(encoding='utf-8'))
    raw=loadmat(p,struct_as_record=False,squeeze_me=False);simple=loadmat(p,simplify_cells=True)
    units={'source':'Actual README cone paragraphs; paper Methods; historical kernel','stimulus':'R*/s','response':'dimensionless published voltage-clamp current normalized by estimated dark current','response_is_pA':False,'sampling_seconds':.0001,'recording_type':'average of 4-5 repeats; actual per-cell number unknown','raw_trials_present':False,'time_vector_present':False,'relative_index_time_convention':'Plot grid t=(MATLAB sample-1)*dt; not established experimental onset or kernel alignment','stimulus_response_phase_alignment':'not explicitly documented','pre_stimulus_history':'missing','absolute_dark_current_per_cell':'missing; cannot recover real pA or per-cell GD; does not alone prevent fixed-model dimensionless normalization','per_cell_species':'missing','per_cell_eccentricity':'missing','per_cell_L_or_M':'missing','biological_cell_ids':'missing; array ordinals only','independent_holdout':'missing; fitting-source data, not independent validation'}
    out={'dataset_doi':'10.5061/dryad.q2bvq83vg','version':3,'version_id':325663,'data_usage':'author_parameterization_source; NOT independent external validation','archive_inventory':'reports/chen2024_archive_inventory.json','source_archive_sha256':archive['archive_sha256'],'files':[{'name':'Figure2-SourceData1.mat','path':p.relative_to(ROOT).as_posix(),'expected_bytes':41490462,'actual_bytes':p.stat().st_size,'published_sha256':MAT_SHA,'actual_sha256':file_sha(p),'exists':True},{'name':'README.md','path':readme.relative_to(ROOT).as_posix(),'expected_bytes':1357,'actual_bytes':readme.stat().st_size,'published_sha256':README_SHA,'actual_sha256':file_sha(readme),'exists':True}],'observed_mat_variables':[{'name':n,'shape':list(s),'matlab_class':c} for n,s,c in whosmat(p)],'observed_cells_and_groups':{k:inspect(v) for k,v in raw.items() if not k.startswith('__')},'semantics':units,'identity_and_structure':'VERIFIED','status':'BLOCKED_INPUT_SEMANTICS_AND_HISTORY','source_data_reproduction':'NOT_RUN','blocker':'Negative nominal light rates remain unresolved; no published recovery/phase/prehistory rule established. Peaks >=50000 are diagnostic biological-applicability warnings, not an author reproduction rejection reason. No clipping, background offset, time shift or invented initial state.'}
    cells=[];arrays={};template=None
    for index,cell in enumerate(simple['PrimateConeResponses'],1):
        if set(cell)!={'AdaptingStepStim','AdaptingStepResponse'}: raise ValueError('Unexpected real fields')
        stim=np.asarray(cell['AdaptingStepStim']);resp=np.asarray(cell['AdaptingStepResponse'])
        if stim.shape!=(80000,) or resp.shape!=stim.shape: raise ValueError('Unexpected real shape')
        negative=stim<0;high=stim>=50000
        row={'data_cell_label':f'PrimateConeResponses{{{index}}}','biological_cell_id':None,'cell_array_index_matlab_1based':index,'stimulus_field':'AdaptingStepStim','response_field':'AdaptingStepResponse','samples':int(stim.size),'sampling_seconds':.0001,'span_between_first_last_samples_seconds':(stim.size-1)*.0001,'nominal_interval_duration_seconds':stim.size*.0001,'stimulus_min_Rstar_per_s':float(stim.min()),'stimulus_max_Rstar_per_s':float(stim.max()),'stimulus_negative_samples':int(negative.sum()),'stimulus_gte50000_samples':int(high.sum()),'stimulus_negative_fraction':float(negative.mean()),'stimulus_gte50000_fraction':float(high.mean()),'response_min_normalized':float(resp.min()),'response_max_normalized':float(resp.max()),'response_mean_normalized':float(resp.mean()),'response_std_normalized_population':float(resp.std()),'stimulus_equal_to_cell1_verified_by_values':True if template is None else bool(np.array_equal(template,stim)),'model_comparison_status':'BLOCKED_INPUT_SEMANTICS_AND_HISTORY','model_ran':False,'model_rmse_normalized':None,'model_mae_normalized':None,'model_max_abs_error_normalized':None,'model_variance_explained':None,'model_pearson_r':None,'fitted_parameters':False,'raw_trial_noise_ceiling':None,'evidence_tag':'published_source_repeat_mean'}
        cells.append(row);arrays[f'cell_{index}_stimulus_Rstar_per_s']=stim;arrays[f'cell_{index}_response_normalized']=resp
        if template is None: template=stim
    arrays['relative_index_seconds']=np.arange(80000)*.0001
    arrays['evidence_tag']=np.array('published_source_repeat_mean');arrays['time_meaning']=np.array(units['relative_index_time_convention'])
    destination=ROOT/'reports/chen2024_real_data';destination.mkdir(exist_ok=True)
    np.savez_compressed(destination/'full_published_traces.npz',**arrays)
    stats={'dataset_sha256':MAT_SHA,'analysis_created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'data_usage':out['data_usage'],'cell_count':6,'cells':cells,'stimulus_anomaly_intervals_from_cell1_verified_common_values':{'negative':intervals(template<0),'gte50000':intervals(template>=50000)},'source_model_parameters':'Unchanged; fixed source parameters; not used on invalid real input','comparison_status':out['status'],'model_run_count':0,'error_metrics_status':'NOT_COMPUTED; null is not zero','plot_selection_rule':'All six cells, complete 80000 samples plus common first 0.5s index-time window; no selection based on fit','independent_physiological_validation':'NOT_RUN','full_traces':'reports/chen2024_real_data/full_published_traces.npz','no_preprocessing':'Exact published arrays; no clipping, offsets, smoothing, time shift, fit or state invention'}
    (destination/'all_cell_metrics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (destination/'all_cell_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(cells[0]));writer.writeheader();writer.writerows(cells)
    previous=ROOT/'reports/chen2024_data_inventory.json';historical=ROOT/'reports/chen2024_data_inventory_before_manual_zip.json'
    if previous.exists() and not historical.exists(): historical.write_bytes(previous.read_bytes())
    previous.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'identity':'VERIFIED','cells':6,'status':out['status'],'negative_samples_per_cell':cells[0]['stimulus_negative_samples'],'gte50000_per_cell':cells[0]['stimulus_gte50000_samples'],'model_run_count':0}))
    return out,stats
if __name__=='__main__': build()
