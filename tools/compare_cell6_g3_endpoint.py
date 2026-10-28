"""Read-only original/endpoint comparison under frozen output theta, no fit."""
from pathlib import Path
import os,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
import numpy as np
from retina_sim.rgc_stage1_cache import read_cache,write
from retina_sim.rgc_stage1_readout import observed_past,objective,free_mean,decode
from retina_sim.group_time_monitor import frozen_budget
from retina_sim.group_time_storage import load_bundle
from tools.fit_rgc_stage1 import verify
OUT=ROOT/'reports/cell6_g3_nonnegative_endpoint_20261028_v1'
AUDIT=ROOT/'reports/rgc_pro_cache_audit_20261028_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat(x):return dict(mean=float(np.mean(x)),std=float(np.std(x)),minimum=float(np.min(x)),maximum=float(np.max(x)),peak_to_peak=float(np.ptp(x)))
def psth(y,p,lo,hi):
    obs=y[:,lo:hi].reshape(len(y),-1,20).sum(axis=(0,2))/(len(y)*.02);pred=p[lo:hi].reshape(-1,20).sum(axis=1)/.02
    return dict(observed_Hz=stat(obs),predicted_Hz=stat(pred),Pearson=None if np.ptp(obs)==0 or np.ptp(pred)==0 else float(np.corrcoef(obs,pred)[0,1]),RMSE_Hz=float(np.sqrt(np.mean((obs-pred)**2))),bin_seconds=.02,lag_seconds=0.,smoothing=False,rescaling=False)
def main():
    if not (OUT/'FORWARD_RESULT.json').exists():raise RuntimeError('Wait for complete endpoint forward; no partial comparison as final')
    c=read(OUT/'CONTRACT.json');spec=verify();case=spec['cases'][3];new=c['case'];theta=np.asarray(c['output_theta'])
    if sha(ROOT/'reports/rgc_stage1_cpu_parallel_full_v1/fit/TRAINING_SELECTION.json')!=c['output_selection_sha256']:raise RuntimeError('Frozen selected output theta changed')
    lo,hi=case['score_bins'];grid=slice(lo*10,hi*10);base=spec['base'];curves={};rows=[];prefix={}
    with np.load(ROOT/'reports/rgc_stage1_complete_v1/actual_observations.npz',allow_pickle=False) as z:y=z[case['counts_key']].copy()
    for label,directory,s,cc in [('original',ROOT/'reports/rgc_stage1_cpu_parallel_full_v1/workers/cell6_g3/cache/cell6_g3',spec,case),('endpoint',OUT/'cache'/new['case_id'],c['spec'],new)]:
        z,m=read_cache(directory,s,cc)
        with z:
            voltage=z[case['population']+':V'][::10,case['model_column']][:-1].copy()
            series={f:z[case['population']+':'+f][grid,case['model_column']].copy() for f in ['V','gE','gI']}
            prefix[label]={f:z[case['population']+':'+f][:10001,case['model_column']].copy() for f in ['V','gE','gI']}
        d=dict(case,voltage=voltage,counts=y,past=observed_past(y));p=free_mean(voltage,decode(theta,base),spec['initial_probability'])
        rows.append(dict(arm=label,cache_identity=m['cache_identity'],manifest_sha256=sha(directory/'manifest.json'),initial_match=m['initial_match'],
            statistics_100us={k:stat(v) for k,v in series.items()},fixed_current_theta_NLL_per_bin=float(objective(theta,[d],base)[0]),
            fixed_initial_theta_NLL_per_bin=float(objective(np.zeros(6),[d],base)[0]),free_PSTH=psth(y,p,lo,hi),raw_absolute_monitor=m['raw_absolute_monitor'],
            physical_domain_failures=m['raw_absolute_monitor'].get('domain_failures'),stimulus_identity=c['case']['stimulus_sha256'],output_parameters_refitted=False))
        curves[label+'_free_probability']=p;curves[label+'_V']=series['V'];curves[label+'_gE']=series['gE'];curves[label+'_gI']=series['gI']
    metadata=read(OUT/'ACTUAL_WEIGHTED_PORTS.json');old=np.load(AUDIT/'ACTUAL_ROUTE_TRACES_V2.npz',allow_pickle=False);newtrace=np.load(OUT/'ACTUAL_WEIGHTED_PORTS.npz',allow_pickle=False)
    mapping={'cones|E':'native_E_pool','cones|Vis':'cone_Vis_pool','cones|Bh':'cone_Bh_pool','cones|Vbar':'cone_Vbar_pool','cones|release':'cone_release_pool','HC|Vh':'HC_Vh_pool',
        'diffuse_BC_ON|drive':'BC_ON_drive_pool','diffuse_BC_ON|generator_activation':'BC_ON_activation_pool','diffuse_BC_ON|voltage':'BC_ON_V_pool','diffuse_BC_ON|effective_release':'BC_ON_effective_pool',
        'diffuse_BC_OFF|drive':'BC_OFF_drive_pool','diffuse_BC_OFF|generator_activation':'BC_OFF_activation_pool','diffuse_BC_OFF|voltage':'BC_OFF_V_pool','diffuse_BC_OFF|effective_release':'BC_OFF_effective_pool',
        'cross_OFF_ON_parasol_AC_v1|gE':'AC_gE','cross_OFF_ON_parasol_AC_v1|voltage':'AC_V','cross_OFF_ON_parasol_AC_v1|p':'AC_p','cross_OFF_ON_parasol_AC_v1|s':'AC_s','parasol_RGC_ON|voltage':'RGC_V','parasol_RGC_ON|gE':'RGC_gE','parasol_RGC_ON|gI':'RGC_gI'}
    weighted=[]
    for j,column in enumerate(metadata['columns']):
        key='|'.join(column);name=mapping.get(key)
        if name is None:continue
        before=stat(old[name]);after=stat(newtrace['values'][grid,j]);weighted.append(dict(field=name,old=before,endpoint=after,peak_to_peak_ratio=None if before['peak_to_peak']==0 else after['peak_to_peak']/before['peak_to_peak']))
    oldcontrast=old['input_fraction'].copy()
    product=ROOT/'reports/rgc_stage1_cpu_parallel_full_v1/workers/cell6_g3/aligned/attempt_0006';m=read(product/'manifest.json');frames=np.load(product/m['shared_point_index']['input_frame']['path'],mmap_mode='r')[grid].copy()
    drive=load_bundle(ROOT/new['drive_bundle'],expected_kind='validation_evidence')['payload'];seed=load_bundle(ROOT/spec['prepared_model']/'initial_seed',expected_kind='source_construction_seed')['payload'];B=np.asarray(seed['input_reference']['native_background_td'])
    fractions={k:float((v/B)[0]-1.) for k,v in drive['native_inputs'].items()};newcontrast=np.array([fractions[int(k)] for k in frames])
    inputstats={label:dict(contrast=stat(x),contrast_RMS=float(np.sqrt(np.mean(x*x))),light_factor=stat(1+x),SD_over_mean_light=float(x.std()/(1+x).mean())) for label,x in [('original',oldcontrast),('endpoint',newcontrast)]}
    old.close();newtrace.close()
    result=dict(status='COMPLETE_SINGLE_TRAINING_G3_ENGINEERING_ENDPOINT_DIAGNOSTIC',arms=rows,weighted_actual_parent_comparison=weighted,
        contract_sha256=sha(OUT/'CONTRACT.json'),theta=theta.tolist(),same_source_raw_waveform=True,changed_input_kappa_only=True,
        baseline_audit_source_sha256=sha(AUDIT/'ACTUAL_ROUTE_AUDIT_V2.json'),score_window_s=[1.,7.],input_score_100us_ZOH_statistics=inputstats,
        first1s_unchanged_gray_target_prefix_exact={f:bool(np.array_equal(prefix['original'][f],prefix['endpoint'][f])) for f in ['V','gE','gI']},selected_by_validation=False,optimizer_called=False,
        inference_scope='One maximal nonnegative affine input coefficient under existing frozen conditional model. Neither physical light calibration, successful fit, nor an upper bound on all intermediate nonlinear responses.',
        unresolved=['raw optical zero/gain/negative-light acquisition encoding','absolute stimulus acquisition delay','Cell6 anatomical registration and exact eccentricity','engineering internal parameter physiology','full-run dynamic numerical-defect receipt and global/reservoir background qualification'],
        further_runs_authorized=False,default_adopted=False,FIT_READY=False)
    if (OUT/'COMPARISON.json').exists():raise RuntimeError('Preserve existing comparison; do not overwrite')
    write(OUT/'COMPARISON.json',result);np.savez_compressed(OUT/'COMPARISON_CURVES.npz',source_time_s=np.arange(lo*10,hi*10)*.0001,probability_bin_start_s=np.arange(7100)*.001,**curves)
    a,b=rows
    lines=['# 单 g3 非负输入端点诊断', '',f"仅改变输入系数0.05→{c['kappa']!r}；一次真实7.1s CPU前向，无优化。",'',
        f"评分窗实际150Hz ZOH输入：调制标准差{inputstats['original']['contrast']['std']:.9g}→{inputstats['endpoint']['contrast']['std']:.9g}，SD/mean(light)={inputstats['original']['SD_over_mean_light']:.9g}→{inputstats['endpoint']['SD_over_mean_light']:.9g}；这仍是工程Td输入，不是实测光标定。",'',
        '| 指标 | 原始 g3 | 端点 g3 |','|---|---:|---:|']
    for name,key,factor in [('RGC V峰峰值(μV)','V',1e6),('RGC gE峰峰值(pS)','gE',1e12),('RGC gI峰峰值(pS)','gI',1e12)]:lines.append(f"|{name}|{a['statistics_100us'][key]['peak_to_peak']*factor:.9g}|{b['statistics_100us'][key]['peak_to_peak']*factor:.9g}|")
    for name,get in [('固定theta条件NLL',lambda x:x['fixed_current_theta_NLL_per_bin']),('20ms自由PSTH相关',lambda x:x['free_PSTH']['Pearson']),('20ms预测率标准差(Hz)',lambda x:x['free_PSTH']['predicted_Hz']['std'])]:lines.append(f'|{name}|{get(a):.10g}|{get(b):.10g}|')
    lines+=['','模型状态域、初态、原absolute监测结果和完整加权链路统计见COMPARISON.json；保留原false。','',f"前向墙时：{read(OUT/'FORWARD_RESULT.json')['wall_seconds']:.3f}s。",'',result['inference_scope'],'','本次到此停止；不自动拟合、扫参、采用端点或宣称第一阶段成功。']
    (OUT/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(dict(status=result['status'],result=str(OUT/'RESULT.md'),V_pp_uV=[r['statistics_100us']['V']['peak_to_peak']*1e6 for r in rows],PSTH_r=[r['free_PSTH']['Pearson'] for r in rows])))
if __name__=='__main__':main()
