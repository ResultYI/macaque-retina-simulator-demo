"""Save interpretation of existing scores and update only the three completed registry rows."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import csv,hashlib,json,math,shutil,sys
ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.durable_research_runtime_v1 import atomic_json
OUT=ROOT/'reports/WP1_LATIMER_CELL2_4_V1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if read(OUT/'COMPLETE.json')['completed_cells']!=[2,3,4]:raise RuntimeError('Not complete')
    interpretation={}
    for cell in (2,3,4):
        target=OUT/'targets'/f'Latimer_Cell_{cell}';groups=read(target/'COMPLETE.json')['development_groups']
        results={g:read(target/f'g{g}_RESULT.json') for g in groups}
        interpretation[str(cell)]=dict(existing_within_stimulus_95_interval_conclusion_counts={baseline:{
            key:dict(Counter(r['paired_V_vs_baseline'][baseline]['metric_conclusions'][key] for r in results.values()))
            for key in ('delta_NLL_95','delta_MSE_95','delta_r_95')} for baseline in ('LN','GLM')},
            repetitions_by_development_group={str(g):r['repeats'] for g,r in results.items()},
            pooled_independent_stimulus_test_performed=False,new_prediction_acceptance_threshold_invented=False)
    atomic_json(OUT/'DEVELOPMENT_INTERPRETATION.json',interpretation)
    report=OUT/'FINAL_REPORT.md';original=report.read_text(encoding='utf-8')
    extra='''
## 结果含义与未通过/未定项

Cell2/4的网络V_GLM相较LN，在各自5个开发刺激的原描述性区间中，NLL/MSE/相关均偏向V_GLM。相较刺激S_GLM，Cell2的5组全部仍属不确定；Cell4的NLL有3/5组偏向V_GLM，另2组不确定，MSE与相关各仅1/5组偏向V_GLM，其余不确定。不能概括成网络全面超过刺激GLM。

Cell3结果混合：相较S_GLM，6/9组NLL区间偏向S_GLM、2/9偏向V_GLM、1组不确定；相关有7/9组偏向S_GLM。g9/g10的V_GLM自由PSTH标准差仅7.02/6.25Hz，而实测分别74.92/71.16Hz。即使这些组相关达到0.780/0.750，也未追回真实调制幅度。没有根据该结果返调。

上述区间是各固定刺激内的既有trial/时间块重采样，不是跨细胞独立泛化试验；不把重复当作新刺激样本。Cell3有单次记录，不能宣称重复试次可靠性。没有预先规定这三细胞统一的预测达标阈值，因此这里只登记完整拟合/评估与优劣，不创造预测PASS。物理/有限性/时间连续及恢复门已通过；全时程全状态tight未做，原生损坏根因未确定，真实位置及生理标定仍未完成。
'''
    if '## 结果含义与未通过/未定项' in original:raise FileExistsError('Interpretation retained')
    report.write_text(original+extra,encoding='utf-8')
    registry=ROOT/'reports/RGC_DATASET_TARGET_MATRIX_V1';jp=registry/'TARGET_MATRIX.json';cp=registry/'TARGET_MATRIX.csv'
    before={p.name:sha(p) for p in (jp,cp)};matrix=read(jp);backup=OUT/'delivery/TARGET_MATRIX_BEFORE_UPDATE'
    backup.mkdir(parents=True,exist_ok=False)
    for path in (jp,cp):shutil.copy2(path,backup/path.name)
    updates={}
    for row in matrix['rows']:
        if row['dataset']!='Latimer2019_Figshare9636854' or row['cell_id'] not in ('Cell_2','Cell_3','Cell_4'):continue
        cell=int(row['cell_id'][-1]);target=OUT/'targets'/f'Latimer_Cell_{cell}';identity=read(target/'SOURCE_IDENTITY.json')
        if row['raw_sha256']!=identity['raw_sha256'] or row['eccentricity_deg'] is not None:raise RuntimeError('Registry source identity differs')
        row.update(fit_status='CONDITIONAL_PROXY_READOUT_COMPLETE',neural_owner=identity['target'],owner_members=2,
            model_proxy_eccentricity_deg=math.hypot(*identity['proxy_coordinate_deg']),
            result_directory=str(target.relative_to(ROOT)),physiological_calibration=False,FIT_READY=False)
        updates[row['cell_id']]=row
    if set(updates)!=set(('Cell_2','Cell_3','Cell_4')):raise RuntimeError('Exact three registry targets required')
    matrix['latest_conditional_proxy_completion']=dict(utc=datetime.now(timezone.utc).isoformat(),cells=[2,3,4],
        experiment=str(OUT.relative_to(ROOT)),all_listed_RGC_targets_complete=False)
    with cp.open(encoding='utf-8-sig',newline='') as stream:reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    for row in rows:
        if row['dataset']=='Latimer2019_Figshare9636854' and row['cell_id'] in updates:
            update=updates[row['cell_id']]
            for key in ('fit_status','neural_owner','owner_members'):row[key]=update[key]
    if any(sha(p)!=before[p.name] for p in (jp,cp)):raise RuntimeError('Concurrent registry update; retained backup, no overwrite')
    atomic_json(jp,matrix)
    temp=cp.with_name('TARGET_MATRIX.completed_cell234.tmp')
    with temp.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temp.replace(cp)
    atomic_json(OUT/'delivery/REGISTRY_UPDATE.json',dict(before_sha256=before,after_sha256={p.name:sha(p) for p in (jp,cp)},
        only_completed_source_cells=[2,3,4],other_target_states_unchanged=True,source_eccentricity_kept_null=True))
    print(json.dumps(dict(delivered_cells=[2,3,4],registry_updated=True,new_simulation_or_fit=False)))


if __name__=='__main__':main()
