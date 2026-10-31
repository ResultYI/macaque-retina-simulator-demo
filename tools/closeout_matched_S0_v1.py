"""Chinese matched-S0 morning delivery from finished arrays and analysis only."""
import csv
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.cross_position_context_v1 import read,write

EXPERIMENT=ROOT/'reports/position_S0_far_originaltarget_20261029_resume_v2'
BASE=EXPERIMENT/'dynamic_S0_v1';ANALYSIS=BASE/'analysis'
CROSS=ROOT/'reports/layerwise_physiology/cross_position_functional_coverage_v1_20261029'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_read(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def main():
    result=read(ANALYSIS/'RESULT.json');rows=read(ANALYSIS/'POSITION_TYPE_SUMMARY.json')
    protocol=read(BASE/'PROTOCOL.json');prior=read(EXPERIMENT/'RESOURCE_LEDGER.json')
    resources={name:read(BASE/name/'RESOURCE.json') for name in ['standard','tight']}
    if not all(item['status']=='PASS' and item['last_event_index']==9000 for item in resources.values()):
        raise ValueError('A permitted900ms run did not finish successfully')
    seconds=sum(item['wall_seconds'] for item in resources.values());total=prior['cumulative_seconds']+seconds
    ledger=dict(historical_cumulative_seconds=prior['cumulative_seconds'],new_model_seconds=seconds,cumulative_seconds=total,
        stages=resources,maximum_process_peak_RSS_bytes=max(x['memory']['process_peak_rss_bytes'] for x in resources.values()),
        approved_process_limit_bytes=24*1024**3,system_free_guard_bytes=8*1024**3,
        last_system_available_memory_is_not_full_run_minimum=True,near_rerun=False,
        attempts=dict(standard=1,tight=1),additional_probes_or_fit=False,
        analysis_and_plot_wall_time_not_added_to_model_counter=True,active_MATLAB_workers_at_close=0,
        archived_failures_and_ledgers_preserved=True,closed_utc=datetime.now(timezone.utc).isoformat())
    write(BASE/'RESOURCE_LEDGER.json',ledger)
    names={'midget_RGC_ON':'midget ON','midget_RGC_OFF':'midget OFF','parasol_RGC_ON':'parasol ON',
        'parasol_RGC_OFF':'parasol OFF','small_bistratified_S_ON':'S-bistratified'}
    near={r['type']:r for r in rows if r['position']=='near_S0'};far={r['type']:r for r in rows if r['position']=='far_S0'}
    baselines=[];responses=[]
    for pop,n in near.items():
        f=far[pop]
        baselines.append(f"| {names[pop]} | {n['groups']}/{n['represented_members']} | {f['groups']}/{f['represented_members']} | {n['baseline_V_mV']:.6f} / {f['baseline_V_mV']:.6f} | {n['baseline_rate_Hz']:.6f} / {f['baseline_rate_Hz']:.6f} |")
        label={'CONDITIONAL_FUNCTIONAL_PASS':'条件功能通过','MIXED_INPUT_FUNCTIONAL_S_ON_UNTESTED':'仅混合输入','FUNCTIONAL_GAP_RECORDED':'有功能缺口'}[f['status']]
        responses.append(f"| {names[pop]} | {n['release_down_deltaV_mV']:+.6f} / {f['release_down_deltaV_mV']:+.6f} | {n['release_up_deltaV_mV']:+.6f} / {f['release_up_deltaV_mV']:+.6f} | {n['late_rod_deltaV_mV']:+.6f} / {f['late_rod_deltaV_mV']:+.6f} | {n['sampled_group_object_events']} / {f['sampled_group_object_events']} | {label} |")
    coords=csv_read(ANALYSIS/'RGC_COORDINATES_AND_MEMBERS.csv');coord_lines=[]
    for row in coords:
        coord_lines.append(f"| {'近' if row['position']=='near_S0' else '远'} | {names[row['type']]} | {row['groups']}/{row['represented_members']} | {float(row['group_RF_ecc_min_deg']):.6f}–{float(row['group_RF_ecc_max_deg']):.6f} | {float(row['fine_member_RF_ecc_min_deg']):.6f}–{float(row['fine_member_RF_ecc_max_deg']):.6f} |")
    gaps='无已记录功能缺口（Sbi仅限混合输入功能）' if not result['functional_gaps'] else json.dumps(result['functional_gaps'],ensure_ascii=False)
    numerical='通过' if result['numerical_passed'] else '未通过；保留具体超限项，未改阈值'
    report=f'''# 同一冻结S0规则下的两点功能结果

新远点的一对standard/tight900ms探针均正常完成，近中心复用已有记录，没有重跑。当前结果：**数值一致性{numerical}；{gaps}**。

远点是原manifest目标parasol_RGC_OFF原行25496/父组8163，准确RF中心 **[4.724986235371699,-0.11169236923054729]°**（半径4.7263061803°）；近点补丁中心为[-0.04510050976293674,-0.045844909154906355]°（半径0.0643102766°）。这些是补丁中心，下面另列实际RGC组与成员RF范围。

共同S0身份：`{protocol['common_S0_id']}`。新远点RGC身份：`{result['far_RGC_identity']}`。光谱/QE/gain、固定LMS与孔径profile相同；按同一construction t=0外部状态生成算法，在各位置独立初始化。旧远点t=4.993环带及旧背景结果保留，不参加本次两点曲线比较。

## 原生孔径问题已解决的范围

核生成复用cMosaic.generateApertureKernel，面积复用computeApertureAreasMetersSquared；通过public setter初始化后重新导入逐cone直径，并在结束时恢复导入数据。保留猕猴D(e)、原坐标、0.05μm分区、裁剪后unit-sum及禁止外推；没有声称整个capture已换成原生compute。130-cone均匀/非均匀完整捕获回归逐值一致，相关MATLAB源码未变。近点仍复用旧Gaussian适配产生的缓存，没有冒称完整近点已重跑新前端。详见[精确复用范围](../../position_S0_entry_preparation_20261029_v1/NATIVE_APERTURE_SCOPE.md)。

## 基线与实际动态

V和源率按代表成员数加权，gE/gI的每成员口径和完整时间曲线另存CSV；源率属于现有有效模型，未拟合生理发放。

| 类型 | 近组/成员 | 远组/成员 | 基线V近/远(mV) | 基线源率近/远(Hz) |
|---|---:|---:|---:|---:|
{chr(10).join(baselines)}

原协议不变：50–200ms Cone release×0.9，350–500ms×1.1，600–750ms Rod端口×2；测量窗口分别100–200、400–500、800–900ms。ΔV相对各自本地基线，不做动态gain归一化。

| 类型 | release×0.9 ΔV近/远(mV) | release×1.1 ΔV近/远(mV) | 晚期Rod ΔV近/远(mV) | 事件近/远 | 远点资格 |
|---|---:|---:|---:|---:|---|
{chr(10).join(responses)}

远点晚期Rod后的膜变化比近点明显，OFF两类约-0.0797mV、Sbi约-0.0386mV；近点对应约-0.0060/-0.0082/-0.00038mV。两位置保留的Rod/RBC/AII及核心成员分布不同；此处是同一工程Rod端口探针下的实际局部网络响应，不能把差异单独归因于偏心度或孔径。

Sbi只有当前混合输入功能，选择性S_ON、蓝黄对抗及偏心生理关系均未验证；没有新增色彩或zero-gI实验。抑制检查只用原记录中gI/Ii符号，没有冒称抑制移除因果实验。详细源率幅值、晚期Rod响应、preprobe50ms与逐类检查见[POSITION_TYPE_SUMMARY.csv](analysis/POSITION_TYPE_SUMMARY.csv)。

![两点曲线](analysis/matched_S0_functional_curves.png)

## 真实位置、成员及事件口径

| 位置 | 类型 | 组/成员 | 组RF偏心范围(°) | 冻结成员RF偏心范围(°) |
|---|---|---:|---:|---:|
{chr(10).join(coord_lines)}

成员RF来自既有源表，不是新生成几何或生物记录配准。组动态并未还原为各成员独立轨迹。实际事件为{result['near_object_events']}近点、{result['far_object_events']}远点，来自55与{result['far_groups']}个组对象的900个1ms半开时间箱；没有按成员复制，也没有叠加tight事件。事件/箱映射通过={result['event_contract_passed']}；远点standard/tight实际history及PCG64一致={result['far_history_and_rng_equal']}。

## 数值与资源

standard保留DOP853 rtol1e-6/atol1e-9、maxstep100μs；tight为1e-10/1e-13、25μs。完整901节点、{result['readout_fields']}读出最大误差/既有冻结预算={result['maximum_readout_error_over_frozen_budget']:.9g}；全部{result['far_inner_states']}个inner状态终点最大={result['maximum_inner_endpoint_error_over_frozen_budget']:.9g}。未修改容差或验收阈值。[逐字段数值表](analysis/NUMERICAL_REFERENCE.csv)与[终点表](analysis/FULL_INNER_ENDPOINT_REFERENCE.csv)保留原值。

standard耗时{resources['standard']['wall_seconds']:.3f}s，tight {resources['tight']['wall_seconds']:.3f}s；本轮新增{seconds:.3f}s，累计 **{total:.6f}s**。峰值{ledger['maximum_process_peak_RSS_bytes']/1024**3:.3f}GiB，24GiB进程/8GiB系统余量保护保留；末次系统可用值未冒称全程最小。只运行这一对，当前无活动计算，旧失败及账本不覆盖。

## 限定结论与唯一下步

本次建立的是同一既有S0输入和边界生成规则下，两点独立局部背景上的功能记录；不是猕猴光学/发放的物理标定。内部门控release探针也不是标定后的外部视觉对比度。几何为既有冻结借用/工程源，孔径0.75°以外仍有工程转移；两个不同成员集合的差异不能归结为已测量的偏心函数。FIT_READY、background_qualified、production_adopted全部false。

唯一建议下一步：在这两份背景资产上，预声明同一外部光学对比度刺激并通过Cone/HC真实输入通路验证，使两点比较从内部分层探针进入同视觉刺激端到端功能验证；不拟合或扩大连续全域。本轮不自动启动该新实验、不外发Chat、不Git。
'''
    with (BASE/'FINAL_REPORT_ZH.md').open('x',encoding='utf-8') as stream:stream.write(report)
    result.update(closed_utc=datetime.now(timezone.utc).isoformat(),cumulative_model_seconds=total,
        new_model_seconds=seconds,report='FINAL_REPORT_ZH.md',
        scope='Same frozen S0 rules and local exterior-generation algorithm; no physiological/eccentricity/visual-contrast calibration')
    write(BASE/'FINAL_RESULT.json',result)
    handoff=CROSS/'MORNING_HANDOFF.md';previous=handoff.read_bytes()
    with (BASE/'MORNING_HANDOFF_before_matched_pair.md').open('xb') as stream:stream.write(previous)
    lead=f'''# 最新结果：同一冻结S0下的两点900ms功能验证已完成

原生孔径核/面积复用已修复；新远点按原目标准确坐标独立构建、捕获和初始化后，standard/tight900ms均正常完成。近点未重跑，同谱/QE/gain/固定LMS及t=0边界生成规则保持。[中文最终结果](../../position_S0_far_originaltarget_20261029_resume_v2/dynamic_S0_v1/FINAL_REPORT_ZH.md)包含真实RF范围、成员数、gE/gI/V/率曲线和事件口径。

数值一致性{numerical}，50读出误差/冻结预算最大{result['maximum_readout_error_over_frozen_budget']:.6g}，{result['far_inner_states']}状态终点最大{result['maximum_inner_endpoint_error_over_frozen_budget']:.6g}；{gaps}。Sbi仍只限混合输入，未获选择性色彩资格。近/远事件{result['near_object_events']}/{result['far_object_events']}来自组对象，不是独立fine-cell spike。

本轮模型{seconds:.3f}s，累计{total:.6f}s，峰值{ledger['maximum_process_peak_RSS_bytes']/1024**3:.3f}GiB，24GiB/系统余量保护未改，当前无活动计算。没有新扫参、zero-gI、色彩实验、拟合或外发。三项科学资格仍false。唯一下步建议为共同外部光学对比度经Cone/HC通路的两点功能验证；本轮没有自动启动。

---

'''
    assert handoff.read_bytes()==previous;handoff.write_text(lead+previous.decode('utf-8'),encoding='utf-8',newline='\n')
    sources=['tools/run_cross_position_rgc_v1.py','tools/analyze_matched_S0_positions_v1.py','tools/plot_matched_S0_positions_v1.py','tools/closeout_matched_S0_v1.py']
    write(BASE/'EXECUTION_SOURCES.json',{p:sha(ROOT/p) for p in sources})
    write(BASE/'DELIVERABLE_SHA256.json',{str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.is_file()})
    print(json.dumps(result))


if __name__=='__main__':main()
