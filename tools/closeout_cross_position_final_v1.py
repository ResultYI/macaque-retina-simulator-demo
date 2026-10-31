"""Close the limited two-position delivery, preserving prior failures and costs."""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.cross_position_context_v1 import read, write
from retina_sim.group_time_storage import load_bundle
from retina_sim.contracts import canonical_hash
from retina_sim.spatial_group_graph import digest

BASE = ROOT / 'reports/layerwise_physiology/cross_position_functional_coverage_v1_20261029'
PREPARED = BASE / 'restored_resume_v1'
DYNAMIC = PREPARED / 'dynamic_24g_v1'
ANALYSIS = DYNAMIC / 'analysis'
OLD = ROOT / 'reports/layerwise_physiology/RGC_functional_v1_laptop_20261029'


def main():
    result = read(ANALYSIS / 'RESULT.json')
    near = read(ANALYSIS / 'NEAR_TYPE_COMPLETION.json')
    units = read(DYNAMIC / 'source_port_checks/SOURCE_PORT_AND_UNIT_CHECK.json')
    init = read(PREPARED / 'background_24g_v3/INITIALIZATION_CHECK.json')
    seed = load_bundle(PREPARED / 'background_24g_v3/seed', expected_kind='source_construction_seed')
    phases = ['background_24g_v1/initialize_RESOURCE.json', 'background_24g_v2/initialize_RESOURCE.json',
        'short_6ms_v1/RESOURCE.json', 'background_24g_v3/initialize_RESOURCE.json',
        'dynamic_24g_v1/short6/RESOURCE.json', 'dynamic_24g_v1/standard/RESOURCE.json',
        'dynamic_24g_v1/tight/RESOURCE.json', 'dynamic_24g_v1/source_port_checks/RESOURCE.json']
    costs = [dict(path=(PREPARED / p).relative_to(ROOT).as_posix(), **read(PREPARED / p)) for p in phases]
    base_cost = 4309.720314500069
    added = sum(r['wall_seconds'] for r in costs)
    ledger = dict(schema='cross-position-resource-closeout-1', historical_recorded_seconds=base_cost,
        newly_recorded_seconds=added, cumulative_recorded_seconds=base_cost + added, stages=costs,
        simulated_inner_seconds_since24GiB_approval=1.812,
        archival_costs_and_failed_attempts_retained=True, previous_deadline_not_reused=True,
        approved_process_RSS_limit_bytes=24 * 1024**3, minimum_available_memory_guard_bytes=8 * 1024**3,
        maximum_new_process_peak_RSS_bytes=max(r['memory']['process_peak_rss_bytes'] for r in costs),
        dynamic_maximum_peak_RSS_bytes=max(r['memory']['process_peak_rss_bytes'] for r in costs if 'dynamic_24g_v1/' in r['path']),
        report_analysis_cost_accounting='Data-only analysis/plotting not included in model/preparation wall counter',
        old_ledgers_overwritten=False, no_Git_operations=True, OS_settings_changed=False)
    new_id = read(DYNAMIC / 'standard/IDENTITY.json'); old_id = read(OLD / 'IDENTITY.json')
    bin_checks = []; event_rows = []
    for position, directory, identity in [('near', DYNAMIC / 'standard', new_id),
                                          ('reference473', OLD / 'source_coupled_v1/standard', old_id)]:
        with np.load(directory / 'RESPONSE.npz', allow_pickle=False) as z:
            for pop, info in identity['types'].items():
                times = z[pop + '__event_time_seconds']; groups = z[pop + '__event_group_index']
                counts = z[pop + '__sampled_bin_counts']; counted = np.zeros_like(counts)
                bins = np.floor(times / .001).astype(int)
                np.add.at(counted, (groups, bins), 1)
                if not np.array_equal(counted, counts): raise ValueError('Event-bin mapping differs')
                bin_checks.append(dict(position=position, type=pop, passed=True, events=len(times),
                    rule='Native event time belongs to original half-open1ms bin; no interpolation'))
                event_rows.extend(dict(position=position, type=pop, group_index=int(g),
                    group_id=info['group_ids'][int(g)], time_seconds=float(t), bin_index=int(b),
                    independent_fine_cell_event=False) for t, g, b in zip(times, groups, bins))
    snapshots = [load_bundle(DYNAMIC / name / 'INNER_STATE_0900', expected_kind='validation_evidence')['payload']
                 for name in ['standard', 'tight']]
    exact_history = all(np.array_equal(snapshots[0]['spike_states'][p]['history'], snapshots[1]['spike_states'][p]['history']) for p in new_id['types'])
    exact_rng = all(canonical_hash(snapshots[0]['spike_states'][p]['rng_state']) == canonical_hash(snapshots[1]['spike_states'][p]['rng_state']) for p in new_id['types'])
    if not exact_history or not exact_rng or not units['passed']: raise ValueError('Final source/history check failed')
    write(ANALYSIS / 'EVENT_HISTORY_AND_BIN_CHECK.json', dict(rows=bin_checks,
        near_standard_tight_actual_history_bitwise_equal=exact_history,
        near_standard_tight_PCG64_state_bitwise_equal=exact_rng, complete_global_checkpoint=False))
    with (ANALYSIS / 'TWO_POSITION_GROUP_OBJECT_EVENTS.csv').open('x', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(event_rows[0])); w.writeheader(); w.writerows(event_rows)
    result.update(status='LIMITED_TWO_POSITION_FUNCTIONAL_COVERAGE_V1_COMPLETE',
        closed_utc=datetime.now(timezone.utc).isoformat(), scientific_review='NEW_TWO_POSITION_RESULTS_NOT_YET_REVIEWED_BY_ORIGINAL_CHAT',
        source_RHS_and_member_units_verified=True, event_bins_and_final_actual_history_verified=True,
        original_evidence_restored=True, near_initialized_model_id=seed['identity']['initialized_model_id'],
        near_RGC_identity=new_id['RGC_identity'], source_inner_identity=new_id['source_inner'],
        near_center_xy_deg=[-.04510050976293674, -.045844909154906355],
        coordinate_qualification='Patch center; actual per-type group RF eccentricities are separately tabulated',
        graph_qualification='Frozen human-borrowed group geometry; no fine-cell reconstruction or lattice regeneration',
        native_support_radius_deg=.6, inner_support_radius_deg=.3, core_radius_deg=.15,
        retained_Cone_groups=466, retained_HC_groups=1107, H1_groups=771, H2_groups=336,
        retained_Rod_RBC_AII_groups=[515, 83, 366], geometric_core_Rod_RBC_AII_groups=[0, 0, 0],
        internal_gap_nnz_including_diagonal=2780, exterior_gap_nnz=773, new_direct_AII_RGC_edges=0,
        background_bridge_profile_identity_is_new=True, input_coefficients_not_fitted_per_position=True,
        complete_reference_ran_uninterrupted=True, continuation_needed=False,
        blue_screen_cause='NOT_DIAGNOSED', hardware_health_claim=False, resource_ledger='FINAL_RESOURCE_LEDGER.json',
        cumulative_recorded_seconds=ledger['cumulative_recorded_seconds'],
        primary_table=(ANALYSIS / 'TWO_POSITION_FUNCTIONAL_COVERAGE.csv').relative_to(BASE).as_posix(),
        primary_curves=(ANALYSIS / 'two_position_functional_curves.png').relative_to(BASE).as_posix(),
        unique_next_recommendation='Original Chat review of this bounded two-position result and input/boundary limits before defining cross-eccentricity calibration; no continuous0-6deg fit now')
    types = {'midget_RGC_ON': 'midget ON', 'midget_RGC_OFF': 'midget OFF', 'parasol_RGC_ON': 'parasol ON',
             'parasol_RGC_OFF': 'parasol OFF', 'small_bistratified_S_ON': 'S-bistratified'}
    table = '\n'.join(f"| {types[r['type']]} | {r['groups']} / {r['represented_members']} | {r['baseline_V_mV']:.6f} | {r['baseline_source_rate_Hz']:.6f} | {r['common_decrease_deltaV_mV']:+.6f} / {r['common_increase_deltaV_mV']:+.6f} | {r['sampled_object_events']} |" for r in near)
    expected_events = sum(r['expected_object_events'] for r in near)
    report = f'''# 跨位置功能覆盖 v1：有限两点交付

已完成固定近中心 **0.0643102766°** 与现有 **4.73°** 的有限功能覆盖。近中心独立背景、6 ms完整耦合检查、900 ms标准与紧参考均完成；4.73°直接复用已验证原耦合数据。四类ON/OFF midget/parasol的极性、因果、有限动态和数值检查通过。S-bistratified混合输入响应已记录，独立S_ON极性及蓝黄对抗仍未验证；不将其列为已标定S通路。

## 实际结果

| 近中心类型 | 组 / 代表成员 | 背景V (mV) | 背景源率 (Hz) | release×0.9 / ×1.1 的ΔV (mV) | 事件 |
|---|---:|---:|---:|---:|---:|
{table}

near的 **1464** 个事件来自55个采样组对象、900个1ms区间，期望总数 **{expected_events:.6f}**。它不是2173个细成员的独立放电总数，也不是标准/紧参考两次计数相加。两者使用配对相同RNG；所有二值分箱计数、最终实际历史及PCG64状态一致，事件时间与分箱逐组核对通过。4.73°原单次运行35组、1071事件。组数和组成不同，不可直接把两个总数当作偏心效应。

[两位置完整对照表]({result['primary_table']}) · [全部曲线CSV]({(ANALYSIS / 'ALL_POSITION_TYPE_TIME_COURSES.csv').relative_to(BASE).as_posix()}) · [事件CSV]({(ANALYSIS / 'TWO_POSITION_GROUP_OBJECT_EVENTS.csv').relative_to(BASE).as_posix()})

![两点曲线]({result['primary_curves']})

图中电压/率变化相对各自本地背景；gE/gI按代表成员口径报告，原始每组总S/A数据保留在RESPONSE.npz。没有按响应幅度或方差归一化，更没有调gain/beta使结果通过。

## 协议、背景和边界

900ms协议保持一致：50–200ms native Cone release×0.9；350–500ms×1.1；600–750ms Rod输入×2。100us事件网格、1ms BC/RGC历史和原PCG64采样器保留。这是分层探针，900ms阶段持有各自Cone/HC背景，不是全光学到全视网膜的瞬态验证。

近中心固定点独立重建，12,602个内层状态，初始最大RHS **{init['maximum_inner_RHS']:.9g}**，首次BC历史变化为0。没有拿4.73°seed当新平衡。新孔径输入使用已有全局固定参考的同一LMS系数和辐射增益，466组/13,898个Cone成员，native参考等效背景范围3.523141336–5.229042234；4.73°旧输入范围13.644691342–19.502018456。输入身份不同，没有建立两者相同物理背景资格，不能从基线/幅度差异直接估计偏心函数。

中心是原parasol OFF组行4276，坐标[-0.0451005098,-0.0458449092]°，按最小RGC中心半径固定选取，未按响应择点。0.15°核心中各RGC组实际半径约0.06431–0.21291°，不是所有细胞位于同一个0.06431°点。参考组约4.60907–4.85844°。[各类型真实坐标范围]({(ANALYSIS / 'RGC_COORDINATE_COVERAGE.csv').relative_to(BASE).as_posix()})保存逐类型口径。父图声明6°输出范围，祖先更广；未重建lattice或制造缺失细胞。

沿用native0.6°/inner0.3°和原bounded外部合同。保留466 Cone、1107 HC（771 H1/336 H2）、515 Rod、83 RBC、366 AII组。几何核心无Rod/RBC/AII，祖先路径仍存在。原电耦合2780个内部非零项含对角、773个外部非零项及对应固定乘积保留；没有删除双向耦合或新增AII直达RGC。HC的同型耦合仍是原封闭局部代理图，非实测电导。Rod晚期响应近中心更弱，但900ms未证明完整恢复，未作机制归因或拟合。

## 数值与来源资格

标准DOP853为rtol1e-6、atol1e-9、max_step100us；紧参考为1e-10、1e-13、25us。50个读出字段最大标准/紧差异为冻结预算的 **{result['maximum_numerical_error_over_frozen_budget']:.9g}**；全部12,602状态终点最大比例 **{result['full_inner_endpoint_maximum_error_over_budget']:.9g}**。901个原1ms节点完整，无插值回放代替原耦合任务。早期旧1ms线性回放失败保留，不作为本轮资格。

实际源RHS与读出电流分解在五类初始点逐数组相同；分组C/gL仅按成员乘一次。55项RGC参数继承比较全部相同：C60pF/member、gL2nS/member、EL−65mV、EE0、EI−80mV、alpha90Hz、mu−53mV、beta1.67mV、历史[-12,-8,-4]mV及1ms分箱。BC/AC、HC参数和现有路由未重调，未引入偏心参数曲线或Cell6归一化。四类跨类型转移仍为D工程来源；数值一致性不等于生理校准。

背景资格仅限独立固定点、首次事件、实测6ms和刺激前50ms。抑制通过实际负电流及原方程核对，本轮没有额外gI清零轨迹。Sbi背景160.872Hz、混合偏OFF响应和未验证的独立S_ON极性仍是限制；不是蓝黄对抗证据。没有真实PSTH拟合、完整全链checkpoint、连续0–6°覆盖或生物学偏心效应证明。FIT_READY/background_qualified/production_adopted全部false。新两点结果尚未交原Chat审阅。

## 资源与保留记录

用户批准24GiB单进程上限，仍保留8GiB系统余量和进程超时。标准约278.588s，紧参考约815.099s，紧参考整段完成，无续接发生。初始化峰值约11.38GiB，动态约2.56GiB；通过不构成硬件健康证明。重启及旧8GiB保护退出、原件缺失、保存字段错误和来源字段修正记录均保留。未改驱动/BIOS/安全或OS设置；未Git提交/push/reset。

从4309.720314500s承接，本轮批准后新增记账 **{added:.9f}s**，累计 **{ledger['cumulative_recorded_seconds']:.9f}s**（含失败/重复准备，不重置历史）。数据整理和绘图另列为非模型工作。见FINAL_RESOURCE_LEDGER.json。4原件已按旧SHA恢复，守卫未取消。来源修正仅将场景SHA改指真实input.mat，固定LMS数值和捕获未变；旧身份记录保留，新实际运行使用background_24g_v3。

唯一下一建议：将本轮有限两点结果、Sbi和输入/边界限制交原Chat审阅，再定义跨偏心校准。此次交付后停止，不自动启动全域拟合。
'''
    with (BASE / 'FINAL_REPORT.md').open('x', encoding='utf8') as f: f.write(report)
    reproduction = '''# Reproduction from the saved current notebook assets

Run from D:/macaque retinal simulator. Preserve current outputs and choose a fresh output directory. This is replay within this source-bound project, not a standalone portable asset package.

```powershell
$prep = 'reports/layerwise_physiology/cross_position_functional_coverage_v1_20261029/restored_resume_v1'
$new = "$prep/rerun_unique_name"
& '.venv/Scripts/python.exe' -B tools/reproduce_cross_position_v1.py --prepared $prep --output $new --historical-seconds <actual_current_cumulative_seconds>
```

Run sequentially and stop on failure; the current worker retains24GiB/8GiB memory protection and820s internal/900s parent-process limits. Tight reference wall time on this notebook was815s; different machine/load can hit protection. Per100ms files are inner experiment evidence, not a validated full global resumable checkpoint. Do not relabel them as one.

Existing source-bound construction/optics are reused, not regenerated for this replay. To rebuild an independent background from the same retained optical capture, use prepare_cross_position_background_v1.py with --mode initialize --prepared $prep --input-background "$prep/background_v1" --output <fresh reports path> --max-rss-gib 24 and carry the actual historical counter. This retains original source guards and computes a new initialization, never reuses4.73 equilibrium.

Plot saved CSV with tools/plot_cross_position_v1.py --directory <analysis directory> using the existing D:/anaconda/envs/retina_rf_snn/python.exe and an explicit writable MPLCONFIGDIR. No model runs during plotting. The explicit analysis entry was replayed from saved arrays into a fresh scratch directory; all9 output files were byte-identical.
'''
    with (BASE / 'REPRODUCE_FINAL.md').open('x', encoding='utf8') as f: f.write(reproduction)
    sources = ['prepare_cross_position_coverage_v1.py', 'cross_position_context_v1.py',
        'prepare_cross_position_background_v1.py', 'capture_cross_position_background_v1.m',
        'run_cross_position_optical_v1.py', 'run_cross_position_rgc_v1.py', 'check_cross_position_ports_v1.py',
        'analyze_cross_position_v1.py', 'plot_cross_position_v1.py', 'reproduce_cross_position_v1.py',
        'closeout_cross_position_final_v1.py']
    result['implementation_sha256'] = {'tools/' + p: digest(ROOT / 'tools' / p) for p in sources}
    result['actual_near_type_results'] = near
    result['supporting_evidence'] = dict(numerical=(ANALYSIS / 'NUMERICAL_REFERENCE.csv').relative_to(BASE).as_posix(),
        full_state=(ANALYSIS / 'FULL_INNER_ENDPOINT_REFERENCE.csv').relative_to(BASE).as_posix(),
        ports=(DYNAMIC / 'source_port_checks/SOURCE_PORT_AND_UNIT_CHECK.json').relative_to(BASE).as_posix(),
        events=(ANALYSIS / 'EVENT_HISTORY_AND_BIN_CHECK.json').relative_to(BASE).as_posix())
    write(BASE / 'FINAL_RESULT.json', result); write(BASE / 'FINAL_RESOURCE_LEDGER.json', ledger)
    pins = [BASE / 'FINAL_REPORT.md', BASE / 'REPRODUCE_FINAL.md', ANALYSIS / 'RESULT.json',
        ANALYSIS / 'TWO_POSITION_FUNCTIONAL_COVERAGE.csv', ANALYSIS / 'ALL_POSITION_TYPE_TIME_COURSES.csv',
        ANALYSIS / 'two_position_functional_curves.png', ANALYSIS / 'two_position_functional_curves.pdf',
        PREPARED / 'background_24g_v3/seed/manifest.json',
        DYNAMIC / 'standard/RESPONSE.npz', DYNAMIC / 'tight/RESPONSE.npz']
    write(BASE / 'FINAL_DELIVERABLE_SHA256.json', {p.relative_to(ROOT).as_posix(): digest(p) for p in pins})
    for _ in range(4):
        size = sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())
        if ledger.get('current_output_tree_bytes') == size: break
        ledger['current_output_tree_bytes'] = size
        (BASE / 'FINAL_RESOURCE_LEDGER.json').write_text(json.dumps(ledger, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(status=result['status'], near_events=result['actual_sampled_object_events'],
        cumulative_recorded_seconds=ledger['cumulative_recorded_seconds'], output_bytes=ledger['current_output_tree_bytes'],
        numerical_passed=result['numerical_passed'], functional_gaps=result['functional_gaps'])), flush=True)


if __name__ == '__main__': main()
