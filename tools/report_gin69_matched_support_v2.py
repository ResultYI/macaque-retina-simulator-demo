"""Deliver one frozen 69#6 fit and matched input comparisons, without refitting."""
from pathlib import Path
import csv
import json
import time
import numpy as np
from retina_sim.spatial_group_graph import digest
from tools.durable_research_runtime_v1 import atomic_json

ROOT = Path(r'D:\macaque retinal simulator')
OUT = ROOT / 'reports/GIN_69_6_HC2_BC1_SUPPORT_V2'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def build(output, guard):
    began = time.perf_counter()
    network_folder = OUT / 'readout'
    baseline_folder = OUT / 'matched_spatial_baselines'
    result = read(baseline_folder / 'FIXED_DEVELOPMENT_RESULT.json')
    network_result = read(network_folder / 'FIXED_DEVELOPMENT_RESULT.json')
    network = read(network_folder / 'MODEL_FROZEN.json')
    baseline = read(baseline_folder / 'MODELS_FROZEN.json')
    if digest(network_folder / 'MODEL_FROZEN.json') != network_result['frozen_model_sha256']:
        raise ValueError('Network frozen model changed')
    if digest(baseline_folder / 'MODELS_FROZEN.json') != result['frozen_models_sha256']:
        raise ValueError('Baseline frozen models changed')
    with np.load(network_folder / 'FIXED_DEVELOPMENT_PREDICTIONS.npz', allow_pickle=False) as z:
        observed = z['observed_counts'][:, 30000:45000].mean(0).reshape(-1, 20).mean(1) * 1000
        network_mc = z['MC_batch_probability'][:, 30000:45000].reshape(16, -1, 20).mean(2) * 1000
    with np.load(baseline_folder / 'FIXED_DEVELOPMENT_PREDICTIONS.npz', allow_pickle=False) as z:
        curves = {name: z[name + '_probability'][30000:45000].reshape(-1, 20).mean(1) * 1000
            for name in result['metrics']}
        stimulus_mc = z['S_GLM_MC_batch_probability'][:, 30000:45000].reshape(16, -1, 20).mean(2) * 1000
    se = dict(V_GLM=network_mc.std(0, ddof=1) / 4, S_GLM=stimulus_mc.std(0, ddof=1) / 4)
    times = 30 + (np.arange(750) + .5) * .02
    with (OUT / 'DEVELOPMENT_CURVES_20ms.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_center_s', 'observed_6trial_Hz', *[name + '_free_Hz' for name in curves], 'V_GLM_MC_SE_Hz', 'S_GLM_MC_SE_Hz'])
        writer.writerows(zip(times, observed, *curves.values(), se['V_GLM'], se['S_GLM']))
    columns = ['model', 'parameters', 'free_r', 'free_MSE_Hz2', 'free_mean_Hz', 'observed_mean_Hz', 'conditional_NLL_nat_per1ms']
    sizes = dict(V_GLM=30, constant=1, S_LN=7889, S_GLM=7902)
    with (OUT / 'FIXED_EVALUATION.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, values in result['metrics'].items():
            writer.writerow(dict(model=name, parameters=sizes[name], **{key: values[key] for key in columns[2:]}))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, layout='constrained')
    axes[0].plot(times, observed, color='black', lw=.85, label='Observed: 6 repeats, 20 ms')
    colors = dict(V_GLM='#1767a5', S_LN='#a36100', S_GLM='#11846d', constant='#888888')
    for name, values in curves.items():
        axes[0].plot(times, values, color=colors[name], lw=.8, label=name + ' free')
        if name in se:
            axes[0].fill_between(times, values - 1.96 * se[name], values + 1.96 * se[name], color=colors[name], alpha=.12)
    axes[0].set_ylabel('Event rate (Hz)')
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].set_title('69#6 M-off, 3.88 deg: fixed development [30,45) s\nMatched spatial input; different model capacities; reference meridian and brightness approximation')
    with np.load(OUT / 'formal_neural/PREFIX_45s.npz', allow_pickle=False) as z:
        select = np.arange(300000, 450001, 10)
        axes[1].plot(z['time_source_s'][select], z['V'][select] * 1000, color='#743880', lw=.85)
    axes[1].set_ylabel('Network membrane V (mV)')
    axes[1].set_xlabel('Source time (s)')
    for ax in axes:
        ax.set_xlim(30, 45)
        ax.grid(alpha=.15)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / ('FIXED_DEVELOPMENT.' + extension), dpi=180)
    plt.close(fig)
    guard()
    timing = dict(neural_segment_worker_seconds=sum(read(OUT / 'formal_neural' / f'SEGMENT_{i:03d}.json')['wall_seconds'] for i in range(90)),
        voltage_training_seconds=network['training_wall_seconds'], baseline_training_seconds=baseline['training_wall_seconds'],
        voltage_evaluation_seconds=network_result['wall_seconds'], baseline_evaluation_seconds=result['wall_seconds'])
    rows = []
    for name, item in result['metrics'].items():
        r = 'NA（常率或零调制）' if item['free_r'] is None else f"{item['free_r']:.6g}"
        rows.append(f"| {name} | {sizes[name]} | {r} | {item['free_MSE_Hz2']:.6g} | {item['free_mean_Hz']:.6g} | {item['conditional_NLL_nat_per1ms']:.7g} |")
    comparisons = []
    for name, item in result['paired_V_GLM_vs_baselines'].items():
        for metric, interval in item['intervals'].items():
            conclusion = '不确定' if interval is None or interval[0] <= 0 <= interval[2] else ('网络读出较好' if interval[0] > 0 else '刺激基线较好')
            comparisons.append(f"- 相对 {name}，{metric}：{interval}；{conclusion}。正值统一表示网络读出较好。")
    report = f'''# 69#6 独立读出拟合与固定开发评估

已完成当前新 HC / BC 网络的 45 秒连续前向、69#6 独立 V_GLM 读出拟合及匹配输入 LN / SGLM 对照。最后 [45,60) 秒仍封存，未用于调参、选择或自动评价。完成的是一个真实目标的条件模型，不是全层生理标定。

真实记录为 69#6、M-off、RF 偏心度 3.88°，网络目标为 OFF-parasol group 7730。+X 是已批准参考经线；4 mm 瞳孔和作者时间协议保留；222.2 td 是已批准的一分钟中央加权亮度近似，不能称恢复了原 CRT 绝对标定。HC / BC 的感光 owner 与物理支持域已分开，超原 1.338 mm 部分明确属于工程外推。有限外部输入及 held 边界仍有限制。

训练 [0,30) 秒，固定开发 [30,45) 秒；六重复相同时间 mask。三个 10 秒训练内留块选择及 200 ms history purge，不把重复当作不同刺激。每次只用训练数据标准化；开发结果不用于返调。完整因果状态从零时刻连续推进，不在开发起点重置。源元信息、重复计数及一分钟亮度输入协变量曾在本次冻结前检查；封存不能追溯宣称从未接触。已批准的光度尺度固定自该一分钟输入校准，不按预测窗口动态归一化，也未用封存 spike 学习尺度。

| 模型 | 参数数 | free r | free MSE (Hz²) | free 均值 (Hz) | conditional NLL (nat/1ms) |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

统计为六重复平均的 20 ms PSTH，NLL 为全部重复、全部评分 bin 的平均。conditional 可以用严格过去观测 spike；free GLM 只用自生成 history 的 4096 条路径、16 批期望概率。阴影为 Monte Carlo 均值 ±1.96 SE，不是实测数据置信区间。

V_GLM 使用网络原膜电压及统计时间/history 核；没有膜 reset 或新物理适应。刺激 LN / SGLM 使用同一网络实际收到的全部 493 个空间光输入、16 个时间基，没有另拟空间 RF 或压缩成全场平均。SGLM 与 V_GLM 采用相同 200 ms history 基与链接函数，但参数数不同，因此比较完整管线预测表现，不能单独归因网络结构。

配对 1000 次重复/200 ms 时间块重采样的描述性 95% 区间：

{chr(10).join(comparisons)}

这是一段固定自然视频的开发评估，不等于新刺激泛化。区间未包含 Monte Carlo 预测噪声；该误差另存原始报告。不能用高预测分数证明内部膜参数或连接生理正确。

计时（秒）：{json.dumps(timing, ensure_ascii=False)}。神经 worker 合计含加载、构建、积分、记录和检查点，不是纯积分；训练/评估独立列出。光学、背景、失败和修复记录留在各自执行收据，不将嵌套计时重复相加。

模型：`readout/MODEL_FROZEN.json` 和 `matched_spatial_baselines/MODELS_FROZEN.json`；训练选择与冻结合同位于各自目录。指标：`FIXED_EVALUATION.csv`；曲线：`DEVELOPMENT_CURVES_20ms.csv`；图：`FIXED_DEVELOPMENT.png`、`.pdf`、`.svg`；原始预测、条件损失、MC 批曲线及配对索引均保留。

`FIT_READY=false`、`background_qualified=false`、`production_adopted=false`。本次结束后不自动启动其他细胞或封存测试。下一步应先根据本细胞固定开发表现及其有限边界决定是否推进相同已注册目标流程，不能自动用调整内部参数提高本次开发成绩。
'''
    (OUT / 'FINAL_REPORT.md').write_text(report, encoding='utf8')
    atomic_json(output / 'RESULT.json', dict(status='69_6_FIT_AND_MATCHED_FIXED_DEVELOPMENT_REPORTED',
        independent_readout_fit_complete=True, internal_network_parameters_fitted=False, matched_stimulus_baselines_complete=True,
        report_sha256=digest(OUT / 'FINAL_REPORT.md'), timing=timing, wall_seconds=time.perf_counter() - began,
        sealed_access=False, FIT_READY=False, background_qualified=False, production_adopted=False))
