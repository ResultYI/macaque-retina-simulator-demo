"""Render saved baseline results only; never fit or simulate here."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(r'D:\macaque retinal simulator')


def deliver(out):
    rows=json.loads((out/'ALL_METRICS.json').read_text(encoding='utf8'))
    frozen=json.loads((out/'MODELS_FROZEN.json').read_text(encoding='utf8'))
    with np.load(out/'BASIS.npz',allow_pickle=False) as saved:
        stimulus=saved['stimulus_native'];history=saved['history_ms']
    with (out/'COMPARISON.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    colors=dict(T='#245a91',LN='#15845f',GLM='#a43b8b',H='#bd7728',constant='#555555')
    fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True,sharey=True)
    comparisons={}
    for g,ax in zip((3,4,5),axes):
        result=json.loads((out/f'g{g}_RESULT.json').read_text(encoding='utf8'))
        comparisons[str(g)]=result['paired_T_vs_baseline']
        with np.load(out/f'g{g}_PREDICTIONS.npz',allow_pickle=False) as z:
            time=z['time_bin_start_seconds'].reshape(-1,20).mean(axis=1)
            obs=z['observed_counts'].mean(axis=0).reshape(-1,20).mean(axis=1)*1000
            ax.plot(time,obs,color='.7',lw=.8,label='Observed3-repeat PSTH')
            for name in ('T','LN','GLM','H','constant'):
                value=z[name+'_probability'].reshape(-1,20).mean(axis=1)*1000
                ax.plot(time,value,color=colors[name],lw=1 if name in ('T','LN','GLM') else .6,
                    ls='-' if name in ('T','LN','GLM') else '--',label=name)
            se=z['GLM_20ms_rate_SE_Hz'];v=z['GLM_probability'][1000:7000].reshape(-1,20).mean(axis=1)*1000
            ax.fill_between(time[50:350],v-2*se,v+2*se,color=colors['GLM'],alpha=.15,label='GLM +/-2 MC SE')
        ax.axvspan(0,1,color='.9');ax.axvspan(7,7.1,color='.9');ax.grid(alpha=.2)
        ax.set_ylabel(f'g{g} rate (Hz)');ax.set_ylim(bottom=0)
    axes[0].legend(ncol=4,fontsize=8);axes[-1].set_xlabel('Original source time (s)')
    fig.suptitle('Frozen Cell6 network T vs raw-stimulus LN / softplus GLM;20ms PSTH')
    fig.tight_layout(rect=(0,0,1,.96))
    for ext in ('png','svg'):fig.savefig(out/('DEVELOPMENT_OVERLAY.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,m in frozen['models'].items():
        theta=np.asarray(m['theta']);scale=np.asarray(m['scale'])
        kernel=stimulus@(theta[1:17]/scale[:16])
        axes[0].plot(np.arange(5000)*.1,kernel,label=name)
        if name=='GLM':axes[1].plot(np.arange(1,201),history@(theta[17:]/scale[16:]),label='GLM')
    axes[0].set(xlabel='Stimulus lag (ms)',ylabel='Delta linear predictor / raw stimulus unit')
    axes[1].set(xlabel='Past spike lag (ms)',ylabel='Delta linear predictor / spike')
    for ax in axes:ax.axhline(0,color='.7',lw=.6);ax.grid(alpha=.2);ax.legend()
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(out/('FILTERS.'+ext),dpi=180)
    plt.close(fig)
    (out/'PAIRED_COMPARISONS.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# Cell6：冻结网络与刺激驱动LN/GLM比较','',
        '本轮没有运行视网膜网络，也未修改原网络或原g5 INCONCLUSIVE结论。仅g0–g2训练，参数冻结后g3–g5各完成一次本轮评价；历史test未访问。',
        '', '**重要输入限制：LN/GLM使用完整原始100μs刺激，原网络T此前使用150Hz零阶保持。** 同一Cell6观测、划分、1ms评分和20ms PSTH，但这比较的是完整预测管线，不是单独控制输入后的网络结构贡献。没有为匹配旧网络而丢弃基线刺激信息。',
        '', '两基线均为softplus强度、1ms Bernoulli事件概率，非指数链接Poisson GLM。LN有500ms/16刺激基函数和截距，共17参数；GLM另加200ms/13自身历史基函数，共30参数。参考T仅本轮时间读出7参数，神经网络内部大量参数被冻结，不能将7作为全网络模型复杂度。',
        '', '正则在g0–g2按整个刺激留一组交叉验证，重复不跨组泄漏；四个预声明候选，每折仅使用折内训练统计，最终标准化仅由g0–g2估计。初态使用已知灰刺激、零pre0 spike并从0秒预热，未在1秒重置。',
        '', 'GLM conditional NLL使用严格过去实测spike；free曲线使用固定4096条自生成历史轨迹的期望估计，不注入实测历史。T/H自由曲线原样复用。LN没有历史信息，因此T与LN的conditional NLL差也包含历史信息集的差异。',
        '', '|组|模型|conditional NLL (nat/1ms)|free r|free MSE (Hz²)|均值Hz|SD Hz|',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['group']<3:continue
        correlation='NA' if row['free_r'] is None else f"{row['free_r']:.5f}"
        lines.append(f"|g{row['group']}|{row['model']}|{row['conditional_NLL_nat_per1ms']:.6f}|{correlation}|{row['free_MSE_Hz2']:.2f}|{row['mean_Hz']:.2f}|{row['SD_Hz']:.2f}|")
    lines += ['', '配对差值定义为baseline−T的NLL/MSE、T−baseline的r，正值支持T。区间复用原1000个重复/200ms块重采样索引，仅为固定刺激内的描述性不确定性；3次重复不是3条独立刺激。没有把区间跨零称为统计等效。',
        '', '|组|基线|ΔNLL 95%|ΔMSE 95% (Hz²)|Δr 95%|', '|---|---|---|---|---|']
    for g,comparisons_g in comparisons.items():
        for name,c in comparisons_g.items():
            fmt=lambda value:'NA' if value is None else '['+', '.join(f'{x:.5g}' for x in (value[0],value[2]))+']'
            lines.append(f"|g{g}|{name}|{fmt(c['delta_NLL_95'])}|{fmt(c['delta_MSE_95'])}|{fmt(c['delta_r_95'])}|")
    lines += ['', '按每一指标区间的符号报告T胜/败/不确定，不将混合结果合成肯定的网络增益结论：']
    for g,comparisons_g in comparisons.items():
        for name,c in comparisons_g.items():lines.append(f"- g{g} 对{name}：{json.dumps(c['metric_conclusions'],ensure_ascii=False)}")
    lines += ['', 'GLM Monte Carlo误差单独列出；上述观测重采样区间条件于固定的MC预测曲线，未把MC误差伪装为零，也未按结果增加采样：']
    for g in (3,4,5):
        mc=json.loads((out/f'g{g}_RESULT.json').read_text())['GLM_Monte_Carlo']
        lines.append(f"- g{g}：20ms率平均SE={mc['mean_20ms_rate_SE_Hz']:.4g}Hz，最大SE={mc['maximum_20ms_rate_SE_Hz']:.4g}Hz；r的MC SE={mc['free_r_MC_SE']}；MSE的MC SE={mc['free_MSE_Hz2_MC_SE']}。")
    lines += ['', '训练复杂度与耗时（最终训练不含CV；完整CV记录见相应JSON）：']
    for name,m in frozen['models'].items():lines.append(f"- {name}：{m['parameters']}参数，ridge={m['ridge']}，收敛={m['converged']}，最终训练墙时={m['wall_seconds']:.4g}秒。")
    lines += ['', '原网络g3是已触碰内部检查、g4/g5是历史开发数据，本轮均不是新未触碰测试。即使某一管线胜出，也不等于证明生理机制、空间结构或新拓扑贡献。FIT_READY/background_qualified/production_adopted均保持false。',
        '', '一般方法来源：[Paninski2004](https://doi.org/10.1088/0954-898X/15/4/002)、[Pillow et al.2008](https://doi.org/10.1038/nature07140)。本轮支持长度与基函数数为预声明工程选择，不冒称复现论文的具体设置或群体耦合模型。',
        '', '产物：COMPARISON.csv、PAIRED_COMPARISONS.json、MODELS_FROZEN.json、PROTOCOL_FROZEN.json、DEVELOPMENT_OVERLAY.png/svg、FILTERS.png/svg。']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    archive=out/'source_archive';archive.mkdir()
    for relative in ('retina_sim/stimulus_baselines_v1.py','tools/cell6_baseline_metrics_v1.py',
        'tools/run_cell6_stimulus_baselines_v1.py','tools/deliver_cell6_stimulus_baselines_v1.py'):
        shutil.copy2(ROOT/relative,archive/Path(relative).name)
    (out/'DELIVERY_SOURCE_IDENTITY.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.iterdir()},indent=2),encoding='utf8')
