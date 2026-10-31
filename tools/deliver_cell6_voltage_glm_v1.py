"""Plot and report the saved voltage-readout diagnostic; no fitting or prediction."""
from pathlib import Path
from datetime import datetime,timezone
import csv
import hashlib
import json
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(r'D:\macaque retinal simulator')


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def interval(value):return 'NA' if value is None else f'[{value[0]:.6g}, {value[2]:.6g}]'


def deliver(out):
    complete=read(out/'COMPUTATION_COMPLETE.json');frozen=read(out/'MODELS_FROZEN.json')
    frozen_hash=sha(out/'MODELS_FROZEN.json')
    for name,expected in frozen['source_pins'].items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Pinned source changed: '+name)
    rows=read(out/'ALL_METRICS.json');results={g:read(out/f'g{g}_RESULT.json') for g in range(6)}
    if complete['groups']!=6 or any(r['models_frozen_sha256']!=frozen_hash for r in results.values()):
        raise RuntimeError('Incomplete frozen evaluation')
    with (out/'COMPARISON.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True,sharey=True)
    for group,axis in zip((3,4,5),axes):
        with np.load(out/f'g{group}_PREDICTIONS.npz',allow_pickle=False) as z:
            times=z['time_bin_start_seconds'].reshape(-1,20).mean(axis=1)
            observed=z['observed_counts'].mean(axis=0).reshape(-1,20).mean(axis=1)*1000
            axis.plot(times,observed,color='.7',lw=.8,label='Observed3-repeat PSTH')
            for name,color in [('T','#bd7728'),('V_GLM','#245a91'),('S_GLM','#a43b8b')]:
                rate=z[name+'_probability'].reshape(-1,20).mean(axis=1)*1000
                axis.plot(times,rate,color=color,lw=1,label=name)
                if name=='V_GLM':
                    se=z['V_GLM_20ms_rate_SE_Hz'];axis.fill_between(times[50:350],rate[50:350]-2*se,rate[50:350]+2*se,color=color,alpha=.15,label='V_GLM +/-2 MC SE')
        axis.axvspan(0,1,color='.9');axis.axvspan(7,7.1,color='.9');axis.grid(alpha=.2)
        axis.set_ylabel(f'g{group} rate (Hz)');axis.set_ylim(bottom=0)
    axes[0].legend(ncol=5,fontsize=8);axes[-1].set_xlabel('Source time (s); gray outside scoring window')
    fig.suptitle('Cell6 development diagnostic: original T / voltage GLM / stimulus GLM; same readout method')
    fig.tight_layout(rect=(0,0,1,.96))
    for ext in ('png','svg'):fig.savefig(out/('DEVELOPMENT_OVERLAY.'+ext),dpi=180)
    plt.close(fig)
    lines=['# Cell6：缓存电压匹配GLM读出开发诊断','',
        '只训练一个新候选V_GLM：原目标8166的缓存膜电压直接进入与刺激S_GLM相同的因果滤波和200 ms历史GLM。原T和S_GLM自由预测全部复用；无神经重跑、无参考模型重新拟合。','',
        '目标为sg2:61468d3761f46cda:parasol_RGC_ON:8166，含2个原成员，是代表组电压而非单细胞独立实测。只使用V字段（伏特），没有使用gE/gI、多层状态或直接刺激特征，也没有叠加旧单指数时间状态。','',
        'V_GLM和S_GLM均为30参数：截距、16个500 ms因果滤波基函数、13个200 ms严格过去spike历史基函数。链接为90×softplus，1 ms Bernoulli事件似然；不是指数Poisson GLM。相同四档ridge、整组训练刺激留一选择、单次初始化、L-BFGS-B、训练内标准化。完整定义在PROTOCOL_FROZEN.json；逐项匹配在METHOD_MATCHING.json。','',
        '电压取实际100 μs节点V[:-1]，同样在1 ms bin起点采样滤波结果；pre0电压固定为自己的初值V0，pre0 spike为零。恒定V0是背景相容的读出边界约定，不是观测到的采集前轨迹。完整0–1 s预热，没有在1 s重置状态。','',
        '仅g0–g2训练和选择，模型冻结后才载入本轮g3–g5电压并各固定评估一次。free采用4096条自生成spike历史、16批×256、与S_GLM相同的固定PCG64种子，无实测开发历史。conditional NLL只使用严格过去实测spike。评分[1,7) s、20 ms PSTH及原重采样索引均不变。','',
        '|组|模型|conditional NLL nat/1 ms|free r|MSE Hz²|均值 Hz|SD Hz|',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['group']<3:continue
        lines.append(f"|g{row['group']}|{row['model']}|{row['conditional_NLL_nat_per1ms']:.6f}|{row['free_r']:.5f}|{row['free_MSE_Hz2']:.2f}|{row['mean_Hz']:.2f}|{row['SD_Hz']:.2f}|")
    lines+=['','差距追回比例仅为描述性点估计：(V−T)/(S−T)用于r，(T−V)/(T−S)用于MSE/NLL。负值表示比原T更差，超过100%表示点估计超过刺激GLM；不确定性须看后续配对CI，不能把CI跨零表述成等效。','',
        '|组|r差距追回|MSE差距追回|NLL差距追回|','|---|---:|---:|---:|']
    for group in (3,4,5):
        recovery=results[group]['gap_recovery']
        values=[recovery[k]['recovered_fraction'] for k in ('free_r','free_MSE_Hz2','conditional_NLL_nat_per1ms')]
        lines.append(f'|g{group}|'+ '|'.join('NA' if v is None else f'{100*v:.1f}%' for v in values)+'|')
    lines+=['','配对CI沿用1000次原重复/200 ms区块索引；正值支持V_GLM（ΔNLL/MSE=参考−V，Δr=V−参考）。这是固定刺激内的描述性不确定性；3次重复不是3个独立刺激。区间条件于已固定MC曲线，MC误差另列。','',
        '|组|V_GLM对照|ΔNLL 95% CI|ΔMSE 95% CI|Δr 95% CI|','|---|---|---|---|---|']
    for group in (3,4,5):
        for name,c in results[group]['paired_V_vs_reference'].items():
            lines.append(f"|g{group}|{name}|{interval(c['delta_NLL_95'])}|{interval(c['delta_MSE_95'])}|{interval(c['delta_r_95'])}|")
    lines+=['','逐指标结论（uncertain表示区间跨零，不表示两者等效）：']
    for group in (3,4,5):
        for name,c in results[group]['paired_V_vs_reference'].items():
            lines.append(f"- g{group} V_GLM对{name}："+json.dumps(c['metric_conclusions'],ensure_ascii=False))
        for name in ('V_GLM','S_GLM'):
            mc=results[group][name+'_Monte_Carlo']
            lines.append(f"- g{group} {name}：20 ms平均/最大率MC SE={mc['mean_20ms_rate_SE_Hz']:.4f}/{mc['maximum_20ms_rate_SE_Hz']:.4f} Hz；r MC SE={mc['free_r_MC_SE']:.6f}，MSE MC SE={mc['free_MSE_Hz2_MC_SE']:.4f} Hz²。")
    fit=frozen['models']['V_GLM']
    lines+=['',f"新候选参数量={fit['parameters']}，最终ridge={fit['ridge']}，收敛={fit['converged']}；最终训练墙时={fit['wall_seconds']:.4f} s（不含12次预定CV）。整轮计算={complete['wall_seconds']:.3f} s，固定预测/评估={complete['evaluation_wall_seconds']:.3f} s，峰值RSS={complete['peak_RSS_bytes']/1024**2:.1f} MiB。每组预测时间在gN_RESULT.json。",'',
        '这次同时替换了滤波与history，只能归因于读出整体差异；已有LN更高r也说明不能把全部差距归于history。若差距缩小，不证明内部生理正确；若仍落后，不构成不可逆信息丢失的数学证明。',
        '', 'g3–g5已经多次用于开发，不是新的独立验证。原test封存，原g5 INCONCLUSIVE不变；FIT_READY/background_qualified/production_adopted均为false。没有追加特征、延迟、超参范围、消融或其他候选。',
        '', '交付：COMPARISON.csv、PAIRED_COMPARISONS.json、GAP_RECOVERY.json、MODELS_FROZEN.json、CROSS_VALIDATION.json、PROTOCOL_FROZEN.json、DEVELOPMENT_OVERLAY.png/svg、gN_RESULT/PREDICTIONS。']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    write(out/'PAIRED_COMPARISONS.json',{str(g):results[g]['paired_V_vs_reference'] for g in (3,4,5)})
    write(out/'GAP_RECOVERY.json',{str(g):results[g]['gap_recovery'] for g in (3,4,5)})
    archive=out/'source_archive';archive.mkdir()
    for relative in ('tools/run_cell6_voltage_glm_v1.py','tools/deliver_cell6_voltage_glm_v1.py',
        'retina_sim/stimulus_baselines_v1.py','tools/cell6_baseline_metrics_v1.py'):
        shutil.copy2(ROOT/relative,archive/Path(relative).name)
    write(out/'DELIVERY_SOURCE_IDENTITY.json',{p.name:sha(p) for p in archive.iterdir()})
    write(out/'DELIVERY.json',dict(saved_arrays_only=True,new_fits=0,new_predictions=0,new_evaluations=0,
        interpreter=sys.executable,utc=datetime.now(timezone.utc).isoformat()))
    write(out/'COMPLETE.json',dict(complete,model_freeze_sha256=frozen_hash,original_results_preserved=True))


if __name__=='__main__':deliver(Path(sys.argv[1]))
