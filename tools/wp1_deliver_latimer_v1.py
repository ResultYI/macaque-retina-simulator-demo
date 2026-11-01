"""Render one independently fitted real-cell target using saved arrays only."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(r'D:\macaque retinal simulator')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def deliver(out):
    finished=read(out/'COMPUTATION_COMPLETE.json');models=read(out/'MODELS_FROZEN.json');rows=read(out/'ALL_METRICS.json')
    for path,expected in models['source_pins'].items():
        if sha(ROOT/path)!=expected:raise RuntimeError('Frozen source changed: '+path)
    with (out/'COMPARISON.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True,sharey=True)
    pairs={}
    for g,axis in zip((3,4,5),axes):
        data=read(out/f'g{g}_RESULT.json');pairs[str(g)]=data['paired_V_vs_baseline']
        with np.load(out/f'g{g}_PREDICTIONS.npz',allow_pickle=False) as z:
            time=z['time_bin_start_seconds'].reshape(-1,20).mean(axis=1)
            observed=z['observed_counts'].mean(axis=0).reshape(-1,20).mean(axis=1)*1000
            axis.plot(time,observed,color='.7',lw=.8,label='Observed3-repeat PSTH')
            for name,color in [('V_GLM','#245a91'),('S_GLM','#a43b8b'),('LN','#15845f'),('constant','#555555')]:
                axis.plot(time,z[name+'_probability'].reshape(-1,20).mean(axis=1)*1000,label=name,color=color,lw=1)
        axis.axvspan(0,1,color='.9');axis.axvspan(7,7.1,color='.9');axis.set_ylabel(f'g{g} rate Hz');axis.grid(alpha=.2);axis.set_ylim(bottom=0)
    axes[0].legend(ncol=5,fontsize=8);axes[-1].set_xlabel('Source time s; scoring window[1,7)')
    fig.suptitle(f"Latimer real Cell{finished['cell']}: independent readout / shared engineering proxy, actual eccentricity unknown")
    fig.tight_layout(rect=(0,0,1,.96))
    for ext in ('png','svg'):fig.savefig(out/('DEVELOPMENT_OVERLAY.'+ext),dpi=180)
    plt.close(fig)
    text=[f"# Latimer Cell{finished['cell']}：独立条件性读出完成",'',
        '真实ON-parasol实验记录；真实偏心度和方位未知。使用模型8166、2成员、约4.738°作为明确登记的条件代理，不能计入真实偏心度覆盖。',
        '', '六条完整光输入均与既有Cell6缓存逐字节匹配，复用同一冻结模型/背景/初态/时钟的电压轨迹；没有从另一偏心网络借用V，也没有重新运行上游。不同生物细胞分别训练其参数和标准化，不广播Cell6参数。',
        '', '沿用原六刺激开发子集：g0–g2训练，冻结后g3–g5各评价一次，18个真实重复。其他四刺激/12试次保持隔离，没有把未使用数据计为完成，也没有打开原Cell6 test观测。此结果不是独立跨刺激总体泛化的证明。',
        '', '主候选V_GLM：原始缓存V直接经500 ms/16基函数因果FIR，加200 ms/13基函数自生成历史，共30参数。S_GLM同结构输入为完全匹配的150 Hz保持刺激；LN无历史，17参数。1 ms Bernoulli、90×softplus、原有限ridge和整组训练留一规则、[1,7)评分、20 ms PSTH均不变。没有叠加旧单指数状态或调上游。',
        '', 'free GLM为预先固定4096条自生成历史/16批；不注入实测评估历史。conditional NLL可用严格过去实测spike。配对CI沿用原1000次重复/200 ms块索引，条件于固定MC曲线；原始重复不是独立刺激。MC误差见各组JSON。','',
        '|组|模型|NLL nat/1 ms|free r|MSE Hz²|均值 Hz|SD Hz|','|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['group']<3:continue
        corr='NA' if row['free_r'] is None else f"{row['free_r']:.5f}"
        text.append(f"|g{row['group']}|{row['model']}|{row['conditional_NLL_nat_per1ms']:.6f}|{corr}|{row['free_MSE_Hz2']:.2f}|{row['mean_Hz']:.2f}|{row['SD_Hz']:.2f}|")
    text+=['','配对区间和逐指标结论见PAIRED_COMPARISONS.json；正值支持V_GLM，CI跨零不代表等效。']
    for g in (3,4,5):
        for name,value in pairs[str(g)].items():text.append(f"- g{g} V_GLM对{name}："+json.dumps(value['metric_conclusions'],ensure_ascii=False))
    text+=['','训练结果：']
    for name,fit in models['models'].items():text.append(f"- {name}: {fit['parameters']}参数，ridge={fit['ridge']}，收敛={fit['converged']}，最终优化{fit['wall_seconds']:.3f}s。")
    text+=['',f"整轮计算{finished['wall_seconds']:.3f}s，峰值RSS{finished['peak_RSS_bytes']/1024**2:.1f}MiB。原Cell6 g5 INCONCLUSIVE不变；三资格false，生产默认未改。"]
    (out/'FINAL_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    write(out/'PAIRED_COMPARISONS.json',pairs);write(out/'COMPLETE.json',dict(finished,delivery_complete=True))


if __name__=='__main__':deliver(Path(sys.argv[1]))
