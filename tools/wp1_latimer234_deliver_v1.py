"""Saved-data-only figures and table for one independently fitted source cell."""
from pathlib import Path
import csv,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(r'D:\macaque retinal simulator/reports/WP1_LATIMER_CELL2_4_V1')


def read(p):return json.loads(p.read_text(encoding='utf8'))


def main(cell):
    out=ROOT/'targets'/f'Latimer_Cell_{cell}';done=read(out/'COMPUTATION_COMPLETE.json');rows=read(out/'ALL_METRICS.json')
    with (out/'COMPARISON.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    shown=done['development_groups'][:3];fig,axes=plt.subplots(len(shown),1,figsize=(13,3*len(shown)),sharex=True,squeeze=False)
    for g,axis in zip(shown,axes[:,0]):
        with np.load(out/f'g{g}_PREDICTIONS.npz',allow_pickle=False) as z:
            t=z['time_bin_start_seconds'].reshape(-1,20).mean(axis=1);counts=z['observed_counts'];n=len(counts)
            axis.plot(t,counts.mean(axis=0).reshape(-1,20).mean(axis=1)*1000,color='.6',label=f'Observed{n} repeat(s)')
            for name in ('V_GLM','S_GLM','LN','constant'):axis.plot(t,z[name+'_probability'].reshape(-1,20).mean(axis=1)*1000,label=name,lw=1)
        axis.axvspan(0,1,color='.9');axis.axvspan(7,7.1,color='.9');axis.set_ylabel(f'g{g} rate Hz');axis.grid(alpha=.2)
    axes[0,0].legend(ncol=5,fontsize=8);axes[-1,0].set_xlabel('Source time(s); scoring[1,7)')
    fig.suptitle(f'Latimer Cell{cell}: independent readout / shared engineering proxy; measured eccentricity unknown')
    fig.tight_layout(rect=(0,0,1,.96))
    for ext in ('png','svg'):fig.savefig(out/('DEVELOPMENT_OVERLAY.'+ext),dpi=180)
    plt.close(fig)
    lines=[f'# Latimer Cell{cell} 条件性读出结果','',
        '真实ON-parasol记录；真实偏心度未知。复用同身份工程代理8166，并非测得的逐细胞解剖模型。各细胞独立训练和标准化，未广播Cell6参数。',
        f"训练组：{done['training_groups']}；固定开发评估组：{done['development_groups']}。按刺激完整SHA排序、响应盲选前半组训练，其余开发；重复不跨分割。原Cell6 test未访问。",'',
        'V_GLM30、刺激S_GLM30、LN17沿用原基函数、链接、正则有限网格和优化规则；标准化及留组选择仅训练。自由预测使用4096条自生成历史，conditional NLL使用严格过去观测。1ms bin，[1,7)s评分，20ms PSTH。',
        '单重复记录仅给实测轨迹与描述性时间块CI，不宣称重复试次可靠性。图固定展示按输入编号排列的前三个开发组，不按表现挑选。所有开发组数字在CSV/JSON。','',
        '|组|模型|NLL nat/1ms|free r|MSE Hz²|均值 Hz|','|---|---|---:|---:|---:|---:|']
    for row in rows:
        r='NA' if row['free_r'] is None else f"{row['free_r']:.5f}"
        lines.append(f"|{row['group']}|{row['model']}|{row['conditional_NLL_nat_per1ms']:.6f}|{r}|{row['free_MSE_Hz2']:.3f}|{row['mean_Hz']:.3f}|")
    lines+=['','FIT_READY/background_qualified/production_adopted均false。性能不证明内部生理正确；不同真实细胞位置仍未知。采样数值资格沿用本轮记录范围，不宣称所有内部状态全时程tight验证。']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (out/'COMPLETE.json').write_text(json.dumps(dict(done,delivery_complete=True),indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main(int(sys.argv[1]))
