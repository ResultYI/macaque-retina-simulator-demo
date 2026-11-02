"""Render the completed fixed evaluation, without fitting or new simulation."""
from pathlib import Path
import json
import csv
import time
import numpy as np
from retina_sim.spatial_group_graph import digest
from tools.durable_research_runtime_v1 import atomic_json

ROOT=Path(r'D:\macaque retinal simulator')
OUT=ROOT/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2'
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def build(output,guard):
    start=time.perf_counter();folder=OUT/'readout'
    result=read(folder/'FIXED_DEVELOPMENT_RESULT.json');frozen=read(folder/'MODEL_FROZEN.json')
    if digest(folder/'MODEL_FROZEN.json')!=result['frozen_model_sha256']:raise ValueError('Evaluation model identity changed')
    with np.load(folder/'FIXED_DEVELOPMENT_PREDICTIONS.npz',allow_pickle=False) as z:
        observed=z['observed_counts'][:,30000:45000].mean(0).reshape(-1,20).mean(1)*1000
        predicted=z['probability'][30000:45000].reshape(-1,20).mean(1)*1000
        constant=z['constant_probability'][30000:45000].reshape(-1,20).mean(1)*1000
        mc=z['MC_batch_probability'][:,30000:45000].reshape(16,-1,20).mean(2)*1000
        se=mc.std(0,ddof=1)/4
    t=30+(np.arange(len(observed))+.5)*.02
    with (folder/'DEVELOPMENT_CURVES_20ms.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream);writer.writerow(['time_center_s','observed_6trial_mean_Hz','V_GLM_free_event_rate_Hz','constant_Hz','Monte_Carlo_SE_Hz'])
        writer.writerows(zip(t,observed,predicted,constant,se))
    columns=['model','free_r','free_MSE_Hz2','free_mean_Hz','observed_mean_Hz','conditional_NLL_nat_per1ms']
    with (OUT/'FIXED_EVALUATION.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for name,metrics in result['metrics'].items():writer.writerow(dict(model=name,**{k:metrics[k] for k in columns[1:]}))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
    axes[0].plot(t,observed,color='black',lw=.9,label='Observed, 6 repeats, 20 ms')
    axes[0].plot(t,predicted,color='#1767a5',lw=.9,label='V-GLM free prediction')
    axes[0].fill_between(t,predicted-1.96*se,predicted+1.96*se,color='#1767a5',alpha=.15,label='MC mean +/-1.96 SE (not data CI)')
    axes[0].plot(t,constant,color='#bb6600',ls='--',label='Training constant')
    axes[0].set_ylabel('Event rate (Hz)');axes[0].legend(fontsize=8,ncol=2)
    with np.load(OUT/'formal_neural/PREFIX_45s.npz',allow_pickle=False) as z:
        selection=np.arange(300000,450001,10)
        axes[1].plot(z['time_source_s'][selection],z['V'][selection]*1000,color='#743880',lw=.9)
    axes[1].set_ylabel('Network membrane V (mV)');axes[1].set_xlabel('Source time (s)')
    axes[0].set_title('69#6, M-off, 3.88 deg: fixed development [30,45) s\nReference meridian +X; literature brightness approximation; physical states not calibrated')
    for ax in axes:ax.grid(alpha=.15);ax.set_xlim(30,45)
    for extension in ('png','svg','pdf'):fig.savefig(folder/('FIXED_DEVELOPMENT.'+extension),dpi=180)
    plt.close(fig);guard()
    m=result['metrics']['V_GLM'];c=result['metrics']['constant'];fit=frozen['model']
    cache_receipts=[read(OUT/'formal_neural'/f'SEGMENT_{i:03d}.json') for i in range(90)]
    compute=sum(x['wall_seconds'] for x in cache_receipts)
    report=f'''# 69#6 独立读出拟合与固定开发结果

训练和固定开发评估已经完成。仅本细胞；[45,60) 秒封存未使用。LN / SGLM 空间视频特征合同尚未闭合，因此该比较未完成，不用恒率冒充匹配刺激基线。

真实记录 69#6，M-off，RF 偏心度 3.88°；当前 OFF-parasol group 7730。经线 +X、4 mm 光学、中央加权一分钟 222.2 td 沿已批准近似。HC / BC 的 owner 与物理支持分离，超原范围部分为工程外推。内部状态可以读出，但不据预测成绩宣称内部生理已标定。

训练 [0,30) 秒、开发 [30,45) 秒；六重复使用同时间 mask。30 参数 V_GLM 独立训练，训练内三块选择既有正则；ridge={fit['ridge']}，收敛={fit['converged']}，迭代={fit['iterations']}。参数／标准化冻结后只做一次开发评估。过去观测 history 仅用于 conditional NLL；free 使用 4096 条自身生成 history 的路径，不注入开发观测 history。统计滤波／history 不解释为膜 reset 或物理适应。

20 ms PSTH 的 free r={m['free_r']}，MSE={m['free_MSE_Hz2']:.8g} Hz²；平均预测={m['free_mean_Hz']:.8g} Hz，实测六重复均值={m['observed_mean_Hz']:.8g} Hz。Conditional NLL={m['conditional_NLL_nat_per1ms']:.8g} nat / 1 ms。

训练恒率参考：MSE={c['free_MSE_Hz2']:.8g} Hz²，NLL={c['conditional_NLL_nat_per1ms']:.8g} nat / 1 ms。恒率相关系数无定义。其他比较未实现为 NA，不能据此声称胜过 LN / SGLM。

正式 90 个 0.5 秒分段累计 worker 墙时={compute:.3f} 秒，包含每段加载、构建、检查点和写出；它不是纯积分时间。训练墙时={frozen['training_wall_seconds']:.3f} 秒，开发 free/conditional 评估墙时={result['wall_seconds']:.3f} 秒。光学、控制器和旧失败开销分别保留在各原始收据，不与嵌套计时重复相加。

参数：`readout/MODEL_FROZEN.json`；训练选择：`readout/CROSS_VALIDATION.json`；预测数组：`readout/FIXED_DEVELOPMENT_PREDICTIONS.npz`；指标：`FIXED_EVALUATION.csv`；图：`readout/FIXED_DEVELOPMENT.png`（另有 SVG/PDF）；可绘图 CSV：`readout/DEVELOPMENT_CURVES_20ms.csv`。

图的阴影只表示 Monte Carlo 估计误差，不是实验数据或独立刺激泛化置信区间。一个连续自然电影的分块开发也不等价新刺激泛化；重复只衡量重复观测。三科学资格继续 false，原生产默认不升级。
'''
    (OUT/'FINAL_REPORT.md').write_text(report,encoding='utf8')
    atomic_json(output/'RESULT.json',dict(status='69_6_FIT_AND_FIXED_DEVELOPMENT_REPORTED',
        independent_readout_fit_complete=True,internal_network_parameters_fitted=False,matched_stimulus_baselines_complete=False,baseline_blocker='Natural-movie spatial feature contract not frozen',
        model_sha256=digest(folder/'MODEL_FROZEN.json'),report_sha256=digest(OUT/'FINAL_REPORT.md'),
        wall_seconds=time.perf_counter()-start,sealed_access=False,FIT_READY=False,background_qualified=False,production_adopted=False))
