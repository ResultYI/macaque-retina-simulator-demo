"""Read completed evidence only; never solve, optimize, or restart a stage."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import sys

ROOT=Path(r'D:\macaque retinal simulator')
sys.path.insert(0,str(ROOT))
from tools.durable_research_runtime_v1 import atomic_json
BASE=ROOT/'reports/CELL6_RAW_REBUILD_CONDITIONAL_READOUT_V1'
OUT=BASE/'runtime_monitor_repair_v1'
DIAG=Path(r'C:\Users\win11-pc\Documents\Codex\2026-10-29\task\cell6_native_repair_v2')


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    atomic_json(path,value)


def main():
    complete=[g for g in range(6) if (OUT/f'cache/g{g}/manifest.json').is_file()]
    fit_done=(OUT/'readout/COMPLETE.json').is_file()
    continuation=read(OUT/'CONTINUATION_TERMINAL.json') if (OUT/'CONTINUATION_TERMINAL.json').exists() else None
    status='COMPLETE' if len(complete)==6 and fit_done and continuation and continuation['status']=='COMPLETE' else 'PARTIAL_NOT_COMPLETE'
    captures=[]
    for path in sorted((OUT/'capture').glob('*/TERMINAL.json')):
        result=read(path)
        captures.append(dict(stage=path.parent.name,**result))
    diagnostic=[]
    for folder in ['direct_prefix_v3','direct_prefix_v4','minimum_inplace_v1','minimum_allocating_v1']:
        path=DIAG/folder/'TERMINAL.json'
        if path.exists():diagnostic.append(dict(attempt=folder,**read(path)))
    nested={name for v in captures for name in v.get('owned_child_stages',[])}
    ledger=dict(schema='cell6-monitor-repair-resources-v1',updated_utc=datetime.now(timezone.utc).isoformat(),
        original_experiment_ledger=dict(path=str(BASE/'RESOURCE_LEDGER.json'),sha256=sha(BASE/'RESOURCE_LEDGER.json')),
        original_history_reset=False,new_attempts=captures,
        new_controller_inclusive_seconds=sum(v['wall_seconds'] for v in captures if v['stage'] not in nested),
        nested_native_children_excluded_from_sum=sorted(nested),
        diagnostic_completed_receipts=diagnostic,
        diagnostic_v2_closed_cost_unknown=True,diagnostic_v2_includes_paused_debugger_wait=True,
        full_historical_cost_closed=False,do_not_add_nested_worker_and_controller_times=True,
        memory_limit_GiB=24,system_reserve_GiB=8,automatic_retry=False)
    write(OUT/'RESOURCE_LEDGER.json',ledger)
    source=read(BASE/'SOURCE_IDENTITY.json')
    repair=read(OUT/'RUNTIME_REPAIR.json')
    identity=dict(schema='cell6-runtime-repair-input-model-v1',source=source,
        model=read(BASE/'halo_model_v2/RESULT.json'),runtime_repair=repair,
        implementation_freeze=read(OUT/'TRAIN_FIRST_IMPLEMENTATION_FREEZE.json'),
        initial_runtime_implementation_freeze=read(OUT/'IMPLEMENTATION_FREEZE.json'),
        target='sg2:61468d3761f46cda:parasol_RGC_ON:8166',members=2,
        target_group_RF_deg=[4.738358014957018,.05661587713139822],
        physical_Cell6_location_verified=False,input_physical_calibration=False,
        FIT_READY=False,background_qualified=False,production_adopted=False)
    write(OUT/'INPUT_AND_MODEL_IDENTITY.json',identity)
    write(OUT/'NEW_CELL6_CACHE_MANIFEST.json',dict(status=status,completed_groups=complete,planned_groups=list(range(6)),
        completed_cache_manifests={str(g):dict(path=f'cache/g{g}/manifest.json',sha256=sha(OUT/f'cache/g{g}/manifest.json')) for g in complete},
        source_preparation=str(BASE/'SOURCE_PREPARATION.npz'),old_topology_cache_reused=False,
        old_failed_experiment_partial_cache_reused=False,verified_current_run_checkpoint_prefixes_reused=True,
        FIT_READY=False,background_qualified=False,production_adopted=False))
    if status!='COMPLETE':
        fault=next((read(p) for p in (OUT/'capture').glob('*/EXACT_NATIVE_EXCEPTION.json')),None)
        stopped=read(OUT/'CONTINUATION_FAILURE.json') if (OUT/'CONTINUATION_FAILURE.json').exists() else None
        write(OUT/'DELIVERY_STATUS.json',dict(status=status,complete_groups=complete,fit_complete=fit_done,
            native_fault=fault,continuation_failure=stopped,root_cause_proven=False,
            unchanged_retry_performed=False))
        (OUT/'FINAL_REPORT.md').write_text(
            '# Cell6运行修复：尚未完成\n\n'
            f'完整缓存清单{len(complete)}/6；拟合完成标记={fit_done}。整体控制器未确认全流程正常退出，不能认定实验完成。\n\n'
            f'终止/阻塞收据：{json.dumps(stopped,ensure_ascii=False)}\n\n'
            '监控统计的独立输出防御性候选已通过完整统计逐位等价测试；两写法各140万次均未复现，原生损坏源未证明。'
            '另已修复MemoryGuard的ctypes类型滞留：500次检查新增指针类型由500降到0，阈值与测量逻辑不变。'
            'g0旧已加载守卫与后续进程新守卫的版本区分见RUNTIME_API_OWNERSHIP_FIX.json。\n\n'
            '保留故障、原始准备、相容初态和检查点；未将部分轨迹拼成完整缓存。受影响模型计算不自动重试；'
            '继续修复须依据新捕获证据推进，不能将停止计算误作整项任务结束。FIT_READY、background_qualified、production_adopted均为false。\n',
            encoding='utf8')
        return
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    with (OUT/'FIXED_EVALUATION.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    fig,axes=plt.subplots(6,3,figsize=(14,13),sharex=True,sharey='col')
    for g in range(6):
        with np.load(OUT/f'cache/g{g}/endogenous.npz',allow_pickle=False) as z:
            t=z['time_source_s'];voltage=z['V']*1000;ge=z['gE']*1e9;gi=z['gI']*1e9
        with np.load(OUT/f'readout/g{g}_PREDICTIONS.npz',allow_pickle=False) as z:
            bin_time=z['time_bin_start_seconds'];free=z['T_probability']*1000
            observed=z['observed_counts'].mean(axis=0)*1000
        axes[g,0].plot(t,voltage,lw=.65,color='#174b83')
        axes[g,0].set_ylabel(f'g{g} V (mV)')
        axes[g,1].plot(t,ge,lw=.65,label='gE');axes[g,1].plot(t,gi,lw=.65,label='gI')
        axes[g,1].set_ylabel('Conductance (nS)')
        # Fixed non-overlapping20ms display bins; never used for forward model or training.
        display_time=bin_time[:7100].reshape(-1,20).mean(axis=1)
        axes[g,2].plot(display_time,observed.reshape(-1,20).mean(axis=1),lw=.6,color='0.6',label='3-repeat PSTH')
        axes[g,2].plot(display_time,free.reshape(-1,20).mean(axis=1),lw=.8,color='#bd4a1f',label='free prediction')
        axes[g,2].set_ylabel('Rate (Hz)')
        for ax in axes[g]:ax.axvspan(0,1,color='0.9');ax.axvspan(7,7.1,color='0.9');ax.grid(alpha=.15)
    for ax in axes[-1]:ax.set_xlabel('Original source time (s)')
    axes[0,1].legend();axes[0,2].legend()
    fig.suptitle('Cell6 engineering input / group8166: frozen new-topology conditional readout',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.97))
    for suffix in ['png','svg']:fig.savefig(figures/('cell6_new_topology_predictions.'+suffix),dpi=180)
    plt.close(fig)
    numerical=read(OUT/'cache/g0/SHORT_NUMERICAL_CHECK.json')
    decision=read(OUT/'READOUT_FIT.json')['development_decision']
    fit=read(OUT/'readout/READOUT_FIT.json');core=read(OUT/'readout/MODEL.json')['core']
    started=read(OUT/'readout/TEMPORAL_OPTIMIZER_STARTED.json')['utc']
    from datetime import timedelta
    started_local=datetime.fromisoformat(started).astimezone(timezone(timedelta(hours=9))).isoformat()
    metric_table='|组|NLL (nat/1 ms)|free r (20 ms)|MSE (Hz²)|ΔNLL H−T|ΔMSE H−T (Hz²)|ΔMSE 95%下界|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for row in rows[3:]:
        metric_table+='|g'+row['group']+'|'+ '|'.join(f'{float(row[key]):.6g}' for key in ['conditional_NLL_nat_per1ms','free_r','free_MSE_Hz2','delta_NLL_vs_H','delta_MSE_vs_H','lower95_delta_MSE_vs_H'])+'|\n'
    report=f'''# Cell6新拓扑条件性读出：本轮结果

完整缓存：6/6，每组7.1秒、71,001个100μs物理端口节点、7,100个1ms率分箱。目标固定为8166、2名原成员；同刺激3次观测重复共享一条确定性神经轨迹。各组沿原相容初态轨迹完整计算或从验证过的本轮检查点续接；未使用旧拓扑响应或原生失败进程的检查点。

本轮沿用监控统计独立输出的防御性候选，监控完整统计逐位等价回归通过。先前小型两写法各140万次均未复现，故**没有证明损坏源**。本轮完成不等于间歇原生故障根因已修复或生产稳定；原始异常、无堆转储与旧版本保留。

另已确认并修复内存守卫的ctypes类型滞留：原500次检查保留500个新指针类型，修复后500次不再增长。每个守卫实例固定持有API和类型，原测量/阈值/停止判断AST完全相同。g0原进程在守卫修改前启动；终点普通边界故障后，从已验证7.0秒检查点恢复的末尾0.1秒及后续进程使用修复版。版本与恢复证据见RUNTIME_API_OWNERSHIP_FIX.json和cache/g0/RECOVERY_RESTORE_CHECK.json。此运行资源实现差异不改变科学参数，也没有证据证明它是原生损坏源。

首次g1在5.3秒遇到0xc0000005、RIP=0；即时调用方是Python解释器FOR_ITER，经保存寄存器与本地PE映射可定位为对PyCell_Type空tp_iternext的调用，不能归因监控minimum。旧失败目录完整封存且没有进入拟合。后续采用原0.5秒完整检查点的正常退出交接：仅正常退出的worker可交给下一进程，每次恢复逐项核验原神经状态、历史、时钟和队列。原方程、精度、门槛与模型时钟不变。这是限制单进程生命周期的防御性措施，未证明损坏起点或根治原生故障。

g3随后在4.8秒出现状态打包的Python索引异常，没有记录新的原生异常。实验入口将打包改成同顺序显式循环，并用等价的无闭包监控控制流；未改生产监控模块、方程、数值门槛或读出。实际完整状态打包与完整监控payload逐位一致验证通过。从前一独立进程正常退出的4.5秒完整检查点续接，失败段没有拼入缓存。拟合文件及标准化哈希保持不变，仅续接未完成的固定评估；没有第二次优化。损坏起点仍未证明。

模型方程、物理参数、输入、汇聚、读出结构和数值门槛未改。g0原短数值对照：{numerical['scope']}；{numerical['nodes']}节点、{numerical['state_values']}状态值，最大原预算比{numerical['maximum_budget_ratio']:.6g}，passed={numerical['passed']}。其余组及全7.1秒没有声称做过全时程tight。

仅g0–g2训练；一次时间读出拟合后冻结标准化、参数与代码，再逐组固定评估。g3是已触碰内部检查，g4/g5是历史开发数据；历史test未访问。conditional NLL使用严格过去观测spike，free预测不注入观测历史。开发结论：**{decision}**。完整指标见FIXED_EVALUATION.csv，实际预测见figures/cell6_new_topology_predictions.png及SVG。

实际优化开始：{started_local}（日本时间），早于g3缓存完成。收敛=True，{fit['iterations']}次迭代、{fit['function_evaluations']}次目标评价；时间读出仅优化一次。训练原始NLL={fit['raw_NLL_nat_per1ms']:.12g} nat/1ms，惩罚目标={fit['penalized_objective']:.12g}。

冻结参数：b={core['b']:.12g}，g_DC={fit['theta'][1]:.12g}，k={fit['theta'][2]:.12g}，eta={core['eta']}，tau={core['tau_s']*1000:.12g} ms。标准化mean={core['mean_train_V']:.17g} V，SD={core['SD_train_V']:.17g} V。g_DC与eta3位于冻结范围边界；局部辨识检查不构成生理参数认定。模型和标准化哈希与训练冻结时完全一致，资格结论另存于本目录READOUT_FIT.json，不改写参数文件。

{metric_table}

评分区间[1,7)秒。Δ为H−T，正值有利于T。g3通过内部点估计检查，g4通过全部预定开发门槛；g5的ΔMSE配对95%下界为{float(rows[5]['lower95_delta_MSE_vs_H']):.6g} Hz²，未通过原定大于零门槛，故结论为INCONCLUSIVE。区间是原冻结重复/时间块重采样的描述性条件区间。本轮没有据此返调或追加拟合。

Windows状态文件短暂共享冲突采用2秒上限的原子提交重试，持续冲突仍失败；这不重试模型积分。g4两次控制器退出查询故障均发生于正常退出、资源通过的原生子进程之后，仅复用这些独立验证过的完整检查点。g5曾因控制状态提交失败停止，从前一正常退出的1.0秒检查点续接。相关旧失败、控制权交接与验证见repair_evidence；未重新评估已经保存的组。

输入仍为继承的工程等效native光输入映射，未做亮度/LMS物理标定；真实Cell6偏心度未核实，当前位置是工程代理。事件及状态是代表组，不能当独立单细胞实测。结构改动的可解释性与预测成绩分开，未重跑旧拓扑对照，不能将成绩差异归因新拓扑。

FIT_READY、background_qualified、production_adopted均为false。本轮结束，不扩展类型、偏心度、上游拟合或历史test。成本见RESOURCE_LEDGER.json；原历史账本未重置，早期暂停调试段总墙时未闭合，未给虚假精确累计。
'''
    (OUT/'FINAL_REPORT.md').write_text(report,encoding='utf8')
    write(OUT/'DELIVERY_STATUS.json',dict(status='COMPLETE',groups=6,temporal_fit_count=1,development_groups_evaluated=[3,4,5],
        decision=decision,root_cause_proven=False,figures=['figures/cell6_new_topology_predictions.png','figures/cell6_new_topology_predictions.svg'],
        FIT_READY=False,background_qualified=False,production_adopted=False))
    print(json.dumps(dict(status=status,decision=decision,report=str(OUT/'FINAL_REPORT.md'))))


if __name__=='__main__':main()
