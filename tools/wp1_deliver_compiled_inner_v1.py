"""Export saved acceleration evidence only; never run a simulation or fitter."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(r'D:\macaque retinal simulator')
BASE=ROOT/'reports/CELL6_COMPILED_INNER_V1';FORMAL=BASE/'formal_g1'
OLD=ROOT/'reports/CELL6_RAW_REBUILD_CONDITIONAL_READOUT_V1/runtime_monitor_repair_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def deliver():
    result=read(BASE/'FORMAL_RESULT.json');pair=read(BASE/'paired_guard_v2/PAIR_RESULT.json');timing=result['timings']
    supplement=read(BASE/'FORMAL_ACCEPTANCE_SUPPLEMENT.json') if (BASE/'FORMAL_ACCEPTANCE_SUPPLEMENT.json').exists() else None
    if supplement and supplement['original_formal_result_sha256']!=sha(BASE/'FORMAL_RESULT.json'):raise RuntimeError('Supplement source identity differs')
    effective_passed=bool(result['passed'] or supplement and supplement['passed'])
    event_scope=read(BASE/'FORMAL_EVENT_SCOPE.json')
    if not (all(v['passed'] for v in result['target_full_grid_comparison']+result['inner_endpoint_comparison']+result['auxiliary_rate_original_budget'])
            and result['native_endpoint_and_retained_intervals_bitwise_equal']
            and result['exact_PCG_and_discrete_history'] and event_scope['identities_order_and_bins_exact']):
        raise RuntimeError('Saved result contains a failure beyond the explicitly reported event-time evidence gap')
    with np.load(OLD/'cache/g1/endogenous.npz',allow_pickle=False) as a,np.load(FORMAL/'cache/g1/endogenous.npz',allow_pickle=False) as b:
        time=a['time_source_s'];fig,axes=plt.subplots(2,2,figsize=(12,8))
        axes[0,0].plot(time,(a['V']-a['V'][0])*1e6,label='Reference',lw=1)
        axes[0,0].plot(time,(b['V']-a['V'][0])*1e6,label='Compiled',lw=.8,ls='--')
        axes[0,0].set(ylabel='V minus original t=0 V (microvolt)',xlabel='Source time (s)');axes[0,0].legend()
        for field,color in [('gE','#15845f'),('gI','#a43b8b')]:
            axes[0,1].plot(time,a[field]*1e9,label=field+' reference',color=color,lw=1)
            axes[0,1].plot(time,b[field]*1e9,label=field+' compiled',color=color,ls='--',lw=.8)
        axes[0,1].set(ylabel='Target group conductance (nS)',xlabel='Source time (s)');axes[0,1].legend(fontsize=8)
        for row in result['target_full_grid_comparison']:
            ratio=abs(a[row['field']]-b[row['field']])/row['original_budget']
            axes[1,0].plot(time,np.ma.masked_equal(ratio,0),label=row['field'],lw=.8)
        axes[1,0].axhline(1,color='red',ls='--',label='Original budget')
        axes[1,0].set(yscale='log',ylabel='Absolute difference / original budget',xlabel='Source time(s); zero differences omitted on log axis');axes[1,0].legend(fontsize=8)
        before=[timing['reference_successful_workers_seconds'],timing['reference_inner_seconds'],timing['reference_snapshot_seconds']]
        after=[timing['compiled_successful_workers_seconds'],timing['compiled_inner_seconds'],timing['compiled_snapshot_seconds']]
        at=np.arange(3);axes[1,1].bar(at-.18,np.asarray(before)/60,.36,label='Reference');axes[1,1].bar(at+.18,np.asarray(after)/60,.36,label='Compiled')
        axes[1,1].set(xticks=at,xticklabels=['Successful workers','Inner integral','Snapshots'],ylabel='Measured wall time(min)');axes[1,1].legend(fontsize=8)
        for axis in axes.flat:axis.grid(alpha=.2)
        fig.suptitle('One formal7.1s cache: same input, model, solver, FP64 and15-segment schedule')
        fig.tight_layout(rect=(0,0,1,.96))
        for ext in ('png','svg'):fig.savefig(BASE/('FORMAL_COMPARISON.'+ext),dpi=180)
        plt.close(fig)
    performance=[dict(scope='controlled50ms_warm_total',reference_seconds=pair['arms']['reference']['warm_wall_seconds'],compiled_seconds=pair['arms']['compiled']['warm_wall_seconds'],speedup=pair['warm_total_speedup']),
        dict(scope='controlled50ms_inner',reference_seconds=pair['arms']['reference']['metrics_delta']['inner_wall_seconds'],compiled_seconds=pair['arms']['compiled']['metrics_delta']['inner_wall_seconds'],speedup=pair['inner_integral_speedup']),
        dict(scope='formal_successful_workers_including_cold',reference_seconds=timing['reference_successful_workers_seconds'],compiled_seconds=timing['compiled_successful_workers_seconds'],speedup=timing['successful_worker_speedup']),
        dict(scope='formal_inner',reference_seconds=timing['reference_inner_seconds'],compiled_seconds=timing['compiled_inner_seconds'],speedup=timing['inner_speedup'])]
    with (BASE/'PERFORMANCE.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(performance[0]));writer.writeheader();writer.writerows(performance)
    count=result['total_compiled_rhs_calls'];cold=result['total_construction_and_first_rhs_seconds']
    lines=['# 内层编译接入：一次配对及一条正式缓存','',
        '**已完成：** 连续内层实际使用CompiledACVSInnerRHS；完整7.1秒g1验证结束。保留原Cone–HC CPU路径、两侧IncrementDenseDOP853、FP64、容差、100 μs Rod/1 ms BC–RGC/150 Hz时钟、事件相位、反馈及随机历史。生产默认未改，未启动其他拟合队列。','',
        '该g1是已有明确Cell6轨迹的正式后端验证，不算新增细胞覆盖；没有重跑Cell1/5拟合或整个数据集。','',
        '|测量范围|原参考秒|编译秒|倍数|','|---|---:|---:|---:|']
    for row in performance:lines.append(f"|{row['scope']}|{row['reference_seconds']:.3f}|{row['compiled_seconds']:.3f}|{row['speedup']:.3f}×|")
    lines+=['',f"正式成功worker用时包含冷启动、编译、原监控、检查点和写出；两边均按15个成功片段求和。历史原控制器总墙时为{timing['reference_total_wall_seconds']:.3f}s，新接续控制器为{timing['compiled_continuation_controller_seconds']:.3f}s，加已完成首段worker为{timing['compiled_prefix_plus_continuation_seconds']:.3f}s。后者是明确标注的组合量，不冒充无中断的整段控制器计时。",'',
        f"本次从首次启动到最终退出的实际历时{timing['interrupted_end_to_end_seconds']:.3f}s，包含一次{timing['failed_worker_seconds']:.3f}s原生失败及修复暂停；没有隐藏失败成本。历史完整运行不是同时交错采样，严格同条件短配对与历史运行级比较分别报告。",'',
        f"实际编译调用累计{count:,}次，连续RHS回退{result['python_continuous_rhs_fallback_calls']}次；每次构建另有一次Direct RHS一致性检查，不是热路径回退。计划覆盖当前BC/AC V+s、legacy h、Rod/RBC/AII、真实电耦合、分组权重及RGC膜；时钟保持/离散状态没有当成连续方程改写。",'',
        f"实际各实例计划构建加首次RHS累计{cold:.3f}s，已含在正式总墙时中；这包含Numba首次执行成本，不把懒注册计时当完整冷编译。峰值RSS{result['peak_RSS_bytes']/1024**3:.3f}GiB，保留24GiB上限与8GiB系统余量。",'',
        '首次短配对的检查器误将派生conditional_rate_Hz也纳入PCG逐位比较；原随机状态、离散历史及事件完全一致。已用原GroupTimeComparator._variable_aux传播预算重判，原失败/原始曲线保留，没有重跑参考或改阈值。',
        '', '正式第二段曾在snapshot→observation_witness→BC三系数固定核校验np.abs(kernel)处捕获0xc0000409。只确认检测模块/调用位置，未证明内存损坏根因。新增防御性修复仅缓存冻结BC核的私有只读副本和原dominant-positive判定，完整原检查仍保留，物理更新、观测/历史语义未改。原源码不动，修复代码进入新缓存运行身份；连续求解后端本身未变，不伪造不同后端检查点身份。',
        '', '227个等价判定用例通过；正常退出的0.5秒前缀又与原参考全内层状态按原预算比较，native及PCG/离散历史一致。恢复仅使用该正常前缀，没有使用崩溃进程状态；修复后的候选短回归复用原参考并通过。原生故障根因仍列未解决，不能据本次成功称生产稳定。',
        '', '数值覆盖：50 ms所有状态501个节点；正式7.1秒目标V/gE/gI全部71001节点、全内层状态终点、原native终点/保留区间、PCG/离散历史和全部组事件。没有声称所有细胞所有状态的7.1秒全时程误差比较。',
        '',f"正式目标三端口原预算均通过；内层终点最大误差/预算={max(v['max_budget_ratio'] for v in result['inner_endpoint_comparison']):.6g}。原期望率最大绝对差={result['original_expected_rate_max_abs_error_Hz']:.6g}Hz，单独报告误差，不编造独立Hz预算。",'',
        f"剩余实测开销：native积分{timing['compiled_native_seconds']:.3f}s、内层积分{timing['compiled_inner_seconds']:.3f}s、snapshot{timing['compiled_snapshot_seconds']:.3f}s。监控、写盘等未独立计时，不能把剩余全部归因于某一项。",'',
        'FIT_READY/background_qualified/production_adopted均false；旧数据、失败、checkpoint、模型和生产入口全部保留。原PID16764已由进程API确认EXITED，没有关闭用户程序。',
        '', '入口和证据：FORMAL_RESULT.json、PERFORMANCE.csv、FORMAL_COMPARISON.png/svg、paired_v1/PAIR_ACCEPTED.json、paired_guard_v2/PAIR_RESULT.json、NATIVE_FAULT_LOCATION.json、KERNEL_GUARD_VALIDATION_V2.json、formal_g1/cache/g1/manifest.json。']
    gap_lines=[
        '**整体资格：INCONCLUSIVE。计算已完成；目标输出和内层终态通过原数值预算；事件时间的原采样见证未闭合。**','',
        f"两侧各{event_scope['reference_count']}条代表组事件，身份、顺序和1ms分箱完全一致。其中{event_scope['changed_timestamp_count']}条时间戳存在差异，最大{event_scope['max_absolute_difference_seconds']:.17g}s（{event_scope['max_reference_ULP_difference']:.1f} ULP）。这不是独立单细胞spike证据。",'',
        '原GroupTimeComparator._draw要求原uniform/rate见证及lambda导出的事件时间区间。当前正式缓存未保留逐bin见证，不能仅凭差异很小或终态PCG相同宣布该门槛通过。FORMAL_RESULT.json的passed=false原样保留；本报告不引入新容差、不重跑模型。详见FORMAL_EVENT_SCOPE.json。','',
    ]
    if supplement and supplement['passed']:
        gap_lines=[
            '**原数值与事件合同已通过补充见证闭合。此前严格逐位比较的失败与INCONCLUSIVE记录保留，见FORMAL_ACCEPTANCE_SUPPLEMENT.json。**','',
            '10871条事件身份、顺序、分箱完全一致。10条时间戳相差1 ULP（最大8.88e-16s），已从双方原有检查点定向回放并精确重现各自原缓存事件；原GroupTimeComparator._draw核验同uniform/PCG/前史/决策、原率/概率预算和lambda导出的事件时间区间，全部通过。没有放宽时间门槛或修改采样器。','',
            '覆盖范围仍为50ms全状态全节点、7.1s目标三端口全时程及全内层终态/原生终态/随机历史/事件。不是7.1s所有内部状态的全时程tight证明。原生内存损坏根因仍未证实。','',
        ]
    lines[2:2]=gap_lines
    (BASE/'FINAL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    first_control=read(FORMAL/'capture/g1_process_handoffs_v1/RESOURCE.json')['wall_seconds']
    paired_control=read(BASE/'paired_v1/TERMINAL.json')['wall_seconds'];regression_control=read(BASE/'paired_guard_v2/TERMINAL.json')['wall_seconds']
    write(BASE/'RESOURCE_LEDGER.json',dict(schema='compiled-inner-package-measured-resource-ledger-v1',
        measured_initial_pair_controller_seconds=paired_control,measured_candidate_only_regression_controller_seconds=regression_control,
        measured_first_formal_controller_including_failed_worker_seconds=first_control,measured_continuation_controller_seconds=timing['compiled_continuation_controller_seconds'],
        added_measured_controller_seconds=paired_control+regression_control+first_control+timing['compiled_continuation_controller_seconds'],
        non_compute_repair_pause_not_hidden=timing['interrupted_end_to_end_seconds'],
        historical_ledger_and_costs_preserved=True,historical_project_cumulative_not_reset_or_recomputed=True,
        historical_baseline_path=str(OLD/'capture/g1_process_handoffs_v1/RESOURCE.json'),peak_RSS_bytes=result['peak_RSS_bytes']))
    if supplement and supplement['passed']:
        ledger=read(BASE/'RESOURCE_LEDGER.json')
        witness_seconds=sum(v['controller_wall_seconds'] for v in supplement['windows'])
        comparator_failure_seconds=read(BASE/'event_witness_v1/window0/TERMINAL.json')['wall_seconds']
        native_witness_failure_seconds=read(BASE/'event_witness_v2/window4/TERMINAL.json')['wall_seconds']
        monitor_test_seconds=read(BASE/'MONITOR_V3_REGRESSION.json')['wall_seconds']
        ledger.update(measured_event_witness_supplement_controller_seconds=witness_seconds,
            measured_failed_witness_comparator_controller_seconds=comparator_failure_seconds,
            measured_failed_native_witness_controller_seconds=native_witness_failure_seconds,
            measured_no_model_monitor_regression_seconds=monitor_test_seconds)
        ledger['added_measured_controller_seconds']+=witness_seconds+comparator_failure_seconds+native_witness_failure_seconds
        ledger['added_measured_execution_seconds']=ledger['added_measured_controller_seconds']+monitor_test_seconds
        write(BASE/'RESOURCE_LEDGER.json',ledger)
    write(BASE/'COMPLETE.json',dict(status='ONE_COMPILED_INNER_PACKAGE_COMPLETE' if effective_passed else 'ONE_COMPILED_INNER_PACKAGE_COMPLETE_WITH_EVENT_EVIDENCE_GAP',formal_cache_validated=effective_passed,
        overall_validation_status='PASSED_ORIGINAL_CONTRACT_WITH_EVENT_WITNESSES' if effective_passed else 'INCONCLUSIVE_ORIGINAL_EVENT_TIME_WITNESS_MISSING',event_scope='FORMAL_EVENT_SCOPE.json',
        event_acceptance_supplement='FORMAL_ACCEPTANCE_SUPPLEMENT.json' if effective_passed else None,
        source_group=1,new_cell_coverage_added=0,other_queues_started=False,original_native_corruption_root_cause_proven=False,
        actual_compiled_rhs_calls=count,success_worker_speedup=timing['successful_worker_speedup'],inner_speedup=timing['inner_speedup'],
        FIT_READY=False,background_qualified=False,production_adopted=False))


if __name__=='__main__':deliver()
