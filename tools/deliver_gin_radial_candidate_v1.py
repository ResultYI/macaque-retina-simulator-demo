"""Figures, actual costs and limited conclusion from already saved short runs."""
from pathlib import Path
import csv,json,sys,time
import numpy as np
from scipy.io import loadmat
from tools.gin_radial_context_v1 import ROOT,BASE,read,write
from tools.prepare_gin_radial_candidate_v1 import sha
from tools.gin_radial_saved_event_check_v1 import check

def main():
    start=time.perf_counter();event=read(BASE/'SAVED_EVENT_CHECK.json') if (BASE/'SAVED_EVENT_CHECK.json').exists() else check();s=read(BASE/'forward/standard/RESULT.json');t=read(BASE/'forward/tight/RESULT.json');n=read(BASE/'SHORT_NUMERICAL_CHECK.json')
    counts=read(BASE/'prepared/STATIC_COUNTS.json');m=read(BASE/'optical/cache/metadata.json');opt=read(BASE/'optical/MATLAB_TIMING.json');init=read(BASE/'background/RESULT.json')
    metrics=s['metrics_instances'][-1];integral=metrics['native_wall_seconds']+metrics['inner_wall_seconds'];opt_per_frame=opt['capture_seconds']/9
    extrapolation={str(seconds):dict(optical_seconds=opt_per_frame*150*seconds,neural_integral_only_seconds=integral/.05*seconds,
        full_measured_loop_seconds=s['loop_wall_seconds_including_monitor_checkpoint_restore']/.05*seconds,
        optical_plus_integral_hours=(opt_per_frame*150*seconds+integral/.05*seconds)/3600,
        optical_plus_full_observed_loop_hours=(opt_per_frame*150*seconds+s['loop_wall_seconds_including_monitor_checkpoint_restore']/.05*seconds)/3600) for seconds in (60,600)}
    costs=dict(optical=opt,optical_launcher_total_seconds=read(BASE/'optical/MATLAB_RESOURCE.json')['wall_seconds'],
        optical_startup_and_controller_residual_seconds=read(BASE/'optical/MATLAB_RESOURCE.json')['wall_seconds']-sum(opt.values()),
        optical_cold_vs_warm='First cache17.65s is measured cold construction;9-frame capture26.43s includes cache loading and first-frame native control. Per-frame warm costs were not instrumented; no separately measured warm frame claim.',
        initialization_seconds=init['initialization_wall_seconds'],standard={k:s[k] for k in ('wall_seconds','load_context_seconds','first_construction_seconds','loop_wall_seconds_including_monitor_checkpoint_restore','write_seconds')},
        standard_pure_integral_seconds=integral,standard_native_seconds=metrics['native_wall_seconds'],standard_inner_seconds=metrics['inner_wall_seconds'],standard_snapshot_seconds=metrics['snapshot_wall_seconds'],
        first_compiled_build_seconds=s['compiled_builds'][0]['construction_plus_first_call_seconds'],
        tight={k:t[k] for k in ('wall_seconds','load_context_seconds','first_construction_seconds','loop_wall_seconds_including_monitor_checkpoint_restore','write_seconds')},
        timing_nesting='Integral/snapshot/write/build are nested components, not additive to loop or trajectory total; metrics cumulative across25ms checkpoint.',
        mechanical_extrapolation=extrapolation,extrapolation_conditions='Only first8liveframes/50ms measured. Optical9-frame average includes background and native control; solver/monitor costs may vary. Full-loop scaling repeats deliberately expensive25ms checkpoints and dense witnesses; it is a cost scenario, not ETA. Integral-only omits observation/IO and is not total runtime.',
        new_long_queue_started=False)
    if not (BASE/'COSTS.json').exists():write(BASE/'COSTS.json',costs)
    import os,runpy
    # Use installed CPython3.11 plotting extras after the pinned scientific libs.
    sys.path.append(r'D:\anaconda\envs\snn_env\Lib\site-packages')
    os.environ['MPLCONFIGDIR']=str(Path(sys.argv[2])/'matplotlib_cache')
    runpy.run_path(str(ROOT/'tools/plot_gin_radial_saved_v1.py'),run_name='__main__')
    response_summary=read(BASE/'TARGET_RESPONSE_SUMMARY.json')['fields']
    root=BASE.parent;entries=[]
    for path in (root/'prepare_RESOURCE.json',BASE/'prepare_RESOURCE.json',BASE/'execution_structure_finish/structure_finish_RESOURCE.json',BASE/'executions/optical_setup/RESOURCE.json',BASE/'executions/initialize/RESOURCE.json',BASE/'executions/forward/RESOURCE.json',BASE/'optical/MATLAB_RESOURCE.json'):
        v=read(path);entries.append(dict(path=str(path.relative_to(ROOT)),wall_seconds=v['wall_seconds'],status=v.get('status','PREPARATION_RECEIPT_INCLUDING_FAILED_ATTEMPT')))
    write(BASE/'RESOURCE_LEDGER.json',dict(schema='gin69-independent-session-cost-v1',stages=entries,stage_receipt_wall_seconds_sum=sum(v['wall_seconds'] for v in entries),
        historical_cost_retained='reports/WP1_LATIMER_CELL2_4_V1 (unchanged); not reset or added to an invented budget',
        CPU_model_peak_bytes=read(BASE/'executions/initialize/RESOURCE.json')['peak_rss_bytes'],
        optical_memory_qualification='First optical wrapper sampled the MATLAB launcher, not the actual MATLAB child. Its10.8MB peak is NOT the MATLAB workload peak, which was not measured. Observed system reserve stayed>79GiB. Future wrapper corrected to verify and monitor self-reported owned MATLAB PID; no optical rerun performed.',
        prior_failures_retained=True,all_scientific_flags_false=True))
    report=f'''69#6径向条件候选：短前向已完成（未拟合）

本轮消除了“真实ecc/type/source能否进入当前局部全链、4mm新前端与冻结拓扑能否相容运行”的工程不确定性。真实源69#6为M off、3.88°；模型选择OFF-parasol group7730，1成员，实际质心[3.864888126°,0.015528208°]，其半径约3.86492°，与源径向3.88°不同，未声称逐细胞解剖空间注册。局部484Cone组/3004成员、625HC组/3136成员，其他层及Rod/RBC/AII全部保留，详细计数见prepared/STATIC_COUNTS.json。四条冻结汇聚路线闭合、完整父图归一化后投影，无局部重归一化。

用户批准的+x参考经线和222.2td亮度近似分别标ASSUMED。222.2td只锚定帧751..9750的首分钟、中央直径15角分、148个像素余弦离散加权的时间均值。单一全局增益{m['fixed_radiance_gain']:.12g}；源灰参考134.81287td，绝未拉齐到222.2。实际源GunSpectra/gamma/256、4mm瞳孔；旧3mm/Latimer光学和神经缓存均未复用。原CRT绝对标定、原2002精确片段与权重身份尚未恢复。

新相容背景初始化PASS：{init['initialization_wall_seconds']:.3f}s，原生最大RHS {init['initial_native_max_rhs']:.3g}，内层100us残差/原预算 {init['initial_inner100us_over_original_budget']:.3g}，首次BC历史更新为0。PASS表示这份有限网络的代数/状态/历史一致，background_qualified=false仍表示未达到全视网膜物理/生理背景资格。外侧继续原有held reservoir；没有把外部全部改为CRT同步输入。以真实预播放帧0独立稳态起步，未回放全部采集pre-roll，因此这50ms不等于原trial完整前史再现。

真实live帧751..758、150Hz连续50ms已完成。标准/tight均正常退出，无原生异常；编译内层实际被调用{s['compiled_RHS_calls']}/{t['compiled_RHS_calls']}次，积分无Python连续RHS回退。标准在25ms完整checkpoint恢复，native/inner/PCG/history/clock/delay/cursor逐类精确一致并实际继续后25ms。全层121字段/12040状态标量×501节点的原数值预算通过，最大预算比{n['maximum_budget_ratio']:.6g}；tight密集重构导数缺陷比{t['dynamic_defect']['maximum_norm']:.6g}。没有放宽容差、精度或门槛。

事件：两组各73个group对象事件，逐条时间并非bitwise相同；已存V与原始PCG/历史可精确重建各自事件及完整末态，再以原采样比较合同判定，结果{event['status']}。具体不同时间数、最大差值和严格lambda区间检查见SAVED_EVENT_CHECK.json。不把代表group事件冒称独立成员实测spike。

实测成本：MATLAB总{costs['optical_launcher_total_seconds']:.3f}s，其中新光学缓存{opt['cache_seconds']:.3f}s、9帧捕获{opt['capture_seconds']:.3f}s，余下{costs['optical_startup_and_controller_residual_seconds']:.3f}s是启动/控制残差；未单独测到逐帧warm耗时。标准全任务{s['wall_seconds']:.3f}s，加载{s['load_context_seconds']:.3f}s、首次构建{s['first_construction_seconds']:.3f}s（编译构建/首调用{costs['first_compiled_build_seconds']:.3f}s）、循环{s['loop_wall_seconds_including_monitor_checkpoint_restore']:.3f}s；循环内纯积分{integral:.3f}s（native{metrics['native_wall_seconds']:.3f}+inner{metrics['inner_wall_seconds']:.3f}），快照{metrics['snapshot_wall_seconds']:.3f}s。累计写出{s['write_seconds']:.3f}s嵌套于循环及收尾，不能另加。tight全任务{t['wall_seconds']:.3f}s；同一worker总{read(BASE/'executions/forward/RESOURCE.json')['wall_seconds']:.3f}s含两组和末尾比较。

只作机械条件外推：60s光学约{extrapolation['60']['optical_seconds']/3600:.2f}h，神经纯积分约{extrapolation['60']['neural_integral_only_seconds']/3600:.2f}h；600s分别约{extrapolation['600']['optical_seconds']/3600:.2f}h/{extrapolation['600']['neural_integral_only_seconds']/3600:.2f}h。若逐样重复本次带高频checkpoint和完整见证的循环开销，总光学+循环约{extrapolation['60']['optical_plus_full_observed_loop_hours']:.2f}h/{extrapolation['600']['optical_plus_full_observed_loop_hours']:.2f}h。这不是排队ETA或性能保证：仅8个live帧/50ms，不代表整分钟的求解步数；9帧平均含一次native校验，完整见证写盘密度也不适合直接复制到长任务。

资源：模型初始化峰值{read(BASE/'executions/initialize/RESOURCE.json')['peak_rss_bytes']/1024**3:.2f}GiB，前向峰值{read(BASE/'executions/forward/RESOURCE.json')['peak_rss_bytes']/1024**3:.2f}GiB，24GiB/8GiB保护未触发。光学首轮守卫误采了launcher，真实MATLAB峰值未测，不能引用其10.8MB为工作负载峰值；系统余量>79GiB。已修正未来wrapper按MATLAB自报PID核身份并监控，保留首轮原收据，不为补峰值重跑。旧间歇原生故障根因仍未证明解决。

代码与证据：tools/prepare_gin_radial_candidate_v1.py、gin_radial_pooling_v1.py、gin_radial_context_v1.py、prepare_gin_crt_optics_v1.m、export_gin_crt_tile_v1.m、gin_radial_initialize_v1.py、gin_radial_forward_v1.py、gin_radial_saved_event_check_v1.py及owned worker。INPUT_APPROXIMATION_MANIFEST.json、SOURCE_SCALE.json、optical/CONFIG.json、background/seed、forward/*/RESPONSE.npz、NUMERICAL_STATES.npz、完整checkpoint、SHORT_NUMERICAL_CHECK.json与SAVED_EVENT_CHECK.json可复查。SHORT_RESPONSE.png/SVG、SOURCE_FRAME751.png、TARGET_RESPONSE.csv为已保存结果的图表，不新增模拟。

下一步最有决策价值：在同一候选做1s warm紧凑缓存成本段，保留原时钟和必要状态，将一次性光学native校验/全量数值见证与正式目标记录成本分开，确认9000帧光学和神经长任务的实际可承受性后，再决定首个60s共享轨迹。当前未启动该段、60s/600s、拟合或任何长队列。FIT_READY/background_qualified/production_adopted仍全部false。原几何、生产默认、失败记录未覆盖。
'''
    (BASE/'FINAL_REPORT.md').write_text(report,encoding='utf8')
    write(BASE/'DELIVERY_RESULT.json',dict(status='COMPLETE_SHORT_RADIAL_CONDITIONAL_CANDIDATE',numerical_passed=n['passed'],event_check=event['status'],source_cell='69#6',source_eccentricity_deg=3.88,model_target=counts['target_group_id'],duration_seconds=.05,long_queue_started=False,fit_started=False,final_report='FINAL_REPORT.md',plot='SHORT_RESPONSE.png',delivery_seconds=time.perf_counter()-start,FIT_READY=False,background_qualified=False,production_adopted=False))
    print(json.dumps(dict(event_status=event['status'],event_max_difference=event['maximum_event_time_difference_s'],standard_seconds=s['wall_seconds'],tight_seconds=t['wall_seconds'],target_response=response_summary,report=str(BASE/'FINAL_REPORT.md'))))

if __name__=='__main__':main()
