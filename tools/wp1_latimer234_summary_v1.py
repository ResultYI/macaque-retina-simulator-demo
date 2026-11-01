"""Summarize completed existing outputs; never start a model, fit or evaluation."""
from datetime import datetime,timezone
from pathlib import Path
import csv,hashlib,json,statistics,sys

ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.durable_research_runtime_v1 import atomic_json,inspect_identity
OUT=ROOT/'reports/WP1_LATIMER_CELL2_4_V1'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    complete=read(OUT/'COMPLETE.json');protocol=read(OUT/'PROTOCOL_FROZEN.json');plan=read(OUT/'QUEUE_PLAN.json')
    if complete['completed_cells']!=[2,3,4] or complete['unique_neural_trajectories']!=17:raise RuntimeError('Cohort is not complete')
    if (OUT/'FINAL_REPORT.md').exists():raise FileExistsError('Final report retained')
    cache_timing=[];all_rows=[];aggregate=[];training={};evaluations={};source=[]
    for g in sorted(plan['groups']):
        folder=OUT/'cache'/f'g{g}';manifest=read(folder/'manifest.json')
        if not manifest['physical_gate_passed'] or not manifest['complete_research_restore']:raise RuntimeError('Cache required gate failed')
        captures=sorted((OUT/'capture').glob(f'g{g}_part*'));receipts=[]
        for capture in captures:
            if not (capture/'SEGMENT_EXECUTION_CONTRACT.json').exists():continue
            contract=read(capture/'SEGMENT_EXECUTION_CONTRACT.json');terminal=read(capture/'TERMINAL.json')
            if terminal['status']!='EXITED_0':raise RuntimeError('Successful stage contract has failed terminal')
            receipts.append((contract['index'],terminal['wall_seconds'],capture))
        if sorted(i for i,_,_ in receipts)!=list(range(15)):raise RuntimeError('Successful cache segment coverage differs')
        metrics=manifest['neural_metrics']
        cache_timing.append(dict(group=g,source_seconds=7.1,successful_worker_wall_seconds=sum(t for _,t,_ in receipts),
            native_integrator_seconds=metrics['native_wall_seconds'],inner_integrator_seconds=metrics['inner_wall_seconds'],
            peak_RSS_bytes=max(read(c/'RESOURCE.json')['peak_rss_bytes'] for _,_,c in receipts),
            manifest_last_worker_wall_seconds=manifest.get('wall_seconds'),samples=manifest['samples'],bins=manifest['bins']))
    for cell in (2,3,4):
        target=OUT/'targets'/f'Latimer_Cell_{cell}';done=read(target/'COMPLETE.json');frozen=read(target/'MODELS_FROZEN.json')
        training[str(cell)]=frozen['training_wall_seconds'];evaluations[str(cell)]=done['wall_seconds']
        source.append(dict(cell=cell,source_identity_sha256=sha(target/'SOURCE_IDENTITY.json'),frozen_model_sha256=sha(target/'MODELS_FROZEN.json'),
            train_groups=done['training_groups'],development_groups=done['development_groups'],source_eccentricity_deg=None,
            target=protocol['cells'][str(cell)]['target'],independent_model_parameters={n:v['parameters'] for n,v in frozen['models'].items()}))
        for g in done['development_groups']:
            result=read(target/f'g{g}_RESULT.json')
            if result['frozen_model_sha256']!=sha(target/'MODELS_FROZEN.json'):raise RuntimeError('Evaluation parameter identity changed')
        rows=read(target/'ALL_METRICS.json');all_rows+=rows
        for model in ('V_GLM','S_GLM','LN','constant'):
            values=[r for r in rows if r['model']==model]
            aggregate.append(dict(cell=cell,model=model,development_stimuli=len(values),
                mean_conditional_NLL_nat_per1ms=statistics.mean(r['conditional_NLL_nat_per1ms'] for r in values),
                mean_free_r=statistics.mean(r['free_r'] for r in values) if all(r['free_r'] is not None for r in values) else None,
                mean_free_MSE_Hz2=statistics.mean(r['free_MSE_Hz2'] for r in values),mean_predicted_rate_Hz=statistics.mean(r['mean_Hz'] for r in values),
                mean_observed_rate_Hz=statistics.mean(r['observed_mean_Hz'] for r in values)))
    failures=[];readout_worker_seconds=0.
    for capture in sorted((OUT/'capture').iterdir()):
        if not capture.is_dir() or not (capture/'TERMINAL.json').exists():continue
        if not ('_part' in capture.name or capture.name.startswith('Cell')):continue
        terminal=read(capture/'TERMINAL.json')
        if terminal['status']!='EXITED_0':failures.append(dict(capture=str(capture.relative_to(OUT)),**terminal))
        elif capture.name.startswith('Cell'):readout_worker_seconds+=terminal['wall_seconds']
    qualification=[]
    for name in ('runtime_isolation_pair_v1','runtime_isolation_pair_v2','runtime_baseline_simd_pair_v1','runtime_portable_codegen_pair_v1'):
        if (OUT/name/'TERMINAL.json').exists():qualification.append(dict(directory=name,**read(OUT/name/'TERMINAL.json')))
    queue=read(OUT/'QUEUE_PROGRESS.json');owner=inspect_identity(queue['controller'])
    if owner['state'] in ('ALIVE','UNKNOWN'):raise RuntimeError('Queue owner not confirmed exited')
    last=read(OUT/('SEGMENT_REQUEST_g'+str(plan['groups'][-1])+'.json'));leaf=Path(last['capture_directory'])
    worker=inspect_identity(read(leaf/'WORKER_READY.json')['identity'])
    if worker['state'] in ('ALIVE','UNKNOWN'):raise RuntimeError('Last model worker not confirmed exited')
    now=datetime.now(timezone.utc);started=datetime.fromisoformat(protocol['created_utc'])
    resource=dict(schema='latimer234-completed-cohort-accounting-v1',successful_cache_workers_seconds=sum(r['successful_worker_wall_seconds'] for r in cache_timing),
        successful_cache_source_seconds=17*7.1,training_seconds_by_cell=training,evaluation_seconds_by_cell=evaluations,
        successful_readout_worker_seconds=readout_worker_seconds,maximum_successful_cache_peak_RSS_bytes=max(r['peak_RSS_bytes'] for r in cache_timing),
        qualification_attempts=qualification,qualification_worker_seconds=sum(r['wall_seconds'] for r in qualification),
        failed_native_or_readout_worker_seconds=sum(r['wall_seconds'] for r in failures),failures=failures,
        elapsed_span_from_frozen_protocol_seconds=(now-started).total_seconds(),span_includes_repairs_waiting_and_delivery=True,
        historical_ledger_preserved=str(ROOT/'reports/CELL6_COMPILED_INNER_V1/RESOURCE_LEDGER.json'),
        historical_project_cumulative_not_reset_or_recomputed=True,cold_compile_in_worker_wall=True,
        worker_times_exclude_supervisor_gaps_and_untimed_short_diagnostics=True,
        numerical_qualification=read(OUT/'runtime_portable_codegen_pair_v1/PAIR_RESULT.json')['passed'],
        final_codegen=read(OUT/'runtime_portable_codegen_pair_v1/ACTUAL_COMPILED_CODEGEN.json'),
        new_interpreter_root_cause_proven=False,queue_exit=owner,last_worker_exit=worker)
    for name,rows in [('ALL_CELL_DEVELOPMENT_METRICS.csv',all_rows),('PER_CELL_DEVELOPMENT_MEANS.csv',aggregate),('CACHE_TIMING.csv',cache_timing)]:
        with (OUT/name).open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(OUT/'RESOURCE_LEDGER.json',resource);atomic_json(OUT/'COHORT_IDENTITY.json',dict(cells=source,protocol_sha256=sha(OUT/'PROTOCOL_FROZEN.json'),
        source_preparation_sha256=sha(OUT/'preparation/SOURCE_PREPARATION.npz'),new_unique_inputs=17,all_flags_false=True))
    lines=['# Cell2–4 条件性读出结果','',
        '已完成17条唯一7.1秒前向、三个细胞各自独立训练及全部固定开发评估。Cell2/4各10个刺激、Cell3为17个刺激；同一刺激的重复始终处于同一划分。',
        '三者均为来源登记的ON-parasol，真实偏心度未知；共享8166工程代理及其2成员，并非三个实测位置的解剖网络。没有将Cell6拟合参数广播。',
        '训练仅用各自冻结训练组选择有限正则；随后冻结，开发集固定评分。V_GLM30、刺激S_GLM30、LN17及恒率参考使用同一150Hz保持输入信息和[1,7)s评分窗、1ms bin、20ms PSTH。自由预测使用自生成历史，条件NLL可使用严格过去观测。',
        '以下为开发刺激等权均值，不是独立泛化结论或跨刺激置信区间；每个刺激的配对CI、MC误差及原始指标见各细胞g*_RESULT.json。单次记录不宣称重复可靠性。','',
        '|细胞|模型|条件NLL nat/1ms|free r均值|MSE Hz²|预测均率 Hz|',
        '|---|---|---:|---:|---:|---:|']
    for row in aggregate:
        r='NA' if row['mean_free_r'] is None else f"{row['mean_free_r']:.5f}"
        lines.append(f"|{row['cell']}|{row['model']}|{row['mean_conditional_NLL_nat_per1ms']:.6f}|{r}|{row['mean_free_MSE_Hz2']:.3f}|{row['mean_predicted_rate_Hz']:.3f}|")
    lines+=['','## 查看结果','']
    for cell in (2,3,4):
        folder=f'targets/Latimer_Cell_{cell}'
        lines.append(f'- Cell{cell}：[完整报告]({folder}/FINAL_REPORT.md)、[全部开发指标]({folder}/COMPARISON.csv)、[真实预测曲线]({folder}/DEVELOPMENT_OVERLAY.png)。图按冻结组序展示前三个开发刺激，未按效果选窗。')
    lines+=['','## 运行与限制','',
        f"成功前向worker累计{resource['successful_cache_workers_seconds']/3600:.3f}小时；三次训练累计{sum(training.values()):.2f}秒；固定评估累计{sum(evaluations.values()):.2f}秒。失败worker耗时另记，修复及等待包含在总时间跨度，不把末段manifest墙时当完整轨迹成本。详见RESOURCE_LEDGER.json及CACHE_TIMING.csv。",
        '原生间歇故障根因未证明。保留原故障、输入哈希启动失败及CPython setobject内部错误。单换已安装CPython构建或单停NumPy X86_V3派发均未消除全部故障；最终缓存进程采用generic CPU代码生成、关闭SVML并保留NumPy基线路径，加载原项目同一科学库。各候选均经原50ms数值/事件/PCG/历史/恢复资格验证；不能把兼容运行结果说成已诊断硬件或已根治原生损坏。正常检查点继承完整状态，失败进程的部分曲线未续算。读出训练与评估保持同一原冻结运行入口。',
        '本轮不是全时程全内部状态tight验证：沿用已接受短代表片段数值对照，并对每条全时程执行原有限性、物理、时间连续和完整恢复门。保留原求解器、FP64、事件时钟和数值门槛。',
        'FIT_READY=false、background_qualified=false、production_adopted=false。旧Cell6 test未访问、旧g5结论不改；没有新类型或真实偏心度覆盖，不证明内部生理参数已标定。',
        '已确认队列及最后模型worker退出，本轮不再自动启动新队列。']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(complete_cells=[2,3,4],cache_groups=17,training_seconds=training,evaluation_seconds=evaluations,
        successful_cache_hours=resource['successful_cache_workers_seconds']/3600,report=str(OUT/'FINAL_REPORT.md'))))


if __name__=='__main__':main()
