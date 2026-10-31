"""Saved-evidence delivery; never starts a model, optimizer or recovery."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import shutil
import struct
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
BASE=ROOT/'reports/CELL6_RAW_REBUILD_CONDITIONAL_READOUT_V1'
WORK=Path(r'C:\Users\win11-pc\Documents\Codex\2026-10-29\task\hc_revision')


def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,v):(BASE/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')


def main():
    source=read(BASE/'SOURCE_IDENTITY.json');model=read(BASE/'halo_model_v2/RESULT.json')
    complete=[g for g in range(6) if (BASE/'cache'/('g'+str(g))/'manifest.json').is_file()]
    captures=[]
    for folder in sorted((BASE/'capture').iterdir()):
        if not (folder/'TERMINAL.json').exists():continue
        terminal=read(folder/'TERMINAL.json');captures.append(dict(stage=folder.name,controller_wall_seconds=terminal['elapsed_seconds'],status=terminal['status']))
    fault=read(BASE/'capture/g0/NATIVE_BREAK.json') if (BASE/'capture/g0/NATIVE_BREAK.json').exists() else None
    if fault:
        rip=int(next(r['value'] for r in fault['registers'] if r['expression']=='@rip'))
        module=next(m for m in fault['modules'] if int(m['base_address_hex'],16)<=rip<int(m['base_address_hex'],16)+m['size'])
        offset=rip-int(module['base_address_hex'],16)
        raw=Path(module['path']).read_bytes();pe=struct.unpack_from('<I',raw,60)[0];sections=struct.unpack_from('<H',raw,pe+6)[0];optional=struct.unpack_from('<H',raw,pe+20)[0]
        for j in range(sections):
            start=pe+24+optional+40*j;vs,va,size,pointer=struct.unpack_from('<IIII',raw,start+8)
            if va<=offset<va+max(vs,size):file_offset=pointer+offset-va;break
        code=raw[file_offset:file_offset+2].hex();rcx=int(next(r['value'] for r in fault['registers'] if r['expression']=='@rcx'))
        failure=dict(status='REPEATED_NATIVE_FAULT_STOPPED',owned_worker_pid=read(BASE/'capture/g0/WORKER_READY.json')['identity']['pid'],
            native_module=module['path'],module_sha256=sha(Path(module['path'])),module_offset=hex(offset),instruction_hex=code,RCX=rcx,
            native_break_reason=11,native_break_reason_name='dbgEventReasonExceptionNotHandled',
            fail_fast_category='FAST_FAIL_STACK_COOKIE_CHECK_FAILURE' if code=='cd29' and rcx==2 else 'unclassified',
            category_source='Installed Windows SDK10.0.26100.0 um/winnt.h line23043: FAST_FAIL_STACK_COOKIE_CHECK_FAILURE2',
            exception_stream_present=False,exact_new_exception_code_not_recorded=True,
            corruption_origin_identified=False,model_equation_fault_proved=False,
            no_heap_dump=dict(path='capture/g0/NATIVE_BREAK_NO_HEAP.dmp',bytes=(BASE/'capture/g0/NATIVE_BREAK_NO_HEAP.dmp').stat().st_size,
                sha256=sha(BASE/'capture/g0/NATIVE_BREAK_NO_HEAP.dmp'),heap_included=False,uploaded=False),
            last_complete_progress=read(BASE/'cache/g0/PROGRESS.json'),last_complete_checkpoint=read(BASE/'cache/g0/LATEST_CHECKPOINT.json'),
            no_automatic_model_retry=True,no_dependency_or_system_change=True)
        write('NATIVE_FAILURE_EVIDENCE.json',failure)
    else:failure=None
    status='BLOCKED_REPEATED_NATIVE_FAULT' if failure else 'COMPLETE' if len(complete)==6 and (BASE/'readout/COMPLETE.json').exists() else 'PARTIAL_NOT_COMPLETE'
    write('INPUT_AND_MODEL_IDENTITY.json',dict(schema='new-raw-cell6-input-model-identity-v1',source=source,
        new_halo_model=model,model_seed_manifest_sha256=sha(BASE/'halo_model_v2/seed/manifest.json'),
        target='sg2:61468d3761f46cda:parasol_RGC_ON:8166',target_local_row=3,members=2,
        patch_center_deg=[4.724986235371699,-.11169236923054729],target_group_RF_deg=[4.738358014957018,.05661587713139822],
        target_is_explicit_halo_owner_not_core=True,old8164_retained=True,
        physical_Cell6_location_verified=False,input_physical_calibration=False,
        new_preparation_not_old_file_restoration=True,all_internal_physiological_values_frozen=True,
        FIT_READY=False,background_qualified=False,production_adopted=False))
    write('NEW_CELL6_CACHE_MANIFEST.json',dict(schema='new-raw-cell6-cache-coverage-v1',status=status,complete=len(complete)==6,
        completed_groups=complete,planned_groups=list(range(6)),
        partial_g0=dict(last_progress_seconds=.8,last_complete_checkpoint_seconds=.5,full_checkpoint_hashes_validated=True,
            checkpoint_is_recovery_candidate_not_complete_trajectory=True,preexisting_memory_corruption_cannot_be_excluded=True) if failure else None,
        no_short_prefix_admitted_as_complete=True,optical_trajectory_reused=False,FIT_READY=False,background_qualified=False,production_adopted=False))
    if not (BASE/'READOUT_FIT.json').exists():
        write('READOUT_FIT.json',dict(status='NOT_STARTED_NO_COMPLETE_TRAINING_CACHES',temporal_fit_count=0,reference_fit_count=0,
            development_evaluation_count=0,parameters_and_statistics_not_estimated=True,implementation='tools/fit_cell6_raw_readout_v1.py',
            actual_fit_or_prediction_success_claimed=False,FIT_READY=False,background_qualified=False,production_adopted=False))
    prior=ROOT/'reports/retinotopic_rf_revision/INTEGRATED_FAR_HC_POOLING_V1/RESOURCE_LEDGER.json'
    write('RESOURCE_LEDGER.json',dict(previous_ledger=dict(path=str(prior.relative_to(ROOT)),sha256=sha(prior)),history_reset=False,
        full_historical_cumulative_closed=False,captures=captures,new_controller_inclusive_wall_seconds=sum(r['controller_wall_seconds'] for r in captures),
        partial_worker_last_heartbeat_seconds=read(BASE/'g0_execution/runtime/state.json')['elapsed_seconds'] if (BASE/'g0_execution/runtime/state.json').exists() else None,
        worker_total_not_closed_due_native_termination=True,controller_and_worker_times_must_not_be_added=True,
        last_recorded_peak_RSS_bytes=read(BASE/'cache/g0/PROGRESS.json')['memory']['process_peak_rss_bytes'],
        last_recorded_available_physical_bytes=read(BASE/'g0_execution/runtime/state.json')['memory']['available_physical_bytes'],
        no_recorded_memory_guard_trip=True,memory_limit_GiB=24,system_reserve_GiB=8,artificial_model_wall_deadline=False,
        output_bytes_before_archive=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())))
    figures=BASE/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    with np.load(BASE/'SOURCE_PREPARATION.npz',allow_pickle=False) as z:
        fig,axes=plt.subplots(3,2,figsize=(12,8),sharex=True,sharey=True)
        for g,ax in enumerate(axes.flat):
            raw=z['g'+str(g)+'_stimulus'];factor=1+.05*(raw-raw[0]);ax.plot(np.arange(71000)*.0001,factor,lw=.7)
            ax.axvspan(0,1,color='gray',alpha=.15);ax.axvspan(7,7.1,color='gray',alpha=.15)
            ax.set_title('g'+str(g)+(' train' if g<3 else ' touched internal' if g==3 else ' development'));ax.set_ylabel('Native Td / local S0');ax.set_xlabel('Source time (s)');ax.grid(alpha=.2)
        fig.suptitle('Cell6原始刺激准备已完成；神经全时程与预测尚未完成',fontsize=14)
        fig.text(.5,.012,'Inherited engineering affine mapping only; no luminance/LMS calibration. Gray regions excluded from scoring.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.035,1,.96))
        for ext in ('png','svg'):fig.savefig(figures/('prepared_stimuli.'+ext),dpi=180)
        plt.close(fig)
    text=f'''# Cell6原始重建实验：已准备，原生故障阻塞

状态：**{status}**。完整神经缓存{len(complete)}/6；时间读出拟合0次，g3–g5评估0次。没有把0.8s进度或0.5s检查点当完整7.1s结果。没有预测曲线、完整FIXED_EVALUATION.csv或完整响应ZIP。

已完成新实验SOURCE_IDENTITY.json、六组实际刺激与3重复1ms分箱、独立新PCG64种子2026103006的1000组配对重采样、原7参数/2–50ms边界/原损失正则和优化规则。18个获准SpikeTimes单元被选择性解码，其他12个未解码；E/I实测电流未解码或注入。g0–g2训练、g3曾触碰内部检查、g4/g5历史开发评估，历史test不重开。原MAT哈希59e1457bc27f7392a3c458902941a234c687b26a525b26d168bc8609aae40025。

输入继承已有Btd×[1+0.05×(raw−raw0)]、150Hz ZOH及原100µs/1ms时钟。此轮新实验模型origin=0由现相容初态定义，不恢复旧派生文件字节或旧绝对时钟。六组全波形已核对正输入，前1s恒定；它是已批准等效光输入，非物理亮度/LMS标定或113.AVI替代。figures/prepared_stimuli.png/svg只展示真实刺激准备，**不是预测结果**。

8166及其2原成员已实际纳入，local row=3；保留8164、原中心、core和支持域，8166明确为halo owner。为它生成冻结锚点公式的BC→parasol行及原cross_OFF_ON V+s池，无新增直接AII→RGC边。原核/面积、HC、BC/AC/AII/RGC方程及物理参数冻结，已比较原对象的参数值和上游选择不变。新相容背景见halo_model_v2/RESULT.json与seed；新增halo模型43个RGC组、39个四池有效AC对象。

首轮构建因原widefield旧路由在ACVS移除前触发closure而停止，保留halo_model/及prepare_execution。v2提前应用原ACVS同一筛选谓词，再对全部活跃路由做原closure检查；没有增加无用旧AC节点、关闭活跃守卫或改变科学机制，独立初始化成功。g0在50ms完成严格checkpoint往返并继续原轨迹；原相同RK增量的防抵消密集插值适配用于本新入口，未改接受步状态、方程、容差或预算。

g0最后完整进度0.8/7.1s，实际墙时413.740s（其中native11?详见PROGRESS，native29.371s、inner249.138s）；最后worker心跳454.363s，controller总墙时462.964s。发生于尚未得到变化刺激有效缓存的阶段。峰值2.599GiB、最后记录系统可用约77GiB，无已记录内存守卫触发，不能据此断言硬件或内存破坏来源已排除。

## 原生现场与停止依据

Native捕获确认为未处理异常（dbgEventReasonExceptionNotHandled），模块NumPy_multiarray_umath.cp311-win_amd64.pyd＋0x2b04c9，与此前原生故障偏移相同；该RIP磁盘指令为int29h，前置mov ecx,2且捕获RCX=2。安装的Windows SDK winnt.h第23043行定义2=FAST_FAIL_STACK_COOKIE_CHECK_FAILURE，确认栈cookie保护失败类别。**造成损坏的起点尚未定位，不能说科学方程错误、普通数值超限或NumPy版本本身已被证明有错。** 无exception stream，故本次准确Windows异常码未记录，不冒用旧0xC0000409作为新记录。

现场：capture/g0/NATIVE_BREAK.json（线程、寄存器、模块）、NATIVE_BREAK_NO_HEAP.dmp（230773bytes，不含堆、不上传）、NATIVE_FAILURE_EVIDENCE.json。清理阶段出现身份不匹配保护拒绝，未强杀身份不明进程；随后只读核实本任务worker/VS均退出。没有依赖安装、系统设置变更或重复模型重试。

0.5s完整研究CHECKPOINT_0005000的所有bundle哈希已验证，5001节点/501个1ms bin、PCG/history、native frontier/输入/队列均保留。它可供后续同身份恢复判断，**并非完整轨迹，也不能单凭哈希排除更早的潜在内存破坏**。更晚0.5–0.8s没有完整落盘状态，不拼接。g1–g5、g0短数值参考、训练标准化/H/S/T拟合、固定开发评估均未运行。

## 可运行入口与剩余工作

tools/prepare_cell6_raw_experiment_v1.py为原始准备脚本；tools/cell6_halo_context_v1.py为新owner/相容初态；tools/run_cell6_raw_stage_v1.py、工作区capture_cell6_raw_v1.ps1为分阶段Native监护入口；tools/finish_cell6_raw_pipeline_v1.py负责串行六组→一次拟合→固定评估；tools/fit_cell6_raw_readout_v1.py复用原时间读出目标/滤波函数。源码快照在source_archive。当前入口会拒绝覆盖已存在运行，**不是让用户直接重复启动失败g0的指令**。

后续唯一阻塞是先处理已捕获的重复原生栈完整性故障，再核验可恢复checkpoint并补剩余新轨迹；不能将未知故障下的部分数据交给拟合。无需再补旧8项派生文件，无需换8164、改上游增益或重定科研方案。本轮按异常停止合同停止新计算，未继续压力复现或扫参。

原响应ZIP流程只接受完整包；本次不伪造complete标志绕过该流程，按现目录保留分类的原始准备、model、partial cache、Native现场与报告。全部三个科学flags仍false、默认生产模型不变、无Git写操作或外发。
'''
    text=text.replace('（其中native11?详见PROGRESS，native29.371s、inner249.138s）','（其中native29.371s、inner249.138s）')
    (BASE/'FINAL_REPORT.md').write_text(text,encoding='utf8')
    archive=BASE/'source_archive';archive.mkdir(exist_ok=True)
    names=['cell6_raw_intake_v1.py','prepare_cell6_raw_experiment_v1.py','cell6_halo_context_v1.py','cell6_raw_runner_v1.py',
        'cell6_raw_cache_v1.py','cell6_raw_numerical_v1.py','run_cell6_raw_stage_v1.py','fit_cell6_raw_readout_v1.py',
        'finish_cell6_raw_pipeline_v1.py','capture_cell6_raw_v1.ps1','cell6_worker_native_gate_v1.py','deliver_cell6_raw_results_v1.py']
    for name in names:
        path=WORK/name
        if path.exists():shutil.copy2(path,archive/name)
    files=[p for p in BASE.iterdir() if p.is_file()]+list(archive.iterdir())+list(figures.iterdir())
    files=[p for p in files if p.name!='ARTIFACT_INDEX.json']
    write('ARTIFACT_INDEX.json',dict(status=status,files=[dict(path=str(p.relative_to(BASE)),bytes=p.stat().st_size,sha256=sha(p)) for p in files],
        native_dump_excluded_from_portable_deliverables=True,no_upload=True,complete_result_claimed=False))
    print(json.dumps(dict(status=status,completed_neural_cases=len(complete),fits=0,fixed_evaluations=0,report=str(BASE/'FINAL_REPORT.md')),ensure_ascii=False))


if __name__=='__main__':main()
