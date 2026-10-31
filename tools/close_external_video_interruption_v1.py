"""Preserve verified captures and prepare pending work after unexplained process loss."""
from datetime import datetime,timezone
import hashlib,html,json,os
from pathlib import Path
import shutil,sys,time
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.build_external_video_viewer_v1 import preview


def read(p):return json.loads(p.read_bytes())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def close(base):
    start=time.perf_counter();plan=read(base/'OPTICAL_PLAN.json');jobs=read(base/'jobs.json')
    completed=[];pending=[];covered=np.zeros((120,len(plan['union_global_rows'])),dtype=bool)
    union={int(r):i for i,r in enumerate(plan['union_global_rows'])};last=None
    source=ROOT/'reports/cone_video_113_4s_eccentric_aperture_0_6deg_20261029_v1'
    previous=loadmat(source/'first_frame/capture.mat')['expected_counts'][0];frame_errors=[]
    for task in plan['tasks']:
        path=Path(task['output']);marker=Path(str(path)+'.ready')
        if not marker.exists():
            if path.exists():raise RuntimeError('Uncertain partial capture exists; retained without overwrite: '+str(path))
            pending.append(dict(configuration=task['configuration'],output=str(path)));continue
        raw=loadmat(path);meta=json.loads(raw['manifest_json'].item());rows=np.load(task['rows']);a,b=task['first'],task['stop']
        if (meta['request_signature']!=task['request_signature'] or meta['aperture_backend']!='isetbio_native_aperture_v1'
            or raw['expected_counts'].shape!=(b-a,len(rows)) or not np.isfinite(raw['expected_counts']).all()
            or np.any(raw['expected_counts']<0)):raise RuntimeError('Completed capture validation failed: '+str(path))
        covered[a:b,[union[int(r)] for r in rows]]=True
        stamp=marker.stat().st_mtime;last=stamp if last is None else max(last,stamp)
        completed.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path),tile=task['tile_index'],
            frames=[a,b],cones=len(rows),ready_utc=datetime.fromtimestamp(stamp,timezone.utc).isoformat()))
        if a==0:frame_errors.append(dict(tile=task['tile_index'],maximum_relative_error=float(np.max(
            abs(raw['expected_counts'][0]-previous[rows])/np.maximum(previous[rows],1e-20)))))
    presence={}
    for position,rows in plan['position_global_rows'].items():
        mask=covered[:,[union[int(r)] for r in rows]]
        presence[position]=dict(cones=len(rows),complete_frames=np.flatnonzero(mask.all(axis=1)).tolist(),
            covered_cone_frame_values=int(mask.sum()),required_cone_frame_values=int(mask.size))
    if len(completed)!=49 or len(pending)!=15:raise RuntimeError('Observed completion changed; inspect before closing')
    first_discovery=datetime.fromisoformat('2026-10-29T19:43:23.3649258+00:00').timestamp()
    optical_start=(base/'optical_stderr.log').stat().st_mtime
    lower=max(0.,last-optical_start);upper=max(lower,first_discovery-optical_start)
    prep=read(base/'PREPARATION_RESOURCE.json')
    interruption=dict(status='ABRUPT_PROCESS_LOSS_WITHOUT_TERMINAL_RECEIPT',
        discovery_utc=datetime.fromtimestamp(first_discovery,timezone.utc).isoformat(),
        optical_log_open_utc=datetime.fromtimestamp(optical_start,timezone.utc).isoformat(),
        last_verified_capture_utc=datetime.fromtimestamp(last,timezone.utc).isoformat(),
        completed_jobs=len(completed),pending_jobs=len(pending),coverage=presence,
        optical_elapsed_clock_estimate_lower_seconds=lower,optical_elapsed_conservative_upper_seconds=upper,
        exact_optical_wall_seconds_unavailable=True,last_observed_process_tree_peak_bytes=3908472832,
        last_observation_is_not_terminal_peak=True,operating_system_uptime_observed_ms=22589156,
        cause='Unknown; no terminal optical/controller receipt. No attribution to model or hardware.',
        neural_4s_runs_started=0,automatic_restart_performed=False,
        stop_rule_source='Existing task rule: stop automatic retries after system/resource/connection abnormalities',
        denial=False,outcome='Unexplained termination; completed files verified, remaining captures uncommitted')
    write(base/'INTERRUPTION.json',interruption);write(base/'VERIFIED_COMPLETED_CAPTURES.json',completed)
    write(base/'PENDING_CAPTURE_JOBS.json',pending);write(base/'PARTIAL_FIRST_FRAME_CHECK.json',frame_errors)
    recovery=base/'pending_optics_v1';recovery.mkdir(exist_ok=False)
    write(recovery/'OPTICAL_PLAN.json',plan);write(recovery/'jobs.json',pending)
    write(recovery/'PREPARED_ONLY.json',dict(status='NOT_EXECUTED',remaining_jobs=15,
        completed_old_jobs_not_repeated=True,same_original_source_physics_and_window=True,
        do_not_start_automatically_after_unexplained_process_loss=True))
    write(base/'RESOURCE_INTERRUPTION_LEDGER.json',dict(historical_cumulative_seconds=prep['historical_cumulative_seconds'],
        measured_preparation_seconds=prep['wall_seconds'],interrupted_optical_cost_exact=False,
        optical_clock_lower_seconds=lower,optical_conservative_upper_seconds=upper,
        cumulative_conservative_upper_seconds=prep['historical_cumulative_seconds']+prep['wall_seconds']+upper,
        neural_forward_seconds=0,failed_or_interrupted_cost_not_reset=True,delivery_cost_excluded=True))
    preview(base)
    report=['# 连续视频任务：49/64光学分块已完成，异常停止现场','',
        '已确认正式manifest是原113.AVI的首4秒/120帧/30Hz，每帧1/30秒曝光。已准备仅覆盖两局部网络的16,461个Cone、8个原tile的捕获和连续状态执行入口。没有把首帧重复成视频，没有新增刺激参数。','',
        '实际已完成49/64个原生孔径光学分块：近点13,898个Cone的120帧齐全；远点2,563个Cone仅部分齐全，还缺15个分块。已完成文件均检查真实数组形状、有限非负值、请求身份及原生孔径backend；可用区域首帧与原捕获相对误差为0。具体SHA及缺项见VERIFIED_COMPLETED_CAPTURES.json和PENDING_CAPTURE_JOBS.json。','',
        '未完成：本轮两个4秒Cone/HC→BC→AC→RGC运行都没有启动。不能把入口、单测、旧33ms结果或近点完整光学捕获称为4秒全链完成。旧单帧真实动态和数据仍完整保留。','',
        '异常：光学监督器和等待它的Python控制器均消失；没有OPTICAL_RESOURCE.json或PIPELINE_FAILED.json等终止收据。最后成功包约19:39UTC，19:43UTC核实无Python/MATLAB进程。系统uptime约6.3小时，不支持刚发生过重启；无证据判定模型、内存硬件或驱动原因。最后已观察峰值约3.64GiB，不等于失联时的实际终端峰值。','',
        '按本任务先前明确的异常停止规则，未自动重启、未扩大内存、未改系统设置或科学参数。当前阻塞是无法解释的后台执行中断，不是缺原始视频/光谱/模型素材，也不是待选新生理参数。',
        f"资源：原历史{prep['historical_cumulative_seconds']:.6f}s保留；新增准备{prep['wall_seconds']:.6f}s可测，光学确切wall-time不可恢复。外部时钟估计下界{lower:.3f}s、保守上界{upper:.3f}s；累计保守记账上界{prep['historical_cumulative_seconds']+prep['wall_seconds']+upper:.3f}s。4秒神经forward为0秒。",'',
        '后续已准备：pending_optics_v1/jobs.json只包含未完成15包，原49包不重复。仍用原run_external_video_optics_v1.py、24GiB/8GiB余量与300秒无进展保护。未执行。先核清执行环境中断并获得符合原异常规则的续跑依据，再只补这些包，然后从各自S0执行单个连续4秒IVP；不逐帧重置history/PCG。',
        '连续记录采用无损按秒NPZ分块，避免备份和整段重复占磁盘；真实事件边界逐位恢复检查通过。原50字段预算单位适配、DOP853稠密对象回归和JavaScript语法检查通过，但4秒功能运行未验证。','',
        '离线预览页显示真实原4秒视频与明确的进度状态，链接旧33ms真实响应。已准备的完整同步查看器代码暂未绑定4秒神经数据。原file://浏览器预览限制仍在，未绕过。',
        '外部储库固定S0，Rod工程背景，Sbi混合/高基线、几何与孔径混合参考等限制均不变。FIT_READY/background_qualified/production_adopted全部false。没有Git、上传或外发Chat。']
    (base/'PARTIAL_REPORT_ZH.md').write_text('\n\n'.join(report)+'\n',encoding='utf8')
    status_html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>113.AVI连续任务 · 已完成输入与待续状态</title><style>body{max-width:980px;margin:40px auto;padding:0 22px;font:16px/1.7 system-ui,"Microsoft YaHei UI",sans-serif;color:#252b31;background:#f7f8fa}h1{line-height:1.3}video{width:100%;max-width:640px;background:#16191d}a{color:#006b9c}strong{color:#9b4500}.note{border-left:3px solid #aab4bf;padding-left:15px}img{width:100%;height:auto}</style><h1>113.AVI首4秒：输入已部分完成，连续神经运行待续</h1><p class="note"><strong>49/64光学分块已完成。近点120帧光学齐全；远点还缺15包。两个4秒神经运行均未启动。</strong><br>后台进程无终止收据地消失；按既定异常停止规则未自动重启。此页不是4秒全链完成结果。</p><video controls preload="metadata" poster="source_first_frame.png" src="source_113_first4s_preview.mp4"></video><p>真实原视频0–4秒、120帧、30Hz；压缩预览仅供显示，模型输入仍为冻结原RGB。没有复制首帧补视频。</p><p><a href="PARTIAL_REPORT_ZH.md">完整进度与中断事实</a> · <a href="INTERRUPTION.json">中断收据</a> · <a href="VERIFIED_COMPLETED_CAPTURES.json">49包实际数据与SHA</a> · <a href="PENDING_CAPTURE_JOBS.json">剩余15包</a> · <a href="RESOURCE_INTERRUPTION_LEDGER.json">保守成本账本</a></p><h2>已完成的旧33.333ms真实外部输入响应</h2><p>下面只对应此前首帧33.333ms，不能映射成上面4秒视频的连续响应。所有group事件均不是独立fine-cell spike。</p><a href="../external_first_frame_two_positions_20261029_v1/FINAL_REPORT_ZH.md"><img src="../external_first_frame_two_positions_20261029_v1/external_first_exposure_curves.png" alt="此前已完成首33ms的真实五类RGC曲线"></a><p>三资格false。固定S0外边界、Rod工程背景和Sbi限制保留。HTML浏览器渲染未验证；没有CDN/网络接口，也没有绕过file://限制。</p></html>'''
    (base/'index.html').write_text(status_html,encoding='utf8')
    handoff=ROOT/'MORNING_HANDOFF.md';shutil.copy2(handoff,base/'MORNING_HANDOFF_before_interrupted_video.md')
    morning=str(base/'index.html')+'\n\n'+report[2]+'\n\n'+report[4]+'\n\n'+report[6]+'\n\n'+report[8]+'\n\n'+report[10]+'\n\n旧33ms真实两点全链报告仍在reports/external_first_frame_two_positions_20261029_v1/FINAL_REPORT_ZH.md。新4秒全链尚未执行，不能称完成。\n'
    handoff.write_text(morning,encoding='utf8');(base/'MORNING_HANDOFF.md').write_text(morning,encoding='utf8')
    names=['prepare_external_video_v1.py','run_external_video_optics_v1.py','run_external_video_v1.py',
        'run_external_first_frame_v1.py','run_external_video_pipeline_v1.py','external_video_records_v1.py',
        'analyze_external_video_v1.py','build_external_video_viewer_v1.py','external_video_viewer_v1.html']
    write(base/'IMPLEMENTATION_SHA256.json',{'tools/'+n:digest(ROOT/'tools'/n) for n in names})
    write(base/'DELIVERY_STATUS.json',dict(status='PARTIAL_DELIVERED_NO_AUTOMATIC_RETRY',completed_jobs=49,pending_jobs=15,
        near_complete_optical_frames=120,far_complete_optical_frames=0,new_4s_neural_runs=0,
        completed_values=int(covered.sum()),required_values=int(covered.size),preview_frames=120,
        wall_seconds_this_data_only_closeout=time.perf_counter()-start,
        FIT_READY=False,background_qualified=False,production_adopted=False))
    print(json.dumps(dict(status='PARTIAL_DELIVERED',completed_jobs=49,pending_jobs=15,new_4s_neural_runs=0)),flush=True)


if __name__=='__main__':close(ROOT/'reports/external_video_two_positions_4s_20261029_v1')
