"""Local-only video and saved-response viewer; never computes neural outputs."""
import argparse,csv,hashlib,html,json,os
from pathlib import Path
import re,shutil,struct,subprocess,sys,zlib
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def read(path):return json.loads(path.read_bytes())
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')


def preview(base):
    source=ROOT/'reports/cone_video_113_4s_eccentric_aperture_0_6deg_20261029_v1'
    bank=loadmat(source/'source_rgb.mat')['source_rgb'];runtime=read(source/'codec_runtime.json')
    env=os.environ.copy();env['PATH']=';'.join(runtime['dll_directories'])+';'+env['PATH']
    if (base/'source_113_first4s_preview.mp4').exists():
        old=read(base/'PREVIEW_IDENTITY.json')
        if old['original_source_rgb_sha256']!=hashlib.sha256((source/'source_rgb.mat').read_bytes()).hexdigest():
            raise RuntimeError('Existing preview source differs; preserve prior artifact')
        probe=subprocess.run([runtime['ffprobe'],'-v','error','-show_streams','-show_format','-of','json',
            str(base/'source_113_first4s_preview.mp4')],capture_output=True,env=env,timeout=30,check=True)
        meta=json.loads(probe.stdout);stream=next(s for s in meta['streams'] if s['codec_type']=='video')
        if int(stream['nb_frames'])!=120 or stream['r_frame_rate']!='30/1' or float(meta['format']['duration'])!=4.:
            raise RuntimeError('Existing preview time contract differs')
        if not (base/'source_first_frame.png').is_file():raise RuntimeError('Existing preview poster absent')
        return
    command=[runtime['ffmpeg'],'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
        '-s','640x360','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','18',
        '-pix_fmt','yuv420p','-threads','1','-movflags','+faststart',str(base/'source_113_first4s_preview.mp4')]
    run=subprocess.run(command,input=bank.transpose(3,0,1,2).tobytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        env=env,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
    if run.returncode:raise RuntimeError(run.stderr.decode(errors='replace'))
    probe=subprocess.run([runtime['ffprobe'],'-v','error','-show_streams','-show_format','-of','json',
        str(base/'source_113_first4s_preview.mp4')],capture_output=True,env=env,timeout=30,check=True)
    meta=json.loads(probe.stdout);stream=next(s for s in meta['streams'] if s['codec_type']=='video')
    assert int(stream['nb_frames'])==120 and stream['r_frame_rate']=='30/1' and float(meta['format']['duration'])==4.
    image=np.ascontiguousarray(bank[:,:,:,0]);payload=b''.join(b'\0'+image[r].tobytes() for r in range(360))
    def chunk(kind,data):return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',640,360,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(payload))+chunk(b'IEND',b'')
    (base/'source_first_frame.png').write_bytes(png)
    write(base/'PREVIEW_IDENTITY.json',dict(original_source_rgb_sha256=hashlib.sha256((source/'source_rgb.mat').read_bytes()).hexdigest(),
        frames=120,rate_Hz=30,duration_seconds=4.,preview_only_lossy_H264=True,
        neural_input_used_frozen_uncompressed_RGB=True,no_audio_model=True,ffprobe=meta))


def deliver(base):
    result=read(base/'FINAL_RESULT.json');metrics=result['type_metrics'];ledger=result['ledger']
    view=read(base/'VIEW_DATA.json');preview(base)
    rows=[]
    for r in metrics:
        rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['position'],r['type'],
            f"{r['groups']}/{r['members']}",f"{r['minimum_group_mean_voltage_mV']:.6f}…{r['maximum_group_mean_voltage_mV']:.6f}",
            f"{r['minimum_group_mean_binary_rate_Hz']:.4f}…{r['maximum_group_mean_binary_rate_Hz']:.4f}",r['group_events']])+'</tr>')
    template=(ROOT/'tools/external_video_viewer_v1.html').read_text(encoding='utf8')
    prep=read(base/'PREPARATION.json')
    for position,oldx,oldy in [('near','318.151','180.872'),('far','460.956','182.841')]:
        center=read(Path(prep['bindings'][position]['prepared'])/'GEOMETRY_SELECTION.json')['center_deg']
        template=template.replace(oldx,repr(319.5+center[0]*359/12)).replace(oldy,repr(179.5-center[1]*359/12))
    text=template.replace('__DATA__',json.dumps(view,separators=(',',':'),allow_nan=False).replace('</','<\\/'))
    text=text.replace('__TABLE__',''.join(rows)).replace('__COST__',f"{ledger['new_seconds']:.3f}s；累计{ledger['cumulative_seconds']:.3f}s")
    (base/'index.html').write_text(text,encoding='utf8')
    lines=['# 113.AVI：冻结首4秒的两位置连续功能交付','',
        '已实际完成：原视频0–4秒、120帧、30Hz、每帧1/30秒曝光。原视频14.9秒；此处复用正式manifest的既有4秒片段，没有改成新900ms刺激，也没有重复首帧补足未来输入。',
        '按原生Gaussian核/面积适配，仅为两个局部网络所需16,461个Cone在8个既有tile内补算真实图像捕获。既有光谱、LCD gamma/QE、固定gain、猕猴位置/直径和输入单位转换不变。第一帧完整捕获对照见CAPTURE_CHECK.json。',
        '每位置一个连续IVP从其独立S0开始；600个150Hz输入是120个原30Hz帧各保持五次。跨帧不重置Cone/HC/BC/AC/RGC、BC历史或PCG。旧单帧实验不拼接入本结果：第一帧在新连续任务中重放一次，不双计旧事件。终点保持原离散时钟提交，光输入用左极限；只发布[0,4)内组对象事件，终点之后的待发事件不计入。','',
        '|位置|类型|组/成员|组均膜范围mV|组均期望二元率范围Hz|组事件|','|---|---|---:|---:|---:|---:|']
    for r in metrics:
        lines.append(f"|{r['position']}|{r['type']}|{r['groups']}/{r['members']}|{r['minimum_group_mean_voltage_mV']:.6f}…{r['maximum_group_mean_voltage_mV']:.6f}|{r['minimum_group_mean_binary_rate_Hz']:.6f}…{r['maximum_group_mean_binary_rate_Hz']:.6f}|{r['group_events']}|")
    lines+=['','数值与衔接：原物理/历史/守恒、接受稠密段导数缺陷规则保留，未新增tight对照。真实覆盖节点及599个150Hz内部边界检查见NUMERICAL_SCOPE.json；与上一轮首33.333ms的50字段/75记录及事件前缀核对见FIRST_FRAME_CONTINUITY.json。原始各层端口与RGC保留同一记录时间轴，由RESPONSE_INDEX.json绑定无损NPZ时间分块；查看器仅显示真实1ms样本，不插值，不改变模型信号。',
        '范围：近点实际中心[-0.04510050976293674,-0.045844909154906355]°（半径0.0643102766°），原远点[4.724986235371699,-0.11169236923054729]°。patch中心不是各细胞偏心度。近55组/2173成员，远42组/191成员；所有图中均值按保留组等权。电流和电导仍为原group物理总量，膜电位保持原V，期望二元率与hazard及抽样组事件明确分开。',
        '原始数据按秒保存在无损NPZ分块，由各运行目录RESPONSE_INDEX.json绑定顺序与SHA。分块在事件左右侧保留相同时间戳，时间轴不重置、不重叠追加；不另存一份重复整段大文件。[近点索引](near_run_v1/RESPONSE_INDEX.json) · [远点索引](far_run_v1/RESPONSE_INDEX.json)。',
        '几何和孔径仍保持原混合参考资格：坐标沿用既有human-borrowed route，孔径近中心来自M838、外围沿用工程迁移；不是实测全域猕猴标定。原human MW光学和显示光谱来源不变。',
        '边界与资格：外部储库冻结于各位置原S0；没有全视网膜图像驱动的外部边界。Rod保持既有独立工程背景。固定LMS为reference-equivalent工程尺度，不是一般photopic Td；原视频gain与S0的约1.8e-7差保留。任意自然图像片段不能单独证明对比极性、Sbi色对抗或偏心效应。所有弱反应和Sbi混合输入/高基线限制保留，没有拟合、任意gain/beta或机制改变。FIT_READY/background_qualified/production_adopted全部false。',
        f"本轮新增可测成本{ledger['newly_measured_seconds']:.6f}s；含旧光学中断保守上界后新增记账{ledger['new_seconds']:.6f}s，历史{ledger['historical_seconds']:.6f}s→累计记账{ledger['cumulative_seconds']:.6f}s。单进程24GiB和8GiB可用内存保护保留。4秒工作进程有限超时6000s；旧首帧未拆分积分计时，先前35–45分钟/位置是粗估，本轮RESOURCE.timings保存实际native/inner积分时间。默认旧单帧入口仍820s。旧失败与成本没有重置。",'',
        '[离线动态查看](index.html) · [全RGC曲线](RGC_video_curves.png) · [上游曲线](upstream_video_curves.png) · [各类型CSV](POSITION_TYPE_SUMMARY.csv) · [原RGC时序CSV](RGC_TIME_COURSES.csv) · [组事件](GROUP_OBJECT_EVENTS.csv) · [复现](REPRODUCE.md)',
        '', '离线页面无CDN、网络API或外部资源。预览MP4仅作显示；模型输入为冻结原RGB，不用重压缩预览视频作为输入。当前工具的file://浏览器URL策略禁止本地页面预览，未绕过；静态科学曲线可直接检查，HTML源码/链接/数据绑定核对另列，不能冒称浏览器渲染已通过。',
        '', '本轮以已有4秒片段为限收束。未扩展原14.9秒、拟合或新增机制。唯一后续研究仍需区分局部固定外边界/图像内容差异与偏心生理效应，当前连续视频结果先作为可复现功能基线。']
    (base/'FINAL_REPORT_ZH.md').write_text(re.sub(r'(\|[^\n]*\|)\n\n(?=\|)',r'\1\n','\n\n'.join(lines))+'\n',encoding='utf8')
    command='@echo off\r\ncd /d "'+str(ROOT)+'"\r\n"'+str(ROOT/'.venv/Scripts/python.exe')+'" -u tools\\run_external_video_pipeline_v1.py\r\npause\r\n'
    (base/'REPRODUCE.cmd').write_text(command,encoding='ascii',newline='')
    (base/'OPEN_VIEWER.cmd').write_text('@echo off\r\nstart "" "%~dp0index.html"\r\n',encoding='ascii',newline='')
    (base/'REPRODUCE.md').write_text('双击REPRODUCE.cmd：在reports下创建带UTC时间的新目录，依次准备→单进程光学→近点→远点→本地交付。旧目录绝不覆盖。此入口会实际重跑冻结4秒片段，预计约两小时，以实测资源账本为准。只查看请打开index.html或OPEN_VIEWER.cmd。所有资格false，无上传。\n\n命令：`.venv\\Scripts\\python.exe tools\\run_external_video_pipeline_v1.py`。无默认全视网膜计算、原视频全长、拟合、扫参或并行模型。\n',encoding='utf8')
    for p in [base/'FINAL_REPORT_ZH.md',base/'index.html']:
        links=re.findall(r'(?:href|src)="([^"#]+)"',p.read_text(encoding='utf8')) if p.suffix=='.html' else re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf8'))
        assert all((base/link).is_file() for link in links),links
    assert not re.search(r'https?://|fetch\(|XMLHttpRequest',text)
    write(base/'OFFLINE_VIEW_CHECK.json',dict(source_links_data_binding='PASS',source_frames=120,
        response_positions=2,saved_data_only=True,CDN=False,network_API=False,
        browser_render_verified=False,browser_policy='file:// preview prohibited; no workaround attempted',
        physical_voltage_rate_and_group_events_distinguished=True,source_preview_is_not_model_input=True))
    write(base/'DELIVERY_SOURCES.json',{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
        'tools/analyze_external_video_v1.py','tools/build_external_video_viewer_v1.py','tools/external_video_viewer_v1.html',
        'tools/external_video_records_v1.py','tests/test_external_video_records_v1.py']})
    handoff=ROOT/'MORNING_HANDOFF.md';shutil.copy2(handoff,base/'MORNING_HANDOFF_before_video4s.md')
    summary=str(base/'index.html')+'\n\n已完成原生孔径接口收敛、同S0两点内部探针及现在同一113.AVI首4秒/120真实帧的两位置连续Cone→HC→BC→AC→RGC运行。跨帧状态、history、PCG不中断；没有重复首帧伪造视频。点击页面看同步视频、分层曲线、物理膜电位、期望率与组事件，原始记录和CSV均可查。\n\n'+f"本轮实测{ledger['new_seconds']:.3f}s，累计{ledger['cumulative_seconds']:.3f}s。三资格false；固定S0外边界、Rod工程背景、Sbi高基线/混合输入和未证明偏心效应限制均保留。只做原reference，未宣称新tight通过。没有Git/上传/外发Chat。\n\n"+'[中文完整报告](reports/'+base.name+'/FINAL_REPORT_ZH.md) · [一键复现说明](reports/'+base.name+'/REPRODUCE.md)。本地HTML浏览器预览受file://策略限制，未绕过；原始科学曲线另存PNG/SVG/PDF。最终进程退出状态由执行器在交付后记录。\n'
    handoff.write_text(summary,encoding='utf8');(base/'MORNING_HANDOFF.md').write_text(summary,encoding='utf8')
    files=[p for p in base.iterdir() if p.is_file() and p.suffix!='.log'
        and not p.name.startswith('PIPELINE_') and not p.name.endswith('_CONTROLLER.json')]
    write(base/'DELIVERABLE_SHA256.json',{p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files})
    print(json.dumps(dict(status='LOCAL_DELIVERED',index=str(base/'index.html'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    deliver(p.parse_args().experiment.resolve())
