"""Final saved-file evidence and chart axis correction, no model computation."""
from pathlib import Path
import json,os,runpy,sys
import numpy as np
from scipy.io import loadmat
from tools.gin_radial_context_v1 import ROOT,BASE,read,write
from tools.prepare_gin_radial_candidate_v1 import sha

def main():
    sys.path.append(r'D:\anaconda\envs\snn_env\Lib\site-packages');os.environ['MPLCONFIGDIR']=str(Path(sys.argv[2])/'matplotlib_cache')
    runpy.run_path(str(ROOT/'tools/plot_gin_radial_saved_v1.py'),run_name='__main__')
    raw=json.loads(loadmat(BASE/'optical/CAPTURE.mat',variable_names=['manifest_json'])['manifest_json'].item())
    write(BASE/'OPTICAL_CHECK_SUMMARY.json',dict(native_control=raw['native_control'],sampling=raw['sampling'],
        exposure_seconds=raw['exposure_seconds'],cached_OI=read(BASE/'optical/cache/metadata.json')))
    a=sorted(read(BASE/'forward/standard/GROUP_OBJECT_EVENTS.json'),key=lambda x:(x['population'],x['cell_index'],x['bin_start_seconds']))
    b=sorted(read(BASE/'forward/tight/GROUP_OBJECT_EVENTS.json'),key=lambda x:(x['population'],x['cell_index'],x['bin_start_seconds']))
    ulps=[abs(np.float64(x['time_seconds']).view(np.int64).item()-np.float64(y['time_seconds']).view(np.int64).item()) for x,y in zip(a,b)]
    write(BASE/'EVENT_ULP_SUMMARY.json',dict(paired_events=len(a),different_times=sum(u>0 for u in ulps),maximum_ULPs=max(ulps),
        interpretation='Time differences induced by standard/tight rate differences; both satisfy unchanged original lambda-derived event interval, full PCG/history/decision witnesses agree',gate='SAVED_EVENT_CHECK.json'))
    report=BASE/'FINAL_REPORT.md';text=report.read_text(encoding='utf8')
    text+='\n补充：39个事件时间有差异，最大10ULP/3.46945e-17s，255组原采样见证均通过，PCG和严格过去历史一致。\n'
    text+='\n短窗目标膜电位−56.526751→−56.529819mV（仅约−3.068µV），期望hazard9.985625→9.968780Hz；这说明接口有真实连续输出，不能用如此短且低调制的片段证明可拟合性或预测质量。父图及光学仍含已有human-borrowed几何/滤光/OS先验，未升级为全猕猴生理标定。下一1s紧凑段应同时判断warm成本和目标V动态范围；尚未执行。\n'
    report.write_text(text,encoding='utf8')
    names=['prepare_gin_radial_candidate_v1.py','gin_radial_pooling_v1.py','gin_radial_context_v1.py','gin_radial_worker_v1.py','gin_radial_initialize_v1.py','gin_radial_forward_v1.py','gin_radial_saved_event_check_v1.py','prepare_gin_crt_optics_v1.m','export_gin_crt_tile_v1.m','deliver_gin_radial_candidate_v1.py','plot_gin_radial_saved_v1.py','finish_gin_radial_evidence_v1.py','capture_gin_radial_v1.ps1','capture_gin_optics_v1.ps1']
    local=['INPUT_APPROXIMATION_MANIFEST.json','SOURCE_SCALE.json','SOURCE_BANK.mat','prepared/local_static/selection_identity.json','POOLING_RESULT.json','optical/CONFIG.json','optical/CAPTURE.mat','background/seed/manifest.json','forward/standard/IDENTITY.json','forward/tight/IDENTITY.json','SHORT_NUMERICAL_CHECK.json','SAVED_EVENT_CHECK.json','FINAL_REPORT.md','SHORT_RESPONSE.png','SHORT_RESPONSE.svg','TARGET_RESPONSE.csv']
    files=[ROOT/'tools'/n for n in names]+[BASE/n for n in local]
    write(BASE/'CODE_AND_DELIVERY_PINS.json',dict(files={p.relative_to(ROOT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in files},
        no_production_file_modified=True,analysis_only_no_new_trajectory=True,source_scope='Separate approved conditional research entry; original strict GIN binding gate/result retained'))
    print(json.dumps(dict(native_control=raw['native_control'],maximum_event_ULPs=max(ulps),new_output_bytes=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file()),final_report=str(report))))

if __name__=='__main__':main()
