"""One short GUI-config -> backend execution, not a long adaptation experiment."""
from pathlib import Path
import sys,json,time,os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datetime import datetime,timezone
import argparse
import tkinter as tk
import numpy as np
from retina_platform.config import RunConfig
from retina_platform.app import RetinaApp
from retina_platform.backend import SimulationSession


def draw_saved_acceptance(out,run):
    """Render saved real coordinates/trace once; no replay or numerical fitting."""
    from PIL import Image,ImageDraw,ImageFont
    m=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    report=json.loads((run/'background_initialization.json').read_text(encoding='utf-8'))
    im=Image.new('RGB',(1200,640),'#f6f8fa');draw=ImageDraw.Draw(im)
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',18)
    title=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',25)
    draw.text((30,20),'Source-limited ROI: real geometry and short recorded trace',font=title,fill='#172c3c')
    draw.text((30,60),'39 components / 20 layers | engineering RGB input | physiology validation NOT_RUN',font=font,fill='#4d5963')
    boxes=[(70,145,565,480),(680,145,1150,480)]
    native=next(c for c in m['components'] if c['id']=='outer_native')
    cones=np.array(native['ports']['R']['geometry']['OPL']['values']);hc=np.array(native['ports']['Vh']['geometry']['OPL']['values'])
    allxy=np.vstack((cones,hc));lo=allxy.min(axis=0);hi=allxy.max(axis=0);span=np.maximum(hi-lo,1e-9)
    for b in boxes:draw.rectangle(b,outline='#8d9da8',width=1)
    def point(xy):
        b=boxes[0];u=(xy-lo)/span
        return b[0]+u[0]*(b[2]-b[0]),b[3]-u[1]*(b[3]-b[1])
    for xy in cones:
        x,y=point(xy);draw.ellipse((x-3,y-3,x+3,y+3),fill='#267aab')
    for xy in hc:
        x,y=point(xy);draw.ellipse((x-5,y-5,x+5,y+5),outline='#b9582b',width=2)
    draw.text((70,105),'OPL physical coordinates: cones / horizontal cells',font=font,fill='#172c3c')
    draw.text((70,490),f'x = {lo[0]:.4f} .. {hi[0]:.4f} mm',font=font,fill='#4d5963')
    draw.text((70,516),f'y = {lo[1]:.4f} .. {hi[1]:.4f} mm; all owner + halo cells',font=font,fill='#4d5963')
    component=next(c for c in m['components'] if c['id']=='AII_AC');ids=component['ports']['voltage']['cell_ids']
    cell=report['dominant']['cell_id'];idx=ids.index(cell)
    with np.load(next((run/'windows').glob('*.npz')),allow_pickle=False) as data:
        t=data['time']*1000.;v=data['AII_AC.voltage'][idx]*1000.
    vlo=float(v.min())-.005;vhi=float(v.max())+.005
    pts=[(boxes[1][0]+(tt-t[0])/(t[-1]-t[0])*(boxes[1][2]-boxes[1][0]),
          boxes[1][3]-(vv-vlo)/(vhi-vlo)*(boxes[1][3]-boxes[1][1])) for tt,vv in zip(t,v)]
    draw.line(pts,fill='#a95222',width=3)
    for x,y in pts:draw.ellipse((x-4,y-4,x+4,y+4),fill='#a95222')
    draw.text((680,105),f'{cell} voltage: {v.min():.3f} .. {v.max():.3f} mV',font=font,fill='#172c3c')
    draw.text((680,490),f'Absolute time {t[0]:.3f} .. {t[-1]:.3f} ms',font=font,fill='#4d5963')
    draw.text((680,516),'Internal origin 0; stimulus origin 0.2 ms',font=font,fill='#4d5963')
    draw.text((30,571),'Background NOT_REACHED: 0.2 ms actual / >= 1250 ms required; original RHS residual also fails.',font=font,fill='#983521')
    draw.text((30,601),'No long adaptation, model fitting, cache, or stochastic stationarity claim.',font=font,fill='#4d5963')
    path=out/'background_flow_response.png';im.save(path);return path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--roi',choices=('peripheral','central'),default='peripheral');args=parser.parse_args()
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    out=ROOT/'reports/background_acceptance'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    image=ROOT/'reports/hc_acceptance/20261011T100207621130Z/engineering_uniform_rgb.png'
    # Only the sampler's numerical mixing budget is reduced for this test. The
    # real density and named source-derived central finite disks are unchanged.
    geometry=out/'central_geometry_fixture.json'
    geometry.write_text(json.dumps({'profile_id':'hc_joint_synthetic_v1','arbor_profile_id':'central_OPL_equal_area_W2000',
        'generator':{'sweeps':2}},indent=2)+'\n',encoding='utf-8')
    requested=RunConfig(media_path=str(image),duration_seconds=.0002,warmup_seconds=0.,window_seconds=.0002,sample_seconds=.0001,
        execution_profile='source_constrained_approximation',ipl_geometry_profile='legacy_point_ipl_v1',
        hc_geometry_profile='hc_joint_synthetic_v1',hc_arbor_profile='central_OPL_equal_area_W2000',hc_geometry_path=str(geometry),
        owner_x_mm=.35,owner_y_mm=0.,owner_width_mm=.002,owner_height_mm=.002,roi_preset='central_OPL_support',
        max_spatial_cells=20000,background_initialization_mode='bounded_convergence',background_max_sim_seconds=.0002)
    if args.roi=='peripheral':
        requested.hc_geometry_profile='source_patch_v1';requested.hc_arbor_profile='explicit_footprints';requested.hc_geometry_path=''
        requested.owner_x_mm=4.6;requested.owner_width_mm=.02;requested.owner_height_mm=.02;requested.roi_preset='peripheral_anchor'
    app=RetinaApp.__new__(RetinaApp);app.config=requested;app.media=None
    app.regional_support=json.loads((ROOT/'configurations/regional_support_v1.json').read_text(encoding='utf-8'))
    app.roi_labels={k:v['label'] for k,v in app.regional_support['roi_presets'].items()}
    tcl=tk.Tcl();app.params={name:tk.StringVar(master=tcl) for name in
        [r[0] for r in RetinaApp.PARAMS]+['media_path','gaze_csv','hc_geometry_path']}
    class Label:
        def configure(self,**kw):pass
    app.gaze_label=Label();app._set_config_vars(requested);config=app._read_config()
    (out/'gui_parsed_config.json').write_text(json.dumps(config.to_dict(),indent=2)+'\n',encoding='utf-8')
    session=SimulationSession(config,ROOT,out/'run').start()
    deadline=time.monotonic()+180
    while session.is_alive():
        if time.monotonic()>deadline:session.stop();raise RuntimeError('Short acceptance wall budget exceeded')
        session._thread.join(.2)
    if session.status!='completed':
        error=(session.path/'error.txt').read_text(encoding='utf-8') if session.path and (session.path/'error.txt').exists() else session.status
        raise RuntimeError(error)
    summary=json.loads((session.path/'summary.json').read_text(encoding='utf-8'))
    report=json.loads((session.path/'background_initialization.json').read_text(encoding='utf-8'))
    manifest=json.loads((session.path/'manifest.json').read_text(encoding='utf-8'))
    hc=json.loads((session.path/'hc_geometry.json').read_text(encoding='utf-8'))
    assert report['status']=='NOT_REACHED' and not report['converged']
    assert report['timing_requirements']['minimum_required_seconds']>=1.25
    assert not report['uncovered_states'] and report['stimulus_origin_seconds']==.0002
    assert summary['stimulus_origin_seconds']==.0002 and summary['grid_origin_seconds']==0.
    assert summary['simulation_time_seconds']==.0004 and summary['windows']==1
    assert hc['configuration']['arbor_profile_id']==requested.hc_arbor_profile
    numeric=0
    for path in (session.path/'windows').glob('*.npz'):
        with np.load(path,allow_pickle=False) as data:
            for key in data.files:
                if np.issubdtype(data[key].dtype,np.number):
                    assert np.all(np.isfinite(data[key]));numeric+=data[key].size
    result={'engineering_execution':'PASS','run_directory':str(session.path),'config_path':str(out/'gui_parsed_config.json'),
        'components':len(manifest['components']),'layers':len(manifest['required_layers']),'finite_numeric_values':numeric,
        'background_status':report['status'],'background_elapsed_seconds':report['elapsed_seconds'],
        'minimum_required_seconds':report['timing_requirements']['minimum_required_seconds'],
        'stimulus_origin_seconds':summary['stimulus_origin_seconds'],'grid_origin_seconds':summary['grid_origin_seconds'],
        'arbor_profile_id':hc['configuration']['arbor_profile_id'],'physiological_validation':'NOT_RUN',
        'long_adaptation_execution':'NOT_RUN','mixing_certified':False}
    result['figure']=str(draw_saved_acceptance(out,session.path))
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (out.parent/'latest.json').write_text(json.dumps({'acceptance_directory':str(out)},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
