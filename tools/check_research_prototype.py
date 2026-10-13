"""One authorized GUI -> full native small-ROI build and short integration."""
from pathlib import Path
import os,sys,json,time
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datetime import datetime,timezone
import tkinter as tk
import numpy as np
from retina_platform.config import RunConfig
from retina_platform.app import RetinaApp
from retina_platform.backend import SimulationSession


def render(out,run,manifest,hc):
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',17)
    title=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',25)
    im=Image.new('RGB',(1200,690),'#f6f8fa');d=ImageDraw.Draw(im)
    d.text((25,15),'Research prototype: actual OPL contact graph and owner RGC trace',font=title,fill='#173244')
    d.text((25,58),f"{len(manifest['components'])} components / {len(manifest['required_layers'])} applicable layers | unscaled source geometry",font=font,fill='#435361')
    a=hc['contact_audit'];left=(65,145,560,500);right=(680,145,1150,500)
    p=np.array([r['pedicle_xy_mm'] for r in a['paired_cones']]);h=np.array([r['arbor_anchor_xy_OPL_mm'] for r in a['hc_geometry']])
    xy=np.vstack((p,h));lo=xy.min(0);span=np.maximum(xy.max(0)-lo,1e-9)
    def point(q):
        u=(q-lo)/span;return left[0]+u[0]*(left[2]-left[0]),left[3]-u[1]*(left[3]-left[1])
    d.rectangle(left,outline='#93a2ad');d.rectangle(right,outline='#93a2ad')
    for q in p:
        x,y=point(q);d.ellipse((x-1,y-1,x+1,y+1),fill='#347fac')
    for q,r in zip(h,a['hc_geometry']):
        x,y=point(q);d.ellipse((x-2,y-2,x+2,y+2),outline='#b6502e' if r['type']=='H1' else '#785392')
    d.text((65,107),'OPL pedicles, independent H1/H2 anchors; all halo retained',font=font,fill='#173244')
    c=next(c for c in manifest['components'] if c['id']=='midget_RGC_ON');port=c['ports']['voltage']
    i=port['is_owner'].index(True);ts=[];vs=[]
    for path in sorted((run/'windows').glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:ts.extend(z['time']*1000);vs.extend(z['midget_RGC_ON.voltage'][i]*1000)
    t=np.asarray(ts);v=np.asarray(vs);vl=float(v.min())-.0001;vr=float(v.max())+.0001
    pts=[(right[0]+(tt-t[0])/(t[-1]-t[0])*(right[2]-right[0]),right[3]-(vv-vl)/(vr-vl)*(right[3]-right[1])) for tt,vv in zip(t,v)]
    d.line(pts,fill='#ae532a',width=3)
    d.text((680,107),'Actual owner midget RGC ON voltage (mV)',font=font,fill='#173244')
    d.text((680,512),f'{t[0]:.1f} .. {t[-1]:.1f} ms; {v.min():.6f} .. {v.max():.6f} mV',font=font,fill='#435361')
    for j,sub in enumerate(('H1','H2')):
        stat=a['population_diagnostics'][sub]
        d.text((25,559+27*j),f"{sub}: owner D_all mean {stat['D_all_owner']['mean']} / D_eligible {stat['D_eligible_owner']['mean']} / C_owner {stat['C_complete_owner_HC']['mean']}",font=font,fill='#435361')
    d.text((25,618),'Finite geometry first; population contact uncalibrated; BC/AC/RGC effective dynamics uncalibrated.',font=font,fill='#8e452b')
    d.text((25,645),'Background NOT_REACHED (0.2 ms cap); bounded anchor extrapolation; physiology validation NOT_RUN.',font=font,fill='#8e452b')
    path=out/'research_prototype_response.png';im.save(path);return str(path)


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    out=ROOT/'reports/central_build_acceptance'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    image=ROOT/'reports/hc_acceptance/20261011T100207621130Z/engineering_uniform_rgb.png'
    requested=RunConfig.from_dict(dict(execution_profile='retina_contact_geometry_effective_v1',
        media_path=str(image),duration_seconds=.020,warmup_seconds=0.,window_seconds=.002,
        sample_seconds=.001,background_initialization_mode='bounded_convergence',background_max_sim_seconds=.0002))
    app=RetinaApp.__new__(RetinaApp);app.config=requested;app.media=None
    app.regional_support=json.loads((ROOT/'configurations/regional_support_v1.json').read_text(encoding='utf-8'))
    app.roi_labels={k:v['label'] for k,v in app.regional_support['roi_presets'].items()}
    tcl=tk.Tcl();app.params={name:tk.StringVar(master=tcl) for name in [r[0] for r in RetinaApp.PARAMS]+['media_path','gaze_csv','hc_geometry_path']}
    class Label:
        def configure(self,**kw):pass
    app.gaze_label=Label();app._set_config_vars(requested);config=app._read_config()
    (out/'gui_parsed_config.json').write_text(json.dumps(config.to_dict(),indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'phase':'GUI_CONFIG_PARSED','acceptance_directory':str(out),'owner_bounds_mm':config.owner_bounds()}),flush=True)
    session=SimulationSession(config,ROOT,out/'run').start();deadline=time.monotonic()+600;last=None;last_window=-1
    while session.is_alive():
        if session.status!=last:print(json.dumps({'phase':session.status,'run_directory':str(session.path)}),flush=True);last=session.status
        if session._windows!=last_window:
            print(json.dumps({'phase':'window_progress','windows':session._windows,'simulation_seconds':session._sim_time}),flush=True);last_window=session._windows
        if time.monotonic()>deadline:session.stop();raise RuntimeError('Bounded prototype acceptance exceeded wall budget')
        session._thread.join(.2)
    if session.status!='completed':
        error=(session.path/'error.txt').read_text(encoding='utf-8') if session.path and (session.path/'error.txt').exists() else session.status
        (out/'acceptance.json').write_text(json.dumps({'status':'BLOCKED','run_directory':str(session.path),'error':error,'time_integration':'see run status; no success claim'},indent=2)+'\n',encoding='utf-8')
        raise RuntimeError(error)
    run=session.path;read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    summary=read('summary.json');manifest=read('manifest.json');hc=read('hc_geometry.json');ipl=read('ipl_geometry.json');background=read('background_initialization.json')
    assert background['status']=='NOT_REACHED' and not background['converged']
    assert abs(summary['simulation_time_seconds']-.0202)<1e-12 and summary['grid_origin_seconds']==0.
    assert hc['contact_audit']['contact_policy']=='finite_geometry_first_v1'
    cert=hc['zero_contact_execution']
    assert cert['support_complete'] and len(cert['rows'])==11 and not cert['cone_population_pruned']
    assert all(r['support_complete'] and r['forward_column_degree']==r['feedback_row_degree']==0 for r in cert['rows'])
    assert ipl['requested_owner_counts']['midget_RGC_ON']>0 and not ipl['degree_deficits']
    assert len({(e['cone_id'],e['hc_id']) for e in hc['contact_audit']['edges']})==hc['contact_audit']['edge_count']
    numeric=0
    for path in (run/'windows').glob('*.npz'):
        with np.load(path,allow_pickle=False) as z:
            for key in z.files:
                if np.issubdtype(z[key].dtype,np.number):assert np.all(np.isfinite(z[key]));numeric+=z[key].size
    result={'status':'PASS','execution_profile':config.execution_profile,'run_directory':str(run),
        'components':len(manifest['components']),'applicable_layers':len(manifest['required_layers']),
        'requested_owner_counts':ipl['requested_owner_counts'],'finite_numeric_values':numeric,
        'integration_seconds':summary['simulation_time_seconds'],'background_status':background['status'],
        'background_elapsed_seconds':background['elapsed_seconds'],'stimulus_seconds':config.duration_seconds,
        'HC_edges':hc['contact_audit']['edge_count'],'population_diagnostics':hc['contact_audit']['population_diagnostics'],
        'zero_feedback_cones':len(cert['rows']),'zero_feedback_owner_cones':sum(r['is_owner'] for r in cert['rows']),
        'zero_contact_background_equilibrium':hc['zero_contact_background_equilibrium'],
        'solver_metrics':summary['solver_metrics'],'wall_seconds':summary['wall_seconds'],
        'stimulus_class':'uniform sRGB128 display approximation; native rate contrast step after declared uniform background',
        'response_scope':'20ms engineering transient covers native tau_E=16.8ms and repeated BC/RGC history ticks; not physiological latency or steady-state validation',
        'old_strict_full_build':'FAILED_PRESERVED','physiology_validation':'NOT_RUN','mixing_certified':False,
        'finite_boundary':'recurrent closure uncertified; all coupled halo states retained'}
    result['figure']=render(out,run,manifest,hc)
    (out/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out.parent/'research_prototype_latest.json').write_text(json.dumps({'acceptance_directory':str(out),'status':'PASS'},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
