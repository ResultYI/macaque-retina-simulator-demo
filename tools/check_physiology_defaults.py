"""One short media-to-retina acceptance; no research controls or training."""
from pathlib import Path
import hashlib,json,queue,sys,time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retina_platform.config import RunConfig
from retina_platform.backend import SimulationSession
from retina_platform.results import ResultStore

def main():
    cfg=RunConfig(media_path=str(ROOT/'examples/platform_media/synthetic_rgb.png'),
        native_background_td=250.,cone_background_rstar_s=2000.,rod_background_rstar_s=15.)
    output=ROOT/'reports/defaults_acceptance'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    session=SimulationSession(cfg,ROOT,output/'runs').start()
    while session.is_alive():
        try:
            event=session.events.get(timeout=1.)
            if event['kind'] in ('window','error','finished') or event.get('message'):
                print(json.dumps(event,ensure_ascii=False),flush=True)
        except queue.Empty:pass
    while not session.events.empty():
        event=session.events.get()
        if event['kind'] in ('error','finished'):print(json.dumps(event,ensure_ascii=False),flush=True)
    if session.status!='completed':raise RuntimeError('Short acceptance did not complete: '+str(session.path))
    store=ResultStore(session.path)
    manifest=store.manifest
    windows=store.windows
    assert len(windows)==3
    assert manifest['execution_profile']=='source_constrained_approximation'
    assert manifest['cone_route']=='native_shared_H1_approximation'
    assert manifest['initialization']['native_background_td']==250.
    assert manifest['initialization']['adaptation_status']=='finite_prerun_NOT_steady_state_verified'
    import numpy as np
    from PIL import Image,ImageDraw
    from retina_platform.plotting import scatter_plot,curve_plot,font
    times,values=store.history('outer_native.release',0)
    checked_scalars=0
    for index in range(len(windows)):
        recorded=store.read_window(index)
        assert np.all(np.isfinite(recorded['time']))
        for name,signal in recorded['signals'].items():
            assert np.all(np.isfinite(signal)),name
            checked_scalars+=signal.size
    resolved_path=session.path/'resolved_profile.json'
    resolved=json.loads(resolved_path.read_text(encoding='utf-8'))
    assert hashlib.sha256(resolved_path.read_bytes()).hexdigest()==manifest['resolved_profile_sha256']
    assert resolved['light_context']['inner_initialization_mode']=='transient_start'
    data=store.read_window(-1)
    info=store.port_info('outer_native.release')
    xy=np.asarray(info['geometry']['RF']['values'])
    owner=np.asarray(manifest['population_owner_mask']['cones'])
    left,mapping=scatter_plot(xy,np.asarray(data['signals']['outer_native.release'])[:,-1],np.arange(len(xy)),(640,400),
        coordinate='RF',coordinate_unit='mm',value_unit='dimensionless',title='Cone patch: outlined owner + finite halo')
    display=np.asarray(mapping)
    transform=np.linalg.lstsq(np.c_[xy,np.ones(len(xy))],display[:,:2],rcond=None)[0]
    x0,x1,y0,y1=cfg.owner_bounds()
    corners=np.c_[[x0,x1,x1,x0],[y0,y0,y1,y1],np.ones(4)]@transform
    draw=ImageDraw.Draw(left)
    draw.line([tuple(p) for p in corners]+[tuple(corners[0])],fill='#e11d48',width=2)
    for px,py,index in mapping:
        if owner[index]:draw.ellipse((px-8,py-8,px+8,py+8),outline='#e11d48',width=2)
    right=curve_plot(times,values,(640,400),title='Actual recorded cone 0: native release proxy',unit='dimensionless')
    figure=Image.new('RGB',(1280,500),'#f8fafc')
    ImageDraw.Draw(figure).text((20,16),'Latest code / transient_start / uncalibrated RGB and dynamics',font=font(19),fill='#243447')
    ImageDraw.Draw(figure).text((20,470),'Red: 20 x 20 um owner at (4.60, 0) mm; other points: 40 um halo. Recorded output; physiology validation NOT_RUN.',font=font(13),fill='#243447')
    figure.paste(left,(0,60));figure.paste(right,(640,60));figure.save(output/'source_roi_response.png')
    store.export_curve('outer_native.release',0,output/'selected_cone.csv')
    # GUI integration against this same completed run; never repeats the simulation.
    import tkinter as tk
    from retina_platform.app import RetinaApp
    root=tk.Tk();root.withdraw()
    app=RetinaApp(root,ROOT)
    app._set_config_vars(cfg)
    assert app._read_config().execution_profile==cfg.execution_profile
    root.destroy()
    result={'passed':True,'run':str(session.path),'windows':len(windows),
        'components':len(manifest['components']),'layers':len(manifest['required_layers']),
        'resource_estimate':manifest['resource_estimate'],
        'light_context':manifest['initialization'],'scientific_validation':'NOT_RUN',
        'figure':'source_roi_response.png','figure_version':2,'gui_profile_roundtrip':True,
        'all_recorded_outputs_finite':True,'checked_recorded_scalars':checked_scalars,
        'resolved_profile_sha256':manifest['resolved_profile_sha256'],
        'source_code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'examples/full_target_network.py','retina_sim/initialization.py','retina_sim/runtime_profiles.py',
            'retina_sim/wiring.py','retina_platform/backend.py','tools/check_physiology_defaults.py')}}
    (output/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output.parent/'latest.json').write_text(json.dumps({'acceptance_directory':str(output),
        'run':str(session.path),'figure':str(output/'source_roi_response.png'),'figure_version':2},indent=2)+'\n',encoding='utf-8')
    print('ACCEPTANCE '+str(output),flush=True)

if __name__=='__main__':main()
