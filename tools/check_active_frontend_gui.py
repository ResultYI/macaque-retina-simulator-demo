"""One short actual Tk Run -> ISET -> fresh bounded background -> retina solver check."""
from pathlib import Path
import argparse,sys,json,time,hashlib
import tkinter as tk
from PIL import Image
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--duration',type=float,default=.0012)
    parser.add_argument('--expect-reuse-background',action='store_true')
    parser.add_argument('--config',type=Path,help='The same RunConfig JSON accepted by the GUI and run_retina.py')
    args=parser.parse_args();rootdir=Path(__file__).resolve().parents[1];sys.path.insert(0,str(rootdir))
    from retina_platform.app import RetinaApp
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    image=output/'controlled_image.png';Image.new('RGB',(128,96),(230,120,30)).save(image)
    root=tk.Tk();root.withdraw();app=RetinaApp(root,rootdir);errors=[]
    app._error=lambda error:errors.append(str(error))
    if args.config:assert app.load_config(args.config)
    assert app.import_media(image)
    for key,value in {'duration_seconds':args.duration,'window_seconds':args.duration,'sample_seconds':.0001}.items():app.params[key].set(str(value))
    started=time.monotonic();last_phase=None
    def finish():
        try:
            if errors:raise AssertionError(errors[-1])
            path=app.session.path;manifest=json.loads((path/'manifest.json').read_bytes());summary=json.loads((path/'summary.json').read_bytes())
            source=manifest['input_frontend'];bg=json.loads((path/'background_initialization.json').read_bytes())
            assert summary['status']=='completed' and summary['components']==len(manifest['components']) and len(manifest['required_layers'])==20
            assert source['producer']['source_id']=='iset_lcd_apple_mw_v1'
            assert source['producer']['capture']['gamma_applied_once'] and not source['producer']['capture']['phototransduction_applied']
            assert source['capture_producer']=='real pinned ISET APIs' and bg['converged'] and bg['cache_reused']==args.expect_reuse_background
            assert summary['simulation_time_seconds']>summary['stimulus_origin_seconds']
            assert summary['stimulus_origin_seconds']>=bg['coverage']['timing_requirements']['minimum_required_seconds']
            store=app.result_store;window=store.read_window(0)
            assert all(np.all(np.isfinite(v)) for v in window['signals'].values())
            assert 'midget_RGC_ON.voltage' in window['signals']
            app.port_var.set('midget_RGC_ON.voltage');app._port_changed();app.selected_cell=0
            app.export_png(output/'response.png')
            result={'status':'PASS','actual_Tk_Run_button':True,'actual_ISET_capture':True,'actual_retina_solver':True,
                'MATLAB_invoked_this_run':source['actual_ISET_call'],'ISET_capture_cache_reused':source['cache_reused'],
                'approved_current_background':True,'background_cache_reused':bg['cache_reused'],
                'stimulus_duration_seconds':args.duration,'run_path':str(path),
                'frontend_signature':source['frontend_signature'],'background_context_hash':source['background_context_hash'],
                'capture_sha256':source['producer']['capture_sha256'],'layers':20,'components':summary['components'],
                'circuit_configuration_path':app.config.circuit_configuration_path,
                'internal_physiological_parameters_changed':False,'physiological_calibration':'APPROVED_MIXED_REFERENCE_NOT_FULL_MACAQUE'}
            (output/'check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            print(json.dumps(result,ensure_ascii=False),flush=True)
        except Exception:
            import traceback
            (output/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise
        finally:app.close()
    def tick():
        nonlocal last_phase
        event=app.last_event or {};phase=(app.state,event.get('phase'),event.get('message'))
        if phase!=last_phase:
            print(json.dumps({'state':app.state,'phase':event.get('phase'),'message':event.get('message'),
                'run_path':str(app.session.path) if app.session and app.session.path else None},ensure_ascii=False),flush=True);last_phase=phase
        if errors or app.session and not app.session.is_alive():finish();return
        if time.monotonic()-started>2000:
            app.session.stop();errors.append('Bounded GUI integration exceeded 2000s');finish();return
        root.after(500,tick)
    app.buttons['start'].invoke();assert app.session is not None
    root.after(500,tick);root.mainloop()
    if errors or (output/'error.txt').exists():raise SystemExit(1)

if __name__=='__main__':main()
