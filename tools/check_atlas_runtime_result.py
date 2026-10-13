"""Verify and export a saved atlas Run in the GUI, without another integration."""
from pathlib import Path
import argparse,json,sys,hashlib,csv
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from retina_platform.results import ResultStore
    from retina_platform.config import RunConfig
    store=ResultStore(args.run);out=args.output
    if (out/'verification.json').exists():raise FileExistsError('Verification is already complete')
    out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((store.path/'summary.json').read_bytes());assert summary['status']=='completed'
    assert len(store.windows)==1 and np.isclose(summary['simulation_time_seconds'],.0012)
    assert summary['components']==34 and len(summary['required_layers'])==20
    win=store.read_window();assert np.allclose(win['time'],np.arange(13)*.0001,rtol=0,atol=1e-15)
    declared={x['key']:x for c in store.manifest['components'] for x in c['ports'].values()}
    for key,signal in win['signals'].items():
        info=declared[key];assert np.all(np.isfinite(signal))
        assert signal.shape[0]==len(info['cell_ids'])
        assert info['cell_ids']==win['metadata']['recorded_cell_ids'][key]
        assert (info['unit'],info['meaning'])==(win['ports'][key]['unit'],win['ports'][key]['meaning'])
        for frame,geo in info['geometry'].items():
            assert geo['unit']=='mm' and len(geo['values'])==signal.shape[0]
        if signal.ndim==2:assert signal.shape[1]==len(win['time'])
    initial=json.loads((store.path/'initial_checkpoint.json').read_bytes())
    final=json.loads((store.path/'final_checkpoint.json').read_bytes())
    assert initial!=final
    rgc_keys=[name+'.voltage' for name in ('midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON')]
    assert all(key in win['signals'] for key in rgc_keys)
    # GUI and CLI must resolve the same frozen production artifact as this saved Run.
    import tkinter as tk
    from retina_platform.app import RetinaApp
    window=tk.Tk();window.attributes('-alpha',0)
    app=RetinaApp(window,root);window.update_idletasks()
    try:
        assert app.import_media(root/'examples/platform_media/synthetic_rgb.png')
        assert app._read_config().execution_profile==RunConfig.project_default(root).execution_profile
        assert str(app.entries['execution_profile'].cget('state'))=='disabled'
        assert len(app.entries['execution_profile'].cget('values'))==1
        assert app._read_config().execution_profile=='atlas_population_revision_v1'
        active=json.loads((root/'configurations/retina_platform_v1.json').read_bytes())
        binding=active['atlas_target']['population_revision']
        assert all(store.manifest['atlas_runtime']['graph_binding'][k]==v for k,v in binding.items())
        assert hashlib.sha256((root/binding['artifact']).read_bytes()).hexdigest()==binding['sha256']
        preview=json.loads((store.path/'population_preview/manifest.json').read_bytes())
        assert preview['artifact_binding']==binding
        assert app.load_results(store.path)
        for layer in summary['required_layers']:
            app.layer_var.set(layer);app.layer_combo.event_generate('<<ComboboxSelected>>');window.update_idletasks()
            assert app.port_records,layer
        app.layer_var.set('全部层');app.layer_combo.event_generate('<<ComboboxSelected>>')
        key='midget_RGC_ON.voltage';app.port_var.set(key);app.port_combo.event_generate('<<ComboboxSelected>>')
        app.coordinate_var.set('RF');app.selected_cell=0;app._render_results()
        assert app.pointmap
        if not (out/'midget_owner_voltage.csv').exists():assert app.export_curve(out/'midget_owner_voltage.csv')
        if not (out/'atlas_runtime_response.png').exists():assert app.export_png(out/'atlas_runtime_response.png')
        with (out/'midget_owner_voltage.csv').open(encoding='utf-8-sig') as stream:rows=list(csv.reader(stream))
        assert np.array_equal(np.array([float(x[1]) for x in rows[1:]]),store.history(key,0)[1])
        if not (out/'exported_run').exists():assert app.export_bundle(out/'exported_run')
        reloaded=ResultStore(out/'exported_run')
        assert (reloaded.path/'final_checkpoint.json').read_bytes()==(store.path/'final_checkpoint.json').read_bytes()
        from PIL import Image,ImageDraw
        from retina_platform.plotting import scatter_plot,curve_plot,font
        figure=Image.new('RGB',(1200,1360),'#f8fafc');draw=ImageDraw.Draw(figure)
        draw.text((18,10),'34 components / 20 layers · actual saved 1.2 ms transient · zero prerun',font=font(17),fill='#243447')
        for i,key in enumerate(rgc_keys):
            info=store.port_info(key);values=win['signals'][key][:,-1];xy=np.asarray(info['geometry']['RF']['values'])
            image,_=scatter_plot(xy,values,np.arange(len(values)),(600,260),coordinate='RF',value_unit=info['unit'],
                title=f'{key} · {len(values)} recorded owner cells')
            figure.paste(image,(0,42+260*i))
            t,y=store.history(key,0)
            figure.paste(curve_plot(t,y,(600,260),title='Recorded cell 0 · absolute seconds',unit=info['unit']),(600,42+260*i))
        draw.text((18,1343),'Saved outputs; no equilibrium or physiological calibration claim.',font=font(11),fill='#64748b')
        figure.save(out/'population_runtime_response.png')
        arrays=final['payload']['dynamic']['state']['ordered_state_arrays']
        report={'status':'PASS','run':str(store.path),'integration_reused':True,
            'checks':['absolute seconds/13 author-grid samples','all finite recorded ports','IDs/row counts/units/coordinates',
                'changed actual checkpoint','20 GUI layers and owner point','CSV equals actual history','PNG export',
                'result bundle/checkpoint reload','one shared active default; no mode switch'],
            'state_blocks':len(arrays),
            'state_scalars':sum(np.asarray(v).size for v in arrays),
            'artifact_binding':binding,'figure':'population_runtime_response.png',
            'short_run_summary':summary,'unsupported_paths':store.manifest['atlas_runtime']['unsupported_paths'],
            'checkpoint_sha256':hashlib.sha256((store.path/'final_checkpoint.json').read_bytes()).hexdigest(),
            'scientific_validation':'NOT_RUN','legacy_rod_fallback_enabled':False}
        (out/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps({'status':'PASS','output':str(out)},ensure_ascii=False),flush=True)
    finally:app.close()


if __name__=='__main__':main()
