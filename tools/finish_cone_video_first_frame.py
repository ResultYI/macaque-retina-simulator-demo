"""Reuse completed captures and compute only missing tiles of source frame zero."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.io import loadmat,savemat
from retina_platform.disk_frontend import _run_batch
from retina_sim.contracts import canonical_hash


def main(source):
    plan=json.loads((source/'capture_plan.json').read_bytes())
    profile=json.loads((source/'optics_config.json').read_bytes())['profile']
    out=source/'first_frame';out.mkdir(exist_ok=True);tasks=[];jobs=[]
    for original in plan['tasks']:
        if original['first']!=0:continue
        task=dict(original)
        if Path(task['output']+'.ready').is_file():
            task['reused_video_capture']=True
        else:
            tx,ty=task['tile_index'];directory=out/f'tile_{tx}_{ty}';directory.mkdir(exist_ok=True)
            c=json.loads((Path(task['output']).parent/'export.json').read_bytes())
            c.update(frame_indices=[1],window_edges_seconds=[0.,1/30])
            c['request_signature']=canonical_hash(dict(parent_request=task['request_signature'],first_frame_configuration=c))
            target=directory/'capture.mat';config=directory/'export.json'
            if config.exists():assert json.loads(config.read_bytes())==c
            else:config.write_text(json.dumps(c,indent=2))
            task.update(output=str(target),configuration=str(config),request_signature=c['request_signature'],reused_video_capture=False)
            if not Path(str(target)+'.ready').is_file():
                assert not target.exists(),'Incomplete capture retained; inspect before retry'
                jobs.append(dict(configuration=str(config),output=str(target)))
        tasks.append(task)
    assert len(tasks)==32
    (out/'plan.json').write_text(json.dumps(dict(tasks=tasks,requested_frames=1,source_time_seconds=0.,remaining_video_paused=True),indent=2))
    reused=sum(t['reused_video_capture'] for t in tasks)
    print(f'First frame: reuse {reused}/32 tiles; compute only {len(jobs)} missing tiles.',flush=True)
    def worker(i):
        owned=jobs[i::8]
        if owned:
            directory=out/f'worker_{i+1}';directory.mkdir(exist_ok=True)
            _run_batch(ROOT,profile,owned,directory,lambda text:print(f'Worker {i+1}: {text}',flush=True),None,
                max_threads=4,batch_function='export_cone_video_aperture_batch',tools_directory=source/'producer_source/tools')
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
    n=plan['geometry']['count'];counts=np.empty((1,n));background=np.empty(n);coverage=np.zeros(n,bool);controls=[]
    for task in tasks:
        assert Path(task['output']+'.ready').is_file()
        r=loadmat(task['output']);meta=json.loads(r['manifest_json'].item());rows=np.load(task['rows'])
        assert meta['request_signature']==task['request_signature'] and not coverage[rows].any()
        counts[0,rows]=r['expected_counts'][0];background[rows]=r['background_counts'].ravel();coverage[rows]=True
        if meta['native_control']:controls.append(meta['native_control'])
    assert coverage.all() and np.isfinite(counts).all() and np.all(counts>=0) and np.all(background>0)
    with np.load(source/'geometry.npz',allow_pickle=False) as g:diameters=g['apertures_um'].copy()
    signature=canonical_hash(dict(frame_tasks=tasks,geometry=plan['geometry']))
    meta=dict(request_signature=signature,phototransduction_applied=False,phototransduction_run=False,downstream_circuit_run=False,
        eye_movements=False,FIT_READY=False,background_qualified=False,native_controls=controls,source_time_seconds=0.,exposure_seconds=1/30)
    capture=out/'capture.mat';assert not capture.exists()
    savemat(capture,dict(manifest_json=json.dumps(meta),expected_counts=counts,background_counts=background,
        diameters_um=diameters,window_edges_seconds=np.array([0.,1/30])),do_compression=True)
    (out/'capture_complete.json').write_text(json.dumps(dict(meta,cones=n,reused_tiles=reused,newly_computed_tiles=len(jobs)),indent=2))
    hd=out/'HD';hd.mkdir(exist_ok=True);shutil.copy2(source/'HD/cone_raster.mat',hd/'cone_raster.mat')
    config=dict(input_mat=str(source/'input.mat'),request_signature=signature,output_directory=str(hd),high_definition=True,
        image_only=True,render_size=[16384,6144],render_dpi=300,render_font_size=32)
    config_path=hd/'export.json';config_path.write_text(json.dumps(config,indent=2))
    q=lambda p:str(p).replace("'","''")
    for folder in ('matlab_preferences','matlab_temp'):(hd/folder).mkdir(exist_ok=True)
    env=dict(os.environ,MATLAB_PREFDIR=str(hd/'matlab_preferences'),TEMP=str(hd/'matlab_temp'),TMP=str(hd/'matlab_temp'))
    with (hd/'render_stdout.log').open('wb') as stdout,(hd/'render_stderr.log').open('wb') as stderr:
        result=subprocess.run([profile['matlab_executable'],'-batch',"maxNumCompThreads(8);addpath('"+q(ROOT/'tools')+"');render_cone_video('"+q(config_path)+"','"+q(capture)+"');"],env=env,
            stdout=stdout,stderr=stderr,timeout=300,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert result.returncode==0,'First-frame render failed; captures retained'
    print((hd/'render_metadata.json').read_text(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    main(parser.parse_args().source.resolve())
