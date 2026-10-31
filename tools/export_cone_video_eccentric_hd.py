"""Render all frozen 0-6 degree cone responses to a lossless RGB master and playback copy."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.cone_video_raster import prepare_raster


def prepare(source):
    out=source/'HD';out.mkdir(exist_ok=True)
    if not (out/'cone_raster.mat').exists():
        with np.load(source/'geometry.npz',allow_pickle=False) as data:
            prepare_raster(out/'cone_raster.mat',data['xy_mm'],data['apertures_um'],
                           [-1.338,1.338,-1.338,1.338],samples=4096,supersampling=2)
    return out


def render(source):
    out=prepare(source);capture=json.loads((source/'capture_complete.json').read_bytes())
    plan=json.loads((source/'capture_plan.json').read_bytes());runtime=json.loads((source/'codec_runtime.json').read_bytes())
    size=[16384,6144];master=out/'cone_video_4s_0_6deg_D_e_16K_RGB_lossless.mkv'
    preview=out/'cone_video_4s_0_6deg_D_e_4K_playback.mp4'
    if master.exists() or preview.exists():raise FileExistsError('Preserve existing video export')
    args=[runtime['ffmpeg'],'-hide_banner','-loglevel','warning','-nostdin','-n',
        '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{size[0]}x{size[1]}',
        '-framerate','30','-i','pipe:0',
        '-map','0:v:0','-c:v','ffv1','-level','3','-g','1','-slices','16','-slicecrc','1',
        '-threads','16','-pix_fmt','bgr0',str(master),
        '-map','0:v:0','-vf','scale=3840:1440:flags=lanczos','-c:v','libx264',
        '-preset','fast','-crf','12','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(preview)]
    config=dict(input_mat=str(source/'input.mat'),request_signature=capture['request_signature'],
        output_directory=str(out),high_definition=True,capture_blocks=capture['capture_blocks'],
        window_edges_seconds=plan['window_edges_seconds'],maximum_rate_Rstar_per_cone_s=capture['maximum_rate_Rstar_per_cone_s'],
        render_size=size,render_dpi=300,render_font_size=32,encoder_arguments=args,
        encoder_dll_paths=runtime['dll_directories'],rgb_master=str(master),playback_video=str(preview))
    config_path=out/'export.json';config_path.write_text(json.dumps(config,indent=2),encoding='utf8')
    p=json.loads((source/'optics_config.json').read_bytes())['profile']
    for folder in ('matlab_preferences','matlab_temp'):(out/folder).mkdir(exist_ok=True)
    environment=dict(os.environ,MATLAB_PREFDIR=str(out/'matlab_preferences'),TEMP=str(out/'matlab_temp'),TMP=str(out/'matlab_temp'))
    q=lambda path:str(path).replace("'","''")
    expression="maxNumCompThreads(8);addpath('"+q(ROOT/'tools')+"');render_cone_video('"+q(config_path)+"','"+q(Path(capture['capture_blocks'][0]['path']))+"');"
    with (out/'render_stdout.log').open('wb') as stdout,(out/'render_stderr.log').open('wb') as stderr:
        result=subprocess.run([p['matlab_executable'],'-batch',expression],env=environment,
            stdout=stdout,stderr=stderr,timeout=1200,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:raise RuntimeError('Renderer failed; all captures and logs retained: '+str(out))
    print((out/'render_metadata.json').read_text(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if args.prepare_only:print(prepare(args.source.resolve()))
    else:render(args.source.resolve())
