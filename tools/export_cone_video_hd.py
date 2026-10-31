"""Re-export saved cone captures in 4K with an uncompressed RGB master."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.io import loadmat
from tools.cone_video_raster import prepare_raster


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('Preserve existing exports: '+str(output))
    saved=json.loads((source/'export.json').read_bytes())
    data=loadmat(source/'input.mat',variable_names=['plot_bounds_mm'])
    with np.load(source/'geometry.npz',allow_pickle=False) as g:
        xy=g['xy_mm'].copy();diameters=g['apertures_um'].copy()
    output.mkdir(parents=True)
    prepare_raster(output/'cone_raster.mat',xy,diameters,data['plot_bounds_mm'],samples=1536,supersampling=2)
    config=dict(input_mat=str(source/'input.mat'),request_signature=saved['request_signature'],
                output_directory=str(output),high_definition=True)
    config_path=output/'export.json';config_path.write_text(json.dumps(config,indent=2),encoding='utf8')
    profile=json.loads((ROOT/'configurations/iset_lcd_apple_mw_v1.json').read_bytes())
    for folder in ('matlab_preferences','matlab_temp'):(output/folder).mkdir()
    environment=dict(os.environ,MATLAB_PREFDIR=str(output/'matlab_preferences'),
                     TEMP=str(output/'matlab_temp'),TMP=str(output/'matlab_temp'))
    q=lambda path:str(path).replace("'","''")
    expression="addpath('"+q(ROOT/'tools')+"');render_cone_video('"+q(config_path)+"','"+q(source/'capture.mat')+"');"
    print('Rendering existing 120 cone-capture frames; no optical computation.',flush=True)
    with (output/'render_stdout.log').open('wb') as stdout,(output/'render_stderr.log').open('wb') as stderr:
        result=subprocess.run([profile['matlab_executable'],'-batch',expression],env=environment,
            stdout=stdout,stderr=stderr,timeout=600,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:raise RuntimeError('Render failed; outputs retained: '+str(output))
    metadata=json.loads((output/'render_metadata.json').read_bytes())
    (output/'README.txt').write_text(
        'Render-only export from existing saved cone captures; no optical calculation was run.\n'
        f'Source results: {source}\n'
        '3840x1440; 120 frames, 30 fps, 4 seconds; central 0-2 deg; no eye movements.\n'
        'RGB AVI master preserves the exact rendered RGB pixels without chroma subsampling or lossy compression.\n'
        'MP4 is a compatible playback copy; use the RGB master or PNG for individual-cone color detail.\n'
        '1536-square physical-aperture layers with 2x supersampling; English panel titles/axes/units only, no subtitles/captions.\n'
        'Fixed color scale 0-15000 expected R*/cone/s; original model assumptions and source image detail retained.\n'
        'Old results and input/capture files are unchanged. No additional validation pass was run, as requested.\n',encoding='utf8')
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
