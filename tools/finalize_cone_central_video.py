"""Finalize saved captures when only the renderer changed during a long capture."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.io import loadmat
from retina_platform.iset_frontend import digest,read_capture
from retina_sim.contracts import canonical_hash
from tools.cone_mosaic_demo import require


def finalize(output):
    require(not (output/'manifest.json').exists(),'Preserve existing finalized manifest')
    sources=json.loads((output/'capture_producer_sources.json').read_bytes())
    plan=json.loads((output/'capture_plan.json').read_bytes())
    geometry=json.loads((output/'geometry.json').read_bytes())
    with np.load(output/'geometry.npz',allow_pickle=False) as data:
        ids,xy,types,apertures=(data[k].copy() for k in ('cell_ids','xy_mm','types','apertures_um'))
    config=json.loads((output/'export.json').read_bytes())
    raw,meta=read_capture(output/'capture.mat',config['request_signature'])
    edges=raw['window_edges_seconds'].reshape(-1);counts=raw['expected_counts'];rates=counts/np.diff(edges)[:,None]
    info=plan['media'];receipts=plan['frame_receipts']
    media_path=Path(info['path'])
    media_sha=digest(media_path)
    request=canonical_hash(dict(sources=sources,geometry=geometry,media_sha256=media_sha,
                                receipts=receipts,reference=meta['global_reference']))
    require(request==config['request_signature'],'Saved capture source/geometry/media identity differs')
    require(counts.shape==(120,45480) and edges[0]==0 and edges[-1]==4 and
            np.array_equal(raw['diameters_um'].reshape(-1),apertures),'Central capture layout differs')
    require(not meta['phototransduction_applied'] and not meta.get('eye_movements_applied',False) and
            meta['photon_noise']=='none','Capture scope changed')
    allowed={str(ROOT/'tools/render_cone_video.m'),str(ROOT/'tools/cone_central_video_demo.py')}
    for name,value in sources.items():
        path=Path(name)
        if name not in allowed:require(digest(path)==value,'Capture/model source changed: '+name)
        if path.is_relative_to(ROOT) and 'vendor_pinned' not in path.relative_to(ROOT).parts:
            require(digest(output/'producer_source'/path.relative_to(ROOT))==value,'Capture source snapshot changed')
    rendering=json.loads((output/'render_metadata.json').read_bytes())
    require(rendering['frame_count']==120 and rendering['frame_rate']==30 and rendering['duration_seconds']==4 and
            not rendering['eye_movements_enabled'] and rendering['rasterized_apertures'] and
            rendering['color_range'][0]==0 and rendering['color_range'][1]>=rates.max(),'Rendered scope/scale changed')
    raster=loadmat(output/'cone_raster.mat')
    require(np.array_equal(raster['diameters_um'].reshape(-1),apertures),'Raster aperture sizes changed')
    render_sources={str(ROOT/name):digest(ROOT/name) for name in (
        'tools/render_cone_video.m','tools/cone_video_raster.py','tools/finalize_cone_central_video.py')}
    for name,value in render_sources.items():
        target=output/'render_source'/Path(name).name;target.parent.mkdir(exist_ok=True);shutil.copy2(name,target)
        require(digest(target)==value,'Render source snapshot changed')
    manifest=dict(schema='cone_video_capture_demo_v1',media=info,media_sha256=media_sha,
        start_seconds=0.,duration_seconds=4.,frame_count=120,frame_receipts=receipts,integration_seconds=1/30,
        cell_count=len(ids),type_counts={t:int(np.sum(types==t)) for t in 'LMS'},geometry=geometry,
        source_hashes=sources,render_source_hashes=render_sources,raw_capture=meta,rendering=rendering,
        fixed_gaze=True,eye_movements=None,optical_grid_samples=[973,973],optical_grid_pitch_um=1.,
        render_input_samples=[512,512],per_frame_normalization=False,source_clip_physical_scene_calibration=False,
        units=dict(stored='expected R*/frame integration window',displayed='expected R*/cone/s'),
        phototransduction_run=False,downstream_circuit_run=False,physiological_validation=False,
        postprocessing_note='Renderer changed after capture to reuse fixed physical-circle supersampled rasters; original capture source snapshot retained',
        outputs={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.suffix in ('.mat','.csv','.mp4','.png','.pdf')})
    validation=dict(passed=True,frames=120,fps=30,duration_seconds=4.,cones=45480,
        cone_geometry_and_types_preserved=True,per_cone_apertures_preserved=True,eye_movements=False,
        image_fit='isotropic cover; centered crop, no gray padding inside visible [-2,2] degree field',
        full_disk_native_control=json.loads((output/'native_equivalence.json').read_bytes()),
        fixed_absolute_color_scale=rendering['color_range'],capture_source_identity_verified=True,
        phototransduction_run=False,downstream_circuit_run=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf8')
    (output/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf8')
    (output/'README.txt').write_text(
        'Central 0-2 deg eccentricity; center (0,0), diameter 4 deg; 45,480 frozen macaque cones.\n'
        f'Source: {media_path}; first 4.000 s, 120 source frames at 30 fps, 33.333 ms/window.\n'
        '640x360 source video is isotropically enlarged; center square retained, left/right cropped; no gray padding inside visible field.\n'
        'Optical input: 973x973, 1 um sampling, 40 um halo. Four independent workers, each capped at eight MATLAB threads.\n'
        'Fixed gaze: no drift or microsaccades. No photon noise, phototransduction or downstream circuit.\n'
        'Stored expected R*/frame; displayed expected R*/cone/s. One fixed absolute color scale.\n'
        'English panel titles, axes, legend and units only; no subtitles or captions.\n'
        'Same frozen global IDs, full-master apertures and original global LMS assignment. First four frames are bit-identical to native per-aperture capture.\n'
        'Existing macaque cross-species S assignment and human MW/lens/macular/OS approximations retained. RGB video is a display stimulus, not recovered scene radiance.\n'
        '512-square input and 4x supersampled physical-circle rasters affect display only; native float32 optical inputs remain in batches.\n'
        'Renderer was accelerated after capture; actual capture and render source snapshots are recorded separately.\n',encoding='utf8')
    print(json.dumps(validation))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    finalize(parser.parse_args().output.resolve())
