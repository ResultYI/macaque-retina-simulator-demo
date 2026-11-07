"""Capture a frame-aligned constant-FPS video clip on the existing cone mosaic."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.io import savemat, loadmat
from retina_platform.media import MediaSource
from retina_platform.iset_frontend import digest, read_capture, source_signature, run_matlab
from retina_sim.contracts import canonical_hash
from retina_sim.isetbio_geometry import cone_apertures
from tools.cone_mosaic_demo import load_geometry, sample_image, require


def frame_timeline(fps: float, frame_count: int, start: float, duration: float = 1.) -> tuple[np.ndarray, np.ndarray]:
    """Require complete source frames and preserve their native exposure intervals."""
    require(np.isfinite(fps) and fps > 0 and np.isfinite(start) and start >= 0, 'Positive FPS and nonnegative start required')
    require(np.isfinite(duration) and duration > 0, 'Positive finite clip duration required')
    count, first = int(round(duration*fps)), int(round(start*fps))
    require(abs(round(fps)-fps) < 1e-8 and abs(first-start*fps) < 1e-8 and
            abs(count-duration*fps) < 1e-8 and count >= 2,
            'This demo requires integer constant FPS and frame-aligned clip boundaries covering at least two frames')
    require(first+count <= frame_count, 'Video does not contain the requested full clip; no looping or padding')
    return np.arange(first, first+count), np.arange(count+1, dtype=float)/fps


def matlab_stage(profile: dict, directory: Path, expression: str, name: str) -> None:
    """Run one isolated, bounded artifact stage and retain its logs."""
    for folder in ('matlab_preferences','matlab_temp'):
        (directory/folder).mkdir(exist_ok=True)
    environment=dict(os.environ,MATLAB_PREFDIR=str(directory/'matlab_preferences'),
                     TEMP=str(directory/'matlab_temp'),TMP=str(directory/'matlab_temp'))
    with (directory/(name+'_stdout.log')).open('wb') as stdout,(directory/(name+'_stderr.log')).open('wb') as stderr:
        result=subprocess.run([profile['matlab_executable'],'-batch',expression],env=environment,
            stdout=stdout,stderr=stderr,timeout=300,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    require(result.returncode==0,name+' failed; preserve the stage outputs and inspect its logs')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media', type=Path, required=True)
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--duration', type=float, default=1.)
    parser.add_argument('--fixational-eye-movements', action='store_true')
    parser.add_argument('--eye-seed', type=int, default=20261028)
    parser.add_argument('--central-2deg', action='store_true', help='Existing fovea-centered radius 2 degree disk; fixed gaze only')
    parser.add_argument('--central-geometry-from', type=Path, help='Reuse saved central geometry after checking its source rules and master')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), 'Output exists; preserve it and choose a fresh directory')
    require(args.central_2deg or args.central_geometry_from is None,'Saved central geometry requires --central-2deg')
    profile, sources = source_signature(ROOT)
    if args.fixational_eye_movements:
        require(0<=args.eye_seed<2**32,'Eye movement seed must be a uint32 integer')
        eye_files=[ROOT/'tools/prepare_cone_eye_movements.m',ROOT/'tools/isetbio_compat/sgolayfilt.m']
        eye_files.extend((Path(profile['isetbio_root'])/'eyemovement').rglob('*.m'))
        sources.update({str(path):digest(path) for path in eye_files})
    for relative in ('tools/cone_video_demo.py', 'tools/render_cone_video.m',
                     'tools/cone_mosaic_demo.py', 'retina_platform/media.py',
                     'retina_sim/isetbio_geometry.py', 'retina_sim/atlas_population_network.py',
                     'retina_sim/atlas_sources.py'):
        path = ROOT/relative; sources[str(path)] = digest(path)
    if args.central_2deg:
        from tools.cone_central_video_demo import run
        run(args,profile,sources)
        return
    xy, ids, types, support, active, apertures, geometry, halo = load_geometry(ROOT)
    margin = float(apertures.max())/2000+.001
    bounds = np.array([xy[:,0].min()-margin, xy[:,0].max()+margin,
                       xy[:,1].min()-margin, xy[:,1].max()+margin])
    center = (bounds[[0,2]]+bounds[[1,3]])/2
    extent = bounds[[1,3]]-bounds[[0,2]]+2*halo
    cols, rows = np.ceil(extent/(profile['scene_sampling_um']*.001)).astype(int)+1
    require(max(rows,cols) <= profile['maximum_scene_samples_per_axis'], 'Existing scene sampling budget exceeded')
    pitch = max(extent[0]/(cols-1),extent[1]/(rows-1))
    x = (np.arange(cols)-(cols-1)/2)*pitch+center[0]
    y = ((rows-1)/2-np.arange(rows))*pitch+center[1]
    xx, yy = np.meshgrid(x,y)
    u = profile['reference_linear_value']
    grey = 12.92*u if u <= .0031308 else 1.055*u**(1/2.4)-.055
    media_sha = digest(args.media)
    frames, receipts = [], []
    eye=None; eye_metadata=None
    quote=lambda path:str(path).replace("'","''")
    with MediaSource(args.media) as media:
        require(media.info['kind']=='video', 'An actual video file is required')
        info = dict(media.info)
        indices, edges = frame_timeline(info['fps'],info['frame_count'],args.start,args.duration)
        if args.fixational_eye_movements:
            output.mkdir(parents=True)
            eye_config=output/'eye_config.json'; eye_path=output/'eye_movements.mat'
            eye_config.write_text(json.dumps(dict(profile=profile,duration_seconds=args.duration,
                frame_rate=info['fps'],seed=args.eye_seed)),encoding='utf8')
            expression="addpath('"+quote(ROOT/'tools')+"');prepare_cone_eye_movements('"+quote(eye_config)+"','"+quote(eye_path)+"');"
            matlab_stage(profile,output,expression,'eye_generation')
            eye=loadmat(eye_path); eye_metadata=json.loads(eye['parameters_json'].item())
            require(eye['em_positions_arcmin'].shape[0]==len(indices),'Eye/video time axes differ')
            shift_mm=np.max(np.abs(eye['em_positions_arcmin']),axis=(0,1))*.223/60
            extra=np.ceil(np.maximum(0.,shift_mm+.005-halo)/pitch).astype(int)
            # Add whole pixel pairs, preserving the old grid and scene registration in its center.
            cols+=2*extra[0]; rows+=2*extra[1]
            require(max(rows,cols)<=profile['maximum_scene_samples_per_axis'],'Eye path exceeds the optical grid budget')
            x=(np.arange(cols)-(cols-1)/2)*pitch+center[0]
            y=((rows-1)/2-np.arange(rows))*pitch+center[1]
            xx,yy=np.meshgrid(x,y)
        require(len(indices)*rows*cols*3*4 <= 128*1024**2, 'Optical RGB scene exceeds the existing 128 MiB budget')
        for index in indices:
            requested = float(index/info['fps'])
            rgb = media.frame_at(requested)
            require(not media.info['timestamp_fallback_used'] and
                    abs(media.last_timestamp_seconds-requested) < 1e-5,
                    'Decoded frame timestamps do not establish the assumed constant-FPS timeline')
            frames.append(sample_image(rgb,bounds,xx,yy,grey).astype(np.float32))
            receipts.append(dict(source_frame_index=int(index), source_time_seconds=requested,
                decoded_time_seconds=media.last_timestamp_seconds,
                decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
        info.update(media.info)
    require(digest(args.media)==media_sha, 'Video changed while decoding')
    require(len({r['decoded_rgb_sha256'] for r in receipts})>1, 'Requested clip has no changing decoded frames')
    rgb_frames = np.stack(frames,axis=3)
    del frames
    request = canonical_hash(dict(profile=profile,sources=sources,geometry=geometry,
        media_sha256=media_sha,start=args.start,duration=args.duration,receipts=receipts,edges=edges,bounds=bounds,
        eye_model=eye_metadata,eye_path_sha256=digest(output/'eye_movements.mat') if eye is not None else None))
    if eye is None: output.mkdir(parents=True)
    source_mat = output/'input.mat'
    input_data=dict(rgb_frames=rgb_frames,window_edges_seconds=edges,
        frame_indices=np.arange(1,len(indices)+1,dtype=float),reference_rgb=np.full(3,grey),
        native_apertures_um=cone_apertures(support)[active],
        support_is_mm=support,active_indices=active.astype(float)+1,is_mm=xy,
        types=np.array([{'L':1.,'M':2.,'S':3.}[t] for t in types]),image_center_deg=center/.223,
        plot_bounds_mm=bounds,plot_x_mm=x,plot_y_mm=y)
    if eye is not None:
        for key in ('em_positions_arcmin','subsample_times_seconds','frame_eye_midpoint_arcmin'):
            input_data[key]=eye[key]
        input_data['eye_time_seconds']=eye['time_seconds']
        input_data['eye_position_arcmin']=eye['eye_position_arcmin']
        input_data['microsaccade_onset_seconds']=eye['microsaccade_onset_seconds']
        probes=[]
        for kind in 'LMS':
            rows_for_type=np.flatnonzero(types==kind); ordered=rows_for_type[np.argsort(xy[rows_for_type,0])]
            probes.extend(ordered[np.round(np.array([.25,.5,.75])*(len(ordered)-1)).astype(int)].tolist())
        input_data['fine_probe_indices']=np.array(probes,dtype=float)+1
    savemat(source_mat,input_data,do_compression=True)
    focal = .223e-3/np.tan(np.pi/180)
    config = dict(profile=profile,input_mat=str(source_mat),
        patch_fov_deg=float(np.rad2deg(2*np.arctan(cols*pitch*.001/(2*focal)))),
        request_signature=request,background_signature=canonical_hash(dict(profile=profile,geometry=geometry,xy=xy)))
    config_path = output/'export.json'
    config_path.write_text(json.dumps(config,indent=2,allow_nan=False),encoding='utf8')
    capture = output/'capture.mat'
    print(f'Actual optical capture: {len(indices)} frames, {len(ids)} cones, {args.duration:.3f} s; no circuit.',flush=True)
    run_matlab(profile,ROOT,config_path,capture,lambda status:print(status,flush=True),None)
    raw, meta = read_capture(capture,request)
    require(raw['expected_counts'].shape==(len(indices),len(ids)), 'Capture frame/cone dimensions changed')
    require(np.array_equal(raw['window_edges_seconds'].reshape(-1),edges), 'Captured video timeline differs')
    require(np.allclose(raw['diameters_um'].reshape(-1),apertures,rtol=1e-11,atol=1e-9), 'Frozen apertures changed')
    require(not meta['phototransduction_applied'] and meta['photon_noise']=='none', 'Capture stage changed')
    require(meta.get('eye_movements_applied',False)==(eye is not None),'Requested eye movement stage was not applied')
    counts = raw['expected_counts']; rates = counts/np.diff(edges)[:,None]
    if eye is not None:
        fine=loadmat(str(capture)+'.eye_substeps.mat')
        probe_indices=input_data['fine_probe_indices'].astype(int)-1
        summed=fine['expected_counts_substeps'].reshape(len(indices),eye_metadata['subsamples_per_frame'],len(probe_indices)).sum(axis=1)
        require(np.allclose(summed,counts[:,probe_indices],rtol=1e-12,atol=1e-12),'Fine captures do not sum to video-frame captures')
        with (output/'eye_movement_path.csv').open('w',newline='',encoding='utf8') as stream:
            writer=csv.writer(stream);writer.writerow(['time_s','x_arcmin','y_arcmin','x_um','y_um'])
            for time,position in zip(eye['time_seconds'].reshape(-1),eye['eye_position_arcmin']):
                if time<=args.duration:
                    writer.writerow([time,*position,*(position*profile['microns_per_degree']/60)])
    require(np.any(np.ptp(rates,axis=0)>0), 'No temporal variation in actual captures')
    with (output/'cone_video_capture.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['video_frame_index','start_s','end_s','cell_id','type','expected_Rstar','expected_Rstar_per_cone_s'])
        for f,index in enumerate(indices):
            writer.writerows((int(index),edges[f],edges[f+1],ids[i],types[i],counts[f,i],rates[f,i]) for i in range(len(ids)))
    expression = "addpath('"+quote(ROOT/'tools')+"');render_cone_video('"+quote(config_path)+"','"+quote(capture)+"');"
    matlab_stage(profile,output,expression,'render')
    rendering = json.loads((output/'render_metadata.json').read_bytes())
    require(rendering['frame_count']==len(indices) and abs(rendering['duration_seconds']-args.duration)<1e-9,
            'Rendered video does not cover the requested duration')
    require(rendering['color_range'][0]==0 and rendering['color_range'][1]>=float(rates.max()), 'Video color scale clips physical rates')
    require(all(digest(Path(path))==value for path,value in sources.items()), 'Source changed during capture/render')
    require(digest(args.media)==media_sha, 'Source video changed during capture/render')
    snapshot = output/'producer_source'; snapshot.mkdir()
    for name,value in sources.items():
        path = Path(name)
        if path.is_relative_to(ROOT) and 'vendor_pinned' not in path.relative_to(ROOT).parts:
            target = snapshot/path.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target); require(digest(target)==value,'Producer snapshot changed')
    manifest = dict(schema='cone_video_capture_demo_v1',media=info,media_sha256=media_sha,
        start_seconds=args.start,duration_seconds=args.duration,frame_receipts=receipts,frame_count=len(indices),
        integration_seconds=1/info['fps'],display_hold='Each decoded RGB frame is held until the next source frame; no temporal interpolation',
        window_edges_seconds=edges.tolist(),cell_count=len(ids),type_counts={t:int(np.sum(types==t)) for t in 'LMS'},
        cone_identity_sha256=canonical_hash(dict(ids=ids,xy=xy,types=types.tolist())),geometry=geometry,
        source_hashes=sources,raw_capture=meta,rendering=rendering,fixed_gaze=eye is None,
        eye_movements=eye_metadata,
        temporal_probe_cell_ids=[ids[int(i)-1] for i in input_data['fine_probe_indices']] if eye is not None else [],
        per_frame_normalization=False,source_clip_physical_scene_calibration=False,
        units=dict(stored='expected R*/frame integration window',displayed='expected R*/cone/s'),
        phototransduction_run=False,downstream_circuit_run=False,physiological_validation=False,
        outputs={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.suffix in ('.mat','.csv','.mp4','.png','.pdf')})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    (output/'README.txt').write_text(
        'Natural video -> existing L/M/S cone mosaic -> expected light capture.\n'
        f'Source: {args.media.resolve()}\nClip: [{args.start:.6f}, {args.start+args.duration:.6f}) s; {len(indices)} frames at {info["fps"]} fps.\n'
        f'Each held display frame integrates for {1000/info["fps"]:.9g} ms. The raw unit is expected R*/frame; plots divide by this interval.\n'
        'Fixed absolute color scale across all frames; no per-frame/type normalization. No photon noise.\n'+
        ('Fixed gaze; no eye movements.\n' if eye is None else
         f'ISETBio drift + microsaccades; fixed seed {args.eye_seed}; {eye_metadata["subsamples_per_frame"]} midpoint substeps per video frame.\n'
         'The native 1 ms eye path is linearly resampled as in ISETBio. Numerical capture quadrature is finer than the model time grid.\n'
         'Macular pigment uses the existing ISETBio frame-mean-eye-position approximation; no per-substep macular update.\n'
         'Panel A shows the midpoint retinal image position. Panel C integrates the entire within-frame path. Cone geometry remains fixed.\n'
         'Extra reference-gray optical support covers the complete eye path; aperture coverage is checked before sampling.\n'
         'Fine temporal captures are saved for nine identified probe cones, alongside every cone\'s frame-integrated result.\n')+
        'English panel names, axes, legend and units only. No subtitles or explanatory figure captions.\n'
        'Uses the existing frozen cone geometry, LCD-Apple display, human MW/lens/macular and typed OS model.\n'
        'Video RGB is treated as a display stimulus; original scene radiance and camera exposure are not recovered.\n'
        'The video is one sequence of optical captures; no phototransduction, neural temporal adaptation, circuit, fitting or training.\n'
        'Decoded frame timestamps, source/producer hashes, optical input, raw captures, per-cell CSV and representative PNG/PDF frames are retained.\n',encoding='utf8')
    print(json.dumps(dict(output=str(output),frames=len(indices),cones=len(ids),duration_seconds=args.duration,color_range=rendering['color_range'])),flush=True)


if __name__=='__main__':
    main()
