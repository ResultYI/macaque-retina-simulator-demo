"""Fixed-gaze central 0-2 degree capture using the existing frozen macaque master."""
import csv
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.ndimage import map_coordinates

from retina_platform.disk_frontend import _capture_request, _run_batch, _finish_capture
from retina_platform.iset_frontend import BUDGET, digest, read_capture
from retina_platform.media import MediaSource
from retina_sim.contracts import canonical_hash
from tools.cone_mosaic_demo import ROOT, load_central_geometry, load_saved_central_geometry, require
from tools.cone_video_demo import frame_timeline, matlab_stage
from tools.cone_video_raster import prepare_raster


def sample_cover(rgb, bounds, x, y, grey):
    """Fill the visible field without stretching; keep gray only beyond the source image."""
    height,width=rgb.shape[:2]
    x0,x1,y0,y1=bounds
    pitch=max((x1-x0)/(width-1),(y1-y0)/(height-1))
    row=(height-1)/2-(y-(y0+y1)/2)/pitch
    col=(width-1)/2+(x-(x0+x1)/2)/pitch
    row=np.where(np.abs(row)<1e-8,0,np.where(np.abs(row-(height-1))<1e-8,height-1,row))
    col=np.where(np.abs(col)<1e-8,0,np.where(np.abs(col-(width-1))<1e-8,width-1,col))
    return np.stack([map_coordinates(rgb[:,:,k].astype(float)/255,[row,col],order=1,
                     mode='constant',cval=grey,prefilter=False) for k in range(3)],axis=-1)


def run(args, profile, sources):
    require(not args.fixational_eye_movements, 'This central-disk run requires fixed gaze')
    output = args.output.resolve()
    print('Loading the existing central mosaic and its global L/M/S label domains.',flush=True)
    xy, ids, types, apertures, geometry = (load_saved_central_geometry(ROOT,args.central_geometry_from)
        if args.central_geometry_from else load_central_geometry(ROOT))
    for name in ('tools/cone_central_video_demo.py', 'tools/export_disk_mixed_reference.m',
                 'tools/export_disk_mixed_batch.m', 'retina_platform/disk_frontend.py',
                 'retina_sim/source_atlas.py', 'retina_sim/atlas_s_partition.py', 'retina_sim/geometry.py',
                 'tools/cone_video_raster.py'):
        sources[str(ROOT/name)] = digest(ROOT/name)
    output.mkdir(parents=True)
    (output/'geometry.json').write_text(json.dumps(geometry,indent=2),encoding='utf8')
    np.savez_compressed(output/'geometry.npz',xy_mm=xy,cell_ids=np.asarray(ids),types=types,apertures_um=apertures)
    bounds = np.array([-.446,.446,-.446,.446])
    pitch = profile['scene_sampling_um']*.001
    samples = int(round((.892+.08)/pitch))+1
    require(samples <= profile['maximum_scene_samples_per_axis'], 'Central disk exceeds optical sampling budget')
    axis = (np.arange(samples)-(samples-1)/2)*pitch
    xx, yy = np.meshgrid(axis,-axis)
    plot_axis = np.linspace(-.446,.446,512)
    px, py = np.meshgrid(plot_axis,-plot_axis)
    u = profile['reference_linear_value']
    grey = 12.92*u if u <= .0031308 else 1.055*u**(1/2.4)-.055
    reference = dict(mode='one_global_foveal_LMS_reference_v1', reference_position_mm=[0.,0.],
                     reference_diameter_um=float(apertures.mean()), blur_um=float(apertures.mean()),
                     reference_illuminance_td=100., physiological_calibration=False,
                     geometry_sha256=canonical_hash(geometry))
    reference['definition_sha256'] = canonical_hash(reference)
    focal = .223e-3/np.tan(np.pi/180)
    fov = float(np.rad2deg(2*np.arctan(samples*pitch*.001/(2*focal))))
    data = dict(rgb_frames=np.full((samples,samples,3,1),grey,dtype=np.float32),
                window_edges_seconds=np.array([0.,1/30]),frame_indices=np.ones(1), reference_rgb=np.full(3,grey),
                is_mm=np.zeros((3,2)), types=np.array([1.,2.,3.]),
                frozen_apertures_um=np.full(3,apertures.mean()),image_center_deg=np.zeros(2))
    export = dict(profile=profile,global_reference=reference,patch_fov_deg=fov,
                  request_signature=canonical_hash(dict(reference=reference,sources=sources)),
                  background_signature=reference['definition_sha256'])
    print('Preparing one fixed foveal radiance/LMS reference.',flush=True)
    (_, ref_meta), _ = _capture_request(ROOT,profile,output/'reference',data,export,
                                       on_status=lambda text:print(text,flush=True))
    reference.update(fixed_radiance_gain=ref_meta['fixed_radiance_gain'],
                     reference_energy_spectrum_hex=ref_meta['reference_energy_spectrum_hex'],
                     q_ref_LMS_rstar_s=ref_meta['q_ref_LMS_rstar_s'])
    media_sha = digest(args.media)
    jobs, tasks, receipts, plot_frames, batch_frames = [], [], [], [], []
    with MediaSource(args.media) as media:
        require(media.info['kind']=='video','Actual video required')
        info = dict(media.info)
        indices, edges = frame_timeline(info['fps'],info['frame_count'],args.start,args.duration)
        require(len(indices)*512*512*3 <= BUDGET,'Rendered input array exceeds memory budget')
        for f, index in enumerate(indices):
            requested = float(index/info['fps']); rgb = media.frame_at(requested)
            require(not media.info['timestamp_fallback_used'] and abs(media.last_timestamp_seconds-requested)<1e-5,
                    'Decoded timestamps differ from the source timeline')
            receipts.append(dict(source_frame_index=int(index),source_time_seconds=requested,
                decoded_time_seconds=media.last_timestamp_seconds,decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
            plot_frames.append(np.rint(sample_cover(rgb,bounds,px,py,grey)*255).astype(np.uint8))
            batch_frames.append(sample_cover(rgb,bounds,xx,yy,grey).astype(np.float32))
            if len(batch_frames)==4 or f==len(indices)-1:
                first = f+1-len(batch_frames)
                frames = np.stack(batch_frames,axis=3)
                require(frames.nbytes<=BUDGET,'Optical RGB batch exceeds unchanged memory budget')
                batch_data = dict(rgb_frames=frames,window_edges_seconds=edges[first:f+2],
                    frame_indices=np.arange(1,len(batch_frames)+1,dtype=float),reference_rgb=np.full(3,grey),
                    is_mm=xy,types=np.array([{'L':1.,'M':2.,'S':3.}[t] for t in types]),
                    frozen_apertures_um=apertures,image_center_deg=np.zeros(2))
                signature = canonical_hash(dict(sources=sources,reference=reference,geometry=geometry,
                    media_sha256=media_sha,receipts=receipts[first:f+1],bounds=bounds,edges=edges[first:f+2]))
                batch_export = dict(export,global_reference=reference,request_signature=signature)
                directory = output/'batches'/f'frames_{first:03d}_{f:03d}'
                (_, pending), _ = _capture_request(ROOT,profile,directory,batch_data,batch_export,defer_jobs=jobs)
                tasks.append((directory,pending,first,f+1)); batch_frames.clear()
        info.update(media.info)
    require(digest(args.media)==media_sha and len({r['decoded_rgb_sha256'] for r in receipts})>1,'Source changed or clip is static')
    (output/'capture_plan.json').write_text(json.dumps(dict(jobs=jobs,geometry=geometry,
        input_sampling=dict(samples=samples,pitch_um=pitch*1000,batch_frames=4,
                            image_fit='isotropic cover; centered crop, no gray in visible field'),
        media=info,frame_receipts=receipts,eye_movements_enabled=False),indent=2),encoding='utf8')
    print(f'Capturing {len(indices)} frames on {len(ids)} cones; {len(jobs)} bounded batches, fixed gaze.',flush=True)
    worker_jobs=[jobs[i::4] for i in range(4)]
    def capture_worker(index):
        directory=output/f'worker_{index+1}';directory.mkdir()
        _run_batch(ROOT,profile,worker_jobs[index],directory,
                   lambda text:print(f'Worker {index+1}: {text}',flush=True),None,max_threads=8)
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(capture_worker,range(4)))
    counts = np.empty((len(indices),len(ids)))
    background = meta = background_dt = None
    for directory,pending,first,end in tasks:
        raw, current = _finish_capture(directory,pending['pending_path'],pending['signature'])
        require(np.array_equal(raw['window_edges_seconds'].reshape(-1),edges[first:end+1]),'Batch time axis changed')
        require(np.array_equal(raw['diameters_um'].reshape(-1),apertures),'Frozen apertures changed')
        require(current['fixed_radiance_gain']==reference['fixed_radiance_gain'] and
                np.array_equal(raw['q_ref_LMS'].reshape(-1),np.asarray(reference['q_ref_LMS_rstar_s'])), 'Batch reference changed')
        batch_dt=float(np.diff(raw['window_edges_seconds'].reshape(-1))[0])
        if background is None: background=raw['background_counts']; meta=current; background_dt=batch_dt
        require(np.allclose(raw['background_counts']/batch_dt,background/background_dt,rtol=1e-12,atol=0),
                'Background rate changed between temporal batches')
        counts[first:end]=raw['expected_counts']
    require(np.all(np.isfinite(counts)) and np.all(counts>=0) and np.any(np.ptp(counts,axis=0)>0),'Invalid or static captures')
    request=canonical_hash(dict(sources=sources,geometry=geometry,media_sha256=media_sha,receipts=receipts,reference=reference))
    meta.update(request_signature=request,eye_movements_applied=False,temporal_batches=len(tasks),
                global_reference=reference,scene_samples=[samples,samples],fixed_gaze=True)
    source_mat=output/'input.mat'; capture=output/'capture.mat'
    savemat(source_mat,dict(rgb_frames=np.stack(plot_frames,axis=3),is_mm=xy,
        types=np.array([{'L':1.,'M':2.,'S':3.}[t] for t in types]),plot_bounds_mm=bounds,
        plot_x_mm=plot_axis,plot_y_mm=-plot_axis),do_compression=True)
    savemat(capture,dict(manifest_json=json.dumps(meta),expected_counts=counts,background_counts=background,
        q_ref_LMS=np.asarray(reference['q_ref_LMS_rstar_s']),window_edges_seconds=edges,diameters_um=apertures),do_compression=True)
    read_capture(capture,request)
    config_path=output/'export.json'
    config_path.write_text(json.dumps(dict(input_mat=str(source_mat),request_signature=request),indent=2),encoding='utf8')
    rates=counts/np.diff(edges)[:,None]
    with (output/'cone_geometry.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream);writer.writerow(['cell_id','type','x_mm','y_mm','eccentricity_deg','aperture_diameter_um'])
        writer.writerows((ids[i],types[i],*xy[i],np.linalg.norm(xy[i])/.223,apertures[i]) for i in range(len(ids)))
    with (output/'cone_video_capture.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream);writer.writerow(['video_frame_index','start_s','end_s','cell_id','type','expected_Rstar','expected_Rstar_per_cone_s'])
        for f,index in enumerate(indices):
            writer.writerows((int(index),edges[f],edges[f+1],ids[i],types[i],counts[f,i],rates[f,i]) for i in range(len(ids)))
    prepare_raster(output/'cone_raster.mat',xy,apertures,bounds)
    quote=lambda path:str(path).replace("'","''")
    expression="addpath('"+quote(ROOT/'tools')+"');render_cone_video('"+quote(config_path)+"','"+quote(capture)+"');"
    matlab_stage(profile,output,expression,'render')
    rendering=json.loads((output/'render_metadata.json').read_bytes())
    require(rendering['frame_count']==len(indices) and rendering['duration_seconds']==args.duration and
            not rendering['eye_movements_enabled'] and rendering['color_range'][1]>=rates.max(),'Rendered schedule/scale differs')
    require(all(digest(Path(path))==value for path,value in sources.items()) and digest(args.media)==media_sha,'Sources changed')
    snapshot=output/'producer_source'
    for name,value in sources.items():
        path=Path(name)
        if path.is_relative_to(ROOT) and 'vendor_pinned' not in path.relative_to(ROOT).parts:
            target=snapshot/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            require(digest(target)==value,'Producer snapshot changed')
    manifest=dict(schema='cone_video_capture_demo_v1',media=info,media_sha256=media_sha,
        start_seconds=args.start,duration_seconds=args.duration,frame_count=len(indices),frame_receipts=receipts,
        integration_seconds=1/info['fps'],cell_count=len(ids),type_counts={t:int(np.sum(types==t)) for t in 'LMS'},
        geometry=geometry,source_hashes=sources,raw_capture=meta,rendering=rendering,fixed_gaze=True,eye_movements=None,
        optical_grid_samples=[samples,samples],optical_grid_pitch_um=pitch*1000,render_input_samples=[512,512],
        per_frame_normalization=False,source_clip_physical_scene_calibration=False,
        phototransduction_run=False,downstream_circuit_run=False,physiological_validation=False,
        outputs={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.suffix in ('.mat','.csv','.mp4','.png','.pdf')})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf8')
    (output/'README.txt').write_text(
        f'Central disk: 0-2 deg eccentricity; center (0,0), diameter 4 deg; {len(ids)} existing frozen cones.\n'
        f'Source: {args.media.resolve()}; [{args.start},{args.start+args.duration}) s, {len(indices)} frames at {info["fps"]} fps.\n'
        f'Optical input: {samples} x {samples}, {pitch*1000:g} um sampling, 40 um halo; four frames per bounded batch.\n'
        'Fixed gaze; no drift or microsaccades. No photon noise, phototransduction or downstream circuit.\n'
        'Video isotropically enlarged to cover [-2,2] deg square; left/right center crop, no gray in visible input.\n'
        'Off-screen optical halo uses source pixels where available and reference gray beyond source boundaries.\n'
        'English panel titles, axes, legend and units only; no subtitles/captions. One absolute color scale for every frame.\n'
        'Stored expected R*/frame; displayed expected R*/cone/s. Per-cell apertures from complete master support.\n'
        'Same global-ID/macro-domain LMS assignment as the earlier local patch; macaque cross-species S rule retained.\n'
        'LCD-Apple and human MW/lens/macular/OS approximations retained; RGB video is a display stimulus, not recovered scene radiance.\n'
        '512-square uint8 input is for rendering only; native float32 optical inputs are retained per batch.\n',encoding='utf8')
    print(json.dumps(dict(output=str(output),frames=len(indices),cones=len(ids),duration_seconds=args.duration,
                          color_range=rendering['color_range'],fixed_gaze=True)),flush=True)
