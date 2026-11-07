"""Full 0-6 degree frozen-cone video using native ISETBio Cone apertures."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.io import loadmat,savemat
from retina_platform.disk_frontend import _run_batch
from retina_platform.iset_frontend import BUDGET,digest
from retina_platform.media import MediaSource
from retina_sim.atlas import global_cell_id
from retina_sim.cone_aperture_eccentricity import resolve_apertures
from retina_sim.contracts import canonical_hash
from retina_sim.disk_input import optical_tiles
from tools.cone_video_demo import frame_timeline


def prepare(out,media_path):
    if (out/'capture_plan.json').exists():raise FileExistsError('Preserve existing capture plan')
    profile=json.loads((ROOT/'configurations/iset_lcd_apple_mw_v1.json').read_bytes())
    aperture=json.loads((ROOT/'configurations/isetbio_native_cone_aperture_v1.json').read_bytes())
    table_path=ROOT/'data/atlas/human_disk3mm_typed_tables_v1/cones/manifest.json'
    table=json.loads(table_path.read_bytes());arrays={}
    for name,binding in [('xy',table['coordinates']['rf']),('types',table['extras']['subtype']),('source_keys',table['source_keys'])]:
        path=ROOT/binding['path'];assert digest(path)==binding['sha256'];arrays[name]=np.load(path,mmap_mode='r')
    selected=np.flatnonzero(np.linalg.norm(arrays['xy'],axis=1)<=1.338)
    xy=np.array(arrays['xy'][selected]);types=arrays['types'][selected].astype(str)
    metadata=table['metadata']
    ids=[global_cell_id(metadata['atlas_id'],metadata['atlas_version'],metadata['source_version'],
                        'cones',str(arrays['source_keys'][i])) for i in selected]
    resolved=resolve_apertures(ids,types,xy,aperture,support_positions_mm=arrays['xy'])
    assert len(ids)==190498 and np.all(resolved['eccentricity_degrees']<=6.)
    np.savez_compressed(out/'geometry.npz',xy_mm=xy,cell_ids=np.asarray(ids),types=types,
        apertures_um=resolved['reference_diameter_um'],
        effective_area_m2=resolved['effective_area_m2'],eccentricity_deg=resolved['eccentricity_degrees'],
        evidence_class=resolved['evidence_class'],source_table_rows=selected)
    geometry=dict(center_deg=[0.,0.],radius_deg=6.,diameter_deg=12.,count=len(ids),
        type_counts={k:int(np.sum(types==k)) for k in 'LMS'},geometry_route=table['source_bindings']['cone_geometry_route'],
        density_species=table['source_bindings']['density_species'],S_rule=table['source_bindings']['S_rule'],
        frozen_table_manifest_sha256=digest(table_path),aperture_profile_sha256=canonical_hash(aperture),
        diameter_range_um=[float(resolved['reference_diameter_um'].min()),float(resolved['reference_diameter_um'].max())],
        aperture='Native ISETBio Pillbox aperture and area; exactly equal native kernels share sampling',
        aperture_evidence='Pinned cMosaic.computeConeApertures on complete source support',
        no_geometry_regeneration=True,physiological_validation=False)
    (out/'geometry.json').write_text(json.dumps(geometry,indent=2),encoding='utf8')
    media_sha=digest(media_path);frames=[];receipts=[]
    with MediaSource(media_path) as media:
        info=dict(media.info);indices,edges=frame_timeline(info['fps'],info['frame_count'],0.,4.)
        assert info['fps']==30 and len(indices)==120
        for i in indices:
            rgb=media.frame_at(float(i/30));assert not media.info['timestamp_fallback_used']
            assert abs(media.last_timestamp_seconds-i/30)<1e-5
            frames.append(rgb.copy());receipts.append(dict(source_frame_index=int(i),source_time_seconds=float(i/30),
                decoded_time_seconds=media.last_timestamp_seconds,decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
        info.update(media.info)
    bank=np.stack(frames,axis=3);assert bank.nbytes<=BUDGET and digest(media_path)==media_sha
    savemat(out/'source_rgb.mat',dict(source_rgb=bank),do_compression=True)
    bounds=np.array([-1.338,1.338,-1.338,1.338]);height,width=bank.shape[:2]
    assert (height,width)==(360,640)
    display=bank[:,140:500,:,:]
    savemat(out/'input.mat',dict(rgb_frames=display,is_mm=xy,types=np.array([{'L':1.,'M':2.,'S':3.}[t] for t in types]),
        plot_bounds_mm=bounds,plot_x_mm=np.linspace(-1.338,1.338,360),plot_y_mm=np.linspace(1.338,-1.338,360)),do_compression=True)
    source_files=['tools/export_cone_video_aperture_tile.m','tools/export_cone_video_aperture_batch.m',
        'tools/cone_video_cached_oi.m','tools/cone_video_gaussian_sampling.m','tools/cone_optics_mosaic.m',
        'tools/matlab/sample_cone_aperture_zones.m','tools/cone_native_aperture.m',
        'tools/cone_video_eccentric_demo.py',
        'retina_sim/cone_aperture_eccentricity.py','retina_sim/native_cone_apertures.py',
        'tools/native_cone_kernel_groups.m','tools/native_cone_apertures_from_positions.m','retina_platform/disk_frontend.py',
        'configurations/iset_lcd_apple_mw_v1.json','configurations/isetbio_native_cone_aperture_v1.json']
    sources={name:digest(ROOT/name) for name in source_files}
    snapshot=out/'producer_source'
    for name,value in sources.items():
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target);assert digest(target)==value
    optics=json.loads((out/'optics_config.json').read_bytes());jobs=[];tasks=[];tiles=optical_tiles(xy)
    for tile in tiles:
        rows=tile['indices'];tx,ty=tile['tile_index'];directory=out/'tiles'/f'tile_{tx}_{ty}';directory.mkdir(parents=True)
        np.save(directory/'global_rows.npy',rows,allow_pickle=False)
        data=dict(is_mm=xy[rows],types=np.array([{'L':1.,'M':2.,'S':3.}[t] for t in types[rows]]),
            reference_diameter_um=resolved['reference_diameter_um'][rows],zone_diameter_um=resolved['zone_diameter_um'][rows])
        savemat(directory/'geometry.mat',data,do_compression=True)
        tile_signature=canonical_hash(dict(sources=sources,geometry=geometry,tile_index=[tx,ty],rows=rows))
        for first in range(0,120,16):
            stop=min(first+16,120);destination=directory/f'frames_{first:03d}_{stop-1:03d}';destination.mkdir()
            signature=canonical_hash(dict(tile=tile_signature,media_sha256=media_sha,first=first,stop=stop))
            config=dict(aperture_model=aperture['profile_id'],profile=profile,optics_cache=str(out/'optics_cache'),source_bank=str(out/'source_rgb.mat'),
                geometry_mat=str(directory/'geometry.mat'),center_mm=tile['center_mm'].tolist(),plot_bounds_mm=bounds.tolist(),
                reference_rgb=optics['reference_rgb'],frame_indices=list(range(first+1,stop+1)),
                window_edges_seconds=edges[first:stop+1].tolist(),native_control=(first==0 and [tx,ty] in ([0,0],[2,0])),
                request_signature=signature,tile_signature=tile_signature,aperture_profile_sha256=canonical_hash(aperture))
            config_path=destination/'export.json';config_path.write_text(json.dumps(config,indent=2),encoding='utf8')
            capture=destination/'capture.mat';jobs.append(dict(configuration=str(config_path),output=str(capture)))
            tasks.append(dict(tile_index=[tx,ty],rows=str(directory/'global_rows.npy'),first=first,stop=stop,
                              output=str(capture),request_signature=signature))
    plan=dict(schema='cone_eccentric_video_plan_v1',media=info,media_sha256=media_sha,frame_receipts=receipts,
        geometry=geometry,sources=sources,jobs=jobs,tasks=tasks,window_edges_seconds=edges.tolist(),
        workers=8,threads_per_worker=4,scene_samples=593,scene_sampling_um=1.,
        phototransduction_run=False,downstream_circuit_run=False,eye_movements=False,FIT_READY=False,background_qualified=False)
    (out/'capture_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
    print(f'Prepared {len(ids)} frozen cones, {len(tiles)} tiles, {len(jobs)} bounded jobs.',flush=True)
    return plan


def capture(out):
    plan=json.loads((out/'capture_plan.json').read_bytes());profile=json.loads((out/'optics_config.json').read_bytes())['profile']
    while not (out/'optics_cache/metadata.json').exists():time.sleep(1)
    tile_keys=list(dict.fromkeys(tuple(task['tile_index']) for task in plan['tasks']))
    def worker(index):
        owned=set(tile_keys[index::8]);jobs=[job for job,task in zip(plan['jobs'],plan['tasks'])
            if tuple(task['tile_index']) in owned and not Path(job['output']+'.ready').exists()]
        if not jobs:return
        directory=out/f'worker_{index+1}';directory.mkdir(exist_ok=True)
        _run_batch(ROOT,profile,jobs,directory,lambda text:print(f'Worker {index+1}: {text}',flush=True),None,
            max_threads=4,batch_function='export_cone_video_aperture_batch',tools_directory=out/'producer_source/tools')
    with ThreadPoolExecutor(max_workers=8) as workers:list(workers.map(worker,range(8)))
    collect(out)


def collect(out):
    plan=json.loads((out/'capture_plan.json').read_bytes());n=plan['geometry']['count'];edges=np.asarray(plan['window_edges_seconds'])
    target=out/'expected_counts.npy';assert not target.exists()
    counts=np.lib.format.open_memmap(target,mode='w+',dtype=np.float64,shape=(120,n));background=np.empty(n)
    coverage=np.zeros((120,n),dtype=bool);controls=[]
    for task in plan['tasks']:
        raw=loadmat(task['output']);meta=json.loads(raw['manifest_json'].item());assert meta['request_signature']==task['request_signature']
        rows=np.load(task['rows']);first,stop=task['first'],task['stop'];values=raw['expected_counts']
        assert values.shape==(stop-first,len(rows)) and np.all(np.isfinite(values)) and np.all(values>=0)
        assert not coverage[first:stop,rows].any();counts[first:stop,rows]=values;coverage[first:stop,rows]=True
        background[rows]=raw['background_counts'].reshape(-1)
        if meta['native_control']:controls.append(meta['native_control'])
    assert coverage.all() and np.all(background>0);counts.flush();del coverage
    with np.load(out/'geometry.npz') as geometry:diameters=geometry['apertures_um'].copy()
    blocks=[];maximum=0.;request=canonical_hash(dict(plan=plan['sources'],geometry=plan['geometry'],media=plan['media_sha256']))
    directory=out/'capture_blocks';directory.mkdir()
    for first in range(0,120,40):
        stop=min(first+40,120);values=np.array(counts[first:stop]);maximum=max(maximum,float((values*30).max()))
        meta=dict(schema='cone_eccentric_aperture_video_v1',request_signature=request,phototransduction_applied=False,
            photon_noise='none',eye_movements_applied=False,aperture_profile_sha256=plan['geometry']['aperture_profile_sha256'],
            direct_native_controls=controls,FIT_READY=False,background_qualified=False)
        path=directory/f'frames_{first:03d}_{stop-1:03d}.mat'
        savemat(path,dict(manifest_json=json.dumps(meta),expected_counts=values,background_counts=background,
            diameters_um=diameters,window_edges_seconds=edges[first:stop+1]),do_compression=True)
        assert path.stat().st_size<BUDGET and values.nbytes+background.nbytes+diameters.nbytes<BUDGET
        blocks.append(dict(path=str(path),first=first,stop=stop))
    result=dict(request_signature=request,frames=120,fps=30.,duration_seconds=4.,cones=n,
        maximum_rate_Rstar_per_cone_s=maximum,capture_blocks=blocks,native_controls=controls,
        phototransduction_run=False,downstream_circuit_run=False,eye_movements=False,
        FIT_READY=False,background_qualified=False)
    (out/'capture_complete.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--media',type=Path,default=Path('D:/macular/examples/video/video/113.AVI'))
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--capture-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if not args.capture_only:prepare(out,args.media)
    if not args.prepare_only:capture(out)
