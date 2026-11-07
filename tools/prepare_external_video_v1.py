"""Reuse the frozen 0..4s/120-frame optical plan for only the two local populations."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys,time
import numpy as np
from scipy.io import loadmat,savemat

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.prepare_external_first_frame_v1 import VIDEO,locations
from tools.cross_position_context_v1 import read,write
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.spatial_group_graph import digest
from retina_sim.disk_input import optical_tiles
from retina_sim.isetbio_input import reference_equivalent_drive

HISTORY=6564.098260100071


def prepare(out):
    start=time.perf_counter();out.mkdir(exist_ok=False);plan=read(VIDEO/'capture_plan.json')
    edges=np.asarray(plan['window_edges_seconds'],dtype=np.float64)
    if not np.array_equal(edges,np.arange(121)/30) or plan['media']['fps']!=30 or plan['media']['frame_count']!=447:
        raise ContractError('Frozen 0..4s timeline differs; do not choose new timing')
    with np.load(VIDEO/'geometry.npz',allow_pickle=False) as z:g={k:z[k] for k in z.files}
    from retina_sim.native_cone_apertures import PROFILE,native_apertures
    support=np.load(ROOT/'data/atlas/human_disk3mm_typed_tables_v1/cones/rf.npy')
    native_diameters,_=native_apertures(support)
    g['apertures_um']=native_diameters[g['source_table_rows']]
    index={int(row):i for i,row in enumerate(g['source_table_rows'])};position_rows={}
    for label,paths in locations().items():
        with np.load(paths['optical']/'APERTURE_GEOMETRY.npz',allow_pickle=False) as z:
            rows=np.asarray([index[int(r)] for r in z['fine_rows']],dtype=np.int64)
            for a,b in [(g['xy_mm'][rows],z['positions_mm']),(g['apertures_um'][rows],z['reference_diameter_um']),
                        (g['types'][rows],z['cone_types'].astype('U1'))]:
                if not np.array_equal(a,b):raise ContractError('Actual local geometry differs from frozen video')
            position_rows[label]=rows
    union=np.unique(np.concatenate(list(position_rows.values())))
    bank=loadmat(VIDEO/'source_rgb.mat')['source_rgb']
    if bank.shape!=(360,640,3,120):raise ContractError('Frozen source frame bank absent or incomplete')
    for i,receipt in enumerate(plan['frame_receipts']):
        if receipt['source_frame_index']!=i or hashlib.sha256(bank[:,:,:,i].tobytes()).hexdigest()!=receipt['decoded_rgb_sha256']:
            raise ContractError('Original decoded source frame identity differs')
    pins={str(p.relative_to(ROOT)):digest(p) for p in [VIDEO/'capture_plan.json',VIDEO/'source_rgb.mat',
        VIDEO/'geometry.npz',VIDEO/'optics_config.json',VIDEO/'optics_cache/model.mat',VIDEO/'first_frame/capture.mat']}
    for p in (VIDEO/'optics_cache').glob('otf_*.mat'):pins[str(p.relative_to(ROOT))]=digest(p)
    if len(list((VIDEO/'optics_cache').glob('otf_*.mat')))!=101:raise ContractError('Frozen wave cache incomplete')
    names=['export_cone_video_aperture_tile.m','export_cone_video_aperture_batch.m','cone_video_cached_oi.m',
        'native_cone_kernel_groups.m','cone_video_gaussian_sampling.m','cone_native_aperture.m','cone_optics_mosaic.m','matlab/sample_cone_aperture_zones.m']
    for name in names:pins['tools/'+name]=digest(ROOT/'tools'/name)
    tasks=[];jobs=[]
    for tile in optical_tiles(g['xy_mm'][union]):
        key=tile['tile_index'];original=next(t for t in plan['tasks'] if t['tile_index']==key and t['first']==0)
        config=read(Path(original['output']).parent/'export.json')
        old_rows=np.load(Path(original['rows']));raw=loadmat(Path(config['geometry_mat']))
        wanted=union[tile['indices']];mask=np.isin(old_rows,wanted);rows=old_rows[mask]
        if not np.array_equal(np.sort(rows),wanted):raise ContractError('Original tile ownership differs')
        folder=out/('tile_%d_%d'%tuple(key));folder.mkdir();np.save(folder/'global_rows.npy',rows)
        data={k:raw[k].reshape((-1,2) if k=='is_mm' else (-1,1))[mask] for k in ['is_mm','types']}
        data.update(reference_diameter_um=g['apertures_um'][rows],zone_diameter_um=g['apertures_um'][rows])
        savemat(folder/'geometry.mat',data,do_compression=True)
        tile_id=canonical_hash(dict(original_tile=config['tile_signature'],rows=rows,geometry=data,
            aperture_native_sources={k:v for k,v in pins.items() if k.startswith('tools/')}))
        for first in range(0,120,16):
            stop=min(first+16,120);directory=folder/f'frames_{first:03d}_{stop-1:03d}';directory.mkdir()
            c=deepcopy(config);c.update(aperture_model=PROFILE['profile_id'],aperture_profile_sha256=canonical_hash(PROFILE),geometry_mat=str(folder/'geometry.mat'),tile_signature=tile_id,
                frame_indices=list(range(first+1,stop+1)),window_edges_seconds=edges[first:stop+1].tolist(),
                native_control=bool(config['native_control'] and first==0))
            c['request_signature']=canonical_hash(dict(parent=original['request_signature'],new_configuration=c,source_pins=pins))
            write(directory/'export.json',c)
            job=dict(configuration=str(directory/'export.json'),output=str(directory/'capture.mat'));jobs.append(job)
            tasks.append(dict(tile_index=key,rows=str(folder/'global_rows.npy'),first=first,stop=stop,**job,
                request_signature=c['request_signature']))
    write(out/'jobs.json',jobs)
    descriptor=dict(schema='two-position-frozen-video-4s-1',source_pins=pins,tasks=tasks,
        source_frame_count=120,frame_indices=list(range(120)),window_edges_seconds=edges.tolist(),
        position_global_rows={k:v.tolist() for k,v in position_rows.items()},union_global_rows=union.tolist(),
        source_time_range_seconds=[0.,4.],source_original_duration_seconds=plan['media']['duration_seconds'],
        selection_reason='Use existing frozen first4s short segment;900ms fallback not applicable',
        first_frame_replay='New continuous IVP starts from each saved S0; old single-frame results not appended or double-counted',
        no_per_frame_state_or_rng_reset=True,no_new_physics_or_stimulus_parameters=True,
        FIT_READY=False,background_qualified=False,production_adopted=False)
    descriptor['experiment_id']=canonical_hash(descriptor);write(out/'OPTICAL_PLAN.json',descriptor)
    write(out/'PREPARATION_RESOURCE.json',dict(wall_seconds=time.perf_counter()-start,historical_cumulative_seconds=HISTORY,
        needed_fine_cones=len(union),optical_tiles=len(tasks)//8,jobs=len(jobs),source_rgb_frames_sha_verified=120))
    print(json.dumps(dict(status='PREPARED',needed_fine_cones=len(union),jobs=len(jobs),duration_seconds=4.)),flush=True)


def collect(out):
    start=time.perf_counter();plan=read(out/'OPTICAL_PLAN.json');union=np.asarray(plan['union_global_rows'])
    lookup={int(r):i for i,r in enumerate(union)};counts=np.empty((120,len(union)));covered=np.zeros_like(counts,dtype=bool);pins={};controls=[]
    for task in plan['tasks']:
        path=Path(task['output'])
        if not Path(str(path)+'.ready').exists():raise ContractError('Capture not completed: '+str(path))
        raw=loadmat(path);meta=json.loads(raw['manifest_json'].item());rows=np.load(task['rows']);dest=[lookup[int(r)] for r in rows]
        if meta['request_signature']!=task['request_signature'] or meta['aperture_backend']!='isetbio_native_aperture_v1':raise ContractError('Optical identity differs')
        a,b=task['first'],task['stop'];values=raw['expected_counts']
        if values.shape!=(b-a,len(rows)) or covered[a:b,dest].any():raise ContractError('Capture axes/ownership differ')
        counts[a:b,dest]=values;covered[a:b,dest]=True;pins[str(path.relative_to(ROOT))]=digest(path)
        if meta['native_control']:controls.append(meta['native_control'])
    if not covered.all() or not np.isfinite(counts).all() or np.any(counts<0):raise ContractError('Finite complete real-frame capture required')
    old=loadmat(VIDEO/'first_frame/capture.mat')['expected_counts'][0,union]
    error=float(np.max(abs(counts[0]-old)/np.maximum(old,1e-20)))
    if error>3e-6:raise ContractError('Original first-frame capture cross-check exceeds existing native tolerance')
    np.savez_compressed(out/'FINE_CAPTURE.npz',expected_counts=counts,global_video_rows=union,
        window_edges_seconds=np.asarray(plan['window_edges_seconds']))
    protocol=deepcopy(read(ROOT/'reports/external_first_frame_two_positions_20261029_v1/PREPARATION.json')['protocol'])
    protocol.update(stop_seconds=4.,checkpoint_seconds=4.,input_frame_indices=list(range(600)))
    bindings={}
    for label,paths in locations().items():
        rows=plan['position_global_rows'][label];columns=[lookup[int(r)] for r in rows]
        with np.load(paths['optical']/'APERTURE_GEOMETRY.npz',allow_pickle=False) as z:
            types=z['cone_types'].astype('U1').tolist();members=z['local_group_counts'];fine=z['fine_rows']
        rates=counts[:,columns]/(1/30);bridge=deepcopy(read(paths['background']/'FIXED_LMS_BRIDGE.json')['profile'])
        scope=canonical_hash(dict(captures=pins,position=label,fine_rows=fine));bridge['capture_scope_sha256']=scope
        drive,conversion=reference_equivalent_drive(rates,types,bridge,capture_scope_sha256=scope)
        cuts=np.r_[0,np.cumsum(members)];group=np.stack([drive[:,cuts[i]:cuts[i+1]].mean(axis=1) for i in range(len(members))],axis=1)
        folder=out/label;folder.mkdir();np.savez_compressed(folder/'DRIVE.npz',group_native_drive=group,fine_rows=fine)
        binding=dict(prepared=str(paths['prepared']),background=str(paths['background']),position=label,
            drive_sha256=digest(folder/'DRIVE.npz'),optical_frames=120,native150Hz_holds_per_source_frame=5,
            fixed_LMS_bridge=bridge,conversion=conversion,member_mean_once=True)
        write(folder/'BINDING.json',binding);bindings[label]=binding
    pins.update(plan['source_pins'])
    descriptor=dict(experiment_id=plan['experiment_id'],source_pins=pins,protocol=protocol,bindings=bindings,
        fixture=dict(input='Original113.AVI frames0..119',duration_seconds=4.,original_optical_plan=str(out/'OPTICAL_PLAN.json'),
            frame_rate=30.,fixed_radiance_gain=read(VIDEO/'optics_config.json')['fixed_radiance_gain'],
            native_aperture_backend=True,no_new_contrast_or_spectrum=True,engineering_continuous_playback=True),
        FIT_READY=False,background_qualified=False,production_adopted=False)
    write(out/'PREPARATION.json',descriptor)
    write(out/'CAPTURE_CHECK.json',dict(status='PASS',covered_frames=120,covered_cones=len(union),
        original_first_frame_maximum_relative_error=error,existing_tolerance=3e-6,native_controls=controls,
        collection_wall_seconds=time.perf_counter()-start))
    print(json.dumps(dict(status='READY_CONTINUOUS',duration_seconds=4.,first_frame_relative_error=error)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--collect',action='store_true');a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'reports'):raise ContractError('Fresh reports output required')
    (collect if a.collect else prepare)(out)
