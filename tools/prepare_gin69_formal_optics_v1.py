"""Exact-frame deduplication for the approved training/development prefix only."""
from pathlib import Path
from copy import deepcopy
import hashlib
import time
import numpy as np
import cv2
from scipy.io import savemat
from retina_sim.contracts import canonical_hash
from retina_sim.spatial_group_graph import digest
from tools.durable_research_runtime_v1 import atomic_json
from tools.gin_hc2_bc1_context_v1 import ROOT,OUT,OLD,read

def build(output,guard,*,source_out=OUT):
    OUT=Path(source_out)
    start=time.perf_counter();gate=read(OUT/'SHORT_NUMERICAL_CHECK.json')
    if not gate['passed']:raise ValueError('New network short numerical gate required')
    split=read(OUT/'SPLIT_CONTRACT.json')
    if split['train']['interval']!=[0,30] or split['development']['interval']!=[30,45] or split['sealed']['automatic_access']:
        raise ValueError('Frozen prefix-only split differs')
    folder=OUT/'formal_optics';folder.mkdir()
    config=read(OUT/'optical/CONFIG.json');source=read(OLD/'INPUT_APPROXIMATION_MANIFEST.json')
    movie=ROOT/source['movie_path']
    if digest(movie)!=source['movie_sha256']:raise ValueError('Frozen source movie differs')
    previous=read(OLD/'SOURCE_SCALE.json')['frame_hashes'];known={h:dict(short_capture_row=int(i)-751+1) for i,h in previous.items()}
    frame_rows=[];jobs=[];pending=[];pending_ids=[];unique=0
    def flush():
        if not pending:return
        job_id=len(jobs);directory=folder/f'job_{job_id:04d}';directory.mkdir()
        bank=directory/'SOURCE_BANK.mat';savemat(bank,dict(source_rgb=np.stack(pending,axis=3)),do_compression=True)
        c=deepcopy(config);c.update(source_bank=str(bank),frame_indices=list(range(1,len(pending)+1)),
            window_edges_seconds=(np.arange(len(pending)+1)/150).tolist(),native_control=job_id==0)
        c.pop('request_signature');c['request_signature']=canonical_hash(c)
        atomic_json(directory/'CONFIG.json',c)
        jobs.append(dict(job=job_id,config=str(directory/'CONFIG.json'),output=str(directory/'CAPTURE.mat'),
            source_frame_zero_based=pending_ids.copy(),count=len(pending),bank_sha256=digest(bank),config_sha256=digest(directory/'CONFIG.json'),
            native_control='existing native control on first actual nonuniform batch; unchanged optics/sampler thereafter'))
        pending.clear();pending_ids.clear()
    cap=cv2.VideoCapture(str(movie))
    try:
        for index in range(751+6750):
            okay,bgr=cap.read()
            if not okay:raise ValueError('Source movie prefix truncated at'+str(index))
            if index<751:continue
            rgb=bgr[:,:,::-1].copy();h=hashlib.sha256(rgb.tobytes()).hexdigest()
            if index<759 and previous[str(index)]!=h:raise ValueError('Short and formal source frame bytes disagree')
            if h not in known:
                known[h]=dict(job=len(jobs),capture_row=len(pending));pending.append(rgb);pending_ids.append(index);unique+=1
                if len(pending)==30:flush();guard()
            frame_rows.append(dict(source_frame_zero_based=index,time_frame_start_seconds=(index-751)/150,
                decoded_RGB_sha256=h,**known[h]))
    finally:cap.release()
    flush()
    if len(frame_rows)!=6750:raise ValueError('Exact45s prefix missing')
    meta=dict(schema='GIN69-exact-movie-prefix-optics-1',split_sha256=digest(OUT/'SPLIT_CONTRACT.json'),
        source_manifest_sha256=digest(OLD/'INPUT_APPROXIMATION_MANIFEST.json'),movie_sha256=source['movie_sha256'],
        actual_simulation_interval_seconds=[0,45],source_frame_range_half_open=[751,7501],playback_Hz=150,
        frame_count=len(frame_rows),additional_distinct_frames=unique,jobs=jobs,frames=frame_rows,
        exact_duplicate_RGB_only=True,changed_physics=False,original_optical_config_sha256=digest(OUT/'optical/CONFIG.json'),
        original_capture_sha256=digest(OUT/'optical/CAPTURE.mat'),same_geometry_optics_SPD_QE_aperture_clock=True,
        stimulus_only_no_spike_data_read=True,sealed_segment_not_generated=True,
        approval='Sentinel_22edff39ac30819183d4ea7c17a01855',FIT_READY=False,background_qualified=False,production_adopted=False)
    meta['identity']=canonical_hash(meta);atomic_json(folder/'MANIFEST.json',meta)
    atomic_json(output/'RESULT.json',dict(status='FORMAL_OPTICAL_PREFIX_READY',frame_count=6750,
        additional_distinct_frames=unique,short_capture_reused_frame_occurrences=sum('short_capture_row' in r for r in frame_rows),
        jobs=len(jobs),preparation_wall_seconds=time.perf_counter()-start,manifest_sha256=digest(folder/'MANIFEST.json')))
    print(str(read(output/'RESULT.json')),flush=True)
