"""Prepare missing exact tile scenes; never run MATLAB or a neural model.

Every source frame/time is retained. Reuse requires byte-identical scene
multipliers on the same frozen tile, at the same capture window and reference.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def exact_scene_column(scene, candidates):
    import numpy as np
    for column in range(candidates.shape[2]):
        if np.array_equal(scene,candidates[:,:,column]):return column
    return None


def prepare(directory,start,stop):
    import numpy as np
    from scipy.io import loadmat,savemat
    from retina_sim.contracts import ContractError,canonical_hash
    from retina_sim.split_chunk_store import write_json
    from retina_sim.split_execution_guard import SplitExecutionGuard
    from retina_sim.useful_runtime_budget import runtime_limits,OVERNIGHT_SCOPE
    from tools.run_vr_useful_split import require_frontend
    from tools.vr_teacher_handoff_v1.fit_integration_v1.capture_preparation_v2.handoff import approved_policy
    from tools.vr_teacher_handoff_v1.fit_integration_v1.capture_preparation_v2.linear_gray import LinearGrayScene
    from tools.vr_teacher_handoff_v1.fit_integration_v1.pilot import Gray8Chunks
    directory=Path(directory).resolve();budget,frontend=require_frontend(directory)
    authorization=runtime_limits(directory,budget)
    if authorization.get('authorized_goal')!=OVERNIGHT_SCOPE or not 0<=start<stop<=9751:
        raise ContractError('Optical extension requires the approved first65s plus genuine next-frame guard')
    guard=SplitExecutionGuard(authorization,output_directory=directory);guard.require(force=True)
    output=directory/'optical_extensions'/f'frames-{start:06d}-{stop:06d}'
    if output.exists():raise FileExistsError('Preserve existing prepared extension; do not regenerate')
    output.mkdir(parents=True)
    approval=json.loads((directory/'input_policy_approval.json').read_bytes())
    _,objects=approved_policy(ROOT,approval)
    received=objects['received_assets'];intake=Path(received['local_intake_root'])
    index_path=intake/received['stimulus_index']['path']
    if hashlib.sha256(index_path.read_bytes()).hexdigest()!=received['stimulus_index']['sha256']:
        raise ContractError('Original raw source index changed')
    reader=Gray8Chunks(intake,json.loads(index_path.read_bytes()))
    prefix=directory/'linear_inputs';old_plan=json.loads((prefix/'capture_plan.json').read_bytes())
    representatives=[];hashes=[];mapping=[];seen={};frames=[]
    for i in range(start,stop):
        frame=reader.frame(i);digest=hashlib.sha256(frame.tobytes()).hexdigest();hashes.append(digest)
        if digest not in seen:
            if len(frames)>=32:raise ContractError('Use smaller input chunks: at most32 distinct raw frames per tile packet')
            seen[digest]=len(frames);frames.append(frame.copy());representatives.append(i)
        elif not np.array_equal(frame,frames[seen[digest]]):
            raise ContractError('Raw hash matched unequal frame bytes')
        mapping.append(seen[digest])
    optical=json.loads((ROOT/'data/atlas/human_disk3mm_input_plan_v2/manifest.json').read_bytes())
    reference=json.loads((directory/'actual_global_reference.json').read_bytes())
    adapter=LinearGrayScene(ROOT,approval);jobs=[];tiles=[]
    progress=dict(stage='PREPARING_EXACT_OPTICAL_EXTENSION',raw_frames=[start,stop],
        unique_frames=len(frames),prepared_tiles=0,matlab_jobs=0,created_unix=time.time(),model_run=False)
    write_json(output/'status.json',progress)
    for i,tile in enumerate(optical['tiles']):
        guard.require()
        old_folder=prefix/f'tile-{i:04d}';old_config=json.loads((old_folder/'export.json').read_bytes())
        old_mat=old_folder/'input.mat'
        signature=canonical_hash(dict(policy=approval['policy_sha256'],tile=tile,reference=reference,
            frame_sha256=old_plan['frame_hashes'],input_mat_sha256=hashlib.sha256(old_mat.read_bytes()).hexdigest()))
        if signature!=old_config['request_signature'] or signature!=old_plan['tiles'][i]['request_signature']:
            raise ContractError('Original exact scene proof changed before reuse')
        data={k:v for k,v in loadmat(old_mat).items() if not k.startswith('__')}
        old_scenes=data['scene_multipliers'];samples=tile['scene_samples'];center=np.asarray(tile['center_mm'])
        axis=np.arange(samples)-(samples-1)/2
        xx,yy=np.meshgrid(axis*.001+center[0],-axis*.001+center[1]);xy=np.stack([xx,yy],axis=-1)
        sources=[];new_scenes=[]
        for frame in frames:
            scene=adapter.reference_multipliers(frame,xy)
            prior=exact_scene_column(scene,old_scenes)
            if prior is not None:
                raw_column=old_plan['unique_frame_indices'].index(prior+1)
                sources.append(dict(kind='prefix_global_counts',column=raw_column))
                continue
            column=next((k for k,old in enumerate(new_scenes) if np.array_equal(scene,old)),None)
            if column is None:column=len(new_scenes);new_scenes.append(scene)
            sources.append(dict(kind='new_tile_capture',column=column))
        item=dict(tile_index=i,global_rows=tile['global_rows'],sources=sources,
            original_scene_proof_sha256=signature)
        if new_scenes:
            folder=output/f'tile-{i:04d}';folder.mkdir()
            scenes=np.stack(new_scenes,axis=2)
            if scenes.nbytes>128*1024**2:raise ContractError('Optical tile packet exceeds128MiB; use fewer frames')
            data.update(scene_multipliers=scenes,window_edges_seconds=np.arange(len(new_scenes)+1)/150.,
                        frame_indices=np.arange(1,len(new_scenes)+1,dtype=float))
            source=folder/'input.mat';savemat(source,data,do_compression=True)
            config=dict(old_config,input_mat=str(source),request_signature=canonical_hash(dict(
                scope='exact_optical_extension_v1',prefix_frontend=canonical_hash(frontend),tile=tile,
                raw_frame_range=[start,stop],frame_sha256=hashes,input_mat_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
            write_json(folder/'export.json',config)
            jobs.append(dict(configuration=str(folder/'export.json'),output=str(folder/'capture.mat')))
            item.update(request_signature=config['request_signature'],capture_output=str(folder/'capture.mat'),
                        new_unique_scenes=len(new_scenes))
        tiles.append(item)
        progress.update(prepared_tiles=i+1,matlab_jobs=len(jobs),updated_unix=time.time(),resources=guard.last)
        write_json(output/'status.json',progress,replace=True)
    write_json(output/'batch.json',jobs)
    record=dict(schema='exact-vr-optical-extension-plan-1',raw_frame_range=[start,stop],
        raw_index_sha256=received['stimulus_index']['sha256'],frame_sha256=hashes,
        unique_frame_representatives=representatives,frame_to_unique=mapping,tiles=tiles,
        input_policy_sha256=approval['policy_sha256'],prefix_frontend_sha256=canonical_hash(frontend),
        actual_capture_window_seconds=1/150.,no_gamma_or_normalization_change=True,
        original_750_frames_retained=True,reference_global_sha256=frontend['global_reference_numeric_sha256'],
        preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        exact_scene_reuse_only=True,fit_eligible=False)
    write_json(output/'plan.json',record)
    progress.update(stage='PREPARED_MISSING_SCENES_MATLAB_NOT_RUN',finished_unix=time.time())
    write_json(output/'status.json',progress,replace=True)
    print(json.dumps(dict(output=str(output),unique_frames=len(frames),missing_tile_jobs=len(jobs),
                          reused_tiles=len(tiles)-len(jobs))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True)
    parser.add_argument('--start-frame',type=int,required=True);parser.add_argument('--stop-frame',type=int,required=True)
    args=parser.parse_args();prepare(args.run_dir,args.start_frame,args.stop_frame)
