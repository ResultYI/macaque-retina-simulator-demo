"""Restore the exact scalar150Hz field from T's stored input identity and time axis."""
from pathlib import Path
import hashlib
import json
import numpy as np
from retina_sim.contracts import canonical_hash

ROOT=Path(r'D:\macaque retinal simulator')
BASE=ROOT/'reports/CELL6_RAW_REBUILD_CONDITIONAL_READOUT_V1'
REFERENCE=BASE/'runtime_monitor_repair_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dictionary(value):return dict(value['dictionary'])


def owned_array(directory,manifest,descriptor):
    name=descriptor['array']
    if Path(name).name!=name or not name.endswith('.npy'):raise ValueError('Local typed array required')
    path=directory/name
    if sha(path)!=manifest['files'][name]['sha256']:raise RuntimeError('Stored input array hash differs')
    return np.load(path,allow_pickle=False)


def matched_input(raw,group,out):
    folder=REFERENCE/'cache'/f'g{group}'
    identity=read(folder/'INPUT_AND_MODEL_IDENTITY.json');mapping=identity['input_mapping']
    if mapping['native_input_fps']!=150. or mapping['formula']!='Btd*(1+0.05*(raw[j]-raw[0]))':
        raise RuntimeError('T input mapping differs; no guessed resampling')
    seed=BASE/'halo_model_v2/seed';manifest=read(seed/'manifest.json')
    reference=dictionary(dictionary(manifest['payload'])['input_reference'])
    background=owned_array(seed,manifest,reference['native_background_td'])
    indices={int(k):int(v) for k,v in identity['frame_source_indices'].items()}
    frames=sorted(indices)
    if frames!=list(range(1065)):raise RuntimeError('Exact declared T frame coverage required')
    factors={f:1+.05*(float(raw[indices[f]])-float(raw[0])) for f in frames}
    program={f:background*factors[f] for f in frames}
    reconstructed=canonical_hash(program)
    if reconstructed!=identity['input_payload_sha256']:
        raise RuntimeError('Reconstructed full T light program SHA mismatch')
    latest=read(folder/'LATEST_CHECKPOINT.json');checkpoint=folder/latest['directory']
    saved=read(checkpoint/'manifest.json')
    anchor=dictionary(dictionary(dictionary(saved['payload'])['native'])['point'])['anchor']
    anchor=dictionary(anchor);epoch=anchor['input_epoch']
    origin=anchor['time_seconds']-epoch/150.
    if origin!=anchor['input_stop_seconds']-(epoch+1)/150.:
        raise RuntimeError('Stored T start/end input clock origin differs')
    actual=owned_array(checkpoint,saved,anchor['light'])
    if not np.array_equal(actual,program[epoch]):raise RuntimeError('Actual saved native input differs')
    cache_manifest=read(folder/'manifest.json');T_result=read(REFERENCE/f'readout/g{group}_RESULT.json')
    if sha(folder/'manifest.json')!=T_result['cache_manifest_sha256']:
        raise RuntimeError('T-used cache manifest differs')
    if sha(folder/'endogenous.npz')!=cache_manifest['arrays_sha256']:
        raise RuntimeError('T time-axis/source artifact differs')
    with np.load(folder/'endogenous.npz',allow_pickle=False) as z:times=z['time_source_s'][:-1].copy()
    if not np.array_equal(times,np.arange(71000)*.0001):raise RuntimeError('Original100us node grid differs')
    starts=np.asarray([origin+f/150. for f in frames]);ends=np.asarray([origin+(f+1)/150. for f in frames])
    frame_at=np.searchsorted(starts,times,side='right')-1
    if not np.all((frame_at>=0)&(times>=starts[frame_at])&(times<ends[frame_at])):
        raise RuntimeError('T right-continuous frame coverage is incomplete')
    scalar=np.asarray([factors[f] for f in frames]);held=scalar[frame_at]
    np.savez_compressed(out/f'g{group}_MATCHED_INPUT.npz',time_source_s=times,relative_Td_factor=held,
        frame_ids=np.asarray(frames),frame_start_s=starts,frame_factors=scalar,
        raw_source_indices=np.asarray([indices[f] for f in frames]))
    witness=dict(group=group,T_input_identity_sha256=sha(folder/'INPUT_AND_MODEL_IDENTITY.json'),
        T_program_sha256=identity['input_payload_sha256'],reconstructed_program_sha256=reconstructed,
        full_program_bitwise_identity_closed=True,saved_native_anchor_light_exact=True,input_origin_s=origin,
        input_fps=150.,actual_T_time_axis_used=True,node_samples=71000,
        mapping='Stored frame_source_indices and scalar1+.05*(raw[index]-raw[0]); Btd vector retained in exact program hash',
        continuity='Same right-continuous150Hz field sampled at T100us nodes; baseline FIR and1ms bin-start rules unchanged',
        scalar_information='Complete common multiplicative illumination factor across original Cone backgrounds; no extra stimulus information',
        baseline_input_units='relative nativeTd factor; not raw acquisition units',retinal_model_solved=False)
    (out/f'g{group}_INPUT_IDENTITY.json').write_text(json.dumps(witness,indent=2),encoding='utf8')
    return held
