"""Bounded native vector-pack reproduction from the actual200ms checkpoint."""
from pathlib import Path
import argparse,faulthandler,hashlib,json,sys,time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/retinotopic_rf_revision/SPATIAL_POOLING_CANDIDATE_V1/packing_diagnosis_v1'


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run(mode):
    import numpy as np
    from retina_sim.background_checkpoint import MemoryGuard
    start=time.perf_counter();memory=MemoryGuard(24*1024**3,8*1024**3)
    def guard():
        if not memory.check():raise MemoryError(str((memory.stop_reason,memory.last)))
        if time.perf_counter()-start>60:raise TimeoutError('Packing reproduction60s bound')
    if mode=='capture':
        from scipy import sparse
        from retina_sim.group_time_storage import load_bundle
        from retina_sim.contracts import canonical_hash
        from tools import run_spatial_pooling_pair_v1 as old
        case=old.OUT;source=old.read(ROOT/'reports/external_video_two_positions_4s_20261029_v1/PREPARATION.json')['bindings']['near']
        a,parent,_=old.load_background(Path(source['prepared']),Path(source['background']));guard()
        seed=load_bundle(case/'function_v2/near/candidate/initial_seed',expected_kind='source_construction_seed')['payload']
        m=a.runtime_factories['inner'](a,seed['inner_parameters']);bindings={}
        for r in old.read(case/'STRUCTURE_RESULT.json')['routes']:
            if r['location']!='near':continue
            p=case/'structure/near'/r['route']/'group_candidate_local.npz';m.mean[r['route']]=sparse.load_npz(p);bindings[r['route']]=old.file_sha(p)
        m.identity=canonical_hash(dict(parent_inner=m.identity,spatial_candidate='RETINOTOPIC_GEOMETRY_ANCHORED_POOLING_V1',location='near',arm='candidate',matrices=bindings,physiology_parameters_unchanged=True))
        checkpoint=case/'function_v3/near/candidate/tight/CHECKPOINT_0200'
        payload=load_bundle(checkpoint,expected_kind='validation_evidence')['payload']
        assert m.identity==payload['model_identity'] and payload['next_event_index']==2000
        state=m.unpack(payload['inner_flat'],copy=False)
        before=canonical_hash(state);rhs=m.rhs(.2,state);m.validate(rhs)
        assert canonical_hash(state)==before
        arrays=[rhs[k] for k in m.state_keys]
        golden=np.concatenate(arrays)
        manual=np.empty(m.nstate,dtype=np.float64)
        for k in m.state_keys:manual[m.slices[k]]=rhs[k]
        assert np.array_equal(golden,manual)
        contracts=[dict(key=list(k),type=type(rhs[k]).__name__,shape=list(rhs[k].shape),strides=list(rhs[k].strides),
            owndata=bool(rhs[k].flags.owndata),aligned=bool(rhs[k].flags.aligned),writeable=bool(rhs[k].flags.writeable),
            c_contiguous=bool(rhs[k].flags.c_contiguous),base_type=type(rhs[k].base).__name__) for k in m.state_keys]
        np.savez_compressed(OUT/'ACTUAL_RHS_0200.npz',**{f'block_{i:03d}':v for i,v in enumerate(arrays)},golden=golden)
        write(OUT/'CAPTURE.json',dict(model_identity=m.identity,checkpoint_sha256=sha(checkpoint/'manifest.json'),
            state_not_mutated_by_rhs=True,concatenate_equals_slice_pack=True,nblocks=len(arrays),nstate=m.nstate,
            contracts=contracts,wall_seconds=time.perf_counter()-start,peak_rss_bytes=memory.peak_rss_bytes,
            forward_model_time_advanced=0,compiled_rhs_enabled=m.compiled_rhs is not None))
        print('ACTUAL_RHS_CAPTURED',len(arrays),m.nstate,flush=True);return
    with np.load(OUT/'ACTUAL_RHS_0200.npz',allow_pickle=False) as z:
        arrays=[z[k].copy() for k in sorted(z.files) if k.startswith('block_')];golden=z['golden'].copy()
    if mode=='views':
        cuts=np.r_[0,np.cumsum([len(v) for v in arrays])];owner=golden.copy();arrays=[owner[a:b] for a,b in zip(cuts[:-1],cuts[1:])]
    elif mode in ('owned','churn'):arrays=[a.copy() for a in arrays]
    else:raise ValueError(mode)
    source_hash=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
    iterations=0;latest=None;retained=[]
    while iterations<200000 and time.perf_counter()-start<25:
        if mode=='churn':
            transient=[a.copy() for a in arrays]
            latest=np.concatenate(transient)
            retained.append(latest)
            if len(retained)>16:del retained[0]
            del transient
        else:latest=np.concatenate(arrays)
        iterations+=1
        if iterations%1000==0:
            guard()
            if not np.array_equal(latest,golden):raise ValueError('Native pack changed bits')
    assert np.array_equal(latest,golden)
    assert source_hash==[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
    write(OUT/(mode+'_RESULT.json'),dict(status='PASS_NOT_REPRODUCED',mode=mode,iterations=iterations,nblocks=len(arrays),nstate=len(golden),
        output_and_inputs_bitwise_stable=True,wall_seconds=time.perf_counter()-start,peak_rss_bytes=memory.peak_rss_bytes,
        numpy_version=np.__version__,native_bug_fixed_claim=False))
    print(mode,'PASS_NOT_REPRODUCED',iterations,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['capture','owned','views','churn'],required=True);args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    with (OUT/(args.mode+'_native_fault.log')).open('x',encoding='utf8') as f:
        faulthandler.enable(file=f,all_threads=True)
        run(args.mode)
