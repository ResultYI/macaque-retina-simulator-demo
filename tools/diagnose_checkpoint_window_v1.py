"""Single diagnostic replay of the recorded200..300ms fault window, unchanged math."""
from pathlib import Path
import faulthandler,json,os,sys,time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/retinotopic_rf_revision/SPATIAL_POOLING_CANDIDATE_V1/packing_diagnosis_v1/window_debug'


def main():
    from copy import deepcopy
    from types import SimpleNamespace
    import numpy as np
    from scipy import sparse
    from scipy.integrate import solve_ivp
    from retina_sim.background_checkpoint import MemoryGuard
    from retina_sim.group_time_storage import load_bundle,save_bundle
    from retina_sim.contracts import canonical_hash
    from retina_sim.rgc_functional_v1 import RGCFunctionalV1
    from tools import run_spatial_pooling_pair_v1 as old
    start=time.perf_counter();memory=MemoryGuard(24*1024**3,8*1024**3);journal={'phase':'entry','model_time_seconds':.2,'pack_calls':0}
    def marker():
        p=OUT/'LAST_CALL.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(journal,indent=2),encoding='utf8');t.replace(p)
    def guard():
        if not memory.check():raise MemoryError(str((memory.stop_reason,memory.last)))
        if time.perf_counter()-start>150:raise TimeoutError('Single fault-window diagnosis150s bound')
    P=old.OUT;source=old.read(ROOT/'reports/external_video_two_positions_4s_20261029_v1/PREPARATION.json')['bindings']['near']
    a,parent,_=old.load_background(Path(source['prepared']),Path(source['background']));guard()
    seed=load_bundle(P/'function_v2/near/candidate/initial_seed',expected_kind='source_construction_seed')['payload']
    m=a.runtime_factories['inner'](a,seed['inner_parameters']);bindings={}
    for r in old.read(P/'STRUCTURE_RESULT.json')['routes']:
        if r['location']!='near':continue
        p=P/'structure/near'/r['route']/'group_candidate_local.npz';m.mean[r['route']]=sparse.load_npz(p);bindings[r['route']]=old.file_sha(p)
    m.identity=canonical_hash(dict(parent_inner=m.identity,spatial_candidate='RETINOTOPIC_GEOMETRY_ANCHORED_POOLING_V1',location='near',arm='candidate',matrices=bindings,physiology_parameters_unchanged=True))
    checkpoint=load_bundle(P/'function_v3/near/candidate/tight/CHECKPOINT_0200',expected_kind='validation_evidence')['payload']
    assert checkpoint['model_identity']==m.identity and checkpoint['next_event_index']==2000
    with np.load(Path(source['background'])/'BACKGROUND.npz',allow_pickle=False) as z:release=z['cone_release'].copy()
    state=m.unpack(checkpoint['inner_flat']);aux=deepcopy(checkpoint['spike_states'])
    bc=old.LocalBC(m,seed,release);layer=object.__new__(RGCFunctionalV1);layer.model=m;layer.origin=bc.origin;layer.ac=SimpleNamespace(bc=bc,assembly=a)
    original_pack=m.pack
    def traced_pack(values):
        journal['pack_calls']+=1
        if journal['pack_calls']%1000==1:
            journal['phase']='before_original_pack'
            journal['contracts']=[dict(key=list(k),dtype=str(values[k].dtype),shape=list(values[k].shape),strides=list(values[k].strides),
                owned=bool(values[k].flags.owndata),aligned=bool(values[k].flags.aligned),base_type=type(values[k].base).__name__) for k in m.state_keys]
            marker()
        return original_pack(values)
    m.pack=traced_pack;nfev=0
    for index in range(2000,3000):
        if index%100==0:
            guard();journal.update(phase='before_original_event',index=index,model_time_seconds=index*.0001);marker()
            print(json.dumps(dict(index=index,wall_seconds=time.perf_counter()-start)),flush=True)
        cone,rod=bc.inputs(index);tick=index%10==0
        state,following,drawn=layer.commit(index,state,cone,rod,aux if tick else None)
        if tick:aux=following
        left,right=bc.origin+index*.0001,bc.origin+(index+1)*.0001
        sol=solve_ivp(layer.rhs,(left,right),m.pack(state),method='DOP853',rtol=1e-10,atol=1e-13,max_step=.000025,first_step=min(.000025,right-left),t_eval=[right])
        if not sol.success:raise ValueError(sol.message)
        state=m.unpack(sol.y[:,-1]);nfev+=sol.nfev
    save_bundle(OUT/'END_INNER_DIAGNOSTIC',kind='validation_evidence',identity=dict(model=m.identity,next_event_index=3000),
        payload=dict(inner_flat=m.pack(state),spike_states=aux,next_event_index=3000),
        qualification=dict(default_adopted=False,fit_eligible=False,FIT_READY=False,background_qualified=False,scope='Diagnostic replay only, not case completion'))
    result=dict(status='WINDOW_FINISHED_NOT_REPRODUCED',original_window_ms=[200,300],original_clock_precision_and_history_preserved=True,
        pythonmalloc=os.environ.get('PYTHONMALLOC'),no_scientific_parameter_changed=True,root_cause_confirmed=False,
        wall_seconds=time.perf_counter()-start,peak_rss_bytes=memory.peak_rss_bytes,nfev=nfev,pack_calls=journal['pack_calls'])
    old.write(OUT/'RESULT.json',result)


if __name__=='__main__':
    OUT.mkdir(exist_ok=False)
    with (OUT/'NATIVE_TRACE.log').open('x',encoding='utf8') as f:
        faulthandler.enable(file=f,all_threads=True)
        main()
