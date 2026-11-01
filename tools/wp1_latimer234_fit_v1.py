"""Independent frozen readouts from approved cached proxy trajectories only."""
from pathlib import Path
import hashlib,json,os,sys,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/WP1_LATIMER_CELL2_4_V1'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')


def choose(cases,family,guard):
    import numpy as np
    from retina_sim import stimulus_baselines_v1 as m
    groups=sorted(cases);records=[]
    for ridge in m.GRID:
        folds=[]
        for hold in groups:
            fit=m.fit(cases,[g for g in groups if g!=hold],family,ridge,guard)
            score=float(m.conditional(cases[hold],fit)[:,m.LO:m.HI].mean()) if fit['converged'] else None
            folds.append(dict(held_out_stimulus=hold,NLL=score,fit=fit))
        scores=[v['NLL'] for v in folds]
        records.append(dict(ridge=ridge,folds=folds,mean_heldout_NLL=None if any(s is None for s in scores) else float(np.mean(scores))))
    valid=[r for r in records if r['mean_heldout_NLL'] is not None]
    if not valid:raise RuntimeError('Original finite ridge selection did not converge; no expanded search')
    best=min(r['mean_heldout_NLL'] for r in valid)
    selected=max(r['ridge'] for r in valid if r['mean_heldout_NLL']<=best+1e-8)
    fit=m.fit(cases,groups,family,selected,guard)
    if not fit['converged']:raise RuntimeError('Final original optimizer did not converge')
    return fit,records


def held_input(raw,g,target):
    import numpy as np
    from retina_sim.contracts import canonical_hash
    from tools.cell6_matched_input_v1 import dictionary,owned_array
    folder=OUT/'cache'/f'g{g}';identity=read(folder/'INPUT_AND_MODEL_IDENTITY.json');mapping=identity['input_mapping']
    if mapping['native_input_fps']!=150. or mapping['formula']!='Btd*(1+0.05*(raw[j]-raw[0]))':raise RuntimeError('Frozen input mapping differs')
    seed=ROOT/'reports/CELL6_RAW_REBUILD_CONDITIONAL_READOUT_V1/halo_model_v2/seed';manifest=read(seed/'manifest.json')
    descriptor=dictionary(dictionary(manifest['payload'])['input_reference'])['native_background_td']
    background=owned_array(seed,manifest,descriptor)
    indices={int(k):int(v) for k,v in identity['frame_source_indices'].items()};frames=sorted(indices)
    if frames!=list(range(1065)):raise RuntimeError('Complete original150Hz input required')
    factors=np.asarray([1+.05*(float(raw[indices[f]])-float(raw[0])) for f in frames])
    if canonical_hash({f:background*factors[f] for f in frames})!=identity['input_payload_sha256']:raise RuntimeError('Full light program hash differs')
    latest=read(folder/'LATEST_CHECKPOINT.json');checkpoint=folder/latest['directory'];saved=read(checkpoint/'manifest.json')
    anchor=dictionary(dictionary(dictionary(saved['payload'])['native'])['point'])['anchor'];anchor=dictionary(anchor)
    epoch=anchor['input_epoch'];origin=anchor['time_seconds']-epoch/150.
    if origin!=anchor['input_stop_seconds']-(epoch+1)/150.:raise RuntimeError('Native input clock origin differs')
    if not np.array_equal(owned_array(checkpoint,saved,anchor['light']),background*factors[epoch]):raise RuntimeError('Saved native illumination differs')
    times=np.arange(71000)*.0001;starts=origin+np.asarray(frames)/150.;ends=origin+(np.asarray(frames)+1)/150.
    frame_at=np.searchsorted(starts,times,side='right')-1
    if not np.all((frame_at>=0)&(times>=starts[frame_at])&(times<ends[frame_at])):raise RuntimeError('Right-continuous frame coverage differs')
    held=factors[frame_at]
    np.savez_compressed(target/f'g{g}_MATCHED_INPUT.npz',time_source_s=times,relative_Td_factor=held)
    return held


def execute(cell,mode,*,runtime_identity_file=None):
    import numpy as np
    from retina_sim import stimulus_baselines_v1 as m
    from tools.wp1_latimer_metrics_v1 import metrics,paired,mc_precision
    from tools.durable_research_runtime_v1 import DurableRun
    from retina_sim.background_checkpoint import MemoryGuard
    target=OUT/'targets'/f'Latimer_Cell_{cell}';contract=read(OUT/'PROTOCOL_FROZEN.json');c=contract['cells'][str(cell)]
    journal=DurableRun(target/('runtime_'+mode),'independent_proxy_readout');memory=MemoryGuard(24*1024**3,8*1024**3)
    began=time.perf_counter();pins={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'PROTOCOL_FROZEN.json',target/'SOURCE_COUNTS.npz',target/'SOURCE_IDENTITY.json',Path(__file__),ROOT/'retina_sim/stimulus_baselines_v1.py',ROOT/'tools/wp1_latimer_metrics_v1.py']}
    def guard():
        if (OUT/'REQUEST_STOP').exists():raise RuntimeError('Explicit STOP')
        if not memory.check():raise MemoryError(str(memory.last))
        journal.heartbeat(memory=memory.last)
    def phase(name,**v):journal.heartbeat(force=True,phase=name,**v);print(json.dumps(dict(cell=cell,phase=name,**v)),flush=True)
    def case(g):
        folder=OUT/'cache'/f'g{g}';manifest=read(folder/'manifest.json')
        if manifest['target']!=c['target'] or manifest['members']!=c['members'] or not manifest['physical_gate_passed']:raise RuntimeError('Proxy binding mismatch')
        if sha(folder/'endogenous.npz')!=manifest['arrays_sha256']:raise RuntimeError('Neural arrays changed')
        for name in ('manifest.json','endogenous.npz','INPUT_AND_MODEL_IDENTITY.json'):pins[str((folder/name).relative_to(ROOT))]=sha(folder/name)
        with np.load(folder/'endogenous.npz',allow_pickle=False) as z:V=z['V'].copy();times=z['time_source_s']
        if not np.array_equal(times,np.arange(71001)*.0001):raise RuntimeError('Source time grid differs')
        with np.load(target/'SOURCE_COUNTS.npz',allow_pickle=False) as z:counts=z[f'g{g}_counts'].copy()
        with np.load(OUT/'preparation/SOURCE_PREPARATION.npz',allow_pickle=False) as z:raw=z[f'g{g}_stimulus'].copy()
        common=dict(group=g,counts=counts,history_features=m.history_features(counts))
        return dict(V=dict(common,stimulus_features=m.stimulus_features(V[:-1])),S=dict(common,stimulus_features=m.stimulus_features(held_input(raw,g,target))))
    try:
        if mode=='train':
            if (target/'MODELS_FROZEN.json').exists():raise FileExistsError('Frozen independent model retained')
            sets={g:case(g) for g in c['training_groups']};models={}
            for name,source,family in [('V_GLM','V','GLM'),('S_GLM','S','GLM'),('LN','S','LN')]:
                phase('TRAINING',model=name)
                models[name],cv=choose({g:sets[g][source] for g in sets},family,guard);write(target/(name+'_CROSS_VALIDATION.json'),cv)
            probability=float(np.concatenate([sets[g]['V']['counts'][:,m.LO:m.HI].ravel() for g in sets]).mean())
            write(target/'MODELS_FROZEN.json',dict(models=models,constant_probability=probability,source_pins=pins,protocol_sha256=sha(OUT/'PROTOCOL_FROZEN.json'),training_groups=c['training_groups'],development_outcomes_used=False,training_wall_seconds=time.perf_counter()-began,
                Python_runtime=read(runtime_identity_file) if runtime_identity_file else None))
            phase('MODELS_FROZEN')
        elif mode=='evaluate':
            freeze=sha(target/'MODELS_FROZEN.json');frozen=read(target/'MODELS_FROZEN.json');rows=[]
            for name,h in frozen['source_pins'].items():
                if sha(ROOT/name)!=h:raise RuntimeError('Training source changed: '+name)
            for g in c['development_groups']:
                if (target/f'g{g}_RESULT.json').exists():raise FileExistsError('Fixed evaluation retained; no repeat selection')
                phase('FIXED_EVALUATION',group=g);sets=case(g);counts=sets['V']['counts'];probabilities={};losses={};mc={};extra={}
                for name,source in [('V_GLM','V'),('S_GLM','S'),('LN','S')]:
                    probabilities[name],batch=m.predict(sets[source],frozen['models'][name],seed=2026103100+cell*100+g,guard=guard)
                    losses[name]=m.conditional(sets[source],frozen['models'][name])
                    if batch is not None:mc[name],extra[name+'_20ms_rate_SE_Hz']=mc_precision(counts,batch)
                q=frozen['constant_probability'];probabilities['constant']=np.full(7100,q);losses['constant']=np.where(counts,-np.log(q),-np.log1p(-q))
                values={name:metrics(counts,p,losses[name]) for name,p in probabilities.items()}
                rng=np.random.Generator(np.random.PCG64(2026103100+cell*100+g));tr=rng.integers(len(counts),size=(1000,len(counts)));bl=rng.integers(30,size=(1000,30))
                comparisons=paired(counts,dict(T=probabilities['V_GLM'],LN=probabilities['LN'],GLM=probabilities['S_GLM']),dict(T=losses['V_GLM'],LN=losses['LN'],GLM=losses['S_GLM']),tr,bl)
                write(target/f'g{g}_RESULT.json',dict(cell=cell,group=g,metrics=values,paired_V_vs_baseline=comparisons,Monte_Carlo=mc,repeats=len(counts),single_repeat_reliability_claimed=False,frozen_model_sha256=freeze,source_eccentricity_deg=None,conditional_proxy=True))
                np.savez_compressed(target/f'g{g}_PREDICTIONS.npz',observed_counts=counts,time_bin_start_seconds=np.arange(7100)*.001,**{k+'_probability':v for k,v in probabilities.items()},**extra)
                np.savez_compressed(target/f'g{g}_BOOTSTRAP.npz',spike_repeats=tr,time_blocks=bl)
                rows.extend(dict(cell=cell,group=g,model=name,**v) for name,v in values.items())
                if sha(target/'MODELS_FROZEN.json')!=freeze:raise RuntimeError('Frozen parameters changed')
            write(target/'ALL_METRICS.json',rows);write(target/'EVALUATION_SOURCE_PINS.json',pins)
            write(target/'COMPUTATION_COMPLETE.json',dict(status='INDEPENDENT_CONDITIONAL_PROXY_READOUT_COMPLETE',cell=cell,training_groups=c['training_groups'],development_groups=c['development_groups'],groups=c['unique_stimuli'],source_eccentricity_deg=None,per_target_independent_fits=True,wall_seconds=time.perf_counter()-began,FIT_READY=False,background_qualified=False,production_adopted=False))
        else:raise ValueError('Unknown readout phase')
        journal.finish('COMPLETE')
    except BaseException as error:
        write(target/(mode+'_FAILURE.json'),dict(error=repr(error)));journal.finish('FAILED');raise


if __name__=='__main__':execute(int(sys.argv[1]),sys.argv[2])
