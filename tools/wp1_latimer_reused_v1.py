"""Fit independent real-cell readouts only when full neural input identity matches."""
from pathlib import Path
import faulthandler,hashlib,json,os,sys,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.run_cell6_matched_baselines_v1 import attachment_gate,read,sha,write,BASE,REFERENCE
STIM=ROOT/'reports/CELL6_STIMULUS_BASELINES_V1_20261031/matched_input'
TARGET='sg2:61468d3761f46cda:parasol_RGC_ON:8166'


def execute(cell,out):
    import numpy as np
    from scipy.io import loadmat
    from retina_sim import stimulus_baselines_v1 as model
    from tools.cell6_baseline_metrics_v1 import metrics,paired,mc_precision
    from tools.cell6_raw_intake_v1 import selected_spikes
    from tools.cell6_matched_input_v1 import matched_input
    from tools.durable_research_runtime_v1 import DurableRun
    from retina_sim.background_checkpoint import MemoryGuard
    if cell not in (1,5):raise RuntimeError('Only exact shared-six-input cells1/5 admitted by this adapter')
    journal=DurableRun(out/'runtime',f'WP1_Latimer_Cell_{cell}');memory=MemoryGuard(24*1024**3,8*1024**3)
    started=time.perf_counter();raw_path=ROOT/f'data/source_data/latimer2019_figshare_v1/Cell_{cell}.mat'
    metadata=read(raw_path.parent/'article_metadata.json');entry=next(v for v in metadata['files'] if v['name']==raw_path.name)
    if raw_path.stat().st_size!=entry['size'] or hashlib.md5(raw_path.read_bytes()).hexdigest()!=entry['computed_md5']:
        raise RuntimeError('Official raw file identity differs')
    def guard():
        if (out/'STOP').exists():raise RuntimeError('Explicit STOP')
        if not memory.check():raise MemoryError(str(memory.last))
        journal.heartbeat(memory=memory.last)
    def phase(name,**values):
        journal.heartbeat(force=True,phase=name,**values);print(json.dumps(dict(cell=cell,phase=name,**values)),flush=True)
    sources={str(p.relative_to(ROOT)):sha(p) for p in [raw_path,Path(__file__),STIM/'PROTOCOL_FROZEN.json',
        BASE/'SOURCE_IDENTITY.json',BASE/'BOOTSTRAP_INDICES.npz',BASE/'halo_model_v2/seed/manifest.json',
        ROOT/'retina_sim/stimulus_baselines_v1.py',ROOT/'tools/cell6_baseline_metrics_v1.py',ROOT/'tools/cell6_raw_intake_v1.py',ROOT/'tools/cell6_matched_input_v1.py']}
    for g in range(6):
        for filename in ('manifest.json','endogenous.npz'):
            path=REFERENCE/f'cache/g{g}'/filename;sources[str(path.relative_to(ROOT))]=sha(path)
    original=read(STIM/'PROTOCOL_FROZEN.json');protocol=dict(original)
    protocol.update(schema='wp1-latimer-shared-input-independent-cell-readout-v1',source_pins=sources,cell_id=f'Cell_{cell}',
        retinal_type='ON-parasol',source_eccentricity_deg=None,source_azimuth_deg=None,
        proxy_target=TARGET,proxy_members=2,proxy_coordinate_deg=[4.738358015,.056615877],
        qualification='Conditional shared engineering proxy, not measured source eccentricity or per-cell anatomy',
        new_candidates=['V_GLM'],paired_baselines=['S_GLM','LN','constant'],
        parameter_counts=dict(V_GLM=30,S_GLM=30,LN=17,constant=1),
        split='Same six exact stimuli as approved Cell6 development subset; groups0..2 train,3..5 fixed evaluation for this cell',
        remaining_source_groups='Four other stimuli quarantined, not silently counted as fitted/evaluated; protects historic Cell6 test-stimulus boundary',
        input='Raw cached8166 V directly,100us nodes; same500ms FIR and200ms causal history as stimulus GLM',
        prehistory='Constant own first V or held stimulus value before0; spike pre0 zero; entire0..1s warmup',
        original_Cell6_results_unchanged=True,retinal_model_solves=0)
    write(out/'PROTOCOL_FROZEN.json',protocol);phase('CONTRACT_FROZEN')
    try:
        stimuli=loadmat(raw_path,variable_names=['SpikeStm'])['SpikeStm']
        if stimuli.shape!=(30,71000):raise RuntimeError('Thirty original trials required')
        hashes=[hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in stimuli]
        original_cases=read(BASE/'SOURCE_IDENTITY.json')['cases'];rows={c['group']:[i for i,h in enumerate(hashes) if h==c['stimulus_sha256']] for c in original_cases}
        if any(len(v)!=3 for v in rows.values()):raise RuntimeError('Shared stimulus repeats do not match')
        spikes=selected_spikes(raw_path,{i for values in rows.values() for i in values})
        arrays={};cases=[]
        for c in original_cases:
            g=c['group'];raw=stimuli[rows[g][0]].copy();counts=np.stack([np.bincount((spikes[i]-1)//10,minlength=7100) for i in rows[g]])
            if counts.shape!=(3,7100) or not np.isin(counts,(0,1)).all():raise RuntimeError('Declared binary1ms source support differs; no clipping')
            arrays[f'g{g}_stimulus']=raw;arrays[f'g{g}_counts']=counts
            cases.append(dict(group=g,stimulus_sha256=c['stimulus_sha256'],original_trial_rows=rows[g],repeats=3,
                source_seconds=7.1,score_seconds=[1,7],split='train' if g<3 else 'fixed_development'))
        np.savez_compressed(out/'SOURCE_PREPARATION.npz',**arrays)
        write(out/'SOURCE_IDENTITY.json',dict(raw_source_sha256=sha(raw_path),cases=cases,source_eccentricity_deg=None,
            source_type='ON-parasol',proxy=TARGET,unique_source_stimuli=10,admitted_unique_stimuli=6,
            admitted_spike_repeats=18,quarantined_trial_rows=[i for i in range(30) if i not in spikes],
            original_Cell6_spikes_decoded=False,matched_cache_reuse_requires_exact_input=True))
        sets={};case_identity=[]
        def load_case(g):
            if g>=3 and not (out/'MODELS_FROZEN.json').exists():raise RuntimeError('Evaluation feature access before freeze')
            raw=arrays[f'g{g}_stimulus'];counts=arrays[f'g{g}_counts'];held=matched_input(raw,g,out)
            folder=REFERENCE/'cache'/f'g{g}';manifest=read(folder/'manifest.json')
            if manifest['target']!=TARGET or manifest['members']!=2 or sha(folder/'endogenous.npz')!=manifest['arrays_sha256']:
                raise RuntimeError('Original proxy cache changed')
            sources[str((folder/'endogenous.npz').relative_to(ROOT))]=manifest['arrays_sha256']
            with np.load(folder/'endogenous.npz',allow_pickle=False) as z:times=z['time_source_s'];V=z['V'].copy()
            if not np.array_equal(times,np.arange(71001)*.0001):raise RuntimeError('Proxy time grid differs')
            common=dict(group=g,counts=counts,history_features=model.history_features(counts))
            result=dict(V=dict(common,stimulus_features=model.stimulus_features(V[:-1])),S=dict(common,stimulus_features=model.stimulus_features(held)))
            case_identity.append(dict(group=g,exact_full_T_input_program=True,neural_cache=manifest['cache_identity'],
                neural_arrays_sha256=manifest['arrays_sha256'],no_new_forward=True))
            return result
        for g in (0,1,2):sets[g]=load_case(g)
        models={};training_started=time.perf_counter()
        for name,source,family in [('V_GLM','V','GLM'),('S_GLM','S','GLM'),('LN','S','LN')]:
            phase('TRAINING',model=name);fit,cv=model.choose_and_fit({g:sets[g][source] for g in (0,1,2)},family,guard)
            models[name]=fit;write(out/(name+'_CROSS_VALIDATION.json'),cv)
        train_probability=float(np.concatenate([arrays[f'g{g}_counts'][:,1000:7000].ravel() for g in (0,1,2)]).mean())
        write(out/'MODELS_FROZEN.json',dict(models=models,constant_probability=train_probability,source_pins=sources.copy(),
            protocol_sha256=sha(out/'PROTOCOL_FROZEN.json'),evaluation_outcomes_used=False,training_wall_seconds=time.perf_counter()-training_started))
        freeze=sha(out/'MODELS_FROZEN.json');phase('MODELS_FROZEN');preparation_training_seconds=time.perf_counter()-started
        with np.load(BASE/'BOOTSTRAP_INDICES.npz',allow_pickle=False) as z:trials=z['spike_repeats'];blocks=z['time_blocks']
        all_rows=[];evaluation_started=time.perf_counter()
        for g in range(6):
            guard()
            if sha(out/'MODELS_FROZEN.json')!=freeze:raise RuntimeError('Frozen model changed')
            if g not in sets:sets[g]=load_case(g)
            counts=arrays[f'g{g}_counts'];probabilities={};losses={};mc={};extra={}
            for name,source in [('V_GLM','V'),('S_GLM','S'),('LN','S')]:
                phase('FIXED_PREDICTION',group=g,model=name)
                probabilities[name],batch=model.predict(sets[g][source],models[name],seed=2026103100+g,guard=guard)
                losses[name]=model.conditional(sets[g][source],models[name])
                if batch is not None:
                    mc[name],extra[name+'_20ms_rate_SE_Hz']=mc_precision(counts,batch)
                    extra[name+'_batch_probability']=batch
            probabilities['constant']=np.full(7100,train_probability)
            losses['constant']=np.where(counts,-np.log(train_probability),-np.log1p(-train_probability))
            values={name:metrics(counts,p,losses[name]) for name,p in probabilities.items()}
            comparisons=paired(counts,dict(T=probabilities['V_GLM'],LN=probabilities['LN'],GLM=probabilities['S_GLM']),
                dict(T=losses['V_GLM'],LN=losses['LN'],GLM=losses['S_GLM']),trials,blocks)
            for c in comparisons.values():
                c['direction']='positive favors V_GLM; reference minus V for loss/MSE, V minus reference for r'
                c['metric_conclusions']={k:{'T_better':'V_GLM_better','baseline_better':'reference_better','uncertain':'uncertain'}[v] for k,v in c['metric_conclusions'].items()}
            write(out/f'g{g}_RESULT.json',dict(cell=cell,group=g,metrics=values,paired_V_vs_baseline=comparisons,
                Monte_Carlo=mc,models_frozen_sha256=freeze,source_eccentricity_deg=None,conditional_proxy=True,
                no_observed_evaluation_history_in_free=True))
            np.savez_compressed(out/f'g{g}_PREDICTIONS.npz',**{k+'_probability':v for k,v in probabilities.items()},
                observed_counts=counts,time_bin_start_seconds=np.arange(7100)*.001,**extra)
            all_rows.extend(dict(cell=cell,group=g,model=name,**v) for name,v in values.items());phase('GROUP_COMPLETE',group=g)
        for name,value in sources.items():
            if sha(ROOT/name)!=value:raise RuntimeError('Source changed: '+name)
        write(out/'ALL_METRICS.json',all_rows);write(out/'CACHE_REUSE.json',case_identity)
        write(out/'COMPUTATION_COMPLETE.json',dict(status='CONDITIONAL_PROXY_READOUT_COMPLETE',cell=cell,source_type='ON-parasol',
            source_eccentricity_deg=None,groups=6,source_stimuli_not_admitted=4,retinal_model_solves=0,
            per_target_independent_fits=True,new_final_fits=3,preparation_training_wall_seconds=preparation_training_seconds,
            evaluation_wall_seconds=time.perf_counter()-evaluation_started,wall_seconds=time.perf_counter()-started,
            peak_RSS_bytes=memory.peak_rss_bytes,original_Cell6_test_accessed=False,FIT_READY=False,background_qualified=False,production_adopted=False))
        journal.finish('COMPLETE')
    except BaseException as error:
        write(out/'FAILURE.json',dict(error=repr(error),original_results_modified=False));journal.finish('FAILED');raise


if __name__=='__main__':
    nonce,folder,number=sys.argv[1:];out=Path(folder)
    with (out/'NATIVE_FAULTHANDLER.log').open('x',encoding='utf8') as stream:
        faulthandler.enable(file=stream,all_threads=True);attachment_gate(nonce,out);execute(int(number),out)
