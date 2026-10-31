"""One frozen voltage-input GLM diagnostic; no neural solve or baseline refit."""
from pathlib import Path
import faulthandler
import os
import sys
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.run_cell6_matched_baselines_v1 import attachment_gate,read,sha,write,BASE,REFERENCE
STIM=ROOT/'reports/CELL6_STIMULUS_BASELINES_V1_20261031/matched_input'
TARGET='sg2:61468d3761f46cda:parasol_RGC_ON:8166'


def execute(out):
    import numpy as np
    from retina_sim import stimulus_baselines_v1 as model
    from retina_sim.temporal_readout_v1 import stable_terms
    from retina_sim.rgc_stage1_readout import observed_past
    from retina_sim.background_checkpoint import MemoryGuard
    from tools.cell6_baseline_metrics_v1 import metrics,paired,mc_precision
    from tools.durable_research_runtime_v1 import DurableRun
    journal=DurableRun(out/'runtime','voltage_matched_GLM_diagnostic')
    memory=MemoryGuard(24*1024**3,8*1024**3);started=time.perf_counter()
    def guard():
        if (out/'STOP').exists():raise RuntimeError('Explicit STOP')
        if not memory.check():raise MemoryError(str(memory.last))
        journal.heartbeat(memory=memory.last)
    def phase(name,**values):
        journal.heartbeat(force=True,phase=name,**values)
        import json
        print(json.dumps(dict(phase=name,**values)),flush=True)
    original=read(STIM/'PROTOCOL_FROZEN.json');sources=dict(original['source_pins'])
    for path in [Path(__file__),ROOT/'tools/deliver_cell6_voltage_glm_v1.py',
        STIM/'PROTOCOL_FROZEN.json',STIM/'MODELS_FROZEN.json',STIM/'BASIS.npz',
        *[STIM/f'g{g}_{suffix}' for g in range(6) for suffix in ('RESULT.json','PREDICTIONS.npz','MATCHED_INPUT.npz','INPUT_IDENTITY.json')]]:
        sources[str(path.relative_to(ROOT))]=sha(path)
    for name in ('retina_sim/stimulus_baselines_v1.py','tools/cell6_baseline_metrics_v1.py'):
        if sha(ROOT/name)!=original['source_pins'][str(Path(name))]:raise RuntimeError('Frozen baseline implementation changed')
    with np.load(STIM/'BASIS.npz',allow_pickle=False) as z:
        if not np.array_equal(z['stimulus_native'],model.stimulus_basis()) or not np.array_equal(z['history_ms'],model.history_basis()):
            raise RuntimeError('Baseline basis mismatch')
    protocol=dict(original)
    protocol.update(schema='cell6-voltage-matched-glm-development-diagnostic-v1',source_pins=sources,
        input='Original target8166 cached V in volts,100us native nodes[0,7.1); no old exponential filter, no gE/gI/stimulus features',
        input_comparison_limit='Same readout family as frozen stimulus GLM; source channels differ, no architecture-causal claim',
        prehistory='Constant first cached V before0; zero spikes before0; full0..1s warmup; no reset at score start',
        parameter_counts=dict(V_GLM=30,S_GLM=30,T_readout=7),
        target=TARGET,members=2,original_V_units='V',voltage_node_sample_rule='Use V[:-1] on exact T100us nodes; same causal FIR and1ms bin-start sample',
        voltage_prehistory_limit='Constant equilibrium-initialized V0 is a readout boundary convention, not measured pre-acquisition voltage',
        candidate_count=1,train_families=['GLM'],stimulus_GLM_refit=False,retinal_model_solves=0,
        original_g5_decision_unchanged='INCONCLUSIVE',development_repeatedly_touched=True)
    preserved=('stimulus_support_ms','stimulus_basis','GLM_history','likelihood','regularization','ridge_candidates',
        'selection','optimizer','standardization','score_window_seconds','PSTH_bin_ms','free_GLM','MC_error','resampling')
    if any(protocol[k]!=original[k] for k in preserved):raise RuntimeError('Undeclared readout-method difference')
    write(out/'PROTOCOL_FROZEN.json',protocol)
    write(out/'METHOD_MATCHING.json',dict(identical_method_keys=list(preserved),all_identical=True,
        stimulus_basis_and_history_arrays_bitwise_identical=True,old_single_exponential_not_used=True,
        input_difference='Scalar membrane V replaces held optical-stimulus factor; boundary constant is own first value'))
    np.savez_compressed(out/'BASIS.npz',stimulus_native=model.stimulus_basis(),history_ms=model.history_basis())
    phase('CONTRACT_FROZEN_TRAINING_NOT_STARTED')
    cases={}
    def load_case(g):
        if g>=3 and not (out/'MODELS_FROZEN.json').exists():raise RuntimeError('New development access before freeze')
        folder=REFERENCE/'cache'/f'g{g}';manifest=read(folder/'manifest.json')
        T=read(REFERENCE/f'readout/g{g}_RESULT.json')
        if manifest['target']!=TARGET or manifest['members']!=2 or manifest['units']['V']!='V':raise RuntimeError('Target/units differ')
        if sha(folder/'manifest.json')!=T['cache_manifest_sha256'] or sha(folder/'endogenous.npz')!=manifest['arrays_sha256']:
            raise RuntimeError('Original T cache identity differs')
        with np.load(folder/'endogenous.npz',allow_pickle=False) as z:times=z['time_source_s'].copy();voltage=z['V'].copy()
        if not np.array_equal(times,np.arange(71001)*.0001) or voltage.shape!=(71001,) or not np.isfinite(voltage).all():
            raise RuntimeError('Actual voltage/time grid differs')
        with np.load(BASE/'SOURCE_PREPARATION.npz',allow_pickle=False) as z:counts=z[f'g{g}_counts'].copy()
        with np.load(STIM/f'g{g}_PREDICTIONS.npz',allow_pickle=False) as z:
            if not np.array_equal(counts,z['observed_counts']):raise RuntimeError('Frozen observations differ')
        write(out/f'g{g}_INPUT_IDENTITY.json',dict(group=g,target=TARGET,members=2,source_array_sha256=manifest['arrays_sha256'],
            T_used_cache_manifest_sha256=T['cache_manifest_sha256'],units='V',native_nodes=71000,node_dt_s=.0001,
            pre0='constant own V0',first_V_V=float(voltage[0]),old_temporal_state_used=False,
            source_fields_used=['V','time_source_s'],retinal_model_solved=False))
        return dict(group=g,counts=counts,stimulus_features=model.stimulus_features(voltage[:-1]),history_features=model.history_features(counts))
    try:
        for g in (0,1,2):cases[g]=load_case(g);phase('TRAINING_FEATURES_READY',group=g)
        phase('TRAINING',model='V_GLM')
        fitted,cv=model.choose_and_fit(cases,'GLM',guard)
        write(out/'CROSS_VALIDATION.json',cv)
        write(out/'MODELS_FROZEN.json',dict(protocol_sha256=sha(out/'PROTOCOL_FROZEN.json'),source_pins=sources,
            models=dict(V_GLM=fitted),development_outcomes_used=False))
        freeze=sha(out/'MODELS_FROZEN.json');phase('MODEL_FROZEN',ridge=fitted['ridge'])
        core=read(REFERENCE/'readout/MODEL.json')['core'];Smodel=read(STIM/'MODELS_FROZEN.json')['models']['GLM']
        with np.load(BASE/'BOOTSTRAP_INDICES.npz',allow_pickle=False) as z:trials=z['spike_repeats'];blocks=z['time_blocks']
        all_rows=[];evaluation_started=time.perf_counter()
        for g in range(6):
            guard()
            if sha(out/'MODELS_FROZEN.json')!=freeze:raise RuntimeError('Candidate changed after freeze')
            if g not in cases:cases[g]=load_case(g)
            case=cases[g];counts=case['counts'];past=observed_past(counts)
            with np.load(REFERENCE/f'readout/g{g}_PREDICTIONS.npz',allow_pickle=False) as z:
                Tprob=z['T_probability'].copy();signal=core['b']+core['w0']*z['x']+core['w1']*z['readout_state_a']
            with np.load(STIM/f'g{g}_PREDICTIONS.npz',allow_pickle=False) as z:Sprob=z['GLM_probability'].copy()
            with np.load(STIM/f'g{g}_MATCHED_INPUT.npz',allow_pickle=False) as z:held=z['relative_Td_factor'].copy()
            Scase=dict(case,stimulus_features=model.stimulus_features(held))
            Tloss=stable_terms(signal[None,:]+past@core['eta'],90.,counts)[0];Sloss=model.conditional(Scase,Smodel)
            reference=read(STIM/f'g{g}_RESULT.json')
            for name,loss in [('T',Tloss),('GLM',Sloss)]:
                if abs(float(loss[:,model.LO:model.HI].mean())-reference['metrics'][name]['conditional_NLL_nat_per1ms'])>1e-12:
                    raise RuntimeError('Frozen reference per-bin loss reconstruction differs')
            phase('FROZEN_VOLTAGE_PREDICTION',group=g)
            tick=time.perf_counter();Vprob,batches=model.predict(case,fitted,seed=2026103100+g,guard=guard)
            Vloss=model.conditional(case,fitted);mc,se=mc_precision(counts,batches)
            prediction_seconds=time.perf_counter()-tick
            probabilities=dict(T=Tprob,V_GLM=Vprob,S_GLM=Sprob);losses=dict(T=Tloss,V_GLM=Vloss,S_GLM=Sloss)
            scores={name:metrics(counts,p,losses[name]) for name,p in probabilities.items()}
            for name,old in [('T','T'),('S_GLM','GLM')]:
                if scores[name]!=reference['metrics'][old]:raise RuntimeError('Frozen reference scores differ')
            contrast=paired(counts,dict(T=Vprob,LN=Tprob,GLM=Sprob),dict(T=Vloss,LN=Tloss,GLM=Sloss),trials,blocks)
            pairs={}
            for alias,name in [('LN','T'),('GLM','S_GLM')]:
                c=contrast[alias];c['direction']='positive favors V_GLM; delta NLL/MSE=reference minus V; delta r=V minus reference'
                c['metric_conclusions']={key:{'T_better':'V_GLM_better','baseline_better':'reference_better','uncertain':'uncertain'}[value]
                    for key,value in c['metric_conclusions'].items()};pairs[name]=c
            recovery={}
            for key,sign in [('free_r',1.),('free_MSE_Hz2',-1.),('conditional_NLL_nat_per1ms',-1.)]:
                gap=sign*(scores['S_GLM'][key]-scores['T'][key]);regained=sign*(scores['V_GLM'][key]-scores['T'][key])
                recovery[key]=dict(original_gap=gap,recovered=regained,recovered_fraction=None if abs(gap)<1e-12 else regained/gap,
                    interpretation='Descriptive point fraction only; paired CI reported separately, not equivalence evidence')
            write(out/f'g{g}_RESULT.json',dict(group=g,split='train' if g<3 else 'repeatedly_touched_development',
                models_frozen_sha256=freeze,metrics=scores,paired_V_vs_reference=pairs,gap_recovery=recovery,
                V_GLM_Monte_Carlo=mc,S_GLM_Monte_Carlo=reference['GLM_Monte_Carlo'],prediction_wall_seconds=prediction_seconds,
                no_observed_spikes_in_free_prediction=True,reference_free_predictions_reused=True,conditional_history_strictly_past=True))
            np.savez_compressed(out/f'g{g}_PREDICTIONS.npz',**{k+'_probability':v for k,v in probabilities.items()},
                observed_counts=counts,time_bin_start_seconds=np.arange(7100)*.001,V_GLM_batch_probability=batches,V_GLM_20ms_rate_SE_Hz=se)
            all_rows.extend(dict(group=g,model=name,**value) for name,value in scores.items());phase('GROUP_COMPLETE',group=g)
        for name,expected in sources.items():
            if sha(ROOT/name)!=expected:raise RuntimeError('Pinned source changed: '+name)
        write(out/'ALL_METRICS.json',all_rows)
        write(out/'COMPUTATION_COMPLETE.json',dict(status='COMPLETE_VOLTAGE_MATCHED_GLM_DIAGNOSTIC',groups=6,
            new_final_fits=1,CV_fits=12,new_development_evaluations=3,retinal_model_solves=0,baseline_refits=0,
            wall_seconds=time.perf_counter()-started,evaluation_wall_seconds=time.perf_counter()-evaluation_started,
            peak_RSS_bytes=memory.peak_rss_bytes,historical_test_accessed=False,original_g5_decision='INCONCLUSIVE',
            FIT_READY=False,background_qualified=False,production_adopted=False))
        journal.finish('COMPLETE')
    except BaseException as error:
        write(out/'FAILURE.json',dict(error=str(error),original_results_modified=False));journal.finish('FAILED');raise


if __name__=='__main__':
    nonce,folder=sys.argv[1:];out=Path(folder)
    with (out/'NATIVE_FAULTHANDLER.log').open('x',encoding='utf8') as stream:
        faulthandler.enable(file=stream,all_threads=True);attachment_gate(nonce,out);execute(out)
