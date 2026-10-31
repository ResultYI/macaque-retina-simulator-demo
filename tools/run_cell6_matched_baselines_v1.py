"""One predeclared LN/GLM comparison using saved Cell6 inputs; no retinal solve."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import faulthandler
import hashlib
import json
import os
import sys
import threading
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
BASE=ROOT/'reports/CELL6_RAW_REBUILD_CONDITIONAL_READOUT_V1'
REFERENCE=BASE/'runtime_monitor_repair_v1'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    from tools.durable_research_runtime_v1 import atomic_json
    atomic_json(path,value)


def attachment_gate(nonce,output):
    from tools.durable_research_runtime_v1 import process_identity
    ctypes.pythonapi.PyThreadState_Get.argtypes=[]
    ctypes.pythonapi.PyThreadState_Get.restype=ctypes.c_void_p
    (output/'PYTHON_TSTATE.txt').write_text(str(os.getpid())+'\t'+str(threading.get_native_id())+'\t'+format(ctypes.pythonapi.PyThreadState_Get(),'x'),encoding='ascii')
    write(output/'WORKER_READY.json',dict(nonce=nonce,identity=process_identity(os.getpid()),argv=sys.argv))
    start=time.monotonic()
    while not (output/'GO.json').exists():
        if time.monotonic()-start>120:raise TimeoutError('Native attachment gate')
        time.sleep(.1)
    go=read(output/'GO.json')
    if go['pid']!=os.getpid() or go['nonce']!=nonce or not ctypes.WinDLL('kernel32').IsDebuggerPresent():
        raise RuntimeError('Own native attachment identity mismatch')


def execute(output):
    import numpy as np
    from retina_sim import stimulus_baselines_v1 as model
    from retina_sim.temporal_readout_v1 import stable_terms
    from retina_sim.rgc_stage1_readout import observed_past
    from retina_sim.background_checkpoint import MemoryGuard
    from tools.durable_research_runtime_v1 import DurableRun
    from tools.cell6_baseline_metrics_v1 import metrics,paired,mc_precision
    from tools.cell6_matched_input_v1 import matched_input
    journal=DurableRun(output/'runtime','stimulus_LN_GLM_comparison')
    memory=MemoryGuard(24*1024**3,8*1024**3)
    def guard():
        if (output/'STOP').exists():raise RuntimeError('Explicit STOP')
        if not memory.check():raise MemoryError(str(memory.last))
        journal.heartbeat(memory=memory.last)
    def phase(name,**values):
        journal.heartbeat(force=True,phase=name,**values)
        print(json.dumps(dict(phase=name,**values)),flush=True)
    sources={str(p.relative_to(ROOT)):sha(p) for p in [BASE/'SOURCE_PREPARATION.npz',BASE/'SOURCE_IDENTITY.json',BASE/'BOOTSTRAP_INDICES.npz',
        REFERENCE/'readout/MODEL.json',REFERENCE/'readout/READOUT_FIT.json',ROOT/'retina_sim/stimulus_baselines_v1.py',
        ROOT/'tools/cell6_baseline_metrics_v1.py',Path(__file__)]}
    for path in [ROOT/'retina_sim/temporal_readout_v1.py',ROOT/'retina_sim/rgc_stage1_readout.py',
        *[REFERENCE/f'readout/g{g}_{suffix}' for g in range(6) for suffix in ('RESULT.json','PREDICTIONS.npz')]]:
        sources[str(path.relative_to(ROOT))]=sha(path)
    parent=ROOT/'reports/CELL6_STIMULUS_BASELINES_V1_20261031'
    original=read(parent/'PROTOCOL_FROZEN.json')
    if sha(ROOT/'retina_sim/stimulus_baselines_v1.py')!=original['source_pins'][r'retina_sim\stimulus_baselines_v1.py'] or sha(ROOT/'tools/cell6_baseline_metrics_v1.py')!=original['source_pins'][r'tools\cell6_baseline_metrics_v1.py']:
        raise RuntimeError('Original method implementation changed')
    extra=[parent/'PROTOCOL_FROZEN.json',ROOT/'tools/cell6_matched_input_v1.py',BASE/'halo_model_v2/seed/manifest.json']
    for g in range(6):
        folder=REFERENCE/'cache'/f'g{g}'
        latest=read(folder/'LATEST_CHECKPOINT.json')
        extra.extend([folder/'INPUT_AND_MODEL_IDENTITY.json',folder/'LATEST_CHECKPOINT.json',folder/'manifest.json',
            folder/'endogenous.npz',folder/latest['directory']/'manifest.json'])
    for path in extra:sources[str(path.relative_to(ROOT))]=sha(path)
    protocol=dict(original)
    protocol.update(schema='cell6-stimulus-ln-glm-matched-input-supplement-v1',source_pins=sources,
        input='Exact T150Hz held scalar relativeTd factor, using stored frame_source_indices, actual saved input clock and T100us nodes; full vector input SHA must match',
        input_comparison_limit='Stimulus information and timing matched to T; GLM200ms history versus T3ms remains a full-pipeline difference, not an architecture causal ablation',
        parent_protocol_sha256=sha(parent/'PROTOCOL_FROZEN.json'),method_changed=False,
        input_identity_must_close_before_training=True)
    excluded={'schema','source_pins','input','input_comparison_limit'}
    unchanged=[k for k in original if k not in excluded]
    if any(protocol[k]!=original[k] for k in unchanged):raise RuntimeError('Unapproved baseline method change')
    write(output/'METHOD_INVARIANTS.json',dict(parent_protocol_sha256=sha(parent/'PROTOCOL_FROZEN.json'),
        unchanged_keys=unchanged,all_method_settings_equal=True,core_and_metrics_code_identical=True))
    write(output/'PROTOCOL_FROZEN.json',protocol);phase('CONTRACT_FROZEN_TRAINING_NOT_STARTED')
    np.savez_compressed(output/'BASIS.npz',stimulus_native=model.stimulus_basis(),history_ms=model.history_basis())
    cases={}
    def load_case(group):
        if group>=3 and not (output/'MODELS_FROZEN.json').exists():raise RuntimeError('Development access before model freeze')
        with np.load(BASE/'SOURCE_PREPARATION.npz',allow_pickle=False) as z:
            raw=z[f'g{group}_stimulus'].copy();counts=z[f'g{group}_counts'].copy()
        if raw.shape!=(71000,) or counts.shape!=(3,7100) or np.any((counts!=0)&(counts!=1)):
            raise ValueError('Original complete stimulus/binary observation layout differs')
        expected=read(BASE/'SOURCE_IDENTITY.json')['cases'][group]['stimulus_sha256']
        if hashlib.sha256(raw.tobytes()).hexdigest()!=expected:raise RuntimeError('Source waveform hash differs')
        held=matched_input(raw,group,output)
        case=dict(group=group,counts=counts,stimulus_features=model.stimulus_features(held),history_features=model.history_features(counts))
        with (output/'DATA_ACCESS.jsonl').open('a',encoding='utf8') as stream:
            stream.write(json.dumps(dict(group=group,models_frozen=(output/'MODELS_FROZEN.json').exists(),utc=datetime.now(timezone.utc).isoformat()))+'\n')
        return case
    started=time.perf_counter()
    try:
        for g in (0,1,2):cases[g]=load_case(g);phase('TRAINING_FEATURES_READY',group=g)
        models={};cv={}
        for family in ('LN','GLM'):
            phase('TRAINING',model=family)
            models[family],cv[family]=model.choose_and_fit(cases,family,guard)
            write(output/(family+'_CROSS_VALIDATION.json'),cv[family]);phase('MODEL_TRAINED',model=family,ridge=models[family]['ridge'])
        frozen=dict(protocol_sha256=sha(output/'PROTOCOL_FROZEN.json'),source_pins=sources,models=models,
            frozen_utc=datetime.now(timezone.utc).isoformat(),development_outcomes_used=False)
        write(output/'MODELS_FROZEN.json',frozen);freeze_sha=sha(output/'MODELS_FROZEN.json');phase('MODELS_FROZEN')
        original_model=read(REFERENCE/'readout/MODEL.json')['core'];original_fit=read(REFERENCE/'readout/READOUT_FIT.json')
        with np.load(BASE/'BOOTSTRAP_INDICES.npz',allow_pickle=False) as z:trials=z['spike_repeats'];blocks=z['time_blocks']
        all_rows=[]
        for g in range(6):
            guard()
            if sha(output/'MODELS_FROZEN.json')!=freeze_sha:raise RuntimeError('Frozen models changed')
            if g not in cases:cases[g]=load_case(g)
            case=cases[g];counts=case['counts'];past=observed_past(counts)
            with np.load(REFERENCE/f'readout/g{g}_PREDICTIONS.npz',allow_pickle=False) as z:
                if not np.array_equal(counts,z['observed_counts']):raise RuntimeError('Reference observations differ')
                probabilities={k:z[k+'_probability'].copy() for k in ('T','H','constant')}
                signal=original_model['b']+original_model['w0']*z['x']+original_model['w1']*z['readout_state_a']
            h=np.asarray(original_fit['history_only_theta']);p=original_fit['constant_probability']
            losses=dict(T=stable_terms(signal[None,:]+past@original_model['eta'],90.,counts)[0],
                H=stable_terms(h[0]+past@h[1:],90.,counts)[0],constant=np.where(counts,-np.log(p),-np.log1p(-p)))
            existing=read(REFERENCE/f'readout/g{g}_RESULT.json')['metrics']
            for name in probabilities:
                if abs(float(losses[name][:,model.LO:model.HI].mean())-existing[name]['conditional_NLL_nat_per1ms'])>1e-12:
                    raise RuntimeError('Reconstructed frozen reference loss differs')
            mc=None;rate_se=None;batch=None
            for family in ('LN','GLM'):
                phase('FROZEN_PREDICTION',group=g,model=family)
                probabilities[family],batches=model.predict(case,models[family],seed=2026103100+g,guard=guard)
                losses[family]=model.conditional(case,models[family])
                if batches is not None:mc,rate_se=mc_precision(counts,batches);batch=batches
            result_metrics={name:metrics(counts,pred,losses[name]) for name,pred in probabilities.items()}
            intervals=paired(counts,probabilities,losses,trials,blocks)
            write(output/f'g{g}_RESULT.json',dict(group=g,split='train' if g<3 else 'touched_internal' if g==3 else 'historical_development',
                models_frozen_sha256=freeze_sha,metrics=result_metrics,paired_T_vs_baseline=intervals,GLM_Monte_Carlo=mc,
                no_observed_spikes_used_in_free_prediction=True,conditional_history_strictly_past=True))
            np.savez_compressed(output/f'g{g}_PREDICTIONS.npz',**{k+'_probability':v for k,v in probabilities.items()},
                observed_counts=counts,time_bin_start_seconds=np.arange(7100)*.001,GLM_batch_probability=batch,GLM_20ms_rate_SE_Hz=rate_se)
            all_rows.extend(dict(group=g,model=name,**value) for name,value in result_metrics.items());phase('GROUP_COMPLETE',group=g)
        for name,expected in sources.items():
            if sha(ROOT/name)!=expected:raise RuntimeError('Frozen source/reference changed during comparison: '+name)
        write(output/'ALL_METRICS.json',all_rows)
        write(output/'COMPUTATION_COMPLETE.json',dict(status='COMPLETE_FIXED_BASELINE_COMPARISON',groups=6,final_baseline_fits=2,
            development_evaluations_per_baseline=3,retinal_model_solves=0,historical_test_accessed=False,
            wall_seconds=time.perf_counter()-started,peak_RSS_bytes=memory.peak_rss_bytes,original_T_decision_unchanged='INCONCLUSIVE'))
        journal.finish('COMPLETE')
    except BaseException as error:
        write(output/'FAILURE.json',dict(error=str(error),original_results_modified=False));journal.finish('FAILED');raise


if __name__=='__main__':
    nonce,folder=sys.argv[1:];output=Path(folder)
    with (output/'NATIVE_FAULTHANDLER.log').open('x',encoding='utf8') as stream:
        faulthandler.enable(file=stream,all_threads=True);attachment_gate(nonce,output);execute(output)
