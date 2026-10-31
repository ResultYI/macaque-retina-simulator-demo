"""Freeze a NEW six-group experiment from original raw source, never old hashes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import sys
import numpy as np
from scipy.io import loadmat

ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.cell6_raw_intake_v1 import selected_spikes
BASE=ROOT/'reports/CELL6_RAW_REBUILD_CONDITIONAL_READOUT_V1'


def write(name,value):(BASE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    BASE.mkdir(exist_ok=True)
    if (BASE/'SOURCE_IDENTITY.json').exists() or (BASE/'SOURCE_PREPARATION.npz').exists():raise FileExistsError('Existing prepared experiment cannot be overwritten')
    raw=ROOT/'data/source_data/latimer2019_figshare_v1/Cell_6.mat'
    if sha(raw)!='59e1457bc27f7392a3c458902941a234c687b26a525b26d168bc8609aae40025':raise ValueError('Raw source SHA differs')
    prior=ROOT/'reports/modeling_scope_20261020/latimer_cell6_rgc_v1/contract.json'
    metadata=json.loads(prior.read_text(encoding='utf8'))
    hashes=metadata['partition']['training_stimulus_sha256'];assert len(hashes)==6
    stimuli=loadmat(raw,variable_names=['SpikeStm'])['SpikeStm']
    if stimuli.shape!=(30,71000) or stimuli.dtype!=np.float64:raise ValueError('Original stimulus layout differs')
    rows={h:[i for i,s in enumerate(stimuli) if hashlib.sha256(s.tobytes()).hexdigest()==h] for h in hashes}
    pairing={p['stimulus_sha256']:p['spike_trial_indices_zero_based'] for p in metadata['pairing']}
    if any(len(rows[h])!=3 or rows[h]!=pairing[h] for h in hashes):raise ValueError('Historical stimulus/repeat identity differs')
    allowed={i for ids in rows.values() for i in ids};spikes=selected_spikes(raw,allowed)
    cases=[];arrays={}
    for group,h in enumerate(hashes):
        v=stimuli[rows[h][0]].copy();counts=np.stack([np.bincount((spikes[i]-1)//10,minlength=7100) for i in rows[h]])
        if counts.shape!=(3,7100) or not np.isin(counts,[0,1]).all():raise ValueError('Original1ms binary repeat layout differs')
        if not np.all(v[:10000]==v[0]):raise ValueError('Actual first1s source prefix differs')
        factors=1+.05*(v-v[0])
        if not np.isfinite(factors).all() or np.any(factors<0):raise ValueError('Original affine native-light mapping leaves domain')
        arrays['g'+str(group)+'_stimulus']=v;arrays['g'+str(group)+'_counts']=counts
        cases.append(dict(group=group,case_id='cell6_g'+str(group),stimulus_sha256=h,spike_trial_indices_zero_based=rows[h],
            split='train' if group<3 else 'previously_touched_internal_check' if group==3 else 'historical_development_evaluation',
            actual_source_samples=71000,source_dt_seconds=.0001,bins=7100,spike_index_origin=1,
            affine_factor_min=float(factors.min()),affine_factor_max=float(factors.max()),first1s_constant=True))
    np.savez_compressed(BASE/'SOURCE_PREPARATION.npz',**arrays)
    rng=np.random.Generator(np.random.PCG64(2026103006))
    trials=rng.integers(0,3,size=(1000,3));blocks=rng.integers(0,30,size=(1000,30))
    np.savez_compressed(BASE/'BOOTSTRAP_INDICES.npz',spike_repeats=trials,time_blocks=blocks)
    source=dict(schema='cell6-raw-rebuild-preparation-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        new_experiment_not_recovered_old_derivatives=True,raw_source=dict(path=str(raw.relative_to(ROOT)),bytes=raw.stat().st_size,sha256=sha(raw)),
        group_mapping_source=dict(path=str(prior.relative_to(ROOT)),sha256=sha(prior),rule='Exact six historical development stimulus hashes, existing ordered source metadata; no response-based new grouping'),
        cases=cases,selected_spike_cells_decoded=sorted(allowed),other_spike_cells_decoded=False,
        stimulus_values_loaded=True,ExcCurrents_and_InhCurrents_decoded=False,historical_test_spikes_binned_or_scored=False,
        input_mapping=dict(formula='Btd*(1+0.05*(raw[j]-raw[0]))',unit='td',physical_calibration=False,
            source='tools/run_bc_ac_fullchain_full_trial.py::prepare',source_sha256=sha(ROOT/'tools/run_bc_ac_fullchain_full_trial.py'),
            native_input_fps=150.,sample_index='max(0,floor((frame/150-model_origin_s)/0.0001))',model_origin_s=0.,
            source_interpretation='inherited approved engineering equivalent input, not measured radiance/LMS calibration',rod='unchanged engineering background'),
        score_window_seconds=[1.,7.],psth='20ms nonoverlapping rectangles; no smoothing, lag or shift',
        resampling=dict(new_indices=True,old_indices_not_recreated=True,seed=2026103006,algorithm='NumPy PCG64',draws=1000,
            rule='paired3 spike repeats with replacement,30 aligned200ms time blocks with replacement',interval='2.5/50/97.5 percentiles, descriptive conditional only'),
        preparation_npz_sha256=sha(BASE/'SOURCE_PREPARATION.npz'),bootstrap_npz_sha256=sha(BASE/'BOOTSTRAP_INDICES.npz'),
        code_pins={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'tools/cell6_raw_intake_v1.py',ROOT/'tools/prepare_cell6_raw_experiment_v1.py',ROOT/'retina_sim/temporal_readout_v1.py']},
        FIT_READY=False,background_qualified=False,production_adopted=False)
    write('SOURCE_IDENTITY.json',source)
    write('READOUT_PROTOCOL_FROZEN.json',dict(schema='new-cell6-conditional-temporal-protocol-v1',
        formula='x=(V-m)/s; tau*da/dt=x-a; z=b+g_DC*x+k*(x-a)+eta@past; lambda=90*softplus(z);p=-expm1(-.001*lambda)',
        coordinates=['b','g_DC','k','eta1','eta2','eta3','log(tau/.010)'],
        bounds=[[-10,10],[0,10],[0,10],[-12,0],[-12,0],[-12,0],[float(np.log(.2)),float(np.log(5.))]],
        regularization='0.5e-4*((g_DC+k)^2+k^2+sum(eta^2)); b/logtau unpenalized',
        optimizer=dict(method='L-BFGS-B',maxiter=200,maxfun=1000,maxls=30,ftol=1e-10,gtol=1e-7,starts=1),
        start='deterministic training-only static-independent S fit using existing b/gamma/eta formula and bounds; k=1,tau10ms. New S start, not old missing coefficients.',
        history_only='existing b+eta@strictly_past3bin; training g0–g2 only; alpha90Hz',constant='training-only mean binary probability',
        statistics='pooled g0–g2 V at1ms bin starts in[1,7)s only; freeze m/s and same gray initial_a for all groups',
        free='exact8-state law with model stationary/declared initial law, no observed evaluation history',
        train_groups=[0,1,2],fixed_development=[3,4,5],historical_test_reopened=False,
        prediction_gates='existing temporal protocol: g3 point deltaNLLvsH>0,r>0,deltaMSEvsH>0; g4/g5 lower95 deltaNLLvsH>0,deltaMSEvsH>0,r>0; failures retained, no changes',
        prior_method='tools/run_temporal_readout_v1.py::prepare/train/objective/metrics/paired',
        qualification='explicit independent experimental fit allowed; FIT_READY=false unchanged',FIT_READY=False,background_qualified=False,production_adopted=False))
    print(json.dumps(dict(status='NEW_RAW_PREPARATION_FROZEN',cases=len(cases),selected_spike_cells=len(allowed),old_test_used=False,output=str(BASE))))


if __name__=='__main__':main()
