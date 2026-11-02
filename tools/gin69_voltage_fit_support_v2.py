"""Frozen30parameter voltage GLM, real69#6 time masks, sealed segment inaccessible."""
from pathlib import Path
import json
import time
import zipfile
import numpy as np
from scipy.optimize import minimize
from retina_sim import stimulus_baselines_v1 as model
from retina_sim.spatial_group_graph import digest
from tools.durable_research_runtime_v1 import atomic_json
from tools.gin_hc2_bc1_context_support_v2 import ROOT, OUT, read

SOURCE=ROOT/'reports/RGC_DATASET_TARGET_MATRIX_V1/source_preparation'
SEED=2026110106


def counts_prefix(n):
    if n not in (30000,45000):raise ValueError('Only approved training or development prefix')
    index=read(SOURCE/'GIN_SOURCE_INDEX.json');path=SOURCE/'GIN_RAW_SPIKES.npz'
    if digest(path)!=index['arrays_sha256']:raise ValueError('Original source observations changed')
    record=next(r for r in index['records'] if r['record_id']=='lss01256_segment0')
    if record['source_cell_key']!='69#6' or record['duplicate']:raise ValueError('Wrong source cell/duplicate')
    result=[]
    with zipfile.ZipFile(path) as archive:
        for key in record['counts_keys']:
            with archive.open(key+'.npy') as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
                elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(stream)
                else:raise ValueError('Unsupported safe numeric source format')
                if shape!=(60000,) or dtype.hasobject or order:raise ValueError('Source count layout differs')
                values=np.frombuffer(stream.read(n*dtype.itemsize),dtype=dtype).copy()
                if len(values)!=n or np.any((values!=0)&(values!=1)):raise ValueError('Binary1ms source contract differs; no clipping')
                result.append(values)
    if len(result)!=6:raise ValueError('Six identical-time-mask source repeats required')
    return np.asarray(result),dict(index_sha256=digest(SOURCE/'GIN_SOURCE_INDEX.json'),arrays_sha256=index['arrays_sha256'],
        record_id=record['record_id'],count_keys=record['counts_keys'],bins_read_per_trial=n,sealed_values_decoded=False)


def source_case(seconds):
    cache=OUT/'formal_neural';last=read(cache/f'SEGMENT_{seconds*2-1:03d}.json')
    execution=Path(last['execution'])
    if read(execution/'TERMINAL.json')['status']!='EXITED_0' or read(execution/'RESOURCE.json')['status']!='PASS':
        raise ValueError('Only complete normally exited neural prefix usable')
    identity=read(cache/'IDENTITY.json')
    if identity['source_cell']!='69#6' or identity['target_group']!=7730 or identity['split_sha256']!=digest(OUT/'SPLIT_CONTRACT.json'):
        raise ValueError('Frozen target/split mismatch')
    path=cache/f'PREFIX_{seconds}s.npz'
    with np.load(path,allow_pickle=False) as data:
        voltage=data['V'].copy();times=data['time_source_s']
    if not np.array_equal(times,np.arange(seconds*10000+1)*.0001) or not np.isfinite(voltage).all():
        raise ValueError('Continuous native voltage/time grid required')
    counts,evidence=counts_prefix(seconds*1000)
    return dict(counts=counts,stimulus_features=model.stimulus_features(voltage[:-1]),
        history_features=model.history_features(counts)),dict(neural_array_sha256=digest(path),neural_identity=identity,
        observations=evidence,voltage_unit='V',source_field='V',pre0='constantV0; zero readout spike history as inherited contract')


def fit_mask(case,mask,ridge,guard):
    started=time.perf_counter();raw=model.unscaled_design(case,'GLM')[:,mask].reshape(-1,29)
    y=case['counts'][:,mask].ravel();mean=raw.mean(axis=0);scale=raw.std(axis=0)
    constant=scale<1e-12;scale[constant]=1.;x=np.column_stack((np.ones(len(y)),(raw-mean)/scale))
    p=float(y.mean())
    if not 0<p<1:raise ValueError('Degenerate training observations')
    theta=np.zeros(30);theta[0]=np.log(np.expm1(-np.log1p(-p)/.09))
    def objective(t):guard();return model.objective(t,x,y,ridge)
    opt=minimize(objective,theta,jac=True,method='L-BFGS-B',options=model.OPTIONS)
    return dict(family='GLM',theta=opt.x.tolist(),mean=mean.tolist(),scale=scale.tolist(),
        constant_columns=np.flatnonzero(constant).tolist(),ridge=ridge,parameters=30,converged=bool(opt.success),
        message=str(opt.message),iterations=int(opt.nit),function_evaluations=int(opt.nfev),
        raw_NLL=float(model.objective(opt.x,x,y,0)[0]),penalized_objective=float(opt.fun),
        wall_seconds=time.perf_counter()-started,scored_bins=int(mask.sum()),starts=1)


def metrics(counts,p,loss,lo,hi):
    observed=counts[:,lo:hi].mean(axis=0).reshape(-1,20).mean(axis=1)*1000
    predicted=p[lo:hi].reshape(-1,20).mean(axis=1)*1000
    r=float(np.corrcoef(observed,predicted)[0,1]) if min(np.std(observed),np.std(predicted))>1e-12 else None
    return dict(free_r=r,free_MSE_Hz2=float(np.mean((predicted-observed)**2)),
        free_mean_Hz=float(predicted.mean()),observed_mean_Hz=float(observed.mean()),
        free_modulation_SD_Hz=float(predicted.std()),conditional_NLL_nat_per1ms=float(loss[:,lo:hi].mean()))


def execute(output,guard,mode):
    started=time.perf_counter();folder=OUT/'readout';folder.mkdir(exist_ok=True)
    protocol_path=folder/'PROTOCOL_FROZEN.json'
    protocol=dict(schema='gin69_voltage30_timeblock_v1',split_sha256=digest(OUT/'SPLIT_CONTRACT.json'),
        source_cell='69#6',target_group=7730,source_input='newHCv2BCv1 actual membraneV100us',
        parameters=30,voltage_bases=16,voltage_support_ms=[0,499.9],history_bases=13,history_lags_ms=[1,200],
        link='lambda90softplus; Bernoulli p=1-exp(-.001lambda)',fit_interval=[0,30],development_interval=[30,45],
        sealed_interval=[45,60],sealed_access=False,train_blocks=[[0,10],[10,20],[20,30]],
        CV='leave-one10s block out, all repeats share masks; purge200ms after held block from fitting to exclude its observed-history features',
        standardization='each fit uses only its training rows; final uses0..30s',
        initial_history='zero spikes before0, inherited readout boundary; pre-live source events remain in raw archive, not injected into free prediction',
        voltage_prehistory='constant equilibrium V0; full continuous feature filtering from0, no split resets',
        optimizer=model.OPTIONS,ridge_candidates=list(model.GRID),selection='minimum mean held-block conditionalNLL; strongest within1e-8; no extra starts',
        free_prediction=dict(paths=4096,batches=16,seed=SEED,observed_history=False),
        PSTH_bin_ms=20,reference='training-only constant Bernoulli probability',
        stimulus_baseline='Separate matched_spatial_baselines: all493 actual native inputs;16 temporal bases; LN/nohistory and GLM/13history bases; capacities explicitly differ',
        code_sha256=digest(Path(__file__)),base_code_sha256=digest(ROOT/'retina_sim/stimulus_baselines_v1.py'),
        FIT_READY=False,background_qualified=False,production_adopted=False)
    if protocol_path.exists():
        if read(protocol_path)!=protocol:raise ValueError('Readout contract changed')
    else:atomic_json(protocol_path,protocol)
    if mode=='train':
        if (folder/'MODEL_FROZEN.json').exists():raise FileExistsError('Existing fitted model retained')
        case,evidence=source_case(30);selection=[];grid=np.arange(30000)
        for ridge in model.GRID:
            folds=[]
            for hold in range(3):
                lo,hi=hold*10000,(hold+1)*10000;mask=(grid<lo)|(grid>=hi)
                mask[(grid>=hi)&(grid<min(30000,hi+200))]=False
                fitted=fit_mask(case,mask,ridge,guard)
                loss=model.conditional(case,fitted)[:,lo:hi]
                folds.append(dict(block=hold,model=fitted,NLL=float(loss.mean()) if fitted['converged'] else None))
                atomic_json(folder/'TRAINING_PROGRESS.json',dict(ridge=ridge,held_block=hold,wall_seconds=time.perf_counter()-started))
            scores=[f['NLL'] for f in folds]
            selection.append(dict(ridge=ridge,folds=folds,score=None if any(s is None for s in scores) else float(np.mean(scores))))
        atomic_json(folder/'CROSS_VALIDATION.json',selection)
        valid=[r for r in selection if r['score'] is not None]
        if not valid:raise ValueError('Original finite ridge candidates did not converge; no scope expansion')
        best=min(r['score'] for r in valid);ridge=max(r['ridge'] for r in valid if r['score']<=best+1e-8)
        fitted=fit_mask(case,np.ones(30000,dtype=bool),ridge,guard)
        if not fitted['converged']:raise ValueError('Original final optimizer failed; no silent retry')
        atomic_json(folder/'MODEL_FROZEN.json',dict(model=fitted,constant_probability=float(case['counts'].mean()),
            protocol_sha256=digest(protocol_path),input_evidence=evidence,development_outcomes_used=False,
            training_wall_seconds=time.perf_counter()-started))
        atomic_json(output/'RESULT.json',dict(status='INDEPENDENT_69_6_VOLTAGE_MODEL_FROZEN',parameters=30,
            selected_ridge=ridge,wall_seconds=time.perf_counter()-started))
    elif mode=='evaluate':
        frozen_path=folder/'MODEL_FROZEN.json';freeze=digest(frozen_path);frozen=read(frozen_path)
        if frozen['protocol_sha256']!=digest(protocol_path):raise ValueError('Frozen model protocol mismatch')
        if (folder/'FIXED_DEVELOPMENT_RESULT.json').exists():raise FileExistsError('Fixed evaluation already completed')
        case,evidence=source_case(45)
        probability,batches=model.predict(case,frozen['model'],seed=SEED,guard=guard)
        loss=model.conditional(case,frozen['model']);q=frozen['constant_probability']
        constant=np.full(45000,q);closs=np.where(case['counts'],-np.log(q),-np.log1p(-q))
        curves=batches[:,30000:45000].reshape(16,-1,20).mean(axis=2)*1000
        mc_se=curves.std(axis=0,ddof=1)/4
        result=dict(status='FIXED_DEVELOPMENT_COMPLETE',interval_seconds=[30,45],repeats=6,
            metrics=dict(V_GLM=metrics(case['counts'],probability,loss,30000,45000),
                constant=metrics(case['counts'],constant,closs,30000,45000)),
            MC=dict(paths=4096,batches=16,seed=SEED,max20ms_rate_SE_Hz=float(mc_se.max()),mean20ms_rate_SE_Hz=float(mc_se.mean())),
            frozen_model_sha256=freeze,input_evidence=evidence,wall_seconds=time.perf_counter()-started,
            stimulus_baseline='Separate fixed matched_spatial_baselines evaluation follows',independent_stimulus_generalization_claim=False,
            full_physiology_validated=False,sealed_values_used=False,FIT_READY=False,background_qualified=False,production_adopted=False)
        if digest(frozen_path)!=freeze:raise ValueError('Model changed during fixed evaluation')
        np.savez_compressed(folder/'FIXED_DEVELOPMENT_PREDICTIONS.npz',time_bin_start_seconds=np.arange(45000)*.001,
            probability=probability,MC_batch_probability=batches,observed_counts=case['counts'],conditional_loss=loss,
            constant_probability=constant,constant_conditional_loss=closs)
        atomic_json(folder/'FIXED_DEVELOPMENT_RESULT.json',result);atomic_json(output/'RESULT.json',result)
    else:raise ValueError('Unknown fixed readout phase')
