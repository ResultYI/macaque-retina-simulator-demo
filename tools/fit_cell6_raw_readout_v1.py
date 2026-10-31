"""One new conditional temporal fit; fixed development evaluation, no retina solve."""
from pathlib import Path
from datetime import datetime,timezone
from types import FunctionType
import csv
import hashlib
import json
import sys
import time
import numpy as np
from scipy.optimize import minimize

ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools import run_temporal_readout_v1 as original
from tools.cell6_halo_context_v1 import BASE,MODEL,TARGET
from retina_sim.temporal_readout_v1 import TemporalReadout,free_probability,stable_terms,digest
from retina_sim.rgc_stage1_readout import observed_past
from retina_sim.group_time_storage import load_bundle

EXPERIMENT=BASE;HERE=EXPERIMENT/'readout';LO,HI=1000,7000
GUARD=lambda:None


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);temp=p.with_name(p.name+'.writing')
    temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x),encoding='utf8');temp.replace(p)


def load_case(group):
    if group>=3 and not (HERE/'MODEL.json').exists():raise RuntimeError('Development outcomes only after model freeze')
    folder=EXPERIMENT/'cache'/('g'+str(group));m=read(folder/'manifest.json');case=read(BASE/'SOURCE_IDENTITY.json')['cases'][group]
    if m['status']!='COMPLETE_NEW_TOPOLOGY_CELL6_CACHE' or m['target']!=TARGET or m['stimulus_sha256']!=case['stimulus_sha256'] or sha(folder/'endogenous.npz')!=m['arrays_sha256']:
        raise RuntimeError('Complete new exact-target neural cache required')
    identity=read(folder/'INPUT_AND_MODEL_IDENTITY.json')
    if (m['group']!=group or m['samples']!=71001 or m['bins']!=7100 or m['duration_seconds']!=7.1
        or m['members']!=2 or not m['physical_gate_passed'] or not m['complete_research_restore']
        or m['old_neural_cache_reused'] or identity['identity']!=m['cache_identity']
        or identity['target']!=TARGET or identity['members']!=2):
        raise RuntimeError('Full new-topology cache identity/physical/restore gate failed')
    if group==0 and not read(folder/'SHORT_NUMERICAL_CHECK.json')['passed']:
        raise RuntimeError('Original g0 short numerical reference failed')
    with np.load(folder/'endogenous.npz',allow_pickle=False) as z:
        if not np.array_equal(z['time_source_s'],np.arange(71001)*.0001):raise RuntimeError('Original100us source time mismatch')
        for name in ('V','gE','gI','expected_rate_base_Hz'):
            if z[name].shape!=((7100,) if name=='expected_rate_base_Hz' else (71001,)) or not np.isfinite(z[name]).all():
                raise RuntimeError('Full finite response cache required: '+name)
        v=z['V'][:-1].copy();prod=z['expected_rate_base_Hz'].copy()*.001
    with np.load(BASE/'SOURCE_PREPARATION.npz',allow_pickle=False) as z:y=z['g'+str(group)+'_counts'].copy()
    with (HERE/'DATA_ACCESS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(group=group,model_frozen=(HERE/'MODEL.json').exists(),cache_identity=m['cache_identity'],utc=datetime.now(timezone.utc).isoformat()))+'\n')
    return dict(group=group,voltage=v,bin_voltage=v[::10],counts=y,past=observed_past(y),production_probability=prod,cache_identity=m['cache_identity'])


def static_objective(theta,cases,stats,static):
    GUARD()
    total=0.;grad=np.zeros(len(theta));n=0
    for c in cases:
        x=(c['bin_voltage'][LO:HI]-stats['mean_V'])/stats['SD_V'];past=c['past'][:,LO:HI]
        z=theta[0]+(theta[1]*x if static else 0)+past@theta[2 if static else 1:]
        loss,g=stable_terms(z,90.,c['counts'][:,LO:HI]);total+=loss.sum();n+=loss.size
        grad[0]+=g.sum()
        if static:grad[1]+=g.sum(axis=0)@x
        grad[2 if static else 1:]+=np.einsum('rt,rtk->k',g,past)
    return float(total/n+.5e-4*(theta[1:]@theta[1:])),grad/n+np.r_[0.,1e-4*theta[1:]]


def core_for(theta,stats,pi):
    return dict(schema='temporal_readout_v1',scope='new raw Cell6 exact8166 halo conditional output; not physical Cell6 registration',
        target=dict(global_id=TARGET,population='parasol_RGC_ON',members=2,halo_owner=True,physical_identity_verified=False),
        mean_train_V=stats['mean_V'],SD_train_V=stats['SD_V'],initial_a=stats['initial_a'],initial_gray_V=stats['initial_gray_V'],
        initial_history_probability=pi,b=float(theta[0]),w0=float(theta[1]+theta[2]),w1=float(-theta[2]),eta=theta[3:6].tolist(),
        tau_s=float(.010*np.exp(theta[6])),alpha_Hz=90.,native_dt_s=.0001,bin_dt_s=.001,train_groups=[0,1,2],
        contract_sha256=sha(BASE/'READOUT_PROTOCOL_FROZEN.json'),inference_code_sha256=sha(ROOT/'retina_sim/temporal_readout_v1.py'),FIT_READY=False)


def necessary(cases,stats,theta,pi):
    globals_=dict(original.necessary_checks.__globals__,HERE=HERE,core_for=lambda t,s:core_for(t,s,pi),write=write)
    fn=FunctionType(original.necessary_checks.__code__,globals_,original.necessary_checks.__name__,original.necessary_checks.__defaults__)
    fn(cases,stats,theta)


def train():
    if (HERE/'READOUT_FIT.json').exists() or (HERE/'MODEL.json').exists():raise FileExistsError('One temporal training only')
    cases=[load_case(g) for g in (0,1,2)];pool=np.concatenate([c['bin_voltage'][LO:HI] for c in cases])
    stats=dict(mean_V=float(pool.mean()),SD_V=float(pool.std()),initial_gray_V=float(cases[0]['voltage'][0]))
    if stats['SD_V']<=0 or any(c['voltage'][0]!=stats['initial_gray_V'] for c in cases):raise RuntimeError('Original shared gray and nonzero train SD required')
    stats['initial_a']=(stats['initial_gray_V']-stats['mean_V'])/stats['SD_V']
    pbar=float(np.concatenate([c['counts'][:,LO:HI].ravel() for c in cases]).mean())
    if not 0<pbar<1:raise RuntimeError('Training constant MLE is degenerate')
    b0=float(np.log(np.expm1(-np.log1p(-pbar)/.09)))
    if not -10<=b0<=10:raise RuntimeError('Declared deterministic start outside original bounds')
    seed=load_bundle(MODEL/'seed',expected_kind='source_construction_seed')['payload'];target=read(MODEL/'RESULT.json')['target_local_row']
    pi=[float(seed['states']['parasol_RGC_ON','RGC_expected_history','history_probability_'+str(i)][target]) for i in range(8)]
    options=dict(maxiter=200,maxfun=1000,maxls=30,ftol=1e-10,gtol=1e-7)
    write(HERE/'TRAINING_START_FROZEN.json',dict(statistics=stats,constant_probability=pbar,history_only_start=[b0,0,0,0],
        S_start_rule='H-trained b/eta, gamma1; one5-coordinate static fit, used only as temporal initialization',
        optimizer=options,initial_history_probability=pi,source_groups=[0,1,2],development_outcomes_accessed_for_training=False,
        new_indices_and_start=True,old_statistics_tau_or_start_reused=False))
    write(HERE/'TRAINING_STARTED.json',dict(utc=datetime.now(timezone.utc).isoformat(),train_groups=[0,1,2],stage='history_reference_then_static_initialization_then_one_temporal_fit'))
    h=minimize(lambda t:static_objective(t,cases,stats,False),np.array([b0,0.,0.,0.]),jac=True,method='L-BFGS-B',bounds=[(-10,10)]+[(-12,0)]*3,options=options)
    write(HERE/'HISTORY_ONLY_FIT.json',dict(converged=bool(h.success),theta=h.x.tolist(),message=str(h.message),iterations=int(h.nit)))
    if not h.success:raise RuntimeError('Declared history-only reference failed; no restart')
    s=minimize(lambda t:static_objective(t,cases,stats,True),np.r_[h.x[0],1.,h.x[1:]],jac=True,method='L-BFGS-B',bounds=[(-10,10),(0,10)]+[(-12,0)]*3,options=options)
    write(HERE/'STATIC_INITIALIZATION_FIT.json',dict(converged=bool(s.success),theta=s.x.tolist(),message=str(s.message),iterations=int(s.nit),role='training-only deterministic start, no development model selection'))
    if not s.success:raise RuntimeError('Declared static initialization fit failed; no restart')
    theta=np.r_[s.x[0],s.x[1],1.,s.x[2:],0.];necessary(cases,stats,theta,pi)
    started=time.perf_counter();trace=[]
    def callback(t):trace.append(dict(iteration=len(trace)+1,theta=t.tolist(),objective=float(original.objective(t,cases,stats)[0])))
    def objective(t):GUARD();return original.objective(t,cases,stats)
    write(HERE/'TEMPORAL_OPTIMIZER_STARTED.json',dict(utc=datetime.now(timezone.utc).isoformat(),train_groups=[0,1,2],temporal_fit_count=1))
    opt=minimize(objective,theta,jac=True,method='L-BFGS-B',bounds=original.BOUNDS,callback=callback,options=options)
    fit=dict(converged=bool(opt.success),message=str(opt.message),theta=opt.x.tolist(),initial_theta=theta.tolist(),statistics=stats,
        iterations=int(opt.nit),function_evaluations=int(opt.nfev),penalized_objective=float(opt.fun),raw_NLL_nat_per1ms=float(original.objective(opt.x,cases,stats,False)[0]),
        wall_seconds=time.perf_counter()-started,trace=trace,constant_probability=pbar,history_only_theta=h.x.tolist(),
        active_bounds=[i for i,(v,(lo,hi)) in enumerate(zip(opt.x,original.BOUNDS)) if min(abs(v-lo),abs(v-hi))<1e-7],
        temporal_fit_count=1,training_groups=[0,1,2],new_experiment=True,FIT_READY=False,background_qualified=False,production_adopted=False)
    write(HERE/'READOUT_FIT.json',fit)
    if not opt.success:raise RuntimeError('One temporal fit did not converge; no restart or expanded bounds')
    fisher=[]
    for c in cases:
        x,a,da=original.features(c,opt.x,stats);past=c['past'][:,LO:HI]
        jac=np.concatenate([np.ones((*past.shape[:2],1)),
            np.broadcast_to(np.column_stack([x[LO:HI],(x-a)[LO:HI]])[None,:,:],(*past.shape[:2],2)),
            past,np.broadcast_to((-opt.x[2]*da[LO:HI])[None,:,None],(*past.shape[:2],1))],axis=2)
        z=opt.x[0]+opt.x[1]*x[None,LO:HI]+opt.x[2]*(x-a)[None,LO:HI]+past@opt.x[3:6]
        exposure=.09*np.logaddexp(0.,z);q=-np.expm1(-exposure)
        weight=np.exp(.5*(2*np.log(.09)-2*np.logaddexp(0.,-z)-exposure-np.log(q)))
        fisher.append((jac*weight[:,:,None]).reshape(-1,7))
    singular=np.linalg.svd(np.vstack(fisher)/np.sqrt(54000),compute_uv=False)
    write(HERE/'IDENTIFIABILITY.json',dict(singular_values=singular.tolist(),relative_minimum=float(singular[-1]/singular[0]),
        threshold=.001,local_rank_passed=bool(singular[-1]/singular[0]>=.001),scope='original conditional training Fisher; no biological identification'))
    core=core_for(opt.x,stats,pi);model=dict(core=core,model_id=digest(core),qualification='FROZEN_NOT_EVALUATED',frozen_utc=datetime.now(timezone.utc).isoformat())
    write(HERE/'MODEL.json',model)
    write(HERE/'FROZEN_IDENTITY.json',dict(model_id=model['model_id'],model_sha256=sha(HERE/'MODEL.json'),
        fit_sha256=sha(HERE/'READOUT_FIT.json'),training_start_sha256=sha(HERE/'TRAINING_START_FROZEN.json'),
        protocol_sha256=sha(BASE/'READOUT_PROTOCOL_FROZEN.json'),inference_sha256=sha(ROOT/'retina_sim/temporal_readout_v1.py'),
        fit_implementation_sha256=sha(Path(__file__)),original_temporal_method_sha256=sha(ROOT/'tools/run_temporal_readout_v1.py'),
        source_preparation_sha256=sha(BASE/'SOURCE_PREPARATION.npz'),bootstrap_indices_sha256=sha(BASE/'BOOTSTRAP_INDICES.npz'),
        cache_manifests={str(g):sha(EXPERIMENT/'cache'/('g'+str(g))/'manifest.json') for g in range(3)},
        cache_manifest_scope='training groups only; development cache identity recorded at fixed evaluation'))
    return model,fit


def intervals(case,prob,loss):
    with np.load(BASE/'BOOTSTRAP_INDICES.npz',allow_pickle=False) as z:trials=z['spike_repeats'];blocks=z['time_blocks']
    observed=case['counts'][:,LO:HI].reshape(3,30,10,20).mean(axis=3)*1000
    rates={k:v[LO:HI].reshape(30,10,20).mean(axis=2)*1000 for k,v in prob.items()}
    losses={k:v[:,LO:HI].reshape(3,30,200).mean(axis=2) for k,v in loss.items()};values={k:dict(nll=[],mse=[],r=[]) for k in ('H','constant')}
    for index,(tr,bl) in enumerate(zip(trials,blocks)):
        if index%100==0:GUARD()
        y=observed[tr][:,bl].mean(axis=0).ravel();t=rates['T'][bl].ravel()
        correlation=np.corrcoef(y,t)[0,1] if y.std()>1e-12 and t.std()>1e-12 else np.nan
        for k,out in values.items():
            q=rates[k][bl].ravel();out['nll'].append(float((losses[k]-losses['T'])[tr][:,bl].mean()))
            out['mse'].append(float(((q-y)**2).mean()-((t-y)**2).mean()));out['r'].append(correlation)
    def quantile(v):
        a=np.asarray(v);a=a[np.isfinite(a)];return None if not len(a) else np.quantile(a,[.025,.5,.975]).tolist()
    return {k:{name:quantile(v) for name,v in out.items()} for k,out in values.items()}


def evaluate(group,model,fit):
    GUARD()
    dest=HERE/('g'+str(group)+'_RESULT.json')
    if dest.exists():raise FileExistsError('One fixed evaluation per group')
    frozen=read(HERE/'FROZEN_IDENTITY.json')
    if frozen['model_sha256']!=sha(HERE/'MODEL.json') or frozen['fit_sha256']!=sha(HERE/'READOUT_FIT.json'):raise RuntimeError('Frozen fit/standardization identity changed')
    if frozen['fit_implementation_sha256']!=sha(Path(__file__)) or frozen['inference_sha256']!=sha(ROOT/'retina_sim/temporal_readout_v1.py') or frozen['bootstrap_indices_sha256']!=sha(BASE/'BOOTSTRAP_INDICES.npz'):
        raise RuntimeError('Frozen evaluation code/uncertainty identity changed')
    if any(sha(EXPERIMENT/'cache'/('g'+g)/'manifest.json')!=expected for g,expected in frozen['cache_manifests'].items()):
        raise RuntimeError('Frozen training cache identity changed')
    c=load_case(group);theta=np.asarray(fit['theta']);x,a,_=original.features(c,theta,fit['statistics']);core=model['core']
    signal=core['b']+core['w0']*x+core['w1']*a;h=np.asarray(fit['history_only_theta'])
    prob=dict(T=TemporalReadout(model).predict(c['voltage'])['probability'],
        H=free_probability(np.full(7100,h[0]),h[1:],core['initial_history_probability'])[0],constant=np.full(7100,fit['constant_probability']))
    loss=dict(T=stable_terms(signal[None,:]+c['past']@core['eta'],90.,c['counts'])[0],H=stable_terms(h[0]+c['past']@h[1:],90.,c['counts'])[0])
    p=fit['constant_probability'];loss['constant']=np.where(c['counts'],-np.log(p),-np.log1p(-p))
    metrics={k:original.metrics(c,v,loss[k]) for k,v in prob.items()};ci=intervals(c,prob,loss)
    row=dict(group=group,split='train' if group<3 else 'touched_internal' if group==3 else 'historical_development',
        conditional_NLL_nat_per1ms=metrics['T']['conditional_NLL_nat_per1ms'],delta_NLL_vs_H=metrics['H']['conditional_NLL_nat_per1ms']-metrics['T']['conditional_NLL_nat_per1ms'],
        delta_NLL_vs_constant=metrics['constant']['conditional_NLL_nat_per1ms']-metrics['T']['conditional_NLL_nat_per1ms'],
        free_r=metrics['T']['free_r'],free_MSE_Hz2=metrics['T']['free_MSE_Hz2'],delta_MSE_vs_H=metrics['H']['free_MSE_Hz2']-metrics['T']['free_MSE_Hz2'],
        mean_Hz=metrics['T']['mean_Hz'],SD_Hz=metrics['T']['SD_Hz'],observed_mean_Hz=metrics['T']['observed_mean_Hz'],observed_SD_Hz=metrics['T']['observed_SD_Hz'],
        lower95_delta_NLL_vs_H=ci['H']['nll'][0],lower95_delta_MSE_vs_H=ci['H']['mse'][0],lower95_free_r=None if ci['H']['r'] is None else ci['H']['r'][0])
    if any(not np.isfinite(v).all() or np.any((v<0)|(v>1)) for v in prob.values()):raise RuntimeError('Prediction probability domain failed')
    np.savez_compressed(HERE/('g'+str(group)+'_PREDICTIONS.npz'),**{k+'_probability':v for k,v in prob.items()},observed_counts=c['counts'],time_bin_start_seconds=np.arange(7100)*.001,readout_state_a=a,x=x)
    write(dest,dict(model_id=model['model_id'],metrics=metrics,paired=ci,row=row,free_uses_no_observed_history=True,conditional_uses_strict_past_only=True,cache_identity=c['cache_identity'],cache_manifest_sha256=sha(EXPERIMENT/'cache'/('g'+str(group))/'manifest.json')))
    print(json.dumps(dict(stage='FIXED_EVALUATION_COMPLETE',**row)),flush=True)
    return row


def wait_for_development_cache(group):
    from tools.durable_research_runtime_v1 import inspect_identity
    terminal=EXPERIMENT/'capture'/('g'+str(group))/'TERMINAL.json'
    while not terminal.exists():
        GUARD()
        owner=read(EXPERIMENT/'TRAIN_FIRST_OWNER.json')
        if inspect_identity(dict(pid=owner['pid'],creation_100ns=owner['creation_100ns'],image=owner['image']))['state']!='ALIVE':
            raise RuntimeError('Owned continuation unavailable; preserve frozen fit without retry')
        if (EXPERIMENT/'CONTINUATION_FAILURE.json').exists():raise RuntimeError('Cache continuation failed; frozen readout retained')
        write(HERE/'WAITING_FOR_CACHE.json',dict(group=group,model_frozen=True,utc=datetime.now(timezone.utc).isoformat()))
        time.sleep(5)
    if read(terminal)['status']!='EXITED_0' or read(terminal.parent/'RESOURCE.json')['status']!='PASS':
        raise RuntimeError('Development cache worker did not complete safely')


def main(guard=lambda:None,*,output_root=None):
    global GUARD,HERE,EXPERIMENT
    EXPERIMENT=BASE if output_root is None else Path(output_root);HERE=EXPERIMENT/'readout'
    GUARD=guard;GUARD()
    HERE.mkdir(exist_ok=False)
    try:
        model,fit=train();rows=[]
        for g in range(6):
            if g>=3 and (EXPERIMENT/'TRAIN_FIRST_OWNER.json').exists():wait_for_development_cache(g)
            rows.append(evaluate(g,model,fit))
        with (EXPERIMENT/'FIXED_EVALUATION.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        internal=rows[3]['delta_NLL_vs_H']>0 and rows[3]['free_r'] is not None and rows[3]['free_r']>0 and rows[3]['delta_MSE_vs_H']>0
        dev={str(g):all(rows[g][name] is not None and rows[g][name]>0 for name in ['lower95_delta_NLL_vs_H','lower95_delta_MSE_vs_H','lower95_free_r']) for g in (4,5)}
        decision='RETAIN_CONDITIONAL' if internal and all(dev.values()) else 'INCONCLUSIVE'
        write(EXPERIMENT/'READOUT_FIT.json',dict(status='COMPLETE_NEW_EXPERIMENT',model_id=model['model_id'],fit=fit,development_decision=decision,internal_g3=internal,development_gates=dev,
            frozen_model_file='readout/MODEL.json',qualification_separate_from_frozen_parameters=True,FIT_READY=False,background_qualified=False,production_adopted=False))
        write(HERE/'COMPLETE.json',dict(status=decision,groups_evaluated=6,historical_test_accessed=False,temporal_fit_count=1,stopped_after_fixed_evaluation=True))
    except BaseException as error:
        write(HERE/'FAILURE.json',dict(error=str(error),failed_without_restart=True));raise


if __name__=='__main__':main()
