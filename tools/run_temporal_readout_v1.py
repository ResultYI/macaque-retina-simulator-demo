"""One declared low-cost conditional temporal readout trial; no retina forward."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import minimize
from retina_sim.contracts import canonical_hash
from retina_sim.rgc_stage1_readout import observed_past
from retina_sim.temporal_readout_v1 import (
    ALPHA, BIN_DT, NATIVE_DT, TemporalReadout, causal_state, digest,
    free_probability, load_model, stable_terms)

HERE = ROOT / 'reports/cell6_temporal_readout_v1_20261028'
HS = ROOT / 'reports/cell6_static_readout_independent_20261028_v1'
CD = ROOT / 'reports/cell6_rgc_cd_reference_control_20261028_v1'
AB = ROOT / 'reports/cell6_reference_encoding_control_20261028_v1'
SPEC = ROOT / 'reports/rgc_stage1_complete_v1/CONTRACT.json'
OBS = ROOT / 'reports/rgc_stage1_complete_v1/actual_observations.npz'
REGISTRY = ROOT / 'reports/rgc_stage1_cpu_parallel_full_v1/CACHE_REGISTRY.json'
CONFIG = ROOT / 'configurations/calibration/temporal_readout_v1_cell6_conditional.json'
LO, HI = 1000, 7000
RIDGE = 1e-4
BOUNDS = [(-10., 10.), (0., 10.), (0., 10.)] + [(-12., 0.)] * 3 + [(np.log(.2), np.log(5.))]


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.writing')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                           default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n', encoding='utf8')
    t.replace(p)


def archive(path, arrays):
    p = Path(path)
    if p.exists():
        raise RuntimeError('Refuse to overwrite evaluation archive: ' + str(p))
    t = p.with_name(p.name + '.writing')
    with t.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    t.replace(p)


def guard():
    if (ROOT / 'REQUEST_STOP').exists() or (HERE / 'REQUEST_STOP').exists():
        raise RuntimeError('Explicit temporal trial STOP')
    if (ROOT / 'reports/ac_vs_redesign_v1/ACTIVE_WORKER.json').exists():
        raise RuntimeError('Unexpected full-model worker; do not overlap')
    if not (ROOT / 'reports/cell6_bc_time_calibration_20261028_v1/REQUEST_STOP').exists():
        raise RuntimeError('BC pause must remain retained')


def check():
    guard()
    c, spec = read(HERE / 'CONTRACT.json'), read(SPEC)
    for path, expected in {**spec['code_pins'], **c['pins']}.items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Frozen code/input changed: ' + path)
    return spec, c


def prepare():
    HERE.mkdir(parents=True, exist_ok=True)
    if (HERE / 'CONTRACT.json').exists():
        raise RuntimeError('Contract already frozen')
    guard()
    paths = [Path(__file__), ROOT / 'retina_sim/temporal_readout_v1.py', SPEC, OBS, REGISTRY,
             HS / 'FROZEN_PARAMETERS.json', HS / 'CURVES.npz', CD / 'FROZEN_CD_PARAMETERS.json',
             CD / 'CURVES.npz', CD / 'BOOTSTRAP_INDICES.npz', AB / 'reference_control.py']
    c = dict(schema='cell6-temporal-readout-v1', frozen_utc=utc(),
        pins={str(p.relative_to(ROOT)): sha(p) for p in paths},
        target=read(SPEC)['target'],
        formula='x=(V-m)/s; tau*da/dt=x-a; z=b+w0*x+w1*a+eta@past; lambda=90*softplus(z); p=-expm1(-.001*lambda)',
        capacity='one effective causal time state, seven trainable coordinates, no new physiological equation',
        parameterization='theta=[b,g_DC,k,eta1,eta2,eta3,log(tau/.010)]; w0=g_DC+k; w1=-k',
        ON_constraint='g_DC>=0,k>=0: nonnegative instantaneous and steady depolarization response with history fixed; effective lead component, not verified adaptation biology',
        bounds=dict(b=[-10,10],g_DC=[0,10],k=[0,10],eta=[-12,0],tau_ms=[2,50]),
        bounds_source='engineering standardized coefficient bounds from prior reference trials; tau resolved at100us and1s context>=20*max tau; not empirical Cell6 time constants or selected using g3',
        tau_training='one jointly optimized positive log coordinate, no time-constant scan or separate models',
        initialization='single existing g0-g2 static-independent S: b,g_DC,eta; k=1 and tau10ms; no restart or D-g3 tuning',
        regularization='0.5e-4*(w0^2+w1^2+sum(eta^2)); b/logtau unpenalized; raw NLL separately reported',
        optimizer=dict(method='L-BFGS-B',maxiter=200,maxfun=1000,maxls=30,ftol=1e-10,gtol=1e-7,
                       failure='stop and retain, no restart/expanded bounds/scans',starts=1),
        train_groups=[0,1,2],internal_development=[3],development_validation=[4,5],
        historical_test='not loaded; g3/g4/g5 are previously development-touched, not independent new test',
        statistics='pooled train g0-g2 V at1ms bin starts within[1,7)s only; same m/s for every group',
        initial_a='shared source-gray V0 from g0 transformed with train m/s; verify all source V0 equal; no per-eval initialization estimate',
        initial_history_probability=read(SPEC)['initial_probability'],
        context='full first1s recorded source voltage before scoring; source initial gray equilibrium, not a claim prefix is all gray',
        integration='exact ZOH100us recurrence on original native V; a at bin start precedes current x; no future interpolation and no additional retina solve',
        observation='original1ms binary spike bins; conditional NLL uses strictly previous1/2/3ms spikes, free prediction exact8state law without observed history',
        score='[1,7)s;20ms rectangles/no smoothing/zero shift/no best lag or posthoc polarity',
        uncertainty='same saved CD1000 paired resampling:3 spike trials and30 aligned200ms blocks; descriptive conditional intervals, not biological replication',
        comparisons='T vs frozen H, D, original source production; positive deltaNLL or deltaMSE favors T; r difference where defined; constant correlation undefined, no equivalence from nonsignificance',
        retain_rule=dict(training='optimizer converged and finite; necessary causal/domain/gradient/restore checks pass',
            g3='point deltaNLLvsH>0, free_r>0, deltaMSEvsH>0; internal consistency only',
            g4_g5='each: paired lower95 deltaNLLvsH>0, lower95 deltaMSEvsH>0, lower95 free_r>0',
            D='comparative diagnostic, not a hyperparameter selection gate',
            uncertain='if output finite but predictive gates incomplete -> INCONCLUSIVE; numerical/optimization failure -> REJECTED; stop, no further models'),
        identifiability='train expected Bernoulli Fisher Jacobian SVD relative1e-3 reported; rank/active bounds limit parameter interpretation, no physiological identity',
        needed_checks=['analytic constant-state solution','past-only native state','logtau gradient','chunk/readout-state restore vs batch','fixed-history and free synthetic voltage-port ON check'],
        controls='H/D frozen0..3 curves reused; H/D fixed parameters evaluated4/5 once only, not fitted. Production is original source base and cached original expectation.',
        expected_cost='one7-coordinate offline fit on54000 observations, plus six short readout evaluations/bootstrap; minutes, no ODE; working arrays small vs source caches',
        production_changed=False,upstream_forward_count=0,BC_scan_paused=True,FIT_READY=False,
        absolute_monitor_false_preserved=True,no_default_switch=True,no_UI_or_publish=True)
    write(HERE / 'CONTRACT.json', c)
    write(CONFIG, dict(schema='conditional-output-opt-in-1',representation='coarse_group_state',
        target=c['target'],default_selected=False,experimental=True,implementation='retina_sim.temporal_readout_v1',
        model=str((HERE/'MODEL.json').relative_to(ROOT)),FIT_READY=False))
    print(json.dumps(dict(stage='CONTRACT_FROZEN',utc=utc(),parameters=7,tau_ms=[2,50],retina_forwards=0)),flush=True)


def load_case(index):
    spec, _ = check()
    if index not in range(6) or index >= 3 and not (HERE/'MODEL.json').exists():
        raise RuntimeError('Heldout read only after candidate freeze; historical test excluded')
    target = spec['target']
    if target['global_id'] != 'sg2:61468d3761f46cda:parasol_RGC_ON:8166' or target['column'] != 1:
        raise RuntimeError('Exact target group required')
    case = spec['cases'][index]
    record = next(x for x in read(REGISTRY)['caches'] if x['case_id'] == case['case_id'])
    directory = Path(record['directory'])
    m = read(directory/'manifest.json')
    if (m['cache_identity'] != canonical_hash(dict(spec=spec,case=case))
            or sha(directory/'endogenous.npz') != m['arrays_sha256']
            or m['units'] != dict(V='V',gE='S',gI='S')):
        raise RuntimeError('Original cache identity/units mismatch')
    with np.load(directory/'endogenous.npz',allow_pickle=False) as z:
        if not np.array_equal(z['time_source_s'],np.arange(71001)*.0001):
            raise RuntimeError('Native sample time mismatch')
        v = z[target['population']+':V'][:,target['column']].copy()
        production = z['expected_rate_base_Hz'][:,target['column']].copy()*.001
    with np.load(OBS,allow_pickle=False) as z:
        y = z[case['counts_key']].copy()
    if v.shape != (71001,) or y.shape != (3,7100) or not np.isin(y,[0,1]).all():
        raise RuntimeError('Original observations/voltage dimensions mismatch')
    if not np.isfinite(v).all() or np.any((v<-.08)|(v>0)):
        raise RuntimeError('Source voltage domain failure')
    with (HERE/'DATA_ACCESS.jsonl').open('a',encoding='utf8') as stream:
        stream.write(json.dumps(dict(utc=utc(),group=index,stimulus_sha256=case['stimulus_sha256'],
            model_frozen=(HERE/'MODEL.json').exists(),cache_identity=m['cache_identity']))+'\n')
    return dict(group=index,voltage=v[:-1],bin_voltage=v[::10][:-1],counts=y,
                past=observed_past(y),production_probability=production,
                stimulus_sha256=case['stimulus_sha256'],cache_identity=m['cache_identity'])


def features(case,theta,stats):
    x = (case['voltage']-stats['mean_V'])/stats['SD_V']
    a,da = causal_state(x,.010*np.exp(theta[6]),stats['initial_a'],True)
    return x[::10],a[:-1:10],da[:-1:10]


def objective(theta,cases,stats,penalty=True):
    loss=0.;grad=np.zeros(7);count=0
    for case in cases:
        x,a,da=features(case,theta,stats)
        signal=theta[0]+theta[1]*x+theta[2]*(x-a)
        z=signal[None,LO:HI]+case['past'][:,LO:HI]@theta[3:6]
        nll,gz=stable_terms(z,ALPHA,case['counts'][:,LO:HI])
        loss+=float(nll.sum());count+=nll.size
        total=gz.sum(axis=0)
        grad[0]+=total.sum()
        grad[1]+=total@x[LO:HI];grad[2]+=total@(x-a)[LO:HI]
        grad[3:6]+=np.einsum('rt,rtk->k',gz,case['past'][:,LO:HI])
        grad[6]+=total@(-theta[2]*da[LO:HI])
    value=loss/count;grad/=count
    if penalty:
        g,k=theta[1:3];eta=theta[3:6]
        value+=.5*RIDGE*((g+k)**2+k*k+eta@eta)
        grad[1]+=RIDGE*(g+k);grad[2]+=RIDGE*(g+2*k);grad[3:6]+=RIDGE*eta
    if not np.isfinite(value) or not np.isfinite(grad).all():
        raise RuntimeError('Nonfinite objective; stop')
    return value,grad


def core_for(theta,stats):
    spec=read(SPEC)
    return dict(schema='temporal_readout_v1',scope='Cell6 fixed-input local conditional grouped functional output',
        target=spec['target'],mean_train_V=stats['mean_V'],SD_train_V=stats['SD_V'],
        initial_a=stats['initial_a'],initial_gray_V=stats['initial_gray_V'],
        initial_history_probability=spec['initial_probability'],
        b=float(theta[0]),w0=float(theta[1]+theta[2]),w1=float(-theta[2]),
        eta=theta[3:6].tolist(),tau_s=float(.010*np.exp(theta[6])),alpha_Hz=90.,
        native_dt_s=.0001,bin_dt_s=.001,train_groups=[0,1,2],
        contract_sha256=sha(HERE/'CONTRACT.json'),
        inference_code_sha256=sha(ROOT/'retina_sim/temporal_readout_v1.py'),
        physiology='effective output filter only; neither channel identification nor physical beta',
        FIT_READY=False)


def necessary_checks(cases,stats,theta):
    constant=1.7;tau=.013;a0=-.3;n=1177
    a=causal_state(np.full(n,constant),tau,a0)
    closed=constant+(a0-constant)*np.exp(-np.arange(n+1)*NATIVE_DT/tau)
    analytic=float(np.max(abs(a-closed)))
    x=np.sin(np.arange(n)/13);future=x.copy();future[600:]+=3
    aa=causal_state(x,tau,a0);bb=causal_state(future,tau,a0)
    causal=bool(np.array_equal(aa[:601],bb[:601]))
    probe=np.array([.1,.3,1.,-2.,-1.,-.2,np.log(1.3)])
    _,g=objective(probe,cases,stats);numeric=[]
    for j in range(7):
        d=np.zeros(7);d[j]=1e-5
        numeric.append((objective(probe+d,cases,stats)[0]-objective(probe-d,cases,stats)[0])/2e-5)
    gradient=float(np.max(abs(g-np.asarray(numeric))))
    core=core_for(theta,stats);model=dict(core=core,model_id=digest(core),qualification='UNQUALIFIED')
    v=cases[0]['voltage'][:3307]
    one=TemporalReadout(model);whole=one.predict(v)
    part=TemporalReadout(model);first=part.predict(v[:1537])
    recovered=TemporalReadout(model,json.loads(json.dumps(part.state())))
    second=recovered.predict(v[1537:])
    chunk=float(np.max(abs(whole['probability']-np.r_[first['probability'],second['probability']])))
    state_error=max(abs(one.a-recovered.a),float(np.max(abs(one.pi-recovered.pi))))
    # Fixed-history response slopes are algebraic; free test acts on voltage, not optics.
    baseline=np.full(10200,stats['initial_gray_V'])
    up=baseline.copy();up[10000:]+=stats['SD_V']
    down=baseline.copy();down[10000:]-=stats['SD_V']
    p0=TemporalReadout(model).predict(baseline)['probability']
    pp=TemporalReadout(model).predict(up)['probability']
    pm=TemporalReadout(model).predict(down)['probability']
    on=bool(np.sum(pp[1000:])-np.sum(p0[1000:])>0 and np.sum(pm[1000:])-np.sum(p0[1000:])<0)
    passed=analytic<1e-12 and causal and gradient<1e-7 and chunk<1e-12 and state_error<1e-12 and on
    result=dict(passed=passed,analytic_state_max_error=analytic,future_causality_exact=causal,
        gradient_max_abs_error=gradient,chunk_restore_probability_max_error=chunk,restore_state_max_error=state_error,
        voltage_port_ON_step_passed=on,ON_scope='depolarizing voltage step +/-one train SD, not full optical ON qualification',
        no_true_history_in_free_inference=True,retina_forwards=0)
    write(HERE/'NECESSARY_CHECKS.json',result)
    if not passed:raise RuntimeError('New readout necessary check failed; no fit')


def train():
    if (HERE/'MODEL.json').exists() or (HERE/'FIT.json').exists():
        raise RuntimeError('One candidate training only; use export-resume for frozen outputs')
    cases=[load_case(i) for i in (0,1,2)]
    pool=np.concatenate([c['bin_voltage'][LO:HI] for c in cases])
    stats=dict(mean_V=float(pool.mean()),SD_V=float(pool.std()),
               initial_gray_V=float(cases[0]['voltage'][0]))
    if stats['SD_V']<=0 or any(c['voltage'][0]!=stats['initial_gray_V'] for c in cases):
        raise RuntimeError('Nonzero train SD and shared source gray initial voltage required')
    stats['initial_a']=(stats['initial_gray_V']-stats['mean_V'])/stats['SD_V']
    old=read(HS/'FROZEN_PARAMETERS.json')
    theta=np.r_[old['b'],old['gamma'],1.,old['eta'],0.]
    write(HERE/'TRAIN_STATISTICS_FROZEN.json',dict(utc=utc(),statistics=stats,initial_theta=theta.tolist()))
    necessary_checks(cases,stats,theta)
    start=time.perf_counter();trace=[];calls=0
    print(json.dumps(dict(stage='TRAINING_STARTED',utc=utc(),parameters=7,groups=[0,1,2])),flush=True)
    def fn(t):
        nonlocal calls
        guard();calls+=1
        return objective(t,cases,stats)
    def cb(t):
        val,_=objective(t,cases,stats)
        trace.append(dict(iteration=len(trace)+1,theta=t.tolist(),penalized_objective=val))
        if len(trace)%10==0:
            print(json.dumps(dict(stage='TRAINING',iteration=len(trace),evaluations=calls,objective=val)),flush=True)
    opt=minimize(fn,theta,jac=True,method='L-BFGS-B',bounds=BOUNDS,callback=cb,
        options=dict(maxiter=200,maxfun=1000,maxls=30,ftol=1e-10,gtol=1e-7))
    fit=dict(utc=utc(),converged=bool(opt.success),message=str(opt.message),theta=opt.x.tolist(),
        iterations=int(opt.nit),function_evaluations=int(opt.nfev),penalized_objective=float(opt.fun),
        raw_NLL=objective(opt.x,cases,stats,False)[0],wall_seconds=time.perf_counter()-start,trace=trace,
        active_bounds=[i for i,(v,(lo,hi)) in enumerate(zip(opt.x,BOUNDS)) if min(abs(v-lo),abs(v-hi))<1e-7])
    write(HERE/'FIT.json',fit)
    if not opt.success:raise RuntimeError('One training failed to converge; no restart')
    jac=[]
    for case in cases:
        x,a,da=features(case,opt.x,stats);past=case['past'][:,LO:HI]
        j=np.concatenate([np.ones((*past.shape[:2],1)),
            np.broadcast_to(np.column_stack([x[LO:HI],(x-a)[LO:HI]])[None,:,:],(*past.shape[:2],2)),
            past,np.broadcast_to((-opt.x[2]*da[LO:HI])[None,:,None],(*past.shape[:2],1))],axis=2)
        z=opt.x[0]+opt.x[1]*x[None,LO:HI]+opt.x[2]*(x-a)[None,LO:HI]+past@opt.x[3:6]
        exposure=.09*np.logaddexp(0.,z);q=-np.expm1(-exposure)
        weight=np.exp(.5*(2*np.log(.09)-2*np.logaddexp(0.,-z)-exposure-np.log(q)))
        jac.append((j*weight[:,:,None]).reshape(-1,7))
    sv=np.linalg.svd(np.vstack(jac)/np.sqrt(54000),compute_uv=False)
    write(HERE/'IDENTIFIABILITY.json',dict(relative_min_singular_value=float(sv[-1]/sv[0]),
        singular_values=sv.tolist(),threshold=.001,local_rank_passed=bool(sv[-1]/sv[0]>=.001),
        scope='conditional training Fisher, not biological identification'))
    core=core_for(opt.x,stats)
    model=dict(model_id=digest(core),core=core,qualification='FROZEN_NOT_EVALUATED',frozen_utc=utc())
    write(HERE/'MODEL.json',model)
    print(json.dumps(dict(stage='MODEL_FROZEN',utc=utc(),model_id=model['model_id'],
        tau_ms=1000*core['tau_s'],wall_seconds=fit['wall_seconds'])),flush=True)


def helpers():
    s=importlib.util.spec_from_file_location('frozen_reference_pure_helpers',AB/'reference_control.py')
    ref=importlib.util.module_from_spec(s);s.loader.exec_module(ref)
    return ref


def metrics(case,p,nll):
    y=case['counts'][:,LO:HI].reshape(3,300,20).mean(axis=2)*1000.
    pred=p[LO:HI].reshape(300,20).mean(axis=1)*1000.;obs=y.mean(axis=0)
    r=None if pred.std()<1e-12 or obs.std()<1e-12 else float(np.corrcoef(pred,obs)[0,1])
    return dict(conditional_NLL_nat_per1ms=float(nll[:,LO:HI].mean()),free_r=r,
        free_MSE_Hz2=float(np.mean((pred-obs)**2)),mean_Hz=float(pred.mean()),SD_Hz=float(pred.std()),
        observed_mean_Hz=float(obs.mean()),observed_SD_Hz=float(obs.std()))


def interval(values):
    x=np.asarray(values);x=x[np.isfinite(x)]
    return None if not len(x) else np.quantile(x,[.025,.5,.975]).tolist()


def paired(case,prob,losses):
    with np.load(CD/'BOOTSTRAP_INDICES.npz',allow_pickle=False) as z:
        trials=z['spike_repeats'];blocks=z['time_blocks']
    obs=case['counts'][:,LO:HI].reshape(3,30,10,20).mean(axis=3)*1000
    p={k:v[LO:HI].reshape(30,10,20).mean(axis=2)*1000 for k,v in prob.items()}
    loss={k:v[:,LO:HI].reshape(3,30,200).mean(axis=2) for k,v in losses.items()}
    samples={k:dict(nll=[],mse=[],r=[],delta_r=[]) for k in ('H','D','production')}
    for tr,bl in zip(trials,blocks):
        y=obs[tr][:,bl].mean(axis=0).ravel();q=p['T'][bl].ravel()
        rt=np.corrcoef(y,q)[0,1] if q.std()>1e-12 and y.std()>1e-12 else np.nan
        for k,record in samples.items():
            other=p[k][bl].ravel()
            ro=np.corrcoef(y,other)[0,1] if other.std()>1e-12 and y.std()>1e-12 else np.nan
            record['nll'].append(float((loss[k]-loss['T'])[tr][:,bl].mean()))
            record['mse'].append(float(np.mean((other-y)**2)-np.mean((q-y)**2)))
            record['r'].append(rt);record['delta_r'].append(rt-ro)
    return {k:dict(delta_NLL_nat_per1ms_95=interval(v['nll']),delta_MSE_Hz2_95=interval(v['mse']),
                   free_r_95=interval(v['r']),delta_r_95=interval(v['delta_r']),
                   paired=True,scope='descriptive conditional block/trial resampling') for k,v in samples.items()}


def evaluate_case(index,model,ref):
    out=HERE/('g'+str(index)+'_RESULT.json');npz=HERE/('g'+str(index)+'_CURVES.npz')
    if out.exists() and npz.exists():
        saved=read(out)
        if saved['model_id']!=model['model_id'] or sha(npz)!=saved['curves_sha256']:
            raise RuntimeError('Completed group output identity changed')
        return saved
    if out.exists() or npz.exists():
        raise RuntimeError('Partial group export; preserve and report before recovery')
    case=load_case(index);core=model['core']
    if case['voltage'][0]!=core['initial_gray_V']:
        raise RuntimeError('Eval source initial voltage differs from frozen shared gray')
    theta=np.asarray(read(HERE/'FIT.json')['theta']);stats=read(HERE/'TRAIN_STATISTICS_FROZEN.json')['statistics']
    x,a,_=features(case,theta,stats)
    signal=core['b']+core['w0']*x+core['w1']*a
    z=signal[None,:]+case['past']@core['eta']
    losses={'T':stable_terms(z,ALPHA,case['counts'])[0]}
    probabilities={'T':TemporalReadout(model).predict(case['voltage'])['probability']}
    if index<4:
        with np.load(HS/'CURVES.npz',allow_pickle=False) as z:
            probabilities['H']=z['g'+str(index)+'_H_probability'].copy()
            losses['H']=z['g'+str(index)+'_H_NLL'].copy()
        with np.load(CD/'CURVES.npz',allow_pickle=False) as z:
            probabilities['D']=z['g'+str(index)+'_D_free_probability'].copy()
            losses['D']=z['g'+str(index)+'_D_conditional_NLL'].copy()
    else:
        h=np.asarray(read(HS/'FROZEN_PARAMETERS.json')['models']['H']['theta'])
        hsignal=np.full(7100,h[0])
        probabilities['H']=free_probability(hsignal,h[1:],core['initial_history_probability'])[0]
        losses['H']=stable_terms(hsignal[None,:]+case['past']@h[1:],90.,case['counts'])[0]
        d=read(CD/'FROZEN_CD_PARAMETERS.json');t=np.asarray(d['models']['D']['theta']);s=d['statistics']['D']
        values=case['voltage'].reshape(7100,10).mean(axis=1)
        feat=(ref.basis((values-s['input_mean'])/s['input_SD'])-s['feature_mean'])/s['feature_SD']
        ds=t[0]+feat@t[1:4]
        probabilities['D']=free_probability(ds,t[4:],core['initial_history_probability'],alpha=100.)[0]
        losses['D']=stable_terms(ds[None,:]+case['past']@t[4:],100.,case['counts'])[0]
    base=read(SPEC)['base']
    probabilities['production']=case['production_probability']
    pz=(case['bin_voltage'][None,:]+case['past']@base['history_kernel']-base['mu'])/base['beta']
    losses['production']=stable_terms(pz,base['alpha'],case['counts'])[0]
    rows={k:metrics(case,p,losses[k]) for k,p in probabilities.items()}
    for k,row in rows.items():
        row['delta_NLL_vs_H']=rows['H']['conditional_NLL_nat_per1ms']-row['conditional_NLL_nat_per1ms']
    arrays={**{k+'_probability':v for k,v in probabilities.items()},
            **{k+'_conditional_NLL':v for k,v in losses.items()},
            'observed_counts':case['counts'],'native_state_a_at_bin_start':a,'x_bin_start':x}
    archive(npz,arrays)
    result=dict(utc=utc(),group=index,model_id=model['model_id'],cache_identity=case['cache_identity'],
        stimulus_sha256=case['stimulus_sha256'],models=rows,paired=paired(case,probabilities,losses),
        curves_sha256=sha(npz),control_evaluation='saved H/D reused' if index<4 else 'frozen H/D evaluated once, no refit',
        valid_probabilities=all(np.isfinite(p).all() and np.all((p>=0)&(p<=1)) for p in probabilities.values()),
        scope='train' if index<3 else 'internal development' if index==3 else 'development validation, previously touched')
    write(out,result)
    print(json.dumps(dict(stage='GROUP_EVALUATED',group=index,utc=utc(),T=rows['T'])),flush=True)
    return result


def finalize():
    spec,c=check();model=load_model(HERE/'MODEL.json',allow_unqualified=True);ref=helpers()
    rows=[evaluate_case(i,model,ref) for i in range(6)]
    g3=rows[3]['models'];r3=g3['T']
    internal=bool(r3['delta_NLL_vs_H']>0 and r3['free_r'] is not None and r3['free_r']>0
                  and r3['free_MSE_Hz2']<g3['H']['free_MSE_Hz2'])
    dev={}
    for index in (4,5):
        p=rows[index]['paired']['H']
        dev['g'+str(index)]=bool(all(p[k] is not None and p[k][0]>0
            for k in ('delta_NLL_nat_per1ms_95','delta_MSE_Hz2_95','free_r_95')))
    keep=internal and all(dev.values()) and all(r['valid_probabilities'] for r in rows)
    status='RETAIN_CONDITIONAL' if keep else 'INCONCLUSIVE'
    model['qualification']=status;write(HERE/'MODEL.json',model)
    result=dict(utc=utc(),status=status,model_id=model['model_id'],internal_g3=internal,development_gates=dev,
        groups=rows,fit=read(HERE/'FIT.json'),identifiability=read(HERE/'IDENTIFIABILITY.json'),
        FIT_READY=False,absolute_monitor_false_preserved=True,production_default_changed=False,
        physical_calibration=False,additional_models_or_scans=0,retina_forwards=0)
    write(HERE/'RESULT.json',result)
    report(result)
    check()
    write(HERE/'FINAL_RECEIPT.json',dict(status='COMPLETE_AND_STOPPED',utc=utc(),decision=status,
        model_id=model['model_id'],report=str(HERE/'REPORT.md'),report_sha256=sha(HERE/'REPORT.md'),
        worker_retained=False,BC_STOP_retained=True,new_fit_count=1,control_refits=0,full_model_runs=0,
        elapsed_seconds=time.perf_counter()-BEGAN,FIT_READY=False))
    print(json.dumps(dict(stage='COMPLETE',decision=status,report=str(HERE/'REPORT.md'),
                         wall_seconds=time.perf_counter()-BEGAN)),flush=True)


def report(r):
    core=read(HERE/'MODEL.json')['core'];fit=r['fit']
    lines=['# Cell6 temporal_readout_v1 条件性功能修复结果','',
        '决定：**'+r['status']+'**。仅评价固定内生膜电压上的独立输出版本；生产模型与生理参数均未改变，FIT_READY=false，原 absolute monitor=false 保留。','',
        '## 冻结实现与训练','',
        '唯一候选为 x=(V−m)/s，τ da/dt=x−a，z=b+w0 x+w1 a+η·过去三个二值 spike，λ=90 softplus(z)，每1ms事件概率 p=1−exp(−λ×0.001)。期望计数率是 p/0.001，不能将 λ 直接当离散放电率。','',
        '采用 w0=g_DC+k、w1=−k、g_DC/k≥0 的预声明 ON 电压端口约束。τ仅在2–50ms内以对数坐标联合训练一次，没有时间常数扫描。a 是有效因果输出滤波状态，不代表已验证动态阈值、适应离子通道或物理膜参数。','',
        'g0–g2确定标准化与全部参数；g3内部开发检查，g4/g5开发验证。它们均已在开发中被触碰，不是全新独立测试。历史 test 未读取；没有根据评估结果改参数。','',
        '100μs原始膜电压按ZOH积分；1ms bin-start使用消费当前样本前的a，严格过去三bin history。所有组共用g0源灰背景初值与原八态历史概率；评分前使用各组首1s真实上下文，并非把这段伪称恒定灰底。','',
        '模型身份：'+r['model_id']+'。合同冻结于 '+read(HERE/'CONTRACT.json')['frozen_utc']+'，参数冻结于 '+read(HERE/'MODEL.json')['frozen_utc']+'。','',
        'τ='+str(1000*core['tau_s'])+'ms；b='+str(core['b'])+'；w0='+str(core['w0'])+'；w1='+str(core['w1'])+'；η='+str(core['eta'])+'。','',
        '训练迭代/目标评价='+str(fit['iterations'])+'/'+str(fit['function_evaluations'])+'，训练墙时='+str(fit['wall_seconds'])+'s；原始NLL='+str(fit['raw_NLL'])+'，惩罚目标='+str(fit['penalized_objective'])+'；活跃边界坐标='+str(fit['active_bounds'])+'。Fisher最小/最大奇异值='+str(r['identifiability']['relative_min_singular_value'])+'，只描述训练局部参数辨识。','',
        '## 结果表','',
        'NLL单位为 nat/单个1ms Bernoulli观测；ΔNLL=NLL_H−NLL_模型。PSTH为[1,7)s原20ms矩形分箱，无平滑/平移/幅值重缩放；自由预测从未使用真实spike history。','',
        '|组/模型|条件NLL|ΔNLL对H|自由r|MSE Hz²|均值 Hz|SD Hz|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in r['groups']:
        for name,m in row['models'].items():
            lines.append('|g'+str(row['group'])+'/'+name+'|'+'|'.join(
                '未定义' if m[k] is None else format(m[k],'.8g')
                for k in ('conditional_NLL_nat_per1ms','delta_NLL_vs_H','free_r','free_MSE_Hz2','mean_Hz','SD_Hz'))+'|')
    lines+=['','## 开发检查与描述性区间','',
        '|组/相对对照|ΔNLL 95%|ΔMSE Hz² 95%|r或Δr 95%|',
        '|---|---|---|---|']
    for row in r['groups'][3:]:
        for name,p in row['paired'].items():
            lines.append('|g'+str(row['group'])+'/'+name+'|'+str(p['delta_NLL_nat_per1ms_95'])+'|'+
                str(p['delta_MSE_Hz2_95'])+'|'+str(p['free_r_95'] if name=='H' else p['delta_r_95'])+'|')
    lines+=['',
        '沿用冻结1000次配对重采样：3个spike重复与30个对齐200ms区块。各模型使用相同索引；正ΔNLL/ΔMSE有利于T。此区间是条件描述，少量重复及时间自相关限制统计解释，不是新生物重复。差异不显著不等于等效，组合门失败也不等于输入无信息。','',
        'g3点判据='+str(r['internal_g3'])+'；g4/g5预冻结区间判据='+str(r['development_gates'])+'。D是比较对照而非调参目标；未要求超过D才能保留。H/D g0–g3使用保存预测，g4/g5只评估其冻结参数一次，无重训。D使用alpha100Hz及三个1–100ms基函数，共7参；本模型alpha90Hz、一个递归状态、7参及ON约束，不能把差异仅归为某一生理机制。','',
        '## 资格与限制','',
        ('保留为显式加载的独立条件性功能输出版本；不切换默认、不接UI、不发布。'
         if r['status']=='RETAIN_CONDITIONAL' else
         '证据未满足全部预冻结功能门，保留实现与结果为不确定版本，停止；不得自动采用或追加状态/扫参。'),
        '电压端口ON检查仅验证固定history的瞬时/稳态斜率及合成去极化阶跃的自由期望响应，不能代替光刺激全链ON极性或生理资格。','',
        '模型电压SD仅约 '+str(core['SD_train_V']*1e6)+' μV；标准化会放大已有信息，但不能证明原电压幅度、生理电导、BC、膜滤波或原读出错误。τ/权重属于功能参数，不是物理beta或通道时间常数。输入物理标定、绝对采集延迟、局部外边界与Cell6解剖注册均未闭合；目标为两个成员的工程代表group状态，不是两条独立逐细胞轨迹。','',
        '不跑全链、不恢复BC扫描、不读历史test、不改变训练数据或生产源；所有旧失败保留。完成后停止。','',
        '## 本地使用','',
        '指定原环境运行 tools/run_temporal_readout_v1.py infer --model reports/cell6_temporal_readout_v1_20261028/MODEL.json --voltage native_voltage_V.npy --output predicted.npz。输入为从源初态开始的100μs一维SI电压，省略记录末端7.1s孤立样本；推理输出1ms概率与期望计数率。完整命令见 README.md。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (HERE/'README.md').write_text(
        '# temporal_readout_v1 显式使用\n\n此版本不改变默认模型。资格见 MODEL.json / RESULT.json。\n\n'
        '训练和评估（本目录结果已存在时禁止重复训练）：\n\n'
        '    D:\\macaque_retina_simulator_envs\\cu126_py311_v1\\Scripts\\python.exe -B -X utf8 tools/run_temporal_readout_v1.py prepare\n'
        '    D:\\macaque_retina_simulator_envs\\cu126_py311_v1\\Scripts\\python.exe -B -X utf8 tools/run_temporal_readout_v1.py run\n\n'
        '显式推理（只允许已保留版本，其他版本将拒绝）：\n\n'
        '    D:\\macaque_retina_simulator_envs\\cu126_py311_v1\\Scripts\\python.exe -B -X utf8 tools/run_temporal_readout_v1.py infer --model reports/cell6_temporal_readout_v1_20261028/MODEL.json --voltage native_voltage_V.npy --output predicted.npz\n\n'
        '输入SI V，100μs一行，从模型声明初态开始。可以用 --state-in / --state-out 分块连续推理；'
        '恢复状态只覆盖输出filter/八态history概率，不是全链checkpoint。输出bin_index为全局1ms索引，'
        'probability为该bin事件概率，expected_count_rate_Hz=probability/.001。不要将group状态复制后声称逐细胞观测。\n\n'
        '原缓存不修改。本轮代码、合同、冻结统计、FIT、必要检查、逐组曲线、配对区间均在当前目录。'
        'export-resume只恢复已冻结模型的缺失评估，不重新训练；完整结果存在时拒绝执行。\n',
        encoding='utf8')


def infer(args):
    model=load_model(args.model)
    state=read(args.state_in) if args.state_in else None
    predictor=TemporalReadout(model,state)
    v=np.load(args.voltage,allow_pickle=False)
    result=predictor.predict(v)
    archive(args.output,{**result,'model_id':np.asarray(model['model_id'])})
    if args.state_out:write(args.state_out,predictor.state())
    print(json.dumps(dict(model_id=model['model_id'],bins=len(result['probability']),output=args.output)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('prepare','run','export-resume'):sub.add_parser(command)
    p=sub.add_parser('infer');p.add_argument('--model',required=True);p.add_argument('--voltage',required=True)
    p.add_argument('--output',required=True);p.add_argument('--state-in');p.add_argument('--state-out')
    args=parser.parse_args()
    if args.command=='prepare':prepare();return
    if args.command=='infer':infer(args);return
    check()
    if (HERE/'FINAL_RECEIPT.json').exists():raise RuntimeError('Completed trial; stop, no duplicate run')
    lock=HERE/'ACTIVE_OFFLINE.json';owner=dict(pid=os.getpid(),utc=utc(),scope='readout-only')
    with lock.open('x') as stream:json.dump(owner,stream)
    try:
        if args.command=='run':train()
        else:load_model(HERE/'MODEL.json',allow_unqualified=True)
        finalize()
    except BaseException as exc:
        write(HERE/('FAILURE_'+datetime.now(timezone.utc).strftime('%H%M%S')+'.json'),
            dict(utc=utc(),reason=str(exc),traceback=traceback.format_exc(),retina_forwards=0,FIT_READY=False))
        raise
    finally:
        if lock.exists() and read(lock)==owner:lock.unlink()


BEGAN=time.perf_counter()
if __name__=='__main__':main()
