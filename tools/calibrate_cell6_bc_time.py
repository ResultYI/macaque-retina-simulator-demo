"""Finite conditional ON diffuse BC effective-time calibration. No default adoption."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.rgc_stage1_cache import write, StageWorker, read_cache
from retina_sim.bc_effective_time_candidate import EffectiveTimeAssembly, POPULATION, CAPACITANCE

OUT = ROOT / 'reports/cell6_bc_time_calibration_20261028_v1'
TARGETS = ROOT / 'reports/layer_calibration/ac_ei_targets_v1'
QUAL = ROOT / 'experiments/local_patch_backend_v1'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    from tools.fit_rgc_stage1 import verify as verify_original
    spec = verify_original()
    protocol = read(OUT/'PRE_REGISTERED.json')
    for relative, digest in protocol['new_code_pins'].items():
        if sha(ROOT/relative) != digest:
            raise ContractError('New calibration implementation changed after freezing: '+relative)
    for relative, digest in protocol['target_pins'].items():
        if sha(TARGETS/relative) != digest:
            raise ContractError('Original E target identity changed')
    if (ROOT/'REQUEST_STOP').exists() or (OUT/'REQUEST_STOP').exists():
        raise ContractError('Explicit calibration STOP')
    return spec, protocol


def prepare():
    from tools.fit_rgc_stage1 import verify as verify_original
    spec = verify_original()
    OUT.mkdir(exist_ok=True)
    destination = OUT/'PRE_REGISTERED.json'
    if destination.exists():
        raise ContractError('Frozen protocol already exists')
    protocol = dict(schema='cell6-on-diffuse-bc-time-calibration-1', frozen_utc=utc(),
        approved_selector='bc.diffuse_ON.tau_eff_s',
        parameter=dict(baseline_s=.020, bounds_s=[.005,.030],
            formula='C_eff_per_member = fixed_gL_per_member * tau_eff',
            interpretation='conditional effective membrane time, not measured capacitance'),
        estimation_groups=['g0','g1','g2'], candidate_check='one g3 after candidate freeze; previously explored training group',
        validation_access=False, historical_test_access=False, input='unchanged original0.05 arm only',
        target=spec['target'], output_theta='original prior g0-g3 TRAINING_SELECTION, fixed; no readout refit',
        score_interval_source_s=[1.,7.], source_dt_s=.0001, primary_grid='100us pre-event native source-grid nodes, [1,7)',
        auxiliary_grids=['1ms mean of10 source nodes','20ms mean of200 source nodes'],
        polarity='model positive gE compared with minus raw E current, sign frozen from prior E definition',
        objective='mean over3 groups of MSE(zscore(gE),zscore(-mean(E repeats))); temporal mean removed and population SD; no shift',
        old_loss='signed DC-retained own-RMS(-gE) versus signed own-RMS(raw E mean)',
        spectrum='fixed0..60Hz rFFT, same mean-subtracted rectangular6s; phase descriptive, no delay optimization',
        numerical=dict(solver='original ACVSInner.advance_reference SciPy DOP853', precision='FP64',
            rtol=1e-6, atol=1e-9, max_step_s=.0001, rod_tick_s=.0001, BC_RGC_tick_s=.001,
            monitor='original absolute false remains; original physical/history/domain guard retained',
            state_comparison='original frozen_budget(initial,unit)[2], normalized error <=1',
            baseline_lineage='new C binding changes full artifact/native source/interval IDs even at20ms; compare readout cursor content after removing ONLY source_version/native_representation_id identity tags; all numerical coefficients/epochs/stateIDs/consumer clocks retained'),
        execution=dict(original_CPU='DirectACVSRHS', candidate='existing opt-in LocalPatchCPURHS, only after real tau20 prefix equivalence',
            causal_closure='all native and all inner populations/contact/gap/history/RNG states evolve; no frozen BC/AII trajectories',
            prior_cache_reuse='baseline20 compact V/gE/gI only after new-path equivalence; no new tau trajectory from old RGC cache',
            native_cache='No exact selected native continuous driver exists in old compact caches; full causal native is initially recomputed. No sparse interpolation substitution.',
            workers=1, memory_reserve_GiB=4, process_RSS_limit_GiB=16,
            single_worker_safety_s=18108,
            safety_reason='2x prior9054s single7.1s case protection; not cumulative optimization limit',
            checkpoint='complete causal payload first50ms then500ms; same-identity resume only'),
        identifiability=dict(signal_SD='must exceed propagated original terminal_release/A state budgets through actual fixed E route',
            output_budget='N*GE*(abs(W)@(B_release + 6*0.75**5*B_A)); denominator>=1, release<=1; conservative mean-value bound',
            loss_numeric_bound='4*d + d*d where d=2*port_budget/(SD-port_budget), requires SD>port_budget',
            flat='objective range cannot exceed original numerical allowance or candidate improvements inconsistent across actual E repeats'),
        optimizer=dict(method='scipy.optimize.minimize_scalar bounded', bounds_s=[.005,.030],
            max_objective_evaluations=8, baseline20_counts_as_evaluation=1,
            remaining_bounded_maxiter=7, xatol_s=.001,
            resolution_reason='1ms BC history/event clock; no sub-ms parameter precision claim',
            initial='baseline20 plus deterministic bounded golden-section points; no15374gain initial',
            selection='lowest training objective among evaluated points including baseline20',
            boundary_stop='selected <=6ms or >=29ms; no g3',
            convergence='bounded termination or evaluation cap; resolution not reached must be explicit'),
        training_gate=dict(required=[
            'each3 group primary loss improvement > propagated numerical shape-loss bound',
            'for each3 group, improvement positive against EACH of3 actual E repeats',
            'mean improvement > SEM of three group improvements; groups not timebins are replication units',
            'repeat autocorrelation integrated positive-lag time reported; no independent-timebin pvalue',
            'old signed DC RMS loss per group not worse beyond propagated numerical loss allowance',
            'finite nonnegative conductance and original state/history domain pass'],
            no_noise_pvalue=True),
        g3_gate='same primary/old-loss/domain rules once; E repeats all-positive improvement; no retuning on g3',
        fixed=['all parameters except ON diffuse BC C_eff','OFF BC C','route gains/topology','AC/AII/RGC membrane/readout',
            'input/seed/states/delay/RNG','BC kernel/bias/gain/release/adaptation','source split'],
        outcome='CONTINUE only conditionally if training and one g3 checks pass; boundary/flat/unresolved/failed ->STOP or UNIDENTIFIABLE',
        original_spec_sha256=sha(ROOT/'reports/rgc_stage1_complete_v1/CONTRACT.json'),
        original_51_code_pins=spec['code_pins'],
        target_pins={n:sha(TARGETS/n) for n in ('target_definition.json','target_manifest.json','verification_report.json','targets_native_100us.npz')},
        new_code_pins={n:sha(ROOT/n) for n in ('retina_sim/bc_effective_time_candidate.py','tools/calibrate_cell6_bc_time.py')},
        FIT_READY=False, default_adopted=False)
    write(destination,protocol)
    write(OUT/'STATUS.json',dict(utc=utc(),stage='FROZEN_BEFORE_NEW_MODEL_RUN', model_workers=0,
        minimum_path='full coupled inner including AII gap and unchanged native; existing compiler pending one necessary equivalence',
        cost_basis='old direct full case59..76min; <=24 training case forward recomputations plus one g3; not8 cache reads',
        conditional_old_path_CPU_hours=[23.8,30.5], new_execution_cost_unmeasured=True))
    print(json.dumps(dict(status='PRE_REGISTERED',directory=str(OUT))),flush=True)


def qualify_old_manifest():
    spec, protocol = verify()
    from experiments.local_patch_backend_v1.qualify import run, compare
    for backend in ('direct','compiled'):
        name='bc_tau20_'+backend+'_v1'
        path=QUAL/'runs'/name
        if not (path/'RESULT.json').exists():
            run(SimpleNamespace(backend=backend,name=name,split=False,resume=None))
    destination=QUAL/'BC_TIME_COMPARISON.json'
    if not destination.exists():
        compare(SimpleNamespace(left=str(QUAL/'runs/bc_tau20_direct_v1'),
            right=str(QUAL/'runs/bc_tau20_compiled_v1'),exact=False,output=str(destination)))
    result=read(destination)
    write(OUT/'TAU20_PREFIX_EQUIVALENCE.json',dict(result, scope='10.1ms actual g0, all120 states/15SI ports; same equations/events; not long-run qualification',
        reference=str(QUAL/'runs/bc_tau20_direct_v1'),candidate=str(QUAL/'runs/bc_tau20_compiled_v1'),
        direct=read(QUAL/'runs/bc_tau20_direct_v1/RESULT.json'),
        compiled=read(QUAL/'runs/bc_tau20_compiled_v1/RESULT.json')))
    if not result['passed']:
        raise ContractError('Existing typed CPU path does not pass original real-model budget')


def qualify():
    """Current dependency-bound check, without changing the old experiment audit."""
    spec,protocol=verify()
    from retina_sim.group_time_runner import GroupTimeArm
    from retina_sim.ac_vs_model import DirectACVSRHS
    from retina_sim.group_time_monitor import GroupTimeMonitor,frozen_budget
    from retina_sim.group_time_storage import save_bundle,load_bundle
    from tools.validate_ac_vs_equilibrium_start import initial_match
    from tools.validate_ac_vs_timescale import calendar
    from tools.validate_ac_vs import causal_hash
    from retina_sim.rgc_cpu_case_isolation import resources
    before=resources()
    if before['available_RAM_bytes']<9*1024**3 or before['commit_headroom_bytes']<5*1024**3:
        raise ContractError('9GiB RAM /5GiB commit one-worker admission unavailable')
    directories={}
    for backend in ('direct','compiled'):
        out=OUT/('prefix20_'+backend+'_v2');out.mkdir(exist_ok=True);directories[backend]=out
        if (out/'RESULT.json').exists():continue
        arm=None;started=time.perf_counter()
        with StageWorker(ROOT,out,OUT,18108) as worker:
            try:
                a,seed,p,binding=case_context(spec,0,.020)
                if backend=='direct':a.runtime_factories=dict(a.runtime_factories,compiled=DirectACVSRHS)
                arm=GroupTimeArm(a,seed,p,mode='reference',guard=worker.check)
                monitor=GroupTimeMonitor(a,seed['states'],parameters=seed['inner_parameters'])
                init,_=initial_match(arm,seed)
                if not init['passed']:raise ContractError('Prefix initial states/history differ')
                units={':'.join((b.population,b.mechanism,b.name)):b.unit for b in
                    (*a.state_blocks,*a.probability_blocks,*a.auxiliary_z1_blocks)}
                states={};ports={};stamps=[]
                def observe(t,side,label):
                    s=arm.snapshot(t,side,queue_node=True)
                    monitor.observe(t,side,s['states'],s['rhs'],transitions=s['transitions'],
                        ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
                    if monitor.domain_failures:raise ContractError('Prefix domain/history failed')
                    stamps.append([t,side,label])
                    for k,v in s['states'].items():states.setdefault(':'.join(k),[]).append(v.copy())
                    for pop in spec['recording']:
                        for name in ('voltage','gE','gI'):
                            ports.setdefault(pop+':'+name,[]).append(s['ports'][pop,name].copy())
                hot=time.perf_counter()
                observe(arm.start,'right','before_initial')
                arm.initial_commit();observe(arm.start,'right','after_initial');arm.ack(arm.start)
                target=arm.start+.0101
                for end,flags in calendar(p):
                    if end>target+1e-13:break
                    begin=arm.inner_time;arm.ensure_native(end,'left');arm.advance_inner(begin,end)
                    observe(end,'left' if flags['frame'] else 'right','pre')
                    if flags['rod'] or flags['bc']:
                        arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],
                            rod_sample_time=max(arm.start,end-.0001+1e-13) if flags['rod'] else None)
                    observe(end,'right','post');arm.ack(end)
                if abs(arm.inner_time-target)>1e-13:raise ContractError('Prefix endpoint differs')
                wall=time.perf_counter()-hot
                payload=arm.checkpoint_payload(monitor=monitor.payload())
                save_bundle(out/'checkpoint',kind='complete_model_checkpoint',identity=dict(arm=arm.identity),
                    payload=payload,qualification=dict(default_adopted=False,fit_eligible=False,FIT_READY=False))
                np.savez_compressed(out/'traces.npz',
                    **{'state__'+k:np.asarray(v) for k,v in states.items()},
                    **{'port__'+k:np.asarray(v) for k,v in ports.items()})
                write(out/'metadata.json',dict(units=units,stamps=stamps))
                write(out/'RESULT.json',dict(backend=backend,source_seconds=arm.inner_time-arm.start,
                    pid=os.getpid(),hot_wall_seconds=wall,total_wall_seconds=time.perf_counter()-started,
                    peak_RSS_bytes=worker.memory.peak_rss_bytes,metrics=arm.metrics,
                    original_absolute=monitor.report(.0101),initial_match=init,binding=binding,
                    resources_before=before,resources_after=resources(),original51pins_match=True))
                print(json.dumps(dict(stage='PREFIX_COMPLETE',backend=backend,wall_seconds=wall)),flush=True)
            except BaseException as exc:
                write(out/('FAILURE_'+str(time.time_ns())+'.json'),dict(reason=str(exc),traceback=traceback.format_exc()))
                raise
            finally:
                if arm is not None:arm.close()
    lm=read(directories['direct']/'metadata.json');rm=read(directories['compiled']/'metadata.json')
    if lm!=rm:raise ContractError('Prefix observation/phase/field metadata differ')
    fields=[]
    with np.load(directories['direct']/'traces.npz',allow_pickle=False) as x,np.load(directories['compiled']/'traces.npz',allow_pickle=False) as y:
        if set(x.files)!=set(y.files):raise ContractError('Prefix field coverage differs')
        for name in x.files:
            category,k=name.split('__',1);a,b=x[name],y[name]
            unit=lm['units'][k] if category=='state' else 'V' if k.endswith(':voltage') else 'S'
            allowance=frozen_budget(a[0],unit)[2]
            norm=float(np.max(np.abs(a-b)/allowance))
            fields.append(dict(field=name,unit=unit,norm=norm,passed=norm<=1,exact=bool(np.array_equal(a,b))))
    lp=load_bundle(directories['direct']/'checkpoint',expected_kind='complete_model_checkpoint')['payload']
    rp=load_bundle(directories['compiled']/'checkpoint',expected_kind='complete_model_checkpoint')['payload']
    def sampled(v):
        return {p:{k:x for k,x in st.items() if k!='conditional_rate_Hz'} for p,st in v.items()}
    discrete={k:causal_hash(lp[k])==causal_hash(rp[k]) for k in ('clocks','input_position','observation_delay','readout_cursor')}
    discrete['RNG_history']=causal_hash(sampled(lp['spike_states']))==causal_hash(sampled(rp['spike_states']))
    for k in ('event_generation','initial_commit_done','last_events','event_log','emitted_events_unrecorded','emitted_until'):
        discrete[k]=causal_hash(lp['controller'][k])==causal_hash(rp['controller'][k])
    coverage=dict(states=sum(r['field'].startswith('state__') for r in fields),
        SI_ports=sum(r['field'].startswith('port__') for r in fields))
    result=dict(passed=coverage==dict(states=120,SI_ports=15) and all(r['passed'] for r in fields) and all(discrete.values()),
        coverage=coverage,fields=fields,discrete_exact=discrete,maximum_norm=max(r['norm'] for r in fields),
        direct=read(directories['direct']/'RESULT.json'),compiled=read(directories['compiled']/'RESULT.json'),
        obsolete_manifest_failure_retained='qualification_v1.log',
        scope='current pre-registered real10.1ms g0; does not retroactively qualify old experiment')
    write(OUT/'TAU20_PREFIX_EQUIVALENCE.json',result)
    if not result['passed']:raise ContractError('New path prefix does not pass original numerical/causal gate')


def original_cache(spec,index):
    registry=read(ROOT/'reports/rgc_stage1_cpu_parallel_full_v1/CACHE_REGISTRY.json')
    entry=next(x for x in registry['caches'] if x['case_id']=='cell6_g'+str(index))
    data,manifest=read_cache(entry['directory'],spec,spec['cases'][index])
    return data,manifest,Path(entry['directory'])


def case_context(spec,index,tau):
    from tools.validate_ac_vs_native_support import load
    from retina_sim.group_time_storage import load_bundle
    from experiments.local_patch_backend_v1.backend import bind_cpu
    assembly,seed,parent=load(ROOT/spec['prepared_model'])
    if canonical_hash(seed)!=spec['source_seed_sha256']:
        raise ContractError('Original complete equilibrium IVP changed')
    case=spec['cases'][index]
    drive=load_bundle(ROOT/case['drive_bundle'],expected_kind='validation_evidence')['payload']
    if canonical_hash(drive)!=case['drive_sha256'] or drive['stimulus_sha256']!=case['stimulus_sha256']:
        raise ContractError('Original input source identity differs')
    overlay=EffectiveTimeAssembly(assembly,seed,tau)
    seed=overlay.seed
    seed['native_inputs']=deepcopy(drive['native_inputs'])
    seed['rod_inputs']=deepcopy(drive['rod_inputs'])
    selected=bind_cpu(overlay)
    protocol=deepcopy(parent['protocol'])
    protocol.update(start_seconds=drive['model_origin_s'],stop_seconds=drive['model_stop_s'],
        checkpoint_seconds=drive['model_stop_s'],input_frame_indices=sorted(drive['native_inputs']))
    return selected,seed,protocol,overlay.binding


def port_budget(arm,seed,spec):
    """Actual route + frozen state allocation, no free conductance unit floor."""
    from retina_sim.group_time_monitor import frozen_budget
    target=spec['target'];pop=target['population']
    row=spec['recording'][pop]['rows'][target['column']]
    routes=[r for r in arm.inner.routes if r.post==pop and r.pre==POPULATION]
    if len(routes)!=1 or float(np.asarray(arm.inner.parameters['route_gains'][routes[0].name]))!=1e-9:
        raise ContractError('Expected one actual ON diffuse BC1nS route')
    w=arm.inner.mean[routes[0].name].getrow(row).toarray().ravel()
    if abs(w.sum()-1)>1e-14 or np.any(w<0):
        raise ContractError('Existing actual mean-weight routing changed')
    rel=seed['states'][POPULATION,'HistoryBipolarTerminal','terminal_release']
    adap=seed['states'][POPULATION,'HistoryBipolarTerminal','A']
    b=frozen_budget(rel,'dimensionless')[2]+6*.75**5*frozen_budget(adap,'dimensionless')[2]
    bound=float(arm.inner.members[pop][row]*1e-9*(np.abs(w)@b))
    return bound,dict(route=routes[0].name,row=int(row),columns=np.flatnonzero(w).tolist(),
        weights=w[w!=0].tolist(),row_sum=float(w.sum()),members=int(arm.inner.members[pop][row]),
        total_conductance_budget_S=bound)


def capture_case(spec,index,tau,destination):
    from retina_sim.group_time_runner import GroupTimeArm
    from retina_sim.group_time_monitor import GroupTimeMonitor
    from retina_sim.group_time_storage import save_bundle,load_bundle
    from tools.validate_ac_vs_equilibrium_start import initial_match
    from tools.validate_ac_vs_timescale import calendar
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    identity=canonical_hash(dict(protocol=sha(OUT/'PRE_REGISTERED.json'),case=spec['cases'][index],
        tau_seconds=tau,execution='local_patch_cpu_typed_fp64_v1'))
    if (out/'RESULT.json').exists():
        result=read(out/'RESULT.json')
        if result['identity']!=identity or result['arrays_sha256']!=sha(out/'compact.npz'):
            raise ContractError('Completed new candidate identity changed')
        return result
    arm=None;started=time.perf_counter();cpu=time.process_time()
    with StageWorker(ROOT,out,OUT,18108) as worker:
        try:
            a,seed,protocol,binding=case_context(spec,index,tau)
            pointer=out/'latest_checkpoint.json'
            saved=None
            if pointer.exists():
                point=read(pointer)
                bundle=load_bundle(out/point['directory'],expected_kind='complete_model_checkpoint')
                if bundle['identity']!=dict(calibration_identity=identity):
                    raise ContractError('New candidate checkpoint identity differs')
                saved=bundle['payload']
                arm=GroupTimeArm.from_checkpoint(a,seed,protocol,saved,mode='reference',guard=worker.check)
                monitor=GroupTimeMonitor.from_payload(a,seed['states'],saved['monitor'],parameters=seed['inner_parameters'])
            else:
                arm=GroupTimeArm(a,seed,protocol,mode='reference',guard=worker.check)
                monitor=GroupTimeMonitor(a,seed['states'],parameters=seed['inner_parameters'])
            budget,route=port_budget(arm,seed,spec)
            count=71001
            arrays={pop+':'+field:np.zeros((count,len(sel['rows'])),dtype=np.float64)
                for pop,sel in spec['recording'].items() for field in ('V','gE','gI')}
            cols=np.asarray(route['columns'],dtype=np.int64)
            for name in ('activation','V','terminal_release','A'):
                arrays['BC:'+name]=np.zeros((count,len(cols)),dtype=np.float64)
            for name in ('V','release','x'):
                arrays['AII:'+name]=np.zeros((count,arm.inner.counts['AII_AC']),dtype=np.float64)
            arrays['BC:gap_current_A']=np.zeros((count,len(cols)),dtype=np.float64)
            recorded=0
            if saved is None:
                init,_=initial_match(arm,seed)
            else:
                rec=saved['recording'];recorded=rec['recorded'];init=rec['initial_match']
                for k,v in rec['arrays'].items():arrays[k][:recorded]=v
            if not init['passed']:
                raise ContractError('Full initial state/history mismatch before initial_commit')
            def observe(t,side):
                s=arm.snapshot(t,side,queue_node=True)
                monitor.observe(t,side,s['states'],s['rhs'],transitions=s['transitions'],
                    ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
                if monitor.domain_failures:
                    raise ContractError('Original physical/history/domain gate failed')
                return s
            def record(s,j):
                nonlocal recorded
                if recorded!=j:raise ContractError('Source row skipped/repeated')
                for pop,sel in spec['recording'].items():
                    for field,port in (('V','voltage'),('gE','gE'),('gI','gI')):
                        arrays[pop+':'+field][j]=s['ports'][pop,port][sel['rows']]
                for field in ('activation','V','terminal_release','A'):
                    arrays['BC:'+field][j]=s['states'][POPULATION,'HistoryBipolarTerminal',field][cols]
                for field in ('V','release','x'):
                    arrays['AII:'+field][j]=arm.inner._mem(s['states'],'AII_AC',field)
                arrays['BC:gap_current_A'][j]=s['ports'][POPULATION,'gap_current'][cols]
                recorded+=1
            def checkpoint(j):
                directory='checkpoint_'+str(j).zfill(7)
                payload=arm.checkpoint_payload(monitor=monitor.payload(),recording=dict(recorded=recorded,
                    initial_match=init,arrays={k:v[:recorded].copy() for k,v in arrays.items()}))
                save_bundle(out/directory,kind='complete_model_checkpoint',identity=dict(calibration_identity=identity),
                    payload=payload,qualification=dict(default_adopted=False,fit_eligible=False,FIT_READY=False))
                write(pointer,dict(directory=directory,source_seconds=j*.0001,identity=identity))
            write(out/'STARTED.json',dict(utc=utc(),pid=os.getpid(),index=index,tau_s=tau,identity=identity,
                binding=binding,route=route,source_seed_initial_match=init,assembly=a.artifact_sha256,
                full_inner_causal_dynamics=True,native_recomputed=True))
            print(json.dumps(dict(stage='CASE_STARTED',g=index,tau_ms=tau*1000,pid=os.getpid(),out=str(out))),flush=True)
            if not arm.initial_commit_done:
                s=observe(arm.start,'right');record(s,0)
                arm.initial_commit();observe(arm.start,'right');arm.ack(arm.start)
            for end,flags in calendar(protocol):
                if end<=arm.inner_time:continue
                begin=arm.inner_time
                arm.ensure_native(end,'left');arm.advance_inner(begin,end)
                before=observe(end,'left' if flags['frame'] else 'right')
                j=int(round((end-arm.start)/.0001))
                tick=flags['rod'] or end==arm.stop
                if tick and j==recorded:record(before,j)
                if flags['rod'] or flags['bc']:
                    arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],
                        rod_sample_time=max(arm.start,end-.0001+1e-13) if flags['rod'] else None)
                observe(end,'right');arm.ack(end)
                if tick and (j==500 or j%5000==0 or end==arm.stop):checkpoint(j)
                if tick and j%1000==0:
                    progress=dict(utc=utc(),pid=os.getpid(),group=index,tau_s=tau,source_seconds=end-arm.start,
                        recorded=recorded,total=count,wall_seconds=time.perf_counter()-started,
                        checkpoint=read(pointer) if pointer.exists() else None)
                    write(out/'progress.json',progress);write(OUT/'STATUS.json',dict(stage='ACTUAL_CASE_RUNNING',**progress))
                    print(json.dumps(progress),flush=True)
            if recorded!=count:
                raise ContractError('Incomplete original source grid')
            np.savez_compressed(out/'compact.npz',time_source_s=np.arange(count)*.0001,**arrays)
            result=dict(status='COMPLETE_NEW_TAU_COUPLED_FORWARD',identity=identity,index=index,tau_s=tau,
                source_seconds=arm.stop-arm.start,recorded=recorded,initial_match=init,binding=binding,route=route,
                port_budget_S=budget,arrays_sha256=sha(out/'compact.npz'),metrics=arm.metrics,
                raw_absolute_monitor=monitor.report(arm.stop-arm.start),domain_failures=monitor.domain_failures,
                elapsed_wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,
                peak_RSS_bytes=worker.memory.peak_rss_bytes,compiled_metrics=arm.compiled.metrics,
                native_recomputed=True,readout_fits=0,FIT_READY=False)
            write(out/'RESULT.json',result)
            return result
        except BaseException as exc:
            failures=out/'failures';failures.mkdir(exist_ok=True)
            write(failures/(str(time.time_ns())+'.json'),dict(utc=utc(),error=type(exc).__name__,reason=str(exc),
                traceback=traceback.format_exc(),source_seconds=None if arm is None else arm.inner_time-arm.start))
            raise
        finally:
            if arm is not None:arm.close()


def baseline_long_comparison(spec,result,directory):
    from retina_sim.group_time_monitor import frozen_budget
    from retina_sim.group_time_storage import load_bundle
    from tools.validate_ac_vs import causal_hash
    old,manifest,path=original_cache(spec,0)
    new=np.load(Path(directory)/'compact.npz',allow_pickle=False)
    rows=[]
    with old,new:
        for pop in spec['recording']:
            for field,unit in (('V','V'),('gE','S'),('gI','S')):
                k=pop+':'+field;x,y=old[k],new[k]
                allowance=frozen_budget(x[0],unit)[2]
                if pop==spec['target']['population'] and field=='gE':
                    allowance=np.minimum(allowance,result['port_budget_S'])
                norm=float(np.max(np.abs(x-y)/allowance))
                rows.append(dict(field=k,normalized_error=norm,passed=norm<=1,exact=bool(np.array_equal(x,y))))
    lpoint=read(path/'latest_checkpoint.json')
    rpoint=read(Path(directory)/'latest_checkpoint.json')
    lp=load_bundle(path/lpoint['directory'],expected_kind='complete_model_checkpoint')['payload']
    rp=load_bundle(Path(directory)/rpoint['directory'],expected_kind='complete_model_checkpoint')['payload']
    a,initial_seed,_,_=case_context(spec,0,.020)
    units={(b.population,b.mechanism,b.name):b.unit for b in (*a.state_blocks,*a.probability_blocks)}
    for k,x in lp['inner'].items():
        b=frozen_budget(initial_seed['states'][k],units[k])[2]
        norm=float(np.max(abs(x-rp['inner'][k])/b))
        rows.append(dict(field='final_inner:'+':'.join(k),normalized_error=norm,passed=norm<=1))
    discrete={k:causal_hash(lp[k])==causal_hash(rp[k]) for k in ('clocks','input_position','observation_delay')}
    def lineage_neutral(value):
        if isinstance(value,dict):
            return {k:lineage_neutral(v) for k,v in value.items() if k not in ('source_version','native_representation_id')}
        if isinstance(value,list):return [lineage_neutral(v) for v in value]
        return value
    discrete['readout_cursor_content']=causal_hash(lineage_neutral(lp['readout_cursor']))==causal_hash(lineage_neutral(rp['readout_cursor']))
    def sampled(v):
        return {p:{k:x for k,x in state.items() if k!='conditional_rate_Hz'} for p,state in v.items()}
    discrete['RNG_sampled_history']=causal_hash(sampled(lp['spike_states']))==causal_hash(sampled(rp['spike_states']))
    verdict=dict(passed=all(x['passed'] for x in rows) and all(discrete.values()),fields=rows,
        discrete_exact=discrete,maximum_norm=max(x['normalized_error'] for x in rows),
        original_complete_case=str(path),candidate=str(directory),same_original_input=True)
    write(OUT/'TAU20_FULL_G0_EQUIVALENCE.json',verdict)
    if not verdict['passed']:
        raise ContractError('Full baseline20 new execution path differs beyond original allocation')


def normalize(x,center=False):
    x=np.asarray(x,dtype=float)
    if center:x=x-np.mean(x)
    scale=float(np.sqrt(np.mean(x*x)))
    if not np.isfinite(scale) or scale<=0:raise ContractError('Unidentifiable zero/nonfinite signal scale')
    return x/scale,scale


def shape_loss(x,y):
    a,_=normalize(x,True);b,_=normalize(y,True)
    return float(np.mean((a-b)**2))


def loss_allowance(scale,budget):
    if scale<=budget:return None
    d=2*budget/(scale-budget)
    return 4*d+d*d


def metrics(spec,index,arrays,budget):
    pop=spec['target']['population'];col=spec['target']['column'];s=slice(10000,70000)
    ge=arrays[pop+':gE'][s,col];gi=arrays[pop+':gI'][s,col];v=arrays[pop+':V'][s,col]
    # Materialize only this approved group's E; never g4/g5 or historical tests.
    with np.load(TARGETS/'targets_native_100us.npz',allow_pickle=False) as target:
        repeats=target['g'+str(index)+'_E_raw_repeats'][:,s].copy()
    y=-repeats.mean(axis=0)
    z,sd=normalize(ge,True);zy,ysd=normalize(y,True)
    if sd<=budget:raise ContractError('E temporal SD below original propagated numerical resolvability')
    gnorm,grms=normalize(-ge);ynorm,yrms=normalize(-y)
    grids={}
    for n,label in ((1,'100us'),(10,'1ms'),(200,'20ms')):
        a=ge.reshape(-1,n).mean(axis=1);b=y.reshape(-1,n).mean(axis=1)
        grids[label]=dict(correlation=float(np.corrcoef(a,b)[0,1]),shape_MSE=shape_loss(a,b))
    # Positive-sequence integral autocorrelation time; describes dependence,
    # never turns60000 bins into independent biological replicates.
    autocorr=[]
    for rep in repeats:
        r=rep-rep.mean();fft=np.fft.rfft(r,n=131072)
        ac=np.fft.irfft(fft*np.conj(fft),n=131072)[:10001];ac/=ac[0]
        nonpositive=np.flatnonzero(ac[1:]<=0)
        last=int(nonpositive[0]+1) if len(nonpositive) else len(ac)
        autocorr.append(float(.0001*(1+2*ac[1:last].sum())))
    f=np.fft.rfftfreq(len(ge),.0001);mask=f<=60
    G=np.fft.rfft(ge-ge.mean());Y=np.fft.rfft(y-y.mean())
    from retina_sim.rgc_stage1_readout import free_mean,decode
    theta=np.asarray(read(ROOT/'reports/rgc_stage1_cpu_parallel_full_v1/fit/TRAINING_SELECTION.json')['theta'])
    volt=arrays[pop+':V'][::10,col][:-1]
    probability=free_mean(volt,decode(theta,spec['base']),spec['initial_probability'])
    with np.load(ROOT/'reports/rgc_stage1_complete_v1/actual_observations.npz',allow_pickle=False) as source:
        counts=source['g'+str(index)+'_counts'][:,1000:7000].copy()
    psth=counts.mean(axis=0).reshape(-1,20).mean(axis=1)*1000
    pred=(probability[1000:7000]*1000).reshape(-1,20).mean(axis=1)
    passive=spec['base'];n=spec['recording'][pop]['members'][col]
    tau=n*passive['C']/(n*passive['gL']+ge+gi)
    summary=dict(group=index,score_source_s=[1,7],primary_MSE=float(np.mean((z-zy)**2)),
        per_E_repeat_shape_MSE=[shape_loss(ge,-r) for r in repeats],
        grids=grids,old_signed_DC_RMS_MSE=float(np.mean((gnorm-ynorm)**2)),
        gE=dict(mean_S=float(ge.mean()),SD_S=sd,ptp_S=float(np.ptp(ge)),min_S=float(ge.min()),max_S=float(ge.max())),
        gI=dict(mean_S=float(gi.mean()),SD_S=float(gi.std()),min_S=float(gi.min())),
        V=dict(mean_V=float(v.mean()),SD_V=float(v.std()),ptp_V=float(np.ptp(v))),
        effective_RGC_tau_s=dict(mean=float(tau.mean()),min=float(tau.min()),max=float(tau.max())),
        E_repeat_autocorrelation_integral_s=autocorr,
        original_port_budget_S=budget,shape_numeric_allowance=loss_allowance(sd,budget),
        old_loss_numeric_allowance=loss_allowance(grms,budget),
        fixed_theta_free_PSTH=dict(correlation=float(np.corrcoef(psth,pred)[0,1]),
            MSE_Hz2=float(np.mean((psth-pred)**2)),predicted_SD_Hz=float(pred.std()),observed_SD_Hz=float(psth.std())),
        absolute_E_scale_identified=False,readout_refitted=False)
    spectral=dict(frequency_Hz=f[mask],model_gE_fft=G[mask],observed_negative_E_fft=Y[mask],
        phase_difference_rad=np.angle(G[mask]*np.conj(Y[mask])))
    return summary,spectral


def training_gate(base,candidate):
    details=[]
    improvements=[]
    for b,c in zip(base,candidate):
        improvement=b['primary_MSE']-c['primary_MSE']
        allowance=b['shape_numeric_allowance']+c['shape_numeric_allowance']
        repeat=[x-y for x,y in zip(b['per_E_repeat_shape_MSE'],c['per_E_repeat_shape_MSE'])]
        old_ok=c['old_signed_DC_RMS_MSE']<=b['old_signed_DC_RMS_MSE']+b['old_loss_numeric_allowance']+c['old_loss_numeric_allowance']
        details.append(dict(group=b['group'],improvement=improvement,numerical_allowance=allowance,
            each_actual_repeat_improvement=repeat,old_DC_loss_nondegrading=old_ok,
            passed=bool(improvement>allowance and min(repeat)>0 and old_ok)))
        improvements.append(improvement)
    sem=float(np.std(improvements,ddof=1)/np.sqrt(len(improvements))) if len(improvements)>1 else 0.
    return dict(passed=all(x['passed'] for x in details) and np.mean(improvements)>sem,
        per_group=details,mean_improvement=float(np.mean(improvements)),group_SEM=sem,
        independent_timebin_test_used=False)


def run():
    spec,protocol=verify()
    if not read(OUT/'TAU20_PREFIX_EQUIVALENCE.json')['passed']:
        raise ContractError('Required baseline20 compiled prefix equivalence missing')
    evaluations=OUT/'evaluations';evaluations.mkdir(exist_ok=True)
    baseline_dir=evaluations/'baseline20_g0'
    baseline_result=capture_case(spec,0,.020,baseline_dir)
    baseline_long_comparison(spec,baseline_result,baseline_dir)
    budget=baseline_result['port_budget_S'];baseline=[];spectra={}
    for index in range(3):
        data,manifest,path=original_cache(spec,index)
        with data:m,s=metrics(spec,index,data,budget)
        baseline.append(m)
        for k,v in s.items():spectra['g'+str(index)+'_'+k]=v
    write(OUT/'BASELINE20_TRAINING.json',dict(metrics=baseline,baseline_reused_after_equivalence=True))
    np.savez_compressed(OUT/'BASELINE20_SPECTRA.npz',**spectra)
    costs=[baseline_result['elapsed_wall_seconds']]
    write(OUT/'MEASURED_COST.json',dict(utc=utc(),one_complete_new_path_g0_wall_seconds=costs[0],
        upper_remaining_training_case_runs=21,conditional_serial_remaining_hours=21*costs[0]/3600,
        g3_extra_if_training_passes_hours=costs[0]/3600,
        native_recomputed_each_candidate=True,no_claim8_cache_reads=True,workers=1))
    history=[dict(evaluation=1,tau_s=.020,objective=float(np.mean([m['primary_MSE'] for m in baseline])),
        group_metrics=baseline,source='original complete matching baseline caches after equivalence')]
    write(OUT/'OBJECTIVE_TRACE.json',history)
    def objective(tau):
        if len(history)>=8:raise ContractError('Frozen8 evaluation limit')
        number=len(history)+1
        group_metrics=[]
        for index in range(3):
            directory=evaluations/('evaluation_'+str(number).zfill(2)+'_g'+str(index))
            result=capture_case(spec,index,float(tau),directory)
            with np.load(directory/'compact.npz',allow_pickle=False) as data:
                m,s=metrics(spec,index,data,result['port_budget_S'])
            group_metrics.append(m)
            np.savez_compressed(directory/'spectra.npz',**s)
        value=float(np.mean([m['primary_MSE'] for m in group_metrics]))
        history.append(dict(evaluation=number,tau_s=float(tau),objective=value,group_metrics=group_metrics))
        write(OUT/'OBJECTIVE_TRACE.json',history)
        print(json.dumps(dict(stage='OBJECTIVE_COMPLETE',evaluation=number,tau_ms=tau*1000,loss=value)),flush=True)
        return value
    from scipy.optimize import minimize_scalar
    optimum=minimize_scalar(objective,bounds=(.005,.030),method='bounded',options=dict(maxiter=7,xatol=.001))
    selected=min(history,key=lambda x:x['objective'])
    gate=training_gate(baseline,selected['group_metrics'])
    boundary=selected['tau_s']<=.006 or selected['tau_s']>=.029
    candidate=dict(utc=utc(),selected=selected,training_gate=gate,boundary=boundary,
        optimizer=dict(success=bool(optimum.success),message=str(optimum.message),nfev=int(optimum.nfev),x=float(optimum.x)),
        evaluations=len(history),frozen_before_g3=True)
    write(OUT/'CANDIDATE_FROZEN.json',candidate)
    if boundary or not gate['passed']:
        finish('STOP','Training gate failed or boundary/unresolved optimum; g3 not run',candidate)
        return
    if not optimum.success:
        finish('UNIDENTIFIABLE','Finite evaluation cap reached without frozen1ms optimization resolution; g3 not run',candidate)
        return
    index=3;data,manifest,path=original_cache(spec,index)
    with data:b,s=metrics(spec,index,data,budget)
    directory=evaluations/'g3_frozen_candidate'
    result=capture_case(spec,index,selected['tau_s'],directory)
    with np.load(directory/'compact.npz',allow_pickle=False) as data:c,s=metrics(spec,index,data,result['port_budget_S'])
    check=training_gate([b],[c]);write(OUT/'G3_CHECK.json',dict(baseline=b,candidate=c,gate=check,
        scope='previously explored training group held out of this estimate; not newtest'))
    np.savez_compressed(directory/'spectra.npz',**s)
    finish('CONTINUE' if check['passed'] else 'STOP','Frozen g3 check passed' if check['passed'] else 'Frozen g3 check failed; no retuning',candidate)


def finish(decision,reason,candidate=None):
    verify()
    final=dict(utc=utc(),decision=decision,reason=reason,candidate=candidate,default_adopted=False,
        FIT_READY=False,raw_absolute_false_preserved=True,readout_refits=0,
        other_groups_run=False,validation_used_to_select=False,historical_test_accessed=False,
        global_model_lease_absent=not (ROOT/'reports/ac_vs_redesign_v1/ACTIVE_WORKER.json').exists(),
        no_static15374gain_used=True)
    write(OUT/'FINAL_RESULT.json',final)
    report='# ON diffuse BC 条件有效时间校准\n\n结论：'+decision+'。\n\n'+reason+'\n\n'
    if candidate is not None:
        report+='候选 tau = '+str(candidate['selected']['tau_s']*1000)+' ms；总评价 '+str(candidate['evaluations'])+' 次（含20ms基线）。\n\n'
        report+='训练门结果与逐组/逐重复差值见 CANDIDATE_FROZEN.json；全部评价见 OBJECTIVE_TRACE.json。\n\n'
    report+='仅改变 ON diffuse BC 的条件有效 C，gL 固定。所有耦合状态随实际方程推进；原1nS兴奋系数、抑制、RGC膜及读出固定。\n\n'
    report+='这不是实测电容识别或全视网膜生理验证。原输入标定、绝对电流单位、外部背景、空间支持及旧 absolute monitor=false 限制保留；FIT_READY=false。没有重拟合读出，也未使用 g4/g5 或历史测试选参。\n'
    (OUT/'REPORT.md').write_text(report,encoding='utf8')
    print(json.dumps(dict(stage='FINAL',decision=decision,reason=reason,report=str(OUT/'REPORT.md'))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','qualify','run'])
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='qualify':qualify()
    else:run()


if __name__=='__main__':
    main()
