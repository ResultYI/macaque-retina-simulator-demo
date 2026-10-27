"""Frozen native060/inner030 process restore and actual native half-step check."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import argparse
import json
import shutil
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import retina_sim.native_multirate as native_module
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.group_time_storage import load_bundle,save_bundle
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.group_time_monitor import GroupTimeMonitor,frozen_budget
from retina_sim.ac_vs_monitor import DynamicDefectMonitor
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.background import NUMERICAL_UNIT_SCALES
from retina_sim.rgc_expectation import STATES
from retina_sim.resident_derived_ports import compare_cell
from retina_sim.spatial_group_model import RGC
from tools.validate_ac_vs import Worker,QUALIFICATION,write,dynamic_observe,causal_hash
from tools.validate_ac_vs_equilibrium_start import verify_sources,initial_match
from tools.validate_ac_vs_timescale import calendar,CoreRecorder,sparse_states
from tools.validate_ac_vs_native_support import load as load_candidate,read,sha

OUT=ROOT/'reports/ac_vs_native_temporal_v1';REG=OUT/'pre_registration_v1.json'


@contextmanager
def native_cap(value):
    if native_module.MAX_STEP!=.0001 or value not in (.0001,.00005):raise ContractError('Explicit original/half native cap required; no nested override')
    original=native_module.MAX_STEP;native_module.MAX_STEP=value
    try:yield
    finally:native_module.MAX_STEP=original


def baseline():
    reg=read(REG)
    for path,digest in reg['baseline_pins'].items():
        if sha(ROOT/path)!=digest:raise ContractError('Verified baseline bytes changed: '+path)
    a,seed,identity=load_candidate(ROOT/reg['candidate_preparation'])
    b=ROOT/reg['baseline_directory']
    if read(b/'model_identity.json')!=identity or not read(b/'result.json')['passed']:raise ContractError('Frozen candidate baseline identity changed')
    cp=load_bundle(b/'final_checkpoint',expected_kind='complete_model_checkpoint',expected_identity=identity)['payload']
    if causal_hash(cp)!=read(b/'result.json')['final_causal_hash']:raise ContractError('Baseline causal checkpoint changed')
    return a,seed,identity,cp


def prepare(output):
    with Worker(output) as worker:
        a,seed,identity,cp=baseline();reg=read(REG)
        memory=MemoryGuard(16*1024**3,4*1024**3);healthy=memory.check();disk=shutil.disk_usage(ROOT)
        old=read(ROOT/reg['baseline_directory']/'resource_receipt.json');projected=old['peak_sampled_RSS_bytes']+512*1024**2
        pre=dict(passed=bool(healthy and memory.last['available_physical_bytes']>4*1024**3+projected and disk.free>512*1024**2),
            actual_memory=memory.last,disk_free_bytes=disk.free,projected_peak_RSS_bytes=projected,projected_disk_bytes=256*1024**2,
            producer_wall_projection_seconds=old['elapsed_seconds']*.5+5,restore_wall_projection_seconds=old['elapsed_seconds']*.5+5,
            native_half_wall_projection_seconds=old['elapsed_seconds']+read(ROOT/reg['baseline_directory']/'result.json')['metrics']['native_wall_seconds'],
            resources=read(ROOT/'configurations/ac_vs_acceptance_v1.json')['resources'])
        write(output/'MEMORY_PREFLIGHT.json',pre)
        if not pre['passed']:raise ContractError('Native temporal resource preflight failed before model construction')
        with native_cap(.0001):
            arm=GroupTimeArm(a,seed,identity['protocol'],mode='reference',guard=worker.check)
            try:
                match,_=initial_match(arm,seed);s=arm.initial_snapshot()
                if arm.identity!=cp['controller']['identity'] or arm.native_backend.reference_identity!=cp['native']['point']['backend_id']:raise ContractError('Original native/controller numerical source changed')
                ports={n:np.asarray(s['ports'][tuple(f['key'])])[f['rows']] for n,f in identity['selection']['fields'].items() if f['category']=='ports'}
                np.savez_compressed(output/'initial_core_ports.npz',**ports)
            finally:arm.close()
        budgets={':'.join(k):frozen_budget(v,next(b.unit for b in (*a.state_blocks,*a.probability_blocks,*a.auxiliary_z1_blocks) if (b.population,b.mechanism,b.name)==k))[2] for k,v in seed['states'].items()}
        np.savez_compressed(output/'state_budgets.npz',**budgets)
        pins=dict(identity['source_pins']);pins.update({str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [Path(__file__),REG,ROOT/'retina_sim/native_multirate.py',ROOT/'tools/validate_ac_vs_timescale.py']})
        result=dict(status='FROZEN_READY_RESTORE_AND_NATIVE_HALF',passed=True,registration=reg,registration_sha256=sha(REG),source_pins=pins,
            baseline_model_identity_sha256=canonical_hash(identity),artifact=a.artifact_sha256,seed_sha256=canonical_hash(seed),
            baseline_controller_identity=cp['controller']['identity'],baseline_native_backend_identity=cp['native']['point']['backend_id'],
            initial_precommit_match=match,state_budgets_sha256=sha(output/'state_budgets.npz'),initial_ports_sha256=sha(output/'initial_core_ports.npz'),
            state_scalar_count=sum(len(v) for v in seed['states'].values()),core_fields=len(identity['selection']['fields']),
            whole_retina_background_qualified=False,FIT_READY=False)
        write(output/'preparation.json',result);return result


def prepared(directory):
    p=Path(directory);raw=read(p/'preparation.json');verify_sources(raw['source_pins'])
    for path,key in [('state_budgets.npz','state_budgets_sha256'),('initial_core_ports.npz','initial_ports_sha256')]:
        if sha(p/path)!=raw[key]:raise ContractError('Frozen numerical scales changed')
    a,seed,identity,cp=baseline()
    if canonical_hash(identity)!=raw['baseline_model_identity_sha256'] or canonical_hash(seed)!=raw['seed_sha256']:raise ContractError('Frozen baseline/source changed')
    return a,seed,identity,raw


def run(directory,phase,output,resume=None):
    a,seed,identity,prep=prepared(directory);protocol=identity['protocol'];cap=.00005 if phase=='native_half' else .0001
    numerical_identity=identity if cap==.0001 else dict(baseline_identity=identity,native_step_cap_seconds=cap,registration_sha256=prep['registration_sha256'])
    with Worker(output) as worker,native_cap(cap):
        if phase=='restore':
            previous=load_bundle(Path(resume)/'checkpoint',expected_kind='complete_model_checkpoint',expected_identity=identity)['payload']
            arm=GroupTimeArm.from_checkpoint(a,seed,protocol,previous,mode='reference',guard=worker.check)
            monitor=GroupTimeMonitor.from_payload(a,seed['states'],previous['monitor'],parameters=seed['inner_parameters']);recorder=CoreRecorder(identity['selection'],previous['recording'])
            restored=arm.checkpoint_payload(monitor=monitor.payload(),recording=recorder.cursor)
            exact={k:causal_hash(v)==causal_hash(restored[k]) for k,v in previous.items()}
            write(output/'restore_entry_match.json',dict(passed=all(exact.values()),causal_fields=exact,expected_hash=causal_hash(previous),actual_hash=causal_hash(restored)))
            if not all(exact.values()):arm.close();raise ContractError('Complete restore causal entry differs before advancing')
            init=read(Path(resume)/'initial_seed_match.json')
        else:
            arm=GroupTimeArm(a,seed,protocol,mode='reference',guard=worker.check);monitor=GroupTimeMonitor(a,seed['states'],parameters=seed['inner_parameters']);recorder=CoreRecorder(identity['selection'])
        dynamic=DynamicDefectMonitor(a,seed['states']);edges=[];begin_segment=arm.inner_time;counter_start=len(arm.metrics['native_counters']);widths=[];native_ids=[]
        write(output/'model_identity.json',numerical_identity)
        write(output/'execution_identity.json',dict(preparation=prep,phase=phase,native_step_cap_seconds=cap,inner_max_step_seconds=a.max_step,protocol_exact_baseline=True,
            native_backend_identity=arm.native_backend.reference_identity,baseline_backend_identity=prep['baseline_native_backend_identity'],
            reference_nominal_method_label_contains_100us=True,actual_hashed_runtime_native_max_step_seconds=cap))
        def observe(t,side,record=False):
            s=arm.snapshot(t,side,queue_node=True)
            monitor.observe(t,side,s['states'],s['rhs'],transitions=s['transitions'],ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
            if monitor.domain_failures:raise ContractError('Temporal physical/history/conservation gate failed')
            if record:recorder.record(s)
            return s
        try:
            if phase!='restore':
                init,_=initial_match(arm,seed);write(output/'initial_seed_match.json',init)
                observe(arm.start,'right');arm.initial_commit();observe(arm.start,'right',True);arm.ack(arm.start)
            for end,flags in calendar(protocol):
                if end<=arm.inner_time:continue
                begin=arm.inner_time;arm.ensure_native(end,'left');interval=arm._interval_at(end,'left')
                if interval is not None and interval.representation_id not in native_ids:
                    native_ids.append(interval.representation_id)
                    widths.extend(s.t1-s.t0 for s in interval.curves['X'].segments)
                advance=arm.advance_inner(begin,end)
                for curve in advance.dense_segments:
                    mid=curve.t0+(curve.t1-curve.t0)/2;witness=dynamic_observe(arm,advance,mid,dynamic,'left' if mid==advance.t1 else 'right')
                    if max(r['maximum_norm'] for r in dynamic.rows.values())>1:
                        save_bundle(output/'failed_dynamic_witness',kind='validation_evidence',identity=numerical_identity,payload=dict(states=witness['states'],rhs=witness['rhs'],time_seconds=mid,dynamic=dynamic.report()),qualification=QUALIFICATION)
                        raise ContractError('Original dynamic numerical defect budget failed')
                before=observe(end,'left' if flags['frame'] else 'right',flags['frame'])
                if flags['rod'] or flags['bc']:arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],rod_sample_time=max(arm.start,end-.0001+1e-13) if flags['rod'] else None)
                last=observe(end,'right',flags['record'] or flags['frame'])
                if flags['frame']:
                    keys=[k for k in last['states'] if k[1] not in ('Chen2024PrimateRodCurrent','RGC_expected_history') and not(k[1]=='HistoryBipolarTerminal' and(k[2].startswith('history_') or k[2]=='activation'))]
                    edges.append(dict(time_seconds=end,continuous_state_max_jump=max((float(np.max(abs(last['states'][k]-before['states'][k]),initial=0.)) for k in keys),default=0.)))
                arm.ack(end)
                if end==protocol['checkpoint_seconds']:
                    sparse_states(output/'full_state_mid.npz',last)
                    if phase=='checkpoint':
                        recorder.save(output/'core_prefix.npz')
                        save_bundle(output/'checkpoint',kind='complete_model_checkpoint',identity=identity,payload=arm.checkpoint_payload(monitor=monitor.payload(),recording=recorder.cursor),qualification=QUALIFICATION)
                        break
            final=arm.checkpoint_payload(monitor=monitor.payload(),recording=recorder.cursor)
            if phase!='checkpoint':
                sparse_states(output/'full_state_final.npz',last);recorder.save(output/('core_suffix.npz' if phase=='restore' else 'core_traces.npz'))
                save_bundle(output/'final_checkpoint',kind='complete_model_checkpoint',identity=numerical_identity,payload=final,qualification=QUALIFICATION)
            counters=arm.metrics['native_counters'][counter_start:];cap_ok=all(width<=cap+16*np.spacing(arm.stop) for width in widths)
            expected=dict(rod=50430,bc=5043,rgc=5043) if phase=='checkpoint' else dict(rod=50930,bc=5093,rgc=5093)
            clock_ok=final['clocks']['indices']==expected
            result=dict(status='NATIVE_TEMPORAL_SEGMENT_COMPLETE',passed=dynamic.report()['passed'] and cap_ok and clock_ok and all(e['continuous_state_max_jump']==0 for e in edges),phase=phase,
                start_seconds=begin_segment,end_seconds=arm.inner_time,actual_simulated_seconds=arm.inner_time-begin_segment,initial_match=init,
                DYNAMIC_NUMERICAL_DEFECT=dynamic.report(),PHYSICAL_DOMAIN_AND_CONSERVATION=dict(passed=not monitor.domain_failures,failures=monitor.domain_failures),
                INPUT_EVENT_AND_HISTORY=dict(passed=clock_ok and all(e['continuous_state_max_jump']==0 for e in edges),edges=edges,clocks=final['clocks']),
                raw_absolute_equilibrium_monitor=monitor.report(.1),metrics=arm.metrics,final_causal_hash=causal_hash(final),core_cursor=recorder.cursor,
                native_stepping=dict(configured_max_step_seconds=cap,inner_max_step_seconds=a.max_step,native_windows_this_segment=len(counters),native_accepted_steps_this_segment=sum(c['steps'] for c in counters),
                    native_nfev_this_segment=sum(c['nfev'] for c in counters),observed_native_dense_step_count=len(widths),observed_native_max_width_seconds=max(widths,default=0.),observed_native_min_width_seconds=min(widths,default=0.),cap_respected=cap_ok,
                    native_backend_identity=arm.native_backend.reference_identity),FIT_READY=False,default_adopted=False)
            write(output/'result.json',result);return result
        except Exception as exc:
            if recorder.stamps:recorder.save(output/'stopped_core_traces.npz')
            write(output/'failure_witness.json',dict(error=type(exc).__name__,message=str(exc),time_seconds=arm.inner_time,dynamic=dynamic.report(),metrics=arm.metrics));raise
        finally:arm.close()


def restore_compare(directory,producer,restored,output):
    _,_,identity,prep=prepared(directory);reg=prep['registration'];base=ROOT/reg['baseline_directory'];r=Path(restored);p=Path(producer)
    if read(r/'model_identity.json')!=identity or read(p/'model_identity.json')!=identity:raise ContractError('Restore arm identity changed')
    full=load_bundle(base/'final_checkpoint',expected_kind='complete_model_checkpoint',expected_identity=identity)['payload']
    continued=load_bundle(r/'final_checkpoint',expected_kind='complete_model_checkpoint',expected_identity=identity)['payload']
    fields={k:causal_hash(v)==causal_hash(continued[k]) for k,v in full.items()}
    with np.load(base/'core_traces.npz') as a,np.load(p/'core_prefix.npz') as prefix,np.load(r/'core_suffix.npz') as suffix:
        if set(a.files)!=set(prefix.files) or set(a.files)!=set(suffix.files):raise ContractError('Restore core trace registry changed')
        traces={k:np.array_equal(a[k],np.concatenate([prefix[k],suffix[k]],axis=0)) for k in a.files}
        lengths=dict(full=len(a['time_seconds']),prefix=len(prefix['time_seconds']),suffix=len(suffix['time_seconds']))
    passed=all(fields.values()) and all(traces.values()) and read(r/'restore_entry_match.json')['passed']
    result=dict(status='PASS_NEW_SELECTION_PROCESS_RESTORE' if passed else 'FAIL_NEW_SELECTION_PROCESS_RESTORE',passed=passed,causal_fields=fields,core_trace_fields=traces,
        baseline_causal_hash=causal_hash(full),restored_causal_hash=causal_hash(continued),records=lengths,
        baseline_trace_reused=True,new_checkpoint_producer=True,checkpoint_seconds=5.043,metrics_only_excluded_from_causal_hash=True,FIT_READY=False)
    write(output/'result.json',result);return result


def numerical_compare(directory,refined,output):
    a,seed,identity,prep=prepared(directory);base=ROOT/prep['registration']['baseline_directory'];r=Path(refined);rows=[];full=[]
    with np.load(Path(directory)/'state_budgets.npz') as z:budgets={k:z[k] for k in z.files}
    with np.load(base/'core_traces.npz') as x,np.load(r/'core_traces.npz') as y,np.load(Path(directory)/'initial_core_ports.npz') as initial:
        for k in ['time_seconds','side','generation']:np.testing.assert_array_equal(x[k],y[k])
        for name,f in identity['selection']['fields'].items():
            av,bv=x[name],y[name];selected=f['rows'];text=':'.join(f['key']);exact=None
            if f['category']=='states':bound=budgets[text][selected][None,:]
            elif f['unit'] in NUMERICAL_UNIT_SCALES:bound=frozen_budget(initial[name],f['unit'])[2][None,:]
            elif f['unit']=='Hz' and f['key'][0] in RGC:
                pop=f['key'][0];p=seed['inner_parameters']['populations'][pop]
                va,vb=x['states__'+pop+':RGC_spike_history:V'],y['states__'+pop+':RGC_spike_history:V']
                kernel=np.asarray(p['spike_history_kernel']);ha=p['alpha']*np.logaddexp(0.,(va[:,:,None]+(STATES@kernel)[None,None,:]-p['mu'])/p['beta'])
                hb=p['alpha']*np.logaddexp(0.,(vb[:,:,None]+(STATES@kernel)[None,None,:]-p['mu'])/p['beta'])
                piB=np.stack([budgets[pop+':RGC_expected_history:history_probability_'+str(i)][selected] for i in range(8)],axis=-1)
                rates=np.maximum(ha,hb) if f['key'][1]=='expected_hazard_Hz' else np.maximum(-np.expm1(-.001*ha),-np.expm1(-.001*hb))/.001
                bound=p['alpha']/p['beta']*budgets[pop+':RGC_spike_history:V'][selected][None,:]*(1+1e-12)+np.sum(piB[None,:,:]*rates,axis=-1)+8192*np.finfo(float).eps*np.maximum(np.maximum(abs(av),abs(bv)),1.)
            elif f['unit']=='events/s':
                pop=f['key'][0];p=seed['inner_parameters']['populations'][pop]
                va,vb=x['states__'+pop+':ReceptorConductanceMembrane:V'],y['states__'+pop+':ReceptorConductanceMembrane:V']
                xa,xb=x['states__'+pop+':ReceptorConductanceMembrane:x'],y['states__'+pop+':ReceptorConductanceMembrane:x']
                Bv=budgets[pop+':ReceptorConductanceMembrane:V'][selected];Bx=budgets[pop+':ReceptorConductanceMembrane:x'][selected]
                bound=np.empty_like(av);exact=True
                for i in range(len(av)):
                    for j in range(len(selected)):
                        result=compare_cell(pool=float(p['Npool']),U=float(p['U']),rmax=float(p['rmax']),r_half=float(p['r_half']),r_slope=float(p['r_slope']),
                            V_reference=float(va[i,j]),V_candidate=float(vb[i,j]),x_reference=float(xa[i,j]),x_candidate=float(xb[i,j]),rate_reference=float(av[i,j]),rate_candidate=float(bv[i,j]),B_V=float(Bv[j]),B_x=float(Bx[j]))
                        bound[i,j]=int(result['allowance_exact'][0])/int(result['allowance_exact'][1]);exact=exact and result['passed']
            else:raise ContractError('Original numerical output budget not registered: '+name+' '+f['unit'])
            norm=float(np.max(abs(av-bv)/bound,initial=0.));rows.append(dict(field=name,unit=f['unit'],maximum_norm=norm,max_abs=float(np.max(abs(av-bv),initial=0.)),passed=norm<=1 if exact is None else exact))
    for suffix in ['mid','final']:
        with np.load(base/('full_state_'+suffix+'.npz')) as x,np.load(r/('full_state_'+suffix+'.npz')) as y:
            for key,budget in budgets.items():
                norm=float(np.max(abs(x[key]-y[key])/budget,initial=0.));full.append(dict(point=suffix,state=key,maximum_norm=norm,max_abs=float(np.max(abs(x[key]-y[key]),initial=0.)),passed=norm<=1))
    old=read(base/'result.json');new=read(r/'result.json');native=new['native_stepping']
    actual_refined=native['native_accepted_steps_this_segment']>sum(c['steps'] for c in old['metrics']['native_counters']) and native['native_windows_this_segment']==old['metrics']['native_advances'] and native['configured_max_step_seconds']==.00005 and native['inner_max_step_seconds']==.0001 and native['cap_respected']
    passed=all(row['passed'] for row in rows+full) and actual_refined and new['passed'];worst=max(rows+full,key=lambda row:row['maximum_norm'])
    result=dict(status='PASS_INDEPENDENT_NATIVE_HALF_STEP' if passed else 'FAIL_INDEPENDENT_NATIVE_HALF_STEP',passed=passed,maximum_norm=worst['maximum_norm'],worst=worst,
        core_fields=rows,sparse_full_state_checks=full,actual_native_refinement_proved=actual_refined,baseline_native_accepted_steps=sum(c['steps'] for c in old['metrics']['native_counters']),refined_native_stepping=native,
        budget='Original frozen_budget and original Hz/events-s propagation, unchanged; no spatial budget substitution',baseline_reused=True,inner_setting_unchanged=True,observation_sampling_unchanged=True,FIT_READY=False)
    write(output/'result.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','restore-compare','numerical-compare']);p.add_argument('--output',required=True);p.add_argument('--prepared')
    p.add_argument('--phase',choices=['checkpoint','restore','native_half']);p.add_argument('--resume');p.add_argument('--producer');p.add_argument('--restored');p.add_argument('--refined');args=p.parse_args();output=Path(args.output).resolve()
    if not output.is_relative_to(OUT) or output.exists():raise ContractError('Fresh independent native temporal evidence directory required')
    output.mkdir(parents=True)
    try:
        if args.mode=='prepare':result=prepare(output)
        elif args.mode=='run':result=run(args.prepared,args.phase,output,args.resume)
        elif args.mode=='restore-compare':result=restore_compare(args.prepared,args.producer,args.restored,output)
        else:result=numerical_compare(args.prepared,args.refined,output)
        print(json.dumps(dict(status=result['status'],passed=result['passed'],output=str(output))))
    except Exception as exc:write(output/'failure.json',dict(error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
