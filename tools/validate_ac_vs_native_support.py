"""Explicit native0.6/inner0.3 graph candidate; one frozen100ms arm."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import shutil
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.ac_vs_model import ACVSAssembly,load_config
from retina_sim.local_patch_graph import LocalPatchAssembly,_verified_base
from retina_sim.local_patch_boundary import BoundedInnerDynamics,BoundedNativePhysics
from retina_sim.group_time_storage import load_bundle,save_bundle
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.group_time_monitor import GroupTimeMonitor
from retina_sim.ac_vs_monitor import DynamicDefectMonitor
from retina_sim.background_checkpoint import MemoryGuard
from tools.validate_ac_vs import Worker,QUALIFICATION,write,dynamic_observe,causal_hash
from tools.validate_ac_vs_equilibrium_start import verify_sources,initial_match
from tools.validate_ac_vs_timescale import calendar,CoreRecorder,sparse_states
from tools.validate_ac_vs_spatial_common import read,sha,assembly,mappings,project_seed,common_boundary,point_check,by_ids

OUT=ROOT/'reports/ac_vs_native_support_v1';REG=OUT/'pre_registration_v1.json'
CONFIG=ROOT/'configurations/local_patch_v2/native_060_inner_030_ac_vs_v1.json'
SCHEMA='ac-vs-native060-inner030-candidate-1'


def checked(binding):
    p=(ROOT/binding['path']).resolve()
    if not p.is_relative_to(ROOT) or sha(p)!=binding['sha256']:raise ContractError('Frozen file changed: '+binding['path'])
    return p


def merge_selection(small,large):
    if set(small)!=set(large):raise ContractError('Different original selection registries')
    result={k:np.array(v,copy=True) for k,v in small.items()}
    for pop in ['cones','HC']:result['population__'+pop]=np.array(large['population__'+pop],copy=True)
    for name,value in result.items():
        if value.dtype!=np.int64 or value.ndim!=1 or np.any(np.diff(value)<=0):raise ContractError('Unique sorted parent-row IDs required: '+name)
        if name.startswith('core__') and not np.isin(value,result['population__'+name[6:]]).all():raise ContractError('Core escaped retained native/inner support')
        value.flags.writeable=False
    return result


def candidate(directory):
    config=read(CONFIG);baseconfig=read(ROOT/config['base_patch_config']);base=_verified_base(ROOT,baseconfig)
    selections=[]
    for kind in ['inner_selection','native_selection']:
        with np.load(checked(config[kind]),allow_pickle=False) as z:selections.append({k:z[k] for k in z.files})
    selected=merge_selection(*selections)
    rows={k[12:]:v for k,v in selected.items() if k.startswith('population__')}
    contacts={k[9:]:v for k,v in selected.items() if k.startswith('contact__')}
    core={k[6:]:v for k,v in selected.items() if k.startswith('core__')}
    descriptor=dict(schema=SCHEMA,configuration=config,parent_artifact=base.artifact_sha256,
        selection_sha256=canonical_hash(selected),implementation_sha256=sha(Path(__file__)),weights_renormalized=False)
    artifact=canonical_hash(descriptor);directory=Path(directory)
    target=read(ROOT/'reports/ac_vs_redesign_v1/functional_static_halo_030_v1/selection_identity.json')['target_parent_row']
    if directory.exists():
        if read(directory/'selection_identity.json')!=descriptor:raise ContractError('Candidate static selection identity changed')
        with np.load(directory/'selection.npz') as z:
            if set(z.files)!=set(selected) or any(not np.array_equal(z[k],v) for k,v in selected.items()):raise ContractError('Candidate selection bytes changed')
    else:
        directory.mkdir(parents=True);write(directory/'selection_identity.json',descriptor);np.savez_compressed(directory/'selection.npz',**selected)
    baseconfig['native_support_override']=deepcopy(config)
    baseconfig['position_provenance']+=' Explicit candidate: native Cone–HC retains existing0.6 sets, all inner populations/contacts retain0.3; not original0.3 qualification.'
    patch=object.__new__(LocalPatchAssembly);patch._assemble(ROOT,directory,base,baseconfig,rows,contacts,core,target,artifact,descriptor)
    patch.runtime_factories=dict(inner=BoundedInnerDynamics,native=BoundedNativePhysics)
    return ACVSAssembly(patch,load_config(ROOT/config['model_config']),max_step=.0001)


def reference():
    reg=read(REG)
    for p,digest in reg['reference_evidence_pins'].items():checked(dict(path=p,sha256=digest))
    checked(reg['original_failure'])
    identity=read(ROOT/reg['reference_directory']/'model_identity.json');verify_sources(identity['common']['sources'])
    evidence=read(ROOT/reg['reference_directory']/'result.json')
    if not evidence['passed'] or not evidence['saved_common_060_background_used'] or evidence['old_030_run_reused']:raise ContractError('Reference dynamic qualification changed')
    bundle=load_bundle(ROOT/reg['reference_preparation']/'seed_060',expected_kind='source_construction_seed')
    if bundle['identity']!=identity or canonical_hash(bundle['payload'])!=identity['seed_sha256']:raise ContractError('Reference full source/input identity changed')
    boundary=load_bundle(ROOT/reg['reference_preparation']/'boundary_060',expected_kind='validation_evidence',expected_identity=identity)['payload']
    if boundary['identity']!=bundle['payload']['patch_boundary']['identity']:raise ContractError('Reference native exterior identity differs')
    return bundle['payload'],identity,boundary


def selection_invariants(a,small,large):
    rows={};contacts={}
    for pop,p in a.populations.items():
        expected=large if pop in ['cones','HC'] else small
        if not np.array_equal(p.group_ids,expected.populations[pop].group_ids):raise ContractError('Unrequested domain change: '+pop)
        if not np.array_equal(p.group_ids[a.core_population_rows[pop]],small.populations[pop].group_ids[small.core_population_rows[pop]]):raise ContractError('Core identity changed: '+pop)
        rows[pop]=dict(candidate=p.group_count,original_030=small.populations[pop].group_count,reference_060=large.populations[pop].group_count)
    for component,value in a.parent_contact_rows.items():
        if not np.array_equal(value,small.parent_contact_rows[component]):raise ContractError('Inner physical contact domain changed: '+component)
        contacts[component]=len(value)
    for path in ['routes/cones_to_HC/mean_weights.npz','routes/HC_to_cones/mean_weights.npz']:
        difference=a.matrix(path)-large.matrix(path);difference.eliminate_zeros()
        if difference.nnz:raise ContractError('Native operator not exact0.6: '+path)
    return dict(populations=rows,contact_counts=contacts,native_operators_exact_060=True,inner_domains_exact_030=True)


def preflight(output):
    source,ref,boundary=reference();a=candidate(OUT/'static_candidate_v1');small,large=assembly('030'),assembly('060')
    checks=selection_invariants(a,small,large);maps=mappings(a,large)
    counts={label:sum(b.count for b in (*m.state_blocks,*m.probability_blocks,*m.auxiliary_z1_blocks)) for label,m in [('candidate',a),('original_030',small),('reference_060',large)]}
    memory=MemoryGuard(16*1024**3,4*1024**3);healthy=memory.check();disk=shutil.disk_usage(ROOT)
    prior=read(ROOT/read(REG)['reference_directory']/'resource_receipt.json');projection=int(prior['peak_sampled_RSS_bytes']+512*1024**2)
    feasible=healthy and memory.last['available_physical_bytes']>4*1024**3+projection and disk.free>512*1024**2
    pins=dict(ref['common']['sources']);pins.update({str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [CONFIG,REG,Path(__file__)]})
    result=dict(status='DIRECTED_NATIVE_SUPPORT_PREFLIGHT',passed=bool(feasible),registration_sha256=sha(REG),candidate_artifact=a.artifact_sha256,source_pins=pins,
        state_scalar_counts=counts,selection_checks=checks,map_identity=maps['identity'],actual_memory=memory.last,disk_free_bytes=disk.free,
        projected_peak_RSS_bytes=projection,projected_new_disk_bytes=256*1024**2,candidate_wall_projection_seconds=prior['elapsed_seconds'],
        reference_model_identity_sha256=canonical_hash(ref),reference_trace_reuse_verified=True,new_equilibrium_solves=0,new_dynamic_arms=1,
        resources=read(ROOT/'configurations/ac_vs_acceptance_v1.json')['resources'])
    write(output/'preflight.json',result)
    if not feasible:raise ContractError('Actual candidate memory/disk preflight failed')
    return result


def prepare(preflight_dir,output):
    before=read(Path(preflight_dir)/'preflight.json');verify_sources(before['source_pins'])
    reg=read(REG)
    if reg['spatial_budget']!=read(ROOT/'configurations/ac_vs_acceptance_v1.json')['spatial_budget']:raise ContractError('Original spatial budget changed')
    with Worker(output) as worker:
        source,ref,boundary=reference();a=candidate(OUT/'static_candidate_v1');large=assembly('060');large.boundary_runtime=boundary
        if a.artifact_sha256!=before['candidate_artifact']:raise ContractError('Post-preflight selection changed')
        maps=mappings(a,large);projected=project_seed(source,a,large,maps)
        # All reference native rows are retained; no native annulus is omitted.
        if len(maps['populations']['cones'])!=large.populations['cones'].group_count:raise ContractError('Native cone support is not full reference0.6')
        native=BoundedNativePhysics(large)
        y,anchor=native.construction_point(SimpleNamespace(states=source['states'],time_seconds=source['time_seconds']),source['input_reference']['native_background_td'],5.,748)
        release=native.evaluate(y,source['time_seconds'],anchor).outputs['release']
        products,descriptor=common_boundary(a,large,source,maps,release)
        for name in ['A_Vbar','A_release','B_Vh']:
            if not np.array_equal(products[name],boundary[name]):raise ContractError('Native boundary product changed: '+name)
        descriptor.pop('identity');descriptor.update(candidate_selection='native060_inner030',native_annulus_omitted_rows=0)
        products['identity']=canonical_hash(descriptor);descriptor['identity']=products['identity']
        a.boundary_runtime=products;projected['patch_boundary']=descriptor
        projected['ac_vs_lineage'].update(candidate_schema=SCHEMA,native060_inner030=True,new_background_solved=False,reference_common_background_reused=True)
        protocol=deepcopy(ref['common']['protocol'])
        check,selections=point_check(a,large,projected,source,maps,protocol,worker.check)
        write(output/'COMMON_INITIAL_AND_BOUNDARY_CHECK.json',check)
        if not check['passed']:raise ContractError('Candidate common equilibrium/initial state algebra failed before dynamics')
        for category,pop in [('native_inputs','cones'),('rod_inputs','rods')]:
            if any(not np.array_equal(value,source[category][frame][maps['populations'][pop]]) for frame,value in projected[category].items()):raise ContractError('Global stimulus projection changed')
        identity=dict(schema=SCHEMA,artifact=a.artifact_sha256,seed_sha256=canonical_hash(projected),source_pins=before['source_pins'],registration=reg,
            registration_sha256=sha(REG),candidate_config=read(CONFIG),projection_map_identity=maps['identity'],selection=selections['030'],protocol=protocol,
            reference_model_identity_sha256=canonical_hash(ref),reference_trace_reused=True,reference_directory=reg['reference_directory'],reference_source_seed_sha256=ref['seed_sha256'],
            old_small_trace_reused=False,initialization='Exact projection of saved common0.6 equilibrium; native060 inner030; no new equilibrium or RNG draw',FIT_READY=False)
        save_bundle(output/'initial_seed',kind='source_construction_seed',identity=identity,payload=projected,qualification=QUALIFICATION)
        save_bundle(output/'boundary',kind='validation_evidence',identity=identity,payload=products,qualification=QUALIFICATION)
        write(output/'preparation.json',dict(status='FROZEN_READY_NATIVE060_INNER030',passed=True,identity=identity,preflight_sha256=sha(Path(preflight_dir)/'preflight.json'),new_equilibrium_solves=0,actual_dynamic_advance_started=False))
        return dict(passed=True,status='FROZEN_READY_NATIVE060_INNER030')


def load(prepared):
    bundle=load_bundle(Path(prepared)/'initial_seed',expected_kind='source_construction_seed');identity=bundle['identity'];seed=bundle['payload']
    verify_sources(identity['source_pins']);source,ref,_=reference()
    if canonical_hash(ref)!=identity['reference_model_identity_sha256']:raise ContractError('Frozen reference identity changed')
    a=candidate(OUT/'static_candidate_v1')
    if a.artifact_sha256!=identity['artifact'] or canonical_hash(seed)!=identity['seed_sha256']:raise ContractError('Frozen candidate identity changed')
    a.boundary_runtime=load_bundle(Path(prepared)/'boundary',expected_kind='validation_evidence',expected_identity=identity)['payload']
    if a.boundary_runtime['identity']!=seed['patch_boundary']['identity']:raise ContractError('Frozen candidate exterior changed')
    return a,seed,identity


def run(prepared,output):
    a,seed,identity=load(prepared);protocol=identity['protocol']
    with Worker(output) as worker:
        arm=GroupTimeArm(a,seed,protocol,mode='reference',guard=worker.check)
        monitor=GroupTimeMonitor(a,seed['states'],parameters=seed['inner_parameters']);dynamic=DynamicDefectMonitor(a,seed['states']);recorder=CoreRecorder(identity['selection']);edges=[]
        write(output/'model_identity.json',identity)
        def observe(t,side,record=False):
            s=arm.snapshot(t,side,queue_node=True)
            monitor.observe(t,side,s['states'],s['rhs'],transitions=s['transitions'],ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
            if monitor.domain_failures:raise ContractError('Candidate physical/history/conservation gate failed')
            if record:recorder.record(s)
            return s
        try:
            init,_=initial_match(arm,seed);write(output/'initial_seed_match.json',init)
            observe(arm.start,'right');arm.initial_commit();observe(arm.start,'right',True);arm.ack(arm.start)
            for end,flags in calendar(protocol):
                if end<=arm.inner_time:continue
                begin=arm.inner_time;arm.ensure_native(end,'left');advance=arm.advance_inner(begin,end)
                for curve in advance.dense_segments:
                    mid=curve.t0+(curve.t1-curve.t0)/2;witness=dynamic_observe(arm,advance,mid,dynamic,'left' if mid==advance.t1 else 'right')
                    if max(r['maximum_norm'] for r in dynamic.rows.values())>1:
                        save_bundle(output/'failed_dynamic_witness',kind='validation_evidence',identity=identity,payload=dict(states=witness['states'],rhs=witness['rhs'],time_seconds=mid,dynamic=dynamic.report()),qualification=QUALIFICATION)
                        raise ContractError('Original candidate dynamic defect budget failed')
                before=observe(end,'left' if flags['frame'] else 'right',flags['frame'])
                if flags['rod'] or flags['bc']:arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],rod_sample_time=max(arm.start,end-.0001+1e-13) if flags['rod'] else None)
                last=observe(end,'right',flags['record'] or flags['frame'])
                if flags['frame']:
                    keys=[k for k in last['states'] if k[1] not in ('Chen2024PrimateRodCurrent','RGC_expected_history') and not(k[1]=='HistoryBipolarTerminal' and(k[2].startswith('history_') or k[2]=='activation'))]
                    edges.append(dict(time_seconds=end,continuous_state_max_jump=max((float(np.max(abs(last['states'][k]-before['states'][k]),initial=0.)) for k in keys),default=0.)))
                arm.ack(end)
                if end==protocol['checkpoint_seconds']:sparse_states(output/'full_state_mid.npz',last)
            sparse_states(output/'full_state_final.npz',last);recorder.save(output/'core_traces.npz')
            payload=arm.checkpoint_payload(monitor=monitor.payload(),recording=recorder.cursor)
            save_bundle(output/'final_checkpoint',kind='complete_model_checkpoint',identity=identity,payload=payload,qualification=QUALIFICATION)
            clocks=payload['clocks'];clock_ok=clocks['indices']==dict(rod=50930,bc=5093,rgc=5093)
            result=dict(status='NATIVE060_INNER030_100MS_ARM_COMPLETE',passed=dynamic.report()['passed'] and clock_ok and all(e['continuous_state_max_jump']==0 for e in edges),
                start_seconds=arm.start,end_seconds=arm.inner_time,actual_simulated_seconds=arm.inner_time-arm.start,initial_match=init,
                DYNAMIC_NUMERICAL_DEFECT=dynamic.report(),PHYSICAL_DOMAIN_AND_CONSERVATION=dict(passed=not monitor.domain_failures,failures=monitor.domain_failures),
                INPUT_EVENT_AND_HISTORY=dict(passed=clock_ok and all(e['continuous_state_max_jump']==0 for e in edges),edges=edges,clocks=clocks),
                raw_absolute_equilibrium_monitor=monitor.report(.1),metrics=arm.metrics,final_causal_hash=causal_hash(payload),core_records=recorder.cursor['records'],
                candidate_native_support='060',inner_support='030',saved_common_060_background_used=True,reference_060_trace_reused=True,old_030_trace_reused=False,FIT_READY=False)
            write(output/'result.json',result);return result
        except Exception as exc:
            if recorder.stamps:recorder.save(output/'stopped_core_traces.npz')
            write(output/'failure_witness.json',dict(error=type(exc).__name__,message=str(exc),time_seconds=arm.inner_time,dynamic=dynamic.report(),metrics=arm.metrics));raise
        finally:arm.close()


def compare_arrays(x,y,unit,budget):
    allowance=budget['voltage_V'] if unit=='V' else budget['conductance_S'] if unit=='S' else budget['expected_rate_Hz'] if unit=='Hz' else budget['general_atol']+budget['general_rtol']*np.maximum(abs(x),abs(y))
    return abs(x-y)/allowance


def compare(candidate_dir,output):
    _,ref,_=reference();reg=read(REG);d=Path(candidate_dir);identity=read(d/'model_identity.json');verify_sources(identity['source_pins'])
    if canonical_hash(ref)!=identity['reference_model_identity_sha256'] or identity['protocol']!=ref['common']['protocol']:raise ContractError('Candidate/reference protocol or source differs')
    old=read(ROOT/reg['original_failure']['path']);budget=reg['spatial_budget'];rows=[];excluded=[]
    with np.load(d/'core_traces.npz') as x,np.load(ROOT/reg['reference_directory']/'core_traces.npz') as y:
        for name in ['time_seconds','side','generation']:np.testing.assert_array_equal(x[name],y[name])
        if set(identity['selection']['fields'])!=set(ref['selection']['fields']):raise ContractError('Core field registry changed')
        for name,f in identity['selection']['fields'].items():
            g=ref['selection']['fields'][name]
            if not f['spatial_comparable']:
                if name!='ports__local_patch_boundary:gap_current':raise ContractError('Unregistered comparison exclusion')
                excluded.append(name);continue
            j=by_ids(g['ids'],f['ids'],name)
            if len(j)!=len(g['ids']) or f['unit']!=g['unit']:raise ContractError('Core group or unit registry changed')
            a,b=x[name],y[name][:,j];norms=compare_arrays(a,b,f['unit'],budget);maximum=float(np.max(norms,initial=0.))
            row=dict(field=name,unit=f['unit'],global_ids=f['ids'],samples=len(a),maximum_norm=maximum,max_abs=float(np.max(abs(a-b),initial=0.)),passed=maximum<=1)
            if norms.size:
                t,k=np.unravel_index(int(np.argmax(norms)),norms.shape)
                row.update(worst_time_seconds=float(x['time_seconds'][t]),worst_side=str(x['side'][t]),worst_global_id=f['ids'][k],candidate_value=float(a[t,k]),reference_value=float(b[t,k]),
                    worst_point_abs_error=float(abs(a[t,k]-b[t,k])),worst_point_allowance=float(abs(a[t,k]-b[t,k])/maximum) if maximum else None)
            rows.append(row)
    passed=all(r['passed'] for r in rows);worst=max(rows,key=lambda r:r['maximum_norm'])
    old_failed={r['field']:r for r in old['per_field'] if not r['passed']}
    result=dict(status='PASS_NATIVE060_INNER030_100MS_SPATIAL' if passed else 'FAIL_NATIVE060_INNER030_100MS_SPATIAL',passed=passed,
        maximum_norm=worst['maximum_norm'],worst=worst,per_field=rows,excluded_noncore_flux=excluded,thresholds=budget,thresholds_changed=False,
        original_030_failure_retained=True,old_small_trace_reused=False,reference_060_trace_reused=True,
        reference_evidence_pins=reg['reference_evidence_pins'],
        prior_failed_fields_remedy=[dict(field=r['field'],prior_norm=old_failed[r['field']]['maximum_norm'],candidate_norm=r['maximum_norm'],passed=r['passed']) for r in rows if r['field'] in old_failed],
        FIT_READY=False,default_adopted=False,reference_060_convergence_proved=False,external_reservoir_background_qualified=False)
    write(output/'result.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['preflight','prepare','run','compare']);p.add_argument('--output',required=True)
    p.add_argument('--preflight');p.add_argument('--prepared');p.add_argument('--candidate-run');args=p.parse_args();output=Path(args.output).resolve()
    if not output.is_relative_to(OUT) or output.exists():raise ContractError('Fresh independent native-support report directory required')
    output.mkdir(parents=True)
    try:
        if args.mode=='preflight':result=preflight(output)
        elif args.mode=='prepare':result=prepare(args.preflight,output)
        elif args.mode=='run':result=run(args.prepared,output)
        else:result=compare(args.candidate_run,output)
        print(json.dumps(dict(status=result['status'],passed=result['passed'],output=str(output))))
    except Exception as exc:write(output/'failure.json',dict(error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
