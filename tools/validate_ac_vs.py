"""Finite opt-in AC V+s qualification. Each run owns one CPU model worker."""
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import argparse
import json
import math
import os
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np

from retina_sim.ac_vs_model import ACVSAssembly,ACVSInner,load_config,redesign_seed,MECHANISM,functional_patch
from retina_sim.ac_vs_monitor import DynamicDefectMonitor
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.group_time_monitor import GroupTimeMonitor,frozen_budget
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.group_time_storage import save_bundle,load_bundle
from retina_sim.local_patch_graph import LocalPatchAssembly,build_local_patch_graph
from retina_sim.local_patch_runner import prepare_seed,make_protocol,state_witness_hash
from retina_sim.spatial_group_graph import digest
from retina_sim.vh2005 import SOURCE_VALUES
from retina_sim.background_checkpoint import MemoryGuard
from tools.validate_group_time_candidate import clock_events
from tools.validate_group_scc_radau import process_identity
from scipy.integrate import solve_ivp
from scipy.special import expit

REPORT=ROOT/'reports/ac_vs_redesign_v1'
QUALIFICATION=dict(default_adopted=False,fit_eligible=False,physiological_calibration=False)
SOURCES=('retina_sim/ac_vs_model.py','retina_sim/ac_vs_monitor.py','retina_sim/group_inner_dynamics.py',
         'retina_sim/group_time_runner.py','retina_sim/ac_vs_background.py','tools/validate_ac_vs.py',
         'configurations/ac_vs_functional_v1.json','configurations/ac_vs_acceptance_v1.json')


def write(path,value):
    path=Path(path);pending=path.with_name(path.name+'.writing')
    pending.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    os.replace(pending,path)


def protocol_config():
    return json.loads((ROOT/'configurations/ac_vs_acceptance_v1.json').read_text(encoding='utf8'))


class Worker:
    def __init__(self,output):
        self.output=output;self.started=perf_counter()
        self.resources=protocol_config()['resources'];self.peak=0
        self.memory=MemoryGuard(self.resources['process_RSS_limit_bytes'],self.resources['available_RAM_floor_bytes'])
        self.path=REPORT/'ACTIVE_WORKER.json'
        self.owner=dict(pid=os.getpid(),creation_identity=process_identity(os.getpid()),output=str(output))

    def __enter__(self):
        with self.path.open('x',encoding='utf8') as f: json.dump(self.owner,f)
        self.check();return self

    def check(self):
        if (REPORT/'REQUEST_STOP').exists() or (ROOT/'REQUEST_STOP').exists():raise ContractError('Explicit AC worker STOP')
        if json.loads(self.path.read_text(encoding='utf8'))!=self.owner:raise ContractError('AC worker ownership changed')
        healthy=self.memory.check();self.peak=self.memory.peak_rss_bytes
        if not healthy:
            raise ContractError('Real AC worker RAM guard')
        if perf_counter()-self.started>self.resources['single_worker_wall_seconds']:
            raise ContractError('Declared finite AC worker timeout')

    def __exit__(self,*exc):
        write(self.output/'resource_receipt.json',dict(self.owner,elapsed_seconds=perf_counter()-self.started,
            peak_sampled_RSS_bytes=self.peak,limits=self.resources,released=True))
        if self.path.exists() and json.loads(self.path.read_text(encoding='utf8'))==self.owner:self.path.unlink()


def assembly_and_seed(halo,half,guard):
    p=protocol_config();config=load_config(ROOT/p['config'])
    directory=REPORT/('functional_static_halo_'+halo+'_v1')
    config_path=ROOT/'configurations/local_patch_v2'/('halo_'+halo+'.json')
    patch=functional_patch(ROOT,json.loads(config_path.read_text(encoding='utf8')),directory,guard)
    old=prepare_seed(ROOT,patch,patch.config,guard=guard)
    assembly=ACVSAssembly(patch,config,max_step=.00005 if half else .0001)
    return assembly,redesign_seed(assembly,old)


def identity(assembly,seed):
    return dict(schema='ac-vs-qualification-run-1',artifact=assembly.artifact_sha256,
        seed=canonical_hash(seed),sources={n:digest(ROOT/n) for n in SOURCES},
        input_frames={str(k):canonical_hash(v) for k,v in seed['native_inputs'].items()},
        time_origin=seed['time_seconds'],old_cache_or_background_reused=False)


def causal_hash(payload):
    def strip(value):
        if isinstance(value,dict):return {k:strip(v) for k,v in value.items() if k!='metrics'}
        if isinstance(value,list):return [strip(v) for v in value]
        if isinstance(value,tuple):return tuple(strip(v) for v in value)
        return value
    return canonical_hash(strip(payload))


def dynamic_observe(arm,advance,t,monitor,side='right'):
    state=advance.evaluate(t,side)
    snapshot=arm.snapshot(t,side,inner_state=state)
    derivative=advance.representation_derivative(t,side)
    interval=arm._interval_at(t,side)
    if interval is None:raise ContractError('Native dense derivative owner missing')
    for name,curve in interval.curves.items():
        k=('HC','reduced_native',name) if name in ('w','z2','Vh') else ('cones','selected_original_z1_auxiliary','z1_aux') if name=='z1_aux' else ('cones','reduced_native',name)
        derivative[k]=np.asarray(curve.derivative(t))
    a=interval.anchor;dt=(t-a.time_seconds)*1000
    tr,te=SOURCE_VALUES['tau_R'],SOURCE_VALUES['tau_E']
    er,ee=np.exp(-dt/tr),np.exp(-dt/te)
    derivative['cones','analytic_RE','R']=1000*(a.light-a.R)*er/tr
    derivative['cones','analytic_RE','E']=1000*((a.light-a.E)*ee/te+(a.R-a.light)*tr/(tr-te)*(-er/tr+ee/te))
    monitor.observe(snapshot['states'],snapshot['rhs'],derivative,time_seconds=t,side='interior_of_ZOH_and_event_interval')
    return snapshot


def save_traces(output,assembly,observations):
    values={'time_seconds':np.array([s['time_seconds'] for s in observations])}
    for category in ('states','rhs','ports'):
        for k in observations[0][category]:
            values[category+'__'+':'.join(k)]=np.stack([s[category][k] for s in observations])
    np.savez_compressed(output/'traces.npz',**values)
    write(output/'trace_registry.json',dict(populations={n:dict(group_ids=p.group_ids.tolist(),
        core_rows=assembly.core_population_rows.get(n,np.empty(0,dtype=int)).tolist()) for n,p in assembly.populations.items()},
        units={':'.join(k):v for k,v in observations[0]['port_units'].items()},
        state_units={':'.join((b.population,b.mechanism,b.name)):b.unit for b in (*assembly.state_blocks,*assembly.probability_blocks,*assembly.auxiliary_z1_blocks)}))


def run(args,output,worker):
    p=protocol_config();assembly,seed=assembly_and_seed(args.halo,args.half,worker.check)
    model_identity=identity(assembly,seed)
    protocol=make_protocol(seed,duration_seconds=p['duration_seconds'],bounded_v2=True)
    protocol['ac_vs_contract_sha256']=digest(ROOT/'configurations/ac_vs_acceptance_v1.json')
    write(output/'model_identity.json',model_identity);write(output/'routing.json',assembly.route_ledger)
    write(output/'protocol.json',protocol)
    save_bundle(output/'construction_seed',kind='source_construction_seed',identity=model_identity,payload=seed,qualification=QUALIFICATION)
    if args.resume:
        prior=Path(args.resume)
        if json.loads((prior.parent/'model_identity.json').read_text())!=model_identity:raise ContractError('New process restore identity differs')
        bundle=load_bundle(prior,expected_kind='complete_model_checkpoint')
        arm=GroupTimeArm.from_checkpoint(assembly,seed,protocol,bundle['payload'],mode='reference',guard=worker.check)
        monitor=GroupTimeMonitor.from_payload(assembly,seed['states'],bundle['payload']['monitor'],parameters=seed['inner_parameters'])
        restore_start_hash=causal_hash(arm.checkpoint_payload(monitor=monitor.payload()))
        restored_expected_hash=causal_hash(bundle['payload'])
        if restore_start_hash!=restored_expected_hash:raise ContractError('Restored causal checkpoint differs before advancing')
    else:
        arm=GroupTimeArm(assembly,seed,protocol,mode='reference',guard=worker.check)
        monitor=GroupTimeMonitor(assembly,seed['states'],parameters=seed['inner_parameters'])
        restore_start_hash=restored_expected_hash=None
    dynamic=DynamicDefectMonitor(assembly,seed['states'])
    observations=[];prepost=[]
    checkpoint_time=seed['time_seconds']+p['checkpoint_offset_seconds']
    def observe(t,side,*,record=False):
        snapshot=arm.snapshot(t,side,queue_node=True)
        monitor.observe(t,side,snapshot['states'],snapshot['rhs'],transitions=snapshot['transitions'],ports=snapshot['ports'],aux=snapshot['aux'],event_generation=snapshot['event_generation'])
        if monitor.domain_failures:raise ContractError('Physical/history/conservation monitor failure '+str(monitor.domain_failures[:3]))
        if record:observations.append(snapshot)
        return snapshot
    try:
        if not args.resume:
            observe(arm.start,'right');arm.initial_commit();observe(arm.start,'right',record=True);arm.ack(arm.start)
        events={t:(rod,bc) for t,rod,bc in clock_events(protocol)}
        for i in range(int(arm.start*150)+1,int(arm.stop*150)+1):
            edge=i/150.
            if arm.start<edge<arm.stop:events.setdefault(edge,(False,False))
        for end,(rod,bc) in sorted(events.items()):
            if end<=arm.inner_time+1e-13:continue
            begin=arm.inner_time;arm.ensure_native(end,'left');advance=arm.advance_inner(begin,end)
            # Every accepted inner polynomial is checked at its interior midpoint.
            for curve in advance.dense_segments:
                midpoint=curve.t0+(curve.t1-curve.t0)/2
                dynamic_observe(arm,advance,midpoint,dynamic,'left' if midpoint==advance.t1 else 'right')
            at_edge=abs(end*150-round(end*150))<1e-10
            before=observe(end,'left' if at_edge else 'right')
            if rod or bc:arm.commit(end,rod=rod,bc=bc,rgc=bc,rod_sample_time=max(arm.start,end-.0001+1e-13) if rod else None)
            after=observe(end,'right',record=True)
            if at_edge:
                differences={':'.join(k):float(np.max(np.abs(before['states'][k]-after['states'][k]),initial=0.)) for k in after['states'] if k[1] not in ('Chen2024PrimateRodCurrent','RGC_expected_history') and not (k[1]=='HistoryBipolarTerminal' and (k[2].startswith('history_') or k[2]=='activation'))}
                prepost.append(dict(time_seconds=end,continuous_state_max_jump=max(differences.values(),default=0.),BC_clock=bc,rod_clock=rod))
            arm.ack(end)
            if not args.resume and math.isclose(end,checkpoint_time,abs_tol=1e-13,rel_tol=0):
                save_bundle(output/'checkpoint',kind='complete_model_checkpoint',identity=model_identity,
                    payload=arm.checkpoint_payload(monitor=monitor.payload()),qualification=QUALIFICATION)
        final_payload=arm.checkpoint_payload(monitor=monitor.payload())
        save_bundle(output/'final_checkpoint',kind='complete_model_checkpoint',identity=model_identity,payload=final_payload,qualification=QUALIFICATION)
        save_traces(output,assembly,observations)
        absolute=monitor.report(protocol['stop_seconds']-protocol['start_seconds'])
        defects=dynamic.report()
        result=dict(status='NEW_MODEL_FINITE_RUN_COMPLETE',identity=model_identity,final_time=arm.inner_time,
            absolute_monitor_preserved=absolute,DYNAMIC_NUMERICAL_DEFECT=defects,
            PHYSICAL_DOMAIN_AND_CONSERVATION=dict(passed=not monitor.domain_failures,failures=monitor.domain_failures),
            INPUT_EVENT_AND_HISTORY=dict(passed=not monitor.domain_failures and all(e['continuous_state_max_jump']==0 for e in prepost),
                input_edges=prepost,clocks=final_payload['clocks'],input_position=final_payload['input_position'],
                frame_hashes=model_identity['input_frames'],stamps=monitor.stamps),
            BACKGROUND_EQUILIBRIUM=dict(passed=False,status='SOURCE_IVP_NOT_COUPLED_BACKGROUND; independent background command required'),
            state_witness_sha256=causal_hash(final_payload),new_process_resume=bool(args.resume),
            restored_start_hash=restore_start_hash,saved_checkpoint_hash=restored_expected_hash,
            runtime_metrics=arm.metrics,**QUALIFICATION)
        write(output/'result.json',result)
        return result
    finally:arm.close()


def prepare(args,output,worker):
    assembly,seed=assembly_and_seed(args.halo,args.half,worker.check)
    result=dict(status='NEW_AC_STATIC_AND_SOURCE_IVP_READY',identity=identity(assembly,seed),
        routing=assembly.route_ledger,states=[vars(b) for b in assembly.state_blocks],
        source_and_units=dict(BC_effective_release='dimensionless bounded fraction; 1-to-1 conversion; no second sigmoid',
            input_conductance='S per effective pool',V='V',s='dimensionless',output='S total target conductance; target members once',
            AII_direct_RGC_enabled=False,old_S_ON_state_and_weights_retained=True),
        backup_receipt=dict(path='D:/macaque_retina_simulator.backup_pre_ac_redesign_20261027_v1/_LOCAL_BACKUP_METADATA/verification.json',
            sha256=digest(Path('D:/macaque_retina_simulator.backup_pre_ac_redesign_20261027_v1/_LOCAL_BACKUP_METADATA/verification.json'))),
        **QUALIFICATION)
    write(output/'result.json',result)
    return result


def compare(args,output,worker):
    left,right=Path(args.left),Path(args.right)
    lreg=json.loads((left/'trace_registry.json').read_text());rreg=json.loads((right/'trace_registry.json').read_text())
    rows=[];p=protocol_config();kind='numerical' if args.numerical else 'spatial'
    with np.load(left/'traces.npz',allow_pickle=False) as l,np.load(right/'traces.npz',allow_pickle=False) as r:
        lt,rt=l['time_seconds'],r['time_seconds']
        times=[(i,int(j[0])) for i,t in enumerate(lt) if len(j:=np.flatnonzero(np.isclose(rt,t,rtol=0,atol=1e-13)))]
        for name in l.files:
            if name=='time_seconds' or name not in r.files or name.startswith('rhs__'):continue
            category,keytext=name.split('__',1);key=keytext.split(':');pop=key[0]
            if pop not in lreg['populations'] or pop not in rreg['populations']:continue
            lp,rp=lreg['populations'][pop],rreg['populations'][pop]
            rid={v:i for i,v in enumerate(rp['group_ids'])}
            selected=list(range(len(lp['group_ids']))) if args.numerical else lp['core_rows']
            pairs=[(i,rid[lp['group_ids'][i]]) for i in selected if lp['group_ids'][i] in rid]
            # Contact states and contact ports have route identities, not population rows.
            if l[name].shape[1]!=len(lp['group_ids']) or r[name].shape[1]!=len(rp['group_ids']):continue
            if not pairs:continue
            a=l[name][np.ix_([i for i,j in times],[i for i,j in pairs])]
            b=r[name][np.ix_([j for i,j in times],[j for i,j in pairs])]
            unit=(lreg['state_units'][keytext] if category=='states' else lreg['units'][keytext])
            if args.numerical:
                if category!='states':continue
                budget=frozen_budget(l[name][0],unit)[2][[i for i,j in pairs]][None,:]
            else:
                budget=p['spatial_budget']['voltage_V'] if unit=='V' else p['spatial_budget']['conductance_S'] if unit=='S' else p['spatial_budget']['expected_rate_Hz'] if unit=='Hz' else p['spatial_budget']['general_atol']+p['spatial_budget']['general_rtol']*np.maximum(np.abs(a),np.abs(b))
            norm=float(np.max(np.abs(a-b)/budget,initial=0.))
            rows.append(dict(field=name,unit=unit,rows=len(pairs),samples=len(times),max_abs=float(np.max(np.abs(a-b),initial=0.)),maximum_norm=norm,passed=norm<=1))
    maximum=max((row['maximum_norm'] for row in rows),default=0.)
    result=dict(category=kind.upper(),passed=bool(rows and maximum<=1),maximum_norm=maximum,
        comparisons=rows,left=str(left),right=str(right),budget_frozen_before_runs=True,
        common_missing_edge_check='Each new channel validated against full parent before either Halo ran; zero missing chemical edges. Original native/gap held exterior limitations retained.')
    write(output/'result.json',result);return result


def isolated(args,output,worker):
    contract=protocol_config();c=contract['isolated_step'];config=load_config(ROOT/contract['config'])
    p=config['parameters_per_effective_pool_SI'];G=config['channels'][0]['G_BA_S']
    boundaries=[0.,*c['switch_seconds'],c['stop_seconds']]
    times=np.linspace(0,c['stop_seconds'],2501)
    results={}
    for step in (contract['base_max_step_seconds'],contract['half_max_step_seconds']):
        y=np.array([c['initial_V_V'],c['initial_s']]);values=[];analytic_error=0.
        for j,(start,stop) in enumerate(zip(boundaries,boundaries[1:])):
            worker.check();gE=p['gE_base']+G*c['release_fraction'][j]
            def f(t,y):
                probability=expit((y[0]-p['theta'])/p['k'])
                return [(p['gL']*(p['EL']-y[0])+gE*(p['EE']-y[0]))/p['C'],p['alpha']*probability*(1-y[1])-p['beta']*y[1]]
            grid=times[(times>=start)&(times<=stop)]
            sol=solve_ivp(f,(start,stop),y,method='DOP853',rtol=1e-6,atol=1e-9,max_step=step,dense_output=True)
            if not sol.success or not np.isfinite(sol.y).all():raise ContractError('Isolated step integration failed')
            sampled=sol.sol(grid)
            Vinf=(p['gL']*p['EL']+gE*p['EE'])/(p['gL']+gE)
            exact=Vinf+(y[0]-Vinf)*np.exp(-(grid-start)*(p['gL']+gE)/p['C'])
            analytic_error=max(analytic_error,float(np.max(np.abs(sampled[0]-exact))))
            values.append(sampled[:,:-1] if j<2 else sampled);y=sol.y[:,-1]
        values=np.concatenate(values,axis=1)
        if values.shape!=(2,len(times)):raise ContractError('Isolated exact sampling coverage mismatch')
        results[str(step)]=dict(values=values,analytic_error=analytic_error)
    a=results[str(contract['base_max_step_seconds'])]['values'];b=results[str(contract['half_max_step_seconds'])]['values']
    budgets=np.array([frozen_budget(np.array([c['initial_V_V']]),'V')[2][0],frozen_budget(np.array([c['initial_s']]),'dimensionless')[2][0]])
    norm=float(np.max(np.abs(a-b)/budgets[:,None]));fractions=np.select([times<c['switch_seconds'][0],times<c['switch_seconds'][1]],[c['release_fraction'][0],c['release_fraction'][1]],default=c['release_fraction'][2])
    gE=G*fractions+p['gE_base'];probability=expit((a[0]-p['theta'])/p['k'])
    polarity=bool(a[0,1400]>a[0,500] and a[1,1400]>a[1,500] and a[0,-1]<a[0,1500] and a[1,-1]<a[1,1500])
    passed=bool(norm<=1 and np.all(gE>=0) and np.all((a[1]>=0)&(a[1]<=1)) and polarity)
    np.savez_compressed(output/'isolated_step.npz',time_seconds=times,gE_S=gE,V_V=a[0],p=probability,s=a[1],half_V_V=b[0],half_s=b[1])
    result=dict(category='ISOLATED_STEP',passed=passed,base_half_maximum_norm=norm,
        membrane_analytic_max_error_V=max(v['analytic_error'] for v in results.values()),
        gate_min=float(a[1].min()),gate_max=float(a[1].max()),polarity_passed=polarity,
        gate_constant_voltage_analytic_test='tests/test_ac_vs_model.py',input_and_parameters_frozen=True)
    write(output/'result.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','run','compare','background','isolated'))
    parser.add_argument('--halo',choices=('030','060'),default='030');parser.add_argument('--half',action='store_true')
    parser.add_argument('--output',required=True);parser.add_argument('--resume');parser.add_argument('--left');parser.add_argument('--right');parser.add_argument('--numerical',action='store_true')
    args=parser.parse_args();output=Path(args.output).resolve()
    if not output.is_relative_to(REPORT) or output.exists():raise ContractError('Fresh output under this new AC report root required')
    output.mkdir(parents=True)
    write(output/'attempt.json',dict(command=sys.argv,pid=os.getpid(),mode=args.mode,qualification=QUALIFICATION))
    try:
        with Worker(output) as worker:
            if args.mode=='background':
                from retina_sim.ac_vs_background import qualify_background
                assembly,seed=assembly_and_seed(args.halo,args.half,worker.check)
                result=qualify_background(assembly,seed,output,worker.check)
            else:result={'prepare':prepare,'run':run,'compare':compare,'isolated':isolated}[args.mode](args,output,worker)
        print(json.dumps(dict(status=result.get('status',result.get('category')),passed=result.get('passed'),output=str(output))))
    except Exception as exc:
        write(output/'failure.json',dict(status='FAILED_PRESERVED',error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
