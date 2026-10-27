"""Frozen100ms AC engineering stage, bounded core storage and serial workers."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import math
import shutil
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.group_time_storage import load_bundle,save_bundle
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.group_time_monitor import GroupTimeMonitor,frozen_budget
from retina_sim.ac_vs_monitor import DynamicDefectMonitor
from retina_sim.group_inner_dynamics import CONTACTS
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.background import NUMERICAL_UNIT_SCALES
from retina_sim.resident_derived_ports import compare_cell
from retina_sim.rgc_expectation import STATES
from retina_sim.spatial_group_model import RGC
from tools.validate_ac_vs import Worker,QUALIFICATION,write,causal_hash,dynamic_observe
from tools.validate_ac_vs_equilibrium_start import load_prepared,initial_match,bind_original_exterior,verify_sources
from tools.validate_group_time_candidate import clock_events,sample_grid

OUT=ROOT/'reports/ac_vs_timescale_v1'
REG=OUT/'pre_registration_v1.json'
SCHEMA='ac-vs-timescale-engineering-1'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    from retina_sim.spatial_group_graph import digest
    return digest(path)


def calendar(protocol):
    events={t:dict(rod=r,bc=b,frame=False,record=False) for t,r,b in clock_events(protocol)}
    for i in range(int(protocol['start_seconds']*150)+1,int(protocol['stop_seconds']*150)+1):
        t=i/150.
        if protocol['start_seconds']<t<protocol['stop_seconds']:
            events.setdefault(t,dict(rod=False,bc=False,frame=False,record=False))['frame']=True
    for t in sample_grid(protocol['start_seconds'],protocol['stop_seconds'],.0005):
        if t>protocol['start_seconds']:
            events.setdefault(t,dict(rod=False,bc=False,frame=False,record=False))['record']=True
    events.setdefault(protocol['checkpoint_seconds'],dict(rod=False,bc=False,frame=False,record=False))['record']=True
    return sorted(events.items())


def selection(assembly,saved):
    """Semantic row selection; contacts are never guessed from array lengths."""
    core=assembly.core_population_rows;fields={}
    contacts={r.contact_component:r for r in assembly.routes.values() if r.contact_component}
    crows={c:np.flatnonzero(np.isin(assembly.array('routes/'+r.name+'/contact_post_group.npy'),core[r.post])) for c,r in contacts.items()}
    units={':'.join((b.population,b.mechanism,b.name)):b.unit for b in (*assembly.state_blocks,*assembly.probability_blocks,*assembly.auxiliary_z1_blocks)}
    for key,value in saved['states'].items():
        pop,mechanism,name=key
        rows=crows[mechanism] if mechanism in CONTACTS else np.empty(0,dtype=int) if mechanism=='selected_original_z1_auxiliary' else core.get(pop,np.empty(0,dtype=int))
        fields['states__'+':'.join(key)]=dict(category='states',key=list(key),rows=np.asarray(rows,dtype=int).tolist(),unit=units[':'.join(key)])
    old=read(ROOT/'reports/ac_vs_equilibrium_start_v1/run_v1/trace_registry.json')
    for text,unit in old['units'].items():
        pop,name=text.split(':')
        if pop=='cones' and name=='z1_aux':rows=np.empty(0,dtype=int)
        elif pop in contacts:
            rows=crows[pop] if name=='contact_conductance' else core[contacts[pop].post]
        elif pop=='local_patch_boundary':rows=np.array([0])
        else:rows=core.get(pop,np.empty(0,dtype=int))
        fields['ports__'+text]=dict(category='ports',key=[pop,name],rows=np.asarray(rows,dtype=int).tolist(),unit=unit)
    return dict(schema='ac-vs-core-recording-1',fields=fields,selection_hash=canonical_hash(fields))


def prepare(output):
    reg=read(REG)
    assembly,saved,_,parent=load_prepared(ROOT/reg['parent_preparation'],ROOT/reg['initial_seed'])
    seed=deepcopy(saved);B=saved['input_reference']['native_background_td']
    mask=np.zeros(len(B),dtype=bool);mask[np.asarray(assembly.core_population_rows['cones'],dtype=int)]=True
    if np.count_nonzero(mask)!=15:raise ContractError('The previously frozen15-cone core changed')
    frames=list(range(int(math.floor(reg['start_seconds']*150)),int(math.ceil(reg['stop_seconds']*150))+1))
    seed['native_inputs']={i:B*(1+.05*mask) if reg['contrast_on_frame']<=i<reg['contrast_off_frame'] else B.copy() for i in frames}
    rod=saved['rod_inputs'][min(saved['rod_inputs'])]
    seed['rod_inputs']={i:rod.copy() for i in frames}
    stimulus=dict(schema='ac-core-grey-pulse-grey-1',unit='td',start_seconds=reg['start_seconds'],stop_seconds=reg['stop_seconds'],
        contrast=.05,on_seconds=reg['contrast_on_seconds'],off_seconds=reg['contrast_off_seconds'],
        mask_group_ids=np.asarray(assembly.populations['cones'].group_ids)[mask].tolist(),mask_sha256=canonical_hash(mask),
        native_frame_sha256={str(i):canonical_hash(v) for i,v in seed['native_inputs'].items()},rod_background_sha256=canonical_hash(rod),real_movie=False)
    stimulus['identity']=canonical_hash(stimulus)
    seed['ac_vs_lineage']=dict(schema=SCHEMA,artifact=assembly.artifact_sha256,saved_equilibrium_seed_used=True,
        saved_seed=parent['saved_seed'],background_receipt=parent['background_receipt'],stimulus_identity=stimulus['identity'],
        whole_retina_background_qualified=False,real_movie_cache_reused=False)
    protocol=deepcopy(parent['protocol']);protocol.update(stop_seconds=reg['stop_seconds'],checkpoint_seconds=reg['checkpoint_seconds'],
        input_frame_indices=frames,equilibrium_start_stimulus_identity=stimulus['identity'],timescale_registration_sha256=sha(REG))
    selected=selection(assembly,saved);events=calendar(protocol)
    baseclock=clock_events(protocol)
    if sum(x['rod'] for t,x in events)!=sum(x[1] for x in baseclock) or sum(x['bc'] for t,x in events)!=sum(x[2] for x in baseclock):
        raise ContractError('Calendar lost an original tick')
    widths=np.diff([reg['start_seconds'],*(t for t,_ in events)])
    if np.any(widths<=0):raise ContractError('Nonincreasing frozen event calendar')
    cells=sum(len(f['rows']) for f in selected['fields'].values())
    nrecords=1+sum(x['record'] or x['frame'] for t,x in events)+sum(x['frame'] for t,x in events)
    raw=cells*nrecords*8;objects=nrecords*len(selected['fields'])*160
    prior=read(ROOT/'reports/ac_vs_equilibrium_start_v1/run_v1/resource_receipt.json')
    baseline=prior['peak_sampled_RSS_bytes'];projected_peak=baseline+128*1024**2+4*(raw+objects)
    projected_disk=3*(raw+objects)+64*1024**2
    memory=MemoryGuard(16*1024**3,4*1024**3);healthy=memory.check();disk=shutil.disk_usage(ROOT)
    feasible=healthy and projected_peak<16*1024**3 and memory.last['available_physical_bytes']>4*1024**3+projected_peak and disk.free>2*projected_disk
    resources=dict(feasible=bool(feasible),actual_memory=memory.last,actual_disk_free_bytes=disk.free,
        state_scalars=sum(len(v) for v in saved['states'].values()),selected_scalar_columns=cells,maximum_core_records=nrecords,
        raw_core_trace_bytes_per_arm=raw,conservative_object_overhead_bytes=objects,projected_peak_RSS_bytes=projected_peak,
        projected_new_disk_bytes=projected_disk,baseline_measured_RSS_bytes=baseline,
        coarse_hot_cost_projection_seconds=(.0735058+.8144596+.4353967)*(.1/.0031),
        limits='existing900s single-worker/16GiB RSS/4GiB free reserve; no aggregate role budget')
    write(output/'MEMORY_PREFLIGHT.json',resources)
    if not feasible:raise ContractError('100ms memory/disk preflight failed; no model started')
    pins=dict(parent['sources']);pins['tools/validate_ac_vs_timescale.py']=sha(Path(__file__));pins[str(REG.relative_to(ROOT)).replace('\\','/')]=sha(REG)
    identity=dict(schema=SCHEMA,artifact=assembly.artifact_sha256,sources=pins,saved_seed=parent['saved_seed'],background_receipt=parent['background_receipt'],
        stimulus=stimulus,protocol=protocol,selection=selected,registration=reg,seed_sha256=canonical_hash(seed),saved_background_reused=True,FIT_READY=False)
    save_bundle(output/'initial_seed',kind='source_construction_seed',identity=identity,payload=seed,qualification=QUALIFICATION)
    write(output/'protocol.json',protocol);write(output/'stimulus.json',stimulus);write(output/'selection.json',selected)
    write(output/'calendar.json',dict(events=[dict(time_seconds=t,**flags) for t,flags in events],original_rod_ticks=sum(x[1] for x in baseclock),
        original_BC_RGC_ticks=sum(x[2] for x in baseclock),near_neighbour_intervals_below_1ps=int(np.sum(widths<1e-12)),minimum_interval_seconds=float(widths.min())))
    result=dict(status='PRE_REGISTERED_MEMORY_READY',identity=identity,resources=resources,real_model_advanced=False)
    write(output/'preparation.json',result);return result


def load(prepared):
    p=Path(prepared).resolve();raw=read(p/'preparation.json');identity=raw['identity'];verify_sources(identity['sources'])
    assembly,saved,_,parent=load_prepared(ROOT/identity['registration']['parent_preparation'],ROOT/identity['registration']['initial_seed'])
    bundle=load_bundle(p/'initial_seed',expected_kind='source_construction_seed',expected_identity=identity)
    if canonical_hash(bundle['payload'])!=identity['seed_sha256'] or assembly.artifact_sha256!=identity['artifact']:
        raise ContractError('Frozen timescale input/assembly changed')
    return assembly,saved,bundle['payload'],identity


class CoreRecorder:
    def __init__(self,selection,cursor=None):
        self.selection=selection;self.values={f:[] for f in selection['fields']};self.stamps=[]
        self.cursor=dict(schema='ac-vs-core-cursor-1',selection_hash=selection['selection_hash'],records=0,last_stamp=None) if cursor is None else deepcopy(cursor)
        if self.cursor['selection_hash']!=selection['selection_hash']:raise ContractError('Core recorder selection changed')
    def record(self,s):
        stamp=[s['time_seconds'],s['side'],s['event_generation']]
        if stamp==self.cursor['last_stamp']:return
        for name,f in self.selection['fields'].items():
            self.values[name].append(np.asarray(s[f['category']][tuple(f['key'])])[f['rows']].copy())
        self.stamps.append(stamp);self.cursor['records']+=1;self.cursor['last_stamp']=stamp
    def save(self,path):
        arrays={n:np.stack(v) for n,v in self.values.items()}
        arrays.update(time_seconds=np.array([s[0] for s in self.stamps]),side=np.array([s[1] for s in self.stamps]),generation=np.array([s[2] for s in self.stamps],dtype=np.int64))
        np.savez_compressed(path,**arrays)


def sparse_states(path,snapshot):
    np.savez_compressed(path,time_seconds=np.array([snapshot['time_seconds']]),**{':'.join(k):v for k,v in snapshot['states'].items()})


def run(prepared,output,arm_name,*,resume=None):
    assembly,saved,seed,identity=load(prepared);assembly.max_step=.0001 if arm_name=='coarse' else .00005
    run_identity=dict(preparation=identity,numerical_arm=arm_name,inner_max_step_seconds=assembly.max_step)
    protocol=deepcopy(identity['protocol']);protocol['timescale_numerical_arm']=arm_name
    with Worker(output) as worker:
        bind_original_exterior(assembly,saved)
        if resume:
            parent=Path(resume).resolve()
            prior=load_bundle(parent,expected_kind='complete_model_checkpoint',expected_identity=run_identity)['payload']
            arm=GroupTimeArm.from_checkpoint(assembly,seed,protocol,prior,mode='reference',guard=worker.check)
            monitor=GroupTimeMonitor.from_payload(assembly,seed['states'],prior['monitor'],parameters=seed['inner_parameters'])
            recorder=CoreRecorder(identity['selection'],prior['recording'])
            if causal_hash(arm.checkpoint_payload(monitor=monitor.payload(),recording=recorder.cursor))!=causal_hash(prior):raise ContractError('Restore entry causal state changed')
            init=read(parent.parent/'initial_seed_match.json')
        else:
            arm=GroupTimeArm(assembly,seed,protocol,mode='reference',guard=worker.check)
            monitor=GroupTimeMonitor(assembly,seed['states'],parameters=seed['inner_parameters']);recorder=CoreRecorder(identity['selection'])
        segment_start=arm.inner_time
        dynamic=DynamicDefectMonitor(assembly,seed['states']);edge_rows=[];writes=[]
        write(output/'model_identity.json',run_identity)
        def observe(t,side,*,record=False):
            s=arm.snapshot(t,side,queue_node=True)
            monitor.observe(t,side,s['states'],s['rhs'],transitions=s['transitions'],ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
            if monitor.domain_failures:raise ContractError('Original physical/history/conservation gate failed '+str(monitor.domain_failures[:3]))
            if record:recorder.record(s)
            return s
        try:
            if not resume:
                init,_=initial_match(arm,saved);write(output/'initial_seed_match.json',init)
                s=observe(arm.start,'right')
                np.savez_compressed(output/'initial_core_ports.npz',**{n:np.asarray(s['ports'][tuple(f['key'])])[f['rows']] for n,f in identity['selection']['fields'].items() if f['category']=='ports'})
                arm.initial_commit();observe(arm.start,'right',record=True);arm.ack(arm.start)
            for end,flags in calendar(protocol):
                if end<=arm.inner_time:continue
                begin=arm.inner_time;arm.ensure_native(end,'left');advance=arm.advance_inner(begin,end)
                for curve in advance.dense_segments:
                    mid=curve.t0+(curve.t1-curve.t0)/2
                    side='left' if mid==advance.t1 else 'right'
                    witness=dynamic_observe(arm,advance,mid,dynamic,side)
                    if max((r['maximum_norm'] for r in dynamic.rows.values()),default=0.)>1:
                        save_bundle(output/'failed_dynamic_witness',kind='validation_evidence',identity=run_identity,
                            payload=dict(time_seconds=mid,side=side,states=witness['states'],rhs=witness['rhs'],ports=witness['ports'],
                                inner_derivative=advance.representation_derivative(mid,side),
                                inner_dense=dict(t0=curve.t0,t1=curve.t1,y0=curve.y0,coefficients=curve.coefficients),
                                native_interval=arm._interval_at(mid,side).payload(),dynamic=dynamic.report()),qualification=QUALIFICATION)
                        raise ContractError('DYNAMIC_NUMERICAL_DEFECT exceeds original budget; accepted dense evidence preserved')
                before=observe(end,'left' if flags['frame'] else 'right',record=flags['frame'])
                if flags['rod'] or flags['bc']:
                    arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],rod_sample_time=max(arm.start,end-.0001+1e-13) if flags['rod'] else None)
                last=observe(end,'right',record=flags['record'] or flags['frame'])
                if flags['frame']:
                    keys=[k for k in last['states'] if k[1] not in ('Chen2024PrimateRodCurrent','RGC_expected_history') and not (k[1]=='HistoryBipolarTerminal' and (k[2].startswith('history_') or k[2]=='activation'))]
                    jump=max((float(np.max(np.abs(last['states'][k]-before['states'][k]),initial=0.)) for k in keys),default=0.)
                    edge_rows.append(dict(time_seconds=end,continuous_state_max_jump=jump))
                arm.ack(end)
                if not resume and end==protocol['checkpoint_seconds']:
                    sparse_states(output/'full_state_mid.npz',last)
                    if arm_name=='coarse':
                        recorder.save(output/'core_prefix.npz')
                        writes.append(save_bundle(output/'checkpoint',kind='complete_model_checkpoint',identity=run_identity,
                            payload=arm.checkpoint_payload(monitor=monitor.payload(),recording=recorder.cursor),qualification=QUALIFICATION))
            final=arm.snapshot(arm.stop,'right');sparse_states(output/'full_state_final.npz',final)
            payload=arm.checkpoint_payload(monitor=monitor.payload(),recording=recorder.cursor)
            writes.append(save_bundle(output/'final_checkpoint',kind='complete_model_checkpoint',identity=run_identity,payload=payload,qualification=QUALIFICATION))
            recorder.save(output/('core_suffix.npz' if resume else 'core_traces.npz'))
            expected_clocks=dict(rod=int(round((arm.stop-protocol['clock_origin_seconds'])/.0001)),
                bc=int(math.floor((arm.stop-protocol['clock_origin_seconds'])/.001+1e-10)),
                rgc=int(math.floor((arm.stop-protocol['clock_origin_seconds'])/.001+1e-10)))
            clocks_ok=payload['clocks']['indices']==expected_clocks
            report=dynamic.report();events_ok=clocks_ok and all(x['continuous_state_max_jump']==0 for x in edge_rows)
            result=dict(status='AC_TIMESCALE_RUN_COMPLETE',passed=init['passed'] and report['passed'] and events_ok,
                model_origin_seconds=arm.start,start_seconds=segment_start,end_seconds=arm.stop,actual_simulated_seconds=arm.inner_time-segment_start,
                complete_protocol_seconds=arm.stop-arm.start,
                initial_match=init,DYNAMIC_NUMERICAL_DEFECT=report,PHYSICAL_DOMAIN_AND_CONSERVATION=dict(passed=not monitor.domain_failures,failures=monitor.domain_failures),
                INPUT_EVENT_AND_HISTORY=dict(passed=events_ok,edges=edge_rows,clocks=payload['clocks'],expected_indices=expected_clocks,all_original_ticks_retained=clocks_ok,input_position=payload['input_position']),
                raw_absolute_equilibrium_monitor=monitor.report(arm.stop-arm.start),final_causal_sha256=causal_hash(payload),
                recording_cursor=recorder.cursor,metrics=arm.metrics,sparse_checkpoints=writes,source_identity=run_identity,
                FIT_READY=False,whole_retina_or_arbitrary_exterior_background_qualified=False,**QUALIFICATION)
            write(output/'result.json',result);return result
        except Exception as exc:
            if recorder.stamps:recorder.save(output/'stopped_core_traces.npz')
            write(output/'failure_witness.json',dict(error=type(exc).__name__,message=str(exc),time_seconds=arm.inner_time,
                dynamic=dynamic.report(),absolute=monitor.report(max(0,arm.inner_time-arm.start)) if monitor.count else None,metrics=arm.metrics))
            raise
        finally:arm.close()


def compare(prepared,coarse,refined,output,restored=None):
    _,saved,seed,identity=load(prepared);fields=identity['selection']['fields'];rows=[]
    budgets={':'.join(k):frozen_budget(v,next(f['unit'] for f in fields.values() if f['category']=='states' and f['key']==list(k)))[2] for k,v in saved['states'].items()}
    with np.load(coarse/'core_traces.npz',allow_pickle=False) as a,np.load(refined/'core_traces.npz',allow_pickle=False) as b,np.load(coarse/'initial_core_ports.npz',allow_pickle=False) as initial:
        for key in ('time_seconds','side','generation'):np.testing.assert_array_equal(a[key],b[key])
        for name,f in fields.items():
            av,bv=a[name],b[name];selected=f['rows'];text=':'.join(f['key'])
            if f['category']=='states':bound=budgets[text][selected][None,:]
            elif f['unit'] in NUMERICAL_UNIT_SCALES:bound=frozen_budget(initial[name],f['unit'])[2][None,:]
            elif f['unit']=='Hz' and f['key'][0] in RGC:
                pop=f['key'][0];p=seed['inner_parameters']['populations'][pop]
                va,vb=a['states__'+pop+':RGC_spike_history:V'],b['states__'+pop+':RGC_spike_history:V']
                kernel=np.asarray(p['spike_history_kernel']);ha=p['alpha']*np.logaddexp(0.,(va[:,:,None]+(STATES@kernel)[None,None,:]-p['mu'])/p['beta'])
                hb=p['alpha']*np.logaddexp(0.,(vb[:,:,None]+(STATES@kernel)[None,None,:]-p['mu'])/p['beta'])
                piB=np.stack([budgets[pop+':RGC_expected_history:history_probability_'+str(i)][selected] for i in range(8)],axis=-1)
                rates=np.maximum(ha,hb) if f['key'][1]=='expected_hazard_Hz' else np.maximum(-np.expm1(-.001*ha),-np.expm1(-.001*hb))/.001
                bound=p['alpha']/p['beta']*budgets[pop+':RGC_spike_history:V'][selected][None,:]*(1+1e-12)+np.sum(piB[None,:,:]*rates,axis=-1)+8192*np.finfo(float).eps*np.maximum(np.maximum(abs(av),abs(bv)),1.)
            elif f['unit']=='events/s':
                pop=f['key'][0];p=seed['inner_parameters']['populations'][pop]
                va,vb=a['states__'+pop+':ReceptorConductanceMembrane:V'],b['states__'+pop+':ReceptorConductanceMembrane:V']
                xa,xb=a['states__'+pop+':ReceptorConductanceMembrane:x'],b['states__'+pop+':ReceptorConductanceMembrane:x']
                Bv=budgets[pop+':ReceptorConductanceMembrane:V'][selected];Bx=budgets[pop+':ReceptorConductanceMembrane:x'][selected]
                bound=np.empty_like(av);exact=True
                for i in range(len(av)):
                    for j in range(len(selected)):
                        r=compare_cell(pool=float(p['Npool']),U=float(p['U']),rmax=float(p['rmax']),r_half=float(p['r_half']),r_slope=float(p['r_slope']),
                            V_reference=float(va[i,j]),V_candidate=float(vb[i,j]),x_reference=float(xa[i,j]),x_candidate=float(xb[i,j]),rate_reference=float(av[i,j]),rate_candidate=float(bv[i,j]),B_V=float(Bv[j]),B_x=float(Bx[j]))
                        bound[i,j]=int(r['allowance_exact'][0])/int(r['allowance_exact'][1]);exact=exact and r['passed']
                rows.append(dict(field=name,maximum_norm=float(np.max(abs(av-bv)/bound,initial=0.)),passed=exact,gate='original_exact_ribbon_bound'));continue
            else:raise ContractError('Unregistered output comparison unit: '+name+' '+f['unit'])
            norm=float(np.max(np.abs(av-bv)/bound,initial=0.));rows.append(dict(field=name,maximum_norm=norm,passed=norm<=1,scalars=av.size))
    full=[]
    for suffix in ('mid','final'):
        with np.load(coarse/('full_state_'+suffix+'.npz'),allow_pickle=False) as a,np.load(refined/('full_state_'+suffix+'.npz'),allow_pickle=False) as b:
            for k,budget in budgets.items():
                norm=float(np.max(np.abs(a[k]-b[k])/budget,initial=0.));full.append(dict(point=suffix,state=k,maximum_norm=norm,passed=norm<=1))
    restore=None
    if restored:
        ia=read(coarse/'model_identity.json');ib=read(restored/'model_identity.json')
        if ia!=ib:raise ContractError('Restore numerical arm/source identity changed')
        a=load_bundle(coarse/'final_checkpoint',expected_kind='complete_model_checkpoint',expected_identity=ia)['payload']
        b=load_bundle(restored/'final_checkpoint',expected_kind='complete_model_checkpoint',expected_identity=ib)['payload']
        equality={k:causal_hash(v)==causal_hash(b[k]) for k,v in a.items()}
        with np.load(coarse/'core_traces.npz',allow_pickle=False) as complete,np.load(coarse/'core_prefix.npz',allow_pickle=False) as prefix,np.load(restored/'core_suffix.npz',allow_pickle=False) as suffix:
            trace_equal={k:np.array_equal(complete[k],np.concatenate((prefix[k],suffix[k]),axis=0)) for k in complete.files}
        restore=dict(passed=all(equality.values()) and all(trace_equal.values()),causal_fields=equality,core_trace_fields=trace_equal,
            complete_causal_sha256=causal_hash(a),restored_causal_sha256=causal_hash(b))
    result=dict(status='AC_TIMESCALE_COMPARISON_COMPLETE',numerical_passed=all(r['passed'] for r in rows+full),
        core_fields=rows,sparse_all_state_checks=full,restore=restore,
        maximum_norm=max((r['maximum_norm'] for r in rows+full),default=0.),
        scope='core trajectory and all-state sparse points; dynamic defects/domains cover full states; unchanged native reference not independently half-refined',FIT_READY=False)
    write(output/'result.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','run','compare'));p.add_argument('--output',required=True)
    p.add_argument('--prepared');p.add_argument('--arm',choices=('coarse','refined'),default='coarse');p.add_argument('--resume');p.add_argument('--coarse');p.add_argument('--refined');p.add_argument('--restored')
    args=p.parse_args();output=Path(args.output).resolve()
    if not output.is_relative_to(OUT) or output.exists():raise ContractError('Fresh independent timescale report output required')
    output.mkdir(parents=True)
    try:
        if args.mode=='prepare':result=prepare(output)
        elif args.mode=='run':result=run(args.prepared,output,args.arm,resume=args.resume)
        else:result=compare(args.prepared,Path(args.coarse),Path(args.refined),output,Path(args.restored) if args.restored else None)
        print(json.dumps(dict(status=result['status'],passed=result.get('passed',result.get('numerical_passed')),output=str(output))))
    except Exception as exc:
        write(output/'failure.json',dict(error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
