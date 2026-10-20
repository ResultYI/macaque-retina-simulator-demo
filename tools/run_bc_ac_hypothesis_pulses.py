"""Fixed-parameter input-scale pulses on the complete engineering feedback graph.

Each light consumer is perturbed in its own units. The common fraction/schedule
is a software experiment, not a calibrated visual contrast or common scene.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,hashlib,sys,time,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retina_sim.contracts import RunMode,Quantity,canonical_hash,ContractError
from retina_sim.assembly import RetinaBuilder
from retina_sim.hybrid import HybridRunner
from retina_sim.network import InputBundle
from retina_sim.runtime_profiles import LightContext
from retina_sim.recording import RecordingPlan
from retina_sim.bc_ac_candidate import CandidateProfile,migrate_bc_ac
from retina_sim.bc_ac_hypothesis import constant_background_channels,prepare_feedback_initial_state
from examples.full_target_network import full_target_network


def pulse_channels(background,*,fraction,start,duration,stop):
    """Explicit step in each consumer's own rate/illuminance domain."""
    result={}
    for name,ch in background.items():
        base=ch.at(0.)
        def value(t,base=base,fraction=fraction):
            f=1+fraction if start<=t<start+duration else 1.
            return Quantity(base.values*f,base.unit,base.meaning,base.evidence_tag)
        result[name]=replace(ch,start_seconds=0.,stop_seconds=stop,evaluator=value,
            breakpoints_seconds=(float(start),float(start+duration)),
            input_contract=dict(ch.input_contract,declared_constant_background=fraction==0,
                pulse_fraction=fraction,pulse_start_seconds=start,pulse_duration_seconds=duration,
                perturbation_domain='independent model consumer units',calibrated_visual_contrast=False,
                common_physical_scene_claim=False,consumer_unit=base.unit,base_values=base.values.tolist()))
    return result


def signal_stats(compiled,state,runner,channels):
    y=np.concatenate(state.ordered_state_arrays)
    _,out=runner.model.evaluate(state.time_seconds,y,InputBundle({k:v.at(state.time_seconds) for k,v in channels.items()}))
    names=('local_feedforward_AC','local_crossover_AC','widefield_AC')
    return {cid:{port:dict(mean=float(out[(cid,port)].values.mean()),minimum=float(out[(cid,port)].values.min()),
        maximum=float(out[(cid,port)].values.max())) for port in ('voltage','activation','receptor_gate')} for cid in names}


def record_segment(runner,state,channels,stop,folder,plan,*,sample_step=.001,chunk=.025,label):
    folder.mkdir(parents=True,exist_ok=False);begin=state.time_seconds;started=time.perf_counter()
    all_times=np.linspace(begin,stop,int(round((stop-begin)/sample_step))+1)
    pieces=[];counts={};events={};declarations={};index=0
    while state.time_seconds<stop-1e-12:
        end=min(stop,begin+(index+1)*chunk)
        times=all_times[(all_times>=state.time_seconds-1e-12)&(all_times<=end+1e-12)]
        # Align requested endpoint stamps within the existing 1e-12 recording
        # tolerance. This changes no event/integration endpoint or state value.
        times=np.clip(times,state.time_seconds,end)
        state,r=runner.advance(state,channels,end,state.solver_config,sample_times=times.tolist(),recording_plan=plan)
        arrays={k:q.values for k,q in r.outputs_with_units.items()}
        np.savez_compressed(folder/f'chunk_{index:03d}.npz',**arrays)
        pieces.append(arrays)
        for k,q in r.outputs_with_units.items():declarations[k]=dict(unit=q.unit,meaning=q.meaning)
        for cid,ev in r.metadata['spike_events'].items():events.setdefault(cid,[]).extend(ev)
        for k,a in arrays.items():
            if k.endswith('.spike_count'):counts[k]=counts.get(k,np.zeros_like(a))+a
        (folder/'state_latest.json').write_text(runner.checkpoint(state),encoding='utf8')
        index+=1
        print(json.dumps(dict(stage=label,relative_ms=round(1000*(state.time_seconds-begin),4),
            wall_seconds=round(time.perf_counter()-started,3),nfev=runner.integration_metrics['nfev'])),flush=True)
    joined={}
    for k in pieces[0]:
        if k.endswith('.spike_count'):joined[k]=counts[k]
        elif k.endswith('.window_rate'):joined[k]=counts[k.replace('.window_rate','.spike_count')]/(stop-begin)
        else:
            axis=0 if k=='time' else 1
            joined[k]=np.concatenate([a[k] if i==0 else a[k][1:] if axis==0 else a[k][:,1:]
                for i,a in enumerate(pieces)],axis=axis)
    if len(joined['time'])!=len(all_times) or not np.allclose(joined['time'],all_times,atol=1e-12,rtol=0):
        raise ContractError('Intermediate sample coverage changed')
    if not all(np.all(np.isfinite(a)) for a in joined.values()):raise ContractError('Nonfinite recorded output')
    np.savez_compressed(folder/'traces.npz',**joined)
    report=dict(status='SEGMENT_COMPLETE',start_seconds=begin,stop_seconds=stop,wall_seconds=time.perf_counter()-started,
        samples=len(all_times),sample_step_seconds=sample_step,all_finite=True,output_units=declarations,
        spike_events=events,solver_metrics=runner.integration_metrics,
        observed_spike_counts_are_one_seed_not_ensemble=True)
    (folder/'segment.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    return state,joined,report


def compare(background,trace,rgcs):
    t=trace['time']-trace['time'][0];summary={}
    for cid in rgcs:
        rows={}
        for port in ('voltage','conditional_rate','gE','gI'):
            key=cid+'.'+port;delta=trace[key]-background[key];mean=delta.mean(axis=0)
            peak=int(np.argmax(np.abs(mean)))
            at={str(ms):float(mean[np.argmin(np.abs(t-ms/1000))]) for ms in (0,5,10,25,50,75,100,150,200)}
            rows[port]=dict(background_subtracted_mean_at_ms=at,peak_mean=float(mean[peak]),peak_time_ms=float(t[peak]*1000),
                end_mean=float(mean[-1]),maximum_absolute_cell_delta=float(np.max(np.abs(delta))),
                final_20ms_mean=float(mean[t>=.18-1e-12].mean()),
                end_to_peak_ratio=float(abs(mean[-1])/abs(mean[peak])) if abs(mean[peak])>1e-15 else None,
                sign_at_50ms='positive' if at['50']>0 else 'negative' if at['50']<0 else 'zero')
        rows['observed_spikes']=dict(background_total=int(background[cid+'.spike_count'].sum()),
            pulse_total=int(trace[cid+'.spike_count'].sum()),not_statistical_validation=True)
        summary[cid]=rows
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume-prepared',action='store_true',help='Reuse the saved identical prepared checkpoint; preserve any incomplete branch directory')
    parser.add_argument('--export-layout-only',action='store_true',help='Reconstruct and verify the recorded cell/port layout; no neural advance')
    args=parser.parse_args()
    if args.resume_prepared or args.export_layout_only:
        if not (args.output/'prepared_state.json').is_file():parser.error('Explicit saved prepared state required')
    else:args.output.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter()
    profile=CandidateProfile(ROOT/'configurations/bc_ac_contact_first_hypothesis_v1.json');data=profile.data['background_spec']
    light=LightContext(**{k:v for k,v in data.items() if k!='common_physical_scene_claim'})
    old,_,audit=full_target_network(ROOT,effective_cones=64,effective_rods=64,duration=.3,
        runtime_profile='effective_dynamics_v1',light_context=light)
    old=RetinaBuilder(audit['registry']).build(old.spec,RunMode.REFERENCE)
    bg=constant_background_channels(old,light,stop=.3)
    new,registry,migration=migrate_bc_ac(old,audit['registry'],profile,background_inputs={k:v.at(0.) for k,v in bg.items()})
    frozen_parameters=canonical_hash({k:c.parameters.parameter_hash for k,c in new.components.items()})
    if args.export_layout_only:
        check=HybridRunner(new,compute_backend='python')
        check.restore((args.output/'prepared_state.json').read_text(encoding='utf8'))
        layout={}
        for cid,c in new.components.items():
            for name,p in c.outputs.items():layout[cid+'.'+name]=dict(cell_ids=list(p.cell_ids),unit=p.unit,meaning=p.meaning)
            if c.mechanism=='HistoryConductanceRGC':
                for name,unit,meaning in (('conditional_rate','Hz','conditional_spike_intensity'),
                        ('spike_count','dimensionless','observed_window_spike_count'),('window_rate','Hz','observed_window_firing_rate')):
                    layout[cid+'.'+name]=dict(cell_ids=list(c.state_specs[0].cell_ids),unit=unit,meaning=meaning)
        with (args.output/'recorded_layout.json').open('x',encoding='utf8') as f:
            json.dump(dict(assembly_hash=new.assembly_hash,parameter_hash=frozen_parameters,
                checkpoint_identity_verified=True,neural_advance=False,ports=layout),f,ensure_ascii=False,indent=2)
        print(json.dumps(dict(stage='RECORDED_LAYOUT_VERIFIED_NO_NEURAL_ADVANCE',wall_seconds=time.perf_counter()-began)),flush=True)
        return
    if args.resume_prepared:
        root_runner=HybridRunner(new,compute_backend='python')
        state=root_runner.restore((args.output/'root_state.json').read_text(encoding='utf8'))
        initialization=json.loads((args.output/'initialization.json').read_bytes())
    else:
        state,initialization,root_runner=prepare_feedback_initial_state(new,bg,max_unknown_scalars=1536)
        (args.output/'root_state.json').write_text(root_runner.checkpoint(state),encoding='utf8')
        (args.output/'initialization.json').write_text(json.dumps(initialization,indent=2),encoding='utf8')
    root_points=signal_stats(new,state,root_runner,bg)
    anchors={cid:new.components[cid].parameters.conditions['background_context']['output_anchor'] for cid in root_points}
    print(json.dumps(dict(stage='INITIAL_BACKGROUND_SOLVED',wall_seconds=time.perf_counter()-began,AC_reference=anchors,
        AC_solved={cid:p['activation']['mean'] for cid,p in root_points.items()})),flush=True)
    targets=migration['targets'];rgcs=[cid for cid,c in new.components.items() if c.mechanism=='HistoryConductanceRGC']
    ports=[]
    for cid in targets:ports += [cid+'.'+p for p in ('voltage','receptor_gate','activation')]+([cid+'.release'] if cid.endswith('BC_ON') or cid.endswith('BC_OFF') or cid=='S_ON_BC' else [])
    for cid in rgcs:ports += [cid+'.'+p for p in ('voltage','conditional_rate','gE','gI','spike_count','window_rate')]
    plan=RecordingPlan(states=(),ports=tuple(ports),max_bytes=16*1024*1024)
    if args.resume_prepared:
        prepared_checkpoint=(args.output/'prepared_state.json').read_text(encoding='utf8')
        prepared=root_runner.restore(prepared_checkpoint)
        with np.load(args.output/'short_wash_in/traces.npz',allow_pickle=False) as z:warm={k:z[k].copy() for k in z.files}
        warm_report=json.loads((args.output/'short_wash_in/segment.json').read_bytes())
    else:
        prepared,warm,warm_report=record_segment(root_runner,state,bg,.1,args.output/'short_wash_in',plan,label='SHORT_BACKGROUND_WASH_IN')
        prepared_checkpoint=root_runner.checkpoint(prepared)
        (args.output/'prepared_state.json').write_text(prepared_checkpoint,encoding='utf8')
    checkpoint_hash=hashlib.sha256(prepared_checkpoint.encode()).hexdigest()
    prepared_points=signal_stats(new,prepared,root_runner,bg)
    branches={};reports={};start_hashes={};branch_folders={}
    for name,fraction in (('background',0.),('plus10',.1),('minus10',-.1)):
        runner=HybridRunner(new,compute_backend='python');branch_state=runner.restore(prepared_checkpoint)
        start_hashes[name]=hashlib.sha256(runner.checkpoint(branch_state).encode()).hexdigest()
        channels=pulse_channels(bg,fraction=fraction,start=.1,duration=.05,stop=.3)
        folder=args.output/name
        if folder.exists():
            if not args.resume_prepared:raise ContractError('Branch output already exists')
            if (folder/'segment.json').is_file():
                with np.load(folder/'traces.npz',allow_pickle=False) as z:traces={k:z[k].copy() for k in z.files}
                report=json.loads((folder/'segment.json').read_bytes())
                branches[name]=traces;reports[name]=report;branch_folders[name]=folder.name
                continue
            folder=args.output/(name+'_retry_01')
        _,traces,report=record_segment(runner,branch_state,channels,.3,folder,plan,label=name)
        branch_folders[name]=folder.name
        branches[name]=traces;reports[name]=report
        assert canonical_hash({k:c.parameters.parameter_hash for k,c in new.components.items()})==frozen_parameters
    if set(start_hashes.values())!={checkpoint_hash}:raise ContractError('Branches did not share exactly the same prepared state/RNG')
    comparison={name:compare(branches['background'],branches[name],rgcs) for name in ('plus10','minus10')}
    drift={cid:dict(root_to_wash_in_mV=float(1000*(warm[cid+'.voltage'][:,-1]-warm[cid+'.voltage'][:,0]).mean()),
        subsequent_control_200ms_mV=float(1000*(branches['background'][cid+'.voltage'][:,-1]-branches['background'][cid+'.voltage'][:,0]).mean()),
        final_wash_in_25ms_mV=float(1000*(warm[cid+'.voltage'][:,-1]-warm[cid+'.voltage'][:,-26]).mean())) for cid in rgcs}
    layer_effects={}
    for name in ('plus10','minus10'):
        layer_effects[name]={}
        for cid in targets:
            layer_effects[name][cid]={}
            for port in ('voltage','receptor_gate','activation'):
                v=branches[name][cid+'.'+port]-branches['background'][cid+'.'+port];mean=v.mean(axis=0)
                layer_effects[name][cid][port]=dict(peak_absolute_mean=float(np.max(np.abs(mean))),at_50ms=float(mean[50]),at_200ms=float(mean[-1]))
    result=dict(status='FIXED_HYPOTHESIS_PULSE_TEST_COMPLETE_NOT_PHYSIOLOGICAL_CALIBRATION',
        model_geometry='1070 engineering cell records; not actual frozen geometry validation',
        stimulus='Independent relative model-input pulses in each consumer unit; not calibrated visual contrast',
        input_consumer_contracts={k:v.input_contract for k,v in pulse_channels(bg,fraction=.1,start=.1,duration=.05,stop=.3).items()},
        pulse_ms=50,recovery_ms=150,short_background_wash_in_ms=100,fully_adapted=False,
        initialization=initialization,AC_reference_output=anchors,AC_after_initialization=root_points,AC_after_short_wash_in=prepared_points,
        RGC_initialization_drift=drift,matched_background_subtracted_RGC=comparison,matched_background_subtracted_layers=layer_effects,
        segments={k:{x:y for x,y in v.items() if x!='spike_events'} for k,v in reports.items()},
        start_checkpoint_hashes=start_hashes,parameter_hash=frozen_parameters,profile_sha256=profile.sha256,
        branch_folders=branch_folders,reused_saved_prepared_state=args.resume_prepared,
        parameters_changed=False,production_default_changed=False,all_feedback_connections_retained=True,
        gap_hash_original=canonical_hash(old.spec.gap_junctions),gap_hash_candidate=canonical_hash(new.spec.gap_junctions),
        counts=dict(components=len(new.components),populations=len(new.populations),cells=sum(len(p.cells.cell_id) for p in new.populations.values()),
            states=sum(s.shape[0] for c in new.components.values() for s in c.state_specs),biological_links=len(old.spec.links),gap_tables=len(old.spec.gap_junctions)),
        wall_seconds_total=time.perf_counter()-began,calibration_or_compensatory_tuning_performed=False,
        limitations=['One stochastic seed; no ensemble statistics','100ms wash-in is not proof of slow adaptation',
            'Full-field coordinated consumer perturbations do not isolate one retinal circuit','Assumed parameters remain uncalibrated'])
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(stage='ALL_PULSE_BRANCHES_COMPLETE',wall_seconds_total=result['wall_seconds_total'],all_finite=True)),flush=True)


if __name__=='__main__':main()
