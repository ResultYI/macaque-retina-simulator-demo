"""One per-type candidate on the unchanged complete graph, with own gray fork.

Uses the existing ON-parasol old-parameter run as the frozen reference. No
cross-parameter checkpoint is restored. Extra chain recording changes no
equation, clock, RNG, graph, optical input or production default.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.rgc_defaults import historical_project_default,historical_reference_file
from tools.run_on_parasol_candidate_network import (
    RunConfig,atlas_population_network,read_source_candidate,merge_parameter_package,
    HybridRunner,constant_background_channels,RecordingPlan,MemoryGuard,record_segment,
    drive_channels,save,sha,seed_background_state,balance_rgc_initial_voltages,SolverConfig,canonical_hash,_plain)
from retina_sim.contracts import ContractError
from retina_sim.rgc_expectation import exact_history_rates,validate_sampler
from retina_sim.source_reference_candidate import CROSS_TYPE_TARGETS

BASE=ROOT/'reports/modeling_scope_20261020'
REFERENCE=BASE/'on_parasol_candidate_network_v1'
RGCS=('midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON')


def chain_ports(compiled):
    ports=[cid+'.'+p for cid in RGCS for p in ('voltage','gE','gI','conditional_rate','spike_count','window_rate')]
    for cid,c in compiled.components.items():
        if cid=='outer_native':names=('Vis','release','q','Vh')
        elif '__contact_occupancy_' in cid:names=('occupied',)
        elif cid in ('midget_BC_ON','midget_BC_OFF','diffuse_BC_ON','diffuse_BC_OFF','S_ON_BC',
                     'local_feedforward_AC','local_crossover_AC','widefield_AC'):
            names=('activation','voltage','release','receptor_gate')
        else:continue
        ports.extend(cid+'.'+p for p in names if p in c.outputs)
    return tuple(ports)


def load_arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def analyze_rates(compiled,trace,aux,folder):
    arrays={};checks={}
    for cid in RGCS:
        n=validate_sampler(compiled,cid)
        rates=exact_history_rates(n,trace[cid+'.voltage'],trace['time'],aux[cid])
        for key,value in rates.items():
            if isinstance(value,np.ndarray):arrays[cid+'.'+key]=value
        checks[cid]=dict(maximum_probability_mass_error=rates['maximum_probability_mass_error'],
            native_bin_start_grid=True,spike_independent_voltage=True,stationary_history_claim=False,
            first_bin_conditioned_on_saved_draw=True)
    np.savez_compressed(folder/'four_rate_expectations.npz',**arrays)
    save(folder/'expectation_contract.json',checks)
    return arrays


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--component',choices=CROSS_TYPE_TARGETS,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();d=args.output.resolve()
    if not d.is_relative_to(ROOT):raise ContractError('Experimental directory must remain in workspace')
    d.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    pins=json.loads((REFERENCE/'protected_before.json').read_bytes())
    for name,h in pins.items():
        if sha(historical_reference_file(ROOT,name,h))!=h:raise ContractError('Protected model/GUI/historical evidence changed: '+name)
    save(d/'protected_before.json',pins)
    recipe,path=read_source_candidate(ROOT,ROOT/'configurations'/(args.component+'_literature_candidate_v1.json'))
    save(d/'candidate_recipe.json',recipe)
    report=dict(status='ASSEMBLY',component=args.component,candidate_sha256=sha(path),
        production_default_changed=False,physiology_calibrated=False,matlab_launches=0,
        prior_ON_parasol_results_reused=True,fully_adapted=False,neural_seconds=0.,
        preparation_seconds=.1,stimulus_seconds=.5,gray_control_seconds=.5)
    save(d/'result.json',report);guard=MemoryGuard(4*1024**3,2*1024**3)
    def boundary(*unused):
        if not guard.check() or time.perf_counter()-start>1800 or (d/'STOP').exists():
            raise ContractError('Per-type short run reached explicit 1800s/4GiB/available-memory/STOP boundary')
    class Runner(HybridRunner):
        def advance(self,*args,**kwargs):
            boundary();kwargs.setdefault('state_observer',boundary)
            return super().advance(*args,**kwargs)
    try:
        cfg=historical_project_default(ROOT)
        original,_,audit=atlas_population_network(ROOT,apply_rgc_output_defaults=False,duration=.6,light_context=cfg.light_context(),
            execution_profile=cfg.execution_profile,configuration_path=cfg.circuit_configuration_path,
            owner_bounds_mm=cfg.owner_bounds(),halo_mm=cfg.halo_mm,max_spatial_cells=cfg.max_spatial_cells,
            seed=cfg.seed,dynamics_candidate_path=ROOT/recipe['base_dynamics_candidate']['path'])
        prior=json.loads((REFERENCE/'result.json').read_bytes())
        if original.assembly_hash!=prior['variants']['old']['assembly_hash']:
            raise ContractError('Frozen old-parameter full-network reference differs from current assembly')
        changed=merge_parameter_package(original,audit['registry'],recipe,applicability=recipe['applicability'],
            allowed_fields=['alpha','mu','beta'],proposal_path=path)
        assert canonical_hash(original.spec.links)==canonical_hash(changed.spec.links)
        assert canonical_hash(original.spec.gap_junctions)==canonical_hash(changed.spec.gap_junctions)
        report.update(assembly_hash=changed.assembly_hash,old_assembly_hash=original.assembly_hash,
            counts=prior['counts'],links_and_gaps_unchanged=True,merge=changed.parameter_package_merge)
        save(d/'actual_RGC_parameters.json',{cid:{key:np.asarray(getattr(changed.components[cid].native_model,key)).tolist()
            for key in ('C','gL','EL','EE','EI','alpha','mu','beta','history_kernel','dt')} for cid in RGCS})
        bg=constant_background_channels(changed,cfg.light_context(),.6,interpolation='zero_order_hold')
        runner=Runner(changed,compute_backend='python')
        state=runner.initialize(SolverConfig(method='DOP853',rtol=1e-7,atol=1e-10,max_step=.0001,input_interpolation='zero_order_hold'))
        seed=seed_background_state(runner,state,bg)
        state,balance=balance_rgc_initial_voltages(runner,seed.state,bg)
        save(d/'initialization.json',dict(seed=seed.audit,RGC_balance=balance,fresh_state=True,
            old_checkpoint_reused=False,fully_adapted=False))
        ports=chain_ports(changed);plan=RecordingPlan(states=(),ports=ports,max_bytes=128*1024**2)
        state,_,warm=record_segment(runner,state,bg,.1,d/'wash_in',plan,label=args.component+'_WASH_IN')
        prepared=runner.checkpoint(state);(d/'prepared_checkpoint.json').write_text(prepared,encoding='utf8')
        aux=json.loads(prepared)['payload']['spike_states'];report['neural_seconds']=.1
        channels=drive_channels(bg,.1,recipe['input_protocol'])
        save(d/'input_contracts.json',{k:v.input_contract for k,v in channels.items()})
        save(d/'recorded_ports.json',{cid+'.'+k:dict(unit=q.unit,meaning=q.meaning,cell_ids=list(q.cell_ids))
            for cid,c in changed.components.items() for k,q in c.outputs.items() if cid+'.'+k in ports})
        state,stim,smeta=record_segment(runner,state,channels,.6,d/'stimulus',plan,label=args.component+'_STIMULUS')
        final=runner.checkpoint(state);(d/'final_checkpoint.json').write_text(final,encoding='utf8')
        prior_final=json.loads((REFERENCE/'old/final_checkpoint.json').read_bytes())['payload']['dynamic']['state']['ordered_state_arrays']
        continuous_equal=all(np.array_equal(a,np.asarray(b)) for a,b in zip(state.ordered_state_arrays,prior_final))
        if not continuous_equal:raise ContractError('Continuous network final states differ from frozen unchanged dynamics')
        expected_stim=analyze_rates(changed,stim,aux,d/'stimulus')
        old=load_arrays(REFERENCE/'old/stimulus/traces.npz')
        equality={cid:{port:bool(np.array_equal(stim[cid+'.'+port],old[cid+'.'+port]))
            for port in ('voltage','gE','gI','conditional_rate','spike_count','window_rate')} for cid in RGCS}
        if not all(all(row.values()) for cid,row in equality.items() if cid!=args.component):
            raise ContractError('An unmodified RGC output differs from the frozen whole-network reference')
        if not all(equality[args.component][k] for k in ('voltage','gE','gI')):
            raise ContractError('Candidate changed graded state or conductance')
        report.update(neural_seconds=.6,status='STIMULUS_COMPLETE',continuous_final_states_bitwise_equal=True,
            signal_equality=equality,wash_in_wall_seconds=warm['wall_seconds'],stimulus_wall_seconds=smeta['wall_seconds'])
        save(d/'result.json',report)
        state=runner.restore(prepared)
        if runner.checkpoint(state)!=prepared:raise ContractError('Own candidate prepared-state restore is not exact')
        state,gray,gmeta=record_segment(runner,state,bg,.6,d/'matched_gray',plan,label=args.component+'_MATCHED_GRAY')
        (d/'matched_gray/final_checkpoint.json').write_text(runner.checkpoint(state),encoding='utf8')
        expected_gray=analyze_rates(changed,gray,aux,d/'matched_gray')
        old_gray_final=json.loads((REFERENCE/'old/matched_gray/final_checkpoint.json').read_bytes())['payload']['dynamic']['state']['ordered_state_arrays']
        if not all(np.array_equal(a,np.asarray(b)) for a,b in zip(state.ordered_state_arrays,old_gray_final)):
            raise ContractError('Gray continuous final states differ from frozen unchanged dynamics')
        responses={}
        for cid in RGCS:
            key=cid+'.expected_binary_count_rate_Hz';dv=(stim[cid+'.voltage']-gray[cid+'.voltage'])*1000
            dr=expected_stim[key]-expected_gray[key]
            np.testing.assert_allclose(dv[:,:100],0.,atol=1e-9,rtol=0)
            np.testing.assert_allclose(dr[:,:100],0.,atol=1e-8,rtol=0)
            phases={}
            for j,phase in enumerate(('gray_before','increment','gray_middle','decrement','gray_after')):
                sl=slice(j*100,(j+1)*100);left=round(.1+j*.1,10);right=round(left+.1,10)
                observed=sum(left<=e['time_seconds']<right for e in smeta['spike_events'][cid])/(dv.shape[0]*.1)
                phases[phase]=dict(mean_voltage_delta_mV=float(dv[:,sl].mean()),
                    mean_expected_count_rate_delta_Hz=float(dr[:,sl].mean()),
                    expected_gray_count_rate_Hz=float(expected_gray[key][:,sl].mean()),
                    actual_one_seed_count_rate_Hz=observed,
                    zero_history_hazard_Hz=float(expected_stim[cid+'.zero_history_hazard_Hz'][:,sl].mean()),
                    mean_history_hazard_Hz=float(expected_stim[cid+'.mean_history_hazard_Hz'][:,sl].mean()),
                    expected_binary_count_rate_Hz=float(expected_stim[key][:,sl].mean()),
                    recorded_conditional_hazard_Hz=float(stim[cid+'.conditional_rate'][:,sl].mean()))
            responses[cid]=dict(phases=phases,peak_abs_expected_delta_Hz=float(abs(dr.mean(axis=0)).max()))
        save(d/'response_summary.json',responses)
        for name,h in pins.items():
            if sha(historical_reference_file(ROOT,name,h))!=h:raise ContractError('Protected file changed during candidate run: '+name)
        report.update(status='COMPLETE_PER_TYPE_CANDIDATE_UNADOPTED',neural_seconds=1.1,
            gray_control_wall_seconds=gmeta['wall_seconds'],gray_continuous_states_bitwise_equal=True,
            own_checkpoint_roundtrip_exact=True,protected_files_unchanged=len(pins),
            wall_seconds=time.perf_counter()-start,maximum_RSS_bytes=guard.peak_rss_bytes,
            output_mapping_pass=True,default_adoption=False,decision='RETAIN_CANDIDATE_ENTER_B2',
            interpretation='Short engineering functional comparison; no matched physiological amplitude criterion')
        save(d/'result.json',report)
        print(json.dumps(dict(stage=report['status'],component=args.component,wall_seconds=report['wall_seconds'],
            target=responses[args.component])),flush=True)
    except Exception as e:
        report.update(status='STOPPED_EXACT_BLOCKER',error_type=type(e).__name__,error=str(e),wall_seconds=time.perf_counter()-start)
        save(d/'result.json',report);raise


if __name__=='__main__':main()
