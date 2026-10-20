"""Shortest shared-main integration check: full graph, 5ms, no optical launch."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_rgc_type_candidate import RunConfig,atlas_population_network,HybridRunner,constant_background_channels,seed_background_state,balance_rgc_initial_voltages,save,SolverConfig,RecordingPlan
from retina_sim.rgc_defaults import apply_rgc_defaults
from retina_sim.contracts import ContractError,canonical_hash


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-integration',type=Path,help='Reuse a completed integration only when both actual assembly identities still match')
    a=p.parse_args();d=a.output.resolve();d.mkdir(parents=True,exist_ok=False)
    cfg=RunConfig.project_default(ROOT)
    old,ch,audit=atlas_population_network(ROOT,duration=.005,light_context=cfg.light_context(),execution_profile=cfg.execution_profile,
        configuration_path=cfg.circuit_configuration_path,owner_bounds_mm=cfg.owner_bounds(),halo_mm=cfg.halo_mm,
        max_spatial_cells=cfg.max_spatial_cells,seed=cfg.seed,apply_rgc_output_defaults=False,apply_rod_reference_defaults=True)
    new,_,new_audit=apply_rgc_defaults((old,ch,audit),ROOT)
    targets=[r['component_id'] for r in new.accepted_rgc_output_defaults]
    declared=json.loads((ROOT/'configurations/retina_platform_v1.json').read_bytes())['rgc_output_defaults']['accepted_types']
    assert set(targets)==set(declared) and 'parasol_RGC_ON' in targets
    assert len(new.components)==len(old.components)==34
    assert canonical_hash(old.spec.links)==canonical_hash(new.spec.links)
    assert canonical_hash(old.spec.gap_junctions)==canonical_hash(new.spec.gap_junctions)
    for cid,c in old.components.items():
        assert new.components[cid].mechanism==c.mechanism
        if cid not in targets:assert canonical_hash(c.parameters.profile)==canonical_hash(new.components[cid].parameters.profile)
    assert new.components['diffuse_BC_ON'].mechanism=='HistoryBipolarTerminal'
    rgcs=[cid for cid,c in new.components.items() if c.mechanism=='HistoryConductanceRGC']
    if a.reuse_integration:
        prior_raw=a.reuse_integration.read_bytes();prior=json.loads(prior_raw)
        if (prior.get('status') in ('PASS_SHARED_MAIN_ON_PARASOL_ENGINEERING_DEFAULT_INTEGRATION','PASS_SHARED_MAIN_ACCEPTED_RGC_ENGINEERING_DEFAULT_INTEGRATION') and
                prior.get('historical_assembly')==old.assembly_hash and prior.get('shared_main_assembly')==new.assembly_hash and
                all(prior.get(k) is True for k in ('continuous_final_states_bitwise_equal','old_checkpoint_rejected','BC_AC_mechanisms_unchanged','links_and_gaps_unchanged'))):
            saved=json.loads((a.reuse_integration.parent/'actual_RGC_parameters.json').read_bytes())
            for name,compiled in [('historical_defaults',old),('shared_main',new)]:
                for cid in rgcs:
                    for key,value in saved[name][cid].items():np.testing.assert_array_equal(np.asarray(getattr(compiled.components[cid].native_model,key)),value)
            save(d/'result.json',dict(status='PASS_REUSED_IDENTICAL_SHARED_MAIN_INTEGRATION',components=34,
                continuous_final_states_bitwise_equal=True,old_checkpoint_rejected=True,BC_AC_mechanisms_unchanged=True,links_and_gaps_unchanged=True,
                actual_assembly_identities_match_completed_check=True,new_neural_seconds=0.,matlab_launches=0,
                reused_result_path=str(a.reuse_integration),reused_result_sha256=hashlib.sha256(prior_raw).hexdigest(),
                historical_assembly=old.assembly_hash,shared_main_assembly=new.assembly_hash,
                acceptance=new_audit['runtime_profile']['accepted_RGC_output_defaults'],physiological_calibration=False,fully_adapted=False))
            print('PASS_REUSED_IDENTICAL_SHARED_MAIN_INTEGRATION',flush=True);return
    plan=RecordingPlan(states=(),ports=tuple(c+'.'+p for c in rgcs for p in ('voltage','gE','gI','conditional_rate','spike_count','window_rate')),max_bytes=64*1024**2)
    results={};states={};checkpoints={};parameters={}
    for name,compiled in [('historical_defaults',old),('shared_main',new)]:
        runner=HybridRunner(compiled,compute_backend='python');state=runner.initialize(SolverConfig(method='DOP853',rtol=1e-7,atol=1e-10,max_step=.0001,input_interpolation='zero_order_hold'))
        bg=constant_background_channels(compiled,cfg.light_context(),.005,interpolation='zero_order_hold')
        seed=seed_background_state(runner,state,bg);state,_=balance_rgc_initial_voltages(runner,seed.state,bg);checkpoints[name]=runner.checkpoint(state)
        state,r=runner.advance(state,bg,.005,sample_times=np.arange(6)*.001,recording_plan=plan)
        results[name]={k:q.values for k,q in r.outputs_with_units.items()};states[name]=state
        np.savez_compressed(d/(name+'.npz'),**results[name])
        parameters[name]={cid:{k:np.asarray(getattr(compiled.components[cid].native_model,k)).tolist() for k in ('alpha','mu','beta','C','gL','EL','EE','EI','history_kernel','dt')} for cid in rgcs}
    assert all(np.array_equal(a,b) for a,b in zip(states['historical_defaults'].ordered_state_arrays,states['shared_main'].ordered_state_arrays))
    for cid in rgcs:
        ports=('voltage','gE','gI') if cid in targets else ('voltage','gE','gI','conditional_rate','spike_count','window_rate')
        for port in ports:assert np.array_equal(results['historical_defaults'][cid+'.'+port],results['shared_main'][cid+'.'+port])
    try:HybridRunner(new,compute_backend='python').restore(checkpoints['historical_defaults'])
    except ContractError:rejected=True
    else:raise AssertionError('Old default checkpoint was accepted across parameter/identity change')
    save(d/'actual_RGC_parameters.json',parameters)
    save(d/'result.json',dict(status='PASS_SHARED_MAIN_ACCEPTED_RGC_ENGINEERING_DEFAULT_INTEGRATION',components=34,
        continuous_final_states_bitwise_equal=True,unchanged_RGC_outputs_bitwise_equal=[cid for cid in rgcs if cid not in targets],old_checkpoint_rejected=rejected,
        BC_AC_mechanisms_unchanged=True,links_and_gaps_unchanged=True,matlab_launches=0,new_neural_seconds=.01,
        rod_reference=audit.get('rod_reference'),
        historical_assembly=old.assembly_hash,shared_main_assembly=new.assembly_hash,
        acceptance=new_audit['runtime_profile']['accepted_RGC_output_defaults'],physiological_calibration=False,
        fully_adapted=False,scope='5ms actual complete shared-main integration; earlier 42-component functional evidence remains separately scoped'))
    print('PASS_SHARED_MAIN_ACCEPTED_RGC_ENGINEERING_DEFAULT_INTEGRATION',flush=True)


if __name__=='__main__':main()
