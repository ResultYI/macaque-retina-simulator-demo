"""Matched short actual-network gray controls using each variant's own state."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.rgc_defaults import historical_project_default,historical_reference_file
from tools.run_on_parasol_candidate_network import (
    read_source_candidate,RunConfig,atlas_population_network,merge_parameter_package,
    HybridRunner,constant_background_channels,RecordingPlan,MemoryGuard,record_segment,expectation,save,sha)
from retina_sim.contracts import ContractError


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True)
    args=p.parse_args();d=args.case.resolve();started=time.perf_counter()
    report=json.loads((d/'result.json').read_bytes())
    if not report['status'].startswith('COMPLETE_'):raise ContractError('Completed stimulus variants required before matched controls')
    if (d/'matched_control_result.json').exists():raise ContractError('Preserve completed or partial control attempt')
    summary=dict(status='ASSEMBLY',controls={},new_neural_seconds=0.,checkpoint_scope='Each variant restores only its own exact prepared checkpoint',
        fully_adapted=False,production_default_changed=False,matlab_launches=0,permission_denials=0)
    save(d/'matched_control_result.json',summary)
    guard=MemoryGuard(4*1024**3,1024**3)
    def boundary(*unused):
        if not guard.check() or time.perf_counter()-started>600 or (d/'STOP').exists():
            raise ContractError('Matched short-control 600s/4GiB/STOP boundary reached')
    class BudgetRunner(HybridRunner):
        def advance(self,*args,**kwargs):
            boundary();kwargs.setdefault('state_observer',boundary)
            return super().advance(*args,**kwargs)
    try:
        cfg=historical_project_default(ROOT)
        recipe,path=read_source_candidate(ROOT,ROOT/'configurations/on_parasol_literature_candidate_v1.json')
        if sha(path)!=report['candidate_sha256']:raise ContractError('Candidate recipe differs from actual stimulus run')
        original,_,audit=atlas_population_network(ROOT,apply_rgc_output_defaults=False,duration=.6,light_context=cfg.light_context(),
            execution_profile=cfg.execution_profile,configuration_path=cfg.circuit_configuration_path,
            owner_bounds_mm=cfg.owner_bounds(),halo_mm=cfg.halo_mm,max_spatial_cells=cfg.max_spatial_cells,
            seed=cfg.seed,dynamics_candidate_path=ROOT/recipe['base_dynamics_candidate']['path'])
        candidate=merge_parameter_package(original,audit['registry'],recipe,applicability=recipe['applicability'],
            allowed_fields=['alpha','mu','beta'],proposal_path=path)
        states={};all_responses={}
        for name,compiled in [('old',original),('candidate',candidate)]:
            if compiled.assembly_hash!=report['variants'][name]['assembly_hash']:raise ContractError('Actual assembly identity differs')
            rgcs=sorted(cid for cid,c in compiled.components.items() if c.mechanism=='HistoryConductanceRGC')
            ports=tuple(cid+'.'+port for cid in rgcs for port in ('voltage','gE','gI','conditional_rate','spike_count','window_rate'))
            plan=RecordingPlan(states=(),ports=ports,max_bytes=64*1024**2)
            cp=(d/name/'prepared_checkpoint.json').read_text(encoding='utf-8')
            runner=BudgetRunner(compiled,compute_backend='python');state=runner.restore(cp)
            assert runner.checkpoint(state)==cp
            aux=json.loads(cp)['payload']['spike_states']
            bg=constant_background_channels(compiled,cfg.light_context(),.6,interpolation='zero_order_hold')
            state,control,segment=record_segment(runner,state,bg,.6,d/name/'matched_gray',plan,label=name+'_MATCHED_GRAY')
            (d/name/'matched_gray/final_checkpoint.json').write_text(runner.checkpoint(state),encoding='utf-8')
            states[name]=[a.copy() for a in state.ordered_state_arrays]
            with np.load(d/name/'stimulus/traces.npz',allow_pickle=False) as z:stim={k:z[k] for k in z.files}
            with np.load(d/name/'exact_expected_count_rates.npz',allow_pickle=False) as z:expected_stim={k:z[k] for k in z.files}
            with (d/name/'stimulus/segment.json').open(encoding='utf-8') as f:stim_meta=json.load(f)
            expected_control={};responses={};delta_arrays={}
            for cid in rgcs:
                native=compiled.components[cid].native_model
                expected_control[cid]=expectation(native,control[cid+'.voltage'],aux[cid])
                dV=1000*(stim[cid+'.voltage']-control[cid+'.voltage'])
                dRate=expected_stim[cid]-expected_control[cid]
                delta_arrays[cid+'.voltage_delta_mV']=dV
                delta_arrays[cid+'.expected_rate_delta_Hz']=dRate
                np.testing.assert_allclose(dV[:,:100],0.,atol=1e-9,rtol=0)
                np.testing.assert_allclose(dRate[:,:100],0.,atol=1e-8,rtol=0)
                phases={}
                for j,phase in enumerate(['gray_before','increment','gray_middle','decrement','gray_after']):
                    sl=slice(j*100,(j+1)*100);left=round(.1+j*.1,10);right=round(left+.1,10)
                    counts={}
                    for tag,meta in [('stimulus',stim_meta),('gray',segment)]:
                        counts[tag]=sum(left<=e['time_seconds']<right for e in meta['spike_events'][cid])/(len(native.alpha)*.1)
                    phases[phase]=dict(mean_voltage_delta_mV=float(dV[:,sl].mean()),
                        mean_expected_count_rate_delta_Hz=float(dRate[:,sl].mean()),
                        one_seed_observed_rate_delta_Hz=counts['stimulus']-counts['gray'],
                        mean_expected_gray_count_rate_Hz=float(expected_control[cid][:,sl].mean()),
                        one_seed_gray_count_rate_Hz=counts['gray'])
                responses[cid]=dict(phases=phases,peak_abs_voltage_delta_mV=float(np.max(abs(dV.mean(axis=0)))),
                    peak_abs_expected_count_rate_delta_Hz=float(np.max(abs(dRate.mean(axis=0)))))
            delta_arrays['time_seconds']=stim['time']
            np.savez_compressed(d/name/'matched_gray/exact_expected_count_rates.npz',**expected_control)
            np.savez_compressed(d/name/'matched_gray/matched_response_deltas.npz',**delta_arrays)
            save(d/name/'matched_gray/response_summary.json',responses)
            summary['controls'][name]=dict(assembly_hash=compiled.assembly_hash,prepared_checkpoint_sha256=sha(d/name/'prepared_checkpoint.json'),
                restored_on_own_exact_assembly=True,checkpoint_roundtrip_exact=True,neural_seconds=.5,
                wall_seconds=segment['wall_seconds'],all_finite=True)
            summary['new_neural_seconds']+=.5;all_responses[name]=responses
            save(d/'matched_control_result.json',summary)
        summary.update(status='COMPLETE_PAIRED_GRAY_CONTROLS',responses=all_responses,
            gray_final_continuous_states_bitwise_equal=all(np.array_equal(a,b) for a,b in zip(states['old'],states['candidate'])),
            wall_seconds=time.perf_counter()-started,
            total_experiment_neural_seconds=report['new_neural_seconds']+summary['new_neural_seconds'],
            observed_rate_scope='One seed; expected rates integrate only future history conditional on each saved prepared history',
            physiology_calibrated=False,matched_background_subtraction=True)
        pins=json.loads((d/'protected_before.json').read_bytes())
        for f,h in pins.items():assert sha(historical_reference_file(ROOT,f,h))==h,f
        summary['protected_files_unchanged']=len(pins)
        save(d/'matched_control_result.json',summary)
        print(json.dumps(dict(stage=summary['status'],ON_parasol={name:all_responses[name]['parasol_RGC_ON'] for name in all_responses},
                             wall_seconds=summary['wall_seconds'])),flush=True)
    except Exception as e:
        summary.update(status='STOPPED_EXACT_BLOCKER',error_type=type(e).__name__,error=str(e),wall_seconds=time.perf_counter()-started)
        save(d/'matched_control_result.json',summary);raise


if __name__=='__main__':main()
