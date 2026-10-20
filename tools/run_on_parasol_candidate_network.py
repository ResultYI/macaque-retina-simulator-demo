"""Actual full-network short comparison of one unadopted source-mapping delta.

One shared production configuration supplies geometry/dynamics. No optical
capture, incompatible checkpoint transfer, fit, or production-default change.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import copy
import csv
import hashlib
import json
import shutil
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_platform.config import RunConfig
from retina_sim.atlas_population_network import atlas_population_network
from retina_sim.source_reference_candidate import read_source_candidate
from retina_sim.calibration_modules import merge_parameter_package,_plain
from retina_sim.contracts import ContractError,SolverConfig,Quantity,canonical_hash
from retina_sim.hybrid import HybridRunner
from retina_sim.recording import RecordingPlan
from retina_sim.background import constant_background_channels
from retina_sim.background_seed import seed_background_state
from retina_sim.bc_ac_hypothesis import balance_rgc_initial_voltages
from retina_sim.background_checkpoint import MemoryGuard
from tools.run_bc_ac_hypothesis_pulses import record_segment
from tools.diagnose_bc_ac_hypothesis_baseline import STATES,INDEX,NEXT0,NEXT1,hazard

BASE=ROOT/'reports/modeling_scope_20261020'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def drive_channels(background,origin,protocol):
    phases=protocol['fractions'];duration=protocol['phase_seconds']
    edges=np.array([round(origin+j*duration,10) for j in range(len(phases)+1)])
    result={}
    for name,ch in background.items():
        base=ch.at(0.)
        def evaluate(t,base=base):
            j=min(max(int(np.searchsorted(edges,t,side='right')-1),0),len(phases)-1)
            return Quantity(base.values*(1+phases[j]),base.unit,base.meaning,base.evidence_tag)
        result[name]=replace(ch,evaluator=evaluate,breakpoints_seconds=tuple(edges[1:-1]),
            input_contract=dict(ch.input_contract,declared_constant_background=False,
                input_class='engineering_independent_consumer_fullfield_steps',fractions=phases,
                phase_edges_seconds=edges.tolist(),calibrated_visual_contrast=False,
                common_physical_scene_claim=False,source='Existing typed input perturbation route; no ISET capture'))
    return result


def expectation(native,V,aux):
    """Exact future-history mean on the actually recorded spike-independent V.

    The first prepared bin has already been drawn; keep that count fixed.
    Last recorded V is an endpoint, outside the observed count interval.
    """
    n,nt=V.shape;h=np.asarray(aux['history'],int);first=np.asarray(aux['last_bin_counts'],int)
    assert h.shape==(3,n) and first.shape==(n,)
    pi=np.zeros((n,8))
    for c in range(n):pi[c,INDEX[(int(first[c]),int(h[0,c]),int(h[1,c]))]]=1.
    rate=np.empty((n,nt-1));rate[:,0]=first/native.dt
    for j in range(1,nt-1):
        lam=hazard(V[:,j,None],(STATES@native.history_kernel)[None,:],native.alpha[0],native.mu[0],native.beta[0])
        q=-np.expm1(-native.dt*lam);rate[:,j]=(pi*q).sum(axis=1)/native.dt
        nxt=np.zeros_like(pi)
        for k in range(8):nxt[:,NEXT0[k]]+=pi[:,k]*(1-q[:,k]);nxt[:,NEXT1[k]]+=pi[:,k]*q[:,k]
        pi=nxt
    return rate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate',type=Path,default=ROOT/'configurations/on_parasol_literature_candidate_v1.json')
    p.add_argument('--resume-startup',action='store_true')
    args=p.parse_args();d=args.output.resolve()
    if not d.is_relative_to(ROOT):raise ContractError('Keep experimental output in workspace')
    if args.resume_startup:
        prior=json.loads((d/'result.json').read_bytes())
        if (prior['status']!='STOPPED_EXACT_BLOCKER' or not prior.get('error','').startswith('Recording requires at least')
            or prior.get('variants') or any(d.rglob('chunk_*.npz'))):
            raise ContractError('Startup-only recovery requires proof that no advance was recorded')
        save(d/'startup_recording_budget_failure.json',prior)
        empty=d/'old/wash_in'
        if empty.exists():empty.rmdir()  # Explicitly empty; never recursive removal.
    else:d.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter()
    frozen=json.loads((BASE/'standard_stimulus_rate_audit_v1/baseline_frozen_identity.json').read_bytes())
    pins={**frozen['files'],**frozen['fixed_parameters_and_equations']}
    pins.update(json.loads((BASE/'standard_stimulus_rate_audit_v1/scope_integrity.json').read_bytes())['source_sha256'])
    for f,h in pins.items():assert sha(ROOT/f)==h,f
    save(d/'protected_before.json',pins)
    recipe,path=read_source_candidate(ROOT,args.candidate);save(d/'candidate_recipe.json',recipe)
    cfg=RunConfig.project_default(ROOT)
    save(d/'main_configuration_reference.json',dict(path=recipe['base_main_configuration']['path'],
        sha256=recipe['base_main_configuration']['sha256'],run_defaults=cfg.to_dict(),
        experiment_overrides=dict(short_wash_in_seconds=.1,drive_seconds=.5,steady_state_claim=False,
                                  input='independent root consumer fullfield steps; no media capture')))
    report=dict(schema='on_parasol_source_candidate_full_network_v1',status='ASSEMBLY',variants={},
        candidate_sha256=sha(path),production_default_changed=False,GUI_changed=False,fit_performed=False,
        input_protocol=recipe['input_protocol'],matlab_launches=0,permissions=dict(denied=0,pending=[],uncertain=[]))
    save(d/'result.json',report)
    guard=MemoryGuard(4*1024**3,1024**3)
    def boundary(*unused):
        if (d/'STOP').exists():raise ContractError('Explicit cooperative STOP')
        if not guard.check() or time.perf_counter()-began>1200:raise ContractError('Short experiment reached 1200s/4GiB bound')
    class BudgetRunner(HybridRunner):
        def advance(self,*args,**kwargs):
            boundary();kwargs.setdefault('state_observer',boundary)
            return super().advance(*args,**kwargs)
    try:
        tick=time.perf_counter()
        original,_,audit=atlas_population_network(ROOT,duration=.6,light_context=cfg.light_context(),
            execution_profile=cfg.execution_profile,configuration_path=cfg.circuit_configuration_path,
            owner_bounds_mm=cfg.owner_bounds(),halo_mm=cfg.halo_mm,max_spatial_cells=cfg.max_spatial_cells,
            seed=cfg.seed,dynamics_candidate_path=ROOT/recipe['base_dynamics_candidate']['path'])
        changed=merge_parameter_package(original,audit['registry'],recipe,applicability=recipe['applicability'],
            allowed_fields=['alpha','mu','beta'],proposal_path=path)
        report['assembly_wall_seconds']=time.perf_counter()-tick
        counts=dict(components=len(original.components),populations=len(original.populations),
            cells=sum(len(x.cells.cell_id) for x in original.populations.values()),
            dynamic_scalars=sum(s.shape[0] for c in original.components.values() for s in c.state_specs),
            links=len(original.spec.links),gaps=len(original.spec.gap_junctions))
        report['counts']=counts
        assert canonical_hash(original.spec.links)==canonical_hash(changed.spec.links)
        assert canonical_hash(original.spec.gap_junctions)==canonical_hash(changed.spec.gap_junctions)
        rgcs=sorted(cid for cid,c in original.components.items() if c.mechanism=='HistoryConductanceRGC')
        assert len(rgcs)==5
        ports=tuple(cid+'.'+name for cid in rgcs for name in ('voltage','gE','gI','conditional_rate','spike_count','window_rate'))
        plan=RecordingPlan(states=(),ports=ports,max_bytes=64*1024**2)
        save(d/'assembly_comparison.json',dict(counts=counts,old_assembly=original.assembly_hash,
            candidate_assembly=changed.assembly_hash,links_and_gaps_unchanged=True,
            component_parameter_profiles={name:{cid:canonical_hash((_plain(c.parameters.records),_plain(c.parameters.profile)))
                for cid,c in compiled.components.items()} for name,compiled in [('old',original),('candidate',changed)]},
            source_candidate_merge=changed.parameter_package_merge))
        traces={};final_states={};snapshots={};rows=[]
        for name,compiled in [('old',original),('candidate',changed)]:
            boundary();folder=d/name;folder.mkdir(exist_ok=args.resume_startup)
            background=constant_background_channels(compiled,cfg.light_context(),.6,interpolation='zero_order_hold')
            runner=BudgetRunner(compiled,compute_backend='python')
            state=runner.initialize(SolverConfig(method='DOP853',rtol=1e-7,atol=1e-10,max_step=.0001,input_interpolation='zero_order_hold'))
            seed=seed_background_state(runner,state,background)
            state,balance=balance_rgc_initial_voltages(runner,seed.state,background)
            save(folder/'initialization.json',dict(fresh_runner=True,prior_checkpoint_restored=False,
                seed=seed.audit,RGC_balance=balance,fully_adapted=False,short_wash_in_seconds=.1))
            root_cp=runner.checkpoint(state)
            if (folder/'root_checkpoint.json').exists():
                assert (folder/'root_checkpoint.json').read_text(encoding='utf-8')==root_cp
            else:(folder/'root_checkpoint.json').write_text(root_cp,encoding='utf-8')
            print(json.dumps(dict(stage=name+'_FRESH_INITIALIZATION',counts=counts,
                                  wall_seconds=time.perf_counter()-began)),flush=True)
            state,_,warm=record_segment(runner,state,background,.1,folder/'wash_in',plan,label=name+'_WASH_IN')
            cp=runner.checkpoint(state);(folder/'prepared_checkpoint.json').write_text(cp,encoding='utf-8')
            # Prepared checkpoints stay with their own assembly. No cross-restore.
            prepared=json.loads(cp)['payload']['spike_states']
            channels=drive_channels(background,.1,recipe['input_protocol'])
            save(folder/'input_contracts.json',{key:value.input_contract for key,value in channels.items()})
            state,trace,segment=record_segment(runner,state,channels,.6,folder/'stimulus',plan,label=name+'_FULL_NETWORK')
            (folder/'final_checkpoint.json').write_text(runner.checkpoint(state),encoding='utf-8')
            traces[name]=trace;final_states[name]=[a.copy() for a in state.ordered_state_arrays]
            expected={};snapshots[name]={}
            for cid in rgcs:
                n=compiled.components[cid].native_model
                expected[cid]=expectation(n,trace[cid+'.voltage'],prepared[cid])
                snapshots[name][cid]={key:np.asarray(getattr(n,key)).tolist() for key in ('C','gL','EL','EE','EI','alpha','mu','beta','history_kernel','dt')}
                for j,phase in enumerate(['gray_before','increment','gray_middle','decrement','gray_after']):
                    sl=slice(j*100,(j+1)*100);left=round(.1+j*.1,10);right=round(left+.1,10)
                    events=segment['spike_events'][cid]
                    observed=sum(left<=e['time_seconds']<right for e in events)/(len(n.alpha)*.1)
                    rows.append(dict(variant=name,component=cid,phase=phase,
                        mean_voltage_mV=float(trace[cid+'.voltage'][:,sl].mean()*1000),
                        mean_conditional_hazard_Hz=float(trace[cid+'.conditional_rate'][:,sl].mean()),
                        expected_binary_count_rate_Hz=float(expected[cid][:,sl].mean()),
                        one_seed_observed_rate_Hz=observed,
                        voltage_phase_drift_mV=float(np.mean(trace[cid+'.voltage'][:,j*100+99]-trace[cid+'.voltage'][:,j*100])*1000)))
            np.savez_compressed(folder/'exact_expected_count_rates.npz',**expected)
            report['variants'][name]=dict(assembly_hash=compiled.assembly_hash,wash_in_wall_seconds=warm['wall_seconds'],
                stimulus_wall_seconds=segment['wall_seconds'],neural_seconds=.6,fully_adapted=False,
                fresh_parameter_matched_state=True,prior_checkpoint_restored=False,all_outputs_finite=True)
            save(d/'result.json',report)
        save(d/'actual_RGC_parameter_snapshots.json',snapshots)
        with (d/'five_RGC_phase_comparison.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        equality={cid:{port:bool(np.array_equal(traces['old'][cid+'.'+port],traces['candidate'][cid+'.'+port]))
            for port in ('voltage','gE','gI','conditional_rate','spike_count','window_rate')} for cid in rgcs}
        assert all(all(check.values()) for cid,check in equality.items() if cid!='parasol_RGC_ON')
        assert equality['parasol_RGC_ON']['voltage'] and equality['parasol_RGC_ON']['gE'] and equality['parasol_RGC_ON']['gI']
        state_equal=all(np.array_equal(a,b) for a,b in zip(final_states['old'],final_states['candidate']))
        report.update(status='COMPLETE_UNADOPTED_LITERATURE_TRANSFER_SHORT_NETWORK_TEST',
            new_neural_seconds=1.2,continuous_final_states_bitwise_equal=state_equal,signal_equality=equality,
            other_four_RGC_outputs_bitwise_equal=True,source_transfer_not_empirical_calibration=True,
            phase_comparison=rows,prepared_checkpoint_mixed=False,fully_adapted=False,
            real_ISET_standard_video_test='NOT_RUN: no compatible capture; denied MATLAB operation not retried',
            observed_rates='One seed and small population; not a physiological estimate',
            expected_rates='Exact history expectation on actually advanced spike-independent network voltage; first prepared bin count fixed',
            wall_seconds=time.perf_counter()-began)
        for f,h in pins.items():assert sha(ROOT/f)==h,f
        report['protected_files_unchanged']=len(pins)
        save(d/'result.json',report)
        shutil.copyfile(ROOT/'tmp/on_parasol_candidate_regression.xml',d/'candidate_regression.xml')
        print(json.dumps(dict(stage=report['status'],wall_seconds=report['wall_seconds'],
            parasol=[x for x in rows if x['component']=='parasol_RGC_ON'])),flush=True)
    except Exception as e:
        report.update(status='STOPPED_EXACT_BLOCKER',error_type=type(e).__name__,error=str(e),wall_seconds=time.perf_counter()-began)
        save(d/'result.json',report)
        raise


if __name__=='__main__':main()
