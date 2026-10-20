"""300ms rod-only engineering flash or own gray, unchanged complete network."""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.rgc_defaults import historical_project_default,historical_reference_file
from tools.run_rgc_type_candidate import (BASE,RunConfig,atlas_population_network,read_source_candidate,merge_parameter_package,
    HybridRunner,constant_background_channels,RecordingPlan,MemoryGuard,record_segment,chain_ports,save,sha)
from retina_sim.contracts import ContractError,Quantity


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--branch',choices=('flash','gray'),required=True)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();d=a.output.resolve();d.mkdir(parents=True,exist_ok=False);began=time.perf_counter()
    report=dict(status='ASSEMBLY',branch=a.branch,neural_seconds=0.,parameter_changes=False,
        production_default_changed=False,rod_reference_changed=False,matlab_launches=0,physiology_calibrated=False)
    save(d/'result.json',report);guard=MemoryGuard(4*1024**3,2*1024**3)
    def boundary(*unused):
        if not guard.check() or time.perf_counter()-began>600 or (d/'STOP').exists():raise ContractError('Rod probe reached 600s/4GiB/STOP boundary')
    class Runner(HybridRunner):
        def advance(self,*args,**kwargs):
            boundary();kwargs.setdefault('state_observer',boundary);return super().advance(*args,**kwargs)
    try:
        cfg=historical_project_default(ROOT);recipe,path=read_source_candidate(ROOT,ROOT/'configurations/on_parasol_literature_candidate_v1.json')
        old,_,audit=atlas_population_network(ROOT,apply_rgc_output_defaults=False,duration=.6,light_context=cfg.light_context(),execution_profile=cfg.execution_profile,
            configuration_path=cfg.circuit_configuration_path,owner_bounds_mm=cfg.owner_bounds(),halo_mm=cfg.halo_mm,
            max_spatial_cells=cfg.max_spatial_cells,seed=cfg.seed,dynamics_candidate_path=ROOT/recipe['base_dynamics_candidate']['path'])
        compiled=merge_parameter_package(old,audit['registry'],recipe,applicability=recipe['applicability'],allowed_fields=['alpha','mu','beta'],proposal_path=path)
        cp=BASE/'pro_b2_b3_v1/recover_gray/final_checkpoint.json';serialized=cp.read_text(encoding='utf8')
        runner=Runner(compiled,compute_backend='python');state=runner.restore(serialized);assert runner.checkpoint(state)==serialized
        origin=state.time_seconds;stop=round(origin+.3,10);bg=constant_background_channels(compiled,cfg.light_context(),stop,interpolation='zero_order_hold');channels=dict(bg)
        if a.branch=='flash':
            ch=bg['rod_light'];base=ch.at(origin);on=round(origin+.05,10);off=round(origin+.15,10)
            def evaluate(t):return Quantity(base.values*(1.2 if on<=t<off else 1.),base.unit,base.meaning,base.evidence_tag)
            channels['rod_light']=replace(ch,evaluator=evaluate,breakpoints_seconds=(on,off),
                input_contract=dict(ch.input_contract,declared_constant_background=False,rod_only=True,
                    base_Rstar_per_s=5.,increment_fraction=.2,on_seconds=on,off_seconds=off,
                    calibrated_visual_contrast=False,cone_input_held=True))
        ports=list(chain_ports(compiled))
        for cid in ('rod','rod_current_SI','rod_membrane','individual_rod_RBC_contacts','rod_BC','AII_AC','RBC_AII_AMPA','AII_FMB_glycine','AII_DB1_glycine'):
            ports.extend(cid+'.'+key for key in compiled.components[cid].outputs)
        plan=RecordingPlan(states=(),ports=tuple(ports),max_bytes=128*1024**2)
        save(d/'input_contracts.json',{k:v.input_contract for k,v in channels.items()})
        state,trace,meta=record_segment(runner,state,channels,stop,d/'trace',plan,label='ROD_'+a.branch.upper())
        (d/'final_checkpoint.json').write_text(runner.checkpoint(state),encoding='utf8')
        report.update(status='COMPLETE_UNCHANGED_ROD_CHAIN_PROBE',neural_seconds=.3,origin_seconds=origin,
            source_checkpoint_sha256=sha(cp),exact_own_checkpoint_restore=True,assembly_hash=compiled.assembly_hash,
            all_values_finite=True,wall_seconds=time.perf_counter()-began,maximum_RSS_bytes=guard.peak_rss_bytes)
        save(d/'result.json',report);print(json.dumps(report),flush=True)
    except Exception as e:
        report.update(status='STOPPED_EXACT_BLOCKER',error_type=type(e).__name__,error=str(e),wall_seconds=time.perf_counter()-began)
        save(d/'result.json',report);raise


if __name__=='__main__':main()
