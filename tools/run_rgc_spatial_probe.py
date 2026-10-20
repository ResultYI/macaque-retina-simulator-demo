"""Short receptor-domain spatial/spectral probes on the unchanged full graph.

These are declared engineering cone-input masks, not ISET or calibrated
physical color contrasts. All rod consumers keep their original background.
Each branch restores only its mapping's own prepared checkpoint.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.rgc_defaults import historical_project_default,historical_reference_file
from tools.run_rgc_type_candidate import (
    BASE,REFERENCE,RGCS,RunConfig,atlas_population_network,read_source_candidate,merge_parameter_package,
    HybridRunner,constant_background_channels,RecordingPlan,MemoryGuard,record_segment,save,sha,
    chain_ports,analyze_rates,load_arrays)
from retina_sim.contracts import ContractError,Quantity

PROTOCOLS=('center05','center20','annulus20','full20','S20','LM20')


def masked_channels(compiled,cfg,background,protocol):
    cones=compiled.populations['cones'].cells;xy=cones.rf_position.values
    if cones.position_unit!='mm':raise ContractError('Explicit RF position in mm required for this diagnostic mask')
    center=compiled.populations['parasol_RGC_ON'].cells.rf_position.values.mean(axis=0)
    r=np.linalg.norm(xy-center,axis=1)
    fraction=.05 if protocol=='center05' else .2
    if protocol.startswith('center'):mask=r<=.015
    elif protocol=='annulus20':mask=(r>.015)&(r<=.045)
    elif protocol=='S20':mask=np.asarray(cones.subtype)=='S'
    elif protocol=='LM20':mask=np.isin(cones.subtype,('L','M'))
    else:mask=np.ones(len(r),bool)
    if not mask.any():raise ContractError('Declared probe mask has no actual cone support')
    edges=np.array([.1,.2,.3,.4,.5,.6]);phases=[0.,fraction,0.,-fraction,0.]
    result={};mask_applied=0
    for key,ch in background.items():
        base=ch.at(0.)
        ext=next(e for e in compiled.spec.external_inputs if e.channel==key)
        if ext.target_component!='outer_native':result[key]=ch;continue
        if tuple(compiled.components['outer_native'].inputs[ext.target_port].cell_ids)!=tuple(cones.cell_id):
            raise ContractError('Cone mask cell order differs from consumed input')
        def evaluate(t,base=base,mask=mask):
            j=min(max(int(np.searchsorted(edges,t,side='right')-1),0),4)
            return Quantity(base.values*(1+phases[j]*mask),base.unit,base.meaning,base.evidence_tag)
        result[key]=replace(ch,evaluator=evaluate,breakpoints_seconds=tuple(edges[1:-1]),
            input_contract=dict(ch.input_contract,declared_constant_background=False,
                probe=protocol,phase_edges_seconds=edges.tolist(),fractions=phases,
                actual_active_cone_ids=[cid for cid,on in zip(cones.cell_id,mask) if on],
                calibrated_visual_contrast=False,common_physical_scene_claim=False,
                spatial_domain='cone RF-position engineering mask; no optical projection',
                center_mm=center.tolist(),center_radius_mm=.015,annulus_outer_radius_mm=.045,
                rod_background_held=True))
        mask_applied+=1
    if mask_applied!=1:raise ContractError('Exactly one actual native cone input required for these probes')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',choices=PROTOCOLS,required=True)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args();d=args.output.resolve()
    if not d.is_relative_to(ROOT):raise ContractError('Probe output must remain local')
    d.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    cid='small_bistratified_S_ON' if args.protocol in ('S20','LM20') else 'parasol_RGC_ON'
    candidate_path=ROOT/'configurations'/('on_parasol_literature_candidate_v1.json' if cid=='parasol_RGC_ON'
                                       else cid+'_literature_candidate_v1.json')
    own=REFERENCE/'candidate' if cid=='parasol_RGC_ON' else BASE/'pro_b0_b2_v1'/cid
    report=dict(status='ASSEMBLY',protocol=args.protocol,mapping_component=cid,neural_seconds=0.,
        production_default_changed=False,physiology_calibrated=False,matlab_launches=0,
        rod_background_held=True,fully_adapted=False)
    save(d/'result.json',report)
    guard=MemoryGuard(4*1024**3,2*1024**3)
    def boundary(*unused):
        if not guard.check() or time.perf_counter()-started>1200 or (d/'STOP').exists():
            raise ContractError('Spatial probe reached explicit 1200s/4GiB/available-memory/STOP boundary')
    class Runner(HybridRunner):
        def advance(self,*args,**kwargs):
            boundary();kwargs.setdefault('state_observer',boundary)
            return super().advance(*args,**kwargs)
    try:
        cfg=historical_project_default(ROOT);recipe,path=read_source_candidate(ROOT,candidate_path)
        original,_,audit=atlas_population_network(ROOT,apply_rgc_output_defaults=False,duration=.6,light_context=cfg.light_context(),
            execution_profile=cfg.execution_profile,configuration_path=cfg.circuit_configuration_path,
            owner_bounds_mm=cfg.owner_bounds(),halo_mm=cfg.halo_mm,max_spatial_cells=cfg.max_spatial_cells,
            seed=cfg.seed,dynamics_candidate_path=ROOT/recipe['base_dynamics_candidate']['path'])
        compiled=merge_parameter_package(original,audit['registry'],recipe,applicability=recipe['applicability'],
            allowed_fields=['alpha','mu','beta'],proposal_path=path)
        runner=Runner(compiled,compute_backend='python');prepared=(own/'prepared_checkpoint.json').read_text(encoding='utf8')
        state=runner.restore(prepared)
        if runner.checkpoint(state)!=prepared:raise ContractError('Own prepared checkpoint roundtrip differs')
        aux=json.loads(prepared)['payload']['spike_states']
        bg=constant_background_channels(compiled,cfg.light_context(),.6,interpolation='zero_order_hold')
        channels=masked_channels(compiled,cfg,bg,args.protocol)
        save(d/'input_contracts.json',{k:v.input_contract for k,v in channels.items()})
        plan=RecordingPlan(states=(),ports=chain_ports(compiled),max_bytes=128*1024**2)
        state,trace,meta=record_segment(runner,state,channels,.6,d/'stimulus',plan,label=args.protocol)
        (d/'final_checkpoint.json').write_text(runner.checkpoint(state),encoding='utf8')
        analyze_rates(compiled,trace,aux,d/'stimulus')
        own_gray=load_arrays(own/'matched_gray/traces.npz')
        for key in (c+'.'+k for c in RGCS for k in ('voltage','gE','gI')):
            np.testing.assert_allclose(trace[key][:,:100],own_gray[key][:,:100],rtol=0,atol=1e-12)
        report.update(status='COMPLETE_UNCALIBRATED_RECEPTOR_DOMAIN_PROBE',neural_seconds=.5,
            assembly_hash=compiled.assembly_hash,own_prepared_checkpoint_sha256=sha(own/'prepared_checkpoint.json'),
            own_gray_reference=str((own/'matched_gray').relative_to(ROOT)),own_checkpoint_roundtrip_exact=True,
            first_gray_segment_matches_control=True,wall_seconds=time.perf_counter()-started,
            maximum_RSS_bytes=guard.peak_rss_bytes,all_values_finite=True)
        save(d/'result.json',report);print(json.dumps(report),flush=True)
    except Exception as e:
        report.update(status='STOPPED_EXACT_BLOCKER',error_type=type(e).__name__,error=str(e),wall_seconds=time.perf_counter()-started)
        save(d/'result.json',report);raise


if __name__=='__main__':main()
