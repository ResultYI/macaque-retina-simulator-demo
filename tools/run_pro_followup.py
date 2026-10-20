"""Bounded recovery continuations and one-cell geometric center/surround probes.

Same unadopted ON-parasol mapping and coupled graph. No parameter change,
checkpoint rebinding, optical capture, gain increase or anatomical change.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.rgc_defaults import historical_project_default,historical_reference_file
from tools.run_rgc_type_candidate import (BASE,REFERENCE,RGCS,RunConfig,atlas_population_network,read_source_candidate,
    merge_parameter_package,HybridRunner,constant_background_channels,RecordingPlan,MemoryGuard,
    record_segment,chain_ports,save,sha,analyze_rates)
from retina_sim.contracts import ContractError,Quantity

BRANCHES=('recover_stimulus','recover_gray','cell_center','cell_annulus')


def single_cell_mask(compiled,cfg,bg,kind):
    cid='parasol_RGC_ON';pop=compiled.populations[cid].cells;coords=pop.rf_position.values
    focus=int(np.argmin(np.linalg.norm(coords-coords.mean(axis=0),axis=1)));gid=pop.cell_id[focus];center=coords[focus]
    adapter=cid+'_excitatory_diffuse_BC_ON'
    bc_link=next(l for l in compiled.spec.links if l.target_component==adapter and l.target_port=='activation')
    pre_bc={a for a,b,w in zip(bc_link.connections.pre_cell_id,bc_link.connections.post_cell_id,bc_link.connections.weight) if b==gid and w>0}
    cone_link=next(l for l in compiled.spec.links if l.target_component=='diffuse_BC_ON' and l.target_port=='drive')
    pre_cones={a for a,b,w in zip(cone_link.connections.pre_cell_id,cone_link.connections.post_cell_id,cone_link.connections.weight) if b in pre_bc and w>0}
    cones=compiled.populations['cones'].cells;xy=cones.rf_position.values;index={g:j for j,g in enumerate(cones.cell_id)}
    if not pre_cones or not pre_cones<=set(index):raise ContractError('Cannot trace actual feedforward cone support for selected RGC')
    distance=np.linalg.norm(xy-center,axis=1);radius=max(.015,1.05*max(distance[index[g]] for g in pre_cones));outer=radius+.03
    if np.any(center-outer<xy.min(axis=0)) or np.any(center+outer>xy.max(axis=0)):
        raise ContractError('Requested ring exceeds conservative recorded cone-support bbox; no ROI expansion or hidden boundary fill')
    mask=distance<=radius if kind=='cell_center' else (distance>radius)&(distance<=outer)
    assert mask.any()
    if kind=='cell_annulus' and any(mask[index[g]] for g in pre_cones):raise ContractError('Annulus includes a traced direct feedforward cone')
    edges=np.arange(6)*.1+.1;phases=[0.,.2,0.,-.2,0.];channels={}
    contract=dict(focus_component=cid,focus_cell_id=gid,focus_cell_index=focus,focus_RF_mm=center.tolist(),
        traced_feedforward_BC_ids=sorted(pre_bc),traced_feedforward_cone_ids=sorted(pre_cones),
        center_radius_mm=float(radius),ring_outer_radius_mm=float(outer),
        direct_feedforward_cones_excluded_from_ring=True,
        active_cone_ids=[g for g,on in zip(cones.cell_id,mask) if on],
        spatial_scope='One actual RGC RF; center covers original direct excitatory cone support; ring adds 30um',
        input='engineering cone receptor-domain mask, rod held; no ISET or physical contrast calibration',
        fully_adapted=False,ROI_or_connectivity_changed=False)
    for key,ch in bg.items():
        ext=next(e for e in compiled.spec.external_inputs if e.channel==key);base=ch.at(0.)
        if ext.target_component!='outer_native':channels[key]=ch;continue
        def evaluate(t,base=base):
            j=min(max(int(np.searchsorted(edges,t,side='right')-1),0),4)
            return Quantity(base.values*(1+phases[j]*mask),base.unit,base.meaning,base.evidence_tag)
        channels[key]=replace(ch,evaluator=evaluate,breakpoints_seconds=tuple(edges[1:-1]),
            input_contract=dict(ch.input_contract,declared_constant_background=False,single_cell_probe=contract,
                fractions=phases,phase_edges_seconds=edges.tolist()))
    return channels,contract


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--branch',choices=BRANCHES,required=True)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();d=a.output.resolve();d.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();report=dict(status='ASSEMBLY',branch=a.branch,neural_seconds=0.,matlab_launches=0,
        production_defaults_changed=False,parameter_changes=False,geometry_changes=False,physiology_calibrated=False)
    save(d/'result.json',report);guard=MemoryGuard(4*1024**3,2*1024**3)
    def boundary(*unused):
        if not guard.check() or time.perf_counter()-started>900 or (d/'STOP').exists():raise ContractError('Bounded followup reached 900s/4GiB/STOP')
    class Runner(HybridRunner):
        def advance(self,*args,**kwargs):
            boundary();kwargs.setdefault('state_observer',boundary)
            return super().advance(*args,**kwargs)
    try:
        cfg=historical_project_default(ROOT);recipe,path=read_source_candidate(ROOT,ROOT/'configurations/on_parasol_literature_candidate_v1.json')
        original,_,audit=atlas_population_network(ROOT,apply_rgc_output_defaults=False,duration=.6,light_context=cfg.light_context(),execution_profile=cfg.execution_profile,
            configuration_path=cfg.circuit_configuration_path,owner_bounds_mm=cfg.owner_bounds(),halo_mm=cfg.halo_mm,
            max_spatial_cells=cfg.max_spatial_cells,seed=cfg.seed,dynamics_candidate_path=ROOT/recipe['base_dynamics_candidate']['path'])
        compiled=merge_parameter_package(original,audit['registry'],recipe,applicability=recipe['applicability'],
            allowed_fields=['alpha','mu','beta'],proposal_path=path)
        if a.branch=='recover_stimulus':cp=BASE/'pro_b0_b2_v1/spatial/full20/final_checkpoint.json';stop=1.1
        elif a.branch=='recover_gray':cp=REFERENCE/'candidate/matched_gray/final_checkpoint.json';stop=1.1
        else:cp=REFERENCE/'candidate/prepared_checkpoint.json';stop=.6
        runner=Runner(compiled,compute_backend='python');serialized=cp.read_text(encoding='utf8');state=runner.restore(serialized)
        if runner.checkpoint(state)!=serialized:raise ContractError('Own checkpoint roundtrip differs')
        bg=constant_background_channels(compiled,cfg.light_context(),stop,interpolation='zero_order_hold')
        if a.branch.startswith('cell_'):
            channels,geometry=single_cell_mask(compiled,cfg,bg,a.branch);save(d/'single_cell_geometry.json',geometry)
        else:channels=bg
        plan=RecordingPlan(states=(),ports=chain_ports(compiled),max_bytes=128*1024**2)
        save(d/'input_contracts.json',{k:v.input_contract for k,v in channels.items()})
        state,trace,meta=record_segment(runner,state,channels,stop,d/'trace',plan,label=a.branch)
        (d/'final_checkpoint.json').write_text(runner.checkpoint(state),encoding='utf8')
        if a.branch.startswith('cell_'):
            analyze_rates(compiled,trace,json.loads(serialized)['payload']['spike_states'],d/'trace')
        report.update(status='COMPLETE_UNCHANGED_MODEL_FOLLOWUP',neural_seconds=.5,wall_seconds=time.perf_counter()-started,
            source_checkpoint=str(cp.relative_to(ROOT)),source_checkpoint_sha256=sha(cp),exact_own_checkpoint_restore=True,
            assembly_hash=compiled.assembly_hash,all_values_finite=True,maximum_RSS_bytes=guard.peak_rss_bytes)
        save(d/'result.json',report);print(json.dumps(report),flush=True)
    except Exception as e:
        report.update(status='STOPPED_EXACT_BLOCKER',error_type=type(e).__name__,error=str(e),wall_seconds=time.perf_counter()-started)
        save(d/'result.json',report);raise


if __name__=='__main__':main()
