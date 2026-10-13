"""Minimal actual-graph coverage, disk interruption equivalence and cache rejection."""
from pathlib import Path
import argparse,json,sys
from dataclasses import replace
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));out=args.output
    if (out/'verification.json').exists():raise FileExistsError('Verification already complete')
    out.mkdir(parents=True,exist_ok=True)
    from retina_platform.config import RunConfig
    from retina_sim.atlas_population_network import atlas_population_network
    from retina_sim.hybrid import HybridRunner
    from retina_sim.contracts import SolverConfig,ContractError
    from retina_sim.background import constant_background_channels
    from retina_sim.background_checkpoint import BackgroundCheckpointStore,MemoryGuard,atomic_json
    from retina_sim.recording import RecordingPlan
    cfg=RunConfig.project_default(root);light=cfg.light_context();policy=cfg.background_policy();guard=MemoryGuard();assert guard.check()
    compiled,_,audit=atlas_population_network(root,light_context=light,duration=11.)
    runner=HybridRunner(compiled);reference=runner.initialize(SolverConfig(**audit['resolved_execution_profile']['solver']))
    store=BackgroundCheckpointStore(root,runner,reference,light,policy);coverage=store.coverage()
    assert coverage['state_blocks']==coverage['covered_blocks']==77 and coverage['state_scalars']==53759 and not coverage['missing']
    channels=constant_background_channels(compiled,light,11.);plan=RecordingPlan(ports=(),states=(),cell_ids=())
    state=runner.restore(runner.checkpoint(reference));state,_=runner.advance(state,channels,.001,recording_plan=plan)
    path=out/'partial_checkpoint.json';store.save(path,state)
    state,_=runner.advance(state,channels,.003,recording_plan=plan);expected=runner.checkpoint(state)
    restored,_=store.load(path);restored,_=runner.advance(restored,channels,.003,recording_plan=plan)
    assert runner.checkpoint(restored)==expected,'Complete checkpoint/RNG/history/delay continuation differs'
    rejected=[]
    def reject(label,action):
        try:action()
        except ContractError:rejected.append(label)
        else:raise AssertionError(label+' unexpectedly accepted')
    original=compiled.links;matrix=original[0].matrix.copy();matrix.data[0]*=1.001
    compiled.links=(replace(original[0],matrix=matrix),)+original[1:]
    altered=BackgroundCheckpointStore(root,runner,reference,light,policy);reject('actual edge weight changed, identical counts',lambda:altered.load(path));compiled.links=original
    altered=BackgroundCheckpointStore(root,runner,reference,replace(light,native_background_td=101.),policy)
    reject('background changed',lambda:altered.load(path));reference.input_position.update(store.expected_context)
    other=runner.initialize(replace(reference.solver_config,rtol=reference.solver_config.rtol*2))
    altered=BackgroundCheckpointStore(root,runner,other,light,policy);reject('solver changed',lambda:altered.load(path))
    value=json.loads(path.read_bytes());value['checkpoint']['payload']['sample_index']+=1;atomic_json(out/'tampered.json',value)
    reject('time/history auxiliary payload changed',lambda:store.load(out/'tampered.json'))
    atomic_json(out/'raw_old_checkpoint.json',json.loads(expected));reject('plain/old Run checkpoint has no background identity',lambda:store.load(out/'raw_old_checkpoint.json'))
    reject('transient cannot claim converged cache',lambda:store.load(path,require_converged=True))
    guard.check();report={'status':'PASS','coverage':coverage,'complete_disk_resume_bitwise_equal':True,
        'equivalence_interval_seconds':[.001,.003],'rejected_cases':rejected,'binding':store.binding,'memory':guard.last,
        'convergence_criteria_changed':False,'physiological_validation':'NOT_RUN'}
    atomic_json(out/'verification.json',report);print(json.dumps({'status':'PASS','output':str(out),'coverage_blocks':77,'minimum_background_seconds':coverage['timing_requirements']['minimum_required_seconds'],'memory':guard.last}),flush=True)

if __name__=='__main__':main()
