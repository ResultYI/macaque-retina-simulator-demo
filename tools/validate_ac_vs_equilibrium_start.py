"""Opt-in saved-equilibrium IVP entry; prepare is static, run is explicit.

No background solving or legacy construction-seed rebuilding occurs here.
The declared exterior reservoirs remain unqualified for arbitrary dynamics.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import math
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

from retina_sim.ac_vs_model import ACVSAssembly,functional_patch,load_config
from retina_sim.ac_vs_monitor import DynamicDefectMonitor
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.group_time_monitor import GroupTimeMonitor
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.group_time_storage import load_bundle,save_bundle
from retina_sim.local_patch_boundary import BoundedInnerDynamics,BoundedNativePhysics,configure_boundary_runtime
from retina_sim.local_patch_runner import make_protocol
from retina_sim.spatial_group_graph import digest
from retina_sim.spatial_group_model import RGC
from tools.validate_ac_vs import (Worker,QUALIFICATION,write,protocol_config,dynamic_observe,
    save_traces,causal_hash)
from tools.validate_group_time_candidate import clock_events

REPORT=ROOT/'reports/ac_vs_equilibrium_start_v1'
SCHEMA='ac-vs-saved-equilibrium-start-1'
ENTRY_PINS=('tools/validate_ac_vs_equilibrium_start.py','retina_sim/group_time_storage.py',
    'retina_sim/local_patch_boundary.py','retina_sim/local_patch_graph.py','retina_sim/group_native_physics.py',
    'retina_sim/native_multirate.py','retina_sim/group_parameter_binding.py','retina_sim/recording.py')


def local_path(value):
    path=Path(value).resolve()
    if not path.is_relative_to(ROOT):raise ContractError('Explicit workspace-local bound path required')
    return path


def seed_directory(value):
    path=local_path(value)
    return path.parent if path.name=='manifest.json' else path


def verify_sources(pins):
    for relative,expected in pins.items():
        path=local_path(ROOT/relative)
        if not path.is_file() or digest(path)!=expected:
            raise ContractError('Saved equilibrium scientific/source pin differs: '+relative)


def static_assembly():
    """Geometry/registry only. No native/inner equation-provider construction."""
    directory=ROOT/'reports/ac_vs_redesign_v1/functional_static_halo_030_v1'
    if not directory.is_dir():raise ContractError('Already frozen Halo030 geometry required; do not build a new graph')
    config=json.loads((ROOT/'configurations/local_patch_v2/halo_030.json').read_text(encoding='utf8'))
    patch=functional_patch(ROOT,config,directory,lambda:None)
    patch.runtime_factories=dict(inner=BoundedInnerDynamics,native=BoundedNativePhysics)
    return ACVSAssembly(patch,load_config(ROOT/'configurations/ac_vs_functional_v1.json'),max_step=.0001)


def validate_saved_background(bundle,receipt,assembly):
    seed=bundle['payload'];identity=bundle['identity']
    if (receipt.get('category')!='BACKGROUND_EQUILIBRIUM' or receipt.get('passed') is not True
            or receipt.get('retained_local_background_equilibrium') is not True
            or receipt.get('whole_retina_or_natural_video_background_qualified') is not False):
        raise ContractError('Passed conditional local-background receipt required')
    if identity.get('artifact')!=assembly.artifact_sha256 or seed['ac_vs_lineage']['artifact']!=assembly.artifact_sha256:
        raise ContractError('Saved background artifact differs from same AC assembly')
    actual=canonical_hash(seed)
    if identity.get('seed')!=actual or receipt.get('seed_sha256')!=actual:
        raise ContractError('Background receipt/payload identity differs')
    verify_sources(identity['sources']);verify_sources(seed['patch_initialization']['source_sha256'])
    blocks=(*assembly.state_blocks,*assembly.probability_blocks,*assembly.auxiliary_z1_blocks)
    specs={(b.population,b.mechanism,b.name):b for b in blocks}
    if set(seed['states'])!=set(specs):raise ContractError('Saved background must cover every declared state block')
    for k,b in specs.items():
        a=np.asarray(seed['states'][k])
        if a.dtype!=np.float64 or a.shape!=(b.count,) or not np.isfinite(a).all():
            raise ContractError('Saved background state shape/dtype/domain differs: '+str(k))
    if set(seed['spike_states'])!=set(RGC):raise ContractError('All five RGC history/RNG fields required')
    for pop,spike in seed['spike_states'].items():
        n=assembly.populations[pop].group_count
        if np.asarray(spike['history']).shape!=(3,n) or not np.isin(spike['history'],(0,1)).all():
            raise ContractError('Saved original three-bin binary history required')
        rng=np.random.Generator(np.random.PCG64());rng.bit_generator.state=deepcopy(spike['rng_state'])
        if spike['last_bin_counts'] is not None or spike['pending_events']:
            raise ContractError('Uncommitted equilibrium source IVP required')
    observation=seed['observation_prehistory'];t=float(seed['time_seconds'])
    if (t!=4.993 or observation['delay_seconds']!=.00282 or observation['origin_seconds']!=t
            or observation['interpolation']!='linear'
            or not np.array_equal(observation['history_times'],np.array([t]))
            or not np.array_equal(observation['history_values'],np.asarray(observation['prehistory'])[None,:])
            or not np.array_equal(observation['prehistory'],seed['states']['HC','reduced_native','Vh'])):
        raise ContractError('Saved constant HC2.82ms history/origin differs')
    B=np.asarray(seed['input_reference']['native_background_td'])
    if B.dtype!=np.float64 or B.shape!=(assembly.populations['cones'].group_count,) or not np.isfinite(B).all() or np.any(B<0):
        raise ContractError('Nonnegative native td background vector required')
    for frame,value in seed['native_inputs'].items():
        if not np.array_equal(value,B):raise ContractError('Initial source package is not constant native background')
        rod=np.asarray(seed['rod_inputs'][frame])
        if not np.array_equal(rod,np.full(assembly.populations['rods'].group_count,seed['input_reference']['rod_background_rstar_s'])):
            raise ContractError('Saved constant rod background differs')
    if canonical_hash(receipt['explicit_exterior'])!=canonical_hash(seed['patch_boundary']):
        raise ContractError('Conditional exterior identity differs')
    return dict(state_blocks=len(specs),state_scalars=sum(b.count for b in blocks),
        time_seconds=t,saved_payload_sha256=actual,artifact=assembly.artifact_sha256,
        background_receipt_scope='conditional_local_fixed_exterior_only',real_model_constructed=False)


def contrast_from_saved_seed(saved,assembly,*,manifest_hash,receipt_sha256):
    """Only replace future native frames and lineage; retain all initial conditions."""
    original=canonical_hash(saved);seed=deepcopy(saved)
    B=np.asarray(saved['input_reference']['native_background_td'])
    rows=np.asarray(assembly.core_population_rows['cones'],dtype=np.int64)
    if (rows.ndim!=1 or not len(rows) or len(np.unique(rows))!=len(rows)
            or np.any(rows<0) or np.any(rows>=len(B))):raise ContractError('Nonempty frozen geometric core cone mask required')
    mask=np.zeros(len(B),dtype=bool);mask[rows]=True
    if not np.any(B[mask]>0):raise ContractError('Core contrast must change a positive background input')
    t=float(saved['time_seconds']);first=int(math.floor(t*150.));edge=(first+1)/150.
    frames=sorted(saved['native_inputs'])
    if frames[0]!=first or first+1 not in frames:raise ContractError('Original exact150Hz frame coverage required')
    for frame in frames:seed['native_inputs'][frame]=B.copy() if frame==first else B*(1.+.05*mask)
    stimulus=dict(schema='local-core-native-td-contrast-1',unit='td',meaning='native_retinal_illuminance',
        stimulus_class='synthetic_engineering_local_contrast_not_real_movie',first_frame=first,
        next_frame_boundary_seconds=edge,contrast=.05,background_sha256=canonical_hash(B),
        mask_rule='already frozen core_population_rows[cones]; no response-based selection',
        core_rows=rows.tolist(),core_cone_group_ids=np.asarray(assembly.populations['cones'].group_ids)[rows].tolist(),
        mask_sha256=canonical_hash(mask),native_frame_sha256={str(f):canonical_hash(v) for f,v in seed['native_inputs'].items()},
        rod_frame_sha256={str(f):canonical_hash(v) for f,v in seed['rod_inputs'].items()},rod_unit='R*/s',rod_background_unchanged=True,
        start_seconds=t,stop_seconds=t+.0031,input_rate_Hz=150.,optical_or_physiological_calibration_claimed=False)
    stimulus['identity']=canonical_hash(stimulus)
    seed['ac_vs_lineage']=dict(schema=SCHEMA,artifact=assembly.artifact_sha256,
        initial_condition_origin='saved_conditional_equilibrium_source_seed',saved_equilibrium_seed_used=True,
        saved_seed_manifest_hash=manifest_hash,saved_seed_payload_sha256=original,
        background_receipt_sha256=receipt_sha256,synthetic_input_identity=stimulus['identity'],
        real_movie_cache_reused=False,whole_retina_background_qualified=False,
        parameter_identity=saved['inner_parameters']['identity'])
    if canonical_hash(saved)!=original:raise ContractError('Saved equilibrium payload mutated')
    return seed,stimulus,mask


def normalized_delay(delay):
    return dict(delay_seconds=delay['delay_seconds'],history_times=np.asarray(delay['history_times'],dtype=np.float64),
        history_values=np.asarray(delay['history_values'],dtype=np.float64),
        prehistory=np.asarray(delay['prehistory']['values']),origin_seconds=delay['prehistory_stop_seconds'],
        unit=delay['prehistory']['unit'],meaning=delay['prehistory']['meaning'],interpolation=delay['interpolation'])


def initial_match(arm,saved):
    """This assertion is legal only before initial_commit; later sampling is normal."""
    if arm.initial_commit_done or arm.event_generation!=0 or arm.last_draws:
        raise ContractError('Initial equality must be captured before any initial_commit or RNG draw')
    snapshot=arm.initial_snapshot();actual=snapshot['states'];expected=saved['states']
    if set(actual)!=set(expected):raise ContractError('Initial state registry differs from saved equilibrium')
    state_checks={':'.join(k):canonical_hash(actual[k])==canonical_hash(expected[k]) for k in expected}
    spike_checks={pop:canonical_hash(arm.spike_states[pop])==canonical_hash(saved['spike_states'][pop]) for pop in RGC}
    observation=saved['observation_prehistory']
    expected_delay={n:observation[n] for n in ('delay_seconds','history_times','history_values','prehistory','origin_seconds','unit','meaning','interpolation')}
    delay=normalized_delay(arm.delay_payload())
    delay_checks={n:canonical_hash(delay[n])==canonical_hash(v) for n,v in expected_delay.items()}
    parameters_match=canonical_hash(arm.inner.parameters)==canonical_hash(saved['inner_parameters'])
    time_exact=float(snapshot['time_seconds'])==float(saved['time_seconds'])
    passed=all(state_checks.values()) and all(spike_checks.values()) and all(delay_checks.values()) and parameters_match and time_exact
    result=dict(passed=passed,before_initial_commit=True,states=state_checks,spike_history_RNG_events=spike_checks,
        HC_delay=delay_checks,parameters_exact=parameters_match,time_exact=time_exact,initial_state_sha256=canonical_hash(actual),
        saved_state_sha256=canonical_hash(expected),time_seconds=snapshot['time_seconds'],
        post_commit_sampling_must_not_be_compared_to_saved_seed=True)
    evidence=dict(receipt=result,states=deepcopy(actual),spike_states=deepcopy(arm.spike_states),
        observation_delay=arm.delay_payload(),inner_parameters=deepcopy(arm.inner.parameters))
    if not passed:raise ContractError('Actual pre-commit initial conditions differ from saved equilibrium: '+json.dumps(result))
    return result,evidence


def construct_arm(assembly,seed,protocol,guard,*,arm_type=GroupTimeArm):
    return arm_type(assembly,seed,protocol,mode='reference',guard=guard)


def prepare(initial_seed,output):
    directory=seed_directory(initial_seed)
    bundle=load_bundle(directory,expected_kind='source_construction_seed')
    receipt_path=directory.parent/'result.json';receipt=json.loads(receipt_path.read_text(encoding='utf8'))
    assembly=static_assembly();check=validate_saved_background(bundle,receipt,assembly)
    seed,stimulus,mask=contrast_from_saved_seed(bundle['payload'],assembly,
        manifest_hash=bundle['manifest_hash'],receipt_sha256=digest(receipt_path))
    protocol=make_protocol(seed,duration_seconds=.0031,bounded_v2=True)
    protocol['initialization']='Saved conditional local equilibrium source IVP; regenerate solver controllers and consumer watermarks; retain all saved states/history/RNG/delay before initial_commit'
    protocol['equilibrium_start_stimulus_identity']=stimulus['identity']
    protocol['ac_vs_contract_sha256']=digest(ROOT/'configurations/ac_vs_acceptance_v1.json')
    pins=dict(bundle['identity']['sources']);pins.update({n:digest(ROOT/n) for n in ENTRY_PINS})
    identity=dict(schema=SCHEMA,artifact=assembly.artifact_sha256,saved_seed=dict(
        path=str((directory/'manifest.json').relative_to(ROOT)).replace('\\','/'),manifest_hash=bundle['manifest_hash'],
        manifest_file_sha256=digest(directory/'manifest.json'),payload_sha256=canonical_hash(bundle['payload'])),
        background_receipt=dict(path=str(receipt_path.relative_to(ROOT)).replace('\\','/'),sha256=digest(receipt_path)),
        sources=pins,seed_sha256=canonical_hash(seed),stimulus=stimulus,protocol=protocol,
        saved_equilibrium_seed_used=True,real_movie_cache_reused=False,default_adopted=False,FIT_READY=False)
    save_bundle(output/'dynamic_initial_seed',kind='source_construction_seed',identity=identity,payload=seed,qualification=QUALIFICATION)
    np.savez_compressed(output/'geometric_core_mask.npz',mask=mask,group_ids=assembly.populations['cones'].group_ids)
    preparation=dict(schema=SCHEMA,status='READY_FOR_COORDINATED_RUN',identity=identity,static_validation=check,
        mask_file_sha256=digest(output/'geometric_core_mask.npz'),
        real_model_constructed=False,real_model_advanced=False,new_background_solved=False,
        actual_precommit_initial_match='PENDING_EXPLICIT_RUN',execution_ownership='parent coordinator must release model worker',
        qualification=QUALIFICATION)
    preparation['preparation_hash']=canonical_hash(preparation)
    write(output/'preparation.json',preparation);write(output/'stimulus_contract.json',stimulus)
    return preparation


def load_prepared(prepared,initial_seed):
    root=local_path(prepared);raw=json.loads((root/'preparation.json').read_text(encoding='utf8'))
    claimed=raw.pop('preparation_hash',None)
    if raw.get('schema')!=SCHEMA or claimed!=canonical_hash(raw):raise ContractError('Frozen preparation identity differs')
    identity=raw['identity'];verify_sources(identity['sources'])
    directory=seed_directory(initial_seed)
    if (directory/'manifest.json')!=local_path(ROOT/identity['saved_seed']['path']):raise ContractError('Different --initial-seed requested')
    if digest(directory/'manifest.json')!=identity['saved_seed']['manifest_file_sha256']:raise ContractError('Saved equilibrium manifest changed')
    original=load_bundle(directory,expected_kind='source_construction_seed')
    if original['manifest_hash']!=identity['saved_seed']['manifest_hash'] or canonical_hash(original['payload'])!=identity['saved_seed']['payload_sha256']:
        raise ContractError('Saved equilibrium payload changed')
    receipt=local_path(ROOT/identity['background_receipt']['path'])
    if digest(receipt)!=identity['background_receipt']['sha256'] or digest(root/'geometric_core_mask.npz')!=raw['mask_file_sha256']:
        raise ContractError('Frozen background receipt or geometric mask changed')
    bundle=load_bundle(root/'dynamic_initial_seed',expected_kind='source_construction_seed',expected_identity=identity)
    if canonical_hash(bundle['payload'])!=identity['seed_sha256']:raise ContractError('Frozen dynamic seed changed')
    assembly=static_assembly()
    validate_saved_background(original,json.loads(receipt.read_text(encoding='utf8')),assembly)
    return assembly,original['payload'],bundle['payload'],identity


def bind_original_exterior(assembly,saved):
    """Execution-only reconstruction of the same fixed reservoirs, no warmup."""
    binding=assembly.patch.config['input_identity'];manifest=local_path(ROOT/binding['path'])
    if digest(manifest)!=binding['sha256']:raise ContractError('Exterior source manifest changed')
    parent=load_bundle(manifest.parent,expected_kind='source_construction_seed')['payload']
    actual=configure_boundary_runtime(assembly.patch,parent,{})
    if canonical_hash(actual)!=canonical_hash(saved['patch_boundary']):raise ContractError('Original fixed exterior products changed')


def run(prepared,initial_seed,output,*,resume=None):
    assembly,saved,seed,identity=load_prepared(prepared,initial_seed)
    protocol=identity['protocol']
    with Worker(output) as worker:
        bind_original_exterior(assembly,saved)
        if resume:
            checkpoint=local_path(resume)
            prior=json.loads((checkpoint.parent/'model_identity.json').read_text(encoding='utf8'))
            if prior!=identity:raise ContractError('Resume equilibrium/input/source identity differs')
            restored=load_bundle(checkpoint,expected_kind='complete_model_checkpoint',expected_identity=identity)['payload']
            arm=GroupTimeArm.from_checkpoint(assembly,seed,protocol,restored,mode='reference',guard=worker.check)
            monitor=GroupTimeMonitor.from_payload(assembly,seed['states'],restored['monitor'],parameters=seed['inner_parameters'])
            if causal_hash(arm.checkpoint_payload(monitor=monitor.payload()))!=causal_hash(restored):raise ContractError('Full restored checkpoint differs')
            init=json.loads((checkpoint.parent/'initial_seed_match.json').read_text(encoding='utf8'))
            if not init['passed']:raise ContractError('Original run has no passed pre-commit initial-match receipt')
        else:
            arm=construct_arm(assembly,seed,protocol,worker.check)
            monitor=GroupTimeMonitor(assembly,seed['states'],parameters=seed['inner_parameters'])
        dynamic=DynamicDefectMonitor(assembly,seed['states']);observations=[];edges=[]
        write(output/'model_identity.json',identity)
        def observe(t,side,record=False):
            snapshot=arm.snapshot(t,side,queue_node=True)
            monitor.observe(t,side,snapshot['states'],snapshot['rhs'],transitions=snapshot['transitions'],
                ports=snapshot['ports'],aux=snapshot['aux'],event_generation=snapshot['event_generation'])
            if monitor.domain_failures:raise ContractError('Physical/history/conservation violation '+str(monitor.domain_failures[:3]))
            if record:observations.append(snapshot)
            return snapshot
        try:
            if not resume:
                init,evidence=initial_match(arm,saved)
                write(output/'initial_seed_match.json',init)
                save_bundle(output/'precommit_initial_conditions',kind='validation_evidence',identity=identity,payload=evidence,qualification=QUALIFICATION)
                observe(arm.start,'right');arm.initial_commit();observe(arm.start,'right',True);arm.ack(arm.start)
                write(output/'post_initial_commit.json',dict(expected_original_sampling=True,initial_commit_done=arm.initial_commit_done,
                    event_generation=arm.event_generation,spike_states_sha256=canonical_hash(arm.spike_states),
                    equality_to_uncommitted_seed_is_not_required=True))
            events={t:(rod,bc) for t,rod,bc in clock_events(protocol)}
            for i in range(int(arm.start*150)+1,int(arm.stop*150)+1):
                edge=i/150.
                if arm.start<edge<arm.stop:events.setdefault(edge,(False,False))
            for end,(rod,bc) in sorted(events.items()):
                if end<=arm.inner_time+1e-13:continue
                begin=arm.inner_time;arm.ensure_native(end,'left');advance=arm.advance_inner(begin,end)
                for curve in advance.dense_segments:
                    midpoint=curve.t0+(curve.t1-curve.t0)/2
                    dynamic_observe(arm,advance,midpoint,dynamic,'left' if midpoint==advance.t1 else 'right')
                frame_edge=abs(end*150-round(end*150))<1e-10
                before=observe(end,'left' if frame_edge else 'right')
                if rod or bc:arm.commit(end,rod=rod,bc=bc,rgc=bc,rod_sample_time=max(arm.start,end-.0001+1e-13) if rod else None)
                after=observe(end,'right',True)
                if frame_edge:
                    continuous=[k for k in after['states'] if k[1] not in ('Chen2024PrimateRodCurrent','RGC_expected_history') and not (k[1]=='HistoryBipolarTerminal' and (k[2].startswith('history_') or k[2]=='activation'))]
                    edges.append(dict(time_seconds=end,continuous_state_max_jump=max((float(np.max(np.abs(before['states'][k]-after['states'][k]),initial=0.)) for k in continuous),default=0.)))
                arm.ack(end)
                if not resume and math.isclose(end,protocol['checkpoint_seconds'],rel_tol=0,abs_tol=1e-13):
                    save_bundle(output/'checkpoint',kind='complete_model_checkpoint',identity=identity,
                        payload=arm.checkpoint_payload(monitor=monitor.payload()),qualification=QUALIFICATION)
            payload=arm.checkpoint_payload(monitor=monitor.payload())
            save_bundle(output/'final_checkpoint',kind='complete_model_checkpoint',identity=identity,payload=payload,qualification=QUALIFICATION)
            save_traces(output,assembly,observations)
            defects=dynamic.report();physical=not monitor.domain_failures;input_pass=all(e['continuous_state_max_jump']==0 for e in edges)
            passed=init['passed'] and defects['passed'] and physical and input_pass and arm.metrics['inner_advances']>0
            result=dict(schema=SCHEMA,status='SAVED_LOCAL_EQUILIBRIUM_CONNECTED_TO_DYNAMIC_ADVANCE' if passed else 'ENGINEERING_GATE_FAILED',
                passed=passed,initial_saved_seed_match=init,from_saved_equilibrium_seed=True,saved_background_reused=True,
                DYNAMIC_NUMERICAL_DEFECT=defects,PHYSICAL_DOMAIN_AND_CONSERVATION=dict(passed=physical,failures=monitor.domain_failures),
                INPUT_EVENT_AND_HISTORY=dict(passed=input_pass,edges=edges,clocks=payload['clocks'],input_position=payload['input_position']),
                raw_absolute_equilibrium_monitor=monitor.report(arm.stop-arm.start),final_causal_sha256=causal_hash(payload),
                resumed_in_new_process=bool(resume),final_time_seconds=arm.inner_time,runtime_metrics=arm.metrics,
                whole_retina_or_arbitrary_exterior_background_qualified=False,real_movie_qualified=False,
                FIT_READY=False,**QUALIFICATION)
            write(output/'result.json',result);return result
        finally:arm.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','run'));parser.add_argument('--initial-seed',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--prepared');parser.add_argument('--resume')
    args=parser.parse_args();output=local_path(args.output)
    if not output.is_relative_to(REPORT) or output.exists():raise ContractError('Fresh output in independent equilibrium-start report root required')
    if args.mode=='prepare' and (args.prepared or args.resume):raise ContractError('Prepare does not run or resume a model')
    if args.mode=='run' and not args.prepared:raise ContractError('Explicit already frozen --prepared directory required')
    output.mkdir(parents=True)
    try:
        result=prepare(args.initial_seed,output) if args.mode=='prepare' else run(args.prepared,args.initial_seed,output,resume=args.resume)
        print(json.dumps(dict(status=result['status'],output=str(output))))
    except Exception as exc:
        write(output/'failure.json',dict(status='FAILED_PRESERVED',error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
