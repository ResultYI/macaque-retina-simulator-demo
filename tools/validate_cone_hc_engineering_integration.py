"""Configured native equivalence and existing full inner-runner port smoke test."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from datetime import datetime,timezone
import argparse
import time
import traceback
import numpy as np
from retina_sim.assembly import RetinaBuilder
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.cone_hc_spatial import ConeHCSpatialRunner,FineMemberReleaseProjection,ConeHCGroupPortRunner
from retina_sim.group_time_storage import save_bundle,load_bundle
from retina_sim.group_inner_dynamics import BC_MECH
from retina_sim.spatial_group_model import BC
from retina_sim.h1_h2_effective import matrix_id
from tools.run_hc_minimal_real_diagnostic_v1 import write,sha,peak_memory_bytes


def downstream(model,source_runner,out,*,source_start_seconds=.02):
    from tools.gin_hc2_bc1_context_support_v2 import load,protocol
    from tools.validate_h1_h2_effective import tick
    assembly,seed=load();p=protocol();origin=p['clock_origin_seconds']
    index0=int(round((seed['time_seconds']-origin)/.0001))
    p['start_seconds']=seed['time_seconds'];p['stop_seconds']=origin+(index0+20)*.0001
    native=assembly.runtime_factories['native'](assembly)
    bg=np.asarray(seed['input_reference']['native_background_td'])
    y,anchor=native.construction_point(SimpleNamespace(states=seed['states'],time_seconds=seed['time_seconds']),bg,seed['time_seconds']+1.,0)
    reference=native.evaluate(y,seed['time_seconds'],anchor).outputs['release']
    projection=FineMemberReleaseProjection(model,assembly.populations['cones'],reference,target_identity=assembly.artifact_sha256)
    options=dict(source_runner=source_runner,projection=projection,source_start_seconds=source_start_seconds)
    arm=ConeHCGroupPortRunner(assembly,seed,p,**options);restored=None;started=time.perf_counter()
    frozen_parameters=canonical_hash(seed['inner_parameters']);rows=[];max_restore=0.;port_checks=0
    h2_identity=matrix_id(arm.native_physics.A[arm.native_physics.h2_rows])
    try:
        initial=arm.snapshot(arm.start,'right',queue_node=True)
        arm.initial_commit();arm.ack(arm.start)
        for k in range(1,21):
            idx=index0+k;end=origin+idx*.0001
            left,right=tick(arm,end,idx)
            for section in ('states','rhs','ports'):
                if any(not np.isfinite(v).all() for v in right[section].values()):raise ContractError('Nonfinite downstream '+section)
            if idx%10==0:
                for pop in BC:
                    route=next(r for r in arm.inner.routes if r.pre=='cones' and r.post==pop)
                    expected=arm.inner.mean[route.name]@left['ports']['cones','release']
                    np.testing.assert_array_equal(expected,left['ports'][pop,'drive'])
                    np.testing.assert_array_equal(expected-arm.inner.par(pop,'baseline'),right['states'][pop,BC_MECH,'history_0'])
                    port_checks+=1
            rows.append(right)
            if k==10:
                identity=dict(execution=arm.identity,source_mode=model.identity,projection=projection.identity)
                save_bundle(out/'downstream_checkpoint',kind='complete_model_checkpoint',identity=identity,
                            payload=arm.checkpoint_payload(),qualification=dict(default_adopted=False,fit_eligible=False,research_only=True))
                saved=load_bundle(out/'downstream_checkpoint',expected_kind='complete_model_checkpoint',expected_identity=identity)
                restored=ConeHCGroupPortRunner.from_checkpoint(assembly,seed,p,saved['payload'],**options)
                bad=deepcopy(saved['payload']);bad['controller']['research_cone_release_port']='different-mode'
                try:ConeHCGroupPortRunner.from_checkpoint(assembly,seed,p,bad,**options)
                except ContractError:pass
                else:raise ContractError('Mismatched downstream release checkpoint accepted')
            elif k>10:
                _,other=tick(restored,end,idx)
                for section in ('states','rhs','ports'):
                    if set(right[section])!=set(other[section]):raise ContractError('Restored registry changed')
                    for key,value in right[section].items():max_restore=max(max_restore,float(np.max(abs(value-other[section][key]),initial=0.)))
                if canonical_hash(arm.spike_states)!=canonical_hash(restored.spike_states):raise ContractError('RGC RNG/history restoration differs')
        if max_restore!=0:raise ContractError('Downstream continuation differs')
        if canonical_hash(seed['inner_parameters'])!=frozen_parameters:raise ContractError('Downstream parameters changed')
        if matrix_id(arm.native_physics.A[arm.native_physics.h2_rows])!=h2_identity:
            raise ContractError('Independent downstream H2 operator changed')
        traces={'time_seconds':np.array([r['time_seconds'] for r in rows])}
        for key in rows[-1]['states']:
            if key[0] not in ('cones','HC'):traces['STATE__'+'__'.join(key)]=np.column_stack([r['states'][key] for r in rows])
        for key in rows[-1]['ports']:traces['PORT__'+'__'.join(key)]=np.column_stack([r['ports'][key] for r in rows])
        np.savez_compressed(out/'DOWNSTREAM_STATES.npz',**traces)
        return dict(passed=True,wall_seconds=time.perf_counter()-started,duration_seconds=.002,
            source_assembly=assembly.artifact_sha256,execution=arm.identity,projection_identity=projection.identity,
            mapped_source_fine_cones=projection.mapped_fine_count,BC_drive_checks=port_checks,
            state_registry_unchanged=True,parameters_unchanged=True,all_finite=True,
            H2_rows=len(arm.native_physics.h2_rows),H2_matrix_identity=h2_identity,H2_unchanged=True,
            exact_restored_max_difference=max_restore,RGC_history_and_RNG_equal=True,
            populations={name:layout.group_count for name,layout in assembly.populations.items()},
            port_units={str(k):v for k,v in rows[-1]['port_units'].items()},
            scope='one-way fine-member release perturbation into existing group ports and unchanged scheduler; parent auxiliary outer states retained separately')
    finally:
        arm.close()
        if restored is not None:restored.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();result=dict(started_utc=datetime.now(timezone.utc).isoformat(),status='RUNNING',modes={})
    try:
        v6=root/'reports/CONE_HC_MINIMAL_REAL_DIAGNOSTIC_20261102_v6/run_02';models={};sources={}
        for mode,condition in [('uniform','A_uniform'),('biexponential','B_biexponential')]:
            model=RetinaBuilder.build_research_cone_hc(root,'configurations/research/cone_hc_'+mode+'.json');models[mode]=model
            runner=model.runner()
            if mode=='uniform':data=runner.advance(.12,np.arange(61)*.002)
            else:
                first=runner.advance(.06,np.arange(31)*.002);checkpoint=runner.checkpoint_payload()
                save_bundle(out/'native_checkpoint',kind='cone_hc_spatial_checkpoint',identity=dict(model=model.identity),payload=checkpoint,
                            qualification=dict(default_adopted=False,fit_eligible=False,research_only=True))
                loaded=load_bundle(out/'native_checkpoint',expected_kind='cone_hc_spatial_checkpoint',expected_identity=dict(model=model.identity))
                resumed=ConeHCSpatialRunner.from_checkpoint(model,loaded['payload'])
                try:ConeHCSpatialRunner.from_checkpoint(models['uniform'],loaded['payload'])
                except ContractError:pass
                else:raise ContractError('Uniform accepted biexponential checkpoint')
                tail=runner.advance(.12,np.arange(31,61)*.002);other=resumed.advance(.12,np.arange(31,61)*.002)
                restart=max(float(np.max(abs(tail[k]-other[k]))) for k in tail)
                if restart>1e-7:raise ContractError('Native resumed trajectory differs')
                data={k:np.concatenate([first[k],tail[k]],axis=1 if first[k].ndim==2 else 0) for k in first}
                result['native_checkpoint']=dict(restored_max_difference=restart,cross_mode_rejected=True)
            np.savez_compressed(out/(mode+'_TRAJECTORY.npz'),**data)
            with np.load(v6/(condition+'_TRAJECTORY.npz')) as reference:
                error={k:float(np.max(abs(data[k]-reference[k]))) for k in ('H1_target','release_core','pooled_release_target','feedback_core','y')}
            if max(error.values())>1e-7:raise ContractError('Configured path differs from independent result: '+str(error))
            result['modes'][mode]=dict(identity=model.identity,native_identity=model.native.source_version,
                background_max_rhs=model.background_rhs,comparison_max_abs=error,runner_metrics=runner.metrics,
                state_layout=model.state_layout,qualification=model.permission)
            sources[mode]=runner
            print('Native mode PASS',mode,error,flush=True)
        result['downstream']=downstream(models['biexponential'],sources['biexponential'],out)
        result.update(status='CONE_HC_BIEXP_ENGINEERING_INTEGRATED',physiological_prediction_qualified=False,
            spatial_convergence='FAIL_RETAINED',production_default='uniform',
            scope='configured pinned-local CPU research runner and one-way existing downstream group-port adapter; no generic full-parent rebuild')
    except Exception as error:
        result.update(status='FAILED_RETAINED',error=str(error),traceback=traceback.format_exc());raise
    finally:
        result.update(ended_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-start,peak_working_set_bytes=peak_memory_bytes());write(out/'RESULT.json',result)
    print(result['status'],result['wall_seconds'],flush=True)


if __name__=='__main__':main()
