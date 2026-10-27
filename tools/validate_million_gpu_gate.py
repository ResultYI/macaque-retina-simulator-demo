"""One immutable, bounded 100us full-graph GPU RHS feasibility gate.

Reference DOP853 supplies every real stage state. The entire inner RHS is
replayed on the GPU, with persistent buffers and mandatory status synchronization.
This is a gate for subsequent solver/lifecycle integration, not a GPU trajectory
or a 7ms end-to-end speed claim. A failed gate cannot be resumed or overwritten.
"""
from pathlib import Path
import json
import os
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from scipy.integrate import DOP853
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.group_time_storage import load_bundle
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.group_time_monitor import frozen_budget,rhs_allowance,_contract
from retina_sim.spatial_group_model import GroupModelAssembly
from retina_sim.spatial_group_graph import digest
from retina_sim.group_gpu_rhs import GroupGpuRHS
from tools.validate_group_reference_backend import protocol,implementation_identity,PREPARATION,ARTIFACT

REPORT=ROOT/'reports/million_gpu_resident_20261025_v1'


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def run():
    out=REPORT/'gate_r1'
    if out.exists():
        raise ContractError('This gate already exists: failed/completed/stopped runs cannot be resumed or overwritten')
    compatibility=json.loads((REPORT/'compatibility.json').read_text(encoding='utf8'))
    if compatibility['status']!='PASS_SELECTED_CUDA126_EXECUTION_ONLY':
        raise ContractError('Selected CUDA 12.6 numerical compatibility must pass first')
    out.mkdir()
    begun=time.perf_counter()
    started=None
    p=protocol()
    gate=dict(schema='retipath-million-gpu-feasibility-1',reference_protocol=p,
        start_seconds=p['start_seconds'],stop_seconds=4.9931,physiological_interval_seconds=.0001,
        expected_groups=301814,expected_inner_states=1835686,
        acceptance=dict(median_full_rhs_including_scalar_check_and_sync_ms=2.0,
                        required_numerical_allocation_sha256=p['numerical_budget']['sha256'],
                        all_original_domain_checks=True),
        measurement='One cold complete RHS separately; every stage and dense RHS of the original 100us reference DOP853, no synthetic state or subgraph',
        host_transfer='State upload and full derivative download for comparison separately timed; they are excluded from resident RHS time',
        followup='Only a passed gate authorizes integration work; failure stops GPU migration',
        gpu_trajectory_integrated=False,legacy_failed_jobs_resumed=False,
        runtime_limits=p['runtime_limits'],laptop_physical_vram_guard='Also stop above actual total minus 512 MiB',
        compatibility_sha256=digest(REPORT/'compatibility.json'))
    write(out/'protocol.json',gate)
    pins=json.loads((ROOT/'reports/spatial_group_graph_v02_20261025_v1/verification.json').read_text())['unchanged_pins']
    protected={name:digest(ROOT/name) for name in pins}
    if protected!=pins:
        raise ContractError('Original STOP/source pin changed before the gate')
    memory=MemoryGuard(16*1024**3,4*1024**3)
    torch=arm=gpu=None
    stage='COLD_PREPARATION'
    samples=[]
    peak_gpu=0
    phases={}
    checks={}
    def guard():
        nonlocal peak_gpu
        if (out/'REQUEST_STOP').exists() or (REPORT/'REQUEST_STOP').exists():
            raise ContractError('Explicit million-state gate STOP')
        elapsed=time.perf_counter()-(begun if started is None else started)
        if elapsed>(600. if started is None else 180.):
            raise ContractError('Original preparation/model wall limit reached; do not retry or expand')
        if not memory.check():
            raise ContractError('Original RAM guard: '+str(memory.stop_reason))
        if torch is not None:
            free,total=torch.cuda.mem_get_info()
            peak_gpu=max(peak_gpu,total-free)
            if total-free>min(12*1024**3,total-512*1024**2):
                raise ContractError('GPU memory guard reached')
    def timed(name,fn):
        now=time.perf_counter()
        value=fn()
        phases[name]=phases.get(name,0.)+time.perf_counter()-now
        return value
    def status():
        guard()
        write(out/'status.json',dict(stage=stage,rhs_samples=len(samples),resources=memory.last,
             elapsed_model_seconds=None if started is None else time.perf_counter()-started,
             whole_gpu_sampled_peak_bytes=peak_gpu))
    try:
        import torch as module
        torch=module
        torch.cuda.init()
        status()
        bundle=timed('seed_load',lambda:load_bundle(PREPARATION/'construction_seed',expected_kind='source_construction_seed'))
        assembly=timed('group_assembly',lambda:GroupModelAssembly(ARTIFACT))
        if sum(pop.group_count for pop in assembly.populations.values())!=301814:
            raise ContractError('Real fixed group count differs')
        arm=timed('reference_arm_build',lambda:GroupTimeArm(assembly,bundle['payload'],p,mode='reference',torch=torch,device='cuda:0',guard=guard))
        if arm.inner.nstate!=1835686:
            raise ContractError('Full inner-state count differs')
        gpu=timed('gpu_compile_and_resident_plan',lambda:GroupGpuRHS(arm.compiled,torch))
        (out/'generated_rhs.cu').write_text(gpu.source,encoding='utf8')
        write(out/'backend.json',dict(identity=gpu.identity,runtime=gpu.module.identity,metrics=gpu.metrics,nvrtc_log=gpu.module.compile_log))
        _contract()
        # Freeze the original state scales before either numerical evaluation.
        budgets={}
        for block in arm.inner.blocks:
            key=(block.population,block.mechanism,block.name)
            scale,_,budget=frozen_budget(bundle['payload']['states'][key],block.unit)
            budgets[key]=(scale,budget)
            checks[':'.join(key)]=dict(max_abs_error=0.,max_allowance_ratio=0.,values_compared=0)
        identity=dict(protocol=canonical_hash(gate),implementation=implementation_identity(),
            tool=digest(Path(__file__)),backend=gpu.identity,graph=assembly.artifact_sha256,
            seed_manifest=digest(PREPARATION/'construction_seed/manifest.json'),protected=protected,
            initial_qualification=bundle['qualification'])
        write(out/'prospective_identity.json',identity)
        phases['cold_prepare']=time.perf_counter()-begun
        stage='ORIGINAL_INITIAL_COMMIT'
        started=time.perf_counter()
        timed('original_initial_bc_rgc_commit',arm.initial_commit)
        initial=arm.inner.pack(arm.inner_state)
        input_device=torch.as_tensor(initial.copy(),device='cuda')
        output_device=torch.empty_like(input_device)
        reference=arm.compiled._reference_interval_rhs(initial)
        # First-call compilation/driver residency is explicitly separate, never hidden.
        cold_reference=timed('cpu_first_rhs',lambda:reference(p['start_seconds'],initial))
        torch.cuda.synchronize()
        timed('gpu_first_rhs_including_sync',lambda:gpu.evaluate_into(p['start_seconds'],input_device,output_device))
        cold_actual=output_device.cpu().numpy()
        def compare(expected,actual):
            if not np.isfinite(actual).all():
                raise ContractError('Nonfinite complete GPU derivative')
            for key,section in arm.inner.slices.items():
                ref,got=expected[section],actual[section]
                scale,budget=budgets[key]
                allowance=rhs_allowance(scale,budget,ref,got)
                error=np.abs(ref-got)
                row=checks[':'.join(key)]
                row['max_abs_error']=max(row['max_abs_error'],float(error.max(initial=0)))
                row['max_allowance_ratio']=max(row['max_allowance_ratio'],float((error/allowance).max(initial=0)))
                row['values_compared']+=len(got)
                if np.any(error>allowance):
                    raise ContractError('Original frozen RHS allocation exceeded: '+':'.join(key))
        compare(cold_reference,cold_actual)
        stage='FULL_100US_REFERENCE_DOP853_GPU_RHS_REPLAY'
        def rhs(t,y):
            guard()
            a=time.perf_counter()
            expected=reference(t,y)
            cpu_seconds=time.perf_counter()-a
            a=time.perf_counter()
            input_device.copy_(torch.from_numpy(y))
            torch.cuda.synchronize()
            upload_seconds=time.perf_counter()-a
            a=time.perf_counter()
            gpu.evaluate_into(t,input_device,output_device)
            gpu_seconds=time.perf_counter()-a
            a=time.perf_counter()
            actual=output_device.cpu().numpy()
            download_seconds=time.perf_counter()-a
            compare(expected,actual)
            samples.append(dict(time_seconds=float(t),cpu_full_rhs_seconds=cpu_seconds,
                gpu_full_rhs_including_sync_seconds=gpu_seconds,
                reference_state_H2D_seconds=upload_seconds,comparison_rhs_D2H_seconds=download_seconds))
            return expected
        status()
        a=time.perf_counter()
        solver=DOP853(rhs,p['start_seconds'],initial,gate['stop_seconds'],rtol=1e-6,atol=1e-9,max_step=.0001)
        accepted=[]
        while solver.status=='running':
            message=solver.step()
            if solver.status=='failed':
                raise ContractError('Original reference DOP853 failed: '+str(message))
            solver.dense_output()
            accepted.append(float(solver.t))
        phases['reference_100us_with_gpu_rhs_replay_and_all_comparisons']=time.perf_counter()-a
        if solver.t!=gate['stop_seconds'] or len(samples)!=solver.nfev:
            raise ContractError('Incomplete real 100us interval or RHS coverage')
        stage='GATE_DECISION'
        status()
        gpu_ms=np.array([s['gpu_full_rhs_including_sync_seconds'] for s in samples])*1000
        cpu_ms=np.array([s['cpu_full_rhs_seconds'] for s in samples])*1000
        passed=bool(np.median(gpu_ms)<=2.)
        result=dict(status='PASS_RHS_FEASIBILITY_ONLY' if passed else 'STOP_GPU_MIGRATION_PERFORMANCE_GATE',
            integration_allowed=passed,physiological_interval_seconds=.0001,reference_accepted_times=accepted,
            reference_nfev=solver.nfev,gpu_complete_rhs_comparisons=len(samples)+1,
            numerical_gate='PASS_ORIGINAL_FULL_STATE_RHS_ALLOCATION',domain_checks='ALL_ORIGINAL_CHECKS_PASS',
            gpu_rhs_median_ms=float(np.median(gpu_ms)),gpu_rhs_p95_ms=float(np.percentile(gpu_ms,95)),
            gpu_rhs_min_ms=float(gpu_ms.min()),gpu_rhs_max_ms=float(gpu_ms.max()),
            cpu_rhs_median_ms=float(np.median(cpu_ms)),cpu_rhs_p95_ms=float(np.percentile(cpu_ms,95)),
            same_laptop_local_rhs_speedup=float(np.median(cpu_ms)/np.median(gpu_ms)),
            full_7ms_neural_speedup=None,complete_gpu_trajectory=False,complete_observer_monitor_checkpoint_integration=False,
            phase_seconds=phases,model_wall_seconds=time.perf_counter()-started,
            whole_gpu_sampled_peak_bytes=peak_gpu,resources=memory.last,backend=gpu.metrics,
            pins_unchanged=all(digest(ROOT/name)==value for name,value in protected.items()))
        if not result['pins_unchanged']:
            raise ContractError('Original STOP/source pins changed during gate')
        write(out/'samples.json',samples)
        write(out/'full_rhs_numerical_checks.json',checks)
        write(out/'result.json',result)
        if not passed:
            write(out/'failure.json',dict(stage=stage,message='Complete synchronized RHS median exceeds prospective 2 ms target',result=result))
            (out/'REQUEST_STOP').write_text('GPU migration stopped at prospective feasibility gate. Do not resume or extend this run.\n',encoding='utf8')
        print(json.dumps(result,indent=2))
        return result
    except Exception as exc:
        write(out/'samples.json',samples)
        write(out/'full_rhs_numerical_checks.json',checks)
        write(out/'failure.json',dict(stage=stage,error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
            phase_seconds=phases,model_wall_seconds=None if started is None else time.perf_counter()-started,
            resources=memory.last,whole_gpu_sampled_peak_bytes=peak_gpu,rhs_samples=len(samples)))
        (out/'REQUEST_STOP').write_text('Failed gate; do not resume or expand budgets.\n',encoding='utf8')
        raise
    finally:
        if gpu is not None: gpu.close()
        if arm is not None: arm.close()


if __name__=='__main__':
    run()
