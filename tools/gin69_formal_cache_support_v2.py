"""Continuous real-movie target trajectory with clean, complete 0.5s handoffs."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from retina_sim.contracts import canonical_hash
from retina_sim.group_time_storage import save_bundle, load_bundle
from retina_sim.spatial_group_graph import digest
from tools.gin69_frame_index_support_v2 import IndexedGIN69Runner as GuardedCompiledCell6Runner
from tools.cell6_compiled_inner_v1 import PHYSICAL
from tools.cell6_runtime_monitor_v3 import Cell6Monitor
from tools.durable_research_runtime_v1 import atomic_json
from tools.validate_ac_vs_timescale import calendar
from tools.gin_hc2_bc1_context_support_v2 import ROOT, OUT, read, load, protocol, background_directory


def run(output, guard):
    began=time.perf_counter();request=read(OUT/'FORMAL_SEGMENT_REQUEST.json')
    if Path(request['output']).resolve()!=output.resolve():raise ValueError('Segment ownership mismatch')
    index=int(request['index']);end_segment=(index+1)*.5
    if not 0<=index<90:raise ValueError('Only approved45s prefix supported')
    cache=OUT/'formal_neural';cache.mkdir(exist_ok=True)
    inputs=read(OUT/'FORMAL_INPUT_LOCATION.json');source=Path(inputs['path'])
    if digest(source/'RESULT.json')!=inputs['result_sha256']:raise ValueError('Formal source receipt changed')
    evidence=read(source/'RESULT.json')
    if digest(source/'NATIVE_INPUTS.npz')!=evidence['array_sha256']:raise ValueError('Formal native input changed')
    if evidence['split_sha256']!=digest(OUT/'SPLIT_CONTRACT.json'):raise ValueError('Split changed')
    assembly,seed=load();seed=deepcopy(seed);p=protocol()
    p.update(stop_seconds=45.,input_frame_indices=list(range(6750)),checkpoint_seconds=.5)
    with np.load(source/'NATIVE_INPUTS.npz',allow_pickle=False) as z:drive=z['drive'].copy()
    if drive.shape!=(6750,assembly.populations['cones'].group_count):raise ValueError('Native member projection shape differs')
    seed['native_inputs']={i:drive[i].copy() for i in range(6750)}
    rod=np.full(assembly.populations['rods'].group_count,float(seed['input_reference']['rod_background_rstar_s']))
    seed['rod_inputs']={i:rod.copy() for i in range(6750)}
    population='parasol_RGC_OFF';pop=assembly.populations[population]
    rows=np.flatnonzero(pop.group_ids==7730)
    if len(rows)!=1:raise ValueError('Previously bound69#6 target7730 absent')
    row=int(rows[0]);members=int(pop.counts[row])
    identity=canonical_hash(dict(source=inputs['result_sha256'],model=digest(background_directory()/'seed/manifest.json'),
        artifact=assembly.artifact_sha256,protocol=p,target=7730,members=members,
        code={name:digest(ROOT/'tools'/name) for name in ('gin69_formal_cache_support_v2.py','gin_hc2_bc1_context_support_v2.py',
            'cell6_compiled_kernel_guard_v2.py','cell6_compiled_inner_v1.py','cell6_raw_runner_v1.py','cell6_runtime_monitor_v3.py','gin69_frame_index_support_v2.py')}))
    identity_path=cache/'IDENTITY.json'
    meta=dict(identity=identity,source_cell='69#6',true_source_ecc_deg=3.88,population=population,target_group=7730,
        local_row=row,members=members,target_centroid_deg=(pop.centroid_mm[row]/.223).tolist(),
        source_plan_sha256=evidence['source_plan_sha256'],split_sha256=evidence['split_sha256'],protocol=p,
        no_observed_spikes_used=True,conditional_reference_meridian=True,brightness_literature_approximation=True,
        authoring_phase='post clock commit and right input side, same as short trajectory; final stop uses left side',
        frame_lookup='cached original float boundaries, bisect, exact original side predicates; no input mutation',
        FIT_READY=False,background_qualified=False,production_adopted=False)
    if identity_path.exists():
        if read(identity_path)!=meta:raise ValueError('Existing formal trajectory identity differs')
    else:atomic_json(identity_path,meta)
    arrays={name:np.empty(450001) for name in ('V','gE','gI')};expected=np.empty(45000)
    recorded=bins=0;arm=None;load_seconds=time.perf_counter()-began
    try:
        construct=time.perf_counter()
        if index:
            previous=read(cache/f'SEGMENT_{index-1:03d}.json')
            predecessor=Path(previous['execution'])
            if read(predecessor/'TERMINAL.json')['status']!='EXITED_0' or read(predecessor/'RESOURCE.json')['status']!='PASS':
                raise ValueError('Only clean predecessor checkpoint admitted; no failed-native prefix reuse')
            checkpoint=cache/previous['checkpoint']
            if previous['identity']!=identity or digest(checkpoint/'manifest.json')!=previous['checkpoint_sha256']:
                raise ValueError('Clean prefix identity changed')
            saved=load_bundle(checkpoint,expected_kind='validation_evidence',expected_identity=dict(cache_identity=identity))['payload']
            arm=GuardedCompiledCell6Runner.from_checkpoint(assembly,seed,p,saved,guard=guard)
            copied=arm.checkpoint_payload(monitor=saved['monitor'],recording=saved['recording'])
            if any(canonical_hash(saved[k])!=canonical_hash(copied[k]) for k in PHYSICAL):
                raise ValueError('Full physical/PCG/history restoration differs')
            monitor=Cell6Monitor.from_payload(assembly,seed['states'],saved['monitor'],parameters=seed['inner_parameters'])
            recording=saved['recording'];recorded=recording['recorded'];bins=recording['bins']
            for name in arrays:arrays[name][:recorded]=recording['arrays'][name]
            expected[:bins]=recording['expected']
            if abs(arm.inner_time-index*.5)>1e-12:raise ValueError('Segment clock discontinuity')
        else:
            arm=GuardedCompiledCell6Runner(assembly,seed,p,precision='standard',guard=guard)
            monitor=Cell6Monitor(assembly,seed['states'],parameters=seed['inner_parameters'])
        construction_seconds=time.perf_counter()-construct
        prior_metrics=deepcopy(arm.metrics)
        def observe(t,side):
            snapshot=arm.snapshot(t,side,queue_node=True)
            monitor.observe(t,side,snapshot['states'],snapshot['rhs'],transitions=snapshot['transitions'],
                ports=snapshot['ports'],aux=snapshot['aux'],event_generation=snapshot['event_generation'])
            if monitor.domain_failures:raise ValueError('Original full-chain finite/domain/history gate failed')
            return snapshot
        def record(snapshot,j):
            nonlocal recorded,bins
            if j!=recorded:raise ValueError('Authoring node repeated or omitted')
            for name,port,unit in [('V','voltage','V'),('gE','gE','S'),('gI','gI','S')]:
                if snapshot['port_units'][population,port]!=unit:raise ValueError('Target port units differ')
                arrays[name][j]=snapshot['ports'][population,port][row]
            if j%10==0 and j<450000:
                expected[bins]=snapshot['ports'][population,'expected_rate_Hz'][row];bins+=1
            recorded+=1
        if not index:
            observe(0.,'right');arm.initial_commit();first=observe(0.,'right');record(first,0);arm.ack(0.)
        phase=time.perf_counter()
        for end,flags in calendar(p):
            if end<=arm.inner_time:continue
            if end>end_segment+1e-12:break
            guard();begin=arm.inner_time;arm.ensure_native(end,'left');arm.advance_inner(begin,end)
            observe(end,'left' if flags['frame'] or end==arm.stop else 'right')
            j=int(round(end/.0001));author=flags['rod'] or end==arm.stop
            if flags['rod'] or flags['bc']:
                arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],
                    rod_sample_time=max(0.,end-.0001+1e-13) if flags['rod'] else None)
            after=observe(end,'left' if end==arm.stop else 'right');arm.ack(end)
            if author:record(after,j)
            if author and j%1000==0:
                progress=dict(stage='neural',segment=index,source_seconds=end,target_seconds=45.,
                    segment_wall_seconds=time.perf_counter()-began,recorded=recorded,
                    cumulative_native_seconds=arm.metrics['native_wall_seconds'],cumulative_inner_seconds=arm.metrics['inner_wall_seconds'])
                atomic_json(cache/'PROGRESS.json',progress);print(str(progress),flush=True)
        loop_seconds=time.perf_counter()-phase;phase=time.perf_counter()
        stop_node=round(end_segment/.0001)
        if recorded!=stop_node+1:raise ValueError('Segment coverage incomplete')
        name=f'CHECKPOINT_{stop_node:07d}'
        payload=arm.checkpoint_payload(monitor=monitor.payload(),recording=dict(identity=identity,recorded=recorded,bins=bins,
            arrays={k:v[:recorded].copy() for k,v in arrays.items()},expected=expected[:bins].copy()))
        save_bundle(cache/name,kind='validation_evidence',identity=dict(cache_identity=identity),payload=payload,
            qualification=dict(default_adopted=False,fit_eligible=False))
        if end_segment in (30.,45.):
            length=round(end_segment/.0001)
            np.savez_compressed(cache/f'PREFIX_{int(end_segment)}s.npz',time_source_s=np.arange(length+1)*.0001,
                **{k:v[:length+1] for k,v in arrays.items()},expected_rate_base_Hz=expected[:int(end_segment*1000)])
            atomic_json(cache/f'END_HISTORY_{int(end_segment)}s.json',arm.spike_states)
            atomic_json(cache/f'GROUP_EVENTS_{int(end_segment)}s.json',
                [dict(population=pop_name,**e) for pop_name,events in arm.event_log.items() for e in events if e['time_seconds']<end_segment])
        write_seconds=time.perf_counter()-phase
        result=dict(status='COMPLETE_CLEAN_SEGMENT_PENDING_NORMAL_EXIT',index=index,identity=identity,execution=str(output),
            source_start=index*.5,source_stop=end_segment,recorded=recorded,bins=bins,checkpoint=name,
            checkpoint_sha256=digest(cache/name/'manifest.json'),checkpoint_restored_exact=bool(index),
            source_seconds=end_segment,load_seconds=load_seconds,construction_seconds=construction_seconds,
            loop_seconds=loop_seconds,write_seconds=write_seconds,wall_seconds=time.perf_counter()-began,
            metrics_before=prior_metrics,metrics_after=arm.metrics,compiled_build=arm.compiled_build,
            compiled_rhs_calls=arm.compiled._calls,python_continuous_fallback_calls=0,
            physical_domain_history_passed=True,FIT_READY=False,background_qualified=False,production_adopted=False)
        atomic_json(cache/f'SEGMENT_{index:03d}.json',result);atomic_json(output/'RESULT.json',result)
    finally:
        if arm is not None:arm.close()
