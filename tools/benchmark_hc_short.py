"""One frozen real graph, short HC diagnosis and one selected equivalent change.

No background preparation, new optics or closure search. Device events and one
instrumented window are diagnostic only; uninstrumented full windows are timed
separately with the same original numerical monitor and selected recording.
"""
from pathlib import Path
from dataclasses import replace
from contextlib import contextmanager
import argparse,copy,gc,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def save(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False,sort_keys=True)+'\n',encoding='utf8')
    temporary.replace(path)


class Instrumentation:
    def __init__(self,runner):
        import torch
        self.torch=torch;self.runner=runner;self.events=[];self.wall={};self.calls={};self.old=[]
    @contextmanager
    def span(self,name):
        tick=time.perf_counter()
        with self.torch.profiler.record_function(name):yield
        self.wall[name]=self.wall.get(name,0.)+time.perf_counter()-tick
        self.calls[name]=self.calls.get(name,0)+1
    def install(self):
        owner=self;native=self.runner.partition.native;torch=self.torch
        class Proxy:
            def __getattr__(self,name):return getattr(torch,name)
            def mv(self,matrix,vector):
                count=sum(label.startswith('HC_SPMV_A') for label,*_ in owner.events)
                label='HC_SPMV_B_h' if matrix is native.B else ('HC_SPMV_A_Vbar' if count%2==0 else 'HC_SPMV_A_z1')
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                with torch.profiler.record_function(label):
                    begin.record();result=torch.mv(matrix,vector);end.record()
                owner.events.append((label,begin,end));return result
        self.old.append((native,'torch',native.torch));native.torch=Proxy()
        original_native=self.runner.partition.native
        class Native:
            def __getattr__(self,name):return getattr(original_native,name)
            def __call__(self,*args):
                with owner.span('HC_NATIVE'):return original_native(*args)
        self.old.append((self.runner.partition,'native',original_native));self.runner.partition.native=Native()
        for obj,name,label in ((self.runner.model,'evaluate','CPU_MODEL_EVALUATE'),
                               (self.runner.model,'rhs_flat','CPU_MONITOR_RHS'),
                               (self.runner.partition,'fn','CPU_INNER_NUMBA')):
            old=getattr(obj,name)
            def wrap(*args,_old=old,_label=label,**kwargs):
                with owner.span(_label):return _old(*args,**kwargs)
            self.old.append((obj,name,old));setattr(obj,name,wrap)
    def restore(self):
        for obj,name,value in reversed(self.old):setattr(obj,name,value)
    def summary(self,profile):
        self.torch.cuda.synchronize();sparse={}
        for label,begin,end in self.events:sparse.setdefault(label,[]).append(float(begin.elapsed_time(end)))
        groups={};scalars=[];cpu_events=0
        for event in profile.events():
            if str(event.device_type).split('.')[-1]!='CPU':continue
            cpu_events+=1;parent=event;names=[]
            while parent is not None:
                names.append(parent.name);parent=parent.cpu_parent
            group=next((name for name in names if name.startswith('HC_SPMV_')),None)
            if group is None:
                group='HC_NATIVE_other_ops' if 'HC_NATIVE' in names else 'other_GPU_ops'
            value=float(getattr(event,'self_device_time_total',getattr(event,'self_cuda_time_total',0.)))
            groups[group]=groups.get(group,0.)+value/1000.
            if event.name in ('aten::item','aten::_local_scalar_dense'):
                scalars.append({'name':event.name,'self_cpu_seconds':float(event.self_cpu_time_total)/1e6,
                    'group':'HC_NATIVE' if 'HC_NATIVE' in names else 'joint_or_integrator'})
        return {'span_wall_seconds':self.wall,'span_calls':self.calls,
            'SpMV_device_events':{key:{'calls':len(values),'total_ms':sum(values),'median_ms':float(np.median(values))}
                for key,values in sparse.items()},'GPU_self_operator_ms':groups,
            'scalar_sync_CPU_events':scalars,'profiler_CPU_events':cpu_events,
            'interpretation':'Diagnostic instrumentation adds overhead. Device-event sparse totals and profiler self-device times are distinct measures, not additive. CPU span times overlap; item waits include preceding GPU work.'}


def csr_storage(runner):
    from retina_sim.compiled_kernel import ROW_CSR,CSR
    rhs=runner.model._compiled_rhs;native=runner.compiled.components['outer_native'].native_model
    records=[]
    for instruction in rhs.program:
        code,dest,n,a,an,ip,dp=map(int,instruction)
        if code not in (ROW_CSR,CSR):continue
        count=int(rhs.pointers[ip+n]-rhs.pointers[ip])
        for label,matrix in (('A',native.A),('B',native.B)):
            if n==matrix.shape[0] and an==matrix.shape[1] and count==matrix.nnz:
                weights=rhs.weights[dp:dp+count];indices=rhs.indices[dp:dp+count];pointers=rhs.pointers[ip:ip+n+1]
                if (np.array_equal(weights,matrix.data) and np.array_equal(indices,matrix.indices) and
                        np.array_equal(pointers,matrix.indptr)):
                    records.append({'matrix':label,'weights_offset':dp,'pointer_offset':ip,'nnz':count,
                        'weights_bytes':weights.nbytes,'indices_bytes':indices.nbytes,'pointers_bytes':pointers.nbytes})
    a=[row for row in records if row['matrix']=='A']
    separate=len(a)==2 and a[0]['weights_offset']+a[0]['nnz']<=a[1]['weights_offset']
    return {'actual_plan_CSR_entries':records,'A_is_independent_duplicate_buffers':separate,
        'potential_raw_array_bytes_saved':sum(a[1][key] for key in ('weights_bytes','indices_bytes','pointers_bytes')) if separate else 0,
        'normalization_per_RHS':False,'compiled_plan_total_array_bytes':sum(getattr(rhs,key).nbytes for key in ('work','program','weights','indices','pointers','derivatives'))}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():raise ValueError('Fresh project output required')
    from tools.benchmark_compiler_snapshot import use_v7_compiler
    compiler=use_v7_compiler()
    from retina_sim.background_checkpoint import MemoryGuard
    from tools.benchmark_execution_procedures import available_windows_commit_bytes
    guard=MemoryGuard(14336*1024**2,8192*1024**2);guard.check()
    if guard.last['available_physical_bytes']<15*1024**3 or available_windows_commit_bytes()<7*1024**3:
        raise MemoryError('Original physical/commit admission failed')
    output.mkdir();started=time.perf_counter();setup={}
    def progress(status,**values):
        save(output/'progress.json',dict(status=status,wall_seconds=time.perf_counter()-started,setup=setup,**values))
        print(status,json.dumps(values),flush=True)
    def check():
        if not guard.check():raise MemoryError('Original memory reserve reached')
    from retina_platform.config import RunConfig
    from retina_sim.atlas_population_network import atlas_population_network
    from retina_sim.execution_backends import make_runner,CUDA
    from retina_sim.contracts import SolverConfig,canonical_hash
    from retina_sim.disk_input_packet import load_global_input
    from retina_sim.background import _WindowMonitor
    from retina_sim.recording import RecordingPlan
    from retina_sim.recording_indices import IndexedRecordingPlan
    from tools.disk3mm_response_check import observation_path
    from tools.benchmark_real_bridge import static_input,binding,aux,persist_state
    original=ROOT/'reports/human_disk3mm_floor_transition_v7/transition'
    cfg=RunConfig.from_dict(json.loads((original/'manifest.json').read_bytes())['configuration'])
    construction=replace(cfg,native_background_values=None,frontend_signature='')
    progress('ASSEMBLY_BEGIN');tick=time.perf_counter()
    compiled,_,audit=atlas_population_network(ROOT,duration=.0001,light_context=construction.light_context(),
        owner_bounds_mm=cfg.owner_bounds(),halo_mm=.04,max_spatial_cells=cfg.max_spatial_cells,seed=cfg.seed,
        execution_profile=cfg.execution_profile,configuration_path=cfg.circuit_configuration_path)
    setup['assembly_seconds']=time.perf_counter()-tick;check();progress('ASSEMBLY_COMPLETE')
    tick=time.perf_counter();runner=make_runner(compiled,CUDA)
    solver=SolverConfig(**audit['resolved_execution_profile']['solver']);initial=runner.initialize(solver)
    initial.input_position={'phase':'fresh_factory_HC_short_only'}
    setup['runner_initialize_seconds']=time.perf_counter()-tick;check()
    tick=time.perf_counter();packet=load_global_input(original/'input.npz',compiled,cfg.light_context(),
        reference_id=compiled.global_disk_circuit.record['reference_id'])
    channels=static_input(packet.channels);setup['existing_input_strict_load_seconds']=time.perf_counter()-tick
    _,chosen,bc,_=observation_path(compiled,audit)
    plan=IndexedRecordingPlan(RecordingPlan(states=tuple(name+'.terminal_release' for name in bc),cell_ids=frozenset(chosen),max_bytes=64*1024**2))
    storage=csr_storage(runner);save(output/'csr_storage.json',storage)
    import torch
    times=np.array([0.,.00005,.0001])
    def run(target,*,diagnostic=None):
        monitor=_WindowMonitor(target,initial,cfg.background_policy());points=[];monitor_seconds=[0.]
        def observe(t,y,bundle):
            check();tick=time.perf_counter()
            with torch.profiler.record_function('NUMERICAL_MONITOR'):
                monitor.observe(t,y,bundle)
            monitor_seconds[0]+=time.perf_counter()-tick;points.append(float(t))
        before=dict(target.integration_metrics);torch.cuda.synchronize();tick=time.perf_counter()
        end,result=target.advance(initial if target is runner else target.initialize(solver),channels,.0001,
            sample_times=times,recording_plan=plan,state_observer=observe)
        torch.cuda.synchronize();wall=time.perf_counter()-tick
        report=monitor.report(.0001)
        return end,result,{'wall_seconds':wall,'monitor_wall_seconds':monitor_seconds[0],
            'observer_times':points,**{key:target.integration_metrics[key]-before[key] for key in ('solver_calls','nfev','integration_wall_seconds')}},report
    progress('ORIGINAL_COMPLETE_WINDOW_WARMUP');warm=run(runner);del warm;gc.collect();check()
    progress('SHORT_DIAGNOSIS_BEGIN');instrumentation=Instrumentation(runner);instrumentation.install()
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            diagnostic_end,diagnostic_result,diagnostic_measure,diagnostic_monitor=run(runner)
    finally:instrumentation.restore()
    diagnosis=instrumentation.summary(profile)
    diagnosis.update(window=diagnostic_measure,CSR_storage=storage,compiler=compiler,
        graph={'cones':runner.partition.native.nc,'HC':runner.partition.native.nh,'states':sum(a.size for a in initial.ordered_state_arrays)},
        no_new_background_or_closure_search=True)
    save(output/'diagnosis.json',diagnosis);profile.export_chrome_trace(str(output/'short_diagnostic_trace.json'))
    del profile,instrumentation;gc.collect();check();progress('DIAGNOSIS_READY',diagnosis='diagnosis.json')
    deadline=time.monotonic()+600
    while not (output/'selection.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('No candidate selected; diagnosis retained')
        time.sleep(.2)
    selection=json.loads((output/'selection.json').read_bytes())
    if selection['candidate']!='cpu_finite_joint_validation':raise ValueError('Only one reviewed equivalent candidate may run')
    from retina_sim.gpu_joint_validation import CPUFiniteJointCudaRunner
    candidate=CPUFiniteJointCudaRunner.from_runner(runner)
    science=binding(runner,initial,channels)
    candidate_initial=candidate.initialize(solver);candidate_initial.input_position=dict(initial.input_position)
    if binding(candidate,candidate_initial,channels)!=science:raise AssertionError('Scientific binding changed')
    # Keep phase metadata identical when run() independently initializes the candidate.
    original_initialize=candidate.initialize
    def initialize(*args,**kwargs):
        state=original_initialize(*args,**kwargs);state.input_position=dict(initial.input_position);return state
    candidate.initialize=initialize
    progress('CANDIDATE_COMPLETE_WINDOW_WARMUP');warm=run(candidate);del warm;gc.collect();check()
    measured=[];endpoints=[];results=[];monitors=[]
    # Candidate first reverses the historical GPU v1/compact order. No reuse of
    # those old 0.514/0.583 timings as a fresh baseline or CPU comparison.
    for label,target in (('candidate',candidate),('original',runner)):
        progress('UNINSTRUMENTED_MEASURE',arm=label);check()
        end,result,metrics,monitor=run(target);measured.append(dict(arm=label,**metrics))
        endpoints.append(end);results.append(result);monitors.append(monitor)
        persist_state(output/label,end)
        np.savez(output/(label+'_outputs.npz'),**{key:value.values for key,value in result.outputs_with_units.items()})
    left,right=endpoints
    for a,b in zip(left.ordered_state_arrays,right.ordered_state_arrays):
        np.testing.assert_array_equal(a.view(np.uint64),b.view(np.uint64))
    if canonical_hash(aux(left))!=canonical_hash(aux(right)):raise AssertionError('Causal auxiliary state changed')
    if canonical_hash(monitors[0])!=canonical_hash(monitors[1]):raise AssertionError('Original full numerical monitoring changed')
    for key,value in results[0].outputs_with_units.items():
        other=results[1].outputs_with_units[key]
        assert (value.unit,value.meaning,value.evidence_tag)==(other.unit,other.meaning,other.evidence_tag)
        np.testing.assert_array_equal(value.values.view(np.uint64),other.values.view(np.uint64))
    metadata=[dict(result.metadata) for result in results]
    for item in metadata:item.pop('compute_backend',None)
    if canonical_hash(metadata[0])!=canonical_hash(metadata[1]):raise AssertionError('Non-backend result metadata changed')
    for key in ('observer_times','solver_calls','nfev'):
        if measured[0][key]!=measured[1][key]:raise AssertionError('Numerical execution schedule changed')
    # Check diagnostic wrappers did not alter state, outputs or causal history.
    for a,b in zip(diagnostic_end.ordered_state_arrays,right.ordered_state_arrays):
        np.testing.assert_array_equal(a.view(np.uint64),b.view(np.uint64))
    if canonical_hash(aux(diagnostic_end))!=canonical_hash(aux(right)):raise AssertionError('Instrumentation changed causal state')
    report={'status':'FULL_SHORT_WINDOW_MONITOR_RECORD_AND_ALL_STATE_EXACT_PASS','scope':'Original real HC graph; one warmed100us matched pair from original factory values, not adapted background',
        'measurement':measured,'speedup':measured[1]['wall_seconds']/measured[0]['wall_seconds'],
        'diagnosis':diagnosis,'selection':selection,'setup':setup,'scientific_binding':science,
        'candidate_identity':candidate.model.compute_metadata['integration_backend'],
        'original_model_hash':runner.model.model_hash,'candidate_model_hash':candidate.model.model_hash,
        'memory':guard.last,'peak_RSS_bytes':guard.peak_rss_bytes,'device_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'full_states_causal_aux_units_IDs_outputs_monitor_schedule_exact':True,
        'CPU_vs_GPU_comparison':False,'default_changed':False,'long_background_performed':False,
        'one_pair_only_no_statistical_or_whole_video_extrapolation':True}
    save(output/'result.json',report);progress('COMPLETE',status_result=report['status'],measurement=measured,speedup=report['speedup'])

if __name__=='__main__':main()
