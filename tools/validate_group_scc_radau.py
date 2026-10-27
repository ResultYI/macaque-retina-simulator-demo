"""Bounded full-network Radau or original GPU-DOP853 acceptance; never starts an old stopped driver.

The public command supervises sequential paired GPU operations in fresh child
processes, with a fixed cumulative budget ledger across necessary code repairs.
"""
from pathlib import Path
import argparse
from collections import defaultdict
from copy import deepcopy
import json
import os
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.group_scc_protocol import (AcceptanceFailure, BudgetLedger, FAILURES, GPU_ACTIVITY_POLICY, LIMITS,
    REPORT_PARENT, SCHEMA, CANDIDATES, FALLBACK_GATE, FALLBACK_ORIGIN, candidate_modes, comparison_status, load_protocol, performance_result,
    validate_candidate_phase, verify_fallback_origin,
    read_json, resource_check, scientific_source_preflight, sha256, write_json)
from retina_sim.group_time_storage import load_bundle, save_bundle
from tools.supervise_resident_validation import memory as owned_process_memory
from tools.validate_group_time_candidate import (cpu, checkpoint_hash, clock_events,
    event_snapshot_sides, sample_grid, summary)

QUALIFICATION = dict(default_adopted=False, fit_eligible=False, coupled_background_qualified=False)
HOT_PHASES = ('native_integration', 'inner_integration', 'snapshot_and_bridge', 'events',
    'monitor', 'recording', 'inner_dense_readout', 'consumer_ack')


def require_live_budget_lease(path, token, phase):
    """Re-read the committed lease through short Windows sharing violations."""
    active = read_json(path, permission_retry_seconds=.25)
    if active.get('active') != token or active.get('stopped'):
        raise AcceptanceFailure('RESOURCE_FAIL', 'Budget lease stopped or changed')
    for category in ('cold', phase):
        if active['used'][category] >= LIMITS[category]:
            raise AcceptanceFailure('RESOURCE_FAIL', 'Cumulative '+category+' budget exhausted')


def mandatory_callback_points(arm, advance, nodes, begin, end):
    points = {t for t in nodes if begin < t < end}
    points.update(t for t in advance.accepted_times if begin < t < end)
    for interval in arm.native_intervals:
        for curve in interval.curves.values():
            points.update(segment.t1 for segment in curve.segments if begin < segment.t1 < end)
    return points


def paired_timing_summary(phases, reference_mirrors, candidate='radau'):
    modes=candidate_modes(candidate)
    hot = {m: sum(phases[m].get(k, 0.) for k in HOT_PHASES) for m in modes}
    raw = dict(hot)
    raw['reference'] += sum(reference_mirrors.values())
    candidate_seconds = hot[candidate]
    return dict(hot_seconds=hot, raw_union_hot_seconds=raw,
        reference_mirror_QA_seconds=dict(reference_mirrors),
        raw_union_speedup=None if candidate_seconds <= 0 else raw['reference']/candidate_seconds,
        conservative_speedup=None if candidate_seconds <= 0 else hot['reference']/candidate_seconds,
        exclusive_phase_seconds={m:dict(v) for m,v in phases.items()},
        adoption_timing_basis='Reference original mandatory callbacks divided by candidate full union-required work; candidate-only reference mirrors are independent QA')


def eligible_timing_repeat(prior, hardware):
    return (prior.get('scientific_status')=='PASS' and prior.get('hardware')==hardware
        and prior.get('restore_status')=='RESTORE_PASS' and prior.get('timing_environment_status')=='PASS')


def preserve_unresolved_restore_hold(outcome,protocol):
    if any(receipt.get('evidence_pins') for receipt in protocol.get('historical_batches',[])):
        outcome=dict(outcome,adoptable=False,adoption_hold='PRIOR_UNEXPLAINED_RESTORE_MISMATCH')
        if outcome.get('performance') is not None:
            outcome['performance']=dict(outcome['performance'],adoptable=False,
                adoption_hold='PRIOR_UNEXPLAINED_RESTORE_MISMATCH')
    return outcome


def implementation_identity():
    paths = list((ROOT/'retina_sim').glob('*.py')) + [Path(__file__),
        ROOT/'tools/validate_group_time_candidate.py', ROOT/'tools/cone_hc_foveal_recording.py',
        ROOT/'tools/supervise_resident_validation.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def identity_for(p):
    candidate='gpu_dop853' if p.get('execution_contract') else 'radau'
    preservation = read_json(ROOT/'reports/solver_architecture_research_20261025_v1/PRESERVATION.json')['files']
    pins = {name: value for name,value in preservation.items() if name.endswith('/REQUEST_STOP')}
    pins.update(read_json(ROOT/'reports/spatial_group_graph_v02_20261025_v1/verification.json')['unchanged_pins'])
    if candidate=='gpu_dop853':
        pins.update({FALLBACK_ORIGIN['output']+'/'+name: value for name,value in FALLBACK_ORIGIN['evidence_sha256'].items()})
        for receipt in p.get('historical_batches',[]):
            if receipt.get('evidence_pins'):
                pins.update(receipt['evidence_pins'])
                pins[receipt['ledger_path']]=receipt['ledger_sha256']
    verify_pins(pins)
    return dict(schema=SCHEMA, candidate=candidate, execution_class='A' if candidate=='gpu_dop853' else 'B', protocol=canonical_hash(p), implementation=implementation_identity(),
        group_manifest=sha256(ROOT/p['group_artifact']/'manifest.json'),
        seed_manifest=sha256(ROOT/p['construction_seed']/'manifest.json'), historical_pins=pins)


def verify_pins(pins):
    for name, expected in pins.items():
        if sha256(ROOT/name) != expected:
            raise AcceptanceFailure('INCONCLUSIVE', 'Historical STOP/scientific pin changed: '+name)


def _windows_kernel():
    import ctypes
    from ctypes import wintypes as W
    if os.name != 'nt':
        raise ContractError('Acceptance supervisor requires the configured Windows host')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    signatures = {
        'OpenProcess': ([W.DWORD, W.BOOL, W.DWORD], W.HANDLE),
        'CloseHandle': ([W.HANDLE], W.BOOL),
        'WaitForSingleObject': ([W.HANDLE, W.DWORD], W.DWORD),
        'GetProcessTimes': ([W.HANDLE]+[ctypes.POINTER(W.FILETIME)]*4, W.BOOL),
        'CreateToolhelp32Snapshot': ([W.DWORD,W.DWORD], W.HANDLE),
        'CreateJobObjectW': ([ctypes.c_void_p,W.LPCWSTR], W.HANDLE),
        'SetInformationJobObject': ([W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD], W.BOOL),
        'AssignProcessToJobObject': ([W.HANDLE,W.HANDLE], W.BOOL),
        'IsProcessInJob': ([W.HANDLE,W.HANDLE,ctypes.POINTER(W.BOOL)], W.BOOL),
        'TerminateJobObject': ([W.HANDLE,W.UINT], W.BOOL),
        'TerminateProcess': ([W.HANDLE,W.UINT], W.BOOL),
    }
    for name,(args,result) in signatures.items():
        fn = getattr(kernel,name); fn.argtypes=args; fn.restype=result
    return kernel


def _process_parents():
    import ctypes
    from ctypes import wintypes as W
    class Entry(ctypes.Structure):
        _fields_ = [('dwSize',W.DWORD),('cntUsage',W.DWORD),('pid',W.DWORD),
            ('heap',ctypes.c_size_t),('module',W.DWORD),('threads',W.DWORD),
            ('parent',W.DWORD),('priority',W.LONG),('flags',W.DWORD),('exe',W.WCHAR*260)]
    kernel = _windows_kernel()
    kernel.Process32FirstW.argtypes = kernel.Process32NextW.argtypes = [W.HANDLE,ctypes.POINTER(Entry)]
    kernel.Process32FirstW.restype = kernel.Process32NextW.restype = W.BOOL
    handle = kernel.CreateToolhelp32Snapshot(2,0)
    if handle == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_last_error(),'Cannot query owned process ancestry')
    try:
        entry=Entry(); entry.dwSize=ctypes.sizeof(entry)
        rows={}
        more=kernel.Process32FirstW(handle,ctypes.byref(entry))
        while more:
            rows[int(entry.pid)]=int(entry.parent)
            more=kernel.Process32NextW(handle,ctypes.byref(entry))
        return rows
    finally:
        kernel.CloseHandle(handle)


def _birth_from_handle(handle):
    import ctypes
    from ctypes import wintypes as W
    values=[W.FILETIME() for _ in range(4)]
    if not _windows_kernel().GetProcessTimes(handle,*[ctypes.byref(v) for v in values]):
        raise OSError(ctypes.get_last_error(),'Cannot bind owned process creation time')
    return (int(values[0].dwHighDateTime)<<32)|int(values[0].dwLowDateTime)


def process_identity(pid):
    import ctypes
    kernel=_windows_kernel()
    handle=kernel.OpenProcess(0x00101000,False,pid)
    if not handle:
        raise OSError(ctypes.get_last_error(),'Cannot open owned process identity')
    try:
        return dict(pid=int(pid),creation_ticks=_birth_from_handle(handle))
    finally:
        kernel.CloseHandle(handle)


def process_alive(pid, creation_ticks=None):
    kernel=_windows_kernel()
    handle=kernel.OpenProcess(0x00101000,False,pid)
    if not handle:
        return False
    try:
        return (kernel.WaitForSingleObject(handle,0)==258
            and (creation_ticks is None or _birth_from_handle(handle)==creation_ticks))
    finally:
        kernel.CloseHandle(handle)


def verified_ancestry(pid, supervisor):
    parents=_process_parents()
    chain=[]
    for _ in range(8):
        current=process_identity(pid)
        if chain and current['creation_ticks'] > chain[-1]['creation_ticks']:
            raise ContractError('Owned process ancestry contains a reused parent PID')
        chain.append(current)
        if current == supervisor:
            return chain
        pid=parents.get(pid)
        if not pid:
            break
    raise ContractError('Worker is not descended from this exact supervisor process')


def worker_handshake(output, job, label):
    own=process_identity(os.getpid())
    chain=verified_ancestry(own['pid'],job['supervisor_process'])
    payload=dict(token=job['token'],label=label,worker=own,parent_pid=os.getppid(),ancestry=chain)
    write_json(Path(output)/(label+'_worker_handshake.json'),payload)
    receipt_path=Path(output)/(label+'_worker_ownership.json')
    deadline=time.monotonic()+20.
    while not receipt_path.exists():
        if (not process_alive(job['supervisor_process']['pid'],job['supervisor_process']['creation_ticks'])
                or (Path(output)/'REQUEST_STOP').exists() or time.monotonic()>=deadline):
            raise ContractError('No live supervisor ownership acknowledgement; worker cannot start')
        time.sleep(.02)
    receipt=read_json(receipt_path,permission_retry_seconds=.25)
    if (not process_alive(job['supervisor_process']['pid'],job['supervisor_process']['creation_ticks'])
            or (Path(output)/'REQUEST_STOP').exists() or time.monotonic()>=deadline):
        raise ContractError('No live supervisor ownership acknowledgement; worker cannot start')
    if (receipt.get('token')!=job['token'] or receipt.get('worker')!=own
            or receipt.get('label')!=label or receipt.get('job_kill_on_close') is not True):
        raise ContractError('Worker ownership acknowledgement differs')
    return payload


class WindowsWorkerJob:
    """Own the actual interpreter and descendants before any CUDA/model work."""
    def __init__(self):
        import ctypes
        from ctypes import wintypes as W
        class Basic(ctypes.Structure):
            _fields_=[('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),
                ('flags',W.DWORD),('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),
                ('active',W.DWORD),('affinity',ctypes.c_size_t),('priority',W.DWORD),('scheduling',W.DWORD)]
        class IO(ctypes.Structure):
            _fields_=[(name,ctypes.c_ulonglong) for name in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
        class Extended(ctypes.Structure):
            _fields_=[('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),
                ('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
        self.kernel=_windows_kernel()
        self.handle=self.kernel.CreateJobObjectW(None,None)
        self.launcher=None; self.worker=None; self.worker_handle=None
        if not self.handle:
            raise OSError(ctypes.get_last_error(),'Cannot create owned worker job')
        limit=Extended(); limit.basic.flags=0x2000
        if not self.kernel.SetInformationJobObject(self.handle,9,ctypes.byref(limit),ctypes.sizeof(limit)):
            self.kernel.CloseHandle(self.handle); self.handle=None
            raise OSError(ctypes.get_last_error(),'Cannot enforce kill-on-close worker ownership')

    def _assign(self, handle):
        import ctypes
        from ctypes import wintypes as W
        inside=W.BOOL()
        if not self.kernel.IsProcessInJob(handle,self.handle,ctypes.byref(inside)):
            raise OSError(ctypes.get_last_error(),'Cannot inspect worker job membership')
        if not inside.value and not self.kernel.AssignProcessToJobObject(self.handle,handle):
            raise OSError(ctypes.get_last_error(),'Cannot attach exact owned process to worker job')

    def bind_launcher(self, process):
        self.launcher=process_identity(process.pid)
        self._assign(process._handle)

    def accept(self, payload, *, token, label, supervisor):
        import ctypes
        own=payload.get('worker',{})
        if payload.get('token')!=token or payload.get('label')!=label or not self.launcher:
            raise ContractError('Worker handshake token/segment/launcher differs')
        chain=verified_ancestry(own.get('pid'),supervisor)
        if own!=chain[0] or self.launcher not in chain or payload.get('ancestry')!=chain:
            raise ContractError('Worker handshake does not belong to the exact owned launcher')
        handle=self.kernel.OpenProcess(0x00101511,False,own['pid'])
        if not handle:
            raise OSError(ctypes.get_last_error(),'Cannot open actual worker for supervision')
        try:
            if _birth_from_handle(handle)!=own['creation_ticks']:
                raise ContractError('Actual worker PID was reused')
            self._assign(handle)
        except BaseException:
            self.kernel.CloseHandle(handle)
            raise
        self.worker=own; self.worker_handle=handle
        return dict(token=token,label=label,worker=own,launcher=self.launcher,
            supervisor=supervisor,job_kill_on_close=True,ancestry=chain)

    def resources(self, launcher_handle):
        handle=self.worker_handle if self.worker_handle is not None else launcher_handle
        if self.kernel.WaitForSingleObject(handle,0)!=258:
            return None
        result=owned_process_memory(handle)
        result['measured_process']=self.worker if self.worker is not None else self.launcher
        result['actual_worker_bound']=self.worker is not None
        return result

    def terminate(self):
        if self.handle is not None:
            self.kernel.TerminateJobObject(self.handle,1)

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle); self.handle=None
        if self.worker_handle is not None:
            self.kernel.CloseHandle(self.worker_handle); self.worker_handle=None


def classify_gpu_activity(rows, owned_pids):
    import math
    import re
    active=[]; invalid=[]; valid=0; owned_active=0
    for row in rows:
        match=re.match(r'^pid_(\d+)_',row.get('name',''))
        pid=int(match.group(1)) if match else None
        value=row.get('utilization_percent')
        if (row.get('counter_status') not in (0,1) or isinstance(value,bool)
                or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0):
            if pid not in owned_pids:
                invalid.append(dict(name=row.get('name'),counter_status=row.get('counter_status')))
            continue
        valid+=1
        if value>0:
            if pid in owned_pids:
                owned_active+=1
            else:
                engine=row['name'].split('_engtype_',1)[-1].lower()
                kind=('compute' if engine.startswith(('compute','cuda')) else 'graphics' if
                    engine.startswith(('3d','copy','video','overlay','graphics')) else 'unknown')
                active.append(dict(pid=pid,name=row['name'],engine_type=engine,activity_kind=kind,utilization_percent=value))
    compute=[r for r in active if r['activity_kind']=='compute']
    graphics=[r for r in active if r['activity_kind']=='graphics']
    unknown=[r for r in active if r['activity_kind']=='unknown']
    status='CONTAMINATED' if compute else 'INCONCLUSIVE' if invalid or unknown or not valid else 'PASS'
    return dict(status=status,engine_instances=len(rows),valid_engine_instances=valid,
        active_external_engines=active,invalid_external_engines=invalid,owned_active_engines=owned_active,
        external_compute_percent=sum(r['utilization_percent'] for r in compute),
        external_graphics_aggregate_percent=sum(r['utilization_percent'] for r in graphics),
        external_graphics_peak_engine_percent=max((r['utilization_percent'] for r in graphics),default=0.0),
        compute_activity_threshold_percent=GPU_ACTIVITY_POLICY['external_compute_limit_percent'])


def gpu_activity_summary(samples):
    measured=[s for s in samples if s.get('label') in ('before','during','after')]
    production=[s for s in measured if s.get('label') in ('during','after')]
    duration=sum(s.get('interval_seconds',0.) for s in production)
    mean=(sum(s.get('external_graphics_aggregate_percent',0.)*s.get('interval_seconds',0.) for s in production)/duration if duration>0 else None)
    peak=max((s.get('external_graphics_aggregate_percent',0.) for s in production),default=None)
    status='PASS'
    if any(s['status']=='CONTAMINATED' for s in measured):
        status='CONTAMINATED'
    elif ({s.get('label') for s in measured} != {'before','during','after'}
            or any(s['status']!='PASS' for s in measured) or mean is None):
        status='INCONCLUSIVE'
    elif (mean>GPU_ACTIVITY_POLICY['quiet_graphics_time_weighted_aggregate_percent']
            or peak>GPU_ACTIVITY_POLICY['graphics_peak_aggregate_percent']):
        status='GRAPHICS_RECHECK_REQUIRED'
    return dict(status=status,graphics_time_weighted_aggregate_percent=mean,
        graphics_peak_aggregate_percent=peak,policy=GPU_ACTIVITY_POLICY,
        note='Idle contexts do not contaminate timings. Routine graphics is measured and screened before net-speed adoption; close gains still require paired timing repeats.')


def gpu_activity_status(samples):
    return gpu_activity_summary(samples)['status']


class GPUActivityWitness:
    """Persistent, language-neutral Windows engine-utilization interval counter."""
    counter_path=r'\GPU Engine(*)\Utilization Percentage'
    def __init__(self, owned_pids):
        import ctypes
        from ctypes import wintypes as W
        self.owned_pids=set(owned_pids); self.query=W.HANDLE(); self.counter=W.HANDLE()
        self.error=None; self.last_unix=time.time(); self.pdh=None
        class Number(ctypes.Union):
            _fields_=[('longValue',W.LONG),('doubleValue',ctypes.c_double),('largeValue',ctypes.c_longlong),('stringValue',W.LPCWSTR)]
        class Value(ctypes.Structure):
            _fields_=[('status',W.DWORD),('number',Number)]
        class Item(ctypes.Structure):
            _fields_=[('name',W.LPCWSTR),('value',Value)]
        self.Item=Item
        try:
            if os.name!='nt':
                raise OSError('Windows GPU Engine counters required')
            self.pdh=ctypes.WinDLL('pdh',use_last_error=True)
            signatures={
                'PdhOpenQueryW':([W.LPCWSTR,ctypes.c_size_t,ctypes.POINTER(W.HANDLE)],W.LONG),
                'PdhAddEnglishCounterW':([W.HANDLE,W.LPCWSTR,ctypes.c_size_t,ctypes.POINTER(W.HANDLE)],W.LONG),
                'PdhCollectQueryData':([W.HANDLE],W.LONG),
                'PdhGetFormattedCounterArrayW':([W.HANDLE,W.DWORD,ctypes.POINTER(W.DWORD),ctypes.POINTER(W.DWORD),ctypes.c_void_p],W.LONG),
                'PdhCloseQuery':([W.HANDLE],W.LONG)}
            for name,(args,result) in signatures.items():
                fn=getattr(self.pdh,name);fn.argtypes=args;fn.restype=result
            for name,args in (
                ('PdhOpenQueryW',(None,0,ctypes.byref(self.query))),
                ('PdhAddEnglishCounterW',(self.query,self.counter_path,0,ctypes.byref(self.counter))),
                ('PdhCollectQueryData',(self.query,))):
                status=getattr(self.pdh,name)(*args)
                if status:
                    raise OSError(name+' failed: '+hex(status&0xffffffff))
            self.last_unix=time.time()
        except (OSError,AttributeError) as error:
            self.error=str(error)
            self.close()

    def sample(self,label):
        import ctypes
        from ctypes import wintypes as W
        start=self.last_unix; now=time.time()
        result=dict(source='Windows PDH GPU Engine Utilization Percentage',counter_path=self.counter_path,
            label=label,sample_start_unix=start,sample_stop_unix=now,interval_seconds=max(0.,now-start),
            owned_pids=sorted(self.owned_pids),scope='All Windows GPU adapters/engines; idle process presence is irrelevant',
            coverage='Interval counters sampled at boundaries and during execution; not an exhaustive trace of short-lived instances')
        if self.error:
            return dict(result,status='INCONCLUSIVE',reason=self.error)
        try:
            status=self.pdh.PdhCollectQueryData(self.query)
            if status:
                raise OSError('PdhCollectQueryData failed: '+hex(status&0xffffffff))
            self.last_unix=now
            for _ in range(3):
                size=W.DWORD();count=W.DWORD()
                status=self.pdh.PdhGetFormattedCounterArrayW(self.counter,0x200,ctypes.byref(size),ctypes.byref(count),None)
                if status&0xffffffff!=0x800007d2 or size.value==0:
                    raise OSError('GPU counter sizing failed: '+hex(status&0xffffffff))
                buffer=ctypes.create_string_buffer(size.value)
                status=self.pdh.PdhGetFormattedCounterArrayW(self.counter,0x200,ctypes.byref(size),ctypes.byref(count),buffer)
                if status&0xffffffff==0x800007d2:
                    continue
                if status:
                    raise OSError('GPU counter formatting failed: '+hex(status&0xffffffff))
                values=ctypes.cast(buffer,ctypes.POINTER(self.Item))
                rows=[dict(name=values[i].name,utilization_percent=values[i].value.number.doubleValue,
                    counter_status=int(values[i].value.status)) for i in range(count.value)]
                return dict(result,**classify_gpu_activity(rows,self.owned_pids))
            raise OSError('GPU counter instances changed during every bounded read')
        except (OSError,ValueError) as error:
            return dict(result,status='INCONCLUSIVE',reason=str(error))

    def close(self):
        if self.pdh is not None and self.query:
            self.pdh.PdhCloseQuery(self.query)
            self.query=type(self.query)()


def warm_exact_initial_rhs(arm):
    """Compile the production signatures without advancing or changing sampler state."""
    warm_started=time.perf_counter()
    before=checkpoint_hash(arm.checkpoint_payload())
    counters_before=deepcopy(getattr(arm.compiled,'execution_profile',{}))
    packed=arm.inner.pack(arm.inner_state)
    rhs_started=time.perf_counter()
    if arm.gpu_rhs is None:
        value=arm.compiled.rhs_flat(arm.start,packed)
        if value.dtype!=np.float64 or not np.isfinite(value).all():
            raise AcceptanceFailure('DOMAIN_FAIL','Cold original CPU RHS warmup is nonfinite or not FP64')
        backend='original_compiled_CPU_RHS'
    else:
        arm._snapshot_gpu_state.copy_(arm.torch.from_numpy(packed))
        arm.gpu_rhs.evaluate_into(arm.start,arm._snapshot_gpu_state,arm._snapshot_gpu_rhs)
        if arm._snapshot_gpu_rhs.dtype!=arm.torch.float64 or not bool(arm.torch.isfinite(arm._snapshot_gpu_rhs).all().item()):
            raise AcceptanceFailure('DOMAIN_FAIL','Cold original GPU RHS warmup is nonfinite or not FP64')
        backend='original_GPU_RHS_existing_buffers'
    rhs_seconds=time.perf_counter()-rhs_started
    after=checkpoint_hash(arm.checkpoint_payload())
    if before!=after:
        raise AcceptanceFailure('INCONCLUSIVE','Cold RHS warmup changed state/event/RNG/controller ownership')
    return dict(backend=backend,wall_seconds=time.perf_counter()-warm_started,rhs_call_wall_seconds=rhs_seconds,
        complete_payload_before=before,complete_payload_after=after,
        state_event_rng_unchanged=True,production_output_reused=False,
        compiled_profile_before=counters_before,compiled_profile_after=deepcopy(getattr(arm.compiled,'execution_profile',{})))


def hardware_identity(torch):
    import scipy
    import numba
    props = torch.cuda.get_device_properties(0)
    return dict(host=platform.node(), platform=platform.platform(), cpu=platform.processor(),
        python=sys.version, numpy=np.__version__, torch=torch.__version__,
        scipy_version=scipy.__version__,numba_version=numba.__version__,
        cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        device_uuid=str(getattr(props, 'uuid', 'unavailable')), total_vram_bytes=props.total_memory,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def recording_content_hash(path):
    """Compare numeric recording content, not ZIP creation timestamps."""
    files = {}
    for folder in ('windows', 'observations', 'events'):
        for source in sorted((Path(path)/folder).glob('*')):
            if source.suffix == '.npz':
                with np.load(source, allow_pickle=False) as z:
                    value = {name: z[name] for name in sorted(z.files)}
                files[source.relative_to(path).as_posix()] = canonical_hash(value)
            elif source.suffix == '.json':
                files[source.relative_to(path).as_posix()] = canonical_hash(read_json(source))
            else:
                raise ContractError('Unknown ordinary recording content '+source.name)
    return canonical_hash(files)


def _failure_status(exc, stage):
    if isinstance(exc, AcceptanceFailure):
        return exc.status
    message = str(exc).lower()
    if any(term in message for term in ('out of memory', 'memory guard', 'wall budget', 'request_stop')):
        return 'RESOURCE_FAIL'
    if stage.startswith('RESTORE'):
        return 'RESTORE_FAIL'
    if any(term in message for term in ('domain', 'nonfinite', 'finite original')):
        return 'DOMAIN_FAIL'
    if any(term in message for term in ('pcg64', 'event chronology', 'strict-past', 'sampler')):
        return 'EVENT_FAIL'
    if isinstance(exc, ContractError) and any(term in message for term in ('converge', 'residual', 'numerical', 'error budget')):
        return 'NUMERICAL_FAIL'
    return 'INCONCLUSIVE'


def arm_execution_options(candidate, mode, torch, guard, shared_plans):
    if mode not in candidate_modes(candidate):
        raise ContractError('Arm mode differs from the selected paired candidate')
    return dict(mode=mode,torch=torch,device='cuda:0',guard=guard,
        shared_plans=None if mode=='reference' else shared_plans)


def checkpoint_identity(identity, mode):
    if mode not in candidate_modes(identity.get('candidate')):
        raise ContractError('Checkpoint arm differs from the selected candidate')
    return dict(identity,mode=mode)


def bind_worker_selection(job, candidate, phase):
    validate_candidate_phase(candidate,phase)
    if job.get('candidate')!=candidate or job.get('phase')!=phase or job.get('identity',{}).get('candidate')!=candidate:
        raise ContractError('Worker candidate/phase does not match its immutable job identity')


def enforce_paired_entry_gate(candidate, token, identity, protocol, started, now, receipt=None):
    if candidate!='gpu_dop853' or started is None:
        return False
    if receipt is None:
        if now-started>=FALLBACK_GATE['wall_limit_seconds']:
            raise AcceptanceFailure('RESOURCE_FAIL','Integral paired first100us wall limit reached')
        return False
    expected=dict(status='PASS',run_id=token,candidate=candidate,identity=canonical_hash(identity),
        hot_started_unix=started,time_seconds=protocol['start_seconds']+FALLBACK_GATE['scope_seconds'],
        numerical_status='PASS',domain_status='PASS',event_status='PASS',monitor_status='PASS',dense_status='PASS',
        recording_and_acknowledgement_complete=True,complete_coverage=True,charge_category='paired')
    if not isinstance(receipt,dict) or any(receipt.get(k)!=v for k,v in expected.items()):
        raise ContractError('Integral paired first100us receipt identity or complete scientific gates differ')
    completed=receipt.get('completed_unix')
    if isinstance(completed,bool) or not isinstance(completed,(int,float)) or not np.isfinite(completed) or not started<=completed<=now:
        raise ContractError('Invalid integral paired first100us completion clock')
    if completed-started>=FALLBACK_GATE['wall_limit_seconds']:
        raise AcceptanceFailure('RESOURCE_FAIL','Integral paired first100us wall limit reached')
    return True


def run_worker(output, *, restore=False, token=None, candidate='radau', phase=None):
    """Internal entry: a live supervisor lease is mandatory; no manual resume."""
    from retina_sim.background_checkpoint import MemoryGuard
    from retina_sim.spatial_group_model import GroupModelAssembly
    from retina_sim.group_time_runner import GroupTimeArm
    from retina_sim.group_time_monitor import GroupTimeMonitor, GroupTimeComparator
    from retina_sim.group_device_observer import GroupDeviceObserver
    from retina_sim.group_source_observer import GroupSourceObserver
    from retina_sim.group_scc_dense_qa import compare_dense_interval
    from retina_sim.group_dense_certificate import prepare_dense_certificate
    output = Path(output).resolve()
    if output.parent != ROOT/REPORT_PARENT:
        raise ContractError('Worker output outside the fixed batch')
    job = read_json(output/'job.json')
    bind_worker_selection(job,candidate,phase)
    modes=candidate_modes(candidate)
    ledger_path = output.parent/'budget_ledger.json'
    ledger = read_json(ledger_path, permission_retry_seconds=.25)
    if token != job['token'] or ledger.get('active') != token or not (output.parent/'budget_ledger.lock').is_file():
        raise ContractError('A live matching parent budget lease is required')
    if not process_alive(job['supervisor_process']['pid'],job['supervisor_process']['creation_ticks']):
        raise ContractError('Orphaned worker cannot continue')
    worker_handshake(output,job,'restored' if restore else 'uninterrupted')
    if any((output/name).exists() for name in ('REQUEST_STOP', 'failure.json', 'result.json')):
        raise ContractError('Stopped, failed or completed attempt cannot resume')
    p = load_protocol(ROOT,candidate)
    identity = identity_for(p)
    restore_import=job.get('restore_source')
    if restore_import is not None:
        from retina_sim.restore_import import inspect_source,bind_identity
        if not restore or candidate!='gpu_dop853' or job['phase']!='paired':
            raise ContractError('Imported checkpoint requires an owned full paired restore worker')
        if inspect_source(ROOT,restore_import['source'],p,identity,REPORT_PARENT)!=restore_import:
            raise ContractError('Imported checkpoint lineage changed')
        identity=bind_identity(identity,restore_import)
    source_output=output if restore_import is None else Path(restore_import['source'])
    source_identity=identity if restore_import is None else restore_import['source_identity']
    if identity != job['identity'] or p != read_json(output/'protocol.json'):
        raise ContractError('Prospective source/protocol identity changed')
    source_preflight=scientific_source_preflight(ROOT)
    if source_preflight['status']!='PASS':
        write_json(output/'failure.json',dict(source_preflight,budget_charged=True))
        write_json(output/'REQUEST_STOP',dict(status='INCONCLUSIVE',run_id=token,
            message='Frozen scientific dependency preflight failed before CUDA/model preparation'))
        raise AcceptanceFailure('INCONCLUSIVE','Frozen scientific dependency preflight failed before CUDA/model preparation')
    phase = job['phase']
    if candidate=='gpu_dop853':
        verify_fallback_origin(ROOT,p,ledger)
    if phase == 'feasibility':
        p = dict(p, stop_seconds=p['start_seconds']+.0001)
    stage = 'COLD_PREPARATION'
    started = None
    public = p['start_seconds']
    last_validated = None
    arms, monitors, recorders = {}, {}, {}
    phases = {m: defaultdict(float) for m in modes}
    auxiliary = defaultdict(float)
    reference_mirrors = defaultdict(float)
    reference_mirror = False
    reference_mirror_callbacks = 0
    comparisons, dense_reports = [], []
    prefix = None
    next_event = 0
    recorded = set()
    oracle = []
    oracle_index = 0
    checkpoint_receipts = {}
    peak_gpu = peak_rss = 0
    guard_tick = -1.0
    ownership_tick = -1.0
    gpu_activity_samples = []
    activity = None
    cold_warmup = {}
    gate_receipt = None
    memory = MemoryGuard(16*1024**3, 4*1024**3)
    torch = None
    process_started = time.perf_counter()
    environment = None
    last_evidence = {}

    def publish():
        write_json(output/'status.json', dict(stage=stage, worker_pid=os.getpid(), restore=restore,
            public_time_seconds=public, last_validated_time_seconds=last_validated,
            hot_started_unix=started, resources=memory.last, whole_GPU_sampled_peak_bytes=peak_gpu,
            process_peak_rss_bytes=peak_rss))

    def guard(force=False):
        if restore_import is None:
            enforce_paired_entry_gate(candidate, token, identity, p, started, time.time(), gate_receipt)
        else:
            enforce_paired_entry_gate(candidate,restore_import['source_token'],source_identity,p,
                restore_import['source_hot_started_unix'],time.time(),restore_import['entry_gate'])
        nonlocal guard_tick, peak_gpu, peak_rss, ownership_tick
        if (output/'REQUEST_STOP').exists() or (output.parent/'REQUEST_STOP').exists():
            raise AcceptanceFailure('RESOURCE_FAIL', 'REQUEST_STOP is permanent for this acceptance run')
        if not process_alive(job['supervisor_process']['pid'],job['supervisor_process']['creation_ticks']):
            raise AcceptanceFailure('INCONCLUSIVE', 'Supervisor exited; orphaned worker stopped')
        now = time.perf_counter()
        if not force and now-guard_tick < .5:
            return
        require_live_budget_lease(ledger_path, token, phase)
        if not memory.check():
            raise AcceptanceFailure('RESOURCE_FAIL', 'CPU memory guard: '+str(memory.stop_reason))
        peak_rss = max(peak_rss, memory.last.get('process_peak_rss_bytes', 0))
        if torch is not None:
            free, total = torch.cuda.mem_get_info(0)
            used = total-free
            resource_check(rss=0, available=4*1024**3, gpu_used=used, gpu_total=total)
            peak_gpu = max(peak_gpu, used)
            if activity is not None and now-ownership_tick >= 1.:
                sample_activity('during' if started is not None else 'preparation')
                ownership_tick = now
        verify_pins(identity['historical_pins'])
        publish(); guard_tick = now

    def sample_activity(label):
        query_started=time.perf_counter()
        try:
            gpu_activity_samples.append(activity.sample(label))
        finally:
            auxiliary['GPU_activity_query'] += time.perf_counter()-query_started

    def timed(mode, name, fn):
        guard()
        if torch is not None:
            torch.cuda.synchronize(0)
        began = time.perf_counter()
        ownership_before = auxiliary['GPU_activity_query']
        try:
            return fn()
        finally:
            if torch is not None:
                torch.cuda.synchronize(0)
            elapsed = time.perf_counter()-began-(auxiliary['GPU_activity_query']-ownership_before)
            if mode == 'reference' and reference_mirror and name in HOT_PHASES:
                reference_mirrors[name] += elapsed
            else:
                (auxiliary if mode is None else phases[mode])[name] += elapsed
            guard()

    def metrics():
        return dict(candidate=candidate, execution_class=identity['execution_class'], paired_entry_gate=gate_receipt,
            fallback_origin=p.get('fallback_origin'), **paired_timing_summary(phases, reference_mirrors, candidate),
            reference_mirror_callbacks_this_process=reference_mirror_callbacks,
            independent_QA_and_IO_seconds=dict(auxiliary), seven_ms_prefix=prefix,
            neural_metrics={m:a.metrics for m,a in arms.items()},
            compiled_execution_profile={m:getattr(a.compiled, 'execution_profile', {}) for m,a in arms.items()},
            compiled_profile_note='Nested cumulative counters include the separately recorded cold warmup; exclusive hot timings determine performance',
            transport_metrics={m:{} if a.transport is None else a.transport.metrics for m,a in arms.items()},
            monitor_host_transfer_metrics={m:getattr(mon, 'host_transfer_metrics', {}) for m,mon in monitors.items()},
            monitor_execution_profile={m:getattr(mon, 'execution_profile', {}) for m,mon in monitors.items()},
            resource_peak=dict(process_rss_bytes=peak_rss, whole_GPU_bytes=peak_gpu), resources=memory.last,
            process_wall_seconds=time.perf_counter()-process_started,
            device_transfer_seconds='Nested in exclusive phases; transport and monitor ingress counters are not added again',
            timing_environment_status=gpu_activity_status(gpu_activity_samples),
            gpu_activity_samples=gpu_activity_samples, gpu_activity_summary=gpu_activity_summary(gpu_activity_samples), cold_warmup=cold_warmup,
            timing_note='Synchronized sequential arm calls. Reference candidate-only mirror callbacks are QA and excluded from its conservative hot denominator; candidate retains all union callbacks. Raw union timings are also reported. No historical cross-machine timing enters either ratio. QA/checkpoint and all mirrors charge the same cumulative wall budget.')

    try:
        guard(True)
        activity = timed(None,'GPU_activity_initialization',lambda:GPUActivityWitness({os.getpid()}))
        def cuda_setup():
            nonlocal torch
            import torch as module
            torch = module
            if not torch.cuda.is_available():
                raise AcceptanceFailure('INCONCLUSIVE', 'Real CUDA unavailable; CPU cannot substitute for this qualification')
            torch.cuda.set_device(0); torch.cuda.init()
        timed(None, 'cuda_initialization', cuda_setup)
        environment = hardware_identity(torch)
        if restore:
            if environment != read_json(source_output/'environment.json')['hardware']:
                raise AcceptanceFailure('RESTORE_FAIL', 'Hardware/software identity changed across processes')
            session = read_json(source_output/'session_checkpoint.json')
            witness = session.pop('session_hash')
            if canonical_hash(session) != witness or session['identity'] != source_identity or session['writer_pid'] == os.getpid():
                raise AcceptanceFailure('RESTORE_FAIL', 'Fresh-process session identity invalid')
            if session.get('candidate')!=candidate:
                raise AcceptanceFailure('RESTORE_FAIL','Checkpoint candidate selection differs')
            started = session['hot_started_unix'] if restore_import is None else job['restore_attempt_started_unix']
            if candidate=='gpu_dop853':
                gate_receipt=read_json(source_output/'paired_entry_gate.json')
                if restore_import is None:
                    enforce_paired_entry_gate(candidate,token,identity,p,started,time.time(),gate_receipt)
                else:
                    enforce_paired_entry_gate(candidate,restore_import['source_token'],source_identity,p,
                        restore_import['source_hot_started_unix'],time.time(),gate_receipt)
            public = session['public_time_seconds']; next_event = session['next_event']
            last_validated = session['last_validated_time_seconds']; recorded = set(session['recorded_times'])
            oracle = read_json(source_output/'uninterrupted_oracle.json')['callbacks']
            stage = 'RESTORE_LOAD'
        cold_warmup['dense_QA']=timed(None,'cold_dense_QA_warmup',prepare_dense_certificate)
        seed_bundle = timed(None, 'seed_load', lambda: load_bundle(ROOT/p['construction_seed'], expected_kind='source_construction_seed'))
        seed = seed_bundle['payload']
        assembly = timed(None, 'group_assembly', lambda: GroupModelAssembly(ROOT/p['group_artifact']))
        from retina_sim.group_time_recording import GroupRecording
        times = sample_grid(p['start_seconds'], p['stop_seconds'], .0005)
        if phase == 'feasibility' or p.get('diagnostic_scope_only'):
            times = (p['start_seconds'], p['stop_seconds'])
        records = set(times)
        edges = {i/150. for i in range(748,752) if p['start_seconds'] < i/150. <= p['stop_seconds']}
        nodes = set(sample_grid(p['start_seconds'], p['stop_seconds'], .0001)) | records | edges | {p['start_seconds'], p['stop_seconds']}
        if not restore:
            (output/'uninterrupted').mkdir()
        for mode in modes:
            observer = (GroupTimeMonitor if mode == 'reference' else
                        GroupSourceObserver if mode == 'gpu_dop853' else GroupDeviceObserver)
            observer_kwargs = dict(parameters=seed['inner_parameters'], torch=torch)
            if mode == candidate:
                observer_kwargs.update(device='cuda:0', allow_host_inputs=True)
            common = arm_execution_options(candidate,mode,torch,guard,
                None if mode=='reference' else arms['reference'].shared_plans)
            if restore:
                bundle = timed(None, 'checkpoint_load', lambda m=mode: load_bundle(source_output/(m+'_checkpoint'),
                    expected_kind='complete_model_checkpoint', expected_identity=checkpoint_identity(source_identity,m)))
                payload = bundle['payload']
                before = timed(None, 'checkpoint_before_restore_hash', lambda: checkpoint_hash(payload))
                if before != session['checkpoint_payload_hashes'][mode]:
                    raise AcceptanceFailure('RESTORE_FAIL', 'Checkpoint original payload hash differs')
                arms[mode] = timed(None, 'arm_restore', lambda: GroupTimeArm.from_checkpoint(assembly, seed, p, payload, **common))
                monitors[mode] = timed(None, 'monitor_restore', lambda: observer.from_payload(assembly, seed['states'], payload['monitor'], **observer_kwargs))
                recorders[mode] = timed(None, 'recording_restore', lambda m=mode: GroupRecording.restore(assembly, output/'restored'/m, payload['recording'], artifactidentity=source_identity))
                rebuilt = timed(None, 'checkpoint_rebuild', lambda m=mode: arms[m].checkpoint_payload(monitor=monitors[m].payload(), recording=recorders[m].checkpoint_payload()))
                after = timed(None, 'checkpoint_rebuilt_hash', lambda: checkpoint_hash(rebuilt))
                if before != after:
                    raise AcceptanceFailure('RESTORE_FAIL', 'Complete ten-field checkpoint payload differs after restore')
                checkpoint_receipts[mode] = dict(original_hash=before, restored_hash=after)
                del payload, rebuilt, bundle
            else:
                arms[mode] = timed(None, 'arm_build', lambda: GroupTimeArm(assembly, seed, p, **common))
                monitors[mode] = timed(None, 'monitor_build', lambda: observer(assembly, seed['states'], **observer_kwargs))
                recorders[mode] = timed(None, 'recording_build', lambda m=mode: GroupRecording(assembly, output/'uninterrupted'/m, np.asarray(times), artifactidentity=identity))
            arms[mode].compiled.profile_enabled = True
        if not restore:
            for mode,arm in arms.items():
                cold_warmup[mode]=timed(None,'cold_RHS_warmup_'+mode,lambda a=arm:warm_exact_initial_rhs(a))
            sample_activity('before')
            write_json(output/'cold_warmup.json',cold_warmup)
            write_json(output/'environment.json', dict(hardware=environment, writer_pid=os.getpid(),
                source_initial_qualification=seed_bundle['qualification'], group_counts={k:v.group_count for k,v in assembly.populations.items()},
                cold_preparation_seconds=time.perf_counter()-process_started))
            started = time.time()
        else:
            sample_activity('before')
            write_json(output/'restore_receipt.json', dict(status='COMPLETE_PAYLOAD_BITWISE_RESTORED',
                writer_pid=session['writer_pid'], restore_pid=os.getpid(), fresh_process=True,
                complete_payload_hashes=checkpoint_receipts, suffix_validation='PENDING'))
        guard(True)
        sample_activity('during')
        initial_ports = None
        comparator = None
        if restore:
            initial_ports = timed(None, 'initial_ports_load', lambda: load_bundle(source_output/'initial_ports', expected_kind='validation_evidence', expected_identity=source_identity)['payload'])
            comparator = GroupTimeComparator(monitors['reference'], initial_ports['values'], initial_ports['units'], parameters=seed['inner_parameters'])

        def observe(t, side, states=None, *, do_record=False):
            nonlocal public, last_validated, last_evidence, initial_ports, comparator, oracle_index
            public = float(t)
            evidence = {}
            for mode, arm in arms.items():
                kwargs = dict(queue_node=t in nodes)
                if states is not None:
                    kwargs['inner_state'] = states[mode]
                evidence[mode] = timed(mode, 'snapshot_and_bridge', lambda a=arm, k=kwargs: a.snapshot(t, side, **k))
                e = evidence[mode]
                timed(mode, 'monitor', lambda m=mode, e=e: monitors[m].observe(t, side, e['states'], e['rhs'],
                    transitions=e['transitions'], ports=e['ports'], event_generation=e['event_generation'], aux=e['aux'], rhs_kind=e['rhs_kind']))
                if monitors[mode].domain_failures:
                    raise AcceptanceFailure('DOMAIN_FAIL', 'Complete original domain/history/gap monitor failed: '+str(monitors[mode].domain_failures[:3]))
                if monitors[mode].missing:
                    raise AcceptanceFailure('INCONCLUSIVE', 'Complete unit coverage missing')
            last_evidence = evidence
            if comparator is None:
                initial_ports = dict(values={k:cpu(v) for k,v in evidence['reference']['ports'].items()}, units=evidence['reference']['port_units'])
                timed(None, 'initial_ports_save', lambda: save_bundle(output/'initial_ports', kind='validation_evidence', identity=identity, payload=initial_ports, qualification=QUALIFICATION))
                comparator = GroupTimeComparator(monitors['reference'], initial_ports['values'], initial_ports['units'], parameters=seed['inner_parameters'])
            compared = timed(None, 'full_numerical_equivalence', lambda: comparator.compare_pair(evidence['reference'], evidence[candidate]))
            comparisons.append(summary(compared))
            if p.get('diagnostic_scope_only'):
                for mode,e in evidence.items():
                    cached = timed(None,'canonical_cached_compatibility',lambda e=e:checkpoint_hash(e))
                    original = timed(None,'canonical_original_compatibility',lambda e=e:checkpoint_hash(e,reuse_arrays=False))
                    if cached != original:
                        raise AcceptanceFailure('RESTORE_FAIL','Cached canonical digest differs from original at '+str(t)+' '+mode)
            if compared['status'] != 'PASS':
                timed(None, 'failure_numeric_witness', lambda: save_bundle(output/'first_failed_numeric_witness', kind='validation_evidence', identity=identity,
                    payload=dict(comparison=compared, **evidence), qualification=QUALIFICATION))
                raise AcceptanceFailure(comparison_status(compared), 'Original scientific gate: '+str(compared['issues'][:3]))
            if phase == 'paired' and t > p['checkpoint_seconds']:
                row = dict(time_hex=float(t).hex(), side=side, hashes={m:timed(None, 'restore_oracle_hash', lambda e=e: checkpoint_hash(e)) for m,e in evidence.items()})
                if restore:
                    if oracle_index >= len(oracle) or row != oracle[oracle_index]:
                        from retina_sim.restore_oracle_diagnostics import capture_mismatch
                        try:
                            expected_evidence=None
                            if oracle_index==15:
                                try:
                                    expected_evidence=load_bundle(ROOT/'reports/overnight_20261026_v1/combined_restore_probe_r1/witness',
                                        expected_kind='validation_evidence')['payload']
                                except Exception:
                                    pass  # An unavailable baseline must not prevent capture of actual evidence.
                            timed(None,'restore_mismatch_capture',lambda:capture_mismatch(output/'first_restore_hash_mismatch',
                                identity,oracle_index,None if oracle_index>=len(oracle) else oracle[oracle_index],row,
                                evidence,monitors=monitors,arms=arms,expected_evidence=expected_evidence))
                        except Exception as capture_error:
                            try:
                                write_json(output/'restore_mismatch_capture_error.json',dict(error=repr(capture_error),
                                    index=oracle_index,actual_row=row,expected_row=None if oracle_index>=len(oracle) else oracle[oracle_index]))
                            except Exception:
                                pass  # Failure evidence is best effort; preserve the original mismatch.
                        raise AcceptanceFailure('INCONCLUSIVE', 'RESTORE_INCONCLUSIVE: exact suffix hash differs; numeric witness comparison required at callback '+str(oracle_index))
                    oracle_index += 1
                else:
                    oracle.append(row)
            last_validated = float(t)
            if do_record and t in records and t not in recorded:
                for mode, arm in arms.items():
                    e = evidence[mode]
                    def record(m=mode, a=arm, e=e):
                        native = a.record_native_values(t, side, snapshot=e)
                        recorders[m].record(t, native, e['inner_outputs'], delayedHC=cpu(e['delayed_HC']), eventlogs=a.event_log)
                        a.event_log = {pop:[] for pop in a.event_log}
                    timed(mode, 'recording', record)
                recorded.add(t)

        if not restore:
            stage = 'INITIAL_OBSERVATION_AND_EVENTS'
            observe(public, 'right')
            for mode, arm in arms.items():
                timed(mode, 'events', arm.initial_commit)
            observe(public, 'right', do_record=True)
            for mode, arm in arms.items():
                timed(mode, 'consumer_ack', lambda a=arm: a.ack(public))
        stage = 'RESTORED_SUFFIX' if restore else 'FULL_PAIRED_ADVANCE'
        events = clock_events(p)
        for event_index in range(next_event, len(events)):
            end, rod, bc = events[event_index]
            begin = arms['reference'].inner_time
            if not begin < end or any(a.inner_time != begin for a in arms.values()):
                raise AcceptanceFailure('EVENT_FAIL', 'Original event chronology differs')
            advances = {}
            for mode, arm in arms.items():
                timed(mode, 'native_integration', lambda a=arm: a.ensure_native(end, 'left'))
                advances[mode] = timed(mode, 'inner_integration', lambda a=arm: a.advance_inner(begin, end))
            certificate = timed(None, 'full_interval_dense_QA', lambda: compare_dense_interval(
                arms['reference'], arms[candidate], advances['reference'], advances[candidate],
                begin, end, monitors['reference'].rows))
            dense_reports.append(certificate)
            timed(None,'dense_progress_save',lambda:write_json(output/'dense_progress.json',
                dict(candidate=candidate,run_id=token,intervals_completed=len(dense_reports),dense_certificates=dense_reports)))
            if certificate['status'] in ('NUMERICAL_FAIL', 'DOMAIN_FAIL'):
                raise AcceptanceFailure(certificate['status'], str(certificate))
            own_points = {m:mandatory_callback_points(arm, advances[m], nodes, begin, end) for m,arm in arms.items()}
            points = own_points['reference'] | own_points[candidate]
            for t in sorted(points):
                reference_mirror = t not in own_points['reference']
                if reference_mirror:
                    reference_mirror_callbacks += 1
                states = {m:timed(m, 'inner_dense_readout', lambda r=r: r.evaluate(t, 'right')) for m,r in advances.items()}
                if t in edges:
                    observe(t, 'left', states)
                observe(t, 'right', states, do_record=t in records)
                for mode, arm in arms.items():
                    timed(mode, 'consumer_ack', lambda a=arm: a.ack(t))
            reference_mirror = False
            pre_side, post_side = event_snapshot_sides(end, edges)
            observe(end, pre_side, {m:r.state for m,r in advances.items()})
            if rod or bc:
                for mode, arm in arms.items():
                    timed(mode, 'events', lambda a=arm: a.commit(end, rod=rod, bc=bc, rgc=bc,
                        rod_sample_time=max(p['start_seconds'], end-.0001+1e-13) if rod else None))
                observe(end, post_side, do_record=True)
            elif end in records:
                observe(end, 'right', do_record=True)
            for mode, arm in arms.items():
                timed(mode, 'consumer_ack', lambda a=arm: a.ack(end))
            public = end
            if candidate=='gpu_dop853' and not restore and gate_receipt is None:
                first_monitor=timed(None,'paired_entry_full_monitor',lambda:comparator.compare_monitors(
                    monitors['reference'],monitors[candidate],end-p['start_seconds']))
                if first_monitor['status']!='PASS':
                    raise AcceptanceFailure(comparison_status(first_monitor),'First 100us full monitor gate failed')
                if not dense_reports or any(r['status']!='PASS' or r.get('complete_coverage') is not True for r in dense_reports):
                    raise AcceptanceFailure('INCONCLUSIVE','First 100us full dense coverage is not qualified')
                gate_receipt=dict(status='PASS',run_id=token,candidate=candidate,identity=canonical_hash(identity),
                    hot_started_unix=started,completed_unix=time.time(),time_seconds=end,
                    numerical_status='PASS',domain_status='PASS',event_status='PASS',monitor_status='PASS',
                    dense_status='PASS',recording_and_acknowledgement_complete=True,complete_coverage=True,
                    charge_category='paired',scope='First100us only; full8ms and fresh-process restoration remain required')
                enforce_paired_entry_gate(candidate,token,identity,p,started,time.time(),gate_receipt)
                write_json(output/'paired_entry_gate.json',gate_receipt)
            if end == 5.0:
                prefix_summary = paired_timing_summary(phases, reference_mirrors, candidate)
                prefix = {m:dict(hot_seconds=prefix_summary['hot_seconds'][m], raw_union_hot_seconds=prefix_summary['raw_union_hot_seconds'][m], exclusive_phase_seconds=dict(phases[m])) for m in modes}
                prefix['reference_mirror_QA_seconds'] = dict(reference_mirrors)
                prefix['conservative_speedup'] = prefix_summary['conservative_speedup']
                prefix['raw_union_speedup'] = prefix_summary['raw_union_speedup']
            if phase == 'paired' and not restore and end == p['checkpoint_seconds']:
                stage = 'CHECKPOINT_SAVE_WITH_UNINTERRUPTED_ORACLE'
                hashes = {}
                for mode, arm in arms.items():
                    cursor = timed(mode, 'recording', recorders[mode].checkpoint_payload)
                    payload = timed(None, 'checkpoint_construct', lambda m=mode,a=arm: a.checkpoint_payload(monitor=monitors[m].payload(), recording=cursor))
                    hashes[mode] = timed(None, 'checkpoint_save_hash', lambda: checkpoint_hash(payload))
                    checkpoint_receipts[mode] = timed(None, 'checkpoint_save', lambda m=mode: save_bundle(output/(m+'_checkpoint'),
                        kind='complete_model_checkpoint', identity=checkpoint_identity(identity,m), payload=payload, qualification=QUALIFICATION))
                    timed(None, 'checkpoint_recording_copy', lambda m=mode: shutil.copytree(output/'uninterrupted'/m, output/'restored'/m))
                    del payload
                session = dict(identity=identity, candidate=candidate, writer_pid=os.getpid(), public_time_seconds=public,
                    last_validated_time_seconds=last_validated, next_event=event_index+1,
                    hot_started_unix=started, recorded_times=sorted(recorded), checkpoint_payload_hashes=hashes)
                write_json(output/'session_checkpoint.json', dict(session, session_hash=canonical_hash(session)))
                write_json(output/'checkpoint_receipt.json', dict(status='SAVED_FOR_NEW_PROCESS_SUFFIX', time_seconds=public, receipts=checkpoint_receipts))
                stage = 'FULL_PAIRED_ADVANCE'
        stage = 'RESTORE_FINAL_COMPARE' if restore else 'FINAL_COMPARE'
        guard(True)
        if recorded != records or any(a.time_seconds != p['stop_seconds'] for a in arms.values()):
            raise AcceptanceFailure('INCONCLUSIVE', 'Full trajectory or 45-field recording coverage incomplete')
        final_monitor = timed(None, 'full_monitor_equivalence', lambda: comparator.compare_monitors(monitors['reference'], monitors[candidate], p['stop_seconds']-p['start_seconds']))
        if final_monitor['status'] != 'PASS':
            raise AcceptanceFailure(comparison_status(final_monitor), 'Complete monitor comparison: '+str(final_monitor['issues'][:3]))
        suffix = 'restored' if restore else 'uninterrupted'
        monitor_hashes = {}
        recording_hashes = {}
        for mode in modes:
            monitor_payload = timed(None, 'monitor_export', monitors[mode].payload)
            monitor_hashes[mode] = timed(None, 'monitor_hash', lambda: checkpoint_hash(monitor_payload))
            timed(None, 'monitor_save', lambda m=mode: save_bundle(output/(suffix+'_'+m+'_monitor'), kind='validation_evidence',
                identity=dict(identity, mode=m), payload=dict(monitor=monitor_payload, report=monitors[m].report(p['stop_seconds']-p['start_seconds'])), qualification=QUALIFICATION))
            timed(mode, 'recording', recorders[mode].checkpoint_payload)
            recording_hashes[mode] = timed(None, 'recording_content_hash', lambda m=mode: recording_content_hash(output/suffix/m))
        replay_receipts = {}
        source = ROOT/assembly.manifest['configuration']['source_run']/'centered_inputs_v2'
        frame_map = np.load(source/'source_frame_to_unique.npy', allow_pickle=False)
        frames = np.load(source/'source_unique_gray8.npy', mmap_mode='r')
        indices = np.arange(748, 752, dtype=np.int64)
        for mode in modes:
            replay_receipts[mode] = timed(mode, 'recording', lambda m=mode: recorders[m].finish(
                np.asarray(frames[frame_map[indices]]), indices.astype(np.float64)/150., indices,
                metadata=dict(input_class=candidate+' full paired acceptance', runtime={'stimulus_origin_seconds':0.},
                    replay=dict(source_construction_initial=True, qualified_background=False, paired_validation=True))))
        sample_activity('after')
        if restore:
            expected = read_json(source_output/'uninterrupted_oracle.json')
            if oracle_index != len(oracle) or monitor_hashes != expected['monitor_hashes'] or recording_hashes != expected['recording_hashes']:
                raise AcceptanceFailure('INCONCLUSIVE', 'RESTORE_INCONCLUSIVE: suffix hash coverage differs; full numeric comparison remains required')
            restore_result = dict(status='RESTORE_PASS', fresh_process=True, writer_pid=session['writer_pid'], restore_pid=os.getpid(),
                oracle_callbacks_checked=oracle_index, monitor_hashes=monitor_hashes, recording_hashes=recording_hashes,
                complete_payload_hashes=checkpoint_receipts, replay_receipts=replay_receipts, **metrics())
            write_json(output/'restore_result.json', restore_result)
            return restore_result
        full = dict(status='FEASIBILITY_NUMERICAL_PASS' if phase == 'feasibility' else 'PAIRED_UNINTERRUPTED_COMPLETE',
            numerical_status='PASS', domain_status='PASS', event_status='PASS', monitor_status='PASS',
            dense_status='PASS' if dense_reports and all(r['status'] == 'PASS' and r.get('complete_coverage') is True for r in dense_reports) else 'INCONCLUSIVE',
            dense_certificates=dense_reports, comparisons=comparisons, scope_seconds=[p['start_seconds'],p['stop_seconds']],
            ordinary_recording_samples=len(recorded), observation_fields=len(assembly.recording_fields),
            complete_monitor_callbacks={m:monitors[m].count for m in modes}, checkpoint_required=phase == 'paired',
            restore_status='PENDING' if phase == 'paired' else 'NOT_EXERCISED', hardware=environment, replay_receipts=replay_receipts, **metrics(), **QUALIFICATION)
        if phase == 'paired':
            write_json(output/'uninterrupted_oracle.json', dict(callbacks=oracle, monitor_hashes=monitor_hashes, recording_hashes=recording_hashes))
        write_json(output/'uninterrupted_result.json', full)
        return full
    except Exception as exc:
        failure = dict(status=_failure_status(exc, stage), stage=stage, error=type(exc).__name__, message=str(exc),
            public_time_seconds=public, last_validated_time_seconds=last_validated, restored_process=restore,
            dense_certificates=dense_reports,comparisons=comparisons,
            dense_summary=dict(intervals_completed=len(dense_reports),
                complete_intervals=sum(r.get('status')=='PASS' and r.get('complete_coverage') is True for r in dense_reports)),
            **metrics(), **QUALIFICATION)
        write_json(output/'failure.json', failure)
        # Resource protection stops immediately; other failures first preserve accepted records.
        if failure['status'] != 'RESOURCE_FAIL':
            for mode, recorder in recorders.items():
                try:
                    recorder.checkpoint_payload()
                except Exception as save_error:
                    failure.setdefault('recording_flush_errors', {})[mode] = str(save_error)
            write_json(output/'failure.json', failure)
        write_json(output/'REQUEST_STOP', dict(status=failure['status'], message=str(exc), run_id=token))
        raise
    finally:
        if activity is not None:
            activity.close()
        for mode in reversed(modes):
            if mode in arms:
                arms[mode].close()

def charge_budget_interval(ledger, phase, begin, end, hot_started, *, uncertain=False):
    if end < begin:
        raise AcceptanceFailure('RESOURCE_FAIL','System wall clock moved backwards; no reset allowed')
    if uncertain:
        if hot_started is None:
            ledger.charge('cold',end-begin)
        ledger.charge(phase,end-begin)
    elif hot_started is None:
        ledger.charge('cold',end-begin)
    else:
        cold_end=min(end,max(begin,hot_started))
        if cold_end>begin:
            ledger.charge('cold',cold_end-begin)
        if end>cold_end:
            ledger.charge(phase,end-cold_end)


def supervise(output, phase, candidate='radau', restore_from=None):
    """A single owner charges process startup, restore, comparison and all retries."""
    validate_candidate_phase(candidate,phase)
    output = Path(output).resolve()
    source_preflight=scientific_source_preflight(ROOT)
    if source_preflight['status']!='PASS':
        return dict(source_preflight,requested_output=str(output),phase=phase)
    protocol = load_protocol(ROOT,candidate)
    identity = identity_for(protocol)
    restore_import=None
    if restore_from is not None:
        from retina_sim.restore_import import inspect_source,bind_identity
        if phase!='paired' or candidate!='gpu_dop853':
            raise ContractError('Restore import requires the full GPU paired protocol')
        restore_import=inspect_source(ROOT,restore_from,protocol,identity,REPORT_PARENT)
        enforce_paired_entry_gate(candidate,restore_import['source_token'],restore_import['source_identity'],protocol,
            restore_import['source_hot_started_unix'],time.time(),restore_import['entry_gate'])
        identity=bind_identity(identity,restore_import)
    with BudgetLedger(ROOT) as ledger:
        if candidate=='gpu_dop853':
            verify_fallback_origin(ROOT,protocol,ledger.data)
        elif phase=='feasibility' and any(a['run_id']==FALLBACK_ORIGIN['run_id'] for a in ledger.data['attempts']):
            raise ContractError('Radau feasibility is closed after the preserved r5 numerical failure')
        token = ledger.begin(output, phase, identity)
        attempt_started=ledger.data['attempts'][-1]['started_unix']
        write_json(output/'job.json', dict(token=token, phase=phase, candidate=candidate, supervisor_pid=os.getpid(),
            supervisor_process=process_identity(os.getpid()), identity=identity,
            **({} if restore_import is None else dict(restore_source=restore_import,restore_attempt_started_unix=attempt_started))))
        write_json(output/'protocol.json', protocol)
        write_json(output/'prospective_identity.json', identity)
        process = None
        owner = None
        hot_started = None if restore_import is None else attempt_started
        last_tick = time.time()
        outcome = None
        entry_gate_receipt = None
        def check_entry_gate():
            nonlocal entry_gate_receipt
            if restore_import is not None:
                enforce_paired_entry_gate(candidate,restore_import['source_token'],restore_import['source_identity'],protocol,
                    restore_import['source_hot_started_unix'],time.time(),restore_import['entry_gate'])
                return
            receipt=entry_gate_receipt
            gate_path=output/'paired_entry_gate.json'
            if candidate=='gpu_dop853' and receipt is None and gate_path.exists():
                try:
                    receipt=read_json(gate_path,permission_retry_seconds=.25)
                except (OSError,ValueError,ContractError):
                    enforce_paired_entry_gate(candidate,token,identity,protocol,hot_started,time.time())
                    raise
            if enforce_paired_entry_gate(candidate,token,identity,protocol,hot_started,time.time(),receipt):
                entry_gate_receipt=receipt
        def account():
            nonlocal last_tick, hot_started
            if protocol.get('diagnostic_scope_only') and time.time()-ledger.data['attempts'][-1]['started_unix'] > protocol['diagnostic_wall_cap_seconds']:
                raise AcceptanceFailure('RESOURCE_FAIL','Explicit short diagnostic wall cap reached')
            status_path = output/'status.json'
            try:
                status = read_json(status_path,permission_retry_seconds=.25) if status_path.exists() else {}
                if not isinstance(status,dict):
                    raise ContractError('Worker status must be a JSON object')
                value=status.get('hot_started_unix')
            except (OSError,ValueError,ContractError):
                now=time.time()
                try:
                    # Unknown transition time is conservatively charged to both applicable caps.
                    charge_budget_interval(ledger,phase,last_tick,now,hot_started,uncertain=True)
                finally:
                    last_tick=now
                check_entry_gate()
                raise
            now = time.time()
            if now < last_tick:
                raise AcceptanceFailure('RESOURCE_FAIL', 'System wall clock moved backwards; no reset allowed')
            try:
                if value is not None:
                    if not isinstance(value, (int,float)) or isinstance(value,bool) or not np.isfinite(value) or value > now:
                        raise ContractError('Invalid worker budget boundary')
                    if hot_started is None:
                        hot_started = value
                        if value < last_tick:
                            # The polling interval was conservatively charged cold; also charge its real model portion.
                            ledger.charge(phase, last_tick-value)
                    elif value != hot_started:
                        raise ContractError('Worker attempted to reset the model budget clock')
            except ContractError:
                try:
                    charge_budget_interval(ledger,phase,last_tick,now,hot_started,uncertain=True)
                finally:
                    last_tick=now
                raise
            try:
                charge_budget_interval(ledger,phase,last_tick,now,hot_started)
            finally:
                last_tick=now
            check_entry_gate()
        def terminate_owned():
            if owner is not None:
                owner.terminate()
            if process is None or process.poll() is not None:
                return
            process.kill()
            process.wait(timeout=10)
        def child(restore):
            nonlocal process, owner
            command = [sys.executable, '-X', 'utf8', '-B', str(Path(__file__)),
                '--output', str(output), '--phase', phase, '--candidate', candidate, '--worker', '--token', token]
            if restore:
                command.append('--restore')
            label = 'restored' if restore else 'uninterrupted'
            owner = WindowsWorkerJob()
            with (output/(label+'_stdout.log')).open('x', encoding='utf8') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                owner.bind_launcher(process)
                handshake_path=output/(label+'_worker_handshake.json')
                while process.poll() is None:
                    account()
                    if owner.worker is None and handshake_path.exists():
                        receipt=owner.accept(read_json(handshake_path),token=token,label=label,supervisor=process_identity(os.getpid()))
                        write_json(output/(label+'_worker_ownership.json'),receipt)
                    if (output.parent/'REQUEST_STOP').exists():
                        raise AcceptanceFailure('RESOURCE_FAIL', 'Parent REQUEST_STOP reached')
                    if (output/'REQUEST_STOP').exists():
                        try:
                            saved = read_json(output/'REQUEST_STOP')
                        except (ValueError, ContractError) as error:
                            raise AcceptanceFailure('RESOURCE_FAIL', 'Explicit or malformed REQUEST_STOP') from error
                        if not isinstance(saved, dict) or saved.get('run_id') != token:
                            raise AcceptanceFailure('RESOURCE_FAIL', 'Explicit REQUEST_STOP without this worker identity')
                        status = saved.get('status', 'RESOURCE_FAIL')
                        if status not in FAILURES:
                            status = 'RESOURCE_FAIL'
                        raise AcceptanceFailure(status, saved.get('message', 'Run stopped'))
                    try:
                        usage = owner.resources(process._handle)
                        if usage is not None:
                            resource_check(rss=usage['peak_rss_bytes'], available=usage['available_physical_bytes'])
                            write_json(output/(label+'_supervisor_resources.json'),usage)
                    except OSError as error:
                        if process.poll() is None:
                            raise AcceptanceFailure('RESOURCE_FAIL', 'Owned worker resource query failed: '+str(error)) from error
                    time.sleep(.2)
                account()
            if owner.worker is None and process.returncode == 0:
                raise AcceptanceFailure('INCONCLUSIVE','Worker exited without verified actual-process ownership')
            owner.close(); owner=None
            if process.returncode:
                if (output/'failure.json').exists():
                    saved = read_json(output/'failure.json')
                    raise AcceptanceFailure(saved['status'], saved['message'])
                raise AcceptanceFailure('INCONCLUSIVE', 'Worker exited without complete failure evidence; see '+label+'_stdout.log')
        try:
            if restore_import is not None:
                from retina_sim.restore_import import copy_recording_prefix,verify_source_pins
                copy_recording_prefix(ROOT,restore_import,output)
                child(True)
                restored=read_json(output/'restore_result.json')
                if restored['status']!='RESTORE_PASS':
                    raise AcceptanceFailure('RESTORE_FAIL','Imported fresh-process suffix did not pass')
                verify_source_pins(ROOT,restore_import)
                account()
                outcome=dict(status='RESTORE_RETRY_PASS',restore_status='RESTORE_PASS',
                    scientific_status='IMPORTED_MAIN_PASS_AND_FULL_SUFFIX_PASS',restore_result=restored,
                    diagnostic_restore_only=True,restore_source=restore_import,adoptable=False,performance=None,
                    original_failure_root_cause='UNRESOLVED',
                    qualification_note='A passing retry preserves the original intermittent failure; no default adoption or unknown-root-cause fix is asserted',
                    cumulative_budget_seconds=dict(ledger.data['used']),**QUALIFICATION)
                write_json(output/'result.json',outcome)
                ledger.finish(outcome['status'])
                return outcome
            child(False)
            first = read_json(output/'uninterrupted_result.json')
            if phase == 'paired':
                if not (output/'session_checkpoint.json').is_file() or not (output/'uninterrupted_oracle.json').is_file():
                    raise AcceptanceFailure('RESTORE_FAIL', 'Uninterrupted run did not produce the full checkpoint/suffix oracle')
                child(True)
                restored = read_json(output/'restore_result.json')
                if restored['status'] != 'RESTORE_PASS':
                    raise AcceptanceFailure('RESTORE_FAIL', 'Fresh-process continuation did not pass')
                pairs = [first['hot_seconds']]
                raw_pairs = [first['raw_union_hot_seconds']]
                for attempt in ledger.data['attempts'][:-1]:
                    previous = Path(attempt['output'])/'result.json'
                    if attempt['phase'] != 'paired' or attempt['identity'] != canonical_hash(identity) or not previous.is_file():
                        continue
                    prior = read_json(previous)
                    if eligible_timing_repeat(prior,first['hardware']):
                        pairs.append(prior['hot_seconds'])
                        raw_pairs.append(prior['raw_union_hot_seconds'])
                performance = dict(performance_result(pairs,candidate), timing_basis='CONSERVATIVE_REFERENCE_MANDATORY_VS_CANDIDATE_UNION')
                raw_performance = dict(performance_result(raw_pairs,candidate), timing_basis='RAW_UNION_DIAGNOSTIC_ONLY', adoptable=False)
                qualified = first['dense_status'] == 'PASS'
                clean_timing = first.get('timing_environment_status') == 'PASS'
                if not clean_timing:
                    reason='TIMING_RECHECK_REQUIRED' if first.get('timing_environment_status')=='GRAPHICS_RECHECK_REQUIRED' else 'GPU_TIMING_ENVIRONMENT_UNCONFIRMED'
                    performance = dict(performance, status='INCONCLUSIVE', adoptable=False, reason=reason,
                        timing_environment_status=first.get('timing_environment_status'))
                outcome = dict(first, status=performance['status'] if qualified else 'INCONCLUSIVE',
                    scientific_status='PASS' if qualified else 'INCONCLUSIVE', restore_status='RESTORE_PASS',
                    restore_result=restored, performance=performance, raw_union_performance=raw_performance,
                    adoptable=qualified and performance['adoptable'],
                    qualification_note='Full relative represented-polynomial, callback, domain, event, monitor and restore gates required; no absolute ODE-accuracy proof or background/fitting qualification')
            else:
                outcome = dict(first, status='FEASIBILITY_PASS' if first['dense_status'] == 'PASS' else 'INCONCLUSIVE',
                    scientific_status='PARTIAL_100US_ONLY', adoptable=False, performance=None,
                    qualification_note='100 microseconds cannot qualify the full 8ms trajectory or restoration')
            if protocol.get('diagnostic_scope_only'):
                outcome.update(diagnostic_scope_only=True,
                    scientific_status='SHORT_SCOPE_PASS' if outcome.get('scientific_status')=='PASS' else outcome.get('scientific_status'),
                    adoptable=False, qualification_note=protocol['diagnostic_note'])
                if outcome.get('performance') is not None:
                    outcome['performance']=dict(outcome['performance'],adoptable=False,short_scope_only=True)
            outcome=preserve_unresolved_restore_hold(outcome,protocol)
            account()
            outcome['cumulative_budget_seconds'] = dict(ledger.data['used'])
            write_json(output/'result.json', outcome)
            retry_kind = 'timing_recheck' if outcome.get('scientific_status') == 'PASS' and outcome.get('performance', {}).get('reason') == 'TIMING_RECHECK_REQUIRED' else None
            ledger.finish(outcome['status'], retry_kind=retry_kind)
            return outcome
        except BaseException as exc:
            terminate_owned()
            try:
                account()
            except BaseException as accounting_error:
                if not isinstance(exc, AcceptanceFailure):
                    exc = accounting_error
            status = _failure_status(exc, 'SUPERVISOR')
            if isinstance(exc, KeyboardInterrupt):
                status = 'RESOURCE_FAIL'
            failure = dict(status=status, error=type(exc).__name__, message=str(exc),
                cumulative_budget_seconds=dict(ledger.data['used']), **QUALIFICATION)
            if not (output/'failure.json').exists():
                progress=output/'dense_progress.json'
                if progress.is_file():
                    try:
                        failure['completed_dense_QA']=read_json(progress,permission_retry_seconds=.25)
                    except (OSError,ValueError,ContractError) as error:
                        failure['dense_progress_read_error']=str(error)
                write_json(output/'failure.json', failure)
            write_json(output/'REQUEST_STOP', dict(status=status, message=str(exc), run_id=token))
            ledger.finish(status)
            raise
        finally:
            terminate_owned()
            if owner is not None:
                owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=CANDIDATES, default='radau')
    parser.add_argument('--phase', choices=('feasibility','paired'), required=True)
    parser.add_argument('--output', required=True, help='Fresh immediate subdirectory of '+REPORT_PARENT)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--restore', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--restore-from', help='Read-only completed main trajectory imported into a new supervised restore-only attempt')
    parser.add_argument('--token', help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    if args.worker:
        if args.restore_from:
            parser.error('Only the supervisor may select a restore source')
        result = run_worker(output, restore=args.restore, token=args.token, candidate=args.candidate, phase=args.phase)
    else:
        if args.restore or args.token:
            parser.error('Manual restore/token overrides are forbidden; the supervisor owns continuation')
        result = supervise(output, args.phase, args.candidate,None if args.restore_from is None else (ROOT/args.restore_from).resolve())
    print(json.dumps({k:result[k] for k in ('status','stage','scientific_status','restore_status','hot_seconds','performance','cumulative_budget_seconds','issues','model_forward_started','cuda_initialized','adoptable','budget_charged') if k in result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
