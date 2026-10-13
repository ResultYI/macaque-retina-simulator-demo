"""Identity-bound complete uniform-background checkpoints; no count-only cache."""
from dataclasses import asdict
from pathlib import Path
import ctypes,hashlib,json,os,time
import numpy as np
from .contracts import ContractError,canonical_hash
from .background import NUMERICAL_UNIT_SCALES,_WindowMonitor,timing_requirements


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n',encoding='utf8')
    os.replace(temporary,path)


class MemoryGuard:
    """Cooperative check at integration boundaries; leaves the last valid state."""
    def __init__(self,max_rss_bytes=4*1024**3,min_available_bytes=1024**3):
        self.max_rss_bytes=int(max_rss_bytes);self.min_available_bytes=int(min_available_bytes)
        self.stop_reason=None;self.last={};self.peak_rss_bytes=0
    def check(self):
        if os.name!='nt':raise ContractError('Memory guard requires the configured Windows host; no unmeasured fallback')
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in
                ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        class Status(ctypes.Structure):
            _fields_=[('dwLength',wintypes.DWORD),('dwMemoryLoad',wintypes.DWORD)]+[(k,ctypes.c_ulonglong) for k in
                ('ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual')]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        c=Counters();c.cb=ctypes.sizeof(c);s=Status();s.dwLength=ctypes.sizeof(s)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb) or not kernel.GlobalMemoryStatusEx(ctypes.byref(s)):
            raise OSError(ctypes.get_last_error(),'Cannot measure safety memory budget')
        self.peak_rss_bytes=max(self.peak_rss_bytes,int(c.PeakWorkingSetSize))
        self.last={'rss_bytes':int(c.WorkingSetSize),'process_peak_rss_bytes':self.peak_rss_bytes,'available_physical_bytes':int(s.ullAvailPhys),
            'rss_limit_bytes':self.max_rss_bytes,'minimum_available_bytes':self.min_available_bytes}
        if self.peak_rss_bytes>self.max_rss_bytes:self.stop_reason='memory_peak_rss_limit'
        elif c.WorkingSetSize>self.max_rss_bytes:self.stop_reason='memory_rss_limit'
        elif s.ullAvailPhys<self.min_available_bytes:self.stop_reason='available_memory_reserve'
        return self.stop_reason is None


class BackgroundCheckpointStore:
    def __init__(self,root,runner,reference_state,light_context,policy):
        self.runner=runner;self.reference=reference_state;self.policy=policy
        self.expected_context={'background_context_hash':light_context.background_context_hash,'phase':'uniform_background'}
        self.reference.input_position.update(self.expected_context)
        c=runner.compiled
        matrices=[{'specification':x.specification,'shape':x.matrix.shape,'data':x.matrix.data,
                   'indices':x.matrix.indices,'indptr':x.matrix.indptr} for x in c.links]
        criteria={k:v for k,v in asdict(policy).items() if k not in ('max_wall_seconds','max_sim_seconds','max_observations','require_converged')}
        self.binding={'assembly_hash':c.assembly_hash,'actual_populations_hash':canonical_hash(tuple(c.populations.values())),
            'actual_links_and_weights_hash':canonical_hash(matrices),'gap_topology_conductance_hash':canonical_hash(c.spec.gap_junctions),
            'components_hash':canonical_hash({k:v.identity for k,v in c.components.items()}),
            'graph_hash':runner.model.graph.graph_hash,'parameter_hash':runner.model.parameters.parameter_hash,
            'model_hash':runner.model.model_hash,'state_schema_hash':canonical_hash(runner.model.state_specs),'clock_hash':canonical_hash(c.clock),
            'solver_hash':canonical_hash(reference_state.solver_config),'background_context_hash':light_context.background_context_hash,
            'criteria_hash':canonical_hash({'policy':criteria,'unit_scales':NUMERICAL_UNIT_SCALES}),
            'initial_monitor_reference_hash':canonical_hash(reference_state.ordered_state_arrays),
            'monitor_code_sha256':hashlib.sha256((Path(root)/'retina_sim/background.py').read_bytes()).hexdigest()}
        self.binding_hash=canonical_hash(self.binding)
    def coverage(self):
        monitor=_WindowMonitor(self.runner,self.reference,self.policy)
        return {'state_blocks':len(self.runner.model.state_specs),'covered_blocks':len(monitor.rows),
            'state_scalars':sum(np.asarray(x).size for x in self.reference.ordered_state_arrays),
            'missing':monitor.missing,'states':[r['spec'].state_name for r in monitor.rows],
            'timing_requirements':timing_requirements(self.runner,self.policy)}
    def save(self,path,state,report=None,*,origin_seconds=0.,safety=None):
        if any(state.input_position.get(k)!=v for k,v in self.expected_context.items()):
            raise ContractError('Only the actual declared uniform-background phase can be cached')
        checkpoint=json.loads(self.runner.checkpoint(state));payload=checkpoint['payload']
        auxiliary={k:payload[k] for k in ('grid_origin_seconds','sample_index','component_tick_indices','spike_states','delay_states','link_delay_states','input_position')}
        value={'schema':'retina-uniform-background-checkpoint-1','binding':self.binding,'binding_hash':self.binding_hash,
            'checkpoint':checkpoint,'complete_checkpoint_hash':canonical_hash(checkpoint),'auxiliary_time_rng_history_hash':canonical_hash(auxiliary),
            'absolute_time_seconds':state.time_seconds,'preparation_origin_seconds':float(origin_seconds),'report':report,'safety':safety}
        value['report_hash']=canonical_hash(report)
        atomic_json(path,value);return value
    def load(self,path,*,require_converged=False):
        value=json.loads(Path(path).read_bytes())
        if value.get('schema')!='retina-uniform-background-checkpoint-1' or value.get('binding')!=self.binding or value.get('binding_hash')!=self.binding_hash:
            raise ContractError('Background checkpoint actual graph/weights/parameters/background/solver/schema identity mismatch; no fallback')
        checkpoint=value['checkpoint'];payload=checkpoint['payload']
        auxiliary={k:payload[k] for k in ('grid_origin_seconds','sample_index','component_tick_indices','spike_states','delay_states','link_delay_states','input_position')}
        if canonical_hash(checkpoint)!=value['complete_checkpoint_hash'] or canonical_hash(auxiliary)!=value['auxiliary_time_rng_history_hash']:
            raise ContractError('Complete checkpoint time/RNG/history/delay identity changed')
        if canonical_hash(value.get('report'))!=value.get('report_hash'):raise ContractError('Background numerical report identity changed')
        state=self.runner.restore(json.dumps(checkpoint),expected_input_context=self.expected_context)
        if state.time_seconds!=value['absolute_time_seconds'] or value['preparation_origin_seconds']>state.time_seconds:
            raise ContractError('Background checkpoint absolute preparation time mismatch')
        if require_converged:self.require_convergence(value)
        return state,value
    def require_convergence(self,value):
        report=value.get('report') or {};coverage=self.coverage();windows=report.get('windows',[])
        criteria={k:v for k,v in asdict(self.policy).items() if k not in ('max_wall_seconds','max_sim_seconds','max_observations','require_converged')}
        if any(report.get('policy',{}).get(k)!=v for k,v in criteria.items()) or report.get('numerical_unit_registry')!=NUMERICAL_UNIT_SCALES:
            raise ContractError('Prepared report numerical thresholds/scales differ from the unchanged policy')
        if report.get('status')!='CONVERGED' or report.get('converged') is not True or report.get('uncovered_states') or report.get('checked_states')!=coverage['states']:
            raise ContractError('Checkpoint does not certify all actual registered states under the original numerical criteria')
        recent=windows[-self.policy.stable_windows:]
        if len(recent)!=self.policy.stable_windows or any(not r.get('passed') or not r.get('complete_window') or r.get('maximum_norm',float('inf'))>1 or r.get('history_consistency_failures') or r.get('domain_failures') for r in recent):
            raise ContractError('Checkpoint lacks the original consecutive complete convergence windows')
        elapsed=value['absolute_time_seconds']-value['preparation_origin_seconds']
        if abs(report.get('elapsed_seconds',-1)-elapsed)>1e-10:
            raise ContractError('Prepared report elapsed time does not match the full checkpoint')
        if elapsed+1e-12<coverage['timing_requirements']['minimum_required_seconds']:
            raise ContractError('Checkpoint has not reached the unchanged minimum timing requirement')


def prepare_with_checkpoint(root,runner,reference,light,policy,destination,*,resume_path=None,
                            guard=None,on_progress=None,on_boundary=None):
    """Same monitor for GUI/CLI; resumption conservatively starts a new 3-window streak."""
    from .background import constant_background_channels,prepare_background
    store=BackgroundCheckpointStore(root,runner,reference,light,policy);coverage=store.coverage()
    if coverage['missing'] or coverage['covered_blocks']!=coverage['state_blocks']:
        raise ContractError('Incomplete registered background state coverage: '+str(coverage['missing']))
    destination=Path(destination);guard=guard or MemoryGuard()
    state=runner.restore(runner.checkpoint(reference),expected_input_context=store.expected_context);origin=reference.time_seconds
    if resume_path:
        state,cached=store.load(resume_path);origin=cached['preparation_origin_seconds']
        if (cached.get('report') or {}).get('converged'):
            store.require_convergence(cached)
            if not guard.check():raise ContractError('Memory safety budget reached before reusing prepared stimulus state')
            report=dict(cached['report'],cache_reused=True,prepared_checkpoint_source=str(resume_path))
            store.save(destination,state,report,origin_seconds=origin,safety=guard.last)
            return state,report,store
    atomic_json(destination.with_name('background_start_checkpoint.json'),json.loads(runner.checkpoint(state)))
    store.save(destination,state,origin_seconds=origin,safety=guard.last)
    channels=constant_background_channels(runner.compiled,light,origin+policy.max_sim_seconds+1.)
    last_save=time.perf_counter();last_window=None;wall_start=last_save
    def progress(current):
        nonlocal last_save
        if not guard.check():return False
        if on_boundary is not None and on_boundary() is False:return False
        if time.perf_counter()-last_save>=45.:
            store.save(destination,current,origin_seconds=origin,safety=guard.last);last_save=time.perf_counter()
            if on_progress:on_progress(current,{'phase':'background','wall_seconds':last_save-wall_start,'last_window':last_window,'memory':guard.last})
        return True
    def window(current,row):
        nonlocal last_save,last_window
        last_window=row;store.save(destination,current,origin_seconds=origin,safety=guard.last);last_save=time.perf_counter()
        if on_progress:on_progress(current,{'phase':'background_window','wall_seconds':last_save-wall_start,'last_window':row,'memory':guard.last})
    state,report=prepare_background(runner,state,channels,policy,on_chunk=progress,on_window=window,
        reference_state=reference,preparation_origin_seconds=origin)
    if guard.stop_reason:report['budget_exit']=guard.stop_reason
    report.update(coverage=coverage,safety=dict(guard.last,cooperative_boundary_checks=True),
        complete_state_checkpoint=str(destination),cache_reused=False)
    store.save(destination,state,report,origin_seconds=origin,safety=guard.last)
    if report['converged']:store.require_convergence(json.loads(destination.read_bytes()))
    return state,report,store
