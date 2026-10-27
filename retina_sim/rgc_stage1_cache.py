"""Identity-bound fixed-model voltage capture, with complete resumable checkpoints."""
from pathlib import Path
from copy import deepcopy
from time import perf_counter
import json,math,os
import numpy as np
from .contracts import ContractError,canonical_hash
from .group_time_runner import GroupTimeArm
from .group_time_monitor import GroupTimeMonitor
from .group_time_storage import save_bundle,load_bundle
from .background_checkpoint import MemoryGuard

SCHEMA='rgc-stage1-fixed-endogenous-cache-1'


def write(path,value):
    p=Path(path);temp=p.with_name(p.name+'.writing');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8');temp.replace(p)


class StageWorker:
    def __init__(self,root,out,phase,wall_limit):
        self.root=Path(root);self.out=Path(out);self.phase=Path(phase);self.started=perf_counter();self.limit=wall_limit
        self.lease=self.root/'reports/ac_vs_redesign_v1/ACTIVE_WORKER.json'
        self.owner=dict(pid=os.getpid(),output=str(out),scope='RGC stage1 fixed endogenous capture; BC/AC queue remains stopped')
        self.memory=MemoryGuard(17179869184,4294967296)
    def __enter__(self):
        if not self.memory.check():raise ContractError('Measured stage1 RAM preflight failed')
        with self.lease.open('x',encoding='utf8') as stream:json.dump(self.owner,stream)
        return self
    def check(self):
        if (self.root/'REQUEST_STOP').exists() or (self.phase/'REQUEST_STOP').exists():raise ContractError('Explicit RGC stage1 STOP')
        if json.loads(self.lease.read_text())!=self.owner:raise ContractError('Stage1 worker ownership changed')
        if not self.memory.check() or perf_counter()-self.started>self.limit:raise ContractError('Stage1 single-worker safety guard')
    def __exit__(self,*args):
        receipts=self.out/'resource_receipts';receipts.mkdir(exist_ok=True)
        write(receipts/(str(os.getpid())+'_'+str(int(self.started*1000))+'.json'),dict(self.owner,elapsed_seconds=perf_counter()-self.started,
            peak_RSS_bytes=self.memory.peak_rss_bytes,released=True,wall_safety_seconds=self.limit))
        if self.lease.exists() and json.loads(self.lease.read_text())==self.owner:self.lease.unlink()


def capture(root,phase,spec,case,*,resume=False,stop_after_checkpoint=False):
    from tools.validate_ac_vs_native_support import load
    from tools.validate_ac_vs_equilibrium_start import initial_match
    from tools.validate_ac_vs_timescale import calendar
    root=Path(root);phase=Path(phase);out=phase/'cache'/case['case_id'];out.mkdir(parents=True,exist_ok=True)
    finished=out/'manifest.json'
    if finished.exists():
        cached,manifest=read_cache(out,spec,case);cached.close();return manifest
    identity=canonical_hash(dict(spec=spec,case=case));pointer=out/'latest_checkpoint.json'
    if pointer.exists() and not resume:raise ContractError('Partial cache exists; explicit resume required')
    with StageWorker(root,out,phase,spec['single_worker_safety_seconds']) as worker:
        a,seed,parent=load(root/spec['prepared_model']);seed=deepcopy(seed)
        if canonical_hash(seed)!=spec['source_seed_sha256']:raise ContractError('Frozen initial state/history identity differs')
        for pop,selection in spec['recording'].items():
            p=a.populations[pop]
            if np.asarray(p.group_ids)[selection['rows']].tolist()!=selection['global_ids']:raise ContractError('Cache RGC group/row identity differs')
        drive=load_bundle(root/case['drive_bundle'],expected_kind='validation_evidence')['payload']
        if canonical_hash(drive)!=case['drive_sha256'] or drive['stimulus_sha256']!=case['stimulus_sha256']:raise ContractError('Actual endogenous drive identity differs')
        protocol=deepcopy(parent['protocol']);protocol.update(start_seconds=drive['model_origin_s'],stop_seconds=drive['model_stop_s'],
            checkpoint_seconds=drive['model_stop_s'],input_frame_indices=sorted(drive['native_inputs']),rgc_stage1_cache_identity=identity)
        seed['native_inputs']=deepcopy(drive['native_inputs']);seed['rod_inputs']=deepcopy(drive['rod_inputs'])
        count=int(round((protocol['stop_seconds']-protocol['start_seconds'])/.0001))+1
        arrays={k:np.zeros((count,len(v['rows'])),dtype=np.float64) for pop,v in spec['recording'].items() for k in [pop+':V',pop+':gE',pop+':gI']}
        expected=np.zeros((int(round((count-1)/10)),len(spec['recording']['parasol_RGC_ON']['rows'])))
        recorded=0;bins=0
        if pointer.exists():
            point=json.loads(pointer.read_text());bundle=load_bundle(out/point['directory'],expected_kind='complete_model_checkpoint')
            if bundle['identity']!=dict(cache_identity=identity):raise ContractError('Checkpoint scientific/input/execution identity differs')
            arm=GroupTimeArm.from_checkpoint(a,seed,protocol,bundle['payload'],mode='reference',guard=worker.check)
            monitor=GroupTimeMonitor.from_payload(a,seed['states'],bundle['payload']['monitor'],parameters=seed['inner_parameters'])
            saved=bundle['payload']['recording'];recorded=saved['recorded'];bins=saved['bins']
            for k,v in saved['arrays'].items():arrays[k][:recorded]=v
            expected[:bins]=saved['expected'];init=saved['initial_match']
        else:
            arm=GroupTimeArm(a,seed,protocol,mode='reference',guard=worker.check)
            monitor=GroupTimeMonitor(a,seed['states'],parameters=seed['inner_parameters']);init,_=initial_match(arm,seed)
        def observe(t,side):
            s=arm.snapshot(t,side,queue_node=True)
            monitor.observe(t,side,s['states'],s['rhs'],transitions=s['transitions'],ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
            if monitor.domain_failures:raise ContractError('Original physical/domain/history gate failed')
            return s
        def record(s,j):
            nonlocal recorded,bins
            if j!=recorded:raise ContractError('Cache source sample skipped or repeated')
            for pop,selection in spec['recording'].items():
                for field,port,unit in [('V','voltage','V'),('gE','gE','S'),('gI','gI','S')]:
                    if s['port_units'][pop,port]!=unit:raise ContractError('Endogenous cache unit differs')
                    arrays[pop+':'+field][j]=s['ports'][pop,port][selection['rows']]
            if j%10==0 and j<count-1:
                if bins!=j//10:raise ContractError('Native RGC bin clock differs')
                expected[bins]=s['ports']['parasol_RGC_ON','expected_rate_Hz'][spec['recording']['parasol_RGC_ON']['rows']];bins+=1
            recorded+=1
        def checkpoint(j):
            name='checkpoint_'+str(j).zfill(7)
            payload=arm.checkpoint_payload(monitor=monitor.payload(),recording=dict(recorded=recorded,bins=bins,
                arrays={k:v[:recorded].copy() for k,v in arrays.items()},expected=expected[:bins].copy(),initial_match=init))
            save_bundle(out/name,kind='complete_model_checkpoint',identity=dict(cache_identity=identity),payload=payload,
                qualification=dict(default_adopted=False,fit_eligible=False))
            write(pointer,dict(directory=name,source_sample=j,cache_identity=identity))
        try:
            if not arm.initial_commit_done:
                s=observe(arm.start,'right');record(s,0);arm.initial_commit();observe(arm.start,'right');arm.ack(arm.start)
            for end,flags in calendar(protocol):
                if end<=arm.inner_time:continue
                begin=arm.inner_time;arm.ensure_native(end,'left');arm.advance_inner(begin,end)
                before=observe(end,'left' if flags['frame'] else 'right')
                # Only author100us clock points are cache nodes; no new ODE endpoints.
                j=int(round((end-arm.start)/.0001))
                author_tick=flags['rod'] or end==arm.stop
                if author_tick and j==recorded:record(before,j)
                if flags['rod'] or flags['bc']:arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],rod_sample_time=max(arm.start,end-.0001+1e-13) if flags['rod'] else None)
                observe(end,'right');arm.ack(end)
                if author_tick and (j in spec.get('additional_checkpoint_ticks',[]) or j==500 or j>0 and j%5000==0 or end==arm.stop):
                    checkpoint(j)
                    if stop_after_checkpoint:raise ContractError('Requested engineering checkpoint stop')
                if j%1000==0:write(out/'progress.json',dict(recorded=recorded,total=count,source_seconds=end-arm.start,last_checkpoint=json.loads(pointer.read_text()) if pointer.exists() else None))
            if recorded!=count or bins!=len(expected):raise ContractError('Complete source/node/bin coverage missing')
            np.savez_compressed(out/'endogenous.npz',time_source_s=np.arange(count)*.0001,expected_rate_base_Hz=expected,**arrays)
            import hashlib
            digest=hashlib.sha256((out/'endogenous.npz').read_bytes()).hexdigest()
            result=dict(schema=SCHEMA,cache_identity=identity,status='COMPLETE_FIXED_ENDOGENOUS_CACHE',case=case,
                arrays_sha256=digest,units=dict(V='V',gE='S',gI='S'),recorded=recorded,bins=bins,initial_match=init,
                raw_absolute_monitor=monitor.report(arm.stop-arm.start),metrics=arm.metrics,readout_parameters_not_used_to_change_upstream=True)
            write(finished,result);return result
        except Exception as exc:
            failures=out/'failure_receipts';failures.mkdir(exist_ok=True)
            write(failures/(str(os.getpid())+'.json'),dict(error=type(exc).__name__,reason=str(exc),source_seconds=arm.inner_time-arm.start,
                recorded=recorded,last_checkpoint=json.loads(pointer.read_text()) if pointer.exists() else None))
            raise
        finally:arm.close()


def read_cache(directory,spec,case):
    import hashlib
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text())
    if m.get('schema')!=SCHEMA or m.get('status')!='COMPLETE_FIXED_ENDOGENOUS_CACHE' or m['cache_identity']!=canonical_hash(dict(spec=spec,case=case)):
        raise ContractError('Complete endogenous cache identity differs; no regional/clamp substitution')
    if hashlib.sha256((directory/'endogenous.npz').read_bytes()).hexdigest()!=m['arrays_sha256']:raise ContractError('Cached trajectory payload changed')
    if m['units']!=dict(V='V',gE='S',gI='S') or not m['initial_match']['passed']:raise ContractError('Cached initial state/unit contract differs')
    return np.load(directory/'endogenous.npz',allow_pickle=False),m
