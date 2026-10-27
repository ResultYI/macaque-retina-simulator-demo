"""One preregistered short before/after CPU comparison; never resumes fit queue."""
from pathlib import Path
from copy import deepcopy
from time import perf_counter
import argparse,json,math,os,sys,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.ac_vs_rhs_readouts import contract,install
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.group_time_monitor import GroupTimeMonitor
from retina_sim.group_time_storage import save_bundle
from retina_sim.background_checkpoint import MemoryGuard
from tools.validate_ac_vs_native_support import load
from tools.validate_ac_vs_equilibrium_start import initial_match
from tools.validate_ac_vs_timescale import calendar
from tools.validate_ac_vs import write

OUT=ROOT/'reports/ac_vs_rhs_readouts_v1'
LEASE=ROOT/'reports/ac_vs_redesign_v1/ACTIVE_WORKER.json'
CODE=['retina_sim/ac_vs_rhs_readouts.py','tools/validate_ac_vs_rhs_readouts.py','retina_sim/bc_ac_fullchain_candidate.py']
def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare():
    if (OUT/'PRE_REGISTERED.json').exists():raise ContractError('Fresh comparison registration required')
    a,seed,parent=load(ROOT/'reports/ac_vs_native_support_v1/prepared_v1')
    protocol=deepcopy(parent['protocol']);protocol['stop_seconds']=protocol['checkpoint_seconds']=seed['time_seconds']+.0101
    frames=range(math.floor(seed['time_seconds']*150),math.floor(np.nextafter(protocol['stop_seconds'],-np.inf)*150)+1)
    protocol['input_frame_indices']=list(frames)
    changes=int(np.count_nonzero(seed['native_inputs'][750]!=seed['native_inputs'][748]))
    registration=dict(schema='ac-vs-rhs-readout-short-pair-1',start=seed['time_seconds'],duration_seconds=.0101,
        protocol=protocol,source_seed_sha256=canonical_hash(seed),artifact=a.artifact_sha256,backend=contract(),
        source_pins=parent['source_pins'],code_pins={n:sha(ROOT/n) for n in CODE},
        input='same frozen15-core-cone +5% step at global5.000s; existing native td and rod R*/s tables',changed_cones_at_step=changes,
        quantities=['every snapshot states/RHS/ports/pure transitions/unit/time/side/generation/aux/draws',
            'complete final causal state/history/RNG/clocks/delay/readout/native controller'],
        tolerances='exact FP64 bytes and exact metadata; no relaxed numerical tolerance',
        expected_identity_difference='compiled backend and arm identity only; compare final payload excluding controller.identity and timing metrics',
        repetitions='one fresh baseline, one fresh candidate; no old reference trace reused',
        resources=dict(single_worker_wall_seconds=120,process_RSS_limit_bytes=17179869184,available_RAM_floor_bytes=4294967296,
            rationale='10.1ms comparison ~4-6s hot per prior1.1s measurements plus setup/hash/storage; finite120s safety headroom, no cumulative role budget'),
        long_queue_paused=True,stop_policy='preserve paused long-queue REQUEST_STOP; authorized short comparison honors root REQUEST_STOP and its own report REQUEST_STOP',
        parameters_fixed=True,fit_steps=0,scope='execution redundancy only; not time-constant fitting or new physiological qualification')
    if changes!=15:raise ContractError('Frozen short comparison step identity differs')
    write(OUT/'PRE_REGISTERED.json',registration);print('SHORT_COMPARISON_FROZEN')


class ShortWorker:
    def __init__(self,directory,limits):
        self.directory=directory;self.limits=limits;self.start=perf_counter()
        self.owner=dict(pid=os.getpid(),output=str(directory),scope='authorized short RHS comparison; paused full fit remains stopped')
        self.memory=MemoryGuard(limits['process_RSS_limit_bytes'],limits['available_RAM_floor_bytes'])
    def __enter__(self):
        if not self.memory.check():raise ContractError('Measured comparison RAM preflight failed')
        with LEASE.open('x',encoding='utf8') as f:json.dump(self.owner,f)
        return self
    def check(self):
        if (ROOT/'REQUEST_STOP').exists() or (OUT/'REQUEST_STOP').exists():raise ContractError('Explicit short-comparison STOP')
        if json.loads(LEASE.read_text(encoding='utf8'))!=self.owner:raise ContractError('Comparison worker ownership changed')
        if not self.memory.check() or perf_counter()-self.start>self.limits['single_worker_wall_seconds']:raise ContractError('Short comparison resource guard')
    def __exit__(self,*exc):
        write(self.directory/'resource_receipt.json',dict(self.owner,elapsed_seconds=perf_counter()-self.start,
            peak_RSS_bytes=self.memory.peak_rss_bytes,released=True,limits=self.limits))
        if LEASE.exists() and json.loads(LEASE.read_text(encoding='utf8'))==self.owner:LEASE.unlink()


def causal(value):
    if isinstance(value,dict):return {k:causal(v) for k,v in value.items() if k!='metrics'}
    if isinstance(value,list):return [causal(v) for v in value]
    if isinstance(value,tuple):return tuple(causal(v) for v in value)
    return value


def run_arm(kind,reg):
    folder=OUT/kind;folder.mkdir(exist_ok=False);traces={};stamps=[];meta=[]
    with ShortWorker(folder,reg['resources']) as worker:
        a,seed,_=load(ROOT/'reports/ac_vs_native_support_v1/prepared_v1')
        if a.artifact_sha256!=reg['artifact'] or canonical_hash(seed)!=reg['source_seed_sha256']:raise ContractError('Frozen short comparison source changed')
        protocol=deepcopy(reg['protocol']);seed=deepcopy(seed)
        for category in ['native_inputs','rod_inputs']:seed[category]={k:seed[category][k] for k in protocol['input_frame_indices']}
        if kind=='candidate':protocol['rhs_execution_backend']=install(a,reg['backend'])
        begun=perf_counter();arm=GroupTimeArm(a,seed,protocol,mode='reference',guard=worker.check)
        monitor=GroupTimeMonitor(a,seed['states'],parameters=seed['inner_parameters'])
        def observe(t,side):
            s=arm.snapshot(t,side,queue_node=True)
            monitor.observe(t,side,s['states'],s['rhs'],transitions=s['transitions'],ports=s['ports'],aux=s['aux'],event_generation=s['event_generation'])
            if monitor.domain_failures:raise ContractError('Original physical/domain/history predicate failed')
            stamps.append((t,side,s['event_generation']))
            meta.append(canonical_hash({k:s[k] for k in ['port_units','aux','draws','event_witness_required','rhs_kind']}))
            for category in ['states','rhs','ports','transitions']:
                for key,value in s[category].items():traces.setdefault(category+'__'+':'.join(key),[]).append(np.array(value,copy=True))
        try:
            init,_=initial_match(arm,seed);observe(arm.start,'right');arm.initial_commit();observe(arm.start,'right');arm.ack(arm.start)
            for end,flags in calendar(protocol):
                if end<=arm.inner_time:continue
                begin=arm.inner_time;arm.ensure_native(end,'left');arm.advance_inner(begin,end)
                observe(end,'left' if flags['frame'] else 'right')
                if flags['rod'] or flags['bc']:arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],rod_sample_time=max(arm.start,end-.0001+1e-13) if flags['rod'] else None)
                observe(end,'right');arm.ack(end)
            payload=arm.checkpoint_payload(monitor=monitor.payload(),recording=dict(records=len(stamps)))
            hot=perf_counter()-begun;normalized=causal(payload);normalized['controller'].pop('identity')
            np.savez_compressed(folder/'traces.npz',time=np.array([t for t,_,_ in stamps]),side=np.array([s for _,s,_ in stamps]),generation=np.array([g for _,_,g in stamps]),metadata_hash=np.asarray(meta),**{k:np.stack(v) for k,v in traces.items()})
            save_bundle(folder/'final_checkpoint',kind='complete_model_checkpoint',identity=dict(common=canonical_hash(reg),arm=arm.identity,kind=kind),payload=payload,qualification=dict(default_adopted=False,fit_eligible=False))
            result=dict(kind=kind,hot_wall_seconds=hot,initial=init,final_causal_hash=canonical_hash(normalized),arm_identity=arm.identity,
                metrics=arm.metrics,raw_absolute_monitor=monitor.report(.0101),snapshots=len(stamps),parameters_unchanged=True)
            write(folder/'result.json',result);return result
        except Exception as exc:
            write(folder/'failure.json',dict(error=type(exc).__name__,reason=str(exc),model_time_s=arm.inner_time));raise
        finally:arm.close()


def run():
    reg=json.loads((OUT/'PRE_REGISTERED.json').read_text(encoding='utf8'))
    for name,digest in dict(reg['source_pins'],**reg['code_pins']).items():
        if sha(ROOT/name)!=digest:raise ContractError('Pre-registered source changed: '+name)
    before=run_arm('baseline',reg);after=run_arm('candidate',reg);rows=[]
    with np.load(OUT/'baseline/traces.npz',allow_pickle=False) as a,np.load(OUT/'candidate/traces.npz',allow_pickle=False) as b:
        if set(a.files)!=set(b.files):raise ContractError('Comparison field registry changed')
        for key in a.files:
            exact=a[key].dtype==b[key].dtype and a[key].shape==b[key].shape and a[key].tobytes()==b[key].tobytes()
            rows.append(dict(field=key,exact=exact,max_abs_difference=float(np.max(abs(a[key]-b[key]),initial=0.)) if a[key].dtype.kind in 'fiu' else None))
    passed=all(r['exact'] for r in rows) and before['final_causal_hash']==after['final_causal_hash']
    result=dict(status='PASS_SHORT_EXACT_COMPARISON' if passed else 'FAIL_CANDIDATE_NOT_APPROVED',passed=passed,fields=rows,
        baseline_hot_wall_seconds=before['hot_wall_seconds'],candidate_hot_wall_seconds=after['hot_wall_seconds'],
        observed_pair_speedup=before['hot_wall_seconds']/after['hot_wall_seconds'],
        limitation='one short pair; no general speed claim, no long queue or fit',final_causal_exact=before['final_causal_hash']==after['final_causal_hash'],
        long_queue_stop_retained=(ROOT/'reports/ac_vs_redesign_v1/REQUEST_STOP').exists())
    write(OUT/'FINAL_RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='fields'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run']);args=p.parse_args()
    prepare() if args.mode=='prepare' else run()
