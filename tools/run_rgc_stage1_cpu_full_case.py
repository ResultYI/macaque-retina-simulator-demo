"""Fresh-process compact capture using unchanged original science code."""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def blas_pools():
    """Read exported thread counters only; never set runtime thread counts."""
    import numpy,scipy
    rows=[]
    for package in [numpy,scipy]:
        directory=Path(package.__file__).parent.parent/(package.__name__+'.libs')
        for path in directory.glob('*openblas*.dll'):
            library=ctypes.WinDLL(str(path));row=dict(library=str(path),threads=None)
            for name in ['scipy_openblas_get_num_threads64_','scipy_openblas_get_num_threads','openblas_get_num_threads64_','openblas_get_num_threads']:
                try:fn=getattr(library,name)
                except AttributeError:continue
                fn.argtypes=[];fn.restype=ctypes.c_int;row.update(threads=fn(),symbol=name);break
            rows.append(row)
    return rows

def checkpoint_identity(directory,spec,case):
    """Validate real payload and original recording clock, without relabelling."""
    from retina_sim.group_time_storage import load_bundle
    from retina_sim.contracts import canonical_hash,ContractError
    import numpy as np
    directory=Path(directory).resolve();b=load_bundle(directory,expected_kind='complete_model_checkpoint');p=b['payload']
    identity=canonical_hash(dict(spec=spec,case=case));r=p['recording'];j=int(r['recorded'])-1
    if b['identity']!=dict(cache_identity=identity) or not r['initial_match']['passed']:
        raise ContractError('Original complete capture checkpoint identity/initial state differs')
    drive=load_bundle(ROOT/case['drive_bundle'],expected_kind='validation_evidence')['payload']
    if abs(p['controller']['time_seconds']-drive['model_origin_s']-j*.0001)>1e-12 or j<0 or j>case['samples_100us']:
        raise ContractError('Original checkpoint source clock differs')
    if r['bins']!=min(j//10+1,case['samples_100us']//10):raise ContractError('Original checkpoint RGC bin coverage differs')
    for pop,selection in spec['recording'].items():
        for field in ['V','gE','gI']:
            a=r['arrays'][pop+':'+field]
            if a.shape!=(j+1,len(selection['rows'])) or a.dtype!=np.float64 or not np.isfinite(a).all():
                raise ContractError('Original partial recording field/shape/domain differs')
    return dict(directory=str(directory),source_sample=j,source_seconds=j*.0001,cache_identity=identity,
        manifest_sha256=sha(directory/'manifest.json'),payload_hash_verified_by_load_bundle=True,
        source='existing complete originalCPU checkpoint, never short-test or GPU',arm_source_protocol_validation='performed by unchanged GroupTimeArm.from_checkpoint at actual capture')

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);args=p.parse_args();job=read(args.job)
    directory=Path(job['directory']).resolve();private=Path(job['private_root']).resolve();phase=Path(job['phase']).resolve()
    if not directory.is_relative_to(private) or not private.is_relative_to(phase.parents[1]/'workers'):
        raise RuntimeError('Explicit private full-case workspace required')
    from tools.validate_group_scc_radau import worker_handshake
    worker_handshake(directory,job,'case')
    from retina_sim.rgc_cpu_case_isolation import ScopedStageWorker,resources,enforce_private_writes
    from retina_sim.contracts import ContractError,canonical_hash
    from retina_sim import rgc_stage1_cache as cache
    from tools.fit_rgc_stage1 import verify
    started=time.perf_counter();cpu_started=time.process_time();blocked=enforce_private_writes(private)
    original_guard=cache.StageWorker;guards=[]
    try:
        spec=verify();case=next(c for c in spec['cases'] if c['case_id']==job['case_id'])
        identity=canonical_hash(dict(spec=spec,case=case))
        if identity!=job['computation_identity']:raise ContractError('Unmodified full CPU spec+case differs')
        pools=blas_pools()
        if any(row['threads'] not in (None,1) for row in pools):raise ContractError('Measured BLAS pool oversubscription before capture')
        target=private/'cache'/case['case_id'];target.mkdir(parents=True,exist_ok=True)
        pointer=target/'latest_checkpoint.json';resume=None
        if pointer.exists():
            point=read(pointer);resume=checkpoint_identity(target/point['directory'],spec,case)
        elif job.get('resume_source'):
            resume=checkpoint_identity(job['resume_source'],spec,case)
            # The real existing checkpoint stays read-only at its real path.
            cache.write(pointer,dict(directory=resume['directory'],source_sample=resume['source_sample'],cache_identity=identity,
                external_readonly_original_checkpoint=True,source_manifest_sha256=resume['manifest_sha256']))
        cache.write(directory/'CAPTURE_START.json',dict(case=case,computation_identity=identity,orchestration_identity=job['orchestration_identity'],
            source_ROOT=str(ROOT),capture_implementation_sha256=sha(ROOT/'retina_sim/rgc_stage1_cache.py'),
            original_capture_function=True,compact_recording=True,restoring=resume,thread_environment={k:os.environ.get(k) for k in job['thread_policy']},
            actual_BLAS_pools=pools,resources=resources(),pid=os.getpid()))
        class FullGuard(ScopedStageWorker):
            def __init__(self,root,out,passed_phase,wall_limit):
                super().__init__(root,out,phase,wall_limit,job)
                self.lease=directory/'ACTIVE_WORKER.json'
                self.owner.update(scope='authorized original compactCPU capture under two-slot coordinator')
                guards.append(self)
            def check(self):
                super().check()
                if (directory/'REQUEST_STOP').exists():raise ContractError('Owned full case STOP')
        # Only lease ownership is rebound. The pinned capture/event/solver code,
        # full scientific spec, source seed, inputs and model execution are reused.
        cache.StageWorker=FullGuard
        before=resources();manifest=cache.capture(ROOT,private,spec,case,resume=pointer.exists());after=resources()
        result=dict(status='COMPLETE_ORIGINAL_COMPACT_CPU_CAPTURE',case_id=case['case_id'],cache_identity=identity,
            cache_directory=str(target),orchestration_identity=job['orchestration_identity'],manifest_sha256=sha(target/'manifest.json'),
            wrapper_wall_seconds=time.perf_counter()-started,wrapper_CPU_seconds=time.process_time()-cpu_started,
            resources_before=before,resources_after=after,peak_RSS_bytes=max([g.memory.peak_rss_bytes for g in guards],default=0),
            metrics=manifest['metrics'],restored_from=resume,shared_write_attempts=blocked,actual_BLAS_pools=pools,
            original_scientific_spec_unchanged=True,GPU_used=False,FIT_READY=False)
        cache.write(directory/'RESULT.json',result);print(json.dumps({k:result[k] for k in ['status','case_id','cache_identity','cache_directory','wrapper_wall_seconds','wrapper_CPU_seconds','peak_RSS_bytes']}),flush=True)
    except BaseException as e:
        cache.write(directory/'FAILURE.json',dict(error=type(e).__name__,reason=str(e),elapsed_seconds=time.perf_counter()-started,blocked_writes=blocked));raise
    finally:cache.StageWorker=original_guard

if __name__=='__main__':main()
