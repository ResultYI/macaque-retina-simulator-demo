"""Isolated original assay; native-fault traces and complete inner checkpoints."""
from pathlib import Path
import argparse,faulthandler,hashlib,inspect,json,sys,time,traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/retinotopic_rf_revision/SPATIAL_POOLING_CANDIDATE_V1'


def main(args):
    import numpy as np
    from scipy import sparse
    from scipy.integrate import solve_ivp
    from copy import deepcopy
    from retina_sim.contracts import canonical_hash
    from retina_sim.background_checkpoint import MemoryGuard
    from retina_sim.group_time_storage import load_bundle,save_bundle
    from tools import run_spatial_pooling_pair_v1 as old
    from tools.durable_research_runtime_v1 import DurableRun
    run_dir=args.output.resolve()
    if not run_dir.is_relative_to(OUT):raise ValueError('Independent local candidate output required')
    run_dir.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();memory=MemoryGuard(24*1024**3,8*1024**3);journal=DurableRun(run_dir/'runtime','isolated_spatial_pooling_assay')
    def guard():
        journal.heartbeat(phase='loading_or_initializing')
        if not memory.check():raise MemoryError(str((memory.stop_reason,memory.last)))
        if time.perf_counter()-started>900:raise TimeoutError('Isolated worker900s guard')
    try:
        source=old.read(ROOT/'reports/external_video_two_positions_4s_20261029_v1/PREPARATION.json')['bindings'][args.position]
        assembly,parent,_=old.load_background(Path(source['prepared']),Path(source['background']));guard()
        with np.load(Path(source['background'])/'BACKGROUND.npz',allow_pickle=False) as z:release=z['cone_release'].copy()
        original_factory=assembly.runtime_factories['inner'];destination=OUT/'function_v2'/args.position/args.arm
        if (destination/'initial_seed/manifest.json').exists():
            seed=load_bundle(destination/'initial_seed',expected_kind='source_construction_seed')['payload']
            model=original_factory(assembly,seed['inner_parameters']);identities={}
            if args.arm=='candidate':
                for r in old.read(OUT/'STRUCTURE_RESULT.json')['routes']:
                    if r['location']!=args.position:continue
                    if not r['dynamic_local_closure']:raise ValueError('Candidate exterior not closed')
                    p=OUT/'structure'/args.position/r['route']/'group_candidate_local.npz'
                    model.mean[r['route']]=sparse.load_npz(p);identities[r['route']]=old.file_sha(p)
            model.identity=canonical_hash(dict(parent_inner=model.identity,spatial_candidate='RETINOTOPIC_GEOMETRY_ANCHORED_POOLING_V1',location=args.position,arm=args.arm,matrices=identities,physiology_parameters_unchanged=True))
            if model.identity!=old.read(destination/'INITIALIZATION.json')['model_identity']:raise ValueError('Existing own initializer identity differs')
        else:
            init_start=time.perf_counter();model,seed,destination=old.initialize(assembly,parent,original_factory,release,args.position,args.arm,guard)
            old.write(destination/'INITIALIZATION_RESOURCE.json',dict(wall_seconds=time.perf_counter()-init_start,peak_rss_bytes=memory.peak_rss_bytes,recovery_worker=True))
        if canonical_hash(seed['inner_parameters'])!=canonical_hash(parent['inner_parameters']):raise ValueError('Frozen parameters differ')
        if args.repro:
            # Fixed recorded endpoint, no new experimental stimulus or long rollout.
            endpoint=OUT/'function_v2/near/legacy/standard/STATE_END.npz'
            with np.load(endpoint,allow_pickle=False) as z:y=z['inner_flat'].copy()
            nfev=0;begin=time.perf_counter()
            def rhs(t,v):return model.pack(model.rhs(t,model.unpack(v,copy=False)))
            reference=rhs(.9,y)
            for i in range(400):
                guard()
                if time.perf_counter()-begin>40:raise TimeoutError('Minimal native-call reproduction40s bound')
                np.testing.assert_array_equal(rhs(.9,y),reference)
                if i<20:
                    sol=solve_ivp(rhs,(.9,.9001),y,method='DOP853',rtol=1e-10,atol=1e-13,max_step=.000025,first_step=.000025,t_eval=[.9001])
                    if not sol.success or not np.isfinite(sol.y).all():raise ValueError('Minimal native call failed')
                    nfev+=sol.nfev
            payload=dict(inner_flat=y,spike_states=deepcopy(seed['spike_states']),next_event_index=9000,clock_origin=0.,model_identity=model.identity)
            save_bundle(run_dir/'checkpoint_roundtrip',kind='validation_evidence',identity=dict(model=model.identity),payload=payload,
                qualification=dict(default_adopted=False,fit_eligible=False,FIT_READY=False,background_qualified=False))
            restored=load_bundle(run_dir/'checkpoint_roundtrip',expected_kind='validation_evidence')['payload']
            if canonical_hash(restored)!=canonical_hash(payload):raise ValueError('Exact state/history/PCG roundtrip differs')
            old.write(run_dir/'RESULT.json',dict(status='PASS_NOT_REPRODUCED',RHS_repetitions=400,tiny_IVP_repetitions=20,nfev=nfev,
                wall_seconds=time.perf_counter()-started,peak_rss_bytes=memory.peak_rss_bytes,checkpoint_roundtrip_exact=True,
                root_cause_confirmed=False,no_scientific_experiment_added=True))
        else:
            original=inspect.getsource(old.simulate)
            modified=original.replace('out=destination/precision;out.mkdir();started=time.perf_counter();nfev=0','out=run_dir;started=time.perf_counter();nfev=0')
            needle="if (index+1)%3000==0:np.savez_compressed(out/f'PARTIAL_{(index+1)//10:04d}.npz',**packet())"
            replacement="""curve_path=out/f'PARTIAL_{(index+1)//10:04d}.npz'
            np.savez_compressed(curve_path,**packet())
            checkpoint_path=out/f'CHECKPOINT_{(index+1)//10:04d}'
            save_bundle(checkpoint_path,kind='validation_evidence',identity=dict(model=model.identity,precision=precision,initial_seed=canonical_hash(seed)),
                payload=dict(inner_flat=model.pack(state),spike_states=deepcopy(auxiliary),next_event_index=index+1,clock_origin=bc.origin,
                    model_identity=model.identity,nfev=nfev,curve_prefix=curve_path.name,curve_sha256=file_sha(curve_path)),
                qualification=dict(default_adopted=False,fit_eligible=False,FIT_READY=False,background_qualified=False,scope='Complete inner held-input assay checkpoint only, not native/global'))"""
            if original.count(needle)!=1 or modified==original:raise ValueError('Frozen loop patch locations changed')
            modified=modified.replace(needle,replacement)
            namespace=dict(old.__dict__,run_dir=run_dir,save_bundle=save_bundle,file_sha=old.file_sha)
            exec(compile(modified,'<original_assay_with_checkpoint_IO>','exec'),namespace)
            old.write(run_dir/'EXECUTION_ADAPTER.json',dict(source_sha256=old.file_sha(ROOT/'tools/run_spatial_pooling_pair_v1.py'),
                original_simulate_sha256=hashlib.sha256(original.encode()).hexdigest(),patched_simulate_sha256=hashlib.sha256(modified.encode()).hexdigest(),
                changes=['output directory','complete state/history/PCG and curve-prefix snapshot each100ms'],
                equations_clocks_tolerances_and_event_loop_unchanged=True,no_native_bug_fix_claim=True))
            namespace['simulate'](assembly,model,seed,release,destination,args.position,args.arm,args.precision,memory,journal)
        journal.finish('PASS',phase='complete')
    except BaseException as e:
        old.write(run_dir/'PYTHON_FAILURE.json',dict(error=str(e),traceback=traceback.format_exc()))
        journal.finish('FAILED',phase='python_exception');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--position',choices=['near','far'],required=True)
    p.add_argument('--arm',choices=['legacy','candidate'],required=True);p.add_argument('--precision',choices=['standard','tight'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repro',action='store_true');args=p.parse_args()
    log_path=args.output.with_name(args.output.name+'_native_fault.log');log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('x',encoding='utf8') as log:
        faulthandler.enable(file=log,all_threads=True)
        main(args)
