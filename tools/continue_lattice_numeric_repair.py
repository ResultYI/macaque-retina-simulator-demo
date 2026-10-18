"""Bounded, exact-state continuation of the unchanged source spring model."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,sys,time,shutil
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
INITIALIZATION=ROOT/'data/atlas/isetbio_macaque_cones_coveragefix_v1_20261015/initialization.mat'
INITIALIZATION_SHA256='20046d18fde2c931b9519d7cc8d66c6e0b6ce1929ca42c05dccd85e32e433f59'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def save(path,value):
    tmp=path.with_suffix(path.suffix+'.writing')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)

def load_state(source):
    if sha(INITIALIZATION)!=INITIALIZATION_SHA256:raise ValueError('Frozen initialization/lookup identity changed')
    if not (source/'coordinates.npz').exists():
        with np.load(source/'checkpoint.npz',allow_pickle=False) as z:
            if str(z['initialization_sha256'])!=sha(INITIALIZATION):raise ValueError('Checkpoint initialization changed')
            return {'current_raw_um':z['current_raw_um'].copy(),'best_raw_um':z['best_raw_um'].copy(),
                'last_step_before_raw_um':z['last_step_before_raw_um'].copy() if 'last_step_before_raw_um' in z else z['current_raw_um'].copy(),
                'generation':json.loads(str(z['generation_json']))}
    with np.load(source/'coordinates.npz',allow_pickle=False) as z:
        current=z['final_raw_um'].copy();best=z['selected_raw_um'].copy()
        before=z['last_step_before_raw_um'].copy() if 'last_step_before_raw_um' in z else current.copy()
        if not np.array_equal(z['initial_raw_um'],loadmat(INITIALIZATION,squeeze_me=True)['initial_raw_um']):
            raise ValueError('Initial point identities or initialization changed')
    result=json.loads((source/'generation.json').read_bytes())
    return {'current_raw_um':current,'best_raw_um':best,'last_step_before_raw_um':before,'generation':result}

def worker(args):
    from retina_sim.lattice_relaxation import SourceSpringField,relax
    source=(ROOT/args.source_directory).resolve();out=(ROOT/args.output_directory).resolve()
    v=loadmat(INITIALIZATION,squeeze_me=True);state=load_state(source)
    field=SourceSpringField(v['lookup_xy_um'],v['lookup_spacing_um'],v['radius_um'])
    previous_motion=None;previous_added=np.empty(0,dtype=np.int64);last_before=state['last_step_before_raw_um'].copy()
    def diagnose_step(before,after,old_springs,new_springs,force,desired,row):
        nonlocal previous_motion,previous_added,last_before
        last_before=before.copy()
        n=len(before)
        # Qhull indices are int32: row*N+column exceeds that range here.
        old_code=old_springs[:,0].astype(np.int64)*n+old_springs[:,1]
        new_code=new_springs[:,0].astype(np.int64)*n+new_springs[:,1]
        removed=np.setdiff1d(old_code,new_code,assume_unique=True)
        added=np.setdiff1d(new_code,old_code,assume_unique=True)
        old_index=np.searchsorted(old_code,removed)
        old_length=np.linalg.norm(before[old_springs[old_index,0]]-before[old_springs[old_index,1]],axis=1)
        removed_repulsion=np.maximum(1.1*desired[old_index]-old_length,0.)
        motion=after-before;at=row['maximum_movement_source_row']
        record={'edges_removed':len(removed),'edges_added':len(added),
            'edges_immediately_reversed':len(np.intersect1d(removed,previous_added,assume_unique=True)),
            'removed_edges_with_repulsion':int(np.sum(removed_repulsion>0)),
            'largest_removed_repulsion_um':float(removed_repulsion.max()) if len(removed_repulsion) else 0.}
        if previous_motion is not None:
            dot=np.sum(motion*previous_motion,axis=1)
            denominator=np.linalg.norm(motion[at])*np.linalg.norm(previous_motion[at])
            record.update(max_mover_previous_direction_cosine=float(dot[at]/denominator) if denominator else None,
                displacement_vector_reversal_count=int(np.sum(dot<0)))
        row['topology_diagnostic']=record;previous_motion=motion;previous_added=added
    def checkpoint(current,best,result):
        tmp=out/'checkpoint.writing.npz'
        # One atomic file binds current, best and the full history. An interrupted
        # JSON/NPZ pair can never masquerade as a resumable checkpoint.
        np.savez_compressed(tmp,current_raw_um=current,best_raw_um=best,
            last_step_before_raw_um=last_before,
            generation_json=np.array(json.dumps(result,allow_nan=False)),
            initialization_sha256=np.array(sha(INITIALIZATION)))
        os.replace(tmp,out/'checkpoint.npz')
        save(out/'progress.json',{k:v for k,v in result.items() if k!='trace'})
        status_path=out.parent/'current_status.json'
        status=json.loads(status_path.read_bytes()) if status_path.exists() else {}
        row=result['trace'][-1]
        reached=bool(row['quality_width_stop'] or row.get('source_motion_stop',False))
        status.update(updated_utc=utc(),source_OR_restored=True,
            last_durable_checkpoint_iteration=result['accepted_iterations'],checkpoint_quality=result['final_quality'],
            source_motion_um=row.get('source_interior_p30_motion_um'),maximum_true_displacement_um=row['actual_max_movement_um'],
            worker_pid=os.getpid(),segment=out.relative_to(ROOT).as_posix(),stopped=reached,
            execution_state='SOURCE_STOP_REACHED' if reached else 'RUNNING' if result['stop_reason']=='checkpoint' else 'SEGMENT_FINISHED_REVIEW_REQUIRED',
            message='Source OR stopping active; current durable checkpoint; maximum true motion is diagnostic only.',physiological_validation=False)
        save(status_path,status)
    def progress(row):
        keys=('iteration','q_0_8','q_min','source_interior_p30_motion_um','source_motion_stop','actual_max_movement_um','actual_p30_movement_um',
            'accepted_step','backtracks','width_condition','elapsed_seconds')
        print(json.dumps({k:row[k] for k in keys}),flush=True)
    best,current,result=relax(v['initial_raw_um'],field,resume=state,
        max_iterations=args.additional_iterations,max_wall_seconds=args.budget_seconds,
        on_checkpoint=checkpoint,on_progress=progress,on_step=diagnose_step,allow_legacy_stop_contract_migration=True)
    np.savez_compressed(out/'coordinates.npz',selected_raw_um=best,final_raw_um=current,
        last_step_before_raw_um=last_before,initial_raw_um=v['initial_raw_um'])
    save(out/'generation.json',result)
    print('SEGMENT_FINISHED',json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)

def launch(args):
    source=(ROOT/args.source_directory).resolve();out=(ROOT/args.output_directory).resolve()
    if out.exists():assert not any(out.iterdir()),'Never overwrite a previous run'
    else:out.mkdir(parents=True)
    state=load_state(source)
    assert state['generation']['stop_reason'] in ('iteration_budget','wall_budget','checkpoint'),'Source is already terminal; diagnose first'
    protected={str(p.relative_to(ROOT)):sha(p) for folder in [ROOT/'data/atlas',ROOT/'configurations']
        for p in folder.rglob('*') if p.is_file()}
    save(out/'protected_existing_assets.json',protected)
    provenance={'authorization':'Sentinel_8ae5dbf4316c819183a2ae5b9bf2b682_2026-10-16T01:33UTC',
        'source_directory':source.relative_to(ROOT).as_posix(),
        'source_coordinates_sha256':sha(source/('coordinates.npz' if (source/'coordinates.npz').exists() else 'checkpoint.npz')),
        'source_generation_sha256':sha(source/('generation.json' if (source/'generation.json').exists() else 'checkpoint.npz')),
        'initialization_sha256':sha(INITIALIZATION),'resume_iteration':state['generation']['accepted_iterations'],
        'resume_state':'final_raw_um, never best_raw_um; full prior quality width history restored',
        'implementation_sha256':sha(ROOT/'retina_sim/lattice_relaxation.py'),
        'model_rules_changed':False,'stopping_contract_restored':'source OR; literal nominal-dt internal p30; maximum true motion diagnostic only',
        'density_domain_seed_point_count_LMS_unchanged':True,
        'physiological_validation':False,'adopted':False}
    save(out/'provenance.json',provenance)
    snapshots=out/'implementation_snapshot';snapshots.mkdir()
    for name in ('retina_sim/lattice_relaxation.py','tools/continue_lattice_numeric_repair.py'):
        shutil.copyfile(ROOT/name,snapshots/Path(name).name)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',
        '--source-directory',str(source.relative_to(ROOT)),'--output-directory',str(out.relative_to(ROOT)),
        '--additional-iterations',str(args.additional_iterations),'--budget-seconds',str(args.budget_seconds)]
    hard=args.budget_seconds+60
    run={'command':command,'start_utc':utc(),'hard_deadline_seconds':hard,
        'cooperative_generation_budget_seconds':args.budget_seconds,
        'additional_iteration_cap':args.additional_iterations,'automatic_retry':False,'status':'STARTING'}
    save(out/'run.json',run);start=time.monotonic()
    with (out/'generation.log').open('xb') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        run.update(pid=p.pid,status='RUNNING');save(out/'run.json',run)
        try:run['returncode']=p.wait(timeout=hard);run['status']='PROCESS_EXITED'
        except subprocess.TimeoutExpired:
            killed=subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,text=True)
            run.update(status='HARD_DEADLINE_TERMINATED',own_process_tree_kill=killed.stdout)
            p.wait(timeout=30);run['returncode']=p.returncode
    run.update(end_utc=utc(),wall_seconds=time.monotonic()-start,
        protected_existing_assets_unchanged=all(sha(ROOT/k)==v for k,v in protected.items()))
    save(out/'run.json',run);print(json.dumps(run),flush=True)
    if run['returncode']:raise SystemExit(run['returncode'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true')
    p.add_argument('--source-directory',required=True);p.add_argument('--output-directory',required=True)
    p.add_argument('--additional-iterations',type=int,default=600);p.add_argument('--budget-seconds',type=float,default=840.)
    args=p.parse_args()
    assert all((ROOT/v).resolve().is_relative_to(ROOT) for v in (args.source_directory,args.output_directory))
    assert 1<=args.additional_iterations<=1000 and 30<=args.budget_seconds<=1800
    worker(args) if args.worker else launch(args)
