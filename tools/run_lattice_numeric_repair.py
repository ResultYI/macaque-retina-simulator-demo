"""One bounded full-domain numerical-repair validation; never adopts a model."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,sys,time
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'data/atlas/isetbio_macaque_cones_numericfix_v3_20261015'
DIAG=ROOT/'reports/lattice_generator_repair_20261015'
OLD=ROOT/'data/atlas/isetbio_macaque_cones_v1'
INITIALIZATION=DIAG/'source_force_diagnostic.mat'
PARITY=DIAG/'force_parity.json'
AUTHORIZATION=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def save(path,value):
    tmp=path.with_suffix(path.suffix+'.writing');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)
def snapshot(best,iteration,quality,trace):
    tmp=OUT/'checkpoint.writing.npz'
    np.savez_compressed(tmp,best_raw_um=best,selected_iteration=iteration)
    os.replace(tmp,OUT/'checkpoint.npz')
    save(OUT/'checkpoint.json',{'selected_iteration':iteration,'quality':quality,'last_iteration':trace[-1]})

def worker():
    from retina_sim.lattice_relaxation import SourceSpringField,relax
    from retina_sim.isetbio_geometry import cone_apertures
    v=loadmat(INITIALIZATION,squeeze_me=True)
    field=SourceSpringField(v['lookup_xy_um'],v['lookup_spacing_um'],v['radius_um'])
    def progress(row):print(json.dumps({k:row[k] for k in ('iteration','q_0_8','q_min','actual_max_movement_um','accepted_step','backtracks','elapsed_seconds')}),flush=True)
    best,final,result=relax(v['initial_raw_um'],field,max_iterations=300,max_wall_seconds=840.,on_snapshot=snapshot,on_progress=progress)
    best_xy=-best/1000;D=cone_apertures(best_xy)
    np.savez_compressed(OUT/'coordinates.npz',selected_raw_um=best,final_raw_um=final,initial_raw_um=v['initial_raw_um'])
    np.savez_compressed(OUT/'positions.npz',positions_mm=best_xy,source_keys=np.array(['iset_lattice_row:'+str(i) for i in range(len(best))]),aperture_diameters_um=D)
    save(OUT/'generation.json',result)
    old=json.loads((OLD/'manifest.json').read_bytes())
    initialization=json.loads(str(v['initialization_json'])) if 'initialization_json' in v else None
    old.update(candidate_version=OUT.name+'_not_adopted',count=len(best),
        payload={'path':(OUT/'positions.npz').relative_to(ROOT).as_posix(),'sha256':sha(OUT/'positions.npz')},
        generator='source_retinalattice_spring_model_with_traced_numerical_repairs',
        source_commit_preserved=old['isetbio_commit'],
        numerical_implementation={'path':'retina_sim/lattice_relaxation.py','sha256':sha(ROOT/'retina_sim/lattice_relaxation.py')},
        exact_initialization={'path':INITIALIZATION.relative_to(ROOT).as_posix(),'sha256':sha(INITIALIZATION),'coverage_metadata':initialization},
        generation_record={'path':(OUT/'generation.json').relative_to(ROOT).as_posix(),'sha256':sha(OUT/'generation.json')},
        radius_um=field.radius,ellipse_axes=[1.,1.2247],
        generation_domain='Same fixed source ellipse; initialization coverage and project numerical repair are explicitly versioned',
        coordinate_precision='float64; selected raw coordinates and quality evaluated together',
        authorization_reference=AUTHORIZATION,
        actual_density_request={'path':'data/atlas/isetbio_macaque_cones_candidate300_20261015/request.mat',
            'sha256':sha(ROOT/'data/atlas/isetbio_macaque_cones_candidate300_20261015/request.mat'),
            'target_grid_unchanged':True},
        candidate_status={'adopted':False,'typed_graph_built':False,'neural_run':False,'physiological_validation':False})
    # These source-generation fields describe the historical MATLAB asset,
    # not this new numerical implementation; preserve them under provenance.
    old['historical_source_generation']={k:old.pop(k) for k in ('generator_output_sha256','generator_request_sha256','exporter_sha256','matlab')}
    old['recipe']=dict(old['recipe'],max_iterations=300,numerical_repairs=result['algorithm'])
    audit_path=INITIALIZATION.parent/'initialization_run.json'
    if audit_path.exists():old['initialization_run_record']={'path':audit_path.relative_to(ROOT).as_posix(),'sha256':sha(audit_path)}
    save(OUT/'manifest.json',old)
    print('GENERATION_FINISHED',json.dumps({k:result[k] for k in ('stop_reason','accepted_iterations','selected_iteration','selected_quality','backtracks_total','point_count')}),flush=True)

def launch():
    if OUT.exists():assert not any(OUT.iterdir()),'Existing run/asset must never be overwritten'
    else:OUT.mkdir()
    parity=json.loads(PARITY.read_bytes());assert parity['force_scaled_max_difference']<1e-11
    initialization_data=loadmat(INITIALIZATION,squeeze_me=True)
    initialization=json.loads(str(initialization_data['initialization_json'])) if 'initialization_json' in initialization_data else None
    from tools.generate_isetbio_geometry import VENDOR
    # These exact bytes were already matched to the pinned Git blobs during
    # the frozen source export. Reading their frozen hashes also works in the
    # workspace sandbox, where git -C into the external vendor may be denied.
    methods=json.loads((OLD/'manifest.json').read_bytes())['source_methods']
    assert all(sha(VENDOR/k)==v['sha256'] for k,v in methods.items()),'Pinned source bytes changed'
    previous=sorted((ROOT/'data/atlas').glob('isetbio_macaque_cones*'))
    protected={str(p.relative_to(ROOT)):sha(p) for folder in previous+[ROOT/'configurations'] if folder!=OUT for p in folder.rglob('*') if p.is_file()}
    save(OUT/'protected_existing_assets.json',protected)
    save(OUT/'provenance.json',{'source_methods':methods,'force_parity':parity,
        'initialization_path':INITIALIZATION.relative_to(ROOT).as_posix(),'initialization_sha256':sha(INITIALIZATION),
        'initialization_coverage':initialization,'authorization_reference':AUTHORIZATION,
        'implementation_sha256':sha(ROOT/'retina_sim/lattice_relaxation.py'),
        'unchanged_model_rules':['pinned source sampler, seed and legacy sampled cells; approved omitted-candidate continuation is explicit','macaque density grid; source64-sample lookup rebuilt from covered initialization','gain1.1 normalized repulsive springs','same ellipse radius/axes and hex phase','RF polarity and .223mm/deg','LMS rule not changed'],
        'numerical_repairs':['invalidate stale Delaunay/spacing caches on current coordinates','reject interior Euler face inversion and machine-coincident swept point encounters; hull membership changes are legal','exclude only roundoff-zero-area Qhull faces while requiring every point remains represented','normalize implicit-surface Newton projection','double-precision coordinate/quality pairing','maximum nominal-step movement cannot pass just because adaptive dt became tiny'],
        'new_biological_minimum_spacing':False,'point_deletion_or_density_change':False,'adopted':False})
    import shutil
    snapshots=OUT/'implementation_snapshot';snapshots.mkdir()
    for name in ('retina_sim/lattice_relaxation.py','tools/run_lattice_numeric_repair.py','tools/export_lattice_initialization_covered.m'):
        shutil.copyfile(ROOT/name,snapshots/Path(name).name)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',
        '--output-directory',str(OUT.relative_to(ROOT)),'--initialization',str(INITIALIZATION.relative_to(ROOT)),
        '--parity-report',str(PARITY.relative_to(ROOT))]
    if AUTHORIZATION:command+=['--authorization',AUTHORIZATION]
    run={'command':command,'start_utc':utc(),'hard_deadline_seconds':900,'cooperative_generation_budget_seconds':840,
        'accepted_iteration_cap':300,'automatic_retry':False,'status':'STARTING'};save(OUT/'run.json',run)
    start=time.monotonic()
    with (OUT/'generation.log').open('xb') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        run.update(pid=p.pid,status='RUNNING');save(OUT/'run.json',run)
        try:run['returncode']=p.wait(timeout=900);run['status']='PROCESS_EXITED'
        except subprocess.TimeoutExpired:
            kill=subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,text=True)
            run.update(status='HARD_DEADLINE_TERMINATED',own_process_tree_kill=kill.stdout);p.wait(timeout=30);run['returncode']=p.returncode
    run.update(end_utc=utc(),wall_seconds=time.monotonic()-start,
        protected_existing_assets_unchanged=all(sha(ROOT/k)==v for k,v in protected.items()))
    save(OUT/'run.json',run);print(json.dumps(run),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true')
    p.add_argument('--output-directory',default=str(OUT.relative_to(ROOT)))
    p.add_argument('--initialization',default=str(INITIALIZATION.relative_to(ROOT)))
    p.add_argument('--parity-report',default=str(PARITY.relative_to(ROOT)))
    p.add_argument('--authorization');a=p.parse_args()
    OUT=(ROOT/a.output_directory).resolve();INITIALIZATION=(ROOT/a.initialization).resolve();PARITY=(ROOT/a.parity_report).resolve()
    assert all(path.is_relative_to(ROOT) for path in (OUT,INITIALIZATION,PARITY)),'All paths must stay in the project'
    AUTHORIZATION=a.authorization
    worker() if a.worker else launch()
