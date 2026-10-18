"""Freeze a source-stopped terminal geometry, preserving every earlier asset."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil,sys
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'data/atlas/isetbio_macaque_cones_coveragefix_v1_20261015'
INITIALIZATION=ROOT/'reports/lattice_coverage_repair_20261015/covered_initialization.mat'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def relative(p):return p.relative_to(ROOT).as_posix()

def freeze(segment,out,report):
    from retina_sim.lattice_relaxation import SourceSpringField,quality_state,source_motion_statistic,project_ellipse,ellipse_distance,STOPPING_CONTRACT
    from retina_sim.isetbio_geometry import cone_apertures,FixedISETConeMaster
    g=json.loads((segment/'generation.json').read_bytes());run=json.loads((segment/'run.json').read_bytes())
    assert run['returncode']==0 and run['protected_existing_assets_unchanged']
    assert g['stopping_contract']==STOPPING_CONTRACT
    assert g['stop_reason'] in ('original_quality_plus_width_condition','original_internal_p30_motion_condition')
    row=g['trace'][-1];assert row['quality_width_stop'] or row['source_motion_stop']
    with np.load(segment/'coordinates.npz',allow_pickle=False) as z:
        current=z['final_raw_um'].copy();best=z['selected_raw_um'].copy()
        initial=z['initial_raw_um'].copy();before=z['last_step_before_raw_um'].copy()
    v=loadmat(INITIALIZATION,squeeze_me=True)
    assert np.array_equal(initial,v['initial_raw_um']) and len(current)==121811
    field=SourceSpringField(v['lookup_xy_um'],v['lookup_spacing_um'],v['radius_um'])
    tri,springs=field.topology(before);f,_,_=field.force(before,springs)
    source_value,mask=source_motion_statistic(before,f,field)
    assert source_value==row['source_interior_p30_motion_um'] and int(mask.sum())==row['source_interior_point_count']
    reconstructed,_=project_ellipse(before+row['accepted_step']*f,field.radius)
    assert np.array_equal(reconstructed,current),'Terminal transition is not reproducible'
    q=quality_state(current,field.topology(current)[0]);assert q==g['final_quality']
    fresh_tri,fresh_springs=field.topology(current);fresh_force,_,_=field.force(current,fresh_springs)
    fresh_source,_=source_motion_statistic(current,fresh_force,field)
    fresh_nominal,_=project_ellipse(current+.2*fresh_force,field.radius)
    fresh_max=float(np.linalg.norm(fresh_nominal-current,axis=1).max())
    assert not out.exists(),'Never overwrite a frozen asset';out.mkdir(parents=True)
    report.mkdir(parents=True,exist_ok=True)
    for name in ('initialization.mat','initialization_run.json','source_request.mat','historical_density_request.mat',
        'historical_density_manifest.json','source_force_parity.json'):
        shutil.copyfile(BASE/name,out/name)
    shutil.copytree(BASE/'source_snapshot',out/'source_snapshot')
    edge_proof=ROOT/'reports/lattice_coverage_repair_20261015/continuation/edge_identity_regression.json'
    if edge_proof.exists():shutil.copyfile(edge_proof,out/'edge_identity_regression.json')
    implementation=out/'implementation_snapshot';implementation.mkdir()
    for p in (segment/'implementation_snapshot').iterdir():
        if p.is_file():shutil.copyfile(p,implementation/p.name)
    for name in ('tools/freeze_source_stopped_lattice.py','tools/check_lattice_numeric_repair.py','tools/plot_lattice_convergence.py'):
        shutil.copyfile(ROOT/name,implementation/Path(name).name)
    shutil.copyfile(BASE/'implementation_snapshot/export_lattice_initialization_covered.m',implementation/'export_lattice_initialization_covered.m')
    # Bind the exact resumed-state chain, not just a claim that it was resumed.
    chain=[];cursor=segment;lineage=out/'lineage';lineage.mkdir()
    while True:
        provenance=json.loads((cursor/'provenance.json').read_bytes())
        destination=lineage/cursor.name;destination.mkdir()
        for name in ('run.json','provenance.json','generation.json','coordinates.npz'):
            if (cursor/name).exists():shutil.copyfile(cursor/name,destination/name)
        state_name='coordinates.npz' if (cursor/'coordinates.npz').exists() else 'checkpoint.npz'
        generation_name='generation.json' if (cursor/'generation.json').exists() else 'checkpoint.npz'
        if state_name=='checkpoint.npz':shutil.copyfile(cursor/state_name,destination/state_name)
        if (cursor/'implementation_snapshot').exists():
            shutil.copytree(cursor/'implementation_snapshot',destination/'implementation_snapshot')
        chain.append({'directory':relative(cursor),'archived_directory':relative(destination),
            'state_file':state_name,'generation_file':generation_name,
            'coordinates_sha256':sha(cursor/state_name),'generation_sha256':sha(cursor/generation_name)})
        if 'source_directory' not in provenance:break
        cursor=ROOT/provenance['source_directory']
        if cursor.name=='segment_003':
            destination=lineage/'resume_1400';destination.mkdir()
            for name in ('run.json','provenance.json','generation.json','coordinates.npz'):
                shutil.copyfile(cursor/name,destination/name)
            shutil.copytree(cursor/'implementation_snapshot',destination/'implementation_snapshot')
            chain.append({'directory':relative(cursor),'archived_directory':relative(destination),
                'coordinates_sha256':sha(cursor/'coordinates.npz'),'generation_sha256':sha(cursor/'generation.json')})
            break
    np.savez_compressed(out/'coordinates.npz',selected_raw_um=current,final_raw_um=current,
        best_quality_raw_um=best,last_step_before_raw_um=before,initial_raw_um=initial)
    xy=-current/1000
    np.savez_compressed(out/'positions.npz',positions_mm=xy,
        source_keys=np.array(['iset_lattice_row:'+str(i) for i in range(len(xy))]),aperture_diameters_um=cone_apertures(xy))
    original_best={'iteration':g['selected_iteration'],'quality':g['selected_quality'],'coordinate_key':'best_quality_raw_um'}
    g.update(selected_iteration=g['accepted_iterations'],selected_quality=q,
        selected_coordinate_policy='Terminal stopping frame; historical best retained separately',best_quality_snapshot=original_best)
    save(out/'generation.json',g);shutil.copyfile(segment/'run.json',out/'run.json')
    stopping={'source_stopping_contract':STOPPING_CONTRACT,'trigger':g['stop_reason'],
        'quality_branch':row['quality_width_stop'],'source_motion_branch':row['source_motion_stop'],
        'source_motion_um':source_value,'source_motion_internal_point_count':int(mask.sum()),
        'threshold_um':g['thresholds']['d_tolerance_um'],'quality':q,'width_condition':row['width_condition'],
        'terminal_transition_bitwise_reproduced':True,'terminal_coordinates_quality_exact':True,
        'fresh_terminal_source_motion_um':fresh_source,'fresh_terminal_maximum_nominal_displacement_um':fresh_max,
        'all_points_stationary_claim':False,'physiological_validation':False,
        'statistic_basis':'Stopping motion is evaluated on force at the last step predecessor and its nominal preprojection interior mask; geometry quality is evaluated on the terminal coordinates',
        'maximum_displacement_is_stop_gate':False,'quality_sample_history':'All retained fresh-topology quality samples; no rolling window',
        'pre_1401_source_motion_history':'Not backfilled or invented; explicit approved contract migration after1400'}
    save(out/'stopping_verification.json',stopping)
    m=json.loads((BASE/'manifest.json').read_bytes())
    m['historical_recipe']=m['recipe'].copy()
    m['recipe']=dict(m['recipe'],numerical_repairs=g['algorithm'],stopping_contract=STOPPING_CONTRACT,
        resumed_from_iteration=1400,accepted_iterations=g['accepted_iterations'],
        source_default_iteration_budget=3000,iteration_budget_policy='Individually protected segments; budget exhaustion is not completion')
    m['recipe'].pop('max_iterations',None)
    m.update(candidate_version=out.name+'_not_adopted',count=len(xy),
        payload={'path':relative(out/'positions.npz'),'sha256':sha(out/'positions.npz')},
        numerical_implementation={'path':relative(implementation/'lattice_relaxation.py'),'sha256':sha(implementation/'lattice_relaxation.py')},
        generation_record={'path':relative(out/'generation.json'),'sha256':sha(out/'generation.json')},
        authorization_reference='Sentinel_8ae5dbf4316c819183a2ae5b9bf2b682_2026-10-16T01:33UTC',
        stopping_contract=STOPPING_CONTRACT,
        candidate_status={'adopted':False,'typed_graph_built':False,'neural_run':False,'physiological_validation':False,
            'initialization_coverage_complete':True,'source_stopping_condition_reached':True,
            'engineering_quality_condition_reached':row['quality_width_stop'],'source_motion_condition_reached':row['source_motion_stop'],
            'maximum_true_motion_equilibrium_reached':fresh_max<.000215,'default_changed':False,'old_input_background_cache_reuse_allowed':False},
        generation_state={'initialization_coverage':'COMPLETE','relaxation':'SOURCE_STOPPING_CONDITION_REACHED',
            'stop_reason':g['stop_reason'],'accepted_iterations':g['accepted_iterations'],'selected_iteration':g['selected_iteration'],
            'quality_threshold':.84,'quality_threshold_reached':q['q_0_8']>.84,'source_motion_condition_reached':row['source_motion_stop'],
            'maximum_true_motion_equilibrium_reached':fresh_max<.000215,'further_run_started':False})
    for key,name in [('exact_initialization','initialization.mat'),('initialization_run_record','initialization_run.json'),
        ('actual_density_request','source_request.mat'),('source_force_parity_archive','source_force_parity.json'),
        ('historical_density_provenance_archive','historical_density_manifest.json')]:
        m[key]=dict(m.get(key,{}),path=relative(out/name),sha256=sha(out/name))
    m['source_snapshot']['directory']=relative(out/'source_snapshot')
    m.pop('validation_record',None)
    for name,identity in m['source_methods'].items():assert sha(out/'source_snapshot'/name)==identity['sha256']
    save(out/'manifest.json',m)
    save(out/'provenance.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'base_manifest_sha256':sha(BASE/'manifest.json'),'lineage':chain,'source_methods':m['source_methods'],
        'density_domain_sampling_LMS_spring_law_unchanged':True,'numeric_guards_retained':True,
        'source_stopping_contract_restored':True,'pre_1401_history_preserved':True,'physiological_validation':False})
    from tools import check_lattice_numeric_repair as checker
    checker.OUT=out;checker.REPORT=report;checker.INITIALIZATION=INITIALIZATION;checker.main()
    validation=json.loads((report/'verification.json').read_bytes())
    metrics=validation['repaired_static_metrics'];invariants=validation['invariants']
    assert metrics['point_count']==121811 and metrics['exact_duplicate_count']==0
    assert invariants['outside_count_at_1e_minus10_um_numerical_roundoff_bound']==0
    assert invariants['coordinate_quality_pairing_exact'] and invariants['legacy_initial_prefix_preserved']
    assert validation['approved_initialization_coverage']['certificate']['missing_candidates_after_union']==0
    machine_bound=64*np.finfo(float).eps*max(1.,float(np.abs(current).max()))
    minimum_nn=metrics['regions']['all_saved_points']['nearest_neighbor_um']['min']
    assert minimum_nn>machine_bound
    stopping.update(full_coverage_candidate_omissions=0,point_count=121811,exact_duplicate_count=0,
        minimum_nearest_neighbor_um=minimum_nn,machine_coincidence_tolerance_um=machine_bound,
        machine_near_coincidence_count=0,outside_count_at_1e_minus10_um_numerical_bound=0)
    save(out/'stopping_verification.json',stopping)
    shutil.copyfile(report/'verification.json',out/'validation.json')
    m['validation_record']={'path':relative(out/'validation.json'),'sha256':sha(out/'validation.json')}
    m['stopping_verification_record']={'path':relative(out/'stopping_verification.json'),'sha256':sha(out/'stopping_verification.json')}
    save(out/'manifest.json',m)
    binding={'path':relative(out/'manifest.json'),'sha256':sha(out/'manifest.json')}
    reader=FixedISETConeMaster(ROOT,binding);assert np.array_equal(reader.positions,xy)
    inventory={relative(p):sha(p) for p in out.rglob('*') if p.is_file()}
    save(out/'asset_inventory.json',{'files':inventory,'manifest_binding':binding,'standard_reader_coordinate_match':True})
    print('FROZEN_SOURCE_STOPPED_ASSET',json.dumps({'binding':binding,'stopping':stopping,'files':len(inventory)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--segment-directory',required=True);p.add_argument('--output-directory',required=True)
    p.add_argument('--report-directory',required=True);a=p.parse_args()
    paths=[(ROOT/v).resolve() for v in (a.segment_directory,a.output_directory,a.report_directory)]
    assert all(p.is_relative_to(ROOT) for p in paths);freeze(*paths)
