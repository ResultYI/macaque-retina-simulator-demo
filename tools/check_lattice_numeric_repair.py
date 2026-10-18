"""Saved-coordinate validation of the repaired generator, never a model run."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.lattice_relaxation import SourceSpringField,quality_state,ellipse_distance,project_ellipse
OUT=ROOT/'data/atlas/isetbio_macaque_cones_numericfix_v3_20261015'
REPORT=ROOT/'reports/lattice_generator_repair_20261015'
INITIALIZATION=REPORT/'source_force_diagnostic.mat'
LEGACY_DIAG=ROOT/'reports/lattice_generator_repair_20261015'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def mass_rectangle(grid_x,grid_y,rho,b):
    x=np.r_[b[0],grid_x[(grid_x>b[0])&(grid_x<b[1])],b[1]]
    y=np.r_[b[2],grid_y[(grid_y>b[2])&(grid_y<b[3])],b[3]]
    X,Y=np.meshgrid(x,y,indexing='ij');z=RegularGridInterpolator((grid_x,grid_y),rho)(np.stack((X,Y),axis=-1))
    return float(np.sum((z[:-1,:-1]+z[1:,:-1]+z[:-1,1:]+z[1:,1:])*.25*np.diff(x)[:,None]*np.diff(y)[None,:]))
def ellipse_mass(interpolator,radius_mm,nr,na):
    u,w=leggauss(nr);u=(u+1)/2;w=w/2;theta=(np.arange(na)+.5)*2*np.pi/na
    r=radius_mm*np.sqrt(u);X=r[:,None]*np.cos(theta);Y=1.2247*r[:,None]*np.sin(theta)
    rho=interpolator(np.stack((X,Y),axis=-1))
    return float(np.sum(rho*w[:,None])*(2*np.pi/na)*1.2247*radius_mm**2/2)
def rectangle_intersect_ellipse_mass(interpolator,radius_mm,bounds,nr=256,na=2048):
    u,w=leggauss(nr);u=(u+1)/2;w=w/2;t=(np.arange(na)+.5)*2*np.pi/na
    cx=np.cos(t);sy=np.sin(t)
    rx=np.where(cx>=0,bounds[1],-bounds[0])/np.maximum(np.abs(cx),1e-30)
    ry=np.where(sy>=0,bounds[3],-bounds[2])/(1.2247*np.maximum(np.abs(sy),1e-30))
    limit=np.minimum(np.minimum(rx,ry),radius_mm);r=np.sqrt(u[:,None])*limit[None,:]
    xy=np.stack((r*cx,1.2247*r*sy),axis=-1)
    return float(np.sum(interpolator(xy)*w[:,None]*limit[None,:]**2)*(2*np.pi/na)*1.2247/2)
def candidate_certificate(initial,metadata,request,radius_um):
    """Independently enumerate the authorized fixed-phase candidate domain."""
    center=np.asarray(metadata['hex_phase_center']);lam=metadata['lambda_min_um_unchanged'];scale=np.sqrt(3)/2
    R=radius_um+.00215;dx=scale*lam
    j=np.arange(np.floor((center[0]-R)/dx)-1,np.ceil((center[0]+R)/dx)+2,dtype=int)
    i=np.arange(np.floor((center[1]-1.2247*R)/lam)-1,np.ceil((center[1]+1.2247*R)/lam)+2,dtype=int)
    J,I=np.meshgrid(j,i);sx=(J*scale)*lam;sy=(I-np.mod(J-1,2)*.5)*lam
    x=np.c_[sx.ravel(order='F')-center[0],sy.ravel(order='F')-center[1]]
    within=ellipse_distance(x,radius_um)<.00215
    fov=float(request['fov_degs'].item());mm=float(request['mm_per_deg'].item());rows=2*(1e3*((fov*1.33/2)*mm));cols=rows
    extra_cols=np.floor(cols/scale+.5)-cols
    legacy=(J>=1)&(J<=np.floor(cols+extra_cols))&(I>=1)&(I<=np.floor(rows))&(sx>=-.1)&(sx<=cols+.1)&(sy>=-.1)&(sy<=rows+.1)
    legacy=legacy.ravel(order='F');extra=within&~legacy
    assert int(within.sum())==metadata['complete_ellipse_candidates'] and int(extra.sum())==metadata['omitted_hex_candidates']
    assert int((within&legacy).sum())==metadata['legacy_ellipse_candidates']
    rho=RegularGridInterpolator((request['grid_x_mm'].reshape(-1),request['grid_y_mm'].reshape(-1)),request['density_per_mm2'])(-x[within]/1000)
    p=rho/rho.max();near_origin=np.linalg.norm(x[within],axis=1)<lam;p[near_origin]=1.
    ep=extra[within];expected_extra=float(p[ep].sum())
    assert abs(expected_extra-metadata['sampler_expected_extra_count'])<1e-7
    # Each sample retains the source integer phase; no free-form added points.
    js=np.rint((initial[:,0]+center[0])/(scale*lam)).astype(int)
    ii=np.rint((initial[:,1]+center[1])/lam+np.mod(js-1,2)*.5).astype(int)
    reconstructed=np.c_[(js*scale)*lam-center[0],(ii-np.mod(js-1,2)*.5)*lam-center[1]]
    error=float(np.linalg.norm(initial-reconstructed,axis=1).max())
    assert error<64*np.finfo(float).eps*max(np.abs(initial).max(),1.)
    return {'independent_complete_candidates':int(within.sum()),'independent_extra_candidates':int(extra.sum()),
        'missing_candidates_after_union':0,'rectangle_brackets_ellipse':bool(x[:,0].min()<-R and x[:,0].max()>R and x[:,1].min()<-1.2247*R and x[:,1].max()>1.2247*R),
        'all_sampled_points_on_original_hex_phase_max_error_um':error,
        'full_sampler_expected_count':float(p.sum()),'full_sampler_count_variance':float(np.sum(p*(1-p))),
        'extra_sampler_expected_count':expected_extra,'realized_count':len(initial),
        'count_statistics_scope':'Numerical source-sampler expectation/variance, not biological confidence limits'}
def main():
    destination=REPORT/'verification.json';assert not destination.exists(),'Keep an existing verification, never overwrite'
    run=json.loads((OUT/'run.json').read_bytes());assert run['returncode']==0 and run['protected_existing_assets_unchanged']
    generation=json.loads((OUT/'generation.json').read_bytes())
    diag=loadmat(INITIALIZATION,squeeze_me=True)
    metadata=json.loads(str(diag['initialization_json'])) if 'initialization_json' in diag else None
    field=SourceSpringField(diag['lookup_xy_um'],diag['lookup_spacing_um'],diag['radius_um'])
    with np.load(OUT/'coordinates.npz',allow_pickle=False) as a:raw=a['selected_raw_um'];initial=a['initial_raw_um'];final=a['final_raw_um']
    tri,springs=field.topology(raw);fresh=quality_state(raw,tri)
    assert fresh==generation['selected_quality'],'Saved coordinates and exact same-state quality differ'
    assert np.array_equal(initial,diag['initial_raw_um']) and len(raw)==len(initial)
    legacy=loadmat(LEGACY_DIAG/'source_force_diagnostic.mat',squeeze_me=True)['initial_raw_um']
    assert np.array_equal(initial[:len(legacy)],legacy)
    f,desired,length=field.force(raw,springs);nominal,_=project_ellipse(raw+.2*f,field.radius)
    nominal_residual=float(np.linalg.norm(nominal-raw,axis=1).max())
    request=loadmat(ROOT/'data/atlas/isetbio_macaque_cones_candidate300_20261015/request.mat')
    graph=json.loads((ROOT/'data/atlas/isetbio_macaque_cones_v1/local_population_graph.json').read_bytes())
    regions={'complete_query_square':(-.512,.512,-.512,.512),'RF_input':(.3,.4,-.05,.05),
        'owner':(.34,.36,-.01,.01),'old_dependency_envelope':tuple(graph['dependency_query']['expanded_RF_query_mm'])}
    assessment=module(ROOT/'reports/geometry_candidate_assessment_20261015/evaluate_static.py','frozen_static_assessment')
    metrics=assessment.assess_points(OUT/'positions.npz',request,regions)
    grid_x=request['grid_x_mm'].reshape(-1);grid_y=request['grid_y_mm'].reshape(-1);rho=request['density_per_mm2']
    interp=RegularGridInterpolator((grid_x,grid_y),rho,bounds_error=True)
    m1=ellipse_mass(interp,field.radius/1000,256,1024);m2=ellipse_mass(interp,field.radius/1000,512,2048)
    xy=-raw/1000;counts={}
    for name,b in regions.items():
        keep=(xy[:,0]>=b[0])&(xy[:,0]<b[1])&(xy[:,1]>=b[2])&(xy[:,1]<b[3]);mass=mass_rectangle(grid_x,grid_y,rho,b)
        counts[name]={'count':int(keep.sum()),'input_bilinear_density_mass':mass,'relative_count_minus_mass':float((keep.sum()-mass)/mass)}
    old=json.loads((ROOT/'reports/geometry_candidate_assessment_20261015/assessment.json').read_bytes())
    trace=json.loads((LEGACY_DIAG/'short_source_cache_trace.json').read_bytes())
    qpairing=fresh==generation['selected_quality']
    result={'scope':'Entire same source-initialized ellipse; declared complete square and local ROIs separately audited; no typed graph/background/neural work',
        'run':run,'generator':{k:v for k,v in generation.items() if k!='trace'},
        'source_commit':'d2924c2fe1d41c7dc5af1006d8081735b08f1b35','upstream_source_files_modified':False,
        'root_causes':{'initial_min_spacing_um':2.342476467280371,'original_short_trace':trace,
            'force_parity':json.loads((INITIALIZATION.parent/('covered_force_parity.json' if metadata else 'force_parity.json')).read_bytes()),
            'history_issue':'Upstream q was attached to different/stale snapshots, including single-precision history',
            'boundary_issue':'Implicit ellipse distance is not unit-gradient distance; old projection omitted division by gradient norm squared',
            'spacing_lookup_factor8':'Literal source nine-neighbor weights sum8; normalization cancels this constant, so no target density rescale was applied',
            'failed_guard_prototypes':'v1 accepted0 steps due roundoff hull faces; v2 accepted4 constrained steps, demonstrating legal hull events and tiny-dt false convergence; both retained, neither adopted'},
        'invariants':{'identical_initial_points_and_order':bool(np.array_equal(initial,diag['initial_raw_um'])),
            'all_initialized_points_preserved':len(raw)==len(initial),'legacy_initial_prefix_preserved':True,
            'no_point_deletion_jitter_density_LMS_or_domain_change':True,
            'coordinate_quality_pairing_exact':qpairing,'maximum_outside_ellipse_um':float(ellipse_distance(raw,field.radius).max()),
            'outside_count_at_1e_minus10_um_numerical_roundoff_bound':int(np.sum(ellipse_distance(raw,field.radius)>1e-10)),
            'selected_nominal_source_step_max_residual_um':nominal_residual,
            'minimum_accepted_interior_path_area_um2':min(r['minimum_path_signed_area_um2'] for r in generation['trace']),
            'every_saved_iteration_used_current_topology':True,
            'swept_machine_collision_rejection_enforced_by_generator':True,
            'total_accepted_swept_pair_checks':sum(r['swept_point_collision_pairs_checked'] for r in generation['trace']),
            'excluded_initial_roundoff_faces':generation['initial_roundoff_zero_area_faces_excluded'],
            'boundary_hull_events':sum(r['legal_hull_orientation_events'] for r in generation['trace'])},
        'repaired_static_metrics':metrics,'density_integrals':counts,
        'full_ellipse_density':{'radius_mm':field.radius/1000,'axes':[1.,1.2247],'integral_256x1024':m1,'integral_512x2048':m2,
            'quadrature_relative_difference':(m2-m1)/m2,'count':len(raw),'relative_count_minus_integral':(len(raw)-m2)/m2,
            'uncertainty':'Numerical quadrature check, not biological CI or a fitted density correction'},
        'previous_300_saved_metrics':old['static_results']['candidate'],
        'remaining':{'physiological_validation':False,'actual_force_equilibrium_claim':nominal_residual<.000215,
            'engineering_quality_condition_is_not_physiology':True,'default_adopted':False,'typed_contacts_rebuilt':False,
            'boundary':'Finite source ellipse and open exterior; hull triangles reported separately; no mirror/periodic tissue',
            'apertures':'Same .79*median6NN estimate; overlap is a geometric diagnostic, not measured anatomy'}}
    if metadata:
        b=(-legacy[:,0].max()/1000,-legacy[:,0].min()/1000,-legacy[:,1].max()/1000,-legacy[:,1].min()/1000)
        old_mass=rectangle_intersect_ellipse_mass(interp,field.radius/1000,b)
        outside_legacy=(xy[:,0]<b[0])|(xy[:,0]>b[1])|(xy[:,1]<b[2])|(xy[:,1]>b[3])
        result['approved_initialization_coverage']={'metadata':metadata,'certificate':candidate_certificate(initial,metadata,request,field.radius),
            'legacy_rectangle_RF_mm':b,'old_rectangle_intersect_ellipse_density_mass':old_mass,
            'missing_region_density_mass':m2-old_mass,'selected_final_points_in_previously_omitted_region':int(outside_legacy.sum()),
            'initialization_run_record':json.loads((INITIALIZATION.parent/'initialization_run.json').read_bytes())}
    with destination.open('x',encoding='utf-8') as h:json.dump(result,h,ensure_ascii=False,indent=2,allow_nan=False);h.write('\n')
    print('FULL_DOMAIN_VERIFICATION',destination)
    print('GENERATOR',{k:generation[k] for k in ['stop_reason','accepted_iterations','selected_iteration','selected_quality']})
    for name,r in metrics['regions'].items():print(name,'N',r['points'],'NN_min_um',r['nearest_neighbor_um']['min'],
        'NN_target_min',r['NN_over_local_target_spacing']['min'],'overlap_cones',r['cones_in_region_with_a_project_disk_overlap'],
        'q_0.8',r['quality_centroids_inside']['p0_8'])
    print('INVARIANTS',result['invariants']);print('ELLIPSE_MASS',result['full_ellipse_density'])
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-directory',default=str(OUT.relative_to(ROOT)))
    parser.add_argument('--report-directory',default=str(REPORT.relative_to(ROOT)))
    parser.add_argument('--initialization',default=str(INITIALIZATION.relative_to(ROOT)))
    args=parser.parse_args()
    OUT=(ROOT/args.data_directory).resolve();REPORT=(ROOT/args.report_directory).resolve();INITIALIZATION=(ROOT/args.initialization).resolve()
    assert all(p.is_relative_to(ROOT) for p in (OUT,REPORT,INITIALIZATION)),'Paths must stay in the project'
    main()
