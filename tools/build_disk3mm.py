"""Build the approved fixed 3 mm disk in resumable, versioned stages.

The previous complete local default is never changed by asset construction.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,subprocess,time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.io import savemat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
REPORT=ROOT/'reports/disk3mm_build_20261016'
ASSET=ROOT/'data/atlas/isetbio_macaque_cones_disk3mm_source_v1'
AXIS=1.2247

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');os.replace(tmp,path)

def polar_mass(field,radius,axis=1.):
    z,w=leggauss(64);rr=(z+1)/2;theta=np.arange(256)*2*np.pi/256
    xy=np.stack(np.broadcast_arrays(radius*rr[:,None]*np.cos(theta),axis*radius*rr[:,None]*np.sin(theta)),axis=-1)
    values=field(xy.reshape(-1,2)).reshape(64,256)
    return float(np.sum(values*(w*rr)[:,None])*np.pi/256*axis*radius**2)

def prepare():
    from retina_sim.source_atlas import SourceDensityAtlas
    from retina_sim.atlas_hc import AtlasHCProvider
    from retina_sim.population_revision import ApprovedPackerRodDensity,ApprovedParasolRFDensity
    from retina_sim.contracts import Quantity
    REPORT.mkdir(parents=True,exist_ok=True);ASSET.mkdir(parents=True,exist_ok=True)
    backup=REPORT/'before';backup.mkdir(exist_ok=True)
    for name in ['retina_platform_v1.json','atlas_isetbio_source_stop_shared_v1.json']:
        path=backup/name
        if not path.exists():path.write_bytes((ROOT/'configurations'/name).read_bytes())
    atlas=SourceDensityAtlas(ROOT,profile_id='retina_atlas_Schein_bridge_research_v1');hc=AtlasHCProvider(ROOT)
    # Longest feedforward route: cone -> BC -> AC -> owner RGC, each retaining
    # the existing40um support. HC footprints add their two-sided finite support.
    owner=3.;hard=.04;names=tuple(atlas.henle.nodes)
    owner_opl=max(float(atlas.henle._radial(owner,[(m,1.)])) for m in names)
    inner_opl=owner_opl+3*hard;cone_opl=inner_opl+2*hc.maximum_envelope_mm
    cone_rf=max(float(atlas.henle._inverse_radial(cone_opl,[(m,1.)])) for m in names)
    macro=.5*np.ceil(cone_rf/.5);query=.032*np.ceil(macro/.032)
    initial_radius=float(np.hypot(query,query/AXIS)+.032)
    source_outer=min(rows[-1]['eccentricity_mm'] for rows in atlas.cones.rows.values())
    if AXIS*initial_radius+.032>=source_outer:raise ValueError('Required generator ellipse exceeds unchanged cone density source; do not extrapolate')
    limits={'owner_disk_radius_mm':owner,'owner_max_OPL_radius_mm':owner_opl,'hard_support_mm':hard,
        'feedforward_support_steps':3,'inner_OPL_support_radius_mm':inner_opl,
        'maximum_HC_footprint_radius_mm':hc.maximum_envelope_mm,'cone_RF_query_radius_mm':cone_rf,
        'HC_master_OPL_radius_mm':inner_opl+hc.maximum_envelope_mm+hc.density.max_interaction_support_mm(),
        'AII_master_IPL_radius_mm':inner_opl+hard,'RBC_master_radius_mm':inner_opl+2*hard,'rod_master_radius_mm':inner_opl+3*hard,
        'S_complete_macro_halfwidth_mm':float(macro),'complete_cone_query_bounds_mm':[-float(query),float(query),-float(query),float(query)],
        'generator_initial_radius_mm':initial_radius,'ellipse_y_over_x':AXIS,'cone_source_outer_min_mm':source_outer,
        'parasol_owner_and_seed_radius_mm':3.,'parasol_halo_extrapolation':False,
        'boundary':'one finite canonical network; no exterior feedback, compensation or crop renormalization',
        'whole_exterior_recurrent_closure':'NOT_CERTIFIED'}
    estimates={'cone_owner_Gauss64_estimate':polar_mass(atlas.cones.array_values,3.),
        'cone_generator_ellipse_estimate':polar_mass(atlas.cones.array_values,initial_radius,AXIS),
        'rod_support_estimate':polar_mass(ApprovedPackerRodDensity(ROOT),limits['rod_master_radius_mm']),
        'RBC_support_estimate':polar_mass(ApprovedPackerRodDensity(ROOT),limits['RBC_master_radius_mm'])/18,
        'AII_support_estimate':np.pi*limits['AII_master_IPL_radius_mm']**2*5000,
        'HC_support_estimate':polar_mass(hc.density.total,limits['HC_master_OPL_radius_mm']),
        'parasol_per_polarity_owner_estimate':polar_mass(ApprovedParasolRFDensity('ON'),3.),
        'meaning':'planning integrals only; unchanged samplers determine actual fixed counts'}
    save(REPORT/'support.json',dict(limits=limits,estimates=estimates,
        approval='User tongyi: full3mm RBC/AII quantities, HC candidate geometry and unchanged effective dynamics; original S/Schein/parasol retained',
        physiological_validation=False,status='PREPARED_NOT_RUNTIME_READY'))
    path=ASSET/'request_full_lookup.mat'
    if path.exists():
        print('REQUEST_ALREADY_PREPARED',flush=True);return
    # Preserve the original physical tabulation resolution. Values outside all
    # cardinal source radii remain NaN and are never queried by the ellipse.
    step=2.2/512
    def grid(extent):
        n=int(np.ceil(2*extent/step))+1
        if n%2==0:n+=1
        return np.linspace(-extent,extent,n)
    # The pinned logarithmic lookup uses the maximum radial extent for BOTH
    # coordinate axes, even though the actual generation ellipse is narrower.
    gx=grid(AXIS*initial_radius+.032);gy=gx.copy()
    rho=np.full((len(gx),len(gy)),np.nan)
    for i,x in enumerate(gx):
        xy=np.column_stack((np.full(len(gy),x),gy));valid=np.linalg.norm(xy,axis=1)<=source_outer
        rho[i,valid]=atlas.cones.array_values(xy[valid])
        if i%256==0:print('DENSITY_ROWS',i,len(gx),flush=True)
    # Check the batching against the unchanged detailed provider before export.
    xy=np.array([[0.,0.],[.01,.02],[.35,0.],[1.,2.],[-2.,1.],[0.,3.],[3.,0.],[-3.,0.]])
    exact=atlas.cones(Quantity(xy,'mm','functional_position','project_approximation','functional_visual_field'))
    np.testing.assert_allclose(atlas.cones.array_values(xy),exact,rtol=4e-15,atol=0)
    profile=json.loads((ROOT/'configurations/iset_lcd_apple_mw_v1.json').read_bytes())
    minimum=float(np.sqrt(2e6/(np.sqrt(3)*np.nanmax(rho))))
    savemat(path,{'isetbio_root':profile['isetbio_root'],'grid_x_mm':gx,'grid_y_mm':gy,'density_per_mm2':rho,
        'fov_degs':2*initial_radius/(1.33*.223),'mm_per_deg':.223,'minimum_spacing_um':minimum,'seed':20261010.,'max_iterations':0.})
    save(ASSET/'request_full_lookup_identity.json',{'request_path':path.relative_to(ROOT).as_posix(),'request_sha256':sha(path),'support':limits,'density_estimates':estimates,
        'density_scale':1.,'tabulation_step_max_mm':max(float(gx[1]-gx[0]),float(gy[1]-gy[0])),
        'source_profile':'retina_atlas_Schein_bridge_research_v1','cone_primary_points':atlas.cones.bundle.manifest()['source_assets']['photoreceptor_points'],
        'generation_algorithm':'unchanged source hex/Bernoulli initialization, complete ellipse coverage, repaired source springs',
        'stopping_contract':'pinned_source_OR_quality_width_or_internal_p30_v1','source_motion_threshold_um':.000215,
        'quality_threshold':.84,'existing121811point_asset_preserved':True})
    print('PREPARED',json.dumps(dict(limits=limits,estimates=estimates)),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','initialize','relax','freeze-cones','snapshot-core','hc','populations','rod-graph','outer-tables','input-plan','bc-tables','midget-tables','hc-contacts','inner-tables','aii-bc-contacts','complete-geometry','geometry-stages','freeze-graph','contact-geometry'])
    p.add_argument('--additional-iterations',type=int,default=300);p.add_argument('--budget-seconds',type=float,default=840.)
    p.add_argument('--until-source-stop',action='store_true');args=p.parse_args()
    try:
        if args.stage=='prepare':prepare()
        elif args.stage=='initialize':initialize()
        elif args.stage=='relax':relax_cones(args)
        elif args.stage=='freeze-cones':freeze_cones()
        elif args.stage=='snapshot-core':snapshot_generation_core()
        elif args.stage=='complete-geometry':complete_geometry()
        elif args.stage=='geometry-stages':build_geometry_stages()
        elif args.stage=='contact-geometry':build_contact_geometry()
        elif args.stage=='freeze-graph':
            from retina_sim.disk_circuit import freeze_disk_circuit
            result=freeze_disk_circuit(ROOT,'reports/disk3mm_build_20261016','data/atlas/disk3mm_circuit_v1')
            save(REPORT/'global_graph.json',result);print('GLOBAL_LOGICAL_DISK_GRAPH_FROZEN',flush=True)
        elif args.stage=='hc':build_hc()
        elif args.stage=='populations':build_populations()
        elif args.stage=='rod-graph':build_primary_graph()
        elif args.stage=='outer-tables':
            from retina_sim.disk_source_geometry import freeze_outer_tables
            result=freeze_outer_tables(ROOT,'retina_atlas_disk3mm_v1',json.loads((REPORT/'support.json').read_bytes()),'data/atlas/disk3mm_typed_tables_v1')
            save(REPORT/'outer_tables.json',result);print(json.dumps(result),flush=True)
        elif args.stage=='input-plan':
            from retina_sim.disk_tables import FrozenDiskTable
            from retina_sim.disk_input import freeze_input_plan
            outer=json.loads((REPORT/'outer_tables.json').read_bytes())
            result=freeze_input_plan(ROOT,FrozenDiskTable(ROOT,outer['cones']),'data/atlas/disk3mm_input_plan_v1')
            save(REPORT/'input_plan.json',result);print(json.dumps(result),flush=True)
        elif args.stage=='bc-tables':
            from retina_sim.disk_inner import freeze_bc_tables
            result=freeze_bc_tables(ROOT,json.loads((REPORT/'outer_tables.json').read_bytes()),
                json.loads((REPORT/'support.json').read_bytes()),'data/atlas/disk3mm_inner_v1')
            save(REPORT/'bc_tables.json',result);print('GLOBAL_BC_TABLES_FROZEN',flush=True)
        elif args.stage=='midget-tables':
            from retina_sim.disk_inner import freeze_midget_rgcs
            result=freeze_midget_rgcs(ROOT,json.loads((REPORT/'bc_tables.json').read_bytes()),'data/atlas/disk3mm_inner_v1')
            save(REPORT/'midget_tables.json',result);print('GLOBAL_MIDGET_TABLES_FROZEN',flush=True)
        elif args.stage=='hc-contacts':
            from retina_sim.disk_outer import freeze_outer_contacts
            result=freeze_outer_contacts(ROOT,json.loads((REPORT/'outer_tables.json').read_bytes()),'data/atlas/disk3mm_outer_connections_v1')
            save(REPORT/'outer_connections.json',result);print('GLOBAL_HC_CONTACTS_FROZEN',json.dumps(result),flush=True)
        elif args.stage=='inner-tables':
            from retina_sim.disk_inner import freeze_remaining_inner
            result=freeze_remaining_inner(ROOT,json.loads((REPORT/'outer_tables.json').read_bytes()),
                json.loads((REPORT/'bc_tables.json').read_bytes()),json.loads((REPORT/'midget_tables.json').read_bytes()),
                json.loads((REPORT/'population_build.json').read_bytes())['fixed_populations'],
                json.loads((REPORT/'support.json').read_bytes()),'data/atlas/disk3mm_inner_v1')
            save(REPORT/'inner_tables.json',result);print('GLOBAL_INNER_TABLES_FROZEN',flush=True)
        elif args.stage=='aii-bc-contacts':
            from retina_sim.disk_inner import freeze_aii_bc_contacts
            result=freeze_aii_bc_contacts(ROOT,json.loads((REPORT/'inner_tables.json').read_bytes()),
                json.loads((REPORT/'population_build.json').read_bytes())['fixed_populations'],'data/atlas/disk3mm_global_routes_v2/AII_BC')
            save(REPORT/'aii_bc_connections.json',result);print('GLOBAL_AII_BC_CONTACTS_FROZEN',flush=True)
    except Exception as exc:
        save(REPORT/(args.stage+'_failure.json'),{'stage':args.stage,'status':'FAILED_OR_PAUSED',
            'error':type(exc).__name__+': '+str(exc),'old_default_changed':False,'existing_checkpoints_retained':True})
        raise

def initialize():
    from scipy.io import loadmat
    from retina_sim.background_checkpoint import MemoryGuard
    receipt=json.loads((ASSET/'request_full_lookup_identity.json').read_bytes());request=ROOT/receipt['request_path']
    if sha(request)!=receipt['request_sha256']:raise ValueError('Generator request changed')
    output=ASSET/'initialization.mat'
    if output.exists():raise FileExistsError('Initialization already exists; never restart it')
    guard=MemoryGuard(min_available_bytes=4*1024**3);guard.check()
    if guard.last['available_physical_bytes']<4*1024**3:raise RuntimeError('Insufficient free RAM for source initializer')
    profile=json.loads((ROOT/'configurations/iset_lcd_apple_mw_v1.json').read_bytes())
    preferences=ASSET/'matlab_preferences';temporary=ASSET/'matlab_temp'
    preferences.mkdir(exist_ok=True);temporary.mkdir(exist_ok=True)
    env=dict(os.environ,MATLAB_PREFDIR=str(preferences),TEMP=str(temporary),TMP=str(temporary))
    quote=lambda p:str(Path(p).resolve()).replace("'","''")
    pending=ASSET/'initialization.pending.mat'
    expression="addpath('"+quote(ROOT/'tools')+"');export_disk3mm_initialization('"+quote(request)+"','"+quote(pending)+"');"
    save(REPORT/'status.json',{'stage':'CONE_INITIALIZATION','status':'RUNNING','request_sha256':sha(request),'started_unix':time.time(),
        'expected_points_estimate':receipt['density_estimates']['cone_generator_ellipse_estimate'],'old_default_changed':False})
    started=time.monotonic()
    with (ASSET/'matlab_initialization_full_lookup.log').open('wb') as log:
        subprocess.run([profile['matlab_executable'],'-batch',expression],stdout=log,stderr=subprocess.STDOUT,env=env,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),check=True,timeout=1800)
    raw=loadmat(pending,squeeze_me=True);xy=np.asarray(raw['initial_raw_um']);radius=float(raw['radius_um'])
    q=receipt['support']['complete_cone_query_bounds_mm'][1]*1000
    if np.hypot(q,q/AXIS)>=radius:raise ValueError('Initialized ellipse does not contain the complete query square')
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.all(np.isfinite(xy)):raise ValueError('Invalid source initialization')
    os.replace(pending,output)
    save(ASSET/'initialization_identity.json',{'sha256':sha(output),'request_sha256':sha(request),'point_count':len(xy),'radius_um':radius,
        'source_metadata':json.loads(str(raw['initialization_json'])),'wall_seconds':time.monotonic()-started,
        'exporter_sha256':sha(ROOT/'tools/export_disk3mm_initialization.m'),'status':'INITIALIZED_NOT_RELAXED'})
    save(REPORT/'status.json',{'stage':'CONE_INITIALIZATION','status':'COMPLETE','point_count':len(xy),'old_default_changed':False})
    print('INITIALIZATION_PASS',len(xy),radius,flush=True)

def relax_cones(args):
    from scipy.io import loadmat
    from retina_sim.lattice_relaxation import SourceSpringField,relax
    from retina_sim.background_checkpoint import MemoryGuard
    initial_path=ASSET/'initialization.mat';identity=json.loads((ASSET/'initialization_identity.json').read_bytes())
    if sha(initial_path)!=identity['sha256']:raise ValueError('Frozen initialization changed')
    raw=loadmat(initial_path,squeeze_me=True);initial=np.asarray(raw['initial_raw_um'])
    field=SourceSpringField(raw['lookup_xy_um'],raw['lookup_spacing_um'],float(raw['radius_um']))
    checkpoint_path=ASSET/'relaxation_checkpoint.npz';state=None
    if checkpoint_path.exists():
        with np.load(checkpoint_path,allow_pickle=False) as z:
            if str(z['initialization_sha256'])!=identity['sha256']:raise ValueError('Checkpoint initialization mismatch')
            state={'current_raw_um':z['current_raw_um'].copy(),'best_raw_um':z['best_raw_um'].copy(),
                'generation':json.loads(str(z['generation_json']))}
    guard=MemoryGuard(max_rss_bytes=8*1024**3,min_available_bytes=2*1024**3);last_progress=0.
    def on_step(before,after,old_edges,new_edges,force,desired,row):
        nonlocal last_progress
        if time.monotonic()-last_progress<15:return
        last_progress=time.monotonic()
        status={'stage':'CONE_RELAXATION','status':'RUNNING','point_count':len(after),'old_default_changed':False,
            **{k:row[k] for k in ['iteration','q_0_8','source_interior_p30_motion_um','source_motion_stop','actual_max_movement_um','elapsed_seconds']}}
        save(REPORT/'status.json',status);print('CONE_PROGRESS',json.dumps(status),flush=True)
    def checkpoint(current,best,result):
        pending=ASSET/'relaxation_checkpoint.writing.npz'
        np.savez(pending,current_raw_um=current,best_raw_um=best,generation_json=np.array(json.dumps(result,allow_nan=False)),
            initialization_sha256=np.array(identity['sha256']))
        os.replace(pending,checkpoint_path)
        save(ASSET/'relaxation_progress.json',{k:v for k,v in result.items() if k!='trace'})
        if not guard.check():
            save(REPORT/'status.json',{'stage':'CONE_RELAXATION','status':'RESOURCE_PAUSE','checkpoint_iteration':result['accepted_iterations'],
                'reason':guard.stop_reason,'memory':guard.last,'old_default_changed':False})
            raise RuntimeError('Memory safety pause after durable cone checkpoint: '+str(guard.last))
    while True:
        best,current,result=relax(initial,field,resume=state,max_iterations=args.additional_iterations,max_wall_seconds=args.budget_seconds,
            on_step=on_step,on_checkpoint=checkpoint,checkpoint_interval_iterations=5)
        terminal=result['stop_reason'] in ('original_quality_plus_width_condition','original_internal_p30_motion_condition')
        save(ASSET/'generation.json',result)
        if terminal:
            save(REPORT/'status.json',{'stage':'CONE_RELAXATION','status':'SOURCE_STOP_REACHED','point_count':len(current),
                'accepted_iterations':result['accepted_iterations'],'stop_reason':result['stop_reason'],'old_default_changed':False})
            print('SOURCE_STOP_REACHED',result['accepted_iterations'],result['stop_reason'],flush=True);return
        if not args.until_source_stop:
            print('DURABLE_SEGMENT_PAUSE',result['accepted_iterations'],result['stop_reason'],flush=True);return
        if result['stop_reason'] not in ('iteration_budget','wall_budget'):raise RuntimeError('Unexpected numerical stop: '+result['stop_reason'])
        state={'current_raw_um':current,'best_raw_um':best,'generation':result}

def build_hc():
    from retina_sim.atlas_hc import AtlasHCProvider
    from retina_sim.atlas_hc_master import materialize_hc_master,FixedJointHCMaster
    from retina_sim.disk_quadrature import memoized_hc_rules
    limits=json.loads((REPORT/'support.json').read_bytes())['limits'];radius=limits['HC_master_OPL_radius_mm']
    base=json.loads((ROOT/'configurations/atlas_hc_joint_master_v2.json').read_bytes())
    research=json.loads((ROOT/base['research_generator_path']).read_bytes())
    # Computation budgets change; density, soft-core energy, temperature, sweep
    # count, integration tolerances and seeds retain their approved values.
    research['generator'].update(max_partitions=1000000,max_proposals=8000000,max_quadrature_points=640000000,max_depth=24)
    research_path=ROOT/'configurations/hc_disk3mm_generator_v3.json'
    if research_path.exists():
        if json.loads(research_path.read_bytes())!=research:raise ValueError('Existing HC computation settings differ')
    else:save(research_path,research)
    config_path=ROOT/'configurations/atlas_hc_disk3mm_v1.json'
    config=dict(base,profile_id='atlas_hc_disk3mm_v1',asset_path='data/atlas/hc_disk3mm_master_v1.json',
        expected_asset_sha256=None,research_generator_path=research_path.relative_to(ROOT).as_posix(),
        canonical_bounds_mm=[-radius,radius,-radius,radius],max_cells=600000,
        mask_source_id='HC_fixed_rectangular_enclosure_of_disk3mm_support_v1',
        scope='Approved disk3mm output with derived finite OPL support; same radial/source-transfer HC density and joint generator, no physiological calibration')
    config['construction_integral_cache']={'implementation_sha256':sha(ROOT/'retina_sim/disk_quadrature.py'),
        'rule':'exact same bounds/order memoization; no quadrature/order/tolerance,partition or RNG changes'}
    if config_path.exists():
        previous=json.loads(config_path.read_bytes())
        if previous.get('expected_asset_sha256'):config=previous
        elif previous.get('canonical_disk_radius_mm') or previous.get('research_generator_path')!=config['research_generator_path']:
            save(REPORT/'before/hc_unfinished_disk_mask_config.json',previous)
            save(config_path,config)
        else:
            previous['construction_integral_cache']=config['construction_integral_cache'];config=previous;save(config_path,config)
    else:save(config_path,config)
    atlas=json.loads((ROOT/'configurations/retina_atlas_Schein_bridge_research_v1.json').read_bytes())
    atlas['HC_master_configuration']=config_path.relative_to(ROOT).as_posix();hc=AtlasHCProvider(ROOT)
    if config['expected_asset_sha256']:
        master=FixedJointHCMaster(ROOT,atlas,hc.density)
        print('HC_ALREADY_FROZEN',len(master.table.source_keys),flush=True);return
    save(REPORT/'hc_status.json',{'status':'GENERATING_GLOBAL_FIXED_HC','radius_OPL_mm':radius,
        'estimated_count':json.loads((REPORT/'support.json').read_bytes())['estimates']['HC_support_estimate'],
        'generator_sweeps':research['generator'].get('sweeps',hc.density.config['generator']['sweeps']),
        'density_or_biological_rules_changed':False,'only_computation_budgets_increased':True})
    with memoized_hc_rules() as quadratures:
        result=materialize_hc_master(ROOT,atlas,hc.density)
        result['exact_integral_cache_hits']=sum(q.rule_cache_hits for q in quadratures)
    save(REPORT/'hc_status.json',dict(result,status='FROZEN_GLOBAL_FIXED_HC'))
    print('HC_FROZEN',json.dumps(result),flush=True)

def build_populations():
    from retina_sim.disk_population import materialize_disk_population,FixedDiskPopulation
    from retina_sim.population_revision import ApprovedPackerRodDensity,ApprovedParasolRFDensity,APPROVAL
    from retina_sim.background_checkpoint import MemoryGuard
    limits=json.loads((REPORT/'support.json').read_bytes())['limits'];rod=ApprovedPackerRodDensity(ROOT)
    fields=[('parasol_RGC_ON',3.,ApprovedParasolRFDensity('ON'),20261014,ApprovedParasolRFDensity('ON').metadata,100000),
        ('parasol_RGC_OFF',3.,ApprovedParasolRFDensity('OFF'),20261015,ApprovedParasolRFDensity('OFF').metadata,100000),
        ('AII_AC',limits['AII_master_IPL_radius_mm'],lambda xy:np.full(len(xy),5000.),20261013,
            {'provider_id':'existing_AII_constant5000','density_cells_mm2':5000,'approval':APPROVAL,'disk3mm_extension':'user tongyi,2026-10-16'},300000),
        ('rod_BC',limits['RBC_master_radius_mm'],lambda xy:rod(xy)/18,20261012,
            {'provider_id':'existing_RBC_rhoRod_over18','density_ratio':1/18,'rod_provider':rod.metadata,'approval':APPROVAL,'disk3mm_extension':'user tongyi,2026-10-16'},400000),
        ('rods',limits['rod_master_radius_mm'],rod,20261011,rod.metadata,4500000)]
    catalog_path=REPORT/'population_build.json';catalog=json.loads(catalog_path.read_bytes()) if catalog_path.exists() else {'fixed_populations':{}}
    for name,radius,field,seed,source,budget in fields:
        guard=MemoryGuard(max_rss_bytes=8*1024**3,min_available_bytes=3*1024**3)
        if not guard.check():raise RuntimeError('Population construction memory reserve reached')
        print('POPULATION_GENERATION',name,radius,flush=True)
        if name.startswith('parasol'):
            mass=field.disk_mass(radius);method='radial_1D_quad_of_unchanged_author_field_epsrel1e-12'
        elif name=='AII_AC':
            mass=float(np.pi*radius**2*5000);method='constant5000_times_exact_disk_area'
        else:
            mass=rod.disk_mass(radius)/(18 if name=='rod_BC' else 1)
            method='exact_cardinal_angular_cos2_integral_times_PCHIP_polynomial_radial_integral'+('_over18' if name=='rod_BC' else '')
        # Retain the already frozen valid parasols, including their existing IDs.
        previous=catalog['fixed_populations'].get(name)
        if name.startswith('parasol') and previous:
            master=FixedDiskPopulation(ROOT,previous)
            residual=abs(master.record['diagnostics']['mass']-mass)/mass
            if master.record['count']==int(np.floor(mass+.5)) and residual<=1e-4:
                previous['accurate_integral_recheck']={'mass':mass,'method':method,'relative_mass_difference':residual,'reused_without_regeneration':True}
                save(catalog_path,catalog);print('POPULATION_REUSED',name,master.record['count'],flush=True);continue
            print('POPULATION_TRIAL_RETAINED_ACCURATE_COUNT_DIFFERS',name,master.record['count'],int(np.floor(mass+.5)),flush=True)
        bound=(float(field(np.zeros((1,2)))[0]) if name.startswith('parasol') else 5000. if name=='AII_AC' else
            max(row['density_per_mm2'] for rows in rod.rows.values() for row in rows)/(18 if name=='rod_BC' else 1))
        master=materialize_disk_population(ROOT,'data/atlas/disk3mm_populations_v2/'+name,name,radius,field,seed,source,
            max_cells=budget,density_upper_bound=bound,density_mass=mass,integral_method=method)
        if previous and previous['path']!=master.binding['path']:
            catalog.setdefault('preserved_superseded_trials',{})[name]=dict(previous,reason='generic_masked_cartesian_integral_replaced_by_accurate_same_field_integral')
        catalog['fixed_populations'][name]=dict(master.binding,count=master.record['count'],radius_mm=radius)
        save(catalog_path,catalog);print('POPULATION_FROZEN',name,master.record['count'],flush=True)

def freeze_cones():
    from scipy.spatial import cKDTree
    from retina_sim.isetbio_geometry import COMMIT,SCHEMA,cone_apertures,FixedISETConeMaster
    checkpoint=ASSET/'relaxation_checkpoint.npz';manifest_path=ASSET/'manifest.json'
    if manifest_path.exists():raise FileExistsError('Cone master already frozen; never replace')
    with np.load(checkpoint,allow_pickle=False) as z:
        generation=json.loads(str(z['generation_json']));current=z['current_raw_um'].copy()
        initialization_sha=str(z['initialization_sha256'])
    last=generation['trace'][-1]
    if not(last['source_motion_stop'] or last['quality_width_stop']):raise RuntimeError('Original source OR stop not reached; no provisional master adoption')
    receipt=json.loads((ASSET/'request_full_lookup_identity.json').read_bytes());identity=json.loads((ASSET/'initialization_identity.json').read_bytes())
    if initialization_sha!=identity['sha256']:raise ValueError('Checkpoint initialization differs')
    snapshot_generation_core()
    xy=-current/1000
    if len(xy)!=identity['point_count'] or not np.all(np.isfinite(xy)):raise ValueError('Invalid final point identity/count')
    nn=cKDTree(xy).query(xy,k=2)[0][:,1]
    if np.any(nn<=0):raise ValueError('Duplicate final cone locations; no deletion or jitter')
    radius=identity['radius_um'];outside=np.sqrt(current[:,0]**2+(current[:,1]/AXIS)**2)-radius
    tolerance=max(1e-10,128*np.finfo(float).eps*radius)
    if np.any(outside>tolerance):raise ValueError('Final cone outside numerical ellipse domain')
    payload=ASSET/'positions.npz'
    if payload.exists():raise FileExistsError('Unregistered partial cone payload retained; inspect before replacement')
    np.savez_compressed(payload,positions_mm=xy,source_keys=np.array(['iset_lattice_row:'+str(i) for i in range(len(xy))]),
        aperture_diameters_um=cone_apertures(xy))
    original=json.loads((ROOT/'data/atlas/isetbio_macaque_cones_source_stop_v1_20261016/manifest.json').read_bytes())
    frozen_generation={k:v for k,v in generation.items() if k!='trace'}
    frozen_generation['best_diagnostic_iteration']=generation.get('selected_iteration')
    frozen_generation['best_diagnostic_quality']=generation.get('selected_quality')
    frozen_generation['selected_iteration']=generation['accepted_iterations']
    frozen_generation['selected_quality']=generation['final_quality']
    frozen_generation['selected_state']='current_original_source_OR_stop_state; best diagnostic not substituted'
    value={'schema':SCHEMA,'isetbio_commit':COMMIT,'population':'cones','density_species':'Macaca nemestrina','human_density_used':False,
        'position_role':'rf_input_inner_segment_project_transfer','axes':'+x nasal/+y superior; unchanged project RF=IS registration',
        'complete_query_bounds_mm':receipt['support']['complete_cone_query_bounds_mm'],'count':len(xy),
        'payload':{'path':payload.relative_to(ROOT).as_posix(),'sha256':sha(payload)},'source_methods':original['source_methods'],
        'density':receipt,'generation_domain':{'ellipse_radius_um':radius,'ellipse_y_over_x':AXIS,'owner_disk_radius_mm':3.},
        'initialization':identity,'stopping_contract':generation['stopping_contract'],'generation':frozen_generation,
        'checkpoint_sha256':sha(checkpoint),'implementation_snapshot':'generation_implementation/manifest.json',
        'claims':{'measured_individual_mosaic':False,'physiological_validation':False,'cone_geometry_covers_disk3mm':True,'full_network_ready':False},
        'diagnostics':{'minimum_nearest_neighbor_um':float(nn.min()*1000),'duplicates':0,'numerical_outside':0,
            'maximum_float_ellipse_excess_um':float(outside.max()),'numeric_domain_tolerance_um':tolerance}}
    save(manifest_path,value);binding={'path':manifest_path.relative_to(ROOT).as_posix(),'sha256':sha(manifest_path)}
    master=FixedISETConeMaster(ROOT,binding);np.testing.assert_array_equal(master.positions,xy)
    atlas=json.loads((ROOT/'configurations/retina_atlas_isetbio_source_stop_v1.json').read_bytes())
    atlas.update(cone_master=binding,HC_master_configuration='configurations/atlas_hc_disk3mm_v1.json')
    path=ROOT/'configurations/retina_atlas_disk3mm_v1.json'
    if path.exists():raise FileExistsError('Disk atlas profile already exists')
    save(path,atlas);save(REPORT/'cone_master.json',dict(binding,count=len(xy),source_stop_reached=True,old_default_changed=False))
    print('CONE_MASTER_FROZEN',len(xy),binding['sha256'],flush=True)


def snapshot_generation_core():
    """Preserve unchanged numerical core independently of edited orchestration."""
    directory=ASSET/'generation_implementation';directory.mkdir(exist_ok=True)
    records={}
    for relative in ('retina_sim/lattice_relaxation.py','retina_sim/lattice_quality.py','tools/export_disk3mm_initialization.m'):
        original=ROOT/relative;target=directory/original.name
        if target.exists():
            if target.read_bytes()!=original.read_bytes():raise ValueError('Numerical core changed after snapshot: '+relative)
        else:target.write_bytes(original.read_bytes())
        records[relative]={'snapshot':target.name,'sha256':sha(target)}
    identity=json.loads((ASSET/'initialization_identity.json').read_bytes())
    if records['tools/export_disk3mm_initialization.m']['sha256']!=identity['exporter_sha256']:
        raise ValueError('Initializer exporter differs from actually executed version')
    import scipy
    value={'schema':'source_generator_numerical_core_snapshot_v1','files':records,
        'versions':{'numpy':np.__version__,'scipy':scipy.__version__},
        'scope':'Unchanged numerical core and actually recorded initializer exporter; construction orchestration edited while worker ran is not claimed as an exact loaded-code snapshot',
        'initialization_sha256':identity['sha256'],'initialization_request':json.loads((ASSET/'request_full_lookup_identity.json').read_bytes())['request_sha256']}
    path=directory/'manifest.json'
    if path.exists():
        previous=json.loads(path.read_bytes())
        # Preserve the earlier pre-run snapshot, which also contains the exact
        # orchestration version loaded by the running source worker.
        if 'numeric_contract' in previous:
            for relative,current in records.items():
                if previous['files'].get(relative)!=current['sha256']:raise ValueError('Pre-run numerical core snapshot differs: '+relative)
            for relative,expected in previous['files'].items():
                if sha(directory/Path(relative).name)!=expected:raise ValueError('Pre-run snapshot file changed: '+relative)
            print('EXISTING_PRE_RUN_NUMERICAL_SNAPSHOT_VERIFIED',sha(path),flush=True);return
        if previous!=value:raise ValueError('Numerical source snapshot identity differs')
    else:save(path,value)
    print('NUMERICAL_CORE_SNAPSHOT_FROZEN',sha(path),flush=True)


def complete_geometry():
    """Continue the current production build when its original source stop arrives.

    This joins already running workers. It never restarts cone/HC sampling,
    changes a stopping condition or adopts a new default.
    """
    receipt=REPORT/'authorized_delivery_request.json'
    if receipt.exists() and json.loads(receipt.read_bytes()).get('status')=='PAUSED_FOR_AUTHORIZED_HUMAN_CONE_REUSE':
        raise RuntimeError('Macaque production continuation is explicitly paused for human cone reuse; preserved checkpoint is not a completed source-stop result')
    started=time.time();progress_path=REPORT/'geometry_pipeline.json'
    while not (ASSET/'manifest.json').exists():
        status=json.loads((REPORT/'status.json').read_bytes())
        if status['status']=='SOURCE_STOP_REACHED':freeze_cones();break
        if status['status']=='RESOURCE_PAUSE' or time.time()-(REPORT/'status.json').stat().st_mtime>300:
            raise RuntimeError('Cone worker paused or no longer progressing; checkpoint retained, no automatic restart')
        save(progress_path,{'status':'WAITING_FOR_UNCHANGED_CONE_SOURCE_STOP','cone_iteration':status.get('iteration'),
            'started_unix':started,'old_default_changed':False,'runtime_ready':False})
        time.sleep(15)
    hc_config=ROOT/'configurations/atlas_hc_disk3mm_v1.json'
    while not json.loads(hc_config.read_bytes()).get('expected_asset_sha256'):
        log=REPORT/'hc_cached_integral_generation.log'
        if log.exists() and 'Traceback (most recent call last)' in log.read_text(encoding='utf8',errors='replace')[-65536:]:
            raise RuntimeError('HC worker failed; original output/log retained, no automatic replacement')
        save(progress_path,{'status':'WAITING_FOR_CURRENT_FIXED_HC_MASTER','started_unix':started,'old_default_changed':False,'runtime_ready':False})
        time.sleep(15)
    # Load the current construction code only when both existing source workers
    # finish; editing non-numerical integration code must not stale this waiter.
    subprocess.run([sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'geometry-stages'],cwd=ROOT,check=True)


def build_geometry_stages():
    started=time.time();progress_path=REPORT/'geometry_pipeline.json'
    from retina_sim.disk_source_geometry import freeze_outer_tables
    from retina_sim.disk_outer import freeze_outer_contacts
    from retina_sim.disk_inner import freeze_bc_tables,freeze_midget_rgcs,freeze_remaining_inner,freeze_aii_bc_contacts
    from retina_sim.disk_input import freeze_input_plan
    from retina_sim.disk_tables import FrozenDiskTable
    from retina_sim.disk_circuit import freeze_disk_circuit
    support=json.loads((REPORT/'support.json').read_bytes());populations=json.loads((REPORT/'population_build.json').read_bytes())['fixed_populations']
    stages=[('outer_tables',lambda:freeze_outer_tables(ROOT,'retina_atlas_disk3mm_v1',support,'data/atlas/disk3mm_typed_tables_v1')),
        ('outer_connections',lambda:freeze_outer_contacts(ROOT,json.loads((REPORT/'outer_tables.json').read_bytes()),'data/atlas/disk3mm_outer_connections_v1')),
        ('bc_tables',lambda:freeze_bc_tables(ROOT,json.loads((REPORT/'outer_tables.json').read_bytes()),support,'data/atlas/disk3mm_inner_v1')),
        ('midget_tables',lambda:freeze_midget_rgcs(ROOT,json.loads((REPORT/'bc_tables.json').read_bytes()),'data/atlas/disk3mm_inner_v1')),
        ('inner_tables',lambda:freeze_remaining_inner(ROOT,json.loads((REPORT/'outer_tables.json').read_bytes()),json.loads((REPORT/'bc_tables.json').read_bytes()),
            json.loads((REPORT/'midget_tables.json').read_bytes()),populations,support,'data/atlas/disk3mm_inner_v1')),
        ('aii_bc_connections',lambda:freeze_aii_bc_contacts(ROOT,json.loads((REPORT/'inner_tables.json').read_bytes()),populations,'data/atlas/disk3mm_global_routes_v2/AII_BC')),
        ('input_plan',lambda:freeze_input_plan(ROOT,FrozenDiskTable(ROOT,json.loads((REPORT/'outer_tables.json').read_bytes())['cones']),'data/atlas/disk3mm_input_plan_v1')),
        ('contact_geometry',build_contact_geometry),
        ('global_graph',lambda:freeze_disk_circuit(ROOT,'reports/disk3mm_build_20261016','data/atlas/disk3mm_circuit_v1'))]
    for name,callback in stages:
        save(progress_path,{'status':'BUILDING_'+name.upper(),'started_unix':started,'old_default_changed':False,'runtime_ready':False})
        result=callback();save(REPORT/(name+'.json'),result);print('GLOBAL_GEOMETRY_STAGE_FROZEN',name,flush=True)
    save(progress_path,{'status':'GLOBAL_GEOMETRY_FROZEN_PENDING_ASSEMBLY_AND_ISET_CAPTURE','started_unix':started,
        'old_default_changed':False,'runtime_ready':False,'physiological_validation':False})
    if (REPORT/'authorized_delivery_request.json').exists():
        save(progress_path,{'status':'RUNNING_AUTHORIZED_REPRESENTATIVE_ACCEPTANCE','started_unix':started,
            'old_default_changed':False,'runtime_ready':False,'physiological_validation':False})
        completed=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'tools/finish_disk3mm.py')],cwd=ROOT)
        delivery_path=REPORT/'final_delivery.json'
        delivery=json.loads(delivery_path.read_bytes()) if delivery_path.exists() else {
            'status':'BLOCKED_DELIVERY_PREFLIGHT','default_changed':False}
        save(progress_path,{'status':delivery['status'],'started_unix':started,
            'old_default_changed':delivery['default_changed'],'runtime_ready':delivery['default_changed'],
            'physiological_validation':False})
        if completed.returncode:raise subprocess.CalledProcessError(completed.returncode,completed.args)


def build_contact_geometry():
    from retina_sim.disk_population import FixedDiskPopulation
    from retina_sim.disk_tables import FrozenDiskTable
    from retina_sim.disk_contact_geometry import materialize_contacts
    populations=json.loads((REPORT/'population_build.json').read_bytes())['fixed_populations']
    primary=json.loads((REPORT/'primary_graph.json').read_bytes())['routes']
    tables={name:FixedDiskPopulation(ROOT,populations[name]) for name in ('rods','rod_BC','AII_AC')}
    cases=[('individual_rod_RBC_contacts','rods','rod_BC',primary['rod_RBC'],'OPL',None,'rod_membrane:'),
        ('RBC_AII_AMPA','rod_BC','AII_AC',primary['RBC_AII_AMPA'],'IPL',.03,'')]
    if (REPORT/'aii_bc_connections.json').exists():
        inner=json.loads((REPORT/'inner_tables.json').read_bytes())['tables']
        chemical=json.loads((REPORT/'aii_bc_connections.json').read_bytes())['chemical_routes']
        for cid,name in (('AII_FMB_glycine','midget_BC_OFF'),('AII_DB1_glycine','diffuse_BC_OFF')):
            tables[name]=FrozenDiskTable(ROOT,inner[name])
            cases.append((cid,'AII_AC',name,chemical['AII_'+name+'_glycine'],'IPL',.03,''))
    result={'contacts':{},'status':'CONSTRUCTING_FIXED_GLOBAL_CONTACT_GEOMETRY'}
    for cid,pre,post,binding,frame,radius,prefix in cases:
        def positions(table):return table.positions if isinstance(table,FixedDiskPopulation) else table.coordinates['ipl_preterminal']
        asset=materialize_contacts(ROOT,'data/atlas/disk3mm_contact_geometry_v1/'+cid,binding,
            positions(tables[pre]),positions(tables[post]),frame=frame,overlap_radius_mm=radius)
        result['contacts'][cid]={'binding':asset,'route':binding,'pre':pre,'post':post,'pre_ID_prefix':prefix,
            'contact_ID_rule':cid+':sha256(actual_global_pre_ID>actual_global_post_ID)[:32]'}
        save(REPORT/'contact_geometry.json',result);print('GLOBAL_CONTACT_GEOMETRY_FROZEN',cid,flush=True)
    result['status']='ALL_PRIMARY_CONTACT_GEOMETRY_FROZEN' if len(cases)==4 else 'PRIMARY_CONTACT_GEOMETRY_FROZEN_PENDING_AII_BC'
    save(REPORT/'contact_geometry.json',result);return result


def build_primary_graph():
    """Freeze global primary-rod routes against complete fixed population rows."""
    from scipy import sparse
    from retina_sim.disk_population import FixedDiskPopulation
    from retina_sim.disk_graph_storage import materialize_bounded_route,SparseRouteWriter
    catalog=json.loads((REPORT/'population_build.json').read_bytes())['fixed_populations']
    path=REPORT/'primary_graph.json'
    result=json.loads(path.read_bytes()) if path.exists() else {'routes':{},'complete':False}
    needed=('rod_BC','AII_AC')
    if not all(name in catalog for name in needed):raise RuntimeError('RBC/AII fixed master not ready')
    tables={name:FixedDiskPopulation(ROOT,catalog[name]) for name in needed}
    def write_route(name,pre,post,k):
        recipe={'route_id':name,'pre_population':pre,'post_population':post,'pre_master':tables[pre].binding,
            'post_master':tables[post].binding,'pre_count':len(tables[pre].positions),'post_count':len(tables[post].positions),
            'hard_support_mm':.04,'selection_budget':k,'gaussian_scale':None,
            'selection':'unchanged bounded_matrix; finite support before nearest selection; original row normalization',
            'deficits':'retained; no distant fill','source':'existing primary pathway project approximation',
            'population_generation':'one complete fixed disk; no ROI resampling'}
        def progress(state):
            save(REPORT/'primary_graph_status.json',{'route':name,'completed_post_rows':state['completed_post_rows'],
                'total_post_rows':recipe['post_count'],'edges':state['edges'],'empty_rows':state['empty_rows'],
                'missing_degree_rows':state['missing_degree_rows'],'status':'CONSTRUCTING'})
        binding=materialize_bounded_route(ROOT,'data/atlas/disk3mm_global_routes_v2/'+name,recipe,
            tables[pre].positions,tables[post].positions,k,on_block=progress)
        if name in result['routes'] and result['routes'][name]!=binding:
            result.setdefault('preserved_previous_route_bindings',{})[name]=result['routes'][name]
        result['routes'][name]=binding;save(path,result)
        print('GLOBAL_ROUTE_FROZEN',name,binding['sha256'],flush=True)
    write_route('RBC_AII_AMPA','rod_BC','AII_AC',3)
    # Preserve exactly the existing consecutive-master-order AII gap rule.
    aii=tables['AII_AC'];xy=aii.positions;n=len(xy)
    recipe={'route_id':'AII_AII_bidirectional_gap','pre_population':'AII_AC','post_population':'AII_AC',
        'pre_master':aii.binding,'post_master':aii.binding,'pre_count':n,'post_count':n,
        'selection':'only consecutive fixed master row pairs i,i+1, reciprocal; unchanged project rule',
        'hard_support_mm':.04,'conductance_S':'.2e-9*exp(-(distance/.03)^2)',
        'weight_unit':'S','normalization':False,'physiological_validation':False}
    writer=SparseRouteWriter(ROOT,'data/atlas/disk3mm_global_routes_v1/AII_AII_bidirectional_gap',recipe)
    if not writer.manifest.exists():
        left=np.arange(n-1);right=left+1;distance=np.linalg.norm(xy[left]-xy[right],axis=1);keep=distance<=.04
        left=left[keep];right=right[keep];g=.2e-9*np.exp(-(distance[keep]/.03)**2)
        matrix=sparse.csr_matrix((np.r_[g,g],(np.r_[left,right],np.r_[right,left])),shape=(n,n))
        for start in range(writer.next_row,n,4096):
            stop=min(n,start+4096);block=matrix[start:stop]
            rows=[{'actual_degree':int(degree),'hard_support_mm':.04,'missing_degree':0,
                'normalization':False,'source_rule':'consecutive_fixed_master_order_only'} for degree in np.diff(block.indptr)]
            writer.append(start,block,rows)
    result['routes']['AII_AII_bidirectional_gap']=writer.freeze();save(path,result)
    if 'rods' not in catalog:
        result.update(status='FROZEN_RBC_AII_AND_AII_GAPS_PENDING_RODS',complete=False);save(path,result)
        print(result['status'],flush=True);return
    tables['rods']=FixedDiskPopulation(ROOT,catalog['rods']);write_route('rod_RBC','rods','rod_BC',40)
    result.update(status='PRIMARY_ROD_GLOBAL_ROUTES_FROZEN_PENDING_CONE_BC_CONNECTIONS',complete=False,
        remaining=['AII_BC_glycine','AII_BC_reciprocal_gaps','global cone and inner routes','input and executable assembly'])
    save(path,result)

if __name__=='__main__':main()
