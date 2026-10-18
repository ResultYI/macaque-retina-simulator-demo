"""Use the authorized official human cone lattice without resampling/warping.

The paused macaque generator/checkpoint is never touched. Other fixed populations,
dynamics and project pooling rules are retained; cone-dependent assets are new.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
REPORT=ROOT/'reports/human_disk3mm_build_20261017'
PREVIOUS=ROOT/'reports/disk3mm_build_20261016'
ASSET=ROOT/'data/atlas/isetbio_human_borrowed_cones_disk3mm_v1'
ATLAS='retina_atlas_human_borrowed_disk3mm_v1'
CIRCUIT='configurations/atlas_human_borrowed_disk3mm_v1.json'
PREFIX='data/atlas/human_disk3mm_'
from retina_sim.disk_graph_storage import save_json,file_sha


def import_official():
    from retina_sim.isetbio_geometry import (FixedISETConeMaster,cone_apertures,
        HUMAN_SCHEMA,HUMAN_ASSET_SHA256,HUMAN_TRANSFORM,COMMIT)
    REPORT.mkdir(parents=True,exist_ok=True)
    profile=json.loads((ROOT/'configurations/iset_lcd_apple_mw_v1.json').read_bytes())
    vendor=Path(profile['isetbio_root'])
    source=vendor/'data/sdr/isetbio-mosaics/cone_lattices/right/nominal-fov-58deg.mat'
    assert file_sha(source)==HUMAN_ASSET_SHA256,'Official cone cache checksum differs; no replacement'
    manifest=json.loads((source.parent.parent/'manifest.json').read_bytes())
    entry=next(x for x in manifest['files'] if x['path']=='right/nominal-fov-58deg.mat')
    assert entry['sha256']==HUMAN_ASSET_SHA256 and entry['coordinate_system']=='retinal microns'
    raw=loadmat(source,variable_names=['rfPositions','whichEye','neuronType','fovDegs'],squeeze_me=True)
    assert raw['whichEye']=='right eye' and raw['neuronType']=='cones'
    original=np.asarray(raw['rfPositions']);all_xy=-original.astype(np.float64)/1000.
    support=json.loads((PREVIOUS/'support.json').read_bytes());bounds=support['limits']['complete_cone_query_bounds_mm']
    x0,x1,y0,y1=bounds
    rows=np.flatnonzero((all_xy[:,0]>=x0)&(all_xy[:,0]<x1)&(all_xy[:,1]>=y0)&(all_xy[:,1]<y1))
    xy=all_xy[rows]
    assert np.all(np.isfinite(xy)) and len(xy)>0 and not np.any(np.all(xy==0,axis=1))
    assert np.unique(xy,axis=0).shape==xy.shape,'Official source contains duplicate points; no jitter/delete repair'
    tree=cKDTree(xy);distances=tree.query(xy,k=7)[0]
    radius=support['limits']['cone_RF_query_radius_mm'];active=np.sum(xy**2,axis=1)<=radius**2
    margin=np.min(np.c_[xy[:,0]-x0,x1-xy[:,0],xy[:,1]-y0,y1-xy[:,1]],axis=1)
    assert np.all(distances[active,-1]<margin[active]),'Aperture neighbors exceed complete imported support'
    apertures=cone_apertures(xy)
    keys=np.array(['sdr_human_54eb0a9ef905_row:'+str(i) for i in rows])
    ASSET.mkdir(parents=True,exist_ok=False)
    payload=ASSET/'positions.npz'
    np.savez_compressed(payload,positions_mm=xy,source_keys=keys,official_source_rows=rows,
                        aperture_diameters_um=apertures)
    value={'schema':HUMAN_SCHEMA,'isetbio_commit':COMMIT,'geometry_route':'official_human_borrowed_approximation_v1',
        'population':'cones','density_species':'Homo sapiens','human_density_used':True,
        'position_role':'rf_input_inner_segment_project_transfer','coordinate_transform':HUMAN_TRANSFORM,
        'official_asset':{'collection':'cone_lattices','path':'right/nominal-fov-58deg.mat','eye':'right eye',
            'nominal_FOV_degrees':58,'actual_MAT_FOV_degrees':float(raw['fovDegs']),
            'sha256':HUMAN_ASSET_SHA256,'bytes':source.stat().st_size,'source_count':len(original),
            'deposit':'https://purl.stanford.edu/vm447vf0975','stored_precision':str(original.dtype)},
        'density_origin':'official default human Curcio/Watson lattice, not re-fitted or reprojected to Packer density',
        'complete_query_bounds_mm':bounds,'count':len(xy),
        'payload':{'path':payload.relative_to(ROOT).as_posix(),'sha256':file_sha(payload)},
        'axes':'+x nasal/+y superior project RF frame; official loader negation and unit conversion only; cross-species planar registration is an approximation',
        'apertures':'existing0.79 times median of six actual neighbor spacings; rebuilt on complete imported support',
        'LMS':'No types supplied in official position asset; retain existing macaque-effective absolute S macro quotas/Packer scores and named L:M1:1 assignment on actual candidates',
        'claims':{'measured_individual_mosaic':False,'physiological_validation':False,'whole_macaque_geometry':False,
                  'resampling':False,'density_scale_or_warp':False,'full_network_ready':False},
        'diagnostics':{'owner_cones_RF_disk3mm':int(np.sum(xy[:,0]**2+xy[:,1]**2<=9)),
            'minimum_NN_um':float(distances[:,1].min()*1000),
            'NN_quantiles_um':np.percentile(distances[:,1]*1000,[0,1,50,99,100]).tolist(),
            'all_active_aperture_neighbors_inside_complete_support':True,
            'source_positions_exact_after_official_transform':bool(np.array_equal(xy,-original[rows].astype(np.float64)/1000.))}}
    save_json(ASSET/'manifest.json',value)
    binding={'path':(ASSET/'manifest.json').relative_to(ROOT).as_posix(),'sha256':file_sha(ASSET/'manifest.json')}
    master=FixedISETConeMaster(ROOT,binding);np.testing.assert_array_equal(master.positions,xy)
    cfg=json.loads((ROOT/'configurations/retina_atlas_isetbio_source_stop_v1.json').read_bytes())
    cfg.update(mosaic_rule='isetbio_fixed_human_borrowed_cone_master_v1',cone_master=binding,
        HC_master_configuration='configurations/atlas_hc_disk3mm_v1.json',
        cone_geometry_route='official_human_borrowed_approximation_v1',
        density_rule_role='Retained Packer provider only for existing S-selection scores/Wool project mapping; does not define imported human positions',
        claim='Official human Curcio/Watson cone positions borrowed unchanged on the composite macaque project frame; other approved project pathways retained; not fully macaque physiological geometry')
    cfgpath=ROOT/'configurations'/(ATLAS+'.json')
    if cfgpath.exists():raise FileExistsError('Human atlas profile already exists; retain existing binding')
    save_json(cfgpath,cfg)
    for name in ('support.json','population_build.json','primary_graph.json'):
        save_json(REPORT/name,json.loads((PREVIOUS/name).read_bytes()))
    save_json(REPORT/'cone_master.json',dict(binding,**value['diagnostics'],count=len(xy),
        geometry_route=value['geometry_route'],paused_macaque_snapshot=json.loads((PREVIOUS/'authorized_delivery_request.json').read_bytes())['macaque_paused_snapshot']))
    print('OFFICIAL_HUMAN_CONES_IMPORTED',json.dumps(value['diagnostics']),len(xy),flush=True)


def contacts():
    from retina_sim.disk_population import FixedDiskPopulation
    from retina_sim.disk_tables import FrozenDiskTable
    from retina_sim.disk_contact_geometry import materialize_contacts
    original=json.loads((PREVIOUS/'contact_geometry.json').read_bytes())
    result={'contacts':dict(original['contacts']),'status':'FROZEN_GLOBAL_CONTACT_GEOMETRY_HUMAN_CONE_DEPENDENCIES'}
    population=json.loads((REPORT/'population_build.json').read_bytes())['fixed_populations']
    aii=FixedDiskPopulation(ROOT,population['AII_AC'])
    inner=json.loads((REPORT/'inner_tables.json').read_bytes())['tables']
    routes=json.loads((REPORT/'aii_bc_connections.json').read_bytes())['chemical_routes']
    for cid,name in (('AII_FMB_glycine','midget_BC_OFF'),('AII_DB1_glycine','diffuse_BC_OFF')):
        post=FrozenDiskTable(ROOT,inner[name]);route=routes['AII_'+name+'_glycine']
        binding=materialize_contacts(ROOT,PREFIX+'contact_geometry_v1/'+cid,route,aii.positions,
            post.coordinates['ipl_preterminal'],frame='IPL',overlap_radius_mm=.03)
        result['contacts'][cid]={'binding':binding,'route':route,'pre':'AII_AC','post':name,'pre_ID_prefix':'',
            'contact_ID_rule':cid+':sha256(actual_global_pre_ID>actual_global_post_ID)[:32]'}
    return result


def geometry():
    from retina_sim.disk_source_geometry import freeze_outer_tables
    from retina_sim.disk_outer import freeze_outer_contacts
    from retina_sim.disk_inner import freeze_bc_tables,freeze_midget_rgcs,freeze_remaining_inner,freeze_aii_bc_contacts
    from retina_sim.disk_input import freeze_input_plan
    from retina_sim.disk_tables import FrozenDiskTable
    from retina_sim.disk_circuit import freeze_disk_circuit
    read=lambda name:json.loads((REPORT/(name+'.json')).read_bytes())
    support=read('support');populations=read('population_build')['fixed_populations'];started=time.time()
    stages=[('outer_tables',lambda:freeze_outer_tables(ROOT,ATLAS,support,PREFIX+'typed_tables_v1')),
        ('outer_connections',lambda:freeze_outer_contacts(ROOT,read('outer_tables'),PREFIX+'outer_connections_v1')),
        ('bc_tables',lambda:freeze_bc_tables(ROOT,read('outer_tables'),support,PREFIX+'inner_v1')),
        ('midget_tables',lambda:freeze_midget_rgcs(ROOT,read('bc_tables'),PREFIX+'inner_v1')),
        ('inner_tables',lambda:freeze_remaining_inner(ROOT,read('outer_tables'),read('bc_tables'),read('midget_tables'),
            populations,support,PREFIX+'inner_v1',atlas_profile=ATLAS)),
        ('aii_bc_connections',lambda:freeze_aii_bc_contacts(ROOT,read('inner_tables'),populations,PREFIX+'routes_v1/AII_BC')),
        ('input_plan',lambda:freeze_input_plan(ROOT,FrozenDiskTable(ROOT,read('outer_tables')['cones']),PREFIX+'input_plan_v1')),
        ('contact_geometry',contacts),
        ('global_graph',lambda:freeze_disk_circuit(ROOT,REPORT.relative_to(ROOT),PREFIX+'circuit_v1',atlas_profile=ATLAS,configuration_path=CIRCUIT))]
    for name,callback in stages:
        save_json(REPORT/'geometry_pipeline.json',{'status':'BUILDING_'+name.upper(),'started_unix':started,'default_changed':False})
        save_json(REPORT/(name+'.json'),callback());print('HUMAN_GEOMETRY_STAGE_FROZEN',name,flush=True)
    save_json(REPORT/'geometry_pipeline.json',{'status':'HUMAN_GEOMETRY_FROZEN_PENDING_ACTUAL_RUNTIME_ACCEPTANCE',
        'default_changed':False,'physiological_validation':False})


def prepare_delivery():
    """Bind the already-authorized delivery to this explicit human route."""
    previous=json.loads((PREVIOUS/'authorized_delivery_request.json').read_bytes())
    if previous['status']!='PAUSED_FOR_AUTHORIZED_HUMAN_CONE_REUSE':
        raise ValueError('The prior macaque automatic route must remain paused')
    authorization=previous['human_geometry_route_authorization'];snapshot=previous['macaque_paused_snapshot']
    if not snapshot['verified'] or file_sha(ROOT/snapshot['path']/'manifest.json')!=snapshot['manifest_sha256']:
        raise ValueError('Verified immutable macaque snapshot changed')
    request={key:previous[key] for key in ('schema','python_executable','media_path','rollback_directory','adoption_identity_guards')}
    request.update(status='APPROVED',authorization=authorization,macaque_paused_snapshot=snapshot,
        circuit_configuration_path=CIRCUIT,atlas_profile_id=ATLAS,
        validation_output='reports/human_disk3mm_validation_v1',
        geometry_claim='Official human Curcio/Watson cone coordinates borrowed unchanged with explicit project registration; retained frozen macaque-effective pathways',
        frozen_identity_guards={path:expected for path,expected in previous['frozen_identity_guards'].items()
            if not path.startswith('tools/')})
    request['focused_regressions']={'legacy_isetbio_geometry':{'passed':13},
        'explicit_human_species_asset_transform_rejections':{'passed':1},
        'disk3mm_structural_input_and_identity_suite':{'passed':15},
        'earlier_background_clock_rng_continuation':{'passed':2},
        'scope':'Observed in this authorized session; fixture/self-consistency checks, not physiological validation'}
    paths=[p for folder in ('retina_sim','retina_platform') for p in (ROOT/folder).glob('*.py')]
    paths += [ROOT/'tools'/name for name in ('build_human_disk3mm.py','finish_disk3mm.py','check_disk3mm.py',
        'disk3mm_response_check.py','export_disk_mixed_batch.m','export_disk_mixed_reference.m','export_mixed_reference.m')]
    paths += [ROOT/CIRCUIT,ROOT/'configurations'/(ATLAS+'.json'),ASSET/'manifest.json',ASSET/'positions.npz',
        ROOT/'configurations/iset_lcd_apple_mw_v1.json']
    before=json.loads((REPORT/'optical_profile_before.json').read_bytes())
    after=json.loads((ROOT/'configurations/iset_lcd_apple_mw_v1.json').read_bytes())
    if {k:v for k,v in before.items() if k!='meaning'}!={k:v for k,v in after.items() if k!='meaning'}:
        raise ValueError('Only optical species-scope description may differ; physical coefficients/paths retained')
    request['optical_profile_metadata_migration']={'changed_field':'meaning only','physical_parameters_changed':False,
        'previous_sha256':file_sha(REPORT/'optical_profile_before.json')}
    for path in paths:request['frozen_identity_guards'][path.relative_to(ROOT).as_posix()]=file_sha(path)
    target=REPORT/'authorized_delivery_request.json'
    if target.exists():raise FileExistsError('Existing human delivery authorization retained; reconcile changes explicitly')
    save_json(target,request);print('HUMAN_DELIVERY_IDENTITIES_BOUND',flush=True)


def midget_floor_revision():
    """User-approved minimum change; preserve all unrelated frozen geometry."""
    from retina_sim.disk_inner import freeze_midget_rgcs,route_recipe
    from retina_sim.disk_graph_storage import FrozenSparseRoute,materialize_bounded_route
    from retina_sim.disk_tables import FrozenDiskTable
    from retina_sim.disk_circuit import freeze_disk_circuit
    output=ROOT/'reports/human_disk3mm_midget_floor_20261017'
    output.mkdir(parents=True,exist_ok=True)
    read=lambda name:json.loads((REPORT/(name+'.json')).read_bytes())
    for name in ('support','population_build','primary_graph','outer_tables','outer_connections','bc_tables',
                 'aii_bc_connections','input_plan','contact_geometry','cone_master'):
        target=output/(name+'.json');value=read(name)
        if target.exists() and json.loads(target.read_bytes())!=value:raise ValueError('Preserved source binding changed')
        save_json(target,value)
    save_json(output/'authorization.json',{'UTC':'2026-10-17T14:57:00Z',
        'user_evidence':'Sentinel_bbe883c54de8819190a2dec058a45fdb: 同意',
        'scope':'floor minimum with original n>=1, actual partners/support/continuous target/deficits retained; existing transition real input/background/response only; central HC unresolved; no whole-disk default adoption',
        'macaque5855_snapshot_untouched':True})
    prefix='data/atlas/human_disk3mm_midget_floor_v1'
    value=freeze_midget_rgcs(ROOT,read('bc_tables'),prefix,minimum_degree_rule='floor_minimum_keep_continuous_deficit_v1')
    save_json(output/'midget_tables.json',value);print('MIDGET_FLOOR_TABLES_FROZEN',flush=True)
    original=read('inner_tables');inner=dict(original,tables=dict(original['tables']),routes=dict(original['routes']))
    inner['tables'].update(value['tables']);inner['routes'].update(value['routes'])
    inner['midget_blocked_candidates']=value['blocked_candidates']
    for polarity,ac,upstream in (('ON','local_crossover_AC','diffuse_BC_OFF'),('OFF','local_feedforward_AC','diffuse_BC_ON')):
        name='midget_RGC_'+polarity;pre=FrozenDiskTable(ROOT,inner['tables'][ac]);post=FrozenDiskTable(ROOT,inner['tables'][name])
        enabled=FrozenSparseRoute(ROOT,original['routes'][upstream+'->'+ac]).ready_rows()
        sigma=.02+.06*np.linalg.norm(post.coordinates['rf'],axis=1);route=ac+'->'+name
        recipe=route_recipe(route,pre,post);recipe.update(selection_budget=1,exact_required_degree=True,
            pre_eligibility='ready upstream typed drive only',gaussian_rule='existing sigma by target RF')
        inner['routes'][route]=materialize_bounded_route(ROOT,prefix+'/inhibitory_routes/'+name,recipe,
            pre.coordinates['ipl_preterminal'],post.coordinates['ipl_post_arbor'],1,gaussian_scale=sigma,
            pre_eligible=np.flatnonzero(enabled),exact_degree=True)
    inner.update(status='GLOBAL_INNER_FLOOR_MINIMUM_REVISION_CENTRAL_HC_UNRESOLVED',
        midget_minimum_degree_rule='floor_minimum_keep_continuous_deficit_v1',population_average_target_satisfied='NOT_ESTABLISHED',
        unchanged_nonmidget_source_catalog_sha256=file_sha(REPORT/'inner_tables.json'))
    save_json(output/'inner_tables.json',inner)
    binding=freeze_disk_circuit(ROOT,output.relative_to(ROOT),prefix+'/circuit',atlas_profile=ATLAS,
        configuration_path='configurations/atlas_human_borrowed_disk3mm_floor_v1.json')
    save_json(output/'global_graph.json',binding)
    save_json(output/'geometry_pipeline.json',{'status':'FLOOR_MINIMUM_FROZEN_TRANSITION_RUNTIME_PENDING',
        'global_graph':binding,'default_changed':False,'central_HC_owner_gaps_retained':7770,
        'blocked_midget_candidates':{k:v['count'] for k,v in value['blocked_candidates'].items()},
        'regenerated_cone_HC_or_other_population_masters':False,'population_average_target_satisfied':'NOT_ESTABLISHED'})
    print('FLOOR_MINIMUM_GLOBAL_GRAPH_FROZEN',flush=True)


def refresh_floor_input():
    """Explicit exporter-identity revision; frozen cone positions are unchanged."""
    from retina_sim.disk_tables import FrozenDiskTable
    from retina_sim.disk_input import freeze_input_plan
    report=ROOT/'reports/human_disk3mm_midget_floor_20261017'
    old=json.loads((ROOT/'configurations/atlas_human_borrowed_disk3mm_floor_v1.json').read_bytes())
    path=ROOT/old['global_graph']['path']
    if file_sha(path)!=old['global_graph']['sha256']:raise ValueError('Original floor graph identity changed')
    graph=json.loads(path.read_bytes());cone=FrozenDiskTable(ROOT,graph['tables']['cones']['binding'])
    plan=freeze_input_plan(ROOT,cone,'data/atlas/human_disk3mm_input_plan_v2')
    graph['source_profiles']=dict(graph['source_profiles'],input_plan=plan)
    graph['input_exporter_identity_revision']={'previous_plan':old['input_plan'],'new_plan':plan,
        'reason':'Exact emitted-spectrum identity with one fixed foveal gain; scene-size luminance rounding does not trigger tile refits',
        'cone_positions_types_apertures_changed':False}
    target=ROOT/'data/atlas/human_disk3mm_midget_floor_v1/circuit_input_v2/manifest.json';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and json.loads(target.read_bytes())!=graph:raise ValueError('Existing input revision retained')
    save_json(target,graph);binding={'path':target.relative_to(ROOT).as_posix(),'sha256':file_sha(target)}
    cfg=dict(old,global_graph=binding,input_plan=plan)
    cfgpath=ROOT/'configurations/atlas_human_borrowed_disk3mm_floor_input_v2.json'
    if cfgpath.exists() and json.loads(cfgpath.read_bytes())!=cfg:raise ValueError('Existing configuration revision retained')
    save_json(cfgpath,cfg);save_json(report/'input_identity_revision_v2.json',{'configuration':cfgpath.relative_to(ROOT).as_posix(),
        'global_graph':binding,'input_plan':plan,'old_actual_failure_retained':True,'default_changed':False})
    print('FLOOR_INPUT_IDENTITY_REVISION_FROZEN',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['import','geometry','prepare-delivery','midget-floor','refresh-input']);args=parser.parse_args()
    if args.stage=='import':import_official()
    elif args.stage=='geometry':geometry()
    elif args.stage=='prepare-delivery':prepare_delivery()
    elif args.stage=='midget-floor':midget_floor_revision()
    else:refresh_floor_input()
