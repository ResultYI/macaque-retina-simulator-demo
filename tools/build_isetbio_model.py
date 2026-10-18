"""Build/freeze a local circuit using an explicit existing cone atlas binding.

Never generates a cone mosaic or advances neural/background time. Defaults keep
the historical binding; migration names independent configuration/output paths.
"""
from pathlib import Path
import argparse
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.atlas import AtlasROI
from retina_sim.atlas_connections import ConnectionAtlas
from retina_sim.population_revision_graph import revise_local_graph
from retina_sim.atlas_population_network import atlas_population_network
from retina_sim.population_catalog import register_existing_masters,SharedPopulationCatalog
from retina_sim.circuit_geometry import freeze_circuit_geometry

TARGETS=('midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n',encoding='utf8')
def project_path(p):
    p=(ROOT/p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Build artifacts must stay inside project')
    return p

def verify(configuration,report_dir,run_config=None):
    config=json.loads(configuration.read_bytes());binding=config['frozen_graph']
    record=json.loads((ROOT/binding['artifact']).read_bytes())
    kwargs={} if run_config is None else dict(duration=run_config.duration_seconds,light_context=run_config.light_context(),
        owner_bounds_mm=run_config.owner_bounds(),halo_mm=run_config.halo_mm,max_spatial_cells=run_config.max_spatial_cells,seed=run_config.seed)
    c,_,audit=atlas_population_network(ROOT,execution_profile='atlas_isetbio_geometry_v1',
        configuration_path=configuration.relative_to(ROOT).as_posix(),**kwargs)
    result={'status':'ASSEMBLY_PASS','components':len(c.components),
        'state_blocks':sum(len(x.state_specs) for x in c.components.values()),
        'populations':{k:len(v.cells.cell_id) for k,v in c.populations.items()},
        'graph':binding,'geometry':record['summary'],'model_geometry':record['isetbio_geometry'],
        'owner_target_contract':audit['atlas_runtime']['owner_target_contract'],
        'empty_owner_outputs':audit['atlas_runtime'].get('empty_owner_outputs',{}),
        'frozen_circuit_geometry':audit['atlas_runtime']['frozen_typed_geometry'],
        'compiled_geometry_identity':audit['atlas_runtime']['compiled_geometry_identity'],
        'parameter_profile':'effective_dynamics_v1','background_or_neural_integration_runs':0,
        'claim':'Executable assembly only; no new physiological validation'}
    save(report_dir/'verification.json',result)
    print('MODEL_ASSEMBLY_PASS',len(c.components),flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--config',type=Path,help='Shared RunConfig JSON for verification; construction uses explicit atlas/output bindings')
    p.add_argument('--configuration',type=Path,default=Path('configurations/atlas_isetbio_geometry_v1.json'))
    p.add_argument('--atlas-profile',default='retina_atlas_isetbio_macaque_v1')
    p.add_argument('--output-dir',type=Path,default=Path('data/atlas/isetbio_macaque_cones_v1'))
    p.add_argument('--report-dir',type=Path,default=Path('reports/isetbio_model_20261015'))
    p.add_argument('--reuse-catalog',type=Path,help='Existing independent non-cone masters; source/coverage checked, no regeneration')
    p.add_argument('--rebuild-uncovered-masters',action='store_true',help='Offline migration only: rebuild uncovered finite domains using unchanged density/sampling laws')
    args=p.parse_args();run_config=None
    if args.config:
        from retina_platform.config import RunConfig
        run_config=RunConfig.from_dict(json.loads(project_path(args.config).read_bytes()))
        if not args.verify_only or run_config.execution_profile!='atlas_isetbio_geometry_v1':
            raise ValueError('--config verifies an existing frozen ISET circuit; use explicit construction bindings to build')
        args.configuration=Path(run_config.circuit_configuration_path or 'configurations/atlas_isetbio_geometry_v1.json')
    configuration=project_path(args.configuration);out=project_path(args.output_dir)
    report=project_path(args.report_dir);report.mkdir(exist_ok=True,parents=True)
    if args.verify_only:return verify(configuration,report,run_config)
    if configuration.exists() or (out/'local_population_graph.json').exists():
        raise RuntimeError('Existing configuration/graph retained; use --verify-only or new paths')
    atlas=ConnectionAtlas(ROOT,atlas_profile_id=args.atlas_profile,s_partition_profile='deMonasterio1985_mulatta_effective_S_v1')
    roi=AtlasROI((.34,.36,-.01,.01),(.30,.40,-.05,.05),atlas.atlas.config['reference_id'])
    record=atlas.preview(roi,max_cells=60000);print('CONE_PREVIEW',record['summary'],flush=True)
    catalog=None
    if args.reuse_catalog:
        path=project_path(args.reuse_catalog)
        catalog=SharedPopulationCatalog(ROOT,{'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)})
        if catalog.reference_id!=roi.reference_id:raise ValueError('Independent population reference differs')
    if args.rebuild_uncovered_masters and catalog is None:raise ValueError('Explicit old catalog required for selective rebuilding')
    record=revise_local_graph(ROOT,connection_record=record,atlas_profile_id=args.atlas_profile,population_masters=catalog,
        rebuild_uncovered_masters=args.rebuild_uncovered_masters)
    record['isetbio_geometry']={'master':dict(atlas.atlas.config['cone_master']),'atlas_profile':args.atlas_profile,
        'midget_RF_rule':'connected_cone_centroid_isetbio_v1','midget_independent_density':'NOT_IMPORTED; existing private/Wool route retained',
        'physiology_parameters_changed':False,'rod_RBC_AII_equations_retained':True,'parasol_SBC_sources_changed':False,
        'independent_population_geometry':'Existing fixed masters reused if recipe matches; dependent finite domains rebuilt under unchanged approved density/sampling laws',
        'old_input_background_caches_valid':False,'cell_identity_policy':'new cone namespace; no old-to-new mapping asserted',
        'variable_ROI_scope':'Existing finite owner and halo only; no peripheral expansion'}
    out.mkdir(exist_ok=True,parents=True);graph_path=out/'local_population_graph.json';save(graph_path,record)
    binding={'artifact':graph_path.relative_to(ROOT).as_posix(),'sha256':sha(graph_path)}
    catalog_binding=register_existing_masters(ROOT,record,out/'population_catalog.json')
    config={'schema_version':'1.0','profile_id':'atlas_isetbio_geometry_v1','atlas_profile':args.atlas_profile,
        'frozen_graph':binding,'owner_bounds_mm':list(roi.owner_bounds_mm),'halo_mm':.04,
        'status':'geometry_built_pending_assembly','background_prepared':False,'population_master_catalog':catalog_binding,
        'migration':{'status':'CANDIDATE_NOT_DEFAULT','canonical_target_outputs':list(TARGETS),
            'current_empty_owner_outputs':[],'ready_for_single_default_switch':False,
            'approved_boundary_contract':'finite_open_boundary; no exterior edges, compensation or crop renormalization'}}
    save(configuration,config)
    # Bootstrap invokes the unchanged spatial laws once; normal assembly then
    # requires the frozen result and cannot silently regenerate it.
    compiled,_,audit=atlas_population_network(ROOT,execution_profile='atlas_isetbio_geometry_v1',
        configuration_path=configuration.relative_to(ROOT).as_posix(),_freeze_geometry_bootstrap=True)
    contract=audit['atlas_runtime']['owner_target_contract']
    config['migration'].update(current_empty_owner_outputs=[k for k,v in contract['status'].items() if v=='EMPTY_SAMPLE'],owner_target_contract=contract)
    compiled.typed_view_domain['empty_owner_outputs']=tuple(config['migration']['current_empty_owner_outputs'])
    config['typed_circuit_geometry']=freeze_circuit_geometry(ROOT,out/'circuit_geometry.json',record,binding,compiled)
    config['status']='assembly_pass_no_background';save(configuration,config)
    verify(configuration,report)

if __name__=='__main__':main()
