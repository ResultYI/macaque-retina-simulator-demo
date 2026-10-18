"""Convert expected ISET cone capture using a supplied fixed reference and valid rod/background packet."""
from pathlib import Path
import argparse,sys,json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path)
    p.add_argument('--template',type=Path,required=True);p.add_argument('--reference-profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--summary',type=Path)
    p.add_argument('--config',type=Path,help='RunConfig JSON shared with GUI and run_retina.py; otherwise current project default')
    args=p.parse_args();root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from retina_platform.config import RunConfig
    from retina_sim.atlas_population_network import atlas_population_network
    from retina_sim.isetbio_input import convert_isetbio_capture,project_rf_equals_is_binding,read_isetbio_capture
    cfg=RunConfig.from_dict(json.loads(args.config.read_bytes())) if args.config else RunConfig.project_default(root);light=cfg.light_context()
    compiled,_,audit=atlas_population_network(root,light_context=light,owner_bounds_mm=cfg.owner_bounds(),
        halo_mm=cfg.halo_mm,max_spatial_cells=cfg.max_spatial_cells,seed=cfg.seed,
        execution_profile=cfg.execution_profile,configuration_path=cfg.circuit_configuration_path or None)
    ref=audit['atlas_runtime']['source_atlas']['reference_id'];meta=read_isetbio_capture(args.capture)['manifest']
    result=convert_isetbio_capture(args.capture,args.template,args.output,compiled,light,reference_id=ref,
        reference_profile=json.loads(args.reference_profile.read_text(encoding='utf8')),
        cone_is_binding=project_rf_equals_is_binding(compiled.populations[meta['cones']['population_id']],reference_id=ref))
    if args.summary:
        with args.summary.open('x',encoding='utf8') as f:json.dump(result.source_summary,f,ensure_ascii=False,allow_nan=False,indent=2)
    print(json.dumps({'status':'PASS','output':str(args.output),'channels':list(result.channels),
        'physiological_calibration':False,'GUI_connected':False}))

if __name__=='__main__':main()
