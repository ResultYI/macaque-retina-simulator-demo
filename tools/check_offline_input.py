"""Validate one offline NPZ against the active assembly and emit its source summary."""
from pathlib import Path
import argparse,sys,json

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('input',type=Path);parser.add_argument('--summary',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from retina_platform.config import RunConfig
    from retina_sim.atlas_population_network import atlas_population_network
    from retina_sim.offline_input import load_offline_inputs
    config=RunConfig.project_default(root);light=config.light_context()
    compiled,_,audit=atlas_population_network(root,light_context=light,owner_bounds_mm=config.owner_bounds(),
        halo_mm=config.halo_mm,max_spatial_cells=config.max_spatial_cells,seed=config.seed)
    bundle=load_offline_inputs(args.input,compiled,light,reference_id=audit['atlas_runtime']['source_atlas']['reference_id'])
    result=json.dumps(bundle.source_summary,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
    if args.summary:
        with args.summary.open('x',encoding='utf8') as stream:stream.write(result)
    print(result,end='')

if __name__=='__main__':main()
