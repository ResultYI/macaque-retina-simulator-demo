"""Inspect an opt-in scope/granularity capability without starting a model."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.computation_resolution import resolve

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scope',choices=['local_patch','full_domain'],default='local_patch')
    p.add_argument('--granularity',choices=['coarse_grouped','per_cell'],default='coarse_grouped');args=p.parse_args()
    config=json.loads((ROOT/'configurations/computation_resolution_v1.json').read_text(encoding='utf8'))
    print(json.dumps(resolve(ROOT,config,args.scope,args.granularity),indent=2))

if __name__=='__main__':main()
