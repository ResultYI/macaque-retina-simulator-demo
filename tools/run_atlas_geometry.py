"""Render the GUI/Run's exact frozen population artifact, without regeneration."""
from pathlib import Path
import argparse
import json
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retina_platform.population_preview import export_population_preview

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--center',nargs=2,type=float,default=(.35,0.))
    parser.add_argument('--size',nargs=2,type=float,default=(.02,.02))
    parser.add_argument('--halo-mm',type=float,default=.04)
    parser.add_argument('--max-cells',type=int,default=60000)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    x,y=args.center;w,h=args.size
    destination=args.output or ROOT/'reports'/'population_preview'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    result=export_population_preview(ROOT,destination,(x-w/2,x+w/2,y-h/2,y+h/2),
        halo_mm=args.halo_mm,max_cells=args.max_cells)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
