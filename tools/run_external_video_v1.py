"""Run one frozen4s continuous video IVP; old first-frame results remain independent."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_external_first_frame_v1 import run
from tools.cross_position_context_v1 import read
from retina_sim.contracts import ContractError


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--position',choices=['near','far'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prep=a.preparation.resolve();out=a.output.resolve();protocol=read(prep/'PREPARATION.json')['protocol']
    if protocol['stop_seconds']!=4. or protocol['input_frame_indices']!=list(range(600)):
        raise ContractError('Only existing frozen0..4s/120-frame source scope is authorized here')
    if not out.is_relative_to(ROOT/'reports'):raise ContractError('Fresh reports output required')
    # Single-frame measured30.84/27.70s include10s setup: continuous4s ~42/36min.
    #6000s is a finite safeguard above that measured estimate, not a new model parameter.
    run(prep,a.position,out,max_wall_seconds=6000)
