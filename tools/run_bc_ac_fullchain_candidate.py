"""Independent BC/AC adapter dry-run or explicitly contracted full forward; no fit."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.bc_ac_fullchain_candidate import read,dry_run,forward
from retina_sim.group_time_storage import load_bundle
from tools.validate_ac_vs import write,Worker
from retina_sim.background_checkpoint import MemoryGuard
import math

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['dry-run','forward']);p.add_argument('--config',default='configurations/calibration/bc_ac_fullchain_candidate_v1.json')
    p.add_argument('--output',required=True);p.add_argument('--drive-bundle');p.add_argument('--values');args=p.parse_args();out=Path(args.output).resolve()
    allowed=ROOT/'reports/bc_ac_fullchain_adapter_v1'
    if not out.is_relative_to(allowed) or out.exists():raise ValueError('Fresh independent adapter output required')
    out.mkdir(parents=True);config=read(ROOT/args.config)
    try:
        if args.mode=='dry-run':result=dry_run(ROOT,config)
        else:
            if not args.drive_bundle or not args.values:raise ValueError('Frozen native/rod drive and exact active-values files required')
            drive=load_bundle(args.drive_bundle,expected_kind='validation_evidence')['payload'];values=read(args.values)
            limits=config['launch']['forward_resource_contract']
            keys=['single_worker_wall_seconds','process_RSS_limit_bytes','available_RAM_floor_bytes']
            if not isinstance(limits,dict) or any(k not in limits or isinstance(limits[k],bool) or not math.isfinite(limits[k]) or limits[k]<=0 for k in keys):
                raise ValueError('Frozen positive finite worker resource contract required')
            worker=Worker(out);worker.resources=dict(worker.resources,**{k:limits[k] for k in keys})
            worker.memory=MemoryGuard(worker.resources['process_RSS_limit_bytes'],worker.resources['available_RAM_floor_bytes'])
            with worker:result=forward(ROOT,config,drive,values,out,guard=worker.check)
        write(out/'result.json',result);print(json.dumps(dict(status=result['status'],model_runs=result.get('model_runs',1 if args.mode=='forward' else 0),output=str(out))))
    except Exception as exc:write(out/'failure.json',dict(error=type(exc).__name__,reason=str(exc),mode=args.mode));raise

if __name__=='__main__':main()
