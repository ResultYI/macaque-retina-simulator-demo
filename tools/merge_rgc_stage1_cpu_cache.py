"""Read-only registry of complete same-contract CPU caches; no short/GPU mixing."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True

def inspect(directory,spec,case):
    from retina_sim.contracts import ContractError
    from retina_sim.rgc_stage1_cache import read_cache
    import numpy as np
    arrays,manifest=read_cache(directory,spec,case)
    try:
        expected=case['samples_100us']+1
        if manifest['case']!=case or manifest['recorded']!=expected or manifest['bins']!=case['samples_100us']//10:
            raise ContractError('Full original case/time coverage differs')
        if not np.array_equal(arrays['time_source_s'],np.arange(expected)*.0001):raise ContractError('Original100us sample clock differs')
        for pop,selection in spec['recording'].items():
            for field in ['V','gE','gI']:
                value=arrays[pop+':'+field]
                if value.shape!=(expected,len(selection['rows'])) or value.dtype!=np.float64 or not np.isfinite(value).all():
                    raise ContractError('Full native endogenous field shape/dtype/domain differs')
        return dict(case_id=case['case_id'],cache_identity=manifest['cache_identity'],directory=str(Path(directory).resolve()),
            payload_sha256=manifest['arrays_sha256'],backend='original reference CPU',readonly=True)
    finally:arrays.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',action='append',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    from tools.fit_rgc_stage1 import verify
    from retina_sim.rgc_stage1_cache import write
    spec=verify();found=[];missing=[]
    for case in spec['cases']:
        matches=[Path(root)/case['case_id'] for root in args.root if (Path(root)/case['case_id']/'manifest.json').exists()]
        if len(matches)>1:raise RuntimeError('Ambiguous multiple caches require explicit source selection')
        if not matches:missing.append(case['case_id']);continue
        found.append(inspect(matches[0],spec,case))
    output=Path(args.output)
    if output.exists():raise RuntimeError('Fresh registry output required')
    write(output,dict(schema='same-scientific-contract-cpu-cache-registry-1',caches=found,missing=missing,
        all_six_complete=not missing,short_prefixes_accepted=False,GPU_caches_accepted=False,FIT_READY=False))

if __name__=='__main__':main()
