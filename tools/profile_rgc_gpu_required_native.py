"""One registered optional readout-elimination comparison; never starts long fitting."""
from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from retina_sim.ac_vs_gpu_native_rhs import bind_required_native
from retina_sim.rgc_stage1_cache import write
from tools import validate_rgc_stage1_gpu as check

OUT=ROOT/'reports/rgc_stage1_gpu_adaptation_v1'

if __name__=='__main__':
    check.VAL=OUT/'native_rhs_required_v1';check.bind_gpu=bind_required_native
    check.run('gpu')
    baseline=OUT/'validation_v4/gpu';candidate=check.VAL/'gpu'
    br=json.loads((baseline/'RESULT.json').read_text(encoding='utf-8'));cr=json.loads((candidate/'RESULT.json').read_text(encoding='utf-8'))
    equal=[]
    with np.load(baseline/'traces.npz') as b,np.load(candidate/'traces.npz') as c:
        if b.files!=c.files:raise RuntimeError('Optional adapter changed recording registry')
        equal=[dict(field=k,bitwise_equal=bool(np.array_equal(b[k],c[k]))) for k in b.files]
    bm=json.loads((baseline/'trace_metadata.json').read_text(encoding='utf-8'));cm=json.loads((candidate/'trace_metadata.json').read_text(encoding='utf-8'))
    passed=all(r['bitwise_equal'] for r in equal) and bm['stamps']==cm['stamps']
    result=dict(status='OPTIONAL_NATIVE_RHS_READOUT_COMPARISON',same_source_trace_exact=passed,fields=equal,
        baseline_hot_seconds=br['hot_seconds'],candidate_hot_seconds=cr['hot_seconds'],single_pair_wall_ratio=br['hot_seconds']/cr['hot_seconds'],
        CPU_reference_hot_seconds=4.322010699979728,candidate_faster_than_CPU=cr['hot_seconds']<4.322010699979728,
        phases={name:dict(baseline=br['runtime_metrics'][name],candidate=cr['runtime_metrics'][name]) for name in ['native_wall_seconds','inner_wall_seconds','snapshot_wall_seconds']},
        source_pins={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['retina_sim/ac_vs_gpu_native_rhs.py','tools/profile_rgc_gpu_required_native.py']},
        default_adopted=False,new_candidate_restore_not_run=True,long_run_started=False,qualification='one short pair, no stable throughput claim; baseGPU restore qualification remains separate')
    write(OUT/'NATIVE_RHS_REDUNDANCY_RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='fields'}))
    if not passed:raise RuntimeError('Optional within-call elimination changed original GPU traces; candidate not adopted')
