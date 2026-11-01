"""Bounded arithmetic and real frozen-layout regression, without model integration."""
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools import cell6_runtime_monitor_v3 as candidate
from tools.cell6_runtime_monitor_v2 import Cell6Monitor as Reference
from tools.cell6_halo_context_v1 import load
from retina_sim.contracts import canonical_hash
from retina_sim.group_time_storage import load_bundle


def main():
    began=time.perf_counter();rng=np.random.Generator(np.random.PCG64(20261031));cases=0
    for length in (0,1,2,7,50,466,6517):
        for _ in range(4):
            a=rng.normal(size=length);b=rng.normal(size=length)
            if length>=7:a[:7]=[0.,-0.,np.inf,-np.inf,np.nan,1.,1.];b[:7]=[-0.,0.,-np.inf,np.inf,1.,np.nan,1.]
            expected=np.maximum(a,b);actual=a.copy();candidate.maximum_into(actual,b)
            if not np.array_equal(actual.view('u8'),expected.view('u8')):raise RuntimeError('IEEE statistic mismatch')
            cases+=length
    assembly,seed=load();root=ROOT/'reports/CELL6_RAW_REBUILD_CONDITIONAL_READOUT_V1/runtime_monitor_repair_v1/cache/g1'
    payload=load_bundle(root/'CHECKPOINT_0065000',expected_kind='validation_evidence')['payload']['monitor']
    a=Reference.from_payload(assembly,seed['states'],payload,parameters=seed['inner_parameters'])
    b=candidate.Cell6Monitor.from_payload(assembly,seed['states'],payload,parameters=seed['inner_parameters'])
    golden=ROOT/'reports/CELL6_COMPILED_INNER_V1/paired_v1/reference_STATES.npz'
    with np.load(golden,allow_pickle=False) as z:states={tuple(k.split('__')[1:]):z[k][:,0].copy() for k in z.files if k.startswith('STATE__')}
    rhs={k:v*.001 for k,v in states.items()};transitions={k:states[k].copy() for k in a.transition_keys}
    before=canonical_hash(dict(states=states,rhs=rhs,transitions=transitions))
    for monitor in (a,b):monitor.observe(6.5,'right',states,rhs,transitions=transitions,event_generation=0)
    if canonical_hash(a.payload())!=canonical_hash(b.payload()):raise RuntimeError('Real-layout original monitor statistics differ')
    if canonical_hash(a.report(.0001))!=canonical_hash(b.report(.0001)):raise RuntimeError('Original monitor decision differs')
    if canonical_hash(dict(states=states,rhs=rhs,transitions=transitions))!=before:raise RuntimeError('Monitor modified model buffers')
    result=dict(passed=True,scalar_cases=cases,real_frozen_checkpoint_layout=True,no_model_advance=True,
        original_payload_bitwise_equal=True,original_report_bitwise_equal=True,model_input_buffers_unchanged=True,
        candidate=candidate.evidence(),wall_seconds=time.perf_counter()-began,
        no_long_pressure_test=True,root_cause_proven=False)
    print(json.dumps(result))
    out=ROOT/'reports/CELL6_COMPILED_INNER_V1/MONITOR_V3_REGRESSION.json';out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
