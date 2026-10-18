"""Decompose the saved short trace; no model execution or new profiling."""
from pathlib import Path
from collections import defaultdict
import argparse,json


def summarize(path):
    events=json.loads(Path(path).read_bytes())['traceEvents']
    cpu=[e for e in events if e.get('ph')=='X' and e.get('cat') in ('cpu_op','user_annotation')]
    by_thread=defaultdict(list)
    for e in cpu:by_thread[(e['pid'],e['tid'])].append(e)
    ancestors={};external={}
    for values in by_thread.values():
        stack=[]
        for e in sorted(values,key=lambda e:(e['ts'],-e['dur'])):
            while stack and stack[-1]['ts']+stack[-1]['dur']<=e['ts']+1e-6:stack.pop()
            key=e.get('args',{}).get('External id')
            if key is not None:
                ancestors[key]=[p['name'] for p in stack]+[e['name']];external[key]=e
            stack.append(e)
    # Original pinned partition: inner NumPy RHS -> FP64 upload/cat -> one
    # isfinite/all/item. This interval locates only the final joint scan.
    joint=[]
    roots=sorted([e for e in cpu if e['cat']=='cpu_op'],key=lambda e:e['ts'])
    for inner in (e for e in cpu if e['name']=='CPU_INNER_NUMBA'):
        start=inner['ts']+inner['dur']
        finite=next(e for e in roots if e['ts']>=start and e['name']=='aten::isfinite')
        item=next(e for e in roots if e['ts']>=finite['ts']+finite['dur'] and e['name']=='aten::item')
        joint.append((finite['ts'],item['ts']+item['dur']))
    checks={'aten::isfinite','aten::any','aten::all','aten::item','aten::_local_scalar_dense',
            'aten::lt','aten::le','aten::__or__','aten::__and__','aten::bitwise_or','aten::bitwise_and'}
    totals=defaultdict(lambda:{'kernel_ms':0.,'kernel_count':0,'copy_ms':0.,'copy_count':0,'copy_bytes':0})
    missing=0;copies=defaultdict(lambda:{'calls':0,'bytes':0,'device_ms':0.})
    for e in events:
        if e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset'):continue
        key=e.get('args',{}).get('External id');names=ancestors.get(key,[]);source=external.get(key)
        label=next((name for name in names if name.startswith('HC_SPMV_')),None)
        if label is None and 'HC_NATIVE' in names:
            label='HC_native_domain_and_output_checks' if checks.intersection(names) else 'HC_native_elementwise_math'
        if label is None and source and any(lo<=source['ts']<hi for lo,hi in joint):
            label='redundant_complete_joint_finite_check'
        if label is None:label='joint_integrator_transfer_and_other'
        if not names:missing+=1
        row=totals[label]
        if e['cat']=='kernel':row['kernel_ms']+=e['dur']/1000.;row['kernel_count']+=1
        else:
            row['copy_ms']+=e['dur']/1000.;row['copy_count']+=1;row['copy_bytes']+=e.get('args',{}).get('bytes',0)
            detail=copies[e['name']];detail['calls']+=1;detail['bytes']+=e.get('args',{}).get('bytes',0);detail['device_ms']+=e['dur']/1000.
    runtime=defaultdict(lambda:{'calls':0,'CPU_span_ms':0.})
    for e in events:
        if e.get('cat')!='cuda_runtime':continue
        if 'Synchronize' not in e['name'] and 'Memcpy' not in e['name']:continue
        row=runtime[e['name']];row['calls']+=1;row['CPU_span_ms']+=e['dur']/1000.
    return {'exclusive_device_activity_groups':dict(totals),'device_copy_kinds':dict(copies),'joint_scan_intervals':len(joint),
        'unassigned_external_ids':missing,'CUDA_runtime_sync_or_copy_CPU_spans':dict(runtime),
        'interpretation':'Saved instrumented trace only. Each device activity counted once. Native comparison/isfinite/Boolean/reduction ops classified as checks; remaining native non-SpMV ops as elementwise math. Joint scan intervals follow the pinned partition call order. CPU synchronization/copy spans include GPU waiting and are not additive to device activity time or full-window time.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args()
    result=summarize(args.directory/'short_diagnostic_trace.json')
    for name in ('diagnosis.json','result.json'):
        path=args.directory/name;value=json.loads(path.read_bytes())
        target=value if name=='diagnosis.json' else value['diagnosis']
        target['saved_trace_breakdown']=result
        path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
