"""Runnable whole-layer software fixture; outputs are explicitly synthetic."""
from pathlib import Path
import sys,argparse,json,hashlib
from dataclasses import is_dataclass,fields
from datetime import datetime,timezone
from enum import Enum
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse
from examples.full_network_fixture import full_fixture
from retina_sim.recording import RecordingPlan,NPZWindowRecorder

def serializable(value):
    if sparse.issparse(value):
        a=sparse.csr_matrix(value);return {'format':'CSR','shape':list(a.shape),'rows':'post','columns':'pre','data':a.data.tolist(),'indices':a.indices.tolist(),'indptr':a.indptr.tolist()}
    if is_dataclass(value):return {f.name:serializable(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Enum):return value.value
    if isinstance(value,dict):return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serializable(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError('Unserializable fixture configuration: '+type(value).__name__)

def main():
    parser=argparse.ArgumentParser(description='Synthetic architecture integration fixture, NOT macaque physiological calibration')
    parser.add_argument('--duration',type=float,default=.006,help='Absolute stop seconds, 0 < duration <= .02 prescribed gaze coverage')
    parser.add_argument('--window',type=float,default=.002,help='Bounded output window in seconds')
    parser.add_argument('--plugin-boundary-fixture',action='store_true',help='Engineering ABI/delay example only; rod/RBC/AII physiology remains unmodeled')
    parser.add_argument('--output',type=Path,default=None);args=parser.parse_args()
    if not 0<args.duration<=.02 or args.window<=0:parser.error('Duration/window outside declared fixture contract')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=args.output or ROOT/'reports/synthetic_full_network'/stamp
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    if args.plugin_boundary_fixture:
        from examples.plugin_boundary_fixture import plugin_fixture
        compiled,channels,builder,spec=plugin_fixture(ROOT)
        aux={'claim':'Engineering plugin ABI/delay fixture only; rod/RBC/AII physiology NOT MODELED','builder_estimate':builder.estimate(spec),'displacement':{}}
    else:compiled,channels,aux=full_fixture(ROOT,presynaptic_loop=True)
    recorder=NPZWindowRecorder(output/'windows')
    state=compiled.stream(compiled.initialize(),channels,args.duration,args.window,recorder,recording_plan=RecordingPlan(max_bytes=32*1024*1024))
    checkpoint=compiled.checkpoint(state);(output/'final_checkpoint.json').write_text(checkpoint,encoding='utf-8')
    manifest={'fixture_class':'synthetic_fixture','biological_calibration':False,'independent_validation':'NOT_RUN','claim':aux['claim'],'model_id':compiled.spec.model_id,'assembly_hash':compiled.assembly_hash,'input_contracts':{k:v.input_contract for k,v in channels.items()},'population_records':[p.record() for p in compiled.populations.values()],'components':serializable(compiled.spec.components),'links':serializable(compiled.spec.links),'component_parameter_evidence':{cid:serializable(c.parameters.records) for cid,c in compiled.components.items()},'builder_resource_estimate':aux['builder_estimate'],'displacement_diagnostics':aux['displacement']}
    (output/'fixture_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    events=0;ports=set()
    for path in sorted((output/'windows').glob('*.npz')):
        with np.load(path,allow_pickle=False) as data:
            record=json.loads(str(data['manifest_json']));ports.update(record['ports'])
            events+=sum(len(e) for e in record['metadata']['spike_events'].values())
    summary={'fixture_class':'synthetic_fixture','fixture_scope':aux['claim'],'clock_contract':compiled.clock,'connection_delay_buffer_count':len(state.link_delay_states),'finished_utc':datetime.now(timezone.utc).isoformat(),'stop_seconds':state.time_seconds,'window_count':recorder.windows,'explicit_cell_count':sum(len(p.cells.cell_id) for p in compiled.populations.values()),'executed_component_count':len(compiled.components),'regions':sorted(set(r for p in compiled.populations.values() for r in p.regions)),'cone_subtypes':sorted(set(compiled.populations['cones'].cells.subtype)) if 'cones' in compiled.populations else [],'recorded_ports_and_states':sorted(ports),'observed_synthetic_spike_events':events,'history_sampling_index':state.sample_index,'checkpoint_sha256':hashlib.sha256(checkpoint.encode()).hexdigest(),'animal_data_fits_run':0,'independent_physiological_validation':'NOT_RUN','physiological_completion':False,'engineering_execution':'PASS','full_target':'Source-specific physiological population/profile bindings still required; fixture does not satisfy full target'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'windows':recorder.windows,'components':len(compiled.components),'cells':summary['explicit_cell_count'],'fixture_class':'synthetic_fixture','physiological_completion':False},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
