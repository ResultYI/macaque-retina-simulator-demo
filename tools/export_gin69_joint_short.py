"""Make saved joint smoke states readable by the existing desktop response importer."""
from pathlib import Path
import json,sys,shutil
import numpy as np
from scipy.io import loadmat

ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.gin69_joint_context import load,OUT
from retina_platform.response_bundle import open_result,export_package,sha
from retina_platform.response_data import ResponseData,RGC


def main():
    base=OUT/'executions/joint_short_01/forward/standard';dest=OUT/'joint_short_view';dest.mkdir(exist_ok=True)
    if any(p.is_file() for p in dest.rglob('*')):raise FileExistsError('Existing view assets retained')
    for sub in ['responses','cells','stimulus','provenance']:(dest/sub).mkdir(exist_ok=True)
    a,seed=load();fields=json.loads((base/'FIELDS.json').read_bytes())
    with np.load(base/'RESPONSE.npz',allow_pickle=False) as z:response={k:z[k].copy() for k in z.files}
    with np.load(base/'NUMERICAL_STATES.npz',allow_pickle=False) as z:states={k:z[k].copy() for k in z.files}
    response['side']=np.full(len(response['time_seconds']),'right',dtype='U5')
    geometry={};populations={};types={}
    for name,p in a.populations.items():
        n=p.group_count;gids=np.asarray(p.group_ids)
        for suffix,value in [('ids',gids),('members',p.counts),('xy',p.centroid_mm/.223),
            ('subtypes',a.types[name] if name in ('cones','HC') else np.full(n,name))]:geometry[name+'__'+suffix]=value
        available={k:m['unit'] for k,m in fields.items() if m.get('population')==name and k in response and response[k].shape[0]==n}
        for suffix in ['voltage','excitatory_conductance','inhibitory_conductance','expected_binary_count_rate']:
            choices=[k for k in available if k.endswith('__'+suffix)]
            if choices:
                preferred=next((k for k in choices if k.startswith('RGC__')),choices[0]);alias=name+'__'+suffix
                response[alias]=response[preferred];available[alias]=available[preferred]
        if name=='cones':
            key=next(k for k in states if k.startswith('STATE__cones__') and k.endswith('__Vis'))
            response['cones__native_voltage']=states[key][:,::10];available['cones__native_voltage']='vH_native_voltage'
        if name=='HC':
            h=np.empty((n,51))
            for suffix,rows in [('h1',a.h1_rows),('h2',a.h2_rows)]:
                key=next(k for k in states if k.startswith('STATE__HC__') and k.endswith('__'+suffix));h[rows]=states[key][:,::10]
            response['HC__native_voltage']=h;available['HC__native_voltage']='vH_native_voltage'
        primary=next((k for k in [name+'__voltage',name+'__native_voltage'] if k in available),None)
        if primary is None:primary=next((k for k in available if k.endswith('__voltage') or k.endswith('__inhibition_gate') or k.endswith('__inhibition_state')),None)
        if primary is None and available:primary=next(iter(available))
        populations[name]=dict(count=n,coordinate_role=p.coordinate_role,fields=available,primary=primary,
            representation='effective_functional_pool' if name.startswith(('same_','cross_')) else 'representative_group',
            missing=[] if available else ['No spatially mapped state saved for this population'])
        if name in RGC:types[name]=dict(group_ids=gids.tolist(),represented_members=p.counts.tolist())
    route_arrays={};routes=[]
    def route(name,pre,post,kind,matrix):
        matrix=matrix.tocsr(copy=True);matrix.eliminate_zeros()
        if not matrix.nnz:return
        token='r'+str(len(routes));routes.append(dict(name=name,pre=pre,post=post,kind=kind,token=token,shape=list(matrix.shape)))
        for suffix,value in [('indptr',matrix.indptr),('indices',matrix.indices),('weights',matrix.data)]:route_arrays[token+'_'+suffix]=value
    for name,r in a.routes.items():
        if name in seed['inner_parameters']['route_gains'] and seed['inner_parameters']['route_gains'][name]==0:continue
        kind='effective_functional_H1_pooling_and_original_H2' if r.pre=='cones' and r.post=='HC' else r.kind
        route(name,r.pre,r.post,kind,a.matrix('routes/'+name+'/mean_weights.npz'))
    q=-a.joint_H2_Q.copy();q.setdiag(0);q.eliminate_zeros();route('actual_H2_homotypic','HC','HC','effective_rate_per_second',q*50.)
    np.savez_compressed(dest/'cells/geometry.npz',**geometry);np.savez_compressed(dest/'cells/connections.npz',**route_arrays)
    np.savez_compressed(dest/'responses/chunk.npz',**response)
    identity=json.loads((base/'IDENTITY.json').read_bytes());identity['types']=types
    (dest/'responses/IDENTITY.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    shutil.copyfile(base/'GROUP_OBJECT_EVENTS.json',dest/'responses/GROUP_OBJECT_EVENTS.json')
    index=dict(chunks=[dict(file='chunk.npz',records=51,sha256=sha(dest/'responses/chunk.npz'))])
    (dest/'responses/RESPONSE_INDEX.json').write_text(json.dumps(index),encoding='utf-8')
    # Genuine source pre-live frame, explicitly reference-only; no fake stimulus movie.
    bank=loadmat(ROOT/'reports/GIN_69_6_RADIAL_CONDITIONAL_V1/candidate_v2/SOURCE_BANK.mat',variable_names=['source_rgb'])['source_rgb']
    np.savez_compressed(dest/'stimulus/reference.npz',rgb=bank[:,:,:,0][None,...])
    files={}
    for p in dest.rglob('*'):
        if p.is_file():files[p.relative_to(dest).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
    manifest=dict(schema='retina-response-bundle-1',status='complete',run_id=identity['identity'],
        stimulus=dict(frames='stimulus/reference.npz',frame_count=1,fps=1,duration_seconds=.05,
            name='Native-input 50ms probe; image is pre-live reference only, not a stimulus video',
            input_protocol=identity['controlled_input'],reference_image_only=True),
        positions=dict(joint=dict(geometry='cells/geometry.npz',connections='cells/connections.npz',
            response_prefix='responses/',populations=populations,routes=routes)),files=files,
        limitations=['Group objects are not independent member recordings','Static exterior grey; no peripheral dynamic validation','Source-native HC voltage has no calibrated mV mapping'],
        FIT_READY=False,background_qualified=False,production_adopted=False)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    source=open_result(dest,ROOT)
    try:
        data=ResponseData(source,'joint');target=next(x for x in data.rgcs if x[0]=='parasol_RGC_OFF' and str(data.geometry[x[0]]['ids'][x[1]]).endswith(':7730'))
        network,_=data.network(target)
        for name,meta in populations.items():
            if meta['primary']:data.field(name,meta['primary'])
        result=dict(status='DESKTOP_RESPONSE_PARSER_PASS',time_nodes=len(data.time),RGC_objects=len(data.rgcs),target=list(target),
            ancestors={k:len(v) for k,v in network.items()},reference_image_only=True,rendering_verified=False)
        package=export_package(source,OUT/'GIN69_joint_short.response.zip');result['package']=str(package)
        (dest/'PARSER_CHECK.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
    finally:source.close()


if __name__=='__main__':main()
