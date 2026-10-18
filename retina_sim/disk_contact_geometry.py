"""Fixed contact anchors/overlap in global CSR entry order, without string bloat."""
from pathlib import Path
import json
import numpy as np
from .contracts import ContractError,canonical_hash
from .disk_graph_storage import FrozenSparseRoute,array_identity,file_sha,save_json

def materialize_contacts(root,directory,route_binding,pre,post,*,frame,overlap_radius_mm=None):
    root=Path(root).resolve();directory=(root/directory).resolve()
    if not directory.is_relative_to(root):raise ContractError('Contact geometry outside project')
    directory.mkdir(exist_ok=True,parents=True);route=FrozenSparseRoute(root,route_binding);r=route.record['recipe']
    pre=np.asarray(pre);post=np.asarray(post)
    if pre.shape!=(r['pre_count'],2) or post.shape!=(r['post_count'],2) or frame not in ('OPL','IPL'):
        raise ContractError('Frozen contact endpoint layout/frame differs')
    recipe={'route':route_binding,'pre_coordinates':array_identity(pre),'post_coordinates':array_identity(post),
        'position_formula':'(pre_global_xy+post_global_xy)/2','coordinate_frame':frame,
        'overlap_radius_mm':overlap_radius_mm,'overlap_formula':'exp(-(distance/radius)^2)' if overlap_radius_mm is not None else None,
        'entry_order':'block order, then CSR post row and canonical global pre column order',
        'state_ID':'existing component prefix+sha256(global pre ID>global post ID)[:32], evaluated only for selected states',
        'no_neuron_soma_claim':True,'physiological_validation':False}
    manifest=directory/'manifest.json';progress=directory/'construction.json';n=route.record['edges']
    if manifest.exists():
        value=FixedContactGeometry(root,{'path':manifest.relative_to(root).as_posix(),'sha256':file_sha(manifest)})
        if value.record['recipe']!=recipe:raise ContractError('Frozen contact recipe differs; retained')
        return value.binding
    state={'recipe_hash':canonical_hash(recipe),'completed_blocks':0,'completed_edges':0}
    if progress.exists():
        state=json.loads(progress.read_bytes())
        if state['recipe_hash']!=canonical_hash(recipe):raise ContractError('Partial contact recipe differs; retained')
    else:save_json(progress,state)
    path=directory/'midpoint_mm.npy'
    positions=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float64,shape=(n,2))
    overlap=None;overlap_path=directory/'physical_overlap.npy'
    if overlap_radius_mm is not None:
        if not np.isfinite(overlap_radius_mm) or overlap_radius_mm<=0:raise ContractError('Positive explicit overlap radius required')
        overlap=np.lib.format.open_memmap(overlap_path,mode='r+' if overlap_path.exists() else 'w+',dtype=np.float64,shape=(n,))
    offset=state['completed_edges']
    for index in range(state['completed_blocks'],len(route.record['blocks'])):
        block=route.record['blocks'][index];matrix,_,_=route.block(block);matrix=matrix.tocoo()
        a=pre[matrix.col];b=post[matrix.row+block['start']];stop=offset+matrix.nnz
        positions[offset:stop]=(a+b)/2
        if overlap is not None:overlap[offset:stop]=np.exp(-(np.linalg.norm(a-b,axis=1)/overlap_radius_mm)**2)
        positions.flush()
        if overlap is not None:overlap.flush()
        state.update(completed_blocks=index+1,completed_edges=stop);save_json(progress,state);offset=stop
    if offset!=n:raise ContractError('Contact entry count differs from frozen route')
    del positions
    if overlap is not None:del overlap
    value={'schema':'frozen_global_contact_arrays_v1','recipe':recipe,'count':n,
        'positions':{'path':path.relative_to(root).as_posix(),'sha256':file_sha(path)},
        'overlap':None if overlap_radius_mm is None else {'path':overlap_path.relative_to(root).as_posix(),'sha256':file_sha(overlap_path)},
        'regenerate_on_ROI':False,'physiological_validation':False}
    save_json(manifest,value);return {'path':manifest.relative_to(root).as_posix(),'sha256':file_sha(manifest)}

class FixedContactGeometry:
    def __init__(self,root,binding):
        self.root=Path(root).resolve();self.binding=dict(binding);path=(self.root/binding['path']).resolve()
        if not path.is_relative_to(self.root) or file_sha(path)!=binding['sha256']:raise ContractError('Contact geometry identity differs')
        self.record=json.loads(path.read_bytes());r=self.record
        if r['schema']!='frozen_global_contact_arrays_v1' or r['regenerate_on_ROI'] is not False:raise ContractError('Unknown frozen contact geometry')
        self.positions=self._array(r['positions']);self.overlap=None if r['overlap'] is None else self._array(r['overlap'])
        if self.positions.shape!=(r['count'],2) or self.overlap is not None and self.overlap.shape!=(r['count'],):
            raise ContractError('Frozen contact array dimensions differ')
    def _array(self,binding):
        path=(self.root/binding['path']).resolve()
        if not path.is_relative_to(self.root) or file_sha(path)!=binding['sha256']:raise ContractError('Frozen contact array changed')
        return np.load(path,mmap_mode='r',allow_pickle=False)

def global_contact_rows(route,post_indices):
    selected=np.unique(np.asarray(post_indices,dtype=np.int64));result=[];offset=0
    for block in route.record['blocks']:
        rows=selected[(selected>=block['start'])&(selected<block['stop'])]
        if len(rows):
            matrix,_,_=route.block(block)
            for row in rows-block['start']:
                result.extend(range(offset+int(matrix.indptr[row]),offset+int(matrix.indptr[row+1])))
        offset+=block['edges']
    return np.asarray(result,dtype=np.int64)
