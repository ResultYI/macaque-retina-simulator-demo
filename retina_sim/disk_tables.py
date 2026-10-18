"""Typed global cell arrays with canonical IDs evaluated for selected rows only."""
from pathlib import Path
import json,os,hashlib
import numpy as np
from .atlas import CanonicalCellTable,COORDINATE_ROLES
from .contracts import ContractError,Quantity,canonical_hash
from .disk_graph_storage import file_sha,save_json

SCHEMA='frozen_global_typed_cell_arrays_v1'
META=('atlas_id','atlas_version','source_version','population','reference_id')

def write_table(root,directory,metadata,count,key_batch,coordinates,*,extras=None,source_bindings=None):
    """key_batch(start,stop) supplies the exact canonical source keys, no new sampler."""
    root=Path(root).resolve();directory=(root/directory).resolve()
    if not directory.is_relative_to(root):raise ContractError('Typed table outside project')
    directory.mkdir(parents=True,exist_ok=True);path=directory/'manifest.json'
    metadata={key:metadata[key] for key in META}
    if path.exists():
        existing=FrozenDiskTable(root,{'path':path.relative_to(root).as_posix(),'sha256':file_sha(path)})
        if existing.metadata!=metadata or existing.count!=count or existing.record['source_bindings']!=(source_bindings or {}):
            raise ContractError('Existing typed table identity differs; retained')
        if set(coordinates)!=set(existing.coordinates) or any(not np.array_equal(q.values,existing.coordinates[role]) for role,q in coordinates.items()):
            raise ContractError('Existing typed coordinates differ; retained')
        for role,q in coordinates.items():
            recorded=existing.record['coordinates'][role]
            if (q.unit,q.meaning,q.coordinate_frame,q.evidence_tag)!=(recorded['unit'],recorded['meaning'],recorded['frame'],recorded['evidence_tag']):
                raise ContractError('Existing typed coordinate declaration differs; retained')
        if set(extras or {})!=set(existing.extras) or any(not np.array_equal(np.asarray(value),existing.extras[name]) for name,value in (extras or {}).items()):
            raise ContractError('Existing typed auxiliary arrays differ; retained')
        for start in range(0,count,8192):
            stop=min(count,start+8192);batch=tuple(key_batch(start,stop))
            frozen=tuple(key.decode('ascii') for key in existing.keys[start:stop])
            if batch!=frozen:raise ContractError('Existing typed canonical source keys differ; retained')
        return existing
    if any(directory.iterdir()):raise ContractError('Partial typed table retained; inspect before replacement')
    files={};keys=directory/'source_keys.npy'
    # The largest existing canonical source key is below this fixed engineering
    # storage width. Reject overflows instead of silently truncating identities.
    key_array=np.lib.format.open_memmap(keys,mode='w+',dtype='S512',shape=(count,))
    seen=set()
    for start in range(0,count,8192):
        stop=min(count,start+8192);batch=tuple(key_batch(start,stop))
        if len(batch)!=stop-start or any(not isinstance(key,str) or not key or len(key.encode('ascii'))>512 for key in batch):
            raise ContractError('Typed canonical source key missing/non-ASCII/too long')
        for key in batch:
            identity=hashlib.sha256(key.encode('ascii')).digest()
            if identity in seen:raise ContractError('Duplicate canonical source key in fixed typed population')
            seen.add(identity)
        key_array[start:stop]=batch
    key_array.flush();del key_array
    files['source_keys']={'path':keys.relative_to(root).as_posix(),'sha256':file_sha(keys)}
    roles={}
    for role,quantity in coordinates.items():
        if role not in COORDINATE_ROLES:raise ContractError('Unknown typed coordinate role')
        meaning,frame=COORDINATE_ROLES[role];quantity.require('mm',meaning,(count,2),frame)
        target=directory/(role+'.npy');np.save(target,quantity.values,allow_pickle=False)
        roles[role]={'path':target.relative_to(root).as_posix(),'sha256':file_sha(target),'unit':'mm',
            'meaning':meaning,'frame':frame,'evidence_tag':quantity.evidence_tag}
    extra_files={}
    for name,value in (extras or {}).items():
        array=np.asarray(value)
        if array.dtype.hasobject or array.shape[:1]!=(count,) or '/' in name or '\\' in name:
            raise ContractError('Invalid typed-table auxiliary array')
        target=directory/(name+'.npy');np.save(target,array,allow_pickle=False)
        extra_files[name]={'path':target.relative_to(root).as_posix(),'sha256':file_sha(target)}
    record={'schema':SCHEMA,'metadata':metadata,'count':count,'source_keys':files['source_keys'],
        'coordinates':roles,'extras':extra_files,'source_bindings':source_bindings or {},
        'regenerate_on_ROI':False,'physiological_validation':False}
    record['logical_identity']=canonical_hash(record);save_json(path,record)
    return FrozenDiskTable(root,{'path':path.relative_to(root).as_posix(),'sha256':file_sha(path)})

class FrozenDiskTable:
    def __init__(self,root,binding):
        self.root=Path(root).resolve();self.binding=dict(binding);path=(self.root/binding['path']).resolve()
        if not path.is_relative_to(self.root) or file_sha(path)!=binding['sha256']:raise ContractError('Typed global table identity mismatch')
        self.record=json.loads(path.read_bytes());r=self.record;copy=dict(r);identity=copy.pop('logical_identity')
        if r['schema']!=SCHEMA or r['regenerate_on_ROI'] is not False or canonical_hash(copy)!=identity:
            raise ContractError('Unsupported typed global table contract')
        self.metadata=r['metadata'];self.count=r['count'];self.keys=self._array(r['source_keys'])
        self.coordinates={role:self._array(value) for role,value in r['coordinates'].items()}
        self.extras={name:self._array(value) for name,value in r['extras'].items()}
        if self.keys.shape!=(self.count,) or any(a.shape!=(self.count,2) for a in self.coordinates.values()):
            raise ContractError('Typed global table dimensions differ')
    def _array(self,binding):
        path=(self.root/binding['path']).resolve()
        if not path.is_relative_to(self.root) or file_sha(path)!=binding['sha256']:raise ContractError('Typed global array identity mismatch')
        return np.load(path,mmap_mode='r',allow_pickle=False)
    def rows(self,indices):
        indices=np.asarray(indices,dtype=np.int64)
        if np.any(indices<0) or np.any(indices>=self.count):raise ContractError('Typed row outside fixed population')
        coordinates={role:Quantity(array[indices],value['unit'],value['meaning'],value['evidence_tag'],value['frame'])
            for role,value in self.record['coordinates'].items() for array in [self.coordinates[role]]}
        return CanonicalCellTable(**self.metadata,source_keys=tuple(key.decode('ascii') for key in self.keys[indices]),coordinates=coordinates)
    def ids(self,start,stop):return self.rows(np.arange(start,stop)).global_cell_ids
    def query_indices(self,roi,*,owner=False):
        if roi.reference_id!=self.metadata['reference_id'] or roi.coordinate_role not in self.coordinates:
            raise ContractError('Typed coordinate query absent/different reference; no fallback')
        selected=[];array=self.coordinates[roi.coordinate_role]
        for start in range(0,self.count,65536):
            values=array[start:start+65536];mask=roi.owner_mask(values) if owner else roi.query_mask(values)
            selected.extend((np.flatnonzero(mask)+start).tolist())
        return np.asarray(selected,dtype=np.int64)
