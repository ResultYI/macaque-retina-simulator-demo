"""Lossless multi-population/route storage of a frozen graph, not a generator.

Only descriptor/index metadata is eager. Population row and edge blocks are
loaded when selected. The original JSON SHA and logical hash remain separate
from the immutable storage manifest hash.
"""
from collections.abc import Mapping,Sequence
from pathlib import Path
from io import BytesIO
import gzip,hashlib,json
import numpy as np
from .contracts import ContractError,canonical_hash

SCHEMA='frozen_retina_graph_blocks_v1'

def _raw(value):return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf8')

def convert_graph(source,destination,*,expected_source_sha256,rows_per_block=256,edges_per_block=512):
    """One full read of the immutable source; no sampling, ID allocation or fitting."""
    source=Path(source);raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_source_sha256:raise ContractError('Frozen source graph SHA mismatch')
    record=json.loads(raw);destination=Path(destination);destination.mkdir(parents=True,exist_ok=False);ordinal=0;files={};counts={}
    def write(value,suffix='json.gz',binary=False):
        nonlocal ordinal
        ordinal+=1;name=f'part_{ordinal:06d}.'+suffix
        data=value if binary else gzip.compress(_raw(value),mtime=0)
        (destination/name).write_bytes(data);files[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)};return name
    def table(value,path):
        n=len(value['global_cell_ids']);roles=value['coordinates'];counts['/'.join(path)]=n;blocks=[]
        for start in range(0,n,rows_per_block):
            stop=min(n,start+rows_per_block)
            data={role:v['values'][start:stop] for role,v in roles.items()}
            boxes={role:[float(np.min(a,axis=0)[0]),float(np.max(a,axis=0)[0]),float(np.min(a,axis=0)[1]),float(np.max(a,axis=0)[1])] for role,a in data.items() if len(a)}
            blocks.append({'start':start,'stop':stop,'file':write(data),'bounds':boxes})
        return {'kind':'table','count':n,'header':write({k:v for k,v in value.items() if k not in ('global_cell_ids','source_keys','coordinates')}),
            'index':write({'global_cell_ids':value['global_cell_ids'],'source_keys':value['source_keys']}),
            'coordinates':{k:{x:y for x,y in v.items() if x!='values'} for k,v in roles.items()},'blocks':blocks}
    def edges(value,pre_key,post_key):
        blocks=[]
        for start in range(0,len(value),edges_per_block):
            blocks.append({'start':start,'stop':min(len(value),start+edges_per_block),'file':write(value[start:start+edges_per_block])})
        stream=BytesIO();np.savez_compressed(stream,pre=np.asarray([v[pre_key] for v in value],dtype=str),post=np.asarray([v[post_key] for v in value],dtype=str))
        return {'kind':'edges','count':len(value),'index':write(stream.getvalue(),'npz',True),'blocks':blocks}
    split_paths={(),('HC',),('inner_tables',),('inner_contacts',),('inner_parents',),('rod_inner',),('rod_inner','tables'),('rod_inner','typed_points')}
    def encode(value,path=()):
        if isinstance(value,dict) and all(k in value for k in ('global_cell_ids','source_keys','coordinates','population')):return table(value,path)
        if isinstance(value,dict) and all(k in value for k in ('edges','pre_ids','post_ids')):
            return {'kind':'contact','header':write({k:v for k,v in value.items() if k!='edges'}),'edges':edges(value['edges'],'pre_id','post_id')}
        if path==('HC_candidate_edges',):return edges(value,'cone_id','hc_id')
        if isinstance(value,dict) and path in split_paths:return {'kind':'map','children':{k:encode(v,path+(k,)) for k,v in value.items()}}
        return {'kind':'json','file':write(value)}
    descriptor=encode(record);manifest={'schema':SCHEMA,'source_json_sha256':expected_source_sha256,'logical_record_hash':canonical_hash(record),
        'source_json_bytes':len(raw),'population_counts':counts,'descriptor':descriptor,'files':files,
        'rules':'original row/edge order and values; lossless storage only; no density/shape/weight/parameter changes'}
    manifest['storage_manifest_hash']=canonical_hash(manifest)
    (destination/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8')
    return {'path':str(destination),'source_json_sha256':expected_source_sha256,'logical_record_hash':manifest['logical_record_hash'],
        'storage_manifest_hash':manifest['storage_manifest_hash'],'files':len(files),'compressed_payload_bytes':sum(f['bytes'] for f in files.values())}

class _Map(Mapping):
    def __init__(self,store,desc):self.store=store;self.desc=desc;self.cache={}
    def __iter__(self):return iter(self.desc['children'])
    def __len__(self):return len(self.desc['children'])
    def __getitem__(self,key):
        if key not in self.cache:self.cache[key]=self.store.decode(self.desc['children'][key])
        return self.cache[key]

class _Column(Sequence):
    def __init__(self,table,role):self.table=table;self.role=role
    def __len__(self):return self.table.desc['count']
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        if index<0:index+=len(self)
        if index<0 or index>=len(self):raise IndexError(index)
        block=next(b for b in self.table.desc['blocks'] if b['start']<=index<b['stop'])
        return self.table.store.json(block['file'])[self.role][index-block['start']]
    def __array__(self,dtype=None):return np.asarray(list(self),dtype=dtype)

class _Table(Mapping):
    def __init__(self,store,desc):self.store=store;self.desc=desc;self.header=store.json(desc['header']);self.index=store.json(desc['index'])
    def __iter__(self):return iter([*self.header,'global_cell_ids','source_keys','coordinates'])
    def __len__(self):return len(self.header)+3
    def __getitem__(self,key):
        if key in self.index:return self.index[key]
        if key=='coordinates':return {r:dict(v,values=_Column(self,r)) for r,v in self.desc['coordinates'].items()}
        return self.header[key]
    def select(self,ids):
        wanted=set(ids);rows=[i for i,g in enumerate(self.index['global_cell_ids']) if g in wanted]
        coords={r:dict(v,values=[]) for r,v in self.desc['coordinates'].items()}
        for block in self.desc['blocks']:
            local=[i for i in rows if block['start']<=i<block['stop']]
            if not local:continue
            data=self.store.json(block['file'])
            for r in coords:coords[r]['values'].extend(data[r][i-block['start']] for i in local)
        return dict(self.header,global_cell_ids=[self.index['global_cell_ids'][i] for i in rows],source_keys=[self.index['source_keys'][i] for i in rows],coordinates=coords)
    def owner_ids(self,bounds):
        x0,x1,y0,y1=bounds;selected=[]
        for block in self.desc['blocks']:
            box=block['bounds'].get('rf')
            if box is None:raise ContractError('Owner RF coordinate absent; no implicit frame substitution')
            if box[1]<x0 or box[0]>=x1 or box[3]<y0 or box[2]>=y1:continue
            data=self.store.json(block['file'])['rf']
            selected.extend(self.index['global_cell_ids'][block['start']+i] for i,p in enumerate(data) if x0<=p[0]<x1 and y0<=p[1]<y1)
        return tuple(selected)

class _Edges(Sequence):
    def __init__(self,store,desc):
        self.store=store;self.desc=desc
        with np.load(BytesIO(store.bytes(desc['index'])),allow_pickle=False) as data:self.pre=data['pre'];self.post=data['post']
    def __len__(self):return self.desc['count']
    def __getitem__(self,i):
        if isinstance(i,slice):return [self[j] for j in range(*i.indices(len(self)))]
        if i<0:i+=len(self)
        if i<0 or i>=len(self):raise IndexError(i)
        block=next(b for b in self.desc['blocks'] if b['start']<=i<b['stop'])
        return self.store.json(block['file'])[i-block['start']]
    def parents_for(self,post_ids):
        wanted=set(post_ids);return (str(a) for a,b in zip(self.pre,self.post) if b in wanted)
    def select(self,pre_ids=None,post_ids=None):
        a=None if pre_ids is None else set(pre_ids);b=None if post_ids is None else set(post_ids)
        rows=[i for i,(p,q) in enumerate(zip(self.pre,self.post)) if (a is None or p in a) and (b is None or q in b)];out=[]
        for block in self.desc['blocks']:
            local=[i for i in rows if block['start']<=i<block['stop']]
            if local:
                values=self.store.json(block['file']);out.extend(values[i-block['start']] for i in local)
        return out

class _Contact(Mapping):
    def __init__(self,store,desc):self.store=store;self.desc=desc;self.edges=_Edges(store,desc['edges']);self.header=None
    def _header(self):
        if self.header is None:self.header=self.store.json(self.desc['header'])
        return self.header
    def __getitem__(self,key):return self.edges if key=='edges' else self._header()[key]
    def __iter__(self):return iter([*self._header(),'edges'])
    def __len__(self):return len(self._header())+1

def materialize(value):
    if isinstance(value,_Table):return value.select(value['global_cell_ids'])
    if isinstance(value,Mapping):return {k:materialize(v) for k,v in value.items()}
    if isinstance(value,(_Edges,_Column)):return list(value)
    if isinstance(value,list):return [materialize(v) for v in value]
    return value

class FrozenGraphReader(_Map):
    def __init__(self,path,*,expected_source_sha256,expected_manifest_hash):
        self.path=Path(path);manifest=json.loads((self.path/'manifest.json').read_bytes());digest=manifest.pop('storage_manifest_hash',None)
        if manifest.get('schema')!=SCHEMA or digest!=expected_manifest_hash or canonical_hash(manifest)!=digest or manifest['source_json_sha256']!=expected_source_sha256:
            raise ContractError('Graph storage/source identity mismatch; no original-file or generated fallback')
        self.manifest=manifest;self.storage_manifest_hash=digest;self.semantic_hash=manifest['logical_record_hash'];self.population_counts=manifest['population_counts'];self._decoded={}
        self._kinds={}
        def kinds(node):
            if node['kind'] in ('table','edges'):
                for b in node['blocks']:self._kinds[b['file']]=node['kind']
            if node['kind']=='map':
                for child in node['children'].values():kinds(child)
            if node['kind']=='contact':kinds(node['edges'])
        kinds(manifest['descriptor'])
        self.io={'files_read':0,'compressed_bytes_read':0,'table_blocks_read':0,'edge_blocks_read':0,'original_json_opened':False}
        super().__init__(self,manifest['descriptor'])
    def bytes(self,name):
        if Path(name).name!=name or name not in self.manifest['files']:raise ContractError('Invalid graph storage file reference')
        raw=(self.path/name).read_bytes();expected=self.manifest['files'][name]
        if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:raise ContractError('Graph storage block bytes changed: '+name)
        self.io['files_read']+=1;self.io['compressed_bytes_read']+=len(raw)
        kind=self._kinds.get(name)
        if kind:self.io['table_blocks_read' if kind=='table' else 'edge_blocks_read']+=1
        return raw
    def json(self,name):
        if name not in self._decoded:self._decoded[name]=json.loads(gzip.decompress(self.bytes(name)))
        return self._decoded[name]
    def decode(self,desc):
        kind=desc['kind']
        if kind=='json':return self.json(desc['file'])
        if kind=='map':return _Map(self,desc)
        if kind=='table':return _Table(self,desc)
        if kind=='edges':return _Edges(self,desc)
        if kind=='contact':return _Contact(self,desc)
        raise ContractError('Unknown frozen graph storage node')
    def eager(self,key):return materialize(self[key])
    def clear_decoded_blocks(self):self._decoded.clear()
