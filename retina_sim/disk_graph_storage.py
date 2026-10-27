"""Frozen global integer-index CSR blocks; ROI reads never rebuild a pool.

Population bindings own stable canonical IDs. Row blocks refer to their exact
global ordering, so millions of repeated string IDs need not reside in RAM.
"""
from pathlib import Path
import hashlib,json,os
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from .contracts import ContractError,canonical_hash
from .disk_pooling import bounded_matrix

SCHEMA='frozen_global_sparse_route_v1'

def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def save_json(path,value):
    path=Path(path);pending=path.with_name(path.name+'.writing')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    os.replace(pending,path)

def array_identity(value):
    array=np.asarray(value);digest=hashlib.sha256()
    if array.ndim==0:digest.update(array.tobytes())
    else:
        for start in range(0,len(array),65536):digest.update(np.ascontiguousarray(array[start:start+65536]).tobytes())
    return {'shape':list(array.shape),'dtype':str(array.dtype),'sha256':digest.hexdigest()}

class SparseRouteWriter:
    def __init__(self,root,directory,recipe):
        self.root=Path(root).resolve();self.directory=(self.root/directory).resolve()
        if not self.directory.is_relative_to(self.root):raise ContractError('Route asset outside project')
        self.directory.mkdir(exist_ok=True,parents=True);self.recipe=dict(recipe)
        self.manifest=self.directory/'manifest.json';self.progress=self.directory/'construction.json'
        self.recipe_hash=canonical_hash(recipe)
        self.state={'schema':SCHEMA,'recipe':self.recipe,'recipe_sha256':self.recipe_hash,'blocks':[],
            'completed_post_rows':0,'edges':0,'missing_degree_rows':0,'empty_rows':0,'status':'CONSTRUCTING'}
        if self.manifest.exists():
            self.state=json.loads(self.manifest.read_bytes())
            if self.state['recipe_sha256']!=self.recipe_hash:raise ContractError('Frozen route recipe differs; retained')
        elif self.progress.exists():
            self.state=json.loads(self.progress.read_bytes())
            if self.state['recipe_sha256']!=self.recipe_hash:raise ContractError('Partial route recipe differs; retained')
            for block in self.state['blocks']:
                if file_sha(self.root/block['path'])!=block['sha256']:raise ContractError('Partial route block changed')
        else:save_json(self.progress,self.state)
    @property
    def next_row(self):return self.state['completed_post_rows']
    def append(self,start,matrix,rows,*,row_status=None,extras=None):
        if self.manifest.exists():raise ContractError('Cannot append a frozen route')
        m=sparse.csr_matrix(matrix);stop=start+m.shape[0]
        if start!=self.next_row or m.shape[1]!=self.recipe['pre_count'] or stop>self.recipe['post_count']:
            raise ContractError('Global CSR block ordering/shape differs')
        if len(rows)!=m.shape[0] or (row_status is not None and len(row_status)!=m.shape[0]):raise ContractError('Row audits differ')
        if not np.all(np.isfinite(m.data)):raise ContractError('Nonfinite frozen route weights')
        path=self.directory/f'rows_{start:09d}_{stop:09d}.npz'
        if path.exists():raise ContractError('Unregistered partial block retained for inspection')
        audits=[dict(row,post_index=start+j) for j,row in enumerate(rows)]
        status=list(row_status or ['READY_PROJECT_FINITE_POOL' if m.indptr[j+1]>m.indptr[j] else
            'INCOMPLETE_NO_INPUT_WITHIN_SUPPORT' for j in range(m.shape[0])])
        payload={'data':m.data,'indices':m.indices,'indptr':m.indptr,'shape':np.asarray(m.shape,dtype=np.int64),
            'audit_json':np.asarray(json.dumps(audits,allow_nan=False,separators=(',',':'))),
            'row_status':np.asarray(status,dtype='U96')}
        for name,value in (extras or {}).items():
            if name in payload:raise ContractError('Duplicate route array name')
            payload[name]=np.asarray(value)
        pending=path.with_suffix('.writing.npz');np.savez(pending,**payload);os.replace(pending,path)
        block={'start':start,'stop':stop,'path':path.relative_to(self.root).as_posix(),'sha256':file_sha(path),'edges':m.nnz}
        self.state['blocks'].append(block);self.state['completed_post_rows']=stop;self.state['edges']+=m.nnz
        self.state['missing_degree_rows']+=sum(row.get('missing_degree',0)>0 for row in rows)
        self.state['empty_rows']+=int(np.count_nonzero(np.diff(m.indptr)==0));save_json(self.progress,self.state)
    def freeze(self):
        if self.next_row!=self.recipe['post_count']:raise ContractError('Route incomplete; cannot freeze')
        self.state.update(status='FROZEN',regenerate_on_ROI=False,crop_normalization=False,
            physiological_validation=False,boundary='finite_open_global_canonical_graph')
        if not self.manifest.exists():save_json(self.manifest,self.state)
        return {'path':self.manifest.relative_to(self.root).as_posix(),'sha256':file_sha(self.manifest)}

class FrozenSparseRoute:
    def __init__(self,root,binding):
        self.root=Path(root).resolve();self.binding=dict(binding);path=(self.root/binding['path']).resolve()
        if not path.is_relative_to(self.root) or file_sha(path)!=binding['sha256']:raise ContractError('Global route identity mismatch')
        self.record=json.loads(path.read_bytes());r=self.record;self._verified={}
        if r['schema']!=SCHEMA or r['status']!='FROZEN' or r['recipe_sha256']!=canonical_hash(r['recipe']) or r['regenerate_on_ROI'] is not False or r['crop_normalization'] is not False:
            raise ContractError('Unsupported frozen route contract')
        next_row=0
        for block in r['blocks']:
            if block['start']!=next_row or block['stop']<=next_row:raise ContractError('Noncontiguous global route')
            next_row=block['stop']
        if next_row!=r['recipe']['post_count']:raise ContractError('Incomplete global route')
    def block(self,record):
        path=(self.root/record['path']).resolve()
        if not path.is_relative_to(self.root):raise ContractError('Route block outside project')
        stat=path.stat();stamp=(stat.st_size,stat.st_mtime_ns)
        if self._verified.get(record['path'])!=stamp:
            if file_sha(path)!=record['sha256']:raise ContractError('Frozen route block changed')
            self._verified[record['path']]=stamp
        with np.load(path,allow_pickle=False) as z:
            matrix=sparse.csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(z['shape']))
            rows=json.loads(str(z['audit_json']));statuses=z['row_status'].tolist()
        return matrix,rows,statuses
    def selected_rows(self,post_indices):
        post=np.unique(np.asarray(post_indices,dtype=np.int64))
        if np.any(post<0) or np.any(post>=self.record['recipe']['post_count']):raise ContractError('Route row outside global table')
        matrices=[];audits=[];statuses=[]
        for block in self.record['blocks']:
            selected=post[(post>=block['start'])&(post<block['stop'])]
            if not len(selected):continue
            m,r,s=self.block(block);local=selected-block['start'];matrices.append(m[local])
            audits.extend(r[i] for i in local);statuses.extend(s[i] for i in local)
        m=sparse.vstack(matrices,format='csr') if matrices else sparse.csr_matrix((0,self.record['recipe']['pre_count']))
        return post,m,audits,statuses
    def dependency_rows(self,post_indices):return np.unique(self.selected_rows(post_indices)[1].indices)
    def ready_rows(self):
        result=np.zeros(self.record['recipe']['post_count'],dtype=bool)
        for block in self.record['blocks']:
            _,_,statuses=self.block(block)
            result[block['start']:block['stop']]=[status.startswith('READY') for status in statuses]
        return result
    def view(self,post_indices,pre_indices,*,require_complete=True):
        post,m,rows,statuses=self.selected_rows(post_indices);pre=np.unique(np.asarray(pre_indices,dtype=np.int64))
        if require_complete and not np.all(np.isin(m.indices,pre)):
            raise ContractError('ROI omits frozen dependencies; crop normalization is forbidden')
        return post,pre,m[:,pre],rows,statuses

def materialize_bounded_route(root,directory,recipe,pre,post,k,*,support_mm=.04,gaussian_scale=None,block_rows=4096,on_block=None,pre_eligible=None,post_enabled=None,exact_degree=False):
    """Same hard support, ordering and original per-row normalization, frozen once."""
    pre=np.asarray(pre);post=np.asarray(post)
    if recipe['pre_count']!=len(pre) or recipe['post_count']!=len(post):raise ContractError('Route table counts differ')
    recipe=dict(recipe,resolved_numerics={'pre_coordinates':array_identity(pre),'post_coordinates':array_identity(post),
        'k':array_identity(k),'support_mm':array_identity(support_mm),
        'gaussian_scale':None if gaussian_scale is None else array_identity(gaussian_scale)})
    if pre_eligible is not None:recipe['resolved_numerics']['eligible_pre_rows']=array_identity(pre_eligible)
    if post_enabled is not None:recipe['resolved_numerics']['enabled_post_rows']=array_identity(post_enabled)
    if exact_degree:recipe['resolved_numerics']['exact_degree_status_gate']=True
    writer=SparseRouteWriter(root,directory,recipe)
    if writer.manifest.exists():return writer.freeze()
    eligible=None if pre_eligible is None else np.asarray(pre_eligible,dtype=np.int64)
    if eligible is not None and (len(np.unique(eligible))!=len(eligible) or np.any(eligible<0) or np.any(eligible>=len(pre))):
        raise ContractError('Eligible presynaptic rows invalid')
    candidate_pre=pre if eligible is None else pre[eligible]
    tree=cKDTree(candidate_pre);radii=np.broadcast_to(support_mm,(len(post),));targets=np.broadcast_to(k,(len(post),))
    scales=None if gaussian_scale is None else np.broadcast_to(gaussian_scale,(len(post),))
    for start in range(writer.next_row,len(post),block_rows):
        stop=min(len(post),start+block_rows)
        matrix,rows=bounded_matrix(candidate_pre,post[start:stop],targets[start:stop],support_mm=radii[start:stop],
            gaussian_scale=None if scales is None else scales[start:stop],pre_tree=tree)
        if eligible is not None:
            coo=matrix.tocoo();matrix=sparse.csr_matrix((coo.data,(coo.row,eligible[coo.col])),shape=(stop-start,len(pre)))
        statuses=['READY_PROJECT_FINITE_POOL' if matrix.indptr[j+1]>matrix.indptr[j] else
            'INCOMPLETE_NO_INPUT_WITHIN_SUPPORT' for j in range(stop-start)]
        if exact_degree or post_enabled is not None:
            matrix=matrix.tolil()
            for j,row in enumerate(rows):
                if post_enabled is not None and not post_enabled[start+j]:statuses[j]='BLOCKED_KNOWN_INELIGIBLE_SEED'
                elif exact_degree and row['missing_degree']:statuses[j]='INCOMPLETE_REQUIRED_DEGREE'
                if not statuses[j].startswith('READY'):matrix.rows[j]=[];matrix.data[j]=[]
            matrix=matrix.tocsr()
        writer.append(start,matrix,rows,row_status=statuses)
        if on_block:on_block(writer.state)
    return writer.freeze()

def dependency_closure(routes,seeds,*,max_nodes=None):
    """Exact closure in this finite graph; supplied reverse gap routes are required."""
    selected={name:set(map(int,rows)) for name,rows in seeds.items()};seen={name:set() for name in selected}
    changed=True
    while changed:
        changed=False
        for pre,post,route in routes:
            pending=selected.get(post,set())-seen.setdefault(route.binding['sha256'],set())
            if not pending:continue
            seen[route.binding['sha256']].update(pending)
            parents=set(map(int,route.dependency_rows(sorted(pending))))
            target=selected.setdefault(pre,set());before=len(target);target.update(parents)
            changed|=len(target)>before
            if max_nodes is not None and sum(map(len,selected.values()))>max_nodes:
                raise ContractError('Exact global closure exceeds execution budget; no dependency truncation')
    return {name:np.asarray(sorted(rows),dtype=np.int64) for name,rows in selected.items()}
