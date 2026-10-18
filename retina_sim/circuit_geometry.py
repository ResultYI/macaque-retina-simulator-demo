"""Frozen spatial results for one already supplied canonical finite circuit.

Stores existing results, never extends tissue, changes laws or certifies physiology.
The source graph owns full BC/AC/IPL tables and anchors; this companion freezes
the runtime A/B, rod-contact and gap results that previously were recomputed.
"""
from pathlib import Path
from copy import deepcopy
import hashlib,json
import numpy as np
from scipy import sparse
from .contracts import ContractError,Quantity,canonical_hash
from .assembly import GapJunction
from .mechanism_contracts import EvidenceValue

SCHEMA='canonical_finite_circuit_geometry_v1'
DERIVED_FIELDS=('inner_tables','inner_parents','inner_contacts','inner_metadata','parasol_anchors','rod_inner')

def require(ok,message):
    if not ok:raise ContractError('CIRCUIT_GEOMETRY: '+message)

def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if hasattr(value,'__dataclass_fields__'):return {k:plain(getattr(value,k)) for k in value.__dataclass_fields__}
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value

def csr_record(matrix):
    m=sparse.csr_matrix(matrix)
    return {'shape':list(m.shape),'data':m.data.tolist(),'indices':m.indices.tolist(),'indptr':m.indptr.tolist(),
        'data_dtype':str(m.data.dtype),'index_dtype':str(m.indices.dtype),'indptr_dtype':str(m.indptr.dtype)}

def csr_value(record):
    return sparse.csr_matrix((np.array(record['data'],dtype=record['data_dtype']),
        np.array(record['indices'],dtype=record['index_dtype']),np.array(record['indptr'],dtype=record['indptr_dtype'])),shape=record['shape'])

def spatial_snapshot(compiled):
    """Actual spatial content only; changing background/seed is not new anatomy."""
    return {'populations':{pid:canonical_hash(p.record()) for pid,p in compiled.populations.items()},
        'links':[canonical_hash({'spec':x.specification,'matrix':csr_record(x.matrix)}) for x in compiled.links],
        'native':{cid:{n:csr_record(getattr(c.native_model,n)) for n in ('A','B')}
            for cid,c in compiled.components.items() if c.mechanism=='VanHateren2005NativeConeH1'},
        'gaps':canonical_hash(compiled.spec.gap_junctions)}

def freeze_circuit_geometry(root,path,record,graph_binding,compiled):
    root=Path(root).resolve();path=Path(path).resolve()
    require(path.is_relative_to(root),'Asset must stay in project')
    from .typed_view import compiled_typed_view
    from .atlas import AtlasROI
    v=compiled_typed_view(compiled);targets=tuple(compiled.typed_view_domain['empty_owner_outputs'])+tuple(
        cid for cid in ('midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF') if cid in compiled.components)
    closure=v.owner_view(AtlasROI(v.owner_domain,v.query_domain,v.reference_id),targets)['closure']
    inventory={}
    for pid,p in compiled.populations.items():
        roles={}
        for role,q in (('rf',p.receptive_field_positions),('soma',p.anatomical_positions),('opl_pedicle',p.pedicle_positions),
                ('terminal',p.terminal_positions),('arbor',p.dendritic_positions)):
            if q is not None:roles[role]={'unit':q.unit,'frame':q.coordinate_frame,'count':len(q.values),
                'point_extent':[q.values.min(0).tolist(),q.values.max(0).tolist()],
                'extent_is_source_coverage':False,'sha256':canonical_hash(q)}
        inventory[pid]={'count':len(p.cells.cell_id),'coordinates':roles}
    value={'schema':SCHEMA,'source_graph':graph_binding,'reference_id':v.reference_id,
        'owner_domain_mm':v.owner_domain,'RF_input_domain_mm':v.query_domain,
        'boundary_mode':'finite_open_boundary','edge_compensation':False,'regenerate_on_ROI':False,
        'exterior_recurrent_closure':'NOT_CERTIFIED','physiological_validation':False,
        'derived_graph_fields':{k:canonical_hash(record.get(k)) for k in DERIVED_FIELDS},
        'full_derived_tables':{k:len(t['global_cell_ids']) for k,t in record['inner_tables'].items()},
        'typed_coordinate_inventory':inventory,'spatial_snapshot':spatial_snapshot(compiled),
        'primary_rod_recipe':compiled.canonical_spatial_recipe,
        'coverage':{'contract':'exact dependencies inside supplied finite canonical circuit; exterior omitted by approved design',
            'owner_closure_nodes':len(closure['nodes']),'owner_closure_edges':len(closure['edges']),
            'external_channel_cells':len(closure['external_cells']),'missing_required_owner_input_rows':0,
            'boundary_policy_already_approved':True}}
    raw=(json.dumps(plain(value),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')
    if path.exists():require(path.read_bytes()==raw,'Existing frozen asset differs; never overwrite')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(raw)
    return {'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(raw).hexdigest()}

class FrozenCircuitGeometry:
    def __init__(self,root,binding,record,graph_binding):
        root=Path(root).resolve();path=(root/binding['path']).resolve()
        require(path.is_relative_to(root),'Asset path outside project')
        raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==binding['sha256'],'Frozen asset hash mismatch')
        self.record=json.loads(raw);self.binding=dict(binding);r=self.record
        require(r.get('schema')==SCHEMA and r['source_graph']==graph_binding,'Wrong canonical source graph/schema')
        require(r['boundary_mode']=='finite_open_boundary' and r['edge_compensation'] is False and r['regenerate_on_ROI'] is False,'Unsupported boundary rule')
        require(r['derived_graph_fields']=={k:canonical_hash(record.get(k)) for k in DERIVED_FIELDS},'Derived tables/anchors/contacts changed')
        require(r['reference_id']==record['atlas']['reference_id'] and r['owner_domain_mm']==record['atlas']['owner_domain'] and
            r['RF_input_domain_mm']==record['dependency_query']['requested_RF_query_mm'],'Canonical domain changed')
        self._tables=deepcopy(record['inner_tables'])
    def table_coordinate(self,name,role,ids=None):
        require(name in self._tables,'Unknown full derived table: '+name)
        table=self._tables[name];require(role in table['coordinates'],'Typed coordinate unavailable; no RF/soma/IPL substitution')
        r=table['coordinates'][role];a=np.asarray(r['values'],dtype=float)
        if ids is not None:
            lookup={gid:i for i,gid in enumerate(table['global_cell_ids'])}
            require(all(gid in lookup for gid in ids),'Cell absent from full derived table')
            a=a[[lookup[gid] for gid in ids]]
        q=Quantity(a,r['unit'],r['meaning'],r.get('evidence_tag','project_approximation'),r['frame'])
        object.__setattr__(q,'values',np.frombuffer(q.values.tobytes(),dtype=q.values.dtype).reshape(q.values.shape))
        return q
    def select_table(self,name,roi):
        from .atlas import AtlasROI
        from .typed_view import _inside
        require(isinstance(roi,AtlasROI) and roi.reference_id==self.record['reference_id'] and roi.coordinate_role=='rf','Same-reference RF ROI required')
        require(_inside(roi.owner_bounds_mm,self.record['owner_domain_mm']) and _inside(roi.query_bounds_mm,self.record['RF_input_domain_mm']),
            'ROI outside current canonical domain; no geometry expansion')
        q=self.table_coordinate(name,'rf');q.require('mm','functional_position',frame='functional_visual_field')
        x0,x1,y0,y1=roi.owner_bounds_mm;xy=q.values
        keep=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
        return tuple(gid for gid,yes in zip(self._tables[name]['global_cell_ids'],keep) if yes)
    def native_matrices(self,cones,hcs):
        matrices=self.record['spatial_snapshot']['native']['outer_native']
        require(canonical_hash(cones.record())==self.record['spatial_snapshot']['populations']['cones'] and
            canonical_hash(hcs.record())==self.record['spatial_snapshot']['populations']['HC'],'Native population identity changed')
        return csr_value(matrices['A']),csr_value(matrices['B'])
    def primary_pool(self,name,pre_ids,post_ids):
        r=self.record['primary_rod_recipe']['pools'][name]
        require(list(pre_ids)==r['pre_ids'] and list(post_ids)==r['post_ids'],'Pool cell identity/order changed: '+name)
        return csr_value(r['matrix']),r['rows'],r.get('overlap')
    def contact_record(self,cid):return self.record['primary_rod_recipe']['contact_populations'][cid]
    def gaps(self):
        result=[]
        for r in self.record['primary_rod_recipe']['gaps']:
            e=dict(r['conductance_evidence']);e['value']=Quantity(**e['value'])
            result.append(GapJunction(r['left_component'],tuple(r['left_cell_ids']),r['right_component'],tuple(r['right_cell_ids']),EvidenceValue(**e),r['geometry_frame']))
        return result
    def validate_compiled(self,compiled):
        require(spatial_snapshot(compiled)==self.record['spatial_snapshot'],'Actual typed populations/contacts/weights/gaps differ from frozen canonical asset')
        return dict(self.record['coverage'],asset=self.binding,full_derived_tables=self.record['full_derived_tables'])
