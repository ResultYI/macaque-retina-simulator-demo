"""Shared immutable population masters; ROI queries never create populations.

This is geometry/boundary infrastructure, not a new density or closure model.
Masters retain their existing nominal RF/physical-chart registration. A query
cannot silently convert coordinates, extrapolate, wrap or compensate edges.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from .contracts import ContractError,Quantity,canonical_hash
from .atlas import AtlasROI
from .equal_mass_master import FixedEqualMassMaster

SCHEMA='shared_population_catalog_v1'
REQUIRED=('rods','rod_BC','AII_AC','parasol_RGC_ON','parasol_RGC_OFF')

def _require(ok,message):
    if not ok:raise ContractError('POPULATION_CATALOG: '+message)

def _path(root,value):
    path=(root/str(value)).resolve()
    _require(path.is_relative_to(root),'Master path must stay in the project')
    return path

def register_existing_masters(root,record,path):
    """Freeze references to existing masters only; no generator is invoked."""
    root=Path(root).resolve();path=Path(path).resolve();_require(path.is_relative_to(root),'Catalog path outside project')
    entries={};masters=record['population_revision']['masters']
    for name in REQUIRED:
        _require(name in masters,'Required shared population missing: '+name)
        m=masters[name];source=_path(root,m['path']);raw=source.read_bytes()
        _require(hashlib.sha256(raw).hexdigest()==m['sha256'],'Source master hash mismatch: '+name)
        payload=json.loads(raw);_require(canonical_hash(payload['recipe'])==m['namespace'],'Source namespace mismatch')
        entries[name]={'path':source.relative_to(root).as_posix(),'sha256':m['sha256'],'namespace':m['namespace'],
            'count':len(payload['source_keys']),'canonical_bounds_mm':payload['recipe']['canonical_bounds_mm'],
            'coordinate_role':'rf','coordinate_frame':'functional_visual_field',
            'coordinate_registration':'existing nominal RF/physical chart convention, not a new soma/IS mapping',
            'source':payload['recipe']['source']}
    catalog={'schema':SCHEMA,'reference_id':record['atlas']['reference_id'],
        'cone_master_binding':record['atlas'].get('cone_master'),'masters':entries,
        'boundary_mode':'finite_open_boundary','recurrent_closure':'NOT_CERTIFIED',
        'global_0_to_3mm_coverage':False,'regenerate_on_query':False,
        'edge_compensation':False,'claim':'Immutable geometry reuse and finite spatial coverage only; not physiological validation'}
    raw=(json.dumps(catalog,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf8')
    if path.exists():_require(path.read_bytes()==raw,'Catalog already frozen with different bindings')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as out:out.write(raw)
    return {'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(raw).hexdigest()}

class SharedPopulationCatalog:
    def __init__(self,root,binding,*,expected_cone_master=None):
        self.root=Path(root).resolve();raw=_path(self.root,binding['path']).read_bytes()
        _require(hashlib.sha256(raw).hexdigest()==binding['sha256'],'Frozen catalog hash mismatch')
        self.record=json.loads(raw);r=self.record
        _require(r.get('schema')==SCHEMA and set(r['masters'])==set(REQUIRED),'Unsupported/incomplete shared population catalog')
        _require(r['boundary_mode']=='finite_open_boundary' and r['regenerate_on_query'] is False and r['edge_compensation'] is False,'Unsupported boundary/regeneration policy')
        if expected_cone_master is not None:_require(r['cone_master_binding']==expected_cone_master,'Catalog belongs to different cone geometry')
        self.binding=dict(binding);self.reference_id=r['reference_id'];self._masters={}
    def master(self,name,*,expected_source=None):
        _require(name in self.record['masters'],'Unknown shared population: '+name)
        binding=self.record['masters'][name]
        if name not in self._masters:
            path=_path(self.root,binding['path']);raw=path.read_bytes()
            _require(hashlib.sha256(raw).hexdigest()==binding['sha256'],'Frozen master changed: '+name)
            payload=json.loads(raw);_require(canonical_hash(payload['recipe'])==binding['namespace'],'Frozen master namespace changed')
            master=FixedEqualMassMaster(payload,path,binding['sha256'])
            _require(master.table.reference_id==self.reference_id and len(master.table.source_keys)==binding['count'],'Master reference/count mismatch')
            _require(list(master.bounds)==binding['canonical_bounds_mm'],'Master domain mismatch')
            self._masters[name]=master
        master=self._masters[name]
        if expected_source is not None:_require(canonical_hash(master.recipe['source'])==canonical_hash(expected_source),'Existing master source differs from requested model: '+name)
        return master
    def require_domain(self,name,bounds,*,coordinate_frame='functional_visual_field'):
        _require(coordinate_frame=='functional_visual_field','Explicit frame conversion required; no RF/OPL/IPL fallback')
        bounds=np.asarray(bounds,dtype=float)
        _require(bounds.shape==(4,) and np.all(np.isfinite(bounds)),'Finite rectangular domain required')
        x0,x1,y0,y1=bounds;b=self.master(name).bounds
        _require(b[0]<=x0<x1<=b[1] and b[2]<=y0<y1<=b[3],
            'MASTER_DOMAIN_INSUFFICIENT: '+name+'; requested='+str(bounds.tolist())+'; available='+str(list(b))+
            '; no regeneration, density change or edge compensation')
        return {'population':name,'query_bounds_mm':bounds.tolist(),'canonical_bounds_mm':list(b),
            'spatial_coverage':'COMPLETE_WITHIN_FIXED_MASTER','recurrent_closure':'NOT_CERTIFIED'}
    def crop(self,name,roi,*,max_cells=60000):
        _require(isinstance(roi,AtlasROI) and roi.reference_id==self.reference_id and roi.coordinate_role=='rf','Query frame/reference differs from shared master')
        self.require_domain(name,roi.query_bounds_mm)
        return self.master(name).crop(roi,max_cells)
    def footprint_coverage(self,name,centers,radius_mm):
        centers.require('mm','functional_position',frame='functional_visual_field')
        xy=centers.values;r=np.broadcast_to(np.asarray(radius_mm,dtype=float),(len(xy),))
        _require(xy.ndim==2 and xy.shape[1]==2 and np.all(np.isfinite(r)) and np.all(r>=0),'Finite centers/nonnegative support required')
        b=self.master(name).bounds
        complete=(xy[:,0]-r>=b[0])&(xy[:,0]+r<=b[1])&(xy[:,1]-r>=b[2])&(xy[:,1]+r<=b[3])
        return {'complete':complete,'rule':'finite disk entirely inside the declared rectangular master',
            'outside':'UNKNOWN_OR_UNAVAILABLE; not biological zero','recurrent_closure':'NOT_CERTIFIED'}
