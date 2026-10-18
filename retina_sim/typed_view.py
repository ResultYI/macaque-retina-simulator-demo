"""Immutable typed views and dependency closure of an existing compiled model.

No cells/coordinates/edges are generated. Closure is exact inside the supplied
finite model; it does not invent or certify missing exterior retinal tissue.
"""
from collections import deque
from types import MappingProxyType
import numpy as np
from .atlas import AtlasROI
from .contracts import ContractError,Quantity,canonical_hash
from .geometry_identity import compiled_geometry_identity

# These existing mechanisms operate on corresponding cell rows. Spatial
# mixing is represented by compiled links/gaps, except the explicit native A/B
# adapter below. Unknown mechanisms must supply an adapter rather than silently
# receiving an assumed per-cell closure rule.
CELL_LOCAL_MECHANISMS=frozenset({'HistoryBipolarTerminal','ResidualAC','ResidualBC_ON','ResidualBC_OFF',
    'SynapticConductance','HistoryConductanceRGC','Chen2024PrimateRodCurrent','ReceptorConductanceMembrane',
    'ExplicitPhysicalUnitConversion','RodContactTRPM1','DoubleExponentialChemicalSynapse'})

class TypedBoundaryError(ContractError):
    def __init__(self,code,detail):self.code=code;super().__init__('TYPED_'+code+': '+detail)

def _require(ok,code,detail):
    if not ok:raise TypedBoundaryError(code,detail)

def _inside(b,outer):return outer[0]<=b[0]<b[1]<=outer[1] and outer[2]<=b[2]<b[3]<=outer[3]

def _array(a):
    a=np.asarray(a);return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)

class CompiledTypedView:
    def __init__(self,compiled,*,reference_id,owner_domain_mm,query_domain_mm,empty_owner_outputs=()):
        self.compiled=compiled;self.reference_id=reference_id
        domain=AtlasROI(owner_domain_mm,query_domain_mm,reference_id)
        self.owner_domain=domain.owner_bounds_mm;self.query_domain=domain.query_bounds_mm
        self.geometry=compiled_geometry_identity(compiled);self.empty_outputs=frozenset(empty_owner_outputs)
        self.populations={};self.cell_coordinates={};self.nodes=set();self.parents={};self.requirements={};self.edges=[]
        for pid,p in compiled.populations.items():
            coordinates={}
            for role,q in [('rf',p.receptive_field_positions),('soma',p.anatomical_positions),('opl_pedicle',p.pedicle_positions),
                ('terminal',p.terminal_positions),('arbor',p.dendritic_positions)]:
                if q is None:continue
                if role=='terminal':role='opl_terminal' if q.coordinate_frame=='OPL' else 'ipl_preterminal'
                if role=='arbor':role='opl_post_arbor' if q.coordinate_frame=='OPL' else 'ipl_post_arbor'
                coordinates[role]=(_array(q.values),q.unit,q.meaning,q.evidence_tag,q.coordinate_frame)
                for index,gid in enumerate(p.cells.cell_id):
                    old=self.cell_coordinates.setdefault(gid,{}).get(role)
                    value=(tuple(q.values[index]),q.unit,q.meaning,q.coordinate_frame)
                    _require(old is None or old==value,'ID_COORDINATE_CONFLICT',gid+' '+role)
                    self.cell_coordinates[gid][role]=value
            self.populations[pid]=(tuple(p.cells.cell_id),MappingProxyType(coordinates))
        for cid,c in compiled.components.items():
            ids={gid for port in tuple(c.inputs.values())+tuple(c.outputs.values()) for gid in port.cell_ids}
            ids.update(gid for s in c.state_specs for gid in s.cell_ids)
            self.nodes.update((cid,gid) for gid in ids)
        external={(e.target_component,e.target_port):e.channel for e in compiled.spec.external_inputs}
        incoming={}
        for k,link in enumerate(compiled.links):
            s=link.specification;incoming.setdefault((s.target_component,s.target_port),[]).append(link)
            pre=compiled.components[s.source_component].outputs[s.source_port].cell_ids
            post=compiled.components[s.target_component].inputs[s.target_port].cell_ids
            coo=link.matrix.tocoo()
            for i,j,w in zip(coo.row,coo.col,coo.data):
                if w!=0:self._edge((s.source_component,pre[j]),(s.target_component,post[i]),('link',k,int(i),int(j)),float(w))
        for cid,c in compiled.components.items():
            native=c.native_model
            if c.mechanism=='VanHateren2005NativeConeH1' and all(hasattr(native,n) for n in ('A','B','cones','hcs')):
                cones=native.cones.cells.cell_id;hc=native.hcs.cells.cell_id
                for name,pre,post in [('A',cones,hc),('B',hc,cones)]:
                    m=getattr(native,name).tocoo()
                    for i,j,w in zip(m.row,m.col,m.data):
                        if w!=0:self._edge((cid,pre[j]),(cid,post[i]),('native',cid,name,int(i),int(j)),float(w))
            elif c.mechanism not in CELL_LOCAL_MECHANISMS or len(c.populations)!=1 or c.internal_dependencies:
                raise TypedBoundaryError('INTERNAL_DEPENDENCY_UNSUPPORTED',cid+' needs an explicit spatial dependency adapter')
            for port,p in c.inputs.items():
                for row,gid in enumerate(p.cell_ids):
                    links=incoming.get((cid,port),())
                    if (cid,port) in external:rule=('external',external[(cid,port)])
                    elif links and any(np.any(link.matrix.getrow(row).data!=0) for link in links):rule=('linked',port)
                    else:rule=('missing',port)
                    self.requirements.setdefault((cid,gid),[]).append(rule)
        for k,gap in enumerate(compiled.spec.gap_junctions):
            for i,(left,right) in enumerate(zip(gap.left_cell_ids,gap.right_cell_ids)):
                a=(gap.left_component,left);b=(gap.right_component,right)
                self._edge(a,b,('gap',k,i,'forward'),None);self._edge(b,a,('gap',k,i,'reverse'),None)
        self.nodes=frozenset(self.nodes)
        self.parents=MappingProxyType({k:frozenset(v) for k,v in self.parents.items()});self.edges=tuple(self.edges)
        self.populations=MappingProxyType(self.populations)
        self.requirements=MappingProxyType({k:tuple(v) for k,v in self.requirements.items()})
    def _assert_current(self):
        _require(compiled_geometry_identity(self.compiled)==self.geometry,'STALE_MODEL','Canonical geometry/configuration changed after this view was bound')
    def _edge(self,pre,post,key,weight):
        _require(pre in self.nodes and post in self.nodes,'DANGLING_EDGE',str(key))
        self.parents.setdefault(post,set()).add(pre);self.edges.append((pre,post,key,weight))
    def coordinate(self,population,role,ids=None):
        _require(population in self.populations,'POPULATION',population)
        all_ids,coords=self.populations[population]
        _require(role in coords,'COORDINATE_ROLE_UNAVAILABLE',population+' '+role+'; no soma/RF/OPL/IPL substitution')
        a,unit,meaning,tag,frame=coords[role]
        if ids is not None:
            lookup={key:i for i,key in enumerate(all_ids)}
            _require(all(key in lookup for key in ids),'CELL_ID',population)
            a=a[[lookup[key] for key in ids]]
        value=Quantity(a,unit,meaning,tag,frame)
        # Quantity's general constructor owns a reversible read-only copy. This
        # immutable atlas view instead uses bytes as the backing owner.
        object.__setattr__(value,'values',_array(value.values))
        return value
    def dependency_closure(self,targets,*,require_exterior_closure=False):
        self._assert_current()
        _require(not require_exterior_closure,'EXTERIOR_CLOSURE_UNCERTIFIED','Only the existing finite-open model is supplied')
        selected=set(targets);_require(selected<=self.nodes,'CELL_ID','Target not instantiated in the canonical compiled model')
        pending=deque(selected);external=set()
        while pending:
            node=pending.popleft()
            for kind,value in self.requirements.get(node,()):
                _require(kind!='missing','MISSING_INPUT_ROW',str(node)+' '+value)
                if kind=='external':external.add((value,node[1]))
            for parent in self.parents.get(node,()):
                if parent not in selected:selected.add(parent);pending.append(parent)
        edges=tuple(e for e in self.edges if e[1] in selected)
        _require(all(e[0] in selected for e in edges),'DEPENDENCY','Closure lost a required upstream node')
        return {'nodes':frozenset(selected),'edges':edges,'external_cells':frozenset(external),
            'scope':'exact dependency closure of the supplied canonical finite model',
            'boundary_mode':'finite_open_boundary','exterior_recurrent_closure':'NOT_CERTIFIED'}
    def owner_view(self,roi,outputs):
        _require(isinstance(roi,AtlasROI) and roi.reference_id==self.reference_id and roi.coordinate_role=='rf','REFERENCE','Explicit same-reference RF ROI required')
        _require(_inside(roi.owner_bounds_mm,self.owner_domain) and _inside(roi.query_bounds_mm,self.query_domain),
            'MODEL_DOMAIN_UNAVAILABLE','Requested owner/query crosses the fixed typed model support; no new cells or boundary driver')
        targets=set();owners={};status={}
        for cid in outputs:
            if cid not in self.compiled.components:
                _require(cid in self.empty_outputs,'TARGET_UNAVAILABLE',cid+' has no supplied mechanism/owner certificate')
                owners[cid]=();status[cid]='EMPTY_SAMPLE';continue
            component=self.compiled.components[cid]
            port=component.outputs.get('voltage')
            _require(port is not None,'TARGET_UNAVAILABLE',cid+' requires an explicit target-port adapter')
            b=roi.owner_bounds_mm;ids=[]
            for gid in port.cell_ids:
                coordinate=self.cell_coordinates.get(gid,{}).get('rf')
                _require(coordinate is not None and coordinate[1]=='mm' and coordinate[3]=='functional_visual_field','COORDINATE_ROLE_UNAVAILABLE',cid+' RF')
                x,y=coordinate[0]
                if b[0]<=x<b[1] and b[2]<=y<b[3]:ids.append(gid);targets.add((cid,gid))
            owners[cid]=tuple(ids);status[cid]='INSTANTIATED_OWNER_PATH' if ids else 'EMPTY_SAMPLE'
        closure=self.dependency_closure(targets)
        # The view selects owners/recording only; it never constructs a smaller
        # dynamical graph or renormalizes existing input rows.
        result={'owners':owners,'owner_status':status,'closure':closure,'geometry_identity':self.geometry,
            'owner_bounds_mm':roi.owner_bounds_mm,'query_bounds_mm':roi.query_bounds_mm,
            'canonical_model_unchanged':True,'simulation_scope':'same whole canonical finite model; owner/recording selection only'}
        result['view_hash']=canonical_hash({'geometry':self.geometry['sha256'],'owner':roi.owner_bounds_mm,'query':roi.query_bounds_mm,'owners':owners})
        return result
    def owner_recording(self,roi,outputs,*,max_bytes=256*1024*1024):
        """A real RecordingPlan; underlying canonical states/edges stay intact."""
        from .recording import RecordingPlan
        view=self.owner_view(roi,outputs)
        ports=tuple(cid+'.voltage' for cid,ids in view['owners'].items() if ids)
        ids=tuple(gid for cells in view['owners'].values() for gid in cells)
        return view,RecordingPlan(ports=ports,states=(),cell_ids=ids,max_bytes=max_bytes)

def compiled_typed_view(compiled):
    domain=getattr(compiled,'typed_view_domain',None)
    _require(domain is not None,'DOMAIN_UNAVAILABLE','Compiled model has no declared fixed owner/query support')
    return CompiledTypedView(compiled,**domain)
