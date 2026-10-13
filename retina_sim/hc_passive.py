"""Approved optional orphan-HC closure; source-driven HC equations stay native.

No references, positive rates or contacts are invented. The caller must supply
a matching prepared gap-off state and complete finite contact-query evidence.
"""
from dataclasses import replace
from pathlib import Path
import hashlib,json
import numpy as np
from types import MappingProxyType

from .contracts import ContractError,Quantity,canonical_hash
from .hc_gap import EffectiveHCGapOperator,with_native_hc_gap
from .vh2005 import NativeConeH1

PROFILE='orphan_hc_background_referenced_passive_v1'


def reference_context(model,background,geometry_graph_hash):
    return canonical_hash({'native_model_hash':model.model_hash,'gap_off_background':background,
        'geometry_graph_hash':geometry_graph_hash,'route':PROFILE})


class BackgroundReferencedPassiveHC:
    def __init__(self,component,gap,hc_types,query_rows,prepared_blocks,preparation,
                 background,geometry_graph_hash,requested_hc_ids,root):
        native=component.native_model
        if not isinstance(native,NativeConeH1):raise ContractError('Passive closure requires an unwrapped gap-off native/finite-contact component')
        config_raw=(Path(root)/'configurations'/(PROFILE+'.json')).read_bytes();self.config=MappingProxyType(json.loads(config_raw))
        self.config_sha256=hashlib.sha256(config_raw).hexdigest();self.native_model=native
        self.evidence_tag=component.evidence_tag
        direct=tuple(native.state_specs[-1].cell_ids);types=dict(hc_types)
        if set(types)!=set(gap.cell_ids) or any(t not in ('H1','H2') for t in types.values()) or not set(direct)<=set(types):
            raise ContractError('All supplied gap HC IDs/types must include the native direct-input states')
        if any(types[g]!=t for g,t in zip(direct,native.hcs.cells.subtype)):
            raise ContractError('Direct HC source types cannot be reassigned for passive closure')
        edges=[(e,float(k)) for e,k in zip(gap.edges,gap.rate) if k>0]
        if any(types[a]!=types[b] for (a,b),_ in edges):raise ContractError('First passive route permits homotypic positive gap components only')
        positive=EffectiveHCGapOperator(gap.cell_ids,[e for e,_ in edges],
            Quantity([k for _,k in edges],'ms^-1','effective_HC_gap_rate',gap.rate_quantity.evidence_tag),
            assumption_profile_name=gap.assumption_profile_name,project_evidence_tag=gap.evidence_tag,
            equal_effective_capacity=True,common_state_scale_reference=True,enabled=True)
        requested=set(requested_hc_ids)
        if not requested<=set(types):raise ContractError('Requested HC output IDs must belong to supplied geometry')
        context=reference_context(native,background,geometry_graph_hash)
        if (preparation.get('status')!='CONVERGED' or preparation.get('gap_enabled') is not False or
                preparation.get('reference_context_hash')!=context):
            raise ContractError('Matching numerically prepared gap-off reference/context required; initializer alone is not a preparation certificate')
        light=background.require('td','native_retinal_illuminance',(native.nc,))
        if not np.all(light==light[0]) or not np.array_equal(light,native.background):
            raise ContractError('Passive reference v1 requires the same uniform declared background')
        blocks=tuple(q.values.copy() if isinstance(q,Quantity) else np.asarray(q,dtype=float).copy() for q in prepared_blocks)
        if len(blocks)!=len(native.state_specs) or any(a.shape!=s.shape or not np.all(np.isfinite(a)) for a,s in zip(blocks,native.state_specs)):
            raise ContractError('Prepared reference state must match every native block/ID and be finite')
        pooled=np.asarray(native.A@blocks[6]).ravel();h=blocks[9]
        if np.any(pooled<=0):raise ContractError('Positive same-type pooled Vbar reference required')
        lookup={g:i for i,g in enumerate(direct)};refs={};included=set(direct);unused=[]
        rows={r['hc_id']:r for r in query_rows}
        for group in positive.connected_components():
            if not (set(group)&requested):unused.extend(g for g in group if g not in lookup);continue
            anchors=[g for g in group if g in lookup];orphans=[g for g in group if g not in lookup]
            if not anchors:raise ContractError('Requested gap component has no direct-input same-type reference; external borrowing refused')
            indices=[lookup[g] for g in anchors];vRef=float(np.mean(pooled[indices]));hRef=float(np.mean(h[indices]));aRef=(vRef/native.p['V_I'])**native.p['mu']
            for g in orphans:
                row=rows.get(g,{})
                if (row.get('direct_input_status')!='COMPLETE_EMPTY_DIRECT_INPUT' or row.get('query_complete_sampled') is not True or
                        row.get('candidate_count')!=0 or row.get('conditional_count')!=0):
                    raise ContractError('Incomplete/unknown direct input cannot become a passive orphan HC')
                refs[g]={'reference_hc_ids':anchors,'vRef':vRef,'hRef':hRef,'aRef':aRef,'uRef':hRef,
                    'evidence_status':'project_approximation','tonic_reference_is_cone_edge':False}
                included.add(g)
        self.ids=tuple(g for g in gap.cell_ids if g in included);self.direct_indices=np.array([self.ids.index(g) for g in direct])
        self.orphan_indices=np.array([self.ids.index(g) for g in refs],dtype=int)
        self.references=MappingProxyType({g:MappingProxyType(dict(ref,reference_hc_ids=tuple(ref['reference_hc_ids']))) for g,ref in refs.items()})
        self.nh=len(self.ids)
        from scipy import sparse
        rows=[native.A[lookup[g],:] if g in lookup else sparse.csr_matrix((1,native.nc)) for g in self.ids]
        columns=[native.B[:,lookup[g]] if g in lookup else sparse.csr_matrix((native.nc,1)) for g in self.ids]
        self.A=sparse.vstack(rows,format='csr');self.B=sparse.hstack(columns,format='csr')
        keep=[(e,k) for e,k in edges if set(e)<=included]
        self.gap=EffectiveHCGapOperator(self.ids,[e for e,_ in keep],
            Quantity([k for _,k in keep],'ms^-1','effective_HC_gap_rate',gap.rate_quantity.evidence_tag),
            assumption_profile_name=gap.assumption_profile_name,project_evidence_tag=gap.evidence_tag,
            equal_effective_capacity=True,common_state_scale_reference=True,enabled=True)
        self.state_specs=tuple(replace(s,cell_ids=self.ids,shape=(len(self.ids),),
            initial_condition_source='prepared direct states + approved background-reference passive subclass')
            if s.state_name in ('z2','Vh') else s for s in native.state_specs)
        self._initial=[]
        for i,(a,s) in enumerate(zip(blocks,self.state_specs)):
            if s.state_name in ('z2','Vh'):
                values=np.empty(len(self.ids));values[self.direct_indices]=a
                for g,ref in refs.items():values[self.ids.index(g)]=ref['hRef']
            else:values=a
            self._initial.append(Quantity(values,s.unit,s.meaning,component.evidence_tag))
        self._initial=tuple(self._initial)
        self.metadata={'profile_id':PROFILE,'config_sha256':self.config_sha256,'reference_context_hash':context,
            'preparation_evidence':preparation,'geometry_graph_hash':geometry_graph_hash,
            'uniform_background_td':float(light[0]),'direct_HC_ids':list(direct),'passive_HC_ids':list(refs),
            'unused_geometry_only_ids':unused,'references':refs,'gap':self.gap.metadata,
            'requires_new_positive_gap_background_preparation':True,'initialization_is_coupled_equilibrium':False,
            'aI_direct_dynamic':True,'physical_conductance_claim':False,'physiological_validation':False}
        self.approximations=list(native.approximations)+[self.metadata]

    def __getattr__(self,name):return getattr(self.native_model,name)

    @property
    def model_hash(self):
        return canonical_hash({'native':self.native_model.model_hash,'passive':self.metadata,
            'edges':self.gap.edges,'rates':self.gap.rate,'initial':self._initial,
            'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def initial_conditions(self):return self._initial

    def _direct_blocks(self,blocks):
        return tuple(np.asarray(a)[self.direct_indices] if i in (8,9) else a for i,a in enumerate(blocks))

    def derivative(self,t,blocks,incoming):
        original=self.native_model.derivative(t,self._direct_blocks(blocks),incoming);result=[]
        for i,(q,s) in enumerate(zip(original,self.state_specs)):
            if i not in (8,9):result.append(q);continue
            values=np.zeros(len(self.ids));values[self.direct_indices]=q.values
            for g,ref in self.references.items():
                j=self.ids.index(g)
                values[j]=1000*(ref['uRef']-blocks[8][j])/(self.config['tau_z2_ms']*ref['aRef']) if i==8 else 1000*(blocks[8][j]-blocks[9][j])/(self.config['tau_h_ms']*ref['aRef'])
            if i==9:values+=self.gap.native_derivative(Quantity(blocks[9],'vH_native_voltage','native_relative_voltage','project_approximation')).values
            result.append(Quantity(values,s.unit+'/s',s.meaning+'_derivative',q.evidence_tag))
        return tuple(result)

    def observe(self,port,t,blocks,incoming):
        if port in ('z2','Vh'):
            i=8 if port=='z2' else 9;s=self.state_specs[i]
            return Quantity(blocks[i],s.unit,s.meaning,self.evidence_tag)
        return self.native_model.observe(port,t,self._direct_blocks(blocks),incoming)


def with_background_referenced_passive_hc(component,gap,*,hc_types,query_rows,prepared_blocks,
        preparation,background,geometry_graph_hash,root,requested_hc_ids=None,hc_population=None):
    if not isinstance(gap,EffectiveHCGapOperator):raise ContractError('Passive v1 supports explicit native effective-k route only')
    direct=tuple(component.state_specs[-1].cell_ids)
    requested=tuple(direct if requested_hc_ids is None else requested_hc_ids)
    if not gap.enabled or not np.any(gap.rate>0):
        if set(requested)-set(direct):raise ContractError('Gap-off orphan requested without an active anchored reference component')
        return component
    adjacency={g:set() for g in gap.cell_ids}
    positive_edges=[(e,float(k)) for e,k in zip(gap.edges,gap.rate) if k>0]
    for (a,b),_ in positive_edges:adjacency[a].add(b);adjacency[b].add(a)
    if not set(requested)<=set(adjacency):raise ContractError('Requested HC output IDs must belong to supplied gap graph')
    needed=set(requested);pending=list(requested)
    while pending:
        for neighbour in adjacency[pending.pop()]-needed:needed.add(neighbour);pending.append(neighbour)
    if not needed-set(direct):
        edges=[(e,k) for e,k in positive_edges if set(e)<=set(direct)]
        if not edges:return component
        operator=EffectiveHCGapOperator(direct,[e for e,_ in edges],
            Quantity([k for _,k in edges],'ms^-1','effective_HC_gap_rate',gap.rate_quantity.evidence_tag),
            assumption_profile_name=gap.assumption_profile_name,project_evidence_tag=gap.evidence_tag,
            equal_effective_capacity=True,common_state_scale_reference=True,enabled=True)
        return with_native_hc_gap(component,operator)
    native=BackgroundReferencedPassiveHC(component,gap,hc_types,query_rows,prepared_blocks,preparation,
        background,geometry_graph_hash,requested,root)
    if not native.references:
        if tuple(native.gap.cell_ids)!=direct:
            # Match the original state order while keeping the same supplied pairs.
            native.gap=EffectiveHCGapOperator(direct,native.gap.edges,
                Quantity(native.gap.rate,'ms^-1','effective_HC_gap_rate',gap.rate_quantity.evidence_tag),
                assumption_profile_name=gap.assumption_profile_name,project_evidence_tag=gap.evidence_tag,
                equal_effective_capacity=True,common_state_scale_reference=True,enabled=True)
        return with_native_hc_gap(component,native.gap)
    if hc_population is None or not set(native.ids)<=set(hc_population.cells.cell_id):
        raise ContractError('Supplied real HC population required for passive state members; no invented cells')
    if hc_population.population_id!=component.native_model.hcs.population_id:
        raise ContractError('Expanded HC population must retain the original binding population ID')
    supplied=dict(zip(hc_population.cells.cell_id,hc_population.cells.subtype))
    if any(supplied[g]!=dict(hc_types)[g] for g in native.ids):
        raise ContractError('Passive HC type map differs from supplied real population')
    outputs={k:replace(p,cell_ids=native.ids) if k in ('z2','Vh') else p for k,p in component.outputs.items()}
    populations=tuple(hc_population if p.population_id==component.native_model.hcs.population_id else p for p in component.populations)
    return replace(component,native_model=native,state_specs=native.state_specs,initial=native.initial_conditions,
        derivative=native.derivative,observe=native.observe,outputs=outputs,populations=populations,
        layers=component.layers+('HC_background_referenced_passive',),
        implementation_hash=canonical_hash({'original':component.implementation_hash,'passive_model':native.model_hash}))
