"""Sparse frame-specific contacts with explicit pooling/count/tail diagnostics."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from .contracts import ContractError,Quantity
from .graph import ConnectionTable

@dataclass(frozen=True)
class SparseRule:
    rule_id: str
    geometry_frame: str
    radius: Quantity
    cutoff_epsilon: float
    normalization: str
    source_profile_id: str
    receptor: str
    evidence_tag: str
    delay_seconds: float
    def __post_init__(self):
        if not self.rule_id or not self.geometry_frame or not 0<self.cutoff_epsilon<1 or self.normalization not in ('unit_pooling','absolute_contact_weight') or not self.source_profile_id or not self.receptor or self.delay_seconds<0: raise ContractError('Complete explicit connection rule required')

def _positions(q,frame):
    if q.unit not in ('mm','m','deg') or q.values.ndim!=2 or q.values.shape[1] not in (2,3) or q.coordinate_frame!=frame: raise ContractError('Connection coordinates must use rule frame and physical/functional unit explicitly')
    return q.values

def sparse_gaussian_contacts(pre_ids,pre_positions,post_ids,post_positions,rule,*,eligible_pre=None,effective_target_count=None,halo_complete=False):
    pre=_positions(pre_positions,rule.geometry_frame);post=_positions(post_positions,rule.geometry_frame)
    if pre_positions.unit!=post_positions.unit: raise ContractError('Geometry unit conversion must be explicit')
    if len(pre)!=len(pre_ids) or len(post)!=len(post_ids): raise ContractError('Contact cell IDs/coordinates mismatch')
    radius=rule.radius.require(pre_positions.unit,'pooling_radius')
    radius=np.broadcast_to(radius,(len(post),))
    if np.any(radius<=0):raise ContractError('Positive sourced pooling radii required')
    if not halo_complete:raise ContractError('Declared halo completeness required; missing contacts cannot be reflected/duplicated')
    eligible=np.ones(len(pre),dtype=bool) if eligible_pre is None else np.asarray(eligible_pre,dtype=bool)
    index=np.flatnonzero(eligible);tree=cKDTree(pre[index]);rcut=radius*np.sqrt(np.log(1/rule.cutoff_epsilon));neighbors=tree.query_ball_point(post,rcut)
    ids=[];pres=[];posts=[];weights=[];diagnostics=[]
    for j,candidates in enumerate(neighbors):
        candidates=index[np.array(candidates,dtype=int)]
        if not len(candidates):raise ContractError('Empty presynaptic contact support; enlarge halo/source radius')
        d=np.linalg.norm(pre[candidates]-post[j],axis=1);order=np.argsort(d,kind='stable');candidates=candidates[order];d=d[order]
        w=np.exp(-(d/radius[j])**2)
        outside_bound=rule.cutoff_epsilon*max(0,len(index)-len(candidates));degree_omission=0.
        if effective_target_count is not None:
            target=float(np.broadcast_to(effective_target_count,(len(post),))[j])
            if target<1 or len(candidates)<int(np.ceil(target)): raise ContractError('Degree target infeasible; no hidden replacement/thinning')
            count=int(np.ceil(target));degree_omission=float(w[count:].sum());candidates=candidates[:count];w=w[:count]
            if target%1:
                degree_omission+=float(w[-1]*(1-target%1));w[-1]*=target%1
        before=float(w.sum())
        if rule.normalization=='unit_pooling':w/=before
        for i,value in zip(candidates,w):
            ids.append(f'{rule.rule_id}:{post_ids[j]}:{pre_ids[i]}');pres.append(pre_ids[i]);posts.append(post_ids[j]);weights.append(float(value))
        diagnostics.append({'post_cell_id':post_ids[j],'edge_count':len(candidates),'effective_count':float(w.sum()**2/(w@w)),'normalization_input_sum':before,'nonuniform_omitted_weight_upper_bound':float(outside_bound+degree_omission),'degree_selection_omitted_weight':degree_omission,'tail_bound_note':'Discrete bound over existing eligible nodes; not uniform2D analytic tail; absent anatomy still requires halo provenance','cutoff_radius':float(rcut[j]),'radius_unit':pre_positions.unit})
    n=len(ids)
    table=ConnectionTable(tuple(ids),tuple(pres),tuple(posts),(rule.receptor,)*n,tuple(weights),('dimensionless',)*n,('nonnegative_weight_mechanism_owns_sign',)*n,(rule.delay_seconds,)*n,(rule.source_profile_id,)*n,(rule.normalization,)*n,(rule.evidence_tag,)*n)
    return table,{'rule_id':rule.rule_id,'frame':rule.geometry_frame,'rows':diagnostics,'graph_storage':'O(N+E), cKDTree candidates, no dense all-pairs distances'}

def private_line_contacts(parent_cone_ids,bc_ids,source_profile_id,evidence_tag,*,receptor,delay_seconds):
    if len(parent_cone_ids)!=len(bc_ids) or len(set(bc_ids))!=len(bc_ids):raise ContractError('One explicit cone parent per private-line BC required')
    n=len(bc_ids)
    return ConnectionTable(tuple('private:'+x for x in bc_ids),tuple(parent_cone_ids),tuple(bc_ids),(receptor,)*n,(1.,)*n,('dimensionless',)*n,('nonnegative_weight_mechanism_owns_sign',)*n,(delay_seconds,)*n,(source_profile_id,)*n,('explicit_private_line',)*n,(evidence_tag,)*n)

def enforce_midget_parent_matrix(matrix,pre_bc_ids,parent_bc_ids,effective_target,*,pre_positions=None,post_positions=None,support_mm=None):
    """Use explicit qualified parent IDs whenever the fitted effective target <=1."""
    if matrix.shape!=(len(parent_bc_ids),len(pre_bc_ids)) or len(set(pre_bc_ids))!=len(pre_bc_ids):
        raise ContractError('Private-line parent layout mismatch')
    index={cell_id:i for i,cell_id in enumerate(pre_bc_ids)}
    result=matrix.tolil()
    geometry=(pre_positions,post_positions,support_mm)
    if any(x is not None for x in geometry) and not all(x is not None for x in geometry):
        raise ContractError('Private-line hard support requires both contact frames and radius')
    radius=None if support_mm is None else np.broadcast_to(support_mm,(len(parent_bc_ids),))
    for row in np.flatnonzero(np.asarray(effective_target)<=1):
        parent=parent_bc_ids[row]
        if parent not in index:raise ContractError('Missing qualified parent BC ID; nearest substitution forbidden')
        if radius is not None:
            distance=np.linalg.norm(np.asarray(pre_positions)[index[parent]]-np.asarray(post_positions)[row])
            if not np.isfinite(radius[row]) or radius[row]<=0 or distance>radius[row]+1e-12:
                raise ContractError('Qualified parent BC lies outside declared contact support; no distant private-line fallback')
        result[row,:]=0.;result[row,index[parent]]=1.
    return result.tocsr()

def wool_midget_profile(temporal_equivalent_mm):
    x=np.asarray(temporal_equivalent_mm,dtype=float)
    if np.any(x<0) or np.any(x>10):raise ContractError('Wool convergence source/profile support0..10mm; not engine-wide bound')
    branch=x<.25;n=.29*x*x+.83*x-.28;effective=np.where(branch,1,np.maximum(1,n));radius=np.where(branch,np.nan,.002738*x**1.327)
    return {'effective_contact_target':effective,'center_radius_mm':radius,'private_line_branch':branch,'raw_nW':n,'gaussian_convention':'exp(-(d/Rc)^2)','evidence':'Published source fit0.25..10 temporal-equivalent; private-line/minimum1 are explicit project conventions','needed_adapter':'Source temporal-equivalence or named iso-density inverse; no global retinal-x substitution'}

def smooth_parameter_field(eccentricity,low_anchor,high_anchor,e0,e1,*,positive,source_context_compatible,approximation_id):
    if e1<=e0 or not source_context_compatible or not approximation_id:raise ContractError('Compatible sourced anchors and declared blending convention required')
    e=np.asarray(eccentricity);u=np.clip((e-e0)/(e1-e0),0,1);w=3*u*u-2*u*u*u
    if positive:
        if np.any(np.asarray(low_anchor)<=0) or np.any(np.asarray(high_anchor)<=0):raise ContractError('Log field requires positive anchors')
        value=np.exp((1-w)*np.log(low_anchor)+w*np.log(high_anchor))
    else:value=(1-w)*low_anchor+w*high_anchor
    return value,{'evidence':'project_approximation continuous field, not measured intermediate physiology','approximation_id':approximation_id,'boundaries_are_conventions':True}
