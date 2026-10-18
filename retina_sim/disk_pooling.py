"""Same finite pooling arithmetic with a reusable global tree for CSR blocks.

The original spatial_profiles module stays byte-stable for existing model/cache
identities. This adapter retains query order, support, targets and normalization.
"""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from .contracts import ContractError

def bounded_matrix(pre,target,k,*,support_mm,gaussian_scale=None,policy='finite_induced_graph',pre_tree=None):
    """Hard support first, then declared pooling normalization; deficits are exported."""
    pre,target=np.asarray(pre),np.asarray(target)
    radius=np.broadcast_to(support_mm,(len(target),))
    if np.any(radius<=0):raise ContractError('Positive explicit hard support required')
    tree=cKDTree(pre) if pre_tree is None else pre_tree
    if pre_tree is not None and (tree.data.shape!=pre.shape or (len(pre)>0 and not np.shares_memory(tree.data,pre))):
        raise ContractError('Reused finite-pool tree must index the exact supplied coordinate array')
    rows=[];cols=[];weights=[];audit=[]
    scale=None if gaussian_scale is None else np.broadcast_to(gaussian_scale,(len(target),))
    desired=np.broadcast_to(k,(len(target),))
    for j,point in enumerate(target):
        candidates=np.array(tree.query_ball_point(point,radius[j]),dtype=int)
        available=len(candidates)
        distance=np.linalg.norm(pre[candidates]-point,axis=1)
        order=np.argsort(distance,kind='stable')[:int(np.ceil(desired[j]))]
        candidates,distance=candidates[order],distance[order]
        w=np.ones(len(candidates)) if scale is None else np.exp(-(distance/scale[j])**2)
        if len(w) and desired[j]%1 and len(w)==int(np.ceil(desired[j])):w[-1]*=desired[j]%1
        before=float(w.sum())
        if before:w/=before
        rows.extend([j]*len(candidates));cols.extend(candidates.tolist());weights.extend(w.tolist())
        audit.append({'post_index':j,'target_degree':float(desired[j]),'actual_degree':len(candidates),'available_candidate_degree':available,
            'edge_count':len(candidates),'sum_weights_before_normalization':before,'sum_weights_after_normalization':float(w.sum()),
            'N_eff':float(w.sum()**2/(w@w)) if len(w) and w@w>0 else 0.,
            'missing_degree':max(0,int(np.ceil(desired[j]))-len(candidates)),
            'hard_support_mm':float(radius[j]),'maximum_distance_mm':float(distance.max()) if len(distance) else None,
            'normalization_input_sum':before,'boundary_policy':policy})
    return sparse.csr_matrix((weights,(rows,cols)),shape=(len(target),len(pre))),audit
