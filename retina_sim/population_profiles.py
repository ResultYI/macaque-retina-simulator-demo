"""Explicit density/effective-population providers; no implicit subtype counts."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from .contracts import ContractError

class HCDensityField:
    def __init__(self,source,subtype,*,profile_id,interval_policy,central_policy,angular_policy):
        if subtype not in ('H1','H2') or not profile_id or interval_policy!='midpoint' or central_policy not in ('source_estimate_constant_cap','no_central_extension') or angular_policy!='axisymmetric_project_extension': raise ContractError('Named HC interval/central/angular profile required')
        self.subtype=subtype;rows=source['table2'];x=[sum(r['eccentricity_interval_mm'])/2 for r in rows];y=[r[subtype] for r in rows]
        if central_policy=='source_estimate_constant_cap':
            peak=source['central_author_estimate'];x=[sum(peak['eccentricity_interval_mm'])/2]+x;y=[peak[subtype]]+y
        self.x=np.array(x);self.p=PchipInterpolator(x,np.log(y),extrapolate=False);self.central_policy=central_policy
        self.metadata={'profile_id':profile_id,'species':'Table2 M.fascicularis across specimens; not one individual retina','frame':'OPL physical sampling under explicit soma/dendrite co-location assumption','interval_policy':interval_policy,'central_policy':central_policy,'angular_policy':angular_policy,'density_unit':'cells/mm^2','extension_beyond_last_source':'not enabled'}
    def __call__(self,xy):
        r=np.linalg.norm(np.asarray(xy),axis=-1)
        if np.any(r>self.x[-1]) or self.central_policy=='no_central_extension' and np.any(r<self.x[0]):raise ContractError('HC density outside selected source-composite support; choose separate profile')
        return np.exp(self.p(np.maximum(r,self.x[0])))

def effective_population_centers(parent_ids,parent_positions,*,resolution_profile,source_id):
    if resolution_profile!='effective_unit' or not source_id:raise ContractError('Unknown anatomical density requires explicit effective-unit resolution')
    return {'parent_ids':tuple(parent_ids),'positions':parent_positions,'evidence_class':'project_approximation','representation':'effective units, not real subtype cell counts','source_id':source_id}

def gap_junction_operator(conductance,voltage):
    from scipy import sparse
    G=sparse.csr_matrix(conductance,dtype=float)
    if G.shape!=(len(voltage),len(voltage)) or np.any(G.data<0) or (G-G.T).nnz or np.any(G.diagonal()!=0):raise ContractError('Gap junctions require symmetric nonnegative sourced physical conductance')
    return G@voltage-np.asarray(G.sum(axis=1)).ravel()*voltage
