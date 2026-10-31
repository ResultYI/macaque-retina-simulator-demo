"""Complete fine-cell HC operators aggregated without hiding unsupported members."""
import numpy as np
from scipy import sparse
from .contracts import ContractError


def _normalize_complete_rows(matrix):
    matrix=sparse.csr_matrix(matrix,dtype=np.float64,copy=True)
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all() or np.any(matrix.data<0):
        raise ContractError('Finite nonnegative complete contact weights required')
    mass=np.asarray(matrix.sum(axis=1)).ravel()
    return (sparse.diags(np.divide(1.,mass,out=np.zeros_like(mass),where=mass>0))@matrix).tocsr(),mass>0


def _validate_mapping(mapping,counts,active,label):
    mapping=np.asarray(mapping);counts=np.asarray(counts)
    if (mapping.shape!=active.shape or mapping.dtype.kind not in 'iu' or counts.ndim!=1
            or counts.dtype.kind not in 'iu' or np.any(counts<=0)
            or np.any(mapping < -1) or np.any(mapping>=len(counts))):
        raise ContractError('Invalid original fine-to-group member map: '+label)
    if np.any((mapping<0)&active):
        raise ContractError('Active contact reaches unrepresented geometry; expand actual dependency: '+label)
    valid=mapping>=0
    if not np.array_equal(np.bincount(mapping[valid],minlength=len(counts)),counts):
        raise ContractError('Declared members differ from actual source rows: '+label)
    return mapping,counts,valid


def _member_mean(matrix,post_map,pre_map,counts,shape):
    coo=matrix.tocoo()
    if np.any(post_map[coo.row]<0) or np.any(pre_map[coo.col]<0):
        raise ContractError('Nonzero dependency omitted by representative map')
    summed=sparse.csr_matrix((coo.data,(post_map[coo.row],pre_map[coo.col])),shape=shape)
    return (sparse.diags(1./counts)@summed).tocsr()


def project_hc_contacts(raw_HC_by_cone,cone_types,HC_types,cone_map,HC_map,cone_counts,HC_counts):
    """Normalize each fine row once, then take the declared member mean.

    H1 excludes S direct input. H2 accepts LMS input, but feedback stays H1->LM
    and H2->S. No group/crop renormalization, radius change, gain or partner fill.
    Empty/partial support is retained and explicitly reported as unsupported.
    """
    raw=sparse.csr_matrix(raw_HC_by_cone,dtype=np.float64,copy=True)
    raw.eliminate_zeros()
    ct,ht=np.asarray(cone_types),np.asarray(HC_types)
    if raw.shape!=(len(ht),len(ct)) or not set(ct)<=set('LMS') or not set(ht)<={'H1','H2'}:
        raise ContractError('Actual resolved HC/cone source types and dimensions required')
    coo=raw.tocoo()
    if np.any((ht[coo.row]=='H1')&(ct[coo.col]=='S')):
        raise ContractError('Raw graph violates H1 S exclusion; do not silently repair')
    fine_A,HC_active=_normalize_complete_rows(raw)
    allowed=((ht[coo.row]=='H1')&(ct[coo.col]!='S'))|((ht[coo.row]=='H2')&(ct[coo.col]=='S'))
    feedback=sparse.csr_matrix((coo.data[allowed],(coo.col[allowed],coo.row[allowed])),shape=raw.shape[::-1])
    fine_B,cone_active=_normalize_complete_rows(feedback)
    cm,cc,cv=_validate_mapping(cone_map,cone_counts,np.asarray(raw.sum(axis=0)).ravel()>0,'cones')
    hm,hc,hv=_validate_mapping(HC_map,HC_counts,HC_active,'HC')
    A=_member_mean(fine_A,hm,cm,hc,(len(hc),len(cc)))
    B=_member_mean(fine_B,cm,hm,cc,(len(cc),len(hc)))
    cone_fraction=np.bincount(cm[cv],weights=cone_active[cv],minlength=len(cc))/cc
    HC_fraction=np.bincount(hm[hv],weights=HC_active[hv],minlength=len(hc))/hc
    if (not np.allclose(np.asarray(A.sum(1)).ravel(),HC_fraction,rtol=0,atol=2e-14)
            or not np.allclose(np.asarray(B.sum(1)).ravel(),cone_fraction,rtol=0,atol=2e-14)):
        raise ContractError('Representative rows must retain actual supported-member fractions')
    return dict(A=A,typed_B=B,cone_supported_member_fraction=cone_fraction,HC_supported_member_fraction=HC_fraction,
        omitted_zero_contact_geometry_HC=int((~hv).sum()),
        normalization='fine complete typed row normalization then all-member mean; no second normalization',
        unsupported_policy='retain explicit fraction/mask; do not claim biological zero or complete support',
        FIT_READY=False,background_qualified=False,production_adopted=False)
