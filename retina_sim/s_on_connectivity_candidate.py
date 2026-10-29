"""Opt-in, one-edge S_ON group-input hypothesis; no fine BB reconstruction.

Only singleton existing groups are eligible for the changed row and sources.
The secondary fraction is an explicit engineering reference, not a contact
count, conductance estimate, or a new production default.
"""
import numpy as np
from scipy import sparse
from .contracts import ContractError,canonical_hash

def one_secondary_S_input(original,cone_types,cone_members,BC_members,*,target_row,secondary_column,secondary_fraction):
    W=sparse.csr_matrix(original,dtype=float,copy=True);W.sort_indices()
    kinds=np.asarray(cone_types);cm=np.asarray(cone_members);bm=np.asarray(BC_members)
    if (kinds.shape!=(W.shape[1],) or cm.shape!=kinds.shape or bm.shape!=(W.shape[0],)
            or np.any(cm<=0) or np.any(bm<=0) or not np.isfinite(W.data).all() or np.any(W.data<0)):
        raise ContractError('Explicit actual group types/capacities and nonnegative input required')
    if not (0<=target_row<W.shape[0] and 0<=secondary_column<W.shape[1]):raise ContractError('Unknown actual group row')
    if not np.isfinite(secondary_fraction) or not 0<secondary_fraction<.5:raise ContractError('Dominant original S input must remain >one half')
    left,right=W.indptr[target_row:target_row+2];columns=W.indices[left:right]
    if len(columns)!=1 or W.data[left]!=1.:raise ContractError('Changed row must have one explicit original unit input')
    primary=int(columns[0]);secondary=int(secondary_column)
    if primary==secondary or kinds[primary]!='S' or kinds[secondary]!='S':raise ContractError('Distinct S-cone inputs required; no L/M substitution')
    if bm[target_row]!=1 or cm[primary]!=1 or cm[secondary]!=1:
        raise ContractError('Do not assign different fine connections inside unresolved multi-member groups')
    result=W.tolil();result[target_row,primary]=1-secondary_fraction;result[target_row,secondary]=secondary_fraction
    result=result.tocsr();result.sort_indices()
    if not np.allclose(result@np.ones(W.shape[1]),W@np.ones(W.shape[1]),rtol=0,atol=1e-15):raise ContractError('Original constant-input gain changed')
    change=result-W;other=change.copy().tolil();other[target_row,:]=0
    if other.tocsr().nnz:raise ContractError('Unselected S_ON group changed')
    descriptor=dict(target_BC_group_row=int(target_row),primary_cone_group_column=primary,secondary_cone_group_column=secondary,
        weights=[float(1-secondary_fraction),float(secondary_fraction)],
        original_matrix_hash=canonical_hash(dict(shape=W.shape,data=W.data,indices=W.indices,indptr=W.indptr)),
        candidate_matrix_hash=canonical_hash(dict(shape=result.shape,data=result.data,indices=result.indices,indptr=result.indptr)),
        interpretation='One secondary S-input hypothesis at group resolution; actual singleton groups, no recovered fine BB identity',
        row_semantics='Convex mean of native release; preserve DC gain, not a synaptic current/contact count',
        column_semantics='Information fanout; column sums need not be1; no photon or transmitter-budget conservation claim',
        secondary_fraction_status='Unfitted engineering reference; source supports dominant/secondary structure but not this fraction',
        default_selected=False,production_adopted=False,FIT_READY=False)
    return result,descriptor
