"""Counts on finite legal partner graphs; references never prescribe degrees."""
import numpy as np
from scipy import sparse


def reference_degrees(points,subtype):
    c,p=(3.7,7.5) if subtype=='H1' else (2.2,3.2)
    return c+(p-c)*np.clip((np.linalg.norm(points[:,:2],axis=1)-.75)/(5.5-.75),0,1)


def _summary(values):
    values=np.asarray(values,int)
    return {'count':len(values),'mean':float(values.mean()) if len(values) else None,
        'zero_rate':float(np.mean(values==0)) if len(values) else None,
        'histogram':{str(int(k)):int(v) for k,v in zip(*np.unique(values,return_counts=True))}}


def finite_contact_diagnostics(raw,pedicles,cone_types,hc_types,owner_cones,owner_hc,hc_rows,
                               normalize,forward_rule,feedback_rule):
    raw=sparse.csr_matrix(raw);owner_cones=np.asarray(owner_cones,bool)
    owner_hc=np.asarray(owner_hc,bool);hc_types=np.asarray(hc_types);cone_types=np.asarray(cone_types)
    complete_hc=np.array([r['terminal_generation_covers_full_envelope'] is True for r in hc_rows])
    stats={};legacy=raw.copy().tolil()
    for subtype in ('H1','H2'):
        hmask=hc_types==subtype;eligible=cone_types!='S' if subtype=='H1' else np.ones(len(cone_types),bool)
        degree=np.asarray((raw[hmask]!=0).sum(0)).ravel().astype(int)
        hdegree=np.diff(raw.indptr)
        u=owner_cones;v=owner_hc&hmask&complete_hc
        ref=reference_degrees(pedicles,subtype)
        all_owner=_summary(degree[u]);eligible_owner=_summary(degree[u&eligible])
        c_owner=_summary(hdegree[v])
        stats[subtype]={'D_all_owner':all_owner,'D_eligible_owner':eligible_owner,'C_complete_owner_HC':c_owner,
            'eligible_types':['L','M'] if subtype=='H1' else ['L','M','S'],
            'D_all_reference_mean':float(ref[u].mean()) if np.any(u) else None,
            'D_all_reference_residual':all_owner['mean']-float(ref[u].mean()) if np.any(u) else None,
            'reference_role':'population_diagnostic; original project interpolant of cross-cohort source means',
            'reference_denominator':'all owner cones; H1 includes ineligible S zero values',
            'source_population_pass_threshold':None,'status':'population_contact_uncalibrated',
            'excluded_incomplete_owner_HC':int(np.sum(owner_hc&hmask&~complete_hc)),
            'D_owner_edge_sum':int(degree[u].sum()),'C_owner_edge_sum':int(hdegree[v].sum()),
            'owner_cone_to_halo_HC_edges':int((raw[(~owner_hc)&hmask][:,u]!=0).nnz),
            'owner_HC_to_halo_cone_edges':int((raw[v][:,~u]!=0).nnz)}
        # Counterfactual old rounded selection on this SAME legal morphology
        # graph, not a successful old source-band-only network or a new target.
        indices=np.flatnonzero(eligible);quota=np.floor(ref[indices]).astype(int)
        extra=int(np.floor(ref[indices].sum()+.5)-quota.sum())
        quota[np.argsort(-(ref[indices]-quota),kind='stable')[:extra]]+=1
        for i,k in zip(indices,quota):
            rows=np.flatnonzero(hmask&(np.asarray(raw[:,i].toarray()).ravel()!=0))
            # Distances are supplied separately by the caller's geometry graph;
            # use actual anchor-distance order installed on hc_rows here.
            rows=sorted(rows,key=lambda j:(hc_rows[j]['cone_distance_lookup'][int(i)],int(j)))
            for j in rows[int(k):]:legacy[j,int(i)]=0
    legacy=legacy.tocsr();legacy.eliminate_zeros()
    before_A=normalize(legacy,forward_rule);before_B=normalize(legacy.T.tocsr(),feedback_rule)
    after_A=normalize(raw,forward_rule);after_B=normalize(raw.T.tocsr(),feedback_rule)
    rr,cc=raw.nonzero();retained=np.asarray(legacy[rr,cc]).ravel()!=0
    kept_r,kept_c=rr[retained],cc[retained]
    impacts={'scope':'counterfactual legacy rounded selector on same legal candidate graph; old strict full build remains failed',
        'legacy_full_network_pass':False,'before_edge_count':legacy.nnz,'after_edge_count':raw.nnz,
        'retained_pairs':int(retained.sum()),'added_pairs':int((~retained).sum()),'removed_pairs':0,
        'retained_hc_indices':kept_r.tolist(),'retained_cone_indices':kept_c.tolist(),
        'retained_forward_weight_before':np.asarray(before_A[kept_r,kept_c]).ravel().tolist(),
        'retained_forward_weight_after':np.asarray(after_A[kept_r,kept_c]).ravel().tolist(),
        'retained_feedback_weight_before':np.asarray(before_B[kept_c,kept_r]).ravel().tolist(),
        'retained_feedback_weight_after':np.asarray(after_B[kept_c,kept_r]).ravel().tolist(),
        'forward_row_sums_before':np.asarray(before_A.sum(1)).ravel().tolist(),
        'forward_row_sums_after':np.asarray(after_A.sum(1)).ravel().tolist(),
        'feedback_row_sums_before':np.asarray(before_B.sum(1)).ravel().tolist(),
        'feedback_row_sums_after':np.asarray(after_B.sum(1)).ravel().tolist(),
        'weight_unit':'dimensionless','weight_policy_id':'existing_unit_pooling_forward_feedback_v1',
        'rescale_to_divergence_target':False,'gain_changed':False,'gap_edges_changed':False}
    return stats,impacts
