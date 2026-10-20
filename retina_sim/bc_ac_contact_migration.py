"""Insert stateless transforms while retaining each biological connection table."""
from dataclasses import replace
import numpy as np
from scipy import sparse
from .assembly import ComponentSpec, PortLink, RetinaBuilder
from .graph import ConnectionTable
from .contracts import ContractError, canonical_hash
from .bc_ac_receptor_components import BC_TYPES
from .bc_ac_contact_components import BC_KEY, AC_KEY, STATIC_KEY
from .contact_first_receptor import ROUTE, POOLING, AGGREGATION, validate_reduction


def contact_first_spec(parent, specs, targets, candidate):
    specs=list(specs); by_id={s.component_id:s for s in specs}
    links=list(parent.spec.links); external=list(parent.spec.external_inputs); added=[]; reports={}
    for cid in targets:
        owner=by_id[cid];old=parent.components[cid];ids=old.inputs['drive'].cell_ids
        incoming=[(j,l) for j,l in enumerate(links) if l.target_component==cid and l.target_port=='drive']
        ext=[(j,e) for j,e in enumerate(external) if e.target_component==cid and e.target_port=='drive']
        if bool(incoming)==bool(ext) or len(ext)>1:
            raise ContractError('Contact-first migration needs explicit linked OR one external raw input')
        matrices=[]; contracts=[]
        def identity(source,source_port,target,target_port,carrier,frame,tag):
            n=len(carrier);table=ConnectionTable(tuple(f'compute_identity:{source}>{target}:{i}' for i in range(n)),
                tuple(carrier),tuple(carrier),('computational_identity_not_biological_contact',)*n,(1.,)*n,
                ('dimensionless',)*n,('identity_transport',)*n,(0.,)*n,(owner.parameter_profile_id,)*n,
                ('no_normalization_identity_only',)*n,(tag,)*n)
            return PortLink(source,source_port,target,target_port,table,frame)
        def adapter(aid,carrier,meaning):
            added.append(ComponentSpec(aid,STATIC_KEY,dict(owner.population_roles),owner.parameter_profile_id,
                dict(candidate_dynamics=owner.options['candidate_dynamics'],carrier_cell_ids=list(carrier),
                     input_meaning=meaning,computational_only=True,biological_owner=cid)))
        for j,l in incoming:
            if any(d!=0 for d in l.connections.delay_seconds):
                raise ContractError('Contact-first v2 does not reinterpret delayed raw-input prehistory')
            pre=parent.components[l.source_component].outputs[l.source_port];carrier=pre.cell_ids
            aid=f'{cid}__contact_occupancy_{len(contracts)}'
            adapter(aid,carrier,pre.meaning)
            # Actual W/edge IDs/endpoints stay in the original table. Only its
            # signal source is the algebraic occupancy on those same pre IDs.
            links[j]=replace(l,source_component=aid,source_port='occupied')
            added_link=identity(l.source_component,l.source_port,aid,'drive',carrier,l.geometry_frame,pre.evidence_tag)
            contracts.append(added_link)
            pi={x:i for i,x in enumerate(carrier)};po={x:i for i,x in enumerate(ids)}
            W=sparse.coo_matrix((l.connections.weight,([po[x] for x in l.connections.post_cell_id],
                [pi[x] for x in l.connections.pre_cell_id])),shape=(len(ids),len(carrier))).tocsr()
            matrices.append(W)
        for j,e in ext:
            aid=cid+'__external_contact_occupancy';adapter(aid,ids,old.inputs['drive'].meaning)
            external[j]=replace(e,target_component=aid)
            # This is an explicitly one-input-per-cell clamp, not inferred contacts.
            contracts.append(identity(aid,'occupied',cid,'drive',ids,'OPL' if cid in BC_TYPES else 'IPL',old.evidence_tag))
            matrices.append(sparse.eye(len(ids),format='csr'))
        W=sparse.hstack(matrices,format='csr')
        family=candidate.data['families'][cid]['parameters'];n=len(ids)
        def array(k):
            row=family[k]
            if row['binding']=='derived_background':
                return np.asarray(owner.options['candidate_dynamics']['background_initial_values'][k],float)
            return np.full(n,float(row['value'])) if row['binding']=='broadcast' else np.asarray(row['value'],float)
        _,totals=validate_reduction(W,array('tau_e'),array('e0'),contract=candidate.data['aggregation_contract'])
        contract=dict(owner.options['candidate_dynamics'],row_sum_W=totals.tolist(),
            aggregation_contract=AGGREGATION,initial_mapping='e0 is supplied sum(W*m0); contact states are not reconstructed',
            shared_K_hill_across_target_cells=True)
        by_id[cid]=replace(owner,mechanism=BC_KEY if cid in BC_TYPES else AC_KEY,
            options=dict(owner.options,candidate_dynamics=contract))
        # Static node uses the same profile identity and approved reduction.
        for k,a in enumerate(added):
            if a.options['biological_owner']==cid:
                added[k]=replace(a,options=dict(a.options,candidate_dynamics=contract))
        links.extend(contracts)
        reports[cid]=dict(row_sum_W=totals.tolist(),initial_state_mapping='explicit_weighted_aggregate',
            fixed_weights=True,fixed_shared_contact_tau=True,extra_dynamic_states=0)
    return replace(parent.spec,components=tuple(by_id[s.component_id] for s in specs)+tuple(added),
        links=tuple(links),external_inputs=tuple(external)),reports


def validate_retained_tables(parent,result):
    old=[canonical_hash(l.connections) for l in parent.spec.links]
    new=[canonical_hash(l.connections) for l in result.spec.links
        if not (l.connections.synapse_type and all(t=='computational_identity_not_biological_contact' for t in l.connections.synapse_type))]
    if sorted(old)!=sorted(new) or canonical_hash(parent.spec.gap_junctions)!=canonical_hash(result.spec.gap_junctions):
        raise ContractError('Contact-first migration changed biological W/IDs or electrical contacts')
