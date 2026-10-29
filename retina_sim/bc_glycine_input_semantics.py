"""Research admission for subtype-specific OFF-BC glycine template borrowing.

Channel names are not target-cell identity. A generic diffuse population cannot
become DB1 just because its retained numerical receptor is called AII_DB1.
This module annotates a retained input; it never drops edges or changes gain.
"""
from .contracts import ContractError, canonical_hash

TEMPLATES={'AII_FMB_glycine':'FMB','AII_DB1_glycine':'DB1'}
OFF_SUBTYPES={'FMB','DB1','DB2','DB3a','DB3b','UNRESOLVED'}


def bind_off_glycine_input(channel_id, target_population, target_group_ids, *,
        target_subtypes=None, borrowed_template=None, borrowing_reason=None):
    """Require an explicit subtype or borrowing declaration; preserve identity.

    The primary evidence is a relative input bias, not an exclusive whitelist.
    A different/unknown target can receive input under an explicit template
    borrowing hypothesis; it must retain its own target identity in the result.
    """
    if channel_id not in TEMPLATES:
        raise ContractError('Only declared retained OFF glycine channels supported')
    if target_population not in ('midget_BC_OFF','diffuse_BC_OFF'):
        raise ContractError('OFF glycine template requires an explicit OFF target')
    ids=tuple(str(x) for x in target_group_ids)
    if not ids or len(set(ids))!=len(ids):
        raise ContractError('Unique nonempty target group IDs required')
    labels=tuple(target_subtypes) if target_subtypes is not None else ('UNRESOLVED',)*len(ids)
    if len(labels)!=len(ids) or any(x not in OFF_SUBTYPES for x in labels):
        raise ContractError('Subtype declarations must align with actual target IDs')
    template=TEMPLATES[channel_id]
    borrow=(borrowed_template==template and isinstance(borrowing_reason,str) and bool(borrowing_reason.strip()))
    if borrowed_template is not None and not borrow:
        raise ContractError('Explicit matching template and nonempty borrowing reason required')
    if any(label!=template for label in labels) and not borrow:
        raise ContractError('Channel name cannot identify generic/different targets; declare template borrowing')
    records=[dict(target_group_id=gid,target_subtype=label,template_subtype=template,
        classification='declared_target_family' if label==template else 'explicit_engineering_borrow',
        target_identity_inferred_from_channel_name=False,
        quantitative_conductance_calibrated=False,other_subtypes_forced_zero=False)
        for gid,label in zip(ids,labels)]
    result=dict(schema='research-BC-glycine-target-semantics-1',channel_id=channel_id,
        target_population=target_population,records=records,borrowed_template=borrowed_template,
        borrowing_reason=borrowing_reason,numeric_policy='retain all existing conductances and edges unchanged',
        evidence_scope='Macaque subtype bias toward FMB/DB1; not exclusivity or per-cell quantitative calibration',
        production_adopted=False,FIT_READY=False)
    result['semantic_binding_id']=canonical_hash(result)
    return result
