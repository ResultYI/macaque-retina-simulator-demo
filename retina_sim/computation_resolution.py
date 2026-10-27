"""Independent spatial-scope and computation-granularity identity contract.

This adapter does not reconstruct fine cells from representatives. The current
real grouped patch uses the existing assembly/runner; unimplemented requests
are explicit unavailable capabilities, not a fallback to grouped dynamics.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from .contracts import ContractError,canonical_hash

SCHEMA='retina-computation-scope-resolution-1'


def resolve(root,config,spatial_scope,granularity):
    if config.get('schema')!=SCHEMA or spatial_scope not in config['spatial_scope_options'] or granularity not in config['granularity_options']:
        raise ContractError('Explicit valid independent scope and granularity required')
    root=Path(root);group=config['validated_grouped_local'];fine=config['fine_graph_expected']
    if granularity=='per_cell':
        fine_path=root/fine['path'];present=fine_path.is_file()
        status='UNAVAILABLE';reason='Fine-source graph missing and per-cell V+s runner not implemented' if not present else 'Per-cell V+s graph closure and runner not implemented or qualified'
        binding=dict(graph=fine,physical_fine_graph_exists=present,mapping=None)
    elif spatial_scope=='full_domain':
        status='UNAVAILABLE';reason='Full-domain grouped V+s assembly and source-bound IVP not registered'
        binding=dict(graph=None,mapping=None)
    else:
        mapping=group['group_mapping'];path=(root/mapping['path']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=mapping['sha256']:
            raise ContractError('Frozen group mapping identity changed')
        status='AVAILABLE';reason=None
        binding=dict(graph_artifact_sha256=group['graph_artifact_sha256'],mapping=mapping,prepared_model=group['prepared_model'])
    descriptor=dict(schema=SCHEMA,spatial_scope=spatial_scope,granularity=granularity,binding=binding)
    return dict(descriptor,identity=canonical_hash(descriptor),status=status,reason=reason,
        cache_checkpoint_namespace='resolution:'+canonical_hash(descriptor),cross_identity_reuse=False,
        group_values_are_independent_cells=False,automatic_fallback=False)


def require_available(resolution):
    if resolution['status']!='AVAILABLE':raise ContractError('Requested computation unavailable: '+str(resolution['reason']))
    return resolution


def grouped_members(population,rows):
    """Group recording semantics and membership, without duplicating states."""
    rows=np.asarray(rows,dtype=np.int64);ids=np.asarray(population.group_ids);counts=np.asarray(population.counts)
    if rows.ndim!=1 or np.any(rows<0) or np.any(rows>=len(ids)) or len(set(rows.tolist()))!=len(rows):raise ContractError('Unique actual group rows required')
    if getattr(population,'coordinate_role',None)=='effective_functional_RF_pool_not_anatomical_AC':
        return [dict(group_id=str(ids[i]),member_count=None,original_member_rows=None,
            paired_target_group_id=str(ids[i]).split(':',1)[1],
            state_semantics='one effective functional AC pool, paired to a target group; no anatomical AC membership or independent cell states') for i in rows]
    offsets=np.asarray(population.member_offsets);members=np.asarray(population.member_global_rows)
    if len(offsets)!=len(ids)+1 or offsets[0]!=0 or offsets[-1]!=len(members) or not np.array_equal(np.diff(offsets),counts):raise ContractError('Group-member source mapping differs')
    return [dict(group_id=str(ids[i]),member_count=int(counts[i]),original_member_rows=members[offsets[i]:offsets[i+1]].astype(int).tolist(),
        state_semantics='one representative group state; no independently solved member state') for i in rows]
