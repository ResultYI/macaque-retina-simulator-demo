"""One explicit identity for the actual compiled geometry and configuration.

ROI recording views are separate from the canonical neural geometry. A shared
identity is byte/configuration evidence, never physiological verification.
"""
from .contracts import ContractError,canonical_hash

SCHEMA='compiled_typed_geometry_identity_v1'

def compiled_geometry_identity(compiled):
    if hasattr(compiled,'global_disk_circuit'):
        from .disk_geometry_identity import global_geometry_identity
        return global_geometry_identity(compiled)
    links=[{'specification':x.specification,'shape':x.matrix.shape,'data':x.matrix.data,
        'indices':x.matrix.indices,'indptr':x.matrix.indptr} for x in compiled.links]
    internal={}
    for cid,c in compiled.components.items():
        matrices={}
        for name in ('A','B'):
            m=getattr(c.native_model,name,None)
            if m is not None and hasattr(m,'indptr'):
                matrices[name]={'shape':m.shape,'data':m.data,'indices':m.indices,'indptr':m.indptr}
        if matrices:internal[cid]=matrices
    fields={'assembly_hash':compiled.assembly_hash,
        'actual_populations_hash':canonical_hash(tuple(compiled.populations.values())),
        'actual_links_and_weights_hash':canonical_hash(links),
        'native_spatial_weights_hash':canonical_hash(internal),
        'gap_topology_conductance_hash':canonical_hash(compiled.spec.gap_junctions),
        'components_hash':canonical_hash({k:c.identity for k,c in compiled.components.items()}),
        'clock_hash':canonical_hash(compiled.clock),
        'source_binding':getattr(compiled,'geometry_source_binding',None)}
    return {'schema':SCHEMA,'sha256':canonical_hash(fields),'fields':fields}

def require_geometry_identity(declared,compiled):
    expected=compiled_geometry_identity(compiled)
    if declared!=expected:raise ContractError('Actual typed geometry/configuration identity mismatch; no old-geometry fallback')
    return expected
