"""Bounded-memory identity of actual global-view arrays and immutable bindings."""
from .contracts import canonical_hash
from .disk_graph_storage import array_identity

def global_geometry_identity(compiled):
    tuples={}
    def ids(values):
        key=id(values)
        if key not in tuples:tuples[key]=canonical_hash(values)
        return tuples[key]
    def quantity(value):
        if value is None:return None
        return {'array':array_identity(value.values),'unit':value.unit,'meaning':value.meaning,
            'evidence_tag':value.evidence_tag,'frame':value.coordinate_frame}
    def matrix(value):
        return {'shape':list(value.shape),'data':array_identity(value.data),'indices':array_identity(value.indices),'indptr':array_identity(value.indptr)}
    populations={}
    for name,pop in compiled.populations.items():
        cells=pop.cells
        populations[name]={'canonical_population_hash':pop._population_hash,
            'cell_tuple_fields':{field:ids(getattr(cells,field)) for field in ('cell_id','cell_type','subtype','species_profile_id','hemiretina','meridian','parameter_profile_id','evidence_tags')},
            'regions':ids(pop.regions),'eccentricity':quantity(cells.eccentricity),
            'coordinates':{field:quantity(getattr(pop,field)) for field in ('anatomical_positions','receptive_field_positions','pedicle_positions','terminal_positions','dendritic_positions')}}
    components={};native={}
    for name,component in compiled.components.items():
        ports={kind:{port_name:{'ids':ids(port.cell_ids),'unit':port.unit,'meaning':port.meaning,'evidence_tag':port.evidence_tag}
            for port_name,port in getattr(component,kind).items()} for kind in ('inputs','outputs')}
        components[name]={'mechanism':component.mechanism,'parameters':component.parameters.parameter_hash,
            'binding_options_hash':component.binding_options_hash,'ports':ports,'implementation':component.implementation_hash,
            'native_model_hash':getattr(component.native_model,'model_hash',None)}
        values={key:matrix(getattr(component.native_model,key)) for key in ('A','B') if hasattr(getattr(component.native_model,key,None),'indptr')}
        if values:native[name]=values
    links=[{'source':[link.specification.source_component,link.specification.source_port],
        'target':[link.specification.target_component,link.specification.target_port],
        'connection_hash':link.specification.connections.graph_hash,'matrix':matrix(link.matrix)} for link in compiled.links]
    fields={'hash_encoding':'actual_global_array_bytes_v1','assembly_hash':compiled.assembly_hash,
        'actual_populations_hash':canonical_hash(populations),'actual_links_and_weights_hash':canonical_hash(links),
        'native_spatial_weights_hash':canonical_hash(native),'components_hash':canonical_hash(components),
        'gap_topology_conductance_hash':canonical_hash(compiled.spec.gap_junctions),'clock_hash':canonical_hash(compiled.clock),
        'source_binding':compiled.geometry_source_binding}
    return {'schema':'compiled_typed_geometry_identity_v1','sha256':canonical_hash(fields),'fields':fields}
