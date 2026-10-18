"""Compact actual-value archives and a recording view of the global model.

Only observations/JSON exports are reduced. Neural states, frozen connections,
parameters, inputs and the solver remain the complete selected dependency view.
"""
from pathlib import Path
from enum import Enum
import numpy as np
from scipy import sparse
from retina_sim.contracts import Quantity
from retina_sim.disk_graph_storage import array_identity,file_sha,save_json

def prepare_global_audit(compiled,audit,directory):
    directory=Path(directory);archive=directory/'global_definition';archive.mkdir(exist_ok=True)
    chosen=set()
    for name,pop in compiled.populations.items():
        if name not in ('cones','HC'):chosen.update(pop.cells.cell_id)
    columns=[link.matrix.indices for link in compiled.links if link.specification.source_component=='outer_native']
    cone_rows=np.unique(np.concatenate(columns)) if columns else np.flatnonzero(audit['population_owner_mask']['cones'])
    native=compiled.components['outer_native'].native_model
    hc_rows=np.unique(native.B[cone_rows].indices)
    chosen.update(compiled.populations['cones'].cells.cell_id[i] for i in cone_rows)
    chosen.update(compiled.populations['HC'].cells.cell_id[i] for i in hc_rows)
    compiled.platform_recording_cell_ids=frozenset(chosen)
    view=compiled.global_disk_view;circuit=compiled.global_disk_circuit
    rows_file=archive/'exact_view_rows.npz'
    with rows_file.open('xb') as stream:np.savez(stream,**view['selected_rows'],recorded_native_local_rows=cone_rows,recorded_HC_local_rows=hc_rows)
    view_binding={'path':str(rows_file),'sha256':file_sha(rows_file),'global_graph':circuit.binding}
    parameter_evidence={};component_specs=[]
    for component_spec in compiled.spec.components:
        component=compiled.components[component_spec.component_id];arrays={}
        def compact(value):
            if isinstance(value,Quantity):return {'values':compact(value.values),'unit':value.unit,'meaning':value.meaning,
                'evidence_tag':value.evidence_tag,'coordinate_frame':value.coordinate_frame}
            if isinstance(value,np.ndarray):
                key='a%06d'%len(arrays);arrays[key]=value
                return {'array_key':key,**array_identity(value)}
            if sparse.issparse(value):
                m=value.tocsr();return {'sparse_CSR':True,'shape':list(m.shape),'data':compact(m.data),'indices':compact(m.indices),'indptr':compact(m.indptr)}
            if isinstance(value,Enum):return value.value
            if hasattr(value,'__dataclass_fields__'):return {key:compact(getattr(value,key)) for key in value.__dataclass_fields__}
            if isinstance(value,dict):return {str(key):compact(item) for key,item in value.items()}
            if isinstance(value,(list,tuple)):return [compact(item) for item in value]
            if isinstance(value,np.generic):return value.item()
            return value
        records=compact(component.parameters.records);options=compact(component_spec.options)
        payload=archive/(component_spec.component_id+'_actual_values.npz')
        with payload.open('xb') as stream:np.savez_compressed(stream,**arrays)
        metadata={'schema':'actual_component_parameter_arrays_v1','component_id':component_spec.component_id,
            'parameter_hash':component.parameters.parameter_hash,'records':records,'binding_options':options,
            'array_file':{'path':str(payload),'sha256':file_sha(payload)},'physiological_validation':False}
        meta_file=payload.with_suffix('.json');save_json(meta_file,metadata)
        parameter_evidence[component_spec.component_id]={'schema':metadata['schema'],'parameter_hash':component.parameters.parameter_hash,
            'metadata':{'path':str(meta_file),'sha256':file_sha(meta_file)},'actual_value_arrays':metadata['array_file']}
        component_specs.append({'id':component_spec.component_id,'mechanism':component_spec.mechanism,
            'population_roles':component_spec.population_roles,'parameter_profile_id':component_spec.parameter_profile_id,
            'actual_values':parameter_evidence[component_spec.component_id]})
    audit['population_records']={name:pop.record(compiled.platform_recording_cell_ids) for name,pop in compiled.populations.items()}
    audit['spatial_profile']['HC_geometry']={'profile_id':'atlas_fixed_joint_master_v2','global_HC_table':circuit.record['tables']['HC'],
        'exact_view_rows':view_binding,'instantiated_count':len(compiled.populations['HC'].cells.cell_id),'recorded_count':len(hc_rows),
        'feedback_geometry':'unchanged frozen global rows; no observation-driven cropping or renormalization','physiological_validation':False}
    source={'schema':'global_compiled_dependency_spec_v1','model_id':compiled.spec.model_id,'assembly_hash':compiled.assembly_hash,
        'global_graph':circuit.binding,'exact_view_rows':view_binding,'components':component_specs,
        'required_mechanisms':list(compiled.spec.required_mechanisms),'scope':compiled.spec.scope,
        'parameters_and_native_options':'actual full arrays in content-identified per-component archives',
        'recording':'BC/AC/RGC/primary contacts and feedforward cone/HC observation cohort; all dependency states still integrated',
        'physiological_validation':False}
    # Release duplicated JSON anatomy retained only for construction/auditing.
    compiled.global_spatial_asset.source={'global_graph':circuit.binding,'exact_view':compiled.global_spatial_asset.binding}
    return {'source_spec':source,'parameter_evidence':parameter_evidence,'view_rows':view_binding}
