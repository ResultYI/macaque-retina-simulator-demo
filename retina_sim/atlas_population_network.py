"""Execute the single frozen approved local population artifact, without fallback."""
from pathlib import Path
import json,hashlib
import numpy as np
from .atlas_runtime import atlas_midget_network,q
from .atlas_primary_rod import attach_primary_rod
from .contracts import ContractError
from .spatial import InputChannel
from .runtime_profiles import LightContext


def circuit_configuration(root, configuration_path=None):
    """Explicit candidate binding; the historical profile remains reproducible."""
    root=Path(root).resolve()
    path=(root/(configuration_path or 'configurations/atlas_isetbio_geometry_v1.json')).resolve()
    if not path.is_relative_to(root):raise ContractError('Circuit configuration must stay inside project')
    configuration=json.loads(path.read_bytes())
    if configuration.get('profile_id')!='atlas_isetbio_geometry_v1':raise ContractError('Unsupported circuit configuration')
    return configuration


def load_population_record(root, *, owner_bounds_mm=None, halo_mm=.04,execution_profile='atlas_population_revision_v1',configuration_path=None):
    """One identity-checked artifact for Run, GUI and CLI; never generate a substitute."""
    root=Path(root);config=json.loads((root/'configurations/retina_platform_v1.json').read_bytes())
    target=config.get('legacy_population_target',config['atlas_target'])
    binding=target['population_revision'];storage=target.get('population_revision_storage')
    if execution_profile=='atlas_isetbio_geometry_v1':
        configuration=circuit_configuration(root,configuration_path)
        if configuration.get('storage_kind')=='global_disk_graph_v1':
            if owner_bounds_mm is None:raise ContractError('Global population record requires an explicit recording ROI; inspect GlobalDiskCircuit for whole-disk geometry')
            from .disk_circuit import GlobalDiskCircuit
            from .disk_runtime import runtime_record
            circuit=GlobalDiskCircuit(root,configuration['global_graph'])
            view=circuit.owner_view(owner_bounds_mm,max_nodes=configuration['budgets']['max_dependency_nodes'])
            record,_=runtime_record(root,circuit,view)
            return record,{'artifact':configuration['global_graph']['path'],'sha256':configuration['global_graph']['sha256'],
                'kind':'global_disk_exact_dependency_view','view_sha256':record.semantic_hash}
        binding=configuration['frozen_graph'];storage=None
    elif execution_profile!='atlas_population_revision_v1':raise ContractError('Unknown population graph profile')
    if storage is not None:
        from .atlas_graph_storage import FrozenGraphReader
        if storage['source_json_sha256']!=binding['sha256']:raise ContractError('Storage does not represent the approved original graph')
        record=FrozenGraphReader(root/storage['path'],expected_source_sha256=binding['sha256'],expected_manifest_hash=storage['storage_manifest_hash'])
        if record.semantic_hash!=storage['logical_record_hash']:raise ContractError('Frozen graph logical identity differs')
    else:
        raw=(root/binding['artifact']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=binding['sha256']:raise ContractError('Approved population graph identity mismatch; no fallback')
        record=json.loads(raw)
    if (owner_bounds_mm is not None and not np.allclose(owner_bounds_mm,record['atlas']['owner_domain'],rtol=0,atol=1e-12)) or abs(halo_mm-.04)>1e-12:
        raise ContractError('LOCAL_POPULATION_ARTIFACT_UNSUPPORTED: owner/halo differ from frozen .35 artifact; no historical fallback')
    return record, binding


def atlas_population_network(root,*,duration=.0012,light_context=None,owner_bounds_mm=(.34,.36,-.01,.01),
        halo_mm=.04,max_spatial_cells=60000,seed=20261010,execution_profile='atlas_population_revision_v1',configuration_path=None,_freeze_geometry_bootstrap=False):
    root=Path(root)
    if execution_profile=='atlas_isetbio_geometry_v1':
        configuration=circuit_configuration(root,configuration_path)
        if configuration.get('storage_kind')=='global_disk_graph_v1':
            from .disk_runtime import disk_network
            return disk_network(root,configuration,duration=duration,light_context=light_context or LightContext(),
                owner_bounds_mm=owner_bounds_mm,halo_mm=halo_mm,max_spatial_cells=max_spatial_cells,seed=seed)
    record,binding=load_population_record(root,owner_bounds_mm=owner_bounds_mm,halo_mm=halo_mm,execution_profile=execution_profile,configuration_path=configuration_path)
    spatial_asset=None
    if execution_profile=='atlas_isetbio_geometry_v1':
        configuration=circuit_configuration(root,configuration_path)
        geometry_binding=configuration.get('typed_circuit_geometry')
        if geometry_binding is not None:
            from .circuit_geometry import FrozenCircuitGeometry
            spatial_asset=FrozenCircuitGeometry(root,geometry_binding,record,binding)
        elif not _freeze_geometry_bootstrap:raise ContractError('Frozen canonical typed circuit geometry required; no runtime spatial regeneration')
    light=light_context or LightContext()
    compiled,channels,audit=atlas_midget_network(root,duration=duration,light_context=light,owner_bounds_mm=owner_bounds_mm,
        halo_mm=halo_mm,max_spatial_cells=max_spatial_cells,seed=seed,connection_record=record,
        atlas_profile_id=configuration['atlas_profile'] if execution_profile=='atlas_isetbio_geometry_v1' else None,
        allow_empty_owner_outputs=execution_profile=='atlas_isetbio_geometry_v1',spatial_asset=spatial_asset)
    compiled,channels,audit=attach_primary_rod(root,compiled,channels,audit,record,light,spatial_asset=spatial_asset)
    if spatial_asset is not None:
        audit['atlas_runtime']['frozen_typed_geometry']=spatial_asset.validate_compiled(compiled)
        compiled.frozen_typed_geometry_binding=spatial_asset.binding
    nr=len(compiled.populations['rod'].cells.cell_id)
    channels['rod_light']=InputChannel(0,duration,lambda t:q(np.full(nr,light.rod_background_rstar_s),'R*/s','photoisomerization_rate'),
        'R*/s','photoisomerization_rate','zero_order_hold','project_approximation',{'source':'declared uniform rod background'})
    audit['atlas_runtime']['population_revision']=record['population_revision']
    audit['atlas_runtime']['graph_binding']={'kind':'one_frozen_approved_population_artifact',**binding}
    audit['execution_profile']=dict(audit['execution_profile'],profile_id=execution_profile,
        execution_mode='source_informed_effective_local_full_typed',scope='34 real components/20 layers on one frozen .35 graph; HC/global/calibration limitations remain')
    audit['resolved_execution_profile']=audit['execution_profile']
    if execution_profile=='atlas_isetbio_geometry_v1':
        from .geometry_identity import compiled_geometry_identity
        configuration=circuit_configuration(root,configuration_path)
        compiled.input_geometry_binding_required=True
        compiled.geometry_source_binding={'execution_profile':execution_profile,'frozen_graph':binding,
            'population_master_catalog':configuration['population_master_catalog'],
            'cone_master':record['isetbio_geometry']}
        if spatial_asset is not None:compiled.geometry_source_binding['typed_circuit_geometry']=spatial_asset.binding
        compiled.typed_view_domain={'reference_id':record['atlas']['reference_id'],
            'owner_domain_mm':tuple(record['atlas']['owner_domain']),
            'query_domain_mm':tuple(record['dependency_query']['requested_RF_query_mm']),
            'empty_owner_outputs':tuple(configuration['migration']['current_empty_owner_outputs'])}
        audit['atlas_runtime']['compiled_geometry_identity']=compiled_geometry_identity(compiled)
        audit['execution_profile']['scope']='New ISET/macaque finite local graph; naturally empty owner types explicitly reported, all source geometry retained'
        audit['execution_profile']['execution_mode']='frozen_finite_present_owner_paths'
        audit['execution_profile']['migration_status']=configuration['migration'].get('runtime_status','NOT_FINAL_MODEL_CONFIGURATION')
    audit['glycine_source_condition']={'source':'existing McLaughlin2021 quotation; peripheral slice transfer remains project approximation','primary_verification':'not implied by byte identity'}
    if hasattr(record,'io'):
        audit['artifact_storage']={'source_json_sha256':binding['sha256'],'logical_record_hash':record.semantic_hash,
            'storage_manifest_hash':record.storage_manifest_hash,
            'io':dict(record.io),'scope':'lazy population/route block reads; selected current circuit still fully compiled and resident'}
    return compiled,channels,audit
