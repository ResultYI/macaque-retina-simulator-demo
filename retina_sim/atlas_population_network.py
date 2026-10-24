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
            owner_options=({'owner_global_rows':configuration['owner_global_rows']}
                           if 'owner_global_rows' in configuration else {})
            view=circuit.owner_view(owner_bounds_mm,max_nodes=configuration['budgets']['max_dependency_nodes'],
                empty_chemical_input_policy=configuration.get('empty_chemical_input_policy','reject_empty_chemical_input_v1'),
                **owner_options)
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
        halo_mm=.04,max_spatial_cells=60000,seed=20261010,execution_profile='atlas_population_revision_v1',configuration_path=None,_freeze_geometry_bootstrap=False,
        dynamics_candidate_path=None,apply_rgc_output_defaults=True,apply_rod_reference_defaults=None,
        bc_output_gain_control_enabled=None,ac_inhibition_enabled=None):
    root=Path(root)
    from .spatial_scope import enforce_owner_scope
    scope_receipt=enforce_owner_scope(root,owner_bounds_mm)
    apply_rod_reference_defaults=apply_rgc_output_defaults if apply_rod_reference_defaults is None else apply_rod_reference_defaults
    rod_reference=None
    if apply_rod_reference_defaults:
        policy=json.loads((root/'configurations/retina_platform_v1.json').read_bytes()).get('rod_reference')
        if policy is not None:
            policy_path=(root/policy['path']).resolve()
            if not policy_path.is_relative_to(root.resolve()):raise ContractError('Rod reference policy must stay inside project')
            raw=policy_path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=policy['sha256']:raise ContractError('Rod fixed-reference policy identity differs')
            rod_reference=dict(json.loads(raw),binding=policy)
    if dynamics_candidate_path is not None:
        candidate_path=Path(dynamics_candidate_path)
        dynamics_candidate_path=candidate_path if candidate_path.is_absolute() else root/candidate_path
    if execution_profile=='atlas_isetbio_geometry_v1':
        configuration=circuit_configuration(root,configuration_path)
        if configuration.get('storage_kind')=='global_disk_graph_v1':
            from .disk_runtime import disk_network
            result=disk_network(root,configuration,duration=duration,light_context=light_context or LightContext(),
                owner_bounds_mm=owner_bounds_mm,halo_mm=halo_mm,max_spatial_cells=max_spatial_cells,seed=seed,
                rod_reference_policy=rod_reference)
            from .rgc_defaults import apply_rgc_defaults
            result=apply_rgc_defaults(_candidate_result(result,dynamics_candidate_path,light_context or LightContext()),root,apply_rgc_output_defaults)
            result[2]['spatial_scope']=scope_receipt
            return _retained_initializer(result,root,apply_rgc_output_defaults,dynamics_candidate_path,bc_output_gain_control_enabled,ac_inhibition_enabled)
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
    compiled,channels,audit=attach_primary_rod(root,compiled,channels,audit,record,light,spatial_asset=spatial_asset,rod_reference_policy=rod_reference)
    if spatial_asset is not None:
        geometry_view=compiled
        if rod_reference is not None:
            # The frozen spatial record includes dynamics-profile labels. Only
            # this approved label is projected back; every spatial byte stays
            # subject to the original full snapshot comparison.
            from copy import copy
            from dataclasses import replace
            membrane=compiled.populations['rod_membrane']
            if set(membrane.cells.parameter_profile_id)!={'effective_rod_membrane_fixed_dark_reference_v1'}:
                raise ContractError('Unexpected rod profile alias in frozen geometry')
            geometry_view=copy(compiled);geometry_view.populations=dict(compiled.populations)
            geometry_view.populations['rod_membrane']=replace(membrane,cells=replace(membrane.cells,
                parameter_profile_id=tuple('effective_rod_membrane' for _ in membrane.cells.cell_id)))
            projected=[]
            for link in compiled.links:
                table=link.specification.connections
                if 'effective_rod_membrane_fixed_dark_reference_v1' in table.source_profile_id:
                    if (link.specification.target_component not in ('rod_membrane','rod_current_SI') or
                            set(table.source_profile_id)!={'effective_rod_membrane_fixed_dark_reference_v1'}):
                        raise ContractError('Unexpected link parameter-label alias in frozen geometry')
                    link=replace(link,specification=replace(link.specification,connections=replace(table,
                        source_profile_id=tuple('effective_rod_membrane' for _ in table.source_profile_id))))
                projected.append(link)
            geometry_view.links=tuple(projected)
        audit['atlas_runtime']['frozen_typed_geometry']=spatial_asset.validate_compiled(geometry_view)
        if rod_reference is not None:audit['atlas_runtime']['frozen_typed_geometry']['authorized_parameter_label_alias']='rod fixed-dark-reference profile only'
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
    from .rgc_defaults import apply_rgc_defaults
    result=apply_rgc_defaults(_candidate_result((compiled,channels,audit),dynamics_candidate_path,light),root,apply_rgc_output_defaults)
    result[2]['spatial_scope']=scope_receipt
    return _retained_initializer(result,root,apply_rgc_output_defaults,dynamics_candidate_path,bc_output_gain_control_enabled,ac_inhibition_enabled)


def _retained_initializer(result,root,apply_rgc_output_defaults,dynamics_candidate_path,bc_output_gain_control_enabled=None,ac_inhibition_enabled=None):
    """The same state-only initializer and acceptance criteria on either geometry."""
    root=Path(root)
    from .bc_gain_control import apply_bc_gain_control
    result=apply_bc_gain_control(result,root,bc_output_gain_control_enabled)
    from .ac_inhibition import apply_ac_inhibition
    result=apply_ac_inhibition(result,root,ac_inhibition_enabled)
    if apply_rgc_output_defaults and dynamics_candidate_path is None:
        policy=json.loads((root/'configurations/retina_platform_v1.json').read_bytes()).get('deterministic_background_initialization')
        if policy is not None:
            path=(root/policy['path']).resolve()
            if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=policy['sha256']:
                raise ContractError('Deterministic background initialization policy identity differs')
            definition=json.loads(path.read_bytes())
            if (definition.get('schema_version')!='retained-background-initialization-1.0' or
                    definition.get('parameters_or_references_changed') is not False or
                    definition.get('existing_preparation_criteria_required') is not True):
                raise ContractError('State-only retained background initialization contract required')
            result[0].retained_background_initializer=dict(binding=policy,definition=definition)
    return result


def _candidate_result(result,path,light=None):
    if path is None:return result
    from .bc_ac_candidate import migrate_bc_ac,CandidateProfile
    compiled,channels,audit=result
    candidate=CandidateProfile(path);background=None
    if candidate.data.get('hypothesis_baseline'):
        from .bc_ac_hypothesis import constant_background_channels
        reference=constant_background_channels(compiled,light or LightContext(),stop=.0001)
        background={k:v.at(0.) for k,v in reference.items()}
    compiled,registry,report=migrate_bc_ac(compiled,audit['registry'],candidate,background_inputs=background)
    from .assembly import RetinaBuilder
    audit=dict(audit,registry=registry,spec=compiled.spec,candidate_dynamics=report,
        resource_estimate=RetinaBuilder(registry).estimate(compiled.spec),
        population_records={key:pop.record() for key,pop in compiled.populations.items()},
        runtime_profile=dict(audit['runtime_profile'],candidate_dynamics=report,
            claim='Retained base profile plus explicit BC/AC receptor candidate; no physiological calibration'))
    audit['execution_profile']=dict(audit['execution_profile'],equation_route=report['equation_route'],
        default_route_changed=False,physiological_calibration=False)
    audit['resolved_execution_profile']=audit['execution_profile']
    return compiled,channels,audit
