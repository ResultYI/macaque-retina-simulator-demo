"""Execute the single frozen approved local population artifact, without fallback."""
from pathlib import Path
import json,hashlib
import numpy as np
from .atlas_runtime import atlas_midget_network,q
from .atlas_primary_rod import attach_primary_rod
from .contracts import ContractError
from .spatial import InputChannel
from .runtime_profiles import LightContext


def load_population_record(root, *, owner_bounds_mm=None, halo_mm=.04):
    """One identity-checked artifact for Run, GUI and CLI; never generate a substitute."""
    root=Path(root);config=json.loads((root/'configurations/retina_platform_v1.json').read_bytes())
    binding=config['atlas_target']['population_revision'];storage=config['atlas_target'].get('population_revision_storage')
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
        halo_mm=.04,max_spatial_cells=60000,seed=20261010):
    root=Path(root)
    record,binding=load_population_record(root,owner_bounds_mm=owner_bounds_mm,halo_mm=halo_mm)
    light=light_context or LightContext()
    compiled,channels,audit=atlas_midget_network(root,duration=duration,light_context=light,owner_bounds_mm=owner_bounds_mm,
        halo_mm=halo_mm,max_spatial_cells=max_spatial_cells,seed=seed,connection_record=record)
    compiled,channels,audit=attach_primary_rod(root,compiled,channels,audit,record,light)
    nr=len(compiled.populations['rod'].cells.cell_id)
    channels['rod_light']=InputChannel(0,duration,lambda t:q(np.full(nr,light.rod_background_rstar_s),'R*/s','photoisomerization_rate'),
        'R*/s','photoisomerization_rate','zero_order_hold','project_approximation',{'source':'declared uniform rod background'})
    audit['atlas_runtime']['population_revision']=record['population_revision']
    audit['atlas_runtime']['graph_binding']={'kind':'one_frozen_approved_population_artifact',**binding}
    audit['execution_profile']=dict(audit['execution_profile'],profile_id='atlas_population_revision_v1',
        execution_mode='source_informed_effective_local_full_typed',scope='34 real components/20 layers on one frozen .35 graph; HC/global/calibration limitations remain')
    audit['resolved_execution_profile']=audit['execution_profile']
    audit['glycine_source_condition']={'source':'existing McLaughlin2021 quotation; peripheral slice transfer remains project approximation','primary_verification':'not implied by byte identity'}
    if hasattr(record,'io'):
        audit['artifact_storage']={'source_json_sha256':binding['sha256'],'logical_record_hash':record.semantic_hash,
            'storage_manifest_hash':record.storage_manifest_hash,
            'io':dict(record.io),'scope':'lazy population/route block reads; selected current circuit still fully compiled and resident'}
    return compiled,channels,audit
