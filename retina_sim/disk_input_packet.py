"""Compact global input binding, checked by the existing strict channel reader.

Unlike the local v1 packet, this schema stores frozen global row indices in
numeric arrays instead of repeating millions of IDs inside the JSON header.
"""
from pathlib import Path
import json,zipfile
import numpy as np
from .contracts import ContractError
from .disk_graph_storage import file_sha
from .geometry_identity import compiled_geometry_identity
from .offline_input import OfflineInputs,_read_channel,_validate_geometry_binding,unique_json

SCHEMA='frozen_global_receptor_input_v1'

def write_global_input(path,compiled,light,*,reference_id,arrays,declarations,producer):
    path=Path(path)
    if path.exists():raise FileExistsError('Run-owned global input snapshot exists')
    metadata={'schema':SCHEMA,'version':'1.0','global_graph':compiled.global_disk_circuit.binding,
        'exact_view':compiled.global_spatial_asset.binding,'geometry_identity':compiled_geometry_identity(compiled),
        'background_context_hash':light.background_context_hash,'reference_id':reference_id,
        'channels':declarations,'producer':producer,'identity_policy':'frozen population binding+exact integer rows+actual compiled geometry; no nearest or index-only fallback'}
    encoded=json.dumps(metadata,allow_nan=False,separators=(',',':'))
    if len(encoded.encode('utf8'))>1024**2:raise ContractError('Compact global input metadata exceeds1MiB bound')
    with path.open('xb') as stream:np.savez_compressed(stream,manifest_json=np.array(encoded),**arrays)

def load_global_input(path,compiled,light,*,reference_id,max_bytes=256*1024**2):
    path=Path(path);before=path.stat()
    if not hasattr(compiled,'global_disk_circuit'):raise ContractError('Global input requires the actual frozen global circuit')
    with zipfile.ZipFile(path) as archive:
        if sum(member.file_size for member in archive.infolist())>max_bytes:raise ContractError('Global input decompressed arrays exceed declared256MiB budget')
    channels={};summaries={};circuit=compiled.global_disk_circuit;view=compiled.global_disk_view
    with np.load(path,allow_pickle=False) as data:
        raw=data['manifest_json']
        if raw.shape!=() or raw.dtype.kind!='U' or len(raw.item().encode('utf8'))>1024**2:raise ContractError('Global input metadata must be bounded scalar JSON')
        manifest=json.loads(raw.item(),object_pairs_hook=unique_json,
            parse_constant=lambda value:(_ for _ in ()).throw(ContractError('Nonfinite global JSON '+value)))
        if manifest.get('schema')!=SCHEMA or manifest.get('version')!='1.0':raise ContractError('Unknown global input schema/version')
        if manifest['global_graph']!=circuit.binding or manifest['exact_view']!=compiled.global_spatial_asset.binding:
            raise ContractError('Global input belongs to a different frozen graph or actual dependency view')
        if manifest['reference_id']!=reference_id or manifest['background_context_hash']!=light.background_context_hash:
            raise ContractError('Global input reference/background differs from initialization')
        geometry=_validate_geometry_binding(manifest.get('geometry_identity'),compiled)
        expected={external.channel:external for external in compiled.spec.external_inputs}
        if set(manifest['channels'])!=set(expected):raise ContractError('Supply exactly the global compiled input channels')
        for name,external in expected.items():
            declared=dict(manifest['channels'][name]);global_name=declared.pop('global_population')
            if global_name not in circuit.record['tables']:raise ContractError('Unknown global input population')
            descriptor=circuit.record['tables'][global_name]
            if declared.pop('population_binding')!=descriptor['binding']:raise ContractError('Global input population asset differs')
            key=declared.pop('global_rows_key');rows=np.asarray(data[key])
            if rows.dtype.kind not in 'iu' or rows.ndim!=1 or not np.array_equal(rows,view['selected_rows'][global_name]):
                raise ContractError('Global input row identity/order differs; no positional substitute')
            port=compiled.components[external.target_component].inputs[external.target_port]
            if len(port.cell_ids)!=len(rows):raise ContractError('Global input cell count differs')
            declared['cell_ids']=list(port.cell_ids)
            native=declared['input_stage']=='reference_equivalent_cone_drive'
            background_key=declared.pop('background_counts_key',None)
            if native:
                types=np.asarray(data[declared.pop('capture_types_key')])
                if types.dtype.kind!='S' or types.shape!=(len(rows),) or not set(types.tolist())<=set((b'L',b'M',b'S')):
                    raise ContractError('Global input LMS labels require a typed one-dimensional class array')
                actual=tuple(value.decode('ascii') for value in types)
                if actual!=tuple(compiled.populations['cones'].cells.subtype):raise ContractError('Global input LMS labels differ')
                declared['capture_types']=list(actual)
            # The existing reader still enforces IDs, units/roles, coordinates,
            # time windows, fixed reference, background and mechanism ownership.
            channel,summary=_read_channel(data,declared,external,compiled,light,reference_id)
            if native:
                from .offline_input import numeric
                from .isetbio_input import reference_equivalent_drive
                counts=numeric(data,background_key,'global native background')
                if counts.shape!=(1,len(rows)) or np.any(counts<=0):raise ContractError('Global native background captures differ')
                actual_bg,_=reference_equivalent_drive(counts/declared['time']['source_grid_step_seconds'],actual,
                    declared['reference_profile'],capture_scope_sha256=declared['capture_scope_sha256'])
                values=light.native_background_values
                expected_bg=np.full(len(rows),light.native_background_td) if values is None else np.asarray(values)
                if expected_bg.shape!=(len(rows),) or not np.allclose(actual_bg[0],expected_bg,rtol=2e-15,atol=0):
                    raise ContractError('Actual global per-cell background differs from initialization')
            channels[name]=channel;summaries[name]=dict(summary,global_population_binding=descriptor['binding'],
                global_rows_sha256=__import__('hashlib').sha256(np.ascontiguousarray(rows).tobytes()).hexdigest())
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ContractError('Global input changed while reading')
    digest=file_sha(path)
    for channel in channels.values():channel.input_contract.update(source_sha256=digest,producer=manifest['producer'])
    return OfflineInputs(channels,{'schema':SCHEMA,'version':'1.0','source_file':str(path.resolve()),'source_sha256':digest,
        'producer':manifest['producer'],'background_context_hash':light.background_context_hash,'channels':summaries,
        'geometry_identity':geometry,'global_graph':circuit.binding,'source_identity_scope':'Frozen bytes/units/IDs only; no primary physiological verification',
        'automatic_optics_gain_noise_or_phototransduction':False})
