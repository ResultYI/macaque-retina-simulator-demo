"""Candidate lossless writer for the existing uniform-background JSON protocol.

Not wired into live runners. It avoids full ndarray/list/JSON round trips,
retains the original nested hashes and is read by the unchanged strict loader.
The caller must hold the checkpoint state quiescent while writing, as at an
existing single-worker integration boundary. No dynamics or criteria change.
"""
from dataclasses import asdict,fields
from pathlib import Path
import hashlib,json,os
import numpy as np
from .canonical_stream import _array_data
from .contracts import ContractError,canonical_hash
from .engine import validate_state

_JSON={'sort_keys':True,'allow_nan':False,'separators':(',',':')}
_PRIMITIVES=(type(None),bool,int,float,str)


def _chunks(value):
    # Legacy checkpoint arrays are JSON lists, unlike the ndarray descriptors
    # used by contracts.canonical_hash for in-memory scientific objects.
    if isinstance(value,np.ndarray):yield from _array_data(value)
    elif isinstance(value,dict):
        if any(type(key) is not str for key in value):
            raise ContractError('Legacy checkpoint object keys must remain strings')
        yield '{'
        for index,key in enumerate(sorted(value)):
            if index:yield ','
            yield json.dumps(key,**_JSON);yield ':';yield from _chunks(value[key])
        yield '}'
    elif isinstance(value,(tuple,list)):
        yield '['
        for start in range(0,len(value),8192):
            block=value[start:start+8192]
            if start:yield ','
            if all(type(item) in _PRIMITIVES for item in block):
                yield json.dumps(block,**_JSON)[1:-1]
            else:
                for index,item in enumerate(block):
                    if index:yield ','
                    yield from _chunks(item)
        yield ']'
    else:yield json.dumps(value,**_JSON)


def _legacy_hash(value):
    digest=hashlib.sha256()
    for chunk in _chunks(value):digest.update(chunk.encode('utf-8'))
    return digest.hexdigest()


def _checkpoint(runner,state,*,hash_value=None):
    hash_value=_legacy_hash if hash_value is None else hash_value
    if state.assembly_hash!=runner.compiled.assembly_hash:
        raise ContractError('Checkpoint state/assembly identity mismatch')
    validate_state(state.dynamic,runner.model)
    data={field.name:getattr(state.dynamic,field.name) for field in fields(state.dynamic)}
    data['solver_config']=asdict(state.solver_config)
    dynamic={'checkpoint_version':'1.0','state':data,'payload_hash':hash_value(data)}
    payload={'assembly_hash':runner.compiled.assembly_hash,'dynamic':dynamic,
        'grid_origin_seconds':state.grid_origin_seconds,'sample_index':state.sample_index,
        'spike_states':state.spike_states,'delay_states':state.delay_states,
        'input_position':state.input_position,'link_delay_states':state.link_delay_states,
        'component_tick_indices':state.component_tick_indices}
    return {'version':'retina-architecture-1.0','payload':payload,'payload_hash':hash_value(payload)}


def save_background_streamed(store,path,state,report=None,*,origin_seconds=0.,safety=None):
    """Write the original protocol atomically; return a compact write receipt.

    This is an opt-in candidate, not a replacement monkeypatch for store.save.
    The on-disk document must be byte-identical to that method for equal input.
    """
    store._require_current_geometry()
    if any(state.input_position.get(key)!=value for key,value in store.expected_context.items()):
        raise ContractError('Only the actual declared uniform-background phase can be cached')
    checkpoint=_checkpoint(store.runner,state);payload=checkpoint['payload']
    auxiliary={key:payload[key] for key in ('grid_origin_seconds','sample_index','component_tick_indices',
        'spike_states','delay_states','link_delay_states','input_position')}
    value={'schema':'retina-uniform-background-checkpoint-1','binding':store.binding,'binding_hash':store.binding_hash,
        'checkpoint':checkpoint,'complete_checkpoint_hash':_legacy_hash(checkpoint),
        'auxiliary_time_rng_history_hash':_legacy_hash(auxiliary),'absolute_time_seconds':state.time_seconds,
        'preparation_origin_seconds':float(origin_seconds),'report':report,'safety':safety,
        'report_hash':canonical_hash(report)}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.writing')
    with temporary.open('w',encoding='utf-8') as stream:
        for chunk in _chunks(value):stream.write(chunk)
        stream.write('\n')
    os.replace(temporary,path)
    return {'schema':value['schema'],'binding_hash':value['binding_hash'],
        'complete_checkpoint_hash':value['complete_checkpoint_hash'],'bytes':path.stat().st_size,
        'codec':'streamed_legacy_background_json_candidate_v1','dynamics_or_criteria_changed':False}
