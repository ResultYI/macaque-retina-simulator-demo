"""Lossless strict checkpoint decode without repeated JSON round trips.

No file, binding, hash, geometry, state or convergence check is omitted. The
original public restore remains the fallback for custom/unreviewed runners.
"""
from pathlib import Path
import copy,hashlib,inspect,json,textwrap
from .contracts import ContractError,canonical_hash
from .engine import restore_payload,validate_state,_state_schema_hash
from .hybrid import HybridRunner,RetinaState

RESTORE_SHA256='931e806a91848c42cd2f36b56f736a1281137003c0aa78508a0475d1339c800f'
_AUX=('grid_origin_seconds','sample_index','spike_states','delay_states','input_position',
      'link_delay_states','component_tick_indices')


def _eligibility(runner):
    method=getattr(runner.restore,'__func__',None)
    if method is not HybridRunner.restore:return 'Custom restore hook retains original public API'
    try:digest=hashlib.sha256(textwrap.dedent(inspect.getsource(method)).encode()).hexdigest()
    except (OSError,TypeError):return 'Restore source is not inspectable'
    if digest!=RESTORE_SHA256:return 'Unreviewed restore source retains original public API'
    return None


def restore_retina_payload(runner,checkpoint,*,expected_input_context=None,mode='auto'):
    if mode not in ('auto','legacy','parsed'):raise ContractError('Unknown checkpoint decoding mode')
    reason='Explicit original JSON round-trip fallback' if mode=='legacy' else _eligibility(runner)
    if reason is None and checkpoint['payload']['dynamic']['state']['solver_config']['method']=='RK2':
        reason='Original RK2 branch retains its public decoder'
    if reason is not None:
        if mode=='parsed':raise ContractError(reason)
        state=runner.restore(json.dumps(checkpoint),expected_input_context=expected_input_context)
        return state,{'procedure':'legacy_JSON_roundtrip','reason':reason}
    d=checkpoint['payload']
    if (checkpoint['version']!='retina-architecture-1.0' or checkpoint['payload_hash']!=canonical_hash(d)
            or d['assembly_hash']!=runner.compiled.assembly_hash):
        raise ContractError('Checkpoint payload/assembly identity mismatch')
    if expected_input_context is not None and any(d['input_position'].get(k)!=v for k,v in expected_input_context.items()):
        raise ContractError('Checkpoint input/background context mismatch')
    expected={'state_spec_hash':_state_schema_hash(runner.model),'graph_hash':runner.model.graph.graph_hash,
        'parameter_hash':runner.model.parameters.parameter_hash,'model_hash':runner.model.model_hash,
        'model_id':runner.model.model_id,'model_version':runner.model.version}
    dynamic=restore_payload(d['dynamic'],expected)
    validate_state(dynamic,runner.model)
    # The old extra JSON parse isolated every mutable auxiliary container from
    # the returned envelope. Keep that isolation; immutable floats/strings may
    # safely be shared, but lists/dicts/history/RNG remain private.
    private=copy.deepcopy({key:d[key] for key in _AUX if key in d})
    state=RetinaState(dynamic,d['assembly_hash'],private['grid_origin_seconds'],private['sample_index'],
        private['spike_states'],private['delay_states'],private['input_position'],private['link_delay_states'],
        private.get('component_tick_indices',{}))
    return state,{'procedure':'parsed_strict_checkpoint_v1','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_restore_sha256':RESTORE_SHA256,'nested_hash_geometry_identity_state_and_convergence_checks':'unchanged',
        'mutable_envelope_state_aliasing':False,'checkpoint_or_numerical_identity_migration':False}
