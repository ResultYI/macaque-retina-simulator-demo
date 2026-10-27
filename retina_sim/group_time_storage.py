"""Atomic typed bundles for new group seeds and complete candidate checkpoints."""
from pathlib import Path
import hashlib
import json
import os
import uuid

import numpy as np

from .contracts import ContractError, canonical_hash

SCHEMA = 'retipath-group-time-binary-bundle-1'
COMPLETE_FIELDS = {'native', 'inner', 'spike_states', 'clocks', 'input_position',
                   'observation_delay', 'controller', 'readout_cursor', 'monitor', 'recording'}


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_bundle(directory, *, kind, identity, payload, qualification):
    target = Path(directory).resolve()
    if target.exists() or not target.parent.is_dir() or kind not in ('source_construction_seed', 'complete_model_checkpoint', 'validation_evidence'):
        raise ContractError('Fresh group bundle directory and explicit new bundle kind required')
    if kind == 'complete_model_checkpoint' and set(payload) != COMPLETE_FIELDS:
        raise ContractError('Complete group checkpoint requires state, every clock/history/controller/consumer/monitor/recording field')
    if qualification.get('default_adopted') is not False or qualification.get('fit_eligible') is not False:
        raise ContractError('Candidate bundle cannot confer production or fitting qualification')
    staging = target.with_name(target.name+'.writing-'+uuid.uuid4().hex)
    staging.mkdir()
    files = {}
    def encode(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject or value.dtype.kind not in 'fiubUS' or value.dtype.kind in 'fi' and not np.isfinite(value).all():
                raise ContractError('Object/nonfinite checkpoint arrays forbidden')
            name = f'array_{len(files):05d}.npy'
            with (staging/name).open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
                stream.flush(); os.fsync(stream.fileno())
            files[name] = dict(sha256=_sha(staging/name), bytes=(staging/name).stat().st_size,
                               dtype=value.dtype.str, shape=list(value.shape))
            return {'array': name}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {'dictionary': [[encode(k), encode(v)] for k,v in value.items()]}
        if isinstance(value, tuple):
            return {'tuple': [encode(v) for v in value]}
        if isinstance(value, list):
            # Keep original sampler's list type without giant scalar JSON trees.
            try:
                a = np.asarray(value)
            except (TypeError, ValueError):
                a = None
            if a is not None and a.size > 1024 and a.dtype.kind in 'fiub':
                return {'numeric_list': encode(a)}
            return [encode(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise ContractError('Unsupported group bundle value type: '+type(value).__name__)
    encoded = encode(payload)
    manifest = dict(schema=SCHEMA, kind=kind, identity=identity, qualification=qualification,
                    files=files, payload=encoded)
    checksum = canonical_hash(manifest)
    with (staging/'manifest.json').open('x', encoding='utf8') as stream:
        json.dump(dict(manifest, manifest_hash=checksum), stream, sort_keys=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    if target.exists():
        raise ContractError('Group bundle destination appeared; refusing replacement')
    os.replace(staging, target)
    return dict(directory=str(target), kind=kind, manifest_hash=checksum,
                bytes=sum(v['bytes'] for v in files.values()))


def load_bundle(directory, *, expected_kind, expected_identity=None):
    root = Path(directory).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf8'))
    expected = manifest.pop('manifest_hash', None)
    if (manifest.get('schema') != SCHEMA or manifest.get('kind') != expected_kind
            or canonical_hash(manifest) != expected
            or expected_identity is not None and canonical_hash(manifest['identity']) != canonical_hash(expected_identity)):
        raise ContractError('Group bundle schema/kind/identity differs; old fine checkpoints are incompatible')
    if (manifest.get('qualification',{}).get('default_adopted') is not False
            or manifest.get('qualification',{}).get('fit_eligible') is not False):
        raise ContractError('Restored candidate cannot confer production/fitting qualification')
    loaded = {}
    for name, row in manifest['files'].items():
        p = (root/name).resolve()
        if (Path(name).name != name or not p.is_relative_to(root) or not p.is_file()
                or p.stat().st_size != row['bytes'] or _sha(p) != row['sha256']):
            raise ContractError('Group bundle array identity differs: '+name)
        a = np.load(p, allow_pickle=False)
        if (a.dtype.str != row['dtype'] or list(a.shape) != row['shape'] or a.dtype.hasobject
                or a.dtype.kind not in 'fiubUS' or a.dtype.kind in 'fi' and not np.isfinite(a).all()):
            raise ContractError('Group bundle array layout differs')
        loaded[name] = a
    claimed = set()
    def decode(value):
        if isinstance(value, list):
            return [decode(v) for v in value]
        if isinstance(value, dict):
            if set(value) == {'array'}:
                name = value['array']
                a = loaded[name]
                # The loader owns these arrays. Transfer a first C-contiguous
                # reference; repeated references still get independent storage.
                if name in claimed or not a.flags.c_contiguous:
                    return a.copy()
                claimed.add(name)
                return a
            if set(value) == {'dictionary'}:
                return {decode(k): decode(v) for k,v in value['dictionary']}
            if set(value) == {'tuple'}:
                return tuple(decode(v) for v in value['tuple'])
            if set(value) == {'numeric_list'}:
                return decode(value['numeric_list']).tolist()
            raise ContractError('Unknown typed group checkpoint node')
        return value
    payload = decode(manifest['payload'])
    if expected_kind == 'complete_model_checkpoint' and set(payload) != COMPLETE_FIELDS:
        raise ContractError('Incomplete group checkpoint payload')
    return dict(identity=manifest['identity'], qualification=manifest['qualification'], payload=payload,
                manifest_hash=expected)
