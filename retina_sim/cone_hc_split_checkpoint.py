"""Atomic, complete diagnostic split checkpoints with binary native arrays.

Scientific identity/content is independent of runtime permits. Loading still
requires a valid fresh permit and all original public state/solver/input guards.
This store never manufactures background qualification or fitting eligibility.
No model/device work occurs at import time.
"""
from pathlib import Path
import hashlib
import json
import os
import uuid

from .contracts import ContractError, canonical_hash


SCHEMA = 'cone-hc-split-complete-diagnostic-directory-2'


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def _flush_bytes(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def save_complete(runner, state, directory):
    """Publish an absent directory only after every complete file is durable.

    A failed write leaves a visibly named .writing directory, never a committed
    checkpoint. No existing checkpoint is overwritten or deleted. Native F/y/f
    use non-pickle NPY to avoid millions of Python float/list objects.
    """
    import numpy as np
    from .hybrid import HybridRunner
    from .cone_hc_split_runner import ConeHCSplitState

    if type(state) is not ConeHCSplitState:
        raise ContractError('Complete split store requires the explicit split state')
    target = Path(directory).resolve()
    if target.exists() or not target.parent.is_dir():
        raise ContractError('Checkpoint target must be absent with an existing parent')
    staging = target.with_name(target.name+'.writing-'+uuid.uuid4().hex)
    if staging.parent.resolve() != target.parent.resolve():
        raise ContractError('Atomic checkpoint staging escaped its target parent')
    staging.mkdir()
    files = {}

    def array(name, tensor):
        path = staging/name
        value = tensor.detach().cpu().numpy()
        if value.dtype != np.float64:
            raise ContractError('Complete split arrays must remain FP64')
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        files[name] = dict(sha256=file_sha(path), bytes=path.stat().st_size,
                           dtype=str(value.dtype), shape=list(value.shape), kind='npy')
        return {'array_file': name}

    with runner._lock, runner.torch.no_grad():
        state.validate(runner)
        cursor = state.native_cursor
        if cursor is not None and any(t != state.time_seconds for t in cursor.consumed.values()):
            raise ContractError('Complete checkpoint needs all synchronous consumers at public time')
        public = HybridRunner.checkpoint(runner, state.export(runner))
        path = staging/'public_state.json'
        _flush_bytes(path, public.encode('utf8'))
        files[path.name] = dict(sha256=file_sha(path), bytes=path.stat().st_size, kind='original_public_codec')
        del public
        native = None
        if cursor is not None:
            s = cursor.solver
            if s.failed:
                raise ContractError('Cannot commit a failed native controller')
            s._check_storage()
            native = dict(solver=dict(t=s.t, y=array('native_y.npy', s.y),
                f=None if s.f is None else array('native_f.npy', s.f), h_abs=s.h_abs,
                rtol=float(s.rtol), atol=float(s.atol), max_step=s.max_step,
                needs_input_restart=s.needs_input_restart, input_epoch=s.input_epoch,
                statistics=dict(s.statistics)), segments=[], consumed=dict(cursor.consumed),
                source_binding=cursor.source_binding, readout_sha256=cursor.readout.sha256)
            for i, segment in enumerate(cursor.segments):
                segment.check_owned()
                native['segments'].append(dict(start=segment.start, stop=segment.stop,
                    input_epoch=segment.input_epoch,
                    y0=array(f'segment_{i}_y0.npy', segment.y0),
                    y1=array(f'segment_{i}_y1.npy', segment.y1),
                    coefficients=array(f'segment_{i}_F.npy', segment.coefficients)))
        manifest = dict(schema=SCHEMA, model_hash=runner.model.model_hash,
            scientific_contract_sha256=runner.contract.sha256, input_binding_hash=runner.binding['binding_hash'],
            time_seconds=state.time_seconds, actual_complete_checkpoint=True, fit_eligible=False,
            qualification='ENGINEERING_DIAGNOSTIC_UNQUALIFIED', files=files, native_cursor=native)
        digest = canonical_hash(manifest)
        _flush_bytes(staging/'manifest.json', json.dumps(dict(manifest, manifest_hash=digest),
                     sort_keys=True, allow_nan=False).encode('utf8'))
        # Same-parent rename is the commit. Never accept a .writing directory.
        if target.exists():
            raise ContractError('Checkpoint target appeared during save; refusing replacement')
        os.replace(staging, target)
    return dict(schema=SCHEMA, directory=str(target), manifest_hash=digest,
                actual_complete_checkpoint=True, fit_eligible=False, time_seconds=state.time_seconds,
                bytes=sum(v['bytes'] for v in files.values()))


def load_complete(runner, directory, *, expected_manifest_hash, expected_input_context=None):
    """Verify external receipt, every file, stable identity, then restore owners."""
    import numpy as np
    runner._check_runtime()
    directory = Path(directory).resolve()
    if '.writing-' in directory.name:
        raise ContractError('Uncommitted checkpoint staging directory')
    raw = json.loads((directory/'manifest.json').read_bytes())
    digest = raw.pop('manifest_hash')
    if (raw.get('schema') != SCHEMA or digest != expected_manifest_hash or canonical_hash(raw) != digest
            or raw.get('model_hash') != runner.model.model_hash
            or raw.get('scientific_contract_sha256') != runner.contract.sha256
            or raw.get('input_binding_hash') != runner.binding['binding_hash']
            or raw.get('actual_complete_checkpoint') is not True or raw.get('fit_eligible') is not False):
        raise ContractError('Complete split checkpoint receipt/scientific/input identity differs')
    for name, item in raw['files'].items():
        path = directory/name
        if (Path(name).name != name or path.resolve().parent != directory or
                not path.is_file() or path.stat().st_size != item['bytes'] or file_sha(path) != item['sha256']):
            raise ContractError('Complete split checkpoint file content differs: '+name)
    loaded = {}

    def materialize(value):
        if isinstance(value, dict) and set(value) == {'array_file'}:
            name = value['array_file']
            if name not in loaded:
                item = raw['files'].get(name, {})
                if item.get('kind') != 'npy':
                    raise ContractError('Checkpoint native array reference is not a verified NPY')
                array = np.load(directory/name, allow_pickle=False, mmap_mode='r')
                if str(array.dtype) != item['dtype'] or list(array.shape) != item['shape']:
                    raise ContractError('Checkpoint native NPY layout differs')
                loaded[name] = array
            return loaded[name]
        if isinstance(value, dict):
            return {k: materialize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [materialize(v) for v in value]
        return value

    state = runner._restore_components((directory/'public_state.json').read_text(encoding='utf8'),
                    materialize(raw['native_cursor']), expected_input_context=expected_input_context)
    if state.time_seconds != raw['time_seconds']:
        raise ContractError('Complete split checkpoint public time differs')
    return state
