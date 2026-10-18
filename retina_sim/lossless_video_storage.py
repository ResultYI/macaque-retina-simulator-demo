"""Opt-in transport changes; original checkpoint schema, validators and values.

Canonical subtrees are encoded once per quiescent checkpoint and spooled to
private files. No state or hash is cached across checkpoints. Recording uses
the original NPZ members without DEFLATE, trading disk space for CPU time.
"""
from collections.abc import Iterator
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import textwrap
import time

import numpy as np
from .checkpoint_streaming import _checkpoint, _chunks
from .contracts import ContractError
from .recording import NPZWindowRecorder

_BUFFER_BYTES = 1024**2
_RECORDER_SOURCE = 'c606d7d2d23b3bf2dcc5d72ab362b0977742ea2ecb46dac6466b0956036ea08e'


class _CanonicalSpools:
    def __init__(self, directory: Path):
        self.directory = directory
        self.entries: list[tuple[object, Path]] = []

    def chunks(self, value: object) -> Iterator[bytes]:
        for source, path in self.entries:
            if source is value:
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(_BUFFER_BYTES), b''):
                        yield block
                return
        if isinstance(value, dict):
            if any(type(key) is not str for key in value):
                raise ContractError('Legacy checkpoint object keys must remain strings')
            yield b'{'
            for index, key in enumerate(sorted(value)):
                if index:
                    yield b','
                for part in _chunks(key):
                    yield part.encode('utf8')
                yield b':'
                yield from self.chunks(value[key])
            yield b'}'
        else:
            for part in _chunks(value):
                yield part.encode('utf8')

    def hash_value(self, value: object) -> str:
        if len(self.entries) >= 2:
            raise ContractError('Review spooled codec after checkpoint hash-schema changes')
        path = self.directory / ('subtree_%d.json' % len(self.entries))
        digest = hashlib.sha256()
        with path.open('xb') as stream:
            for block in self.chunks(value):
                stream.write(block)
                digest.update(block)
        self.entries.append((value, path))
        return digest.hexdigest()


def checkpoint_file_spooled(runner, state, path):
    """Same original builder and checks; each canonical subtree encoded once."""
    tick = time.perf_counter()
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    with tempfile.TemporaryDirectory(prefix='canonical_checkpoint_', dir=path.parent) as directory:
        cache = _CanonicalSpools(Path(directory))
        checkpoint = _checkpoint(runner, state, hash_value=cache.hash_value)
        with temporary.open('wb') as stream:
            for block in cache.chunks(checkpoint):
                stream.write(block)
            stream.write(os.linesep.encode('ascii'))
        os.replace(temporary, path)
        receipt = {'path': str(path), 'bytes': path.stat().st_size,
                   'canonical_subtrees': len(cache.entries), 'buffer_bytes': _BUFFER_BYTES,
                   'codec': 'once_encoded_original_canonical_checkpoint_v1'}
    return dict(receipt, seconds=time.perf_counter() - tick)


class StoredNPZWindowRecorder(NPZWindowRecorder):
    def __init__(self, directory):
        source = textwrap.dedent(inspect.getsource(NPZWindowRecorder.consume))
        if hashlib.sha256(source.encode()).hexdigest() != _RECORDER_SOURCE:
            raise ContractError('Review stored NPZ transport after original recording protocol changes')
        super().__init__(directory)

    def consume(self, result):
        filename = self.directory / f'window_{self.windows:08d}.npz'
        arrays = {name: q.values for name, q in result.outputs_with_units.items()}
        manifest = {name: {'unit': q.unit, 'meaning': q.meaning, 'evidence_tag': q.evidence_tag}
                    for name, q in result.outputs_with_units.items()}
        arrays['manifest_json'] = np.array(json.dumps({'ports': manifest, 'metadata': result.metadata}, sort_keys=True))
        np.savez(filename, **arrays)
        self.windows += 1
        return filename
