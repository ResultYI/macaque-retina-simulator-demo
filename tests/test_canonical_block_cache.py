"""Exact old JSON digests, fresh bytes, bounded storage and alias ownership."""
from dataclasses import dataclass
from enum import Enum
import json

import numpy as np
import pytest

from retina_sim.canonical_block_cache import CanonicalBlockCache
from retina_sim.canonical_stream import stream_canonical_hash
from tools import validate_group_time_candidate as validation


class Label(str, Enum):
    VALUE = 'value'


@dataclass
class Record:
    values: object


@pytest.mark.parametrize('dtype', [np.float64, np.float32, np.int64, np.uint8, np.bool_, '>f8'])
def test_same_canonical_bytes_with_layout_dtype_metadata_and_duplicates(dtype):
    array = np.arange(30000).astype(dtype).reshape(100, 300)[:, ::3]
    value = {('key', 2): [array, array.copy(), Record(array)],
             7: 'discarded collision', '7': [Label.VALUE, None, -0.0, 1.1e-300]}
    cache = CanonicalBlockCache(4*1024**2)
    assert stream_canonical_hash(value, block_cache=cache) == stream_canonical_hash(value)
    assert cache.hits > 0
    assert cache.payload_bytes <= cache.max_payload_bytes


def test_mutation_everywhere_and_event_metadata_cannot_hit_stale_contents():
    array = np.linspace(-1., 1., 18000)
    value = dict(time=4.997, side='right', generation=12, values=[array, array])
    cache = CanonicalBlockCache()
    previous = stream_canonical_hash(value, block_cache=cache)
    for index in (0, 8191, 8192, len(array)-1):
        array[index] = np.nextafter(array[index], np.inf)
        actual = stream_canonical_hash(value, block_cache=cache)
        assert actual == stream_canonical_hash(value)
        assert actual != previous
        previous = actual
    for key, item in (('time', 4.998), ('side', 'left'), ('generation', 13)):
        value[key] = item
        actual = stream_canonical_hash(value, block_cache=cache)
        assert actual == stream_canonical_hash(value) != previous
        previous = actual


def test_capacity_eviction_and_nonfinite_errors_are_preserved():
    cache = CanonicalBlockCache(512)
    for start in range(10):
        value = np.arange(start, start+32, dtype=np.float64)
        assert stream_canonical_hash(value, block_cache=cache) == stream_canonical_hash(value)
        assert cache.payload_bytes <= 512
    assert cache.evictions > 0
    for value in (np.array([np.nan]), np.array([np.inf])):
        with pytest.raises(ValueError):
            stream_canonical_hash(value, block_cache=cache)


def test_cache_keys_compare_complete_bytes_even_when_python_hashes_collide():
    class Collision(bytes):
        def __hash__(self):
            return 7
    cache = CanonicalBlockCache()
    a = Collision(np.array([1., -0.]).tobytes())
    b = Collision(np.array([2., 0.]).tobytes())
    assert hash(a) == hash(b) and a != b
    assert cache.encode(np.float64, a) == '1.0,-0.0'
    assert cache.encode(np.float64, b) == '2.0,0.0'
    assert cache.encode(np.float64, a) == '1.0,-0.0'
    assert cache.hits == 1


def test_each_tensor_occurrence_is_read_again_and_next_call_is_fresh(monkeypatch):
    torch = pytest.importorskip('torch')
    tensor = torch.tensor([1., 2.], dtype=torch.float64)
    value = dict(states=tensor, ports=tensor, outputs=[tensor])
    original = validation.cpu
    calls = []
    def tracked(v):
        calls.append(v)
        return original(v)
    monkeypatch.setattr(validation, 'cpu', tracked)
    details = {}
    first = validation.checkpoint_hash(value, diagnostics=details)
    assert len(calls) == 3 and details['tensor_current_reads'] == 3
    assert details['tensor_conversion_reuse'] is False
    assert first == validation.checkpoint_hash(value, reuse_arrays=False)
    tensor.add_(1.)
    assert validation.checkpoint_hash(value) == validation.checkpoint_hash(value, reuse_arrays=False) != first


def test_numpy_shared_storage_write_without_tensor_version_change_is_detected(monkeypatch):
    torch = pytest.importorskip('torch')
    array = np.array([1., 2.])
    tensor = torch.from_numpy(array)
    version = tensor._version
    original = validation.cpu
    count = 0
    def changed(v):
        nonlocal count
        output = original(v)
        count += 1
        if count == 1:
            array[0] = 9.
            assert tensor._version == version
        return output
    monkeypatch.setattr(validation, 'cpu', changed)
    with pytest.raises(ValueError, match='storage changed during capture'):
        validation.checkpoint_hash([tensor, tensor])
    assert count == 2


def test_tensor_mutation_during_frozen_capture_is_not_silently_reused(monkeypatch):
    torch = pytest.importorskip('torch')
    tensor = torch.ones(2, dtype=torch.float64)
    original = validation.cpu
    def changed(v):
        output = original(v)
        v.add_(1.)
        return output
    monkeypatch.setattr(validation, 'cpu', changed)
    with pytest.raises(ValueError, match='changed during capture'):
        validation.checkpoint_hash([tensor, tensor])


def test_repeated_values_keep_exact_float_tokens_and_nonfinite_rejection():
    values = np.resize(np.array([-0., 0., 1.e-300, -1.e-300, 1.e300, np.nextafter(1., 2.)]), 18000)
    cache = CanonicalBlockCache()
    assert stream_canonical_hash(values, block_cache=cache) == stream_canonical_hash(values)
    assert cache.deduplicated_blocks > 0
    values[1234] = np.nan
    with pytest.raises(ValueError):
        stream_canonical_hash(values, block_cache=cache)


def test_inference_tensor_without_version_keeps_old_digest():
    torch = pytest.importorskip('torch')
    with torch.inference_mode():
        tensor = torch.tensor([1., 2.], dtype=torch.float64)
    assert validation.checkpoint_hash([tensor, tensor]) == validation.checkpoint_hash([tensor, tensor], reuse_arrays=False)
