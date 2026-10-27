"""Exact-context snapshot reuse with owned values and bounded retention.

Caching saves recomputation, not transfer/copy cost. Every read returns an owned
copy, so a caller cannot mutate a witness later used by a different consumer.
State/history versions must be incremented by their authoritative owners.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from .contracts import ContractError


@dataclass(frozen=True)
class SnapshotKey:
    time_hex: str
    side: str
    state_version: int
    input_epoch: int
    event_generation: int
    history_version: int
    sampler_identity: str
    operator_identity: str

    def __post_init__(self):
        try:
            time = float.fromhex(self.time_hex)
        except (TypeError, ValueError, OverflowError):
            raise ContractError('An exact finite hexadecimal snapshot time is required') from None
        if not math.isfinite(time) or time.hex() != self.time_hex or self.side not in ('left', 'right'):
            raise ContractError('An exact finite snapshot time and side are required')
        for name in ('state_version', 'input_epoch', 'event_generation', 'history_version'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ContractError('A nonnegative original snapshot counter is required: ' + name)
        for name in ('sampler_identity', 'operator_identity'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ContractError('An explicit snapshot context identity is required: ' + name)

    @classmethod
    def create(cls, t, side, *, state_version, input_epoch, event_generation,
               history_version, sampler_identity, operator_identity):
        return cls(float(t).hex(), side, state_version, input_epoch, event_generation,
                   history_version, sampler_identity, operator_identity)


def _owned_tree(value):
    if isinstance(value, np.ndarray):
        return deepcopy(value) if value.dtype.hasobject else value.copy()
    if hasattr(value, 'detach'):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _owned_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_owned_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_owned_tree(item) for item in value)
    return deepcopy(value)


class ExactSnapshotCache:
    """Small owned cache; a retained key is published once until invalidated or evicted."""

    def __init__(self, capacity=1):
        if type(capacity) is not int or capacity < 1:
            raise ContractError('A positive bounded snapshot-cache capacity is required')
        self.capacity = capacity
        self._entries = OrderedDict()

    def put(self, key, snapshot):
        if not isinstance(key, SnapshotKey):
            raise ContractError('A complete SnapshotKey is required')
        required = {'time_seconds', 'side', 'event_generation', 'rhs_kind',
                    'states', 'rhs', 'transitions', 'ports', 'aux'}
        if not isinstance(snapshot, dict) or not required <= snapshot.keys():
            raise ContractError('A complete original snapshot dictionary is required')
        if key in self._entries:
            raise ContractError('An exact snapshot key cannot be republished with another state')
        if (float(snapshot['time_seconds']).hex() != key.time_hex or snapshot['side'] != key.side
                or type(snapshot['event_generation']) is not int
                or snapshot['event_generation'] != key.event_generation
                or snapshot['rhs_kind'] != 'original_equation'):
            raise ContractError('Snapshot stamp or original-equation witness differs from cache key')
        for name in ('state_version', 'input_epoch', 'history_version', 'sampler_identity', 'operator_identity'):
            if name in snapshot and (type(snapshot[name]) is not type(getattr(key, name))
                                     or snapshot[name] != getattr(key, name)):
                raise ContractError('Snapshot owner context differs from cache key: ' + name)
        self._entries[key] = _owned_tree(snapshot)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def get(self, key):
        if not isinstance(key, SnapshotKey):
            raise ContractError('A complete SnapshotKey is required')
        value = self._entries.get(key)
        if value is None:
            return None
        self._entries.move_to_end(key)
        return _owned_tree(value)

    def clear(self):
        self._entries.clear()

    def __len__(self):
        return len(self._entries)
