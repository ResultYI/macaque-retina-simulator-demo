"""CPU continuous readouts and two-slot ownership contract, without CUDA code.

Staged arrays may be read only after an injected readiness callback completes.
This does not implement or validate a GPU transfer stream/event adapter.
"""
from dataclasses import dataclass
import math
from numbers import Real
from threading import RLock
from types import MappingProxyType

import numpy as np
from scipy import sparse

from .contracts import ContractError, canonical_hash
from .continuous_state import AcceptedRepresentation, from_packet, _immutable, PACKET_SCHEMA as CONTINUOUS_PACKET_SCHEMA
from .vh2005 import SOURCE_VALUES

CONSUMERS = ('inner', 'monitor', 'observation', 'recording')
PACKET_SCHEMA = 'retipath-continuous-release-packet-1'
BRIDGE_CHECKPOINT_SCHEMA = 'retipath-continuous-bridge-ownership-1'


def _finite_time(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 0


class ContinuousBridgeSelection:
    """Original necessary B rows and their exact source HC columns, no renorm."""
    def __init__(self, B, cone_rows, *, parameters=None):
        if (not sparse.isspmatrix_csr(B) or B.dtype != np.float64 or min(B.shape) < 1
                or not np.isfinite(B.data).all() or np.any(B.data < 0)):
            raise ContractError('Finite nonnegative original FP64 CSR B required')
        rows = np.asarray(cone_rows)
        if (rows.ndim != 1 or not len(rows) or rows.dtype.kind not in 'iu'
                or np.any(rows < 0) or np.any(rows >= B.shape[0]) or len(np.unique(rows)) != len(rows)):
            raise ContractError('Distinct necessary original cone group rows required')
        self.cone_rows = _immutable(rows.astype(np.int64))
        chosen = B[self.cone_rows].tocsr(copy=True)
        self.hc_rows = _immutable(np.unique(chosen.indices).astype(np.int64))
        self.B = sparse.csr_matrix((chosen.data.copy(), np.searchsorted(self.hc_rows, chosen.indices),
                                    chosen.indptr.copy()), shape=(len(rows), len(self.hc_rows)))
        for field in ('data', 'indices', 'indptr'):
            setattr(self.B, field, _immutable(getattr(self.B, field)))
        params = dict(SOURCE_VALUES if parameters is None else parameters)
        if params != SOURCE_VALUES:
            raise ContractError('Continuous release readout retains native source parameters')
        self.parameters = MappingProxyType(params)
        self.sha256 = canonical_hash(dict(schema=PACKET_SCHEMA, full_shape=B.shape,
            cone_rows=self.cone_rows, hc_rows=self.hc_rows, data=self.B.data,
            indices=self.B.indices, indptr=self.B.indptr, parameters=params,
            order='continuous Vis/Vbar/h, original B@h, native sigmoid and adaptation'))


def _preflight(selection, representations):
    if set(representations) != {'Vis', 'Vbar', 'Vh'}:
        raise ContractError('Vis, Vbar and current HC representations are required')
    raw, interval = {}, None
    for name, representation in representations.items():
        if name == 'Vh' and not len(selection.hc_rows):
            if representation is not None:
                raise ContractError('Zero-column B readout transfers no unnecessary HC states')
            raw[name] = None
            continue
        packet = representation.packet() if isinstance(representation, AcceptedRepresentation) else representation
        if not isinstance(packet, dict) or set(packet) != {'metadata', 'arrays'}:
            raise ContractError('Typed continuous representation packet required')
        meta, arrays = packet['metadata'], packet['arrays']
        if (not isinstance(meta, dict) or not isinstance(arrays, dict) or meta.get('accepted') is not True
                or meta.get('schema') != CONTINUOUS_PACKET_SCHEMA
                or meta.get('dtype') != 'float64' or meta.get('kind') not in ('dop853_dense', 'exponential_forced', 'discrete_hold', 'quintic_hermite')
                or not _finite_time(meta.get('t0')) or not _finite_time(meta.get('t1'))
                or meta['t1'] <= meta['t0'] or not _integer(meta.get('epoch'))):
            raise ContractError('Accepted FP64 continuous packet interval/epoch/type required')
        required_meta = {'schema', 'kind', 't0', 't1', 't0_hex', 't1_hex', 'epoch', 'dtype',
                         'side', 'accepted', 'original_rhs_id', 'representation_id'}
        if (set(meta) != required_meta or meta['t0_hex'] != float(meta['t0']).hex()
                or meta['t1_hex'] != float(meta['t1']).hex()
                or meta['side'] != dict(t0='right', t1='left', interior='either')):
            raise ContractError('Continuous staged metadata boundary contract differs')
        expected = selection.hc_rows if name == 'Vh' else selection.cone_rows
        ids = np.asarray(arrays.get('state_ids'))
        if ids.dtype.kind not in 'iu' or not np.array_equal(ids, expected):
            raise ContractError('Transfer only exact necessary selected group rows: '+name)
        if any(np.asarray(v).dtype != np.float64 for k, v in arrays.items() if k != 'state_ids'):
            raise ContractError('Staged continuous values and coefficients must be FP64')
        expected_arrays = {'state_ids', 'y0'}
        if meta['kind']=='quintic_hermite':
            expected_arrays.update(('f0','ym','fm','y1','f1'))
        elif meta['kind'] != 'discrete_hold':
            expected_arrays.add('coefficients')
        if meta['kind'] == 'exponential_forced':
            expected_arrays.add('decay')
        shapes = dict(state_ids=(len(expected),), y0=(len(expected),), decay=(len(expected),),
                      coefficients=(7 if meta['kind'] == 'dop853_dense' else 8, len(expected)),
                      **{n:(len(expected),) for n in ('f0','ym','fm','y1','f1')})
        if set(arrays) != expected_arrays or any(np.asarray(value).shape != shapes[k] for k, value in arrays.items()):
            raise ContractError('Continuous staged array shapes/layout differ')
        current = (meta['t0'], meta['t1'], meta['epoch'])
        if interval is not None and current != interval:
            raise ContractError('Native readout representation intervals or epochs differ')
        interval = current
        # Staged value arrays remain unread until readiness. Copy metadata and
        # the array mapping now so a producer cannot replace identities later.
        raw[name] = dict(metadata=dict(meta), arrays=dict(arrays))
    return raw, interval


class ContinuousReleasePacket:
    def __init__(self, selection, representations, *, source_version, segment_id=0,
                 ready_waiter=None, lease_check=None):
        if (not isinstance(source_version, str) or not source_version or not _integer(segment_id)
                or ready_waiter is not None and not callable(ready_waiter)):
            raise ContractError('Source version, segment identity and valid readiness callback required')
        self.selection = selection
        self._raw, (self.start, self.stop, self.input_epoch) = _preflight(selection, representations)
        self.source_version, self.segment_id = source_version, int(segment_id)
        self._waiter, self._lease_check = ready_waiter, lease_check
        self._ready, self._released, self._representations = False, False, None
        self._lock = RLock()

    def _check(self):
        if self._released or self._lease_check is not None and not self._lease_check():
            raise ContractError('Continuous bridge lease has been released or reused')

    def wait(self):
        with self._lock:
            self._check()
            if not self._ready:
                if self._waiter is not None:
                    self._waiter()
                materialized = {name: None if packet is None else from_packet(packet) for name, packet in self._raw.items()}
                # Recheck the readout selection after staged IDs become owned.
                _preflight(self.selection, materialized)
                self._representations = materialized
                self._raw = None
                self._ready = True
            return self

    @property
    def available_until(self):
        self._check()
        return self.stop

    def state_at(self, t, side):
        self.wait()
        return tuple(np.empty(0, dtype=np.float64) if self._representations[name] is None
                     else self._representations[name].evaluate(t, side) for name in ('Vis', 'Vbar', 'Vh'))

    def signals_at(self, t, side):
        v, vbar, h = self.state_at(t, side)
        if np.any(v < 0) or np.any(vbar <= 0):
            raise ContractError('Continuous release native voltage/adaptation domain violated; no clipping')
        bh = self.selection.B@h
        p = self.selection.parameters
        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
            q = 1./(1.+np.exp(-((v-bh-p['V_k'])/p['V_n'])))
            release = q/(vbar/p['V_I'])**p['mu']
        if not np.isfinite(bh).all() or not np.isfinite(release).all():
            raise ContractError('Continuous release became nonfinite')
        return dict(Vis=v, Vbar=vbar, Vh=h, Bh=bh, q=q, release=release)

    def release_at(self, t, side):
        return self.signals_at(t, side)['release']

    def payload(self):
        self.wait()
        packets = {}
        for name, representation in self._representations.items():
            if representation is None:
                packets[name] = None
                continue
            packet = representation.packet()
            packets[name] = dict(metadata=dict(packet['metadata']),
                                 arrays={k: v.copy() for k, v in packet['arrays'].items()})
        return dict(metadata=dict(schema=PACKET_SCHEMA, selected_ids_hash=self.selection.sha256,
            start=self.start, stop=self.stop, input_epoch=self.input_epoch,
            source_version=self.source_version, segment_id=self.segment_id), representations=packets)

    @classmethod
    def from_payload(cls, selection, payload):
        if not isinstance(payload, dict) or set(payload) != {'metadata', 'representations'}:
            raise ContractError('Explicit owned continuous release payload required')
        m = payload['metadata']
        keys = {'schema', 'selected_ids_hash', 'start', 'stop', 'input_epoch', 'source_version', 'segment_id'}
        if (not isinstance(m, dict) or set(m) != keys or m['schema'] != PACKET_SCHEMA
                or m['selected_ids_hash'] != selection.sha256):
            raise ContractError('Continuous release payload selection/schema differs')
        packet = cls(selection, payload['representations'], source_version=m['source_version'], segment_id=m['segment_id'])
        if (packet.start, packet.stop, packet.input_epoch) != (m['start'], m['stop'], m['input_epoch']):
            raise ContractError('Continuous release payload interval/epoch differs')
        return packet.wait()


@dataclass
class _Slot:
    generation: int = 0
    packet: ContinuousReleasePacket | None = None


class ContinuousPacketBridge:
    """Two CPU protocol slots; four consumers jointly govern their reuse."""
    def __init__(self, selection, *, start_time=0.):
        if not _finite_time(start_time):
            raise ContractError('Finite bridge origin required')
        self.selection, self.origin = selection, float(start_time)
        self.frontier, self.last_epoch = self.origin, -1
        self.consumed = dict.fromkeys(CONSUMERS, self.origin)
        self._slots, self._lock = [_Slot(), _Slot()], RLock()

    @property
    def common_watermark(self):
        return min(self.consumed.values())

    def _attach(self, slot, representations, **kwargs):
        slot.generation += 1
        generation = slot.generation
        packet = ContinuousReleasePacket(self.selection, representations,
            lease_check=lambda: slot.generation == generation and slot.packet is packet, **kwargs)
        slot.packet = packet
        return packet

    def submit(self, representations, *, source_version, segment_id=0, ready_waiter=None):
        with self._lock:
            _, (start, stop, epoch) = _preflight(self.selection, representations)
            if start != self.frontier or epoch < self.last_epoch:
                raise ContractError('Continuous bridge requires contiguous accepted intervals and monotone input epochs')
            slot = next((s for s in self._slots if s.packet is None), None)
            if slot is None:
                raise ContractError('Both continuous bridge slots remain owned by consumers')
            packet = self._attach(slot, representations, source_version=source_version,
                                  segment_id=segment_id, ready_waiter=ready_waiter)
            self.frontier, self.last_epoch = stop, epoch
            return packet

    def release(self, packet):
        with self._lock:
            slot = next((s for s in self._slots if s.packet is packet), None)
            if slot is None or self.common_watermark < packet.stop:
                raise ContractError('All four consumer watermarks must pass an owned packet before release')
            packet.wait()
            packet._released = True
            slot.packet = None

    def acknowledge(self, consumer, t):
        with self._lock:
            if (consumer not in CONSUMERS or not _finite_time(t)
                    or not self.consumed[consumer] <= t <= self.frontier):
                raise ContractError('Invalid continuous consumer watermark')
            self.consumed[consumer] = float(t)
            for slot in self._slots:
                if slot.packet is not None and slot.packet.stop <= self.common_watermark:
                    self.release(slot.packet)

    def checkpoint_payload(self):
        with self._lock:
            return dict(schema=BRIDGE_CHECKPOINT_SCHEMA, scope='bridge_ownership_only_not_runner_checkpoint',
                full_runner_restore_supported=False, selection_sha256=self.selection.sha256,
                origin=self.origin, frontier=self.frontier, last_epoch=self.last_epoch,
                consumed=dict(self.consumed), slots=[dict(generation=s.generation,
                    packet=None if s.packet is None else s.packet.payload()) for s in self._slots])

    @classmethod
    def from_checkpoint_payload(cls, selection, payload):
        keys = {'schema', 'scope', 'full_runner_restore_supported', 'selection_sha256',
                'origin', 'frontier', 'last_epoch', 'consumed', 'slots'}
        if (not isinstance(payload, dict) or set(payload) != keys
                or payload['schema'] != BRIDGE_CHECKPOINT_SCHEMA
                or payload['scope'] != 'bridge_ownership_only_not_runner_checkpoint'
                or payload['full_runner_restore_supported'] is not False
                or payload['selection_sha256'] != selection.sha256 or len(payload['slots']) != 2):
            raise ContractError('Explicit matching bridge ownership checkpoint required')
        result = cls(selection, start_time=payload['origin'])
        if (not _finite_time(payload['frontier']) or payload['frontier'] < result.origin
                or not isinstance(payload['consumed'], dict) or set(payload['consumed']) != set(CONSUMERS)
                or any(not _finite_time(t) or not result.origin <= t <= payload['frontier'] for t in payload['consumed'].values())
                or type(payload['last_epoch']) is not int or payload['last_epoch'] < -1
                or payload['frontier'] > result.origin and payload['last_epoch'] < 0
                or payload['frontier'] == result.origin and payload['last_epoch'] != -1):
            raise ContractError('Bridge clock/watermark checkpoint differs')
        result.frontier, result.last_epoch = float(payload['frontier']), payload['last_epoch']
        result.consumed = dict(payload['consumed'])
        active = []
        for slot, record in zip(result._slots, payload['slots']):
            if not isinstance(record, dict) or set(record) != {'generation', 'packet'} or not _integer(record['generation']):
                raise ContractError('Bridge generation checkpoint differs')
            slot.generation = int(record['generation'])
            if record['packet'] is not None:
                if slot.generation < 1:
                    raise ContractError('Active bridge lease has no generation')
                owned = ContinuousReleasePacket.from_payload(selection, record['packet'])
                # Keep the original generation counter; new lease checks bind
                # to this restored bridge, never to an old producer's memory.
                slot.generation -= 1
                packet = result._attach(slot, owned.payload()['representations'],
                    source_version=owned.source_version, segment_id=owned.segment_id)
                packet.wait()
                if (packet.start < result.origin or packet.stop > result.frontier
                        or packet.stop <= result.common_watermark or packet.input_epoch > result.last_epoch):
                    raise ContractError('Bridge checkpoint lost causal packet ownership')
                active.append(packet)
        active.sort(key=lambda packet: packet.start)
        if (active and (active[0].start > result.common_watermark or active[-1].stop != result.frontier
                        or active[-1].input_epoch != result.last_epoch
                        or any(a.stop != b.start or a.input_epoch > b.input_epoch for a, b in zip(active, active[1:])))) or (not active and result.common_watermark != result.frontier):
            raise ContractError('Bridge checkpoint has a missing unconsumed interval')
        return result
