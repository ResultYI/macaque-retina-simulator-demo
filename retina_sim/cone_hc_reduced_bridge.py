"""Accepted-segment FP64 CPU release bridge for the reduced native candidate.

Only construction of PinnedSegmentBridge performs device work.  Packet queries
are CPU-only: reconstruct Vis/Vbar/h, apply the original B rows, then release.
"""
from dataclasses import dataclass
from numbers import Real
from threading import RLock
from time import perf_counter
from types import MappingProxyType

import numpy as np
from scipy import sparse

from .contracts import ContractError, canonical_hash
from .vh2005 import SOURCE_VALUES


PACKET_SCHEMA = 'cone-hc-reduced-cpu-segment-1'


def _indices(values, size, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in 'iu' or not len(raw):
        raise ContractError(name+' must be a nonempty integer vector')
    result = raw.astype(np.int64, copy=True)
    if (np.any(result < 0) or np.any(result >= size)
            or len(np.unique(result)) != len(result)):
        raise ContractError(name+' contains duplicate/out-of-range rows')
    result.setflags(write=False)
    return result


def _ids(values, size, name):
    if values is None:
        return None
    result = tuple(values)
    if len(result) != size or len(set(result)) != size:
        raise ContractError(name+' must cover the complete distinct cell IDs')
    return result


class BridgeSelection:
    """Frozen actual C*/H* and exact within-row B associations.

    cone_rows order is the compact CPU inner input order; it is never sorted.
    HC columns are remapped without changing any B weight or row association.
    """

    def __init__(self, B, cone_rows, *, parameters=None, cone_ids=None, hc_ids=None):
        if not sparse.isspmatrix_csr(B) or B.dtype != np.float64:
            raise ContractError('CPU bridge requires the original FP64 CSR B')
        self.nc, self.nh = B.shape
        if not self.nc or not self.nh or not np.all(np.isfinite(B.data)) or np.any(B.data < 0):
            raise ContractError('CPU bridge B has invalid shape/weights')
        self.cone_rows = _indices(cone_rows, self.nc, 'C*')
        cone_ids = _ids(cone_ids, self.nc, 'Cone IDs')
        hc_ids = _ids(hc_ids, self.nh, 'HC IDs')
        p = dict(SOURCE_VALUES if parameters is None else parameters)
        if p != SOURCE_VALUES:
            raise ContractError('CPU bridge requires unchanged native source parameters')
        self.parameters = MappingProxyType(p)

        # No sort_indices/sum_duplicates: preserve the supplied execution order.
        chosen = B[self.cone_rows].tocsr(copy=True)
        for begin, end in zip(chosen.indptr[:-1], chosen.indptr[1:]):
            if begin == end or len(np.unique(chosen.indices[begin:end])) != end-begin:
                raise ContractError('CPU bridge requires nonempty distinct original B rows')
        self.hc_rows = np.unique(chosen.indices).astype(np.int64)
        self.hc_rows.setflags(write=False)
        mapped = np.searchsorted(self.hc_rows, chosen.indices)
        self.B = sparse.csr_matrix((chosen.data.copy(), mapped.copy(), chosen.indptr.copy()),
                                   shape=(len(self.cone_rows), len(self.hc_rows)))
        for array in (self.B.data, self.B.indices, self.B.indptr):
            array.setflags(write=False)
        self.count, self.hc_count = self.B.shape
        self.width = 2*self.count+self.hc_count
        self.native_size = 5*self.nc+3*self.nh
        self.rows = np.concatenate((2*self.nc+self.cone_rows, 4*self.nc+self.cone_rows,
                                    5*self.nc+2*self.nh+self.hc_rows))
        self.rows.setflags(write=False)
        self.identity = dict(schema=PACKET_SCHEMA, nc=self.nc, nh=self.nh,
            cone_rows=self.cone_rows, hc_rows=self.hc_rows, state_rows=self.rows,
            selected_cone_ids=None if cone_ids is None else tuple(cone_ids[i] for i in self.cone_rows),
            selected_hc_ids=None if hc_ids is None else tuple(hc_ids[i] for i in self.hc_rows),
            B_indptr=self.B.indptr.copy(), B_indices=self.B.indices.copy(), B_data=self.B.data.copy(),
            parameters=dict(p), array_layout='three_rows_x0_f0_x1__Vis_Vbar_h',
            evaluation_order='quadratic_state_then_original_B_rows_then_sigmoid_then_adaptation')
        self.sha256 = canonical_hash(self.identity)

    @property
    def packet_bytes(self):
        return 8*(6*self.count+3*self.hc_count)

    def _metadata(self, segment, *, segment_id, input_side, source_version):
        if (any(isinstance(t, (bool, np.bool_)) or not isinstance(t, Real)
                for t in (segment.start, segment.stop))
                or not np.isfinite(segment.start) or not np.isfinite(segment.stop)
                or segment.stop <= segment.start):
            raise ContractError('CPU bridge requires a positive accepted segment')
        if input_side not in ('right', 'left') or not isinstance(source_version, str) or not source_version:
            raise ContractError('CPU bridge segment needs explicit input side/source version')
        epoch = segment.input_epoch
        if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)) or epoch < 0:
            raise ContractError('CPU bridge input epoch must be a nonnegative integer')
        if isinstance(segment_id, bool) or not isinstance(segment_id, (int, np.integer)) or segment_id < 0:
            raise ContractError('CPU bridge segment ID must be a nonnegative integer')
        return dict(schema=PACKET_SCHEMA, start=float(segment.start), stop=float(segment.stop),
                    segment_id=int(segment_id), input_epoch=int(epoch), input_side=input_side,
                    source_version=str(source_version), selected_ids_hash=self.sha256)

    def project(self, segment, *, segment_id=0, input_side='right', source_version=None):
        """CPU/reference projection, with a private owned immutable packet."""
        version = source_version if source_version is not None else getattr(segment, 'source_version', '')
        metadata = self._metadata(segment, segment_id=segment_id,
                                  input_side=input_side, source_version=version)
        arrays = tuple(np.asarray(getattr(segment, name)) for name in ('y0', 'f0', 'y1'))
        if any(a.dtype != np.float64 or a.ndim != 1 or len(a) < self.native_size for a in arrays):
            raise ContractError('CPU bridge source must retain complete reduced FP64 native layout')
        if len({a.shape for a in arrays}) != 1:
            raise ContractError('CPU bridge source x0/f0/x1 layouts differ')
        packed = np.stack([a[self.rows] for a in arrays])
        return SegmentPacket(self, packed, metadata)

    def from_payload(self, raw):
        if (not isinstance(raw, dict) or set(raw) != {'metadata', 'values'}
                or not isinstance(raw['metadata'], dict)
                or set(raw['metadata']) != {'schema', 'start', 'stop', 'segment_id', 'input_epoch',
                                             'input_side', 'source_version', 'selected_ids_hash'}
                or raw['metadata'].get('selected_ids_hash') != self.sha256
                or raw['metadata'].get('schema') != PACKET_SCHEMA):
            raise ContractError('CPU bridge checkpoint selection/version differs')
        meta = raw['metadata']
        expected = self._metadata(type('Segment', (), meta)(), segment_id=meta['segment_id'],
                                  input_side=meta['input_side'], source_version=meta['source_version'])
        if meta != expected:
            raise ContractError('CPU bridge checkpoint metadata is incomplete or changed')
        values = np.asarray(raw['values'])
        if values.dtype != np.float64:
            raise ContractError('CPU bridge checkpoint must remain FP64')
        return SegmentPacket(self, values.copy(), dict(meta))


class SegmentPacket:
    """An owned CPU packet, or one explicitly leased pinned-host slot.

    A lease is readable only after its ready event and until explicit release.
    payload() waits and copies, so checkpoint data never aliases a reusable slot.
    """

    def __init__(self, selection, packed, metadata, *, waiter=None, lease_check=None, ready_event=None):
        if packed.dtype != np.float64 or packed.shape != (3, selection.width) or not packed.flags.c_contiguous:
            raise ContractError('CPU bridge packet FP64 SoA shape differs')
        if metadata.get('selected_ids_hash') != selection.sha256:
            raise ContractError('CPU bridge packet selected layout differs')
        self.selection = selection
        self.metadata = dict(metadata)
        self.start, self.stop = metadata['start'], metadata['stop']
        self.input_epoch = metadata['input_epoch']
        self.segment_id = metadata['segment_id']
        self.source_version = metadata['source_version']
        self.ready_event = ready_event
        self._packed = packed
        self._packed.setflags(write=False)
        self._waiter, self._lease_check = waiter, lease_check
        self._ready = False
        self._released = False
        if waiter is None:
            self.wait()

    def _check(self):
        if self._released or (self._lease_check is not None and not self._lease_check()):
            raise ContractError('CPU bridge packet slot has been released/reused')

    def wait(self):
        self._check()
        if not self._ready:
            if self._waiter is not None:
                self._waiter()
            if not np.all(np.isfinite(self._packed)):
                raise ContractError('CPU bridge packet contains nonfinite accepted state/derivative')
            self._ready = True
        return self

    @property
    def values(self):
        self.wait()
        return self._packed

    @property
    def nbytes(self):
        return self._packed.nbytes

    def state_at(self, t):
        self.wait()
        at = float(t)
        if not np.isfinite(at) or at < self.start or at > self.stop:
            raise ContractError('CPU bridge query is outside its accepted segment; no extrapolation')
        x0, f0, x1 = self._packed
        if at == self.start:
            value = x0.copy()
        elif at == self.stop:
            value = x1.copy()
        else:
            duration = self.stop-self.start
            theta = (at-self.start)/duration
            value = x0+theta*duration*f0+theta**2*(x1-x0-duration*f0)
        if not np.all(np.isfinite(value)):
            raise ContractError('CPU bridge quadratic state became nonfinite')
        c = self.selection.count
        return value[:c], value[c:2*c], value[2*c:]

    def signals_at(self, t):
        v, vbar, h = self.state_at(t)
        if np.any(v < 0) or np.any(vbar <= 0):
            raise ContractError('CPU bridge native voltage/adaptation domain violated; no clipping')
        p = self.selection.parameters
        bh = self.selection.B @ h
        # Match the original sigmoid expression, including finite saturation.
        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
            q = 1./(1.+np.exp(-((v-bh-p['V_k'])/p['V_n'])))
            release = q/(vbar/p['V_I'])**p['mu']
        if not np.all(np.isfinite(bh)) or not np.all(np.isfinite(release)):
            raise ContractError('CPU bridge nonlinear release became nonfinite')
        return dict(Vis=v, Vbar=vbar, Vh=h, Bh=bh, q=q, release=release)

    def release_at(self, t):
        return self.signals_at(t)['release']

    def payload(self):
        self.wait()
        return dict(metadata=dict(self.metadata), values=self._packed.copy())


@dataclass
class _Slot:
    host: object
    device: object
    event: object
    copy_start: object
    generation: int = 0
    packet: object = None


class PinnedSegmentBridge:
    """Exactly two pinned slots, one asynchronous D2H per accepted segment.

    The producer must call submit on the CUDA stream producing the segment.
    The caller releases a packet only once its relevant consumer watermark has
    passed it. A third outstanding packet is rejected, never silently replaced.
    """

    def __init__(self, selection, torch, device):
        self.selection, self.torch = selection, torch
        self.rows = torch.as_tensor(selection.rows.copy(), dtype=torch.int64, device=device)
        self.device = self.rows.device
        if self.device.type != 'cuda':
            raise ContractError('Pinned segment bridge requires an explicitly selected CUDA device')
        shape = (3, selection.width)
        self._slots = [_Slot(torch.empty(shape, dtype=torch.float64, device='cpu', pin_memory=True),
                             torch.empty(shape, dtype=torch.float64, device=self.device),
                             torch.cuda.Event(enable_timing=True),
                             torch.cuda.Event(enable_timing=True)) for _ in range(2)]
        self._lock = RLock()
        self.metrics = dict(submitted_segments=0, D2H_bytes=0, submit_wall_seconds=0.,
                            wait_wall_seconds=0., CUDA_copy_seconds=0.)

    def submit(self, segment, *, segment_id=0, input_side='right', source_version=None):
        version = source_version if source_version is not None else getattr(segment, 'source_version', '')
        metadata = self.selection._metadata(segment, segment_id=segment_id,
                                           input_side=input_side, source_version=version)
        with self._lock:
            slot = next((s for s in self._slots if s.packet is None), None)
            if slot is None:
                raise ContractError('Both CPU bridge slots remain owned by unconsumed segments')
            values = tuple(getattr(segment, name) for name in ('y0', 'f0', 'y1'))
            if any(v.dtype != self.torch.float64 or v.ndim != 1 or v.numel() < self.selection.native_size
                   or v.device != self.device for v in values):
                raise ContractError('GPU bridge source state/device/FP64 layout differs')
            if len({tuple(v.shape) for v in values}) != 1:
                raise ContractError('GPU bridge source x0/f0/x1 layouts differ')
            began = perf_counter()
            with self.torch.no_grad():
                for i, value in enumerate(values):
                    self.torch.index_select(value, 0, self.rows, out=slot.device[i])
                slot.copy_start.record(self.torch.cuda.current_stream(self.device))
                slot.host.copy_(slot.device, non_blocking=True)
                slot.event.record(self.torch.cuda.current_stream(self.device))
            slot.generation += 1
            generation = slot.generation

            def wait():
                start = perf_counter()
                slot.event.synchronize()
                self.metrics['wait_wall_seconds'] += perf_counter()-start
                self.metrics['CUDA_copy_seconds'] += slot.copy_start.elapsed_time(slot.event)*.001

            packet = SegmentPacket(self.selection, slot.host.numpy(), metadata, waiter=wait,
                                   lease_check=lambda: slot.generation == generation and slot.packet is packet,
                                   ready_event=slot.event)
            slot.packet = packet
            self.metrics['submitted_segments'] += 1
            self.metrics['D2H_bytes'] += self.selection.packet_bytes
            self.metrics['submit_wall_seconds'] += perf_counter()-began
            return packet

    def release(self, packet):
        with self._lock:
            slot = next((s for s in self._slots if s.packet is packet), None)
            if slot is None:
                raise ContractError('CPU bridge packet is not owned by this active slot lease')
            packet.wait()
            packet._released = True
            slot.packet = None

    def payloads(self):
        with self._lock:
            return [s.packet.payload() for s in self._slots if s.packet is not None]

