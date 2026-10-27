"""Selected device coefficients to owned CPU packets through two pinned slots.

Published packets own their arrays. GPU slots may be reused after materializing
that immutable copy; four model consumers retain the owned packets separately.
No CUDA work or Torch import occurs at import time.
"""
from dataclasses import dataclass
from threading import RLock
from time import perf_counter

import numpy as np

from .contracts import ContractError
from .continuous_state import Dop853Dense, ExponentialForced, DiscreteHold, QuinticHermite
from .continuous_bridge import ContinuousReleasePacket, CONSUMERS


@dataclass
class _TransferSlot:
    host: object = None
    packed: object = None
    done: object = None
    began: object = None
    descriptors: object = None
    source_refs: object = None
    generation: int = 0
    segment_id: int = 0


class PinnedContinuousTransport:
    def __init__(self, selection, torch, device, *, source_version):
        self.selection, self.torch, self.device = selection, torch, torch.device(device)
        if self.device.type != 'cuda' or not isinstance(source_version, str) or not source_version:
            raise ContractError('Explicit CUDA transport and candidate source identity required')
        self.source_version = source_version
        self.slots = [_TransferSlot(), _TransferSlot()]
        self.lock = RLock()
        self.next_id = 0
        self.metrics = dict(packets=0, D2H_bytes=0, submit_seconds=0., wait_seconds=0.,
                            materialize_seconds=0., CUDA_copy_seconds=0.)

    def _flatten(self, chunk):
        torch = self.torch
        if set(chunk) != {'Vis', 'Vbar', 'Vh'}:
            raise ContractError('All three selected native readout declarations required')
        descriptors, flat, refs, interval = [], [], [], None
        offset = 0
        for name in ('Vis', 'Vbar', 'Vh'):
            rep = chunk[name]
            rows = self.selection.hc_rows if name == 'Vh' else self.selection.cone_rows
            if not len(rows):
                if rep is not None:
                    raise ContractError('No unnecessary HC coefficients may be transferred')
                descriptors.append((name, None))
                continue
            if rep is None or not np.array_equal(np.asarray(rep.state_ids), rows):
                raise ContractError('Device coefficient selection differs: '+name)
            now = (rep.t0, rep.t1, rep.input_epoch)
            if interval is not None and now != interval:
                raise ContractError('Native readout representations must share exact intervals/epochs')
            interval = now
            values = [(n,getattr(rep,n)) for n in ('y0','f0','ym','fm','y1','f1')] if rep.kind=='quintic_hermite' else [('y0', rep.y0)]
            if rep.kind in ('dop853_dense', 'exponential_forced'):
                values.append(('coefficients', rep.coefficients))
            elif rep.kind not in ('discrete_hold','quintic_hermite'):
                raise ContractError('Unsupported native device representation kind')
            if rep.kind == 'exponential_forced':
                decay = rep.decay
                if not hasattr(decay, 'device'):
                    decay = torch.full((len(rows),), float(decay), dtype=torch.float64, device=self.device)
                elif decay.ndim == 0:
                    decay = decay.expand(len(rows))
                values.append(('decay', decay))
            shapes = dict(y0=(len(rows),), coefficients=(7 if rep.kind == 'dop853_dense' else 8, len(rows)), decay=(len(rows),),
                **{n:(len(rows),) for n in ('f0','ym','fm','y1','f1')})
            fields = []
            for key, value in values:
                if value.dtype != torch.float64 or value.device != self.device or tuple(value.shape) != shapes[key]:
                    raise ContractError('Selected device FP64 coefficient shape differs: '+name+'.'+key)
                count = value.numel()
                fields.append((key, offset, offset+count, shapes[key]))
                flat.append(value.reshape(-1))
                refs.append(value)
                offset += count
            descriptors.append((name, dict(kind=rep.kind, t0=rep.t0, t1=rep.t1,
                epoch=rep.input_epoch, state_ids=np.array(rows, copy=True), fields=fields)))
        return torch.cat(flat), descriptors, (chunk, refs)

    def _finish(self, slot):
        if slot.descriptors is None:
            return None
        started = perf_counter()
        slot.done.synchronize()
        self.metrics['wait_seconds'] += perf_counter()-started
        self.metrics['CUDA_copy_seconds'] += slot.began.elapsed_time(slot.done)*.001
        started = perf_counter()
        # This read starts AFTER the ready event. Constructors make immutable
        # owned byte copies and calculate packet hashes only now.
        values = slot.host.numpy()
        representations = self.materialize(values, slot.descriptors)
        packet = ContinuousReleasePacket(self.selection, representations,
            source_version=self.source_version, segment_id=slot.segment_id).wait()
        self.metrics['materialize_seconds'] += perf_counter()-started
        slot.descriptors = slot.source_refs = slot.packed = None
        return packet

    @staticmethod
    def materialize(values, descriptors):
        if values.dtype != np.float64 or values.ndim != 1 or not np.isfinite(values).all():
            raise ContractError('Completed transfer must contain finite FP64 coefficients')
        classes = dict(dop853_dense=Dop853Dense, exponential_forced=ExponentialForced, discrete_hold=DiscreteHold, quintic_hermite=QuinticHermite)
        result = {}
        for name, row in descriptors:
            if row is None:
                result[name] = None
                continue
            arrays = {key: values[first:last].reshape(shape) for key, first, last, shape in row['fields']}
            result[name] = classes[row['kind']](t0=row['t0'], t1=row['t1'],
                epoch=row['epoch'], state_ids=row['state_ids'], **arrays)
        return result

    def transfer(self, chunks):
        """Synchronous public return, overlapped copies internally; no slot alias."""
        with self.lock, self.torch.no_grad():
            finished = []
            try:
                for i, chunk in enumerate(chunks):
                    slot = self.slots[i % 2]
                    packet = self._finish(slot)
                    if packet is not None:
                        finished.append(packet)
                    started = perf_counter()
                    packed, descriptors, refs = self._flatten(chunk)
                    if slot.host is None or slot.host.numel() != packed.numel():
                        slot.host = self.torch.empty(packed.numel(), dtype=self.torch.float64,
                                                     device='cpu', pin_memory=True)
                    slot.packed, slot.source_refs, slot.descriptors = packed, refs, descriptors
                    slot.generation += 1
                    slot.segment_id, self.next_id = self.next_id, self.next_id+1
                    slot.began = self.torch.cuda.Event(enable_timing=True)
                    slot.done = self.torch.cuda.Event(enable_timing=True)
                    stream = self.torch.cuda.current_stream(self.device)
                    slot.began.record(stream)
                    slot.host.copy_(slot.packed, non_blocking=True)
                    slot.done.record(stream)
                    self.metrics['packets'] += 1
                    self.metrics['D2H_bytes'] += packed.numel()*8
                    self.metrics['submit_seconds'] += perf_counter()-started
                for slot in self.slots:
                    packet = self._finish(slot)
                    if packet is not None:
                        finished.append(packet)
            except Exception:
                # A failed copy still owns its producer storage until completion.
                for slot in self.slots:
                    if slot.descriptors is not None and slot.done is not None:
                        slot.done.synchronize()
                raise
            return tuple(sorted(finished, key=lambda packet: (packet.start, packet.segment_id)))


class OwnedReadoutWindow:
    """Four consumers share immutable packets, independently of pinned slots."""
    def __init__(self, packets):
        self.packets = tuple(packets)
        if not self.packets or any(a.stop != b.start for a, b in zip(self.packets, self.packets[1:])):
            raise ContractError('Contiguous owned native readout intervals required')
        self.start, self.stop = self.packets[0].start, self.packets[-1].stop
        self.consumed = dict.fromkeys(CONSUMERS, self.start)

    def at(self, t, side):
        if side not in ('left', 'right') or not self.start <= t <= self.stop:
            raise ContractError('Readout window query outside accepted interval/side')
        for packet in self.packets:
            if ((packet.start < t < packet.stop) or (t == packet.start and side == 'right')
                    or (t == packet.stop and side == 'left')):
                return packet.release_at(t, side)
        raise ContractError('No accepted packet owns this exact boundary side')

    def acknowledge(self, consumer, t):
        if consumer not in self.consumed or not self.consumed[consumer] <= t <= self.stop:
            raise ContractError('Invalid readout consumer watermark')
        self.consumed[consumer] = t

    def require_consumed(self):
        if any(t != self.stop for t in self.consumed.values()):
            raise ContractError('Unconsumed owned readout window cannot be discarded')
