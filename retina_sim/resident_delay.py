"""Opt-in device storage for the original native observation delay only.

No torch import, device allocation or CUDA initialization occurs on import.
Construction requires the explicitly selected resident runner's torch object.
Rows are owned snapshots; forks share rows that this class never mutates.
This module does not implement or enable delayed link operators.
Runtime/numerical equivalence has not yet been tested.
"""
from bisect import bisect_right
from pathlib import Path
import hashlib
import numpy as np
from .contracts import ContractError


RECORDING_SOURCE_SHA256 = 'd27508a6c8904704d4916f18d04ef6d356d1b3707fac3fbdda9f3f0ecfe4bd75'


def _check_source():
    actual = hashlib.sha256(Path(__file__).with_name('recording.py').read_bytes()).hexdigest()
    if actual != RECORDING_SOURCE_SHA256:
        raise ContractError('Unreviewed TimestampedDelay source change; resident delay unavailable')


class ResidentTimestampedDelay:
    """Same timestamp/interpolation/pruning protocol, device-owned values.

    history_times is CPU metadata. Numerical rows and prehistory are private;
    at() returns an independent device result and payload() explicitly exports
    a complete CPU checkpoint payload. No advance path calls payload().
    """

    @classmethod
    def from_payload(cls, payload, torch, device):
        _check_source()
        self = cls.__new__(cls)
        self.torch = torch
        self.device = torch.device(device)
        self.delay_seconds = float(payload['delay_seconds'])
        self.interpolation = payload['interpolation']
        if self.delay_seconds < 0 or self.interpolation not in ('linear', 'zero_order_hold'):
            raise ContractError('Explicit sourced delay/interpolation required')
        self.history_times = list(payload['history_times'])
        prehistory = payload['prehistory']
        self.unit = prehistory['unit']
        self.meaning = prehistory['meaning']
        self.evidence_tag = prehistory['evidence_tag']
        self._prehistory = torch.tensor(
            prehistory['values'], dtype=torch.float64, device=self.device)
        self.shape = tuple(self._prehistory.shape)
        if len(self.shape) != 1 or not self.unit or not self.meaning or not self.evidence_tag:
            raise ContractError('Observation delay requires one-dimensional declared Quantity metadata')
        if not torch.isfinite(self._prehistory).all().item():
            raise ContractError('Nonfinite array')
        self._values = [torch.tensor(row, dtype=torch.float64, device=self.device)
                        for row in payload['history_values']]
        if len(self.history_times) != len(self._values) or any(tuple(row.shape) != self.shape for row in self._values):
            raise ContractError('Observation delay history layout differs')
        self.prehistory_stop_seconds = payload.get('prehistory_stop_seconds')
        if self.prehistory_stop_seconds is None:
            if not self.history_times:
                raise ContractError('Explicit delay prehistory boundary required')
            self.prehistory_stop_seconds = float(self.history_times[0])
        return self

    def fork(self):
        other = type(self).__new__(type(self))
        other.__dict__ = dict(self.__dict__)
        other.history_times = list(self.history_times)
        other._values = list(self._values)
        # Both containers only append/prune; existing tensors are never changed.
        return other

    def append(self, t, values, unit, meaning):
        torch = self.torch
        if (unit, meaning) != (self.unit, self.meaning) or tuple(values.shape) != self.shape:
            raise ContractError('Observation delay Quantity contract differs')
        if values.dtype != torch.float64 or values.device != self.device:
            raise ContractError('Observation delay values must remain device FP64')
        if not torch.isfinite(values).all().item():
            raise ContractError('Nonfinite array')
        if self.history_times and t <= self.history_times[-1]:
            raise ContractError('Delay history timestamps must increase')
        self.history_times.append(float(t))
        self._values.append(values.clone())
        while len(self.history_times) > 2 and self.history_times[1] < t - self.delay_seconds:
            self.history_times.pop(0)
            self._values.pop(0)

    def at(self, t):
        torch = self.torch
        target = t - self.delay_seconds
        if target <= self.prehistory_stop_seconds:
            return self._prehistory.clone()
        if not self.history_times or target < self.history_times[0]:
            raise ContractError('Delay history was pruned for this readout; no substituted background')
        if target > self.history_times[-1]:
            raise ContractError('Delay requires current timestamp, no future extrapolation')
        index = min(bisect_right(self.history_times, target) - 1, len(self.history_times) - 1)
        left = self._values[index]
        if self.interpolation == 'zero_order_hold' or target == self.history_times[index]:
            value = left.clone()
        else:
            right = self._values[index + 1]
            # Preserve recording.TimestampedDelay.at's slope-then-offset order.
            slope = (right - left) / (self.history_times[index + 1] - self.history_times[index])
            value = slope * (target - self.history_times[index]) + left
            missing = torch.isnan(value)
            if missing.any().item():
                value[missing] = slope[missing] * (target - self.history_times[index + 1]) + right[missing]
                missing = torch.isnan(value) & (left == right)
                value[missing] = left[missing]
        # The reference returns Quantity, whose construction rejects nonfinite.
        if not torch.isfinite(value).all().item():
            raise ContractError('Nonfinite array')
        return value

    def history_failures(self, now, policy, registry):
        """Original _history_consistency suffixes without history D2H."""
        torch = self.torch
        failures = []
        times = self.history_times
        if len(times) and (np.any(np.diff(times) < 0) or times[-1] > now + 1e-12):
            failures.append('invalid_timestamp_order')
        if self._values and not torch.stack([
                torch.isfinite(row).all() for row in self._values]).all().item():
            failures.append('nonfinite_values')
        if self.unit not in registry:
            failures.append('unregistered_queue_scale')
        elif len(self._values) > 1:
            entry = registry[self.unit]
            scale = torch.maximum(self._prehistory.abs(), torch.full_like(self._prehistory, entry['scale_floor']))
            lo = self._values[0]
            hi = self._values[0]
            for row in self._values[1:]:
                lo = torch.minimum(lo, row)
                hi = torch.maximum(hi, row)
            if ((hi - lo) > entry['atol'] + policy.relative_tolerance * scale).any().item():
                failures.append('queue_not_constant_within_tolerance')
        return failures

    def payload(self,*,arrays=False):
        """Explicit complete export; never used for ordinary resident stepping."""
        return {
            'delay_seconds': self.delay_seconds,
            'history_times': list(self.history_times),
            'history_values': list(self._values) if arrays else [row.cpu().numpy().tolist() for row in self._values],
            'prehistory': {
                'values': self._prehistory if arrays else self._prehistory.cpu().numpy().tolist(),
                'unit': self.unit,
                'meaning': self.meaning,
                'evidence_tag': self.evidence_tag,
            },
            'interpolation': self.interpolation,
            'prehistory_stop_seconds': self.prehistory_stop_seconds,
        }
