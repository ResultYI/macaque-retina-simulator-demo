"""Persistent full-registry FP64 monitoring without changing its frozen contract.

Reports/checkpoints retain the original schema and first-argmax rule. Gap-current
conservation deliberately keeps the original NumPy reduction order; sampled
history/RNG/pending-event predicates remain the original CPU checks.
"""
import math
from time import perf_counter

import numpy as np

from .contracts import ContractError
from .group_time_monitor import GroupTimeMonitor, _cpu, _history_failures, _name
from .resident_monitor import UNIT_INTERVAL_MEANINGS, NONNEGATIVE_MEANINGS
from .spatial_group_model import BC, RGC


class GroupDeviceObserver(GroupTimeMonitor):
    """Own contiguous statistics on one device; consume complete snapshot dicts.

    ``torch=None`` provides the same owner on NumPy for small synthetic tests.
    With torch supplied, every numeric state/RHS/transition must already be FP64
    on ``device``. ``allow_host_inputs=True`` explicitly permits FP64 NumPy
    witnesses, packed into reusable host storage and uploaded in contiguous runs.
    Runtime uploads are blocking and counted in ``host_transfer_metrics`` inside
    monitor wall time; constructor/restore transfers are setup costs. Mutable BC
    parameter bounds are refreshed only when their actual binding changes.
    """

    def __init__(self, assembly, initial_states, *, parameters=None, torch=None,
                 relative_tolerance=.0001, device=None, allow_host_inputs=False):
        if type(allow_host_inputs) is not bool:
            raise ContractError('Explicit boolean host-input permission required')
        self.allow_host_inputs = allow_host_inputs
        self.host_transfer_metrics = dict(bytes=0, seconds=0., transfers=0, packing_seconds=0.)
        super().__init__(assembly, initial_states, parameters=parameters, torch=torch,
                         relative_tolerance=relative_tolerance)
        if torch is None:
            if device not in (None, 'cpu'):
                raise ContractError('A torch owner is required for a non-CPU monitor')
            self.device = None
        else:
            first = next(iter(initial_states.values()), None)
            self.device = torch.device(device if device is not None else
                                       getattr(first, 'device', 'cpu'))
        self._slices = {}
        offset = 0
        for key, row in self.rows.items():
            self._slices[key] = slice(offset, offset + row['spec'].count)
            offset += row['spec'].count
        self._buffers = {}
        self._cpu_work = np.empty(0, dtype=np.float64)

        def owned(name, value):
            value = np.asarray(value)
            result = value.copy() if torch is None else torch.as_tensor(
                value, device=self.device).clone()
            self._buffers[name] = result
            return result

        self._statistics = owned('statistics', np.zeros((4, offset), dtype=np.float64))
        if torch is not None:
            self.device = self._statistics.device
        self._values = owned('states', np.empty(offset, dtype=np.float64))
        self._rhs_values = owned('rhs', np.empty(offset, dtype=np.float64))
        self._transition_values = owned('transitions', np.empty(offset, dtype=np.float64))
        self._work = owned('work', np.empty(offset, dtype=np.float64))
        periods = np.ones(offset, dtype=np.float64)
        interval = np.zeros(offset, dtype=bool)
        nonnegative = np.zeros(offset, dtype=bool)
        upper = np.full(offset, np.inf, dtype=np.float64)
        row_index = np.empty(offset, dtype=np.int64)
        self._adaptation_scales = {}
        for index, (key, row) in enumerate(self.rows.items()):
            section = self._slices[key]
            spec = row['spec']
            row_index[section] = index
            if row['period'] is not None:
                periods[section] = row['period']
            interval[section] = spec.unit == 'dimensionless' and (
                spec.meaning in UNIT_INTERVAL_MEANINGS or spec.mechanism == 'RGC_expected_history')
            nonnegative[section] = spec.meaning in NONNEGATIVE_MEANINGS
            if spec.name == 'A' and spec.population in BC:
                if parameters is None:
                    raise ContractError('Actual BC adaptation scale binding required by full monitor')
                scale = np.asarray(parameters['populations'][spec.population]['adaptation_scale'])
                self._adaptation_scales[key] = scale.copy()
                upper[section] = scale + 1e-10
        self._periods = owned('periods', periods)
        self._interval = owned('interval', interval)
        self._nonnegative = owned('nonnegative', nonnegative)
        self._upper = owned('adaptation_upper', upper)
        self._row_index = owned('row_index', row_index)
        # CUDA scatter amax supports integer 0/1 flags, but not Bool in Torch 2.10.
        self._domain_cells = owned('domain_cells', np.empty((3, offset), dtype=np.int32))
        self._domain_rows = owned('domain_rows', np.zeros((3, len(self.rows)), dtype=np.int32))
        self._host_staging_pinned = bool(torch is not None and allow_host_inputs and self.device.type == 'cuda')
        if self._host_staging_pinned:
            self._host_tensor = torch.empty(offset, dtype=torch.float64, device='cpu', pin_memory=True)
            self._host_staging = self._host_tensor.numpy()
        else:
            self._host_staging = np.empty(offset if torch is not None and allow_host_inputs else 0, dtype=np.float64)
            self._host_tensor = None if torch is None else torch.from_numpy(self._host_staging)

    def storage_accounting(self):
        """Actual persistent owner allocations; temporary operation results excluded."""
        allocations = {name: (value.nbytes if self.torch is None else
                             value.numel() * value.element_size())
                       for name, value in self._buffers.items()}
        frozen_host_bytes = sum(value.nbytes for row in self.rows.values()
                                for name in ('initial', 'scale', 'den', 'budget')
                                if (value := row[name]) is not None)
        binding_host_bytes = sum(value.nbytes for value in self._adaptation_scales.values())
        return dict(device='numpy' if self.torch is None else str(self.device),
                    allocations=allocations, frozen_host_bytes=frozen_host_bytes,
                    binding_host_bytes=binding_host_bytes, host_staging_bytes=self._host_staging.nbytes,
                    host_staging_pinned=self._host_staging_pinned,
                    total_bytes=sum(allocations.values()) + frozen_host_bytes + binding_host_bytes
                    + self._host_staging.nbytes)

    def _copy(self, target, source):
        if self.torch is None:
            np.copyto(target, source, casting='no')
        else:
            target.copy_(source.detach())

    def _validate_array(self, value, key, label):
        shape = (self.rows[key]['spec'].count,)
        if self.torch is None:
            value = np.asarray(value)
            valid = value.dtype == np.float64 and value.shape == shape
        else:
            valid = (isinstance(value, self.torch.Tensor) and value.dtype == self.torch.float64
                     and tuple(value.shape) == shape and value.device == self.device)
            if self.allow_host_inputs and isinstance(value, np.ndarray):
                valid = value.dtype == np.float64 and value.shape == shape
        if not valid:
            raise ContractError('Original FP64 monitor ' + label + ' layout/device differs: ' + _name(key))
        return value

    def _upload(self, target, host):
        started = perf_counter()
        target.copy_(host, non_blocking=False)
        self.host_transfer_metrics['seconds'] += perf_counter() - started
        self.host_transfer_metrics['bytes'] += host.numel() * host.element_size()
        self.host_transfer_metrics['transfers'] += 1

    def _ingest(self, target, values, keys, label):
        host_rows = []
        for key in keys:
            section = self._slices[key]
            value = self._validate_array(values[key], key, label)
            if self.torch is not None and isinstance(value, np.ndarray):
                if section.stop > section.start:
                    host_rows.append((section, value))
            else:
                self._copy(target[section], value)
        if not host_rows:
            return
        started = perf_counter()
        runs = []
        for section, value in host_rows:
            np.copyto(self._host_staging[section], value, casting='no')
            if runs and runs[-1].stop == section.start:
                runs[-1] = slice(runs[-1].start, section.stop)
            else:
                runs.append(section)
        self.host_transfer_metrics['packing_seconds'] += perf_counter() - started
        for section in runs:
            self._upload(target[section], self._host_tensor[section])

    def _finite(self, value):
        if self.torch is None:
            return bool(np.isfinite(value).all())
        return bool(self.torch.isfinite(value).all().item())

    def _bind_rows(self):
        for key, row in self.rows.items():
            for index, field in enumerate(('lo', 'hi', 'rhs', 'defect')):
                row[field] = self._statistics[index, self._slices[key]]
            row['native_tensor'] = self.torch is not None
            row.pop('restored_cpu_statistics', None)

    def observe_snapshot(self, snapshot):
        return self.observe(snapshot['time_seconds'], snapshot['side'], snapshot['states'],
                            snapshot['rhs'], transitions=snapshot['transitions'],
                            ports=snapshot['ports'], event_generation=snapshot['event_generation'],
                            aux=snapshot['aux'], rhs_kind=snapshot['rhs_kind'])

    def observe(self, t, side, states, rhs, *, transitions, ports=None,
                event_generation=0, aux=None, rhs_kind='original_equation'):
        if (not math.isfinite(t) or side not in ('left', 'right')
                or type(event_generation) is not int or event_generation < 0
                or rhs_kind != 'original_equation'):
            raise ContractError('Exact stamp/side/generation and original-equation RHS required')
        if (set(states) != set(self.specs) or set(rhs) != set(self.specs)
                or set(transitions) != set(self.transition_keys)):
            raise ContractError('Full state, true RHS and pure one-step transition coverage required')
        self._ingest(self._values, states, self.rows, 'state')
        self._ingest(self._rhs_values, rhs, self.rows, 'RHS')
        self._copy(self._transition_values, self._values)
        self._ingest(self._transition_values, transitions, self.transition_keys, 'pure transition')
        if self.torch is None:
            finite = (np.isfinite(self._values).all() and np.isfinite(self._rhs_values).all()
                      and np.isfinite(self._transition_values).all())
        else:
            finite = bool((self.torch.isfinite(self._values).all()
                           & self.torch.isfinite(self._rhs_values).all()
                           & self.torch.isfinite(self._transition_values).all()).item())
        if not finite:
            for key in self.rows:
                section = self._slices[key]
                if not self._finite(self._values[section]) or not self._finite(self._rhs_values[section]):
                    raise ContractError('Nonfinite original state/RHS: ' + _name(key))
                if key in transitions and not self._finite(self._transition_values[section]):
                    raise ContractError('Pure transition layout differs: ' + _name(key))
        failures = self._domain_failures()
        # Keep the original host reduction sequence, including per-port sums.
        if ports is not None:
            currents = [_cpu(v) for k, v in ports.items() if k[-1] == 'gap_current']
            if currents:
                total = sum(float(np.sum(a)) for a in currents)
                magnitude = sum(float(np.sum(np.abs(a))) for a in currents)
                if abs(total) > 1e-16 + 1e-10 * magnitude:
                    failures.append('reciprocal_gap_current_conservation')
        if aux is not None:
            failures.extend(_history_failures(aux, t))
        self._update_statistics()
        self.domain_failures.extend(failures)
        self.count += 1
        self.stamps.append(dict(time_hex=float(t).hex(), side=side, event_generation=event_generation))

    def _refresh_adaptation_bounds(self):
        for key, previous in self._adaptation_scales.items():
            scale = np.asarray(self.parameters['populations'][key[0]]['adaptation_scale'])
            if scale.dtype == previous.dtype and np.array_equal(scale, previous, equal_nan=True):
                continue
            bound = np.array(np.broadcast_to(scale + 1e-10, (self.rows[key]['spec'].count,)),
                             dtype=np.float64, copy=True)
            if self.torch is not None:
                self._upload(self._upper[self._slices[key]], self.torch.from_numpy(bound))
            else:
                self._copy(self._upper[self._slices[key]], bound)
            self._adaptation_scales[key] = scale.copy()

    def _domain_failures(self):
        self._refresh_adaptation_bounds()
        value = self._values
        self._domain_cells[0] = ((value < -1e-10) | (value > 1 + 1e-10)) & self._interval
        self._domain_cells[1] = (value < -1e-10) & self._nonnegative
        self._domain_cells[2] = value > self._upper
        if self.torch is None:
            self._domain_rows.fill(0)
            for index in range(3):
                np.logical_or.at(self._domain_rows[index], self._row_index, self._domain_cells[index])
            flags = self._domain_rows
            masses = [bool(np.any(np.abs(sum(self._values[self._slices[pop, 'RGC_expected_history',
                                                   'history_probability_' + str(i)]]
                                             for i in range(8)) - 1.) > 1e-12)) for pop in RGC]
        else:
            self._domain_rows.zero_()
            self._domain_rows.scatter_reduce_(1, self._row_index.expand(3, -1),
                                              self._domain_cells, reduce='amax', include_self=True)
            masses = [((sum(self._values[self._slices[pop, 'RGC_expected_history', 'history_probability_' + str(i)]]
                            for i in range(8)) - 1.).abs() > 1e-12).any() for pop in RGC]
            combined = self.torch.cat((self._domain_rows.flatten(),
                                       self.torch.stack(masses).to(dtype=self._domain_rows.dtype)))
            host = combined.cpu().numpy()
            flags = host[:3 * len(self.rows)].reshape(3, len(self.rows))
            masses = host[3 * len(self.rows):]
        failures = []
        for index, key in enumerate(self.rows):
            if flags[0, index]:
                failures.append(_name(key))
            if flags[1, index]:
                failures.append(_name(key) + ':negative_domain')
            if flags[2, index]:
                failures.append(_name(key) + ':above_scale')
        failures.extend(pop + ':history_probability_mass' for pop, failed in zip(RGC, masses) if failed)
        return failures

    def _update_statistics(self):
        lo, hi, rhs, defect = self._statistics
        if not self.count:
            self._copy(lo, self._values)
            self._copy(hi, self._values)
            self._bind_rows()
        if self.torch is None:
            np.minimum(lo, self._values, out=lo)
            np.maximum(hi, self._values, out=hi)
            np.abs(self._rhs_values, out=self._work)
            np.maximum(rhs, self._work, out=rhs)
            np.subtract(self._transition_values, self._values, out=self._work)
            np.abs(self._work, out=self._work)
            np.divide(self._work, self._periods, out=self._work)
            np.maximum(defect, self._work, out=defect)
        else:
            self.torch.minimum(lo, self._values, out=lo)
            self.torch.maximum(hi, self._values, out=hi)
            self.torch.abs(self._rhs_values, out=self._work)
            self.torch.maximum(rhs, self._work, out=rhs)
            self.torch.sub(self._transition_values, self._values, out=self._work)
            self._work.abs_().div_(self._periods)
            self.torch.maximum(defect, self._work, out=defect)

    def payload(self):
        result = super().payload()
        result['stamps'] = [dict(stamp) for stamp in result['stamps']]
        return result

    @classmethod
    def from_payload(cls, assembly, initial_states, payload, *, parameters, torch=None, device=None,
                     allow_host_inputs=False):
        checked = GroupTimeMonitor.from_payload(assembly, initial_states, payload,
                                               parameters=parameters)
        result = cls(assembly, initial_states, parameters=parameters, torch=torch, device=device,
                     allow_host_inputs=allow_host_inputs)
        for key, row in checked.rows.items():
            for index, field in enumerate(('lo', 'hi', 'rhs', 'defect')):
                value = row[field]
                if torch is not None:
                    value = torch.as_tensor(value, dtype=torch.float64, device=result.device)
                result._copy(result._statistics[index, result._slices[key]], value)
        result._bind_rows()
        result.count = checked.count
        result.stamps = [dict(stamp) for stamp in checked.stamps]
        result.domain_failures = checked.domain_failures
        return result
