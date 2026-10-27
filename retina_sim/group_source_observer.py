"""Complete original monitoring on each snapshot row's current storage device.

Only native cone/HC rows may be tensors in this explicit mixed policy. They
keep CUDA statistics. Already downloaded inner rows keep NumPy
statistics, avoiding a second full state/RHS/transition upload for monitoring.
Every callback performs the original predicates and arithmetic; no value cache
or time/generation shortcut is used.
"""
import math
from time import perf_counter

import numpy as np

from .contracts import ContractError
from .group_time_monitor import GroupTimeMonitor, _name


class GroupSourceObserver(GroupTimeMonitor):
    def __init__(self, assembly, initial_states, *, parameters=None, torch=None,
                 relative_tolerance=.0001, device=None, allow_host_inputs=False):
        if type(allow_host_inputs) is not bool:
            raise ContractError('Explicit boolean host-input permission required')
        super().__init__(assembly, initial_states, parameters=parameters, torch=torch,
                         relative_tolerance=relative_tolerance)
        self.allow_host_inputs = allow_host_inputs
        if torch is None:
            if device not in (None, 'cpu'):
                raise ContractError('A torch owner is required for a non-CPU monitor')
            self.device = None
        else:
            bound = torch.device('cpu' if device is None else device)
            self.device = torch.device('cpu') if bound.type == 'cpu' else torch.device(
                bound.type, torch.cuda.current_device() if bound.index is None else bound.index)
        self._placement = {}
        self.host_transfer_metrics = dict(bytes=0, transfers=0, seconds=0., packing_seconds=0.)
        self.execution_profile = dict(policy='original_arithmetic_on_snapshot_source_storage',
            fully_device_resident=False, input_bytes_uploaded=0, observed_host_bytes=0,
            observed_device_bytes=0, restored_statistic_bytes_uploaded=0)

    def _storage(self, value, key, label):
        shape = (self.rows[key]['spec'].count,)
        if self.torch is not None and isinstance(value, self.torch.Tensor):
            if key[0] not in ('cones', 'HC'):
                raise ContractError('Mixed monitor requires NumPy inner witnesses; use the device observer for device inner states')
            if value.dtype == self.torch.float64 and tuple(value.shape) == shape and value.device == self.device:
                return str(value.device), value.numel()*value.element_size()
        elif isinstance(value, np.ndarray):
            if value.dtype == np.float64 and value.shape == shape and (self.torch is None or self.allow_host_inputs):
                return 'numpy', value.nbytes
        raise ContractError('Original FP64 monitor '+label+' layout/device differs: '+_name(key))

    def observe(self, t, side, states, rhs, *, transitions, ports=None,
                event_generation=0, aux=None, rhs_kind='original_equation'):
        if (not math.isfinite(t) or side not in ('left', 'right') or type(event_generation) is not int
                or event_generation < 0 or rhs_kind != 'original_equation'):
            raise ContractError('Exact stamp/side/generation and original-equation RHS required')
        if set(states) != set(self.specs) or set(rhs) != set(self.specs) or set(transitions) != set(self.transition_keys):
            raise ContractError('Full state, true RHS and pure one-step transition coverage required')
        current = {}
        counts = dict(numpy=0, device=0)
        for key in self.rows:
            placement, size = self._storage(states[key], key, 'state')
            current[key] = placement
            values = [('RHS', rhs[key])]
            if key in transitions:
                values.append(('pure transition', transitions[key]))
            for label, value in values:
                other, length = self._storage(value, key, label)
                if other != placement:
                    raise ContractError('Monitor row state/RHS/transition storage differs: '+_name(key))
                size += length
            if key in self._placement and self._placement[key] != placement:
                raise ContractError('Monitor state changed storage device: '+_name(key))
            counts['numpy' if placement == 'numpy' else 'device'] += size
        # Restore transfers only existing native-row statistics, never full inputs.
        # They remain inside the production monitor phase and are counted.
        for key, placement in current.items():
            row = self.rows[key]
            if placement != 'numpy' and row.get('restored_cpu_statistics'):
                begin = perf_counter()
                for field in ('lo', 'hi', 'rhs', 'defect'):
                    value = row[field]
                    row[field] = self.torch.as_tensor(value, dtype=states[key].dtype, device=states[key].device).clone()
                    if states[key].device.type != 'cpu':
                        self.host_transfer_metrics['bytes'] += value.nbytes
                        self.host_transfer_metrics['transfers'] += 1
                        self.execution_profile['restored_statistic_bytes_uploaded'] += value.nbytes
                self.host_transfer_metrics['seconds'] += perf_counter()-begin
                row['native_tensor'] = True
                row.pop('restored_cpu_statistics')
        super().observe(t, side, states, rhs, transitions=transitions, ports=ports,
            event_generation=event_generation, aux=aux, rhs_kind=rhs_kind)
        self._placement = current
        self.execution_profile['observed_host_bytes'] += counts['numpy']
        self.execution_profile['observed_device_bytes'] += counts['device']

    def observe_snapshot(self, snapshot):
        return self.observe(snapshot['time_seconds'], snapshot['side'], snapshot['states'], snapshot['rhs'],
            transitions=snapshot['transitions'], ports=snapshot['ports'], event_generation=snapshot['event_generation'],
            aux=snapshot['aux'], rhs_kind=snapshot['rhs_kind'])

    def payload(self):
        result = super().payload()
        result['stamps'] = [dict(stamp) for stamp in result['stamps']]
        return result

    @classmethod
    def from_payload(cls, assembly, initial_states, payload, *, parameters, torch=None,
                     device=None, allow_host_inputs=False):
        checked = GroupTimeMonitor.from_payload(assembly, initial_states, payload, parameters=parameters, torch=torch)
        result = cls(assembly, initial_states, parameters=parameters, torch=torch,
                     device=device, allow_host_inputs=allow_host_inputs)
        result.rows = checked.rows
        result.count = checked.count
        result.stamps = [dict(stamp) for stamp in checked.stamps]
        result.domain_failures = list(checked.domain_failures)
        return result
