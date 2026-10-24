"""Original full background predicates split across native GPU and inner CPU.

Only the explicitly constructed resident runner supplies torch/device storage.
Every original observation still executes its predicates. Per-cell extrema and
frozen denominators are retained; no instantaneous scalar-summary substitute.
This implementation has only been statically reviewed, not numerically tested.
"""
from pathlib import Path
import hashlib
import numpy as np
from .contracts import ContractError
from .background import NUMERICAL_UNIT_SCALES
from .resident_delay import ResidentTimestampedDelay


BACKGROUND_SOURCE_SHA256 = '776a83f1c62488efa609f0b0a482cf0c48f0bd58f9106b310b67cdc24b9440cf'
UNIT_INTERVAL_MEANINGS = frozenset((
    'excitatory_receptor_gate', 'inhibitory_receptor_gate', 'excitatory_synaptic_gate',
    'release_probability', 'sodium_activation_gate', 'potassium_activation_gate',
    'synaptic_resource_fraction', 'effective_release_activation',
    'effective_inhibitory_activity', 'adaptation_gate'))
NONNEGATIVE_MEANINGS = frozenset((
    'native_illuminance', 'native_precursor', 'native_os_current', 'native_conductance',
    'effective_opsin_activity', 'effective_pde_activity', 'cgmp_concentration',
    'outer_segment_calcium', 'cgmp_synthesis_cache', 'source_scaled_cgmp',
    'source_scaled_calcium', 'source_scaled_cgmp_synthesis', 'receptor_gate',
    'release_adaptation_activity'))


def _check_source():
    actual = hashlib.sha256(Path(__file__).with_name('background.py').read_bytes()).hexdigest()
    if actual != BACKGROUND_SOURCE_SHA256:
        raise ContractError('Unreviewed background monitor source change; resident monitor unavailable')


class ResidentWindowMonitor:
    """Counter/report-compatible implementation of background._WindowMonitor.

    The numerical equivalence contract is enforced by the separate QA gate,
    not by declaring a migration of arithmetic intrinsically equivalent.
    """

    def __init__(self, runner, reference_state, policy, unit_scales=None,
                 explicit_scales=None, absolute_fraction=None):
        _check_source()
        self.runner = runner
        self.torch = runner.torch
        self.policy = policy
        self.native_stop = int(runner.partition.offset)
        self.native_cid = runner.partition.cid
        self.rows = []
        self.missing = []
        self.device = runner.partition.native.A.device
        native = runner.compiled.components[self.native_cid]
        if native.time_contract.kind != 'continuous':
            raise ContractError('Resident monitor requires the continuous native component')
        registry = NUMERICAL_UNIT_SCALES if unit_scales is None else unit_scales
        specs = runner.model.state_specs
        initial_arrays = reference_state.ordered_state_arrays
        if len(specs) != len(initial_arrays):
            raise ContractError('Background reference state registry differs')
        offset = 0
        for spec, initial in zip(specs, initial_arrays):
            sl = slice(offset, offset + spec.shape[0])
            offset = sl.stop
            native_row = sl.stop <= self.native_stop
            if sl.start < self.native_stop < sl.stop:
                raise ContractError('Monitor state row crosses the native partition')
            cid = spec.state_name.split('.', 1)[0]
            if native_row != (cid == self.native_cid):
                raise ContractError('Monitor native-first state layout differs')
            if tuple(initial.shape) != spec.shape:
                raise ContractError('Background reference state shape differs')
            if explicit_scales is not None:
                if spec.state_name not in explicit_scales:
                    continue
                scale = explicit_scales[spec.state_name]
                scale.require(spec.unit, spec.meaning, spec.shape)
                values = scale.values
                atol = absolute_fraction * values
            elif spec.unit in registry:
                entry = registry[spec.unit]
                values = np.maximum(np.abs(initial), entry['scale_floor'])
                atol = entry['atol']
            else:
                self.missing.append({'state': spec.state_name, 'unit': spec.unit})
                continue
            den = np.asarray(atol + policy.relative_tolerance * values)
            if np.any(~np.isfinite(den)) or np.any(den <= 0):
                raise ContractError('Positive numerical state scales required')
            if native_row:
                den = self.torch.as_tensor(den, dtype=self.torch.float64, device=self.device).clone()
            self.rows.append({
                'spec': spec, 'slice': sl, 'den': den, 'native': native_row,
                'local_slice': slice(sl.start - runner.model.slices[cid].start,
                                     sl.stop - runner.model.slices[cid].start),
                'inner_slice': slice(sl.start - self.native_stop, sl.stop - self.native_stop),
            })
        self.nstate = offset
        self.reset()

    def reset(self):
        for row in self.rows:
            zeros = (lambda: self.torch.zeros(row['spec'].shape, dtype=self.torch.float64,
                                             device=self.device)) if row['native'] else (
                         lambda: np.zeros(row['spec'].shape))
            row.update(lo=None, hi=None, rhs=zeros(), defect=zeros())
        self.count = 0
        self.domain_failures = []
        self.coverage_limited = False

    def observe(self, t, y, bundle):
        self.count += 1
        torch = self.torch
        if y.dtype != torch.float64 or tuple(y.shape) != (self.nstate,) or y.device != self.device:
            raise ContractError('Resident background state layout differs')
        if not torch.isfinite(y).all().item():
            raise ContractError('Nonfinite background state; no clipping')
        evidence = self.runner.evaluate(t, y, bundle, sampling=True, require_rhs=True)
        native_rhs = evidence.native_rhs
        inner_rhs = evidence.inner_rhs
        inner_y = evidence.inner_y
        if (tuple(native_rhs.shape) != (self.native_stop,) or native_rhs.dtype != torch.float64
                or native_rhs.device != self.device or inner_rhs.dtype != np.float64
                or inner_rhs.shape != (self.nstate - self.native_stop,)
                or inner_y.dtype != np.float64 or inner_y.shape != inner_rhs.shape):
            raise ContractError('Resident monitor RHS/state evidence layout differs')
        if not torch.isfinite(native_rhs).all().item() or np.any(~np.isfinite(inner_rhs)):
            raise ContractError('Nonfinite original background RHS')
        inputs, outputs = evidence.inputs, evidence.outputs
        gap = [q.values for (cid, port), q in outputs.items() if port == 'gap_current']
        if gap:
            total = sum(float(np.sum(a)) for a in gap)
            magnitude = sum(float(np.sum(np.abs(a))) for a in gap)
            if abs(total) > 1e-16 + 1e-10 * magnitude:
                self.domain_failures.append('reciprocal_gap_current_conservation')
        transitions = {}
        for cid, component in self.runner.compiled.components.items():
            kind = component.time_contract.kind
            if kind in ('causal_history', 'author_discrete'):
                if cid == self.native_cid:
                    raise ContractError('Resident monitor cannot omit a native discrete transition')
                local = evidence.local_blocks(cid)
                if kind == 'causal_history':
                    new = component.native_model.transition(
                        local, inputs[cid], {p: outputs[(cid, p)] for p in component.outputs})
                else:
                    new = component.native_model.transition(local, inputs[cid])
                transitions[cid] = np.concatenate(new)
        for row in self.rows:
            spec = row['spec']
            cid = spec.state_name.split('.', 1)[0]
            if row['native']:
                a = y[row['slice']]
                rhs = native_rhs[row['slice']]
                row['lo'] = a.clone() if row['lo'] is None else torch.minimum(row['lo'], a)
                row['hi'] = a.clone() if row['hi'] is None else torch.maximum(row['hi'], a)
                row['rhs'] = torch.maximum(row['rhs'], rhs.abs())
                any_value = lambda value: bool(value.any().item())
            else:
                a = inner_y[row['inner_slice']]
                rhs = inner_rhs[row['inner_slice']]
                row['lo'] = a.copy() if row['lo'] is None else np.minimum(row['lo'], a)
                row['hi'] = a.copy() if row['hi'] is None else np.maximum(row['hi'], a)
                row['rhs'] = np.maximum(row['rhs'], np.abs(rhs))
                any_value = lambda value: bool(np.any(value))
            if spec.unit == 'dimensionless' and spec.meaning in UNIT_INTERVAL_MEANINGS:
                if any_value((a < -1e-10) | (a > 1 + 1e-10)):
                    self.domain_failures.append(spec.state_name)
            if spec.meaning in NONNEGATIVE_MEANINGS:
                if any_value(a < -1e-10):
                    self.domain_failures.append(spec.state_name + ':negative_domain')
            if spec.meaning == 'release_adaptation_activity':
                control = self.runner.compiled.components[cid].native_model.bc_output_gain_control
                if any_value(a > control.scale + 1e-10):
                    self.domain_failures.append(spec.state_name + ':above_scale')
            if cid in transitions:
                # All discrete/history rows are inner CPU rows, by construction.
                period = self.runner.periods[cid]
                row['defect'] = np.maximum(
                    row['defect'], np.abs(transitions[cid][row['local_slice']] - a) / period)

    def report(self, width):
        result = []
        dominant = None
        worst = 0.
        for row in self.rows:
            spec, den = row['spec'], row['den']
            if row['lo'] is None or row['hi'] is None:
                raise ContractError('Background report requires an original observation')
            norms = [(row['hi'] - row['lo']) / den,
                     width * row['rhs'] / den, width * row['defect'] / den]
            labels = ('excursion', 'original_rhs', 'discrete_one_step_defect')
            if row['native']:
                maxima = [float(self.torch.max(a).item()) if a.numel() else 0. for a in norms]
            else:
                maxima = [float(np.max(a)) if len(a) else 0. for a in norms]
            result.append({
                'state': spec.state_name, 'unit': spec.unit,
                'maximum_observed_excursion_norm': maxima[0],
                'original_rhs_norm': maxima[1], 'discrete_defect_norm': maxima[2],
            })
            for label, array, value in zip(labels, norms, maxima):
                if value > worst:
                    index = int(self.torch.argmax(array).item()) if row['native'] else int(np.argmax(array))
                    worst = value
                    dominant = {'state': spec.state_name,
                                'component': spec.state_name.split('.', 1)[0],
                                'cell_id': spec.cell_ids[index], 'criterion': label, 'norm': value}
        return {'states': result, 'maximum_norm': worst, 'dominant': dominant,
                'observations': self.count, 'domain_failures': sorted(set(self.domain_failures)),
                'passed': worst <= 1 and not self.domain_failures}


def resident_history_consistency(state, runner, policy, unit_scales=None):
    """Original history predicates; device observation queues stay resident."""
    _check_source()
    failures = []
    registry = NUMERICAL_UNIT_SCALES if unit_scales is None else unit_scales
    for cid, aux in state.spike_states.items():
        h = np.asarray(aux['history'])
        counts = aux.get('last_bin_counts')
        if np.any(~np.isfinite(h)) or np.any(h < 0) or np.any(h != np.floor(h)):
            failures.append(cid + ':invalid_spike_history')
        if counts is not None and (np.any(np.asarray(counts) < 0) or np.any(~np.isfinite(counts))):
            failures.append(cid + ':invalid_bin_counts')
        if any(e['time_seconds'] < state.time_seconds - 1e-12 for e in aux['pending_events']):
            failures.append(cid + ':past_pending_event')
        if not aux.get('rng_state'):
            failures.append(cid + ':missing_rng')
    for kind, queues in [('observation', state.delay_states), ('link', state.link_delay_states)]:
        for key, payload in queues.items():
            prefix = kind + ':' + key + ':'
            if isinstance(payload, ResidentTimestampedDelay):
                if kind != 'observation':
                    raise ContractError('Resident observation delay cannot replace a delayed link')
                failures.extend(prefix + item for item in payload.history_failures(state.time_seconds, policy, registry))
                continue
            times = payload.get('history_times', payload.get('times', []))
            if len(times) and (np.any(np.diff(times) < 0) or times[-1] > state.time_seconds + 1e-12):
                failures.append(prefix + 'invalid_timestamp_order')
            values = np.asarray(payload.get('history_values', payload.get('values', [])))
            if np.any(~np.isfinite(values)):
                failures.append(prefix + 'nonfinite_values')
            unit = payload['prehistory']['unit']
            if unit not in registry:
                failures.append(prefix + 'unregistered_queue_scale')
            elif len(values) > 1:
                entry = registry[unit]
                scale = np.maximum(np.abs(payload['prehistory']['values']), entry['scale_floor'])
                if np.any(np.ptp(values, axis=0) > entry['atol'] + policy.relative_tolerance * scale):
                    failures.append(prefix + 'queue_not_constant_within_tolerance')
    for cid, period in runner.periods.items():
        if state.grid_origin_seconds + state.component_tick_indices[cid] * period < state.time_seconds - 1e-12:
            failures.append(cid + ':lost_tick_phase')
    return failures
