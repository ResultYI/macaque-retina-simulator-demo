"""Selected ordinary SimulationResult recording for the resident candidate.

Only requested device rows are exported at recording times. Event/history
updates remain owned by the runner; this object never draws RNG or advances it.
This module is an unvalidated implementation, not a backend acceptance result.
"""
from dataclasses import asdict
import numpy as np

from .contracts import ContractError, Quantity, SimulationResult, canonical_hash
from .network import execution_metadata
from .provenance import CoverageReport
from .recording import RecordingPlan


class ResidentRecording:
    def __init__(self, runner, state, stimulus, plan, times, stop):
        if type(plan) is not RecordingPlan:
            raise ContractError("Resident recording requires an explicit RecordingPlan")
        self.runner, self.state, self.stimulus, self.plan = runner, state, stimulus, plan
        self.times, self.start, self.stop = times, state.time_seconds, float(stop)
        self.empty = plan.ports == () and plan.states == () and plan.cell_ids == ()
        self.rows, self.declarations, self.record_ids = {}, {}, {}
        self.recorded = []
        self.events = {cid: [] for cid in state.spike_states}
        # Match the original allocation guard before allocating trace rows.
        scalar_count = sum(len(plan.indices(s.cell_ids)) for s in runner.model.state_specs
                           if plan.states is None or s.state_name in plan.states)
        scalar_count += sum(len(plan.indices(p.cell_ids))
                            for cid, c in runner.compiled.components.items()
                            for name, p in c.outputs.items()
                            if plan.ports is None or cid + '.' + name in plan.ports)
        extra = sum(len(plan.indices(c.outputs['Vh'].cell_ids))
                    for c in runner.compiled.components.values()
                    if hasattr(c.native_model, 'observation_interval_seconds') and
                    (plan.ports is None or c.component_id + '.observed_Vh' in plan.ports))
        spike_cells = sum(len(plan.indices(c.state_specs[0].cell_ids))
                          for c in runner.compiled.components.values()
                          if hasattr(c.native_model, 'sample_spikes'))
        event_bound = (spike_cells * (int(np.ceil((stop-self.start)/runner.history_dt))+2)
                       if runner.history_dt else 0)
        metadata = 1024 * (len(runner.compiled.components) + sum(
            len(plan.indices(p.cells.cell_id)) for p in runner.compiled.populations.values()))
        self.estimate = (8*len(times)*(scalar_count+extra+spike_cells+1) +
                         128*event_bound + 16*spike_cells + metadata)
        if self.estimate > plan.max_bytes:
            raise ContractError(
                f'Recording requires at least {self.estimate} bytes; select cells/windows or stream, never thin anatomy')

    def _selected(self, values, indices):
        if isinstance(values, self.runner.torch.Tensor):
            selected = self.runner.torch.as_tensor(indices, dtype=self.runner.torch.int64,
                                                   device=values.device)
            return self.runner.torch.index_select(values, 0, selected).detach().cpu().numpy().copy()
        return np.asarray(values)[indices].copy()

    def flush_events(self, t):
        for cid, aux in self.state.spike_states.items():
            ids = self.runner.compiled.components[cid].state_specs[0].cell_ids
            for event in aux['pending_events']:
                if (self.start < event['time_seconds'] <= t and
                        (self.plan.cell_ids is None or ids[event['cell_index']] in self.plan.cell_ids)):
                    self.events[cid].append(dict(event, cell_id=ids[event['cell_index']]))
            aux['pending_events'] = [e for e in aux['pending_events'] if e['time_seconds'] > t]

    def record(self, t, y):
        if not self.empty:
            runner, plan = self.runner, self.plan
            tin = min(t, np.nextafter(self.stop, self.start)) if self.stimulus.interpolation == 'zero_order_hold' else t
            evidence = runner.evaluate(t, y, self.stimulus.at(tin))
            values = {}

            def add(key, data, ids, unit, meaning, tag):
                indices = plan.indices(ids)
                if len(indices):
                    values[key] = Quantity(self._selected(data, indices), unit, meaning, tag)
                    self.record_ids[key] = [ids[i] for i in indices]

            offset = 0
            for spec in runner.model.state_specs:
                end = offset + spec.shape[0]
                if plan.states is None or spec.state_name in plan.states:
                    add('state:' + spec.state_name, y[offset:end], spec.cell_ids,
                        spec.unit, spec.meaning, runner.model.evidence_tag)
                offset = end
            for cid, component in runner.compiled.components.items():
                for port, declaration in component.outputs.items():
                    key = cid + '.' + port
                    if plan.ports is None or key in plan.ports:
                        if cid == runner.partition.cid:
                            data = evidence.native_outputs[port]
                        else:
                            data = evidence.outputs[(cid, port)].values
                        add(key, data, declaration.cell_ids, declaration.unit,
                            declaration.meaning, declaration.evidence_tag)
            for cid, delay in self.state.delay_states.items():
                key = cid + '.observed_Vh'
                if plan.ports is None or key in plan.ports:
                    declaration = runner.compiled.components[cid].outputs['Vh']
                    add(key, delay.at(t), declaration.cell_ids, delay.unit,
                        delay.meaning, delay.evidence_tag)
            for cid, aux in self.state.spike_states.items():
                key = cid + '.conditional_rate'
                component = runner.compiled.components[cid]
                if plan.ports is None or key in plan.ports:
                    add(key, aux['conditional_rate_Hz'], component.state_specs[0].cell_ids,
                        'Hz', 'conditional_spike_intensity', component.evidence_tag)
            for key, q in values.items():
                self.rows.setdefault(key, []).append(q.values)
                self.declarations[key] = (q.unit, q.meaning, q.evidence_tag)
        self.recorded.append(float(t))

    def record_requested(self, t, y):
        index = len(self.recorded)
        if index < len(self.times) and abs(t-self.times[index]) <= 1e-12:
            self.record(float(self.times[index]), y)

    def finish(self, config):
        if (len(self.recorded) != len(self.times) or
                not np.allclose(self.recorded, self.times, rtol=0, atol=1e-12)):
            raise ContractError('Recording timestamp coverage lost')
        if self.empty:
            return None
        runner, plan, state = self.runner, self.plan, self.state
        outputs = {key: Quantity(np.stack(rows, axis=1), *self.declarations[key])
                   for key, rows in self.rows.items()}
        for cid, aux in state.spike_states.items():
            component = runner.compiled.components[cid]
            ids = component.state_specs[0].cell_ids
            indices = plan.indices(ids)
            counts = np.bincount([e['cell_index'] for e in self.events[cid]], minlength=len(ids))[indices]
            for suffix, data, unit, meaning in (
                    ('spike_count', counts, 'dimensionless', 'observed_window_spike_count'),
                    ('window_rate', counts/(self.stop-self.start), 'Hz', 'observed_window_firing_rate')):
                key = cid + '.' + suffix
                if len(indices) and (plan.ports is None or key in plan.ports):
                    outputs[key] = Quantity(data, unit, meaning, component.evidence_tag)
                    self.record_ids[key] = [ids[i] for i in indices]
        outputs['time'] = Quantity(self.recorded, 's', 'absolute_time', runner.model.evidence_tag)
        metadata = execution_metadata(runner.compiled, plan.cell_ids)
        metadata.update({
            'recorded_cell_ids': self.record_ids, 'spike_events': self.events,
            'spike_state_metadata': {cid: {
                'conditional_rate_Hz': np.asarray(aux['conditional_rate_Hz'])[
                    plan.indices(runner.compiled.components[cid].state_specs[0].cell_ids)].tolist(),
                'observation_scheme': aux.get('observation_scheme')}
                for cid, aux in state.spike_states.items()},
            'event_count': {cid: len(events) for cid, events in self.events.items()},
            'event_window_seconds': [self.start, self.stop],
            'conditional_rate_is_not_observed_window_rate': True,
            'clock_contract': runner.compiled.clock,
            'link_delay_buffer_count': len(state.link_delay_states),
            'history_interval_seconds': runner.history_dt,
            'observation_interval_seconds': runner.observation_dt,
            'sampling_index': state.sample_index, 'recording_estimate_bytes': self.estimate,
            'observation_delay': 'Source final delay is timestamped readout only; interpolation explicitly linear; not inserted in feedback',
            'global_state_layout': [{'name': s.state_name, 'size': s.shape[0],
                                     'unit': s.unit, 'meaning': s.meaning} for s in runner.model.state_specs],
            'compute_backend': runner.model.compute_metadata,
            'resident_candidate_validation': 'UNVALIDATED',
        })
        return SimulationResult(
            runner.compiled.run_mode, runner.compiled.spec.model_id,
            canonical_hash({cid: c.parameters.source_manifest_hash for cid, c in runner.compiled.components.items()}),
            self.stimulus.input_contract, outputs,
            CoverageReport(True, (), claim_limit='Executable architecture/reference or explicit software fixture; no physiological completion'),
            {'solver': asdict(config), 'history_grid_held_fixed': True,
             'resident_numerical_acceptance': 'UNVALIDATED'},
            'source_native_or_software_integration', metadata)
