"""Bounded-memory, full-primary-field recording for the future foveal demo.

Importing this module does not assemble a model or initialize CUDA. This is an
observation sink, not a source of simulated values or an execution permission.
"""
from pathlib import Path
from time import perf_counter
import json
import numpy as np

from tools.cone_hc_demo_recording import geometry, selected_rows, sha


class ReplayWriter:
    def __init__(self, directory, components, times, *, buffer_bytes=64 * 1024**2):
        self.path = Path(directory)
        self.times = np.asarray(times, dtype=np.float64)
        if (self.times.ndim != 1 or len(self.times) < 2
                or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0)):
            raise ValueError('Exact increasing observation times required')
        self.components = components
        self.ports = {p['key']: p for c in components for p in c['ports'].values()}
        if len(self.ports) != sum(len(c['ports']) for c in components):
            raise ValueError('Duplicate recorded port')
        width = sum(len(p['cell_ids']) for p in self.ports.values())
        if not width or buffer_bytes < width * 8:
            raise ValueError('Recording buffer cannot hold one full observation')
        self.window_samples = min(len(self.times),max(1, int(buffer_bytes // (width * 8))))
        self.estimated_value_bytes = width * 8 * len(self.times)
        self.path.mkdir(parents=False, exist_ok=False)
        (self.path / 'windows').mkdir()
        self.buffer = {key: np.empty((len(port['cell_ids']),self.window_samples),dtype=np.float64)
                       for key,port in self.ports.items()}
        self.baseline, self.extrema, self.relative_peak = {}, {}, {}
        self.files, self.pending_times = {}, []
        self.index = self.window_index = 0
        self.write_seconds = 0.
        self.closed = False

    def append(self, t, values):
        if self.closed or self.index >= len(self.times) or t != self.times[self.index]:
            raise ValueError('Missing, repeated or unexpected actual observation time')
        if set(values) != set(self.ports):
            raise ValueError('Actual observation fields differ')
        # Validate the whole observation before changing the recording cursor.
        checked = {}
        for key, port in self.ports.items():
            a = np.asarray(values[key])
            if (a.dtype != np.float64 or a.shape != (len(port['cell_ids']),)
                    or not np.isfinite(a).all()):
                raise ValueError('Nonfinite or non-FP64 actual observation: ' + key)
            checked[key] = a
        for key, a in checked.items():
            if key not in self.baseline:
                self.baseline[key] = a.copy()
                self.extrema[key] = [float(a.min()), float(a.max())]
                self.relative_peak[key] = 0.
            self.extrema[key][0] = min(self.extrema[key][0], float(a.min()))
            self.extrema[key][1] = max(self.extrema[key][1], float(a.max()))
            self.relative_peak[key] = max(self.relative_peak[key], float(np.max(np.abs(a-self.baseline[key]))))
            self.buffer[key][:,len(self.pending_times)] = a
        self.pending_times.append(float(t))
        self.index += 1
        if len(self.pending_times) >= self.window_samples:
            self.flush()

    def flush(self):
        if not self.pending_times:
            return
        tick = perf_counter()
        relative = 'windows/window_%04d.npz' % self.window_index
        # Geometry/IDs occur once in the final manifest, not in every window.
        metadata = {'ports': {k: {f: p[f] for f in ('unit', 'meaning')} for k, p in self.ports.items()}}
        np.savez_compressed(self.path / relative, time=np.asarray(self.pending_times),
            manifest_json=json.dumps(metadata), **{k: v[:,:len(self.pending_times)] for k, v in self.buffer.items()})
        self.files[relative] = sha(self.path / relative)
        self.pending_times.clear()
        self.window_index += 1
        self.write_seconds += perf_counter()-tick

    def finish(self, manifest, frames, frame_times, source_indices):
        if self.closed or self.index != len(self.times):
            raise ValueError('Incomplete recording cannot be published as a replay')
        frames = np.asarray(frames); frame_times = np.asarray(frame_times, dtype=np.float64)
        source_indices = np.asarray(source_indices)
        if (frames.dtype != np.uint8 or frames.ndim != 3 or frames.shape[1:] != (256, 256)
                or len(frames) != len(frame_times) or len(frames) != len(source_indices)
                or source_indices.dtype.kind not in 'iu' or not len(frames)
                or not np.isfinite(frame_times).all() or np.any(np.diff(frame_times) <= 0)
                or np.any(np.diff(source_indices) != 1)
                or frame_times[0] > self.times[0] or frame_times[-1] < self.times[-1]):
            raise ValueError('Original stimulus must cover the complete segment including its final boundary')
        self.flush()
        for key, port in self.ports.items():
            port['display_limits_absolute'] = self.extrema[key]
            peak = self.relative_peak[key]
            port['display_limits_relative'] = [-peak, peak]
        np.savez_compressed(self.path / 'display_baseline.npz', **self.baseline)
        np.savez_compressed(self.path / 'stimulus.npz', frames=frames,
                            time=frame_times, source_index=source_indices)
        self.files['display_baseline.npz'] = sha(self.path / 'display_baseline.npz')
        result = dict(manifest, schema='retina-platform-1',
            demo_schema='retipath-foveal-dense-replay-1', components=self.components,
            required_layers=['cone', 'HC', 'BC', 'AC', 'RGC'],
            display_reference={'path': 'display_baseline.npz', 'time_seconds': float(self.times[0]),
                               'sha256': self.files['display_baseline.npz']},
            portable_stimulus={'path': 'stimulus.npz', 'sha256': sha(self.path / 'stimulus.npz'),
                'fps': 150., 'time': 'absolute simulation seconds', 'source_indices': source_indices.tolist()},
            files=self.files)
        with (self.path / 'manifest.json').open('x', encoding='utf8') as stream:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        self.closed = True
        return dict(path=str(self.path), samples=self.index, windows=self.window_index,
                    estimated_uncompressed_value_bytes=self.estimated_value_bytes,
                    window_write_seconds=self.write_seconds, sha256=sha(self.path / 'manifest.json'))


class FovealDemoRecorder:
    """All cells for each primary field; fixed spatial samples for auxiliary fields."""
    def __init__(self, runner, times, output):
        from retina_sim.cone_hc_reduced_evaluation import native_registry
        self.runner = runner
        self.sources, components = {}, []
        specs = {s.state_name: s for s in native_registry(runner)}

        def component(cid, layer, pop, declarations, primary):
            coord, positions = geometry(pop)
            auxiliary = selected_rows(positions.values[:, :2])
            ports = {}
            for name, spec, source in declarations:
                if tuple(spec.cell_ids) != tuple(pop.cells.cell_id):
                    raise ValueError('Original port and population order differ: '+cid+'.'+name)
                rows = np.arange(len(pop.cells.cell_id)) if name == primary else auxiliary
                key = cid+'.'+name
                ports[name] = dict(key=key, unit=spec.unit, meaning=spec.meaning, layer=layer,
                    cell_ids=[str(pop.cells.cell_id[i]) for i in rows],
                    cell_types=[str(pop.cells.cell_type[i]) for i in rows],
                    subtypes=[str(pop.cells.subtype[i]) for i in rows],
                    geometry={coord: dict(values=positions.values[rows, :2].tolist(), unit=positions.unit,
                                         frame=positions.coordinate_frame, meaning=positions.meaning)},
                    temporal_kind='sampled', cell_identity_status='recorded',
                    recorded_count=len(rows), population_count=len(pop.cells.cell_id),
                    selection='all actual selected cells' if name == primary else 'fixed spatial auxiliary sample',
                    primary_field=name == primary)
                device_rows = runner.torch.as_tensor(rows, dtype=runner.torch.int64, device=runner.native.device) if source != 'inner' else rows
                self.sources[key] = source, (cid, name), device_rows
            components.append(dict(id=cid, layers=[layer], ports=ports))

        native_id = runner.partition.cid
        native_ports = runner.compiled.components[native_id].outputs
        component(native_id, 'cone', runner.native.model.cones,
            [(n, specs[n], 'native') for n in ('Vis', 'X', 'C', 'g_i', 'Vbar', 'R', 'E')]
            + [(n, native_ports[n], 'native_output') for n in ('q', 'release', 'Ios')], 'Vis')
        component(native_id, 'HC', runner.native.model.hcs,
            [(n, specs[n], 'native') for n in ('Vh', 'w', 'z2')], 'Vh')
        # Distinct component labels; signal keys retain their original identities.
        components[0]['id'], components[1]['id'] = 'cone', 'HC'
        for cid, comp in runner.compiled.components.items():
            layer = 'RGC' if cid in runner.periods and 'RGC' in cid else 'BC' if 'BC' in cid else 'AC' if 'AC' in cid else None
            if layer is None or cid not in runner.compiled.populations:
                continue
            names = [n for n in ('voltage', 'activation', 'inhibition_state', 'effective_release', 'release') if n in comp.outputs]
            preferred = 'inhibition_state' if layer == 'AC' else 'voltage'
            if not names:
                raise ValueError('No existing primary observable for '+cid)
            component(cid, layer, runner.compiled.populations[cid],
                [(n, comp.outputs[n], 'inner') for n in names], preferred if preferred in names else names[0])
        if {c['layers'][0] for c in components} != {'cone', 'HC', 'BC', 'AC', 'RGC'}:
            raise ValueError('All five actual layers required')
        self.writer = ReplayWriter(Path(output)/'desktop_replay', components, times)
        self.native_transfer_seconds = self.record_seconds = 0.

    def record(self, t, point, evidence, state):
        from retina_sim.cone_hc_reduced_evaluation import state_arrays
        tick = perf_counter()
        native, values = state_arrays(self.runner, point), {}
        for key, (source, (cid, name), rows) in self.sources.items():
            if source == 'inner':
                value = np.asarray(evidence.outputs[cid, name].values)[rows]
            else:
                value = native[name] if source == 'native' else evidence.native_outputs[name]
                transfer = perf_counter()
                value = value.index_select(0, rows).detach().cpu().numpy()
                self.native_transfer_seconds += perf_counter()-transfer
            values[key] = value
        self.writer.append(t, values)
        self.record_seconds += perf_counter()-tick
