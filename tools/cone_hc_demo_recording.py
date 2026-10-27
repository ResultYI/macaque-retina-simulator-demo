"""Selected, lossless FP64 observations from the real reduced runner.

This changes recording only. The full graph, solver and monitors still run.
The portable result uses the existing desktop ResultStore format.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

SCHEMA = 'retipath-reduced-demo-1'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def selected_rows(xy, limit=900):
    """Spatially distributed real cells; no response-dependent selection."""
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('Finite original cell coordinates required')
    if len(xy) <= limit:
        return np.arange(len(xy))
    lo, hi = xy.min(0), xy.max(0)
    bins = np.minimum(((xy-lo)/np.maximum(hi-lo, 1e-12)*29).astype(int), 29)
    _, rows = np.unique(bins[:, 0]*30+bins[:, 1], return_index=True)
    return np.sort(rows[:limit])


def geometry(pop):
    if pop.cells.rf_position is not None:
        return 'RF', pop.cells.rf_position
    for name, attr in [('RF', 'receptive_field_positions'), ('OPL', 'pedicle_positions'),
                       ('IPL', 'dendritic_positions'), ('IPL', 'terminal_positions')]:
        value = getattr(pop, attr, None)
        if value is not None:
            return name, value
    raise ValueError('Original population coordinates unavailable')


class DemoRecorder:
    def __init__(self, runner, protocol, output):
        from retina_sim.cone_hc_reduced_evaluation import native_registry
        self.runner, self.protocol = runner, protocol
        self.path = Path(output)/'desktop_replay'
        self.path.mkdir()
        self.rows, self.meta, self.samples = {}, {}, {}
        self.times = []
        self.wanted = {i*.0001 for i in range(64000, 65001, 5)}
        self.wanted.update((protocol['start_seconds'], protocol['stop_seconds']))
        self.components = []
        # Build only selected-ID geometry, avoiding a second full-retina ID map.
        native = runner.native.model
        for layer, pop, names in [('cone', native.cones, ('X','C','Vis','g_i','Vbar','R','E')),
                                  ('HC', native.hcs, ('w','z2','Vh'))]:
            coord, positions = geometry(pop)
            rows = selected_rows(positions.values[:, :2])
            specs = {s.state_name: s for s in native_registry(runner)}
            ports = {}
            for name in names:
                spec = specs[name]
                key = runner.partition.cid+'.'+name
                ports[name] = self.add(key, spec, layer, pop, rows, coord, positions, 'native', name)
            if layer == 'cone':
                for name in ('q','release','Ios'):
                    spec = runner.compiled.components[runner.partition.cid].outputs[name]
                    key = runner.partition.cid+'.'+name
                    ports[name] = self.add(key, spec, layer, pop, rows, coord, positions, 'native_output', name)
            self.components.append(dict(id=layer, layers=[layer], ports=ports))
        for cid, component in runner.compiled.components.items():
            if cid == runner.partition.cid:
                continue
            layer = 'RGC' if cid in runner.periods and 'RGC' in cid else ('BC' if 'BC' in cid else 'AC' if 'AC' in cid else None)
            if not layer or cid not in runner.compiled.populations:
                continue
            pop = runner.compiled.populations[cid]
            coord, positions = geometry(pop)
            rows = selected_rows(positions.values[:, :2])
            ports = {}
            for name in ('voltage','activation','inhibition_state','effective_release','release'):
                if name not in component.outputs:
                    continue
                spec = component.outputs[name]
                if tuple(spec.cell_ids) != tuple(pop.cells.cell_id):
                    raise ValueError('Port and population cell order differ: '+cid+'.'+name)
                key = cid+'.'+name
                ports[name] = self.add(key, spec, layer, pop, rows, coord, positions, 'inner', (cid,name))
            if ports:
                self.components.append(dict(id=cid, layers=[layer], ports=ports))
        if {x['layers'][0] for x in self.components} != {'cone','HC','BC','AC','RGC'}:
            raise ValueError('All five observed layers are required')

    def add(self, key, spec, layer, pop, rows, coord, positions, source, field):
        ids = [str(pop.cells.cell_id[i]) for i in rows]
        if len(spec.cell_ids) != len(pop.cells.cell_id):
            raise ValueError('State and cell dimensions differ: '+key)
        meta = dict(key=key, unit=spec.unit, meaning=spec.meaning, layer=layer,
            cell_ids=ids, cell_types=[str(pop.cells.cell_type[i]) for i in rows],
            subtypes=[str(pop.cells.subtype[i]) for i in rows],
            geometry={coord:dict(values=positions.values[rows,:2].tolist(), unit=positions.unit,
                                frame=positions.coordinate_frame, meaning=positions.meaning)},
            cell_identity_status='recorded', temporal_kind='sampled',
            recorded_count=len(rows), population_count=len(pop.cells.cell_id),
            selection='all cells' if len(rows)==len(pop.cells.cell_id) else 'fixed spatial coverage; original cells only')
        self.meta[key] = meta
        device_rows = self.runner.torch.as_tensor(rows, device=self.runner.native.device, dtype=self.runner.torch.int64) if source.startswith('native') else rows
        self.rows[key] = source, field, device_rows
        self.samples[key] = []
        return meta

    def record(self, t, point, evidence, state):
        if t not in self.wanted or (self.times and t == self.times[-1]):
            return
        from retina_sim.cone_hc_reduced_evaluation import state_arrays
        native = state_arrays(self.runner, point)
        self.times.append(float(t))
        for key, (source, field, rows) in self.rows.items():
            value = native[field] if source=='native' else evidence.native_outputs[field] if source=='native_output' else evidence.outputs[field].values
            value = value.index_select(0, rows).detach().cpu().numpy() if source.startswith('native') else np.asarray(value)[rows]
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite actual demo observation: '+key)
            self.samples[key].append(value.copy())

    def finish(self, outcome, events):
        from tools.vr_teacher_handoff_v1.fit_integration_v1.pilot import Gray8Chunks
        root = Path(__file__).resolve().parents[1]
        protocol = self.protocol
        if set(self.times) != self.wanted or np.any(np.diff(self.times)<=0):
            raise ValueError('Incomplete demo time coverage')
        intake = root/'tools/vr_teacher_handoff_v1/intake_drive_20261023_v1'
        binding = json.loads((intake/'received_asset_binding.json').read_bytes())
        index_path = intake/binding['stimulus_index']['path']
        if sha(index_path) != binding['stimulus_index']['sha256']:
            raise ValueError('Original stimulus index identity differs')
        source = Gray8Chunks(intake, json.loads(index_path.read_bytes()))
        origin = protocol['raw_zero_seconds']
        edges = origin+np.arange(9752,dtype=float)/150
        first = int(np.searchsorted(edges, self.times[0], side='right')-1)
        last = int(np.searchsorted(edges, self.times[-1], side='right')-1)
        indices = np.arange(first, last+1)
        frames = np.stack([source.frame(int(i)) for i in indices])
        np.savez_compressed(self.path/'stimulus.npz', frames=frames,
            time=edges[indices], source_index=indices)
        (self.path/'windows').mkdir()
        arrays = {k:np.stack(v,axis=1) for k,v in self.samples.items()}
        window_meta = dict(ports=self.meta,metadata=dict(spike_events=events))
        window = self.path/'windows/window_0000.npz'
        np.savez_compressed(window, time=np.array(self.times),
            manifest_json=json.dumps(window_meta,ensure_ascii=False), **arrays)
        manifest = dict(schema='retina-platform-1', demo_schema=SCHEMA,
            required_layers=['cone','HC','BC','AC','RGC'], components=self.components,
            input_class='真实150 Hz刺激 · 简化候选100 ms',
            runtime=dict(stimulus_origin_seconds=0.),
            demo=dict(start=self.times[0],stop=self.times[-1], sample_seconds=.0005,
                backend='analytic R/E + w=A z1 + FP64 exponential midpoint / original CPU inner',
                model_identity=self.runner.identity, protocol_sha256=protocol['protocol_sha256'],
                scientific_status=outcome['status'], fitted=False, candidate_default=False,
                simulation_scope=dict(cones=self.runner.native.nc,HC=self.runner.native.nh,
                    components=len(self.runner.compiled.components)),
                events=events, source_frame_range=[first,last], source_index_sha256=sha(index_path),
                original_observation_only_delay_ms=2.82,
                display_note='空间图只显示记录的真实细胞；原完整网络仍参与计算。Δ为相对本段起点的变化，非拟合或归一化模型。'),
            portable_stimulus=dict(path='stimulus.npz',sha256=sha(self.path/'stimulus.npz'),
                time='absolute simulation seconds',fps=150.,source_indices=indices.tolist()),
            files={'windows/window_0000.npz':sha(window)},
            replay=dict(source='简化模型真实GPU计算 · 100 ms回放',parameter_version=self.runner.identity,
                background_status='原合法初态迁移',elapsed_wall_seconds=outcome['measurements']['candidate']['neural_advance_seconds'],
                peak_rss_bytes=outcome['resource']['process_peak_rss_bytes']))
        (self.path/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
        return dict(path=str(self.path),samples=len(self.times),signals=len(arrays),
            stimulus_frames=len(frames),sha256=sha(self.path/'manifest.json'),fitted=False)
