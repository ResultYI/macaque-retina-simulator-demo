"""Real hybrid-retina execution in a background thread, continuously across media frames."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import numpy as np
from scipy.ndimage import map_coordinates
from .config import RunConfig, AUTHOR_STEP, input_assumptions
from .media import MediaSource

def _replace_atomic(temporary, destination):
    """Windows readers briefly hold non-share-delete handles; keep publication atomic."""
    deadline = time.monotonic() + 2.
    delay = .01
    while True:
        try:
            os.replace(temporary, destination)
            return
        except OSError as error:
            if not isinstance(error, PermissionError) and getattr(error, 'winerror', None) not in (5, 32):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(.1, delay * 2)

def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    # Large source assemblies repeat explicit IDs/units extensively. Compact
    # machine manifests preserve every value while avoiding indentation bloat.
    compact=path.name in ('manifest.json','resolved_profile.json')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=None if compact else 2,
        separators=(',',':') if compact else None,allow_nan=False)+'\n',encoding='utf-8')
    _replace_atomic(temporary, path)

def _atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    _replace_atomic(temporary, path)

def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _record_preparation_metadata(manifest,resolved,report):
    """Actual duration/status supersede the build-time fixed-prerun request."""
    initial=dict(manifest['initialization'])
    initial.update(requested_fixed_prerun_seconds=initial['prerun_seconds'],prerun_seconds=report['elapsed_seconds'],
        adaptation_status=report['status'],inner_initialization_mode='bounded_uniform_background_preparation',
        whole_network_steady_state_claim=False)
    manifest['initialization']=initial
    manifest['input_assumptions']['light_context']=dict(initial)
    manifest['background_initialization']=report
    resolved['light_context']=dict(initial);resolved['background_initialization']=report

def _gaze(config):
    if not config.gaze_csv:
        def fixed(t):
            return np.array([config.gaze_x_deg, config.gaze_y_deg])
        return fixed, {'kind': 'fixed', 'x_deg': config.gaze_x_deg, 'y_deg': config.gaze_y_deg}
    with Path(config.gaze_csv).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'time_seconds', 'x_deg', 'y_deg'} <= set(reader.fieldnames or []):
            raise ValueError('眼位 CSV 需要 time_seconds,x_deg,y_deg 列')
        values = np.array([[float(row[k]) for k in ('time_seconds', 'x_deg', 'y_deg')] for row in reader], dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError('眼位轨迹至少两行有限数值')
    if np.any(np.diff(values[:, 0]) <= 0) or values[0, 0] > 0 or values[-1, 0] < config.duration_seconds:
        raise ValueError('眼位时间必须递增，并覆盖刺激 0 到记录结束；禁止外推')
    def prescribed(t):
        return np.array([np.interp(t, values[:, 0], values[:, k]) for k in (1, 2)])
    return prescribed, {'kind': 'prescribed_csv', 'path': str(Path(config.gaze_csv).resolve()),
                        'sha256': _file_sha(config.gaze_csv), 'columns': ['time_seconds', 'x_deg', 'y_deg'],
                        'interpolation': 'linear', 'time_reference': 'stimulus time, excluding warmup'}

def _port_cells(compiled, ids, owner_masks=None):
    stamp=tuple((pid,id(pop)) for pid,pop in compiled.populations.items())
    cached=getattr(compiled,'_platform_port_lookup',None)
    if cached is None or cached[0]!=stamp:
        lookup={cell_id:(pid,pop,i) for pid,pop in compiled.populations.items() for i,cell_id in enumerate(pop.cells.cell_id)}
        compiled._platform_port_lookup=(stamp,lookup)
    else:lookup=cached[1]
    entries = [lookup.get(cell_id) for cell_id in ids]
    if not all(entry is not None for entry in entries):
        return {'population': None, 'cell_ids': list(ids), 'cell_types': ['contact'] * len(ids),
                'subtypes': ['contact'] * len(ids), 'regions': ['unspecified'] * len(ids), 'geometry': {},'is_owner':[None]*len(ids)}
    fields = {'population': ','.join(dict.fromkeys(e[0] for e in entries)), 'cell_ids': list(ids),
              'cell_types': [e[1].cells.cell_type[e[2]] for e in entries],
              'subtypes': [e[1].cells.subtype[e[2]] for e in entries],
               'regions': [e[1].regions[e[2]] for e in entries], 'geometry': {},
               'is_owner':[bool(owner_masks[e[0]][e[2]]) if owner_masks and e[0] in owner_masks else None for e in entries]}
    selectors = {
        'RF': lambda p: p.receptive_field_positions if p.receptive_field_positions is not None else p.cells.rf_position,
        'Soma': lambda p: p.anatomical_positions if p.anatomical_positions is not None else p.cells.soma_position,
        'OPL': lambda p: next((q for q in (p.pedicle_positions, p.dendritic_positions, p.terminal_positions)
                              if q is not None and q.coordinate_frame == 'OPL'), None),
        'IPL': lambda p: next((q for q in (p.terminal_positions, p.dendritic_positions)
                              if q is not None and q.coordinate_frame == 'IPL'), None),
    }
    for name, select in selectors.items():
        quantities = [select(e[1]) for e in entries]
        if all(q is not None for q in quantities) and len({(q.unit, q.coordinate_frame) for q in quantities}) == 1:
            fields['geometry'][name] = {'values': [q.values[e[2]].tolist() for q, e in zip(quantities, entries)],
                                        'unit': quantities[0].unit, 'frame': quantities[0].coordinate_frame,
                                        'meaning': quantities[0].meaning,
                                        'evidence_tag': quantities[0].evidence_tag}
    return fields

def _components(compiled,owner_masks=None):
    result = []
    for cid, component in compiled.components.items():
        ports = {}
        for name, port in component.outputs.items():
            selected=getattr(compiled,'platform_recording_cell_ids',None)
            ids=port.cell_ids if selected is None else tuple(gid for gid in port.cell_ids if gid in selected)
            ports[name] = {'key': cid + '.' + name, 'unit': port.unit, 'meaning': port.meaning,
                            'evidence_tag': port.evidence_tag, 'temporal_kind': 'sampled', **_port_cells(compiled, ids,owner_masks)}
        main = next(iter(ports.values()), {'population': None, 'cell_ids': [], 'cell_types': [], 'subtypes': [], 'regions': [], 'geometry': {}})
        result.append({'id': cid, 'layers': list(component.layers), 'mechanism': component.mechanism,
                       'profile_id': component.parameters.profile.profile_id, 'ports': ports,
                       **{k: main[k] for k in ('population', 'cell_ids', 'cell_types', 'subtypes', 'regions', 'geometry')}})
    return result

def _runtime_ports(manifest, compiled, state):
    """Declare actual HybridRunner observations absent from static mechanism ports."""
    by_id = {c['id']: c for c in manifest['components']}
    for cid in state.delay_states:
        port = compiled.components[cid].outputs['Vh']
        selected=getattr(compiled,'platform_recording_cell_ids',None)
        ids=port.cell_ids if selected is None else tuple(gid for gid in port.cell_ids if gid in selected)
        by_id[cid]['ports']['observed_Vh'] = {
            'key': cid + '.observed_Vh', 'unit': port.unit, 'meaning': port.meaning,
            'evidence_tag': port.evidence_tag, 'temporal_kind': 'sampled',
            **_port_cells(compiled, ids,manifest.get('population_owner_mask'))}
    for cid in state.spike_states:
        component = compiled.components[cid]
        cells = _port_cells(compiled, component.state_specs[0].cell_ids,manifest.get('population_owner_mask'))
        for name, unit, meaning, temporal_kind in (
            ('conditional_rate', 'Hz', 'conditional_spike_intensity', 'sampled'),
            ('spike_count', 'dimensionless', 'observed_window_spike_count', 'window_aggregate'),
            ('window_rate', 'Hz', 'observed_window_firing_rate', 'window_aggregate')):
            by_id[cid]['ports'][name] = {'key': cid + '.' + name, 'unit': unit, 'meaning': meaning,
                'evidence_tag': component.evidence_tag, 'temporal_kind': temporal_kind, **cells}

class _MediaChannels:
    """Each window caches only sampled input rates; full video is never resident."""
    def __init__(self, config, compiled, media, gaze, stimulus_origin_seconds=None):
        self.config, self.compiled, self.media, self.gaze = config, compiled, media, gaze
        self.stimulus_origin_seconds=config.warmup_seconds if stimulus_origin_seconds is None else float(stimulus_origin_seconds)
        # Use the same author-tick endpoint as the execution windows. Adding
        # origin + duration can round one ULP below tick * AUTHOR_STEP, leaving
        # the last otherwise valid window outside the declared input coverage.
        self.total = int(round((self.stimulus_origin_seconds + config.duration_seconds) / AUTHOR_STEP)) * AUTHOR_STEP
        self.assumptions = input_assumptions(config)
        if config.background_initialization_mode=='bounded_convergence':
            self.assumptions['light_context'].update(requested_fixed_prerun_seconds=config.warmup_seconds,
                prerun_seconds=self.stimulus_origin_seconds,adaptation_status='see_background_initialization_report')
        self.assumptions['actual_input_population_counts']={pid:len(pop.cells.cell_id) for pid,pop in compiled.populations.items()
            if pid in ('cones','peripheral_physical_cones','rod')}
        self.declarations = [
            ('native_light', 'cones', config.native_background_td, 'td', 'native_retinal_illuminance', False),
            ('physical_cone_light', 'peripheral_physical_cones', config.cone_background_rstar_s, 'R*/s', 'photoisomerization_rate', True),
            ('rod_light', 'rod', config.rod_background_rstar_s, 'R*/s', 'photoisomerization_rate', True)]
        self.declarations=[row for row in self.declarations if row[1] in compiled.populations]
        if config.rgb_adapter=='iset_lcd_apple_mw_v1':self.declarations=[r for r in self.declarations if r[0]=='rod_light']
        self.last_input_summary = {}

    def window(self, start, stop):
        from retina_sim.contracts import Quantity
        from retina_sim.spatial import InputChannel
        first, last = int(round(start / AUTHOR_STEP)), int(round(stop / AUTHOR_STEP))
        if start < 0. or stop <= start or stop > self.total + 1e-12:
            raise ValueError('Media window outside declared author-tick duration')
        cell_count = sum(len(self.compiled.populations[pid].cells.cell_id) for _, pid, *_ in self.declarations)
        estimate_bytes = (last - first + 1) * (cell_count * 40 + 40)
        if estimate_bytes > 128 * 1024 * 1024:
            raise ValueError('输入窗口缓存超过 128 MiB；请减小窗口长度，不会自动删减细胞')
        grid = np.arange(first, last + 1, dtype=np.int64).astype(float) * AUTHOR_STEP
        grid[0], grid[-1] = start, stop
        stimulus = np.maximum(0., grid - self.stimulus_origin_seconds)
        pose = np.array([self.gaze(float(t)) for t in stimulus])
        matrices, outside_counts = {}, {}
        weights = self.assumptions['weights']
        width, height = self.media.info['width'], self.media.info['height']
        ppd = width / self.config.field_width_deg
        coords, matrix_weights = {}, {}
        for name, pid, base, unit, meaning, discrete in self.declarations:
            pop = self.compiled.populations[pid]
            rf = pop.cells.rf_position.values / .223
            coords[name] = np.stack((width / 2 + ppd * (rf[None, :, 0] + pose[:, None, 0]),
                                    height / 2 + ppd * (rf[None, :, 1] + pose[:, None, 1])), axis=-1)
            kinds = ['rod'] * len(rf) if pid == 'rod' else pop.cells.subtype
            matrix_weights[name] = np.array([weights[k] for k in kinds])
            matrices[name] = np.empty((len(grid), len(rf)), dtype=float)
            pixels = coords[name]
            outside_counts[name] = float(((pixels[..., 0] < 0) | (pixels[..., 0] > width - 1) |
                                          (pixels[..., 1] < 0) | (pixels[..., 1] > height - 1)).mean())
        previous_index, linear = None, None
        timestamp_records = []
        for row, t in enumerate(stimulus):
            if (grid[row] < self.stimulus_origin_seconds or stop<=self.stimulus_origin_seconds) and self.config.adaptation_policy == 'uniform_declared_background':
                for name, pid, base, unit, meaning, discrete in self.declarations:
                    matrices[name][row] = base
                continue
            media_time = self.config.start_seconds + float(t)
            frame_index = 0 if self.media.info['kind'] == 'image' else min(int(media_time * self.media.info['fps'] + 1e-8), self.media.info['frame_count'] - 1)
            if frame_index != previous_index:
                rgb = self.media.frame_at(media_time).astype(float) / 255.
                linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
                previous_index = frame_index
                timestamp_records.append({'selected_frame_index': frame_index, 'requested_media_seconds': media_time,
                                          'decoded_timestamp_seconds': self.media.last_timestamp_seconds})
            for name, pid, base, unit, meaning, discrete in self.declarations:
                pixels = coords[name][row]
                sampled = np.stack([map_coordinates(linear[:, :, k], [pixels[:, 1], pixels[:, 0]],
                                        order=1, mode='constant', cval=self.config.reference_linear_value if self.config.rgb_adapter in ('contrast_reference_v1','iset_lcd_apple_mw_v1') else .5, prefilter=False) for k in range(3)], axis=-1)
                proxy=(sampled * matrix_weights[name]).sum(axis=-1)
                matrices[name][row] = (base*proxy/self.config.reference_linear_value if self.config.rgb_adapter in ('contrast_reference_v1','iset_lcd_apple_mw_v1') else base*(.6+.8*proxy))
        channels = {}
        for name, pid, base, unit, meaning, discrete in self.declarations:
            values = matrices[name]
            evidence='project_approximation' if self.config.execution_profile in ('atlas_midget_research_v1','atlas_population_revision_v1','atlas_isetbio_geometry_v1') else 'synthetic_fixture'
            def evaluator(t, values=values, discrete=discrete, unit=unit, meaning=meaning,base=base,evidence=evidence):
                # Author transitions at the first media tick consume the preceding
                # interval. Retain its real background even when this window's
                # cache starts at stimulus onset; never clamp it to the new frame.
                if t<self.stimulus_origin_seconds-1e-12 and self.config.adaptation_policy=='uniform_declared_background':
                    return Quantity(np.full(values.shape[1],base),unit,meaning,evidence)
                position = min(max(float(t) / AUTHOR_STEP - first, 0.), len(values) - 1)
                i = min(int(np.floor(position + (1e-9 if discrete else 0.))), len(values) - 1)
                if discrete:
                    result = values[i]
                else:
                    j, fraction = min(i + 1, len(values) - 1), position - i
                    result = (1 - fraction) * values[i] + fraction * values[j]
                return Quantity(result, unit, meaning,evidence)
            contract = dict(self.assumptions, population_id=pid, grid_step_seconds=AUTHOR_STEP,
                            interval_count=int(round(self.total / AUTHOR_STEP)),
                             stimulus_origin_seconds=self.stimulus_origin_seconds,
                            window_cache_seconds=[start, stop])
            channels[name] = InputChannel(0., self.total, evaluator, unit, meaning,
                                         'zero_order_hold' if discrete else 'linear',evidence, contract,
                                         ())  # Existing author/observation clocks already visit the 100-us grid.
        self.last_input_summary = {'outside_fraction': outside_counts, 'frame_timestamps': timestamp_records,
                                   'media_timestamp_fallback_used': self.media.info.get('timestamp_fallback_used', False)}
        return channels

class SimulationSession:
    def __init__(self, config, project_root, output_root=None):
        self.project_root = Path(project_root).resolve()
        self.config = (RunConfig.project_default(self.project_root) if config is None else
            RunConfig.from_dict(config.to_dict() if isinstance(config, RunConfig) else config))
        self.output_root = Path(output_root) if output_root else self.project_root / 'reports' / 'platform_runs'
        self.events = queue.Queue()
        self.path = None
        self.status = 'ready'
        self._thread = None
        self._condition = threading.Condition()
        self._pause = False
        self._stop = False
        self._sim_time = 0.
        self._windows = 0
        self._stimulus_origin=self.config.warmup_seconds
        self._total=self._stimulus_origin+self.config.duration_seconds

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self._thread is not None:
            raise RuntimeError('每个运行会话只能启动一次；重复运行建立新会话')
        self.config.validate()
        self._thread = threading.Thread(target=self._run, name='retina-platform-worker', daemon=True)
        self._thread.start()
        return self

    def _event(self, kind, status, message='', **extra):
        self.status = status
        event = {'kind': kind, 'status': status, 'time': self._sim_time,
                 'total': self._total,
                 'path': str(self.path) if self.path else '', 'windows': self._windows,
                 'stimulus_time': max(0., self._sim_time - self._stimulus_origin) + self.config.start_seconds,
                 'message': message, **extra}
        self.events.put(event)
        if self.path and threading.current_thread() is self._thread:
            _atomic_json(self.path / 'status.json', event)
            with (self.path / 'progress.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')

    def pause(self):
        with self._condition:
            if self.is_alive():
                self._pause = True
                self._event('status', 'pause_requested')

    def resume(self):
        with self._condition:
            self._pause = False
            self._condition.notify_all()

    def stop(self):
        with self._condition:
            if self.is_alive():
                self._stop = True
                self._pause = False
                self._event('status', 'stopping')
                self._condition.notify_all()

    def _boundary(self):
        with self._condition:
            if self._stop:
                return False
            if self._pause:
                self._event('status', 'paused')
                while self._pause and not self._stop:
                    self._condition.wait(.25)
                if self._stop:
                    return False
                self._event('status', 'running')
            return True

    def _run(self):
        media, runner, state = None, None, None
        wall_start = time.perf_counter()
        try:
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
            from examples.full_target_network import full_target_network
            from retina_sim.contracts import SolverConfig,canonical_hash
            from retina_sim.execution_backends import make_runner
            from retina_sim.recording import RecordingPlan, NPZWindowRecorder
            from tools.run_full_network_fixture import serializable
            media = MediaSource(self.config.media_path)
            self.config.validate(media)
            gaze, gaze_metadata = _gaze(self.config)
            total = (round(self.config.warmup_seconds / AUTHOR_STEP) + round(self.config.duration_seconds / AUTHOR_STEP)) * AUTHOR_STEP
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]
            self.path = self.output_root / stamp
            self.path.mkdir(parents=True, exist_ok=False)
            _atomic_json(self.path / 'config.json', self.config.to_dict())
            self._event('status', 'building')
            if self.config.execution_profile in ('atlas_population_revision_v1','atlas_isetbio_geometry_v1'):
                from retina_sim.atlas_population_network import atlas_population_network
                compiled,_,audit=atlas_population_network(self.project_root,duration=total,light_context=self.config.light_context(),
                    owner_bounds_mm=self.config.owner_bounds(),halo_mm=self.config.halo_mm,
                    max_spatial_cells=int(self.config.max_spatial_cells),seed=int(self.config.seed),execution_profile=self.config.execution_profile,
                    configuration_path=self.config.circuit_configuration_path or None)
            elif self.config.execution_profile=='atlas_midget_research_v1':
                from retina_sim.atlas_runtime import atlas_typed_candidate_network
                compiled,_,audit=atlas_typed_candidate_network(self.project_root,duration=total,light_context=self.config.light_context(),
                    owner_bounds_mm=self.config.owner_bounds(),halo_mm=self.config.halo_mm,
                    max_spatial_cells=int(self.config.max_spatial_cells),seed=int(self.config.seed))
            else:
                compiled, _, audit = full_target_network(self.project_root, effective_cones=int(self.config.effective_cones),
                effective_rods=int(self.config.effective_rods), seed=int(self.config.seed), duration=total,
                adaptation_seconds=self.config.warmup_seconds, light_context=self.config.light_context(),
                runtime_profile=self.config.parameter_profile, execution_profile=self.config.execution_profile,
                owner_bounds_mm=self.config.owner_bounds(),halo_mm=self.config.halo_mm,max_spatial_cells=int(self.config.max_spatial_cells),
                hc_geometry_profile=self.config.hc_geometry_profile,hc_geometry_path=self.config.hc_geometry_path,
                hc_h1_radius_mm=self.config.hc_h1_radius_mm,hc_h2_radius_mm=self.config.hc_h2_radius_mm,
                hc_arbor_profile=self.config.hc_arbor_profile,ipl_geometry_profile=self.config.ipl_geometry_profile,
                requested_owner_outputs=(self.config.requested_owner_output,))
            if self.config.requested_owner_output in audit.get('atlas_runtime',{}).get('empty_owner_outputs',{}):
                raise ValueError('该ROI的目标owner为EMPTY_SAMPLE；未补点或改变密度')
            self._event('status','building',message='实体 %d · 边 %d · 估算 %.1f MiB' % (
                audit['resource_estimate']['explicit_cells'],audit['resource_estimate']['estimated_edges'],
                audit['resource_estimate']['estimated_compile_bytes']/1024**2))
            missing = [layer for layer in compiled.spec.required_mechanisms
                       if not any(layer in c.layers for c in compiled.components.values())]
            if missing:
                raise RuntimeError('缺少真实核心机制: ' + ','.join(missing))
            from .iset_frontend import MixedReferenceFrontend
            self._event('status','building',message='ISET 混合参考输入准备')
            frontend=MixedReferenceFrontend
            if hasattr(compiled,'global_disk_circuit'):
                from .disk_frontend import DiskMixedReferenceFrontend
                frontend=DiskMixedReferenceFrontend
            channels_source=frontend(self.config,compiled,media,gaze,self.project_root,
                input_path=self.path/'input.npz',
                on_status=lambda message:self._event('status','building',message=message),on_boundary=self._boundary)
            audit['light_context']=self.config.light_context().manifest(self.config.warmup_seconds)
            _atomic_json(self.path/'config.json',self.config.to_dict())
            global_export=None
            if hasattr(compiled,'global_disk_circuit'):
                from .disk_run_audit import prepare_global_audit
                global_export=prepare_global_audit(compiled,audit,self.path)
            manifest = {'schema': 'retina-platform-1', 'config': self.config.to_dict(), 'media': dict(media.info),
                        'platform_configuration':json.loads((self.project_root/'configurations/retina_platform_v1.json').read_bytes()),
                        'media_sha256': _file_sha(self.config.media_path), 'gaze': gaze_metadata,
                        'input_assumptions': input_assumptions(self.config), 'assembly_hash': compiled.assembly_hash,
                        'required_layers': list(compiled.spec.required_mechanisms), 'components': _components(compiled,audit['population_owner_mask']),
                        'parameter_evidence':global_export['parameter_evidence'] if global_export is not None else {cid: serializable(c.parameters.records) for cid, c in compiled.components.items()},
                        'source_records': serializable(tuple(audit['registry'].sources.values())),
                        'source_spec':global_export['source_spec'] if global_export is not None else serializable(compiled.spec), 'scientific_validation': 'NOT_RUN',
                        'recording': {'ports': 'actual ports on declared global recording cohort' if global_export is not None else 'all configured actual output ports', 'states': [],
                                      'sample_seconds': self.config.sample_seconds, 'window_seconds': self.config.window_seconds},
                        'core_hashes': {str(p.relative_to(self.project_root)): _file_sha(p) for p in (self.project_root / 'retina_sim').glob('*.py')},
                        'initialization': audit['light_context'], 'runtime_parameter_profile': audit['runtime_profile'],
                        'execution_profile': audit['execution_profile'],'spatial_profile':audit['spatial_profile'],
                        'connectivity_support':audit['connectivity_support'],'cone_route':audit['cone_route'],
                        'population_owner_mask':audit['population_owner_mask'],'resource_estimate':audit['resource_estimate'],
                        'rod_inner_geometry':audit['rod_inner_geometry'],'IPL_geometry':audit['IPL_geometry'],
                        'population_records':audit['population_records'],
                        'density_integrals':audit['density_integrals'],
                        'regional_support':audit['regional_support'],'physical_reference_applicability':audit['physical_reference_applicability'],
                        'glycine_source_condition':audit['glycine_source_condition'],
                        'runtime': {'started_utc': datetime.now(timezone.utc).isoformat(), 'state_reset_between_windows': False}}
            if 'atlas_runtime' in audit:manifest['atlas_runtime']=audit['atlas_runtime']
            if 'artifact_storage' in audit:manifest['artifact_storage']=audit['artifact_storage']
            manifest['input_frontend']=channels_source.summary
            _atomic_json(self.path / 'manifest.json', manifest)
            _atomic_json(self.path/'hc_geometry.json',audit['spatial_profile'].get('HC_geometry',{'profile_id':'engineering_preview'}))
            _atomic_json(self.path/'ipl_geometry.json',audit['IPL_geometry'])
            runner = make_runner(compiled,self.config.neural_integration_backend)
            manifest['numerical_execution']=runner.model.compute_metadata
            _atomic_json(self.path/'manifest.json',manifest)
            state = runner.initialize(SolverConfig(**audit['resolved_execution_profile']['solver']))
            state.input_position={'background_context_hash':self.config.light_context().background_context_hash,
                'stimulus_origin_seconds':self.config.warmup_seconds}
            _runtime_ports(manifest, compiled, state)
            _atomic_json(self.path / 'manifest.json', manifest)
            initial = runner.checkpoint(state)
            _atomic_text(self.path / 'initial_checkpoint.json', initial)
            _atomic_text(self.path / 'latest_checkpoint.json', initial)
            resolved={'schema_version':'1.0','execution':audit['resolved_execution_profile'],
                'light_context':audit['light_context'],'configuration':self.config.to_dict(),
                'parameter_defaults':audit['registry'].runtime_profile.data,
                'resolved_parameters':manifest['parameter_evidence'],'source_records':manifest['source_records'],
                'geometry':audit['spatial_profile'],'rod_inner_geometry':audit['rod_inner_geometry'],'IPL_geometry':audit['IPL_geometry'],
                'resource_estimate':audit['resource_estimate'],'ports':manifest['components'],
                'population_records':audit['population_records'],'density_integrals':audit['density_integrals'],
                'regional_support':audit['regional_support'],'physical_reference_applicability':audit['physical_reference_applicability'],
                'glycine_source_condition':audit['glycine_source_condition']}
            if 'atlas_runtime' in audit:resolved['atlas_runtime']=audit['atlas_runtime']
            _atomic_json(self.path/'resolved_profile.json',resolved)
            manifest['resolved_profile_sha256']=_file_sha(self.path/'resolved_profile.json')
            manifest['initialization_diagnostics']=[]
            preparation_report=None
            if self.config.background_initialization_mode=='bounded_convergence':
                from retina_sim.background import constant_background_channels,prepare_background
                policy=self.config.background_policy()
                self._stimulus_origin=policy.max_sim_seconds;self._total=policy.max_sim_seconds+self.config.duration_seconds
                self._event('status','running',phase='background_preparation')
                background=constant_background_channels(compiled,self.config.light_context(),policy.max_sim_seconds+self.config.duration_seconds)
                from retina_sim.background_checkpoint import prepare_with_checkpoint,MemoryGuard,BackgroundCheckpointStore
                guard=MemoryGuard(int(self.config.background_memory_limit_mb*1024**2),int(self.config.background_min_available_mb*1024**2))
                guard.check()
                cache_key=canonical_hash(BackgroundCheckpointStore(self.project_root,runner,state,self.config.light_context(),policy).binding)
                automatic_cache=self.project_root/'reports/iset_background_cache'/cache_key/'background_checkpoint.json'
                resume=Path(self.config.background_checkpoint_path) if self.config.background_checkpoint_path else automatic_cache
                if resume is not None and not resume.is_absolute():resume=self.project_root/resume
                if not self.config.background_checkpoint_path and not resume.is_file():resume=None
                def preparation_progress(current,row):
                    nonlocal state
                    state=current;self._sim_time=current.time_seconds
                    _atomic_text(self.path/'latest_checkpoint.json',runner.checkpoint(current))
                    _atomic_json(self.path/'background_progress.json',dict(row,absolute_background_time_seconds=current.time_seconds,
                        complete_checkpoint_file='background_checkpoint.json',monitoring_criteria_changed=False))
                    if row['phase']=='background_window':
                        with (self.path/'background_windows.jsonl').open('a',encoding='utf8') as stream:
                            stream.write(json.dumps(row['last_window'],ensure_ascii=False,allow_nan=False)+'\n')
                    self._event('progress','running',phase='background_preparation',
                        preparation_wall_seconds=row['wall_seconds'],solver_metrics=dict(runner.integration_metrics))
                state,preparation_report,_=prepare_with_checkpoint(self.project_root,runner,state,self.config.light_context(),policy,
                    self.path/'background_checkpoint.json',resume_path=resume,guard=guard,
                    on_progress=preparation_progress,on_boundary=self._boundary,
                    checkpoint_interval_seconds=self.config.background_checkpoint_interval_seconds)
                self._sim_time=state.time_seconds;self._stimulus_origin=state.time_seconds
                total=self._total=state.time_seconds+self.config.duration_seconds
                _record_preparation_metadata(manifest,resolved,preparation_report)
                _atomic_json(self.path/'resolved_profile.json',resolved)
                manifest['resolved_profile_sha256']=_file_sha(self.path/'resolved_profile.json')
                manifest['initialization_diagnostics']=preparation_report['windows']
                manifest['runtime']['stimulus_origin_seconds']=state.time_seconds
                manifest['runtime']['grid_origin_seconds']=state.grid_origin_seconds
                _atomic_json(self.path/'background_initialization.json',preparation_report)
                _atomic_json(self.path/'manifest.json',manifest)
                _atomic_text(self.path/'prepared_checkpoint.json',runner.checkpoint(state))
                _atomic_text(self.path/'latest_checkpoint.json',runner.checkpoint(state))
                if preparation_report['converged']:
                    automatic_cache.parent.mkdir(parents=True,exist_ok=True)
                    _atomic_json(automatic_cache,json.loads((self.path/'background_checkpoint.json').read_bytes()))
                    manifest['runtime']['prepared_background_cache']=str(automatic_cache)
                _atomic_json(self.path/'background_progress.json',{'status':preparation_report['status'],'absolute_background_time_seconds':state.time_seconds,'budget_exit':preparation_report['budget_exit'],'complete_checkpoint_file':'background_checkpoint.json'})
                if policy.require_converged and not preparation_report['converged']:
                    _atomic_text(self.path/'final_checkpoint.json',runner.checkpoint(state))
                    _atomic_json(self.path/'summary.json',{'status':'initialization_not_reached','time_seconds':state.time_seconds,
                        'windows':0,'background_initialization_status':preparation_report['status'],
                        'budget_exit':preparation_report['budget_exit'],'last_valid_state_retained':True,
                        'physiological_calibration':'NOT_COMPLETE','independent_animal_validation':'NOT_RUN'})
                    self._event('finished','initialization_not_reached',message=preparation_report['status']);return
            state.input_position.update(phase='media_stimulus',stimulus_origin_seconds=self._stimulus_origin)
            channels_source.set_origin(self._stimulus_origin,state.grid_origin_seconds)
            recorder = NPZWindowRecorder(self.path / 'windows')
            warmup_ticks = int(round(self._stimulus_origin / AUTHOR_STEP))
            total_ticks = int(round(total / AUTHOR_STEP))
            window_ticks = int(round(self.config.window_seconds / AUTHOR_STEP))
            sample_ticks = int(round(self.config.sample_seconds / AUTHOR_STEP))
            def aligned(tick):
                nominal = tick * AUTHOR_STEP
                candidates = [round(nominal / p) * p for p in runner.periods.values()
                              if abs(round(nominal / p) * p - nominal) < 1e-12]
                return min(candidates) if candidates else nominal
            self._event('status', 'running', phase='warmup' if warmup_ticks else 'recording')
            while int(round(state.time_seconds / AUTHOR_STEP)) < total_ticks:
                if not self._boundary():
                    break
                first = int(round(state.time_seconds / AUTHOR_STEP))
                warming = first < warmup_ticks
                last = min(first + window_ticks, warmup_ticks if warming else total_ticks)
                stop = aligned(last)
                channels = channels_source.window(state.time_seconds, stop)
                if warming:
                    before=[a.copy() for a in state.ordered_state_arrays]
                    state, _ = runner.advance(state, channels, stop, recording_plan=RecordingPlan(ports=(), states=(), cell_ids=()))
                    drift=[]
                    scales={'V':.01,'A':1e-12,'S':1e-9,'F':1e-12,'dimensionless':1.}
                    for spec,old,new in zip(runner.model.state_specs,before,state.ordered_state_arrays):
                        scale=scales.get(spec.unit,max(1.,float(np.max(np.abs(old)))))
                        denominator=1e-8*scale+1e-4*np.maximum(np.abs(old),np.abs(new))
                        drift.append({'state':spec.state_name,'unit':spec.unit,
                            'normalized_window_drift':float(np.max(np.abs(new-old)/denominator))})
                    maximum=max((row['normalized_window_drift'] for row in drift),default=0.)
                    manifest['initialization_diagnostics'].append({'window_end_seconds':stop,
                        'numerical_policy':{'rtol':1e-4,'atol_scale_fraction':1e-8},'maximum_normalized_drift':maximum,
                        'status':'INIT_NOT_CONVERGED_TRANSIENT_ALLOWED' if maximum>1 else 'window_drift_pass_RHS_and_recurrence_unverified',
                        'allowed_approximation_id':'finite_background_prerun_v1','states':drift,
                        'stochastic_spike_history':'seeded background history retained; not part of static drift criterion'})
                else:
                    ticks = list(range(first, last, sample_ticks)) + [last]
                    times = np.array([aligned(tick) for tick in ticks]); times[0], times[-1] = state.time_seconds, stop
                    state, result = runner.advance(state, channels, stop, sample_times=times,
                        recording_plan=RecordingPlan(states=(),cell_ids=getattr(compiled,'platform_recording_cell_ids',None),max_bytes=audit['resolved_execution_profile']['budgets'].get(
                            'max_recording_window_bytes',64*1024*1024)))
                    if not all(np.all(np.isfinite(q.values)) for q in result.outputs_with_units.values()):
                        raise RuntimeError('核心输出出现非有限数值')
                    declared = {p['key']: p for c in manifest['components'] for p in c['ports'].values()}
                    for key, quantity in result.outputs_with_units.items():
                        if key == 'time':
                            continue
                        metadata = declared.get(key)
                        if metadata is None or metadata['cell_ids'] != result.metadata['recorded_cell_ids'][key]:
                            raise RuntimeError('输出细胞与坐标声明不一致: ' + key)
                        if len(metadata['cell_ids']) != quantity.values.shape[0] or (metadata['unit'], metadata['meaning']) != (quantity.unit, quantity.meaning):
                            raise RuntimeError('输出行数或单位声明不一致: ' + key)
                    # Write complete window to temporary folder first; readers never see half-written NPZ.
                    temporary_recorder = NPZWindowRecorder(self.path / 'window_pending')
                    temporary = temporary_recorder.consume(result)
                    target = recorder.directory / ('window_%08d.npz' % recorder.windows)
                    os.replace(temporary, target)
                    recorder.windows += 1
                    self._windows = recorder.windows
                self._sim_time = state.time_seconds
                _atomic_text(self.path / 'latest_checkpoint.json', runner.checkpoint(state))
                self._event('status' if warming else 'window', 'running', phase='warmup' if warming else 'recording',
                            input_summary=channels_source.last_input_summary, solver_metrics=dict(runner.integration_metrics))
            checkpoint = runner.checkpoint(state)
            _atomic_text(self.path / 'final_checkpoint.json', checkpoint)
            stopped = int(round(state.time_seconds / AUTHOR_STEP)) < total_ticks
            manifest['media'] = dict(media.info)
            _atomic_json(self.path / 'manifest.json', manifest)
            _atomic_json(self.path / 'summary.json', {'status': 'stopped' if stopped else 'completed',
                         'simulation_time_seconds': state.time_seconds, 'windows': self._windows,
                         'wall_seconds': time.perf_counter() - wall_start, 'required_layers': manifest['required_layers'],
                         'components': len(compiled.components), 'unimplemented_layers': missing,
                         'state_reset_between_windows': False, 'checkpoint_sha256': hashlib.sha256(checkpoint.encode()).hexdigest(),
                         'scientific_validation': 'NOT_RUN', 'solver_metrics': dict(runner.integration_metrics),
                         'stimulus_origin_seconds':self._stimulus_origin,'grid_origin_seconds':state.grid_origin_seconds,
                         'background_initialization_status':preparation_report['status'] if preparation_report else 'FIXED_PRERUN_NO_CONVERGENCE_CLAIM'})
            self._event('finished', 'stopped' if stopped else 'completed')
        except Exception as error:
            if self.path:
                (self.path / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
                partial=getattr(error,'partial_build_audit',None)
                if partial is not None:
                    _atomic_json(self.path/'partial_build_audit.json',partial)
                    if 'IPL_geometry' in partial:_atomic_json(self.path/'ipl_geometry.json',partial['IPL_geometry'])
                if runner is not None and state is not None:
                    _atomic_text(self.path / 'latest_checkpoint.json', runner.checkpoint(state))
            self._event('error', 'error', str(error))
        finally:
            if media:
                media.close()
