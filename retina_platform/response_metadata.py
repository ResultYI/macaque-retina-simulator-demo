"""Read saved bindings and static operators only. No solver is constructed/advanced."""
from pathlib import Path
import json
import hashlib
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def metadata(root, base, temporary):
    # Reuse the exact assembly adapters used by the completed run. Construction
    # below recovers geometry/operators/parameters, never advances a state.
    from tools.cross_position_context_v1 import load_background
    from retina_sim.parameter_system import data
    preparation = read(base / 'PREPARATION.json')
    files, positions = {}, {}
    for position, binding in preparation['bindings'].items():
        folder = base / (position + '_run_v1')
        identity, result = read(folder / 'IDENTITY.json'), read(folder / 'RESULT.json')
        if (result['reached_seconds'] != preparation['protocol']['stop_seconds']
                or not result.get('physical_history_conservation_passed')
                or not result.get('dynamic_numerical_defect', {}).get('passed')):
            raise ValueError('响应未完成：' + position)
        assembly, seed, source = load_background(Path(binding['prepared']), Path(binding['background']))
        if source['initialized_model_id'] != identity['initialized_model']:
            raise ValueError('响应与初始化身份不一致')
        native = assembly.lateral_native_factory(assembly, bound=False)
        if native.source_version != assembly.lateral_configuration['selected_native_execution']:
            raise ValueError('保存的原生连接来源与当前适配器不一致')
        geometry, populations = {}, {}
        scale = assembly.manifest['configuration']['mm_per_degree']
        with np.load(folder / read(folder / 'RESPONSE_INDEX.json')['chunks'][0]['file'], allow_pickle=False) as saved:
            field_shapes = {key: saved[key].shape[0] for key in saved.files if key not in ('time_seconds', 'side')}
        units = read(folder / 'PORT_UNITS.json')
        for name, population in assembly.populations.items():
            count = population.group_count
            for suffix, value in [('ids', population.group_ids), ('members', population.counts),
                                  ('xy', population.centroid_mm / scale)]:
                geometry[name + '__' + suffix] = np.asarray(value)
            geometry[name + '__subtypes'] = (np.asarray(assembly.types[name]) if name in ('cones', 'HC')
                                              else np.full(count, name))
            if hasattr(population, 'member_offsets'):
                geometry[name + '__member_offsets'] = population.member_offsets
                geometry[name + '__member_global_rows'] = population.member_global_rows
            fields = {}
            for key, rows in field_shapes.items():
                prefix = name + '__'
                port = 'ports__' + name + ':'
                if key.startswith(prefix) and rows == count:
                    fields[key] = identity['fields'][key]['unit']
                elif key.startswith(port) and rows == count:
                    fields[key] = units[key[7:]]
            primary = (name + '__voltage' if name in identity['types'] else
                       'ports__' + name + ':' + ('release' if name == 'cones' else 'Vh' if name == 'HC'
                       else 'inhibition_state' if name == 'local_feedforward_AC' else 'voltage'))
            populations[name] = dict(count=count, coordinate_role=population.coordinate_role,
                fields=fields, primary=primary if primary in fields else None,
                representation='effective_functional_pool' if name.startswith(('same_', 'cross_')) else 'representative_group',
                missing=[] if fields else ['没有保存该群的可空间映射状态'])
            if name in identity['types']:
                expected = identity['types'][name]
                if expected['group_ids'] != population.group_ids.tolist() or expected['represented_members'] != population.counts.tolist():
                    raise ValueError('RGC响应行与真实几何身份不一致：' + name)
        route_arrays, routes = {}, []
        gains = seed['inner_parameters']['route_gains']
        for name, route in assembly.routes.items():
            matrix = assembly.matrix('routes/' + name + '/mean_weights.npz').copy().tocsr()
            if name in gains and gains[name] == 0:
                continue
            matrix.eliminate_zeros()
            if not matrix.nnz:
                continue
            token = 'r' + str(len(routes))
            route_arrays[token + '_indptr'] = matrix.indptr
            route_arrays[token + '_indices'] = matrix.indices
            route_arrays[token + '_weights'] = matrix.data
            routes.append(dict(token=token, name=name, pre=route.pre, post=route.post,
                               kind=route.kind, shape=list(matrix.shape)))
        matrix = native.coupling_neighbor.copy().tocsr()
        matrix.setdiag(0); matrix.eliminate_zeros()
        if native.coupling_rate and matrix.nnz:
            token = 'r' + str(len(routes))
            route_arrays.update({token + '_indptr': matrix.indptr, token + '_indices': matrix.indices,
                                 token + '_weights': matrix.data})
            routes.append(dict(token=token, name='actual_HC_homotypic_coupling', pre='HC', post='HC',
                               kind='effective_homotypic_coupling', shape=list(matrix.shape)))
        for kind, arrays in [('geometry', geometry), ('connections', route_arrays)]:
            path = temporary / (position + '_' + kind + '.npz')
            np.savez_compressed(path, **arrays)
            files['cells/' + position + '/' + kind + '.npz'] = path
        parameter_path = temporary / (position + '_parameters.json')
        parameter_path.write_text(json.dumps(data(dict(inner_parameters=seed['inner_parameters'],
            HC_binding=assembly.lateral_configuration, initialization_identity=source,
            native_connection_identity=native.identity)), ensure_ascii=False, allow_nan=False), encoding='utf8')
        files['provenance/' + position + '/parameters.json'] = parameter_path
        files['provenance/' + position + '/selection_identity.json'] = Path(binding['prepared']) / 'local_static/selection_identity.json'
        for name in ['IDENTITY.json', 'PORT_UNITS.json', 'RESULT.json', 'RESOURCE.json',
                     'RESPONSE_INDEX.json', 'GROUP_OBJECT_EVENTS.json']:
            files['responses/' + position + '/' + name] = folder / name
        for chunk in read(folder / 'RESPONSE_INDEX.json')['chunks']:
            files['responses/' + position + '/' + chunk['file']] = folder / chunk['file']
        positions[position] = dict(populations=populations, routes=routes,
            geometry='cells/' + position + '/geometry.npz', connections='cells/' + position + '/connections.npz',
            response_prefix='responses/' + position + '/',
            center_deg=read(Path(binding['prepared']) / 'GEOMETRY_SELECTION.json')['center_deg'])
        del native, assembly, seed
    optical = root / 'reports/cone_video_113_4s_eccentric_aperture_0_6deg_20261029_v1'
    plan = read(optical / 'capture_plan.json')
    pins = {str(key).replace('\\','/'): value for key,value in preparation['source_pins'].items()}
    for name in ['capture_plan.json','source_rgb.mat']:
        path = optical / name
        expected = pins[path.relative_to(root).as_posix()]
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest() != expected:
                raise ValueError('刺激来源SHA不一致：' + name)
    from scipy.io import loadmat
    frame_path = temporary / 'frames.npz'
    frames = loadmat(optical / 'source_rgb.mat')['source_rgb'].transpose(3, 0, 1, 2)
    np.savez_compressed(frame_path, rgb=frames)
    files['stimulus/frames.npz'] = frame_path
    for name in ['PREPARATION.json', 'EXECUTION_SOURCES.json', 'RESOURCE_LEDGER.json', 'NUMERICAL_SCOPE.json', 'FIRST_FRAME_CONTINUITY.json']:
        if (base / name).is_file(): files['provenance/' + name] = base / name
    for name in ['FINAL_REPORT_ZH.md', 'POSITION_TYPE_SUMMARY.csv', 'TIMING_SUMMARY.md', 'RGC_video_curves.png', 'upstream_video_curves.png']:
        if (base / name).is_file(): files['reports/' + name] = base / name
    for path in (base / 'recovery_v2').glob('*.log'):
        files['logs/recovery_v2/' + path.name] = path
    for path in base.glob('*.log'):
        if not path.name.startswith('package_'): files['logs/pipeline/' + path.name] = path
    for position in positions:
        for name in ['state.json', 'terminal.json', 'events.jsonl']:
            path = base / (position + '_run_v1/runtime_model') / name
            if path.is_file(): files['logs/' + position + '/' + name] = path
    manifest = dict(schema='retina-response-bundle-1', status='complete', run_id=preparation['experiment_id'],
        stimulus=dict(name='113.AVI', original_sha256=plan['media_sha256'], frames='stimulus/frames.npz',
                      fps=30., frame_count=120, source_start_seconds=0., duration_seconds=4., exposure_seconds=1/30),
        positions=positions, time_rule='Latest saved sample at or before requested time; rightmost side; no interpolation',
        representation='Actual representative groups; member states are not independent recordings',
        missing=['未保存的状态不重建', '外部储库固定S0，非全视网膜动态边界', 'Rod为固定工程背景',
                 'Sbi混合输入和高基线未标定', '并非完整可恢复计算checkpoint'],
        qualification=dict(FIT_READY=False, background_qualified=False, production_adopted=False))
    return manifest, files
