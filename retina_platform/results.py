"""On-demand window reads and faithful export of actual platform runs."""
from pathlib import Path
import csv
import json
import shutil
import numpy as np

def _legacy_cells(ids, populations):
    lookup = {cell_id: (pop, i) for pop in populations
              for i, cell_id in enumerate(pop.get('cell_ids', []))}
    rows = [lookup.get(cell_id) for cell_id in ids]
    value = {'cell_ids': list(ids), 'cell_types': [], 'subtypes': [], 'regions': [], 'geometry': {}}
    for field, default in (('cell_types', 'unknown'), ('subtypes', 'unknown'), ('regions', 'unspecified')):
        value[field] = [row[0].get(field, [default] * len(row[0]['cell_ids']))[row[1]] if row else default for row in rows]
    value['population'] = ','.join(dict.fromkeys(row[0]['population_id'] for row in rows if row)) or None
    def select(pop, name):
        if name == 'RF':
            return pop.get('receptive_field_positions') or pop.get('table_rf')
        if name == 'Soma':
            return pop.get('anatomical_positions') or pop.get('table_soma')
        candidates = ('pedicle_positions', 'dendritic_positions', 'terminal_positions') if name == 'OPL' else ('terminal_positions', 'dendritic_positions')
        return next((pop[field] for field in candidates if pop.get(field) and pop[field].get('frame') == name), None)
    for name in ('RF', 'Soma', 'OPL', 'IPL'):
        quantities = [select(row[0], name) if row else None for row in rows]
        if quantities and all(q and len(q.get('values', [])) > row[1] for q, row in zip(quantities, rows)):
            if len({(q['unit'], q.get('frame')) for q in quantities}) == 1:
                value['geometry'][name] = {**{k: quantities[0][k] for k in ('unit', 'frame', 'meaning', 'evidence_tag') if k in quantities[0]},
                                           'values': [q['values'][row[1]] for q, row in zip(quantities, rows)]}
    return value

def _legacy_manifest(original, first_window):
    """Derive display metadata solely from existing recorded IDs and population rows."""
    with np.load(first_window, allow_pickle=False) as archive:
        saved = json.loads(str(archive['manifest_json']))
        metadata = saved.get('metadata', {})
        populations = metadata.get('populations', [])
        recorded_ids = metadata.get('recorded_cell_ids', {})
        component_records = metadata.get('executed_components', [])
        if not populations or not recorded_ids or not component_records:
            raise ValueError('历史结果缺少真实细胞/组件记录，无法恢复空间展示')
        components = []
        for recorded in component_records:
            cid = recorded['component_id']
            ports = {}
            for key, declaration in saved['ports'].items():
                if not key.startswith(cid + '.') or key not in archive or key not in recorded_ids:
                    continue
                cells = _legacy_cells(recorded_ids[key], populations)
                array = archive[key]
                if array.ndim not in (1, 2) or array.shape[0] != len(cells['cell_ids']):
                    raise ValueError('历史输出与记录的细胞 ID 行数不一致: ' + key)
                ports[key[len(cid) + 1:]] = {'key': key, **declaration, **cells,
                    'temporal_kind': 'window_aggregate' if array.ndim == 1 else 'sampled'}
            main = next(iter(ports.values()), {'population': None, 'cell_ids': [], 'cell_types': [], 'subtypes': [], 'regions': [], 'geometry': {}})
            components.append({'id': cid, 'layers': recorded.get('layers', []), 'ports': ports,
                               'mechanism': recorded.get('mechanism'), 'profile_id': recorded.get('profile_id'),
                               'source_ids': recorded.get('source_ids', []),
                               **{k: main[k] for k in ('population', 'cell_ids', 'cell_types', 'subtypes', 'regions', 'geometry')}})
    return {**original, 'schema': 'retina-platform-1', 'derived_read_only_adapter': True,
            'original_manifest': original, 'legacy_source_run': str(first_window.parent.parent),
            'required_layers': original['spec']['required_mechanisms'], 'components': components,
            'input_assumptions': {'profile': original.get('input_class') or 'legacy_source_run',
                                  'source': 'original input_class/input_contracts retained; no new calibration'},
            'config': {}, 'media': {'kind': 'legacy_source_run'}}

class ResultStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self._manifest_signature = None
        self.refresh()

    def refresh(self):
        self.windows = sorted((self.path / 'windows').glob('window_*.npz'))
        manifest_path = self.path / 'manifest.json'
        stat = manifest_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size,
                     (str(self.windows[0]), self.windows[0].stat().st_mtime_ns) if self.windows else None)
        if signature != self._manifest_signature:
            original = json.loads(manifest_path.read_text(encoding='utf-8'))
            if original.get('schema') == 'retina-platform-1':
                self.manifest = original
            elif not original.get('schema') and original.get('spec', {}).get('required_mechanisms') and self.windows:
                self.manifest = _legacy_manifest(original, self.windows[0])
            else:
                raise ValueError('结果格式不支持，或没有可读取的真实窗口与细胞记录')
            self._manifest_signature = signature
        return self

    def read_window(self, index=-1):
        self.refresh()
        if not self.windows:
            return {'time': np.empty(0), 'signals': {}, 'ports': {}}
        with np.load(self.windows[index], allow_pickle=False) as archive:
            meta = json.loads(str(archive['manifest_json']))
            return {'time': archive['time'].copy(),
                    'signals': {k: archive[k].copy() for k in archive.files if k not in ('time', 'manifest_json')},
                    'ports': meta['ports'], 'metadata': meta.get('metadata', {})}

    def port_info(self, key):
        for component in self.manifest['components']:
            for value in component['ports'].values():
                if value['key'] == key:
                    return value
        raise KeyError(key)

    def history(self, key, cell_index):
        self.refresh()
        chunks_t, chunks_v, last = [], [], -float('inf')
        for path in self.windows:
            with np.load(path, allow_pickle=False) as archive:
                if key not in archive:
                    continue
                t = archive['time']
                signal = archive[key]
                if signal.ndim not in (1, 2) or not 0 <= cell_index < signal.shape[0]:
                    raise ValueError('细胞索引超出所选信号范围')
                if signal.ndim == 1:
                    # Actual window aggregate: its endpoint denotes the completed interval, not an instantaneous sample.
                    if t[-1] > last + 1e-12:
                        chunks_t.append(np.array([t[-1]]))
                        chunks_v.append(np.array([signal[cell_index]]))
                        last = float(t[-1])
                    continue
                keep = t > last + 1e-12
                if np.any(keep):
                    chunks_t.append(t[keep].copy())
                    chunks_v.append(signal[cell_index, keep].copy())
                    last = float(t[keep][-1])
        return (np.concatenate(chunks_t), np.concatenate(chunks_v)) if chunks_t else (np.empty(0), np.empty(0))

    def export_curve(self, key, cell_index, path):
        destination = Path(path)
        if destination.exists():
            raise FileExistsError('导出文件已存在')
        info = self.port_info(key)
        profile = self.manifest.get('input_assumptions', {}).get('profile') or self.manifest.get('input_class') or 'legacy_source_run'
        t, v = self.history(key, cell_index)
        if not len(t):
            raise ValueError('没有已完成窗口可导出')
        with destination.open('x', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            time_label = 'window_end_seconds' if info.get('temporal_kind') == 'window_aggregate' else 'time_seconds'
            writer.writerow([time_label, key + ' [' + info['unit'] + ']', 'cell_id', 'meaning', 'profile'])
            for ti, vi in zip(t, v):
                writer.writerow([repr(float(ti)), repr(float(vi)), info['cell_ids'][cell_index], info['meaning'], profile])
        return destination

    def export_bundle(self, destination):
        self.refresh()
        status_path = self.path / 'status.json'
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding='utf-8')).get('status')
            if status not in ('paused', 'completed', 'stopped', 'error'):
                raise ValueError('请暂停或停止运行后导出完整结果，确保窗口与 checkpoint 同一边界')
        elif self.manifest.get('derived_read_only_adapter'):
            summary_path = self.path / 'summary.json'
            summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
            completed = summary.get('engineering_execution') == 'PASS' or summary.get('completed') is True or summary.get('status') in ('completed', 'stopped', 'error')
            if not completed:
                legacy_status_path = self.path / 'run_status.json'
                if legacy_status_path.is_file():
                    phase = json.loads(legacy_status_path.read_text(encoding='utf-8')).get('phase')
                    if phase not in ('completed', 'stopped', 'error', 'paused'):
                        raise ValueError('历史运行尚未结束，请等待或停止后导出')
                # Metadata-only stability check for historical runs without a completion ledger.
                import time
                def fingerprint():
                    return [(str(p.relative_to(self.path)), p.stat().st_size, p.stat().st_mtime_ns)
                            for p in sorted(self.path.rglob('*')) if p.is_file()]
                before = fingerprint()
                time.sleep(.05)
                if not self.windows or fingerprint() != before:
                    raise ValueError('历史结果仍在写入，请稍后导出')
        destination = Path(destination).resolve()
        if destination.exists():
            raise FileExistsError('结果导出目录已存在')
        if destination == self.path or self.path in destination.parents:
            raise ValueError('导出目录不能位于原结果目录内')
        self.refresh()
        shutil.copytree(self.path, destination)
        return destination

def _check_legacy(path):
    """Read-only acceptance of legacy metadata; does not execute any core dynamics."""
    import hashlib
    def hashes():
        return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in path.rglob('*') if p.is_file()}
    before = hashes()
    result = ResultStore(path)
    assert result.manifest['derived_read_only_adapter']
    assert len(result.manifest['required_layers']) == 20
    assert len(result.manifest['components']) == 39
    info = result.port_info('outer_native.Vh')
    assert len(info['cell_ids']) == 32 and all(x == 'HC' for x in info['cell_types'])
    assert len(info['geometry']['RF']['values']) == 32
    window = result.read_window(0)
    assert window['signals']['outer_native.Vh'].shape[0] == 32
    for component in result.manifest['components']:
        for port in component['ports'].values():
            assert not port['key'].startswith('state:')
            assert len(port['cell_ids']) == window['signals'][port['key']].shape[0]
    t, v = result.history('outer_native.Vh', 0)
    assert len(t) == len(v) and np.all(np.diff(t) > 0)
    cached = result.manifest
    assert cached is result.refresh().manifest  # Repeated reads reuse unchanged manifest metadata.
    assert hashes() == before, 'Legacy source files changed'
    print(json.dumps({'legacy_read_only_adapter': 'PASS', 'file_hashes_unchanged': len(before),
                      'layers': 20, 'components': 39, 'Vh_HC_rows': 32,
                      'windows': len(result.windows), 'history_samples': len(t)}, ensure_ascii=False))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read-only legacy result acceptance')
    parser.add_argument('--check-legacy', type=Path, required=True)
    _check_legacy(parser.parse_args().check_legacy.resolve())
