"""Read-only spatial and temporal adapter for portable complete responses."""
from collections import deque
import math
import zipfile
import numpy as np
from .response_bundle import read, sha, safe_name

RGC = ('midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON')
LAYERS = [('Cone', ('cones',)), ('HC · H1/H2', ('HC',)),
          ('BC', ('midget_BC_ON','midget_BC_OFF','S_ON_BC','diffuse_BC_ON','diffuse_BC_OFF')),
          ('普通AC · 功能RF池', ('same_ON_midget_AC_v1','same_OFF_midget_AC_v1','cross_OFF_ON_parasol_AC_v1','cross_ON_OFF_parasol_AC_v1')),
          ('S通路AC', ('local_feedforward_AC',)), ('Rod', ('rods',)),
          ('RBC', ('rod_BC',)), ('AII', ('AII_AC',)), ('RGC', RGC)]


def checked_npz(path):
    # NPZ is itself an archive: bound its expanded arrays before numpy can allocate.
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 512 or sum(entry.file_size for entry in entries) > 2 * 2**30:
            raise ValueError('NPZ数组数量或展开体积超限')
        for entry in entries:
            if not entry.filename.endswith('.npy') or '/' in entry.filename or '\\' in entry.filename or entry.file_size > 512 * 2**20:
                raise ValueError('NPZ包含非法数组路径或超大数组')
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0): shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0): shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else: raise ValueError('不支持的安全NPY版本')
                if dtype.hasobject or math.prod(shape) * dtype.itemsize > 512 * 2**20:
                    raise ValueError('禁止对象数组或过大数组')
    return np.load(path, allow_pickle=False)


def latest_index(time, seconds):
    if not math.isfinite(seconds): raise ValueError('时间必须有限')
    return max(0, min(len(time)-1, int(np.searchsorted(time, seconds + 1e-12, side='right')-1)))


def ancestors(populations, incoming, target):
    """Exact nonzero directed predecessors, including recurrent/gap components."""
    seen = {name: set() for name in populations}
    seen[target[0]].add(target[1]); queue = deque([target]); direct = set()
    while queue:
        population, row = queue.popleft()
        for pre, indptr, indices in incoming.get(population, []):
            for index in indices[indptr[row]:indptr[row+1]]:
                item = (pre, int(index))
                if (population, row) == target: direct.add(item)
                if item[1] not in seen[pre]:
                    seen[pre].add(item[1]); queue.append(item)
    return {name: np.asarray(sorted(rows), dtype=int) for name, rows in seen.items()}, direct


class ResponseData:
    def __init__(self, source, position, progress=lambda text: None):
        self.source, self.position = source, position
        self.manifest = source.manifest
        self.meta = source.manifest['positions'][position]
        self.populations = self.meta['populations']
        self.prefix = self.meta['response_prefix']
        self.values, self.geometry, self.incoming, self.ranges = {}, {}, {}, {}
        with checked_npz(source.files[self.meta['geometry']]) as saved:
            for name, meta in self.populations.items():
                count = meta['count']
                if not isinstance(count, int) or not 0 <= count <= 100000: raise ValueError('群数量超限')
                arrays = {key: saved[name + '__' + key] for key in ('ids', 'members', 'xy', 'subtypes')}
                if arrays['xy'].shape != (count, 2) or not np.isfinite(arrays['xy']).all() or arrays['ids'].shape != (count,) or len(np.unique(arrays['ids'])) != count:
                    raise ValueError('细胞坐标或身份形状不一致：' + name)
                if arrays['members'].shape != (count,) or np.any(arrays['members'] < 1): raise ValueError('成员数量非法')
                self.geometry[name] = arrays
        with checked_npz(source.files[self.meta['connections']]) as saved:
            for route in self.meta['routes']:
                pre, post, token = route['pre'], route['post'], route['token']
                indptr, indices, weights = (saved[token + '_' + key] for key in ('indptr','indices','weights'))
                if indptr.shape != (self.populations[post]['count']+1,) or indptr[0] != 0 or indptr[-1] != len(indices) or np.any(np.diff(indptr)<0) or np.any(indices<0) or np.any(indices>=self.populations[pre]['count']) or len(weights) != len(indices) or not np.isfinite(weights).all() or np.any(weights == 0):
                    raise ValueError('真实连接CSR损坏：' + route['name'])
                self.incoming.setdefault(post, []).append((pre, indptr, indices))
        identity = read(source.files[self.prefix + 'IDENTITY.json'])
        for name, expected in identity['types'].items():
            actual = self.geometry[name]
            if actual['ids'].tolist() != expected['group_ids'] or actual['members'].tolist() != expected['represented_members']:
                raise ValueError('响应与空间行身份不一致')
        self.index = read(source.files[self.prefix + 'RESPONSE_INDEX.json'])
        if (len(self.index['chunks']) > 256 or sum(chunk['records'] for chunk in self.index['chunks']) > 2_000_000
                or any(chunk['records'] <= 0 for chunk in self.index['chunks'])):
            raise ValueError('v1查看器记录数超限（256块/200万节点）')
        self.events = read(source.files[self.prefix + 'GROUP_OBJECT_EVENTS.json'])
        self.stop = float(source.manifest['stimulus']['duration_seconds'])
        if not 0 < self.stop <= 3600: raise ValueError('播放时长非法')
        self.chunks, times, sides = [], [], []
        for chunk in self.index['chunks']:
            name = self.prefix + safe_name(chunk['file'])
            path = source.files[name]
            progress('核对响应块：' + chunk['file'])
            if sha(path) != chunk['sha256']: raise ValueError('响应块SHA不一致')
            with checked_npz(path) as saved:
                t, side = saved['time_seconds'], saved['side']
                if len(t) != chunk['records'] or side.shape != t.shape: raise ValueError('响应时间轴形状不一致')
                times.append(t); sides.append(side)
            self.chunks.append(path)
        time, side = np.concatenate(times), np.concatenate(sides)
        if not np.isfinite(time).all() or np.any(np.diff(time)<0) or abs(time[0])>1e-12 or abs(time[-1]-self.stop)>1e-12:
            raise ValueError('响应时间覆盖或顺序非法')
        # Only actual saved 1ms nodes are used, preferring the rightmost event side.
        rounded = np.rint(time*1000).astype(int)
        lookup = {}
        for i in np.flatnonzero(np.abs(time-rounded/1000)<1e-12): lookup[int(rounded[i])] = i
        indices = np.asarray([lookup[k] for k in sorted(lookup)], dtype=int)
        self.time = time[indices]
        self.selection = []
        offset = 0
        for block in times:
            self.selection.append(indices[(indices>=offset)&(indices<offset+len(block))]-offset)
            offset += len(block)
        if self.time[0] != 0 or abs(self.time[-1]-self.stop)>1e-12: raise ValueError('缺少真实播放端点')
        rows = sum(meta['count'] for meta in self.populations.values() if meta['primary'])
        rows += 3 * sum(self.populations[name]['count'] for name in RGC if name in self.populations)
        if rows * len(self.time) * 8 > 1024 * 2**20:
            raise ValueError('空间播放状态预计超过1GiB，拒绝无界分配')
        self.available_fields = {name: dict(meta['fields']) for name, meta in self.populations.items()}
        for name, meta in self.populations.items():
            fields = [meta['primary']] if meta['primary'] else []
            if name in RGC: fields += [name + '__expected_binary_count_rate', name + '__excitatory_conductance', name + '__inhibitory_conductance']
            for key in fields:
                if key in meta['fields']:
                    progress('读取逐组状态：' + name)
                    self.field(name, key)
        with checked_npz(source.files[self.manifest['stimulus']['frames']]) as saved:
            self.frames = saved['rgb']
        stimulus = self.manifest['stimulus']
        if self.frames.dtype != np.uint8 or self.frames.ndim != 4 or self.frames.shape[0] != stimulus['frame_count'] or self.frames.shape[-1] != 3:
            raise ValueError('原始视频帧形状或类型非法')
        self.rgcs = [(name, i) for name in RGC if name in identity['types'] for i in range(self.populations[name]['count'])]

    def field(self, name, key):
        if key not in self.populations[name]['fields']: raise ValueError('该状态未保存或无法映射到细胞组')
        if key not in self.values:
            blocks = []
            for path, selection in zip(self.chunks, self.selection):
                with checked_npz(path) as saved:
                    value = saved[key]
                    if value.ndim != 2 or value.shape[0] != self.populations[name]['count'] or not np.isfinite(value).all():
                        raise ValueError('状态数组形状/有限性不合法：' + key)
                    blocks.append(value[:, selection])
            self.values[key] = np.concatenate(blocks, axis=1)
        return self.values[key]

    def network(self, target):
        return ancestors(self.populations, self.incoming, target)

    def field_range(self, population, key, delta):
        token = (key, delta)
        if token not in self.ranges:
            value = self.values[key]
            unit = self.populations[population]['fields'][key]
            scale = 1000 if unit == 'V' else 1
            shown = value-value[:,0,None] if delta else value
            low,high=float(shown.min())*scale,float(shown.max())*scale
            if delta:high=max(abs(low),abs(high));low=-high
            self.ranges[token]=(low,high,scale,'mV' if unit=='V' else unit)
        return self.ranges[token]

    def frame_at(self, seconds):
        fps = self.manifest['stimulus']['fps']
        return self.frames[min(len(self.frames)-1, max(0, int(seconds*fps+1e-8)))]

    def identity_text(self, population, row):
        g = self.geometry[population]; meta = self.populations[population]
        return f"{g['ids'][row]}\n类型：{g['subtypes'][row]}\n坐标：({g['xy'][row,0]:.7f}, {g['xy'][row,1]:.7f}) °\n坐标语义：{meta['coordinate_role']}\n代表成员：{g['members'][row]}\n{meta['representation']}；没有独立成员状态"
