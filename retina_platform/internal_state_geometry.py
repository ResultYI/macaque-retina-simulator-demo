"""Read the frozen GIN69 member geometry and project its saved sparse routes.

No model builder, spatial sampler, connectivity kernel or time integrator is used.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy import sparse

from retina_sim.atlas import global_cell_id
from retina_sim.contracts import canonical_hash
from .response_data import ancestors, RGC


BC = ('midget_BC_ON', 'midget_BC_OFF', 'S_ON_BC', 'diffuse_BC_ON', 'diffuse_BC_OFF')
REMOVED = ('local_crossover_AC', 'widefield_AC')


def _read(path):
    return json.loads(Path(path).read_bytes())


def _arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _dictionary(node):
    return dict(node['dictionary'])


class SavedGeometry:
    """One point per solved object; original members remain a separate geometry."""

    def __init__(self, project_root, current_source):
        self.root = Path(project_root).resolve()
        self.current_source = Path(current_source).resolve()
        if not self.current_source.is_relative_to(self.root):
            raise ValueError('结果目录必须属于当前项目')
        self.report = self.current_source.parent
        self.group_root = self.report / 'executions/assemble_groups_01'
        self.assets = self.root / 'reports/CONE_HC_FINAL_IMPLEMENTATION_2026-11-02/eccentric_assets'
        background = _read(self.current_source / 'BACKGROUND.json')
        self.identity = background['model_identity']
        self._verified = set()
        seed_path = self.current_source / 'seed/manifest.json'
        self._check(seed_path, background['seed_manifest_sha256'])
        self.seed_metadata = _dictionary(_read(seed_path)['payload'])
        self._verify_current_identity(background)
        self.group_record = _read(self.group_root / 'GROUP_ARTIFACT.json')
        self.members = _arrays(self._asset('MEMBERS.npz'))
        self.geometry, self.populations, self.incoming, self.routes = {}, {}, {}, {}
        self._layouts, self._table_records = {}, {}
        bc_path = self.root / 'reports/BC_PHYSIOLOGY_IMPLEMENTATION_2026-11-02/BC_CELL_IDENTITIES.npz'
        self.bc = _arrays(bc_path)
        for name, meta in self.group_record['populations'].items():
            if name in REMOVED:
                continue
            relative = f'populations/{name}/MEMBERSHIP.npz'
            path = self._group_file(relative)
            layout = _arrays(path)
            self._layouts[name] = layout
            record = self._table_record(meta['source_table'])
            self._table_records[name] = record
            rows = layout['member_global_rows']
            member_ids, keys, member_xy = self._source_members(record, rows, meta['coordinate_role'])
            member_group = layout['fine_to_group'][rows]
            subtypes = np.full(len(layout['counts']), name, dtype='U64')
            if name in ('cones', 'HC'):
                token = 'cone' if name == 'cones' else 'hc'
                for key, value in [(token+'_parent_rows', rows), (token+'_group', member_group),
                                   (token+'_counts', layout['counts'])]:
                    np.testing.assert_array_equal(self.members[key], value)
                np.testing.assert_array_equal(self.members[token+'_keys'].astype('U'), keys)
                for group in range(len(subtypes)):
                    labels = np.unique(self.members[token+'_types'][member_group == group])
                    if len(labels) != 1:
                        raise ValueError('保存的分组混合不同细胞类型')
                    subtypes[group] = labels[0]
            geometry = dict(ids=layout['group_ids'].astype('U'), xy_mm=layout['centroid_mm'],
                members=layout['counts'], subtypes=subtypes, coordinate_role=meta['coordinate_role'],
                representation='representative_group', member_ids=member_ids, member_keys=keys,
                member_rows=rows, member_xy_mm=member_xy, member_group=member_group,
                member_coordinate_role=meta['coordinate_role'])
            if name == 'cones':
                geometry.update(member_xy_mm=self.members['cone_IS_mm'],
                    member_coordinate_role='cone_inner_segment',
                    member_opl_mm=self.members['cone_pedicle_mm'])
            if name == 'HC':
                np.testing.assert_array_equal(member_xy, self.members['hc_anchor_mm'])
            if name in BC:
                prefix = name+'__'
                ids = self.bc[prefix+'global_cell_ids']
                np.testing.assert_array_equal(ids, member_ids)
                np.testing.assert_array_equal(self.bc[prefix+'source_rows'], rows)
                np.testing.assert_array_equal(self.bc[prefix+'original_group_rows'], member_group)
                xy = self.bc[prefix+'functional_ancestry_mm']
                np.testing.assert_array_equal(xy, member_xy)
                geometry.update(ids=ids, xy_mm=xy, members=np.ones(len(rows), dtype=int),
                    subtypes=np.full(len(rows), name), representation='independently_solved_BC_cell',
                    member_group=np.arange(len(rows)), input_anchor_OPL_mm=self.bc[prefix+'input_anchor_OPL_mm'])
            self._add_geometry(name, geometry)
        self._load_functional_pools()
        self.h1_rows = np.flatnonzero(self.geometry['HC']['subtypes'] == 'H1')
        self.h2_rows = np.flatnonzero(self.geometry['HC']['subtypes'] == 'H2')
        self._load_routes()
        self.cone_release_rows = np.unique(np.concatenate([
            route['matrix'].indices for route in self.routes.values()
            if route['pre'] == 'cones' and route['post'] in BC]))
        parents = _arrays(self._group_file('PARENT_ROWS.npz'))
        self.parent_rows = parents
        for name, geometry in self.geometry.items():
            if name in parents:
                rows = self.bc[name+'__original_group_rows'] if name in BC else np.arange(len(geometry['ids']))
                geometry['parent_rows'] = parents[name][rows]
                geometry['parent_ids'] = self._layouts[name]['group_ids'][rows].astype('U')
        target_rows = np.flatnonzero(parents['parasol_RGC_OFF'] == 7730)
        if len(target_rows) != 1 or self.geometry['parasol_RGC_OFF']['members'][target_rows[0]] != 1:
            raise ValueError('69#6 原始目标成员映射不一致')
        self.target = ('parasol_RGC_OFF', int(target_rows[0]))
        self.rgcs = [(name, row) for name in RGC for row in range(self.populations[name]['count'])]
        with np.load(self._asset('H1_COMPLETE_ROW_WITNESSES.npz'), allow_pickle=False) as saved:
            self.h1_witness_rows = saved['fine_rows'].copy()

    def _check(self, path, expected=None):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError('缺失或项目外的几何资产：'+str(path))
        token = (str(path), expected)
        if expected and token not in self._verified:
            with path.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError('保存的几何资产身份改变：'+str(path))
            self._verified.add(token)
        return path

    def _verify_current_identity(self, background):
        metrics = _read(self.current_source/'METRICS.json')
        native = _dictionary(self.seed_metadata['h1_h2_model'])
        if (metrics['neural']['model_identity'] != self.identity
                or metrics['neural']['identity'] != background['identity']
                or metrics['neural']['checks']['artifact'] != native['artifact']):
            raise ValueError('当前神经结果、初态及空间模型身份不一致')
        for relative, expected in metrics['reproducibility']['source_sha256'].items():
            if relative.startswith('configurations/research/'):
                self._check(self.root/relative, expected)
        priority = self.root/'reports/CONE_HC_FINAL_IMPLEMENTATION_2026-11-02/h2_priority_assets'
        receipt = _read(priority/'RECEIPT.json')
        if (native['H2_priority'] != receipt['H2_weight_identity']
                or dict(native['H1_calibration']['dictionary']) != receipt['H1_calibration']):
            raise ValueError('当前初态与保存的 H1/H2 空间资产不一致')
        eccentric_relative = (self.assets/'SPATIAL_RECEIPT.json').relative_to(self.root).as_posix()
        self._check(self.assets/'SPATIAL_RECEIPT.json', receipt['source_files'][eccentric_relative])
        self.identity_evidence = dict(model_identity=self.identity, native_artifact=native['artifact'],
            H2_weight_identity=receipt['H2_weight_identity'], A_identity=receipt['A_identity'],
            B_identity=receipt['B_identity'], source_seed_manifest_sha256=background['seed_manifest_sha256'],
            geometry_regenerated=False, model_built=False)

    def _asset(self, name):
        receipt = _read(self.assets / 'SPATIAL_RECEIPT.json')
        return self._check(self.assets / name, receipt['output_files'][name])

    def _group_file(self, relative):
        return self._check(self.group_root / relative, self.group_record['files'][relative]['sha256'])

    def _table_record(self, binding):
        return _read(self._check(self.root / binding['path'], binding['sha256']))

    def _source_members(self, record, rows, role):
        keys = np.load(self._check(self.root/record['source_keys']['path']), mmap_mode='r', allow_pickle=False)[rows].astype('U')
        if record['schema'] == 'frozen_global_typed_cell_arrays_v1':
            metadata = record['metadata']
            coordinate = record['coordinates'][role]
        else:
            metadata = dict(atlas_id='macaque_composite_source_atlas', atlas_version='1.0',
                source_version=record['namespace'], population=record['recipe']['population'])
            coordinate = record['positions']
            if role != 'rf':
                raise ValueError('固定群体没有所要求的坐标角色')
        xy = np.load(self._check(self.root/coordinate['path']), mmap_mode='r', allow_pickle=False)[rows]
        ids = np.asarray([global_cell_id(metadata['atlas_id'], metadata['atlas_version'],
            metadata['source_version'], metadata['population'], str(key)) for key in keys])
        return ids, keys, xy

    def _add_geometry(self, name, geometry):
        count = len(geometry['ids'])
        if (geometry['xy_mm'].shape != (count, 2) or not np.isfinite(geometry['xy_mm']).all()
                or len(np.unique(geometry['ids'])) != count):
            raise ValueError('保存的空间坐标或 ID 不一致：'+name)
        self.geometry[name] = geometry
        self.populations[name] = dict(count=count, coordinate_role=geometry['coordinate_role'],
            representation=geometry['representation'], fields={}, primary=None,
            member_states_independent=geometry['representation']=='independently_solved_BC_cell')

    def _load_functional_pools(self):
        # The AC lineage is frozen unchanged across the two saved results. Only
        # its geometry/matrices are borrowed, never the historical neural states.
        old_seed = _dictionary(_read(self.report/'joint_gray_background/seed/manifest.json')['payload'])
        if self.seed_metadata['ac_vs_lineage'] != old_seed['ac_vs_lineage']:
            raise ValueError('已保存的 AC 功能核不属于当前谱系')
        self.view_root = self.report/'joint_short_view'
        manifest = _read(self.view_root/'manifest.json')
        self.view = manifest['positions']['joint']
        for relative in (self.view['geometry'], self.view['connections']):
            self._check(self.view_root/relative, manifest['files'][relative]['sha256'])
        with np.load(self.view_root/self.view['geometry'], allow_pickle=False) as saved:
            for name, meta in self.view['populations'].items():
                if not name.startswith(('same_', 'cross_')):
                    continue
                self._add_geometry(name, dict(ids=saved[name+'__ids'].astype('U'),
                    xy_mm=saved[name+'__xy']*.223, members=saved[name+'__members'],
                    subtypes=saved[name+'__subtypes'].astype('U'),
                    coordinate_role='effective_functional_RF_pool_not_anatomical_AC',
                    representation='effective_functional_pool', member_ids=np.array([], dtype='U1'),
                    member_rows=np.array([], dtype=int), member_xy_mm=np.empty((0, 2)),
                    member_group=np.array([], dtype=int), member_coordinate_role='not_anatomical'))

    def _add_route(self, name, pre, post, matrix, semantics):
        matrix = matrix.tocsr(copy=True)
        matrix.eliminate_zeros()
        expected = (self.populations[post]['count'], self.populations[pre]['count'])
        if matrix.shape != expected or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
            raise ValueError('保存的稀疏连接形状或权重不一致：'+name)
        self.routes[name] = dict(pre=pre, post=post, matrix=matrix, semantics=semantics)
        self.incoming.setdefault(post, []).append((pre, matrix.indptr, matrix.indices))

    def _member_map(self, name):
        if name not in BC:
            return self._layouts[name]['fine_to_group']
        rows = self.geometry[name]['member_rows']
        result = np.full(len(self._layouts[name]['fine_to_group']), -1, dtype=int)
        result[rows] = np.arange(len(rows))
        return result

    def _load_routes(self):
        fine_path = self.report/'executions/assemble_fine_01/FINE_ASSEMBLY_RESULT.json'
        configuration = _read(self.root/'configurations/research/BC_macaque_fullchain.json')
        self._check(fine_path, configuration['fine_routes']['sha256'])
        fine = _read(fine_path)
        removed_pairs = {'cones_to_HC', 'HC_to_cones'}
        for route in self.group_record['routes']:
            name, pre, post = route['name'], route['pre'], route['post']
            if pre not in self.populations or post not in self.populations or name in removed_pairs:
                continue
            if pre == 'local_feedforward_AC' and post != 'small_bistratified_S_ON':
                continue
            if pre in BC or post in BC:
                record = fine['routes'][name]
                directory = fine_path.parent/'fine_routes'/name
                declared = np.load(self._check(directory/'post_source_rows.npy', record['post_rows_sha256']), allow_pickle=False)
                weights = sparse.load_npz(self._check(directory/'weights.npz', record['weights_sha256'])).tocsr()
                target = np.sort(self.geometry[post]['member_rows'])
                indices = np.searchsorted(declared, target)
                if np.any(indices >= len(declared)) or not np.array_equal(declared[indices], target):
                    raise ValueError('保存连接缺少真实目标成员：'+name)
                edges = weights[indices].tocoo()
                pg = self._member_map(post)[target[edges.row]]
                cg = self._member_map(pre)[edges.col]
                included = cg >= 0
                if route['kind'] != 'gap' and not np.all(included):
                    raise ValueError('保存连接缺少化学祖先：'+name)
                shape = (self.populations[post]['count'], self.populations[pre]['count'])
                matrix = sparse.csr_matrix((edges.data[included], (pg[included], cg[included])), shape=shape)
                matrix = sparse.diags(1/self.geometry[post]['members']) @ matrix
                semantics = 'frozen fine contacts projected by saved membership; no crop renormalization'
            else:
                with np.load(self._group_file('routes/'+name+'/mean_weights.npz'), allow_pickle=False) as saved:
                    matrix = sparse.csr_matrix((saved['data'], saved['indices'], saved['indptr']), shape=saved['shape'])
                semantics = 'saved grouped route'
            self._add_route(name, pre, post, matrix, semantics)
        priority = self.root/'reports/CONE_HC_FINAL_IMPLEMENTATION_2026-11-02/h2_priority_assets'
        receipt = _read(priority/'RECEIPT.json')
        matrix = sparse.load_npz(self._check(priority/'A_group.npz', receipt['output_files']['A_group.npz']))
        actual_identity = canonical_hash(dict(shape=matrix.shape, data=matrix.data,
            indices=matrix.indices, indptr=matrix.indptr))
        if actual_identity != self.identity_evidence['A_identity']:
            raise ValueError('当前 H1/H2 输入矩阵内容身份不一致')
        self._add_route('cones_to_HC', 'cones', 'HC', matrix, 'current eccentric H1 and complete-contact H2 priority')
        feedback = sparse.load_npz(self._asset('B_group.npz'))
        actual_identity = canonical_hash(dict(shape=feedback.shape, data=feedback.data,
            indices=feedback.indices, indptr=feedback.indptr))
        if actual_identity != self.identity_evidence['B_identity']:
            raise ValueError('当前 HC 反馈矩阵内容身份不一致')
        self._add_route('HC_to_cones', 'HC', 'cones', feedback, 'saved native feedback')
        q = -sparse.load_npz(self._asset('H2_Q.npz')).tocsr()
        q.setdiag(0); q.eliminate_zeros()
        edge = q.tocoo()
        matrix = sparse.csr_matrix((edge.data, (self.h2_rows[edge.row], self.h2_rows[edge.col])),
            shape=(len(self.geometry['HC']['ids']),)*2)
        self._add_route('H2_homotypic', 'HC', 'HC', matrix, 'saved H2 coupling topology')
        with np.load(self.view_root/self.view['connections'], allow_pickle=False) as saved:
            for route in self.view['routes']:
                pre, post = route['pre'], route['post']
                if not (pre.startswith(('same_', 'cross_')) or post.startswith(('same_', 'cross_'))):
                    continue
                token = route['token']
                matrix = sparse.csr_matrix((saved[token+'_weights'], saved[token+'_indices'], saved[token+'_indptr']),
                    shape=route['shape'])
                if pre in BC:
                    original = self.bc[pre+'__original_group_rows']
                    parent_matrix = matrix
                    matrix = parent_matrix[:, original] @ sparse.diags(1/self._layouts[pre]['counts'][original])
                    incidence = sparse.csr_matrix((np.ones(len(original)), (np.arange(len(original)), original)),
                        shape=(len(original), parent_matrix.shape[1]))
                    if np.max(np.abs((matrix@incidence-parent_matrix).data), initial=0.) > 2e-14:
                        raise ValueError('AC 原始成员均值投影改变了保存的功能核')
                self._add_route(route['name'], pre, post, matrix,
                    'unchanged AC lineage saved functional weights; original-member mean')

    def network(self, target=None):
        return ancestors(self.populations, self.incoming, target or self.target)

    def h1_inputs(self, row=None):
        """Return a saved complete fine-H1 witness, not an inferred group kernel."""
        row = int(self.h1_witness_rows[0] if row is None else row)
        if row not in self.h1_witness_rows:
            raise ValueError('该 H1 成员的完整短/长分量未保存；可选成员行：'+str(self.h1_witness_rows.tolist()))
        prefix = 'row_'+str(row)+'__'
        with np.load(self._asset('H1_COMPLETE_ROW_WITNESSES.npz'), allow_pickle=False) as saved:
            values = {key: saved[prefix+key] for key in ('official_source_rows', 'cone_pedicle_mm',
                'cone_types', 'distance_mm', 'short_unitmass', 'long_unitmass', 'A')}
        short, long = .15*values['short_unitmass'], .85*values['long_unitmass']
        np.testing.assert_allclose(short+long, values['A'], rtol=0, atol=2e-17)
        np.testing.assert_allclose([short.sum(), long.sum()], [.15, .85], rtol=0, atol=2e-14)
        source_record = _read(self.root/'reports/CONE_HC_OUTER_SOURCE_EXTENSION_20261102_v3/cones_01/RESULT.json')
        key_template = source_record['source_key_rule']
        ids = np.asarray([key_template.replace('<official_source_row>', str(int(i))) for i in values['official_source_rows']])
        group = int(self.members['hc_group'][row])
        return dict(xy_mm=values['cone_pedicle_mm'], ids=ids, subtypes=values['cone_types'],
            short=short, long=long, total=values['A'], short_unitmass=values['short_unitmass'],
            long_unitmass=values['long_unitmass'], distance_mm=values['distance_mm'],
            anchor_mm=self.members['hc_anchor_mm'][row], coordinate_role='OPL_pedicle_mm', unit='dimensionless',
            metadata=dict(fine_row=row, hc_id=str(self.geometry['HC']['member_ids'][row]),
                hc_source_key=str(self.members['hc_keys'][row].decode('ascii')),
                group_row=group, group_id=str(self.geometry['HC']['ids'][group]),
                group_members=int(self.geometry['HC']['members'][group]),
                lambda_short_mm=float(self.members['lambda_short_mm'][row]),
                lambda_long_mm=float(self.members['lambda_long_mm'][row]),
                support_radius_mm=float(self.members['support_radius_mm'][row]),
                short_contribution=float(short.sum()), long_contribution=float(long.sum()),
                meaning='saved complete input weights for one actual H1 member; response belongs to its representative group',
                response_independently_recorded=False, anatomical_connections_added=False))


def load_geometry(project_root, current_source):
    return SavedGeometry(project_root, current_source)
