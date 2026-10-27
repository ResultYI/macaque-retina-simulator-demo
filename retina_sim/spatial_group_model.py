"""Solver-independent assembly of the v0.2 representative spatial model.

This is a structural interface, not a simulator. It allocates no numerical initial
state, does not import a device backend and never promotes fine checkpoints to a
group checkpoint. Observation times and mechanism clocks are separate concepts.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy import sparse

from .contracts import ContractError
from .spatial_group_graph import digest


SCHEMA = 'retipath-static-spatial-group-graph-2'
BC = ('midget_BC_ON', 'midget_BC_OFF', 'S_ON_BC', 'diffuse_BC_ON', 'diffuse_BC_OFF')
AC = ('local_feedforward_AC', 'local_crossover_AC', 'widefield_AC')
RGC = ('midget_RGC_ON', 'midget_RGC_OFF', 'parasol_RGC_ON', 'parasol_RGC_OFF',
       'small_bistratified_S_ON')
POPULATIONS = ('cones', 'HC', 'rods', 'rod_BC', 'AII_AC', *BC, *AC, *RGC)
GEOMETRY_FIELDS = ('leaf_level', 'leaf_width_deg', 'polygon_deg', 'covariance_deg2',
    'lambda_max_deg2', 'centroid_deg', 'leaf_keys', 'precision_status', 'requested_fmax_cpd',
    'covariance_linear_bound', 'admissible_linear_average_fmax_cpd', 'original_singleton_row')
CONTACT_STATES = {'individual_rod_RBC_contacts': ('m',), 'RBC_AII_AMPA': ('a', 'b'),
                  'AII_FMB_glycine': ('a', 'b'), 'AII_DB1_glycine': ('a', 'b')}
REQUIRED_ROUTE_PAIRS = (
    *(('cones', n) for n in BC),
    ('midget_BC_ON', 'midget_RGC_ON'), ('midget_BC_OFF', 'midget_RGC_OFF'),
    ('diffuse_BC_ON', 'local_feedforward_AC'), ('diffuse_BC_OFF', 'local_crossover_AC'),
    ('diffuse_BC_ON', 'widefield_AC'), ('S_ON_BC', 'small_bistratified_S_ON'),
    ('local_crossover_AC', 'midget_RGC_ON'), ('local_feedforward_AC', 'midget_RGC_OFF'),
    ('local_feedforward_AC', 'small_bistratified_S_ON'), ('diffuse_BC_OFF', 'small_bistratified_S_ON'),
    ('widefield_AC', 'parasol_RGC_ON'), ('diffuse_BC_ON', 'parasol_RGC_ON'),
    ('widefield_AC', 'parasol_RGC_OFF'), ('diffuse_BC_OFF', 'parasol_RGC_OFF'),
    ('rods', 'rod_BC'), ('rod_BC', 'AII_AC'), ('AII_AC', 'midget_BC_OFF'),
    ('AII_AC', 'diffuse_BC_OFF'), ('cones', 'HC'), ('HC', 'cones'), ('AII_AC', 'AII_AC'),
    *((pre, post) for n in ('midget_BC_ON', 'diffuse_BC_ON', 'S_ON_BC')
      for pre, post in (('AII_AC', n), (n, 'AII_AC'))))


def recording_contract():
    """The existing FovealDemoRecorder field/selection policy, as plain JSON.

    Primary fields cover every computed representative, while auxiliary fields
    retain the recorder's frozen original-fine-cell selection. Fine members remain
    traceable but are never labelled as independently recorded cells.
    """
    inner = {n: ['voltage', 'activation', 'effective_release'] for n in BC}
    inner.update({n: ['activation', 'inhibition_state'] for n in AC})
    inner.update({n: ['voltage'] for n in RGC})
    inner.update({n: ['voltage', 'activation', 'release'] for n in ('rod_BC', 'AII_AC')})
    fields = {'cones': ['Vis', 'X', 'C', 'g_i', 'Vbar', 'R', 'E', 'q', 'release', 'Ios'],
              'HC': ['Vh', 'w', 'z2'], **inner}
    primary = {n: 'voltage' for n in inner}
    primary.update({n: 'inhibition_state' for n in AC})
    primary.update(cones='Vis', HC='Vh')
    return dict(schema='retipath-group-observation-contract-1', fields=fields,
                primary=primary, observation_interval_seconds=0.0005,
                observation_interval_role='recording schedule only; not a solver step',
                primary_selection='all computed groups',
                auxiliary_selection='tools.cone_hc_demo_recording.selected_rows',
                auxiliary_selection_coordinates='original selected fine-cell positions; mapped to groups without deduplicating',
                auxiliary_limit=900, dtype='float64',
                identity='representative_group; fine member predictions are not independent recordings',
                events='independent sampled-object events and RNG, when requested; no replicated fine spikes',
                history='retain causal history, delay queues, analytic anchors, event/RNG state and phase',
                z1='only explicitly selected auxiliary cone nodes; w does not recover arbitrary z1 history',
                checkpoint='new group schema must bind artifact hash, state layout, parameter binding, '
                           'all clocks/queues/anchors/events/RNG; fine checkpoints are incompatible',
                storage='bounded time windows; exact observation times; geometry and IDs stored once')


@dataclass(frozen=True)
class StateBlock:
    population: str
    mechanism: str
    name: str
    unit: str
    meaning: str
    count: int
    offset: int
    role: str = 'dynamic'

    @property
    def state_slice(self):
        return slice(self.offset, self.offset + self.count)


@dataclass(frozen=True)
class PopulationLayout:
    name: str
    group_ids: np.ndarray
    counts: np.ndarray
    member_offsets: np.ndarray
    member_global_rows: np.ndarray
    fine_to_group: np.ndarray
    centroid_mm: np.ndarray
    coordinate_role: str
    source_table: dict
    source_keys: dict
    parameter_source: str
    unresolved_parameter_fields: tuple
    mesh_geometry: dict
    recording_auxiliary_fine_rows: np.ndarray
    recording_auxiliary_group_rows: np.ndarray

    @property
    def group_count(self):
        return len(self.counts)

    def group_for_fine_row(self, row):
        if not isinstance(row, (int, np.integer)) or row < 0 or row >= len(self.fine_to_group):
            raise ContractError('Original row outside the frozen source population')
        result = int(self.fine_to_group[row])
        if result < 0:
            raise ContractError('Original row is outside this experiment closure')
        return result


@dataclass(frozen=True)
class RouteLayout:
    name: str
    pre: str
    post: str
    kind: str
    shape: tuple
    group_pair_entries: int
    contact_component: str | None
    contact_groups: int
    fine_contacts: int
    coefficient_sum_S: float | None
    sum_semantics: str
    mean_semantics: str


@dataclass(frozen=True)
class RecordingField:
    population: str
    component: str
    name: str
    unit: str
    meaning: str
    rows: np.ndarray
    primary: bool
    coordinate_role: str
    original_fine_rows: np.ndarray | None = None
    identity: str = 'representative_group'

    @property
    def key(self):
        return self.component + '.' + self.name


@dataclass(frozen=True)
class AdvanceRequest:
    """Exact required boundaries; intentionally no global integration step."""
    start_seconds: float
    stop_seconds: float
    observation_times: tuple
    input_boundaries: tuple

    def __post_init__(self):
        if (not math.isfinite(self.start_seconds) or not math.isfinite(self.stop_seconds)
                or self.stop_seconds <= self.start_seconds):
            raise ContractError('Finite increasing advance interval required')
        for label, values in (('observation', self.observation_times), ('input', self.input_boundaries)):
            a = np.asarray(values, dtype=np.float64)
            if (a.ndim != 1 or not np.isfinite(a).all() or np.any(np.diff(a) <= 0)
                    or np.any(a < self.start_seconds) or np.any(a > self.stop_seconds)):
                raise ContractError('Exact ordered ' + label + ' boundaries within the interval required')


class TimeAdvanceBackend(Protocol):
    """Future backends own their time policy and must respect all exact clocks."""
    def advance(self, assembly: 'GroupModelAssembly', prepared_state, inputs,
                request: AdvanceRequest): ...


def _verify_artifact(directory):
    root = Path(directory).resolve()
    raw = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
    expected = raw.pop('manifest_sha256', None)
    actual = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    if (raw.get('schema') != SCHEMA or expected != actual
            or raw.get('runtime_ready') is not False
            or raw.get('model_forward_started') is not False):
        raise ContractError('Completed static v2 artifact identity/classification required')
    if set(raw.get('populations', {})) != set(POPULATIONS):
        raise ContractError('All 18 neuronal populations, including complete rod upstream, required')
    for name, row in raw.get('files', {}).items():
        path = (root / name).resolve()
        if (not path.is_relative_to(root) or not path.is_file()
                or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']):
            raise ContractError('Static v2 array identity differs: ' + name)
    return root, raw, actual


class GroupModelAssembly:
    """Verified geometry/connectivity and uninitialized state/observation layout."""
    def __init__(self, directory):
        self.root, self.manifest, self.artifact_sha256 = _verify_artifact(directory)
        self.populations = {}
        for name in POPULATIONS:
            meta = self.manifest['populations'][name]
            arrays = {key: self.array('populations/' + name + '/' + key + '.npy')
                      for key in ('group_ids', 'counts', 'member_offsets', 'member_global_rows',
                                  'fine_to_group', 'centroid_mm')}
            n, fine = meta['group_count'], meta['fine_neurons']
            ids, counts, offsets, members, mapping, xy = arrays.values()
            if (ids.shape != (n,) or ids.dtype.kind not in 'US' or len(np.unique(ids)) != n
                    or counts.shape != (n,) or counts.dtype.kind not in 'iu' or np.any(counts <= 0)
                    or offsets.shape != (n + 1,) or offsets.dtype.kind not in 'iu'
                    or offsets[0] != 0 or np.any(np.diff(offsets) != counts)
                    or members.shape != (fine,) or members.dtype.kind not in 'iu'
                    or len(np.unique(members)) != fine or int(offsets[-1]) != fine
                    or mapping.shape != (meta['full_asset_count'],) or mapping.dtype.kind != 'i'
                    or np.any(mapping < -1) or np.any(mapping >= n)
                    or np.any(members < 0) or np.any(members >= len(mapping))
                    or xy.shape != (n, 2) or xy.dtype != np.float64 or not np.isfinite(xy).all()):
                raise ContractError('Group membership/geometry layout differs: ' + name)
            if (np.count_nonzero(mapping >= 0) != fine
                    or not np.array_equal(mapping[members], np.repeat(np.arange(n), counts))):
                raise ContractError('Fine membership does not cover the complete selected upstream: ' + name)
            auxiliary_fine = self.array('populations/' + name + '/recording_auxiliary_fine_rows.npy')
            auxiliary_group = self.array('populations/' + name + '/recording_auxiliary_group_rows.npy')
            if (auxiliary_fine.ndim != 1 or not 0 < len(auxiliary_fine) <= 900
                    or auxiliary_fine.dtype.kind not in 'iu' or auxiliary_group.dtype.kind not in 'iu'
                    or auxiliary_group.shape != auxiliary_fine.shape or np.any(np.diff(auxiliary_fine) <= 0)
                    or np.any(auxiliary_fine < 0) or np.any(auxiliary_fine >= len(mapping))
                    or np.any(auxiliary_group < 0) or np.any(auxiliary_group >= n)
                    or not np.array_equal(mapping[auxiliary_fine], auxiliary_group)):
                raise ContractError('Frozen original auxiliary recording selection differs: ' + name)
            self.populations[name] = PopulationLayout(name, **arrays,
                coordinate_role=meta['coordinate_role'], source_table=meta['source_table'],
                source_keys=meta['source_keys'], parameter_source=meta['parameter_class_source'],
                unresolved_parameter_fields=tuple(meta['derived_parameter_fields']),
                mesh_geometry=self._geometry(name, n, counts, members, offsets),
                recording_auxiliary_fine_rows=auxiliary_fine,
                recording_auxiliary_group_rows=auxiliary_group)
        self.routes = self._routes()
        self.state_blocks = self._states()
        self.state_scalar_count = sum(b.count for b in self.state_blocks)
        self.probability_blocks = self._probability_blocks()
        self.probability_scalar_count = sum(b.count for b in self.probability_blocks)
        selected_z1_count = len(self.populations['cones'].recording_auxiliary_fine_rows)
        self.auxiliary_z1_blocks = (StateBlock('cones', 'selected_original_z1_auxiliary', 'z1_aux',
            'vH_native_voltage', 'selected_original_cone_transmitter_drive', selected_z1_count, 0,
            'uninitialized_selected_original_auxiliary'),)
        self.auxiliary_z1_scalar_count = selected_z1_count
        budget = self.manifest['budget_audit']
        if (self.state_scalar_count != budget['core_state_scalars']
                or self.probability_scalar_count != budget['extra_RGC_history_probability_scalars']
                or self.auxiliary_z1_scalar_count != budget['extra_selected_z1_auxiliary_scalars']
                or sum(p.group_count for p in self.populations.values()) != budget['cell_slots']):
            raise ContractError('State layout differs from the actual static capacity ledger')
        self.recording_contract = recording_contract()
        if ('recording_contract' in self.manifest
                and self.manifest['recording_contract'] != self.recording_contract):
            raise ContractError('Recorded field/selection contract changed')
        self.recording_fields = self._recording_fields()
        self.mechanism_clocks = self.manifest.get('mechanism_clocks', {})
        self.unresolved_bindings = tuple((n, p.unresolved_parameter_fields)
                                        for n, p in self.populations.items())
        self.runtime_ready = False

    def _geometry(self, population, count, members_per_group, members, offsets):
        prefix = 'populations/' + population + '/'
        geometry = {name: self.array(prefix + name + '.npy') for name in GEOMETRY_FIELDS}
        keys = self.array(prefix + 'keys.npy')
        leaves = geometry['leaf_keys']
        if (keys.ndim != 2 or keys.shape[0] != count or keys.dtype.kind not in 'iu'
                or leaves.ndim != 2 or leaves.shape[1] != 3 or leaves.dtype.kind not in 'iu'
                or np.any(keys[:, 0] < 0) or np.any(keys[:, 0] >= len(leaves))):
            raise ContractError('Explicit quadtree leaf membership required: ' + population)
        shapes = {'centroid_deg': (count, 2), 'polygon_deg': (count, 4, 2),
                  'covariance_deg2': (count, 2, 2)}
        for name, value in geometry.items():
            if name != 'leaf_keys' and value.shape != shapes.get(name, (count,)):
                raise ContractError('Group geometry shape differs: ' + population + '.' + name)
            if value.dtype.kind == 'f' and name != 'admissible_linear_average_fmax_cpd' and not np.isfinite(value).all():
                raise ContractError('Nonfinite group geometry: ' + population + '.' + name)
        if (not np.array_equal(geometry['leaf_level'], leaves[keys[:, 0], 0])
                or np.any(geometry['leaf_width_deg'] <= 0) or np.any(geometry['lambda_max_deg2'] < 0)
                or np.any(geometry['requested_fmax_cpd'] < 0)
                or np.any(geometry['admissible_linear_average_fmax_cpd'] < 0)
                or np.isnan(geometry['admissible_linear_average_fmax_cpd']).any()
                or not np.array_equal(geometry['original_singleton_row'],
                    np.where(members_per_group == 1, members[offsets[:-1]], -1))):
            raise ContractError('Group leaf/precision/member identity differs: ' + population)
        return geometry

    def describe(self):
        """Small JSON receipt of this assembly; contains no simulated values."""
        return dict(schema='retipath-spatial-group-assembly-1', artifact_sha256=self.artifact_sha256,
            runtime_ready=False, model_forward_started=False, state_values_initialized=False,
            numerical_dynamics_implemented=False, qualified_initial_state_available=False,
            time_backend=dict(bound=False, integration_step_seconds=None,
                              policy='replaceable; exact input, observation and mechanism boundaries'),
            mechanism_clocks=self.mechanism_clocks,
            populations={n: dict(group_count=p.group_count, fine_members=len(p.member_global_rows),
                source_table=p.source_table, source_keys=p.source_keys, coordinate_role=p.coordinate_role,
                singleton_groups=int(np.count_nonzero(p.counts == 1)),
                parameter_source=p.parameter_source, unresolved_parameter_fields=list(p.unresolved_parameter_fields),
                geometry_fields=list(p.mesh_geometry)) for n, p in self.populations.items()},
            core_state_scalars=self.state_scalar_count, state_blocks=[asdict(b) for b in self.state_blocks],
            extra_RGC_history_probability_scalars=self.probability_scalar_count,
            probability_blocks=[asdict(b) for b in self.probability_blocks],
            extra_selected_z1_auxiliary_scalars=self.auxiliary_z1_scalar_count,
            auxiliary_z1_blocks=[asdict(b) for b in self.auxiliary_z1_blocks],
            routes={n: asdict(r) for n, r in self.routes.items()},
            observations=[dict(population=f.population, key=f.key, unit=f.unit, meaning=f.meaning,
                recorded_group_count=len(f.rows), primary=f.primary, coordinate_role=f.coordinate_role,
                identity=f.identity, original_selected_fine_rows=None if f.original_fine_rows is None
                else f.original_fine_rows.tolist()) for f in self.recording_fields],
            recording_contract=self.recording_contract,
            missing_registration=self.manifest.get('target_refinement', {}),
            unresolved=['derived parameter bindings', 'qualified group initial state and complete history',
                        'time advancement implementation and numerical/spatial acceptance'])

    def _path(self, relative):
        if relative not in self.manifest['files']:
            raise ContractError('Array is not bound by the static manifest: ' + relative)
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ContractError('Static array path escapes the artifact')
        return path

    def array(self, relative):
        return np.load(self._path(relative), allow_pickle=False, mmap_mode='r')

    def matrix(self, relative):
        with np.load(self._path(relative), allow_pickle=False) as z:
            if set(z.files) != {'data', 'indices', 'indptr', 'shape'}:
                raise ContractError('Explicit CSR array layout required')
            data, indices, indptr, shape = (z[k] for k in ('data', 'indices', 'indptr', 'shape'))
        if (data.dtype != np.float64 or not np.isfinite(data).all() or indices.dtype.kind not in 'iu'
                or indptr.dtype.kind not in 'iu' or shape.shape != (2,) or shape.dtype.kind not in 'iu'
                or np.any(shape < 0) or indptr.shape != (int(shape[0]) + 1,)
                or indptr[0] != 0 or indptr[-1] != len(data) or len(indices) != len(data)
                or np.any(np.diff(indptr) < 0) or np.any(indices < 0) or np.any(indices >= shape[1])):
            raise ContractError('Invalid finite FP64 CSR matrix: ' + relative)
        matrix = sparse.csr_matrix((data, indices, indptr), shape=tuple(shape))
        matrix.data.flags.writeable = matrix.indices.flags.writeable = matrix.indptr.flags.writeable = False
        return matrix

    def _routes(self):
        result, contacts, pairs = {}, {}, set()
        for meta in self.manifest['routes']:
            name, pre, post = meta['name'], meta['pre'], meta['post']
            if name in result or pre not in self.populations or post not in self.populations:
                raise ContractError('Unknown or repeated group route')
            pairs.add((pre, post))
            shape = (self.populations[post].group_count, self.populations[pre].group_count)
            total = self.matrix('routes/' + name + '/sum_weights.npz')
            mean = self.matrix('routes/' + name + '/mean_weights.npz')
            if total.shape != shape or mean.shape != shape or np.any(total.data < 0) or np.any(mean.data < 0):
                raise ContractError('Group route shape/sign differs: ' + name)
            expected = sparse.diags(1. / self.populations[post].counts) @ total
            residual = mean - expected
            if residual.nnz and np.max(np.abs(residual.data)) > 1e-12 * max(1., np.max(np.abs(mean.data), initial=0.)):
                raise ContractError('Post-member mean normalization differs: ' + name)
            contact = meta.get('contact', {})
            component = contact.get('component')
            if component:
                if component in contacts or component not in CONTACT_STATES:
                    raise ContractError('Unknown or repeated physical contact component')
                count = contact['groups']
                contacts[component] = count
                cp = 'routes/' + name + '/'
                post_rows, pre_rows, offsets = (self.array(cp + k + '.npy') for k in
                    ('contact_post_group', 'contact_pre_group', 'contact_member_offsets'))
                fine, mapping, members = (self.array(cp + k + '.npy') for k in
                    ('fine_contact_global_rows', 'fine_contact_to_group', 'contact_member_global_rows'))
                coefficient = self.matrix(cp + 'contact_coefficient_S.npz')
                if (post_rows.shape != (count,) or pre_rows.shape != (count,)
                        or post_rows.dtype.kind not in 'iu' or pre_rows.dtype.kind not in 'iu'
                        or offsets.shape != (count + 1,) or offsets.dtype.kind not in 'iu' or offsets[0] != 0
                        or np.any(np.diff(offsets) <= 0) or offsets[-1] != contact['fine_contacts']
                        or np.any(post_rows < 0) or np.any(post_rows >= shape[0])
                        or np.any(pre_rows < 0) or np.any(pre_rows >= shape[1])
                        or coefficient.shape != shape or np.any(coefficient.data < 0)
                        or contact['coefficient_unit'] != 'S'
                        or not np.isclose(coefficient.sum(), contact['coefficient_sum'], rtol=1e-12, atol=0.)):
                    raise ContractError('Physical contact layout/totals differ: ' + name)
                if (fine.shape != (contact['fine_contacts'],) or members.shape != fine.shape
                        or mapping.shape != fine.shape or fine.dtype.kind not in 'iu'
                        or members.dtype.kind not in 'iu' or mapping.dtype.kind not in 'iu'
                        or np.any(fine < 0) or len(np.unique(fine)) != len(fine)
                        or np.any(mapping >= count) or np.any(mapping < 0)
                        or len(np.unique(post_rows * shape[1] + pre_rows)) != count
                        or not np.array_equal(members, fine[np.argsort(mapping, kind='stable')])
                        or not np.array_equal(np.bincount(mapping, minlength=count), np.diff(offsets))):
                    raise ContractError('Physical contact member mapping differs: ' + name)
            if total.nnz != meta['group_pair_entries']:
                raise ContractError('Group route pair ledger differs: ' + name)
            result[name] = RouteLayout(name, pre, post, meta['kind'], shape, total.nnz,
                component, contact.get('groups', 0), contact.get('fine_contacts', 0),
                contact.get('coefficient_sum'), meta['sum_semantics'], meta['mean_semantics'])
        if contacts != self.manifest['budget_audit']['contact_groups'] or set(contacts) != set(CONTACT_STATES):
            raise ContractError('Complete physical contact dynamics layout required')
        if pairs != set(REQUIRED_ROUTE_PAIRS):
            raise ContractError('Complete original feedforward, inhibition and reciprocal feedback routes required')
        gap = self.manifest['gap']
        shape = sum(self.populations[n].group_count for n in gap['population_order'])
        laplacian = self.matrix('gap/laplacian_S.npz')
        if laplacian.shape != (shape, shape) or laplacian.nnz != gap['laplacian_entries']:
            raise ContractError('Electrical feedback operator layout differs')
        asymmetry = laplacian - laplacian.T
        scale = np.max(np.abs(laplacian.data), initial=0.)
        if ((asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1e-12 * scale)
                or np.max(np.abs(np.asarray(laplacian.sum(axis=1))), initial=0.) > 1e-12 * scale):
            raise ContractError('Electrical feedback symmetry or current conservation differs')
        if ('runtime_edge_entries' in self.manifest['budget_audit'] and
                sum(r.group_pair_entries for r in result.values() if r.kind != 'gap') + laplacian.nnz
                != self.manifest['budget_audit']['runtime_edge_entries']):
            raise ContractError('Runtime edge ledger differs from retained routes')
        return result

    def _states(self):
        result, offset = [], 0

        def add(pop, mechanism, declarations, role='dynamic', count=None):
            nonlocal offset
            count = self.populations[pop].group_count if count is None else count
            for name, unit, meaning in declarations:
                result.append(StateBlock(pop, mechanism, name, unit, meaning, count, offset, role))
                offset += count

        add('cones', 'reduced_native', [('X', 'vH_native_precursor', 'native_precursor'),
            ('C', 'vH_native_current', 'native_os_current'), ('Vis', 'vH_native_voltage', 'native_relative_voltage'),
            ('g_i', 'vH_native_conductance', 'native_conductance'), ('Vbar', 'vH_native_voltage', 'native_relative_voltage')])
        add('cones', 'analytic_RE', [(n, 'td', 'native_illuminance') for n in ('R', 'E')], 'analytic_anchor')
        add('HC', 'reduced_native', [('w', 'vH_native_voltage', 'aggregated_transmitter_drive'),
            ('z2', 'vH_native_voltage', 'effective_transmitter_drive'), ('Vh', 'vH_native_voltage', 'native_relative_voltage')])
        add('rods', 'Chen2024PrimateRodCurrent', list(zip(('R', 'P', 'G', 'C', 'S'),
            ('s^-2', 's^-1', 'Chen_scaled_concentration', 'Chen_scaled_concentration', 'Chen_scaled_concentration/s'),
            ('effective_opsin_activity', 'effective_pde_activity', 'source_scaled_cgmp', 'source_scaled_calcium', 'source_scaled_cgmp_synthesis'))))
        membrane = [('V', 'V', 'physical_membrane_voltage'), ('release', 'dimensionless', 'effective_release_activation')]
        add('rods', 'rod_membrane', membrane)
        for n in ('rod_BC', 'AII_AC'):
            add(n, 'ReceptorConductanceMembrane', membrane + [('x', 'dimensionless', 'adaptation_gate')])
        for n in BC:
            add(n, 'HistoryBipolarTerminal', [('history_' + str(i), 'dimensionless', 'causal_synaptic_drive_history')
                for i in range(3)] + [('activation', 'dimensionless', 'effective_synaptic_activation')], 'causal_history')
            add(n, 'HistoryBipolarTerminal', [('V', 'V', 'physical_membrane_voltage'),
                ('terminal_release', 'dimensionless', 'effective_release_activation'),
                ('A', 'dimensionless', 'release_adaptation_activity')])
        for n in AC:
            add(n, 'DynamicAmacrineInhibition', [('h', 'dimensionless', 'effective_inhibitory_activity')])
        for n in RGC:
            add(n, 'RGC_spike_history', [('V', 'V', 'physical_membrane_voltage')])
        for route in self.routes.values():
            if route.contact_component:
                add(route.pre, route.contact_component,
                    [(n, 'dimensionless', 'receptor_gate') for n in CONTACT_STATES[route.contact_component]],
                    count=route.contact_groups)
        return tuple(result)

    def _probability_blocks(self):
        result, offset = [], 0
        for pop in RGC:
            for i in range(8):
                count = self.populations[pop].group_count
                result.append(StateBlock(pop, 'RGC_expected_history', 'history_probability_' + str(i),
                    'dimensionless', 'expected_bin_spike_probability', count, offset, 'probability_history'))
                offset += count
        return tuple(result)

    def _recording_fields(self):
        units = {(b.population, b.name): b.unit for b in self.state_blocks if b.mechanism not in CONTACT_STATES}
        meanings = {(b.population, b.name): b.meaning for b in self.state_blocks if b.mechanism not in CONTACT_STATES}
        units.update({('cones', n): 'dimensionless' for n in ('q', 'release')})
        units['cones', 'Ios'] = 'vH_native_current'
        result = []
        for pop, names in self.recording_contract['fields'].items():
            layout = self.populations[pop]
            for name in names:
                primary = name == self.recording_contract['primary'][pop]
                rows = np.arange(layout.group_count) if primary else layout.recording_auxiliary_group_rows
                rows.flags.writeable = False
                unit = units.get((pop, name), 'V' if name == 'voltage' else 'dimensionless')
                ports = {'voltage': 'physical_membrane_voltage', 'q': 'sigmoid_activation',
                    'release': 'effective_synaptic_drive' if pop == 'cones' else 'effective_synaptic_activation',
                    'effective_release': 'effective_release_activation', 'activation': 'effective_synaptic_activation',
                    'inhibition_state': 'effective_inhibitory_activity', 'Ios': 'native_outer_segment_current'}
                meaning = ports[name] if name in ports else meanings[pop, name]
                result.append(RecordingField(pop, 'outer_native' if pop in ('cones', 'HC') else pop,
                    name, unit, meaning, rows, primary, layout.coordinate_role,
                    original_fine_rows=None if primary else layout.recording_auxiliary_fine_rows,
                    identity='representative_group' if primary else 'group_prediction_at_original_selected_cell'))
        return tuple(result)

    def advance(self, prepared_state, inputs, request: AdvanceRequest, *, backend: TimeAdvanceBackend | None = None):
        if backend is None:
            raise ContractError('Time advancement backend is not provided; no numerical state was initialized')
        raise ContractError('Static group assembly cannot advance: derived parameter bindings, qualified '
                            'group initial/history state and numerical backend acceptance remain unresolved')
