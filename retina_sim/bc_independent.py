"""Independent BC cells on the existing finite graph and accepted Cone-HC model.

Only BC membership is refined. Fine anatomical routes are reprojected once;
the existing effective AC kernels still average the same original members.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
from scipy import sparse

from .contracts import ContractError, canonical_hash
from .disk_tables import FrozenDiskTable
from .disk_population import FixedDiskPopulation
from .group_inner_dynamics import BC_MECH
from .group_parameter_binding import bind_bc_physiology
from .hc_eccentric import checked_path, digest
from .h1_h2_effective import matrix_id
from .cone_hc_eccentric import _initialize_inner, EccentricHCFullChainRunner
from .spatial_group_model import BC


DEFAULT_CONFIGURATION = 'configurations/research/BC_macaque_fullchain.json'


def temporal_profile(configuration, population, eccentricity_deg):
    """Query a typed constant profile without inventing an eccentricity trend."""
    e = np.asarray(eccentricity_deg, dtype=np.float64)
    if population not in BC or not np.isfinite(e).all() or np.any((e < 0) | (e > 6)):
        raise ContractError('BC temporal profiles require a known type and 0..6 degree ancestry')
    p = deepcopy(configuration['temporal_profiles'][population])
    kernel = np.asarray(p['kernel'], dtype=np.float64)
    if (p['eccentricity_dependence'] != 'constant_unidentified'
            or p['status'] != 'retained_engineering_baseline'
            or kernel.shape != (3,) or not np.isfinite(kernel).all()
            or kernel[np.argmax(abs(kernel))] <= 0
            or p['sample_interval_seconds'] != .001
            or not np.isfinite([p['tau_release_seconds'], p['tau_adaptation_seconds']]).all()
            or p['tau_release_seconds'] <= 0 or p['tau_adaptation_seconds'] <= 0):
        raise ContractError('Explicit finite three-tap BC profile and original 1ms clock required')
    return p


def member_incidence(parent, child):
    old = parent.fine_to_group[child.member_global_rows]
    if np.any(old < 0):
        raise ContractError('BC refinement introduced a member outside its frozen parent')
    return sparse.csr_matrix((np.ones(len(old)), (old, np.arange(len(old)))),
                             shape=(parent.group_count, child.group_count))


def sparse_error(actual, expected):
    if actual.shape != expected.shape:
        raise ContractError('BC reprojection dimensions differ')
    return float(np.max(abs((actual-expected).data), initial=0.))


class IndependentBCAssembly:
    def __init__(self, parent, root, configuration):
        self.parent = parent
        self.repo_root = Path(root).resolve()
        self.manifest = deepcopy(parent.manifest)
        self.populations = dict(parent.populations)
        self._arrays, self._matrices = {}, {}
        self.bc_cells, self.contact_parent_rows = {}, {}
        self.projection_ledger = {}
        fine = configuration['fine_routes']
        receipt_path = checked_path(self.repo_root, fine['receipt'], fine['sha256'])
        receipt = json.loads(receipt_path.read_bytes())
        directory = receipt_path.parent / 'fine_routes'
        if receipt['status'] != 'FINE_BC_ROUTES_AND_GROUP_SELECTION_SAVED':
            raise ContractError('Completed frozen fine BC routes required')
        self.tables = {n: FrozenDiskTable(self.repo_root, parent.populations[n].source_table) for n in BC}
        cone = FrozenDiskTable(self.repo_root, parent.populations['cones'].source_table)
        for name in BC:
            old, table = parent.populations[name], self.tables[name]
            rows = old.member_global_rows.copy()
            n = len(rows)
            mapping = np.full(len(old.fine_to_group), -1, dtype=np.int64)
            mapping[rows] = np.arange(n)
            original_groups = old.fine_to_group[rows]
            ids = np.asarray(table.rows(rows).global_cell_ids)
            if len(np.unique(ids)) != n:
                raise ContractError('BC singleton identities are not unique')
            layout = replace(old, group_ids=ids, counts=np.ones(n, dtype=np.int64),
                member_offsets=np.arange(n+1), member_global_rows=rows, fine_to_group=mapping,
                centroid_mm=table.coordinates['rf'][rows].copy(),
                mesh_geometry={k: (v if k == 'leaf_keys' else v[original_groups])
                               for k, v in old.mesh_geometry.items()},
                recording_auxiliary_fine_rows=rows.copy(),
                recording_auxiliary_group_rows=np.arange(n))
            self.populations[name] = layout
            owner = table.extras['cone_row'][rows]
            unique_owner, owner_index = np.unique(owner, return_inverse=True)
            owner_ids = np.asarray(cone.rows(unique_owner).global_cell_ids)[owner_index]
            self.bc_cells[name] = dict(global_cell_ids=ids, source_rows=rows,
                original_group_rows=original_groups, source_cone_rows=owner.copy(),
                source_cone_ids=owner_ids,
                functional_ancestry_mm=table.coordinates['rf'][rows].copy(),
                input_anchor_OPL_mm=cone.coordinates['opl_pedicle'][owner].copy(),
                eccentricity_deg=np.linalg.norm(table.coordinates['rf'][rows], axis=1)/.223,
                soma_available=False, state_semantics='independently_solved_BC_cell',
                upstream_semantics='retained_grouped_Cone_and_HC_states')
            self.manifest['populations'][name].update(group_count=n, fine_neurons=n, singleton_groups=n)
            self.manifest['populations'][name]['coordinate'] = table.record['coordinates']['rf']
            self.manifest['populations'][name]['parent_population'] = None
            self._arrays[f'populations/{name}/keys.npy'] = parent.array(f'populations/{name}/keys.npy')[original_groups].copy()
            for key in ('counts', 'member_offsets', 'member_global_rows', 'fine_to_group', 'centroid_mm'):
                self._arrays[f'populations/{name}/{key}.npy'] = getattr(layout, key)
            for field in ('C', 'gL'):
                self.manifest['files'].pop(f'populations/{name}/{field}_total.npy', None)
        self.incidence = {n: member_incidence(parent.populations[n], self.populations[n]) for n in BC}
        self.routes = dict(parent.routes)
        gap_degrees, gap_exterior = {}, {}
        profiles = json.loads((self.repo_root/'configurations/effective_dynamics_v1.json').read_bytes())['profiles']
        aii = None
        for name, route in parent.routes.items():
            if route.pre not in BC and route.post not in BC:
                continue
            pre, post = self.populations[route.pre], self.populations[route.post]
            shape = (post.group_count, pre.group_count)
            prefix = 'routes/'+name+'/'
            if name not in receipt['routes']:
                if route.pre not in BC or route.post not in parent.ac_populations:
                    raise ContractError('Missing frozen fine BC route: '+name)
                average = sparse.diags(1/parent.populations[route.pre].counts) @ self.incidence[route.pre]
                mean = (parent.matrix(prefix+'mean_weights.npz') @ average).tocsr()
                total = (sparse.diags(post.counts, dtype=np.float64) @ mean).tocsr()
                semantics = 'retained effective AC kernel; equal original-member mean before AC equation'
                edges = None
            else:
                record = receipt['routes'][name]
                wp = checked_path(self.repo_root, directory/name/'weights.npz', record['weights_sha256'])
                rp = checked_path(self.repo_root, directory/name/'post_source_rows.npy', record['post_rows_sha256'])
                declared = np.load(rp, allow_pickle=False)
                target = np.sort(post.member_global_rows)
                idx = np.searchsorted(declared, target)
                if np.any(idx >= len(declared)) or not np.array_equal(declared[idx], target):
                    raise ContractError('Incomplete frozen BC target rows: '+name)
                full = sparse.load_npz(wp)[idx].tocsr()
                edges = full.tocoo()
                pg = post.fine_to_group[target[edges.row]]
                cg = pre.fine_to_group[edges.col]
                included = cg >= 0
                if route.kind != 'gap' and not np.all(included):
                    raise ContractError('BC chemical ancestor missing; no crop renormalization: '+name)
                total = sparse.csr_matrix((edges.data[included], (pg[included], cg[included])), shape=shape)
                mean = (sparse.diags(1/post.counts) @ total).tocsr()
                semantics = 'frozen fine edges; all original postsynaptic members; no crop renormalization'
                if route.kind == 'gap' and route.post in BC:
                    gap_degrees[route.post] = np.bincount(pg, weights=edges.data, minlength=shape[0])
                    gap_exterior[route.post] = np.bincount(pg[~included], weights=edges.data[~included], minlength=shape[0])
                if route.pre == 'cones':
                    meta = self.bc_cells[route.post]
                    distances = np.linalg.norm(meta['input_anchor_OPL_mm'][pg]-cone.coordinates['opl_pedicle'][edges.col], axis=1)
                    radius = np.zeros(shape[0]); np.maximum.at(radius, pg, distances)
                    meta.update(actual_cone_count=np.bincount(pg, minlength=shape[0]),
                                actual_radius_mm=radius, input_cone_source_rows=edges.col.copy(),
                                input_post_rows=pg.copy(), input_weights=edges.data.copy())
            self._matrices[prefix+'sum_weights.npz'] = total
            self._matrices[prefix+'mean_weights.npz'] = mean
            self.routes[name] = replace(route, shape=shape, group_pair_entries=total.nnz, mean_semantics=semantics)
            left = self.incidence[route.post] if route.post in BC else sparse.eye(shape[0], format='csr')
            right = self.incidence[route.pre] if route.pre in BC else sparse.eye(shape[1], format='csr')
            old_total = sparse.diags(parent.populations[route.post].counts, dtype=np.float64) @ parent.matrix(prefix+'mean_weights.npz')
            error = sparse_error(left @ total @ right.T, old_total)
            if error > 2e-12:
                raise ContractError('BC refinement changed the original aggregate route: '+name)
            self.projection_ledger[name] = dict(aggregate_max_abs_error=error, independent_shape=shape,
                mean_identity=matrix_id(mean), semantics=semantics)
            if not route.contact_component:
                continue
            if aii is None:
                aii = FixedDiskPopulation(self.repo_root, parent.populations['AII_AC'].source_table).positions
            distance = np.linalg.norm(self.tables[route.post].coordinates['ipl_preterminal'][target[edges.row]]-aii[edges.col], axis=1)
            gain = profiles['effective_'+route.contact_component]['g_peak']['value'] * np.exp(-(distance/.03)**2)
            coefficient = sparse.csr_matrix((gain, (pg, cg)), shape=shape)
            pairs = coefficient.tocoo()
            self._matrices[prefix+'contact_coefficient_S.npz'] = coefficient
            self._arrays[prefix+'contact_post_group.npy'] = pairs.row.copy()
            self._arrays[prefix+'contact_pre_group.npy'] = pairs.col.copy()
            old_post = parent.array(prefix+'contact_post_group.npy')
            old_pre = parent.array(prefix+'contact_pre_group.npy')
            old_keys = old_post * parent.populations[route.pre].group_count + old_pre
            new_old_post = self.bc_cells[route.post]['original_group_rows'][pairs.row]
            keys = new_old_post * parent.populations[route.pre].group_count + pairs.col
            lookup = {int(k): i for i, k in enumerate(old_keys)}
            self.contact_parent_rows[route.contact_component] = np.array([lookup[int(k)] for k in keys])
            cerr = sparse_error(left @ coefficient @ right.T, parent.matrix(prefix+'contact_coefficient_S.npz'))
            if cerr > 1e-21:
                raise ContractError('Refinement changed physical glycine coefficients: '+name)
            self.projection_ledger[name]['contact_aggregate_max_abs_error_S'] = cerr
            self.routes[name] = replace(self.routes[name], contact_groups=len(pairs.data),
                fine_contacts=route.fine_contacts, coefficient_sum_S=float(coefficient.sum()))
        self._gap(parent, gap_degrees, gap_exterior, profiles)
        blocks, offset = [], 0
        contact_counts = {r.contact_component: r.contact_groups for r in self.routes.values() if r.contact_component}
        for block in parent.state_blocks:
            count = contact_counts.get(block.mechanism, self.populations[block.population].group_count if block.population in BC else block.count)
            blocks.append(replace(block, count=count, offset=offset)); offset += count
        self.state_blocks, self.state_scalar_count = tuple(blocks), offset
        self.probability_blocks = parent.probability_blocks
        self.recording_fields = tuple(replace(field, rows=np.arange(self.populations[field.population].group_count),
            original_fine_rows=self.populations[field.population].member_global_rows.copy(),
            identity='independently_solved_cell') if field.population in BC else field for field in parent.recording_fields)
        self.core_population_rows = dict(parent.core_population_rows)
        for name in BC:
            self.core_population_rows[name] = np.flatnonzero(np.isin(self.bc_cells[name]['original_group_rows'], parent.core_population_rows[name]))
        self.layout = dict(parent.layout, BC_state_resolution='one_state_per_original_BC',
                           BC_members={n:self.populations[n].member_global_rows for n in BC})
        self.artifact_sha256 = canonical_hash(dict(parent=parent.artifact_sha256, configuration=configuration,
            routes=self.projection_ledger, layout=self.layout, boundary=self.boundary_runtime,
            implementation=digest(Path(__file__))))
        self.runtime_factories = dict(parent.runtime_factories)

    def _gap(self, parent, degrees, exterior, profiles):
        names = parent.manifest['gap']['population_order']
        sizes = [self.populations[n].group_count for n in names]
        old_sizes = [parent.populations[n].group_count for n in names]
        old_offsets = np.r_[0, np.cumsum(old_sizes)]
        blocks = [[sparse.csr_matrix((a,b)) for b in sizes] for a in sizes]
        for route in self.routes.values():
            if route.kind == 'gap':
                i, j = names.index(route.post), names.index(route.pre)
                blocks[i][j] = blocks[i][j] + self.matrix('routes/'+route.name+'/sum_weights.npz')
        diagonal, missing, products = [], [], []
        old = parent.matrix('gap/laplacian_S.npz')
        for i, name in enumerate(names):
            section = slice(old_offsets[i],old_offsets[i+1])
            if name in BC:
                diagonal.append(degrees[name]); missing.append(exterior[name])
                products.append(-exterior[name]*profiles['effective_AII_AC']['V0']['value'])
            else:
                # Intra-group AII edges cancel in the old Laplacian diagonal.
                diagonal.append(old.diagonal()[section]+blocks[i][i].diagonal())
                missing.append(parent.boundary_runtime['gap_missing_conductance_S'][section])
                products.append(parent.boundary_runtime['gap_product'][section])
        lap = (sparse.diags(np.concatenate(diagonal))-sparse.bmat(blocks, format='csr')).tocsr()
        aggregate = sparse.block_diag([self.incidence[n] if n in BC else sparse.eye(s) for n,s in zip(names,sizes)], format='csr')
        error = sparse_error(aggregate @ lap @ aggregate.T, old)
        asymmetry = sparse_error(lap,lap.T)
        if error > 1e-21 or asymmetry > 1e-21:
            raise ContractError(f'Independent BC gap current reciprocity or original degree changed: {error=}, {asymmetry=}')
        product, miss = np.concatenate(products), np.concatenate(missing)
        np.testing.assert_allclose(aggregate @ product, parent.boundary_runtime['gap_product'], rtol=2e-13, atol=1e-23)
        np.testing.assert_allclose(np.asarray(lap.sum(1)).ravel(),miss,rtol=2e-12,atol=1e-23)
        self._matrices['gap/laplacian_S.npz'] = lap
        self.boundary_runtime = dict(parent.boundary_runtime, gap_product=product, gap_missing_conductance_S=miss)
        self.boundary_runtime['identity'] = canonical_hash(dict(parent=parent.boundary_runtime['identity'],
            gap=matrix_id(lap), gap_product=product, gap_missing_conductance_S=miss))
        self.projection_ledger['gap'] = dict(aggregate_max_abs_error_S=error, reciprocal=True,
            full_incident_degree_retained=True, held_exterior_preserved=True)
        self.manifest['gap'] = dict(parent.manifest['gap'], group_counts=sizes, laplacian_entries=lap.nnz)

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def array(self, relative):
        return self._arrays[relative] if relative in self._arrays else self.parent.array(relative)

    def matrix(self, relative):
        return self._matrices[relative] if relative in self._matrices else self.parent.matrix(relative)


class IndependentBCModel:
    def __init__(self, repo_root, configuration=DEFAULT_CONFIGURATION):
        from .assembly import RetinaBuilder
        self.repo_root = Path(repo_root).resolve()
        path = checked_path(self.repo_root, configuration)
        cfg = json.loads(path.read_bytes()); self.config = cfg
        if (cfg['schema'] != 'macaque-independent-bc-fullchain' or cfg['execution_mode'] != 'local_cpu'
                or cfg['research_only'] is not True or cfg['production_qualified'] is not False
                or set(cfg['temporal_profiles']) != set(BC)):
            raise ContractError('Explicit five-type independent BC research configuration required')
        upstream = RetinaBuilder.build_research_cone_hc(self.repo_root, cfg['cone_hc_configuration'])
        original_hash = canonical_hash(upstream.seed)
        assembly = IndependentBCAssembly(upstream.assembly, self.repo_root, cfg)
        seed = deepcopy(upstream.seed)
        parameters = seed['inner_parameters']
        for name in BC:
            groups = assembly.bc_cells[name]['original_group_rows']
            p = parameters['populations'][name]
            for key, value in list(p.items()):
                if key != 'kernel' and np.shape(value) == (upstream.assembly.populations[name].group_count,):
                    p[key] = np.asarray(value)[groups].copy()
            for key in seed['states']:
                if key[0] == name: seed['states'][key] = seed['states'][key][groups].copy()
        for component, groups in assembly.contact_parent_rows.items():
            old_count = len(groups) and max(groups)+1
            for key, value in list(parameters['contacts'][component].items()):
                if np.ndim(value) == 1:
                    if len(value) < old_count: raise ContractError('Contact parameter lineage incomplete')
                    parameters['contacts'][component][key] = np.asarray(value)[groups].copy()
            for key in seed['states']:
                if key[1] == component: seed['states'][key] = seed['states'][key][groups].copy()
        seed['h1_h2_model']['artifact'] = assembly.artifact_sha256
        seed['bc_parent_binding'] = seed.pop('bc_physiology')
        self.temporal_bindings = {}
        for name in BC:
            profile = temporal_profile(cfg,name,assembly.bc_cells[name]['eccentricity_deg'])
            p = parameters['populations'][name]
            p.update(kernel=profile['kernel'], tau_release=profile['tau_release_seconds'],
                     tau_a_seconds=profile['tau_adaptation_seconds'])
            self.temporal_bindings[name] = profile
        release = upstream.native.evaluate(upstream.y0,0.,upstream.start_anchor).outputs['release']
        for name in BC:
            p = parameters['populations'][name]
            h = assembly.matrix('routes/cones_to_'+name+'/mean_weights.npz') @ release-p['baseline']
            sign = 1 if name.endswith('_OFF') else -1
            for i in range(3): seed['states'][name,BC_MECH,'history_'+str(i)] = h.copy()
            seed['states'][name,BC_MECH,'activation'] = np.logaddexp(0.,p['bias']+sign*np.sum(p['kernel'])*h)
        # Refining actual gap/glycine geometry requires the SAME conditional Rin
        # decomposition per cell; it is a background solve, not a new fit.
        assembly, seed = bind_bc_physiology(assembly, seed, self.repo_root, cfg['membrane_configuration'])
        seed['inner_parameters']['provenance']['BC_independent_configuration'] = digest(path)
        seed['inner_parameters']['identity'] = canonical_hash(dict(parent=seed['inner_parameters']['identity'],
            configuration=cfg, populations=seed['inner_parameters']['populations']))
        self.protocol = deepcopy(upstream.protocol)
        self.seed = _initialize_inner(assembly,seed,self.repo_root,self.protocol)
        self.seed['bc_physiology']['parameter_binding_id'] = assembly.parameter_binding_id
        self.seed['hc_eccentric_binding']['BC_independent_configuration'] = digest(path)
        self.seed['native_inputs'] = deepcopy(upstream.seed['native_inputs'])
        self.assembly, self.upstream = assembly, upstream
        self.calibration = upstream.calibration
        self.reference_bc_identity = upstream.reference_bc_identity
        if canonical_hash(upstream.seed) != original_hash:
            raise ContractError('Independent BC construction mutated the accepted HC model')
        self.identity = canonical_hash(dict(artifact=assembly.artifact_sha256, seed=self.seed,
            configuration=cfg, upstream=upstream.identity))

    def runner(self, protocol=None, *, execution_mode='local_cpu'):
        if execution_mode != 'local_cpu':
            raise ContractError('Independent BC supports the existing CPU FP64 path only')
        return EccentricHCFullChainRunner(self.assembly,self.seed,self.protocol if protocol is None else protocol)
