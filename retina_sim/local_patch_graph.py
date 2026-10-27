"""Exact dependency views of the frozen representative graph; no new equations.

Fine rows remain source provenance. Multi-member groups are never split or
labelled as independent cells. Native and electrical feedback components remain
whole, so a local output request can require broad dynamic support.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .contracts import ContractError, canonical_hash
from .local_patch import (validate_patch_config, requested_rf_center, BOUNDED_BOUNDARY, FULL_REFERENCE)
from .spatial_group_graph import digest
from .spatial_group_model import (GroupModelAssembly, POPULATIONS,
    RGC, BC, AC, CONTACT_STATES, StateBlock, recording_contract)

SCHEMA = 'local-retinal-patch-aggregate-graph-1'


def _frozen(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _verified_base(repo_root, config):
    root = Path(repo_root).resolve()
    if config.get('representation') != 'frozen_representative_groups_v1':
        raise ContractError('Aggregate graph builder requires explicit representative-group mode')
    path = (root / config['source_artifact']['path']).resolve()
    if (not path.is_relative_to(root) or path.name != 'manifest.json'
            or not path.is_file() or digest(path) != config['source_artifact']['sha256']):
        raise ContractError('Local patch parent artifact binding differs')
    base = GroupModelAssembly(path.parent)
    if config['eccentricity_deg'] > base.manifest['configuration']['output_radius_deg']:
        raise ContractError('Requested RF eccentricity exceeds parent output scope')
    return base


def _select_exact(base, config, guard=None):
    center = np.asarray(requested_rf_center(config), dtype=np.float64)
    scale = base.manifest['configuration']['mm_per_degree']
    radius = config['patch_geometry']['support_radius_deg']
    core_radius = config['patch_geometry']['core_radius_deg']
    chosen = {n: np.zeros(p.group_count, dtype=bool) for n, p in base.populations.items()}
    for name in RGC:
        pop = base.populations[name]
        if pop.coordinate_role != 'rf':
            raise ContractError('RGC selection requires original functional RF coordinates')
        chosen[name] = np.linalg.norm(pop.centroid_mm / scale - center, axis=1) <= radius
        if not np.any(chosen[name]):
            raise ContractError('Support contains no original '+name+' group; enlarge explicitly, no forced cell')
    target = base.populations[config['target_rgc_type']]
    identity = config['target_rgc_id']
    if identity is None:
        target_row = int(np.argmin(np.linalg.norm(target.centroid_mm / scale - center, axis=1)))
    else:
        match = re.fullmatch(r'original-row:(\d+)', identity)
        if match:
            target_row = target.group_for_fine_row(int(match.group(1)))
        else:
            rows = np.flatnonzero(target.group_ids == identity)
            if len(rows) != 1:
                raise ContractError('Target ID is not an original source-row or frozen group ID')
            target_row = int(rows[0])
        if target.counts[target_row] != 1:
            raise ContractError('Explicit cell ID belongs to an aggregate; fine splitting is unavailable')
    target_distance = np.linalg.norm(target.centroid_mm[target_row] / scale - center)
    if target_distance > core_radius:
        raise ContractError('Selected actual target lies outside the declared RF core')
    chosen[config['target_rgc_type']][target_row] = True
    # Native bipartite graph is one connected component in this frozen asset.
    # Retaining all native rows also retains source unit-DC A/B without a clamp.
    chosen['cones'][:] = True
    chosen['HC'][:] = True
    matrices = {}
    contact_pairs = {}
    for route in base.routes.values():
        if guard is not None:
            guard()
        matrices[route.name] = base.matrix('routes/'+route.name+'/sum_weights.npz')
        if route.contact_component:
            prefix = 'routes/'+route.name+'/'
            contact_pairs[route.name] = (base.array(prefix+'contact_post_group.npy'),
                                        base.array(prefix+'contact_pre_group.npy'))
    private_parents = {}
    for name in POPULATIONS:
        metadata = base.manifest['populations'][name]
        parent = metadata.get('parent_population')
        if parent:
            column = metadata['key_columns'].index('parent_group')
            keys = base.array('populations/'+name+'/keys.npy')
            source_rows = keys[:, column]
            if np.any(source_rows < 0) or np.any(source_rows >= base.populations[parent].group_count):
                raise ContractError('Original categorical parent identity differs: '+name)
            private_parents[name] = (parent, source_rows)
    gap_names = base.manifest['gap']['population_order']
    gap_offsets = np.r_[0, np.cumsum([base.populations[n].group_count for n in gap_names])]
    gap_blocks = {(r.post, r.pre): matrices[r.name] for r in base.routes.values() if r.kind == 'gap'}
    gap = sparse.bmat([[gap_blocks.get((post, pre), sparse.csr_matrix(
        (base.populations[post].group_count, base.populations[pre].group_count)))
        for pre in gap_names] for post in gap_names], format='csr')
    number, labels = connected_components(gap, directed=False)
    iterations = []
    for iteration in range(sum(p.group_count for p in base.populations.values()) + 1):
        if guard is not None:
            guard()
        before = sum(int(a.sum()) for a in chosen.values())
        # Categorical ancestry also carries spectral identity, including any
        # original zero-weight private-line row absent from a sparse operator.
        for child, (parent, source_rows) in private_parents.items():
            chosen[parent][source_rows[chosen[child]]] = True
        joined = np.concatenate([chosen[n] for n in gap_names])
        retained = np.isin(labels, np.unique(labels[joined]))
        for i, name in enumerate(gap_names):
            chosen[name] |= retained[gap_offsets[i]:gap_offsets[i+1]]
        for route in base.routes.values():
            if route.kind in ('native', 'gap'):
                continue
            post = np.flatnonzero(chosen[route.post])
            chosen[route.pre][np.unique(matrices[route.name][post].indices)] = True
            if route.name in contact_pairs:
                posts, pres = contact_pairs[route.name]
                chosen[route.pre][pres[chosen[route.post][posts]]] = True
        after = sum(int(a.sum()) for a in chosen.values())
        iterations.append(after)
        if before == after:
            break
    else:
        raise ContractError('Patch dependency closure did not reach a fixed point')
    rows = {n: _frozen(np.flatnonzero(chosen[n]).astype(np.int64)) for n in POPULATIONS}
    if any(not len(v) for v in rows.values()):
        raise ContractError('Patch route closure leaves an enabled population empty; enlarge support explicitly')
    contacts = {base.routes[name].contact_component: _frozen(
        np.flatnonzero(chosen[base.routes[name].post][posts]).astype(np.int64))
        for name, (posts, pres) in contact_pairs.items()}
    if any(not len(v) for v in contacts.values()):
        raise ContractError('Patch lacks an enabled physical contact population; enlarge support')
    core = {}
    for name in POPULATIONS:
        pop = base.populations[name]
        mask = chosen[name] & (np.linalg.norm(pop.centroid_mm / scale - center, axis=1) <= core_radius)
        core[name] = np.flatnonzero(mask) if pop.coordinate_role == 'rf' else np.empty(0, np.int64)
    # HC has soma coordinates. Its core observations follow actual feedback
    # edges of core cones, never a fictitious HC functional-RF circle.
    B = matrices['HC_to_cones']
    core['HC'] = np.unique(B[core['cones']].indices).astype(np.int64)
    for route in base.routes.values():
        if base.populations[route.pre].coordinate_role != 'rf' and route.pre != 'HC' and len(core[route.post]):
            sources = np.unique(matrices[route.name][core[route.post]].indices)
            core[route.pre] = np.union1d(core[route.pre], sources).astype(np.int64)
    core[config['target_rgc_type']] = np.union1d(core[config['target_rgc_type']], [target_row]).astype(np.int64)
    audit = dict(closure_group_counts=iterations, native_groups_retained=base.populations['cones'].group_count+base.populations['HC'].group_count,
        native_policy='all original native rows, no exterior clamp or renormalization',
        electrical_components=int(number), largest_electrical_component=int(np.bincount(labels).max()),
        closure_scope='all original group routes plus physical contact classes, including zero-coefficient classes',
        omitted_incoming_edges=0, weights_renormalized=False, exterior_inputs_substituted=False,
        scientific_equivalence='exact retained-equation dependency view of representative parent graph; no fine-cell or absolute physiological claim')
    return rows, contacts, core, target_row, audit


def _target_row(base, config):
    pop = base.populations[config['target_rgc_type']]
    scale = base.manifest['configuration']['mm_per_degree']
    center = np.asarray(requested_rf_center(config))
    identity = config['target_rgc_id']
    if identity is None:
        row = int(np.argmin(np.linalg.norm(pop.centroid_mm / scale - center, axis=1)))
    else:
        match = re.fullmatch(r'original-row:(\d+)', identity)
        if match:
            row = pop.group_for_fine_row(int(match.group(1)))
        else:
            found = np.flatnonzero(pop.group_ids == identity)
            if len(found) != 1:
                raise ContractError('Target ID is not an original source-row or frozen group ID')
            row = int(found[0])
        if pop.counts[row] != 1:
            raise ContractError('Explicit cell ID belongs to an aggregate; fine splitting is unavailable')
    if np.linalg.norm(pop.centroid_mm[row] / scale - center) > config['patch_geometry']['core_radius_deg']:
        raise ContractError('Selected actual target lies outside the declared RF core')
    return row


def _select_bounded(base, config, guard=None):
    """Finite native/gap support; missing linear inputs are explicit boundary ports.

    Spatial disks apply only to original functional RF populations. HC soma and
    AC IPL coordinates never substitute for RF: those rows follow actual graph
    adjacency. Chemical contact classes keep their original pre/post states.
    """
    center = np.asarray(requested_rf_center(config))
    scale = base.manifest['configuration']['mm_per_degree']
    core_radius = config['patch_geometry']['core_radius_deg']
    support_radius = config['patch_geometry']['support_radius_deg']
    target_row = _target_row(base, config)
    full = config['boundary_policy'] == FULL_REFERENCE
    chosen = {}
    for name, pop in base.populations.items():
        if full:
            mask = np.ones(pop.group_count, bool)
        elif name in RGC:
            mask = ((np.linalg.norm(pop.centroid_mm / scale - center, axis=1) <= core_radius)
                    if name == config['target_rgc_type'] else np.zeros(pop.group_count, bool))
        elif pop.coordinate_role == 'rf':
            mask = np.linalg.norm(pop.centroid_mm / scale - center, axis=1) <= support_radius
        else:
            mask = np.zeros(pop.group_count, bool)
        chosen[name] = mask
    chosen[config['target_rgc_type']][target_row] = True
    matrices = {}
    contact_pairs = {}
    for route in base.routes.values():
        if guard:
            guard()
        matrices[route.name] = base.matrix('routes/'+route.name+'/sum_weights.npz')
        if route.contact_component:
            prefix = 'routes/'+route.name+'/'
            contact_pairs[route.name] = (base.array(prefix+'contact_post_group.npy'),
                                        base.array(prefix+'contact_pre_group.npy'))
    parents = {}
    for name in POPULATIONS:
        meta = base.manifest['populations'][name]
        if meta.get('parent_population'):
            column = meta['key_columns'].index('parent_group')
            parents[name] = (meta['parent_population'], base.array('populations/'+name+'/keys.npy')[:,column])
    iterations = []
    core_aii_rows = np.empty(0, dtype=np.int64)
    direct_bc_gap_rows = {}
    if not full:
        # Promote only direct BC gap inputs of RF-defined core AII, once.
        # Their chemical/contact/parent ancestry uses the existing exact closure;
        # newly recruited AII do not recursively expand the electrical component.
        aii = base.populations['AII_AC']
        if aii.coordinate_role != 'rf':
            raise ContractError('Core AII gap support requires original functional RF identities')
        core_aii_rows = np.flatnonzero(chosen['AII_AC'] &
            (np.linalg.norm(aii.centroid_mm / scale - center, axis=1) <= core_radius))
        for route in base.routes.values():
            if route.kind == 'gap' and route.post == 'AII_AC' and route.pre in BC:
                adjacent = np.unique(matrices[route.name][core_aii_rows].indices)
                chosen[route.pre][adjacent] = True
                direct_bc_gap_rows[route.pre] = np.union1d(
                    direct_bc_gap_rows.get(route.pre, np.empty(0, dtype=np.int64)), adjacent).astype(np.int64)
        # All other native/electrical return paths retain the explicit boundary.
        # Chemical predecessors and immutable categorical parents remain exact.
        for _ in range(len(POPULATIONS)+1):
            if guard:
                guard()
            before = sum(int(v.sum()) for v in chosen.values())
            for child, (parent, indices) in parents.items():
                chosen[parent][indices[chosen[child]]] = True
            for route in base.routes.values():
                if route.kind in ('native', 'gap'):
                    continue
                posts = np.flatnonzero(chosen[route.post])
                chosen[route.pre][np.unique(matrices[route.name][posts].indices)] = True
                if route.name in contact_pairs:
                    post, pre = contact_pairs[route.name]
                    chosen[route.pre][pre[chosen[route.post][post]]] = True
            after = sum(int(v.sum()) for v in chosen.values())
            iterations.append(after)
            if before == after:
                break
        else:
            raise ContractError('Bounded chemical ancestry did not reach fixed point')
        # HC pool nodes are selected using original feedback connectivity of
        # retained cones. Incoming exterior cones become boundary pool drives.
        chosen['HC'][np.unique(matrices['HC_to_cones'][np.flatnonzero(chosen['cones'])].indices)] = True
    rows = {n:_frozen(np.flatnonzero(v).astype(np.int64)) for n,v in chosen.items()}
    contacts = {base.routes[name].contact_component:_frozen(np.flatnonzero(chosen[base.routes[name].post][post]).astype(np.int64))
                for name,(post,pre) in contact_pairs.items()}
    core = {}
    for name,pop in base.populations.items():
        inside = np.linalg.norm(pop.centroid_mm / scale - center, axis=1) <= core_radius
        core[name] = (np.flatnonzero(chosen[name] & inside).astype(np.int64)
                      if pop.coordinate_role == 'rf' and (name not in RGC or name == config['target_rgc_type'])
                      else np.empty(0,np.int64))
    core['HC'] = np.unique(matrices['HC_to_cones'][core['cones']].indices).astype(np.int64)
    for route in base.routes.values():
        if base.populations[route.pre].coordinate_role != 'rf' and route.pre != 'HC' and len(core[route.post]):
            source = np.unique(matrices[route.name][core[route.post]].indices)
            core[route.pre] = np.union1d(core[route.pre], source[chosen[route.pre][source]]).astype(np.int64)
    core[config['target_rgc_type']] = np.union1d(core[config['target_rgc_type']], [target_row]).astype(np.int64)
    omitted = {}
    for route in base.routes.values():
        original = base.matrix('routes/'+route.name+'/mean_weights.npz')[rows[route.post]]
        kept = original[:,rows[route.pre]]
        mass = np.asarray(abs(original).sum(axis=1)).ravel()-np.asarray(abs(kept).sum(axis=1)).ravel()
        omitted[route.name] = dict(missing_weight_l1_sum=float(mass.sum()), missing_weight_l1_max=float(mass.max(initial=0.)),
            missing_pair_entries=int(original.nnz-kept.nnz), linear_operator=route.mean_semantics)
    audit = dict(closure_group_counts=iterations,
        native_groups_retained=len(rows['cones'])+len(rows['HC']),
        native_policy='all retained native dynamics; original missing weighted pools supplied by source-bound boundary vectors',
        closure_scope='chemical/contact/categorical ancestors; native and gap return paths terminate at explicit boundary',
        omitted_routes=omitted, weights_renormalized=False, exterior_inputs_substituted=not full,
        output_population=config['target_rgc_type'], output_radius_deg=core_radius,
        halo_request_radius_deg=support_radius, non_rf_selection='original adjacency; HC soma and AC IPL not RF-cropped',
        scientific_equivalence='full original representative reference' if full else 'finite halo approximation; empirical qualification requires matching-input reference comparison',
        boundary_applicability='source-bound exterior construction/background state only; arbitrary exterior video unqualified')
    if not full:
        audit['core_AII_direct_BC_gap_support'] = dict(
            core_AII_parent_rows=core_aii_rows.tolist(),
            direct_BC_parent_rows={name:value.tolist() for name,value in direct_bc_gap_rows.items()},
            expansion='one direct gap hop from original RF core AII, then chemical/contact/categorical ancestry',
            whole_gap_component_retained=False)
    return rows,contacts,core,target_row,audit


def _select(base, config, guard=None):
    if config['boundary_policy'] in (BOUNDED_BOUNDARY,FULL_REFERENCE):
        return _select_bounded(base,config,guard)
    return _select_exact(base,config,guard)


class LocalPatchAssembly(GroupModelAssembly):
    """Small persisted row selection with an identity-verified parent graph."""
    def __init__(self, directory, repo_root=None, *, guard=None):
        directory = Path(directory).resolve()
        raw = json.loads((directory/'manifest.json').read_text(encoding='utf8'))
        expected = raw.pop('artifact_sha256', None)
        if raw.get('schema') != SCHEMA or canonical_hash(raw) != expected:
            raise ContractError('Local patch manifest identity differs')
        root = Path(repo_root or raw['repository_root']).resolve()
        if not directory.is_relative_to(root):
            raise ContractError('Local patch graph must remain within the active repository')
        if digest(Path(__file__)) != raw['implementation_sha256']:
            raise ContractError('Local patch graph implementation identity differs')
        config = validate_patch_config(raw['configuration'])
        base = _verified_base(root, config)
        if base.artifact_sha256 != raw['parent_artifact_sha256']:
            raise ContractError('Local patch parent logical graph identity differs')
        path = directory/'selection.npz'
        if digest(path) != raw['selection_sha256']:
            raise ContractError('Local patch row selection bytes differ')
        rows, contacts, core, target_row, audit = _select(base, config, guard)
        required = ({'population__'+n: v for n, v in rows.items()}
                    | {'contact__'+n: v for n, v in contacts.items()}
                    | {'core__'+n: v for n, v in core.items()})
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != set(required) or any(
                    stored[n].dtype != np.int64 or not np.array_equal(stored[n], value)
                    for n, value in required.items()):
                raise ContractError('Stored patch selection differs from recomputed exact closure')
        if audit != raw['closure_audit'] or target_row != raw['target_parent_row']:
            raise ContractError('Local patch closure or target ledger differs')
        self._assemble(root, directory, base, config, rows, contacts, core, target_row, expected, raw)

    def _assemble(self, root, directory, base, config, rows, contacts, core, target_row, identity, descriptor):
        self.root, self.repo_root, self.base = directory, root, base
        self.config = deepcopy(config)
        self.boundary_policy = config['boundary_policy']
        self.bounded_local = self.boundary_policy == BOUNDED_BOUNDARY
        self.parent_artifact_sha256 = base.artifact_sha256
        self.artifact_sha256 = identity
        self.parent_population_rows, self.parent_contact_rows = rows, contacts
        self.parent_auxiliary_rows = {}
        self.parent_population_keys = {}
        self._arrays, self._matrices = {}, {}
        self._population_maps = {}
        self.populations = {}
        self.manifest = deepcopy(base.manifest)
        self.manifest['local_patch'] = deepcopy(descriptor)
        self.manifest['files_registry_scope'] = 'parent source files; array/matrix apply hash-bound patch selections'
        for name in POPULATIONS:
            original = base.populations[name]
            keep = rows[name]
            self.parent_population_keys[name] = _frozen(base.array('populations/'+name+'/keys.npy')[keep])
            mapping = np.full(original.group_count, -1, np.int64)
            mapping[keep] = np.arange(len(keep))
            self._population_maps[name] = mapping
            counts = _frozen(original.counts[keep])
            members = _frozen(original.member_global_rows[np.repeat(mapping >= 0, original.counts)])
            offsets = _frozen(np.r_[0, np.cumsum(counts)].astype(np.int64))
            fine_map = np.full(len(original.fine_to_group), -1, np.int32)
            fine_map[members] = np.repeat(np.arange(len(keep)), counts)
            fine_map = _frozen(fine_map)
            aux_mask = mapping[original.recording_auxiliary_group_rows] >= 0
            aux_parent = np.flatnonzero(aux_mask)
            aux_fine = original.recording_auxiliary_fine_rows[aux_mask]
            if not len(aux_fine) and name != 'cones':
                aux_fine = np.sort(members[offsets[:-1]])[:900]
            self.parent_auxiliary_rows[name] = _frozen(aux_parent.astype(np.int64))
            aux_fine = _frozen(aux_fine)
            aux_group = _frozen(fine_map[aux_fine])
            geometry = {k: _frozen(v if k == 'leaf_keys' else v[keep])
                        for k, v in original.mesh_geometry.items()}
            pop = replace(original, group_ids=_frozen(original.group_ids[keep]), counts=counts,
                member_offsets=offsets, member_global_rows=members, fine_to_group=fine_map,
                centroid_mm=_frozen(original.centroid_mm[keep]), mesh_geometry=geometry,
                recording_auxiliary_fine_rows=aux_fine, recording_auxiliary_group_rows=aux_group)
            self.populations[name] = pop
            for key in ('group_ids','counts','member_offsets','member_global_rows','fine_to_group',
                        'centroid_mm','recording_auxiliary_fine_rows','recording_auxiliary_group_rows'):
                self._arrays['populations/'+name+'/'+key+'.npy'] = getattr(pop, key)
            self.manifest['populations'][name].update(group_count=len(keep), fine_neurons=len(members),
                singleton_groups=int(np.count_nonzero(counts == 1)),
                state_semantics='original representative group; singleton original rows remain singleton model cells',
                auxiliary_original_cell_count=len(aux_fine), auxiliary_distinct_group_count=len(np.unique(aux_group)))
        self.core_population_rows = {n: _frozen(self._population_maps[n][core[n]]) for n in POPULATIONS}
        self.target_population = config['target_rgc_type']
        self.target_parent_row = int(target_row)
        self.target_local_row = int(self._population_maps[self.target_population][target_row])
        contact_counts = {}
        route_metadata = []
        for old in base.manifest['routes']:
            meta = deepcopy(old)
            name, pre, post = old['name'], old['pre'], old['post']
            total = self.matrix('routes/'+name+'/sum_weights.npz')
            mean = self.matrix('routes/'+name+'/mean_weights.npz')
            original_mean = np.asarray(base.matrix('routes/'+name+'/mean_weights.npz')[rows[post]].sum(axis=1)).ravel()
            kept_mean = np.asarray(mean.sum(axis=1)).ravel()
            if not self.bounded_local and not np.allclose(original_mean, kept_mean, rtol=0, atol=1e-12):
                raise ContractError('Patch omits original incoming weight: '+name)
            meta.update(group_pair_entries=total.nnz, fine_edge_entries=None,
                        fine_edge_entries_scope='not reconstructable for noncontact selected rows from aggregate-only source')
            component = old.get('contact', {}).get('component')
            if component:
                parent_rows = contacts[component]
                prefix = 'routes/'+name+'/'
                old_mapping = base.array(prefix+'fine_contact_to_group.npy')
                class_map = np.full(old['contact']['groups'], -1, np.int64)
                class_map[parent_rows] = np.arange(len(parent_rows))
                fine_mask = class_map[old_mapping] >= 0
                fine = base.array(prefix+'fine_contact_global_rows.npy')[fine_mask]
                contact_map = class_map[old_mapping[fine_mask]]
                member_counts = np.bincount(contact_map, minlength=len(parent_rows))
                values = dict(contact_post_group=self._population_maps[post][base.array(prefix+'contact_post_group.npy')[parent_rows]],
                    contact_pre_group=self._population_maps[pre][base.array(prefix+'contact_pre_group.npy')[parent_rows]],
                    contact_member_offsets=np.r_[0,np.cumsum(member_counts)], fine_contact_global_rows=fine,
                    fine_contact_to_group=contact_map, contact_member_global_rows=fine[np.argsort(contact_map,kind='stable')])
                if np.any(values['contact_pre_group'] < 0):
                    raise ContractError('Patch contact source escaped the dependency closure')
                for key, value in values.items():
                    self._arrays[prefix+key+'.npy'] = _frozen(np.asarray(value,dtype=np.int64))
                coefficient = self.matrix(prefix+'contact_coefficient_S.npz')
                meta['contact'].update(groups=len(parent_rows),fine_contacts=len(fine),coefficient_sum=float(coefficient.sum()))
                contact_counts[component] = len(parent_rows)
            route_metadata.append(meta)
        self.manifest['routes'] = route_metadata
        self._gap_parent_rows = self._gap_rows()
        lap = self.matrix('gap/laplacian_S.npz')
        self.manifest['gap'].update(group_counts=[len(rows[n]) for n in self.manifest['gap']['population_order']],
                                    laplacian_entries=lap.nnz)
        cell_states = (7*len(rows['cones'])+3*len(rows['HC'])+7*len(rows['rods'])+
            3*len(rows['rod_BC'])+3*len(rows['AII_AC'])+7*sum(len(rows[n]) for n in BC)+
            sum(len(rows[n]) for n in AC)+sum(len(rows[n]) for n in RGC))
        contact_states = contact_counts['individual_rod_RBC_contacts']+2*sum(
            contact_counts[n] for n in CONTACT_STATES if n != 'individual_rod_RBC_contacts')
        self.manifest['budget_audit'] = dict(core_state_scalars=cell_states+contact_states,
            cell_states_including_analytic_RE=cell_states,contact_states=contact_states,
            extra_RGC_history_probability_scalars=8*sum(len(rows[n]) for n in RGC),
            extra_selected_z1_auxiliary_scalars=len(self.populations['cones'].recording_auxiliary_fine_rows),
            cell_slots=sum(len(v) for v in rows.values()), contact_groups=contact_counts,
            runtime_edge_entries=sum(r['group_pair_entries'] for r in route_metadata if r['kind']!='gap')+lap.nnz)
        self.routes = self._routes()
        self.state_blocks = self._states()
        self.state_scalar_count = sum(b.count for b in self.state_blocks)
        self.probability_blocks = self._probability_blocks()
        self.probability_scalar_count = sum(b.count for b in self.probability_blocks)
        count = len(self.populations['cones'].recording_auxiliary_fine_rows)
        self.auxiliary_z1_blocks = (StateBlock('cones','selected_original_z1_auxiliary','z1_aux',
            'vH_native_voltage','selected_original_cone_transmitter_drive',count,0,
            'uninitialized_selected_original_auxiliary'),)
        self.auxiliary_z1_scalar_count = count
        self.recording_contract = recording_contract()
        self.recording_fields = self._recording_fields()
        self.mechanism_clocks = deepcopy(base.mechanism_clocks)
        self.unresolved_bindings = tuple((n,p.unresolved_parameter_fields) for n,p in self.populations.items())
        self.runtime_ready = False

    def _routes(self):
        if not self.bounded_local:
            return super()._routes()
        # Parent operators/contact memberships were validated in full. Every
        # local operator is an exact identity-preserving row/column restriction.
        # Current conservation includes the explicit exterior Dirichlet term.
        lap = self.matrix('gap/laplacian_S.npz')
        exterior = self.external_matrix('gap/laplacian_S.npz')
        asymmetry = lap-lap.T
        scale = np.max(np.abs(lap.data),initial=0.)
        residual = np.asarray(lap.sum(axis=1)).ravel()+np.asarray(exterior.sum(axis=1)).ravel()
        if ((asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1e-12*scale)
                or np.max(np.abs(residual),initial=0.) > 1e-12*scale):
            raise ContractError('Bounded electrical operator plus exterior current is inconsistent')
        result = {}
        for meta in self.manifest['routes']:
            original = self.base.routes[meta['name']]
            mean = self.matrix('routes/'+original.name+'/mean_weights.npz')
            total = self.matrix('routes/'+original.name+'/sum_weights.npz')
            expected = sparse.diags(1./self.populations[original.post].counts)@total
            difference = mean-expected
            if (difference.nnz and np.max(np.abs(difference.data)) > 1e-12*max(1.,np.max(np.abs(mean.data),initial=0.))):
                raise ContractError('Bounded post-member normalization changed')
            if original.kind == 'inner_chemical' and self.external_matrix('routes/'+original.name+'/mean_weights.npz').nnz:
                raise ContractError('Bounded chemical predecessor closure is incomplete: '+original.name)
            contact = meta.get('contact',{})
            result[original.name] = replace(original,shape=total.shape,group_pair_entries=total.nnz,
                contact_groups=contact.get('groups',0),fine_contacts=contact.get('fine_contacts',0),
                coefficient_sum_S=contact.get('coefficient_sum'))
        return result

    def _gap_rows(self):
        selected, offset = [], 0
        for name in self.base.manifest['gap']['population_order']:
            selected.append(self.parent_population_rows[name]+offset)
            offset += self.base.populations[name].group_count
        return np.concatenate(selected)

    def array(self, relative):
        if relative in self._arrays:
            return self._arrays[relative]
        parts = Path(relative).as_posix().split('/')
        original = self.base.array(relative)
        if parts[0] == 'populations':
            name, filename = parts[1:]
            if filename == 'leaf_keys.npy':
                result = original
            elif filename == 'keys.npy':
                result = np.array(self.parent_population_keys[name], copy=True)
                metadata = self.base.manifest['populations'][name]
                parent = metadata.get('parent_population')
                if parent:
                    column = metadata['key_columns'].index('parent_group')
                    remapped = self._population_maps[parent][result[:, column]]
                    if np.any(remapped < 0):
                        raise ContractError('Patch categorical parent escaped its closure: '+name)
                    result[:, column] = remapped
            else:
                result = original[self.parent_population_rows[name]]
        elif parts[0] == 'routes':
            route = self.base.routes[parts[1]]
            result = original[self.parent_population_rows[route.post]]
        elif parts[0] == 'input':
            if len(self.parent_population_rows['cones']) != self.base.populations['cones'].group_count:
                if original.ndim == 1 and len(original) == self.base.populations['cones'].group_count:
                    result = original[self.parent_population_rows['cones']]
                else:
                    raise ContractError('Bounded patch requires an explicit source-bound input projection')
            else:
                result = original
        else:
            raise ContractError('Unmapped local patch array '+relative)
        result = _frozen(result)
        self._arrays[relative] = result
        return result

    def matrix(self, relative):
        if relative in self._matrices:
            return self._matrices[relative]
        parts = Path(relative).as_posix().split('/')
        original = self.base.matrix(relative)
        if parts[0] == 'routes':
            route = self.base.routes[parts[1]]
            result = original[self.parent_population_rows[route.post]][:,self.parent_population_rows[route.pre]].tocsr()
        elif parts[0] == 'gap':
            rows = self._gap_rows()
            result = original[rows][:,rows].tocsr()
        else:
            raise ContractError('Unmapped local patch matrix '+relative)
        for value in (result.data,result.indices,result.indptr):
            value.flags.writeable = False
        self._matrices[relative] = result
        return result


    def external_matrix(self, relative):
        """Original selected post rows, with retained presynaptic columns zero.

        Columns keep original source identities and units. This is the omitted
        linear term before any original nonlinear pooling/gating; never a
        renormalized substitute. The gap diagonal remains in the local block.
        """
        cache = getattr(self, '_external_matrices', None)
        if cache is None:
            self._external_matrices = cache = {}
        if relative in cache:
            return cache[relative]
        parts = Path(relative).as_posix().split('/')
        original = self.base.matrix(relative)
        if parts[0] == 'routes':
            route = self.base.routes[parts[1]]
            rows, columns = self.parent_population_rows[route.post], self.parent_population_rows[route.pre]
        elif parts[0] == 'gap':
            rows = columns = self._gap_rows()
        else:
            raise ContractError('Unmapped boundary matrix '+relative)
        external = np.ones(original.shape[1],dtype=np.float64)
        external[columns] = 0.
        result = original[rows].multiply(external).tocsr()
        result.eliminate_zeros()
        for value in (result.data,result.indices,result.indptr):
            value.flags.writeable=False
        cache[relative]=result
        return result


def build_local_patch_graph(repo_root, config, output_directory, *, guard=None):
    root = Path(repo_root).resolve()
    config = validate_patch_config(config)
    output = Path(output_directory)
    output = (root/output).resolve() if not output.is_absolute() else output.resolve()
    if not output.is_relative_to(root) or output == root or output.exists():
        raise ContractError('A new graph directory inside the repository is required')
    base = _verified_base(root, config)
    rows, contacts, core, target_row, audit = _select(base, config, guard)
    output.mkdir(parents=True)
    arrays = ({'population__'+n:v for n,v in rows.items()} |
              {'contact__'+n:v for n,v in contacts.items()} |
              {'core__'+n:np.asarray(v,dtype=np.int64) for n,v in core.items()})
    with (output/'selection.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    raw = dict(schema=SCHEMA, repository_root=str(root), configuration=config,
        parent_artifact_sha256=base.artifact_sha256, parent_source_manifest=config['source_artifact'],
        selection_sha256=digest(output/'selection.npz'), implementation_sha256=digest(Path(__file__)),
        target_parent_row=target_row, closure_audit=audit,
        representation='aggregate-only source-group dependency view; no new spatial aggregation',
        fine_source_reconstruction=False, fine_cell_precision_qualified=False,
        runtime_started=False, default_adopted=False)
    identity = canonical_hash(raw)
    with (output/'manifest.json').open('x',encoding='utf8') as stream:
        json.dump(dict(raw,artifact_sha256=identity),stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    return LocalPatchAssembly(output, root, guard=guard)


def load_local_patch_graph(repo_root, directory, *, guard=None):
    return LocalPatchAssembly(directory, repo_root, guard=guard)
