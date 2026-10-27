"""Frozen scalar dependency plans; construction never evaluates a neural state.

Inner edges are structural reads in the existing typed instruction stream, not
nonzero Jacobian entries at a sampled state. Native descriptors are separately
source-traced to ReducedNative._coefficients; they are not labelled typed IR.
"""
from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import heapq
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .compiled_kernel import (ADD, SUB, MUL, DIV, POW, SIGMOID, LOG, EXP,
    MINIMUM, MAXIMUM, CSR, ROW_CSR, TAKE, SCATTER, ZERO, FINITE, NONNEG,
    POSITIVE, POSITIVE_MASK)
from .contracts import ContractError, canonical_hash
from .group_inner_dynamics import RODS, MEMBRANE, BC_MECH, AC_MECH, RGC_MECH
from .spatial_group_model import CONTACT_STATES
from .vh2005 import SOURCE_VALUES


def _frozen(values, dtype=np.int64):
    value = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class FrozenCSR:
    """Immutable boolean CSR payload; rows are targets unless stated otherwise."""
    shape: tuple
    indptr: np.ndarray
    indices: np.ndarray

    @classmethod
    def from_matrix(cls, matrix):
        matrix = matrix.astype(bool).tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return cls(matrix.shape, _frozen(matrix.indptr), _frozen(matrix.indices))

    @property
    def nnz(self):
        return len(self.indices)

    @property
    def nbytes(self):
        return self.indptr.nbytes + self.indices.nbytes

    def as_scipy(self):
        return sparse.csr_matrix((np.ones(self.nnz, dtype=bool),
            self.indices.copy(), self.indptr.copy()), shape=self.shape)


@dataclass(frozen=True)
class SCCBlock:
    """Batch of independent scalar SCCs with the same registry key signature."""
    name: str
    state_keys: tuple
    full_indices: np.ndarray
    component_ids: np.ndarray
    parent_names: tuple
    parameter_identity: str
    mode: str = 'numerical'
    semantic_name: str = ''
    coupled: bool = False


@dataclass(frozen=True)
class NativeBlock:
    name: str
    state_keys: tuple
    parent_names: tuple
    scalar_count: int
    component_count: int
    component_sizes: np.ndarray
    provenance: str


@dataclass(frozen=True)
class NativeLayout:
    full_keys: tuple
    full_slices: object
    numerical_indices: np.ndarray
    analytic_indices: np.ndarray
    blocks: tuple
    identity: str
    provenance: str


@dataclass(frozen=True)
class SCCPlan:
    full_keys: tuple
    full_slices: object
    active_indices: np.ndarray
    held_indices: np.ndarray
    analytic_indices: np.ndarray
    numerical_indices: np.ndarray
    state_dependencies: FrozenCSR
    component_of_full: np.ndarray
    component_offsets: np.ndarray
    component_indices: np.ndarray
    component_dependencies: FrozenCSR
    component_order: np.ndarray
    blocks: tuple
    block_order: tuple
    native: NativeLayout
    route_owners: tuple
    counts: object
    memory_bytes: object
    typed_ir_identity: str
    identity: str

    @property
    def nstate(self):
        return len(self.component_of_full)

    @property
    def slices(self):
        return self.full_slices

    @property
    def plan_hash(self):
        return self.identity

    def component(self, component_id):
        if not 0 <= component_id < len(self.component_offsets) - 1:
            raise ContractError('SCC component outside the frozen plan')
        return self.component_indices[self.component_offsets[component_id]:
                                      self.component_offsets[component_id + 1]]


def _registry(inner):
    keys = tuple(inner.state_keys)
    if len(keys) != len(set(keys)):
        raise ContractError('Duplicate SCC state registry key')
    offset, slices = 0, {}
    for key in keys:
        size, part = int(inner.sizes[key]), inner.slices[key]
        if size < 0 or part != slice(offset, offset + size):
            raise ContractError('SCC state registry must preserve the full pack order')
        slices[key] = part
        offset += size
    if offset != inner.nstate:
        raise ContractError('SCC state registry count differs')
    return keys, slices


def _trace_inner(inner, compiled):
    """Propagate sparse row provenance with allocation liveness, never scalar sets."""
    keys, slices = _registry(inner)
    program = np.asarray(compiled.program)
    deriv = np.asarray(compiled.derivatives)
    if (program.ndim != 2 or program.shape[1] != 7 or program.dtype.kind not in 'iu'
            or deriv.shape != (len(keys), 3) or deriv.dtype.kind not in 'iu'
            or int(compiled.nstate) != inner.nstate):
        raise ContractError('SCC typed IR layout differs')
    for key, (dest, source, count) in zip(keys, deriv):
        part = slices[key]
        if dest != part.start or count != part.stop - part.start or source < -1:
            raise ContractError('SCC derivative registry differs')
    checks = {FINITE, NONNEG, POSITIVE}
    unary = {SIGMOID, LOG, EXP, POSITIVE_MASK}
    binary = {ADD, SUB, MUL, DIV, POW, MINIMUM, MAXIMUM}
    supported = checks | unary | binary | {CSR, ROW_CSR, TAKE, SCATTER, ZERO}
    sizes = {part.start: part.stop - part.start for part in slices.values()
             if part.stop > part.start}
    constants = dict(compiled._scratch.constants)
    sizes.update({int(start): len(value) for start, value in constants.items()})
    external = {}
    nexternal = int(getattr(getattr(compiled, 'plan', None), 'nexternal', 0))
    if nexternal:
        rod_key = ('rods', RODS, 'G')
        part = slices.get(rod_key)
        if (part is None or nexternal != part.stop - part.start
                or getattr(compiled, '_rod_start', None) != part.start
                or getattr(compiled, '_rod_count', None) != nexternal):
            raise ContractError('Unknown SCC external source binding')
        sizes[inner.nstate] = nexternal
        external[inner.nstate] = part.start
    for code, dest, n, a, an, b, bn in program:
        if int(code) not in supported:
            raise ContractError('Unsupported SCC typed opcode: ' + str(code))
        if int(code) not in checks | {SCATTER}:
            if int(dest) in sizes or n < 0:
                raise ContractError('SCC typed IR requires fresh output allocations')
            sizes[int(dest)] = int(n)
    starts = sorted(sizes)

    def owner(address, count):
        index = bisect_right(starts, int(address)) - 1
        start = starts[index] if index >= 0 else -1
        if start < 0 or count < 0 or address + count > start + sizes[start]:
            raise ContractError('SCC typed read crosses an allocation')
        return start

    def reads(row):
        code, dest, n, a, an, b, bn = map(int, row)
        if code == ZERO:
            return ()
        if code in checks:
            return ((a, n),)
        if code == SCATTER:
            return ((a, n), (dest, 1))
        if code in {CSR, ROW_CSR, TAKE}:
            return ((a, an),)
        return ((a, an),) if code in unary else ((a, an), (b, bn))

    last = {}
    for i, row in enumerate(program):
        for address, count in reads(row):
            last[owner(address, count)] = i
    for _, source, count in deriv:
        if source >= 0:
            last[owner(source, count)] = len(program)
    cache = {}

    def empty(count):
        return sparse.csr_matrix((count, inner.nstate), dtype=bool)

    def read(address, count):
        start = owner(address, count)
        if start not in cache:
            n = sizes[start]
            if start < inner.nstate or start in external:
                source = external.get(start, start)
                cache[start] = sparse.csr_matrix((np.ones(n, dtype=bool),
                    np.arange(source, source + n), np.arange(n + 1)), shape=(n, inner.nstate))
            elif start in constants:
                cache[start] = empty(n)
            else:
                raise ContractError('SCC typed read precedes its definition')
        matrix = cache[start]
        left = int(address) - start
        return matrix if left == 0 and count == sizes[start] else matrix[left:left + count]

    indices, pointers = np.asarray(compiled.indices), np.asarray(compiled.pointers)
    if indices.dtype.kind not in 'iu' or pointers.dtype.kind not in 'iu':
        raise ContractError('SCC typed sparse indices must be integers')

    def selected(start, count, upper):
        values = indices[start:start + count]
        if start < 0 or len(values) != count or np.any(values < 0) or np.any(values >= upper):
            raise ContractError('SCC typed sparse index outside its source')
        return values

    peak = 0
    for i, row in enumerate(program):
        code, dest, n, a, an, b, bn = map(int, row)
        if code in checks:
            read(a, n)
        elif code == ZERO:
            cache[dest] = empty(n)
        elif code == SCATTER:
            target = owner(dest, 1)
            if owner(a, n) == target:
                raise ContractError('Overlapping SCC scatter source and target is unsupported')
            previous, source = read(target, sizes[target]), read(a, n)
            rows = selected(b, n, sizes[target] - (dest - target)) + dest - target
            scatter = sparse.csr_matrix((np.ones(n, dtype=bool), (rows, np.arange(n))),
                                        shape=(sizes[target], n))
            cache[target] = previous.maximum(scatter @ source)
        elif code in {CSR, ROW_CSR}:
            ptr = pointers[b:b + n + 1]
            if (b < 0 or len(ptr) != n + 1 or ptr[0] != 0 or np.any(np.diff(ptr) < 0)):
                raise ContractError('SCC typed CSR pointer layout differs')
            cols = selected(bn, int(ptr[-1]), an)
            matrix = sparse.csr_matrix((np.ones(len(cols), dtype=bool), cols, ptr), shape=(n, an))
            cache[dest] = matrix @ read(a, an)
        elif code == TAKE:
            cache[dest] = read(a, an)[selected(b, n, an)]
        else:
            if an not in (1, n) or (code in binary and bn not in (1, n)):
                raise ContractError('SCC typed broadcast shape differs')
            left = read(a, an)
            if an == 1 and n != 1:
                left = left[np.zeros(n, dtype=np.int64)]
            if code in unary:
                cache[dest] = left
            else:
                right = read(b, bn)
                if bn == 1 and n != 1:
                    right = right[np.zeros(n, dtype=np.int64)]
                cache[dest] = left.maximum(right)
        peak = max(peak, sum(v.data.nbytes + v.indices.nbytes + v.indptr.nbytes for v in cache.values()))
        for address, count in reads(row):
            start = owner(address, count)
            if last[start] == i:
                cache.pop(start, None)
        if code not in checks | {SCATTER} and dest not in last:
            cache.pop(dest, None)
    graph = sparse.vstack([empty(int(count)) if source < 0 else read(int(source), int(count))
                           for _, source, count in deriv], format='csr')
    active = np.concatenate([np.arange(dest, dest + count, dtype=np.int64)
                             for dest, source, count in deriv if source >= 0]) if np.any(deriv[:, 1] >= 0) else np.empty(0, dtype=np.int64)
    ir_identity = canonical_hash(dict(program=program, derivatives=deriv,
        weights=np.asarray(compiled.weights), indices=indices, pointers=pointers,
        constants=tuple((int(k), v) for k, v in sorted(constants.items())),
        source_model=inner.identity, source_compiled=getattr(compiled, 'identity', None),
        external_bindings=tuple(sorted(external.items())),
        external_source='group_inner_compiled._source_rod_current' if external else None))
    return graph, active, ir_identity, peak


def _components(graph, active):
    count, labels = connected_components(graph[active][:, active], directed=True, connection='strong')
    minima = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(minima, labels, active)
    rename = np.empty(count, dtype=np.int64)
    rename[np.argsort(minima)] = np.arange(count)
    labels = rename[labels]
    full = np.full(graph.shape[0], -1, dtype=np.int64)
    full[active] = labels
    ordered = active[np.argsort(labels, kind='stable')]
    offsets = np.r_[0, np.cumsum(np.bincount(labels, minlength=count))]
    edges = graph[active].tocoo()
    parent, child = full[edges.col], labels[edges.row]
    keep = (parent >= 0) & (parent != child)
    dag = sparse.csr_matrix((np.ones(np.count_nonzero(keep), dtype=bool),
                            (parent[keep], child[keep])), shape=(count, count))
    dag.sum_duplicates()
    dag.sort_indices()
    degree = np.asarray(dag.sum(axis=0)).ravel().astype(np.int64)
    ready = np.flatnonzero(degree == 0).tolist()
    heapq.heapify(ready)
    order = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for child in dag.indices[dag.indptr[node]:dag.indptr[node + 1]]:
            degree[child] -= 1
            if degree[child] == 0:
                heapq.heappush(ready, int(child))
    if len(order) != count:
        raise ContractError('SCC condensation is cyclic')
    return full, offsets, ordered, dag, np.asarray(order, dtype=np.int64)


def _name(keys, coupled=False):
    if coupled and all(k[2] == 'V' for k in keys):
        return 'gap_voltage'
    if len(keys) > 1:
        off = [k[0] for k in keys if k[1] == BC_MECH and k[2] == 'V' and k[0].endswith('_OFF')]
        if off:
            return 'off_bc_glycine:' + '+'.join(sorted(off))
        if all(k[2] == 'V' for k in keys):
            return 'gap_voltage'
        return 'coupled:' + '|'.join('.'.join(k) for k in keys)
    pop, mech, state = keys[0]
    if pop == 'rods' and mech == 'rod_membrane':
        return 'rod_voltage' if state == 'V' else 'rod_release'
    if mech == 'individual_rod_RBC_contacts':
        return 'rod_contact_trpm1'
    if mech == 'RBC_AII_AMPA':
        return 'rbc_aii_ampa_' + state
    if mech.startswith('AII_') and state == 'a':
        return 'glycine_a:' + mech
    if pop in ('rod_BC', 'AII_AC') and mech == MEMBRANE:
        stem = 'rod_bc' if pop == 'rod_BC' else 'aii'
        return stem + '_' + {'V': 'voltage', 'x': 'resource', 'release': 'filtered_release'}[state]
    if mech == BC_MECH:
        return {'V': 'bc_voltage', 'terminal_release': 'bc_terminal_release', 'A': 'bc_adaptation'}[state] + ':' + pop
    if mech == AC_MECH:
        return 'ac_inhibition:' + pop
    if mech == RGC_MECH:
        return 'rgc_voltage:' + pop
    return '.'.join(keys[0])


def _blocks(inner, full, offsets, ordered, dag, order, analytic):
    keys = tuple(inner.state_keys)
    key_of_full = np.empty(inner.nstate, dtype=np.int64)
    for i, key in enumerate(keys):
        key_of_full[inner.slices[key]] = i
    level = np.zeros(len(offsets) - 1, dtype=np.int64)
    for source in order:
        targets = dag.indices[dag.indptr[source]:dag.indptr[source + 1]]
        level[targets] = np.maximum(level[targets], level[source] + 1)
    signature_groups = {}
    for component in range(len(offsets) - 1):
        values = ordered[offsets[component]:offsets[component + 1]]
        signature = int(level[component]), tuple(np.unique(key_of_full[values]).tolist()), len(values) > 1
        signature_groups.setdefault(signature, []).append(component)
    signatures = tuple(signature_groups)
    semantics = [_name(tuple(keys[i] for i in sig[1]), sig[2]) for sig in signatures]
    names = [name if semantics.count(name) == 1 else name + ':level' + str(signatures[i][0])
             + ':batch' + str(i) for i, name in enumerate(semantics)]
    component_to_block = np.empty(len(offsets) - 1, dtype=np.int64)
    for i, signature in enumerate(signatures):
        component_to_block[signature_groups[signature]] = i
    parents = [set() for _ in signatures]
    for source in range(dag.shape[0]):
        for target in dag.indices[dag.indptr[source]:dag.indptr[source + 1]]:
            a, b = component_to_block[source], component_to_block[target]
            if a != b:
                parents[b].add(int(a))
    blocks = []
    analytic_set = set(analytic.tolist())
    for i, signature in enumerate(signatures):
        ids = np.asarray(signature_groups[signature], dtype=np.int64)
        selected = ordered[np.isin(full[ordered], ids)]
        state_keys = tuple(keys[j] for j in signature[1])
        blocks.append(SCCBlock(names[i], state_keys, _frozen(np.sort(selected)), _frozen(ids),
            tuple(names[j] for j in sorted(parents[i])),
            canonical_hash(dict(model=inner.identity, keys=state_keys)),
            'analytic' if len(selected) and all(int(j) in analytic_set for j in selected) else 'numerical',
            semantics[i], signature[2]))
    block_order = tuple(names[i] for i in sorted(range(len(signatures)),
                                                key=lambda i: (signatures[i][0], i)))
    return tuple(blocks), block_order


def _native_layout(inner):
    assembly = getattr(inner, 'assembly', None)
    declarations = tuple(b for b in getattr(assembly, 'state_blocks', ()) if b.population in ('cones', 'HC'))
    declarations += tuple(getattr(assembly, 'auxiliary_z1_blocks', ()))
    keys, slices, numerical, analytic, offset = [], {}, [], [], 0
    for block in declarations:
        key = (block.population, block.mechanism, block.name)
        if key in slices:
            raise ContractError('Duplicate native SCC registry key')
        keys.append(key)
        slices[key] = slice(offset, offset + block.count)
        (analytic if block.mechanism == 'analytic_RE' else numerical).extend(range(offset, offset + block.count))
        offset += block.count
    provenance = 'source:cone_hc_reduced_native.ReducedNative._coefficients; not inner typed IR'
    blocks, operators = [], {}
    if declarations:
        nc = assembly.populations['cones'].group_count
        nh = assembly.populations['HC'].group_count
        A, B = (assembly.matrix('routes/' + name + '/mean_weights.npz').tocsr()
                for name in ('cones_to_HC', 'HC_to_cones'))
        if A.shape != (nh, nc) or B.shape != (nc, nh):
            raise ContractError('Native SCC operator layout differs')
        # Retain w->z2->h->release(B)->w(A), without forming a dense AB.
        I = sparse.eye(nh, dtype=bool, format='csr')
        Z = sparse.csr_matrix((nh, nh), dtype=bool)
        Q = sparse.csr_matrix((nc, nc), dtype=bool)
        HQ = sparse.csr_matrix((nh, nc), dtype=bool)
        QH = sparse.csr_matrix((nc, nh), dtype=bool)
        support = sparse.bmat([[Z, Z, Z, A.astype(bool)], [I, Z, Z, HQ],
                               [Z, I, Z, HQ], [QH, QH, B.astype(bool), Q]], format='csr')
        _, label = connected_components(support, directed=True, connection='strong')
        _, hc_sizes = np.unique(label[:3 * nh], return_counts=True)
        descriptors = (
            ('native_RE', ('R', 'E'), (), 2 * nc, np.ones(2 * nc)),
            ('native_XC', ('X', 'C'), ('native_RE',), 2 * nc, np.full(nc, 2)),
            ('native_Vis_gi', ('Vis', 'g_i'), ('native_XC',), 2 * nc, np.full(nc, 2)),
            ('native_Vbar', ('Vbar',), ('native_Vis_gi',), nc, np.ones(nc)),
            ('native_HC', ('w', 'z2', 'Vh'), ('native_Vis_gi', 'native_Vbar'), 3 * nh, hc_sizes),
            ('native_z1_aux', ('z1_aux',), ('native_Vis_gi', 'native_Vbar', 'native_HC'),
             sum(b.count for b in declarations if b.name == 'z1_aux'),
             np.ones(sum(b.count for b in declarations if b.name == 'z1_aux'))))
        for name, states, parents, count, sizes in descriptors:
            selected = tuple(key for key in keys if key[2] in states)
            if count != sum(slices[key].stop - slices[key].start for key in selected):
                raise ContractError('Native source descriptor differs from state registry')
            if count:
                blocks.append(NativeBlock(name, selected, parents, count, len(sizes), _frozen(sizes), provenance))
        operators = {name: dict(shape=matrix.shape, data=matrix.data,
            indices=matrix.indices, indptr=matrix.indptr) for name, matrix in (('A', A), ('B', B))}
    source = Path(__file__).with_name('cone_hc_reduced_native.py')
    cone = getattr(assembly, 'populations', {}).get('cones')
    identity = canonical_hash(dict(keys=keys, sizes=[s.stop - s.start for s in slices.values()],
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), operators=operators,
        source_parameters=dict(SOURCE_VALUES),
        group_binding_sha256=hashlib.sha256(source.with_name('group_native_physics.py').read_bytes()).hexdigest(),
        auxiliary_original_rows=getattr(cone, 'recording_auxiliary_fine_rows', ()),
        auxiliary_group_rows=getattr(cone, 'recording_auxiliary_group_rows', ()),
        artifact=getattr(assembly, 'artifact_sha256', None), provenance=provenance))
    return NativeLayout(tuple(keys), MappingProxyType(slices), _frozen(numerical),
                        _frozen(analytic), tuple(blocks), identity, provenance)


def build_scc_plan(inner, compiled=None):
    """Build exact inner row SCCs and separate native source descriptors.

    ``active_indices`` includes analytic rod V; ``numerical_indices`` excludes
    it. All indices address inner.pack order. Native indices address the separate
    ``native.full_slices`` registry. No contact state or auxiliary row is removed.
    """
    if compiled is None:
        compiled = getattr(inner, 'compiled_rhs', None)
        if compiled is None:
            from .group_inner_compiled import CompiledGroupInnerRHS
            compiled = CompiledGroupInnerRHS(inner)
    if getattr(compiled, 'model', inner) is not inner:
        raise ContractError('SCC typed IR belongs to a different inner model')
    keys, slices = _registry(inner)
    graph, active, ir_identity, peak = _trace_inner(inner, compiled)
    held = np.setdiff1d(np.arange(inner.nstate, dtype=np.int64), active, assume_unique=True)
    rod = slices.get(('rods', 'rod_membrane', 'V'), slice(0, 0))
    analytic = np.arange(rod.start, rod.stop, dtype=np.int64)
    if not np.isin(analytic, active).all():
        raise ContractError('Analytic rod voltage is not an active state')
    numerical = np.setdiff1d(active, analytic, assume_unique=True)
    full, offsets, ordered, dag, order = _components(graph, active)
    blocks, block_order = _blocks(inner, full, offsets, ordered, dag, order, analytic)
    native = _native_layout(inner)
    route_owners = tuple((name, row['route'].name, row['route'].pre, row['route'].post,
        _frozen(row['pre']), _frozen(row['post']), canonical_hash(row['coefficient']))
        for name, row in sorted(getattr(inner, 'contacts', {}).items()))
    dependencies, component_dependencies = FrozenCSR.from_matrix(graph), FrozenCSR.from_matrix(dag)
    counts = dict(inner_total=inner.nstate, inner_active=len(active), inner_held=len(held),
        inner_analytic=len(analytic), inner_numerical=len(numerical),
        inner_contact_scalars=sum(slices[k].stop - slices[k].start for k in keys if k[1] in CONTACT_STATES),
        inner_scc_count=len(offsets) - 1, inner_dependency_edges=dependencies.nnz,
        inner_largest_scc=int(np.max(np.diff(offsets), initial=0)),
        inner_nontrivial_scc_count=int(np.count_nonzero(np.diff(offsets) > 1)),
        native_numerical=len(native.numerical_indices), native_analytic=len(native.analytic_indices))
    counts['logical_total'] = inner.nstate + counts['native_numerical'] + counts['native_analytic']
    maps = (active, held, analytic, numerical, full, offsets, ordered, order)
    memory = dict(layout_arrays=sum(x.nbytes for x in maps),
        block_arrays=sum(b.full_indices.nbytes + b.component_ids.nbytes for b in blocks),
        native_arrays=native.numerical_indices.nbytes + native.analytic_indices.nbytes
            + sum(b.component_sizes.nbytes for b in native.blocks),
        route_index_arrays=sum(row[4].nbytes + row[5].nbytes for row in route_owners),
        dependency_payload=dependencies.nbytes + component_dependencies.nbytes,
        inner_three_stage_numerical_fp64=3 * len(numerical) * 8,
        native_three_stage_numerical_fp64=3 * len(native.numerical_indices) * 8,
        inner_full_state_fp64=inner.nstate * 8, max_retained_provenance_payload=peak)
    memory['owned_array_payload'] = sum(memory[k] for k in ('layout_arrays',
        'block_arrays', 'native_arrays', 'route_index_arrays', 'dependency_payload'))
    identity = canonical_hash(dict(schema='retipath-group-scc-plan-1', typed_ir=ir_identity,
        keys=keys, slices=tuple((s.start, s.stop) for s in slices.values()),
        active=active, held=held, analytic=analytic, component_of_full=full,
        component_dependencies=component_dependencies, component_order=order,
        blocks=blocks, native=native.identity, routes=route_owners, counts=counts,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    return SCCPlan(keys, MappingProxyType(slices), *(_frozen(x) for x in (active, held, analytic, numerical)),
        dependencies, _frozen(full), _frozen(offsets), _frozen(ordered), component_dependencies,
        _frozen(order), blocks, block_order, native, route_owners, MappingProxyType(counts),
        MappingProxyType(memory), ir_identity, identity)
