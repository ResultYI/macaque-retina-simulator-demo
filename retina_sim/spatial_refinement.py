"""Static upstream precision requests. These marks never prune model edges."""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .contracts import ContractError
from .spatial_multiresolution import priority_sources


def private_singletons(singletons, parents):
    """parents maps child population to (parent population, per-child source row)."""
    changed = True
    while changed:
        changed = False
        for child, (parent, source_rows) in parents.items():
            rows = np.unique(np.asarray(source_rows)[singletons[child]])
            if np.any(rows < 0) or np.any(rows >= len(singletons[parent])):
                raise ContractError('Independent cell has missing private-line ancestry')
            before = np.count_nonzero(singletons[parent][rows])
            singletons[parent][rows] = True
            changed |= before != len(np.unique(rows))


def propagate_precision(levels, routes, *, fraction=.99, gap_routes=(), native_routes=(), singleton_requests=None):
    """Monotone precision closure on feedforward routes and small gap cohorts.

    Native Cone/HC feedback is deliberately bounded to B then A, rather than a
    connected-component closure of the giant outer network. All matrices remain
    intact. Separate route objects prevent excitation/opponent cancellation.
    """
    if not 0 < fraction <= 1:
        raise ContractError('Refinement contribution fraction outside (0,1]')
    result = {k: np.array(v, dtype=np.int64, copy=True) for k, v in levels.items()}
    for a in result.values():
        if a.ndim != 1 or np.any(a < -1):
            raise ContractError('Per-fine-row levels must be -1 or a quadtree level')
    gap_names = sorted({n for pre, post, _ in gap_routes for n in (pre, post)})
    labels = None
    if gap_names:
        sizes = [len(result[n]) for n in gap_names]
        blocks = {(post, pre): sparse.csr_matrix(m) for pre, post, m in gap_routes}
        adjacency = sparse.bmat([[blocks.get((post, pre), sparse.csr_matrix((len(result[post]), len(result[pre]))))
                                 for pre in gap_names] for post in gap_names], format='csr')
        _, labels = connected_components(adjacency, directed=False)
        if np.max(np.bincount(labels), initial=0) > 42:
            raise ContractError('Electrical cohort exceeds frozen 42-node evidence; inspect asset')

    def apply(pre, post, matrix):
        matrix = sparse.csr_matrix(matrix)
        if matrix.shape != (len(result[post]), len(result[pre])):
            raise ContractError('Precision route shape differs from original row space')
        for level in np.unique(result[post][result[post] >= 0]):
            selected_rows = np.flatnonzero(result[post] == level)
            sources = priority_sources(matrix.indptr, matrix.indices, matrix.data,
                                       selected_rows, fraction=fraction)
            result[pre][sources] = np.maximum(result[pre][sources], level)
            if singleton_requests is not None:
                for row in selected_rows:
                    a, b = matrix.indptr[row:row+2]
                    footprint = np.unique(matrix.indices[a:b][matrix.data[a:b] != 0])
                    if len(footprint) == 1:
                        singleton_requests[pre][footprint] = True

    for _ in range(sum(len(v) for v in result.values()) + 1):
        before = {k: v.copy() for k, v in result.items()}
        for pre, post, matrix in routes:
            apply(pre, post, matrix)
        if labels is not None:
            values = np.concatenate([result[n] for n in gap_names])
            maxima = np.full(int(labels.max()) + 1, -1, dtype=np.int64)
            np.maximum.at(maxima, labels, values)
            values = maxima[labels]
            offset = 0
            for n in gap_names:
                result[n][:] = values[offset:offset + len(result[n])]
                offset += len(result[n])
        if all(np.array_equal(before[k], result[k]) for k in result):
            break
    else:
        raise ContractError('Static precision propagation did not converge')
    # Propagation follows actual B feedback from refined cones into HC, then
    # actual A inputs into cones. No repeated native feedback closure.
    for pre, post, matrix in sorted(native_routes, key=lambda r: r[0] != 'HC'):
        apply(pre, post, matrix)
    return result
