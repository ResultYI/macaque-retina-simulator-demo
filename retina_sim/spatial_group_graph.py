"""Static spatial restriction/prolongation; no simulator, device or solver imports.

Group matrices describe a new model approximation. They are never registered as
an equivalent replacement for the original fine graph or its checkpoints.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from scipy import sparse

from .contracts import ContractError

SCHEMA = 'retipath-static-spatial-group-graph-1'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified_path(root, binding):
    root = Path(root).resolve()
    path = (root / binding['path']).resolve()
    if (not path.is_relative_to(root) or not path.is_file()
            or digest(path) != binding['sha256']):
        raise ContractError('Static group source identity differs: '+str(binding['path']))
    return path


def polar_cells(xy, bands, mm_per_degree):
    """Closed outer radial boundaries; angle zero is +x, increasing toward +y."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ContractError('Finite two-dimensional source coordinates required')
    if not math.isfinite(mm_per_degree) or mm_per_degree <= 0:
        raise ContractError('Positive explicit coordinate scale required')
    lower, offset = 0., 0
    radius = np.hypot(xy[:, 0], xy[:, 1])/mm_per_degree
    angle = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2*np.pi)
    cells = np.full(len(xy), -1, dtype=np.int32)
    for band in bands:
        upper, rings, sectors = band['outer_deg'], band['radial'], band['angular']
        if (not math.isfinite(upper) or upper <= lower or
                type(rings) is not int or type(sectors) is not int or min(rings,sectors) < 1):
            raise ContractError('Ordered positive radial bands required')
        mask = (radius <= upper) & (cells < 0)
        ring = np.minimum(np.floor((radius[mask]-lower)/(upper-lower)*rings).astype(np.int64), rings-1)
        sector = np.minimum(np.floor(angle[mask]/(2*np.pi)*sectors).astype(np.int64), sectors-1)
        cells[mask] = offset+ring*sectors+sector
        lower, offset = upper, offset+rings*sectors
    if np.any(cells < 0):
        raise ContractError('Source coordinate outside approved outer grouping band; no clipping')
    return cells


def group_members(global_rows, cells, *, subtype=None, parent=None, eligibility=None):
    """Each unique categorical key owns a group; no per-fine-cell dictionaries."""
    rows, cells = np.asarray(global_rows), np.asarray(cells)
    if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or len(rows) != len(cells)
            or not len(rows) or np.any(np.diff(rows) <= 0) or np.any(cells < 0)):
        raise ContractError('Nonempty ordered unique global rows and spatial cells required')
    columns = [cells]
    for value in (subtype, parent, eligibility):
        a = np.full(len(rows), -1, np.int64) if value is None else np.asarray(value)
        if a.shape != rows.shape or a.dtype.kind not in 'iub':
            raise ContractError('Integer subtype/parent/eligibility classes required')
        columns.append(a)
    keys, inverse = np.unique(np.column_stack(columns), axis=0, return_inverse=True)
    inverse = inverse.astype(np.int32)
    counts = np.bincount(inverse, minlength=len(keys)).astype(np.int64)
    offsets = np.r_[np.int64(0), np.cumsum(counts)]
    members = rows[np.argsort(inverse, kind='stable')]
    return dict(keys=keys.astype(np.int64), fine_to_group=inverse,
                counts=counts, member_offsets=offsets, member_global_rows=members)


def aggregate_edges(post_groups, pre_groups, values, shape):
    post, pre, data = np.asarray(post_groups), np.asarray(pre_groups), np.asarray(values, dtype=np.float64)
    if (post.ndim != 1 or pre.shape != post.shape or data.shape != post.shape
            or post.dtype.kind not in 'iu' or pre.dtype.kind not in 'iu'
            or not np.isfinite(data).all() or np.any(data < 0)
            or np.any(post < 0) or np.any(post >= shape[0])
            or np.any(pre < 0) or np.any(pre >= shape[1])):
        raise ContractError('Finite nonnegative typed group edges required')
    result = sparse.coo_matrix((data, (post,pre)), shape=shape).tocsr()
    result.eliminate_zeros()
    return result


def mean_matrix(total, post_counts):
    counts = np.asarray(post_counts)
    if counts.shape != (total.shape[0],) or np.any(counts <= 0):
        raise ContractError('Every group must contain original members')
    return sparse.diags(1./counts) @ total


def moments(values, groups, counts):
    """Stable population moments in original value units, including near constants."""
    v, g, n = np.asarray(values, dtype=np.float64), np.asarray(groups), np.asarray(counts)
    if (v.shape != g.shape or v.ndim != 1 or not np.isfinite(v).all()
            or np.any(g < 0) or np.any(g >= len(n)) or np.any(n <= 0)
            or not np.array_equal(np.bincount(g, minlength=len(n)), n)):
        raise ContractError('Complete finite values and group membership required')
    mean = np.bincount(g, weights=v, minlength=len(n))/n
    variance = np.bincount(g, weights=(v-mean[g])**2, minlength=len(n))/n
    lo, hi = np.full(len(n), np.inf), np.full(len(n), -np.inf)
    np.minimum.at(lo,g,v)
    np.maximum.at(hi,g,v)
    return mean, variance, lo, hi


def gap_laplacian(adjacency):
    g = sparse.csr_matrix(adjacency, dtype=np.float64)
    if g.shape[0] != g.shape[1] or np.any(g.data < 0) or not np.isfinite(g.data).all():
        raise ContractError('Square nonnegative physical conductance adjacency required')
    delta = g-g.T
    tolerance = 1e-12*max(float(np.max(g.data, initial=0.)), np.finfo(float).tiny)
    if delta.nnz and np.max(np.abs(delta.data)) > tolerance:
        raise ContractError('Group gap conductance lost reciprocity')
    result = sparse.diags(np.asarray(g.sum(axis=1)).ravel())-g
    result.eliminate_zeros()  # Internal group gap currents cancel for common V.
    return result.tocsr()


def save_csr(path, matrix):
    m = sparse.csr_matrix(matrix)
    np.savez(path, data=m.data, indices=m.indices.astype(np.int64),
             indptr=m.indptr.astype(np.int64), shape=np.array(m.shape, dtype=np.int64))


def load_static_graph(directory):
    """Verify an immutable completed static artifact; this never creates a runtime."""
    directory = Path(directory).resolve()
    raw = json.loads((directory/'manifest.json').read_text(encoding='utf8'))
    expected = raw.pop('manifest_sha256', None)
    actual = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()
    if (raw.get('schema') != SCHEMA or expected != actual or raw.get('runtime_ready') is not False
            or raw.get('model_forward_started') is not False):
        raise ContractError('Static graph identity/schema/classification differs')
    for name, row in raw['files'].items():
        path = (directory/name).resolve()
        if (not path.is_relative_to(directory) or not path.is_file()
                or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']):
            raise ContractError('Static graph array identity differs: '+name)
    return raw
