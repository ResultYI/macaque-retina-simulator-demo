"""Frozen dyadic geometry and refinement marks; never constructs dynamics.

The fixed origin is (0, 0), leaves are half-open squares, and only leaves
containing original cells exist.  IDs are deterministic within an artifact;
the actual identity is the integer (level, x, y) key.
"""
import math

import numpy as np

from .contracts import ContractError


def _positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ContractError('Positive finite '+name+' required')
    return float(value)


def _coordinates(xy):
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ContractError('Nonempty finite 2D original-cell coordinates required')
    return xy


def _tiles(xy, levels, base_deg):
    widths = np.ldexp(base_deg, -levels)
    quotient = xy/widths[:, None]
    if np.max(np.abs(quotient), initial=0.) >= 2**50:
        raise ContractError('Quadtree coordinates exceed reliable FP64 grid precision')
    tile = np.floor(quotient).astype(np.int64)
    # A division can round an adjacent ULP onto a grid line.  Compare the
    # represented boundaries directly so their half-open ownership is stable.
    tile -= (xy < tile*widths[:, None])
    tile += (xy >= (tile+1)*widths[:, None])
    return np.column_stack((levels, tile))


def _partition(xy, levels, base_deg):
    keys, inverse = np.unique(_tiles(xy, levels, base_deg), axis=0, return_inverse=True)
    return keys, inverse.astype(np.int32), np.bincount(inverse, minlength=len(keys))


def _unbalanced_coarse(keys):
    """Mark coarser touching leaves (edge or corner) with >1 level difference."""
    lookup = {tuple(int(v) for v in row): i for i, row in enumerate(keys)}
    marked = set()
    for level, x, y in keys:
        level, x, y = int(level), int(x), int(y)
        for coarse in range(level-1):
            scale = 1 << (level-coarse)
            bx, by = x//scale, y//scale
            xs = ([bx-1] if x % scale == 0 else []) + ([bx+1] if (x+1) % scale == 0 else [])
            ys = ([by-1] if y % scale == 0 else []) + ([by+1] if (y+1) % scale == 0 else [])
            candidates = [(xx, by) for xx in xs] + [(bx, yy) for yy in ys]
            candidates += [(xx, yy) for xx in xs for yy in ys]
            for xx, yy in candidates:
                other = lookup.get((coarse, xx, yy))
                if other is not None:
                    marked.add(other)
    return np.fromiter(sorted(marked), dtype=np.int64)


def build_quadtree(xy_deg, min_levels, *, singleton=None, base_deg=.4):
    """Build/refine then 2:1-balance nonempty leaves of one population.

    ``min_levels`` are per-original-cell requests, not a global resolution.
    A single original cell already has exact spatial identity and needs no
    further precision splits; balancing can still refine its geometric leaf.
    No minimum leaf width is imposed.  Requests that exceed representable
    FP64/integer geometry fail explicitly instead of being silently clamped.
    """
    xy = _coordinates(xy_deg)
    base = _positive(base_deg, 'base width')
    requested = np.asarray(min_levels)
    if (requested.shape != (len(xy),) or requested.dtype.kind not in 'iu'
            or np.any(requested < 0) or np.any(requested > 48)):
        raise ContractError('Per-cell integer levels within FP64 geometry precision required')
    keep = np.zeros(len(xy), dtype=bool) if singleton is None else np.asarray(singleton)
    if keep.shape != requested.shape or keep.dtype.kind != 'b':
        raise ContractError('Per-cell boolean singleton requests required')
    levels = np.zeros(len(xy), dtype=np.int64)
    split_count = 0
    while True:
        keys, cells, counts = _partition(xy, levels, base)
        strictest = np.zeros(len(keys), dtype=np.int64)
        np.maximum.at(strictest, cells, requested)
        independent = np.zeros(len(keys), dtype=bool)
        np.logical_or.at(independent, cells, keep)
        split = (counts > 1) & ((strictest > keys[:, 0]) | independent)
        if not np.any(split):
            break
        if np.any(keys[split, 0] >= 48):
            raise ContractError('Cannot isolate requested original cells in representable geometry')
        split_count += int(np.count_nonzero(split))
        levels += split[cells]
    balance_passes = 0
    while True:
        coarse = _unbalanced_coarse(keys)
        if not len(coarse):
            break
        split = np.zeros(len(keys), dtype=bool)
        split[coarse] = True
        if np.any(keys[split, 0] >= 48):
            raise ContractError('2:1 balance exceeds representable geometry')
        levels += split[cells]
        split_count += len(coarse)
        balance_passes += 1
        keys, cells, counts = _partition(xy, levels, base)
    return dict(cells=cells, leaf_keys=keys, leaf_counts=counts.astype(np.int64),
                balance_passes=balance_passes, split_count=split_count,
                adjacency='edge_or_corner_of_nonempty_geometric_leaves',
                fixed_origin_deg=[0., 0.], base_deg=base)


def required_level(*, fmax_cpd=None, center_radius_deg=None, epsilon_avg=.05, base_deg=.4):
    """Largest dyadic width satisfying both bound inputs; no RF is invented."""
    epsilon = _positive(epsilon_avg, 'spatial-average design bound')
    base = _positive(base_deg, 'base width')
    bounds = []
    if fmax_cpd is not None:
        bounds.append(math.sqrt(6*epsilon)/(math.pi*_positive(fmax_cpd, 'spatial bandwidth')))
    if center_radius_deg is not None:
        bounds.append(_positive(center_radius_deg, 'registered RF center radius')/2)
    if not bounds:
        raise ContractError('Registered RF scale or declared spatial bandwidth required')
    width = min(bounds)
    level = max(0, int(math.ceil(math.log2(base/width))))
    if level > 48:
        raise ContractError('Required resolution exceeds representable geometry')
    return level


def group_geometry(xy_deg, groups, counts, leaf_keys, group_leaf_ids, *, base_deg=.4, member_weights=None):
    """Actual member moments plus containing polygons, after categorical split.

    Equal original-cell weights are the default.  Alternative nonnegative
    weights are normalized independently within each group.
    """
    xy = _coordinates(xy_deg)
    g, n = np.asarray(groups), np.asarray(counts)
    leaves, ids = np.asarray(leaf_keys), np.asarray(group_leaf_ids)
    if (g.shape != (len(xy),) or g.dtype.kind not in 'iu' or n.ndim != 1
            or np.any(g < 0) or np.any(g >= len(n)) or np.any(n <= 0)
            or not np.array_equal(np.bincount(g, minlength=len(n)), n)
            or leaves.ndim != 2 or leaves.shape[1] != 3 or leaves.dtype.kind not in 'iu'
            or ids.shape != n.shape or ids.dtype.kind not in 'iu'
            or np.any(ids < 0) or np.any(ids >= len(leaves))):
        raise ContractError('Complete member/group/leaf geometry required')
    weights = np.ones(len(g)) if member_weights is None else np.asarray(member_weights, dtype=np.float64)
    if weights.shape != g.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ContractError('Nonnegative finite member weights required')
    total = np.bincount(g, weights=weights, minlength=len(n))
    if np.any(total <= 0):
        raise ContractError('Each group requires positive total member weight')
    centroid = np.column_stack([np.bincount(g, weights=weights*xy[:, k], minlength=len(n))/total for k in range(2)])
    centered = xy-centroid[g]
    covariance = np.empty((len(n), 2, 2), dtype=np.float64)
    for i, j in ((0, 0), (0, 1), (1, 1)):
        covariance[:, i, j] = np.bincount(g, weights=weights*centered[:, i]*centered[:, j], minlength=len(n))/total
    covariance[:, 1, 0] = covariance[:, 0, 1]
    lam = np.maximum(np.linalg.eigvalsh(covariance)[:, 1], 0.)
    key = leaves[ids]
    width = np.ldexp(_positive(base_deg, 'base width'), -key[:, 0])
    lower = key[:, 1:]*width[:, None]
    polygon = lower[:, None, :] + width[:, None, None]*np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])[None, :, :]
    return dict(centroid_deg=centroid, covariance_deg2=covariance, lambda_max_deg2=lam,
                polygon_deg=polygon, leaf_level=key[:, 0], leaf_width_deg=width,
                normalized_member_weights=weights/total[g])


def covariance_bound(lambda_max_deg2, fmax_cpd, *, epsilon_avg=.05):
    """Linear spatial-average bound only; never a neural/numerical acceptance."""
    lam = np.asarray(lambda_max_deg2, dtype=np.float64)
    frequency = np.asarray(fmax_cpd, dtype=np.float64)
    if (not np.isfinite(lam).all() or np.any(lam < 0)
            or not np.isfinite(frequency).all() or np.any(frequency <= 0)):
        raise ContractError('Finite nonnegative covariance and positive bandwidth required')
    epsilon = _positive(epsilon_avg, 'spatial-average design bound')
    value = 2*np.pi**2*frequency**2*lam
    return dict(value=value, satisfied=value <= epsilon, epsilon_avg=epsilon,
                scope='linear_spatial_average_operator_only')


def priority_sources(indptr, indices, values, post_rows, *, fraction=.99):
    """Mark each row's cumulative absolute contribution; preserve all CSR data."""
    ptr, col, data, rows = (np.asarray(v) for v in (indptr, indices, values, post_rows))
    if (ptr.ndim != 1 or ptr.dtype.kind not in 'iu' or len(ptr) < 1 or ptr[0] != 0
            or np.any(np.diff(ptr) < 0) or ptr[-1] != len(col) or col.ndim != 1
            or col.dtype.kind not in 'iu' or data.shape != col.shape or not np.isfinite(data).all()
            or rows.ndim != 1 or rows.dtype.kind not in 'iu'
            or np.any(rows < 0) or np.any(rows >= len(ptr)-1)
            or not math.isfinite(fraction) or not 0 < fraction <= 1):
        raise ContractError('Valid CSR, selected rows and contribution fraction required')
    marked = []
    for row in np.unique(rows):
        a, b = int(ptr[row]), int(ptr[row+1])
        weight = np.abs(data[a:b])
        total = float(np.sum(weight))
        if not total:
            continue
        order = np.lexsort((col[a:b], -weight))
        stop = min(int(np.searchsorted(np.cumsum(weight[order]), fraction*total, side='left'))+1, len(order))
        marked.append(col[a:b][order[:stop]])
    return np.unique(np.concatenate(marked)) if marked else np.empty(0, dtype=col.dtype)


def propagate_component_levels(component_labels, levels):
    """Use the strictest requested level throughout each supplied gap component."""
    labels, requested = np.asarray(component_labels), np.asarray(levels)
    if (labels.ndim != 1 or labels.dtype.kind not in 'iu' or requested.shape != labels.shape
            or requested.dtype.kind not in 'iu' or np.any(labels < 0) or np.any(requested < 0)):
        raise ContractError('Nonnegative component labels and integer precision levels required')
    _, inverse = np.unique(labels, return_inverse=True)
    maximum = np.zeros(int(np.max(inverse, initial=-1))+1, dtype=np.int64)
    np.maximum.at(maximum, inverse, requested)
    return maximum[inverse]
