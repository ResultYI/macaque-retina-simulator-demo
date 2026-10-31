"""Explicit engineering spatial pooling; no physiological RF claim."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def anchored_radius(eccentricity, near_eccentricity, far_eccentricity, near_radius, far_radius):
    if not (0 <= near_eccentricity < far_eccentricity and near_radius > 0 and far_radius > 0):
        raise ValueError('Positive, ordered, explicit geometry anchors required')
    values = np.asarray(eccentricity, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Finite nonnegative target eccentricity required')
    t = np.clip((values - near_eccentricity) / (far_eccentricity - near_eccentricity), 0, 1)
    return np.exp((1-t)*np.log(near_radius) + t*np.log(far_radius))


def complete_d8(tree: cKDTree, points, complete_support_radius):
    xy = np.asarray(points, dtype=float)
    if tree.n < 8:
        return np.full(len(xy), np.nan), np.zeros(len(xy), bool)
    distance = tree.query(xy, k=8, workers=1)[0][:, -1]
    valid = np.isfinite(distance) & (distance > 0) & (np.linalg.norm(xy, axis=1) + distance <= complete_support_radius)
    return distance, valid


def select_row(tree: cKDTree, point, radius, old_mass, complete_support_radius):
    """Return all in-radius source-local rows; caller retains old row on failure."""
    point = np.asarray(point, dtype=float)
    if not np.isfinite(radius) or radius <= 0 or not np.isfinite(old_mass) or old_mass <= 0:
        return None, None, 'LEGACY_FALLBACK_UNSUPPORTED_INVALID_ROW'
    if np.linalg.norm(point) + radius > complete_support_radius:
        return None, None, 'LEGACY_FALLBACK_UNSUPPORTED_PARENT_BOUNDARY'
    indices = np.asarray(sorted(tree.query_ball_point(point, radius, workers=1)), dtype=np.int64)
    if not len(indices):
        return None, None, 'LEGACY_FALLBACK_UNSUPPORTED_NO_SOURCE'
    distance = np.linalg.norm(tree.data[indices] - point, axis=1)
    raw = np.exp(-(distance / radius)**2)
    return indices, old_mass * raw / raw.sum(), 'ENGINEERING_CANDIDATE_VALID'
