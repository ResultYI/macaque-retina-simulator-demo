"""Finite cone-pedicle/HC-arbor geometry; no conductance or kinetic parameters.

Rows of the returned feedforward matrix are HC cells, columns are cones. The
feedback matrix has the transpose contact topology, with independent pooling
normalization. Neither matrix is a conductance. Footprints remain explicitly
unconfigured until finite spans and their provenance are supplied.
"""
from collections.abc import Mapping
import hashlib
import json

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree, Delaunay, QhullError

from .contracts import ContractError, Quantity, canonical_hash


_EVIDENCE = {'measurement', 'published_fit', 'author_assumption',
             'project_approximation'}
_TOL = 1e-12


class HCArborConfigurationError(ContractError):
    """A finite arbor is missing; this is not a claim of absent HC anatomy."""
    status = 'HC_ARBOR_UNCONFIGURED'


def _positive(value, name, *, allow_zero=False):
    if isinstance(value, bool):
        raise ContractError(f'{name} must be a finite physical scalar')
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f'{name} must be a finite physical scalar') from error
    if not np.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ContractError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')
    return number


def _provenance(row, label):
    if not row.get('source_profile_id') or row.get('evidence_class') not in _EVIDENCE:
        raise ContractError(f'{label} requires source_profile_id and evidence_class')
    if row['evidence_class'] == 'project_approximation' and not row.get('approximation_id'):
        raise ContractError(f'{label} project approximation requires a named approximation_id')
    if row['evidence_class'] in ('measurement', 'published_fit'):
        proof = row.get('primary_verification')
        if not isinstance(proof, Mapping) or proof.get('verified') is not True or not all(
                proof.get(key) for key in ('material_path', 'material_sha256', 'locator',
                                          'condition', 'geometry_sha256')):
            raise ContractError(f'{label} scientific geometry requires primary material, locator, condition and frozen geometry verification; a source name is insufficient')
        if row.get('radial_support_mm') is None:
            raise ContractError(f'{label} scientific geometry requires its own source radial support')
    return {key: row.get(key) for key in ('source_profile_id', 'evidence_class',
        'approximation_id', 'source_locator', 'primary_verification',
        'cross_animal_transfer', 'cross_species_transfer', 'meridian_transfer')}


def _finite_shape(row, label):
    """A physical footprint in the HC anchor's OPL plane, never RF space."""
    kind = row.get('kind')
    if row.get('unit', 'mm') != 'mm' or row.get('coordinate_frame', 'OPL') != 'OPL':
        raise ContractError(f'{label} footprint requires mm in OPL')
    if kind == 'disk':
        radius = _positive(row.get('radius_mm'), f'{label}.radius_mm')
        return {'kind': kind, 'radius_mm': radius, 'envelope_mm': radius}
    if kind == 'ellipse':
        axes = np.asarray(row.get('axes_mm'), dtype=float)
        if axes.shape != (2,) or not np.all(np.isfinite(axes)) or np.any(axes <= 0):
            raise ContractError(f'{label}.axes_mm requires two finite positive semi-axes')
        angle = float(row.get('angle_rad', 0.))
        if not np.isfinite(angle):
            raise ContractError(f'{label}.angle_rad must be finite')
        return {'kind': kind, 'axes_mm': axes.tolist(), 'angle_rad': angle,
                'envelope_mm': float(axes.max())}
    raise ContractError(f'{label} requires a disk or ellipse finite domain')


def _footprint(row, label):
    if not isinstance(row, Mapping):
        raise ContractError(f'{label} requires a finite footprint object')
    result = dict(row)
    result['provenance'] = _provenance(row, label)
    kind = row.get('kind')
    if kind in ('disk', 'ellipse'):
        result.update(_finite_shape(row, label))
    elif kind == 'morphology_nodes':
        if row.get('unit', 'mm') != 'mm' or row.get('coordinate_frame', 'OPL') != 'OPL':
            raise ContractError(f'{label} morphology requires mm in OPL')
        nodes = np.asarray(row.get('nodes_mm'), dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] not in (2, 3) or not len(nodes) or not np.all(np.isfinite(nodes)):
            raise ContractError(f'{label}.nodes_mm requires finite nonempty xy or xyz nodes')
        coordinates = row.get('node_coordinates')
        if coordinates not in ('relative_to_anchor', 'absolute_OPL'):
            raise ContractError(f'{label} must declare relative_to_anchor or absolute_OPL nodes')
        threshold = _positive(row.get('contact_threshold_mm'), f'{label}.contact_threshold_mm')
        extension = row.get('allowed_extension')
        if extension is not None:
            extension = _finite_shape(extension, f'{label}.allowed_extension')
            extension_length = _positive(row.get('max_extension_mm'), f'{label}.max_extension_mm')
            if row.get('extension_policy') != 'nearest_branch_to_pedicle_in_finite_domain':
                raise ContractError(f'{label} finite extension requires an explicit branch-to-pedicle policy')
        else:
            extension_length = 0.
        if coordinates == 'relative_to_anchor':
            envelope = float(np.linalg.norm(nodes[:, :2], axis=1).max()) + threshold
            if extension is not None:
                envelope = max(envelope, extension['envelope_mm'])
            if row.get('envelope_mm') is not None and _positive(row['envelope_mm'], f'{label}.envelope_mm') + _TOL < envelope:
                raise ContractError(f'{label}.envelope_mm cannot clip its nodes/contact threshold')
        else:
            # Absolute morphology is verified against each real anchor at build.
            envelope = _positive(row.get('envelope_mm'), f'{label}.envelope_mm')
            if extension is not None:
                envelope = max(envelope, extension['envelope_mm'])
        node_regions = row.get('node_regions')
        if node_regions is not None and len(node_regions) != len(nodes):
            raise ContractError(f'{label}.node_regions must match nodes')
        if nodes.shape[1] == 3 and 'z_tolerance_mm' not in row:
            raise ContractError(f'{label} xyz nodes require explicit z_tolerance_mm')
        result.update(nodes_mm=nodes.tolist(), node_coordinates=coordinates,
                      contact_threshold_mm=threshold, allowed_extension=extension,
                      max_extension_mm=extension_length, envelope_mm=envelope)
    else:
        raise ContractError(f'{label} footprint kind is not implemented')
    if 'z_tolerance_mm' in row:
        result['z_tolerance_mm'] = _positive(row['z_tolerance_mm'], f'{label}.z_tolerance_mm', allow_zero=True)
        if kind != 'morphology_nodes':
            raise ContractError(f'{label} z_tolerance applies to morphology nodes; use z_interval_mm for a footprint')
    if row.get('z_interval_mm') is not None:
        z = np.asarray(row['z_interval_mm'], dtype=float)
        if z.shape != (2,) or not np.all(np.isfinite(z)) or z[1] < z[0]:
            raise ContractError(f'{label}.z_interval_mm requires finite ordered OPL depths')
        result['z_interval_mm'] = z.tolist()
    spacing = row.get('min_contact_spacing_mm', 0.)
    result['min_contact_spacing_mm'] = _positive(spacing, f'{label}.min_contact_spacing_mm', allow_zero=True)
    kernel = dict(row.get('kernel', {'kind': 'uniform'}))
    if kernel.get('kind') == 'gaussian':
        kernel['scale_mm'] = _positive(kernel.get('scale_mm'), f'{label}.kernel.scale_mm')
    elif kernel.get('kind') != 'uniform':
        raise ContractError(f'{label} only uniform or explicitly scaled finite Gaussian weights are supported')
    result['kernel'] = kernel
    result['unit'] = 'mm'
    result['coordinate_frame'] = 'OPL'
    if row.get('radial_support_mm') is not None:
        result['radial_support_mm'] = list(_interval(row['radial_support_mm'], f'{label} morphology source'))
    return result


def validate_footprints(footprints, *, require_configured=True,
                        required_types=('H1', 'H2')):
    """Validate finite spans without creating any cells or contacts.

    Per-cell overrides may be supplied as ``by_hc_id``. A missing type is kept
    as None in candidate-only mode; it is never filled from a central floor.
    """
    footprints = {} if footprints is None else footprints
    if not isinstance(footprints, Mapping):
        raise ContractError('HC footprints must be an H1/H2 mapping')
    unknown = set(footprints) - {'H1', 'H2', 'by_hc_id'}
    if unknown:
        raise ContractError(f'Unknown HC footprint fields: {sorted(unknown)}')
    result = {}
    for subtype in required_types:
        if subtype not in ('H1', 'H2'):
            raise ContractError('HC arbor types must be H1/H2')
        row = footprints.get(subtype)
        if row is None:
            if require_configured:
                raise HCArborConfigurationError(f'{subtype} HC_ARBOR_UNCONFIGURED: supply a sourced or named finite footprint; no inferred central radius')
            result[subtype] = None
        else:
            result[subtype] = _footprint(row, subtype)
    overrides = footprints.get('by_hc_id', {})
    if not isinstance(overrides, Mapping) or any(not isinstance(key, str) or not key for key in overrides):
        raise ContractError('by_hc_id requires nonempty stable HC IDs')
    result['by_hc_id'] = {key: _footprint(row, key) for key, row in overrides.items()}
    return result


def footprint_extent_mm(footprints, *, require_configured=True):
    """Maximum finite OPL anchor envelope for generation-halo planning.

    This is a conservative physical envelope, not a physiological RF radius.
    None is returned if there is no configured footprint in permissive mode.
    """
    rows = validate_footprints(footprints, require_configured=require_configured)
    configured = [row for key, row in rows.items() if key != 'by_hc_id' and row is not None]
    configured.extend(rows['by_hc_id'].values())
    return max((row['envelope_mm'] for row in configured), default=None)


def _coordinates(quantity, ids, meanings, frame, label):
    if not isinstance(quantity, Quantity) or quantity.unit != 'mm' or quantity.meaning not in meanings or quantity.coordinate_frame != frame:
        raise ContractError(f'{label} requires explicit mm/{"|".join(meanings)}/{frame}')
    values = quantity.values
    if values.ndim != 2 or values.shape[0] != len(ids) or values.shape[1] not in (2, 3):
        raise ContractError(f'{label} cell IDs and xy/xyz coordinates do not match')
    return values


def _ids(values, label):
    values = tuple(values)
    if any(not isinstance(value, str) or not value for value in values) or len(set(values)) != len(values):
        raise ContractError(f'{label} requires unique stable nonempty string IDs')
    return values


def _interval(interval, label):
    values = np.asarray(interval, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)) or values[0] < 0 or values[1] <= values[0]:
        raise ContractError(f'{label} requires an explicit finite radial support in mm')
    return float(values[0]), float(values[1])


def _bounds(bounds, label):
    if bounds is None:
        return None
    values = np.asarray(bounds, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)) or values[1] <= values[0] or values[3] <= values[2]:
        raise ContractError(f'{label} requires finite (xmin,xmax,ymin,ymax) OPL bounds')
    return tuple(float(value) for value in values)


def _inside_bounds(points, bounds):
    x0, x1, y0, y1 = bounds
    return ((points[:, 0] >= x0 - _TOL) & (points[:, 0] <= x1 + _TOL) &
            (points[:, 1] >= y0 - _TOL) & (points[:, 1] <= y1 + _TOL))


def _inside_shape(delta, shape):
    if shape['kind'] == 'disk':
        return np.linalg.norm(delta, axis=1) <= shape['radius_mm'] + _TOL
    angle = shape['angle_rad']
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    local = delta @ rotation
    return np.sum((local / np.asarray(shape['axes_mm'])) ** 2, axis=1) <= 1. + _TOL


def _shape_extent(anchor, shape):
    if shape['kind'] == 'disk':
        radius = shape['radius_mm']
        r = float(np.linalg.norm(anchor[:2]))
        return (max(0., r - radius), r + radius), (
            float(anchor[0] - radius), float(anchor[0] + radius),
            float(anchor[1] - radius), float(anchor[1] + radius))
    ax, ay = shape['axes_mm']
    angle = shape['angle_rad']
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    local_anchor = anchor[:2] @ rotation
    # Boundary stationary radii: with u=tan(t/2), the derivative of
    # |anchor + R(a*cos(t),b*sin(t))|^2 is a quartic in u. Enumerating
    # its real roots avoids assuming the enclosing circle is the arbor.
    A, B, C = ax * local_anchor[0], ay * local_anchor[1], ay * ay - ax * ax
    polynomial = np.array([-B, -2 * A - 2 * C, 0., -2 * A + 2 * C, B])
    nonzero = np.flatnonzero(polynomial != 0.)
    roots = [] if not len(nonzero) else np.roots(polynomial[nonzero[0]:])
    angles = [0., np.pi]
    angles.extend(2 * np.arctan(root.real) for root in roots if abs(root.imag) <= 1e-9 * max(1., abs(root.real)))
    t = np.asarray(angles)
    boundary = anchor[:2] + np.c_[ax * np.cos(t), ay * np.sin(t)] @ rotation.T
    radii = np.linalg.norm(boundary, axis=1)
    contains_origin = bool(_inside_shape(-anchor[None, :2], shape)[0])
    minimum = 0. if contains_origin else float(radii.min())
    dx, dy = np.hypot(ax * c, ay * s), np.hypot(ax * s, ay * c)
    return (minimum, float(radii.max())), (
        float(anchor[0] - dx), float(anchor[0] + dx),
        float(anchor[1] - dy), float(anchor[1] + dy))


def _support_extent(anchor, footprint):
    """Finite shape/node support in OPL, independent of density at its anchor."""
    if footprint['kind'] in ('disk', 'ellipse'):
        return _shape_extent(anchor, footprint)
    nodes = np.asarray(footprint['nodes_mm'])
    if footprint['node_coordinates'] == 'relative_to_anchor':
        if nodes.shape[1] == 3 and len(anchor) != 3:
            raise ContractError('Relative xyz morphology requires an explicit xyz HC anchor')
        nodes = nodes + anchor[:nodes.shape[1]]
    threshold = footprint['contact_threshold_mm']
    if np.any(np.linalg.norm(nodes[:, :2] - anchor[:2], axis=1) + threshold > footprint['envelope_mm'] + _TOL):
        raise ContractError('OPL nodes exceed their declared finite HC anchor envelope')
    r = np.linalg.norm(nodes[:, :2], axis=1)
    radial = (max(0., float(r.min()) - threshold), float(r.max()) + threshold)
    rectangle = (float(nodes[:, 0].min() - threshold), float(nodes[:, 0].max() + threshold),
                 float(nodes[:, 1].min() - threshold), float(nodes[:, 1].max() + threshold))
    if footprint['allowed_extension'] is not None:
        other_r, other_box = _shape_extent(anchor, footprint['allowed_extension'])
        radial = (min(radial[0], other_r[0]), max(radial[1], other_r[1]))
        rectangle = (min(rectangle[0], other_box[0]), max(rectangle[1], other_box[1]),
                     min(rectangle[2], other_box[2]), max(rectangle[3], other_box[3]))
    return radial, rectangle


def _candidate_contacts(tree, pedicles, anchor, footprint, cone_ids,
                        cone_regions, eligible_cones):
    candidates = np.asarray(tree.query_ball_point(anchor[:2], footprint['envelope_mm'] + _TOL), dtype=int)
    branch = np.full(len(candidates), -1, dtype=int)
    extension = np.zeros(len(candidates))
    delta = pedicles[candidates, :2] - anchor[:2]
    distance = np.linalg.norm(delta, axis=1)
    if footprint['kind'] in ('disk', 'ellipse'):
        keep = _inside_shape(delta, footprint)
    else:
        nodes = np.asarray(footprint['nodes_mm'])
        if footprint['node_coordinates'] == 'relative_to_anchor':
            if nodes.shape[1] == 3 and len(anchor) != 3:
                raise ContractError('Relative xyz morphology requires an explicit xyz HC anchor')
            nodes = nodes + anchor[:nodes.shape[1]]
        if np.any(np.linalg.norm(nodes[:, :2] - anchor[:2], axis=1) + footprint['contact_threshold_mm'] > footprint['envelope_mm'] + _TOL):
            raise ContractError('Absolute OPL nodes exceed the declared finite HC anchor envelope')
        if nodes.shape[1] == 3 and pedicles.shape[1] != 3:
            raise ContractError('Morphology layer/depth check requires xyz cone pedicle coordinates')
        if 'z_tolerance_mm' in footprint and nodes.shape[1] != 3:
            raise ContractError('z_tolerance_mm requires real xyz morphology nodes')
        node_regions = footprint.get('node_regions')
        allowed = footprint.get('allowed_post_regions')
        eligible_nodes = np.arange(len(nodes))
        if allowed is not None:
            if node_regions is None:
                raise ContractError('Post-region constraints require explicit morphology node_regions')
            eligible_nodes = np.asarray([j for j, region in enumerate(node_regions) if region in allowed], dtype=int)
        keep = np.zeros(len(candidates), dtype=bool)
        threshold = footprint['contact_threshold_mm']
        search_radius = threshold + footprint['max_extension_mm']
        if len(eligible_nodes):
            node_tree = cKDTree(nodes[eligible_nodes, :2])
            for row, point in enumerate(pedicles[candidates]):
                near = np.asarray(node_tree.query_ball_point(point[:2], search_radius + _TOL), dtype=int)
                possible = eligible_nodes[near]
                if nodes.shape[1] == 3:
                    possible = possible[np.abs(nodes[possible, 2] - point[2]) <= footprint['z_tolerance_mm'] + _TOL]
                if not len(possible):
                    continue
                xy_distance = np.linalg.norm(nodes[possible, :2] - point[:2], axis=1)
                order = np.lexsort((possible, xy_distance))
                j, d = int(possible[order[0]]), float(xy_distance[order[0]])
                direct = d <= threshold + _TOL
                may_extend = (footprint['allowed_extension'] is not None and
                              bool(_inside_shape(delta[row:row + 1], footprint['allowed_extension'])[0]) and
                              d <= search_radius + _TOL)
                if direct or may_extend:
                    keep[row], branch[row], distance[row] = True, j, d
                    extension[row] = max(0., d - threshold)
    allowed_pre = footprint.get('allowed_pre_regions')
    if allowed_pre is not None:
        if cone_regions is None:
            raise ContractError('Pre-region constraints require explicit cone_regions')
        keep &= np.asarray([cone_regions[i] in allowed_pre for i in candidates])
    if footprint.get('z_interval_mm') is not None:
        if pedicles.shape[1] != 3:
            raise ContractError('OPL z interval requires xyz cone pedicle coordinates')
        lo, hi = footprint['z_interval_mm']
        keep &= (pedicles[candidates, 2] >= lo - _TOL) & (pedicles[candidates, 2] <= hi + _TOL)
    keep &= eligible_cones[candidates]
    candidates, distance, branch, extension = (values[keep] for values in (candidates, distance, branch, extension))
    order = sorted(range(len(candidates)), key=lambda j: (float(distance[j]), cone_ids[candidates[j]]))
    candidates, distance, branch, extension = (values[order] for values in (candidates, distance, branch, extension))
    spacing = footprint['min_contact_spacing_mm']
    if spacing:
        selected = []
        for j, cone in enumerate(candidates):
            if not selected or np.all(np.linalg.norm(pedicles[candidates[selected], :2] - pedicles[cone, :2], axis=1) >= spacing - _TOL):
                selected.append(j)
        candidates, distance, branch, extension = (values[selected] for values in (candidates, distance, branch, extension))
    return candidates, distance, branch, extension


def _quotas(divergence, cone_ids, cone_types, pedicles):
    if divergence is None:
        return None, {'status': 'DIVERGENCE_UNSELECTED_CANDIDATE_GRAPH',
                      'degree_targets_are_physiological': False}
    if not isinstance(divergence, Mapping):
        raise ContractError('Divergence requires an explicit sourced/project profile')
    provenance = _provenance(divergence, 'divergence')
    if divergence.get('coordinate_frame') != 'OPL' or divergence.get('denominator') != 'all_cones':
        raise ContractError('Divergence source location must be OPL and its denominator all_cones')
    support = _interval(divergence.get('radial_support_mm'), 'divergence')
    r = np.linalg.norm(pedicles[:, :2], axis=1)
    if np.any(r < support[0] - _TOL) or np.any(r > support[1] + _TOL):
        raise ContractError('CONTACT_DOMAIN_UNSUPPORTED: pedicle location exceeds selected divergence profile')
    if divergence.get('rounding_policy', 'balanced_largest_remainder_all_cone_total') != 'balanced_largest_remainder_all_cone_total':
        raise ContractError('Unsupported divergence integer quota convention')
    values = divergence.get('values')
    if not isinstance(values, Mapping) or set(values) != {'H1', 'H2'}:
        raise ContractError('Divergence requires H1 and H2 scalar/per-cone values')
    result, audit = {}, {}
    n = len(cone_ids)
    for subtype in ('H1', 'H2'):
        try:
            value = np.broadcast_to(np.asarray(values[subtype], dtype=float), (n,)).copy()
        except ValueError as error:
            raise ContractError(f'Divergence {subtype} shape must follow cone IDs') from error
        if not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ContractError('Divergence requires finite nonnegative degree targets')
        eligible = np.flatnonzero(cone_types != 'S') if subtype == 'H1' else np.arange(n)
        target = int(np.floor(value.sum() + .5))
        quota = np.zeros(n, dtype=int)
        impossible = target if not len(eligible) or not value[eligible].sum() else 0
        expected = np.zeros(n)
        if not impossible and target:
            expected[eligible] = value[eligible] * (target / value[eligible].sum())
            quota[eligible] = np.floor(expected[eligible]).astype(int)
            remainder = target - int(quota.sum())
            order = sorted(eligible, key=lambda i: (-(expected[i] - quota[i]), cone_ids[i]))
            for i in order[:remainder]:
                quota[i] += 1
        result[subtype] = quota
        audit[subtype] = {'raw_all_cone_edge_target': float(value.sum()),
            'integer_all_cone_edge_target': target, 'eligible_cone_count': len(eligible),
            'integer_eligible_quotas': quota.tolist(), 'eligible_expected_degree': expected.tolist(),
            'unallocatable_no_eligible_mass': impossible,
            'denominator': 'all_cones', 'eligible_redistribution': 'preserve all-cone total over permitted types; stable-ID tie breaks'}
    return result, {'status': 'EXPLICIT_DIVERGENCE_QUOTAS', 'provenance': provenance,
        'radial_support_mm': list(support), 'rounding_policy': 'balanced_largest_remainder_all_cone_total',
        'quota_scope': 'supplied complete OPL cone list; build canonical contacts then crop, independent per-tile quota recomputation does not guarantee common edges',
        'subtypes': audit, 'degree_targets_are_physiological': False}


def _normalize(matrix, rule):
    if rule == 'absolute_contact_weight':
        return matrix.tocsr()
    if rule != 'unit_pooling':
        raise ContractError('Geometry normalization must be unit_pooling or absolute_contact_weight')
    total = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1., total, out=np.zeros_like(total), where=total > 0)
    return (sparse.diags(inverse) @ matrix).tocsr()


def build_hc_arbor_contacts(cone_ids, cone_input_quantity, cone_pedicle_quantity,
        hc_ids, hc_types, hc_soma_quantity, hc_anchor_quantity, owner_cone_mask,
        footprints, *, cone_types, source_radial_support_mm=(0., 7.1),
        cone_pedicle_generation_bounds_mm=None, hc_generation_bounds_mm=None,
        divergence_profile=None, require_configured=True, require_owner_quota=False,
        feedforward_normalization='unit_pooling', feedback_normalization='unit_pooling',
        cone_pedicle_ids=None, cone_regions=None, require_complete_contact_halo=False,
        anchor_applicability_band_mm=None,source_anchor_band_mm=None,
        final_contact_distance_mm=None,contact_policy=None,owner_hc_mask=None):
    """Query actual pedicles throughout finite arbors, then make sparse contacts.

    ``None`` footprints are exported as UNCONFIGURED in permissive mode. A
    configured mode rejects them instead of inventing central spans. Supplying
    bounds enables real support-truncation diagnostics; unspecified bounds are
    never described as halo-complete. Density at an anchor is not an input.
    """
    cone_ids, hc_ids = _ids(cone_ids, 'cone'), _ids(hc_ids, 'HC')
    if set(cone_ids) & set(hc_ids):
        raise ContractError('Cone and HC stable ID namespaces must be distinct')
    if cone_pedicle_ids is not None and tuple(cone_pedicle_ids) != cone_ids:
        raise ContractError('Cone RF/pedicle ID order mismatch; index-only reassociation forbidden')
    q = _coordinates(cone_input_quantity, cone_ids, ('functional_position',), 'functional_visual_field', 'cone input')
    p = _coordinates(cone_pedicle_quantity, cone_ids, ('pedicle_position',), 'OPL', 'cone pedicle')
    soma = _coordinates(hc_soma_quantity, hc_ids, ('anatomical_position',), 'anatomical_retina', 'HC soma')
    anchors = _coordinates(hc_anchor_quantity, hc_ids, ('arbor_anchor_position', 'dendritic_position'), 'OPL', 'HC arbor anchor')
    types, htypes = np.asarray(cone_types), np.asarray(hc_types)
    if types.shape != (len(cone_ids),) or np.any(~np.isin(types, ('L', 'M', 'S'))):
        raise ContractError('Cone types must follow the same L/M/S cone IDs')
    if htypes.shape != (len(hc_ids),) or np.any(~np.isin(htypes, ('H1', 'H2'))):
        raise ContractError('HC types must follow the same H1/H2 HC IDs')
    owner = np.asarray(owner_cone_mask)
    if owner.shape != (len(cone_ids),) or owner.dtype.kind != 'b':
        raise ContractError('Input owner mask must be boolean and follow cone IDs')
    if cone_regions is not None and len(cone_regions) != len(cone_ids):
        raise ContractError('cone_regions must follow cone IDs')
    configured = validate_footprints(footprints, require_configured=False)
    unknown_overrides = set(configured['by_hc_id']) - set(hc_ids)
    if unknown_overrides:
        raise ContractError('Morphology override refers to an unknown stable HC ID')
    support = _interval(source_radial_support_mm, 'HC/contact source')
    anchor_band=None if anchor_applicability_band_mm is None else _interval(anchor_applicability_band_mm,'HC anchor applicability')
    geometry_first=contact_policy=='finite_geometry_first_v1'
    if anchor_band is not None and not geometry_first:
        raise ContractError('Separate anchor applicability requires the named finite-geometry-first policy')
    if geometry_first and (divergence_profile is not None or require_owner_quota):
        raise ContractError('Geometry-first uses references diagnostically, never per-cone quotas')
    if geometry_first and (anchor_band is None or source_anchor_band_mm is None or not require_configured):
        raise ContractError('Geometry-first requires explicit anchor/source bands and configured finite arbors')
    source_anchor_band=None if source_anchor_band_mm is None else _interval(source_anchor_band_mm,'original HC source anchor band')
    if final_contact_distance_mm is not None:final_contact_distance_mm=_positive(final_contact_distance_mm,'final contact cutoff')
    hc_owner=np.zeros(len(hc_ids),bool) if owner_hc_mask is None else np.asarray(owner_hc_mask)
    if hc_owner.shape!=(len(hc_ids),) or hc_owner.dtype.kind!='b':raise ContractError('HC owner mask must follow HC IDs')
    terminal_bounds = _bounds(cone_pedicle_generation_bounds_mm, 'cone pedicle generation domain')
    hc_bounds = _bounds(hc_generation_bounds_mm, 'HC generation domain')
    checked=((p, 'cone pedicle'), (soma, 'HC soma'), (anchors, 'HC anchor')) if anchor_band is None else ((anchors,'HC anchor'),)
    for points, label in checked:
        radial = np.linalg.norm(points[:, :2], axis=1)
        band=support if anchor_band is None else anchor_band
        if np.any(radial < band[0] - _TOL) or np.any(radial > band[1] + _TOL):
            domain_label='physical source radial support' if anchor_band is None else 'named project anchor applicability band'
            raise ContractError(f'CONTACT_DOMAIN_UNSUPPORTED: {label} exceeds {domain_label}')
    if terminal_bounds is not None and not np.all(_inside_bounds(p, terminal_bounds)):
        raise ContractError('Cone pedicle outside declared OPL terminal generation domain')
    if hc_bounds is not None and not np.all(_inside_bounds(soma, hc_bounds)):
        raise ContractError('HC soma outside declared physical HC generation domain')
    quotas, divergence_audit = _quotas(divergence_profile, cone_ids, types, p)
    tree = cKDTree(p[:, :2])
    rows, cols, distances, weights, edge_metadata, hc_rows = [], [], [], [], [], []
    for j, (cell_id, subtype) in enumerate(zip(hc_ids, htypes)):
        footprint = configured['by_hc_id'].get(cell_id, configured[subtype])
        if footprint is None:
            if require_configured:
                raise HCArborConfigurationError(f'{cell_id} {subtype} HC_ARBOR_UNCONFIGURED: no finite span selected')
            hc_rows.append({'hc_id': cell_id, 'subtype': str(subtype), 'status': 'HC_ARBOR_UNCONFIGURED', 'candidate_count': 0})
            continue
        radial_range, rectangle = _support_extent(anchors[j], footprint)
        if anchor_band is None and (radial_range[0] < support[0] - _TOL or radial_range[1] > support[1] + _TOL):
            raise ContractError(f'CONTACT_DOMAIN_UNSUPPORTED: finite arbor of {cell_id} extends beyond source support; never clip its halo')
        if footprint.get('radial_support_mm') is not None:
            own_support = footprint['radial_support_mm']
            if radial_range[0] < own_support[0] - _TOL or radial_range[1] > own_support[1] + _TOL:
                raise ContractError(f'CONTACT_DOMAIN_UNSUPPORTED: finite arbor of {cell_id} exceeds its own morphology source domain')
        complete = None if terminal_bounds is None else bool(np.all(_inside_bounds(np.array([[rectangle[0], rectangle[2]], [rectangle[1], rectangle[3]]]), terminal_bounds)))
        if require_complete_contact_halo and complete is not True:
            raise ContractError(f'CONTACT_BOUNDARY_MISSING: full finite arbor of {cell_id} is not covered by the terminal generation domain')
        eligible_cones = types != 'S' if subtype == 'H1' else np.ones(len(types), bool)
        candidates, distance, branch, extension = _candidate_contacts(tree, p, anchors[j], footprint, cone_ids, cone_regions, eligible_cones)
        if final_contact_distance_mm is not None:
            inside=np.linalg.norm(p[candidates,:2]-anchors[j,:2],axis=1)<=final_contact_distance_mm+_TOL
            candidates,distance,branch,extension=(v[inside] for v in (candidates,distance,branch,extension))
        if geometry_first and (hc_owner[j] or np.any(owner[candidates])) and complete is not True:
            raise ContractError(f'CONTACT_REQUIRED_QUERY_TRUNCATED: required owner HC/contact {cell_id} lacks complete pedicle query envelope')
        for i, d, node, length in zip(candidates, distance, branch, extension):
            kernel = footprint['kernel']
            weight = 1. if kernel['kind'] == 'uniform' else float(np.exp(-(d / kernel['scale_mm']) ** 2))
            # Keep finite geometric edges even when a Gaussian underflows. An
            # underflow cannot remove the common forward/feedback topology.
            weight = max(weight, np.finfo(float).tiny)
            rows.append(j); cols.append(int(i)); distances.append(float(d)); weights.append(weight)
            edge_metadata.append({'morphology_node_index': None if node < 0 else int(node),
                                  'extension_length_mm': float(length)})
        status = ('CONTACT_CANDIDATES' if len(candidates) else
                  'CONTACT_BOUNDARY_MISSING' if complete is False else
                  'ZERO_GENERATED_TERMINALS_HALO_UNDECLARED' if complete is None else
                  'ZERO_LEGAL_TERMINALS_IN_FINITE_SUPPORT')
        hc_rows.append({'hc_id': cell_id, 'subtype': str(subtype), 'status': status,
            'candidate_count': len(candidates), 'arbor': footprint,
            'finite_envelope_radial_mm': list(radial_range), 'finite_envelope_bounds_mm': list(rectangle),
            'support_extent_policy': 'disk or ellipse extent; morphology node contact neighborhoods plus complete allowed extension domain',
            'terminal_generation_covers_full_envelope': complete,
            'boundary_status': 'CONTACT_HALO_UNDECLARED' if complete is None else ('COMPLETE_ENVELOPE' if complete else 'CONTACT_BOUNDARY_MISSING'),
            'zero_candidate_interpretation': 'finite generated nodes only; no anchor-density inference',
            'span_is_functional_RF': False})
        if geometry_first:hc_rows[-1]['cone_distance_lookup']={int(i):float(d) for i,d in zip(candidates,distance)}
    # Per-cone selection is confined to this already finite candidate graph.
    selected = np.ones(len(rows), dtype=bool)
    cone_rows = []
    for subtype in ('H1', 'H2'):
        by_cone = [[] for _ in cone_ids]
        for edge, (j, i) in enumerate(zip(rows, cols)):
            if htypes[j] == subtype:
                by_cone[i].append(edge)
        for i, choices in enumerate(by_cone):
            quota = None if quotas is None else int(quotas[subtype][i])
            ordered = sorted(choices, key=lambda edge: (distances[edge], hc_ids[rows[edge]]))
            kept = ordered if quota is None else ordered[:quota]
            if quota is not None:
                selected[ordered[quota:]] = False
            missing = 0 if quota is None else max(0, quota - len(kept))
            cone_rows.append({'cone_id': cone_ids[i], 'subtype': subtype, 'input_owner': bool(owner[i]),
                'candidate_count': len(choices), 'target_degree': quota, 'actual_degree': len(kept),
                'missing_degree': missing, 'status': 'CONTACT_QUOTA_INFEASIBLE' if missing else ('CANDIDATE_GRAPH_UNCALIBRATED' if quota is None else 'GEOMETRIC_QUOTA_SATISFIED'),
                'maximum_contact_distance_mm': max((distances[edge] for edge in kept), default=None),
                'input_eccentricity_mm': float(np.linalg.norm(q[i, :2])),
                'pedicle_eccentricity_mm': float(np.linalg.norm(p[i, :2]))})
            if require_owner_quota and owner[i] and missing:
                raise ContractError(f'CONTACT_QUOTA_INFEASIBLE: owner {cone_ids[i]} lacks {missing} finite {subtype} contacts; no radius enlargement/nearest fallback')
    if require_owner_quota and quotas is None:
        raise ContractError('Owner quota enforcement requires a selected divergence profile')
    if require_owner_quota and any(row['unallocatable_no_eligible_mass'] for row in divergence_audit['subtypes'].values()):
        raise ContractError('CONTACT_QUOTA_INFEASIBLE: all-cone divergence target has no eligible subtype mass')
    rr, cc, ww = (np.asarray(values)[selected] for values in (rows, cols, weights))
    raw = sparse.csr_matrix((ww, (rr.astype(int), cc.astype(int))), shape=(len(hc_ids), len(cone_ids)))
    feedforward = _normalize(raw, feedforward_normalization)
    feedback = _normalize(raw.T.tocsr(), feedback_normalization)
    edges = []
    for edge in np.flatnonzero(selected):
        j, i = rows[edge], cols[edge]
        key = hashlib.sha256(json.dumps([hc_ids[j], cone_ids[i]], separators=(',', ':')).encode()).hexdigest()
        edges.append({'edge_id': 'hc_arbor:' + key, 'cone_id': cone_ids[i], 'hc_id': hc_ids[j],
            'hc_type': str(htypes[j]), 'distance_mm': distances[edge],
            'distance_metric': 'nearest_allowed_morphology_node_xy' if edge_metadata[edge]['morphology_node_index'] is not None else 'arbor_anchor_xy',
            'arbor_anchor_distance_mm': float(np.linalg.norm(p[i, :2] - anchors[j, :2])),
            'raw_contact_weight': weights[edge], 'weight_unit': 'dimensionless',
            'feedforward_weight': float(feedforward[j, i]), 'feedback_weight': float(feedback[i, j]),
            **edge_metadata[edge]})
    actual_hc_degree = np.diff(raw.indptr)
    for j, row in enumerate(hc_rows):
        row['actual_input_cone_count'] = int(actual_hc_degree[j])
        row['raw_weight_sum'] = float(raw.getrow(j).sum())
    paired = [{'cone_id': cell_id, 'input_xy_mm': q[i, :2].tolist(),
               'pedicle_xy_mm': p[i, :2].tolist(), 'input_owner': bool(owner[i])}
              for i, cell_id in enumerate(cone_ids)]
    topology = sorted((edge['hc_id'], edge['cone_id']) for edge in edges)
    audit = {'profile_id': 'finite_hc_arbor_pedicle_contacts_v1',
        'implementation_status': 'STATIC_IMPLEMENTATION_NOT_RUN',
        'geometry_status': 'HC_ARBOR_UNCONFIGURED' if any(row['status'] == 'HC_ARBOR_UNCONFIGURED' for row in hc_rows) else 'FINITE_GEOMETRIC_GRAPH',
        'claim': 'configured geometric candidate/contact graph; no morphology, dynamics or physiological calibration claim',
        'source_radial_support_mm': list(support), 'coordinate_unit': 'mm',
        'input_frame': 'functional_visual_field', 'contact_frame': 'OPL', 'soma_frame': 'anatomical_retina',
        'input_and_pedicle_identity': 'one supplied cone ID follows input q to pedicle p; no cone resampling or ID substitution',
        'paired_cones': paired, 'paired_cone_hash': canonical_hash(sorted(paired, key=lambda row: row['cone_id'])),
        'hc_geometry': [{'hc_id': cell_id, 'type': str(htypes[j]), 'soma_xy_mm': soma[j, :2].tolist(),
                         'arbor_anchor_xy_OPL_mm': anchors[j, :2].tolist()} for j, cell_id in enumerate(hc_ids)],
        'input_owner_definition': 'supplied RF owner mask; terminal or HC ownership is a separate geometry question',
        'cone_pedicle_generation_bounds_mm': terminal_bounds, 'hc_generation_bounds_mm': hc_bounds,
        'h1_eligibility': 'L/M only; zero S contacts is named project simplification H1_zero_S_contact_v1',
        'h2_eligibility': 'L/M/S; geometric eligibility does not set S-specific gain',
        'query_policy': 'actual pedicles throughout each complete configured footprint, independent of anchor density',
        'contact_topology_hash': canonical_hash(topology), 'edge_count': len(edges),
        'pair_multiplicity': 'one contact per cone/HC pair', 'edges': edges, 'hc_rows': hc_rows,
        'cone_rows': cone_rows, 'divergence': divergence_audit,
        'feedforward': {'shape': list(feedforward.shape), 'normalization': feedforward_normalization, 'unit': 'dimensionless'},
        'feedback': {'shape': list(feedback.shape), 'normalization': feedback_normalization, 'unit': 'dimensionless', 'topology': 'same contacts reversed'},
        'sparse_storage': 'cKDTree finite candidates and O(N+E) sparse matrices; no dense all-pairs distance matrix',
        'geometry_is_conductance': False, 'discrete_sum_jacobian_multiplier': False,
        'unconfigured_spans_are_inferred': False, 'outside_support_nearest_fallback': False}
    if geometry_first:
        from .hc_contact_diagnostics import finite_contact_diagnostics
        stats,impact=finite_contact_diagnostics(raw,p,types,htypes,owner,hc_owner,hc_rows,
            _normalize,feedforward_normalization,feedback_normalization)
        def extent(points,mask):
            a=points[mask,:2]
            return {'count':len(a),'radial_mm':[float(np.linalg.norm(a,axis=1).min()),float(np.linalg.norm(a,axis=1).max())] if len(a) else None,
                'bounds_mm':[float(a[:,0].min()),float(a[:,0].max()),float(a[:,1].min()),float(a[:,1].max())] if len(a) else None}
        def footprint_extent(mask):
            a=[r for j,r in enumerate(hc_rows) if mask[j]]
            return {'count':len(a),'radial_mm':[min(r['finite_envelope_radial_mm'][0] for r in a),
                max(r['finite_envelope_radial_mm'][1] for r in a)] if a else None,
                'bounds_mm':[min(r['finite_envelope_bounds_mm'][0] for r in a),max(r['finite_envelope_bounds_mm'][1] for r in a),
                    min(r['finite_envelope_bounds_mm'][2] for r in a),max(r['finite_envelope_bounds_mm'][3] for r in a)] if a else None}
        required=hc_owner|(np.asarray((raw[:,owner]!=0).sum(1)).ravel()>0)
        if len(topology)!=len(set(topology)):raise ContractError('Duplicate cone/HC partner in finite graph')
        for row in hc_rows:row.pop('cone_distance_lookup',None)
        audit.update(contact_policy=contact_policy,per_cone_exact_divergence=False,
            divergence_reference_role='population_diagnostic',population_diagnostics=stats,graph_weight_impact=impact,
            pair_dedup=True,rescale_to_divergence_target=False,final_contact_distance_mm=final_contact_distance_mm,
            source_anchor_band_mm=source_anchor_band_mm,application_anchor_band_mm=list(anchor_band) if anchor_band else None,
            actual_HC_anchor_extent={t:{scope:extent(anchors,(htypes==t)&(hc_owner if scope=='owner' else ~hc_owner))
                for scope in ('owner','halo')} for t in ('H1','H2')},
            actual_HC_footprint_extent={t:{scope:footprint_extent((htypes==t)&(hc_owner if scope=='owner' else ~hc_owner))
                for scope in ('owner','halo')} for t in ('H1','H2')},
            required_pedicle_query_domain={'materialized_bounds_mm':terminal_bounds,
                'required_owner_dependency_footprint_extent':footprint_extent(required),
                'owner_required_HC_complete':True,'halo_incomplete_count':sum(r['terminal_generation_covers_full_envelope'] is not True for r in hc_rows),
                'boundary_policy':'existing finite_induced_graph; incomplete outer halo is explicit, no recurrent closure claim'},
            extrapolation_label_by_HC=[('bounded_edge_extrapolation' if not source_anchor_band[0]<=r<=source_anchor_band[1]
                else 'source_band_equal_area_project_prior') for r in np.linalg.norm(anchors[:,:2],axis=1)])
    return feedforward, feedback, audit


def homotypic_gap_candidates(hc_ids, hc_types, soma_quantity,
                            max_distance_mm=None, *, source_profile_id=None,
                            approximation_id='HC_homotypic_Delaunay_candidates_v1'):
    """Export bounded homotypic soma neighbor candidates, never electrical edges.

    An explicit distance gate is mandatory for a candidate graph. This helper
    neither supplies a coupling conductance nor changes the dynamics graph.
    Collinear populations use adjacent projected nodes; small populations use
    their only possible pair. No joggle, copied cells or distant nearest fill.
    """
    hc_ids = _ids(hc_ids, 'HC gap candidate')
    points = _coordinates(soma_quantity, hc_ids, ('anatomical_position',),
                          'anatomical_retina', 'HC gap candidate soma')
    htypes = np.asarray(hc_types)
    if htypes.shape != (len(hc_ids),) or np.any(~np.isin(htypes, ('H1', 'H2'))):
        raise ContractError('Gap candidate types must follow H1/H2 HC IDs')
    if max_distance_mm is None:
        gates = {'H1': None, 'H2': None}
    elif isinstance(max_distance_mm, Mapping):
        if set(max_distance_mm) - {'H1', 'H2'}:
            raise ContractError('Gap candidate distance gates support H1/H2 only')
        gates = {subtype: max_distance_mm.get(subtype) for subtype in ('H1', 'H2')}
    else:
        gates = dict.fromkeys(('H1', 'H2'), max_distance_mm)
    if any(gate is not None for gate in gates.values()) and not source_profile_id:
        raise ContractError('Configured HC gap geometry requires explicit source_profile_id; it does not supply a conductance')
    edges, audits = [], []
    for subtype in ('H1', 'H2'):
        gate = gates[subtype]
        indices = np.flatnonzero(htypes == subtype)
        if gate is None:
            audits.append({'subtype': subtype, 'status': 'GAP_DISTANCE_UNCONFIGURED',
                           'cell_count': len(indices), 'edge_count': 0})
            continue
        gate = _positive(gate, f'{subtype}.gap_candidate_max_distance_mm')
        xy = points[indices, :2]
        pairs, method, status = set(), 'small_count', 'FINITE_GAP_CANDIDATES'
        if len(indices) == 2:
            pairs.add((0, 1))
        elif len(indices) > 2:
            centered = xy - xy.mean(axis=0)
            rank = np.linalg.matrix_rank(centered)
            if rank == 0:
                method, status = 'none', 'COINCIDENT_SOMA_GEOMETRY_UNRESOLVED'
            elif rank == 1:
                axis = np.linalg.svd(centered, full_matrices=False)[2][0]
                projection = xy @ axis
                order = sorted(range(len(xy)), key=lambda i: (float(projection[i]), hc_ids[indices[i]]))
                pairs.update(tuple(sorted(pair)) for pair in zip(order[:-1], order[1:]))
                method = 'collinear_projected_adjacent'
            else:
                try:
                    triangles = Delaunay(xy).simplices
                    for triangle in triangles:
                        for a, b in ((0, 1), (1, 2), (0, 2)):
                            pairs.add(tuple(sorted((int(triangle[a]), int(triangle[b])))))
                    method = 'homotypic_Delaunay'
                except QhullError:
                    method, status = 'none', 'DEGENERATE_DELAUNAY_NOT_BUILT'
        candidates = len(pairs)
        accepted = 0
        for a, b in sorted(pairs, key=lambda pair: tuple(sorted((hc_ids[indices[pair[0]]], hc_ids[indices[pair[1]]])))):
            distance = float(np.linalg.norm(xy[a] - xy[b]))
            if distance > gate + _TOL:
                continue
            first, second = sorted((hc_ids[indices[a]], hc_ids[indices[b]]))
            edges.append({'hc_a': first, 'hc_b': second, 'subtype': subtype,
                          'distance_mm': distance, 'candidate_only': True})
            accepted += 1
        audits.append({'subtype': subtype, 'status': status, 'cell_count': len(indices),
            'method': method, 'max_distance_mm': gate, 'ungated_candidate_count': candidates,
            'edge_count': accepted, 'omitted_outside_finite_gate': candidates - accepted})
    return {'profile_id': 'HC_homotypic_gap_candidates_v1',
        'status': 'GAP_DISTANCE_UNCONFIGURED' if all(gate is None for gate in gates.values()) else 'EXPORTED_GEOMETRIC_CANDIDATES_ONLY',
        'implementation_status': 'STATIC_IMPLEMENTATION_NOT_RUN',
        'source_profile_id': source_profile_id, 'approximation_id': approximation_id,
        'evidence_class': 'project_approximation', 'frame': 'anatomical_retina', 'unit': 'mm',
        'edges': edges, 'subtypes': audits, 'edge_count': len(edges),
        'symmetry': 'undirected pairs; an electrical consumer would require symmetric nonnegative sourced conductance',
        'conductance': 'UNBOUND', 'connected_to_dynamics': False,
        'geometry_is_nS': False, 'topology_hash': canonical_hash(edges)}
