"""User-selected functional input-anchor density on registered Cone pedicles."""
from fractions import Fraction
import hashlib
import numpy as np
from scipy.spatial import cKDTree
from .contracts import ContractError

RATIOS = {'diffuse_BC_ON': Fraction(10514, 26512),
          'diffuse_BC_OFF': Fraction(23327, 26512), 'S_ON_BC': Fraction(2702, 1452)}

def cone_bc_densities(rho_lm, rho_s):
    lm, s = np.broadcast_arrays(np.asarray(rho_lm, dtype=np.float64), np.asarray(rho_s, dtype=np.float64))
    if np.any(~np.isfinite(lm)) or np.any(~np.isfinite(s)) or np.any(lm < 0) or np.any(s < 0):
        raise ContractError('Registered pedicle densities must be finite, nonnegative, in cells/mm2')
    total = lm + s
    return {'midget_BC_ON': lm.copy(), 'midget_BC_OFF': total,
            'diffuse_BC_ON': total * float(RATIOS['diffuse_BC_ON']),
            'diffuse_BC_OFF': total * float(RATIOS['diffuse_BC_OFF']),
            'S_ON_BC': s * float(RATIOS['S_ON_BC'])}

def anchor_rows(ids, labels, eligible):
    """Allocate exact finite-population quotas; no temporal-subtype duplication.

    Hash ranking is an engineering placement rule on the supplied parent set.
    It is fixed before responses exist and is never rerun on an ROI crop.
    """
    labels = np.asarray(labels).astype('U7')
    eligible = np.asarray(eligible, dtype=np.int64)
    if (labels.shape != (len(ids),) or len(set(ids)) != len(ids) or
            np.any(eligible < 0) or np.any(eligible >= len(ids)) or len(np.unique(eligible)) != len(eligible)):
        raise ContractError('Unique registered parent IDs/rows and matching labels required')
    if np.any(~np.isin(labels, ('L', 'M', 'S'))):
        raise ContractError('All parent pedicles need registered L/M/S identities')
    result = {}
    for name in ('midget_BC_ON', 'midget_BC_OFF', 'diffuse_BC_ON', 'diffuse_BC_OFF', 'S_ON_BC'):
        base = np.asarray(eligible, dtype=np.int64)
        if name == 'midget_BC_ON':
            base = base[labels[base] != 'S']
        elif name == 'S_ON_BC':
            base = base[labels[base] == 'S']
        if name.startswith('midget'):
            rows = [(int(i), 0) for i in base]
        else:
            ratio = RATIOS[name]
            # Exact rational round-half-up avoids floating-point quota drift.
            target = (2 * len(base) * ratio.numerator + ratio.denominator) // (2 * ratio.denominator)
            order = sorted(base, key=lambda i: (hashlib.sha256((ids[i] + '|' + name + '|BC_anchor_quota_v1').encode()).digest(), ids[i]))
            whole, remainder = divmod(target, len(base)) if len(base) else (0, 0)
            rows = [(int(i), slot) for i in base for slot in range(whole)]
            rows.extend((int(i), whole) for i in order[:remainder])
        rows.sort(key=lambda row: (ids[row[0]], row[1]))
        result[name] = (np.array([r[0] for r in rows], dtype=np.int64),
                        np.array([r[1] for r in rows], dtype=np.int64))
    return result

def crop_anchor_rows(groups, pedicles, query):
    if query is None:
        return groups
    if query.get('unit') != 'mm' or query.get('frame') != 'OPL':
        raise ContractError('BC anchor crop requires explicit mm/OPL bounds')
    bounds = np.asarray(query['bounds_mm'], dtype=np.float64)
    if bounds.shape != (4,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ContractError('Finite increasing BC anchor rectangle required')
    x0,x1,y0,y1 = bounds
    result = {}
    for name,(indices,slots) in groups.items():
        p = pedicles[indices]
        keep = (p[:,0]>=x0)&(p[:,0]<x1)&(p[:,1]>=y0)&(p[:,1]<y1)
        result[name] = indices[keep],slots[keep]
    return result


class RegisteredPedicleField:
    """Finite 2-D pedicle census, distinct from an IS or HC density curve."""
    def __init__(self, cone, config):
        record = cone.record['coordinates'].get('opl_pedicle', {})
        if (record.get('unit'), record.get('meaning'), record.get('frame')) != ('mm', 'pedicle_position', 'OPL'):
            raise ContractError('BC density requires registered mm OPL pedicle coordinates')
        source = cone.record['source_bindings']
        if not source.get('Schein'):
            raise ContractError('Named registered IS-to-pedicle transform required; no IS-density substitution')
        self.xy = cone.coordinates['opl_pedicle']
        self.rf = cone.coordinates['rf']
        self.labels = cone.extras['subtype'].astype('U1')
        if (self.xy.shape != self.rf.shape or not np.isfinite(self.xy).all() or
                not np.isfinite(self.rf).all() or np.any(~np.isin(self.labels, ('L', 'M', 'S')))):
            raise ContractError('Complete finite registered pedicle geometry and L/M/S identities required')
        self.complete_radius = float(source['fixed_RF_query_radius_mm'])
        if (not np.isfinite(self.complete_radius) or self.complete_radius <= 0 or
                np.any(np.linalg.norm(self.rf, axis=1) > self.complete_radius + 1e-12)):
            raise ContractError('Frozen canonical parent RF-query disk identity required')
        if np.any(np.linalg.norm(self.xy, axis=1) + 1e-12 < np.linalg.norm(self.rf, axis=1)):
            raise ContractError('Outward Schein registration prerequisite failed; explicit pedicle query-domain proof required')
        self.window = float(config['local_density']['radius_mm'])
        self.cap = float(config['geometry']['S_ON_max_contact_mm'])
        self.scope = float(config['application_anchor_radius_mm'][1])
        self.owner_support = config.get('owner_support')
        if self.window != .04 or self.cap != .04:
            raise ContractError('Frozen existing40um support and new BC application domain required')
        if self.owner_support is None:
            if self.scope != 1.338:
                raise ContractError('Explicit owner/support extension authorization required')
            self.owner_rows = None
        else:
            domain = self.owner_support
            if (domain.get('authorization') != 'Sentinel_5a05022065f081919c81f4f9db2d6697' or
                    domain.get('owner_RF_radius_mm') != 1.338 or
                    domain.get('BC_anchor_radius_mm') != self.scope or
                    getattr(cone, 'binding', None) != domain['cone_source_table']):
                raise ContractError('Source-bound owner/support contract differs')
            self.owner_rows = np.flatnonzero(np.linalg.norm(self.rf, axis=1) <= domain['owner_RF_radius_mm'])
            if (len(self.owner_rows) != domain['owner_cones'] or
                    np.max(np.linalg.norm(self.xy[self.owner_rows], axis=1)) != self.scope):
                raise ContractError('Registered owner/pedicle extent changed; explicit new domain required')
        self.trees = {kind: cKDTree(self.xy[self.eligible(kind)]) for kind in ('LM', 'S')}
        self.rows = {kind: np.flatnonzero(self.eligible(kind)) for kind in ('LM', 'S')}

    def eligible(self, kind):
        return self.labels == 'S' if kind == 'S' else self.labels != 'S'

    def density(self, center, kind):
        if np.linalg.norm(center) + self.window > self.complete_radius:
            raise ContractError('BC_LOCAL_PEDICLE_DENSITY_QUERY_INCOMPLETE: no cropped-area rescaling')
        return len(self.trees[kind].query_ball_point(center, self.window)) / (np.pi * self.window**2)

    def contacts(self, center, name):
        center = np.asarray(center, dtype=np.float64)
        if (center.shape != (2,) or not np.isfinite(center).all() or
                np.linalg.norm(center) > self.scope or name not in ('diffuse_BC_ON', 'diffuse_BC_OFF', 'S_ON_BC')):
            raise ContractError('Supported current BC anchor/type required; no radial extrapolation')
        kind = 'S' if name == 'S_ON_BC' else 'LM'
        density = self.density(center, kind)
        if density == 0:
            return np.empty(0, dtype=np.int64), {'status': 'INCOMPLETE_NO_ELIGIBLE_PEDICLE', 'eligible_density_cells_mm2': 0.}
        k = 1.7 if kind == 'S' else 8.
        raw_radius = np.sqrt(k / (np.pi * density))
        radius = min(raw_radius, self.cap) if kind == 'S' else raw_radius
        if np.linalg.norm(center) + radius > self.complete_radius:
            return np.empty(0, dtype=np.int64), {'status': 'INCOMPLETE_CONTACT_QUERY_DOMAIN', 'radius_mm': float(radius)}
        found = self.rows[kind][self.trees[kind].query_ball_point(center, radius)]
        found.sort()
        return found, {'status': 'READY_PROJECT_CONTACT_DISK' if len(found) else 'INCOMPLETE_EMPTY_CONTACT_DISK',
            'existing_external_RF_cone_inputs': int(np.sum(np.linalg.norm(self.rf[found],axis=1)>self.owner_support['owner_RF_radius_mm'])) if self.owner_support else None,
            'external_input_policy': 'existing finite source only; outside RF owners held, never silently promoted to optical/dynamic owner',
            'eligible_density_cells_mm2': float(density), 'density_window_mm': self.window,
            'uncapped_radius_mm': float(raw_radius), 'radius_mm': float(radius),
            'cap_applied': bool(radius < raw_radius), 'K_engineering_reference': k,
            'actual_degree': len(found), 'kernel': 'uniform_disk', 'selection_budget': None}
