"""Explicit outer HC research profile; central v2 is evaluated unchanged."""
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from .contracts import ContractError, canonical_hash
from .hc_mosaic import HCDensityProfile, _radial_rectangle_range
from .hc_joint_v2 import JointV2HCDensityProfile
from .atlas_hc import _configuration


class OuterHCCandidate(HCDensityProfile):
    def __init__(self, root):
        super().__init__(root, profile_path='configurations/hc_joint_synthetic_v1.json')
        self.central = JointV2HCDensityProfile(root, profile_path='configurations/hc_density_rf_joint_v2.json')
        anchors, anchor_hash = _configuration(root)
        import json
        verified = json.loads((self.root / anchors['verified_density_rows_path']).read_bytes())['rows']
        for row in self.source_rows:
            for kind in ('H1', 'H2'):
                matches = [v for v in verified if v['source_object'] == 'Table 2'
                           and [v['eccentricity_min_mm'], v['eccentricity_max_mm']] == row['eccentricity_interval_mm']
                           and v['cell_type'] == kind and v['density_cells_per_mm2'] == row[kind]]
                if len(matches) != 1:
                    raise ContractError('Outer HC density differs from verified original table')
        self.join = 1.338
        self.support_radius_mm = 6.
        nodes = np.array([np.mean(r['eccentricity_interval_mm']) for r in self.source_rows])
        self.curves = {}
        for kind, fraction in (('H1', .8), ('H2', .2)):
            values = np.log([r[kind] for r in self.source_rows])
            slopes = PchipInterpolator(nodes, values).derivative()(nodes)
            left = np.log(float(self.central.radial(self.join)) * fraction)
            derivative = float(self.central.derivative(self.join) / self.central.radial(self.join))
            self.curves[kind] = CubicHermiteSpline(np.r_[self.join, nodes], np.r_[left, values],
                                                 np.r_[derivative, slopes], extrapolate=False)
        self.contacts = {}
        for kind in ('H1', 'H2'):
            left = float(self.central.contact_radius_mm(self.join, kind))
            slope = float(self.central.contact_radius_derivative_mm(self.join, kind)) / left
            outer = np.sqrt(anchors['peripheral']['project_selected_convergence'][kind]
                            / (np.pi * anchors['peripheral']['reported_cone_density_cells_mm2']))
            self.contacts[kind] = CubicHermiteSpline([self.join, 5., 6.], np.log([left, outer, outer]),
                                                     [slope, 0., 0.], extrapolate=False)
        extrema = np.concatenate([c.derivative().roots(extrapolate=False) for c in self.curves.values()])
        self.nodes_mm = np.unique(np.r_[0., .6, self.join, nodes[nodes <= 6], 6.,
                                        extrema[(extrema >= self.join) & (extrema <= 6)]])
        self.config = dict(profile_id='HC_outer_W2000_C1_research_v1',
            central_profile_sha256=self.central.profile_hash, anchor_configuration_sha256=anchor_hash,
            density_table_sha256=self.source_asset_hash, central_unchanged_through_mm=self.join,
            density_join_mm=[self.join, float(nodes[0])], contact_join_mm=[self.join, 5.],
            contact_outer_nodes_mm=[5., 6.], original_density_rows=list(self.source_rows),
            density_rule='log subtype Hermite; central analytic slope, outer Table2 PCHIP slopes',
            contact_rule='log radius Hermite; central analytic slope to stationary source-density equal-area anchor',
            peripheral_convergence=anchors['peripheral']['project_selected_convergence'],
            peripheral_cone_density=anchors['peripheral']['reported_cone_density_cells_mm2'],
            user_choice='2026-11-02 central unchanged; literature subtype density/contact anchors; explicit smooth join',
            evidence='mixed-animal interval midpoint and axisymmetric transfers; effective filled disks, not branch anatomy',
            generator=dict(self.central.config['generator']), production_selected=False)
        self.profile_id = self.config['profile_id']
        self.profile_hash = canonical_hash(self.config)
        self.metadata = dict(self.config, profile_sha256=self.profile_hash, physiological_validation=False)

    def _types(self, r):
        r = self._radius(r)
        inner = r <= self.join
        central = self.central.radial(np.minimum(r, self.join))
        outer = np.maximum(r, self.join)
        h1 = central * .8
        return [np.where(inner, value, np.exp(self.curves[k](outer)))
                for k, value in (('H1', h1), ('H2', central-h1))]

    def radial(self, r):
        a, b = self._types(r)
        r = self._radius(r)
        return np.where(r <= self.join, self.central.radial(np.minimum(r, self.join)), a+b)

    def radial_fraction(self, r):
        a, b = self._types(r)
        return np.where(self._radius(r) <= self.join, self.central.central_fraction, a/(a+b))

    def type_density(self, xy, cell_type):
        if cell_type not in ('H1', 'H2'):
            raise ContractError('HC subtype must be H1/H2')
        return self._types(np.linalg.norm(self._xy(xy), axis=-1))[cell_type == 'H2']

    def bounds_for_rectangle(self, bounds):
        lo, hi = _radial_rectangle_range(bounds)
        self._radius([lo, hi])
        r = np.r_[lo, hi, self.nodes_mm[(self.nodes_mm >= lo) & (self.nodes_mm <= hi)]]
        a, b = self._types(r)
        return dict(H1_min=float(a.min()), H1_max=float(a.max()), H2_min=float(b.min()),
                    H2_max=float(b.max()), total_min=float(a.min()+b.min()), total_max=float(a.max()+b.max()))

    def contact_radius_mm(self, r, cell_type):
        if cell_type not in self.contacts:
            raise ContractError('HC subtype must be H1/H2')
        r = self._radius(r)
        return np.where(r <= self.join,
            self.central.contact_radius_mm(np.minimum(r, self.join), cell_type),
            np.exp(self.contacts[cell_type](np.maximum(r, self.join))))
