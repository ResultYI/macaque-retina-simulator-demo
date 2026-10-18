"""Frozen real-source loading and bounded four-meridian density queries.

Source observations remain separate from the interpolation used by an atlas.
No S curve, central rod curve, or HC arbor is fabricated from scalar abstracts.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from types import MappingProxyType
import numpy as np

from .contracts import ContractError, Quantity
from .geometry import PackerDensityField
from .runtime_profiles import frozen_packer_table

PACKAGE_DIRECTORY = Path('data/source_packages/20261012')
PACKAGES = {
    'S_followup': ('macaque_s_cone_public_followup_v1.zip',
                   '2346039c095ab7391ab441112fade80252a1edceb2fd9e35371067b225463e7b',
                   'libfile_f6f6d7425b1c81919703d6f33cf690d1'),
    'hc': ('macaque_hc_sources_20261012.zip',
           '6e6ac21073ecc189fed5e1d1736f22321e53af397e2930b24faf60cd8ceab58d',
           'libfile_5e985d65b9608191b2ca1ad2f491991e'),
    'henle': ('Schein1988_Henle_source_verified.zip',
              '3bcbf83d595a2dd6cc08fe5e3832c6351d93f881e0e8ef6fdfaef4a1c1ce55f6',
              'libfile_be5f4f7217d881918c0df9dc3468ba79'),
    'photoreceptors': ('macaque_photoreceptor_source_data_v1.zip',
                      'bf7fd26555ef324240c12d8b81a63189a40d86a109b2384e23854b19d0e9ee84',
                      'libfile_95e2cea7310481918f7464f44545d75c')}
FROZEN_JSON = {
    'S_single_field': ('macaque_s_cone_public_followup_v1/martin/martin2000_mnemestrina_single_field.json',
                       '1399e1264b4ca29dbefbbc306e51706efcea737c590aef693381d7fa1eccc1c5'),
    'S_followup_provenance': ('macaque_s_cone_public_followup_v1/provenance.json',
                             '70103037ea4d3e44547df0d41e5f960cd67a68a8d6b13e85ec6bccefbccc4989'),
    'hc_rows': ('hc/hc_density_wassle2000.json', '850f0abc91f1bc3b38b9ad966d3c2d2e6a5cac29547fe0a59f95588c0832e832'),
    'hc_anchors': ('hc/hc_density_rohrenbeck1989_abstract_anchors.json', '8b5e5caa67830922150634b70f090c437f1d70fa2ad55ac1817fafd6e96b2fde'),
    'hc_gaps': ('hc/coverage_status.json', '270698f265640be8f03590169c833a5cce8cc4cca6429c60d4345cb9f6b7c7b5'),
    'henle_curves': ('Schein1988_Henle_source_verified/schein1988_fig15_digitized_curves.json', 'b1444893bcb7a4f811718180609f53acce16e1192030ef08e6d756faf6e27323'),
    'henle_provenance': ('Schein1988_Henle_source_verified/provenance.json', '25623970569d3f978b80102585521533a93ad9ca25f5204f51510819cd2829bd'),
    'henle_thresholds': ('Schein1988_Henle_source_verified/textual_anchors.json', '5288d094cbe232bc11c16a2082309aa6b63b6200f23704b4a5ed735d9b208d76'),
    'photoreceptor_points': ('photoreceptors/packer1989_digitized_points_raw.json', 'd33be2bed39bdf12610edf4e3549438985b59db4fc756ccccb130231abf824f6'),
    'photoreceptor_provenance': ('photoreceptors/packer1989_provenance.json', 'f4f9eb0e5faa73b3628f18e7d6167603ffc96767af31ceed2cfebf056dc21faa'),
    'S_constraints': ('photoreceptors/s_cones/s_cone_extraction.json', 'c4f73735c73a4aae2f803d91a00929a30298c12bb839a62fed95ae41fd33bca2')}


def verified_bytes(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ContractError('Frozen atlas source identity mismatch: ' + str(path))
    return raw


class AtlasSourceBundle:
    """Retain original rows, species, intervals, uncertainty and source gaps."""
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root / PACKAGE_DIRECTORY
        self.package_receipts = {}
        for key, (name, sha, library_id) in PACKAGES.items():
            verified_bytes(self.directory/name, sha)
            self.package_receipts[key] = {'path': str(PACKAGE_DIRECTORY/name),
                'sha256': sha, 'library_file_id': library_id,
                'local_bytes_verified': True, 'verification': 'byte identity, not physiological validation'}
        self.data = {key: json.loads(verified_bytes(self.directory/name, sha))
                     for key, (name, sha) in FROZEN_JSON.items()}
        if (len(self.data['hc_rows']['rows']) != 23 or
                len(self.data['henle_curves']['data']) != 343 or
                len(self.data['photoreceptor_points']) != 116):
            raise ContractError('Atlas source extraction row counts changed')
        if self.data['hc_gaps']['direct_anatomical_arbor_measurements']:
            raise ContractError('Unexpected arbor data in density-only source package')

    def manifest(self):
        return {'packages': self.package_receipts, 'source_assets': {
            key: {'path': str(PACKAGE_DIRECTORY/name), 'sha256': sha}
            for key, (name, sha) in FROZEN_JSON.items()},
            'counts': {'HC_density_rows': 23, 'Rohrenbeck_abstract_anchors': 2,
                       'Henle_derived_digitized_curve_points': 343,
                       'Packer_derived_digitized_map_points': 116},
            'S_single_field_reference_only': self.data['S_single_field'],
            'missing': {'HC_arbor': self.data['hc_gaps']['unsupported'],
                        'S_density_curves': self.data['S_constraints']['missing'],
                        'central_rod_density_curve_below_0p4mm': True,
                        'Henle_below_0p025_and_3_to_6mm': True}}


MERIDIANS = ('nasal', 'superior', 'temporal', 'inferior')


@dataclass(frozen=True)
class DensitySamples:
    density: Quantity | None
    supported_values: tuple
    status: tuple
    meridian_weights: tuple
    source_rows: tuple
    provenance: dict


class PackerFourMeridianDensity:
    """Cardinal log-linear interpolation is a project field, never measured 2D.

The optional central cone bridge preserves the older code-based cap, with its
own evidence. Rods have no bridge and no automatic hard absence mask.
"""
    def __init__(self, root, cell_class, *, central_policy='missing', rf_transfer=None):
        if cell_class not in ('all_cones', 'rod'):
            raise ContractError('Packer identifies all cones or rods, not L/M/S')
        if central_policy not in ('missing', 'legacy_Packer_published_code_central_bridge_v1'):
            raise ContractError('Unknown central density policy')
        if cell_class == 'rod' and central_policy != 'missing':
            raise ContractError('No source-resolved central rod bridge is available')
        if rf_transfer not in (None, 'physical_retinal_mm_equals_planar_RF_mm_project_transfer_v1'):
            raise ContractError('Unknown RF/physical-retina registration')
        self.bundle = AtlasSourceBundle(root)
        self.cell_class, self.central_policy, self.rf_transfer = cell_class, central_policy, rf_transfer
        self.rows = MappingProxyType({m: tuple(MappingProxyType(dict(r)) for r in sorted((r for r in self.bundle.data['photoreceptor_points']
            if r['meridian'] == m and r['cell_class'] == cell_class),
            key=lambda r: r['eccentricity_mm'])) for m in MERIDIANS})
        for rows in self.rows.values():
            x = np.array([r['eccentricity_mm'] for r in rows])
            if len(x) < 2 or np.any(np.diff(x) <= 0) or any(r['density_per_mm2'] <= 0 for r in rows):
                raise ContractError('Invalid cardinal density nodes')
        self.central = (PackerDensityField(frozen_packer_table(root), central_policy='central_constant_cap',
                        angular_policy='log_NT_cosine') if central_policy != 'missing' else None)

    def array_values(self, points):
        """Batch the same radial/log and cardinal/linear rules, without row provenance.

        This is for large fixed-asset construction. No new density, extrapolation
        or angular symmetry is introduced; detailed queries still use sample().
        """
        xy=np.asarray(points,dtype=float)
        if xy.ndim!=2 or xy.shape[1]!=2 or not np.all(np.isfinite(xy)):
            raise ContractError('Density array requires finite Nx2 mm coordinates')
        radius=np.linalg.norm(xy,axis=1)
        angle=np.mod(np.arctan2(xy[:,1],xy[:,0]),2*np.pi)/(np.pi/2)
        sector=np.floor(angle).astype(int)%4;fraction=angle-np.floor(angle)
        result=np.zeros(len(xy));missing=np.zeros(len(xy),bool);central_limit=np.zeros(len(xy))
        for side in (0,1):
            weight=1-fraction if side==0 else fraction
            for k,name in enumerate(MERIDIANS):
                mask=((sector+side)%4==k)&((fraction>1e-12) if side else True)
                if not np.any(mask):continue
                rows=self.rows[name];x=np.array([row['eccentricity_mm'] for row in rows])
                y=np.log([row['density_per_mm2'] for row in rows])
                rr=radius[mask];missing[mask]|=(rr<x[0])|(rr>x[-1])
                central_limit[mask]=np.maximum(central_limit[mask],x[0])
                result[mask]+=weight[mask]*np.interp(rr,x,y)
        central=missing&(radius<central_limit)
        if np.any(missing&~central) or (np.any(central) and self.central is None):
            raise ContractError('Packer density array outside original radial support; no extrapolation')
        result=np.exp(result)
        if np.any(central):result[central]=self.central(xy[central])
        return result

    def sample(self, positions, *, allow_missing=False):
        if not isinstance(positions, Quantity):
            raise ContractError('Density query requires typed coordinates')
        is_rf = positions.coordinate_frame == 'functional_visual_field'
        if is_rf:
            positions.require('mm', 'functional_position', frame='functional_visual_field')
            if self.rf_transfer is None:
                raise ContractError('RF query requires explicit physical-retina registration')
        else:
            positions.require('mm', 'anatomical_position', frame='anatomical_retina')
        xy = positions.values
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ContractError('Density coordinates require shape(n,2)')
        radius = np.linalg.norm(xy, axis=1)
        sector = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2*np.pi)/(np.pi/2)
        values, statuses, weights, source_rows = [], [], [], []
        for point, r, angle in zip(xy, radius, sector):
            k = int(np.floor(angle)) % 4
            f = float(angle - np.floor(angle))
            pairs = [(MERIDIANS[k], 1-f)]
            if f > 1e-12:
                pairs.append((MERIDIANS[(k+1) % 4], f))
            samples, brackets, exact = [], [], not is_rf and len(pairs) == 1
            missing = False
            for m, w in pairs:
                rows = self.rows[m]
                x = np.array([row['eccentricity_mm'] for row in rows])
                if r < x[0] or r > x[-1]:
                    missing = True
                    break
                j = int(np.searchsorted(x, r))
                i = j if j < len(x) and x[j] == r else j-1
                j = min(j, len(x)-1)
                y = np.array([row['density_per_mm2'] for row in rows])
                samples.append(w*np.interp(r, x, np.log(y)))
                brackets.append((m, rows[i]['point_id'], rows[j]['point_id']))
                exact &= i == j
            if missing and self.central is not None and r < max(self.rows[m][0]['eccentricity_mm'] for m, _ in pairs):
                value = float(self.central(np.asarray([point]))[0])
                status, brackets = 'project_central_bridge', [('Packer_published_code_NT',)]
            elif missing:
                if not allow_missing:
                    raise ContractError(f'Packer {self.cell_class} curve unsupported at r={r:.9g}mm; no extrapolation')
                value, status, brackets = None, 'unsupported', []
            else:
                value = float(np.exp(sum(samples)))
                status = 'source_supported_digitized_node' if exact else 'project_interpolation_or_registration'
            values.append(value); statuses.append(status); weights.append(tuple(pairs)); source_rows.append(tuple(brackets))
        # A query with any missing point has no runnable Quantity: missing never becomes zero.
        density = None if any(v is None for v in values) else Quantity(values,
            'cells/mm^2', 'population_density', 'source_informed_project_field')
        return DensitySamples(density, tuple(values), tuple(statuses), tuple(weights), tuple(source_rows),
            {'source': 'Packer1989', 'cell_class': self.cell_class, 'species': 'Macaca nemestrina',
             'source_frame': 'physical retinal spherical reconstruction mm',
             'application_frame': positions.coordinate_frame, 'rf_transfer': self.rf_transfer,
             'axes': '+x nasal, +y superior; source figure nasal-left reflection registered explicitly',
             'interpolation': 'log-linear radius and cardinal angular interpolation, project choice',
             'central_policy': self.central_policy, 'raw_evidence': 'digitized_author_map_resampling_not_raw_counts',
             'uncertainty': 'Per-row readout errors retained in source asset; no biological CI supplied'})

    def __call__(self, positions):
        return self.sample(positions).density.values
