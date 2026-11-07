"""Expected Rod capture from shared retinal optics; no neural equations or defaults.

All spatial quadrature is at frozen individual Rod positions. Group inputs are
member means, preserving the existing representative-population approximation.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
from scipy import sparse

from .contracts import ContractError, MissingCalibrationError, Quantity, canonical_hash
from .photometry import (integrate_cone_photoisomerization_rate,
                         photon_irradiance_to_energy)


def _require(condition, message):
    if not condition:
        raise ContractError('ROD_CAPTURE: '+message)


def _contract(config, aperture, efficiency):
    c = deepcopy(config)
    _require(c.get('mode') == 'retinal_spectral', 'Explicit retinal_spectral selection required')
    _require(c.get('qualification') in ('synthetic_physical_fixture', 'conditional_reconstruction',
                                      'calibrated_absolute'), 'Explicit input qualification required')
    if aperture is None or efficiency is None:
        raise MissingCalibrationError('ROD_CAPTURE: sourced Rod aperture and absolute efficiency missing')
    for field in ('input_identity', 'radiometry_source', 'optics_source', 'coordinate_registration',
                  'exterior_background_source'):
        _require(isinstance(c.get(field), str) and bool(c[field].strip()), 'Missing '+field)
    _require(c.get('photon_noise') == 'none', 'Expected irradiance only; no capture noise sampling')
    _require(c.get('temporal_semantics') == 'window_mean_zero_order_hold', 'Window mean/ZOH required')
    _require(c.get('coordinate_frame') == 'functional_visual_field', 'Explicit global retinal chart required')
    _require(c.get('retinal_reference_id'), 'Missing retinal reference identity')
    _require(c.get('aperture_species') and c.get('efficiency_species') and c.get('spatial_kernel_source'),
             'Rod aperture, efficiency species and spatial quadrature provenance required')
    factors = c.get('factors_already_in_irradiance')
    _require(isinstance(factors, list) and len(set(factors)) == len(factors), 'Explicit optical factors required')
    _require(set(factors) == set(efficiency.already_in_irradiance), 'Efficiency/optics ownership differs')
    _require({'pupil_geometry', 'lens_transmittance'} <= set(factors), 'Input must be post-eye-optics irradiance')
    _require({'rod_absorptance', 'isomerization_yield'} <= set(efficiency.included_factors),
             'Relative sensitivity is not absolute absorption times isomerization yield')
    _require('macular_transmittance' in set(factors) | set(efficiency.included_factors),
             'Declare macular transmission (including any explicit unity assumption)')
    _require(not (set(factors) & set(efficiency.included_factors)), 'Repeated optical efficiency factor')
    if c['qualification'] == 'calibrated_absolute':
        _require(aperture.scope != 'synthetic_physical_fixture'
                 and efficiency.efficiency.evidence_tag != 'synthetic_fixture',
                 'Synthetic parameters cannot qualify real absolute capture')
        records = c.get('absolute_calibration')
        required = ('spectral_radiometry', 'rod_collection_area', 'rod_absolute_efficiency',
                    'ocular_transmission', 'spatial_registration')
        if not isinstance(records, dict) or any(not records.get(k) for k in required):
            raise MissingCalibrationError('ROD_CAPTURE: absolute calibration evidence missing: '+', '.join(required))
        if c.get('unresolved_assumptions') != []:
            raise MissingCalibrationError('ROD_CAPTURE: unresolved physical assumptions prohibit absolute qualification')
    return c


@dataclass(frozen=True)
class RodCapture:
    global_rows: np.ndarray
    cell_ids: tuple
    positions_mm: np.ndarray
    table_binding: dict
    window_edges_seconds: np.ndarray
    rod_photon_rate: Quantity
    provenance: dict
    identity: str


class RodCaptureSampler:
    """Batched bilinear sampling at caller-supplied Rod aperture quadrature nodes.

    Grid coordinates and nodes are global physical metres in the existing nominal
    RF chart; no degree conversion, centering, flip or clipping is inferred.
    Nodes/weights must describe the complete aperture, including boundary nodes.
    """
    def __init__(self, root, table_binding, global_rows, spatial_support_m, sample_offsets_m,
                 aperture, efficiency, configuration):
        from .disk_population import FixedDiskPopulation

        self.config = _contract(configuration, aperture, efficiency)
        self.aperture, self.efficiency = aperture, efficiency
        rows = np.asarray(global_rows)
        _require(rows.ndim == 1 and len(rows) > 0 and rows.dtype.kind in 'iu'
                 and len(np.unique(rows)) == len(rows), 'Unique integer global Rod rows required')
        source = FixedDiskPopulation(root, table_binding)
        _require(source.record['recipe']['population'] == 'rods', 'Source table must contain Rods')
        _require(np.all(rows >= 0) and np.all(rows < source.record['count']), 'Rod row outside frozen master')
        table = source.rows(rows)
        _require(table.reference_id == self.config['retinal_reference_id'], 'Retinal coordinate reference differs')
        self.rows = rows.astype(np.int64, copy=True)
        self.ids = table.global_cell_ids
        self.xy = table.coordinate('rf').values.copy()
        self.binding = deepcopy(table_binding)
        self.support = np.array(spatial_support_m, dtype=float, copy=True)
        s = self.support
        _require(s.ndim == 3 and s.shape[2] == 2 and min(s.shape[:2]) >= 2
                 and np.isfinite(s).all(), 'Finite row/column/xy grid in metres required')
        x, y = s[0, :, 0], s[:, 0, 1]
        _require(np.array_equal(s[:, :, 0], np.broadcast_to(x, s.shape[:2]))
                 and np.array_equal(s[:, :, 1], np.broadcast_to(y[:, None], s.shape[:2])),
                 'Rectilinear optical support required')
        _require((np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0))
                 and (np.all(np.diff(y) > 0) or np.all(np.diff(y) < 0)), 'Monotone optical axes required')
        offsets = np.asarray(sample_offsets_m, dtype=float)
        _require(offsets.ndim in (2, 3) and offsets.shape[-1] == 2 and offsets.shape[-2] > 0
                 and np.isfinite(offsets).all(), 'Explicit finite Rod aperture quadrature offsets required')
        self.offsets = np.broadcast_to(offsets, (len(rows), offsets.shape[-2], 2)).copy()
        nodes = self.xy[:, None, :]*1e-3+self.offsets
        _require(np.all(nodes[:, :, 0] >= min(x)) and np.all(nodes[:, :, 0] <= max(x))
                 and np.all(nodes[:, :, 1] >= min(y)) and np.all(nodes[:, :, 1] <= max(y)),
                 'Incomplete optical aperture support; no crop renormalization or exterior fill')
        self.nodes_per_rod = nodes.shape[1]
        flat = nodes.reshape(-1, 2)
        # Reverse descending axes in index space, preserving ISET row orientation.
        ix = np.arange(len(x)) if x[0] < x[-1] else np.arange(len(x)-1, -1, -1)
        iy = np.arange(len(y)) if y[0] < y[-1] else np.arange(len(y)-1, -1, -1)
        xs, ys = x[ix], y[iy]
        col = np.clip(np.searchsorted(xs, flat[:, 0], side='right')-1, 0, len(x)-2)
        row = np.clip(np.searchsorted(ys, flat[:, 1], side='right')-1, 0, len(y)-2)
        fx = (flat[:, 0]-xs[col])/(xs[col+1]-xs[col])
        fy = (flat[:, 1]-ys[row])/(ys[row+1]-ys[row])
        columns = np.stack([iy[row]*len(x)+ix[col], iy[row]*len(x)+ix[col+1],
                            iy[row+1]*len(x)+ix[col], iy[row+1]*len(x)+ix[col+1]], axis=1)
        weights = np.stack([(1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy], axis=1)
        self.sampling = sparse.csr_matrix((weights.ravel(),
            (np.repeat(np.arange(len(flat)), 4), columns.ravel())), shape=(len(flat), len(x)*len(y)))

    def capture(self, irradiance, wavelength_nm, quadrature_weight_nm, window_edges_seconds):
        """Input shape time/row/column/wavelength; output expected R*/rod/s."""
        edges = np.array(window_edges_seconds, dtype=float, copy=True)
        _require(edges.ndim == 1 and len(edges) >= 2 and np.isfinite(edges).all()
                 and np.allclose(np.diff(edges), 1/150, rtol=0, atol=1e-12), '150 Hz integration windows required')
        wave = wavelength_nm.require('nm', 'wavelength')
        _require(irradiance.values.shape == (len(edges)-1, *self.support.shape[:2], len(wave)),
                 'Optical time/space/wavelength shape differs')
        tag = irradiance.evidence_tag
        _require(tag == wavelength_nm.evidence_tag == quadrature_weight_nm.evidence_tag,
                 'Spectral evidence tags differ')
        rates = np.empty((len(edges)-1, len(self.rows)), dtype=np.float64)
        for i, frame in enumerate(irradiance.values):
            q = Quantity(frame, irradiance.unit, irradiance.meaning, tag)
            if q.unit == 'photons/(s m^2 nm)':
                q = photon_irradiance_to_energy(q, wavelength_nm)
            energy = q.require('W/(m^2 nm)', 'retinal_spectral_irradiance')
            _require(np.all(energy >= 0), 'Negative retinal irradiance')
            sampled = (self.sampling @ energy.reshape(-1, len(wave))).reshape(
                len(self.rows), self.nodes_per_rod, len(wave))
            rates[i] = integrate_cone_photoisomerization_rate(
                Quantity(sampled, q.unit, q.meaning, tag), wavelength_nm, quadrature_weight_nm,
                self.aperture, self.efficiency).values
        rate = Quantity(rates, 'R*/s', 'photoisomerization_rate', tag)
        provenance = dict(configuration=self.config, aperture=vars(self.aperture),
            efficiency=vars(self.efficiency), sample_offsets_m=self.offsets, spatial_support_m=self.support,
            wavelength_nm=wave, quadrature_weight_nm=quadrature_weight_nm.values,
            irradiance_sha256=canonical_hash(irradiance),
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            photometry_sha256=hashlib.sha256(Path(__file__).with_name('photometry.py').read_bytes()).hexdigest(),
            absolute_capture_qualified=self.config['qualification'] == 'calibrated_absolute',
            physiological_validation=False, photon_noise='none')
        payload = dict(global_rows=self.rows.copy(), cell_ids=self.ids, positions_mm=self.xy.copy(),
            table_binding=self.binding, window_edges_seconds=edges, rod_photon_rate=rate,
            provenance=deepcopy(provenance))
        return RodCapture(**payload, identity=canonical_hash(payload))


def isetbio_retinal_support(capture, optical_center_m):
    """Pinned cMosaic.compute registration: centre OI axes then add global centre."""
    center = np.asarray(optical_center_m, dtype=float)
    _require(center.shape == (2,) and np.isfinite(center).all(), 'Explicit optical centre required')
    support = np.array(capture['spatial_support_m'], dtype=float, copy=True)
    _require(support.ndim == 3 and support.shape[2] == 2, 'OI row/column/xy support required')
    support[:, :, 0] += center[0]-np.mean(support[0, :, 0])
    support[:, :, 1] += center[1]-np.mean(support[:, 0, 1])
    return support


def capture_isetbio_retinal_photons(sampler, capture, quadrature_weight_nm, *,
                                   optical_center_m, source_origin_seconds):
    """Use read_isetbio_capture's raw OI, applying cMosaic's row flip exactly once.

    OI image rows point downward; the project's retinal y axis points upward.
    No Cone pigment, aperture, macular filter, or output count is reused here.
    """
    _require(capture.get('manifest', {}).get('schema') == 'isetbio_cone_capture_v1',
             'Raw pinned ISET capture required, not an already flipped or Cone-only array')
    _require(np.isfinite(source_origin_seconds), 'Explicit source epoch required')
    support = isetbio_retinal_support(capture, optical_center_m)
    _require(support.shape == sampler.support.shape and np.allclose(support, sampler.support, rtol=0, atol=1e-15),
             'Registered ISET support differs from sampler')
    _require(sampler.config['input_identity'] == capture['sha256'], 'Capture source identity differs')
    tag = sampler.efficiency.efficiency.evidence_tag
    flux = Quantity(np.moveaxis(capture['retinal_photons'][::-1, :, :, :], -1, 0),
        'photons/(s m^2 nm)', 'retinal_spectral_photon_irradiance', tag)
    return sampler.capture(flux, Quantity(capture['wave_nm'], 'nm', 'wavelength', tag),
        quadrature_weight_nm, capture['edges']+source_origin_seconds)


def bind_rod_capture(seed, assembly, protocol, capture, *, initial_state_policy):
    """Return an independent diagnostic seed; never migrate a background/checkpoint.

    A step from the old initial condition is useful for interface verification,
    but it is explicitly not a newly equilibrated physical experiment.
    """
    _require(initial_state_policy == 'diagnostic_step_from_frozen_state',
             'Physical experiment needs independently reconstructed affected background states')
    payload = {k: v for k, v in vars(capture).items() if k != 'identity'}
    _require(canonical_hash(payload) == capture.identity, 'Capture content changed after integration')
    p = assembly.populations['rods']
    _require(capture.table_binding == p.source_table, 'Rod master binding differs')
    members = np.asarray(p.member_global_rows)
    _require(set(capture.global_rows.tolist()) == set(members.tolist()) and len(members) == len(capture.cell_ids),
             'Every local Rod member exactly once; no missing-member background substitution')
    _require(protocol['input_fps'] == 150. and protocol['rod_step_seconds'] == .0001,
             'Original frame/Chen clocks required')
    frames = sorted(protocol['input_frame_indices'])
    _require(frames and frames == list(range(frames[0], frames[-1]+1)), 'Contiguous declared frame indices required')
    expected_edges = protocol['input_origin_seconds']+np.arange(frames[0], frames[-1]+2)/150
    _require(capture.window_edges_seconds.shape == expected_edges.shape and
             np.allclose(capture.window_edges_seconds, expected_edges, rtol=0, atol=1e-12),
             'Capture windows differ from source frame epoch')
    _require(set(seed['rod_inputs']) == set(seed['native_inputs']) == set(frames), 'Seed frame scope differs')
    lookup = {int(row): i for i, row in enumerate(capture.global_rows)}
    order = np.array([lookup[int(row)] for row in members])
    offsets = np.asarray(p.member_offsets)
    counts = np.asarray(p.counts)
    _require(offsets.shape == (len(counts)+1,) and offsets[0] == 0 and offsets[-1] == len(members)
             and np.array_equal(np.diff(offsets), counts) and np.all(counts > 0), 'Invalid frozen membership')
    values = capture.rod_photon_rate.require('R*/s', 'photoisomerization_rate', (len(frames), len(members)))
    grouped = np.add.reduceat(values[:, order], offsets[:-1], axis=1)/counts[None, :]
    result = deepcopy(seed)
    result['rod_inputs'] = {frame: grouped[i].copy() for i, frame in enumerate(frames)}
    result['rod_capture'] = dict(mode='retinal_spectral', identity=capture.identity,
        qualification=capture.provenance['configuration']['qualification'],
        initial_state_policy=initial_state_policy, parent_seed_sha256=canonical_hash(seed),
        group_semantics='mean expected R*/rod/s over all frozen members; no contact renormalization',
        exterior_background_source=capture.provenance['configuration']['exterior_background_source'],
        background_qualified=False, FIT_READY=False, production_adopted=False)
    return result
