"""Source-constrained synthetic HC density and joint marked soft-core geometry.

This module implements the named project construction in
``docs/hc_distribution_proposal.md``.  It is not an implementation of the
Eglen/Wong local dmin model, nor a measured whole-retina HC map.  Generation
does not certify MCMC mixing or biological spatial statistics.  It never
rescales density to satisfy a cell/resource budget.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline

from .contracts import ContractError, Quantity
from .geometry import RetinalDomain, keyed_rng


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ContractError(f"{name} must be finite positive")
    return value


def _integer(value, name, minimum=1):
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ContractError(f"{name} must be an integer >= {minimum}") from error
    if isinstance(value, bool) or number != value or number < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return number


def _radial_rectangle_range(bounds):
    x0, x1, y0, y1 = map(float, bounds)
    nearest_x = max(x0, min(0.0, x1))
    nearest_y = max(y0, min(0.0, y1))
    lower = math.hypot(nearest_x, nearest_y)
    upper = max(math.hypot(x, y) for x in (x0, x1) for y in (y0, y1))
    return lower, upper


class HCDensityProfile:
    """Continuous total density and H1 fraction in the physical soma plane.

    The original Table 2 rows remain separate from their representative
    interpolation nodes.  The central smoothstep, axial symmetry, interval
    midpoints and cross-animal interpolation are project approximations.
    ``radial`` takes mm; array coordinate methods take physical mm, or an
    explicitly qualified ``Quantity``.  Out-of-support queries raise instead
    of clamping/extrapolating.
    """

    def __init__(self, root, profile_path=None, profile_data=None):
        self.root = Path(root)
        self.profile_path = Path(profile_path) if profile_path is not None else (
            self.root / "configurations" / "hc_joint_synthetic_v1.json")
        if not self.profile_path.is_absolute():
            self.profile_path = self.root / self.profile_path
        if profile_data is None and not self.profile_path.is_file():
            raise ContractError(f"Named HC synthetic profile is missing: {self.profile_path}")
        raw = self.profile_path.read_bytes() if self.profile_path.is_file() else None
        # A caller may resolve GUI/user geometry into the named file's content.
        # Canonicalizing through JSON makes an isolated, JSON-only copy and
        # prevents later caller mutations from changing this profile silently.
        loaded = json.loads(raw.decode("utf-8-sig")) if profile_data is None else profile_data
        self.config = json.loads(json.dumps(loaded, allow_nan=False))
        self.raw_profile_asset_hash = hashlib.sha256(raw).hexdigest() if raw is not None else None
        self.profile_hash = _json_hash(self.config)
        self.profile_id = self.config.get("profile_id", self.config.get("name"))
        if not self.profile_id:
            raise ContractError("HC synthetic profile requires profile_id")
        source_path = Path(self.config.get(
            "density_table_path", "sources/spatial/wassle2000_HC_density_tables.json"))
        self.source_path = source_path if source_path.is_absolute() else self.root / source_path
        source_raw = self.source_path.read_bytes()
        expected_hash = self.config.get('expected_density_table_sha256')
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or hashlib.sha256(source_raw).hexdigest() != expected_hash:
            raise ContractError('HC density transcription identity differs from the frozen named profile; this check is not primary verification')
        self.source_asset_hash = hashlib.sha256(source_raw).hexdigest()
        self.source = json.loads(source_raw.decode("utf-8-sig"))
        if self.source.get("unit") != "cells/mm^2":
            raise ContractError("HC Table 2 requires cells/mm^2")
        rows = self.source.get("table2", [])
        if len(rows) != 6:
            raise ContractError("Synthetic v1 HC profile requires the six original Table 2 rows")
        self.source_rows = tuple(dict(row) for row in rows)
        centers, h1, h2 = [], [], []
        for row in rows:
            interval = row.get("eccentricity_interval_mm", ())
            if len(interval) != 2 or not row.get("animal"):
                raise ContractError("Each HC source row needs animal and original eccentricity interval")
            lower, upper = map(float, interval)
            if not np.isfinite(lower + upper) or not 0 <= lower <= upper:
                raise ContractError("Invalid HC source eccentricity interval")
            centers.append((lower + upper) / 2)
            h1.append(_positive(row["H1"], "source H1 density"))
            h2.append(_positive(row["H2"], "source H2 density"))
        central = self.config.get("central", {})
        self.peak_radius_mm = _positive(central.get("peak_radius_mm", .6), "peak radius")
        self.center_total = _positive(central.get("center_total_density_cells_mm2", 250), "central density")
        self.peak_total = _positive(central.get("peak_total_density_cells_mm2", 23000), "peak density")
        self.central_fraction = float(central.get("H1_fraction", .8))
        self.support_radius_mm = _positive(self.config.get("support_radius_mm", 7.1), "source support")
        if not 0 < self.central_fraction < 1 or self.center_total > self.peak_total:
            raise ContractError("HC synthetic profile requires positive subtypes and an ascending central field")
        if not self.peak_radius_mm < centers[0] or np.any(np.diff(centers) <= 0):
            raise ContractError("HC synthetic interpolation nodes must increase after the central peak")
        if self.support_radius_mm > centers[-1]:
            raise ContractError("HC synthetic support cannot extend beyond the last source row")
        self.nodes_mm = np.array([self.peak_radius_mm, *centers], dtype=float)
        self.total_nodes = np.array([self.peak_total, *np.add(h1, h2)], dtype=float)
        self.fraction_nodes = np.array([self.central_fraction,
                                       *np.divide(h1, np.add(h1, h2))], dtype=float)
        log_total = np.log(self.total_nodes)
        logit_fraction = np.log(self.fraction_nodes / (1 - self.fraction_nodes))
        # PCHIP gives the interior/end derivative policy.  Explicitly replacing
        # the first derivative with zero joins the flat central peak endpoint.
        total_slopes = PchipInterpolator(self.nodes_mm, log_total).derivative()(self.nodes_mm)
        fraction_slopes = PchipInterpolator(self.nodes_mm, logit_fraction).derivative()(self.nodes_mm)
        total_slopes[0] = 0.0
        fraction_slopes[0] = 0.0
        self._log_total = CubicHermiteSpline(self.nodes_mm, log_total, total_slopes, extrapolate=False)
        self._logit_fraction = CubicHermiteSpline(self.nodes_mm, logit_fraction,
                                                 fraction_slopes, extrapolate=False)
        reference = self.config.get("source_records", {}).get("EglenWong2008", {})
        self.reference_counts = np.array([reference.get("H1_count", 206),
                                          reference.get("H2_count", 86)], dtype=float)
        self.reference_window_mm = np.array(reference.get("area_mm", [.298, .300]), dtype=float)
        self.reference_dmin_mean_um = np.array(reference.get("dmin_mean_um", [21, 32, 12]), dtype=float)
        if (self.reference_counts.shape != (2,) or self.reference_window_mm.shape != (2,) or
                self.reference_dmin_mean_um.shape != (3,) or
                not all(np.all(np.isfinite(a)) and np.all(a > 0) for a in (
                    self.reference_counts, self.reference_window_mm, self.reference_dmin_mean_um))):
            raise ContractError("HC soft-core reference requires explicit positive field-B counts/window/dmin means")
        self.reference_density = self.reference_counts / np.prod(self.reference_window_mm)
        # The conversion is explicit and stored once; original means remain in um.
        self.reference_lengths_mm = self.reference_dmin_mean_um * .001
        self.reference_dmin_sd_um = list(reference.get("dmin_SD_um", [4, 8, 2.5]))
        self.reference_lower_truncation_um = reference.get("normal_lower_truncation_um", 5)
        self.metadata = {
            "profile_id": self.profile_id, "profile_sha256": self.profile_hash,
            "resolved_profile_sha256": self.profile_hash,
            "raw_profile_asset_sha256": self.raw_profile_asset_hash,
            "source_asset_path": str(self.source_path), "source_asset_sha256": self.source_asset_hash,
            "unit": "cells/mm^2", "coordinate_frame": "anatomical_retina",
            "plane": "HC soma physical projection", "radial_support_mm": [0.0, self.support_radius_mm],
            "evidence_class": "project_approximation",
            "original_source_records": [dict(row) for row in self.source_rows],
            "original_source_evidence_class": self.source.get("evidence_class"),
            "profile_source_quotes": self.config.get("source_records", {}),
            "central_author_estimate_reference_only": self.source.get("central_author_estimate"),
            "central_author_estimate_evidence_class": self.source.get("central_evidence_class"),
            "representative_node_mm": centers,
            "representative_node_policy": "interval midpoint; point records unchanged; project_approximation",
            "central_policy": "log-smoothstep total; constant H1 fraction; project_approximation",
            "outer_policy": "log-total/logit-fraction shape-preserving cubic; zero left derivative; project_approximation",
            "central_endpoint_constraints": {
                "center_total_cells_mm2": self.center_total,
                "peak_total_cells_mm2": self.peak_total, "peak_radius_mm": self.peak_radius_mm,
                "H1_fraction": self.central_fraction,
                "source_note": "Rohrenbeck1989 abstract-level total anchors; fixed peak node and central type fraction are project choices",
                "primary_verification": "not certified by automatic loading"},
            "cross_animal_transfer": True, "cross_species_transfer": self.config.get("assumptions", {}).get(
                "cross_species_transfer", self.config.get(
                    "cross_species_transfer", "target-species-dependent; Table2 fascicularis source")),
            "meridian_transfer": "axisymmetry is project_approximation",
            "derived_center_H1_H2_not_measurements": [self.center_total * self.central_fraction,
                                                      self.center_total * (1 - self.central_fraction)],
            "implementation_status": "IMPLEMENTED_NOT_RUN",
            "calibration_status": "synthetic source-constrained field; not measured whole-retina map",
        }

    def _radius(self, radius_mm):
        radius = np.asarray(radius_mm, dtype=float)
        if not np.all(np.isfinite(radius)) or np.any(radius < 0) or np.any(radius > self.support_radius_mm):
            raise ContractError(f"HC_DENSITY_DOMAIN_UNSUPPORTED: physical radius must be 0..{self.support_radius_mm} mm")
        return radius

    def _xy(self, xy):
        if isinstance(xy, Quantity):
            xy = xy.require("mm", "anatomical_position", frame="anatomical_retina")
        points = np.asarray(xy, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 2 or not np.all(np.isfinite(points)):
            raise ContractError("HC density requires finite physical (..., 2) mm positions")
        return points

    def radial(self, radius_mm):
        radius = self._radius(radius_mm)
        inner = radius <= self.peak_radius_mm
        u = np.minimum(radius / self.peak_radius_mm, 1.0)
        smooth = 3 * u ** 2 - 2 * u ** 3
        central = np.exp(np.log(self.center_total) + (np.log(self.peak_total) - np.log(self.center_total)) * smooth)
        # The central branch supplies values below the outer interpolation domain.
        outer_radius = np.maximum(radius, self.peak_radius_mm)
        return np.where(inner, central, np.exp(self._log_total(outer_radius)))

    def radial_fraction(self, radius_mm):
        radius = self._radius(radius_mm)
        logit = self._logit_fraction(np.maximum(radius, self.peak_radius_mm))
        fraction = 1.0 / (1.0 + np.exp(-logit))
        return np.where(radius <= self.peak_radius_mm, self.central_fraction, fraction)

    def __call__(self, xy):
        return self.total(xy)

    def total(self, xy):
        return self.radial(np.linalg.norm(self._xy(xy), axis=-1))

    def fraction(self, xy):
        return self.radial_fraction(np.linalg.norm(self._xy(xy), axis=-1))

    def type_density(self, xy, cell_type):
        if cell_type not in ("H1", "H2"):
            raise ContractError("HC density subtype must be H1 or H2")
        points = self._xy(xy)
        total = self.total(points)
        fraction = self.fraction(points)
        return total * (fraction if cell_type == "H1" else 1 - fraction)

    def bounds_for_rectangle(self, bounds):
        """Conservative positive density bounds, used for exact rejection draws."""
        lower, upper = _radial_rectangle_range(bounds)
        self._radius(np.array([lower, upper]))
        radii = np.unique(np.r_[lower, upper, self.nodes_mm[(self.nodes_mm >= lower) & (self.nodes_mm <= upper)]])
        if lower <= self.peak_radius_mm <= upper:
            radii = np.unique(np.r_[radii, self.peak_radius_mm])
        total = self.radial(radii)
        fraction = self.radial_fraction(radii)
        # Monotonic Hermite segments cannot exceed their node extrema; taking
        # independent total/fraction bounds is conservative for their product.
        return {"total_min": float(total.min()), "total_max": float(total.max()),
                "H1_min": float(total.min() * fraction.min()),
                "H1_max": float(total.max() * fraction.max()),
                "H2_min": float(total.min() * (1 - fraction.max())),
                "H2_max": float(total.max() * (1 - fraction.min()))}

    def max_interaction_support_mm(self):
        """Conservative finite pair range over the supported radial field.

        A canonical buffer must cover both homotypic and heterotypic proposals;
        this uses source-field minima, with no numerical density floor.
        """
        radii = np.unique(np.r_[0., self.support_radius_mm,
            self.nodes_mm[self.nodes_mm <= self.support_radius_mm]])
        total = self.radial(radii)
        fraction = self.radial_fraction(radii)
        minima = np.array([total.min() * fraction.min(),
            total.min() * (1. - fraction.max()), total.min()])
        reference = np.r_[self.reference_density, self.reference_density.sum()]
        return float(np.max(self.reference_lengths_mm * np.sqrt(reference / minima)))


@dataclass(frozen=True)
class JointHCMosaic:
    positions: Quantity
    types: np.ndarray
    cell_ids: tuple
    owner: np.ndarray
    diagnostics: dict

    def __post_init__(self):
        n = len(self.cell_ids)
        self.positions.require("mm", "anatomical_position", (n, 2), "anatomical_retina")
        labels = np.array(self.types, dtype="<U2", copy=True)
        owner = np.array(self.owner, dtype=bool, copy=True)
        if labels.shape != (n,) or owner.shape != (n,) or np.any(~np.isin(labels, ["H1", "H2"])):
            raise ContractError("Joint HC mosaic requires aligned H1/H2 labels and ownership")
        if len(set(self.cell_ids)) != n:
            raise ContractError("Joint HC mosaic cell IDs must be unique")
        labels.setflags(write=False)
        owner.setflags(write=False)
        object.__setattr__(self, "types", labels)
        object.__setattr__(self, "owner", owner)

    def crop(self, domain):
        """Crop this canonical instance, without regenerating points or IDs."""
        canonical = self.diagnostics.get("view_bounds_mm", self.diagnostics["canonical_bounds_mm"])
        if any(not np.isfinite(v) for v in domain.bounds_mm):
            raise ContractError("HC crop bounds must be finite")
        x0, x1, y0, y1 = canonical
        a, b, c, d = domain.bounds_mm
        if not x0 <= a < b <= x1 or not y0 <= c < d <= y1:
            raise ContractError("HC crop must lie inside the available view of its canonical instance")
        if domain.mask_source_id != self.diagnostics["mask_source_id"]:
            raise ContractError("HC crop must retain the same declared tissue mask identity")
        selected = domain.contains(self.positions.values)
        points = self.positions.values[selected]
        ids = tuple(identifier for identifier, keep in zip(self.cell_ids, selected) if keep)
        return JointHCMosaic(
            Quantity(points, "mm", "anatomical_position", "project_approximation", "anatomical_retina"),
            self.types[selected], ids, domain.contains(points, owner=True),
            dict(self.diagnostics, view_bounds_mm=list(domain.bounds_mm),
                 owner_bounds_mm=list(domain.owner_bounds_mm), crop_label=domain.label,
                 view_count=len(ids), owner_count=int(domain.contains(points, owner=True).sum()),
                 view_policy="crop of one canonical instance; coordinates/types/global IDs unchanged"))

    def select_type(self, cell_type):
        if cell_type not in ("H1", "H2"):
            raise ContractError("Joint HC mosaic subtype must be H1 or H2")
        selected = self.types == cell_type
        return JointHCMosaic(
            Quantity(self.positions.values[selected], "mm", "anatomical_position",
                     "project_approximation", "anatomical_retina"),
            self.types[selected], tuple(i for i, keep in zip(self.cell_ids, selected) if keep),
            self.owner[selected], dict(self.diagnostics, selected_type=cell_type,
                                       view_count=int(selected.sum()), owner_count=int(self.owner[selected].sum())))


@dataclass(frozen=True)
class _Partition:
    key: str
    bounds: tuple
    mass: tuple
    error: tuple
    depth: int


class _Quadrature:
    """Adaptive tensor Gauss quadrature with an explicit unresolved-error stop."""

    def __init__(self, profile, domain, order, relative_tolerance, absolute_tolerance, max_depth, max_points):
        self.profile, self.domain = profile, domain
        self.order, self.rtol, self.atol, self.max_depth = order, relative_tolerance, absolute_tolerance, max_depth
        self.max_points = max_points
        self.rules = {n: np.polynomial.legendre.leggauss(n) for n in (order, 2 * order)}
        self.evaluated_points = 0

    def rule(self, bounds, order):
        x0, x1, y0, y1 = bounds
        nodes, weights = self.rules[order]
        x = (x0 + x1) / 2 + nodes * (x1 - x0) / 2
        y = (y0 + y1) / 2 + nodes * (y1 - y0) / 2
        points = np.array(np.meshgrid(x, y, indexing="ij")).reshape(2, -1).T
        if self.evaluated_points + len(points) > self.max_points:
            raise ContractError("HC_QUADRATURE_BUDGET_EXCEEDED: declare a smaller canonical domain or larger integration budget")
        weights2d = np.multiply.outer(weights, weights).ravel() * (x1 - x0) * (y1 - y0) / 4
        active = np.asarray(self.domain.mask(points), dtype=bool)
        if active.shape != (len(points),):
            raise ContractError("Physical tissue mask must return one boolean per HC query point")
        self.evaluated_points += len(points)
        if not np.any(active):
            return np.zeros(2)
        return np.array([np.dot(weights2d[active], self.profile.type_density(points[active], kind))
                         for kind in ("H1", "H2")])

    def integrate(self, bounds, depth=0, absolute_tolerance=None):
        allowance = self.atol if absolute_tolerance is None else absolute_tolerance
        coarse, fine = self.rule(bounds, self.order), self.rule(bounds, 2 * self.order)
        error = np.abs(fine - coarse)
        if np.all(error <= allowance + self.rtol * np.abs(fine)):
            return fine, error
        if depth >= self.max_depth:
            raise ContractError(
                f"HC_QUADRATURE_UNRESOLVED: mass={fine.tolist()}, estimated_error={error.tolist()}, "
                f"bounds_mm={bounds}; increase declared quadrature/depth or change ROI, never thin density")
        x0, x1, y0, y1 = bounds
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        children = ((x0, xm, y0, ym), (xm, x1, y0, ym),
                    (x0, xm, ym, y1), (xm, x1, ym, y1))
        results = [self.integrate(child, depth + 1, allowance / 4) for child in children]
        return np.sum([v[0] for v in results], axis=0), np.sum([v[1] for v in results], axis=0)


def _partitions(profile, domain, seed, options, quadrature, total_mass, total_error):
    leaves = []
    pending = [(_Partition("root", tuple(domain.bounds_mm), tuple(total_mass), tuple(total_error), 0))]
    while pending:
        partition = pending.pop()
        bounds = profile.bounds_for_rectangle(partition.bounds)
        density_contrast = max(math.log(bounds[kind + "_max"] / bounds[kind + "_min"])
                               for kind in ("total", "H1", "H2"))
        needs_split = sum(partition.mass) > options["partition_target_mass"] or (
            sum(partition.mass) > options["integration_absolute_tolerance"] and
            density_contrast > options["partition_log_density_range"])
        if not needs_split:
            leaves.append(partition)
            continue
        if partition.depth >= options["max_depth"]:
            raise ContractError("HC_PARTITION_UNRESOLVED: declared depth cannot meet local mass/gradient partition rule")
        if len(leaves) + len(pending) + 2 > options["max_partitions"]:
            raise ContractError("HC_PARTITION_BUDGET_EXCEEDED: change canonical ROI/budget; density is not reduced")
        x0, x1, y0, y1 = partition.bounds
        random = keyed_rng(seed, "HC-irregular-partition:" + partition.key)
        # Seeded non-midpoint bisection gives irregular regions, never a cell lattice.
        fraction = float(random.uniform(.35, .65))
        if x1 - x0 >= y1 - y0:
            cut = x0 + fraction * (x1 - x0)
            children = ((x0, cut, y0, y1), (cut, x1, y0, y1))
        else:
            cut = y0 + fraction * (y1 - y0)
            children = ((x0, x1, y0, cut), (x0, x1, cut, y1))
        child_parts = []
        for index, child in enumerate(children):
            mass, error = quadrature.integrate(child)
            child_parts.append(_Partition(partition.key + str(index), child, tuple(mass),
                                          tuple(error), partition.depth + 1))
        pending.extend(reversed(child_parts))
    return tuple(sorted(leaves, key=lambda partition: partition.key))


def _cumulative_quotas(masses, uniform):
    cumulative = np.cumsum(np.asarray(masses, dtype=float))
    counts = np.diff(np.r_[0, np.floor(cumulative + uniform)]).astype(np.int64)
    if np.any(counts < 0):
        raise ContractError("HC cumulative quotas produced a negative count")
    return counts


def _sample_density(profile, domain, bounds, cell_type, random, max_attempts):
    envelope = profile.bounds_for_rectangle(bounds)[cell_type + "_max"]
    x0, x1, y0, y1 = bounds
    for attempt in range(1, max_attempts + 1):
        point = np.array([random.uniform(x0, x1), random.uniform(y0, y1)])
        if not bool(domain.contains(point.reshape(1, 2))[0]):
            continue
        density = float(profile.type_density(point, cell_type))
        if density > envelope * (1 + 1e-12):
            raise ContractError("HC rejection envelope invalid; refuse biased density sampling")
        if float(random.random()) * envelope < density:
            return point, attempt
    raise ContractError("HC_DENSITY_DRAW_BUDGET_EXCEEDED: mask/local mass needs a larger declared sampling budget")


class _SpatialIndex:
    def __init__(self, positions, width):
        self.positions, self.width = positions, width
        self.buckets = {}
        for i, point in enumerate(positions):
            self.buckets.setdefault(self.key(point), set()).add(i)

    def key(self, point):
        return tuple(np.floor(point / self.width).astype(np.int64))

    def nearby(self, point):
        x, y = self.key(point)
        found = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                found.update(self.buckets.get((x + dx, y + dy), ()))
        return found

    def move(self, index, point):
        old, new = self.key(self.positions[index]), self.key(point)
        if old != new:
            self.buckets[old].remove(index)
            if not self.buckets[old]:
                del self.buckets[old]
            self.buckets.setdefault(new, set()).add(index)
        self.positions[index] = point


def _lengths(profile, point):
    h1 = float(profile.type_density(point, "H1"))
    h2 = float(profile.type_density(point, "H2"))
    ref1, ref2 = profile.reference_density
    return profile.reference_lengths_mm * np.sqrt([ref1 / h1, ref2 / h2,
                                                   (ref1 + ref2) / (h1 + h2)])


def _point_energy(index, point, local_lengths, positions, labels, lengths, spatial, amplitudes):
    energy = 0.0
    for other in sorted(spatial.nearby(point)):
        if other == index:
            continue
        distance = float(np.linalg.norm(point - positions[other]))
        if distance == 0:
            return math.inf
        kind = 0 if labels[index] == labels[other] == "H1" else (
            1 if labels[index] == labels[other] == "H2" else 2)
        scale = (local_lengths[kind] + lengths[other, kind]) / 2
        ratio = distance / scale
        if ratio < 1:
            energy += amplitudes[kind] * (1 - ratio) ** 2
    return energy


def generate_joint_hc(profile, domain, seed, max_cells, options=None):
    """Generate one bounded canonical instance; no mixing/biology certification.

    ``domain`` is a physical soma-plane canonical rectangle plus a named tissue
    mask.  Its owner rectangle does not alter the identity or point generation.
    Independent canonical domains produce distinct instances even for the same
    seed.  Reuse ``mosaic.crop`` for adjoining/overlapping views.
    """
    if not isinstance(profile, HCDensityProfile) or not isinstance(domain, RetinalDomain):
        raise ContractError("HC generation requires HCDensityProfile and physical RetinalDomain")
    if not all(np.isfinite(v) for v in (*domain.bounds_mm, *domain.owner_bounds_mm)):
        raise ContractError("HC canonical and owner bounds must be finite")
    if not domain.mask_source_id:
        raise ContractError("HC canonical generation requires concrete tissue mask identity")
    max_cells = _integer(max_cells, "max_cells", minimum=0)
    # Conservative rectangle support check prevents a quadrature/sampling rule
    # from silently trimming an unsupported halo.  This v1 backend explicitly
    # accepts canonical rectangles entirely inside the radial source support.
    profile.bounds_for_rectangle(domain.bounds_mm)
    resolved = dict(partition_target_mass=8, quadrature_order=8,
                    integration_relative_tolerance=1e-3, integration_absolute_tolerance=1e-5,
                    partition_log_density_range=.35, sweeps=8, temperature=1,
                    a11=4, a22=4, a12=4, max_partitions=2048,
                    max_proposals=500000, max_depth=12, max_draw_attempts=100000,
                    max_quadrature_points=2000000)
    allowed = set(resolved)
    configured = profile.config.get("generator", {})
    if not set(configured) <= allowed or options and not set(options) <= allowed:
        raise ContractError("Unknown HC generator option")
    resolved.update(configured)
    resolved.update(options or {})
    for name in ("partition_target_mass", "integration_relative_tolerance", "integration_absolute_tolerance",
                 "partition_log_density_range", "temperature"):
        resolved[name] = _positive(resolved[name], name)
    for name in ("a11", "a22", "a12"):
        value = float(resolved[name])
        if not np.isfinite(value) or value < 0:
            raise ContractError("Soft-core amplitudes must be finite nonnegative")
        resolved[name] = value
    for name in ("quadrature_order", "max_partitions", "max_proposals", "max_depth", "max_draw_attempts", "max_quadrature_points"):
        resolved[name] = _integer(resolved[name], name)
    resolved["sweeps"] = _integer(resolved["sweeps"], "sweeps", minimum=0)
    if resolved["quadrature_order"] < 2:
        raise ContractError("HC quadrature requires order >=2")
    seed = _integer(seed, "seed", minimum=0)
    quadrature = _Quadrature(profile, domain, resolved["quadrature_order"],
                             resolved["integration_relative_tolerance"],
                             resolved["integration_absolute_tolerance"], resolved["max_depth"],
                             resolved["max_quadrature_points"])
    whole_mass, whole_error = quadrature.integrate(tuple(domain.bounds_mm))
    if float(whole_mass.sum()) >= max_cells + 2:
        raise ContractError(f"HC_CELL_BUDGET_EXCEEDED: density mass={whole_mass.sum()}, max_cells={max_cells}; no density thinning")
    partitions = _partitions(profile, domain, seed, resolved, quadrature, whole_mass, whole_error)
    masses = np.array([partition.mass for partition in partitions], dtype=float).reshape(-1, 2)
    partition_mass, partition_error = masses.sum(axis=0), np.sum([p.error for p in partitions], axis=0)
    reconciliation = np.abs(partition_mass - whole_mass)
    reconciliation_allowance = whole_error + partition_error + resolved["integration_absolute_tolerance"]
    if np.any(reconciliation > reconciliation_allowance):
        raise ContractError("HC_QUADRATURE_PARTITION_MISMATCH: declared local/global integrations disagree; "
            f"whole_mass={whole_mass.tolist()}, partition_mass={partition_mass.tolist()}, "
            f"difference={reconciliation.tolist()}, allowance={reconciliation_allowance.tolist()}; "
            "refine declared numerical integration, no density rescaling")
    offsets = {kind: float(keyed_rng(seed, "HC-cumulative-round:" + kind).random()) for kind in ("H1", "H2")}
    quotas = np.stack([_cumulative_quotas(masses[:, i], offsets[kind])
                       for i, kind in enumerate(("H1", "H2"))], axis=1)
    count = int(quotas.sum())
    if count > max_cells:
        raise ContractError(f"HC_CELL_BUDGET_EXCEEDED: integer target={count}, max_cells={max_cells}; no density thinning")
    proposals = count * resolved["sweeps"]
    if proposals > resolved["max_proposals"]:
        raise ContractError(f"HC_PROPOSAL_BUDGET_EXCEEDED: requested={proposals}; declare ROI/budget/sweeps explicitly")
    identity = {"profile_id": profile.profile_id, "profile_sha256": profile.profile_hash,
                "source_asset_sha256": profile.source_asset_hash, "seed": int(seed),
                "canonical_bounds_mm": list(map(float, domain.bounds_mm)),
                "mask_source_id": domain.mask_source_id,
                "mask_identity": profile.config.get("tissue_mask_identity", domain.mask_source_id),
                "generator_options": resolved}
    instance_hash = _json_hash(identity)
    instance_id = profile.profile_id + ":" + instance_hash[:24]
    random = keyed_rng(seed, "HC-instance:" + instance_hash)
    positions, labels, cell_ids, cell_partitions = [], [], [], []
    sampling_attempts = 0
    for part_index, partition in enumerate(partitions):
        for type_index, kind in enumerate(("H1", "H2")):
            for local_index in range(int(quotas[part_index, type_index])):
                point, attempts = _sample_density(profile, domain, partition.bounds, kind,
                                                  random, resolved["max_draw_attempts"])
                positions.append(point)
                labels.append(kind)
                cell_partitions.append(part_index)
                cell_ids.append(f"{instance_id}:hc:{kind}:{partition.key}:{local_index:06d}")
                sampling_attempts += attempts
    positions = np.array(positions, dtype=float).reshape(-1, 2)
    if len({tuple(point) for point in positions}) != count:
        raise ContractError("HC_EXACT_COINCIDENCE: reject duplicate initialization rather than permit a zero-distance pair")
    labels = np.array(labels, dtype="<U2")
    cell_partitions = np.array(cell_partitions, dtype=np.int64)
    conservative = profile.bounds_for_rectangle(domain.bounds_mm)
    ref1, ref2 = profile.reference_density
    maximum_interaction = float(np.max(profile.reference_lengths_mm * np.sqrt([
        ref1 / conservative["H1_min"], ref2 / conservative["H2_min"],
        (ref1 + ref2) / conservative["total_min"]])))
    lengths = np.array([_lengths(profile, point) for point in positions]).reshape(-1, 3)
    spatial = _SpatialIndex(positions, maximum_interaction)
    amplitudes = np.array([resolved["a11"], resolved["a22"], resolved["a12"]])
    accepted, coincident_rejections = 0, 0
    sweep_energy = []
    for sweep in range(resolved["sweeps"]):
        for index in random.permutation(count):
            partition = partitions[cell_partitions[index]]
            candidate, attempts = _sample_density(profile, domain, partition.bounds, str(labels[index]),
                                                   random, resolved["max_draw_attempts"])
            sampling_attempts += attempts
            candidate_lengths = _lengths(profile, candidate)
            old_energy = _point_energy(index, positions[index], lengths[index], positions,
                                       labels, lengths, spatial, amplitudes)
            new_energy = _point_energy(index, candidate, candidate_lengths, positions,
                                       labels, lengths, spatial, amplitudes)
            if not np.isfinite(new_energy):
                coincident_rejections += 1
                continue
            log_acceptance = min(0.0, -(new_energy - old_energy) / resolved["temperature"])
            if math.log(max(float(random.random()), np.finfo(float).tiny)) < log_acceptance:
                spatial.move(index, candidate)
                lengths[index] = candidate_lengths
                accepted += 1
        energy = .5 * sum(_point_energy(i, p, lengths[i], positions, labels, lengths, spatial,
                                       amplitudes) for i, p in enumerate(positions))
        sweep_energy.append(float(energy))
    if not np.all(domain.contains(positions)):
        raise ContractError("Joint HC sampler escaped the canonical physical mask")
    owner = domain.contains(positions, owner=True)
    diagnostics = {
        **identity, "canonical_instance_id": instance_id, "canonical_instance_sha256": instance_hash,
        "algorithm": "fixed-irregular-partition-quota finite-temperature marked soft-core",
        "algorithm_evidence_class": "project_approximation", "density_profile": profile.metadata,
        "positions_frame": "anatomical_retina", "positions_meaning": "anatomical_position",
        "arbor_anchor_policy": "caller may explicitly choose co_located_lumped_anchor; no functional RF assignment",
        "canonical_cell_count": count, "view_count": count, "owner_count": int(owner.sum()),
        "owner_bounds_mm": list(domain.owner_bounds_mm), "halo_count": int((~owner).sum()),
        "partition_count": len(partitions), "partition_rule": "seeded recursive non-midpoint rectangle split; local mass and log density gradient",
        "partitions": [{"id": p.key, "bounds_mm": list(p.bounds), "H1_H2_mass": list(p.mass),
                        "estimated_absolute_quadrature_error": list(p.error),
                        "H1_H2_integer_quota": quotas[i].tolist(), "depth": p.depth}
                       for i, p in enumerate(partitions)],
        "total_H1_H2_mass": partition_mass.tolist(), "whole_domain_H1_H2_mass": whole_mass.tolist(),
        "total_H1_H2_counts": quotas.sum(axis=0).tolist(), "randomized_rounding_offsets": offsets,
        "count_minus_density_mass_H1_H2": (quotas.sum(axis=0) - partition_mass).tolist(),
        "partition_count_minus_mass": (quotas - masses).tolist(),
        "maximum_partition_integer_error_bound": "strictly less than one cell relative to estimated local integral",
        "quadrature_estimated_absolute_error_H1_H2": partition_error.tolist(),
        "quadrature_global_local_difference_H1_H2": reconciliation.tolist(),
        "quadrature_evaluated_points": quadrature.evaluated_points,
        "rounding_policy": "one seeded U per type and cumulative partition mass; no tile/ring independent rounding; no minimum count",
        "initialization_and_proposal": "continuous rejection sampling from rho_type within own partition and physical mask",
        "target_distribution": "proportional to product rho_type(x) exp(-E/T), conditioned on selected partition/type counts",
        "MH_acceptance": "min(1, exp(-delta_E/T)); density factors cancel against independent density proposal",
        "pair_scale_policy": "mean of endpoint density-scaled Eglen field-B lengths; new project soft-core conversion",
        "reference_window_mm": profile.reference_window_mm.tolist(),
        "reference_H1_H2_counts": profile.reference_counts.tolist(),
        "reference_dmin_mean_um": profile.reference_dmin_mean_um.tolist(),
        "reference_dmin_sd_um_not_sampled": profile.reference_dmin_sd_um,
        "reference_lower_truncation_um_not_used": profile.reference_lower_truncation_um,
        "maximum_interaction_support_mm": maximum_interaction,
        "proposals_requested": proposals, "proposals_attempted": proposals,
        "accepted_proposals": accepted, "acceptance_fraction": accepted / proposals if proposals else None,
        "coincident_proposal_rejections": coincident_rejections, "density_draw_attempts": sampling_attempts,
        "energy_after_each_sweep": sweep_energy,
        "mixing_status": "NOT_CERTIFIED; fixed bounded sweeps do not establish stationary sampling",
        "generation_status": "GENERATED_WITH_DECLARED_BOUNDED_WORK; not a statistical validation",
        "statistical_validation_status": "NOT_RUN; no Eglen reproduction, arbitrary-window intensity or soma nonoverlap claim",
        "implementation_status": "IMPLEMENTED_NOT_RUN",
        "canonical_view_policy": "crop this one complete instance; independently generated domains are different identities",
        "boundary_policy": "finite canonical domain; pair neighborhoods outside canonical bounds absent; no mirrored/duplicated cells",
        "quota_scope_limit": "partition integer quotas only; soft-core can change within-partition first-order intensity; no arbitrary-window exact-density claim",
    }
    return JointHCMosaic(Quantity(positions, "mm", "anatomical_position",
                                  "project_approximation", "anatomical_retina"),
                         labels, tuple(cell_ids), owner, diagnostics)
