"""Frame-aware asset coverage; coverage never certifies physiological validation."""
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
import hashlib
import json

import numpy as np

from .contracts import ContractError, Quantity


COVERAGE_STATUSES = ("source_supported", "project_approximation", "unsupported")


def _finite_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and np.isfinite(value)


@dataclass(frozen=True)
class SpatialDomain:
    coordinate_frame: str
    unit: str
    position_meaning: str
    radial_interval: tuple
    ray_angles_deg: tuple = ()
    sampled_radii: tuple = ()
    bounds: tuple | None = None

    def __post_init__(self):
        if (not isinstance(self.coordinate_frame, str) or not self.coordinate_frame or self.unit not in ("mm", "deg")
                or not isinstance(self.position_meaning, str) or not self.position_meaning):
            raise ContractError("Spatial domain requires frame, position meaning and mm/deg unit")
        for name in ("radial_interval", "ray_angles_deg", "sampled_radii"):
            if not isinstance(getattr(self, name), (tuple, list)):
                raise ContractError("Spatial domain intervals/rays/samples require sequences")
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.radial_interval) != 2:
            raise ContractError("Radial interval requires lower and upper bounds")
        lo, hi = self.radial_interval
        if not _finite_number(lo) or lo < 0 or (hi is not None and (not _finite_number(hi) or hi < lo)):
            raise ContractError("Invalid radial domain bounds")
        if any(not _finite_number(a) or not 0 <= a < 360 for a in self.ray_angles_deg):
            raise ContractError("Ray angles must lie in [0,360) degrees")
        if (any(not _finite_number(r) or r < lo or (hi is not None and r > hi) for r in self.sampled_radii)
                or len(set(self.sampled_radii)) != len(self.sampled_radii)):
            raise ContractError("Sample radii must be unique and inside the declared source domain")
        if self.bounds is not None:
            if not isinstance(self.bounds, (tuple, list)) or len(self.bounds) != 4 or not all(_finite_number(x) for x in self.bounds):
                raise ContractError("Query rectangle requires four finite bounds")
            object.__setattr__(self, "bounds", tuple(self.bounds))
            x0, x1, y0, y1 = self.bounds
            if x0 >= x1 or y0 >= y1:
                raise ContractError("Query rectangle requires positive 2D extent")

    @classmethod
    def from_dict(cls, row):
        return cls(row["coordinate_frame"], row["unit"], row["position_meaning"],
                   tuple(row["radial_interval"]), tuple(row.get("ray_angles_deg", ())),
                   tuple(row.get("sampled_radii", ())),
                   None if row.get("bounds") is None else tuple(row["bounds"]))

    def contains(self, positions):
        if not isinstance(positions, Quantity):
            raise ContractError("Coverage query requires a typed Quantity; arrays have no frame")
        xy = positions.require(self.unit, self.position_meaning, frame=self.coordinate_frame)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ContractError("Coverage positions require shape (n,2)")
        r = np.linalg.norm(xy, axis=1)
        lo, hi = self.radial_interval
        inside = (r >= lo) & (True if hi is None else r <= hi)
        if self.sampled_radii:
            # Literal nodes remain discrete; a radial bounding interval cannot fill gaps.
            inside &= np.isin(r, self.sampled_radii)
        if self.ray_angles_deg:
            angles = np.mod(np.degrees(np.arctan2(xy[:, 1], xy[:, 0])), 360)
            difference = np.abs((angles[:, None] - np.asarray(self.ray_angles_deg) + 180) % 360 - 180)
            inside &= np.any(difference <= 1e-12, axis=1)
        if self.bounds is not None:
            x0, x1, y0, y1 = self.bounds
            inside &= (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
        return inside

    def compatible_with(self, other):
        return (self.coordinate_frame, self.unit, self.position_meaning) == (
            other.coordinate_frame, other.unit, other.position_meaning)


@dataclass(frozen=True)
class CoverageRecord:
    provider_id: str
    source_domain: tuple
    application_domain: SpatialDomain | None
    provenance: dict
    active: bool
    data_status: str
    evidence_status: str
    transfer: dict | None

    @classmethod
    def from_dict(cls, provider_id, row):
        required = ("species", "specimen", "source", "source_frame", "source_version",
                    "extrapolation_rule", "evidence_status", "data_status", "representation")
        if any(key not in row for key in required) or not row["source_version"]:
            raise ContractError("Coverage record is missing explicit provenance: " + provider_id)
        active = row.get("active")
        if not isinstance(active, bool):
            raise ContractError("Coverage active flag must be boolean")
        domains = tuple(SpatialDomain.from_dict(d) for d in row.get("source_domain", ()))
        application = None if row.get("application_domain") is None else SpatialDomain.from_dict(row["application_domain"])
        if any(d.coordinate_frame != row["source_frame"] for d in domains):
            raise ContractError("Source domain frame disagrees with source_frame")
        if domains and any(not d.compatible_with(domains[0]) for d in domains):
            raise ContractError("One provider must use one source coordinate type/unit")
        transfer = row.get("transfer")
        if application is not None and any(not d.compatible_with(application) for d in domains):
            if not transfer or not transfer.get("transfer_id") or transfer.get("evidence_status") != "project_approximation":
                raise ContractError("Cross-frame/unit application requires a named project transfer")
        if row["evidence_status"] not in ("published_code_tabulation", "published_fit", "measurement",
                "source_quotation", "direct_morphology", "digitized_author_derived", "digitized_author_map_resampling",
                "published_table_measurement", "digitized_published_replot", "author_assumption", "project_approximation", "unavailable"):
            raise ContractError("Unknown source evidence status")
        if row["data_status"] not in ("ready", "source_quotation_only", "pending_digitization",
                "pending_primary_data", "inactive_candidate"):
            raise ContractError("Unknown data status")
        if active and (row["data_status"] in ("pending_digitization", "pending_primary_data", "inactive_candidate")
                       or row["evidence_status"] == "unavailable"):
            raise ContractError("Missing data or an inactive candidate cannot enable a provider")
        if any(d.radial_interval[1] is None for d in domains):
            raise ContractError("Source domain cannot claim unbounded data coverage")
        if row["representation"] in ("effective_disk", "continuous_project_field", "legacy_displacement_adapter"):
            if row["evidence_status"] not in ("author_assumption", "project_approximation", "unavailable"):
                raise ContractError("A fitted field/effective disk/legacy displacement is not measured morphology")
        return cls(provider_id, domains, application, {key: row[key] for key in row
                   if key not in ("source_domain", "application_domain", "active", "transfer")},
                   active, row["data_status"], row["evidence_status"], transfer)


@dataclass(frozen=True)
class CoverageResult:
    provider_id: str
    status: tuple
    reason: tuple
    source_contains: tuple
    application_contains: tuple
    query_support_contains: tuple
    source_domain: tuple
    application_domain: dict | None
    query_support: dict | None
    provenance: dict
    domain_role: str

    @property
    def statuses(self):
        return self.status

    def manifest(self):
        return dict(asdict(self), status_counts={s: self.status.count(s) for s in COVERAGE_STATUSES},
                    coverage_meaning="support by specified asset/domain, not physiological validation",
                    query_support_meaning="point containment only; complete arbor/recurrent dependency closure not certified")


class CoverageCatalog:
    def __init__(self, data, *, path=None, sha256=None):
        if data.get("schema_version") != "1.0" or data.get("profile_id") != "atlas_source_domains_v1":
            raise ContractError("Unknown atlas source-domain catalog identity/version")
        self.data = data
        self.path, self.sha256 = path, sha256
        self.records = MappingProxyType({key: CoverageRecord.from_dict(key, row)
                                        for key, row in data["providers"].items()})

    def query(self, provider_id, positions, *, query_support=None, domain_role="application"):
        if provider_id not in self.records:
            raise ContractError("Unknown atlas coverage provider " + provider_id)
        if domain_role not in ("application", "source"):
            raise ContractError("Coverage domain_role must be application or source")
        record = self.records[provider_id]
        if not isinstance(positions, Quantity) or positions.values.ndim != 2 or positions.values.shape[1] != 2:
            raise ContractError("Coverage positions require a typed Quantity with shape (n,2)")
        n = len(positions.values)
        selected = record.source_domain if domain_role == "source" else (() if record.application_domain is None else (record.application_domain,))
        if selected:
            # Typed checks run even for pending/inactive records, rather than accepting a wrong frame.
            selected[0].contains(positions)
        else:
            if positions.coordinate_frame != record.provenance["source_frame"] and domain_role == "source":
                raise ContractError("Pending source still requires its declared source frame")
            unit = record.provenance.get("source_unit") if domain_role == "source" else record.provenance.get("application_unit")
            frame = record.provenance.get("application_frame") if domain_role == "application" else None
            meaning = record.provenance.get("source_position_meaning" if domain_role == "source" else "application_position_meaning")
            if (unit is not None and positions.unit != unit or frame is not None and positions.coordinate_frame != frame
                    or meaning is not None and positions.meaning != meaning):
                raise ContractError("Pending coverage coordinate type/unit mismatch")
        if query_support is not None and not isinstance(query_support, SpatialDomain):
            raise ContractError("query_support requires a SpatialDomain")
        query_inside = np.ones(n, dtype=bool) if query_support is None else query_support.contains(positions)
        app_inside = None
        if record.application_domain is not None and (positions.coordinate_frame, positions.unit, positions.meaning) == (
                record.application_domain.coordinate_frame, record.application_domain.unit, record.application_domain.position_meaning):
            app_inside = record.application_domain.contains(positions)
        source_inside = None
        if record.source_domain and (positions.coordinate_frame, positions.unit, positions.meaning) == (
                record.source_domain[0].coordinate_frame, record.source_domain[0].unit, record.source_domain[0].position_meaning):
            source_inside = np.logical_or.reduce([d.contains(positions) for d in record.source_domain])
        status, reasons = [], []
        direct_evidence = record.evidence_status not in ("project_approximation", "author_assumption", "unavailable")
        for i in range(n):
            if not record.active:
                label, reason = "unsupported", record.data_status
            elif not query_inside[i]:
                label, reason = "unsupported", "outside_query_support"
            elif domain_role == "source":
                if source_inside is None or not source_inside[i]:
                    label, reason = "unsupported", "outside_source_domain_or_missing_curve"
                else:
                    label = "source_supported" if direct_evidence else "project_approximation"
                    reason = "specified_source_asset" if direct_evidence else "source_derived_project_representation"
            elif app_inside is None or not app_inside[i]:
                label, reason = "unsupported", "outside_application_domain_or_missing_curve"
            elif source_inside is not None and source_inside[i] and direct_evidence and record.transfer is None:
                label, reason = "source_supported", "specified_source_asset"
            else:
                label, reason = "project_approximation", "named_application_interpolation_or_transfer"
            status.append(label)
            reasons.append(reason)
        return CoverageResult(provider_id, tuple(status), tuple(reasons),
            (None,) * n if source_inside is None else tuple(bool(x) for x in source_inside),
            (None,) * n if app_inside is None else tuple(bool(x) for x in app_inside),
            tuple(bool(x) for x in query_inside), tuple(asdict(d) for d in record.source_domain),
            None if record.application_domain is None else asdict(record.application_domain),
            None if query_support is None else asdict(query_support),
            dict(record.provenance, active=record.active, transfer=record.transfer), domain_role)

    def manifest(self):
        return dict(self.data, path=self.path, sha256=self.sha256,
                    validation_meaning="provenance/domain engineering only; not physiological validation")


def load_atlas_source_domains(root):
    root = Path(root)
    path = root / "configurations/atlas_source_domains_v1.json"
    raw = path.read_bytes()
    data = json.loads(raw)
    for artifact in data.get("existing_artifacts", ()):
        if hashlib.sha256((root / artifact["path"]).read_bytes()).hexdigest() != artifact["sha256"]:
            raise ContractError("Atlas provenance asset differs from frozen identity " + artifact["path"])
    return CoverageCatalog(data, path=str(path), sha256=hashlib.sha256(raw).hexdigest())
