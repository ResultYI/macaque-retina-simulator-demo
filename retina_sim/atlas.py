"""Immutable supplied anatomy and ROI references in one global retinal atlas.

This module does not sample density, infer displacement or certify source support.
Canonical source keys belong to a fixed atlas asset, never a requested ROI/tile.
"""
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .contracts import ContractError, Quantity, canonical_hash


COORDINATE_ROLES = MappingProxyType({
    "rf": ("functional_position", "functional_visual_field"),
    "opl_pedicle": ("pedicle_position", "OPL"),
    "ipl_preterminal": ("terminal_position", "IPL"),
    "ipl_post_arbor": ("dendritic_position", "IPL"),
    "soma": ("anatomical_position", "anatomical_retina"),
})


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a nonempty string")
    return value


def global_cell_id(atlas_id, atlas_version, source_version, population, canonical_source_key):
    """Identity depends only on the canonical source namespace and key."""
    parts = {name: _text(value, name) for name, value in (
        ("atlas_id", atlas_id), ("atlas_version", atlas_version),
        ("source_version", source_version), ("population", population),
        ("canonical_source_key", canonical_source_key))}
    return "atlas:" + canonical_hash(parts)


def _immutable_array(values, dtype):
    array = np.asarray(values, dtype=dtype)
    # A bytes-backed view cannot be made writable by a caller of the atlas.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _bounds(value, name):
    try:
        bounds = tuple(float(x) for x in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ContractError(f"{name} requires four finite mm bounds") from error
    if (len(bounds) != 4 or not np.all(np.isfinite(bounds)) or
            bounds[1] <= bounds[0] or bounds[3] <= bounds[2]):
        raise ContractError(f"{name} requires increasing finite mm bounds")
    return bounds


def _inside(points, bounds):
    x0, x1, y0, y1 = bounds
    return ((points[:, 0] >= x0) & (points[:, 0] < x1) &
            (points[:, 1] >= y0) & (points[:, 1] < y1))


@dataclass(frozen=True)
class AtlasROI:
    """Owner and finite query rectangles in the same explicit coordinate role.

    Half-open ownership lets neighboring tiles share a boundary without claiming
    the same owner twice. These bounds do not assert biological/source coverage.
    """
    owner_bounds_mm: tuple
    query_bounds_mm: tuple
    reference_id: str
    coordinate_role: str = "rf"
    unit: str = "mm"

    def __post_init__(self):
        _text(self.reference_id, "reference_id")
        if self.unit != "mm" or self.coordinate_role not in COORDINATE_ROLES:
            raise ContractError("Atlas ROI requires mm and an explicit coordinate role")
        owner = _bounds(self.owner_bounds_mm, "owner_bounds_mm")
        query = _bounds(self.query_bounds_mm, "query_bounds_mm")
        if not (query[0] <= owner[0] < owner[1] <= query[1] and
                query[2] <= owner[2] < owner[3] <= query[3]):
            raise ContractError("Owner bounds must lie within query bounds")
        object.__setattr__(self, "owner_bounds_mm", owner)
        object.__setattr__(self, "query_bounds_mm", query)

    @property
    def coordinate_frame(self):
        return COORDINATE_ROLES[self.coordinate_role][1]


@dataclass(frozen=True)
class CanonicalCellTable:
    """Supplied fixed rows, with distinct typed coordinates where available.

    Missing coordinates remain absent rather than using a soma/RF fallback.
    Empty rod/S tables are legal. A reference ID must identify the common eye,
    origin and axis convention in the surrounding atlas configuration.
    """
    atlas_id: str
    atlas_version: str
    source_version: str
    population: str
    source_keys: tuple
    coordinates: object
    reference_id: str
    global_cell_ids: tuple = field(init=False)
    table_hash: str = field(init=False)

    def __post_init__(self):
        for name in ("atlas_id", "atlas_version", "source_version", "population", "reference_id"):
            _text(getattr(self, name), name)
        keys = tuple(_text(value, "canonical_source_key") for value in self.source_keys)
        if len(set(keys)) != len(keys):
            raise ContractError("Canonical source keys must be unique")
        if not hasattr(self.coordinates, "items") or not self.coordinates:
            raise ContractError("Canonical table requires explicitly typed coordinates")
        coordinates = {}
        for role, quantity in self.coordinates.items():
            if role not in COORDINATE_ROLES or not isinstance(quantity, Quantity):
                raise ContractError("Unknown coordinate role or untyped coordinate values")
            meaning, frame = COORDINATE_ROLES[role]
            quantity.require("mm", meaning, (len(keys), 2), frame)
            copied = Quantity(quantity.values, "mm", meaning, quantity.evidence_tag, frame)
            object.__setattr__(copied, "values", _immutable_array(copied.values, np.float64))
            coordinates[role] = copied
        ids = tuple(global_cell_id(self.atlas_id, self.atlas_version, self.source_version,
                                   self.population, key) for key in keys)
        object.__setattr__(self, "source_keys", keys)
        object.__setattr__(self, "coordinates", MappingProxyType(coordinates))
        object.__setattr__(self, "global_cell_ids", ids)
        object.__setattr__(self, "table_hash", canonical_hash(self.record(include_hash=False)))

    def coordinate(self, role):
        if role not in self.coordinates:
            raise ContractError(f"Coordinate {role} is not supplied; coordinate fallback forbidden")
        return self.coordinates[role]

    def query(self, roi, *, biological_absence_source_id=None):
        """Select existing rows, with an optional source-defined owner absence.

        The caller must obtain an absence declaration from its biological mask
        provider. Zero sampled owners alone never establishes biological absence.
        Halo rows remain references even when the owner is biologically absent.
        """
        if not isinstance(roi, AtlasROI) or roi.reference_id != self.reference_id:
            raise ContractError("Atlas query must use the same global reference")
        points = self.coordinate(roi.coordinate_role).values
        indices = np.flatnonzero(_inside(points, roi.query_bounds_mm))
        owner = _inside(points[indices], roi.owner_bounds_mm)
        if biological_absence_source_id is not None:
            _text(biological_absence_source_id, "biological_absence_source_id")
            if np.any(owner):
                raise ContractError("Biological absence conflicts with supplied owner cells")
        status = ("BIOLOGICAL_ABSENCE" if biological_absence_source_id is not None else
                  "POPULATED" if np.any(owner) else "EMPTY_SAMPLE")
        return AtlasSelection(self, roi, _immutable_array(indices, np.int64),
                              _immutable_array(owner, np.bool_), status,
                              biological_absence_source_id)

    def record(self, *, include_hash=True):
        record = {name: getattr(self, name) for name in (
            "atlas_id", "atlas_version", "source_version", "population", "reference_id")}
        record.update(source_keys=list(self.source_keys), global_cell_ids=list(self.global_cell_ids),
                      coordinates={role: {"values": quantity.values.tolist(), "unit": quantity.unit,
                          "meaning": quantity.meaning, "frame": quantity.coordinate_frame,
                          "evidence_tag": quantity.evidence_tag}
                          for role, quantity in self.coordinates.items()})
        if include_hash:
            record["table_hash"] = self.table_hash
        return record


@dataclass(frozen=True)
class AtlasSelection:
    """A row-index view; no independent anatomy is generated for owner/halo."""
    table: CanonicalCellTable
    roi: AtlasROI
    row_indices: np.ndarray
    owner: np.ndarray
    owner_status: str
    biological_absence_source_id: str | None

    @property
    def global_cell_ids(self):
        return tuple(self.table.global_cell_ids[i] for i in self.row_indices)

    @property
    def source_keys(self):
        return tuple(self.table.source_keys[i] for i in self.row_indices)

    def coordinate(self, role):
        quantity = self.table.coordinate(role)
        return Quantity(quantity.values[self.row_indices], quantity.unit, quantity.meaning,
                        quantity.evidence_tag, quantity.coordinate_frame)
