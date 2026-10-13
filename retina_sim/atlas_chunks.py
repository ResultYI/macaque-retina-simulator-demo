"""Lossless spatial storage of existing frozen masters; never a mosaic generator.

Original JSON remains the identity/backup. Blocks partition rows, not density
mass. Only the manifest/index is eager; metadata and intersecting blocks are lazy.
"""
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np

from .atlas import AtlasROI, CanonicalCellTable, _bounds, _inside, _immutable_array, _text
from .atlas_store import load_atlas_asset
from .contracts import ContractError, Quantity, canonical_hash
from .equal_mass_master import FixedEqualMassMaster


SCHEMA = "frozen_atlas_spatial_chunks_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _verified_bytes(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ContractError("Chunked atlas file identity differs: " + str(path))
    return raw


def read_frozen_master(source_json, *, expected_source_sha256):
    """Read a supplied equal-mass JSON or a single-population canonical asset."""
    path = Path(source_json)
    raw = _verified_bytes(path, expected_source_sha256)
    payload = json.loads(raw)
    if "recipe" in payload and "positions_mm" in payload:
        master = FixedEqualMassMaster(payload, path, expected_source_sha256)
        table = master.table
        metadata = {k: v for k, v in payload.items() if k not in ("source_keys", "positions_mm")}
        return table, tuple([table.population] * len(table.source_keys)), master.bounds, "rf", 0.0, metadata
    asset = load_atlas_asset(path, expected_sha256=expected_source_sha256)
    if len(asset.tables) != 1:
        raise ContractError("Convert one frozen population per chunk store")
    table = next(iter(asset.tables.values()))
    metadata = asset.metadata_record()
    types = metadata["source"].pop("types", None)
    types = tuple(types) if types is not None else tuple([table.population] * len(table.source_keys))
    role = "soma" if "soma" in table.coordinates else "rf"
    return (table, types, tuple(metadata["boundary"]["canonical_bounds_mm"]), role,
            float(metadata["boundary"].get("pair_support_context_mm", 0.0)), metadata)


def convert_frozen_master(source_json, destination, *, expected_source_sha256, block_size_mm=0.1):
    """Partition one immutable JSON into new files; refuse overwrite and sampling.

    Conversion is a one-time full JSON read. Runtime queries do not open it.
    No generator, per-block quota, density callback or configuration is changed.
    """
    source = Path(source_json).resolve()
    destination = Path(destination).resolve()
    table, types, bounds, role, pair_context, metadata = read_frozen_master(
        source, expected_source_sha256=expected_source_sha256)
    bounds = _bounds(bounds, "canonical_bounds_mm")
    size = float(block_size_mm)
    points = table.coordinate(role).values
    if not np.isfinite(size) or size <= 0 or not np.all(_inside(points, bounds)):
        raise ContractError("Finite block width and all supplied rows inside fixed bounds required")
    if len(types) != len(points) or any(not isinstance(t, str) or not t for t in types):
        raise ContractError("Frozen cell types must align with all rows")
    scaled = np.floor(points / size)
    if np.any(np.abs(scaled) > np.iinfo(np.int64).max // 2):
        raise ContractError("Block grid exceeds integer range")
    tiles = scaled.astype(np.int64)
    # A stable grouping keeps each original row index, including joint marks.
    order = np.lexsort((np.arange(len(points)), tiles[:, 1], tiles[:, 0]))
    grouped = tiles[order]
    boundaries = np.r_[0, np.flatnonzero(np.any(np.diff(grouped, axis=0), axis=1)) + 1, len(order)]
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ContractError("Immutable chunk destination already exists; overwrite refused") from error
    metadata_raw = _json_bytes(metadata)
    (destination / "metadata.json").write_bytes(metadata_raw)
    entries = []
    keys, ids, marks = np.asarray(table.source_keys, dtype=str), np.asarray(table.global_cell_ids, dtype=str), np.asarray(types, dtype=str)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        rows = order[start:end]
        if not len(rows):
            continue
        arrays = {"row_indices": rows, "source_keys": keys[rows], "global_cell_ids": ids[rows], "cell_types": marks[rows]}
        arrays.update({"coordinate_" + r: q.values[rows] for r, q in table.coordinates.items()})
        name = f"block_{len(entries):07d}.npz"
        with (destination / name).open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        xy = points[rows]
        entries.append({"tile": tiles[rows[0]].tolist(), "file": name, "count": len(rows),
                        "point_bounds_mm": [float(xy[:, 0].min()), float(xy[:, 0].max()),
                                            float(xy[:, 1].min()), float(xy[:, 1].max())],
                        "sha256": file_sha256(destination / name),
                        "file_bytes": (destination / name).stat().st_size,
                        "decoded_array_bytes": sum(a.nbytes for a in arrays.values())})
    manifest = {"schema_version": SCHEMA, "source_json_path": str(source),
                "source_json_sha256": expected_source_sha256, "source_json_bytes": source.stat().st_size,
                "full_table_hash": table.table_hash, "count": len(points),
                "canonical_bounds_mm": list(bounds), "index_role": role,
                "pair_support_context_mm": pair_context, "block_size_mm": size,
                "table": {name: getattr(table, name) for name in
                          ("atlas_id", "atlas_version", "source_version", "population", "reference_id")},
                "coordinates": {r: {"unit": q.unit, "meaning": q.meaning,
                                     "frame": q.coordinate_frame, "evidence_tag": q.evidence_tag}
                                for r, q in table.coordinates.items()},
                "metadata_file": "metadata.json", "metadata_sha256": hashlib.sha256(metadata_raw).hexdigest(),
                "blocks": entries, "conversion": "existing rows only; original order/IDs/float64/types retained",
                "source_coverage_claim": "unchanged original metadata; spatial index adds no source support"}
    manifest["storage_manifest_hash"] = canonical_hash(manifest)
    # The manifest is published last; an interrupted conversion is not loadable.
    with (destination / "manifest.json").open("xb") as stream:
        stream.write(_json_bytes(manifest))
    return {"path": str(destination), "source_json_sha256": expected_source_sha256,
            "storage_manifest_hash": manifest["storage_manifest_hash"], "count": len(points),
            "block_count": len(entries), "storage_bytes": sum(p.stat().st_size for p in destination.iterdir())}


@dataclass(frozen=True)
class ChunkedSelection:
    """A canonical selected table plus original master row indices and marks."""
    table: CanonicalCellTable
    row_indices: np.ndarray
    cell_types: tuple
    owner: np.ndarray
    owner_status: str
    roi: AtlasROI
    biological_absence_source_id: str | None
    io_stats: dict

    @property
    def global_cell_ids(self):
        return self.table.global_cell_ids

    @property
    def source_keys(self):
        return self.table.source_keys

    def coordinate(self, role):
        return self.table.coordinate(role)


class ChunkedAtlasReader:
    """An index and bounded query workspace, with no full-master row cache.

    Pin source SHA256; also pin storage_manifest_hash in integration receipts.
    Biological/source masks and recurrent graph closure remain caller contracts.
    """
    def __init__(self, directory, *, expected_source_sha256, expected_manifest_hash=None):
        self.directory = Path(directory)
        data = json.loads((self.directory / "manifest.json").read_bytes())
        digest = data.pop("storage_manifest_hash")
        if (data.get("schema_version") != SCHEMA or canonical_hash(data) != digest or
                data.get("source_json_sha256") != expected_source_sha256 or
                (expected_manifest_hash is not None and digest != expected_manifest_hash)):
            raise ContractError("Chunked master schema/source/storage manifest identity differs")
        self.manifest = dict(data, storage_manifest_hash=digest)
        self.bounds = _bounds(data["canonical_bounds_mm"], "canonical_bounds_mm")
        self.sha256 = expected_source_sha256
        self.full_table_hash = data["full_table_hash"]
        self.reference_id = data["table"]["reference_id"]
        self.index_role = data["index_role"]
        self._size = float(data["block_size_mm"])
        if not np.isfinite(self._size) or self._size <= 0 or self.index_role not in data["coordinates"]:
            raise ContractError("Chunk spatial index width/role invalid")
        self._blocks = {}
        for entry in data["blocks"]:
            tile = tuple(entry["tile"])
            if tile in self._blocks or not re.fullmatch(r"block_[0-9]{7}\.npz", entry["file"]):
                raise ContractError("Duplicate spatial tile or invalid block filename")
            self._blocks[tile] = entry
        if sum(e["count"] for e in self._blocks.values()) != data["count"]:
            raise ContractError("Spatial index counts differ from full master")

    def metadata_record(self):
        """Read unchanged source/density/generator/provenance metadata on demand.

        HC source.types is row data: use selection.cell_types for queried marks.
        """
        return json.loads(_verified_bytes(self.directory / "metadata.json", self.manifest["metadata_sha256"]))

    def query(self, roi, *, max_cells=60000, require_pair_context=True, biological_absence_source_id=None):
        if not isinstance(roi, AtlasROI) or roi.reference_id != self.reference_id or roi.coordinate_role != self.index_role:
            raise ContractError("Chunk query requires the indexed role and same global reference")
        if isinstance(max_cells, bool) or int(max_cells) != max_cells or max_cells < 0:
            raise ContractError("Query cell budget requires a nonnegative integer")
        b = roi.query_bounds_mm
        halo = float(self.manifest["pair_support_context_mm"]) if require_pair_context else 0.0
        context = (b[0] - halo, b[1] + halo, b[2] - halo, b[3] + halo)
        tolerance = 0.0 if halo else 1e-12
        if (context[0] < self.bounds[0] - tolerance or context[1] > self.bounds[1] + tolerance or
                context[2] < self.bounds[2] - tolerance or context[3] > self.bounds[3] + tolerance):
            raise ContractError("FIXED_MASTER_UNSUPPORTED: query/context crosses frozen master; no regeneration")
        # Neighbor margin guards floating division exactly on tile boundaries.
        x0, x1 = int(np.floor(b[0] / self._size)) - 1, int(np.floor(b[1] / self._size)) + 1
        y0, y1 = int(np.floor(b[2] / self._size)) - 1, int(np.floor(b[3] / self._size)) + 1
        if (x1 - x0 + 1) * (y1 - y0 + 1) < len(self._blocks):
            candidates = (self._blocks.get((x, y)) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))
        else:
            candidates = (e for (x, y), e in self._blocks.items() if x0 <= x <= x1 and y0 <= y <= y1)
        roles = tuple(self.manifest["coordinates"])
        required = {"row_indices", "source_keys", "global_cell_ids", "cell_types", *("coordinate_" + r for r in roles)}
        pieces = {key: [] for key in required}
        stats = {"blocks_total": len(self._blocks), "blocks_read": 0, "rows_decoded": 0,
                 "compressed_bytes_read": 0, "decoded_array_bytes_sum": 0, "largest_block_decoded_bytes": 0,
                 "pair_context_checked": bool(require_pair_context), "source_json_sha256": self.sha256}
        count = 0
        for entry in candidates:
            if entry is None:
                continue
            lo, hi, bottom, top = entry["point_bounds_mm"]
            if hi < b[0] or lo >= b[1] or top < b[2] or bottom >= b[3]:
                continue
            raw = _verified_bytes(self.directory / entry["file"], entry["sha256"])
            with np.load(BytesIO(raw), allow_pickle=False) as archive:
                if set(archive.files) != required:
                    raise ContractError("Chunk fields differ from declared coordinate/identity schema")
                arrays = {key: archive[key] for key in required}
            n = entry["count"]
            rows = arrays["row_indices"]
            if (rows.shape != (n,) or rows.dtype.kind != "i" or np.any(rows < 0) or
                    np.any(rows >= self.manifest["count"])):
                raise ContractError("Chunk original row indices invalid")
            for key in ("source_keys", "global_cell_ids", "cell_types"):
                if arrays[key].shape != (n,) or arrays[key].dtype.kind != "U":
                    raise ContractError("Chunk ordered identity/type fields invalid")
            for role in roles:
                a = arrays["coordinate_" + role]
                if a.shape != (n, 2) or a.dtype != np.dtype("float64") or not np.all(np.isfinite(a)):
                    raise ContractError("Chunk typed coordinates must be finite float64 Nx2")
            mask = _inside(arrays["coordinate_" + self.index_role], b)
            count += int(mask.sum())
            if count > max_cells:
                raise ContractError("Fixed master query exceeds cell budget; no thinning")
            for key, array in arrays.items():
                pieces[key].append(array[mask])
            decoded = sum(a.nbytes for a in arrays.values())
            stats["blocks_read"] += 1
            stats["rows_decoded"] += n
            stats["compressed_bytes_read"] += len(raw)
            stats["decoded_array_bytes_sum"] += decoded
            stats["largest_block_decoded_bytes"] = max(stats["largest_block_decoded_bytes"], decoded)
        empty = {"row_indices": np.empty(0, dtype=np.int64), "source_keys": np.empty(0, dtype=str),
                 "global_cell_ids": np.empty(0, dtype=str), "cell_types": np.empty(0, dtype=str)}
        empty.update({"coordinate_" + r: np.empty((0, 2), dtype=np.float64) for r in roles})
        arrays = {key: np.concatenate(parts) if parts else empty[key] for key, parts in pieces.items()}
        order = np.argsort(arrays["row_indices"], kind="stable")
        arrays = {key: a[order] for key, a in arrays.items()}
        rows = arrays["row_indices"]
        if np.any(np.diff(rows) <= 0):
            raise ContractError("Duplicate original rows across spatial chunks")
        coordinates = {r: Quantity(arrays["coordinate_" + r], m["unit"], m["meaning"], m["evidence_tag"], m["frame"])
                       for r, m in self.manifest["coordinates"].items()}
        table = CanonicalCellTable(**self.manifest["table"], source_keys=tuple(arrays["source_keys"]), coordinates=coordinates)
        if table.global_cell_ids != tuple(arrays["global_cell_ids"]):
            raise ContractError("Chunk stored IDs differ from original canonical source identities")
        owner = _inside(table.coordinate(self.index_role).values, roi.owner_bounds_mm)
        if biological_absence_source_id is not None:
            _text(biological_absence_source_id, "biological_absence_source_id")
            if np.any(owner):
                raise ContractError("Biological absence conflicts with supplied owner cells")
        status = ("BIOLOGICAL_ABSENCE" if biological_absence_source_id is not None else
                  "POPULATED" if np.any(owner) else "EMPTY_SAMPLE")
        stats["rows_returned"] = count
        return ChunkedSelection(table, _immutable_array(rows, np.int64), tuple(arrays["cell_types"]),
                                _immutable_array(owner, np.bool_), status, roi, biological_absence_source_id, stats)

    def crop(self, roi, max_cells=60000):
        """Equal-mass crop-shaped API; query() also exposes types and global rows."""
        return self.query(roi, max_cells=max_cells).table
