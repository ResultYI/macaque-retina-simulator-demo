"""Frozen supplied atlas assets; instance archives never imply global coverage."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np

from .atlas import CanonicalCellTable
from .contracts import ContractError, Quantity


SCHEMA_VERSION = "canonical_cell_asset_v1"
ASSET_KINDS = ("instance_archive", "canonical_atlas")


def _freeze_json(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value):
    if hasattr(value, "items"):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _validate_header(record):
    if record.get("schema_version") != SCHEMA_VERSION or record.get("asset_kind") not in ASSET_KINDS:
        raise ContractError("Unsupported canonical asset schema/kind")
    if not isinstance(record.get("asset_id"), str) or not record["asset_id"].strip():
        raise ContractError("Canonical asset requires an explicit asset_id")
    for name in ("source", "provenance", "boundary"):
        if not isinstance(record.get(name), dict):
            raise ContractError(f"Canonical asset requires {name} JSON metadata")


def _table_mapping(tables):
    if isinstance(tables, CanonicalCellTable):
        tables = (tables,)
    elif hasattr(tables, "items"):
        if any(not isinstance(table, CanonicalCellTable) or key != table.population
               for key, table in tables.items()):
            raise ContractError("Table mapping keys must equal population names")
        tables = tuple(tables.values())
    else:
        tables = tuple(tables)
    if not tables or any(not isinstance(table, CanonicalCellTable) for table in tables):
        raise ContractError("Asset requires supplied canonical tables, including legal empty tables")
    names = [table.population for table in tables]
    if len(set(names)) != len(names):
        raise ContractError("Asset population names must be unique")
    references = {(table.atlas_id, table.atlas_version, table.reference_id) for table in tables}
    if len(references) != 1:
        raise ContractError("Asset tables must share one atlas version and global reference")
    return {table.population: table for table in tables}


@dataclass(frozen=True)
class AtlasAsset:
    asset_id: str
    asset_kind: str
    tables: object
    source: object
    provenance: object
    boundary: object
    file_sha256: str

    def metadata_record(self):
        return {"schema_version": SCHEMA_VERSION, "asset_id": self.asset_id,
                "asset_kind": self.asset_kind, "file_sha256": self.file_sha256,
                **{name: _thaw_json(getattr(self, name))
                   for name in ("source", "provenance", "boundary")}}


def save_atlas_asset(path, tables, *, asset_id, asset_kind, source, provenance, boundary):
    """Write a new explicit snapshot, refusing to replace an existing asset.

    ``instance_archive`` preserves a fixed local build. It does not establish a
    globally generated density mosaic or source support outside that build.
    Metadata is retained as supplied, not promoted to verified physiology.
    """
    table_map = _table_mapping(tables)
    record = {"schema_version": SCHEMA_VERSION, "asset_id": asset_id,
              "asset_kind": asset_kind, "source": source, "provenance": provenance,
              "boundary": boundary,
              "tables": [table_map[name].record() for name in sorted(table_map)]}
    _validate_header(record)
    try:
        raw = (json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ContractError("Asset metadata must be finite JSON data") from error
    try:
        with Path(path).open("xb") as stream:
            stream.write(raw)
    except FileExistsError as error:
        raise ContractError("Immutable atlas asset already exists; overwrite refused") from error
    return hashlib.sha256(raw).hexdigest()


def _read_table(record):
    required = {"atlas_id", "atlas_version", "source_version", "population", "reference_id",
                "source_keys", "global_cell_ids", "coordinates", "table_hash"}
    if not isinstance(record, dict) or set(record) != required:
        raise ContractError("Canonical table record fields differ from the frozen schema")
    if not isinstance(record["source_keys"], list) or not isinstance(record["global_cell_ids"], list):
        raise ContractError("Canonical keys and stored global IDs require ordered arrays")
    if not isinstance(record["coordinates"], dict):
        raise ContractError("Canonical table coordinates must retain typed fields")
    coordinates = {}
    for role, item in record["coordinates"].items():
        if not isinstance(item, dict) or set(item) != {"values", "unit", "meaning", "frame", "evidence_tag"}:
            raise ContractError("Canonical coordinate fields including frame must be explicit")
        try:
            values = np.asarray(item["values"], dtype=float)
            # JSON [] has no second axis; an empty table still has explicit 2-D positions.
            if not record["source_keys"] and item["values"] == []:
                values = np.empty((0, 2))
            coordinates[role] = Quantity(values, item["unit"], item["meaning"],
                                         item["evidence_tag"], item["frame"])
        except (TypeError, ValueError, OverflowError) as error:
            raise ContractError(f"Invalid canonical coordinate {role}: {error}") from error
    table = CanonicalCellTable(record["atlas_id"], record["atlas_version"], record["source_version"],
                               record["population"], tuple(record["source_keys"]), coordinates,
                               record["reference_id"])
    if tuple(record["global_cell_ids"]) != table.global_cell_ids:
        raise ContractError("Stored global IDs differ from canonical source identities")
    if record["table_hash"] != table.table_hash:
        raise ContractError("Canonical table_hash differs from supplied coordinates/metadata")
    return table


def load_atlas_asset(path, *, expected_sha256):
    """Verify the frozen file identity and each reconstructed canonical table."""
    if not isinstance(expected_sha256, str) or re.fullmatch("[0-9a-f]{64}", expected_sha256) is None:
        raise ContractError("A frozen lowercase SHA256 is required to load an atlas asset")
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ContractError("Atlas asset SHA256 differs from frozen expected identity")
    try:
        record = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        json.dumps(record, allow_nan=False)
    except (UnicodeDecodeError, ValueError) as error:
        raise ContractError("Atlas asset must contain finite UTF-8 JSON") from error
    if not isinstance(record, dict):
        raise ContractError("Atlas asset must be a JSON object")
    _validate_header(record)
    if not isinstance(record.get("tables"), list):
        raise ContractError("Atlas asset requires canonical table records")
    table_map = _table_mapping([_read_table(table) for table in record["tables"]])
    return AtlasAsset(record["asset_id"], record["asset_kind"], MappingProxyType(table_map),
                      *(_freeze_json(record[name]) for name in ("source", "provenance", "boundary")),
                      digest)
