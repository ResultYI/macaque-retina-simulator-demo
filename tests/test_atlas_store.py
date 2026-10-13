"""Frozen source-key snapshots are archives, not density-generation tests."""
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from retina_sim.atlas import AtlasROI, CanonicalCellTable
from retina_sim.atlas_store import load_atlas_asset, save_atlas_asset
from retina_sim.contracts import ContractError, Quantity


REFERENCE = "left_eye:foveal_origin:nasal_positive_x:superior_positive_y"


def tables():
    return [CanonicalCellTable("fixture", "v1", "source_v1", population,
        tuple(f"canonical:{i}" for i in range(len(points))),
        {"rf": Quantity(points, "mm", "functional_position", "synthetic_fixture", "functional_visual_field"),
         "opl_pedicle": Quantity(np.asarray(points) + [.1, 0], "mm", "pedicle_position", "synthetic_fixture", "OPL")},
        REFERENCE) for population, points in (
            ("cone", np.array([[.1, 0], [.2, 0], [.3, 0]])), ("rod", np.empty((0, 2))))]


def save(path):
    return save_atlas_asset(path, tables(), asset_id="supplied_local_fixture",
                           asset_kind="instance_archive",
                           source={"source_id": "software_fixture"},
                           provenance={"status": "project_approximation", "notes": ["not animal data"]},
                           boundary={"mode": "finite_open", "global_coverage_claim": False})


def rewrite(path, record):
    raw = json.dumps(record, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_roundtrip_preserves_overlap_id_positions_empty_tables_and_metadata(tmp_path):
    path = tmp_path / "snapshot.json"
    digest = save(path)
    loaded = load_atlas_asset(path, expected_sha256=digest)
    assert loaded.asset_kind == "instance_archive"
    assert loaded.file_sha256 == digest
    assert loaded.boundary["global_coverage_claim"] is False
    assert loaded.metadata_record()["provenance"]["notes"] == ["not animal data"]
    assert loaded.tables["rod"].coordinate("rf").values.shape == (0, 2)
    original = tables()[0]
    for owner, query in (((0, .25, -.1, .1), (0, .4, -.1, .1)),
                         ((.2, .4, -.1, .1), (.05, .4, -.1, .1))):
        roi = AtlasROI(owner, query, REFERENCE)
        a, b = original.query(roi), loaded.tables["cone"].query(roi)
        assert a.global_cell_ids == b.global_cell_ids
        assert a.source_keys == b.source_keys
        np.testing.assert_array_equal(a.owner, b.owner)
        np.testing.assert_array_equal(a.coordinate("opl_pedicle").values,
                                      b.coordinate("opl_pedicle").values)
    with pytest.raises(TypeError):
        loaded.tables["cone"] = original
    with pytest.raises(TypeError):
        loaded.boundary["mode"] = "closed"


def test_existing_asset_cannot_be_overwritten_and_file_tampering_is_rejected(tmp_path):
    path = tmp_path / "snapshot.json"
    digest = save(path)
    with pytest.raises(ContractError, match="overwrite refused"):
        save(path)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ContractError, match="SHA256"):
        load_atlas_asset(path, expected_sha256=digest)


@pytest.mark.parametrize("changed_field", ["position", "global_id", "frame", "table_hash"])
def test_table_validation_even_with_a_new_file_digest(tmp_path, changed_field):
    path = tmp_path / "snapshot.json"
    save(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    cone = next(table for table in record["tables"] if table["population"] == "cone")
    if changed_field == "position":
        cone["coordinates"]["rf"]["values"][0][0] = .11
    elif changed_field == "global_id":
        cone["global_cell_ids"][0] = "fabricated"
    elif changed_field == "frame":
        del cone["coordinates"]["opl_pedicle"]["frame"]
    else:
        cone["table_hash"] = "0" * 64
    digest = rewrite(path, record)
    with pytest.raises(ContractError):
        load_atlas_asset(path, expected_sha256=digest)


def test_shared_reference_and_explicit_snapshot_metadata_are_required(tmp_path):
    rows = tables()
    with pytest.raises(ContractError, match="population names"):
        save_atlas_asset(tmp_path / "duplicates.json", [rows[0], rows[0]], asset_id="x",
                         asset_kind="instance_archive", source={}, provenance={}, boundary={})
    with pytest.raises(ContractError, match="schema/kind"):
        save_atlas_asset(tmp_path / "kind.json", rows, asset_id="x",
                         asset_kind="whole_retina_validated", source={}, provenance={}, boundary={})
    with pytest.raises(ContractError, match="global reference"):
        save_atlas_asset(tmp_path / "reference.json", [rows[0], replace(rows[1], reference_id="other_eye")],
                         asset_id="x", asset_kind="instance_archive", source={}, provenance={}, boundary={})
    with pytest.raises(ContractError, match="frozen lowercase SHA256"):
        load_atlas_asset(tmp_path / "missing.json", expected_sha256=None)
