"""Supplied numerical rows test atlas identity; no animal anatomy is inferred."""
import numpy as np
import pytest

from retina_sim.atlas import (AtlasROI, CanonicalCellTable, COORDINATE_ROLES,
                              global_cell_id)
from retina_sim.contracts import ContractError, Quantity


REFERENCE = "left_eye:foveal_origin:nasal_positive_x:superior_positive_y"


def table(population="cone", points=None, keys=None, coordinates=None):
    points = np.array([[0, 0], [.1, 0], [.2, 0], [.3, 0], [.4, 0]]) if points is None else np.asarray(points)
    keys = tuple(f"source-row:{i}" for i in range(len(points))) if keys is None else keys
    if coordinates is None:
        coordinates = {role: Quantity(points + [i * .05, i * .02], "mm", meaning,
                                       "synthetic_fixture", frame)
                       for i, (role, (meaning, frame)) in enumerate(COORDINATE_ROLES.items())}
    return CanonicalCellTable("test_atlas", "v1", "source_v1", population,
                              keys, coordinates, REFERENCE)


def roi(owner, query=None, **options):
    return AtlasROI(owner, owner if query is None else query, REFERENCE, **options)


def test_overlap_and_tile_order_reference_identical_canonical_rows():
    master = table()
    left_roi = roi((-.05, .2, -.1, .1), (-.1, .35, -.1, .1))
    right_roi = roi((.2, .45, -.1, .1), (.05, .5, -.1, .1))
    left, right = master.query(left_roi), master.query(right_roi)
    repeated_right, repeated_left = master.query(right_roi), master.query(left_roi)
    assert left.table is right.table is master
    assert repeated_left.global_cell_ids == left.global_cell_ids
    assert repeated_right.global_cell_ids == right.global_cell_ids
    owner_ids = [cell for view in (left, right)
                 for cell, owner in zip(view.global_cell_ids, view.owner) if owner]
    assert len(owner_ids) == len(set(owner_ids)) == 5
    common = set(left.global_cell_ids) & set(right.global_cell_ids)
    for role in COORDINATE_ROLES:
        a = dict(zip(left.global_cell_ids, left.coordinate(role).values))
        b = dict(zip(right.global_cell_ids, right.coordinate(role).values))
        for cell in common:
            np.testing.assert_array_equal(a[cell], b[cell])
    assert not np.array_equal(master.coordinate("rf").values,
                              master.coordinate("opl_pedicle").values)


def test_ids_depend_on_source_key_and_versions_not_row_order_or_position():
    master = table()
    shuffled = table(points=master.coordinate("rf").values[::-1], keys=master.source_keys[::-1])
    assert shuffled.global_cell_ids == master.global_cell_ids[::-1]
    moved = table(points=master.coordinate("rf").values + [.01, 0], keys=master.source_keys)
    assert moved.global_cell_ids == master.global_cell_ids
    assert moved.table_hash != master.table_hash
    assert global_cell_id("test_atlas", "v1", "source_v2", "cone", "source-row:0") != master.global_cell_ids[0]
    assert global_cell_id("test_atlas", "v2", "source_v1", "cone", "source-row:0") != master.global_cell_ids[0]
    assert table(population="rod").global_cell_ids[0] != master.global_cell_ids[0]


def test_canonical_arrays_mapping_and_query_indices_are_immutable():
    incoming = Quantity([[0, 0]], "mm", "functional_position", "synthetic_fixture", "functional_visual_field")
    master = table(points=[[0, 0]], coordinates={"rf": incoming})
    incoming.values.setflags(write=True)
    incoming.values[:] = 10
    np.testing.assert_array_equal(master.coordinate("rf").values, [[0, 0]])
    with pytest.raises(ValueError):
        master.coordinate("rf").values.setflags(write=True)
    with pytest.raises(TypeError):
        master.coordinates["soma"] = incoming
    selected = master.query(roi((-.1, .1, -.1, .1)))
    with pytest.raises(ValueError):
        selected.row_indices.setflags(write=True)
    with pytest.raises(ValueError):
        selected.owner[0] = False


def test_empty_masked_rod_and_s_are_legal_and_not_empty_sample_claims():
    small = roi((-.01, .01, -.01, .01))
    for population in ("rod", "S_cone"):
        empty = table(population=population, points=np.empty((0, 2)))
        sample = empty.query(small)
        mask = empty.query(small, biological_absence_source_id="source_defined_central_mask")
        assert sample.owner_status == "EMPTY_SAMPLE"
        assert mask.owner_status == "BIOLOGICAL_ABSENCE"
        assert mask.global_cell_ids == () and mask.coordinate("rf").values.shape == (0, 2)
    halo = table(population="rod", points=[[.025, 0]]).query(
        roi((-.01, .01, -.01, .01), (-.04, .04, -.04, .04)),
        biological_absence_source_id="source_defined_central_mask")
    assert halo.owner_status == "BIOLOGICAL_ABSENCE" and len(halo.row_indices) == 1
    assert not np.any(halo.owner)
    with pytest.raises(ContractError, match="conflicts"):
        table().query(small, biological_absence_source_id="source_defined_central_mask")


@pytest.mark.parametrize("unit,meaning,frame", [
    ("deg", "functional_position", "functional_visual_field"),
    ("mm", "anatomical_position", "functional_visual_field"),
    ("mm", "functional_position", "anatomical_retina"),
])
def test_coordinate_unit_meaning_and_frame_are_strict(unit, meaning, frame):
    with pytest.raises(ContractError):
        table(points=[[0, 0]], coordinates={"rf": Quantity([[0, 0]], unit, meaning, "synthetic_fixture", frame)})


def test_shape_keys_reference_and_missing_role_are_rejected():
    with pytest.raises(ContractError, match="unique"):
        table(keys=("same",) * 5)
    with pytest.raises(ContractError, match="Shape"):
        table(coordinates={"rf": Quantity([[0, 0]], "mm", "functional_position", "synthetic_fixture", "functional_visual_field")})
    with pytest.raises(ContractError, match="Nonfinite"):
        table(points=[[np.nan, 0]])
    with pytest.raises(ContractError, match="explicit coordinate role"):
        roi((0, 1, 0, 1), unit="um")
    with pytest.raises(ContractError, match="within"):
        roi((0, 1, 0, 1), (.1, .9, 0, 1))
    master = table()
    with pytest.raises(ContractError, match="global reference"):
        master.query(AtlasROI((0, 1, 0, 1), (0, 1, 0, 1), "other_eye"))
    only_rf = table(coordinates={"rf": master.coordinate("rf")})
    with pytest.raises(ContractError, match="fallback forbidden"):
        only_rf.query(roi((0, 1, 0, 1), coordinate_role="soma"))
