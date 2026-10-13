"""Existing effective inner geometry on supplied synthetic atlas coordinates."""
from pathlib import Path

import numpy as np
import pytest

from retina_sim.atlas import CanonicalCellTable
from retina_sim.atlas_inner import build_effective_inner
from retina_sim.contracts import ContractError, Quantity
from retina_sim.ipl_geometry import RESEARCH, read_ipl_geometry


ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION = read_ipl_geometry(ROOT, RESEARCH)
REFERENCE = "left_eye:foveal_origin:nasal_positive_x:superior_positive_y"


def cones(rf=None, pedicle=None, keys=None, extra_coordinates=None):
    rf = np.asarray([[.03, 0], [.04, 0], [.045, 0]] if rf is None else rf, dtype=float).reshape(-1, 2)
    pedicle = np.asarray([[.4, 0], [.401, 0], [.402, 0]] if pedicle is None else pedicle, dtype=float).reshape(-1, 2)
    coordinates = {"rf": Quantity(rf, "mm", "functional_position", "synthetic_fixture", "functional_visual_field"),
        "opl_pedicle": Quantity(pedicle, "mm", "pedicle_position", "synthetic_fixture", "OPL")}
    coordinates.update(extra_coordinates or {})
    return CanonicalCellTable("fixture", "v1", "source_v1", "cones",
        tuple(f"row:{i}" for i in range(len(rf))) if keys is None else tuple(keys), coordinates, REFERENCE)


def build(master=None, **options):
    master = cones() if master is None else master
    return build_effective_inner(master, configuration=CONFIGURATION,
        temporal_equivalent_mm=options.pop("temporal_equivalent_mm", np.full(len(master.source_keys), .1)),
        temporal_equivalence_source_id=options.pop("temporal_equivalence_source_id", "named_fixture_temporal_equivalence"),
        **options)


def test_known_lm_private_ancestry_and_typed_ipl_keep_new_pedicle_without_soma():
    master = cones(extra_coordinates={"soma": Quantity([[.9, 0], [.91, 0], [.92, 0]],
        "mm", "anatomical_position", "synthetic_fixture", "anatomical_retina")})
    layout = build(master, cone_subtypes=("L", "M", "L"))
    for polarity in ("ON", "OFF"):
        bc_name, rgc_name = "midget_BC_" + polarity, "midget_RGC_" + polarity
        bc, rgc = layout.tables[bc_name], layout.tables[rgc_name]
        assert layout.parent_ids[bc_name] == master.global_cell_ids
        assert layout.parent_ids[rgc_name] == bc.global_cell_ids
        assert "soma" not in rgc.coordinates and "soma" not in bc.coordinates
        np.testing.assert_array_equal(bc.coordinate("ipl_preterminal").values, master.coordinate("opl_pedicle").values)
        np.testing.assert_array_equal(rgc.coordinate("ipl_post_arbor").values, bc.coordinate("ipl_preterminal").values)
        bc_contact = layout.contacts["cones->" + bc_name]
        assert bc_contact["frame"] == "OPL"
        assert bc_contact["post_positions"].meaning == "dendritic_position"
        assert bc_contact["post_positions"].coordinate_frame == "OPL"
        contact = layout.contacts[bc_name + "->" + rgc_name]
        assert contact["frame"] == "IPL"
        np.testing.assert_array_equal(contact["matrix"].toarray(), np.eye(3))
    assert layout.tables["S_ON_BC"].coordinate("rf").values.shape == (0, 2)
    assert layout.metadata["global_coverage_claim"] is False


def test_unknown_subtypes_are_structural_only_and_block_s_pooling():
    master = cones()
    layout = build(master, cone_subtypes=("L", None, "S"))
    bc = layout.tables["midget_BC_ON"]
    assert layout.parent_ids["midget_BC_ON"] == master.global_cell_ids[:2]
    direct = layout.contacts["cones->midget_BC_ON"]
    assert direct["row_status"][1] == "BLOCKED_UNKNOWN_CONE_SUBTYPE"
    assert direct["matrix"].getrow(1).nnz == 0
    rgc = layout.contacts["midget_BC_ON->midget_RGC_ON"]
    assert rgc["matrix"].getrow(1).nnz == 0
    s = layout.contacts["S_ON_BC->small_bistratified_S_ON"]
    assert s["row_status"] == ("BLOCKED_UNKNOWN_S_PARTITION_IN_SUPPORT",)
    assert s["matrix"].nnz == 0
    assert len(bc.source_keys) == 2
    no_labels = build(master)
    assert no_labels.contacts["cones->midget_BC_ON"]["matrix"].nnz == 0


def test_effective_s_relay_is_minimum_one_and_local_ac_id_is_independent():
    layout = build(cone_subtypes=("L", "M", "S"))
    contact = layout.contacts["S_ON_BC->small_bistratified_S_ON"]
    assert contact["matrix"].shape == (1, 1) and contact["matrix"][0, 0] == 1
    assert contact["rows"][0]["legacy_selection_budget"] == 8
    assert contact["rows"][0]["minimum_selected_inputs"] == 1
    a, b = layout.tables["local_feedforward_AC"], layout.tables["local_crossover_AC"]
    assert not set(a.global_cell_ids) & set(b.global_cell_ids)
    np.testing.assert_array_equal(a.coordinate("ipl_preterminal").values, b.coordinate("ipl_preterminal").values)
    assert "soma" not in a.coordinates
    assert all("rod" not in key for key in a.source_keys)


def test_wool_scope_and_finite_support_are_enforced_without_nearest_filling():
    master = cones(pedicle=[[.4, 0], [.405, 0], [.6, 0]])
    layout = build(master, cone_subtypes=("L", "M", "L"), temporal_equivalent_mm=[2, 2, 10.1])
    contact = layout.contacts["midget_BC_ON->midget_RGC_ON"]
    assert contact["row_status"][0] == "INCOMPLETE_MIDGET_CONVERGENCE_FINITE_SUPPORT"
    assert contact["row_status"][2] == "BLOCKED_WOOL_TEMPORAL_EQUIVALENCE_MISSING_OR_UNSUPPORTED"
    assert contact["matrix"].nnz == 0
    for row in contact["rows"][:2]:
        assert row["hard_support_mm"] <= .04
        assert row["maximum_distance_mm"] <= row["hard_support_mm"]
    missing = build(master, cone_subtypes=("L", "M", "L"), temporal_equivalent_mm=None)
    assert missing.contacts["midget_BC_ON->midget_RGC_ON"]["matrix"].nnz == 0
    with pytest.raises(ContractError, match="named source/project"):
        build(master, cone_subtypes=("L", "M", "L"), temporal_equivalence_source_id=None)
    dense = cones(rf=np.c_[np.linspace(.03, .05, 10), np.zeros(10)],
                  pedicle=np.c_[np.linspace(.4, .403, 10), np.zeros(10)])
    complete = build(dense, cone_subtypes=("L",) * 10, temporal_equivalent_mm=np.full(10, 2))
    ready = complete.contacts["midget_BC_ON->midget_RGC_ON"]
    assert set(ready["row_status"]) == {"READY_PROJECT_WOOL_CONVERGENCE"}
    np.testing.assert_allclose(np.asarray(ready["matrix"].sum(axis=1)).ravel(), 1)
    assert np.all(np.diff(ready["matrix"].indptr) == 3)


def test_canonical_subset_preserves_parent_child_and_ac_identity():
    master = cones()
    full = build(master, cone_subtypes=("L", "M", "L"))
    partial = cones(rf=master.coordinate("rf").values[1:],
        pedicle=master.coordinate("opl_pedicle").values[1:], keys=master.source_keys[1:])
    subset = build(partial, cone_subtypes=("M", "L"))
    for population in ("midget_BC_ON", "midget_RGC_ON", "local_feedforward_AC"):
        lookup = dict(zip(full.tables[population].source_keys, full.tables[population].global_cell_ids))
        for key, identity in zip(subset.tables[population].source_keys, subset.tables[population].global_cell_ids):
            assert lookup[key] == identity


def test_coordinate_and_subtype_contract_and_empty_population():
    empty = cones(rf=np.empty((0, 2)), pedicle=np.empty((0, 2)))
    assert build(empty).tables["midget_RGC_ON"].coordinate("ipl_post_arbor").values.shape == (0, 2)
    with pytest.raises(ContractError, match="subtype array"):
        build(cone_subtypes=("L",))
    with pytest.raises(ContractError, match="no implicit non-S"):
        build(cone_subtypes=("L", "M", "?"))
    bad = dict(CONFIGURATION, hard_support_mm=.08)
    with pytest.raises(ContractError, match="40um"):
        build_effective_inner(cones(), configuration=bad)
    master = cones()
    only_rf = CanonicalCellTable(master.atlas_id, master.atlas_version, master.source_version,
        master.population, master.source_keys, {"rf": master.coordinate("rf")}, master.reference_id)
    with pytest.raises(ContractError, match="fallback forbidden"):
        build(only_rf)
    beyond = build(cones(pedicle=[[10.1, 0], [10.2, 0], [10.3, 0]]), cone_subtypes=("L", "M", "L"))
    assert set(beyond.contacts["cones->midget_BC_ON"]["row_status"]) == {"BLOCKED_MIDGET_OPL_ABOVE_10MM"}
    record = build(cone_subtypes=("L", "M", "S")).record()
    assert record["contacts"]["midget_BC_ON->midget_RGC_ON"]["edges"][0]["unit"] == "dimensionless"


def test_explicit_diffuse_seeds_connect_local_ac_and_midget_inhibition_with_old_budgets():
    master = cones(rf=np.c_[np.linspace(.03, .045, 20), np.zeros(20)],
                  pedicle=np.c_[np.linspace(.4, .409, 20), np.zeros(20)])
    layout = build(master, cone_subtypes=("L",) * 20,
                   diffuse_seed_cone_ids=master.global_cell_ids, include_widefield_ac=True)
    for route, budget in (("cones->diffuse_BC_ON", 8),
                          ("diffuse_BC_ON->local_feedforward_AC", 4),
                          ("diffuse_BC_OFF->local_crossover_AC", 4),
                          ("local_crossover_AC->midget_RGC_ON", 1),
                          ("local_feedforward_AC->midget_RGC_OFF", 1)):
        contact = layout.contacts[route]
        assert set(contact["row_status"]) == {"READY_PROJECT_FINITE_POOL"}
        assert np.all(np.diff(contact["matrix"].indptr) == budget)
        np.testing.assert_allclose(np.asarray(contact["matrix"].sum(axis=1)).ravel(), 1)
        assert all(row["hard_support_mm"] == .04 and row["maximum_distance_mm"] <= .04 for row in contact["rows"])
    assert "diffuse_BC_ON" not in layout.parent_ids
    assert layout.metadata["diffuse_sampling_seed_cone_ids"] == sorted(master.global_cell_ids)
    assert set(layout.tables["widefield_AC"].coordinates) == {"ipl_preterminal", "ipl_post_arbor"}
    wide_contact = layout.contacts["diffuse_BC_ON->widefield_AC"]
    assert all(row["selection_budget"] == 16 for row in wide_contact["rows"])
    assert any(status.startswith("READY") for status in wide_contact["row_status"])
    assert not any(blocker.get("reason") == "EFFECTIVE_DIFFUSE_BC_DRIVE_NOT_SUPPLIED" for blocker in layout.blockers)
    assert any(blocker.get("reason") == "INDEPENDENT_CANONICAL_EFFECTIVE_RGC_SEEDS_NOT_SUPPLIED" for blocker in layout.blockers)


def test_unknown_diffuse_pool_blocks_upstream_chain_and_rejects_noncanonical_seeds():
    master = cones(rf=np.c_[np.linspace(.03, .045, 9), np.zeros(9)],
                  pedicle=np.c_[np.linspace(.4, .408, 9), np.zeros(9)])
    layout = build(master, cone_subtypes=("L",) * 8 + (None,), diffuse_seed_cone_ids=master.global_cell_ids)
    for route in ("cones->diffuse_BC_ON", "diffuse_BC_ON->local_feedforward_AC",
                  "local_feedforward_AC->midget_RGC_OFF"):
        assert layout.contacts[route]["matrix"].nnz == 0
        assert all(status.startswith("BLOCKED") for status in layout.contacts[route]["row_status"])
    assert layout.contacts["cones->diffuse_BC_ON"]["rows"][0]["selected_candidate_pre_ids"]
    with pytest.raises(ContractError, match="canonical cone IDs"):
        build(master, cone_subtypes=("L",) * 9, diffuse_seed_cone_ids=("not_supplied",))
    with pytest.raises(ContractError, match="unique"):
        build(master, cone_subtypes=("L",) * 9, diffuse_seed_cone_ids=(master.global_cell_ids[0],) * 2)


def test_partial_finite_effective_pool_is_retained_and_distant_drive_never_fills():
    master = cones(pedicle=[[.4, 0], [.5, 0], [.6, 0]])
    layout = build(master, cone_subtypes=("L", "M", "L"), diffuse_seed_cone_ids=(master.global_cell_ids[0],))
    direct = layout.contacts["cones->diffuse_BC_ON"]
    assert direct["matrix"].nnz == 1 and direct["matrix"].sum() == 1
    assert direct["rows"][0]["missing_degree"] == 7
    assert direct["rows"][0]["exact_budget_enforcement"] is False
    local = layout.contacts["diffuse_BC_ON->local_feedforward_AC"]
    for j, position in enumerate(local["post_positions"].values):
        if position[0] > .44:
            assert local["matrix"].getrow(j).nnz == 0
            assert local["row_status"][j] == "INCOMPLETE_NO_INPUT_WITHIN_40UM"
    dense = cones(rf=np.c_[np.linspace(.03, .04, 8), np.zeros(8)],
                  pedicle=np.c_[np.linspace(.4, .407, 8), np.zeros(8)])
    full = build(dense, cone_subtypes=("L",) * 8, include_widefield_ac=True)
    subset = cones(rf=dense.coordinate("rf").values[3:],
        pedicle=dense.coordinate("opl_pedicle").values[3:], keys=dense.source_keys[3:])
    small = build(subset, cone_subtypes=("L",) * 5, include_widefield_ac=True)
    lookup = dict(zip(full.tables["widefield_AC"].global_cell_ids,
                      full.tables["widefield_AC"].coordinate("ipl_preterminal").values))
    for cell, point in zip(small.tables["widefield_AC"].global_cell_ids,
                           small.tables["widefield_AC"].coordinate("ipl_preterminal").values):
        np.testing.assert_array_equal(point, lookup[cell])
