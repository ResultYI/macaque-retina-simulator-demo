"""Small geometry queries exercise actual frozen rod data, never dynamics."""
from pathlib import Path

import numpy as np
import pytest

from retina_sim.atlas import AtlasROI
from retina_sim.atlas_rod_inner import build_effective_rod_inner
from retina_sim.contracts import ContractError
from retina_sim.source_atlas import SourceDensityAtlas


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def atlas():
    return SourceDensityAtlas(ROOT)


def roi(atlas, owner=(1., 1.04, -.02, .02), query=(.96, 1.08, -.06, .06)):
    return AtlasROI(owner, query, atlas.config["reference_id"])


def test_central_rod_unsupported_keeps_independent_aii_without_fake_absence(atlas):
    result = build_effective_rod_inner(atlas, roi(atlas, (-.04, .04, -.04, .04), (-.08, .08, -.08, .08)))
    assert set(result["tables"]) == {"AII_AC"}
    assert result["tables"]["AII_AC"]["global_cell_ids"]
    assert result["metadata"]["source_status"] == "unsupported"
    assert "Packer rod curve unsupported at " in result["metadata"]["rod_source_error"]
    assert result["blockers"][0]["not_biological_absence"] is True
    assert result["metadata"]["AII_independent_of_rod_count"] is True
    assert result["metadata"]["population_ownership"]["AII_AC"]["owner_status"] == "POPULATED"
    assert result["contacts"] == {}


def test_supported_peripheral_actual_edges_keep_spherule_frames_and_original_budgets(atlas):
    result = build_effective_rod_inner(atlas, roi(atlas))
    assert set(result["tables"]) == {"rods", "rod_BC", "AII_AC"}
    assert "opl_pedicle" not in result["tables"]["rods"]["coordinates"]
    spherule = result["typed_points"]["rods"]["OPL_spherule"]
    assert spherule["frame"] == "OPL" and spherule["meaning"] == "terminal_position"
    assert result["typed_points"]["rods"]["cone_Henle_applied"] is False
    assert result["typed_points"]["rod_BC"]["OPL_dendrite"]["meaning"] == "dendritic_position"
    for route, pre, post, budget, frame in (("rods->rod_BC", "rods", "rod_BC", 40, "OPL"),
                                           ("rod_BC->AII_AC", "rod_BC", "AII_AC", 3, "IPL")):
        contact = result["contacts"][route]
        assert contact["selection_budget"] == budget and contact["frame"] == frame
        assert contact["edges"]
        pre_lookup = dict(zip(result["tables"][pre]["global_cell_ids"], contact["pre_positions"]["values"]))
        post_lookup = dict(zip(result["tables"][post]["global_cell_ids"], contact["post_positions"]["values"]))
        for edge in contact["edges"]:
            distance = np.linalg.norm(np.asarray(pre_lookup[edge["pre_id"]]) - post_lookup[edge["post_id"]])
            assert distance <= .04 + 1e-12
        assert all(row["hard_support_mm"] == .04 for row in contact["rows"])
        assert any(row["support_envelope_complete"] for row in contact["rows"])
    assert result["metadata"]["RBC_to_rod_density_ratio"] == 1 / 18
    assert result["metadata"]["RBC_density_basis_degree"] == 36
    assert result["metadata"]["rod_RBC_selection_budget"] == 40


def test_overlap_queries_keep_fixed_global_ids_positions_and_order_independence(atlas):
    first = roi(atlas)
    second = roi(atlas, (1.02, 1.06, -.02, .02), (.98, 1.10, -.06, .06))
    a = build_effective_rod_inner(atlas, first)
    b = build_effective_rod_inner(atlas, second)
    repeat = build_effective_rod_inner(atlas, first)
    for population, role in (("rods", "rf"), ("rod_BC", "soma"), ("AII_AC", "ipl_post_arbor")):
        row_a, row_b = a["tables"][population], b["tables"][population]
        assert row_a["global_cell_ids"] == repeat["tables"][population]["global_cell_ids"]
        points_a = dict(zip(row_a["global_cell_ids"], row_a["coordinates"][role]["values"]))
        points_b = dict(zip(row_b["global_cell_ids"], row_b["coordinates"][role]["values"]))
        common = points_a.keys() & points_b.keys()
        assert common
        for cell in common:
            np.testing.assert_array_equal(points_a[cell], points_b[cell])
        assert row_a["source_version"] == row_b["source_version"]


def test_resource_budget_refuses_work_without_density_thinning(atlas):
    with pytest.raises(ContractError, match="budget exceeded"):
        build_effective_rod_inner(atlas, roi(atlas), max_cells=1)
    with pytest.raises(ContractError, match="Positive integer"):
        build_effective_rod_inner(atlas, roi(atlas), max_cells=True)
    wrong = AtlasROI((1., 1.04, -.02, .02), (.96, 1.08, -.06, .06), "another_reference")
    with pytest.raises(ContractError, match="global reference"):
        build_effective_rod_inner(atlas, wrong)
