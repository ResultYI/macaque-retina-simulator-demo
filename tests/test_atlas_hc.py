from pathlib import Path
import math

import numpy as np
import pytest

from retina_sim.atlas_hc import AtlasHCProvider, make_hc_soma_density
from retina_sim.contracts import ContractError, Quantity


ROOT = Path(__file__).resolve().parents[1]


def anchors(xy, *, frame="OPL", unit="mm"):
    return Quantity(xy, unit, "anatomical_position", "synthetic_fixture", frame)


def test_default_keeps_central_reference_and_reports_unselected_candidates():
    result = AtlasHCProvider(ROOT).footprints(("inward", "central", "outer"), ("H1",) * 3,
                                             anchors([[.1, 0], [.75, 0], [3, 0]]))
    assert list(result["footprints"]["by_hc_id"]) == ["central"]
    assert result["footprints"]["by_hc_id"]["central"]["radius_mm"] == .00804
    assert [row["reason"] for row in result["blocked_records"]] == [
        "CENTRAL_INWARD_CANDIDATE_NOT_SELECTED", "PERIPHERAL_INTERPOLATION_NOT_SELECTED"]
    assert not result["manifest"]["geometry_ready"]


def test_central_inward_candidate_is_explicit_low_evidence_preview_only():
    provider = AtlasHCProvider(ROOT)
    with pytest.raises(ContractError):
        provider.footprints(("a",), ("H2",), anchors([[.1, 0]]), enable_central_candidate=True)
    result = provider.footprints(("a",), ("H2",), anchors([[.1, 0]]),
                                 mode="connection_preview", enable_central_candidate=True)
    row = result["per_hc_coverage"][0]
    assert row["coverage_status"] == "project_approximation"
    assert row["evidence_status"] == "EXTRAPOLATED_UNCALIBRATED"
    assert row["low_evidence"] and row["radius_mm"] == .01222
    assert not result["manifest"]["inward_candidate_completes_coverage"]


def test_peripheral_band_keeps_source_interval_and_bounds_interpolated_radii():
    provider = AtlasHCProvider(ROOT)
    result = provider.footprints(("interpolated", "peripheral", "outside"), ("H2",) * 3,
        anchors([[3, 0], [-5.4, 0], [6.00001, 0]]), mode="connection_preview")
    expected = math.sqrt(19 / (math.pi * 6000))
    assert result["footprints"]["by_hc_id"]["peripheral"]["radius_mm"] == pytest.approx(expected)
    assert 0 < result["footprints"]["by_hc_id"]["interpolated"]["radius_mm"] <= expected
    assert result["blocked_records"][0]["reason"] == "HC_FOOTPRINT_OUTSIDE_BOUNDED_ANCHOR_APPLICATION"
    source = result["manifest"]["source_domains"][1]
    assert source["radial_interval"] == [5, 6] and source["meridian"] == "temporal"
    assert source["coordinate_frame"] == "anatomical_retina"
    assert result["manifest"]["finite_anchor_curves"][0]["radial_knots_mm"] == [1, 5, 6]
    assert not result["manifest"]["per_cone_integer_divergence_imposed"]


def test_actual_density_is_queried_at_supported_anchor_nodes_not_each_query_point():
    calls = []
    def density(q):
        q.require("mm", "anatomical_position", frame="OPL")
        calls.append(q.values.copy())
        return Quantity(np.full(len(q.values), 10000), "cells/mm^2", "population_density", "synthetic_fixture", "OPL")
    result = AtlasHCProvider(ROOT).footprints(("a",), ("H1",), anchors([[2, 2]]),
        pedicle_density=density, mode="connection_preview")
    assert result["manifest"]["geometry_ready"]
    assert len(calls) == 2
    assert all(np.allclose(np.linalg.norm(points, axis=1), [5, 6]) for points in calls)
    row = result["per_hc_coverage"][0]
    assert not row["radius_is_inverse_density_at_query"]
    assert row["radius_mm"] <= row["maximum_anchor_radius_mm"]


@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_density_blocks_instead_of_inverse_expansion_or_fallback(value):
    def density(q):
        return Quantity(np.full(len(q.values), value), "cones/mm^2", "population_density", "synthetic_fixture", "OPL")
    result = AtlasHCProvider(ROOT).footprints(("a",), ("H1",), anchors([[2, 0]]),
        pedicle_density=density, mode="connection_preview")
    assert not result["footprints"]["by_hc_id"]
    blocked = result["blocked_records"][0]
    assert blocked["coverage_status"] == "unsupported"
    assert blocked["anchor_failures"][0]["error"] == "PEDICLE_DENSITY_NONPOSITIVE_AT_ANCHOR"
    assert blocked["coordinate_frame"] == "OPL" and blocked["unit"] == "mm"


def test_missing_or_untyped_actual_density_never_falls_back_to_source6000():
    provider = AtlasHCProvider(ROOT)
    for callback in (None, lambda q: np.full(len(q.values), 6000)):
        result = provider.footprints(("a",), ("H2",), anchors([[2, 0]]), mode="connection_preview",
                                     density_mode="actual_local_pedicle_density", pedicle_density=callback)
        assert result["blocked_records"][0]["reason"] == "NO_SUPPORTED_FINITE_EQUAL_AREA_ANCHOR"
        assert result["manifest"]["maximum_generated_envelope_mm"] is None


def test_direct_branch_morphology_stays_nodes_and_finite_contact_threshold():
    morphology = {"kind": "morphology_nodes", "nodes_mm": [[0, 0], [.002, 0]],
        "node_coordinates": "relative_to_anchor", "contact_threshold_mm": .0003,
        "unit": "mm", "coordinate_frame": "OPL", "application_anchor_band_mm": [.5, 1],
        "source_profile_id": "synthetic_branch_fixture", "evidence_class": "project_approximation",
        "approximation_id": "synthetic_branch_fixture"}
    result = AtlasHCProvider(ROOT).footprints(("a",), ("H1",), anchors([[.75, 0]]),
        morphology_by_hc_id={"a": morphology})
    shape = result["footprints"]["by_hc_id"]["a"]
    assert shape["kind"] == "morphology_nodes" and shape["nodes_mm"] == morphology["nodes_mm"]
    assert shape["contact_threshold_mm"] == .0003
    assert shape["allowed_extension"] is None
    assert result["per_hc_coverage"][0]["representation"] == "branch_morphology"


@pytest.mark.parametrize("q", [anchors([[.75, 0]], frame="functional_visual_field"), anchors([[.75, 0]], unit="deg")])
def test_rf_or_degrees_cannot_be_misread_as_opl_anchor_mm(q):
    with pytest.raises(ContractError):
        AtlasHCProvider(ROOT).footprints(("a",), ("H1",), q)


def test_reused_joint_density_preserves_measurement_intervals_and_project_field():
    density = make_hc_soma_density(ROOT)
    q = anchors([[0, 0], [.6, 0], [3, 0]], frame="anatomical_retina")
    assert np.all(density(q) > 0)
    assert np.allclose(density.type_density(q, "H1") + density.type_density(q, "H2"), density(q))
    assert density.metadata["evidence_class"] == "project_approximation"
    rows = density.metadata["verified_Table2_original_rows"]
    assert len(rows) == 12
    interval = [row for row in rows if row["patch"] == 1]
    assert all(row["eccentricity_mm"] is None for row in interval)
    assert all((row["eccentricity_min_mm"], row["eccentricity_max_mm"]) == (2, 3) for row in interval)
    assert not density.metadata["physiological_validation"]
