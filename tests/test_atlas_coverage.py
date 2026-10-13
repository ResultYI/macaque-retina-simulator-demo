from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from retina_sim.contracts import ContractError, Quantity
from retina_sim.coverage import CoverageCatalog, SpatialDomain, load_atlas_source_domains


ROOT = Path(__file__).resolve().parents[1]


def positions(xy, frame="functional_visual_field", unit="mm", meaning="functional_position"):
    return Quantity(xy, unit, meaning, "synthetic_fixture", frame)


def test_frozen_catalog_loads_and_stage_target_is_not_source_support():
    catalog = load_atlas_source_domains(ROOT)
    assert catalog.sha256 and catalog.data["atlas_targets"]["phase1_radius"] == 3
    result = catalog.query("HC_W2000_central_equal_area_current",
                           positions([[3, 0]], "OPL", meaning="anatomical_position"))
    assert result.status == ("unsupported",)


@pytest.mark.parametrize("interval", [("0", 3), (True, 3), (-1, 3), (0, float("inf")), (3, 2)])
def test_domain_rejects_untyped_or_nonfinite_bounds(interval):
    with pytest.raises(ContractError):
        SpatialDomain("functional_visual_field", "mm", "functional_position", interval)


def test_packer_literal_nodes_are_physical_and_discrete_not_a_measured_disk():
    catalog = load_atlas_source_domains(ROOT)
    record = catalog.records["packer_cone_nasal_tabulation"]
    r = record.source_domain[0].sampled_radii[0]
    q = positions([[r, 0], [r + 1e-8, 0], [0, r], [-r, 0]],
                  "anatomical_retina", meaning="anatomical_position")
    result = catalog.query(record.provider_id, q, domain_role="source")
    assert result.status == ("source_supported", "unsupported", "unsupported", "unsupported")
    assert result.provenance["evidence_status"] == "published_code_tabulation"
    assert "not physiological validation" in result.manifest()["coverage_meaning"]


def test_packer_rf_application_is_named_transfer_even_at_a_literal_radius():
    catalog = load_atlas_source_domains(ROOT)
    r = catalog.records["packer_cone_nasal_tabulation"].source_domain[0].sampled_radii[0]
    result = catalog.query("packer_cone_nasal_tabulation", positions([[r, 0]]))
    assert result.status == ("project_approximation",)
    assert result.source_contains == (None,)
    assert result.provenance["transfer"]["transfer_id"]


@pytest.mark.parametrize("q", [
    positions([[.75, 0]]),
    positions([[.75, 0]], "OPL", unit="deg", meaning="anatomical_position"),
    positions([[.75, 0]], "OPL", meaning="functional_position"),
    np.array([[.75, 0]]),
])
def test_coverage_rejects_wrong_frame_unit_meaning_or_untyped_array(q):
    with pytest.raises(ContractError):
        load_atlas_source_domains(ROOT).query("HC_W2000_central_equal_area_current", q)


def test_hc_anchor_application_and_source_band_are_not_query_halo_extent():
    catalog = load_atlas_source_domains(ROOT)
    radii = [.47, .5, 1, 1.04, np.nextafter(.47, 0), np.nextafter(1.04, np.inf)]
    q = positions([[r, 0] for r in radii], "OPL", meaning="anatomical_position")
    query = SpatialDomain("OPL", "mm", "anatomical_position", (0, 1.1))
    result = catalog.query("HC_W2000_central_equal_area_current", q, query_support=query)
    assert result.status == ("project_approximation",) * 4 + ("unsupported",) * 2
    assert result.source_contains == (False, True, True, False, False, False)
    assert result.query_support_contains == (True,) * 6
    assert result.source_domain[0]["radial_interval"] == (.5, 1)
    assert result.application_domain["radial_interval"] == (.47, 1.04)
    assert result.provenance["representation"] == "effective_disk"


def test_query_support_can_restrict_but_never_promote_application_coverage():
    catalog = load_atlas_source_domains(ROOT)
    query = SpatialDomain("OPL", "mm", "anatomical_position", (0, 2), bounds=(.7, .8, -.1, .1))
    result = catalog.query("HC_W2000_central_equal_area_current",
                           positions([[.75, 0], [1, 0]], "OPL", meaning="anatomical_position"),
                           query_support=query)
    assert result.status == ("project_approximation", "unsupported")
    assert result.reason[1] == "outside_query_support"
    with pytest.raises(ContractError):
        catalog.query("HC_W2000_central_equal_area_current",
                      positions([[.75, 0]], "OPL", meaning="anatomical_position"), query_support={})


def test_missing_s_density_is_distinct_from_source_quoted_absence():
    catalog = load_atlas_source_domains(ROOT)
    q = positions([[.02, 0]], "anatomical_retina", meaning="anatomical_position")
    absence = catalog.query("MartinGrunert1999_S_free_mask", q, domain_role="source")
    assert absence.status == ("source_supported",)
    assert absence.provenance["biological_mask"] == "S_absent"
    assert not absence.provenance["absence_is_empty_sample"]
    nonzero = catalog.query("MartinGrunert1999_S_density", positions([[.06, 0]]))
    assert nonzero.status == ("unsupported",)
    assert nonzero.reason == ("pending_digitization",)
    with pytest.raises(ContractError):
        catalog.query("MartinGrunert1999_S_density", positions([[.06, 0]], unit="deg"))


def test_rod_hard_mask_keeps_project_class_and_original_sparse_observation():
    result = load_atlas_source_domains(ROOT).query("Packer1989_rod_free_mask", positions([[.01, 0]]))
    assert result.status == ("project_approximation",)
    assert result.provenance["source_values"]["occasional_closest_rod_radius_mm"] == .02
    assert "not a universal" in result.provenance["mask_scope"]


def test_inward_hc_candidate_and_unregistered_henle_application_stay_unsupported():
    catalog = load_atlas_source_domains(ROOT)
    result = catalog.query("HC_central_constant_footprint_inward_candidate",
                           positions([[.3, 0]], "OPL", meaning="anatomical_position"))
    assert result.status == ("unsupported",) and result.application_domain is None
    assert result.reason == ("inactive_candidate",)
    henle = catalog.query("Schein1988_direct_Henle", positions([[.35, 0]]))
    assert henle.status == ("unsupported",)
    assert henle.provenance["reported_digitized_points"] == 343
    assert not henle.provenance["digitized_points_are_cells"]
    assert henle.provenance["output_frames"] == ["OPL"]


def test_verified_henle_points_preserve_derived_evidence_and_missing_tail():
    catalog = load_atlas_source_domains(ROOT)
    provider = "Schein1988_Henle_nasal_digitized"
    source = catalog.query(provider, positions([[.35, 0], [.351, 0], [3, 0]],
                           "anatomical_retina", meaning="anatomical_position"), domain_role="source")
    assert source.status == ("source_supported", "unsupported", "unsupported")
    assert source.provenance["evidence_status"] == "digitized_author_derived"
    assert source.provenance["specimen"] == "M1 right retina"
    assert source.provenance["source_package_identity_verified"]
    assert not source.provenance["raw_cell_data"]
    application = catalog.query(provider, positions([[.351, 0], [3, 0]]))
    assert application.status == ("project_approximation", "unsupported")


def test_original_packer_markers_keep_cell_class_and_native_rays():
    catalog = load_atlas_source_domains(ROOT)
    provider = "Packer1989_rod_digitized_superior"
    record = catalog.records[provider]
    r = record.source_domain[0].sampled_radii[0]
    source = catalog.query(provider, positions([[0, r], [r, 0], [0, .1]],
                           "anatomical_retina", meaning="anatomical_position"), domain_role="source")
    assert source.status == ("source_supported", "unsupported", "unsupported")
    assert source.provenance["evidence_status"] == "digitized_author_map_resampling"
    assert source.provenance["cell_class"] == "rod"
    assert sum(record.provenance["reported_digitized_points"] for name, record in catalog.records.items()
               if name.startswith("Packer1989_") and "_digitized_" in name) == 116


def test_hc_measurement_rows_keep_intervals_without_synthesizing_a_spatial_field():
    catalog = load_atlas_source_domains(ROOT)
    record = catalog.records["Wassle2000_HC_density_measurements"]
    rows = record.provenance["measurement_rows"]
    assert len(rows) == 23
    interval_rows = [row for row in rows if row["source_object"] == "Table 2" and row["patch"] == 1]
    assert len(interval_rows) == 2
    assert all(row["eccentricity_mm"] is None for row in interval_rows)
    assert all((row["eccentricity_min_mm"], row["eccentricity_max_mm"]) == (2, 3) for row in interval_rows)
    assert all(row["retinal_direction"] == "not specified per table row" for row in interval_rows)
    source = catalog.query(record.provider_id, positions([[2.5, 0]],
                           "anatomical_retina", meaning="anatomical_position"), domain_role="source")
    assert source.status == ("unsupported",)
    assert source.provenance["evidence_status"] == "published_table_measurement"


@pytest.mark.parametrize("change", ["measured_disk", "missing_transfer", "activate_pending"])
def test_catalog_rejects_evidence_promotion_and_unnamed_transfer(change):
    data = deepcopy(load_atlas_source_domains(ROOT).data)
    if change == "measured_disk":
        data["providers"]["HC_W2000_central_equal_area_current"]["evidence_status"] = "measurement"
    elif change == "missing_transfer":
        data["providers"]["packer_cone_nasal_tabulation"]["transfer"] = None
    else:
        data["providers"]["MartinGrunert1999_S_density"]["active"] = True
    with pytest.raises(ContractError):
        CoverageCatalog(data)
