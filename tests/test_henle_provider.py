from pathlib import Path
import shutil

import numpy as np
import pytest

from retina_sim.contracts import ContractError, Quantity
from retina_sim.henle import ScheinHenleDisplacement


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path("data/source_packages/20261012/Schein1988_Henle_source_verified")


def is_position(values):
    return Quantity(values, "mm", "cone_inner_segment_position", "test_coordinates", "anatomical_retina")


def test_cardinal_values_and_source_identity_are_preserved():
    provider = ScheinHenleDisplacement(ROOT)
    p = is_position([[.5, 0], [-.5, 0], [0, .5], [0, -.5]])
    output = provider.apply(p)
    np.testing.assert_allclose(output.values, [[.795, 0], [-.778, 0], [0, .805], [0, -.761]], atol=1e-12)
    output.require("mm", "pedicle_position", (4, 2), "OPL")
    report = provider.coverage(p)
    assert all(row["status"] == "source_supported" for row in report["points"])
    assert report["provenance"]["specimen"] == "M1 right retina"
    assert report["provenance"]["species"] == "Macaca fascicularis"
    assert report["provenance"]["curve_point_count"] == 343
    assert report["provenance"]["evidence_status"] == "author_derived_then_digitized"
    assert not report["provenance"]["raw_cell_measurement"]
    assert not report["provenance"]["discrete_mapping_jacobian_applied"]
    assert provider.curves["nasal"].domain_mm == (.025, 2.475)
    with pytest.raises(ValueError):
        provider.curves["nasal"].displacement_mm[0] = 0
    with pytest.raises(TypeError):
        provider.curves["nasal"] = provider.curves["temporal"]


def test_missing_source_is_not_zero_displacement():
    provider = ScheinHenleDisplacement(ROOT)
    p = is_position([[0, 0], [.02, 0], [3, 0], [0, 2], [2.4, 0]])
    report = provider.coverage(p)
    assert [row["status"] for row in report["points"]] == ["unsupported"] * 4 + ["source_supported"]
    assert all(row["displacement_mm"] is None for row in report["points"][:4])
    with pytest.raises(ContractError, match="4 points.*no central bridge or tail fill"):
        provider.apply(p)


def test_angular_interpolation_requires_both_meridians():
    provider = ScheinHenleDisplacement(ROOT)
    diagonal = np.array([[1, 1]]) / np.sqrt(2)
    p = is_position(diagonal * .5)
    output = provider.apply(p)
    np.testing.assert_allclose(np.linalg.norm(output.values, axis=1), [.5 + (.295 + .305) / 2], atol=1e-12)
    row = provider.coverage(p)["points"][0]
    assert row["status"] == "project_approximation"
    assert row["angular_interpolation"] and not row["radial_interpolation"]
    assert row["meridian_weights"] == {"nasal": .5, "superior": .5}
    unsupported = is_position(diagonal * 2)
    with pytest.raises(ContractError, match="superior"):
        provider.apply(unsupported)
    # A pure nasal query need not borrow the shorter superior curve.
    assert provider.coverage(is_position([[2, 0]]))["supported"]


def test_radial_interpolation_is_a_named_project_rule():
    provider = ScheinHenleDisplacement(ROOT)
    p = is_position([[.5125, 0]])
    curve = provider.curves["nasal"]
    expected = np.interp(.5125, curve.inner_segment_radius_mm, curve.displacement_mm)
    np.testing.assert_allclose(provider.apply(p).values, [[.5125 + expected, 0]], atol=1e-12)
    row = provider.coverage(p)["points"][0]
    assert row["status"] == "project_approximation" and row["radial_interpolation"]
    assert provider.provenance["radial_interpolation_rule"] == "linear_between_digitized_curve_samples_v1"


def test_rf_transfer_units_and_coordinate_roles_are_explicit():
    rf = Quantity([[.5, 0]], "mm", "functional_position", "test_coordinates", "functional_visual_field")
    with pytest.raises(ContractError, match="explicit project transfer"):
        ScheinHenleDisplacement(ROOT).apply(rf)
    provider = ScheinHenleDisplacement(ROOT, rf_transfer="rf_equals_inner_segment_mm_v1")
    np.testing.assert_allclose(provider.apply(rf).values, [[.795, 0]])
    assert provider.coverage(rf)["points"][0]["status"] == "project_approximation"
    with pytest.raises(ContractError, match="Expected mm"):
        provider.apply(Quantity([[.5, 0]], "deg", "functional_position", "test", "functional_visual_field"))
    with pytest.raises(ContractError, match="Coordinate frame"):
        provider.apply(Quantity([[.5, 0]], "mm", "cone_inner_segment_position", "test", "OPL"))
    with pytest.raises(ContractError, match="Expected mm/cone_inner_segment_position"):
        provider.apply(Quantity([[.5, 0]], "mm", "anatomical_position", "test", "anatomical_retina"))
    with pytest.raises(ContractError, match="shape"):
        provider.apply(is_position([.5, 0]))


def test_inverse_and_empty_mapping_preserve_count_without_jacobian():
    provider = ScheinHenleDisplacement(ROOT)
    radii = np.array([.025, .4, .5, .537, 1.2, 1.675])
    angles = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, -np.pi / 2])
    xy = radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    output = provider.apply(is_position(xy))
    assert output.values.shape == xy.shape
    np.testing.assert_allclose(provider.inverse(output).values, xy, atol=1e-12)
    empty = provider.apply(is_position(np.empty((0, 2))))
    assert empty.values.shape == (0, 2)
    assert provider.inverse(empty).values.shape == (0, 2)
    with pytest.raises(ContractError, match="outside mapped source domain"):
        provider.inverse(Quantity([[0, 0]], "mm", "pedicle_position", "test", "OPL"))


def test_frozen_hash_does_not_accept_modified_curves(tmp_path):
    destination = tmp_path / SOURCE_DIR
    destination.mkdir(parents=True)
    for name in ("schein1988_fig15_digitized_curves.json", "provenance.json", "figure15_original_crop.png", "textual_anchors.json"):
        shutil.copyfile(ROOT / SOURCE_DIR / name, destination / name)
    curve_file = destination / "schein1988_fig15_digitized_curves.json"
    curve_file.write_bytes(curve_file.read_bytes() + b"\n")
    with pytest.raises(ContractError, match="source identity mismatch"):
        ScheinHenleDisplacement(tmp_path)
