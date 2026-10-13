"""Regression specifications for HC synthetic geometry, written but NOT RUN.

The current user stop-running instruction prohibits executing this file.
When execution is authorized, these check source/domain contracts and geometry
invariants; they do not establish biological spatial calibration or MCMC mixing.
"""
from pathlib import Path
import numpy as np
import pytest

from retina_sim.contracts import ContractError, Quantity
from retina_sim.geometry import RetinalDomain
from retina_sim.hc_mosaic import HCDensityProfile, JointHCMosaic, _cumulative_quotas, generate_joint_hc


ROOT = Path(__file__).resolve().parents[1]


def _domain(bounds=(-.01, .01, -.01, .01), owner=(-.005, .005, -.005, .005), empty=False):
    return RetinalDomain(bounds, owner, lambda xy: np.full(len(xy), not empty, dtype=bool),
                         "test_named_physical_mask", "test_physical_canonical")


def test_density_source_nodes_and_derived_central_endpoints():
    profile = HCDensityProfile(ROOT)
    assert float(profile.radial(0)) == pytest.approx(250)
    assert float(profile.radial(.6)) == pytest.approx(23000)
    assert float(profile.radial_fraction(0)) == pytest.approx(.8)
    for row in profile.source_rows:
        interval = row["eccentricity_interval_mm"]
        radius = sum(interval) / 2
        xy = np.array([[radius, 0]])
        assert profile.type_density(xy, "H1")[0] == pytest.approx(row["H1"])
        assert profile.type_density(xy, "H2")[0] == pytest.approx(row["H2"])
    assert profile.metadata["evidence_class"] == "project_approximation"
    assert profile.metadata["original_source_records"][0]["eccentricity_interval_mm"] == [2, 3]


def test_density_domain_and_coordinate_frames_are_enforced():
    profile = HCDensityProfile(ROOT)
    for unsupported in (-.0001, 7.10001, float("nan")):
        with pytest.raises(ContractError, match="DOMAIN_UNSUPPORTED"):
            profile.radial(unsupported)
    optical = Quantity([[.1, 0]], "mm", "functional_position", "test", "functional_visual_field")
    with pytest.raises(ContractError):
        profile.total(optical)
    with pytest.raises(ContractError):
        profile.type_density([[.1, 0]], "unidentified_HC")


def test_density_is_positive_and_has_no_peak_segment_overshoot():
    profile = HCDensityProfile(ROOT)
    inner = profile.radial(np.linspace(0, .6, 101))
    assert np.all(inner > 0) and np.all(np.diff(inner) >= 0)
    for index in range(len(profile.nodes_mm) - 1):
        grid = np.linspace(profile.nodes_mm[index], profile.nodes_mm[index + 1], 101)
        total, fraction = profile.radial(grid), profile.radial_fraction(grid)
        assert np.all(total >= min(profile.total_nodes[index:index + 2]) * (1 - 1e-12))
        assert np.all(total <= max(profile.total_nodes[index:index + 2]) * (1 + 1e-12))
        assert np.all((fraction > 0) & (fraction < 1))


def test_cumulative_rounding_preserves_global_and_local_counts_including_zero_one():
    masses = np.array([.03, .06, .97, .14, 2.31, .01])
    for offset in (.0, .11, .49, .99):
        counts = _cumulative_quotas(masses, offset)
        assert counts.sum() == np.floor(masses.sum() + offset)
        assert np.all(np.abs(counts - masses) < 1)
        assert np.any(counts == 0)
        assert np.any(counts == 1)


def test_crop_reuses_complete_master_identity_and_never_fabricates_functional_rf():
    master = JointHCMosaic(
        Quantity([[-.005, 0], [.005, 0]], "mm", "anatomical_position", "project_approximation", "anatomical_retina"),
        np.array(["H1", "H2"]), ("canonical:H1:0", "canonical:H2:1"), np.array([True, True]),
        {"canonical_bounds_mm": [-.01, .01, -.01, .01], "mask_source_id": "test_named_physical_mask",
         "canonical_instance_id": "one_saved_instance"})
    view = master.crop(_domain(bounds=(-.01, 0, -.01, .01), owner=(-.007, -.003, -.002, .002)))
    assert view.cell_ids == (master.cell_ids[0],)
    assert np.array_equal(view.positions.values, master.positions.values[:1])
    assert view.types.tolist() == ["H1"]
    assert view.diagnostics["canonical_instance_id"] == "one_saved_instance"
    assert view.positions.coordinate_frame == "anatomical_retina"
    assert view.positions.meaning == "anatomical_position"
    with pytest.raises(ContractError):
        view.crop(_domain())  # Cannot expand an already cropped view into absent cells.


def test_empty_mask_has_zero_cells_no_hidden_minimum_and_no_mixing_claim():
    profile = HCDensityProfile(ROOT)
    mosaic = generate_joint_hc(profile, _domain(empty=True), 17, max_cells=0,
                               options={"sweeps": 0})
    assert mosaic.positions.values.shape == (0, 2)
    assert mosaic.cell_ids == ()
    assert mosaic.diagnostics["total_H1_H2_counts"] == [0, 0]
    assert "NOT_CERTIFIED" in mosaic.diagnostics["mixing_status"]


def test_instance_identity_ignores_owner_but_changes_with_canonical_domain():
    profile = HCDensityProfile(ROOT)
    master = generate_joint_hc(profile, _domain(), 29, max_cells=32, options={"sweeps": 0})
    changed_owner = _domain(owner=(-.004, .004, -.004, .004))
    same = generate_joint_hc(profile, changed_owner, 29, max_cells=32, options={"sweeps": 0})
    assert master.cell_ids == same.cell_ids
    assert np.array_equal(master.positions.values, same.positions.values)
    different = generate_joint_hc(profile, _domain(bounds=(-.011, .011, -.01, .01)), 29,
                                 max_cells=32, options={"sweeps": 0})
    assert master.diagnostics["canonical_instance_id"] != different.diagnostics["canonical_instance_id"]


def test_generator_refuses_unsupported_halo_and_unknown_options_before_sampling():
    profile = HCDensityProfile(ROOT)
    outside = _domain(bounds=(7.0, 7.2, -.01, .01), owner=(7.01, 7.02, -.005, .005))
    with pytest.raises(ContractError, match="DOMAIN_UNSUPPORTED"):
        generate_joint_hc(profile, outside, 1, max_cells=64)
    with pytest.raises(ContractError, match="Unknown"):
        generate_joint_hc(profile, _domain(), 1, max_cells=64, options={"hidden_density_scale": .1})
