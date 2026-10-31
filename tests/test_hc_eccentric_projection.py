"""Actual disk membership, OPL eccentricity and empty-row behavior."""
from pathlib import Path

import numpy as np
import pytest

from retina_sim.atlas_hc import AtlasHCProvider
from retina_sim.contracts import ContractError, Quantity
from retina_sim.disk_hc_functional import eccentric_projection


ROOT = Path(__file__).resolve().parents[1]


def sizes(points, kinds):
    query = AtlasHCProvider(ROOT).footprints(
        tuple(str(i) for i in range(len(points))), kinds,
        Quantity(points, 'mm', 'arbor_anchor_position', 'fixture', 'OPL'),
        mode='connection_preview', enable_central_candidate=True)
    assert not query['blocked_records']
    return np.array([row['radius_mm'] for row in query['per_hc_coverage']])


def test_eccentric_size_changes_actual_connections_and_preserves_types():
    hc = np.array([[.75, 0.], [3., 0.], [.75, .1], [3., .1]])
    kinds = np.array(['H1', 'H1', 'H2', 'H2'])
    radii = sizes(hc, kinds)
    # The same offsets fall outside the central disk and inside the outer disk.
    cones = np.concatenate([p + [[.004, 0], [.014, 0], [.02, 0]] for p in hc])
    types = np.tile(['L', 'S', 'M'], 4)
    raw, audit = eccentric_projection(cones, types, hc, kinds,
                                     np.ones(12, bool), 4., radii)
    assert radii[1] > radii[0] and radii[3] > radii[2]
    assert raw[:, 0].nnz == 1 and raw[:, 1].nnz == 2
    assert raw[:, 2].nnz == 1 and raw[:, 3].nnz == 3
    assert raw[types == 'S'][:, kinds == 'H1'].nnz == 0
    assert audit['maximum_distance_over_HC_radius'] <= 1 + 1e-9
    assert audit['integer_divergence_imposed'] is False


def test_uses_HC_radius_and_keeps_empty_cones_without_nearest_fill():
    raw, audit = eccentric_projection(
        [[.75, 0], [.9, 0]], ['L', 'S'], [[.75, 0], [.75, .05]],
        ['H1', 'H2'], np.ones(2, bool), 1., [.00804, .01222])
    assert raw.nnz == 1 and raw[1].nnz == 0
    assert raw[:, 1].nnz == 0
    assert audit['outside_radius_fallback'] is False


def test_radius_boundary_is_included_and_shape_errors_are_rejected():
    raw, _ = eccentric_projection([[.75804, 0]], ['M'], [[.75, 0]], ['H1'],
                                 np.ones(1, bool), 1., [.00804])
    assert raw.nnz == 1
    with pytest.raises(ContractError, match='radii'):
        eccentric_projection([[.75, 0]], ['L'], [[.75, 0]], ['H1'],
                             np.ones(1, bool), 1., [float('nan')])
    with pytest.raises(ContractError, match='exceeds certified'):
        eccentric_projection([[.999, 0]], ['L'], [[.75, 0]], ['H1'],
                             np.ones(1, bool), 1., [.00804])


def test_source_profile_is_positive_bounded_and_outside_range_blocks():
    radial = np.array([0., .5, 1., 1.338, 3., 5., 6.])
    for kind in ('H1', 'H2'):
        radii = sizes(np.c_[radial, np.zeros(len(radial))], [kind]*len(radial))
        assert np.all(np.diff(radii) >= 0)
        assert radii[-1] == radii[-2] and radii[3] > radii[2]
    query = AtlasHCProvider(ROOT).footprints(('outside',), ('H1',),
        Quantity([[6.01, 0]], 'mm', 'arbor_anchor_position', 'fixture', 'OPL'),
        mode='connection_preview', enable_central_candidate=True)
    assert query['blocked_records']


def test_shared_gain_background_selector_keeps_both_HC_states_and_feedback():
    from scipy import sparse
    from retina_sim.h1_h2_effective import joint_equilibrium
    from retina_sim.vh2005 import SOURCE_VALUES
    A = sparse.csr_matrix([[.75, .25], [.2, .8]])
    B = sparse.eye(2, format='csr')
    parameters = dict(input_scale=SOURCE_VALUES['g_t'], feedback_scale=1.)
    original = joint_equilibrium(A, B, np.array([0]), np.array([1]), np.array([10.,100.]), parameters)
    shared = joint_equilibrium(A, B, np.array([0]), np.array([], dtype=int), np.array([10.,100.]), parameters)
    np.testing.assert_allclose(original['h'], shared['h'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(original['release'], shared['release'], rtol=0, atol=1e-12)
    assert shared['feedback'][1,1] == 1 and shared['h'][1] > 0
