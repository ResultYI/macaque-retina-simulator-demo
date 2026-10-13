"""Finite HC contact regression tests, written under the user's stop-run order.

These tests have deliberately NOT BEEN EXECUTED in this implementation pass.
They are for the later, explicitly authorized engineering verification phase.
"""
import numpy as np
import pytest

from retina_sim.contracts import ContractError, Quantity
from retina_sim.hc_arbor import (HCArborConfigurationError,
    build_hc_arbor_contacts, footprint_extent_mm, homotypic_gap_candidates)


def geometry(points, meaning, frame):
    return Quantity(np.asarray(points, dtype=float), 'mm', meaning,
                    'synthetic_fixture', frame)


def disk(radius):
    return {'kind': 'disk', 'radius_mm': radius, 'unit': 'mm',
            'coordinate_frame': 'OPL', 'source_profile_id': 'fixture_finite_arbor',
            'evidence_class': 'project_approximation',
            'approximation_id': 'fixture_finite_arbor'}


def contacts(inputs, pedicles, hc_points, *, cone_types=None, hc_types=None,
             footprints=None, **options):
    n, m = len(inputs), len(hc_points)
    return build_hc_arbor_contacts(tuple(f'cone:{i}' for i in range(n)),
        geometry(inputs, 'functional_position', 'functional_visual_field'),
        geometry(pedicles, 'pedicle_position', 'OPL'),
        tuple(f'hc:{i}' for i in range(m)),
        tuple(hc_types or ['H1'] * m),
        geometry(hc_points, 'anatomical_position', 'anatomical_retina'),
        geometry(hc_points, 'arbor_anchor_position', 'OPL'),
        np.ones(n, dtype=bool),
        footprints if footprints is not None else {'H1': disk(.05), 'H2': disk(.05)},
        cone_types=tuple(cone_types or ['L'] * n), **options)


def test_displaced_input_cone_contacts_local_pedicle_hc_not_rf_origin():
    A, B, audit = contacts([[0., 0.]], [[.7, 0.]], [[.71, 0.], [.01, 0.]])
    assert A.toarray().tolist() == [[1.], [0.]]
    assert B.toarray().tolist() == [[1., 0.]]
    assert audit['paired_cones'] == [{'cone_id': 'cone:0',
        'input_xy_mm': [0., 0.], 'pedicle_xy_mm': [.7, 0.], 'input_owner': True}]
    assert [(edge['cone_id'], edge['hc_id']) for edge in audit['edges']] == [('cone:0', 'hc:0')]
    assert audit['discrete_sum_jacobian_multiplier'] is False


def test_anchor_in_pedicle_free_center_can_reach_terminal_in_finite_arbor():
    # No density function is evaluated at this anchor. The entire configured
    # arbor is queried; an empty anchor neighborhood is not an exclusion mask.
    A, B, audit = contacts([[0., 0.]], [[.20, 0.]], [[0., 0.]],
                          footprints={'H1': disk(.25), 'H2': disk(.25)})
    assert A.nnz == B.nnz == 1
    assert audit['edges'][0]['arbor_anchor_distance_mm'] == pytest.approx(.20)
    assert audit['outside_support_nearest_fallback'] is False


def test_absent_span_stays_unconfigured_with_no_central_radius_default():
    assert footprint_extent_mm({'H1': None, 'H2': None}, require_configured=False) is None
    with pytest.raises(HCArborConfigurationError, match='HC_ARBOR_UNCONFIGURED'):
        contacts([[.15, 0.]], [[.7, 0.]], [[.71, 0.]],
                 footprints={'H1': None, 'H2': None})
    A, B, audit = contacts([[.15, 0.]], [[.7, 0.]], [[.71, 0.]],
        footprints={'H1': None, 'H2': None}, require_configured=False)
    assert A.nnz == B.nnz == 0
    assert audit['geometry_status'] == 'HC_ARBOR_UNCONFIGURED'


def test_source_support_rejects_full_arbor_overrun_instead_of_clipping():
    with pytest.raises(ContractError, match='finite arbor.*extends beyond source'):
        contacts([[7.05, 0.]], [[7.05, 0.]], [[7.05, 0.]],
                 footprints={'H1': disk(.1), 'H2': disk(.1)})


def test_real_source_frames_and_id_order_are_checked():
    with pytest.raises(ContractError, match='cone pedicle requires explicit'):
        build_hc_arbor_contacts(('cone:a',),
            geometry([[0., 0.]], 'functional_position', 'functional_visual_field'),
            geometry([[.7, 0.]], 'pedicle_position', 'functional_visual_field'),
            ('hc:a',), ('H1',),
            geometry([[.71, 0.]], 'anatomical_position', 'anatomical_retina'),
            geometry([[.71, 0.]], 'arbor_anchor_position', 'OPL'),
            np.ones(1, bool), {'H1': disk(.05), 'H2': disk(.05)}, cone_types=('L',))
    with pytest.raises(ContractError, match='RF/pedicle ID order mismatch'):
        contacts([[0., 0.]], [[.7, 0.]], [[.71, 0.]], cone_pedicle_ids=('cone:other',))


def test_h1_all_cone_divergence_conserves_total_when_s_is_ineligible():
    divergence = {'source_profile_id': 'fixture_all_cone_divergence',
        'evidence_class': 'project_approximation',
        'approximation_id': 'fixture_balanced_eligible_quotas',
        'coordinate_frame': 'OPL', 'radial_support_mm': [0., 7.1],
        'denominator': 'all_cones', 'values': {'H1': 1., 'H2': 0.}}
    points = [[.50, 0.], [.51, 0.], [.52, 0.], [.53, 0.]]
    A, B, audit = contacts(points, points, [[.5, .001], [.51, .001], [.52, .001]],
        cone_types=('L', 'M', 'S', 'S'),
        footprints={'H1': disk(.1), 'H2': disk(.1)}, divergence_profile=divergence)
    summary = audit['divergence']['subtypes']['H1']
    assert summary['integer_all_cone_edge_target'] == 4
    assert summary['integer_eligible_quotas'] == [2, 2, 0, 0]
    assert A.nnz == B.nnz == 4
    assert B.getrow(2).nnz == B.getrow(3).nnz == 0


def test_quota_deficit_is_reported_without_outside_support_nearest_fill():
    divergence = {'source_profile_id': 'fixture_divergence',
        'evidence_class': 'project_approximation', 'approximation_id': 'fixture_quota',
        'coordinate_frame': 'OPL', 'radial_support_mm': [0., 7.1],
        'denominator': 'all_cones', 'values': {'H1': 3., 'H2': 0.}}
    A, B, audit = contacts([[.5, 0.]], [[.5, 0.]], [[.51, 0.], [1., 0.]],
                          divergence_profile=divergence)
    assert A.nnz == B.nnz == 1
    row = next(row for row in audit['cone_rows'] if row['subtype'] == 'H1')
    assert row['missing_degree'] == 2
    assert row['status'] == 'CONTACT_QUOTA_INFEASIBLE'
    with pytest.raises(ContractError, match='CONTACT_QUOTA_INFEASIBLE'):
        contacts([[.5, 0.]], [[.5, 0.]], [[.51, 0.], [1., 0.]],
                 divergence_profile=divergence, require_owner_quota=True)


def test_common_contact_topology_has_independent_forward_feedback_pooling():
    A, B, audit = contacts([[.5, 0.], [.51, 0.]], [[.5, 0.], [.51, 0.]], [[.5, 0.]],
                          feedback_normalization='absolute_contact_weight')
    assert A.toarray().tolist() == [[.5, .5]]
    assert B.toarray().tolist() == [[1.], [1.]]
    assert set(zip(*A.nonzero())) == {(j, i) for i, j in zip(*B.nonzero())}
    assert all(edge['weight_unit'] == 'dimensionless' for edge in audit['edges'])
    assert audit['geometry_is_conductance'] is False


def test_morphology_nodes_use_actual_branch_and_depth_not_anchor_nearest():
    morphology = {'kind': 'morphology_nodes', 'nodes_mm': [[.15, 0., .01]],
        'node_coordinates': 'relative_to_anchor', 'contact_threshold_mm': .01,
        'z_tolerance_mm': .002, 'node_regions': ['OPL_branch'],
        'allowed_post_regions': ['OPL_branch'],
        'source_profile_id': 'fixture_morphology', 'evidence_class': 'project_approximation',
        'approximation_id': 'fixture_finite_branch'}
    # The first pedicle is in x/y contact with a branch far from the anchor;
    # the second is at the same x/y but at an unsupported OPL depth.
    A, B, audit = contacts([[0., 0.], [.01, 0.]], [[.151, 0., .011], [.151, 0., .05]],
        [[0., 0., 0.]], footprints={'H1': morphology, 'H2': disk(.05)})
    assert A.nnz == B.nnz == 1
    assert audit['edges'][0]['morphology_node_index'] == 0
    assert audit['edges'][0]['arbor_anchor_distance_mm'] > .1
    assert audit['edges'][0]['distance_mm'] < .01


def test_finite_extension_only_contacts_pedicle_in_declared_allowed_domain():
    morphology = {'kind': 'morphology_nodes', 'nodes_mm': [[.02, 0.]],
        'node_coordinates': 'relative_to_anchor', 'contact_threshold_mm': .005,
        'allowed_extension': {'kind': 'disk', 'radius_mm': .06},
        'max_extension_mm': .04,
        'extension_policy': 'nearest_branch_to_pedicle_in_finite_domain',
        'source_profile_id': 'fixture_morphology', 'evidence_class': 'project_approximation',
        'approximation_id': 'fixture_finite_extension'}
    A, B, audit = contacts([[0., 0.], [.01, 0.]], [[.055, 0.], [.075, 0.]], [[0., 0.]],
        footprints={'H1': morphology, 'H2': disk(.05)})
    assert footprint_extent_mm({'H1': morphology, 'H2': disk(.05)}) == pytest.approx(.06)
    assert A.nnz == B.nnz == 1
    assert audit['edges'][0]['extension_length_mm'] == pytest.approx(.03)


def test_ellipse_source_extent_does_not_use_enclosing_circle_as_arbor():
    ellipse = {**disk(.1), 'kind': 'ellipse', 'axes_mm': [.1, .01],
               'angle_rad': np.pi / 2, 'radial_support_mm': [0., 7.1]}
    # A tangentially oriented ellipse fits even though its enclosing disk
    # would reach beyond the 7.1 mm source boundary.
    A, B, audit = contacts([[7.05, 0.]], [[7.05, 0.]], [[7.05, 0.]],
        footprints={'H1': ellipse, 'H2': disk(.01)})
    assert A.nnz == B.nnz == 1
    assert audit['hc_rows'][0]['finite_envelope_radial_mm'][1] < 7.1


def test_generation_boundary_is_distinct_from_zero_legal_candidates():
    A, B, audit = contacts([[.5, 0.]], [[.5, 0.]], [[.52, 0.]],
        cone_pedicle_generation_bounds_mm=(.49, .53, -.01, .01))
    assert A.nnz == B.nnz == 1
    assert audit['hc_rows'][0]['boundary_status'] == 'CONTACT_BOUNDARY_MISSING'
    with pytest.raises(ContractError, match='CONTACT_BOUNDARY_MISSING'):
        contacts([[.5, 0.]], [[.5, 0.]], [[.52, 0.]],
            cone_pedicle_generation_bounds_mm=(.49, .53, -.01, .01),
            require_complete_contact_halo=True)


def test_gap_candidate_graph_requires_gate_and_never_binds_conductance():
    ids, types = ('h1:a', 'h1:b', 'h2:a'), ('H1', 'H1', 'H2')
    soma = geometry([[.5, 0.], [.51, 0.], [.505, 0.]],
                    'anatomical_position', 'anatomical_retina')
    pending = homotypic_gap_candidates(ids, types, soma)
    assert pending['status'] == 'GAP_DISTANCE_UNCONFIGURED'
    assert pending['edges'] == []
    with pytest.raises(ContractError, match='source_profile_id'):
        homotypic_gap_candidates(ids, types, soma, .02)
    audit = homotypic_gap_candidates(ids, types, soma, .02,
        source_profile_id='fixture_finite_gap_candidates')
    assert [(edge['hc_a'], edge['hc_b']) for edge in audit['edges']] == [('h1:a', 'h1:b')]
    assert audit['conductance'] == 'UNBOUND'
    assert audit['connected_to_dynamics'] is False


def test_collinear_gap_neighbors_are_bounded_adjacent_not_infinite_nearest():
    soma = geometry([[.5, 0.], [.51, 0.], [1., 0.]],
                    'anatomical_position', 'anatomical_retina')
    audit = homotypic_gap_candidates(('a', 'b', 'c'), ('H1',) * 3, soma, .02,
        source_profile_id='fixture_finite_gap_candidates')
    assert audit['subtypes'][0]['method'] == 'collinear_projected_adjacent'
    assert audit['edge_count'] == 1
    assert audit['subtypes'][0]['omitted_outside_finite_gate'] == 1


def test_measurement_label_requires_frozen_primary_geometry_and_source_domain():
    claimed = disk(.05)
    claimed['evidence_class'] = 'measurement'
    with pytest.raises(ContractError, match='primary material'):
        footprint_extent_mm({'H1': claimed, 'H2': disk(.05)})
