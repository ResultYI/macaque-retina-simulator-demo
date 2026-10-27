import numpy as np
import pytest
from scipy import sparse

from retina_sim.contracts import ContractError
from retina_sim.continuous_state import ExponentialForced
from retina_sim.continuous_operators import project_common_exponential


def test_shared_basis_matches_actual_spatial_projection_at_internal_times():
    matrix = sparse.csr_matrix([[.3, .7, 0.], [0., .25, .75]])
    coefficients = np.arange(24., dtype=np.float64).reshape(8, 3)*.01
    source = ExponentialForced(t0=.1, t1=.101, state_ids=np.arange(3),
        y0=np.array([1., 2., 3.]), decay=np.full(3, 200.), coefficients=coefficients)
    projected = project_common_exponential(matrix, source, output_state_ids=np.arange(2))
    for t, side in ((.1, 'right'), (.100173, 'left'), (.1009, 'right'), (.101, 'left')):
        expected = matrix@source.evaluate(t, side)
        np.testing.assert_allclose(projected.evaluate(t, side), expected, rtol=3e-15, atol=0)
        # Power must still act on the pooled value, not on original sources.
        assert not np.allclose(projected.evaluate(t, side)**2, matrix@(source.evaluate(t, side)**2))
    with pytest.raises(ContractError, match='original-equation'):
        projected.rhs_at(.1005, 'right')


def test_different_decays_never_moved_through_spatial_operator():
    source = ExponentialForced(t0=0., t1=.001, state_ids=np.arange(2), y0=np.ones(2),
        decay=np.array([100., 101.]), coefficients=np.zeros((8, 2)))
    with pytest.raises(ContractError, match='Shared decay'):
        project_common_exponential(sparse.csr_matrix([[.5, .5]]), source, output_state_ids=[0])
