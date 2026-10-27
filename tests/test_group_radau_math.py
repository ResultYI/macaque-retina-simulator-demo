"""Synthetic CPU algebra checks, not model trajectories or acceptance evidence."""
import numpy as np
import pytest

from retina_sim.contracts import ContractError
from retina_sim.group_radau_math import affine_stages, batched_newton, gmres, tableau


def test_tableau_collocation_and_embedded_order_conditions():
    a, b, c, gamma, bhat = tableau()
    np.testing.assert_allclose(a @ np.ones(3), c, rtol=0, atol=2e-16)
    np.testing.assert_array_equal(a[-1], b)
    for degree in range(5):
        assert b @ c**degree == pytest.approx(1/(degree+1), abs=3e-16)
    for degree in range(3):
        assert gamma*(degree == 0) + bhat @ c**degree == pytest.approx(1/(degree+1), abs=3e-16)
    assert abs(np.linalg.det(a-gamma*np.eye(3))) < 2e-17
    assert gamma == pytest.approx(0.274888829595677, abs=1e-15)
    assert all(value.dtype == np.float64 for value in (a, b, c, bhat))
    a[:] = 0
    assert np.any(tableau()[0])


def test_affine_stages_use_stage_dependent_coefficients_per_state():
    y0 = np.array([0.2, 3.0, -0.6, 1.7])
    q = np.array([[1., 2., 3., 4.], [5., 3., 2., 4.], [2., 6., 1., 8.]])
    k = np.array([[10., 8., 2., 1.], [2., 3., 7., 4.], [1., 9., 5., 3.]])
    before = (y0.copy(), q.copy(), k.copy())
    dt = 0.3
    stages = affine_stages(y0, q, k, dt)
    a = tableau()[0]
    np.testing.assert_allclose(stages-y0[None, :]-dt*a @ (q-k*stages), 0, rtol=0, atol=2e-15)
    for old, current in zip(before, (y0, q, k)):
        np.testing.assert_array_equal(old, current)


@pytest.mark.parametrize("dt", [0.0, 0.01, 100.0])
def test_affine_stability_function_and_broadcasting(dt):
    y0 = np.array([0.3, -0.7, 9.0])
    k = np.array([2., 5., 200.])
    q = 1.2
    stages = affine_stages(y0, q, k, dt)
    z = -dt*k
    stability = (1+2*z/5+z*z/20)/(1-3*z/5+3*z*z/20-z*z*z/60)
    expected = q/k + stability*(y0-q/k)
    np.testing.assert_allclose(stages[-1], expected, rtol=2e-13, atol=2e-15)
    stage_q = np.array([[1.], [2.], [4.]])
    np.testing.assert_allclose(affine_stages(y0, stage_q, 0., dt),
                               y0[None, :]+dt*tableau()[0] @ np.broadcast_to(stage_q, (3, 3)),
                               atol=2e-14, rtol=1e-15)


def test_affine_rejects_bad_dtype_nonfinite_and_bad_shapes():
    with pytest.raises(ContractError, match="FP64"):
        affine_stages(np.ones(2, dtype=np.float32), 1., 1., 0.1)
    with pytest.raises(ContractError, match="Nonfinite"):
        affine_stages(np.array([np.nan]), 1., 1., 0.1)
    with pytest.raises(ContractError):
        affine_stages(np.ones(2), np.ones((2, 2)), 1., 0.1)
    with pytest.raises(ContractError):
        affine_stages(np.ones(2), 1., 1., -0.1)


def test_gmres_nonsymmetric_system_and_fixed_right_preconditioning():
    matrix = np.array([[4., 1., 0., 2.], [-2., 7., 1., 0.],
                       [0., 3., 5., -1.], [1., 0., 2., 6.]])
    expected = np.array([[1., -2.], [0.4, 2.3]])
    rhs = (matrix @ expected.ravel()).reshape(expected.shape)
    counts = {}
    result, info = gmres(lambda v: (matrix @ v.ravel()).reshape(v.shape), rhs,
                        lambda v: (v.ravel()/np.diag(matrix)).reshape(v.shape),
                        rtol=1e-13, atol=0., restart=4, maxiter=8, counters=counts)
    assert info["converged"]
    assert info["iterations"] <= 4
    assert info["residual"] == pytest.approx(np.linalg.norm(rhs.ravel()-matrix @ result.ravel()), abs=1e-15)
    np.testing.assert_allclose(result, expected, rtol=2e-13, atol=1e-14)
    assert counts["gmres_iterations"] == info["iterations"]
    assert counts["gmres_matvecs"] == info["iterations"] + counts["gmres_true_residual_checks"]
    assert counts["gmres_preconditioner_applications"] == info["iterations"] + info["cycles"]
    assert counts.get("host_scalar_reads", 0) == 0


def test_gmres_zero_rhs_and_happy_breakdown():
    result, info = gmres(lambda v: v.copy(), np.zeros(4), rtol=1e-12, atol=0.)
    assert info["converged"] and info["iterations"] == 0
    np.testing.assert_array_equal(result, 0.)
    rhs = np.arange(1., 6.)
    result, info = gmres(lambda v: 7*v, rhs, rtol=1e-14, atol=0.)
    assert info["converged"] and info["iterations"] == 1
    np.testing.assert_allclose(result, rhs/7, atol=1e-15, rtol=1e-15)


def test_gmres_restarted_solve_and_iteration_bound():
    matrix = np.array([[4., 1., 0., 0.], [0., 3., 0.5, 0.],
                       [0.1, 0., 2., 0.4], [0., 0.2, 0., 5.]])
    rhs = np.arange(1., 5.)
    result, info = gmres(lambda v: matrix @ v, rhs, rtol=1e-11, atol=0., restart=2, maxiter=60)
    assert info["converged"] and info["cycles"] > 1
    np.testing.assert_allclose(result, np.linalg.solve(matrix, rhs), rtol=2e-11, atol=1e-11)
    result, info = gmres(lambda v: matrix @ v, rhs, rtol=1e-14, atol=0., restart=2, maxiter=1)
    assert not info["converged"] and info["reason"] == "iteration_limit"
    assert info["iterations"] == 1
    assert np.linalg.norm(rhs-matrix @ result) > info["tolerance"]


def test_gmres_does_not_accept_singular_breakdown_or_scaled_preconditioned_residual():
    rhs = np.ones(3)
    _, info = gmres(lambda v: 0*v, rhs, rtol=1e-12, atol=0.)
    assert not info["converged"] and info["reason"] == "breakdown"
    matrix = np.diag([1., 2., 5.])
    _, info = gmres(lambda v: matrix @ v, rhs, lambda v: 1e-20*v,
                    rtol=1e-12, atol=0., maxiter=1)
    assert not info["converged"]
    assert info["residual"] > 0.1


def test_gmres_rejects_guard_nonfinite_and_callback_shape():
    calls = []
    def guard():
        calls.append(1)
        return len(calls) < 3
    with pytest.raises(ContractError, match="guard"):
        gmres(lambda v: v, np.ones(3), rtol=1e-12, atol=0., guard=guard)
    with pytest.raises(ContractError, match="Nonfinite"):
        gmres(lambda v: v*np.nan, np.ones(3), rtol=1e-12, atol=0.)
    with pytest.raises(ContractError, match="shape"):
        gmres(lambda v: v[None, :], np.ones(3), rtol=1e-12, atol=0.)


def test_newton_independent_systems_component_scales_and_final_residual():
    target = np.array([[4., 9.], [1., 25.], [0.01, 1.]])
    initial = np.array([[1., 1.], [1., 3.], [0.5, 1.]])
    atol = np.array([[1e-11, 1e-10]])
    visited = []
    counts = {}
    def residual(x):
        visited.append(x.copy())
        return x*x-target
    def jacobian(x):
        return 2*x[:, :, None]*np.eye(2)[None, :, :]
    result, info = batched_newton(initial, residual, jacobian, atol=atol,
                                  counters=counts, domain=lambda x: (x >= 0).all(axis=1))
    assert info["converged"] and info["iterations"] <= 8
    np.testing.assert_allclose(result, np.sqrt(target), rtol=0, atol=5e-10)
    np.testing.assert_array_equal(visited[-1], result)
    assert np.max(np.abs(result*result-target)/atol) <= 1
    assert counts["newton_iterations"] == info["iterations"]
    assert counts["newton_residual_evaluations"] == len(visited)


def test_newton_never_evaluates_invalid_backtracking_candidates():
    initial = np.array([[10.], [1.]])
    def residual(x):
        assert np.all(x > 0)
        return np.log(x)
    result, info = batched_newton(initial, residual, lambda x: (1/x)[:, :, None],
                                  atol=1e-11, domain=lambda x: (x > 0).all(axis=1))
    assert info["converged"]
    np.testing.assert_allclose(result, 1., atol=1e-11, rtol=0)


def test_newton_reports_bounded_and_line_search_failures():
    initial = np.array([[0.1]])
    residual = lambda x: x*x+1.
    jacobian = lambda x: (2*x)[:, :, None]
    _, info = batched_newton(initial, residual, jacobian, atol=1e-12)
    assert not info["converged"] and info["reason"] == "line_search_failed"
    assert info["iterations"] == 1
    _, info = batched_newton(np.array([[2.]]), lambda x: x*x-2,
                             jacobian, atol=1e-12, maxiter=1)
    assert not info["converged"] and info["reason"] == "iteration_limit"
    assert info["iterations"] == 1


def test_newton_freezes_fully_converged_rows_with_singular_jacobians():
    initial = np.array([[0.], [1.]])
    target = np.array([[0.], [4.]])
    result, info = batched_newton(initial, lambda x: x*x-target,
                                  lambda x: (2*x)[:, :, None], atol=1e-10)
    assert info["converged"]
    np.testing.assert_allclose(result, [[0.], [2.]], atol=1e-10, rtol=0)


def test_newton_rejects_invalid_domain_scale_guard_and_nonfinite():
    initial = np.ones((2, 1))
    residual = lambda x: x
    jacobian = lambda x: np.ones((2, 1, 1))
    with pytest.raises(ContractError, match="domain"):
        batched_newton(initial, residual, jacobian, atol=1e-10, domain=lambda x: False)
    with pytest.raises(ContractError, match="scales"):
        batched_newton(initial, residual, jacobian, atol=0.)
    with pytest.raises(ContractError, match="guard"):
        batched_newton(initial, residual, jacobian, atol=1e-10, guard=lambda: False)
    with pytest.raises(ContractError, match="Nonfinite"):
        batched_newton(initial, lambda x: x*np.nan, jacobian, atol=1e-10)
    with pytest.raises(ContractError, match="eight"):
        batched_newton(initial, residual, jacobian, atol=1e-10, maxiter=9)


def test_newton_rejects_singular_active_system_and_negative_atol():
    initial = np.array([[0., 1.]])
    with pytest.raises(ContractError, match="Singular"):
        batched_newton(initial, lambda x: x*x-np.array([[0., 4.]]),
                       lambda x: 2*x[:, :, None]*np.eye(2)[None, :, :], atol=1e-10)
    with pytest.raises(ContractError, match="nonnegative"):
        batched_newton(np.ones((1, 1)), lambda x: x, lambda x: np.ones((1, 1, 1)),
                       atol=-1., rtol=2.)


def test_affine_rejects_singular_and_overflowed_stage_systems():
    with pytest.raises(ContractError, match="Singular"):
        affine_stages(np.ones(1), 0., np.array([[0.], [0.], [-9.]]), 1.)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ContractError, match="Nonfinite"):
            affine_stages(np.ones(1), 0., np.array([1e308]), 1e308)


@pytest.mark.parametrize("options", [{"restart": 33}, {"maxiter": 97}])
def test_gmres_enforces_frozen_b_iteration_limits(options):
    with pytest.raises(ContractError, match="limits"):
        gmres(lambda x: x, np.ones(2), rtol=1e-12, atol=0., **options)


def test_empty_local_batches_and_affine_populations_are_supported():
    stages = affine_stages(np.empty(0), 1., 2., 0.1)
    assert stages.shape == (3, 0)
    result, info = batched_newton(np.empty((0, 2)), lambda x: x,
                                  lambda x: np.empty((0, 2, 2)), atol=1e-10)
    assert info["converged"] and result.shape == (0, 2)


def test_gmres_workspace_reuses_banks_and_matches_ephemeral_solve(monkeypatch):
    import retina_sim.group_radau_math as algebra
    matrix = np.array([[4., 1., 0., 0.], [0., 3., 0.5, 0.],
                       [0.1, 0., 2., 0.4], [0., 0.2, 0., 5.]])
    rhs = np.arange(1., 5.)
    precondition = lambda v: v/np.diag(matrix)
    expected, expected_info = gmres(lambda v: matrix @ v, rhs, precondition,
                                    rtol=1e-12, atol=0., restart=2, maxiter=60)
    workspace = algebra.GMRESWorkspace(rhs, restart=2)
    pointers = tuple(a.ctypes.data for a in workspace.arrays)
    assert workspace.owned_bytes == sum(a.nbytes for a in workspace.arrays)
    assert workspace.host_bytes == workspace.owned_bytes
    assert workspace.device_bytes == 0
    assert workspace.shape == rhs.shape and workspace.restart == 2
    def reject_bank_allocation(*args, **kwargs):
        raise AssertionError("The persistent GMRES banks must not be reallocated")
    monkeypatch.setattr(algebra, "_zeros", reject_bank_allocation)
    counts = {}
    result, info = gmres(lambda v: matrix @ v, rhs, precondition, rtol=1e-12,
                         atol=0., restart=2, maxiter=60, workspace=workspace, counters=counts)
    np.testing.assert_array_equal(result, expected)
    assert info == expected_info
    saved = result.copy()
    result2, info2 = gmres(lambda v: matrix @ v, 2*rhs, precondition, rtol=1e-12,
                           atol=0., restart=2, maxiter=60, workspace=workspace, counters=counts)
    np.testing.assert_allclose(result2, 2*expected, atol=1e-14, rtol=1e-14)
    np.testing.assert_array_equal(result, saved)
    assert info2["converged"]
    assert tuple(a.ctypes.data for a in workspace.arrays) == pointers
    assert all(not np.shares_memory(result, a) for a in workspace.arrays)
    assert all(not np.shares_memory(result2, a) for a in workspace.arrays)
    assert workspace.uses == 2
    assert counts["gmres_workspace_uses"] == 2
    assert "gmres_workspace_reuses" not in counts
    assert counts["gmres_workspace_bytes_peak"] == workspace.owned_bytes


@pytest.mark.parametrize("rhs, options", [
    (np.ones(5), {}),
    (np.ones((2, 2)), {}),
    (np.ones(4), {"restart": 3}),
    (np.ones(4, dtype=np.float32), {}),
])
def test_gmres_workspace_rejects_shape_dtype_or_restart_growth(rhs, options):
    from retina_sim.group_radau_math import GMRESWorkspace
    workspace = GMRESWorkspace(np.ones(4), restart=2)
    pointers = tuple(a.ctypes.data for a in workspace.arrays)
    kwargs = dict(rtol=1e-12, atol=0., restart=2, workspace=workspace)
    kwargs.update(options)
    with pytest.raises(ContractError):
        gmres(lambda v: v, rhs, **kwargs)
    assert tuple(a.ctypes.data for a in workspace.arrays) == pointers
    assert workspace.uses == 0


def test_gmres_workspace_rejects_aliases_including_host_rotation_banks():
    from retina_sim.group_radau_math import GMRESWorkspace
    workspace = GMRESWorkspace(np.ones(3), restart=2)
    for rhs in (workspace._x, workspace._residual, workspace._g):
        with pytest.raises(ContractError, match="alias"):
            gmres(lambda v: v, rhs, rtol=1e-12, atol=0., restart=2, workspace=workspace)
    assert workspace.uses == 0


def test_gmres_workspace_releases_after_callback_exception_and_reentry():
    from retina_sim.group_radau_math import GMRESWorkspace
    rhs = np.arange(1., 5.)
    workspace = GMRESWorkspace(rhs, restart=2)
    kwargs = dict(rtol=1e-12, atol=0., restart=2, workspace=workspace)
    def failing(v):
        raise RuntimeError("operator failure")
    with pytest.raises(RuntimeError, match="operator"):
        gmres(failing, rhs, **kwargs)
    def reentrant(v):
        return gmres(lambda x: x, rhs, **kwargs)[0]
    with pytest.raises(ContractError, match="already leased"):
        gmres(reentrant, rhs, **kwargs)
    with pytest.raises(ContractError, match="guard"):
        gmres(lambda v: v, rhs, guard=lambda: False, **kwargs)
    result, info = gmres(lambda v: v, rhs, **kwargs)
    assert info["converged"]
    np.testing.assert_allclose(result, rhs, atol=1e-15, rtol=1e-15)


@pytest.mark.parametrize("rhs, maxiter", [(np.zeros(3), 96), (np.ones(3), 0), (np.ones(3), 1)])
def test_gmres_workspace_early_or_failed_results_never_alias_scratch(rhs, maxiter):
    from retina_sim.group_radau_math import GMRESWorkspace
    workspace = GMRESWorkspace(rhs, restart=2)
    result, _ = gmres(lambda v: 0*v, rhs, rtol=1e-12, atol=0., restart=2,
                      maxiter=maxiter, workspace=workspace)
    assert all(not np.shares_memory(result, a) for a in workspace.arrays)
    saved = result.copy()
    workspace._x[:] = 123.
    np.testing.assert_array_equal(result, saved)


def test_gmres_workspace_rejects_bank_shape_mutation_and_oversize_restart():
    from retina_sim.group_radau_math import GMRESWorkspace
    with pytest.raises(ContractError, match="limit"):
        GMRESWorkspace(np.ones(3), restart=33)
    workspace = GMRESWorkspace(np.ones(3), restart=2)
    workspace._h.shape = (2, 3)
    with pytest.raises(ContractError, match="bank shape"):
        gmres(lambda v: v, np.ones(3), rtol=1e-12, atol=0., restart=2, workspace=workspace)


def test_gmres_workspace_rejects_bank_stride_mutation():
    torch = pytest.importorskip("torch")
    from retina_sim.group_radau_math import GMRESWorkspace
    rhs = torch.ones(3, dtype=torch.float64)
    workspace = GMRESWorkspace(rhs, restart=2)
    workspace._h.as_strided_(workspace._h.shape, (1, 1))
    with pytest.raises(ContractError, match="layout"):
        gmres(lambda v: v, rhs, rtol=1e-12, atol=0., restart=2, workspace=workspace)


def test_gmres_workspace_torch_cpu_parity_residency_and_host_alias():
    torch = pytest.importorskip("torch")
    from retina_sim.group_radau_math import GMRESWorkspace
    matrix = torch.tensor([[4., 1., 0., 0.], [0., 3., 0.5, 0.],
                           [0.1, 0., 2., 0.4], [0., 0.2, 0., 5.]], dtype=torch.float64)
    rhs = torch.arange(1., 5., dtype=torch.float64)
    precondition = lambda v: v/torch.diagonal(matrix)
    expected, expected_info = gmres(lambda v: matrix @ v, rhs, precondition,
                                    rtol=1e-12, atol=0., restart=2, maxiter=60)
    workspace = GMRESWorkspace(rhs, restart=2)
    pointers = tuple(workspace._pointer(a) for a in workspace.arrays)
    result, info = gmres(lambda v: matrix @ v, rhs, precondition, rtol=1e-12,
                         atol=0., restart=2, maxiter=60, workspace=workspace)
    assert torch.equal(result, expected)
    assert info == expected_info
    assert result.dtype == torch.float64 and result.device.type == "cpu"
    assert workspace.device_bytes == 0 and workspace.owned_bytes == workspace.host_bytes
    saved = result.clone()
    second, _ = gmres(lambda v: matrix @ v, 2*rhs, precondition, rtol=1e-12,
                       atol=0., restart=2, maxiter=60, workspace=workspace)
    assert torch.equal(result, saved)
    assert torch.allclose(second, 2*expected, rtol=1e-14, atol=1e-14)
    assert tuple(workspace._pointer(a) for a in workspace.arrays) == pointers
    assert result.untyped_storage().data_ptr() not in pointers
    host_alias_workspace = GMRESWorkspace(torch.ones(3, dtype=torch.float64), restart=2)
    with pytest.raises(ContractError, match="alias"):
        gmres(lambda v: v, torch.from_numpy(host_alias_workspace._g), rtol=1e-12, atol=0.,
              restart=2, workspace=host_alias_workspace)
