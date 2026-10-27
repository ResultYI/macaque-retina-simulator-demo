"""Synthetic CPU equation checks; these do not qualify the fixed retina graph."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from scipy.integrate import solve_ivp

from retina_sim.contracts import ContractError
from retina_sim.group_native_physics import GroupNativePhysics
from retina_sim.group_native_radau import GroupNativeRadau


def fixture(*, auxiliary=True):
    A = sparse.csr_matrix([[.5, .5, 0.], [0., .2, .8]])
    B = sparse.csr_matrix([[1., 0.], [.7, .3], [0., 1.]])
    cones = SimpleNamespace(
        group_count=3, group_ids=np.array(['c0', 'c1', 'c2']),
        recording_auxiliary_group_rows=np.array([0, 0, 2] if auxiliary else [], dtype=np.int64),
        recording_auxiliary_fine_rows=np.array([1, 4, 9] if auxiliary else [], dtype=np.int64))
    hcs = SimpleNamespace(group_count=2, group_ids=np.array(['h0', 'h1']))
    assembly = SimpleNamespace(
        populations={'cones': cones, 'HC': hcs}, artifact_sha256='synthetic-radau-fixture',
        matrix=lambda path: A if path == 'routes/cones_to_HC/mean_weights.npz' else B)
    model = GroupNativePhysics(assembly)
    values = {
        ('cones', 'reduced_native', 'X'): [4., 5., 6.],
        ('cones', 'reduced_native', 'C'): [3., 4., 5.],
        ('cones', 'reduced_native', 'Vis'): [18., 20., 22.],
        ('cones', 'reduced_native', 'g_i'): [.25, .3, .35],
        ('cones', 'reduced_native', 'Vbar'): [17., 19., 21.],
        ('cones', 'analytic_RE', 'R'): [90., 100., 110.],
        ('cones', 'analytic_RE', 'E'): [95., 96., 97.],
        ('HC', 'reduced_native', 'w'): [34., 40.],
        ('HC', 'reduced_native', 'z2'): [27., 30.],
        ('HC', 'reduced_native', 'Vh'): [28., 31.],
        ('cones', 'selected_original_z1_auxiliary', 'z1_aux'): [100., 99., 120.] if auxiliary else [],
    }
    seed = SimpleNamespace(states={key: np.array(value) for key, value in values.items()},
                           time_seconds=0.)
    y, anchor = model.construction_point(seed, np.array([105., 90., 130.]), .001, 7)
    return model, y, anchor


def advance_fixture():
    model, y, anchor = fixture()
    solver = GroupNativeRadau(model, algebraic_rtol=1e-13, algebraic_atol=1e-13)
    before = y.copy()
    result = solver.step(y, 0., .0001, anchor)
    np.testing.assert_array_equal(y, before)
    return model, y, anchor, result


def radau_tableau():
    root6 = np.sqrt(6.)
    A = np.array([
        [(88 - 7*root6)/360, (296 - 169*root6)/1800, (-2 + 3*root6)/225],
        [(296 + 169*root6)/1800, (88 + 7*root6)/360, (-2 - 3*root6)/225],
        [(16 - root6)/36, (16 + root6)/36, 1/9],
    ])
    return A, np.array([(4 - root6)/10, (4 + root6)/10, 1.])


def test_actual_parent_stages_satisfy_complete_native_collocation():
    model, initial, anchor, result = advance_fixture()
    A, c = radau_tableau()
    assert len(result.stage_cache) == 2
    start = initial
    for half in result.stage_cache:
        assert half.stages.shape == (3, model.nstate)
        actual_rhs = np.stack([model.rhs(stage, half.start + node*half.dt, anchor)
                               for stage, node in zip(half.stages, c)])
        np.testing.assert_allclose(half.rhs, actual_rhs, rtol=2e-13, atol=2e-11)
        np.testing.assert_allclose(half.stages, start + half.dt*(A @ actual_rhs),
                                   rtol=2e-13, atol=2e-11)
        start = half.stages[-1]
    np.testing.assert_array_equal(result.y, start)
    assert result.y.dtype == np.float64


def test_small_native_trajectory_and_quintic_match_independent_dop853():
    model, initial, anchor, result = advance_fixture()
    reference = solve_ivp(lambda t, y: model.rhs(y, t, anchor), (0., .0001), initial,
                          method='DOP853', rtol=2e-13, atol=2e-14,
                          max_step=.00001, dense_output=True)
    assert reference.success
    for time in (0., .0000113, .000037, .00005, .0000731, .0001):
        np.testing.assert_allclose(result.dense.at(time), reference.sol(time),
                                   rtol=3e-11, atol=3e-10)
    np.testing.assert_allclose(result.y, reference.y[:, -1], rtol=2e-12, atol=2e-11)


def test_duplicate_auxiliary_drives_keep_independent_filter_memories():
    model, initial, _, result = advance_fixture()
    A, _ = radau_tableau()
    decay = 1000/model.p['tau_1']
    scalar_stages = np.linalg.solve(np.eye(3) + .00005*decay*A, np.ones(3))
    expected_difference = scalar_stages[-1]**2
    aux = model.slices['z1_aux']
    assert model.aux_indices == (0, 0, 2)
    assert model.aux_cell_ids == ('original_cone_row:1', 'original_cone_row:4',
                                  'original_cone_row:9')
    assert initial[aux][0] - initial[aux][1] == 1.
    assert result.y[aux][0] - result.y[aux][1] == pytest.approx(expected_difference,
                                                                            abs=5e-13)


def test_quintic_anchor_derivatives_are_fresh_original_rhs():
    model, initial, anchor, result = advance_fixture()
    midpoint = result.stage_cache[0].stages[-1]
    for time, expected_state in ((0., initial), (.00005, midpoint), (.0001, result.y)):
        np.testing.assert_allclose(result.dense.at(time), expected_state, rtol=0., atol=2e-13)
        expected_rhs = model.rhs(expected_state, time, anchor)
        np.testing.assert_allclose(result.dense.representation_derivative(time), expected_rhs,
                                   rtol=2e-11, atol=2e-8)


def test_native_step_rejects_input_boundary_crossing_and_invalid_domain():
    model, initial, anchor = fixture()
    solver = GroupNativeRadau(model)
    with pytest.raises(ContractError):
        solver.advance(initial, 0., np.nextafter(anchor.input_stop_seconds, np.inf), anchor)
    bad = initial.copy()
    bad[model.slices['g_i']][0] = 0.
    with pytest.raises(ContractError):
        solver.advance(bad, 0., .0001, anchor)



def test_native_jacobian_action_matches_original_equation_difference():
    model, initial, anchor = fixture()
    solver = GroupNativeRadau(model)
    direction = np.linspace(-.15, .15, model.nstate)*np.maximum(np.abs(initial), 1.)
    epsilon = 1e-6
    expected = (model.rhs(initial + epsilon*direction, .000023, anchor)
                - model.rhs(initial - epsilon*direction, .000023, anchor))/(2*epsilon)
    actual = solver.jacobian_action(initial, .000023, anchor, direction)
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=2e-6)


def test_embedded_errors_use_full_native_shifted_jacobian():
    model, initial, anchor, result = advance_fixture()
    A, _ = radau_tableau()
    gamma = float(np.real(np.linalg.eigvals(A)[np.argmin(np.abs(np.imag(np.linalg.eigvals(A))))]))
    root6 = np.sqrt(6.)
    difference_weights = -gamma*np.array([(2 + 3*root6)/6, (2 - 3*root6)/6, 1/3])
    state = initial
    for half in result.stage_cache:
        jacobian = np.empty((model.nstate, model.nstate))
        for column in range(model.nstate):
            epsilon = 1e-6*max(abs(state[column]), 1.)
            offset = np.zeros(model.nstate)
            offset[column] = epsilon
            jacobian[:, column] = (model.rhs(state + offset, half.start, anchor)
                                    - model.rhs(state - offset, half.start, anchor))/(2*epsilon)
        defect = half.dt*(gamma*model.rhs(state, half.start, anchor)
                          + difference_weights @ half.rhs)
        expected = np.linalg.solve(np.eye(model.nstate) - gamma*half.dt*jacobian, defect)
        np.testing.assert_allclose(half.error, expected, rtol=2e-5, atol=3e-12)
        state = half.stages[-1]



def test_advance_accepts_a_real_short_remainder_with_consistent_history():
    model, initial, anchor = fixture()
    solver = GroupNativeRadau(model, algebraic_rtol=1e-13, algebraic_atol=1e-13)
    stop = .000071
    result = solver.advance(initial, 0., stop, anchor, retain_stages=True)
    assert result.accepted_times[0] == 0.
    assert result.accepted_times[-1] == stop
    assert np.all(np.diff(result.accepted_times) > 0.)
    assert len(result.stage_cache) == 2*(len(result.accepted_times) - 1)
    np.testing.assert_array_equal(result.dense.at(stop), result.y)
    with pytest.raises(ContractError):
        result.dense.at(np.nextafter(stop, np.inf))



def test_cpu_torch_tensor_path_matches_numpy_without_cuda(monkeypatch):
    from copy import copy

    torch = pytest.importorskip('torch')
    from retina_sim import resident_cusparse
    from retina_sim.cone_hc_reduced_native import AnalyticREAnchor

    def reject_cuda_initialization():
        pytest.fail('Synthetic tensor parity must not initialize CUDA')

    monkeypatch.setattr(torch.cuda, '_lazy_init', reject_cuda_initialization)

    class CpuCSRPlan:
        def __init__(self, matrix, api, torch_module):
            assert matrix.device.type == 'cpu'
            self.matrix, self.api, self.torch = matrix, api, torch_module
            self.device, self.shape, self.dtype = matrix.device, matrix.shape, matrix.dtype

        def mv(self, vector):
            assert vector.device.type == 'cpu'
            assert vector.dtype == self.torch.float64
            return self.torch.mv(self.matrix, vector)

    monkeypatch.setattr(resident_cusparse, 'NativeCSRPlan', CpuCSRPlan)
    model, initial, anchor = fixture()
    expected = GroupNativeRadau(model).step(initial, 0., .0001, anchor)
    tensor_model = copy(model)
    tensor_model.torch, tensor_model.device = torch, torch.device('cpu')
    tensor_model.statistics = dict(model.statistics)

    def plan(matrix):
        tensor = torch.sparse_csr_tensor(
            torch.tensor(matrix.indptr.copy(), dtype=torch.int64),
            torch.tensor(matrix.indices.copy(), dtype=torch.int64),
            torch.tensor(matrix.data.copy(), dtype=torch.float64),
            size=matrix.shape, dtype=torch.float64, device='cpu')
        return CpuCSRPlan(tensor, None, torch)

    tensor_model.A, tensor_model.B = plan(model.A), plan(model.B)
    tensor_model._aux = torch.tensor(model.aux_indices, dtype=torch.int64)
    tensor_model._topology = tensor_model._topology_signature()
    tensor_initial = torch.tensor(initial.copy(), dtype=torch.float64)
    tensor_anchor = AnalyticREAnchor(
        anchor.time_seconds, torch.tensor(anchor.R.copy()), torch.tensor(anchor.E.copy()),
        torch.tensor(anchor.light.copy()), anchor.input_stop_seconds, anchor.input_epoch)
    solver = GroupNativeRadau(
        tensor_model, error_weights=torch.full((model.nstate,), 1e-5, dtype=torch.float64),
        budgets=torch.full((model.nstate,), 1e-3, dtype=torch.float64))
    actual = solver.step(tensor_initial, 0., .0001, tensor_anchor)
    assert actual.y.device.type == 'cpu'
    assert actual.y.dtype == torch.float64
    np.testing.assert_allclose(actual.y.numpy(), expected.y, rtol=3e-13, atol=3e-12)
    for torch_half, numpy_half in zip(actual.stage_cache, expected.stage_cache):
        np.testing.assert_allclose(torch_half.stages.numpy(), numpy_half.stages,
                                   rtol=3e-13, atol=3e-12)
        np.testing.assert_allclose(torch_half.error.numpy(), numpy_half.error,
                                   rtol=1e-5, atol=3e-12)
    np.testing.assert_allclose(actual.dense.at(.000037).numpy(), expected.dense.at(.000037),
                               rtol=3e-13, atol=3e-12)



def test_controller_keeps_frozen_scales_and_cumulative_error_ledger():
    model, initial, anchor = fixture()
    weights = np.full(model.nstate, 1e-5)
    budgets = np.full(model.nstate, 1e-3)
    solver = GroupNativeRadau(model, error_weights=weights, budgets=budgets)
    weights.fill(99.)
    budgets.fill(99.)
    first = solver.advance(initial, 0., .0001, anchor)
    first_ledger = solver.error_ledger.copy()
    second = solver.advance(first.y, .0001, .0002, anchor)
    np.testing.assert_array_equal(solver.error_weights, np.full(model.nstate, 1e-5))
    np.testing.assert_array_equal(solver.budgets, np.full(model.nstate, 1e-3))
    np.testing.assert_allclose(first_ledger, first.error, rtol=0., atol=0.)
    np.testing.assert_allclose(solver.error_ledger, first.error + second.error,
                               rtol=3e-15, atol=1e-25)
    assert second.metrics['coordinate_error_ratio'] > 0.
    assert second.metrics['certified'] is False


def test_embedded_third_order_difference_is_only_a_budget_diagnostic():
    model, initial, anchor = fixture()
    solver = GroupNativeRadau(model, error_weights=np.ones(model.nstate),
                             budgets=np.full(model.nstate, 1e-30))
    result = solver.advance(initial, 0., .0001, anchor)
    assert result.accepted_times[-1] == .0001
    assert result.metrics['error_rms'] <= 1.
    assert result.metrics['coordinate_error_ratio'] > 1.
    assert result.metrics['coordinate_error_ratio_is_diagnostic'] is True
    assert result.metrics['certified'] is False



def test_controller_restore_reproduces_continuation_and_rejects_changed_policy():
    import json

    model, initial, anchor = fixture()
    options = dict(error_weights=np.full(model.nstate, 1e-5),
                   budgets=np.full(model.nstate, 1e-3))
    original = GroupNativeRadau(model, **options)
    first = original.advance(initial, 0., .000071, anchor)
    saved = original.controller_state()
    saved['identity'] = json.loads(json.dumps(saved['identity']))
    independent = original.controller_state()
    independent['identity']['original_auxiliary_ids'][0] = 'mutated-export'
    assert original.identity['original_auxiliary_ids'][0] == model.aux_cell_ids[0]
    restored = GroupNativeRadau(model, **options)
    restored.restore_controller(saved)
    expected = original.advance(first.y, .000071, .00017, anchor)
    actual = restored.advance(first.y, .000071, .00017, anchor)
    assert actual.accepted_times == expected.accepted_times
    np.testing.assert_array_equal(actual.y, expected.y)
    np.testing.assert_array_equal(restored.error_ledger, original.error_ledger)
    assert actual.metrics == expected.metrics
    invalid_identity = dict(saved, identity=dict(saved['identity'], method='different-method'))
    with pytest.raises(ContractError, match='identity'):
        GroupNativeRadau(model, **options).restore_controller(invalid_identity)
    invalid_weights = dict(saved, error_weights=2*saved['error_weights'])
    with pytest.raises(ContractError, match='frozen coordinate scale'):
        GroupNativeRadau(model, **options).restore_controller(invalid_weights)


def adapter_fixture(stop=.00025):
    from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
    from retina_sim.group_native_radau_adapter import GroupNativeRadauBackend

    model, initial, original_anchor = fixture()
    anchor = AnalyticREAnchor(0., original_anchor.R, original_anchor.E,
                              original_anchor.light, stop, original_anchor.input_epoch)
    backend = GroupNativeRadauBackend(model)
    return model, initial, anchor, backend


def test_adapter_interval_lifecycle_preserves_state_and_original_equations():
    model, initial, anchor, backend = adapter_fixture()
    original = initial.copy()
    state = backend.initial_state(initial, anchor)
    initial[0] += 1.
    np.testing.assert_array_equal(state.y, original)
    result = backend.advance_one(state, anchor.input_stop_seconds)
    assert len(result.intervals) == 1
    interval = result.intervals[0]
    assert result.state.time_seconds == anchor.input_stop_seconds
    assert interval.start == 0. and interval.stop == anchor.input_stop_seconds
    np.testing.assert_array_equal(interval.evaluate(0., 'right'), original)
    np.testing.assert_array_equal(interval.evaluate(interval.stop, 'left'), result.state.y)
    time = .000137
    rows = np.array([4, 0, 7])
    value = interval.evaluate(time)
    expected = model.evaluate(value, time, anchor)
    np.testing.assert_array_equal(interval.evaluate(time, rows=rows), value[rows])
    np.testing.assert_array_equal(interval.rhs_at(time), expected.rhs)
    np.testing.assert_array_equal(interval.rhs_at(time, rows=rows), expected.rhs[rows])
    outputs = interval.outputs_at(time)
    for name in ('R', 'E', 'q', 'release', 'pooled_aI'):
        np.testing.assert_array_equal(outputs[name], expected.outputs[name])
    for time, side in ((0., 'left'), (interval.stop, 'right'), (-1e-9, 'right')):
        with pytest.raises(ContractError):
            interval.evaluate(time, side)
    backend.close()
    with pytest.raises(ContractError):
        backend.advance_one(result.state, .0005)


def test_adapter_readout_chunks_preserve_selected_rows_and_all_six_anchors():
    model, initial, anchor, backend = adapter_fixture()
    result = backend.advance_one(backend.initial_state(initial, anchor), anchor.input_stop_seconds)
    interval = result.intervals[0]
    cone_rows, hc_rows = np.array([2, 0]), np.array([1])
    packets = tuple(interval.readout_chunks(cone_rows, hc_rows))
    assert len(packets) == len(interval.dense.segments)
    assert len(packets) >= 3
    for packet, segment in zip(packets, interval.dense.segments):
        for name in ('Vis', 'Vbar', 'Vh'):
            representation = packet[name]
            rows = hc_rows if name == 'Vh' else cone_rows
            assert representation.kind == 'quintic_hermite'
            assert representation.t0 == segment.start
            assert representation.t1 == segment.stop
            assert representation.input_epoch == anchor.input_epoch
            np.testing.assert_array_equal(representation.state_ids, rows)
            for attribute in ('y0', 'f0', 'ym', 'fm', 'y1', 'f1'):
                np.testing.assert_array_equal(getattr(representation, attribute),
                    getattr(segment, attribute)[model.slices[name]][rows])
    for name, curve in interval.curves.items():
        assert curve.segments[0].t0 == interval.start
        assert curve.segments[-1].t1 == interval.stop
        np.testing.assert_array_equal(curve.at(.000137),
                                       interval.evaluate(.000137)[model.slices[name]])
    assert all(packet['Vh'] is None for packet in interval.readout_chunks(
        cone_rows, np.array([], dtype=np.int64)))
    for invalid in (np.array([0, 0]), np.array([3]), np.array([.5]), np.array([], dtype=np.int64)):
        with pytest.raises(ContractError):
            tuple(interval.readout_chunks(invalid, hc_rows))
    backend.close()


def test_adapter_input_refresh_keeps_memory_at_exact_boundary():
    model, initial, anchor, backend = adapter_fixture()
    state = backend.initial_state(initial, anchor)
    with pytest.raises(ContractError):
        backend.refresh_input(state, anchor.light*2, .0005, anchor.input_epoch+1)
    reached = backend.advance_one(state, anchor.input_stop_seconds).state
    with pytest.raises(ContractError):
        backend.refresh_input(reached, anchor.light*2, .0005, anchor.input_epoch)
    updated = backend.refresh_input(reached, anchor.light*2, .0005, anchor.input_epoch+1)
    np.testing.assert_array_equal(updated.y, reached.y)
    before_re = anchor.at(reached.time_seconds)
    after_re = updated.anchor.at(reached.time_seconds)
    for before, after in zip(before_re, after_re):
        np.testing.assert_array_equal(after, before)
    assert updated.anchor.input_epoch == anchor.input_epoch+1
    np.testing.assert_array_equal(updated.anchor.light, 2*anchor.light)
    next_result = backend.advance_one(updated, .0005)
    assert next_result.state.time_seconds == .0005
    backend.close()



def test_adapter_payload_restores_exact_anchors_and_rejects_malformed_records():
    from copy import deepcopy

    model, initial, anchor, backend = adapter_fixture()
    interval = backend.advance_one(backend.initial_state(initial, anchor),
                                   anchor.input_stop_seconds).intervals[0]
    payload = interval.payload()
    restored = backend.interval_from_payload(payload)
    assert restored.representation_id == interval.representation_id
    assert restored.source_version == interval.source_version
    assert len(restored.dense.segments) == len(interval.dense.segments)
    for before, after in zip(interval.dense.segments, restored.dense.segments):
        assert (after.start, after.stop) == (before.start, before.stop)
        for name in ('y0', 'f0', 'ym', 'fm', 'y1', 'f1'):
            np.testing.assert_array_equal(getattr(after, name), getattr(before, name))
    for time in (0., .0000137, .0001, .000137, interval.stop):
        side = 'left' if time == interval.stop else 'right'
        np.testing.assert_array_equal(restored.evaluate(time, side), interval.evaluate(time, side))
        np.testing.assert_array_equal(restored.representation_derivative(time, side),
                                       interval.representation_derivative(time, side))
        np.testing.assert_array_equal(restored.rhs_at(time, side), interval.rhs_at(time, side))
    for field, value in (('source_version', 'different-source'), ('dtype', 'float32'),
                         ('start_hex', float(.000001).hex()), ('extra_field', True)):
        invalid = deepcopy(payload)
        invalid[field] = value
        with pytest.raises(ContractError):
            backend.interval_from_payload(invalid)
    invalid = deepcopy(payload)
    invalid['segments'][0]['ym'] = invalid['segments'][0]['ym'][:-1]
    with pytest.raises(ContractError, match='shape'):
        backend.interval_from_payload(invalid)
    invalid = deepcopy(payload)
    del invalid['segments'][0]['fm']
    with pytest.raises(ContractError, match='schema'):
        backend.interval_from_payload(invalid)
    invalid = deepcopy(payload)
    invalid['segments'][0]['kind'] = 'dop853_dense'
    with pytest.raises(ContractError):
        backend.interval_from_payload(invalid)
    backend.close()


def test_adapter_preserves_existing_reference_identity():
    from retina_sim.native_multirate import NativeMultirateBackend

    model, _, _, backend = adapter_fixture()
    reference_backend = NativeMultirateBackend(model, rtol=1e-6, atol=1e-9)
    assert backend.reference_identity == reference_backend.reference_identity
    assert backend.identity != reference_backend.identity
    assert backend.identity != backend.reference_identity
    reference_backend.close()
    backend.close()


@pytest.mark.parametrize('time_type', [float, np.float64], ids=['python-float', 'numpy-float64'])
def test_adapter_shifted_time_keeps_exact_contiguous_macro_boundaries(time_type):
    from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
    from retina_sim.group_native_radau_adapter import GroupNativeRadauBackend

    model, initial, original_anchor = fixture()
    start = time_type(4.993)
    stop = time_type(start + .0005)
    anchor = AnalyticREAnchor(start, original_anchor.R, original_anchor.E,
                              original_anchor.light, stop, original_anchor.input_epoch)
    backend = GroupNativeRadauBackend(model)
    result = backend.advance_one(backend.initial_state(initial, anchor), stop)
    interval = result.intervals[0]
    assert result.state.time_seconds == stop
    assert interval.start == start and interval.stop == stop
    assert type(result.state.time_seconds) is float
    restored_core = GroupNativeRadau(model)
    restored_core.restore_controller(backend.core.controller_state())
    assert type(restored_core.controller_state()['window_end']) is float
    np.testing.assert_array_equal(restored_core.error_ledger, backend.core.error_ledger)
    previous = start
    for segment in interval.dense.segments:
        assert segment.start == previous
        assert segment.start < segment.stop <= segment.start + .0001
        previous = segment.stop
    assert previous == stop
    restored = backend.interval_from_payload(interval.payload())
    for time in (start, start + .000137, stop):
        side = 'left' if time == stop else 'right'
        np.testing.assert_array_equal(restored.evaluate(time, side), interval.evaluate(time, side))
    backend.close()


@pytest.mark.parametrize('invalid', [np.nan, np.inf], ids=['nan', 'infinity'])
def test_controller_rejects_nonfinite_filtered_error_without_accepting(monkeypatch, invalid):
    model, initial, anchor = fixture()
    solver = GroupNativeRadau(model)
    monkeypatch.setattr(solver, '_filtered_error',
                        lambda *args: np.full(model.nstate, invalid))
    with pytest.raises(ContractError, match='[Nn]onfinite'):
        solver.advance(initial, 0., .0001, anchor)
    assert solver.statistics.get('accepted_macros', 0) == 0
    np.testing.assert_array_equal(solver.error_ledger, np.zeros(model.nstate))



def test_cpu_preconditioner_survives_torch_first_import_in_fresh_process():
    """Torch's OpenMP runtime must not make the CPU fallback abort the process."""
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip('torch')
    script = """
import os
if os.name == 'nt':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
import torch
assert not torch.cuda.is_initialized()
def reject_cuda(*args, **kwargs):
    raise AssertionError('CPU preconditioner regression must never initialize CUDA')
torch.cuda._lazy_init = reject_cuda
import numpy as np
from retina_sim.group_native_radau import GroupNativeRadau
solver = object.__new__(GroupNativeRadau)
solver.torch = None
solver.statistics = {}
for width in (9, 3):
    matrix = np.stack((2*np.eye(width)+np.diag(np.full(width-1, .4), 1),
                       3*np.eye(width)-np.diag(np.full(width-1, .1), -1)))
    original = matrix.copy()
    solve = solver._factor(matrix)
    rhs = np.arange(2*width, dtype=np.float64).reshape(2, width)
    expected = np.linalg.solve(original, rhs[..., None])[..., 0]
    first = solve(rhs)
    np.testing.assert_allclose(first, expected, rtol=3e-15, atol=3e-15)
    np.testing.assert_allclose((original @ first[..., None])[..., 0], rhs,
                               rtol=3e-15, atol=3e-15)
    matrix.fill(np.nan)
    np.testing.assert_array_equal(solve(rhs), first)
    np.testing.assert_allclose(solve(2*rhs+1), 2*first+solve(np.ones_like(rhs)),
                               rtol=3e-15, atol=3e-15)
assert solver.statistics['preconditioner_factorizations'] == 2
assert not torch.cuda.is_initialized()
print('fixed CPU preconditioners passed after Torch import')
"""
    result = subprocess.run([sys.executable, '-B', '-X', 'faulthandler', '-c', script],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    assert 'fixed CPU preconditioners passed' in result.stdout


@pytest.mark.parametrize('matrix', [np.zeros((2, 3, 3)), np.full((2, 9, 9), np.nan)])
def test_cpu_preconditioner_rejects_singular_or_nonfinite_blocks(matrix):
    solver = object.__new__(GroupNativeRadau)
    solver.torch = None
    solver.statistics = {}
    with pytest.raises(ContractError, match='singular|nonfinite'):
        solver._factor(matrix)



def native_plan_fixture(model):
    from retina_sim.group_scc_plan import _native_layout

    declarations = []
    for population, mechanism, names, count in (
            ('cones', 'reduced_native', ('X', 'C', 'Vis', 'g_i', 'Vbar'), model.nc),
            ('cones', 'analytic_RE', ('R', 'E'), model.nc),
            ('HC', 'reduced_native', ('w', 'z2', 'Vh'), model.nh)):
        declarations.extend(SimpleNamespace(population=population, mechanism=mechanism,
                                             name=name, count=count) for name in names)
    cones = SimpleNamespace(
        group_count=model.nc,
        recording_auxiliary_group_rows=np.array(model.aux_indices, dtype=np.int64),
        recording_auxiliary_fine_rows=np.array(
            [int(name.split(':')[1]) for name in model.aux_cell_ids], dtype=np.int64))
    assembly = SimpleNamespace(
        state_blocks=tuple(declarations),
        auxiliary_z1_blocks=(SimpleNamespace(population='cones',
            mechanism='selected_original_z1_auxiliary', name='z1_aux', count=len(model.aux_indices)),),
        populations={'cones': cones, 'HC': SimpleNamespace(group_count=model.nh)},
        artifact_sha256=model.identity['artifact_sha256'],
        matrix=lambda path: model.model.A if path == 'routes/cones_to_HC/mean_weights.npz' else model.model.B)
    return _native_layout(SimpleNamespace(assembly=assembly))


def test_native_plan_drives_actual_stages_independently_of_block_storage(monkeypatch):
    from dataclasses import replace

    model, initial, anchor = fixture()
    plan = native_plan_fixture(model)
    first = GroupNativeRadau(model, native_plan=plan)
    permuted = GroupNativeRadau(model, native_plan=replace(plan, blocks=plan.blocks[::-1]))
    assert first.execution_order == permuted.execution_order
    assert first.execution_order == ('native_RE', 'native_XC', 'native_Vis_gi',
                                     'native_Vbar', 'native_HC', 'native_z1_aux')
    assert first.schedule_identity == permuted.schedule_identity
    assert first.identity['native_schedule_identity'] == first.schedule_identity
    assert first.identity['native_layout_identity'] == plan.identity
    assert first.identity['native_schedule_provenance'] == plan.provenance
    calls = []
    execute = first._stage_family

    def recording_family(name, *args, **kwargs):
        calls.append(name)
        return execute(name, *args, **kwargs)

    monkeypatch.setattr(first, '_stage_family', recording_family)
    expected = first.step(initial, 0., .0001, anchor)
    actual = permuted.step(initial, 0., .0001, anchor)
    assert tuple(calls) == first.execution_order*2
    np.testing.assert_array_equal(actual.y, expected.y)
    for actual_half, expected_half in zip(actual.stage_cache, expected.stage_cache):
        np.testing.assert_array_equal(actual_half.stages, expected_half.stages)
        np.testing.assert_array_equal(actual_half.rhs, expected_half.rhs)


def invalid_native_plan(plan, fault):
    from dataclasses import replace

    blocks = list(plan.blocks)
    xc_index = next(index for index, block in enumerate(blocks) if block.name == 'native_XC')
    xc = blocks[xc_index]
    if fault == 'missing-block':
        return replace(plan, blocks=tuple(block for block in blocks if block is not xc))
    if fault == 'extra-block':
        return replace(plan, blocks=tuple(blocks) + (replace(xc, name='future_native_state'),))
    if fault == 'duplicate-block':
        return replace(plan, blocks=tuple(blocks) + (xc,))
    if fault == 'missing-parent':
        blocks[xc_index] = replace(xc, parent_names=())
    elif fault == 'unknown-parent':
        blocks[xc_index] = replace(xc, parent_names=('native_RE', 'future_native_feedback'))
    elif fault == 'future-feedback':
        blocks[xc_index] = replace(xc, parent_names=('native_RE', 'native_HC'))
    elif fault == 'wrong-block-key':
        blocks[xc_index] = replace(xc, state_keys=(('cones', 'reduced_native', 'future_X'), xc.state_keys[1]))
    elif fault == 'layout-provenance':
        return replace(plan, provenance='inner typed IR')
    elif fault == 'block-provenance':
        blocks[xc_index] = replace(xc, provenance='inner typed IR')
    elif fault == 'scalar-count':
        blocks[xc_index] = replace(xc, scalar_count=xc.scalar_count + 1)
    elif fault == 'component-count':
        blocks[xc_index] = replace(xc, component_count=xc.component_count + 1)
    elif fault == 'component-sizes':
        blocks[xc_index] = replace(xc, component_sizes=np.array([1, 2, 3], dtype=np.int64))
    elif fault == 'hc-component-split':
        index = next(index for index, block in enumerate(blocks) if block.name == 'native_HC')
        blocks[index] = replace(blocks[index], component_count=2,
                                component_sizes=np.array([3, 3], dtype=np.int64))
    elif fault == 'wrong-registry-key':
        previous, replacement = plan.full_keys[0], ('cones', 'future_mechanism', 'X')
        slices = dict(plan.full_slices)
        slices[replacement] = slices.pop(previous)
        return replace(plan, full_keys=(replacement,) + plan.full_keys[1:], full_slices=slices)
    elif fault == 'missing-registry-key':
        slices = dict(plan.full_slices)
        del slices[plan.full_keys[0]]
        return replace(plan, full_keys=plan.full_keys[1:], full_slices=slices)
    elif fault == 'overlapping-slice':
        slices = dict(plan.full_slices)
        original = slices[plan.full_keys[1]]
        slices[plan.full_keys[1]] = slice(original.start - 1, original.stop - 1)
        return replace(plan, full_slices=slices)
    elif fault == 'analytic-overlap':
        return replace(plan, analytic_indices=np.r_[plan.analytic_indices, plan.numerical_indices[0]])
    elif fault == 'analytic-missing':
        return replace(plan, analytic_indices=plan.analytic_indices[:-1])
    elif fault == 'analytic-wrong-mechanism':
        analytic, numerical = plan.analytic_indices.copy(), plan.numerical_indices.copy()
        analytic[0], numerical[0] = numerical[0], analytic[0]
        return replace(plan, analytic_indices=np.sort(analytic), numerical_indices=np.sort(numerical))
    elif fault == 'numerical-missing':
        return replace(plan, numerical_indices=plan.numerical_indices[:-1])
    elif fault == 'numerical-extra':
        count = max(part.stop for part in plan.full_slices.values())
        return replace(plan, numerical_indices=np.r_[plan.numerical_indices, count])
    elif fault == 'numerical-duplicate':
        return replace(plan, numerical_indices=np.r_[plan.numerical_indices[:-1], plan.numerical_indices[0]])
    else:
        raise AssertionError('Unknown native plan test mutation: ' + fault)
    return replace(plan, blocks=tuple(blocks))


@pytest.mark.parametrize('fault', [
    'missing-block', 'extra-block', 'duplicate-block', 'missing-parent', 'unknown-parent',
    'future-feedback', 'wrong-block-key', 'layout-provenance', 'block-provenance',
    'scalar-count', 'component-count', 'component-sizes',
    'hc-component-split', 'wrong-registry-key', 'missing-registry-key', 'overlapping-slice',
    'analytic-overlap', 'analytic-missing', 'analytic-wrong-mechanism',
    'numerical-missing', 'numerical-extra', 'numerical-duplicate'])
def test_native_plan_malformed_descriptors_and_partitions_fail_closed(fault):
    model, _, _ = fixture()
    plan = native_plan_fixture(model)
    malformed = invalid_native_plan(plan, fault)
    assert malformed.identity == plan.identity
    with pytest.raises(ContractError):
        GroupNativeRadau(model, native_plan=malformed)


def test_native_source_plan_accepts_zero_auxiliary_rows_without_inventing_a_block():
    model, initial, anchor = fixture(auxiliary=False)
    plan = native_plan_fixture(model)
    assert not any(block.name == 'native_z1_aux' for block in plan.blocks)
    solver = GroupNativeRadau(model, native_plan=plan)
    assert solver.execution_order[-1] == 'native_HC'
    result = solver.step(initial, 0., .0001, anchor)
    complete_model, complete_initial, complete_anchor = fixture()
    complete = GroupNativeRadau(complete_model, native_plan=native_plan_fixture(complete_model))
    expected = complete.step(complete_initial, 0., .0001, complete_anchor)
    assert result.y.shape == (5*model.nc + 3*model.nh,)
    np.testing.assert_array_equal(result.y, expected.y[:len(result.y)])
    assert model.aux_cell_ids == ()


def test_native_cuda_metadata_requires_plan_before_any_device_access():
    from copy import copy

    model, _, _ = fixture()
    fake_device_model = copy(model)

    class NoDeviceAccess:
        def __getattr__(self, name):
            pytest.fail('Missing NativeLayout must fail before device access: ' + name)

    fake_device_model.torch = NoDeviceAccess()
    fake_device_model.device = SimpleNamespace(type='cuda')
    with pytest.raises(ContractError, match='(?i)(plan|layout)'):
        GroupNativeRadau(fake_device_model)


def test_native_gmres_banks_are_reused_and_accepted_outputs_do_not_alias_them():
    model, initial, anchor = fixture()
    solver = GroupNativeRadau(model, native_plan=native_plan_fixture(model))
    banks = solver.gmres_workspaces
    assert set(banks) == {1, 3}
    assert banks[3].shape == (3, 3, model.nh)
    assert banks[1].shape == (1, 3, model.nh)
    pointers = {key: tuple(array.ctypes.data for array in workspace.arrays)
                for key, workspace in banks.items()}
    accounting = solver.workspace_metrics()
    assert accounting['count'] == 2
    assert accounting['owned_bytes'] == sum(workspace.owned_bytes for workspace in banks.values())
    assert accounting['host_bytes'] == accounting['owned_bytes']
    assert accounting['device_bytes'] == 0
    first = solver.step(initial, 0., .0001, anchor)
    saved_y, saved_error = first.y.copy(), first.error.copy()
    saved_dense = first.dense.at(.000037)
    uses = {key: workspace.uses for key, workspace in banks.items()}
    assert all(count > 0 for count in uses.values())
    second = solver.step(first.y, .0001, .0001, anchor)
    assert solver.workspace_metrics() == accounting
    for key, workspace in banks.items():
        assert solver.gmres_workspaces[key] is workspace
        assert tuple(array.ctypes.data for array in workspace.arrays) == pointers[key]
        assert workspace.uses > uses[key]
        for array in workspace.arrays:
            assert not np.shares_memory(first.y, array)
            assert not np.shares_memory(first.error, array)
            assert not np.shares_memory(second.y, array)
    np.testing.assert_array_equal(first.y, saved_y)
    np.testing.assert_array_equal(first.error, saved_error)
    np.testing.assert_array_equal(first.dense.at(.000037), saved_dense)


