"""Representation queries/checkpoints only, without a RetiPath model advance."""
import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import DOP853

from retina_sim.contracts import ContractError
from retina_sim.continuous_state import (AcceptedSeries, DiscreteHold, Dop853Dense,
    ExponentialForced, AffineMembraneState, from_packet, load_representation, save_representation)


def test_dop853_packet_reproduces_scipy_dense_at_internal_nodes_and_derivative():
    def equation(t, y):
        return np.array([-3.*y[0], 2.*t])
    solver = DOP853(equation, 0., np.array([2., .1]), .001, max_step=.0001,
                    rtol=1e-9, atol=1e-12)
    solver.step()
    dense = solver.dense_output()
    representation = Dop853Dense.from_scipy(dense, ['a', 'b'])
    for t in np.linspace(dense.t_old, dense.t, 19):
        side = 'left' if t == dense.t else 'right'
        np.testing.assert_array_equal(representation.evaluate(t, side), dense(t))
        np.testing.assert_allclose(representation.representation_derivative(t, side), equation(t, dense(t)),
                                   rtol=2e-10, atol=3e-11)
    assert representation.kind == 'dop853_dense'
    assert representation.available_until == dense.t
    with pytest.raises(ContractError, match='original-equation'):
        representation.rhs_at(.5*(dense.t_old+dense.t), 'right')


def test_forced_exponential_uses_one_expression_for_endpoints_and_internal_points():
    coefficients = np.zeros((8, 2)); coefficients[0] = [2., 1.]; coefficients[7, 1] = .3
    packet = ExponentialForced(t0=0., t1=.01, state_ids=np.array([4, 9]),
        y0=np.array([.5, .25]), coefficients=coefficients, decay=np.array([3., 0.]))
    for t in np.linspace(0., .01, 17):
        side = 'left' if t == .01 else 'right'
        expected = [.5*np.exp(-3*t)+2*(-np.expm1(-3*t))/3,
                    .25+t+.3*t**8/(8*.01**7)]
        np.testing.assert_allclose(packet.evaluate(t, side), expected, rtol=2e-14, atol=1e-15)
        expected_rhs = [2.-3.*expected[0], 1.+.3*(t/.01)**7]
        np.testing.assert_allclose(packet.representation_derivative(t, side), expected_rhs, rtol=2e-14, atol=1e-14)


def test_exact_boundary_side_and_adjacent_ULP_without_snapping():
    first = DiscreteHold(t0=0., t1=.001, state_ids=['x'], y0=np.array([1.]))
    second = DiscreteHold(t0=.001, t1=.002, state_ids=['x'], y0=np.array([7.]), epoch=1)
    series = AcceptedSeries([first, second])
    assert series.evaluate(.001, 'left')[0] == 1.
    assert series.evaluate(.001, 'right')[0] == 7.
    assert series.evaluate(np.nextafter(.001, 0), 'right')[0] == 1.
    assert series.evaluate(np.nextafter(.001, np.inf), 'left')[0] == 7.
    for t, side in ((0., 'left'), (.002, 'right'), (np.nextafter(0., -np.inf), 'right'),
                    (np.nextafter(.002, np.inf), 'left')):
        with pytest.raises(ContractError):
            series.evaluate(t, side)
    with pytest.raises(ContractError, match='side'):
        first.evaluate(.001, 'right')
    with pytest.raises(ContractError, match='contiguous'):
        AcceptedSeries([first, DiscreteHold(t0=np.nextafter(.001, np.inf), t1=.002,
                                           state_ids=['x'], y0=np.array([7.]))])


def test_unaccepted_prediction_is_never_visible_or_checkpointed(tmp_path):
    prediction = DiscreteHold(t0=0., t1=1., state_ids=['x'], y0=np.array([2.]), accepted=False)
    for operation in (lambda: prediction.evaluate(.2, 'right'), lambda: prediction.rhs_at(.2, 'right'),
                      lambda: prediction.available_until, lambda: AcceptedSeries([prediction]),
                      lambda: save_representation(tmp_path/'prediction', prediction)):
        with pytest.raises(ContractError):
            operation()
    assert not (tmp_path/'prediction').exists()


def test_ids_owned_immutable_and_reordered_without_per_cell_dicts():
    original = np.array([4., 9.])
    packet = DiscreteHold(t0=0., t1=1., state_ids=['b', 'a'], y0=original)
    original[:] = 123.
    np.testing.assert_array_equal(packet.evaluate(.2, 'right', ['a', 'b']), [9., 4.])
    with pytest.raises(ValueError):
        packet.y0.flags.writeable = True
    with pytest.raises(ValueError):
        packet.state_ids.flags.writeable = True
    for ids in (['a', 'a'], ['missing'], [1]):
        with pytest.raises(ContractError):
            packet.evaluate(.2, 'right', ids)
    with pytest.raises(ContractError, match='Duplicate'):
        DiscreteHold(t0=0., t1=1., state_ids=['a', 'a'], y0=original)
    with pytest.raises(ContractError, match='FP64'):
        DiscreteHold(t0=0., t1=1., state_ids=['a'], y0=np.array([1.], np.float32))


@pytest.mark.parametrize('kind', ['discrete_hold', 'exponential_forced', 'dop853_dense', 'integrating_factor'])
def test_representation_only_checkpoint_roundtrip_all_kinds(tmp_path, kind):
    common = dict(t0=.001, t1=.002, state_ids=np.array([3, 8]), y0=np.array([2., 3.]), epoch=4)
    if kind == 'discrete_hold':
        original = DiscreteHold(**common)
    elif kind == 'exponential_forced':
        original = ExponentialForced(**common, decay=np.array([2., 3.]), coefficients=np.arange(16., dtype=np.float64).reshape(8, 2))
    elif kind == 'dop853_dense':
        original = Dop853Dense(**common, coefficients=np.arange(14., dtype=np.float64).reshape(7, 2))
    else:
        a = np.zeros((8, 2)); a[0] = [2., 3.]; a[1] = [1., .2]
        b = np.zeros((8, 2)); b[0] = [3., 4.]; b[1] = [.4, .8]
        original = AffineMembraneState(**common, a_coefficients=a, b_coefficients=b,
            quadrature_budget=np.array([1e-11, 1e-11]), state_unit='V')
    folder = tmp_path/kind
    save_representation(folder, original)
    restored = load_representation(folder)
    assert original.representation_id == restored.representation_id
    for t, side in ((.001, 'right'), (.00137, 'left'), (.002, 'left')):
        np.testing.assert_array_equal(original.evaluate(t, side), restored.evaluate(t, side))
    meta = json.loads((folder/'manifest.json').read_text())
    assert meta['scope'] == 'representation_only_not_runner_checkpoint'
    assert meta['full_runner_restore_supported'] is False
    result = subprocess.run([sys.executable, '-B', '-c',
        "import json,sys; from retina_sim.continuous_state import load_representation; "
        "r=load_representation(sys.argv[1]); print(json.dumps(dict(id=r.representation_id, "
        "value=r.evaluate(.00137,'left').tolist())))", str(folder)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    fresh = json.loads(result.stdout)
    assert fresh['id'] == original.representation_id
    np.testing.assert_array_equal(fresh['value'], original.evaluate(.00137, 'left'))
    with pytest.raises(ContractError, match='Fresh'):
        save_representation(folder, original)


def test_original_rhs_callback_is_separate_and_must_be_explicitly_rebound(tmp_path):
    calls = []
    def true_rhs(t, side, y):
        calls.append((t, side, y.copy()))
        return y+100.
    original = DiscreteHold(t0=0., t1=1., state_ids=['b', 'a'], y0=np.array([2., 5.]),
                            original_rhs=true_rhs, original_rhs_id='original-equations-sha256')
    np.testing.assert_array_equal(original.representation_derivative(.3, 'left'), [0., 0.])
    np.testing.assert_array_equal(original.rhs_at(.3, 'left', ['a']), [105.])
    assert calls[0][0:2] == (.3, 'left')
    folder = tmp_path/'rhs'; save_representation(folder, original)
    with pytest.raises(ContractError, match='original-equation'):
        load_representation(folder).rhs_at(.3, 'left')
    restored = load_representation(folder, original_rhs=true_rhs, original_rhs_id='original-equations-sha256')
    np.testing.assert_array_equal(restored.rhs_at(.3, 'left'), [102., 105.])
    with pytest.raises(ContractError, match='identity'):
        load_representation(folder, original_rhs=true_rhs, original_rhs_id='other-equations')


def test_packet_or_checkpoint_tampering_rejected(tmp_path):
    original = DiscreteHold(t0=0., t1=1., state_ids=['a'], y0=np.array([1.]))
    packet = original.packet()
    packet['arrays'] = dict(packet['arrays'], y0=np.array([2.]))
    with pytest.raises(ContractError, match='identity'):
        from_packet(packet)
    packet = original.packet(); packet['metadata']['t1'] = 2.
    with pytest.raises(ContractError, match='schema/boundary'):
        from_packet(packet)
    folder = tmp_path/'tampered'; save_representation(folder, original)
    with (folder/'arrays.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ContractError, match='payload'):
        load_representation(folder)


def test_import_has_no_torch_or_gpu_dependency():
    result = subprocess.run([sys.executable, '-B', '-c',
        "import retina_sim.continuous_state,sys; assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_variable_membrane_uniform_query_matches_manufactured_solution_and_original_rhs():
    H = .001
    y0, c1, c2 = np.array([-.06, -.04]), np.array([.2, -.1]), np.array([2., 3.])
    b = np.zeros((8, 2)); b[0] = [100., 80.]; b[1] = [20., 10.]
    a = np.zeros((8, 2))
    for k in range(2):
        product = np.polynomial.polynomial.polymul(b[:2, k], [y0[k], c1[k]*H, c2[k]*H**2])
        a[:len(product), k] = product
    a[0] += c1; a[1] += 2*c2*H
    packet = AffineMembraneState(t0=0., t1=H, state_ids=['r2', 'r1'], y0=y0,
        a_coefficients=a, b_coefficients=b, quadrature_budget=np.array([1e-13, 1e-13]), state_unit='V')
    for t in np.linspace(0., H, 13):
        side = 'left' if t == H else 'right'
        expected = y0+c1*t+c2*t**2
        np.testing.assert_allclose(packet.evaluate(t, side), expected, rtol=0, atol=4e-16)
        np.testing.assert_allclose(packet.evaluate(t, side, ['r1']), expected[1:], rtol=0, atol=4e-16)
        np.testing.assert_allclose(packet.representation_derivative(t, side), c1+2*c2*t,
                                   rtol=0, atol=5e-14)
        value, error = packet.evaluate_with_error(t, side)
        np.testing.assert_array_equal(value, packet.evaluate(t, side))
        assert np.all(error <= packet.quadrature_budget)
    assert packet.kind == 'integrating_factor' and packet.available_until == H
    with pytest.raises(ContractError, match='original-equation'):
        packet.rhs_at(.0003, 'right')
    rebound = from_packet(packet.packet(), original_rhs=None)
    assert rebound.representation_id == packet.representation_id
    for t, side in ((0., 'left'), (H, 'right'), (np.nextafter(H, np.inf), 'left')):
        with pytest.raises(ContractError):
            packet.evaluate(t, side)


def test_affine_membrane_keeps_original_rhs_and_unit_budget_identity(tmp_path):
    a = np.zeros((8, 1)); a[0, 0] = 2.
    b = np.zeros((8, 1)); b[0, 0] = 3.; b[1, 0] = 1.
    original = AffineMembraneState(t0=0., t1=.001, state_ids=np.array([42]), y0=np.array([-.05]),
        a_coefficients=a, b_coefficients=b, quadrature_budget=np.array([1e-12]), state_unit='V',
        original_rhs=lambda t, side, y: np.full_like(y, 7.), original_rhs_id='true-global-RHS-binding')
    np.testing.assert_array_equal(original.rhs_at(.0003, 'right'), [7.])
    assert original.representation_derivative(.0003, 'right')[0] != 7.
    folder = tmp_path/'affine-original-rhs'
    save_representation(folder, original)
    with pytest.raises(ContractError, match='original-equation'):
        load_representation(folder).rhs_at(.0003, 'right')
    restored = load_representation(folder, original_rhs=lambda t, side, y: np.full_like(y, 7.),
                                   original_rhs_id='true-global-RHS-binding')
    np.testing.assert_array_equal(restored.rhs_at(.0003, 'right'), [7.])
    packet = original.packet(); packet['metadata']['state_unit'] = 'mV'
    with pytest.raises(ContractError, match='identity'):
        from_packet(packet)
    with pytest.raises(ContractError, match='budget'):
        AffineMembraneState(t0=0., t1=.001, state_ids=np.array([42]), y0=np.array([-.05]),
            a_coefficients=a, b_coefficients=b, quadrature_budget=np.array([0.]), state_unit='V')
