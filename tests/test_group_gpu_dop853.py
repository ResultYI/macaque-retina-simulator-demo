"""Small ODE execution only: the CUDA facade stores every tensor on CPU."""
from contextlib import nullcontext
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.integrate import DOP853, solve_ivp
import torch as real_torch

from retina_sim.contracts import ContractError
from retina_sim import group_gpu_dop853 as adapter
from retina_sim import resident_dop853 as resident
from retina_sim import gpu_dop853 as original_transport


class _DeviceTensor(real_torch.Tensor):
    @property
    def device(self):
        return real_torch.device('cuda:0')


class _CpuTorch:
    """Expose CUDA metadata without a CUDA call or a CUDA allocation."""
    cuda = SimpleNamespace(device=lambda *_: nullcontext(),
                           current_stream=lambda *_: SimpleNamespace(cuda_stream=7),
                           current_device=lambda: 0)

    def __getattr__(self, name):
        return getattr(real_torch, name)

    def _make(self, name, *args, **kwargs):
        kwargs['device'] = 'cpu'
        return getattr(real_torch, name)(*args, **kwargs).as_subclass(_DeviceTensor)

    def empty(self, *args, **kwargs):
        return self._make('empty', *args, **kwargs)

    def as_tensor(self, *args, **kwargs):
        return self._make('as_tensor', *args, **kwargs)

    def zeros_like(self, *args, **kwargs):
        return self._make('zeros_like', *args, **kwargs)

    def empty_like(self, *args, **kwargs):
        return self._make('empty_like', *args, **kwargs)


@pytest.fixture
def cpu(monkeypatch):
    facade = _CpuTorch()
    monkeypatch.setitem(sys.modules, 'torch', facade)
    return facade


def _ode(t, y):
    return np.array([5e4*(np.sin(2e5*t)-y[0])+100*y[1]**2,
                     -1.3e4*y[1]+7e3*np.sin(4e4*t)+1e-7*y[2], 0.], dtype=np.float64)


def _rhs(cpu):
    return lambda t, y: cpu.as_tensor(_ode(t, y.detach().numpy()))


def _scipy(initial, start=0., stop=1e-4):
    solver = DOP853(_ode, start, initial.copy(), stop,
                    rtol=1e-6, atol=1e-9, max_step=1e-4)
    segments = []
    while solver.status == 'running':
        solver.step()
        assert solver.status != 'failed'
        segments.append(solver.dense_output())
    return solver, segments


def _dense_value(segment, t):
    x = (t-segment['t_old'])/(segment['t']-segment['t_old'])
    value = np.zeros_like(segment['y_old'].numpy())
    for j, row in enumerate(segment['F'].numpy()[::-1]):
        value += row
        value *= x if j % 2 == 0 else 1-x
    return value+segment['y_old'].numpy()


def test_default_transport_matches_scipy_requested_samples(cpu):
    initial = np.array([.2, -.4, 1e6])
    times = np.array([0., 3.3e-5, 1e-4])
    reference = solve_ivp(_ode, (0., 1e-4), initial, method='DOP853',
                          rtol=1e-6, atol=1e-9, max_step=1e-4, t_eval=times)
    y = cpu.as_tensor(initial.copy())
    result = resident.integrate_device(_rhs(cpu), 0., y, 1e-4,
                rtol=1e-6, atol=1e-9, max_step=1e-4, t_eval=times)
    assert result['dense_segments'] == ()
    legacy = original_transport.integrate(_rhs(cpu), 0., y, 1e-4,
                rtol=1e-6, atol=1e-9, max_step=1e-4, t_eval=times)
    assert result['nfev'] == legacy['nfev']
    np.testing.assert_array_equal(np.stack([v.numpy() for v in result['samples']], axis=1),
                                  legacy['y'])
    np.testing.assert_array_equal(y.numpy(), initial)
    np.testing.assert_allclose(np.stack([v.numpy() for v in result['samples']], axis=1),
                               reference.y, rtol=1e-11, atol=2e-13)
    assert result['nfev'] == 2+12*result['attempts']+3*result['dense_polynomials']


def test_dense_into_schedule_and_ownership_match_original_dop853(cpu):
    initial = np.array([0., 0., 1e6])
    y = cpu.as_tensor(initial.copy())
    workspace = resident.Dop853Workspace(y, 1e-9)
    reference, reference_dense = _scipy(initial)
    calls = []

    def rhs_into(t, state, out):
        assert out.data_ptr() != state.data_ptr()
        calls.append(t)
        out.copy_(_rhs(cpu)(t, state))
        return out

    result = resident.integrate_device(None, 0., y, 1e-4,
        rtol=1e-6, atol=1e-9, max_step=1e-4, workspace=workspace,
        dense_output=True, fun_into=rhs_into)
    ordinary = resident.integrate_device(_rhs(cpu), 0., y, 1e-4,
        rtol=1e-6, atol=1e-9, max_step=1e-4, dense_output=True)
    np.testing.assert_array_equal(result['final'].numpy(), ordinary['final'].numpy())
    assert result['nfev'] == ordinary['nfev'] == len(calls) == reference.nfev
    for direct, original in zip(result['dense_segments'], ordinary['dense_segments']):
        assert direct['t_old'] == original['t_old'] and direct['t'] == original['t']
        np.testing.assert_array_equal(direct['F'].numpy(), original['F'].numpy())
    assert result['steps'] == len(reference_dense) > 1
    assert result['rejected_attempts'] > 0
    np.testing.assert_allclose([s['t'] for s in result['dense_segments']],
                               [s.t for s in reference_dense], rtol=2e-9, atol=2e-14)
    np.testing.assert_allclose(result['final'].numpy(), reference.y, rtol=2e-10, atol=1e-12)
    saved = []
    for segment, expected in zip(result['dense_segments'], reference_dense):
        middle = (segment['t_old']+segment['t'])/2
        np.testing.assert_allclose(_dense_value(segment, middle), expected(middle),
                                   rtol=3e-10, atol=2e-12)
        np.testing.assert_array_equal(segment['F'][:, 2].numpy(), 0.)
        saved.append((segment['y_old'].clone(), segment['F'].clone()))
    buffers = tuple(v.data_ptr() for v in workspace.rhs_buffers)
    storage = (workspace.K.data_ptr(), workspace.F.data_ptr())
    later = cpu.as_tensor(np.array([.7, -.1, 2e6]))
    resident.integrate_device(None, 1e-4, later, 2e-4,
        rtol=1e-6, atol=1e-9, max_step=1e-4, workspace=workspace,
        dense_output=True, fun_into=rhs_into)
    assert buffers == tuple(v.data_ptr() for v in workspace.rhs_buffers)
    assert storage == (workspace.K.data_ptr(), workspace.F.data_ptr())
    for segment, (anchor, coefficients) in zip(result['dense_segments'], saved):
        np.testing.assert_array_equal(segment['y_old'].numpy(), anchor.numpy())
        np.testing.assert_array_equal(segment['F'].numpy(), coefficients.numpy())
    np.testing.assert_array_equal(y.numpy(), initial)
    assert result['solver_scalar_D2H'] == 4+2*result['attempts']


class _Inner:
    state_keys = (('toy', 'dynamic', 'value'), ('toy', 'held', 'history'))
    nstate = 3
    slices = {state_keys[0]: slice(0, 2), state_keys[1]: slice(2, 3)}

    def validate(self, state):
        if set(state) != set(self.state_keys):
            raise ContractError('Incomplete toy state')
        for key, size in zip(self.state_keys, (2, 1)):
            value = np.asarray(state[key])
            if value.shape != (size,) or value.dtype != np.float64 or not np.isfinite(value).all():
                raise ContractError('Invalid toy state')

    def pack(self, state):
        self.validate(state)
        return np.concatenate([state[k] for k in self.state_keys])

    def unpack(self, state, *, copy=True):
        return {k: state[s].copy() if copy else state[s] for k, s in self.slices.items()}

    def rhs(self, t, state):
        return self.unpack(_ode(t, self.pack(state)))


class _CpuRhs:
    def __init__(self, compiled, torch):
        self.torch, self.compiled = torch, compiled
        self.device, self.identity = torch.device('cuda:0'), 'test_cpu_rhs'
        self.metrics = {'kernel_launches_per_rhs': 5, 'resident_bytes': 0}
        self.closed = False

    def evaluate_into(self, t, state, out):
        if self.compiled.fail:
            raise ContractError('Test domain failure')
        out.copy_(self.torch.as_tensor(_ode(t, state.numpy())))
        return out

    def close(self):
        self.closed = True


@pytest.fixture
def execution(cpu, monkeypatch):
    monkeypatch.setattr(adapter, 'GroupGpuRHS', _CpuRhs)
    inner = _Inner()
    compiled = SimpleNamespace(model=inner, nstate=3, fail=False)
    gate = SimpleNamespace(allow=True)
    solver = adapter.GroupGpuDop853(inner, compiled, cpu, guard=lambda: gate.allow)
    return inner, compiled, gate, solver


def test_adapter_commit_refresh_dense_original_rhs_and_metrics(execution):
    inner, compiled, gate, solver = execution
    state = inner.unpack(np.array([.2, -.4, 1e6]))
    old = inner.pack(state)
    first = solver.advance(state, 0., 1e-4, rtol=1e-6, atol=1e-9)
    query = first.evaluate(3e-5)
    np.testing.assert_array_equal(inner.pack(first.rhs_at(3e-5)), _ode(3e-5, inner.pack(query)))
    with pytest.raises(ContractError, match='representation derivatives'):
        first.representation_derivative(3e-5)
    for t, side in ((0., 'left'), (1e-4, 'right'), (-1e-8, 'right')):
        with pytest.raises(ContractError):
            first.evaluate(t, side)
    committed = {k: v.copy() for k, v in first.state.items()}
    committed[inner.state_keys[1]][0] = 2e6
    second = solver.advance(committed, 1e-4, 2e-4, rtol=1e-6, atol=1e-9)
    ref, _ = _scipy(inner.pack(committed), 1e-4, 2e-4)
    np.testing.assert_allclose(inner.pack(second.state), ref.y, rtol=3e-10, atol=2e-12)
    np.testing.assert_array_equal(inner.pack(first.evaluate(3e-5)), inner.pack(query))
    np.testing.assert_array_equal(inner.pack(state), old)
    assert first.accepted_times == (0., *[s['t'] for s in first.dense_segments])
    metrics = first.metrics
    assert metrics['nfev'] == 2+metrics['main_stage_rhs_calls']+metrics['dense_stage_rhs_calls']
    assert metrics['rhs_kernel_launches'] == 5*metrics['nfev']
    assert metrics['scalar_readback_sync_points'] == metrics['nfev']+4+2*metrics['attempts']
    assert metrics['H2D_bytes'] == old.nbytes
    assert metrics['state_D2H_bytes'] == metrics['full_state_D2H']*old.nbytes
    assert metrics['D2H_bytes'] == metrics['state_D2H_bytes']+metrics['scalar_D2H_bytes']
    assert metrics['blocking_state_transfer_calls'] == 1+metrics['full_state_D2H']
    assert metrics['full_state_D2H_per_stage'] == metrics['rhs_output_clones'] == 0
    assert metrics['held_in_initial_step_and_error_norm']
    assert metrics['error_norm_dimension'] == 3


def test_adapter_guards_failure_ownership_and_recovery(execution):
    inner, compiled, gate, solver = execution
    state = inner.unpack(np.array([.2, -.4, 1e6]))
    original = inner.pack(state)
    for rtol, atol in ((1e-5, 1e-9), (1e-6, 1e-8)):
        with pytest.raises(ContractError, match='tolerances'):
            solver.advance(state, 0., 1e-4, rtol=rtol, atol=atol)
    gate.allow = False
    with pytest.raises(ContractError, match='guard'):
        solver.advance(state, 0., 1e-4, rtol=1e-6, atol=1e-9)
    gate.allow = True
    compiled.fail = True
    with pytest.raises(ContractError, match='domain'):
        solver.advance(state, 0., 1e-4, rtol=1e-6, atol=1e-9)
    np.testing.assert_array_equal(inner.pack(state), original)
    compiled.fail = False
    result = solver.advance(state, 0., 1e-4, rtol=1e-6, atol=1e-9)
    assert result.state[inner.state_keys[1]][0] == 1e6
    solver.close()
    solver.close()
    with pytest.raises(ContractError, match='closed'):
        solver.advance(state, 0., 1e-4, rtol=1e-6, atol=1e-9)


def test_downloaded_state_views_outlive_workspace_and_are_owned_per_query(execution):
    inner, _, _, solver = execution
    state = inner.unpack(np.array([.2, -.4, 1e6]))
    first = solver.advance(state, 0., 1e-4, rtol=1e-6, atol=1e-9)
    query = first.evaluate(3e-5)
    expected = inner.pack(query)
    again = first.evaluate(3e-5)
    for key in inner.state_keys:
        assert not np.shares_memory(query[key], again[key])
        assert not np.shares_memory(query[key], first.state[key])
        query[key][:] = 99.
    solver.advance(first.state, 1e-4, 2e-4, rtol=1e-6, atol=1e-9)
    np.testing.assert_array_equal(inner.pack(again), expected)
    np.testing.assert_array_equal(inner.pack(first.evaluate(3e-5)), expected)


def test_guard_after_integration_prevents_publication(execution, monkeypatch):
    inner, compiled, gate, solver = execution
    state = inner.unpack(np.array([.2, -.4, 1e6]))
    original = inner.pack(state)
    integrate = adapter.integrate_device

    def finish_then_stop(*args, **kwargs):
        result = integrate(*args, **kwargs)
        gate.allow = False
        return result

    monkeypatch.setattr(adapter, 'integrate_device', finish_then_stop)
    with pytest.raises(ContractError, match='guard'):
        solver.advance(state, 0., 1e-4, rtol=1e-6, atol=1e-9)
    np.testing.assert_array_equal(inner.pack(state), original)


def test_borrowed_rhs_remains_owned_by_caller(cpu, monkeypatch):
    monkeypatch.setattr(adapter, 'GroupGpuRHS', _CpuRhs)
    inner = _Inner()
    compiled = SimpleNamespace(model=inner, nstate=3, fail=False)
    rhs = _CpuRhs(compiled, cpu)
    rhs.metrics['resident_bytes'] = 1234
    solver = adapter.GroupGpuDop853(inner, compiled, cpu, gpu_rhs=rhs)
    state = inner.unpack(np.array([.2, -.4, 1e6]))
    result = solver.advance(state, 0., 1e-4, rtol=1e-6, atol=1e-9)
    assert result.metrics['shared_rhs_device_bytes'] == 1234
    assert solver.rhs is rhs
    solver.close()
    assert not rhs.closed
    wrong = SimpleNamespace(model=inner, nstate=3)
    with pytest.raises(ContractError, match='Borrowed'):
        adapter.GroupGpuDop853(inner, wrong, cpu, gpu_rhs=rhs)
