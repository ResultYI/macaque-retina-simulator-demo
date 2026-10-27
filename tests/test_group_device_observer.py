"""CPU synthetic coverage only; no graph construction, simulation or CUDA run."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.group_device_observer import GroupDeviceObserver
from retina_sim.group_source_observer import GroupSourceObserver
from retina_sim.group_device_snapshot import ExactSnapshotCache, SnapshotKey
from retina_sim.group_time_monitor import GroupTimeMonitor
from retina_sim.spatial_group_model import BC, RGC


@pytest.fixture
def synthetic():
    specs = []
    initial = {}

    def state(pop, mechanism, name, unit, meaning, values):
        specs.append(SimpleNamespace(population=pop, mechanism=mechanism, name=name,
                                     unit=unit, meaning=meaning, count=len(values)))
        initial[pop, mechanism, name] = np.array(values, dtype=np.float64)

    state('cones', 'native', 'X', 'vH_native_precursor', 'native_precursor', [1., 1., 1.])
    state('cones', 'native', 'q', 'dimensionless', 'release_probability', [.2, .3, .4])
    state('rods', 'Chen2024PrimateRodCurrent', 'G', 'uM', 'cgmp_concentration', [2., 2., 2.])
    state(BC[0], 'BC', 'A', 'dimensionless', 'release_adaptation_activity', [.1, .2, .3])
    state(BC[0], 'BC', 'V', 'V', 'voltage', [-.04, -.04, -.04])
    state('contacts', 'contact', 'gate', 'dimensionless', 'receptor_gate', [.1, .2])
    for pop in RGC:
        for i in range(8):
            state(pop, 'RGC_expected_history', 'history_probability_' + str(i),
                  'dimensionless', 'history_probability', [1. if i == 0 else 0.] * 3)
    assembly = SimpleNamespace(
        state_blocks=specs, probability_blocks=(), auxiliary_z1_blocks=(),
        artifact_sha256='synthetic-device-monitor',
        populations={pop: SimpleNamespace(group_ids=np.array([pop + ':' + str(i) for i in range(3)]))
                     for pop in {key[0] for key in initial}})
    parameters = {'populations': {BC[0]: {'adaptation_scale': np.array([.5, .5, .5])}}}
    return assembly, initial, parameters


def point(monitor, initial, index=0):
    states = {key: value.copy() for key, value in initial.items()}
    states['cones', 'native', 'X'] += index * 1e-8
    rhs = {key: np.full_like(value, (-1.) ** index * index * 1e-6) for key, value in states.items()}
    transitions = {key: states[key] + index * 1e-12 for key in monitor.transition_keys}
    aux = {'spike_states': {'synthetic': {'history': np.array([0, 1, 2]),
                                         'last_bin_counts': np.array([0., 1., 1.]),
                                         'rng_state': {'state': 17},
                                         'pending_events': [{'time_seconds': 1.}]}}}
    return dict(time_seconds=index * .0001, side='left' if index % 2 else 'right',
                event_generation=index, states=states, rhs=rhs, transitions=transitions,
                ports={('AII_AC', 'gap_current'): np.array([1e-12, -1e-12])},
                aux=aux, rhs_kind='original_equation')


def observe(monitor, snapshot):
    monitor.observe(snapshot['time_seconds'], snapshot['side'], snapshot['states'], snapshot['rhs'],
                    transitions=snapshot['transitions'], ports=snapshot['ports'], aux=snapshot['aux'],
                    event_generation=snapshot['event_generation'], rhs_kind=snapshot['rhs_kind'])


def tensor_point(snapshot, torch):
    converted = deepcopy(snapshot)
    for name in ('states', 'rhs', 'transitions', 'ports'):
        converted[name] = {key: torch.as_tensor(value).clone() for key, value in snapshot[name].items()}
    return converted


@pytest.mark.parametrize('backend', ['numpy', 'torch_cpu'])
def test_complete_raw_statistics_frozen_budgets_restore_and_owned_export(synthetic, backend):
    assembly, initial, parameters = synthetic
    torch = pytest.importorskip('torch') if backend == 'torch_cpu' else None
    old = GroupTimeMonitor(assembly, initial, parameters=parameters)
    new = GroupDeviceObserver(assembly, initial, parameters=parameters, torch=torch, device='cpu')
    frozen = {key: tuple(value[name].tobytes() for name in ('scale', 'den', 'budget'))
              for key, value in new.rows.items()}
    buffer_ids = {name: id(value) for name, value in new._buffers.items()}
    saved = None
    for index in range(5):
        snapshot = point(old, initial, index)
        observe(old, snapshot)
        new.observe_snapshot(snapshot if torch is None else tensor_point(snapshot, torch))
        assert canonical_hash(old.payload()) == canonical_hash(new.payload())
        assert old.report(.0005) == new.report(.0005)
        assert {name: id(value) for name, value in new._buffers.items()} == buffer_ids
        assert {key: tuple(value[name].tobytes() for name in ('scale', 'den', 'budget'))
                for key, value in new.rows.items()} == frozen
        if saved is not None:
            assert canonical_hash(saved) == saved_hash
        saved = new.payload()
        saved_hash = canonical_hash(saved)
        for value in snapshot['states'].values():
            value[:] = 999.
    restored = GroupDeviceObserver.from_payload(assembly, initial, new.payload(), parameters=parameters,
                                               torch=torch, device='cpu')
    snapshot = point(old, initial, 5)
    observe(old, snapshot)
    for monitor in (new, restored):
        monitor.observe_snapshot(snapshot if torch is None else tensor_point(snapshot, torch))
    assert canonical_hash(old.payload()) == canonical_hash(new.payload()) == canonical_hash(restored.payload())
    assert new.report(.0005)['dominant'] == old.report(.0005)['dominant']
    accounting = new.storage_accounting()
    assert accounting['total_bytes'] == (sum(accounting['allocations'].values())
                                         + accounting['frozen_host_bytes']
                                         + accounting['binding_host_bytes']
                                         + accounting['host_staging_bytes']) > 0


@pytest.mark.parametrize('backend', ['numpy', 'torch_cpu'])
def test_all_domain_conservation_and_history_gates_keep_original_order(synthetic, backend):
    assembly, initial, parameters = synthetic
    torch = pytest.importorskip('torch') if backend == 'torch_cpu' else None
    old = GroupTimeMonitor(assembly, initial, parameters=parameters)
    new = GroupDeviceObserver(assembly, initial, parameters=parameters, torch=torch, device='cpu')
    snapshot = point(old, initial)
    snapshot['states']['cones', 'native', 'q'][1] = np.nextafter(1. + 1e-10, np.inf)
    snapshot['states']['cones', 'native', 'X'][0] = np.nextafter(-1e-10, -np.inf)
    snapshot['states'][BC[0], 'BC', 'A'][:] = [-.1, .6, .3]
    snapshot['states'][RGC[0], 'RGC_expected_history', 'history_probability_0'][0] = 1.1
    snapshot['ports']['AII_AC', 'gap_current'][:] = 1e-7
    history = snapshot['aux']['spike_states']['synthetic']
    history.update(history=np.array([-1., .5, np.nan]), last_bin_counts=np.array([-1., np.nan]),
                   rng_state={}, pending_events=[{'time_seconds': -.001}])
    observe(old, snapshot)
    new.observe_snapshot(snapshot if torch is None else tensor_point(snapshot, torch))
    assert new.domain_failures == old.domain_failures
    assert len(new.domain_failures) == 11
    assert canonical_hash(new.payload()) == canonical_hash(old.payload())
    assert new.report(.0001) == old.report(.0001)


@pytest.mark.parametrize('case', ['missing_state', 'missing_rhs', 'missing_transition', 'nan_state',
                                 'inf_rhs', 'nan_transition', 'fp32_state', 'fp32_transition',
                                 'bad_rhs_shape', 'bad_time', 'bad_side', 'bool_generation', 'dense_rhs'])
def test_invalid_witness_fails_without_publishing_partial_observation(synthetic, case):
    assembly, initial, parameters = synthetic
    monitor = GroupDeviceObserver(assembly, initial, parameters=parameters)
    monitor.observe_snapshot(point(monitor, initial))
    before = canonical_hash(monitor.payload())
    snapshot = point(monitor, initial, 1)
    key = next(iter(initial))
    transition_key = monitor.transition_keys[0]
    if case == 'missing_state':
        snapshot['states'].pop(key)
    elif case == 'missing_rhs':
        snapshot['rhs'].pop(key)
    elif case == 'missing_transition':
        snapshot['transitions'].pop(transition_key)
    elif case == 'nan_state':
        snapshot['states'][key][0] = np.nan
    elif case == 'inf_rhs':
        snapshot['rhs'][key][0] = np.inf
    elif case == 'nan_transition':
        snapshot['transitions'][transition_key][0] = np.nan
    elif case == 'fp32_state':
        snapshot['states'][key] = snapshot['states'][key].astype(np.float32)
    elif case == 'fp32_transition':
        snapshot['transitions'][transition_key] = snapshot['transitions'][transition_key].astype(np.float32)
    elif case == 'bad_rhs_shape':
        snapshot['rhs'][key] = snapshot['rhs'][key][None, :]
    elif case == 'bad_time':
        snapshot['time_seconds'] = np.nan
    elif case == 'bad_side':
        snapshot['side'] = 'interior'
    elif case == 'bool_generation':
        snapshot['event_generation'] = True
    elif case == 'dense_rhs':
        snapshot['rhs_kind'] = 'representation_derivative'
    with pytest.raises(ContractError):
        monitor.observe_snapshot(snapshot)
    assert canonical_hash(monitor.payload()) == before


def test_first_argmax_uses_original_row_then_criterion_then_cell_order(synthetic):
    assembly, initial, parameters = synthetic
    old = GroupTimeMonitor(assembly, initial, parameters=parameters)
    new = GroupDeviceObserver(assembly, initial, parameters=parameters)
    snapshot = point(old, initial)
    key = ('cones', 'native', 'X')
    snapshot['rhs'][key][:] = 10.
    for monitor in (old, new):
        observe(monitor, snapshot)
    assert new.report(.0001) == old.report(.0001)
    dominant = new.report(.0001)['dominant']
    assert dominant['state'] == ':'.join(key) and dominant['cell_index'] == 0
    assert dominant['criterion'] == 'original_rhs'


def cache_key():
    return SnapshotKey.create(0., 'right', state_version=2, input_epoch=3, event_generation=0,
                              history_version=4, sampler_identity='PCG64-owner-1', operator_identity='plan-1')


@pytest.mark.parametrize('field,value', [('time_hex', float(np.nextafter(0., 1.)).hex()), ('side', 'left'),
                                       ('state_version', 3), ('input_epoch', 4), ('event_generation', 1),
                                       ('history_version', 5), ('sampler_identity', 'PCG64-owner-2'),
                                       ('operator_identity', 'plan-2')])
def test_snapshot_cache_never_reuses_other_context(synthetic, field, value):
    assembly, initial, parameters = synthetic
    monitor = GroupDeviceObserver(assembly, initial, parameters=parameters)
    cache = ExactSnapshotCache()
    key = cache_key()
    cache.put(key, point(monitor, initial))
    assert cache.get(key) is not None
    assert cache.get(replace(key, **{field: value})) is None


@pytest.mark.parametrize('backend', ['numpy', 'torch_cpu'])
def test_cache_nested_ownership_eviction_and_republication(synthetic, backend):
    assembly, initial, parameters = synthetic
    monitor = GroupDeviceObserver(assembly, initial, parameters=parameters)
    snapshot = point(monitor, initial)
    torch = pytest.importorskip('torch') if backend == 'torch_cpu' else None
    if torch is not None:
        snapshot = tensor_point(snapshot, torch)
    cache = ExactSnapshotCache(capacity=1)
    key = cache_key()
    cache.put(key, snapshot)
    snapshot['aux']['spike_states']['synthetic']['rng_state']['state'] = 99
    first_key = next(iter(initial))
    snapshot['states'][first_key][0] = 99
    result = cache.get(key)
    assert result['states'][first_key][0] == 1.
    assert result['aux']['spike_states']['synthetic']['rng_state']['state'] == 17
    result['states'][first_key][0] = 17
    result['aux']['spike_states']['synthetic']['rng_state']['state'] = 99
    assert cache.get(key)['states'][first_key][0] == 1.
    assert cache.get(key)['aux']['spike_states']['synthetic']['rng_state']['state'] == 17
    with pytest.raises(ContractError, match='republished'):
        cache.put(key, result)
    other = replace(key, state_version=3)
    cache.put(other, snapshot)
    assert cache.get(key) is None and cache.get(other) is not None and len(cache) == 1
    cache.clear()
    assert len(cache) == 0


def test_cache_validates_context_and_snapshot_stamp(synthetic):
    assembly, initial, parameters = synthetic
    monitor = GroupDeviceObserver(assembly, initial, parameters=parameters)
    snapshot = point(monitor, initial)
    for changes in ({'state_version': True}, {'event_generation': -1}, {'sampler_identity': ''},
                    {'time_hex': 'nan'}, {'time_hex': '0.0'}):
        with pytest.raises(ContractError):
            replace(cache_key(), **changes)
    for field, value in [('side', 'left'), ('event_generation', 1), ('rhs_kind', 'dense_derivative'),
                         ('state_version', 99)]:
        wrong = deepcopy(snapshot)
        wrong[field] = value
        with pytest.raises(ContractError):
            ExactSnapshotCache().put(cache_key(), wrong)


def test_payload_stamp_and_restore_ownership(synthetic):
    assembly, initial, parameters = synthetic
    monitor = GroupDeviceObserver(assembly, initial, parameters=parameters)
    monitor.observe_snapshot(point(monitor, initial))
    payload = monitor.payload()
    restored = GroupDeviceObserver.from_payload(assembly, initial, payload, parameters=parameters)
    original_hash = canonical_hash(monitor.payload())
    payload['stamps'][0]['side'] = 'left'
    payload['rows']['cones:native:X']['lo'][0] = 99.
    assert canonical_hash(monitor.payload()) == original_hash
    assert canonical_hash(restored.payload()) == original_hash


def test_incomplete_cache_snapshot_and_noncanonical_counter_rejected(synthetic):
    assembly, initial, parameters = synthetic
    monitor = GroupDeviceObserver(assembly, initial, parameters=parameters)
    snapshot = point(monitor, initial)
    snapshot.pop('states')
    with pytest.raises(ContractError, match='complete original snapshot'):
        ExactSnapshotCache().put(cache_key(), snapshot)
    snapshot = point(monitor, initial)
    snapshot['state_version'] = np.int64(2)
    with pytest.raises(ContractError, match='owner context'):
        ExactSnapshotCache().put(cache_key(), snapshot)


def test_device_alias_uses_actual_allocation_and_rejects_mixed_witness(synthetic):
    torch = pytest.importorskip('torch')
    assembly, initial, parameters = synthetic
    monitor = GroupDeviceObserver(assembly, initial, parameters=parameters, torch=torch, device='cpu:0')
    snapshot = tensor_point(point(monitor, initial), torch)
    monitor.observe_snapshot(snapshot)
    assert monitor.device == torch.device('cpu')
    original_hash = canonical_hash(monitor.payload())
    snapshot['rhs'][next(iter(initial))] = np.zeros(3, dtype=np.float64)
    with pytest.raises(ContractError, match='layout/device'):
        monitor.observe_snapshot(snapshot)
    assert canonical_hash(monitor.payload()) == original_hash


@pytest.mark.parametrize('backend', ['numpy', 'torch_cpu'])
def test_mutable_adaptation_binding_matches_original_without_refreezing_budgets(synthetic, backend):
    assembly, initial, parameters = synthetic
    torch = pytest.importorskip('torch') if backend == 'torch_cpu' else None
    old = GroupTimeMonitor(assembly, initial, parameters=parameters)
    new = GroupDeviceObserver(assembly, initial, parameters=parameters, torch=torch, device='cpu')
    key = (BC[0], 'BC', 'A')
    frozen = tuple(new.rows[key][name].tobytes() for name in ('scale', 'den', 'budget'))
    snapshot = point(old, initial)
    snapshot['states'][key][:] = .6
    parameters['populations'][BC[0]]['adaptation_scale'][:] = 1.
    observe(old, snapshot)
    new.observe_snapshot(snapshot if torch is None else tensor_point(snapshot, torch))
    assert new.domain_failures == old.domain_failures == []
    assert canonical_hash(new.payload()) == canonical_hash(old.payload())
    parameters['populations'][BC[0]]['adaptation_scale'] = np.array([.4, .5, .4])
    observe(old, snapshot)
    new.observe_snapshot(snapshot if torch is None else tensor_point(snapshot, torch))
    assert new.domain_failures == old.domain_failures == [':'.join(key) + ':above_scale']
    assert canonical_hash(new.payload()) == canonical_hash(old.payload())
    assert tuple(new.rows[key][name].tobytes() for name in ('scale', 'den', 'budget')) == frozen


def test_explicit_mixed_host_ingress_keeps_all_statistics_and_counts_real_bytes(synthetic):
    torch = pytest.importorskip('torch')
    assembly, initial, parameters = synthetic
    original = GroupTimeMonitor(assembly, initial, parameters=parameters)
    monitor = GroupDeviceObserver(assembly, initial, parameters=parameters, torch=torch, device='cpu',
                                  allow_host_inputs=True)
    staging_id = id(monitor._host_staging)
    tensor_pointer = monitor._host_tensor.data_ptr()
    total_bytes = 0
    for index in range(3):
        expected = point(original, initial, index)
        mixed = deepcopy(expected)
        for name in ('states', 'rhs'):
            for key in initial:
                if key[0] == 'cones':
                    mixed[name][key] = torch.from_numpy(mixed[name][key]).clone()
        # Noncontiguous NumPy input must be staged without a dtype conversion.
        key = ('contacts', 'contact', 'gate')
        mixed['states'][key] = mixed['states'][key][::-1]
        expected['states'][key] = mixed['states'][key].copy()
        total_bytes += sum(value.nbytes for name in ('states', 'rhs', 'transitions')
                           for value in mixed[name].values() if isinstance(value, np.ndarray))
        observe(original, expected)
        monitor.observe_snapshot(mixed)
        assert canonical_hash(original.payload()) == canonical_hash(monitor.payload())
        assert original.report(.0003) == monitor.report(.0003)
        assert id(monitor._host_staging) == staging_id
        assert monitor._host_tensor.data_ptr() == tensor_pointer
        assert monitor.host_transfer_metrics['bytes'] == total_bytes
    metrics = monitor.host_transfer_metrics
    assert metrics['transfers'] < len(initial) and metrics['seconds'] > 0.
    assert metrics['packing_seconds'] > 0.
    accounting = monitor.storage_accounting()
    assert accounting['host_staging_bytes'] == sum(value.nbytes for value in initial.values())
    restored = GroupDeviceObserver.from_payload(assembly, initial, monitor.payload(), parameters=parameters,
                                               torch=torch, device='cpu', allow_host_inputs=True)
    assert restored.allow_host_inputs is True and restored.host_transfer_metrics['bytes'] == 0
    mixed = point(original, initial, 3)
    observe(original, mixed)
    restored.observe_snapshot(mixed)
    assert canonical_hash(original.payload()) == canonical_hash(restored.payload())


@pytest.mark.parametrize('fault', ['fp32', 'wrong_device', 'python_list'])
def test_host_ingress_permission_never_converts_invalid_witnesses(synthetic, fault):
    torch = pytest.importorskip('torch')
    assembly, initial, parameters = synthetic
    monitor = GroupDeviceObserver(assembly, initial, parameters=parameters, torch=torch, device='cpu',
                                  allow_host_inputs=True)
    monitor.observe_snapshot(point(monitor, initial))
    before = canonical_hash(monitor.payload())
    snapshot = point(monitor, initial, 1)
    key = next(iter(initial))
    if fault == 'fp32':
        snapshot['states'][key] = snapshot['states'][key].astype(np.float32)
    elif fault == 'wrong_device':
        snapshot['states'][key] = torch.empty(3, dtype=torch.float64, device='meta')
    else:
        snapshot['states'][key] = snapshot['states'][key].tolist()
    with pytest.raises(ContractError, match='layout/device'):
        monitor.observe_snapshot(snapshot)
    assert canonical_hash(monitor.payload()) == before


def test_torch_native_inputs_do_not_report_host_uploads(synthetic):
    torch = pytest.importorskip('torch')
    assembly, initial, parameters = synthetic
    monitor = GroupDeviceObserver(assembly, initial, parameters=parameters, torch=torch, device='cpu',
                                  allow_host_inputs=True)
    monitor.observe_snapshot(tensor_point(point(monitor, initial), torch))
    assert monitor.host_transfer_metrics == dict(bytes=0, seconds=0., transfers=0, packing_seconds=0.)
    parameters['populations'][BC[0]]['adaptation_scale'][:] = .75
    monitor.observe_snapshot(tensor_point(point(monitor, initial), torch))
    assert monitor.host_transfer_metrics['bytes'] == 3 * 8
    assert monitor.host_transfer_metrics['transfers'] == 1


def test_mixed_ingress_preserves_domain_mass_conservation_and_history_failures(synthetic):
    torch = pytest.importorskip('torch')
    assembly, initial, parameters = synthetic
    original = GroupTimeMonitor(assembly, initial, parameters=parameters)
    monitor = GroupDeviceObserver(assembly, initial, parameters=parameters, torch=torch, device='cpu',
                                  allow_host_inputs=True)
    expected = point(original, initial)
    expected['states']['cones', 'native', 'q'][0] = 1.1
    expected['states'][BC[0], 'BC', 'A'][0] = .6
    expected['states'][RGC[0], 'RGC_expected_history', 'history_probability_0'][0] = 1.1
    expected['ports']['AII_AC', 'gap_current'][:] = 1e-7
    expected['aux']['spike_states']['synthetic']['rng_state'] = {}
    mixed = deepcopy(expected)
    for name in ('states', 'rhs'):
        for key in initial:
            if key[0] == 'cones':
                mixed[name][key] = torch.from_numpy(mixed[name][key]).clone()
    observe(original, expected)
    monitor.observe_snapshot(mixed)
    assert len(original.domain_failures) == 6
    assert monitor.domain_failures == original.domain_failures
    assert canonical_hash(monitor.payload()) == canonical_hash(original.payload())
    assert monitor.report(.0001) == original.report(.0001)


def test_domain_flag_reduction_uses_cuda_supported_integer_dtype(synthetic, monkeypatch):
    torch = pytest.importorskip('torch')
    assembly, initial, parameters = synthetic
    original = GroupTimeMonitor(assembly, initial, parameters=parameters)
    monitor = GroupDeviceObserver(assembly, initial, parameters=parameters, torch=torch, device='cpu',
                                  allow_host_inputs=True)
    scatter_reduce = torch.Tensor.scatter_reduce_
    calls = []

    def cuda_dtype_guard(destination, dimension, indices, source, *args, **kwargs):
        if destination.dtype == torch.bool or source.dtype == torch.bool:
            raise NotImplementedError('CUDA scatter reduction does not implement Bool')
        assert destination.dtype == source.dtype == torch.int32
        calls.append(destination.dtype)
        return scatter_reduce(destination, dimension, indices, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'scatter_reduce_', cuda_dtype_guard)
    for index in range(2):
        snapshot = point(original, initial, index)
        if index:
            snapshot['states']['cones', 'native', 'q'][0] = 1.1
            snapshot['states'][BC[0], 'BC', 'A'][:] = [-.1, .6, .3]
            snapshot['states'][RGC[0], 'RGC_expected_history', 'history_probability_0'][0] = 1.1
        observe(original, snapshot)
        monitor.observe_snapshot(snapshot)
        assert canonical_hash(monitor.payload()) == canonical_hash(original.payload())
        assert monitor.domain_failures == original.domain_failures
        assert monitor.report(.0001) == original.report(.0001)
    assert calls == [torch.int32, torch.int32]
    assert monitor._domain_cells.dtype == monitor._domain_rows.dtype == torch.int32
    assert monitor._statistics.dtype == monitor._values.dtype == torch.float64


@pytest.mark.parametrize('backend', ['numpy', 'torch_cpu_mixed'])
def test_source_placement_reads_each_callback_and_preserves_full_original_evidence(synthetic, backend):
    assembly, initial, parameters = synthetic
    torch = pytest.importorskip('torch') if backend == 'torch_cpu_mixed' else None
    original = GroupTimeMonitor(assembly, initial, parameters=parameters, torch=torch)
    source = GroupSourceObserver(assembly, initial, parameters=parameters, torch=torch,
                                 device='cpu', allow_host_inputs=True)
    expected_host_bytes = 0
    for index in range(3):
        snapshot = point(original, initial, index)
        if torch is not None:
            for name in ('states', 'rhs'):
                snapshot[name] = {key:torch.from_numpy(value) if key[0]=='cones' else value
                                  for key,value in snapshot[name].items()}
        # Same exact stamp/generation on the next call must still read changed values.
        if index == 2:
            snapshot['time_seconds'] = .0001
            snapshot['side'] = 'left'
            snapshot['event_generation'] = 1
            snapshot['states'][BC[0], 'BC', 'A'][0] = .6
            snapshot['states'][RGC[0], 'RGC_expected_history', 'history_probability_0'][0] = 1.1
            snapshot['ports']['AII_AC', 'gap_current'][:] = 1e-7
            snapshot['aux']['spike_states']['synthetic']['rng_state'] = {}
        expected_host_bytes += sum(value.nbytes for name in ('states','rhs','transitions')
                                   for value in snapshot[name].values() if isinstance(value,np.ndarray))
        observe(original, snapshot)
        source.observe_snapshot(snapshot)
        assert canonical_hash(source.payload()) == canonical_hash(original.payload())
        assert source.report(.0003) == original.report(.0003)
        assert source.domain_failures == original.domain_failures
    assert source.execution_profile['observed_host_bytes'] == expected_host_bytes
    assert source.execution_profile['input_bytes_uploaded'] == source.host_transfer_metrics['bytes'] == 0
    assert source.execution_profile['fully_device_resident'] is False


def test_source_placement_restore_export_aliases_and_input_mutation_are_owned(synthetic):
    torch = pytest.importorskip('torch')
    assembly, initial, parameters = synthetic
    source = GroupSourceObserver(assembly, initial, parameters=parameters, torch=torch,
                                 device='cpu', allow_host_inputs=True)
    def mixed(index):
        snapshot = point(source, initial, index)
        for name in ('states','rhs'):
            snapshot[name] = {key:torch.from_numpy(value) if key[0]=='cones' else value
                              for key,value in snapshot[name].items()}
        return snapshot
    first = mixed(0)
    source.observe_snapshot(first)
    payload = source.payload()
    restored = GroupSourceObserver.from_payload(assembly, initial, payload, parameters=parameters,
                         torch=torch, device='cpu', allow_host_inputs=True)
    original = GroupTimeMonitor.from_payload(assembly, initial, deepcopy(payload), parameters=parameters, torch=torch)
    before = canonical_hash(source.payload())
    first['states'][next(iter(initial))][0] = 99.
    payload['stamps'][0]['side'] = 'left'
    payload['rows']['cones:native:X']['lo'][0] = 99.
    assert canonical_hash(source.payload()) == canonical_hash(restored.payload()) == before
    for index in (1,2):
        point_value = mixed(index)
        observe(original, point_value)
        restored.observe_snapshot(point_value)
        assert canonical_hash(original.payload()) == canonical_hash(restored.payload())
        assert original.report(.0003) == restored.report(.0003)
    assert restored.host_transfer_metrics['bytes'] == 0  # torch CPU needs no H2D


def test_source_policy_rejects_device_inner_rows_instead_of_hidden_fallback(synthetic):
    torch = pytest.importorskip('torch')
    assembly, initial, parameters = synthetic
    source = GroupSourceObserver(assembly, initial, parameters=parameters, torch=torch,
                                 device='cpu', allow_host_inputs=True)
    snapshot = point(source, initial)
    key = (BC[0], 'BC', 'A')
    snapshot['states'][key] = torch.from_numpy(snapshot['states'][key])
    with pytest.raises(ContractError, match='NumPy inner witnesses'):
        source.observe_snapshot(snapshot)
    assert source.count == 0


@pytest.mark.parametrize('fault', ['fp32','shape','wrong_device','python_list','storage_change','missing_rhs','transition_fp32'])
def test_source_placement_rejects_invalid_or_changed_layout_before_statistics(synthetic, fault):
    torch = pytest.importorskip('torch')
    assembly, initial, parameters = synthetic
    source = GroupSourceObserver(assembly, initial, parameters=parameters, torch=torch,
                                 device='cpu', allow_host_inputs=True)
    source.observe_snapshot(point(source,initial))
    before = canonical_hash(source.payload())
    snapshot = point(source,initial,1)
    key = next(iter(initial))
    if fault == 'fp32':snapshot['states'][key] = snapshot['states'][key].astype(np.float32)
    elif fault == 'shape':snapshot['rhs'][key] = snapshot['rhs'][key][:1]
    elif fault == 'wrong_device':snapshot['states'][key] = torch.empty(3,dtype=torch.float64,device='meta')
    elif fault == 'python_list':snapshot['states'][key] = snapshot['states'][key].tolist()
    elif fault == 'storage_change':
        snapshot['states'][key] = torch.from_numpy(snapshot['states'][key])
        snapshot['rhs'][key] = torch.from_numpy(snapshot['rhs'][key])
    elif fault == 'missing_rhs':snapshot['rhs'].pop(key)
    else:
        key = next(iter(snapshot['transitions']))
        snapshot['transitions'][key] = snapshot['transitions'][key].astype(np.float32)
    with pytest.raises(ContractError):source.observe_snapshot(snapshot)
    assert canonical_hash(source.payload()) == before
