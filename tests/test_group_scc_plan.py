"""Static manufactured IR checks; no simulation, device launch or benchmark."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from retina_sim.compiled_kernel import Plan, Ref, SIGMOID
from retina_sim.contracts import ContractError
from retina_sim.group_scc_plan import build_scc_plan


def fixture_ir(declarations):
    keys = tuple((name, 'fixture', 'state') for name, _ in declarations)
    slices, sizes, offset = {}, {}, 0
    for key, (_, count) in zip(keys, declarations):
        slices[key] = slice(offset, offset + count)
        sizes[key] = count
        offset += count
    model = SimpleNamespace(state_keys=keys, slices=slices, sizes=sizes,
        nstate=offset, identity='static-IR-fixture', parameters={}, contacts={})
    network = SimpleNamespace(compiled=SimpleNamespace(spec=SimpleNamespace(external_inputs=())),
        _local_slices={k: (v,) for k, v in slices.items()},
        state_specs=tuple(SimpleNamespace(shape=(sizes[k],)) for k in keys))
    plan = Plan(network)

    def finish(outputs):
        compiled = SimpleNamespace(model=model, nstate=model.nstate, plan=plan,
            program=np.asarray(plan.program, dtype=np.int64).reshape(-1, 7),
            derivatives=np.asarray([(slices[k].start, -1 if value is None else value.offset,
                sizes[k]) for k, value in zip(keys, outputs)], dtype=np.int64),
            weights=np.asarray(plan.weights, dtype=np.float64),
            indices=np.asarray(plan.indices, dtype=np.int64),
            pointers=np.asarray(plan.pointers, dtype=np.int64), _scratch=plan.workspace())
        return model, compiled

    return plan, tuple(plan.states[k][0] for k in keys), finish


def dependencies(plan):
    return plan.state_dependencies.as_scipy().toarray()


def test_scalar_feedback_is_not_expanded_to_whole_vectors_and_held_are_parents():
    p, (voltage, gate, held), finish = fixture_ir((('voltage', 2), ('gate', 2), ('held', 1)))
    model, compiled = finish((gate + held, voltage * 2, None))
    plan = build_scc_plan(model, compiled)
    assert [plan.component(i).tolist() for i in range(2)] == [[0, 2], [1, 3]]
    np.testing.assert_array_equal(plan.active_indices, [0, 1, 2, 3])
    np.testing.assert_array_equal(plan.held_indices, [4])
    expected = np.zeros((5, 5), dtype=bool)
    expected[0, [2, 4]] = expected[1, [3, 4]] = True
    expected[2, 0] = expected[3, 1] = True
    np.testing.assert_array_equal(dependencies(plan), expected)
    assert plan.component_of_full[4] == -1


def test_take_scatter_and_offset_alias_keep_exact_rows():
    p, (x, y), finish = fixture_ir((('x', 3), ('y', 2)))
    target = p.zero(3)
    p.scatter(target, y, np.array([2, 0]))
    alias = Ref(p, x.offset + 1, 2)
    dy = p.take(p.unary(SIGMOID, alias), np.array([1, 0]))
    model, compiled = finish((target, dy))
    plan = build_scc_plan(model, compiled)
    expected = np.zeros((5, 5), dtype=bool)
    expected[0, 4] = expected[2, 3] = expected[3, 2] = expected[4, 1] = True
    np.testing.assert_array_equal(dependencies(plan), expected)
    assert plan.component_of_full[2] == plan.component_of_full[3]
    assert plan.component_of_full[0] != plan.component_of_full[4]


def test_repeated_scatter_preserves_reads_before_later_writes():
    p, (x, y), finish = fixture_ir((('x', 3), ('y', 2)))
    target = p.zero(2)
    p.scatter(target, x, np.array([0, 1, 0]))
    snapshot = target + 0.
    p.scatter(target, y, np.array([1, 0]))
    model, compiled = finish((p.take(snapshot, np.array([0, 1, 0])), target))
    graph = dependencies(build_scc_plan(model, compiled))
    assert not graph[:3, 3:].any()
    np.testing.assert_array_equal(np.flatnonzero(graph[0]), [0, 2])
    np.testing.assert_array_equal(np.flatnonzero(graph[3]), [0, 2, 4])
    np.testing.assert_array_equal(np.flatnonzero(graph[4]), [1, 3])


def test_overlapping_scatter_fails_closed_instead_of_assuming_a_snapshot():
    p, (x,), finish = fixture_ir((('x', 2),))
    target = p.zero(2)
    p.scatter(target, x, np.array([0, 1]))
    p.scatter(target, target, np.array([1, 0]))
    model, compiled = finish((target,))
    with pytest.raises(ContractError, match='Overlapping.*scatter'):
        build_scc_plan(model, compiled)


def test_csr_uses_pointer_and_index_bases_with_empty_rows_and_duplicates():
    p, (x,), finish = fixture_ir((('x', 3),))
    p.take(x, np.array([1, 1]))
    p.csr(sparse.eye(3, format='csr'), x)
    matrix = sparse.csr_matrix((np.array([1., 0., 2.]), np.array([2, 2, 0]),
                                np.array([0, 2, 2, 3])), shape=(3, 3))
    model, compiled = finish((p.csr(matrix, x),))
    graph = dependencies(build_scc_plan(model, compiled))
    expected = np.zeros((3, 3), dtype=bool)
    expected[0, 2] = expected[2, 0] = True
    np.testing.assert_array_equal(graph, expected)


def test_same_population_gap_scc_is_separate_from_singleton_voltage():
    p, (x,), finish = fixture_ir((('AII', 3),))
    model, compiled = finish((p.csr(sparse.csr_matrix([[0., 1., 0.],
        [1., 0., 0.], [0., 0., 1.]]), x),))
    key = ('AII_AC', 'ReceptorConductanceMembrane', 'V')
    model.state_keys = (key,)
    model.slices, model.sizes = {key: slice(0, 3)}, {key: 3}
    plan = build_scc_plan(model, compiled)
    gap = next(b for b in plan.blocks if b.semantic_name == 'gap_voltage')
    single = next(b for b in plan.blocks if b.semantic_name == 'aii_voltage')
    assert gap.full_indices.tolist() == [0, 1] and gap.coupled
    assert single.full_indices.tolist() == [2] and not single.coupled


def test_plan_identity_and_immutable_maps_bind_ir_not_sampled_values():
    p, (x,), finish = fixture_ir((('x', 2),))
    model, compiled = finish((p.take(x, np.array([1, 0])),))
    first, second = build_scc_plan(model, compiled), build_scc_plan(model, compiled)
    assert first.identity == second.identity
    for array in (first.active_indices, first.component_offsets, first.state_dependencies.indices):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        first.full_slices[model.state_keys[0]] = slice(1, 3)
    compiled.indices = np.array([0, 1], dtype=np.int64)
    assert build_scc_plan(model, compiled).identity != first.identity
    model.identity = 'changed-parameters'
    assert build_scc_plan(model, compiled).typed_ir_identity != first.typed_ir_identity


def test_unknown_opcode_and_cross_allocation_read_fail_closed():
    p, (x,), finish = fixture_ir((('x', 2),))
    model, compiled = finish((x + 1.,))
    compiled.program = compiled.program.copy()
    compiled.program[0, 0] = 999
    with pytest.raises(ContractError, match='opcode'):
        build_scc_plan(model, compiled)
    compiled.program[0, 0] = 0
    compiled.program[0, 3] = 1
    with pytest.raises(ContractError, match='crosses an allocation'):
        build_scc_plan(model, compiled)


@pytest.fixture(scope='module')
def static_group():
    path = Path(__file__).with_name('test_group_inner_dynamics.py')
    spec = importlib.util.spec_from_file_location('scc_static_group_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixture_model()[0]


def test_current_small_group_plan_preserves_gap_glycine_rod_and_native(static_group, monkeypatch):
    import retina_sim.group_inner_compiled as lowering
    # Construct the original typed plan only; no JIT execution or RHS evaluation.
    monkeypatch.setattr(lowering, 'dispatcher', lambda: (None, 0., 'static-test'))
    compiled = lowering.CompiledGroupInnerRHS(static_group)
    plan = build_scc_plan(static_group, compiled)
    s = static_group.slices
    g = s['rods', 'Chen2024PrimateRodCurrent', 'G']
    v = s['rods', 'rod_membrane', 'V']
    np.testing.assert_array_equal(plan.analytic_indices, np.arange(v.start, v.stop))
    np.testing.assert_array_equal(plan.numerical_indices,
        np.setdiff1d(plan.active_indices, plan.analytic_indices))
    graph = dependencies(plan)
    assert graph[v.start, g.start] and not graph[v.start, g.start + 1]
    assert graph[v.start + 1, g.start + 1] and not graph[v.start + 1, g.start]
    for pop, mechanism in (('midget_BC_OFF', 'AII_FMB_glycine'), ('diffuse_BC_OFF', 'AII_DB1_glycine')):
        voltage = s[pop, 'HistoryBipolarTerminal', 'V'].start
        b = s['AII_AC', mechanism, 'b'].start
        a = s['AII_AC', mechanism, 'a'].start
        assert plan.component_of_full[voltage] == plan.component_of_full[b]
        assert plan.component_of_full[voltage] != plan.component_of_full[a]
    aii = s['AII_AC', 'ReceptorConductanceMembrane', 'V'].start
    on = s['midget_BC_ON', 'HistoryBipolarTerminal', 'V'].start
    assert plan.component_of_full[aii] == plan.component_of_full[on]
    assert plan.component_of_full[aii + 1] != plan.component_of_full[on]
    assert plan.counts['inner_total'] == len(plan.active_indices) + len(plan.held_indices)
    assert plan.counts['inner_contact_scalars'] == 7
    assert plan.counts['native_numerical'] == 8
    assert plan.counts['native_analytic'] == 2
    hc = next(block for block in plan.native.blocks if block.name == 'native_HC')
    assert hc.component_sizes.tolist() == [3]
    rank = {int(node): i for i, node in enumerate(plan.component_order)}
    dag = plan.component_dependencies.as_scipy().tocoo()
    assert all(rank[int(a)] < rank[int(b)] for a, b in zip(dag.row, dag.col))
    block_rank = {name: i for i, name in enumerate(plan.block_order)}
    for block in plan.blocks:
        assert all(block_rank[parent] < block_rank[block.name] for parent in block.parent_names)
    compiled._rod_start += 1
    with pytest.raises(ContractError, match='external source'):
        build_scc_plan(static_group, compiled)


def test_native_identity_binds_source_parameters_and_auxiliary_row_identity(static_group, monkeypatch):
    import retina_sim.group_scc_plan as planning
    first = planning._native_layout(static_group)
    monkeypatch.setitem(planning.SOURCE_VALUES, 'tau_m', planning.SOURCE_VALUES['tau_m'] * 2)
    changed_parameters = planning._native_layout(static_group)
    assert changed_parameters.identity != first.identity
    cone = static_group.assembly.populations['cones']
    monkeypatch.setattr(cone, 'recording_auxiliary_group_rows', np.array([0, 0]), raising=False)
    monkeypatch.setattr(cone, 'recording_auxiliary_fine_rows', np.array([4, 9]), raising=False)
    assert planning._native_layout(static_group).identity != changed_parameters.identity
