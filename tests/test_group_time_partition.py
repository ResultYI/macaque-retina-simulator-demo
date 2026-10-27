"""Source/state/dependency tests only; no neural initial state or advancement."""
from pathlib import Path
from types import SimpleNamespace
import shutil

import numpy as np
import pytest
from scipy import sparse

from retina_sim.contracts import ContractError
from retina_sim.group_time_partition import compile_time_partition, guard_source_dependencies, EQUATION_SOURCE_SHA256
from retina_sim.spatial_group_model import (GroupModelAssembly, StateBlock, POPULATIONS,
    REQUIRED_ROUTE_PAIRS, BC, AC, RGC)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def layout():
    counts = {n: 1 for n in POPULATIONS}
    counts.update(AII_AC=50, midget_BC_ON=2)
    populations = {n: SimpleNamespace(group_count=k) for n,k in counts.items()}
    contact_pairs = {('rods','rod_BC'):'individual_rod_RBC_contacts',
        ('rod_BC','AII_AC'):'RBC_AII_AMPA', ('AII_AC','midget_BC_OFF'):'AII_FMB_glycine',
        ('AII_AC','diffuse_BC_OFF'):'AII_DB1_glycine'}
    gap_names = ('AII_AC','midget_BC_ON','diffuse_BC_ON','S_ON_BC')
    gap_pairs = {('AII_AC','AII_AC'), *(p for n in gap_names[1:] for p in (('AII_AC',n),(n,'AII_AC')))}
    routes = {}
    for pre,post in REQUIRED_ROUTE_PAIRS:
        name=pre+'_to_'+post
        kind='gap' if (pre,post) in gap_pairs else 'native' if (pre,post) in (('cones','HC'),('HC','cones')) else 'primary_chemical' if (pre,post) in contact_pairs else 'inner_chemical'
        routes[name]=SimpleNamespace(name=name,pre=pre,post=post,kind=kind,
            contact_component=contact_pairs.get((pre,post)),contact_groups=2 if (pre,post) in contact_pairs else 0)
    size=sum(counts[n] for n in gap_names)
    # The NEW aggregate graph has a 51-node component, deliberately > old42.
    edges=np.arange(50)
    adj=sparse.coo_matrix((np.ones(100),(np.r_[edges,edges+1],np.r_[edges+1,edges])),shape=(size,size)).tocsr()
    lap=sparse.diags(np.asarray(adj.sum(axis=1)).ravel())-adj
    def matrix(path):
        if path=='gap/laplacian_S.npz':
            return lap
        for r in routes.values():
            if path=='routes/'+r.name+'/contact_coefficient_S.npz':
                return sparse.csr_matrix((np.array([1.]),(np.array([0]),np.array([0]))),shape=(counts[r.post],counts[r.pre]))
        raise AssertionError(path)
    model=SimpleNamespace(populations=populations,routes=routes,matrix=matrix,
        manifest={'gap':{'population_order':gap_names}},artifact_sha256='fixture-artifact')
    model.state_blocks=GroupModelAssembly._states(model)
    model.probability_blocks=GroupModelAssembly._probability_blocks(model)
    model.auxiliary_z1_blocks=(StateBlock('cones','selected_original_z1_auxiliary','z1_aux','vH_native_voltage',
        'selected original z1',1,0,'uninitialized_selected_original_auxiliary'),)
    model.state_scalar_count=sum(b.count for b in model.state_blocks)
    model.probability_scalar_count=sum(b.count for b in model.probability_blocks)
    model.auxiliary_z1_scalar_count=1
    return model


def test_all_declared_scalars_exactly_partitioned_and_no_solver(layout):
    plan=compile_time_partition(layout,ROOT)
    assert len(plan.nodes)==len(layout.state_blocks)+len(layout.probability_blocks)+1
    assert sum(len(p.rows) for p in plan.row_partitions)==layout.state_scalar_count+layout.probability_scalar_count+1
    description=plan.describe()
    assert description['numerical_backend_enabled'] is False
    assert description['model_forward_started'] is False
    assert description['solver_step_seconds'] is None
    assert description['scalar_counts']['selected_original_z1']==1
    with pytest.raises(ValueError):
        plan.row_partitions[0].rows[0]=99


def test_gap_components_from_aggregate_allow_greater_than_original_42(layout):
    plan=compile_time_partition(layout,ROOT)
    assert plan.gap_component_sizes.max()==51
    assert plan.describe()['aggregate_gap']['inherits_original_42_node_bound'] is False
    parts=[p for p in plan.row_partitions if p.state_id=='midget_BC_ON:HistoryBipolarTerminal:V']
    assert {p.category for p in parts}=={'fast_joint','analytic_affine'}
    np.testing.assert_array_equal(next(p.rows for p in parts if p.category=='fast_joint'),[0])
    np.testing.assert_array_equal(next(p.rows for p in parts if p.category=='analytic_affine'),[1])


def test_native_HC_feedback_enters_release_not_phototransduction(layout):
    plan=compile_time_partition(layout,ROOT)
    by={n.state_id:n for n in plan.nodes}
    for name in ('X','C','Vis','g_i','Vbar'):
        assert not any(d.startswith('HC:') for d in by['cones:reduced_native:'+name].rhs_dependencies)
    ports={p.port_id:p for p in plan.ports}
    assert 'HC:reduced_native:Vh' in ports['port:cones:q'].dependencies
    assert ports['port:HC:p'].expression=='(A@Vbar/VI)^mu'
    assert next(p.category for p in plan.row_partitions if p.state_id=='HC:reduced_native:z2')=='fast_joint'


def test_event_history_is_not_a_continuous_ODE_and_BC_A_is_feedforward(layout):
    plan=compile_time_partition(layout,ROOT)
    nodes={n.state_id:n for n in plan.nodes}
    h=nodes['midget_BC_ON:HistoryBipolarTerminal:history_1']
    assert h.rhs_dependencies==() and h.commit_events==('BC_tick',)
    assert h.commit_dependencies==('midget_BC_ON:HistoryBipolarTerminal:history_0',)
    V=nodes['midget_BC_ON:HistoryBipolarTerminal:V']
    assert 'port:cones:release' not in V.rhs_dependencies
    assert not any(d.endswith(':A') for d in V.rhs_dependencies)
    A=nodes['midget_BC_ON:HistoryBipolarTerminal:A']
    assert A.rhs_dependencies==('midget_BC_ON:HistoryBipolarTerminal:A','midget_BC_ON:HistoryBipolarTerminal:terminal_release')
    assert all(n.rhs_dependencies==() for n in plan.nodes if n.mechanism=='Chen2024PrimateRodCurrent')
    assert all(not any('history_probability' in d for d in n.rhs_dependencies) for n in plan.nodes if n.mechanism=='RGC_spike_history')


def test_glycine_current_voltage_normalization_and_resource_rate_preserved(layout):
    plan=compile_time_partition(layout,ROOT)
    ports={p.port_id:p for p in plan.ports}
    gly=ports['port:AII_FMB_glycine:conductance']
    assert 'midget_BC_OFF:HistoryBipolarTerminal:V' in gly.dependencies
    assert 'Z(current_postsynaptic_V)' in gly.expression
    assert ports['port:rod_BC:event_rate'].dependencies==('rod_BC:ReceptorConductanceMembrane:V','rod_BC:ReceptorConductanceMembrane:x')
    assert next(p.category for p in plan.row_partitions if ':AII_FMB_glycine:b' in p.state_id)=='fast_joint'
    assert next(p.category for p in plan.row_partitions if ':AII_FMB_glycine:a' in p.state_id)=='exponential_forced'


def test_unknown_states_or_changed_route_dependency_reject(layout):
    layout.state_blocks+= (StateBlock('cones','reduced_native','unexpected','x','x',1,0),)
    layout.state_scalar_count+=1
    with pytest.raises(ContractError,match='Unknown cone equation'):
        compile_time_partition(layout,ROOT)
    layout.state_blocks=layout.state_blocks[:-1]
    layout.state_scalar_count-=1
    layout.routes.pop(next(iter(layout.routes)))
    with pytest.raises(ContractError,match='route dependency'):
        compile_time_partition(layout,ROOT)


def test_existing_route_with_changed_mechanism_or_contact_role_rejects(layout):
    route=layout.routes['cones_to_midget_BC_ON']
    route.kind='gap'
    with pytest.raises(ContractError,match='mechanism/port role'):
        compile_time_partition(layout,ROOT)
    route.kind='inner_chemical'
    route.contact_component='RBC_AII_AMPA'
    with pytest.raises(ContractError,match='contact mechanisms|mechanism/port role'):
        compile_time_partition(layout,ROOT)


def test_equation_source_drift_rejected_without_importing_model(tmp_path):
    for relative in EQUATION_SOURCE_SHA256:
        destination=tmp_path/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,destination)
    assert guard_source_dependencies(tmp_path)==dict(EQUATION_SOURCE_SHA256)
    with (tmp_path/'retina_sim/rod_pathway.py').open('a',encoding='utf8') as stream:
        stream.write('\n# changed equation implementation identity\n')
    with pytest.raises(ContractError,match='re-audit required'):
        guard_source_dependencies(tmp_path)
