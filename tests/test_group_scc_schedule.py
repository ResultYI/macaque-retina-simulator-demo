"""Tiny typed-IR schedule checks; no neural trajectory or device execution."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from retina_sim.contracts import ContractError
from retina_sim.group_scc_plan import FrozenCSR, build_scc_plan
from retina_sim.group_scc_schedule import FamilySpec, bind_scc_schedule, topological_order


_fixture_spec=importlib.util.spec_from_file_location('scc_schedule_ir_fixture',Path(__file__).with_name('test_group_scc_plan.py'))
_fixture_module=importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
fixture_ir=_fixture_module.fixture_ir


def chain(*, reverse=False):
    p,(x,y,z,held),finish=fixture_ir((('x',2),('y',2),('z',2),('held',1)))
    outputs=(x+y,y+z,z+held,None) if reverse else (x+held,y+x,z+y,None)
    model,compiled=finish(outputs)
    plan=build_scc_plan(model,compiled)
    kx,ky,kz,kh=model.state_keys
    families=(
        FamilySpec('z_upstream',(kx,),((kx,kx),(kx,kh),(kx,ky))),
        FamilySpec('m_middle',(ky,),((ky,ky),(ky,kx),(ky,kz))),
        FamilySpec('a_downstream',(kz,),((kz,kz),(kz,ky),(kz,kh))),
    )
    return plan,families,(kx,ky,kz,kh)


def execute_values(schedule,held_key):
    calls=[];U={};F={}
    def handler(family):
        def publish(u,f):
            calls.append(family.name)
            value=np.ones((3,2))
            slope=np.zeros((3,2))
            for parent in family.parent_keys:
                value=value+u[parent]
                slope=slope+f[parent]
            for key in family.writes:
                u[key]=value.copy();f[key]=slope.copy()
        return publish
    handlers={name:handler(family) for name,family in schedule.families.items()}
    schedule.execute(handlers,U,F,held_state={held_key:np.ones(1)})
    return tuple(calls),U,F


def test_actual_typed_dependencies_drive_execution_independent_of_declaration_order():
    plan,families,keys=chain()
    first=bind_scc_schedule(plan,families)
    reordered=bind_scc_schedule(plan,families[::-1])
    assert first.order==reordered.order==('z_upstream','m_middle','a_downstream')
    assert first.edges==(('m_middle','a_downstream'),('z_upstream','m_middle'))
    assert first.families['m_middle'].parent_keys==(keys[0],)
    assert first.families['z_upstream'].held_reads==(keys[3],)
    for schedule in (first,reordered):
        calls,U,F=execute_values(schedule,keys[3])
        assert calls==schedule.order
        for k,value in zip(keys[:3],(1.,2.,3.)):
            np.testing.assert_array_equal(U[k],np.full((3,2),value))
            np.testing.assert_array_equal(F[k],np.zeros((3,2)))
            assert not U[k].flags.writeable and not F[k].flags.writeable


def test_permitted_changed_graph_reverses_family_execution_without_hand_order():
    forward,families,keys=chain()
    reverse,_,_=chain(reverse=True)
    a=bind_scc_schedule(forward,families)
    b=bind_scc_schedule(reverse,families)
    assert a.order==tuple(reversed(b.order))
    assert a.identity!=b.identity
    calls,U,_=execute_values(b,keys[3])
    assert calls==('a_downstream','m_middle','z_upstream')
    for k,value in zip(keys[:3],(3.,2.,1.)):
        np.testing.assert_array_equal(U[k],np.full((3,2),value))


def test_missing_active_coverage_and_held_writes_fail_closed():
    plan,families,keys=chain()
    with pytest.raises(ContractError,match='cover every active'):
        bind_scc_schedule(plan,families[:-1])
    held=FamilySpec('held',(keys[3],),())
    with pytest.raises(ContractError,match='without held writes'):
        bind_scc_schedule(plan,(*families,held))


def test_overlapping_family_ownership_fails_closed():
    plan,families,keys=chain()
    overlap=FamilySpec('duplicate_owner',(keys[0],),families[0].allowed_dependencies)
    with pytest.raises(ContractError,match='ownership overlaps'):
        bind_scc_schedule(plan,(*families,overlap))


def test_a_true_scalar_scc_cannot_be_split_between_equation_families():
    p,(x,y),finish=fixture_ir((('x',2),('y',2)))
    model,compiled=finish((x+y,y+x));plan=build_scc_plan(model,compiled)
    kx,ky=model.state_keys
    families=(FamilySpec('x',(kx,),((kx,kx),(kx,ky))),FamilySpec('y',(ky,),((ky,ky),(ky,kx))))
    assert plan.component_of_full[model.slices[kx].start]==plan.component_of_full[model.slices[ky].start]
    with pytest.raises(ContractError,match='true scalar SCC is split'):
        bind_scc_schedule(plan,families)


def test_cross_row_feedback_requires_explicit_internal_coupling_labels():
    p,(x,),finish=fixture_ir((('x',3),))
    model,compiled=finish((p.take(x,np.array([1,0,2])),));plan=build_scc_plan(model,compiled)
    key=model.state_keys[0]
    pointwise=FamilySpec('pointwise',(key,),((key,key),))
    with pytest.raises(ContractError,match='within-family scalar coupling'):
        bind_scc_schedule(plan,(pointwise,))
    coupled=replace(pointwise,row_groups=((key,np.array([0,0,1],dtype=np.int64)),))
    bound=bind_scc_schedule(plan,(coupled,))
    assert bound.order==('pointwise',)
    assert len(bound.families['pointwise'].component_ids)==2


def test_unregistered_parent_dependency_fails_even_if_ownership_is_complete():
    plan,families,keys=chain()
    missing_parent=replace(families[1],allowed_dependencies=((keys[1],keys[1]),))
    with pytest.raises(ContractError,match='Unregistered equation dependency'):
        bind_scc_schedule(plan,(families[0],missing_parent,families[2]))


@pytest.mark.parametrize('change',('remove','add'))
def test_component_parent_graph_must_agree_with_actual_scalar_dependencies(change):
    plan,families,keys=chain()
    dag=plan.component_dependencies.as_scipy().tolil()
    if change=='remove':
        row,col=dag.tocoo().row[0],dag.tocoo().col[0]
        dag[row,col]=False
        # A whole family edge consists of both manufactured rows.
        parent=plan.component_of_full[plan.full_slices[keys[0]].start]
        child=plan.component_of_full[plan.full_slices[keys[1]].start]
        for i in range(2):
            p=plan.component_of_full[plan.full_slices[keys[0]].start+i]
            c=plan.component_of_full[plan.full_slices[keys[1]].start+i]
            dag[p,c]=False
    else:
        parent=plan.component_of_full[plan.full_slices[keys[2]].start]
        child=plan.component_of_full[plan.full_slices[keys[0]].start]
        dag[parent,child]=True
    altered=replace(plan,component_dependencies=FrozenCSR.from_matrix(dag.tocsr()))
    with pytest.raises(ContractError,match='parent graph disagrees'):
        bind_scc_schedule(altered,families)


def test_cyclic_family_quotient_is_rejected_even_when_scalar_graph_is_acyclic():
    p,(x,y,z),finish=fixture_ir((('x',1),('y',1),('z',1)))
    model,compiled=finish((x,y+x,z+y));plan=build_scc_plan(model,compiled)
    kx,ky,kz=model.state_keys
    families=(FamilySpec('outer',(kx,kz),((kx,kx),(kz,kz),(kz,ky))),
              FamilySpec('middle',(ky,),((ky,ky),(ky,kx))))
    with pytest.raises(ContractError,match='dependency cycle'):
        bind_scc_schedule(plan,families)


def valid_handlers(schedule):
    handlers={}
    for name,family in schedule.families.items():
        def write(u,f,family=family):
            for key in family.writes:u[key]=np.ones((3,2));f[key]=np.zeros((3,2))
        handlers[name]=write
    return handlers


def test_execute_requires_parent_states_and_slopes_to_be_published():
    plan,families,keys=chain();schedule=bind_scc_schedule(plan,families)
    misordered=replace(schedule,order=tuple(reversed(schedule.order)))
    with pytest.raises(ContractError,match='parent stages/slopes are not published'):
        misordered.execute(valid_handlers(schedule),{}, {},held_state={keys[3]:np.ones(1)})


@pytest.mark.parametrize('violation',('missing_state','missing_slope','wrong_read','foreign_write','overwrite_parent','repeat_write'))
def test_execute_rejects_incomplete_or_unowned_family_operations(violation):
    plan,families,keys=chain();schedule=bind_scc_schedule(plan,families);handlers=valid_handlers(schedule)
    kx,ky,kz,kh=keys
    name='m_middle' if violation=='overwrite_parent' else 'z_upstream'
    def invalid(u,f):
        if violation=='missing_state':f[kx]=np.zeros((3,2))
        elif violation=='missing_slope':u[kx]=np.ones((3,2))
        elif violation=='wrong_read':u[ky]
        elif violation=='foreign_write':u[ky]=np.ones((3,2))
        elif violation=='overwrite_parent':u[kx]=np.ones((3,2))
        else:u[kx]=np.ones((3,2));u[kx]=np.ones((3,2))
    handlers[name]=invalid
    with pytest.raises(ContractError):schedule.execute(handlers,{}, {},held_state={kh:np.ones(1)})


def test_execute_requires_held_sources_and_complete_handler_registry():
    plan,families,keys=chain();schedule=bind_scc_schedule(plan,families)
    with pytest.raises(ContractError,match='held source is absent'):
        schedule.execute(valid_handlers(schedule),{}, {},held_state={})
    with pytest.raises(ContractError,match='every bound family handler'):
        schedule.execute({}, {}, {},held_state={keys[3]:np.ones(1)})


def test_published_numpy_parents_cannot_be_mutated_by_a_downstream_family():
    plan,families,keys=chain();schedule=bind_scc_schedule(plan,families);handlers=valid_handlers(schedule)
    def mutate(u,f):u[keys[0]][0,0]=99.
    handlers['m_middle']=mutate
    with pytest.raises(ValueError,match='read-only'):
        schedule.execute(handlers,{}, {},held_state={keys[3]:np.ones(1)})


@pytest.mark.parametrize('names,edges,message',[
    (('a','a'),(),'Distinct'),
    (('a','b'),(('a','missing'),),'unknown'),
    (('a',),(('a','a'),),'self cycle'),
    (('a','b'),(('a','b'),('b','a')),'dependency cycle'),
])
def test_generic_topological_order_rejects_invalid_native_or_inner_graphs(names,edges,message):
    with pytest.raises(ContractError,match=message):topological_order(names,edges)


def test_generic_topological_order_uses_graph_and_deterministic_ready_ties():
    names=('z','a','b','c')
    assert topological_order(names,(('z','c'),('a','b'),('a','b')))==('a','b','z','c')
    assert topological_order(tuple(reversed(names)),(('a','b'),('z','c')))==('a','b','z','c')


def test_omitting_required_held_source_mapping_fails_closed():
    plan,families,_=chain();schedule=bind_scc_schedule(plan,families)
    handlers={name:lambda u,f:None for name in schedule.order}
    with pytest.raises(ContractError,match='held source is absent'):
        schedule.execute(handlers,{}, {})
