"""Small original-equation graphs only; no production trajectory or GPU run."""
from copy import deepcopy
from pathlib import Path
import importlib.util

import numpy as np
import pytest

from retina_sim.contracts import ContractError
from retina_sim.group_inner_radau import GroupInnerRadau
from retina_sim.group_radau_math import tableau
from retina_sim.group_inner_dynamics import BC_MECH, MEMBRANE, RODS


@pytest.fixture
def model_state():
    spec=importlib.util.spec_from_file_location('inner_radau_fixture',Path(__file__).with_name('test_group_inner_dynamics.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.fixture_model()


def prepared(model,state):
    solver=GroupInnerRadau(model)
    solver._weights={k:1e-10+1e-7*abs(v) for k,v in state.items()}
    solver._budgets={k:v.copy() for k,v in solver._weights.items()}
    return solver


def test_every_stage_satisfies_actual_original_equations(model_state):
    model,state=model_state
    state['midget_BC_OFF',BC_MECH,'V'][:]=-.08
    state['AII_AC',MEMBRANE,'V'][:]=[-.043,-.059]
    for c,row in model.contacts.items():
        if c!='individual_rod_RBC_contacts':
            state[row['route'].pre,c,'a'][:]=.01
            state[row['route'].pre,c,'b'][:]=.1
    solver=prepared(model,state);h=2e-5
    end,U,F=solver._half(state,h);A,_,nodes,_,_=tableau()
    for j,time in enumerate(nodes*h):
        stage={k:U[k][j] if k in U else v for k,v in state.items()}
        original=model.rhs(time,stage)
        for k in solver.active:
            np.testing.assert_allclose(F[k][j],original[k],rtol=3e-14,atol=3e-13,err_msg=str(k))
    for k in solver.active:
        if k==solver.rod_v:continue
        residual=U[k]-state[k]-h*(A@F[k])
        assert np.max(abs(residual)/solver._budgets[k])<.011,k
        np.testing.assert_array_equal(end[k],U[k][-1])
    assert set(U)==set(solver.active)
    assert set(U).isdisjoint(solver.held)


def test_filtered_estimator_includes_original_downstream_jacobian(model_state):
    model,state=model_state;solver=prepared(model,state)
    for c,row in model.contacts.items():
        if c!='individual_rod_RBC_contacts':
            state[row['route'].pre,c,'a'][:]=.01
            state[row['route'].pre,c,'b'][:]=.1
    active=np.concatenate([np.arange(model.slices[k].start,model.slices[k].stop) for k in solver.active])
    packed=model.pack(state);J=np.zeros((len(active),len(active)))
    for column,index in enumerate(active):
        delta=1e-6*max(abs(packed[index]),1e-3)
        plus=packed.copy();minus=packed.copy();plus[index]+=delta;minus[index]-=delta
        J[:,column]=(model.pack(model.rhs(0.,model.unpack(plus)))[active]-model.pack(model.rhs(0.,model.unpack(minus)))[active])/(2*delta)
    rng=np.random.default_rng(41)
    d={k:rng.normal(size=v.shape)*1e-5 for k,v in state.items() if k in solver.active}
    d[solver.rod_v][:]=0.
    shift=4e-5;expected=np.linalg.solve(np.eye(len(active))-shift*J,np.concatenate([d[k] for k in solver.active]))
    actual=np.concatenate([solver._filter(state,d,shift)[k] for k in solver.active])
    np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=5e-13)


def test_two_half_step_output_matches_independent_reference_and_exact_held_states(model_state):
    model,state=model_state;before=deepcopy(state);stop=4e-5
    reference=model.advance_reference(state,0.,stop,rtol=2e-12,atol=2e-14)
    solver=GroupInnerRadau(model)
    result=solver.advance(state,0.,stop,rtol=1e-7,atol=1e-10)
    for t,side in ((0.,'right'),(1.7e-5,'left'),(stop,'left')):
        actual=result.evaluate(t,side);expected=reference.evaluate(t,side)
        for k in model.state_keys:
            np.testing.assert_allclose(actual[k],expected[k],rtol=1e-7,atol=1e-10,err_msg=str(k))
        for k in solver.held:np.testing.assert_array_equal(actual[k],before[k])
    assert result.metrics['steps']<=64 and result.metrics['attempts']<=127
    assert result.metrics['filtered_solves']==2*result.metrics['attempts']
    assert result.metrics['original_full_dimension']==model.nstate
    assert result.metrics['all_contact_states_retained']
    assert result.metrics['uniform_error_certificate']=='not_established'
    derivative=result.representation_derivative(stop,'left');original=result.rhs_at(stop,'left')
    for k in model.state_keys:np.testing.assert_allclose(derivative[k],original[k],rtol=2e-10,atol=2e-11)
    for k in state:np.testing.assert_array_equal(state[k],before[k])
    result.state[solver.rod_v][:]=9.
    np.testing.assert_array_equal(state[solver.rod_v],before[solver.rod_v])
    with pytest.raises(ContractError,match='wrong side'):result.evaluate(stop,'right')


def test_rod_analytic_anchors_publish_polynomial_dense_and_author_state_is_held(model_state):
    model,state=model_state;solver=GroupInnerRadau(model)
    result=solver.advance(state,0.,2e-5,rtol=1e-7,atol=1e-10)
    decay=model.gL['rods']/model.C['rods']
    equilibrium=model.par('rods','EL')-model.members['rods']*(model.rod_current(state)-model.par('rods','current_anchor'))/model.gL['rods']
    t=1.113e-5;expected=state[solver.rod_v]-np.expm1(-decay*t)*(equilibrium-state[solver.rod_v])
    actual=result.evaluate(t,'right')
    assert all(rep.kind=='quintic_hermite' for rep in result.representations[solver.rod_v][0][1].representations)
    np.testing.assert_allclose(actual[solver.rod_v],expected,rtol=0,atol=2e-17)
    for n in ('R','P','G','C','S'):np.testing.assert_array_equal(actual['rods',RODS,n],state['rods',RODS,n])


def test_failure_and_guard_leave_caller_state_unchanged(model_state):
    model,state=model_state;before=deepcopy(state);solver=GroupInnerRadau(model)
    invalid=deepcopy(state);invalid['AII_AC','AII_FMB_glycine','a'][:]=1.
    with pytest.raises(ContractError,match='conductance'):solver.advance(invalid,0.,1e-5,rtol=1e-6,atol=1e-9)
    calls=[]
    def guard():
        calls.append(1)
        return len(calls)<5
    solver=GroupInnerRadau(model,guard=guard)
    with pytest.raises(ContractError,match='guard'):solver.advance(state,0.,1e-5,rtol=1e-6,atol=1e-9)
    assert not hasattr(solver,'_temporal_used')
    for k in state:np.testing.assert_array_equal(state[k],before[k])


def test_frozen_weights_do_not_follow_candidate_state(model_state):
    model,state=model_state;solver=GroupInnerRadau(model)
    solver.advance(state,0.,1e-5,rtol=1e-6,atol=1e-9)
    expected={k:v.copy() for k,v in solver._weights.items()}
    changed=deepcopy(state);changed['rods','rod_membrane','V'][:]=-.04
    solver.advance(changed,1e-5,2e-5,rtol=1e-6,atol=1e-9)
    for k in expected:np.testing.assert_array_equal(solver._weights[k],expected[k])


def test_off_glycine_normalization_jacobian_on_all_clamp_branches(model_state):
    model,state=model_state;solver=prepared(model,state)
    p='midget_BC_OFF';c='AII_FMB_glycine';a=np.array([.013]);b=np.array([.12]);rate=np.array([3.]);target=solver._target(p,state)
    for voltage in (-.12,-.053,.02):
        V=np.array([voltage]);terms=solver._off_terms(p,c,V,b,a,rate,target)
        fV,fb,dv,vb,bv,_=terms;eps=1e-7
        plus=solver._off_terms(p,c,V+eps,b,a,rate,target)
        minus=solver._off_terms(p,c,V-eps,b,a,rate,target)
        np.testing.assert_allclose((plus[0]-minus[0])/(2*eps),dv,rtol=2e-8,atol=1e-7)
        np.testing.assert_allclose((plus[1]-minus[1])/(2*eps),bv,rtol=2e-8,atol=1e-7)
        plus=solver._off_terms(p,c,V,b+eps,a,rate,target)
        minus=solver._off_terms(p,c,V,b-eps,a,rate,target)
        np.testing.assert_allclose((plus[0]-minus[0])/(2*eps),vb,rtol=2e-8,atol=1e-7)


def test_off_arrowhead_retains_two_independent_incoming_contacts(model_state):
    model,state=model_state;c='AII_FMB_glycine';row=model.contacts[c]
    # Extend the manufactured equation fixture with a second distinct contact.
    row.update(pre=np.array([0,1]),post=np.array([0,0]),coefficient=np.array([1e-10,3e-10]))
    for name,value in (('a',[.01,.02]),('b',[.12,.17])):
        k=('AII_AC',c,name);state[k]=np.array(value,dtype=np.float64);model.sizes[k]=2
    offset=0
    for k in model.state_keys:model.slices[k]=slice(offset,offset+model.sizes[k]);offset+=model.sizes[k]
    model.nstate=offset
    solver=prepared(model,state);h=3e-5;end,U,F=solver._half(state,h);A,_,nodes,_,_=tableau()
    for k in (('midget_BC_OFF',BC_MECH,'V'),('AII_AC',c,'b')):
        assert np.max(abs(U[k]-state[k]-h*(A@F[k]))/solver._budgets[k])<.011
    assert U['AII_AC',c,'b'].shape==(3,2)
    assert not np.array_equal(U['AII_AC',c,'b'][:,0],U['AII_AC',c,'b'][:,1])
    for i,time in enumerate(nodes*h):
        stage={k:U[k][i] if k in U else v for k,v in state.items()}
        original=model.rhs(time,stage)
        for k in (('midget_BC_OFF',BC_MECH,'V'),('AII_AC',c,'b')):
            np.testing.assert_allclose(F[k][i],original[k],rtol=2e-14,atol=2e-13)


def test_torch_cpu_stages_and_device_original_rhs_hook_match_numpy(model_state):
    torch=pytest.importorskip('torch');model,state=model_state;calls=[]
    def original(t,packed):
        calls.append(t)
        value=model.pack(model.rhs(t,model.unpack(packed.cpu().numpy())))
        return torch.as_tensor(value,dtype=torch.float64)
    numpy_result=GroupInnerRadau(model).advance(state,0.,1e-5,rtol=1e-7,atol=1e-10)
    torch_result=GroupInnerRadau(model,torch=torch,device='cpu',original_rhs=original).advance(state,0.,1e-5,rtol=1e-7,atol=1e-10)
    assert len(calls)==torch_result.metrics['fresh_original_rhs_calls']
    for k in state:np.testing.assert_allclose(numpy_result.state[k],torch_result.state[k],rtol=1e-13,atol=1e-15)


def test_controller_checkpoint_restores_frozen_scales_schedule_and_counters(model_state):
    model,state=model_state;solver=GroupInnerRadau(model)
    prefix=solver.advance(state,0.,2e-5,rtol=1e-7,atol=1e-10)
    checkpoint=solver.controller_state();saved_counts=dict(checkpoint['statistics'])
    restored=GroupInnerRadau(model);restored.restore_controller(checkpoint)
    expected=solver.advance(prefix.state,2e-5,4e-5,rtol=1e-7,atol=1e-10)
    actual=restored.advance(prefix.state,2e-5,4e-5,rtol=1e-7,atol=1e-10)
    assert actual.accepted_times==expected.accepted_times
    for k in state:np.testing.assert_array_equal(actual.state[k],expected.state[k])
    for name,count in saved_counts.items():assert restored.counters[name]>=count
    a=solver.controller_state();b=restored.controller_state()
    for name in ('next_step','window_end','window_attempts','window_accepts','statistics'):assert a[name]==b[name]
    for name in ('error_weights','budgets','error_ledger'):np.testing.assert_array_equal(a[name],b[name])
    checkpoint['error_weights'][:]=9.
    assert not np.all(restored.controller_state()['error_weights']==9.)
    corrupt=restored.controller_state();corrupt['budgets']*=2
    with pytest.raises(ContractError,match='frozen scale'):restored.restore_controller(corrupt)
    np.testing.assert_array_equal(restored.controller_state()['budgets'],b['budgets'])


def test_cuda_contact_inverse_rows_keep_original_summation_order():
    from retina_sim.group_inner_radau_kernels import stable_contact_rows
    post=np.array([1,0,1,0,1],dtype=np.int64)
    indices,pointers=stable_contact_rows(post,3)
    np.testing.assert_array_equal(indices,[1,3,0,2,4])
    np.testing.assert_array_equal(pointers,[0,2,5,5])
    values=np.array([1e16,3.,1.,4.,-1e16])
    pooled=np.zeros(3)
    for row in range(3):
        for i in indices[pointers[row]:pointers[row+1]]:pooled[row]+=values[i]
    np.testing.assert_array_equal(pooled,np.bincount(post,weights=values,minlength=3))
    with pytest.raises(ContractError,match='outside'):stable_contact_rows(np.array([3]),3)


def test_prebuilt_plan_is_identity_checked_without_second_graph_assembly(model_state,monkeypatch):
    from dataclasses import replace
    from retina_sim.group_inner_compiled import CompiledGroupInnerRHS
    from retina_sim.group_scc_plan import build_scc_plan
    import retina_sim.group_scc_plan as planning
    model,state=model_state;compiled=CompiledGroupInnerRHS(model);plan=build_scc_plan(model,compiled)
    def forbidden(*args,**kwargs):raise AssertionError('A supplied bound plan must be reused')
    monkeypatch.setattr(planning,'build_scc_plan',forbidden)
    solver=GroupInnerRadau(model,compiled=compiled,scc_plan=plan)
    assert solver.plan is plan
    assert solver.identity['schedule']==solver.schedule.identity
    with pytest.raises(ContractError,match='typed IR identity'):
        GroupInnerRadau(model,compiled=compiled,scc_plan=replace(plan,typed_ir_identity='different'))


def test_halfstep_dispatches_bound_graph_order_not_handler_registration_order(model_state):
    model,state=model_state;solver=prepared(model,state);calls=[]
    actions={}
    for name,(action,args) in reversed(tuple(solver._family_actions.items())):
        def record(y,h,U,F,cache,*args,action=action,name=name):
            calls.append(name);return action(y,h,U,F,cache,*args)
        actions[name]=(record,args)
    solver._family_actions=actions
    solver._half(state,2e-5)
    assert tuple(calls)==solver.schedule.order
    assert tuple(calls)!=tuple(actions)
    assert solver.schedule.families['gap_voltage'].parents==('rbc_aii_ampa',)
    assert set(solver.schedule.families['midget_BC_OFF_glycine'].writes)=={
        ('midget_BC_OFF',BC_MECH,'V'),('AII_AC','AII_FMB_glycine','b')}


def test_fixed_gap_workspaces_are_reused_without_aliasing_accepted_outputs(model_state):
    model,state=model_state;solver=prepared(model,state)
    banks=(solver._gap_stage_workspace,solver._gap_filter_workspace)
    pointers=tuple(tuple(a.__array_interface__['data'][0] for a in bank.arrays) for bank in banks)
    result=solver.advance(state,0.,2e-5,rtol=1e-7,atol=1e-10)
    saved={k:v.copy() for k,v in result.state.items()}
    solver.advance(result.state,2e-5,4e-5,rtol=1e-7,atol=1e-10)
    assert pointers==tuple(tuple(a.__array_interface__['data'][0] for a in bank.arrays) for bank in banks)
    assert all(bank.uses>0 for bank in banks)
    assert result.metrics['gmres_owned_workspace_bytes']==sum(bank.owned_bytes for bank in banks)
    for k in saved:np.testing.assert_array_equal(result.state[k],saved[k])
