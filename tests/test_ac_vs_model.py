"""Independent equations, legacy isolation and path ablation checks."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.special import expit

from retina_sim.ac_vs_model import (ACVSInner, MECHANISM, LEGACY_S, load_config,
    equilibrium, full_input_kernel, redesign_seed)
from retina_sim.group_inner_dynamics import AC_MECH, BC_MECH, key
from retina_sim.spatial_group_model import StateBlock, AC, RGC
from retina_sim.contracts import ContractError
from retina_sim.ac_vs_monitor import DynamicDefectMonitor
from retina_sim.group_time_monitor import frozen_budget

ROOT=Path(__file__).resolve().parents[1]


def new_fixture():
    spec=importlib.util.spec_from_file_location('old_ac_fixture',Path(__file__).with_name('test_group_inner_dynamics.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old,state=module.fixture_model()
    a=deepcopy(old.assembly)
    config=load_config(ROOT/'configurations/ac_vs_functional_v1.json')
    a.ac_config=config;a.max_step=.0001
    a.ac_populations=(LEGACY_S,*(c['pool'] for c in config['channels']))
    a.routes={n:r for n,r in a.routes.items() if (r.pre not in AC and r.post not in AC) or r.post==LEGACY_S or (r.pre==LEGACY_S and r.post=='small_bistratified_S_ON')}
    original_matrix=a.matrix;matrices={}
    blocks=[b for b in a.state_blocks if b.population not in ('local_crossover_AC','widefield_AC')]
    for c in config['channels']:
        pop=c['pool'];a.populations[pop]=SimpleNamespace(group_count=1,counts=np.array([1]))
        for pre,post,matrix in ((c['pre'],pop,sparse.csr_matrix(np.ones((1,old.counts[c['pre']]))/old.counts[c['pre']])),(pop,c['target'],sparse.eye(1,format='csr'))):
            name=pre+'_to_'+post
            a.routes[name]=SimpleNamespace(name=name,pre=pre,post=post,kind='inner_chemical',contact_component=None)
            matrices['routes/'+name+'/mean_weights.npz']=matrix
        blocks += [StateBlock(pop,MECHANISM,'V','V','effective_functional_membrane_voltage',1,0),StateBlock(pop,MECHANISM,'s','dimensionless','effective_inhibitory_activity',1,0)]
    a.state_blocks=tuple(blocks)
    a.matrix=lambda path:matrices[path] if path in matrices else original_matrix(path)
    size=sum(old.counts[n] for n in old.gap_names)
    a.boundary_runtime=dict(gap_product=np.zeros(size),gap_missing_conductance_S=np.zeros(size),identity='fixture_boundary')
    seed=redesign_seed(a,dict(states=deepcopy(state),inner_parameters=deepcopy(old.parameters)))
    model=ACVSInner(a,seed['inner_parameters'])
    return old,state,model,seed['states']


def test_analytic_equilibrium_gate_and_membrane_step():
    p=load_config(ROOT/'configurations/ac_vs_functional_v1.json')['parameters_per_effective_pool_SI']
    gE=1.8e-9;V,popen,s=equilibrium(gE,p)
    assert abs((p['gL']*(p['EL']-V)+gE*(p['EE']-V))/p['C']) < 1e-14
    times=np.linspace(0,.2,1001)
    solution=solve_ivp(lambda t,y:[(p['gL']*(p['EL']-y[0])+gE*(p['EE']-y[0]))/p['C'],p['alpha']*popen*(1-y[1])-p['beta']*y[1]],(0,.2),[p['EL'],0.],t_eval=times,rtol=1e-11,atol=1e-13)
    expected_V=V+(p['EL']-V)*np.exp(-times*(p['gL']+gE)/p['C'])
    expected_s=s*(1-np.exp(-times*(p['alpha']*popen+p['beta'])))
    np.testing.assert_allclose(solution.y,[expected_V,expected_s],rtol=1e-9,atol=1e-11)
    assert np.all((solution.y[1]>=0)&(solution.y[1]<=1))


def test_legacy_S_and_upstream_unchanged_and_main_paths_independent():
    old,original,new,state=new_fixture()
    old_rhs=old.rhs(0.,original);rhs=new.rhs(0.,state)
    for k in old_rhs:
        if k[0] not in (*AC,*RGC) or k[0] in (LEGACY_S,'small_bistratified_S_ON'):
            np.testing.assert_array_equal(rhs[k],old_rhs[k])
    ports=new.outputs(0.,state)
    for c in new.channels:
        target=c['target'];pop=c['pool']
        assert ports[target,'gI_'+c['relation']][0] > 0
        assert ports[target,'I_'+c['relation']][0] < 0
        np.testing.assert_array_equal(ports[target,'gI_AII_direct'],0.)
        modified=deepcopy(state);modified[pop,MECHANISM,'s'][:]=0.
        off=new.outputs(0.,modified)
        np.testing.assert_array_equal(off[target,'gI'],0.)
        for bc in ('midget_BC_ON','midget_BC_OFF','diffuse_BC_ON','diffuse_BC_OFF'):
            np.testing.assert_array_equal(off[bc,'effective_release'],ports[bc,'effective_release'])


def test_fraction_input_no_double_sigmoid_and_target_members_once():
    _,_,model,state=new_fixture()
    c=model.channels[0];pre=c['pre'];pool=c['pool']
    state[pre,BC_MECH,'terminal_release'][:]=.2
    state[pre,BC_MECH,'A'][:]=0.
    ports=model.outputs(0.,state)
    np.testing.assert_allclose(ports[pool,'gE'],c['G_BA_S']*.2,rtol=1e-15)
    assert model.C[pool][0]==5e-11
    np.testing.assert_allclose(ports[c['target'],'gI'],model.members[c['target']]*c['G_AR_S']*state[pool,MECHANISM,'s'],rtol=1e-15)


def test_full_parent_kernel_counts_and_crop_weight_loss_visible():
    geometry=dict(mm_per_degree=1.,input_sigma_deg=.04,input_support_deg=.1)
    pre=SimpleNamespace(group_count=2,centroid_mm=np.array([[0.,0.],[.01,0.]]),counts=np.array([1,3]))
    post=SimpleNamespace(group_count=1,centroid_mm=np.array([[0.,0.]]))
    matrix,empty=full_input_kernel(pre,post,geometry)
    assert not empty
    np.testing.assert_allclose(matrix.sum(1),1.,atol=1e-15)
    assert matrix[:,[0]].sum()<1
    np.testing.assert_allclose(matrix[0,1]/matrix[0,0],3*np.exp(-.01**2/(2*.04**2)))


def test_gate_vector_field_points_inward_and_domain_failure_is_not_clipped():
    _,_,model,state=new_fixture();pool=model.channels[0]['pool']
    for gate in (0.,1.):
        state[pool,MECHANISM,'s'][:]=gate
        value=model.rhs(0.,state)[pool,MECHANISM,'s'][0]
        assert value>=0 if gate==0 else value<=0
    state[pool,MECHANISM,'s'][:]=1.01
    with pytest.raises(ContractError,match='no clipping'):model.rhs(0.,state)
    assert state[pool,MECHANISM,'s'][0]==1.01


def test_nonzero_exact_motion_is_not_a_dynamic_defect_and_wrong_derivative_fails():
    k=('pool',MECHANISM,'V');initial={k:np.array([-.05])}
    a=SimpleNamespace(state_blocks=(StateBlock(*k,'V','effective_functional_membrane_voltage',1,0),),probability_blocks=(),auxiliary_z1_blocks=())
    monitor=DynamicDefectMonitor(a,initial)
    rhs={k:np.array([2.5])}
    monitor.observe(initial,rhs,{k:rhs[k].copy()},time_seconds=0.,side='right')
    assert monitor.report()['passed']
    scale,den,budget=frozen_budget(initial[k],'V')
    assert .0031*abs(rhs[k][0])/den[0] > 1  # The old stationarity RHS score fails on exact motion.
    broken=DynamicDefectMonitor(a,initial)
    broken.observe(initial,rhs,{k:np.zeros(1)},time_seconds=0.,side='right')
    assert not broken.report()['passed']
