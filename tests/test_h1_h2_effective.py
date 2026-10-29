"""Small mathematical fixtures; actual typed local network is checked by the CLI."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from scipy.integrate import solve_ivp

from retina_sim.contracts import ContractError
from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
from retina_sim.h1_h2_effective import (global_operators,joint_equilibrium,H1H2Native,
    H1H2GroupRunner,MODEL,MECHANISM,LEGACY,runner)
from retina_sim.vh2005 import SOURCE_VALUES


def fixture():
    A = sparse.csr_matrix([[.5,.5,0.],[.25,.25,.5]])
    B = sparse.csr_matrix([[.8,.2],[.6,.4],[0.,1.]])
    types = dict(cones=np.array(['L','M','S']),HC=np.array(['H1','H2']))
    full = SimpleNamespace(matrix=lambda path:A if 'cones_to_HC' in path else B)
    A,B,h1,h2 = global_operators(full,types)
    pars = dict(tau_seconds=.020,input_scale=151.1,feedback_scale=1.)
    eq = joint_equilibrium(A,B,h1,h2,np.array([10.,20.,12.]),pars)
    pop = lambda ids:SimpleNamespace(group_count=len(ids),group_ids=np.array(ids))
    cones = pop(['cL','cM','cS']);cones.recording_auxiliary_group_rows=np.array([2]);cones.recording_auxiliary_fine_rows=np.array([2])
    a = SimpleNamespace(populations=dict(cones=cones,HC=pop(['h1','h2'])),
        matrix=lambda path:A if 'cones_to_HC' in path else B,artifact_sha256='manufactured',
        h1_rows=h1,h2_rows=h2,h2_parameters=pars,layout=dict(model=MODEL),connection_identity=dict(fixture=True))
    native = H1H2Native(a)
    native.boundary = dict(A_release=np.zeros(2),A_Vbar=np.zeros(2),B_h=np.zeros(3))
    y = np.concatenate([eq['cone'][n] for n in ('X','C','Vis','g_i','Vbar')]+
        [eq['w'],eq['z2'],eq['h'],SOURCE_VALUES['g_t']*eq['release'][[2]]])
    light = eq['cone']['R']
    anchor = AnalyticREAnchor(0.,light,light,light,.1,0)
    return native,y,anchor,eq


def test_four_operators_keep_actual_types_and_global_normalization():
    n,_,_,_=fixture()
    np.testing.assert_array_equal(n.B.toarray(),[[1,0],[1,0],[0,1]])
    assert n.A[n.h1_rows][:,2].nnz == 0
    assert n.A[n.h2_rows][:,2].nnz and n.A[n.h2_rows][:,:2].nnz
    assert n.sizes[5:8] == (1,1,2)  # No unused H2 w/z2 states.
    with pytest.raises(ContractError):
        global_operators(SimpleNamespace(),dict(cones=np.array(['UNKNOWN']),HC=np.array(['H1','H2'])))


def test_joint_fixed_point_and_h2_independent_derivative():
    n,y,anchor,eq=fixture()
    assert eq['residual'] <= 1e-8
    np.testing.assert_allclose(n.rhs(y,0.,anchor),0.,atol=1e-8,rtol=0.)
    perturbed=y.copy();perturbed[n.slices['Vis']][2] -= .1
    f=n.rhs(perturbed,0.,anchor);base=n.rhs(y,0.,anchor)
    assert f[n.slices['Vh']][1] < base[n.slices['Vh']][1]
    np.testing.assert_array_equal(f[n.slices['w']],base[n.slices['w']])
    with pytest.raises(ContractError,match='cannot migrate'):n.migrate(None)
    with pytest.raises(TypeError):n.parameters['tau_seconds']=.1
    with pytest.raises(ValueError):n.boundary['B_h'][0]=1.
    with pytest.raises(ContractError,match='cannot refresh'):n.boundary=dict()


def test_selective_s_and_lm_light_have_same_sign_h2_without_h1_s_drive():
    n,y,old,eq=fixture();delta={}
    for label,rows in [('S',[2]),('LM',[0,1])]:
        light=old.light.copy();light[rows]*=1.1
        anchor=AnalyticREAnchor(0.,old.R,old.E,light,.1,0)
        solution=solve_ivp(lambda t,v:n.rhs(v,t,anchor),(0.,.05),y,method='DOP853',
            rtol=1e-6,atol=1e-9,max_step=.0001)
        assert solution.success
        delta[label]=solution.y[n.slices['Vh'],-1]-y[n.slices['Vh']]
    assert delta['S'][1] < 0 and delta['LM'][1] < 0 and delta['LM'][0] < 0
    assert abs(delta['S'][0]) < 1e-8


@pytest.mark.parametrize('mode',['candidate','radau','gpu_dop853'])
def test_unsupported_backends_explicitly_reject(mode):
    with pytest.raises(ContractError,match='cannot bypass H2'):
        H1H2GroupRunner(None,{}, {},mode=mode)


def test_old_or_absent_h2_seed_rejected():
    with pytest.raises(ContractError,match='independently equilibrated'):
        H1H2GroupRunner(SimpleNamespace(artifact_sha256='new'),{}, {})
    with pytest.raises(ContractError,match='legacy identity'):
        runner(LEGACY,SimpleNamespace(),dict(h1_h2_model={}),{})
