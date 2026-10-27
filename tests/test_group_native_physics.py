from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from retina_sim.contracts import ContractError
from retina_sim.group_native_physics import GroupNativePhysics
from retina_sim.group_parameter_binding import source_native_equilibrium


def test_original_auxiliary_ids_keep_duplicate_group_drives_and_distinct_memory():
    A=sparse.csr_matrix([[.5,.5,0],[0,.2,.8]])
    B=sparse.csr_matrix([[1.,0],[.7,.3],[0,1.]])
    cones=SimpleNamespace(group_count=3,group_ids=np.array(['c0','c1','c2']),
        recording_auxiliary_group_rows=np.array([0,0,2]),recording_auxiliary_fine_rows=np.array([1,4,9]))
    hcs=SimpleNamespace(group_count=2,group_ids=np.array(['h0','h1']))
    assembly=SimpleNamespace(populations={'cones':cones,'HC':hcs},artifact_sha256='fixture',
        matrix=lambda path:A if path=='routes/cones_to_HC/mean_weights.npz' else B)
    model=GroupNativePhysics(assembly)
    assert model.aux_indices==(0,0,2) and len(set(model.aux_cell_ids))==3
    eq=source_native_equilibrium(A,B,np.full(3,14.9))
    state={('cones','reduced_native',n):eq[n] for n in ('X','C','Vis','g_i','Vbar')}
    state.update({('cones','analytic_RE',n):eq[n] for n in ('R','E')})
    state.update({('HC','reduced_native',n):v for n,v in [('w',A@eq['z1']),('z2',eq['z2']),('Vh',eq['Vh'])]})
    state['cones','selected_original_z1_auxiliary','z1_aux']=eq['z1'][[0,0,2]]+np.array([1.,0.,0.])
    seed=SimpleNamespace(states=state,time_seconds=0.)
    y,anchor=model.construction_point(seed,np.full(3,14.9),.001,0)
    derivative=model.blocks(model.rhs(y,0.,anchor))['z1_aux']
    assert derivative[0]-derivative[1]==pytest.approx(-1000/model.p['tau_1'])
    with pytest.raises(ContractError,match='rejected midpoint'):
        model.step(None)
