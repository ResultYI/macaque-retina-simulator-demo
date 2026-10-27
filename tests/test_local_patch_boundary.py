"""Pure toy algebra for v2 forcing; no real retina advance or unguarded RHS."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from retina_sim.contracts import ContractError
from retina_sim.group_native_physics import GroupNativePhysics
from retina_sim.local_patch_boundary import AffineProduct, BoundedNativePhysics, BoundedRecording
from retina_sim.local_patch_runner import _indices, make_protocol


def toy_assembly(A, B, identity):
    nc, nh = np.shape(B)
    pop = lambda n:SimpleNamespace(group_count=n, group_ids=np.arange(n),
        recording_auxiliary_group_rows=np.array([], dtype=np.int64),
        recording_auxiliary_fine_rows=np.array([], dtype=np.int64))
    operators = {'routes/cones_to_HC/mean_weights.npz':sparse.csr_matrix(A),
        'routes/HC_to_cones/mean_weights.npz':sparse.csr_matrix(B)}
    return SimpleNamespace(populations=dict(cones=pop(nc), HC=pop(nh)),
        artifact_sha256=identity, matrix=operators.__getitem__)


def point(native, values):
    from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
    V, Vbar = map(np.asarray, values)
    nc, nh = native.nc, native.nh
    y = np.concatenate([np.ones(nc),np.ones(nc),V,np.ones(nc),Vbar,
        np.ones(nh),np.ones(nh),np.ones(nh)])
    anchor = AnalyticREAnchor(0.,np.ones(nc),np.ones(nc),np.ones(nc),.01,0)
    return y, anchor


def test_native_exterior_enters_both_original_pools_before_nonlinearity():
    full = GroupNativePhysics(toy_assembly([[.25,.75]],[[1.],[1.]],'full'))
    fy, fa = point(full, ([12.,10.],[11.,8.]))
    original = full.evaluate(fy,0.,fa)
    assembly = toy_assembly([[.25]],[[1.]],'local')
    assembly.boundary_runtime = dict(identity='held-test',A_Vbar=np.array([.75*8.]),
        A_release=.75*original.outputs['release'][1:],B_Vh=np.zeros(1))
    local = BoundedNativePhysics(assembly)
    ly, la = point(local, ([12.],[11.]))
    current = local.evaluate(ly,0.,la)
    for name, section in local.slices.items():
        expected = original.rhs[full.slices[name]][:len(current.rhs[section])]
        np.testing.assert_allclose(current.rhs[section],expected,rtol=1e-14,atol=1e-12)
    np.testing.assert_array_equal(current.outputs['pooled_aI'],original.outputs['pooled_aI'])
    assert np.asarray(local.A.sum(axis=1))[0,0] == .25


def test_affine_operator_preserves_linear_matrix_and_owns_offset():
    matrix=sparse.csr_matrix([[2.,-1.],[-1.,4.]])
    offset=np.array([.3,-.2]); before=matrix.copy()
    operator=AffineProduct(matrix,offset);offset[:]=9.
    np.testing.assert_array_equal(operator@np.array([1.,2.]),np.array([.3,6.8]))
    np.testing.assert_array_equal(operator.tocsr().toarray(),before.toarray())
    assert not operator.offset.flags.writeable
    with pytest.raises(ContractError):AffineProduct(matrix,np.array([0.]))


def test_empty_original_selection_requires_explicit_allowance():
    empty=np.empty(0,np.int64)
    with pytest.raises(ContractError):_indices(empty,4,'test')
    np.testing.assert_array_equal(_indices(empty,4,'test',allow_empty=True),empty)


def test_v2_protocol_covers_delay_without_changing_original_tolerances():
    seed=dict(time_seconds=4.993,native_inputs={748:None,749:None,750:None})
    with pytest.raises(ContractError):make_protocol(seed,duration_seconds=.0031)
    protocol=make_protocol(seed,duration_seconds=.0031,bounded_v2=True)
    assert protocol['stop_seconds']==4.9961
    assert protocol['native_observation_delay_seconds']==.00282
    assert protocol['reference']==dict(method='joint_DOP853',rtol=1e-6,atol=1e-9,max_step=.0001)
    with pytest.raises(ContractError):make_protocol(seed,duration_seconds=.0041,bounded_v2=True)


def test_empty_recording_fields_are_declared_not_fabricated(monkeypatch):
    from retina_sim.group_time_recording import GroupRecording
    def configure(self,assembly,identity):
        self.fields=(SimpleNamespace(key='actual',rows=np.array([0])),SimpleNamespace(key='empty',rows=np.empty(0,int)))
        self.rows={field.key:field.rows for field in self.fields};self.field_by_key={f.key:f for f in self.fields}
        self.components=[dict(id='a',ports=dict(x=dict(recorded_count=1))),dict(id='b',ports=dict(y=dict(recorded_count=0)))]
        self.layout_identity='original'
    monkeypatch.setattr(GroupRecording,'_configure',configure)
    recorder=BoundedRecording.__new__(BoundedRecording)
    recorder._configure(None,None)
    assert [field.key for field in recorder.fields]==['actual']
    assert [component['id'] for component in recorder.components]==['a']
    assert recorder.layout_identity!='original'
