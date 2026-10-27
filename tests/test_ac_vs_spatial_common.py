"""Small common-state projection and boundary algebra tests; no model advance."""
from copy import deepcopy
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from retina_sim.contracts import ContractError,canonical_hash
from tools.validate_ac_vs_spatial_common import by_ids,nested,annulus_product,project_seed,spatial_stimulus
import tools.validate_ac_vs_spatial_common as stage


def test_ids_are_ordered_global_identity_and_reject_missing_or_duplicates():
    np.testing.assert_array_equal(by_ids(['c','a','b'],['b','c'],'test'),[2,0])
    with pytest.raises(ContractError):by_ids(['a','a'],['a'],'test')
    with pytest.raises(ContractError):by_ids(['a'],['missing'],'test')


def test_nested_annulus_products_preserve_weights_and_gap_conservation():
    weights=csr_matrix([[.2,.3,.5],[.6,.1,.3],[.4,.4,.2]])
    rows=np.array([2,0]);x=np.array([2.,3.,7.]);outside=np.array([.7,.8,.9])
    local=weights[rows][:,rows]
    nested(local,weights,rows,rows,'synthetic native')
    small_boundary=outside[rows]+annulus_product(weights,rows,rows,x)
    np.testing.assert_allclose(local@x[rows]+small_boundary,(weights@x+outside)[rows],rtol=0,atol=1e-15)
    with pytest.raises(ContractError):nested(local*1.01,weights,rows,rows,'renormalized')
    L=csr_matrix([[3.,-1.,-2.],[-1.,4.,-3.],[-2.,-3.,5.]])
    local=L[rows][:,rows];offset=annulus_product(L,rows,rows,x)
    missing=-annulus_product(L,rows,rows,np.ones(3))
    np.testing.assert_allclose(local@x[rows]+offset,(L@x)[rows])
    reservoir=missing@x[rows]+sum(offset)
    np.testing.assert_allclose(-sum(local@x[rows]+offset)+reservoir,0.,atol=1e-14)


def projection_fixture():
    keys=[('cones','reduced_native','X'),('HC','reduced_native','Vh'),('RGC','membrane','V'),
        ('cones','selected_original_z1_auxiliary','z1_aux')]
    maps=dict(populations={'cones':np.array([2,0]),'HC':np.array([1]),'RGC':np.array([0,1])},contacts={},auxiliary=np.array([1]),identity='map')
    states={k:np.arange(n,dtype=float) for k,n in zip(keys,[3,2,2,2])}
    rng=np.random.PCG64(51).state
    source=dict(states=states,inner_parameters=dict(identity='parameters',provenance={},populations={
        'cones':dict(V0=np.array([1.,2.,3.]),kernel=np.array([9.,8.,7.])),
        'RGC':dict(reference_input=np.array([.1,.2]),spike_history_kernel=np.array([1.,2.]))},contacts={},route_gains={'r':1.}),
        spike_states={'RGC':dict(history=np.array([[1.,0.],[0.,1.],[1.,1.]]),conditional_rate_Hz=np.array([2.,3.]),rng_state=rng,pending_events=[],last_bin_counts=None)},
        observation_prehistory=dict(history_times=np.array([1.,2.]),history_values=np.array([[10.,11.],[12.,13.]]),prehistory=np.array([8.,9.])),
        native_inputs={748:np.array([4.,5.,6.])},rod_inputs={748:np.array([7.,8.])},
        input_reference=dict(native_background_td=np.array([4.,5.,6.])),patch_initialization=dict(source_sha256={'source':'hash'}))
    maps['populations']['rods']=np.array([1])
    small=NS(state_blocks=[NS(population=k[0],mechanism=k[1],name=k[2],count=n) for k,n in zip(keys,[2,1,2,1])],probability_blocks=[],auxiliary_z1_blocks=[],
        parent_population_rows={},parent_contact_rows={},parent_auxiliary_rows={'cones':np.array([1])},artifact_sha256='small')
    large=NS(populations={'cones':NS(group_count=3),'RGC':NS(group_count=2)})
    return source,small,large,maps


def test_complete_projection_preserves_kernels_rng_history_and_source(monkeypatch):
    source,small,large,maps=projection_fixture();original=canonical_hash(source)
    monkeypatch.setattr(np.random,'Generator',lambda *a,**k:pytest.fail('Projection drew random numbers'))
    result=project_seed(source,small,large,maps)
    assert canonical_hash(source)==original
    np.testing.assert_array_equal(result['states']['cones','reduced_native','X'],[2.,0.])
    np.testing.assert_array_equal(result['inner_parameters']['populations']['cones']['V0'],[3.,1.])
    np.testing.assert_array_equal(result['inner_parameters']['populations']['cones']['kernel'],[9.,8.,7.])
    assert canonical_hash(result['spike_states'])==canonical_hash(source['spike_states'])
    np.testing.assert_array_equal(result['observation_prehistory']['history_values'],[[11.],[13.]])
    np.testing.assert_array_equal(result['native_inputs'][748],[6.,4.])
    np.testing.assert_array_equal(result['rod_inputs'][748],[8.])


def test_projection_rejects_unregistered_array_and_uncommitted_sampler():
    source,small,large,maps=projection_fixture();source['inner_parameters']['populations']['cones']['unknown']=np.ones(3)
    with pytest.raises(ContractError,match='Unregistered parameter'):project_seed(source,small,large,maps)
    source,small,large,maps=projection_fixture();source['spike_states']['RGC']['pending_events']=[1]
    with pytest.raises(ContractError,match='Uncommitted'):project_seed(source,small,large,maps)


def test_stimulus_resolves_global_ids_under_permutation(monkeypatch):
    ids=['cone'+str(i) for i in range(20)];frozen=ids[2:17];ordered=ids[::-1]
    monkeypatch.setattr(stage,'read',lambda path:{'mask_group_ids':frozen})
    seed=dict(input_reference=dict(native_background_td=np.arange(20,dtype=float)+1,rod_background_rstar_s=4.))
    a=NS(populations={'cones':NS(group_ids=ordered,group_count=20),'rods':NS(group_count=2)})
    result,meta=spatial_stimulus(seed,a,{'input':{'global_cone_ids_from':'unused'}})
    changed=np.flatnonzero(result['native_inputs'][750]!=result['native_inputs'][749])
    assert set(np.array(ordered)[changed])==set(frozen)
    np.testing.assert_array_equal(result['native_inputs'][749],result['native_inputs'][759])
    assert all(np.array_equal(v,[4.,4.]) for v in result['rod_inputs'].values())
