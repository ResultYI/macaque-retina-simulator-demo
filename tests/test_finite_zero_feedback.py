from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from retina_sim.contracts import ContractError,RunMode,Quantity
from retina_sim.core_sources import load_core_registry
from retina_sim.vh2005 import NativeConeH1,PROFILE
from retina_sim.finite_feedback_native import FiniteContactConeH1,certify_zero_contacts,graph_hash
from retina_sim.hc_geometry import read_hc_geometry,hc_contact_runtime_view
from examples.full_network_fixture import population

ROOT=Path(__file__).resolve().parents[1]

def fixture(empty=True,bounds=(.65,.90,-.05,.05)):
    registry=load_core_registry(ROOT);p=registry.load_parameters(PROFILE,RunMode.NUMERIC_FIXTURE)
    xy=np.array([[.75,0],[.8,0]])
    c=population('c','cone',('L','M'),xy,('center',)*2,PROFILE,p.profile.species)
    h=population('h','HC',('H1',),xy[:1],('center',),PROFILE,p.profile.species)
    A=sparse.csr_matrix([[1.,0.]]) if empty else sparse.csr_matrix([[.5,.5]])
    B=sparse.csr_matrix([[1.],[0.]]) if empty else sparse.csr_matrix([[1.],[1.]])
    cfg=read_hc_geometry(ROOT,'hc_joint_synthetic_v1',arbor_profile_id='W2000_equal_area_bounded_edges_047_104')
    contacts={'paired_cones':[{'input_owner':True},{'input_owner':False}]}
    cert=certify_zero_contacts(c,h,A,B,contacts,cfg,bounds)
    cert['runtime_graph_hash']=graph_hash(c.cells.cell_id,h.cells.cell_id,A,B)
    args=(p,c,h,A,B,Quantity([100.,100.],'td','native_retinal_illuminance','synthetic_fixture'))
    kw={'pooling_policy':'pool_Vbar_before_aI','geometry_frame':'OPL','observation_interval_seconds':.0001}
    return args,kw,cert

def test_complete_true_zero_retains_cone_and_solves_same_no_feedback_rhs():
    args,kw,cert=fixture();model=FiniteContactConeH1(*args,**kw,zero_contact_certificate=cert)
    assert model.cones.cells.cell_id==args[1].cells.cell_id and len(model.initial_conditions()[0].values)==2
    blocks=tuple(q.values for q in model.initial_conditions())
    residual=model.derivative(0,blocks,{'light':args[-1]})
    assert max(float(np.max(np.abs(q.values))) for q in residual)<1e-7
    from scipy.special import expit
    expected=expit((blocks[4][1]-model.p['V_k'])/model.p['V_n'])/(blocks[6][1]/model.p['V_I'])**model.p['mu']
    assert model.observe('release',0,blocks,{}).values[1]==pytest.approx(expected,rel=1e-13)
    assert (model.B@blocks[-1])[1]==0
    assert model.background_equilibrium_audit['maximum_residual_native']<1e-8
    # Strict original component still rejects exactly this topology.
    with pytest.raises(ContractError,match='unit-DC'):NativeConeH1(*args,**kw)
    with pytest.raises(ContractError,match='UNCOVERED_CONES'):hc_contact_runtime_view(args[2],args[3],args[4])

def test_nonempty_source_math_and_background_are_unchanged():
    args,kw,cert=fixture(False);old=NativeConeH1(*args,**kw);new=FiniteContactConeH1(*args,**kw,zero_contact_certificate=cert)
    for a,b in zip(old.initial_conditions(),new.initial_conditions()):np.testing.assert_array_equal(a.values,b.values)
    blocks=[q.values.copy() for q in old.initial_conditions()];blocks[7]*=1.01
    for a,b in zip(old.derivative(.001,blocks,{'light':args[-1]}),new.derivative(.001,blocks,{'light':args[-1]})):
        np.testing.assert_array_equal(a.values,b.values)

def test_domain_hole_is_rejected_and_certificate_binds_exact_graph():
    with pytest.raises(ContractError,match='SUPPORT_INCOMPLETE'):fixture(bounds=(.65,.805,-.05,.05))
    args,kw,cert=fixture();cert=dict(cert,runtime_graph_hash='wrong')
    with pytest.raises(ContractError,match='certificate'):FiniteContactConeH1(*args,**kw,zero_contact_certificate=cert)

def test_nonempty_nonunit_rows_and_nonreciprocal_graph_are_rejected():
    args,kw,cert=fixture();broken=list(args);broken[4]=sparse.csr_matrix([[.5],[0.]])
    with pytest.raises(ContractError,match='unit-DC'):FiniteContactConeH1(*broken,**kw,zero_contact_certificate=cert)
    broken=list(args);broken[3]=sparse.csr_matrix([[.5,.5]])
    with pytest.raises(ContractError,match='partner graph'):FiniteContactConeH1(*broken,**kw,zero_contact_certificate=cert)
