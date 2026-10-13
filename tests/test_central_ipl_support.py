"""Physical support coverage and exact empty contact sums; no new physiology."""
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from retina_sim.geometry import RetinalDomain
from retina_sim.spatial_profiles import aii_ipl_support_domain,finite_support_coverage,bounded_matrix
from retina_sim.contracts import ContractError,RunMode
from retina_sim.assembly import ComponentSpec,ModelSpec,ExternalInput,RetinaBuilder
from retina_sim.rod_pathway import chemical_component
from retina_sim.core_sources import load_core_registry
from retina_sim.spatial import InputChannel
from examples.full_network_fixture import population,fixture_profile,q

ROOT=Path(__file__).resolve().parents[1]

def test_central_ipl_owner_support_is_not_the_rf_rectangle():
    d=RetinalDomain((.309,.391,-.041,.041),(.349,.351,-.001,.001),lambda xy:np.ones(len(xy),bool),'fixture','RF')
    resolved,audit=aii_ipl_support_domain(d,(.746,.751,-.002,.002),support_mm=.04)
    assert audit['domain_extended'] and audit['density_scale']==1 and not audit['density_value_changed']
    assert not finite_support_coverage(np.array([[.748,0]]),.04,[d.bounds_mm])[0]
    assert finite_support_coverage(np.array([[.748,0]]),.04,audit['generation_rectangles_mm'])[0]
    assert resolved.mask(np.array([[.748,0],[.35,0],[.53,0]])).tolist()==[True,True,True]
    assert 'intermediate tissue consumes unchanged density/budget' in audit['mask_policy']
    # Zero candidates in an unavailable source view are not an animal zero.
    old,_=bounded_matrix(np.array([[.35,0]]),np.array([[.748,0]]),3,support_mm=.04)
    new,_=bounded_matrix(np.array([[.748,0]]),np.array([[.748,0]]),3,support_mm=.04)
    assert old.nnz==0 and new.nnz==1

def test_covered_original_domain_is_preserved():
    d=RetinalDomain((4.55,4.65,-.05,.05),(4.59,4.61,-.01,.01),lambda xy:np.ones(len(xy),bool),'fixture','physical')
    resolved,audit=aii_ipl_support_domain(d,(4.591,4.609,-.009,.009),support_mm=.04)
    assert resolved is d and not audit['domain_extended']

def empty_fixture():
    registry=load_core_registry(ROOT)
    fixture_profile(registry,'target_existing_profile',2,{})
    fixture_profile(registry,'empty_contact_sum',0,{})
    pop=population('target','BC',('BC','BC'),np.array([[.7,0],[.71,0]]),('center','center'),
        'target_existing_profile','Macaque_software_fixture')
    spec=ComponentSpec('empty','DoubleExponentialChemicalSynapse',{'cells':'target'},'empty_contact_sum',
        {'receptor':'inhibitory','empty_contact_operator':True,'empty_support_verified':True,
         'target_population_profile_id':'target_existing_profile'})
    return registry,pop,spec

def test_empty_contact_operator_has_zero_contribution_without_fake_cells():
    registry,pop,spec=empty_fixture();c=chemical_component(spec,{'target':pop},registry,RunMode.NUMERIC_FIXTURE)
    assert c.state_specs==() and c.initial()==() and c.derivative(0,(),{})==()
    assert c.inputs=={}
    assert c.populations==(pop,) and c.outputs['conductance'].cell_ids==pop.cells.cell_id
    np.testing.assert_array_equal(c.observe('conductance',0,(),{}).values,[0,0])
    compiled=RetinaBuilder(registry).build(ModelSpec('empty_sum_fixture',(pop,),(spec,),
        required_mechanisms=('chemical_inhibitory_conductance',)),RunMode.NUMERIC_FIXTURE)
    state,result=compiled.advance(compiled.initialize(),{},.0001)
    assert state.ordered_state_arrays==()
    np.testing.assert_array_equal(result.outputs_with_units['empty.conductance'].values,np.zeros((2,2)))

def test_unverified_empty_support_is_rejected():
    registry,pop,spec=empty_fixture()
    with pytest.raises(ContractError,match='verified complete finite support'):
        chemical_component(replace(spec,options=dict(spec.options,empty_support_verified=False)),{'target':pop},registry,RunMode.NUMERIC_FIXTURE)
