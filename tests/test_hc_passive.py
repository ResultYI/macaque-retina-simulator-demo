from pathlib import Path
import numpy as np
import pytest
from test_hc_gap import native_component,q,PROJECT
from examples.full_target_network import population
from retina_sim.contracts import ContractError
from retina_sim.hc_gap import EffectiveHCGapOperator
from retina_sim.hc_passive import with_background_referenced_passive_hc,reference_context

ROOT=Path(__file__).resolve().parents[1]
def fixture(rate=.001):
    component,light=native_component();model=component.native_model
    xy=np.array([[.7,0],[.71,0],[.72,0],[.73,0]])
    full=population('h','HC',('H1',)*4,xy,('center',)*4,model.parameters.profile.profile_id,model.parameters.profile.species)
    ids=full.cells.cell_id;assert ids[:3]==model.hcs.cells.cell_id
    gap=EffectiveHCGapOperator(ids,[(ids[0],ids[3])],q([rate],'ms^-1','effective_HC_gap_rate'),
        assumption_profile_name='test_only_uncalibrated_k',project_evidence_tag=PROJECT,
        equal_effective_capacity=True,common_state_scale_reference=True,enabled=True)
    geometry='complete_finite_fixture_graph_v1'
    kwargs=dict(hc_types=dict(zip(ids,full.cells.subtype)),query_rows=[{'hc_id':ids[3],
        'candidate_count':0,'conditional_count':0,'query_complete_sampled':True,'direct_input_status':'COMPLETE_EMPTY_DIRECT_INPUT'}],
        prepared_blocks=component.initial(),preparation={'status':'CONVERGED','gap_enabled':False,
            'reference_context_hash':reference_context(model,light,geometry),
            'method':'numeric fixture source equilibrium; no real background experiment'},
        background=light,geometry_graph_hash=geometry,root=ROOT,hc_population=full)
    return component,gap,kwargs

def test_zero_k_returns_exact_original_component_without_new_states_or_reference():
    c,g,k=fixture(0);k.update(prepared_blocks=None,preparation=None,background=None,hc_population=None)
    assert with_background_referenced_passive_hc(c,g,**k) is c

def test_uniform_background_reference_has_no_artificial_zero_sink_or_fake_contact():
    c,g,k=fixture();wrapped=with_background_referenced_passive_hc(c,g,**k);m=wrapped.native_model
    ref=next(iter(m.references.values()));assert ref['vRef']>0 and ref['hRef']>0 and ref['aRef']!=1
    assert m.A[-1].nnz==0 and m.B[:,-1].nnz==0
    blocks=[x.values.copy() for x in wrapped.initial()];rhs=wrapped.derivative(0,blocks,{'light':k['background']})
    assert rhs[8].values[-1]==0 and rhs[9].values[-1]==0
    assert blocks[8][-1]==blocks[9][-1]==ref['hRef']
    assert m.metadata['requires_new_positive_gap_background_preparation']
    assert not m.metadata['initialization_is_coupled_equilibrium']
    assert wrapped.observe('Vh',0,blocks,{}).evidence_tag==wrapped.outputs['Vh'].evidence_tag

def test_direct_dynamic_adaptation_unchanged_and_gap_only_flow_conservative():
    c,g,k=fixture();w=with_background_referenced_passive_hc(c,g,**k);blocks=[x.values.copy() for x in w.initial()]
    blocks[6][0]+=1;blocks[9][0]+=2
    base_blocks=[a[:3] if i in (8,9) else a for i,a in enumerate(blocks)]
    old=c.derivative(0,base_blocks,{'light':k['background']});new=w.derivative(0,blocks,{'light':k['background']})
    for a,b in zip(old[:8],new[:8]):np.testing.assert_array_equal(a.values,b.values)
    np.testing.assert_array_equal(new[8].values[:3],old[8].values)
    delta=new[9].values[:3]-old[9].values
    np.testing.assert_allclose(delta,[-2,0,0],atol=1e-12);assert new[9].values[-1]==pytest.approx(2)
    assert float(np.sum(np.r_[delta,new[9].values[-1]]))==pytest.approx(0,abs=1e-12)

def test_incomplete_query_bad_reference_context_and_missing_anchor_refuse():
    c,g,k=fixture();k['query_rows'][0]['query_complete_sampled']=False
    with pytest.raises(ContractError,match='Incomplete'):with_background_referenced_passive_hc(c,g,**k)
    c,g,k=fixture();k['preparation']['reference_context_hash']='other_geometry'
    with pytest.raises(ContractError,match='Matching'):with_background_referenced_passive_hc(c,g,**k)
    c,g,k=fixture();k['requested_hc_ids']=(g.cell_ids[-1],)
    isolated=EffectiveHCGapOperator(g.cell_ids,[],q([],'ms^-1','effective_HC_gap_rate'),
        assumption_profile_name='test_no_reference',project_evidence_tag=PROJECT,equal_effective_capacity=True,
        common_state_scale_reference=True,enabled=True)
    with pytest.raises(ContractError,match='Gap-off orphan'):with_background_referenced_passive_hc(c,isolated,**k)

def test_nonuniform_background_and_heterotypic_orphan_are_outside_v1():
    c,g,k=fixture();k['background']=q([100,110,100],'td','native_retinal_illuminance')
    k['preparation']['reference_context_hash']=reference_context(c.native_model,k['background'],k['geometry_graph_hash'])
    with pytest.raises(ContractError,match='uniform'):with_background_referenced_passive_hc(c,g,**k)
    c,g,k=fixture();k['hc_types'][g.cell_ids[-1]]='H2'
    with pytest.raises(ContractError,match='homotypic'):with_background_referenced_passive_hc(c,g,**k)

def test_positive_orphan_only_requested_component_cannot_borrow_external_reference():
    c,g,k=fixture();model=c.native_model
    xy=np.array([[.7,0],[.71,0],[.72,0],[.73,0],[.74,0]])
    full=population('h','HC',('H1',)*5,xy,('center',)*5,model.parameters.profile.profile_id,model.parameters.profile.species)
    ids=full.cells.cell_id
    isolated=EffectiveHCGapOperator(ids,[(ids[3],ids[4])],q([.001],'ms^-1','effective_HC_gap_rate'),
        assumption_profile_name='test_missing_reference',project_evidence_tag=PROJECT,
        equal_effective_capacity=True,common_state_scale_reference=True,enabled=True)
    k.update(hc_population=full,hc_types=dict(zip(ids,full.cells.subtype)),requested_hc_ids=(ids[3],))
    with pytest.raises(ContractError,match='no direct-input same-type reference'):
        with_background_referenced_passive_hc(c,isolated,**k)
    k['requested_hc_ids']=model.hcs.cells.cell_id
    assert with_background_referenced_passive_hc(c,isolated,**k) is c
