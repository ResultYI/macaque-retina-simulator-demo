"""Finite morphology, bounded anchors and population diagnostics regressions."""
from pathlib import Path
import numpy as np
import pytest
from retina_sim.contracts import ContractError,Quantity
from retina_sim.hc_geometry import read_hc_geometry
from retina_sim.hc_arbor import build_hc_arbor_contacts
from retina_sim.ipl_geometry import read_ipl_geometry,RESEARCH,TYPED

ROOT=Path(__file__).resolve().parents[1]

def q(a,meaning,frame):return Quantity(np.asarray(a,float),'mm',meaning,'synthetic_fixture',frame)

def graph(p,h,types=('L',),htypes=None,**kw):
    cfg=read_hc_geometry(ROOT,'hc_joint_synthetic_v1',arbor_profile_id='W2000_equal_area_bounded_edges_047_104')
    options={'anchor_applicability_band_mm':cfg['anchor_applicability_band_mm'],
        'source_anchor_band_mm':cfg['source_anchor_band_mm'],'contact_policy':cfg['contact_policy'],
        'source_radial_support_mm':cfg['contact_source_radial_support_mm'],'final_contact_distance_mm':.04,
        'cone_pedicle_generation_bounds_mm':(.44,1.07,-.04,.04),'owner_hc_mask':np.ones(len(h),bool)}
    footprints=kw.pop('footprints',cfg['footprints']);options.update(kw)
    return build_hc_arbor_contacts(tuple('c'+str(i) for i in range(len(p))),q(p,'functional_position','functional_visual_field'),
        q(p,'pedicle_position','OPL'),tuple('h'+str(i) for i in range(len(h))),htypes or ('H1',)*len(h),
        q(h,'anatomical_position','anatomical_retina'),q(h,'arbor_anchor_position','OPL'),np.ones(len(p),bool),
        footprints,cone_types=types,**options)

def test_anchor_band_does_not_clip_whole_disk_and_old_strict_is_preserved():
    A,B,a=graph([[.465,0]],[[.47,0]])
    assert A.nnz==B.nnz==1
    assert a['actual_HC_footprint_extent']['H1']['owner']['radial_mm'][0]<.47
    assert a['extrapolation_label_by_HC']==['bounded_edge_extrapolation']
    with pytest.raises(ContractError,match='CONTACT_DOMAIN_UNSUPPORTED'):
        graph([[.465,0]],[[.47,0]],anchor_applicability_band_mm=None,contact_policy=None)
    with pytest.raises(ContractError,match='CONTACT_DOMAIN_UNSUPPORTED'):
        graph([[.465,0]],[[.469,0]])

def test_required_query_domain_is_independent_and_truncation_rejects():
    with pytest.raises(ContractError,match='CONTACT_REQUIRED_QUERY_TRUNCATED'):
        graph([[.47,0]],[[.47,0]],cone_pedicle_generation_bounds_mm=(.47,.5,-.02,.02))
    with pytest.raises(ContractError,match='never per-cone quotas'):
        graph([[.75,0]],[[.75,0]],require_owner_quota=True)

def test_all_legal_partners_preserve_denominators_and_weight_algorithm():
    h=[[.748+i*.001,0] for i in range(5)]+[[.75,0]]
    A,B,a=graph([[.75,0],[.751,0]],h,types=('L','S'),htypes=('H1',)*5+('H2',))
    assert A.nnz==7 and B.nnz==7
    stat=a['population_diagnostics']['H1']
    assert stat['D_all_owner']['mean']==2.5 and stat['D_eligible_owner']['mean']==5
    assert stat['C_complete_owner_HC']['mean']==1 and stat['source_population_pass_threshold'] is None
    impact=a['graph_weight_impact']
    assert impact['added_pairs']==1 and impact['removed_pairs']==0
    assert impact['gain_changed'] is False and impact['rescale_to_divergence_target'] is False
    assert len(set((e['cone_id'],e['hc_id']) for e in a['edges']))==7
    np.testing.assert_allclose(np.asarray(A.sum(1)).ravel(),1)
    np.testing.assert_allclose(np.asarray(B.sum(1)).ravel(),1)

def test_narrow_phase_ellipse_and_final_40um_are_both_enforced():
    ellipse={'kind':'ellipse','axes_mm':[.06,.004],'unit':'mm','coordinate_frame':'OPL',
        'source_profile_id':'fixture','evidence_class':'project_approximation','approximation_id':'fixture'}
    A,B,a=graph([[.75,0],[.79,0],[.791,0],[.75,.005]],[[.75,0]],types=('L',)*4,
        footprints={'H1':ellipse,'H2':ellipse},cone_pedicle_generation_bounds_mm=(.68,.82,-.01,.01))
    assert [e['cone_id'] for e in a['edges']]==['c0','c1']
    assert A.nnz==B.nnz==2

def test_only_named_s_on_relay_relaxes_enforcement_not_selection_or_other_routes():
    old=read_ipl_geometry(ROOT,TYPED);new=read_ipl_geometry(ROOT,RESEARCH)
    assert old['RGC_degree']==new['RGC_degree']
    assert 'S_ON_effective_contract' not in old
    relay=new['S_ON_effective_contract']
    assert relay['legacy_selection_budget']==8 and relay['minimum_legal_selected_inputs']==1
    assert relay['exact_target_enforcement'] is False and relay['is_anatomical_convergence'] is False
    assert old['strata']==new['strata'] and old['boundary']==new['boundary']
