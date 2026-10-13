from pathlib import Path
import numpy as np
import pytest
from retina_sim.contracts import Quantity,ContractError
from retina_sim.henle import ScheinHenleDisplacement
from retina_sim.henle_bridge import ScheinBoundedBridge
from retina_sim.source_atlas import SourceDensityAtlas
from retina_sim.atlas import AtlasROI
from retina_sim.atlas_connections import ConnectionAtlas

ROOT=Path(__file__).resolve().parents[1]
def q(x,rf=False):return Quantity(x,'mm','functional_position' if rf else 'cone_inner_segment_position','project_approximation','functional_visual_field' if rf else 'anatomical_retina')
@pytest.fixture(scope='module')
def bridge():return ScheinBoundedBridge(ROOT)

def test_unchanged_source_nodes_and_approved_C0_bridge_slopes(bridge):
    strict=ScheinHenleDisplacement(ROOT)
    angles={'nasal':0,'temporal':np.pi,'superior':np.pi/2,'inferior':3*np.pi/2}
    for m,theta in angles.items():
        r=bridge.curves[m].inner_segment_radius_mm;xy=r[:,None]*np.array([[np.cos(theta),np.sin(theta)]])
        np.testing.assert_allclose(bridge.apply(q(xy)).values,strict.apply(q(xy)).values,atol=1e-14,rtol=0)
        rr,p=bridge.nodes[m]
        assert (p[1]-p[0])/(rr[1]-rr[0])==pytest.approx({'nasal':.64,'temporal':.76,'superior':.60,'inferior':.80}[m])
        assert (p[-2]-p[-3])/(rr[-2]-rr[-3])==pytest.approx({'nasal':.8448,'temporal':.672,'superior':.4,'inferior':.8}[m])
    assert 'C0' in bridge.provenance['continuity'] and bridge.provenance['strict_source_retained']

def test_positive_radius_inverse_monotone_and_identity_to6mm(bridge):
    r=np.r_[np.linspace(.0001,.025,20),np.linspace(.025,6,200)]
    for theta in np.linspace(0,2*np.pi,13)[:-1]:
        xy=r[:,None]*np.array([[np.cos(theta),np.sin(theta)]]);p=bridge.apply(q(xy))
        assert np.all(np.diff(np.linalg.norm(p.values,axis=1))>=-1e-12)
        assert np.all(np.linalg.norm(p.values,axis=1)>=r-1e-12)
        np.testing.assert_allclose(bridge.inverse(p).values,xy,atol=2e-14,rtol=0)
    np.testing.assert_allclose(bridge.apply(q([[6,0]])).values,[[6,0]],atol=0)
    with pytest.raises(ContractError,match='application'):bridge.apply(q([[6.01,0]]))

def test_origin_is_radial_limit_and_requires_explicit_angle(bridge):
    assert bridge.radial_limit()['status']=='DIRECTION_UNDEFINED'
    assert bridge.coverage(q([[0,0]]))['points'][0]['status']=='DIRECTION_UNDEFINED'
    with pytest.raises(ContractError,match='DIRECTION_UNDEFINED'):bridge.apply(q([[0,0]]))
    p=bridge.apply(q([[0,0]]),origin_direction_policy='fixed_angle_project_origin_v1',origin_direction_radians=np.pi/2)
    np.testing.assert_allclose(p.values,[[0,.15]],atol=1e-16)
    with pytest.raises(ContractError,match='inverse'):bridge.inverse(Quantity([[.14,0]],'mm','pedicle_position','project_approximation','OPL'))

def test_conservative_inverse_enclosure_contains_all_sampled_supported_points(bridge):
    box=(.13,.24,-.08,.08);bound=bridge.inverse_support_bounds(box)['bounds_RF_mm']
    rng=np.random.default_rng(33);points=rng.uniform([box[0],box[2]],[box[1],box[3]],(500,2));points=points[np.linalg.norm(points,axis=1)>.15]
    rf=bridge.inverse(Quantity(points,'mm','pedicle_position','project_approximation','OPL')).values
    assert np.all((rf[:,0]>=bound[0])&(rf[:,0]<=bound[1])&(rf[:,1]>=bound[2])&(rf[:,1]<=bound[3]))
    assert bridge.inverse_support_bounds((-.04,.04,-.04,.04))['model_pedicle_free']

def test_bridge_changes_coordinate_cache_identity_without_resampling_RF():
    strict=SourceDensityAtlas(ROOT);new=SourceDensityAtlas(ROOT,profile_id='retina_atlas_Schein_bridge_research_v1')
    roi=AtlasROI((-.004,.004,-.004,.004),(-.009,.009,-.009,.009),strict.config['reference_id'])
    a=strict.query(roi);b=new.query(roi)
    assert a.selection.global_cell_ids==b.selection.global_cell_ids
    np.testing.assert_array_equal(a.selection.coordinate('rf').values,b.selection.coordinate('rf').values)
    assert strict.source_version!=new.source_version and not len(a.mapped_table.source_keys)
    assert len(b.mapped_table.source_keys)==len(b.selection.source_keys)

def test_actual_original_HC_cohort_receives_full_upstream_query_without_new_HC():
    c=ConnectionAtlas(ROOT,atlas_profile_id='retina_atlas_Schein_bridge_research_v1')
    roi=AtlasROI((.033,.037,-.002,.002),(.026,.044,-.009,.009),c.atlas.config['reference_id']);r=c.preview(roi)
    assert r['summary']['HC_cells']==11 and r['summary']['owner_cones']==4
    assert r['dependency_query']['expanded_RF_query_mm']!=list(roi.query_bounds_mm)
    assert all(x['query_complete_sampled'] and x['candidate_count']>0 for x in r['HC_rows'])
    assert r['summary']['zero_owner_candidate_rows']==4  # no radius/density repair
    assert r['HC']['metadata']['query_pair_context_complete']
