from pathlib import Path
import numpy as np
import pytest

from retina_sim.atlas import AtlasROI
from retina_sim.atlas_connections import ConnectionAtlas
from retina_sim.contracts import ContractError,Quantity
from retina_sim.coverage import load_atlas_source_domains

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def connection_atlas():return ConnectionAtlas(ROOT)

def roi(c,x=.35,y=0,w=.02,h=.02,halo=.04):
    return AtlasROI((x-w/2,x+w/2,y-h/2,y+h/2),
        (x-w/2-halo,x+w/2+halo,y-h/2-halo,y+h/2+halo),c.atlas.config['reference_id'])

def test_actual_inner_private_and_finite_HC_zero_is_not_completion(connection_atlas):
    c=connection_atlas;r=c.preview(roi(c,.035,w=.004,h=.004,halo=.007))
    assert r['summary']['mapped_owner_cones']>0 and r['summary']['inner_weighted_edges']>0
    assert set(r['cone_subtypes'])<= {'L','M'}
    route=r['inner_contacts']['midget_BC_ON->midget_RGC_ON']
    assert route['edges'] and all(e['weight']==1 for e in route['edges'])
    degree={g:0 for g,xy in zip(r['cones']['global_cell_ids'],r['cones']['coordinates']['rf']['values'])
        if .033<=xy[0]<.037000000000000005 and -.002<=xy[1]<.002}
    for e in r['HC_candidate_edges']:
        if e['cone_id'] in degree:degree[e['cone_id']]+=1
    assert r['summary']['zero_owner_candidate_rows']==sum(n==0 for n in degree.values())
    assert r['summary']['HC_mosaic_mode']=='fixed_joint_master_v2'
    assert not r['summary']['network_run_ready'] and not r['gap']['enabled']
    original=r['inner_tables']['midget_RGC_ON'];view=r['inner_soma_views']['midget_RGC_ON']
    assert set(view['global_cell_ids'])<=set(original['global_cell_ids'])
    assert not r['inner_soma_provenance']['old_15_percent_split_used']
    assert 'soma' in view['coordinates'] and 'soma' not in original['coordinates']

def test_unknown_S_is_conditional_not_retitled_LM_and_edges_stay_finite(connection_atlas):
    c=connection_atlas;r=c.preview(roi(c))
    assert set(r['cone_subtypes'])=={'UNKNOWN'}
    assert r['summary']['HC_candidate_edges']>0 and r['summary']['HC_conditional_subtype_edges']>0
    assert r['summary']['inner_weighted_edges']==0
    radii={row['hc_id']:row.get('arbor_radius_mm') for row in r['HC_rows']}
    assert all(e['distance_mm']<=radii[e['hc_id']]+1e-12 for e in r['HC_candidate_edges'])
    assert all(e['feedforward_weight'] is None and not e['dynamic_connection_enabled'] for e in r['HC_candidate_edges'])
    assert len({e['edge_id'] for e in r['HC_candidate_edges']})==len(r['HC_candidate_edges'])
    assert 'soma' not in r['inner_tables']['widefield_AC']['coordinates']
    assert 'rf' not in r['inner_tables']['widefield_AC']['coordinates']

def test_HC_fixed_joint_master_overlap_is_same_physical_anatomy(connection_atlas):
    c=connection_atlas
    a=c.hc_patch((.6,.65,-.04,.04));b=c.hc_patch((.62,.67,-.04,.04))
    da={g:(p,t) for g,p,t in zip(a.table.global_cell_ids,a.anchor.values,a.types)}
    db={g:(p,t) for g,p,t in zip(b.table.global_cell_ids,b.anchor.values,b.types)}
    assert da.keys()&db.keys()
    for gid in da.keys()&db.keys():
        np.testing.assert_array_equal(da[gid][0],db[gid][0]);assert da[gid][1]==db[gid][1]
    with pytest.raises(ContractError,match='budget'):c.hc_patch((0,1,0,1),max_cells=1)

def test_3mm_HC_reference_does_not_supply_missing_RF_pedicles(connection_atlas):
    c=connection_atlas;r=c.preview(roi(c,3),phase='phase2',hc_reference_bounds_mm=(2.95,3.05,-.05,.05))
    assert r['summary']['owner_cones']>0 and r['summary']['mapped_owner_cones']==0
    assert r['summary']['HC_cells']==0 and r['summary']['HC_candidate_edges']==0
    assert r['summary']['HC_query_status']=='HC_FIXED_MASTER_UNSUPPORTED' and r['HC'] is None
    assert r['summary']['owner_coordinate_unsupported']>0 and r['summary']['zero_owner_candidate_rows']==0
    assert any(b.get('status')=='HC_FIXED_MASTER_UNSUPPORTED' for b in r['closure_blockers'])
    assert not r['summary']['network_run_ready']
    assert any(b['frame']=='RF -> OPL' for b in r['closure_blockers'])
    rod=r['rod_inner']
    assert rod['metadata']['source_status']=='source_informed_project_geometry'
    assert rod['contacts']['rods->rod_BC']['edges']
    assert rod['contacts']['rod_BC->AII_AC']['edges']
    assert not rod['typed_points']['rods']['cone_Henle_applied']
    assert r['summary']['rod_inner_weighted_edges']>0

def test_S_single_field_reference_has_no_2D_interpolation_or_synthetic_curve():
    catalog=load_atlas_source_domains(ROOT)
    provider='Martin2000_nemestrina_single_field_reference'
    q=Quantity([[2.2,0],[0,2.2]],'mm','anatomical_position','test','anatomical_retina')
    result=catalog.query(provider,q,domain_role='source')
    assert result.status==('unsupported','unsupported')
    row=catalog.records[provider].provenance['measurement_rows'][0]
    assert row['s_cone_density_cells_per_mm2']==1119 and row['sample_points']==333
    assert row['retinal_meridian'] is None
