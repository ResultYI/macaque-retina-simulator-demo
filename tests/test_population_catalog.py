"""Existing population identity and finite-boundary reuse, no new populations."""
from pathlib import Path
import json
import numpy as np
import pytest
from retina_sim.population_catalog import SharedPopulationCatalog,REQUIRED
from retina_sim.atlas import AtlasROI
from retina_sim.contracts import ContractError,Quantity

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def catalog():
    c=json.loads((ROOT/'configurations/atlas_isetbio_geometry_v1.json').read_bytes())
    a=json.loads((ROOT/'configurations/retina_atlas_isetbio_macaque_v1.json').read_bytes())
    return SharedPopulationCatalog(ROOT,c['population_master_catalog'],expected_cone_master=a['cone_master'])

@pytest.mark.parametrize('name',REQUIRED)
def test_overlapping_roi_uses_same_cells_without_generation(catalog,name,monkeypatch):
    import retina_sim.equal_mass_master as module
    monkeypatch.setattr(module,'generate_density',lambda *a,**kw:pytest.fail('ROI attempted population regeneration'))
    m=catalog.master(name);x0,x1,y0,y1=m.bounds;w=x1-x0;h=y1-y0
    a=(x0+.1*w,x0+.7*w,y0+.1*h,y0+.9*h);b=(x0+.3*w,x0+.9*w,y0+.1*h,y0+.9*h)
    def query(box):return catalog.crop(name,AtlasROI(box,box,catalog.reference_id))
    p,q=query(a),query(b);again=query(a)
    pp=dict(zip(p.global_cell_ids,map(tuple,p.coordinate('rf').values)))
    qq=dict(zip(q.global_cell_ids,map(tuple,q.coordinate('rf').values)));common=pp.keys()&qq.keys()
    assert common and all(pp[k]==qq[k] for k in common)
    assert p.global_cell_ids==again.global_cell_ids
    np.testing.assert_array_equal(p.coordinate('rf').values,again.coordinate('rf').values)
    assert p.source_version==q.source_version==m.namespace

def test_boundary_and_frame_reject_extrapolation(catalog):
    b=catalog.master('rods').bounds;cx=(b[0]+b[1])/2;cy=(b[2]+b[3])/2
    q=Quantity([[cx,cy],[b[0]+.005,cy]],'mm','functional_position','synthetic_fixture','functional_visual_field')
    np.testing.assert_array_equal(catalog.footprint_coverage('rods',q,.01)['complete'],[True,False])
    with pytest.raises(ContractError,match='MASTER_DOMAIN_INSUFFICIENT'):catalog.require_domain('rods',(b[0],b[1]+.001,b[2],b[3]))
    with pytest.raises(ContractError,match='frame conversion'):catalog.require_domain('rods',b,coordinate_frame='OPL')

def test_source_and_catalog_identity_reject_mix(catalog):
    with pytest.raises(ContractError,match='source differs'):catalog.master('rods',expected_source={'provider_id':'different'})
    with pytest.raises(ContractError,match='cone geometry'):SharedPopulationCatalog(ROOT,catalog.binding,expected_cone_master={'sha256':'different'})
    with pytest.raises(ContractError,match='catalog hash'):SharedPopulationCatalog(ROOT,dict(catalog.binding,sha256='0'*64))

def test_actual_model_rebuild_uses_all_shared_masters(catalog,monkeypatch):
    import retina_sim.population_revision_graph as graph
    monkeypatch.setattr(graph,'materialize_equal_mass_master',lambda *a,**kw:pytest.fail('Shared rebuild generated a population'))
    # A distinct, narrower owner/query inside the original supported region.
    # This exercises real ROI rebuilding rather than repeating the seed request.
    owner=(.341,.359,-.009,.009);query=(.301,.399,-.049,.049)
    record=graph.rebuild_roi_from_shared_masters(ROOT,AtlasROI(owner,query,catalog.reference_id),catalog)
    for name in REQUIRED:
        actual=record['population_revision']['masters'][name];expected=catalog.record['masters'][name]
        assert actual['sha256']==expected['sha256'] and actual['namespace']==expected['namespace']
    assert len(record['rod_inner']['tables']['rods']['global_cell_ids'])==catalog.record['masters']['rods']['count']
    assert record['population_revision']['shared_catalog']==catalog.binding

def test_canonical_target_contract_does_not_hide_empty_owner_path():
    from retina_sim.atlas_population_network import atlas_population_network
    c,_,audit=atlas_population_network(ROOT,execution_profile='atlas_isetbio_geometry_v1')
    contract=audit['atlas_runtime']['owner_target_contract']
    assert len(contract['canonical_outputs'])==5 and len(contract['active_owner_outputs'])==4
    assert contract['status']['small_bistratified_S_ON']=='EMPTY_SAMPLE'
    assert not contract['all_canonical_owner_paths_executed'] and not contract['synthetic_cells_added']
    assert 'small_bistratified_S_ON' not in c.components
    assert len(c.components)==29 and sum(len(x.state_specs) for x in c.components.values())==70
    assert sum(s.shape[0] for x in c.components.values() for s in x.state_specs)==59107
    assert audit['execution_profile']['migration_status']=='NOT_FINAL_MODEL_CONFIGURATION'
