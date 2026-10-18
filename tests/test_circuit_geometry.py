"""Canonical asset/ROI identity and coverage only; no neural time advances."""
from copy import copy,deepcopy
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from retina_sim.atlas import AtlasROI
from retina_sim.atlas_population_network import atlas_population_network,load_population_record
from retina_sim.circuit_geometry import FrozenCircuitGeometry,spatial_snapshot
from retina_sim.contracts import ContractError,Quantity
from retina_sim.typed_view import compiled_typed_view
from retina_sim.lattice_quality import triangle_quality,frozen_lattice_quality

ROOT=Path(__file__).resolve().parents[1]
OUTPUTS=('midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON')

@pytest.fixture(scope='module')
def asset():
    record,graph=load_population_record(ROOT,execution_profile='atlas_isetbio_geometry_v1')
    binding=json.loads((ROOT/'configurations/atlas_isetbio_geometry_v1.json').read_bytes())['typed_circuit_geometry']
    return FrozenCircuitGeometry(ROOT,binding,record,graph),record,graph

def test_production_build_uses_frozen_pools_anchors_and_gaps(monkeypatch,asset):
    import retina_sim.atlas_primary_rod as rod
    import retina_sim.ipl_geometry as ipl
    from retina_sim.atlas_connections import ConnectionAtlas
    import retina_sim.population_revision_graph as revision
    def forbidden(*a,**kw):raise AssertionError('Spatial generation/query reached during canonical assembly')
    for module,name in [(rod,'bounded_matrix'),(rod,'cKDTree'),(ipl,'freeze_centroid_anchors'),
            (ipl,'local_typed_sites'),(ipl,'widefield_grid'),(revision,'revise_local_graph'),
            (ConnectionAtlas,'preview'),(ConnectionAtlas,'_types')]:monkeypatch.setattr(module,name,forbidden)
    c,_,audit=atlas_population_network(ROOT,execution_profile='atlas_isetbio_geometry_v1')
    coverage=asset[0].validate_compiled(c)
    assert coverage['missing_required_owner_input_rows']==0 and coverage['owner_closure_nodes']==6928
    assert c.frozen_typed_geometry_binding==asset[0].binding
    assert len(c.components)==29 and sum(len(x.state_specs) for x in c.components.values())==70
    view=compiled_typed_view(c);before=spatial_snapshot(c)
    for b in [(.343,.36,-.01,.01),(.34,.357,-.01,.01),(.343,.36,-.01,.01)]:
        selected,plan=view.owner_recording(AtlasROI(b,view.query_domain,view.reference_id),OUTPUTS)
        assert plan.states==() and plan.ports and plan.cell_ids
        for cid,ids in selected['owners'].items():
            if ids:assert tuple(c.components[cid].outputs['voltage'].cell_ids[i] for i in plan.indices(c.components[cid].outputs['voltage'].cell_ids))==ids
    assert spatial_snapshot(c)==before
    assert audit['atlas_runtime']['frozen_typed_geometry']['boundary_policy_already_approved']

def test_full_derived_tables_reuse_ids_coordinates_and_empty_owner(asset):
    a=asset[0];ref=a.record['reference_id'];query=tuple(a.record['RF_input_domain_mm'])
    left=AtlasROI((.34,.357,-.01,.01),query,ref);right=AtlasROI((.343,.36,-.01,.01),query,ref)
    for name,role in [('midget_BC_ON','ipl_preterminal'),('local_feedforward_AC','ipl_post_arbor'),('midget_RGC_ON','ipl_post_arbor')]:
        common=sorted(set(a.select_table(name,left))&set(a.select_table(name,right)));assert common
        x=a.table_coordinate(name,role,common);y=a.table_coordinate(name,role,common)
        assert x.values.tobytes()==y.values.tobytes()
        with pytest.raises(ValueError):x.values.setflags(write=True)
    assert a.table_coordinate('S_ON_BC','rf').values.shape==(116,2)
    assert a.select_table('small_bistratified_S_ON',left)==()
    with pytest.raises(ContractError,match='coordinate unavailable'):a.table_coordinate('widefield_AC','soma')
    with pytest.raises(ContractError,match='outside current canonical domain'):
        a.select_table('midget_BC_ON',AtlasROI((.339,.36,-.01,.01),query,ref))

def test_asset_hash_and_derived_geometry_changes_rejected(asset):
    a,record,graph=asset
    with pytest.raises(ContractError,match='asset hash mismatch'):
        FrozenCircuitGeometry(ROOT,dict(a.binding,sha256='0'*64),record,graph)
    altered=dict(record);altered['inner_tables']=dict(record['inner_tables'])
    t=deepcopy(altered['inner_tables']['S_ON_BC']);t['coordinates']['rf']['values'][0][0]+=.001
    altered['inner_tables']['S_ON_BC']=t
    with pytest.raises(ContractError,match='Derived tables/anchors/contacts changed'):
        FrozenCircuitGeometry(ROOT,a.binding,altered,graph)

def test_actual_weight_or_gap_change_is_rejected(asset):
    c,_,_=atlas_population_network(ROOT,execution_profile='atlas_isetbio_geometry_v1');a=asset[0]
    changed=copy(c);links=list(c.links);matrix=links[0].matrix.copy();matrix.data[0]*=.99
    links[0]=replace(links[0],matrix=matrix);changed.links=tuple(links)
    with pytest.raises(ContractError,match='Actual typed populations/contacts/weights/gaps differ'):a.validate_compiled(changed)
    changed=copy(c);gaps=list(c.spec.gap_junctions);e=gaps[0].conductance_evidence;q=e.value
    gaps[0]=replace(gaps[0],conductance_evidence=replace(e,value=Quantity(q.values*1.01,q.unit,q.meaning,q.evidence_tag,q.coordinate_frame)))
    changed.spec=replace(c.spec,gap_junctions=tuple(gaps))
    with pytest.raises(ContractError,match='Actual typed populations/contacts/weights/gaps differ'):a.validate_compiled(changed)

def test_saved_coordinate_mesh_q_definition_and_regions():
    xy=np.array([[0.,0.],[1.,0.],[.5,np.sqrt(3)/2]])
    np.testing.assert_allclose(triangle_quality(xy,[[0,1,2]]),[1.],rtol=0,atol=1e-15)
    thin=np.array([[0.,0.],[1.,0.],[.5,.000001]])
    assert triangle_quality(thin,[[0,1,2]])[0]<1e-9
    result=frozen_lattice_quality(xy,{'inside':(-1.,2.,-1.,2.),'empty':(2.,3.,2.,3.)})
    assert result['regions'][1]['triangles']==1 and result['regions'][2]['triangles']==0
    assert not result['physiological_acceptance'] and not result['algorithm_threshold_changed']
