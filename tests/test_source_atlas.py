from pathlib import Path
import hashlib
import numpy as np
import pytest

from retina_sim.atlas import AtlasROI
from retina_sim.atlas_sources import AtlasSourceBundle,PackerFourMeridianDensity
from retina_sim.contracts import ContractError,Quantity
from retina_sim.source_atlas import SourceDensityAtlas

ROOT=Path(__file__).resolve().parents[1]

def physical(points):return Quantity(points,'mm','anatomical_position','test','anatomical_retina')
def rf(points):return Quantity(points,'mm','functional_position','test','functional_visual_field')

def test_real_packages_preserve_intervals_and_gaps():
    b=AtlasSourceBundle(ROOT)
    rows=b.data['hc_rows']['rows']
    intervals=[r for r in rows if r['source_object']=='Table 2' and r['eccentricity_mm'] is None]
    assert {(r['eccentricity_min_mm'],r['eccentricity_max_mm']) for r in intervals}=={(2,3),(4,5)}
    assert b.data['hc_gaps']['direct_anatomical_arbor_measurements']==[]
    assert all(s['curve_points']==[] for s in b.data['S_constraints']['sources'])
    assert len(b.data['photoreceptor_points'])==116

def test_density_literal_and_registration():
    field=PackerFourMeridianDensity(ROOT,'rod')
    row=field.rows['nasal'][2]
    result=field.sample(physical([[row['eccentricity_mm'],0]]))
    assert result.density.values[0]==pytest.approx(row['density_per_mm2'])
    assert result.status==('source_supported_digitized_node',)
    with pytest.raises(ContractError,match='registration'):field.sample(rf([[1,0]]))
    with pytest.raises(ContractError):field.sample(Quantity([[1,0]],'deg','anatomical_position','test','anatomical_retina'))

def test_missing_rod_is_not_numeric_zero_or_absence():
    field=PackerFourMeridianDensity(ROOT,'rod')
    result=field.sample(physical([[.1,0],[1,0]]),allow_missing=True)
    assert result.density is None and result.supported_values[0] is None
    assert result.status[0]=='unsupported' and result.supported_values[1]>0
    with pytest.raises(ContractError,match='no extrapolation'):field(physical([[.1,0]]))

def test_central_cone_bridge_is_explicit_project_rule():
    field=PackerFourMeridianDensity(ROOT,'all_cones',central_policy='legacy_Packer_published_code_central_bridge_v1')
    result=field.sample(physical([[0,0]]))
    assert result.status==('project_central_bridge',) and result.density.values[0]>0
    with pytest.raises(ContractError):PackerFourMeridianDensity(ROOT,'rod',central_policy='legacy_Packer_published_code_central_bridge_v1')

def cell_dict(patch,owner=False):
    return {gid:xy for gid,xy,own in zip(patch.selection.global_cell_ids,
        patch.selection.coordinate('rf').values,patch.selection.owner) if not owner or own}

def test_overlap_query_order_partition_and_mapping_identity():
    a=SourceDensityAtlas(ROOT);b=SourceDensityAtlas(ROOT)
    ref=a.config['reference_id']
    left=AtlasROI((.34,.35,-.01,.01),(.30,.39,-.05,.05),ref)
    right=AtlasROI((.35,.36,-.01,.01),(.31,.40,-.05,.05),ref)
    whole=AtlasROI((.34,.36,-.01,.01),(.30,.40,-.05,.05),ref)
    al,ar=a.query(left),a.query(right)
    br,bl=b.query(right),b.query(left)
    for x,y in [(al,bl),(ar,br)]:
        dx,dy=cell_dict(x),cell_dict(y)
        assert dx.keys()==dy.keys()
        for gid in dx:np.testing.assert_array_equal(dx[gid],dy[gid])
    dl,dr=cell_dict(al),cell_dict(ar)
    assert dl.keys()&dr.keys()
    for gid in dl.keys()&dr.keys():np.testing.assert_array_equal(dl[gid],dr[gid])
    assert cell_dict(al,True).keys()|cell_dict(ar,True).keys()==cell_dict(a.query(whole),True).keys()
    assert set(al.mapped_table.global_cell_ids)==set(al.selection.global_cell_ids)
    assert al.manifest['density_scale']==1 and not al.manifest['network_run_ready']

def test_source_limits_and_budget_do_not_thin_density():
    a=SourceDensityAtlas(ROOT);ref=a.config['reference_id']
    central=a.query(AtlasROI((-.004,.004,-.004,.004),(-.01,.01,-.01,.01),ref))
    assert len(central.mapped_table.source_keys)==0
    assert set(central.mapping_status)=={'unsupported'}
    with pytest.raises(ContractError,match='budget'):a.query(AtlasROI((.34,.36,-.01,.01),(.30,.4,-.05,.05),ref),max_cells=1)
    with pytest.raises(ContractError,match='unsupported'):a.query(AtlasROI((.09,.11,-.01,.01),(.08,.12,-.02,.02),ref),population='rods')

def test_coordinate_metadata_revision_does_not_resample_receptors(monkeypatch):
    import retina_sim.source_atlas as module
    original=module.ScheinHenleDisplacement
    a=SourceDensityAtlas(ROOT)
    class RevisedCoordinateProvider(original):
        @property
        def provenance(self):return dict(super().provenance,coordinate_metadata_revision='test revision')
    monkeypatch.setattr(module,'ScheinHenleDisplacement',RevisedCoordinateProvider)
    b=SourceDensityAtlas(ROOT)
    roi=AtlasROI((.34,.36,-.01,.01),(.30,.40,-.05,.05),a.config['reference_id'])
    ap,bp=a.query(roi),b.query(roi)
    assert a.source_version!=b.source_version and a.mosaic_namespace==b.mosaic_namespace
    assert ap.selection.source_keys==bp.selection.source_keys
    assert ap.selection.global_cell_ids==bp.selection.global_cell_ids
    np.testing.assert_array_equal(ap.selection.coordinate('rf').values,bp.selection.coordinate('rf').values)

def test_old_successful_profile_and_source_equations_unchanged():
    assert hashlib.sha256((ROOT/'configurations/retina_contact_geometry_effective_v1.json').read_bytes()).hexdigest()=='ab63ff04001615a6a1e48bdf8ceb9af5da01596361dc273ab735d39d3db9f893'
    assert hashlib.sha256((ROOT/'retina_sim/vh2005.py').read_bytes()).hexdigest()=='cfc4ce4f9eb1015755cd790b180faaec83d2a9a7b8da083f3a471a00d36a4900'
    from retina_platform.config import RunConfig
    config=RunConfig()
    assert config.owner_x_mm==.35 and config.hc_arbor_profile=='W2000_equal_area_bounded_edges_047_104'
