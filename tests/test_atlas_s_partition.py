from pathlib import Path
import json
import numpy as np
import pytest
from retina_sim.atlas_connections import ConnectionAtlas
from retina_sim.atlas import AtlasROI
from retina_sim.atlas_s_partition import EffectiveMulattaSPartition
from retina_sim.geometry import SConeDensityField

ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope='module')
def atlas():return ConnectionAtlas(ROOT,atlas_profile_id='retina_atlas_Schein_bridge_research_v1',s_partition_profile='deMonasterio1985_mulatta_effective_S_v1')

def test_complete_legacy_density_and_square_rule_unchanged(atlas):
    raw=json.loads((ROOT/'sources/spatial/demonasterio1985_Scone_reference.json').read_bytes())
    old=SConeDensityField(raw,.223,magnification_source_id='fixture',central_policy='assay_square_PCHIP')
    xy=np.array([[.001,.001],[.035,0],[.35,0],[-.35,0],[0,1],[3,0]])
    np.testing.assert_array_equal(atlas.s_partition.field(xy),old(xy))
    assert atlas.s_partition.config['source_species']=='Macaca mulatta'
    assert not atlas.s_partition.config['Godat_or_Martin_hole_spliced']
    assert atlas.s_partition.field(np.array([[.01,.01]]))[0]==0

def test_global_label_domain_overlap_stability_and_no_forced_local_S(atlas):
    ref=atlas.atlas.config['reference_id'];a=AtlasROI((.030,.040,.001,.004),(.025,.045,.0001,.009),ref)
    b=AtlasROI((.034,.044,.001,.004),(.029,.049,.0001,.009),ref)
    ta=atlas.atlas.query(a).mapped_table;tb=atlas.atlas.query(b).mapped_table
    la,meta=atlas._types(ta);lb,_=atlas._types(tb);new=EffectiveMulattaSPartition(atlas.atlas);repeat,_=new.assign(ta)
    assert la==repeat
    da=dict(zip(ta.global_cell_ids,la));db=dict(zip(tb.global_cell_ids,lb))
    assert da.keys()&db.keys() and all(da[g]==db[g] for g in da.keys()&db.keys())
    assert set(la)<= {'L','M','S'} and meta['cross_species_transfer'] and meta['empty_subtype_samples_allowed']
    assert not meta['Martin50um_mask_applied'] and not meta['Godat_curve_used']
