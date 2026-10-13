from pathlib import Path
import json
import numpy as np
import pytest

from retina_sim.atlas_connections import ConnectionAtlas
from retina_sim.atlas_hc_master import HCMasterCoverageError
from retina_sim.contracts import ContractError

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def atlas():return ConnectionAtlas(ROOT)

def test_default_frozen_joint_keeps_approved_sampler_seed_settings_and_types(atlas):
    m=atlas.hc_master;source=m.metadata['source'];o=source['generator_options']
    expected=json.loads((ROOT/'configurations/hc_joint_synthetic_v1.json').read_bytes())['generator']
    expected.update(json.loads((ROOT/'configurations/hc_research_prototype_geometry_v1.json').read_bytes())['generator'])
    assert o==expected and source['seed']==20261010
    assert o['sweeps']==2 and o['a11']==o['a22']==o['a12']==4 and o['temperature']==1
    assert source['mosaic_rule']=='generate_joint_hc_fixed_canonical_master_then_crop'
    assert len(m.metadata['provenance']['joint_generation']['energy_after_each_sweep'])==2
    assert 'exp(-E/T)' in m.metadata['provenance']['joint_generation']['target_distribution']
    assert all(':hc:'+t+':' in k for k,t in zip(m.table.source_keys,m.types))
    assert not m.metadata['Poisson_fallback']

def test_frozen_master_request_order_crops_exact_same_ids_positions_types(atlas):
    boxes=((.16,.24,-.04,.04),(.18,.26,-.04,.04))
    a,b=[atlas.hc_patch(x) for x in boxes]
    other=ConnectionAtlas(ROOT)
    bb,aa=[other.hc_patch(x) for x in reversed(boxes)]
    for one,two in ((a,aa),(b,bb)):
        assert one.table.global_cell_ids==two.table.global_cell_ids and one.types==two.types
        np.testing.assert_array_equal(one.anchor.values,two.anchor.values)
    pa=dict(zip(a.table.global_cell_ids,a.anchor.values));pb=dict(zip(b.table.global_cell_ids,b.anchor.values))
    assert pa.keys()&pb.keys()
    for g in pa.keys()&pb.keys():np.testing.assert_array_equal(pa[g],pb[g])

def test_pair_context_outside_master_is_refused_without_automatic_resampling(atlas):
    with pytest.raises(HCMasterCoverageError,match='no Poisson fallback'):
        atlas.hc_patch((.78,.8,-.01,.01))
    with pytest.raises(ContractError,match='budget'):atlas.hc_patch((.16,.24,-.04,.04),max_cells=1)

def test_old_poisson_is_explicit_independent_preview_and_never_default(atlas):
    old=ConnectionAtlas(ROOT,hc_mosaic_mode='historical_marked_Poisson_preview_v1')
    p=old.hc_patch((.13,.23,-.08,.08))
    assert old.hc_master is None and p.metadata['canonical_rule']=='fixed_marked_Poisson_HC_preview_v1'
    assert 'not the approved joint' in p.metadata['role']
    assert p.table.source_version!=atlas.hc_master.table.source_version

def test_replot_S_candidate_is_registered_without_enabling_main_partition(atlas):
    r=atlas.atlas.coverage_catalog.records['Godat2024_MartinGrunert_replot_candidate_only']
    assert not r.active and r.evidence_status=='digitized_published_replot'
    assert len(r.provenance['readouts'])==11 and r.provenance['dense_pixel_trace_rows']==305
    assert not r.source_domain and r.application_domain is None
    assert atlas.atlas.config['S_partition_rule']=='MartinGrunert1999_curve_pending_no_deMonasterio_substitution'
