"""New source-stopped geometry binds all consumers without replacing old assets."""
from pathlib import Path
import json
import numpy as np
import pytest
from retina_platform.config import RunConfig
from retina_sim.atlas_population_network import atlas_population_network,load_population_record,circuit_configuration
from retina_sim.isetbio_geometry import input_geometry_support
from retina_sim.contracts import ContractError

ROOT=Path(__file__).resolve().parents[1]
CONFIG='configurations/atlas_isetbio_source_stop_shared_v1.json'
MASTER='5ab2429aa920e9bd07f7a7d1a335edeffbbf8f263ddce731352809e7d6218a3a'

@pytest.fixture(scope='module')
def candidate():
    return atlas_population_network(ROOT,execution_profile='atlas_isetbio_geometry_v1',configuration_path=CONFIG)

def test_new_master_reaches_graph_contacts_frozen_geometry_and_input(candidate):
    compiled,_,audit=candidate
    record,graph=load_population_record(ROOT,execution_profile='atlas_isetbio_geometry_v1',configuration_path=CONFIG)
    old,_=load_population_record(ROOT,execution_profile='atlas_isetbio_geometry_v1')
    assert record['atlas']['cone_master']['sha256']==MASTER
    assert record['atlas']['owner_domain']==old['atlas']['owner_domain']
    assert not set(record['cones']['global_cell_ids'])&set(old['cones']['global_cell_ids'])
    ids,xy=input_geometry_support(ROOT,record);assert len(ids)==121811
    lookup={key:i for i,key in enumerate(ids)};pop=compiled.populations['cones']
    np.testing.assert_array_equal(xy[[lookup[x] for x in pop.cells.cell_id]],pop.receptive_field_positions.values)
    assert compiled.geometry_source_binding['frozen_graph']==graph
    assert audit['atlas_runtime']['frozen_typed_geometry']['missing_required_owner_input_rows']==0
    assert compiled.input_geometry_binding_required
    assert record['inner_contacts']['cones->midget_BC_ON']['edges']
    assert compiled.links and compiled.frozen_typed_geometry_binding
    old_masters=old['population_revision']['masters'];new_masters=record['population_revision']['masters']
    for name in ('parasol_RGC_ON','parasol_RGC_OFF'):
        assert old_masters[name]['sha256']==new_masters[name]['sha256']
    assert set(record['population_revision']['explicit_migration_rebuilt_masters'])=={'rods','rod_BC','AII_AC'}

def test_runconfig_explicit_binding_roundtrip_and_path_guard():
    cfg=RunConfig.from_dict({'execution_profile':'atlas_isetbio_geometry_v1','circuit_configuration_path':CONFIG})
    assert RunConfig.from_dict(cfg.to_dict()).circuit_configuration_path==CONFIG
    assert circuit_configuration(ROOT,CONFIG)['atlas_profile']=='retina_atlas_isetbio_source_stop_v1'
    with pytest.raises(ContractError,match='inside project'):circuit_configuration(ROOT,'../outside.json')
    with pytest.raises(ValueError,match='requires the frozen'):
        RunConfig.from_dict({'execution_profile':'engineering_preview','circuit_configuration_path':CONFIG,'media_path':__file__}).validate()

def test_old_binding_remains_independent():
    a=circuit_configuration(ROOT);b=circuit_configuration(ROOT,CONFIG)
    assert a['frozen_graph']!=b['frozen_graph']
    assert a['typed_circuit_geometry']!=b['typed_circuit_geometry']
    assert json.loads((ROOT/'configurations/retina_atlas_isetbio_macaque_v1.json').read_bytes())['cone_master']['sha256']!=MASTER

def test_adopted_default_and_legacy_loader_keep_distinct_graphs():
    cfg=RunConfig.project_default(ROOT)
    assert cfg.execution_profile=='atlas_isetbio_geometry_v1' and cfg.circuit_configuration_path==CONFIG
    assert cfg.background_require_converged and cfg.background_checkpoint_path==''
    _,legacy=load_population_record(ROOT,execution_profile='atlas_population_revision_v1')
    assert legacy['sha256']=='af456b72ce087c372c29f30a0c74198d2602d1bb3db32211eccc8691c534dd6b'
    assert legacy!=circuit_configuration(ROOT,CONFIG)['frozen_graph']
