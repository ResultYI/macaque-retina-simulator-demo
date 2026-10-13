"""Assembly contracts only; completed integration is checked from its saved run."""
import json
from pathlib import Path
import numpy as np
import pytest
from retina_sim.atlas_runtime import atlas_midget_network
from retina_sim.contracts import ContractError
from retina_platform.config import RunConfig

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def network():
    return atlas_midget_network(ROOT)


def test_actual_ids_types_contact_weights_and_private_line(network):
    compiled,_,audit=network
    original=json.loads((ROOT/'data/atlas/atlas_r035_bridge_mulatta_connections_v1.json').read_bytes())
    assert audit['atlas_runtime']['ancestor_population_counts']['cones']==1465
    assert audit['atlas_runtime']['ancestor_population_counts']['HC']==1041
    assert all(g in original['cones']['global_cell_ids'] for g in compiled.populations['cones'].cells.cell_id)
    kinds=dict(zip(original['cones']['global_cell_ids'],original['cone_subtypes']))
    hc_types=dict(zip(original['HC']['table']['global_cell_ids'],original['HC']['types']))
    for e in audit['atlas_runtime']['HC_edges']:
        assert not (kinds[e['cone_id']]=='S' and hc_types[e['hc_id']]=='H1')
    for name,edges in audit['connectivity_support']['runtime_binding_edges'].items():
        actual={(e['pre_id'],e['post_id']):e['weight'] for e in original['inner_contacts'][name]['edges']}
        assert all(actual[e['pre_id'],e['post_id']]==e['weight'] for e in edges)
        if name in ('cones->midget_BC_ON','cones->midget_BC_OFF','midget_BC_ON->midget_RGC_ON','midget_BC_OFF->midget_RGC_OFF'):
            assert len({e['post_id'] for e in edges})==len(edges)==26
    assert audit['atlas_runtime']['passive_HC_instantiated']==0
    assert not any('rod' in k or 'AII' in k for k in compiled.components)
    assert audit['atlas_runtime']['unsupported_paths']['rod_pathway']['status']=='unsupported'


def test_no_zero_feedback_certificate_or_density_reduction():
    records=json.loads((ROOT/'reports/atlas_connections/20261012T065905260400Z/connections.json').read_bytes())
    record=records[1];bounds=record['atlas']['owner_domain']
    with pytest.raises(ContractError,match='JOINT_PROFILE_UNSUPPORTED'):
        atlas_midget_network(ROOT,owner_bounds_mm=bounds,connection_record=record)
    with pytest.raises(ContractError,match='budget'):
        atlas_midget_network(ROOT,max_spatial_cells=100)


def test_frozen_graph_change_is_rejected(monkeypatch):
    real=Path.read_bytes
    def read(path):
        if path.name=='atlas_r035_bridge_mulatta_connections_v1.json':return b'corrupted graph'
        return real(path)
    monkeypatch.setattr(Path,'read_bytes',read)
    with pytest.raises(ContractError,match='identity mismatch'):atlas_midget_network(ROOT)


def test_shared_active_default_kept_and_long_background_not_enabled():
    common=RunConfig.project_default(ROOT)
    assert common.execution_profile=='retina_contact_geometry_effective_v1'
    candidate=RunConfig.from_dict({'execution_profile':'atlas_midget_research_v1',
        'media_path':str(ROOT/'examples/platform_media/synthetic_rgb.png')})
    candidate.validate()
    assert candidate.warmup_seconds==0
    candidate.background_initialization_mode='bounded_convergence'
    with pytest.raises(ValueError,match='no long background'):candidate.validate()


def test_available_typed_owner_routes_bind_and_known_ineligible_diffuse_is_removed():
    compiled,_,audit=atlas_midget_network(ROOT,typed_completion=True)
    assert len(compiled.components)==19
    assert 'S_ON_BC' in compiled.populations and 'small_bistratified_S_ON' in compiled.populations
    assert len(compiled.populations['widefield_AC'].cells.cell_id)==3
    assert all(x in compiled.components for x in ('small_bistratified_S_ON_excitatory_S_ON_BC',
        'small_bistratified_S_ON_excitatory_diffuse_BC_OFF','small_bistratified_S_ON_inhibitory_local_feedforward_AC'))
    binding=audit['connectivity_support']['runtime_binding_edges']
    for name in ('local_feedforward_AC','local_crossover_AC','widefield_AC'):
        route=('diffuse_BC_OFF' if name=='local_crossover_AC' else 'diffuse_BC_ON')+'->'+name
        assert binding[route] and len({e['post_id'] for e in binding[route]})==len(compiled.populations[name].cells.cell_id)
    assert audit['atlas_runtime']['full_target_readiness']=='BLOCKED'
    assert not any('parasol' in x for x in compiled.components)
    assert audit['atlas_runtime']['unsupported_paths']['parasol']['kind']=='data_and_unapproved_source_transfer'
