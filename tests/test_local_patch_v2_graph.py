"""Finite-halo operators preserve parent identities, weights and exterior terms."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from retina_sim.local_patch import load_patch_config
from retina_sim.local_patch_graph import build_local_patch_graph
from retina_sim.spatial_group_model import RGC

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def bounded_graphs(tmp_path_factory):
    out = tmp_path_factory.mktemp('bounded_graphs')
    config = load_patch_config(ROOT/'configurations/local_patch_v1/aggregate_small.json')
    config['boundary_policy'] = 'bounded_background_halo_v2'
    results = []
    for radius in (.3,.6):
        request = deepcopy(config)
        request['patch_geometry']['support_radius_deg'] = radius
        results.append(build_local_patch_graph(ROOT,request,out/str(radius)))
    return tuple(results)


def test_support_reduces_and_scales_with_halo(bounded_graphs):
    small,large = bounded_graphs
    assert small.populations['cones'].group_count == 201
    assert small.populations['HC'].group_count == 229
    assert large.populations['cones'].group_count == 480
    assert large.populations['HC'].group_count == 420
    for name,rows in small.parent_population_rows.items():
        assert np.isin(rows,large.parent_population_rows[name]).all()
    assert small.state_scalar_count < large.state_scalar_count < 2274749
    for name in RGC:
        assert small.populations[name].group_count == (4 if name == 'parasol_RGC_OFF' else 0)


def test_every_local_and_boundary_product_equals_parent(bounded_graphs):
    rng = np.random.Generator(np.random.PCG64(261026))
    for patch in bounded_graphs:
        for route in patch.routes.values():
            for kind in ('sum_weights','mean_weights'):
                name = 'routes/'+route.name+'/'+kind+'.npz'
                original = patch.base.matrix(name)
                x = rng.standard_normal(original.shape[1])
                parent = (original@x)[patch.parent_population_rows[route.post]]
                local = patch.matrix(name)@x[patch.parent_population_rows[route.pre]]
                exterior = patch.external_matrix(name)@x
                np.testing.assert_allclose(local+exterior,parent,rtol=3e-15,atol=3e-15)
        name = 'gap/laplacian_S.npz'
        x = rng.standard_normal(patch.base.matrix(name).shape[1])
        rows = patch._gap_rows()
        np.testing.assert_allclose(patch.matrix(name)@x[rows]+patch.external_matrix(name)@x,
                                   (patch.base.matrix(name)@x)[rows],rtol=3e-15,atol=1e-25)


def test_chemical_and_contact_sources_are_dynamic(bounded_graphs):
    for patch in bounded_graphs:
        assert patch.external_matrix('routes/HC_to_cones/mean_weights.npz').nnz == 0
        assert patch.external_matrix('routes/cones_to_HC/mean_weights.npz').nnz > 0
        for route in patch.routes.values():
            if route.kind == 'inner_chemical':
                assert patch.external_matrix('routes/'+route.name+'/mean_weights.npz').nnz == 0
            if route.contact_component:
                prefix = 'routes/'+route.name+'/'
                pre = patch.array(prefix+'contact_pre_group.npy')
                assert (pre >= 0).all()
                assert (pre < patch.populations[route.pre].group_count).all()


def test_gap_retains_original_diagonal_and_full_current_balance(bounded_graphs):
    for patch in bounded_graphs:
        name = 'gap/laplacian_S.npz'
        local,external = patch.matrix(name),patch.external_matrix(name)
        np.testing.assert_array_equal(local.diagonal(),patch.base.matrix(name).diagonal()[patch._gap_rows()])
        assert external.nnz > 0
        residual = np.asarray(local.sum(axis=1)).ravel()+np.asarray(external.sum(axis=1)).ravel()
        assert np.max(np.abs(residual),initial=0.) < 3e-15*np.max(np.abs(local.data),initial=0.)


def test_recording_and_singleton_identity_are_retained(bounded_graphs,tmp_path):
    from retina_sim.group_time_recording import GroupRecording
    patch = bounded_graphs[0]
    assert patch.target_parent_row == 8163
    assert patch.populations[patch.target_population].counts[patch.target_local_row] == 1
    recorder = GroupRecording(patch,tmp_path/'recording',[0.,.0005,.001])
    assert sum(len(item['ports']) for item in recorder.components) == 45
    assert len(patch.populations['cones'].recording_auxiliary_fine_rows) == 0


def test_full_reference_policy_retains_every_frozen_state(tmp_path):
    config = load_patch_config(ROOT/'configurations/local_patch_v1/aggregate_small.json')
    config['boundary_policy'] = 'full_reference_v2'
    patch = build_local_patch_graph(ROOT,config,tmp_path/'reference')
    assert sum(p.group_count for p in patch.populations.values()) == 301814
    assert patch.state_scalar_count+patch.probability_scalar_count+patch.auxiliary_z1_scalar_count == 2274749
    for name,rows in patch.parent_population_rows.items():
        np.testing.assert_array_equal(rows,np.arange(patch.base.populations[name].group_count))
    assert patch.external_matrix('gap/laplacian_S.npz').nnz == 0


def test_core_aii_direct_bc_gap_inputs_are_dynamic_without_component_expansion(bounded_graphs):
    from retina_sim.spatial_group_model import BC
    for patch in bounded_graphs:
        rows=patch.parent_population_rows['AII_AC'][patch.core_population_rows['AII_AC']]
        for route in patch.base.routes.values():
            if route.kind=='gap' and route.post=='AII_AC' and route.pre in BC:
                needed=np.unique(patch.base.matrix('routes/'+route.name+'/sum_weights.npz')[rows].indices)
                assert np.isin(needed,patch.parent_population_rows[route.pre]).all()
        assert patch.external_matrix('gap/laplacian_S.npz').nnz>0
        assert patch.state_scalar_count<2274749
