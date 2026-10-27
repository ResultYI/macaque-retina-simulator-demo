from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy import sparse

from retina_sim.contracts import ContractError
from retina_sim.group_parameter_binding import build_group_construction_seed, source_native_equilibrium
from retina_sim.spatial_group_graph import digest
from retina_sim.spatial_group_model import GroupModelAssembly, POPULATIONS, REQUIRED_ROUTE_PAIRS, RGC
from retina_platform.config import RunConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def assembly():
    populations = {name:SimpleNamespace(group_count=2,counts=np.array([3,5]),
        recording_auxiliary_group_rows=np.array([0,1,1])) for name in POPULATIONS}
    contacts = {('rods','rod_BC'):'individual_rod_RBC_contacts',('rod_BC','AII_AC'):'RBC_AII_AMPA',
        ('AII_AC','midget_BC_OFF'):'AII_FMB_glycine',('AII_AC','diffuse_BC_OFF'):'AII_DB1_glycine'}
    routes = {pre+'_to_'+post:SimpleNamespace(name=pre+'_to_'+post,pre=pre,post=post,
        contact_component=contacts.get((pre,post)),contact_groups=2 if (pre,post) in contacts else 0)
        for pre,post in REQUIRED_ROUTE_PAIRS}
    source_run='reports/cone_hc_reduced_foveal_6deg_run_20261024_v1'
    a=SimpleNamespace(populations=populations,routes=routes,artifact_sha256='synthetic-two-group-fixture',
        manifest=dict(parameter_sources=dict(main_configuration_sha256=digest(ROOT/'configurations/retina_platform_v1.json'),
            effective_profile_sha256=digest(ROOT/'configurations/effective_dynamics_v1.json')),
            configuration=dict(source_run=source_run)))
    def matrix(path):
        if path.endswith('contact_coefficient_S.npz'):return sparse.diags(np.array([3.,5.])*1e-9,format='csr')
        if path.endswith('sum_weights.npz'):return sparse.diags([3.,5.],format='csr')
        if path.endswith('mean_weights.npz'):return sparse.eye(2,format='csr')
        raise AssertionError(path)
    def array(path):
        if path=='input/group_background_native_td.npy':return np.array([10.,12.])
        if path.endswith('contact_pre_group.npy') or path.endswith('contact_post_group.npy'):return np.array([0,1])
        raise AssertionError(path)
    a.matrix,a.array=matrix,array
    a.state_blocks=GroupModelAssembly._states(a)
    a.probability_blocks=GroupModelAssembly._probability_blocks(a)
    from retina_sim.spatial_group_model import StateBlock
    a.auxiliary_z1_blocks=(StateBlock('cones','selected_original_z1_auxiliary','z1_aux','vH_native_voltage',
        'selected original nodes',3,0,'uninitialized_selected_original_auxiliary'),)
    return a


def construct(assembly):
    p=ROOT/assembly.manifest['configuration']['source_run']/'input_policy.json'
    return build_group_construction_seed(assembly,ROOT,input_policy_binding=dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p)),
        rng_seed=RunConfig.project_default(ROOT).seed)


def test_complete_source_seed_is_legal_but_never_qualified_background(assembly):
    result=construct(assembly)
    expected=(*assembly.state_blocks,*assembly.probability_blocks,*assembly.auxiliary_z1_blocks)
    assert set(result.states)=={(b.population,b.mechanism,b.name) for b in expected}
    assert all(result.states[b.population,b.mechanism,b.name].shape==(b.count,) for b in expected)
    assert result.source_construction_legal and not result.coupled_background_qualified
    assert result.identity['neural_model_advanced'] is False
    with pytest.raises(ContractError,match='not a qualified'):
        result.require_qualified_background()
    np.testing.assert_array_equal(result.states['cones','analytic_RE','R'],[10.,12.])
    assert result.input_reference['rod_background_rstar_s']==pytest.approx(.7450980392156863)
    for pop in RGC:
        assert result.inner_parameters['populations'][pop]['alpha']==90.
        assert result.spike_states[pop]['last_bin_counts'] is None
        assert result.spike_states[pop]['pending_events']==[]
        probabilities=np.array([result.states[pop,'RGC_expected_history','history_probability_'+str(i)] for i in range(8)])
        np.testing.assert_array_equal(probabilities.sum(0),[1.,1.])
    np.testing.assert_array_equal(result.states['cones','selected_original_z1_auxiliary','z1_aux'][1:],
        np.repeat(result.states['cones','selected_original_z1_auxiliary','z1_aux'][1],2))


def test_derived_references_re_evaluated_not_means_and_totals_counted_once(assembly):
    seed=construct(assembly);p=seed.inner_parameters['populations']
    v=seed.states['cones','reduced_native','Vis'];h=seed.states['HC','reduced_native','Vh'];vb=seed.states['cones','reduced_native','Vbar']
    from retina_sim.vh2005 import SOURCE_VALUES
    from scipy.special import expit
    release=expit((v-h-SOURCE_VALUES['V_k'])/SOURCE_VALUES['V_n'])/(vb/SOURCE_VALUES['V_I'])**SOURCE_VALUES['mu']
    np.testing.assert_allclose(p['midget_BC_ON']['baseline'],release,rtol=0,atol=0)
    assert p['midget_BC_ON']['C']==2e-11
    assert p['midget_BC_ON']['gL']==1e-9
    np.testing.assert_allclose(p['local_feedforward_AC']['reference_input'],[.5,.5],rtol=0,atol=0)
    assert p['local_feedforward_AC']['tau_A_seconds']==.01
    assert seed.observation_prehistory['delay_seconds']==.00282
    assert seed.observation_prehistory['feedback_uses_current_state'] is True


def test_native_source_equilibrium_rejects_crop_renormalization():
    with pytest.raises(ContractError,match='unit-DC'):
        source_native_equilibrium(sparse.csr_matrix([[.5]]),sparse.csr_matrix([[1.]]),np.array([10.]))


def test_source_identity_and_seed_substitution_are_rejected(assembly):
    assembly.manifest['parameter_sources']['effective_profile_sha256']='wrong'
    with pytest.raises(ContractError,match='identity'):
        construct(assembly)
    with pytest.raises(ContractError,match='seed'):
        build_group_construction_seed(assembly,ROOT,input_policy_binding={},rng_seed=1)
