"""Analytic optical fixtures plus unchanged Chen/Runner wiring; no physiology fit."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest

from retina_sim.contracts import ContractError, MissingCalibrationError, Quantity, canonical_hash
from retina_sim.photometry import (Aperture, QuantumEfficiencyProfile,
                                  photoisomerization_window_mean_rate)
from retina_sim.rod_capture import RodCaptureSampler, bind_rod_capture, capture_isetbio_retinal_photons


ROOT = Path(__file__).resolve().parents[1]
TAG = 'synthetic_fixture'


def q(value, unit, meaning):
    return Quantity(value, unit, meaning, TAG)


def parameters(area=2e-12, convention='normalized_mean'):
    # These manufactured values have no species interpretation or adoption.
    spatial = (q([[.5/area, .5/area]], 'm^-2', 'normalized_aperture_weight')
               if convention == 'normalized_mean' else q([[.5, .5]], 'dimensionless', 'aperture_mask'))
    aperture = Aperture(convention, spatial, q([[area, area]], 'm^2', 'spatial_quadrature_weight'),
        q(area, 'm^2', 'effective_collection_area'), 'analytic_test_definition', 'synthetic_physical_fixture')
    efficiency = QuantumEfficiencyProfile(q([.1, .2, .1], 'dimensionless', 'photoisomerization_efficiency'),
        'analytic_test_definition: explicit absolute probabilities, not normalized sensitivity',
        'synthetic_physical_fixture', ('rod_absorptance', 'isomerization_yield', 'macular_transmittance'),
        ('pupil_geometry', 'lens_transmittance'), 'Manufactured absorption*yield with unity macular transmission')
    config = json.loads((ROOT/'configurations/research/rod_capture.json').read_text())
    config.update(qualification='synthetic_physical_fixture', input_identity='analytic_fixture',
        radiometry_source='Manufactured W/m2/nm field', aperture_species='none: analytic fixture',
        efficiency_species='none: analytic fixture', spatial_kernel_source='two equal quadrature nodes at +/-0.1 um',
        exterior_background_source='unchanged source seed; diagnostic step only')
    return aperture, efficiency, config


@pytest.fixture
def frozen(tmp_path):
    recipe = dict(population='rods', radius_mm=2.)
    xy = np.array([[.4, -.05], [.6, .03]])
    arrays = {}
    for name, value in [('positions', xy), ('source_keys', np.array(['frozen:rod:0', 'frozen:rod:1'], dtype='S24'))]:
        path = tmp_path/(name+'.npy'); np.save(path, value, allow_pickle=False)
        arrays[name] = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    record = dict(schema='fixed_disk_population_arrays_v1', recipe=recipe,
        namespace=canonical_hash(recipe), regenerate_on_query=False, count=2,
        reference_id='composite_macaque_nasal_positive_superior_positive_planar_v1', **arrays)
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(record))
    return tmp_path, dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def sample(frozen, *, area=2e-12, convention='normalized_mean', reverse=False):
    aperture, efficiency, config = parameters(area, convention)
    xx, yy = np.meshgrid(np.array([.3, .5, .7])*1e-3, np.array([-.1, 0., .1])*1e-3)
    support = np.stack([xx, yy], axis=-1)
    if reverse: support = support[::-1, ::-1]
    sampler = RodCaptureSampler(*frozen, np.array([0, 1]), support,
        np.array([[-1e-7, 0.], [1e-7, 0.]]), aperture, efficiency, config)
    return sampler


def capture(sampler, multiplier=1., spatial=False):
    support = sampler.support
    profile = np.array([2., 3., 5.])*1e-8
    scale = 1+support[:, :, 0]*1000+2*support[:, :, 1]*1000 if spatial else np.ones(support.shape[:2])
    energy = np.stack([scale[:, :, None]*profile, 2*scale[:, :, None]*profile])*multiplier
    return sampler.capture(q(energy, 'W/(m^2 nm)', 'retinal_spectral_irradiance'),
        q([400., 500., 600.], 'nm', 'wavelength'), q([50., 100., 50.], 'nm', 'spectral_quadrature_weight'),
        np.arange(3)/150)


def test_analytic_units_area_intensity_and_mask_equivalence(frozen):
    base = capture(sample(frozen))
    # Independent rectangular/trapezoidal hand sum, SI constants written explicitly.
    expected = 2e-12*(2e-8*.1*400e-9*50+3e-8*.2*500e-9*100+5e-8*.1*600e-9*50)/(6.62607015e-34*299792458.)
    np.testing.assert_allclose(base.rod_photon_rate.values, np.array([[expected]*2, [2*expected]*2]), rtol=3e-15)
    np.testing.assert_allclose(capture(sample(frozen, area=4e-12)).rod_photon_rate.values,
                               base.rod_photon_rate.values*2, rtol=3e-15)
    np.testing.assert_allclose(capture(sample(frozen), multiplier=3).rod_photon_rate.values,
                               base.rod_photon_rate.values*3, rtol=3e-15)
    np.testing.assert_allclose(capture(sample(frozen, convention='dimensionless_mask')).rod_photon_rate.values,
                               base.rod_photon_rate.values, rtol=3e-15)
    assert base.provenance['absolute_capture_qualified'] is False


def test_space_orientation_determinism_and_iset_photon_bridge(frozen):
    sampler = sample(frozen, reverse=True)
    result = capture(sampler, spatial=True)
    uniform = capture(sampler).rod_photon_rate.values
    np.testing.assert_allclose(result.rod_photon_rate.values/uniform, [[1.3, 1.66]]*2, rtol=3e-15)
    assert result.identity == capture(sampler, spatial=True).identity
    wave = np.array([400., 500., 600.])
    sampler = sample(frozen)
    scale = 1+sampler.support[:, :, 0]*1000+2*sampler.support[:, :, 1]*1000
    energy = np.broadcast_to(np.array([2., 3., 5.])*1e-8, (2, 3, 3, 3)).copy()*scale[None, :, :, None]
    energy[1] *= 2
    photon = energy/(6.62607015e-34*299792458./(wave*1e-9))
    raw = dict(spatial_support_m=sampler.support, sha256='analytic_fixture',
        manifest={'schema': 'isetbio_cone_capture_v1'},
        retinal_photons=np.moveaxis(photon[:, ::-1], 0, -1), wave_nm=wave, edges=np.arange(3)/150)
    actual = capture_isetbio_retinal_photons(sampler, raw,
        q([50, 100, 50], 'nm', 'spectral_quadrature_weight'), optical_center_m=[.0005, 0], source_origin_seconds=0.)
    np.testing.assert_allclose(actual.rod_photon_rate.values, result.rod_photon_rate.values, rtol=3e-15)
    counts = q(uniform/150, 'R*/window', 'expected_photoisomerization_count')
    np.testing.assert_allclose(photoisomerization_window_mean_rate(counts,
        q(np.arange(3)/150, 's', 'integration_window_edges')).values, uniform)


def test_spectral_efficiency_direction(frozen):
    sampler = sample(frozen)
    wave = q([400, 500, 600], 'nm', 'wavelength')
    rates = []
    for k in range(3):
        photons = np.zeros((1, 3, 3, 3)); photons[:, :, :, k] = 1e12
        result = sampler.capture(q(photons, 'photons/(s m^2 nm)', 'retinal_spectral_photon_irradiance'),
            wave, q([1, 1, 1], 'nm', 'spectral_quadrature_weight'), [0, 1/150])
        rates.append(result.rod_photon_rate.values[0, 0])
    np.testing.assert_allclose(rates, [.2, .4, .2], rtol=3e-15)


def test_fail_closed(frozen):
    sampler = sample(frozen)
    aperture, efficiency, config = parameters()
    args = (*frozen, np.array([0, 1]), sampler.support, np.array([[0., 0.]]))
    with pytest.raises(MissingCalibrationError):
        RodCaptureSampler(*args, None, efficiency, config)
    config['qualification'] = 'calibrated_absolute'
    with pytest.raises(ContractError):
        RodCaptureSampler(*args, aperture, efficiency, config)
    config = parameters()[2]; config['factors_already_in_irradiance'].append('rod_absorptance')
    with pytest.raises(ContractError):
        RodCaptureSampler(*args, aperture, efficiency, config)
    with pytest.raises(ContractError, match='support'):
        RodCaptureSampler(*args[:4], np.array([[1., 0.]]), aperture, efficiency, parameters()[2])
    with pytest.raises(ContractError, match='150 Hz'):
        sampler.capture(q(np.zeros((1, 3, 3, 3)), 'W/(m^2 nm)', 'retinal_spectral_irradiance'),
            q([400, 500, 600], 'nm', 'wavelength'), q([1, 1, 1], 'nm', 'spectral_quadrature_weight'), [0, .01])


def test_chen_runner_binding_and_checkpoint(frozen):
    from tests.test_group_time_runner import toy
    from retina_sim.group_time_runner import GroupTimeArm
    from retina_sim.group_inner_dynamics import RODS
    from retina_sim.rod_pathway import rod_source_values

    sampler = sample(frozen)
    two = capture(sampler)
    # A single 150 Hz frame suffices for three original 100 us author ticks.
    result = sampler.capture(q(np.ones((1, 3, 3, 3))*1e-8, 'W/(m^2 nm)', 'retinal_spectral_irradiance'),
        q([400, 500, 600], 'nm', 'wavelength'), q([50, 100, 50], 'nm', 'spectral_quadrature_weight'), [0, 1/150])
    assembly, old, protocol = toy()
    p = assembly.populations['rods']; p.source_table = result.table_binding
    p.member_global_rows = np.array([1, 0]); p.member_offsets = np.array([0, 1, 2]); p.counts = np.array([1, 1])
    source = rod_source_values(ROOT)
    source['q'] = source['beta']*source['cdark']/(source['k']*source['gdark']**source['n'])
    source['smax'] = source['eta']/source['phi']*source['gdark']*(1+(source['cdark']/source['kGC'])**source['m'])
    old['inner_parameters']['rod_phototransduction'] = source
    before = canonical_hash(old)
    seed = bind_rod_capture(old, assembly, protocol, result, initial_state_policy='diagnostic_step_from_frozen_state')
    assert before == canonical_hash(old)
    np.testing.assert_array_equal(seed['rod_inputs'][0], result.rod_photon_rate.values[0, ::-1])
    arm = GroupTimeArm(assembly, seed, protocol)
    legacy = GroupTimeArm(assembly, old, protocol)
    assert arm.identity != legacy.identity
    np.testing.assert_array_equal(legacy.rod_rate_at(0.), [100., 100.])
    arm.initial_commit(); arm.ensure_native(.0001); arm.advance_inner(0., .0001)
    prior_R = arm.inner_state['rods', RODS, 'R'].copy()
    arm.snapshot(.0001, 'left', queue_node=True)
    arm.commit(.0001, rod=True, rod_sample_time=1e-13)
    expected_R = prior_R+.0001*(-source['sigma']*prior_R)+source['gamma']*(seed['rod_inputs'][0]*.0001)
    np.testing.assert_array_equal(arm.inner_state['rods', RODS, 'R'], expected_R)
    arm.snapshot(.0001, 'right', queue_node=True); arm.ack(.0001)
    payload = arm.checkpoint_payload()
    restored = GroupTimeArm.from_checkpoint(assembly, seed, protocol, payload)
    with pytest.raises(ContractError):
        GroupTimeArm.from_checkpoint(assembly, old, protocol, payload)
    for target in (.0002, .0003):
        for owner in (arm, restored):
            owner.ensure_native(target); owner.advance_inner(owner.inner_time, target)
            owner.snapshot(target, 'left', queue_node=True)
            owner.commit(target, rod=True, rod_sample_time=target-.0001+1e-13)
            owner.snapshot(target, 'right', queue_node=True); owner.ack(target)
        for k in arm.inner_state:
            np.testing.assert_array_equal(arm.inner_state[k], restored.inner_state[k])
    outputs = arm.inner.outputs(.0003, arm.inner_state)
    assert np.isfinite(arm.inner.rod_current(arm.inner_state)).all()
    assert np.isfinite(outputs['rods', 'voltage']).all()
    assert np.isfinite(outputs['rods', 'release']).all()
    with pytest.raises(ContractError, match='windows'):
        bind_rod_capture(old, assembly, protocol, two, initial_state_policy='diagnostic_step_from_frozen_state')


def test_frozen_research_rods_and_source_seed():
    from retina_sim.group_time_storage import load_bundle
    from retina_sim.rod_pathway import ChenPrimateRod, rod_source_values
    from retina_sim.group_inner_dynamics import RODS
    from conftest import NUMERIC_METRICS

    folder = ROOT/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2'
    manifest = folder/'executions/assemble_groups_01/GROUP_ARTIFACT.json'
    if not manifest.exists():
        pytest.skip('Local frozen research assets are not distributed with unit fixtures')
    descriptor = json.loads(manifest.read_text())['populations']['rods']
    with np.load(folder/'executions/assemble_groups_01/populations/rods/MEMBERSHIP.npz', allow_pickle=False) as z:
        membership = {k: z[k].copy() for k in ('member_global_rows', 'member_offsets', 'counts')}
    rows = membership['member_global_rows']
    aperture, efficiency, config = parameters()
    xx, yy = np.meshgrid([.0005, .001, .0015], [-.0003, 0., .0003])
    support = np.stack([xx, yy], axis=-1)
    started = perf_counter()
    sampler = RodCaptureSampler(ROOT, descriptor['source_table'], rows, support,
        np.array([[-1e-7, 0.], [1e-7, 0.]]), aperture, efficiency, config)
    build_seconds = perf_counter()-started
    started = perf_counter(); result = capture(sampler, spatial=True); capture_seconds = perf_counter()-started
    xy = sampler.xy
    expected = 2e-12*(2e-8*.1*400e-9*50+3e-8*.2*500e-9*100+5e-8*.1*600e-9*50)/(6.62607015e-34*299792458.)
    reference = np.array([1., 2.])[:, None]*(1+xy[:, 0]+2*xy[:, 1])[None, :]*expected
    np.testing.assert_allclose(result.rod_photon_rate.values, reference, rtol=4e-15)
    seed_folder = folder/'current_ac_source/seed'
    original = load_bundle(seed_folder, expected_kind='source_construction_seed')['payload']
    prior_hash = canonical_hash(original)
    source = rod_source_values(ROOT)
    actual = original['inner_parameters']['rod_phototransduction']
    for key, value in source.items():
        if key in actual: assert actual[key] == value
    # Restrict only the independent diagnostic copy's input frame scope.
    seed = dict(original, rod_inputs={i:original['rod_inputs'][i] for i in (0, 1)},
                native_inputs={i:original['native_inputs'][i] for i in (0, 1)})
    assembly = SimpleNamespace(populations={'rods': SimpleNamespace(source_table=descriptor['source_table'], **membership)})
    protocol = dict(input_fps=150., rod_step_seconds=.0001, input_origin_seconds=0., input_frame_indices=[0, 1])
    bound = bind_rod_capture(seed, assembly, protocol, result,
                             initial_state_policy='diagnostic_step_from_frozen_state')
    means = np.stack([reference[:, membership['member_offsets'][i]:membership['member_offsets'][i+1]].mean(axis=1)
                      for i in range(len(membership['counts']))], axis=1)
    np.testing.assert_allclose(np.stack(list(bound['rod_inputs'].values())), means, rtol=5e-15)
    assert prior_hash == canonical_hash(original)
    states = tuple(original['states']['rods', RODS, name].copy() for name in ('R', 'P', 'G', 'C', 'S'))
    model = SimpleNamespace(p=actual, q=actual['q'], smax=actual['smax'])
    for tick in range(100):
        frame = int(np.floor(tick*.0001*150))
        states = ChenPrimateRod._step(model, states, bound['rod_inputs'][frame])
    current = -actual['k']*states[2]**actual['n']
    assert np.isfinite(current).all()
    NUMERIC_METRICS['rod_capture'] = dict(
        frozen_master=descriptor['source_table'], local_rods=len(rows), groups=len(membership['counts']),
        first_three_global_ids=list(result.cell_ids[:3]), first_three_global_rows=rows[:3].tolist(),
        first_three_positions_mm=xy[:3].tolist(),
        maximum_relative_analytic_error=float(np.max(np.abs(result.rod_photon_rate.values/reference-1))),
        two_frame_capture_seconds=capture_seconds, verified_master_and_sampler_seconds=build_seconds,
        fine_capture_bytes=result.rod_photon_rate.values.nbytes,
        grouped_rate_bytes=sum(a.nbytes for a in bound['rod_inputs'].values()),
        sparse_sampler_bytes=sum(a.nbytes for a in (sampler.sampling.data, sampler.sampling.indices, sampler.sampling.indptr)),
        rate_range_Rstar_per_rod_per_s=[float(result.rod_photon_rate.values.min()), float(result.rod_photon_rate.values.max())],
        source_seed_manifest_sha256=hashlib.sha256((seed_folder/'manifest.json').read_bytes()).hexdigest(),
        source_seed_unchanged=True, author_ticks=100, author_duration_seconds=.01,
        current_range_pA=[float(current.min()), float(current.max())],
        capture_identity=result.identity, input_kind='synthetic_physical_fixture_on_actual_Rod_geometry',
        absolute_capture_qualified=False, physiological_validation=False,
        background_qualified=False, FIT_READY=False, production_adopted=False)
