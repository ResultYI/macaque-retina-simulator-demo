"""Synthetic entry checks only; no real equation provider or model advance."""
from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pytest

import tools.validate_ac_vs_equilibrium_start as entry
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.spatial_group_model import RGC


def fixture():
    states={('cones','reduced_native','X'):np.array([1.,2.,3.]),
        ('HC','reduced_native','Vh'):np.array([-.03]),('pool','FunctionalAmacrineVoltageGate_v1','V'):np.array([-.04])}
    seed=dict(states=states,spike_states={p:dict(history=np.zeros((3,1)),last_bin_counts=None,
        rng_state=deepcopy(np.random.PCG64(i).state),pending_events=[],conditional_rate_Hz=np.array([10.])) for i,p in enumerate(RGC)},
        inner_parameters=dict(identity='synthetic_fixed_parameters'),time_seconds=4.993,
        observation_prehistory=dict(delay_seconds=.00282,origin_seconds=4.993,history_times=np.array([4.993]),
            history_values=np.array([[-.03]]),prehistory=np.array([-.03]),unit='vH_native_voltage',
            meaning='native_relative_voltage',interpolation='linear'),
        input_reference=dict(native_background_td=np.array([100.,200.,300.]),rod_background_rstar_s=.745),
        native_inputs={i:np.array([100.,200.,300.]) for i in (748,749,750)},
        rod_inputs={i:np.full(2,.745) for i in (748,749,750)},patch_initialization=dict(source_sha256={}),
        patch_boundary=dict(identity='synthetic_boundary'),ac_vs_lineage=dict(artifact='synthetic_artifact',old_background_reused=False))
    populations={n:SimpleNamespace(group_count=1) for n in RGC}
    populations.update(cones=SimpleNamespace(group_count=3,group_ids=np.array(['cone0','cone1','cone2'])),rods=SimpleNamespace(group_count=2))
    blocks=tuple(SimpleNamespace(population=k[0],mechanism=k[1],name=k[2],count=len(v)) for k,v in states.items())
    assembly=SimpleNamespace(artifact_sha256='synthetic_artifact',populations=populations,core_population_rows={'cones':np.array([1])},
        state_blocks=blocks,probability_blocks=(),auxiliary_z1_blocks=())
    return seed,assembly


def bundle_and_receipt(seed,assembly):
    identity=dict(artifact=assembly.artifact_sha256,seed=canonical_hash(seed),sources={})
    return dict(payload=seed,identity=identity),dict(category='BACKGROUND_EQUILIBRIUM',passed=True,
        retained_local_background_equilibrium=True,whole_retina_or_natural_video_background_qualified=False,
        seed_sha256=canonical_hash(seed),explicit_exterior=deepcopy(seed['patch_boundary']))


class FakeArm:
    def __init__(self,seed):
        self.seed=deepcopy(seed);self.initial_commit_done=False;self.event_generation=0;self.last_draws={}
        self.spike_states=deepcopy(seed['spike_states']);self.inner=SimpleNamespace(parameters=deepcopy(seed['inner_parameters']))
        self.time=seed['time_seconds']
    def initial_snapshot(self):return dict(states=self.seed['states'],time_seconds=self.time)
    def delay_payload(self):
        d=self.seed['observation_prehistory']
        return dict(delay_seconds=d['delay_seconds'],history_times=d['history_times'].tolist(),history_values=list(d['history_values']),
            prehistory=dict(values=d['prehistory'],unit=d['unit'],meaning=d['meaning']),
            prehistory_stop_seconds=d['origin_seconds'],interpolation=d['interpolation'])


def test_frozen_geometry_contrast_retains_all_initial_conditions_and_rod():
    seed,assembly=fixture();original=canonical_hash(seed)
    prepared,stimulus,mask=entry.contrast_from_saved_seed(seed,assembly,manifest_hash='saved_manifest',receipt_sha256='saved_receipt')
    assert canonical_hash(seed)==original
    for field in ('states','spike_states','inner_parameters','observation_prehistory','rod_inputs'):
        assert canonical_hash(prepared[field])==canonical_hash(seed[field])
    np.testing.assert_array_equal(prepared['native_inputs'][748],seed['input_reference']['native_background_td'])
    np.testing.assert_array_equal(prepared['native_inputs'][749],np.array([100.,210.,300.]))
    np.testing.assert_array_equal(mask,[False,True,False])
    assert stimulus['next_frame_boundary_seconds']==749/150
    assert stimulus['stimulus_class']=='synthetic_engineering_local_contrast_not_real_movie'
    assert prepared['ac_vs_lineage']['saved_equilibrium_seed_used']
    assert 'old_background_reused' not in prepared['ac_vs_lineage']
    assert stimulus['native_frame_sha256']['748']!=stimulus['native_frame_sha256']['749']


def test_constructor_uses_saved_states_without_old_seed_factory():
    seed,assembly=fixture();captured={}
    def factory(a,s,p,**kwargs):captured.update(assembly=a,seed=s,protocol=p,kwargs=kwargs);return FakeArm(s)
    arm=entry.construct_arm(assembly,seed,{'frozen':True},lambda:None,arm_type=factory)
    assert captured['seed'] is seed and captured['assembly'] is assembly
    receipt,evidence=entry.initial_match(arm,seed)
    assert receipt['passed'] and receipt['before_initial_commit']
    assert canonical_hash(evidence['states'])==canonical_hash(seed['states'])


@pytest.mark.parametrize('field',['state','RNG','delay','parameters','time'])
def test_precommit_match_rejects_any_initial_condition_change(field):
    seed,_=fixture();arm=FakeArm(seed)
    if field=='state':arm.seed['states']['pool','FunctionalAmacrineVoltageGate_v1','V'][0]+=.001
    elif field=='RNG':arm.spike_states[RGC[0]]['rng_state']=deepcopy(np.random.PCG64(99).state)
    elif field=='delay':arm.seed['observation_prehistory']['prehistory'][0]+=.001
    elif field=='parameters':arm.inner.parameters['identity']='different'
    else:arm.time+=.0001
    with pytest.raises(ContractError,match='pre-commit'):entry.initial_match(arm,seed)


def test_normal_post_commit_sampling_is_outside_initial_equality_contract():
    seed,_=fixture();arm=FakeArm(seed)
    assert entry.initial_match(arm,seed)[0]['post_commit_sampling_must_not_be_compared_to_saved_seed']
    arm.initial_commit_done=True;arm.event_generation=1
    arm.spike_states[RGC[0]]['history'][0,0]=1
    with pytest.raises(ContractError,match='before any initial_commit'):entry.initial_match(arm,seed)


def test_static_validation_rejects_receipt_artifact_input_and_source_changes(tmp_path):
    seed,assembly=fixture();bundle,receipt=bundle_and_receipt(seed,assembly)
    assert entry.validate_saved_background(bundle,receipt,assembly)['real_model_constructed'] is False
    wrong=deepcopy(receipt);wrong['passed']=False
    with pytest.raises(ContractError,match='receipt'):entry.validate_saved_background(bundle,wrong,assembly)
    wrong=deepcopy(bundle);wrong['identity']['artifact']='wrong'
    with pytest.raises(ContractError,match='artifact'):entry.validate_saved_background(wrong,receipt,assembly)
    changed=deepcopy(seed);changed['native_inputs'][749][0]=101.
    b,r=bundle_and_receipt(changed,assembly)
    with pytest.raises(ContractError,match='constant native background'):entry.validate_saved_background(b,r,assembly)
    path=tmp_path/'pinned.txt';path.write_text('original',encoding='utf8')
    relative=str(path.relative_to(entry.ROOT));bundle['identity']['sources'][relative]=entry.digest(path)
    path.write_text('changed',encoding='utf8')
    with pytest.raises(ContractError,match='source pin'):entry.validate_saved_background(bundle,receipt,assembly)


def test_prepare_never_constructs_real_arm_or_solves_background(tmp_path,monkeypatch):
    seed,assembly=fixture();bundle,receipt=bundle_and_receipt(seed,assembly)
    directory=tmp_path/'background'/'source_seed';directory.mkdir(parents=True)
    (directory/'manifest.json').write_text('{}',encoding='utf8')
    (directory.parent/'result.json').write_text(json.dumps(receipt),encoding='utf8')
    bundle['manifest_hash']='synthetic_bundle'
    monkeypatch.setattr(entry,'static_assembly',lambda:assembly)
    monkeypatch.setattr(entry,'load_bundle',lambda *args,**kwargs:bundle)
    def forbidden(*args,**kwargs):raise AssertionError('No real model allowed during prepare')
    monkeypatch.setattr(entry,'construct_arm',forbidden)
    monkeypatch.setattr(entry,'bind_original_exterior',forbidden)
    output=tmp_path/'prepared';output.mkdir()
    ready=entry.prepare(directory,output)
    assert ready['status']=='READY_FOR_COORDINATED_RUN'
    assert ready['real_model_constructed'] is False and ready['new_background_solved'] is False
    assert ready['actual_precommit_initial_match']=='PENDING_EXPLICIT_RUN'
    assert (output/'dynamic_initial_seed'/'manifest.json').is_file()
