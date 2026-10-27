"""Manufactured CPU integration checks; no real network or CUDA validation."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from retina_sim.contracts import ContractError, Quantity, canonical_hash
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.group_inner_dynamics import RODS
from retina_sim.group_time_storage import save_bundle,load_bundle
from retina_sim.spatial_group_model import StateBlock, RGC
from retina_sim.vh2005 import NativeConeH1,SOURCE_VALUES
from tests.test_group_inner_dynamics import fixture_model


def toy():
    model,state=fixture_model();assembly=model.assembly
    for name,p in assembly.populations.items():
        p.group_ids=np.array([name+':'+str(i) for i in range(p.group_count)])
    assembly.populations['cones'].recording_auxiliary_group_rows=np.array([0],dtype=np.int64)
    assembly.populations['cones'].recording_auxiliary_fine_rows=np.array([0],dtype=np.int64)
    assembly.auxiliary_z1_blocks=(StateBlock('cones','selected_original_z1_auxiliary','z1_aux','vH_native_voltage','selected original z1',1,0),)
    native=SimpleNamespace(p=SOURCE_VALUES,A=assembly.matrix('routes/cones_to_HC/mean_weights.npz'),
        B=assembly.matrix('routes/HC_to_cones/mean_weights.npz'),nc=1,nh=1,tag='synthetic_fixture',
        state_specs=tuple(SimpleNamespace(unit='td' if n in ('R','E') else 'vH_native_voltage',meaning=n)
            for n in ('R','E','X','C','Vis','g_i','Vbar','z1','z2','Vh')))
    values=NativeConeH1.equilibrium(native,Quantity([100.],'td','native_retinal_illuminance','synthetic_fixture'))
    native_state={n:np.asarray(v.values) for n,v in zip(('R','E','X','C','Vis','g_i','Vbar','z1','z2','Vh'),values)}
    states={k:v.copy() for k,v in state.items()}
    for n in ('R','E'):states['cones','analytic_RE',n]=native_state[n]
    for n in ('X','C','Vis','g_i','Vbar'):states['cones','reduced_native',n]=native_state[n]
    for n in ('z2','Vh'):states['HC','reduced_native',n]=native_state[n]
    states['HC','reduced_native','w']=native.A@native_state['z1']
    states['cones','selected_original_z1_auxiliary','z1_aux']=native_state['z1'].copy()
    spikes={}
    for pop in RGC:
        n=model.counts[pop];rng=np.random.Generator(np.random.PCG64(52))
        spikes[pop]=dict(history=[np.zeros(n).tolist() for _ in range(3)],last_bin_counts=None,
            rng_state=deepcopy(rng.bit_generator.state),pending_events=[],conditional_rate_Hz=np.zeros(n))
    seed=dict(states=states,spike_states=spikes,inner_parameters=model.parameters,time_seconds=0.,
        native_inputs={0:np.array([100.],dtype=np.float64)},rod_inputs={0:np.full(2,100.,dtype=np.float64)},
        observation_prehistory=dict(delay_seconds=.00282,origin_seconds=0.,history_times=np.array([0.]),
            history_values=native_state['Vh'][None,:],prehistory=native_state['Vh'],unit='vH_native_voltage',
            meaning='native_relative_voltage',interpolation='linear'))
    protocol=dict(start_seconds=0.,stop_seconds=.0003,input_fps=150.,input_origin_seconds=0.,clock_origin_seconds=0.,
        rod_step_seconds=.0001,BC_RGC_period_seconds=.001,recording_step_seconds=.0005,native_observation_delay_seconds=.00282,
        input_frame_indices=[0],reference=dict(rtol=1e-6,atol=1e-9,max_step=.0001),
        candidate=dict(rtol=1e-6,atol=1e-9,fast_max_step=.0001))
    return assembly,seed,protocol


@pytest.mark.parametrize('mode',['reference','candidate'])
def test_initial_snapshot_does_not_integrate_or_consume_rng_then_one_native_window_ahead(mode):
    assembly,seed,protocol=toy()
    arm=GroupTimeArm(assembly,seed,protocol,mode=mode)
    before=canonical_hash(arm.spike_states)
    initial=arm.snapshot(0.,'right',queue_node=True)
    assert arm.native_intervals==[] and arm.native_state.time_seconds==0.
    assert canonical_hash(arm.spike_states)==before
    assert set(initial['states'])==set(seed['states'])
    arm.initial_commit()
    assert set(arm.last_draws)==set(RGC)
    for pop,row in arm.last_draws.items():
        assert row['draw_index'].shape == row['u'].shape
        assert np.all(np.diff(row['draw_index'])==1)
    arm.ensure_native(.0001,'left')
    assert arm.native_state.time_seconds==protocol['stop_seconds'] > .0001
    accepted=arm.advance_inner(0.,.0001)
    interior=arm.snapshot(.000037,'right',inner_state=accepted.evaluate(.000037),queue_node=False)
    assert arm.delay.history_times==[0.]
    end=arm.snapshot(.0001,'left',queue_node=True)
    assert arm.delay.history_times==[0.,.0001]
    assert set(end['draws'])==set(RGC)
    assert 'history_values' in end['aux']['observation_delay']
    arm.commit(.0001,rod=True,rod_sample_time=1e-13)
    arm.snapshot(.0001,'right',queue_node=True)
    assert arm.delay.history_times==[0.,.0001]
    arm.ack(.0001)
    assert arm.time_seconds==.0001 and len(arm.native_intervals)<=2


@pytest.mark.parametrize('mode',['reference','candidate'])
def test_checkpoint_keeps_future_native_and_all_source_state(tmp_path,mode):
    assembly,seed,protocol=toy()
    arm=GroupTimeArm(assembly,seed,protocol,mode=mode)
    initial_snapshot=arm.initial_snapshot()
    arm.initial_commit();arm.ensure_native(.0001)
    arm.advance_inner(0.,.0001)
    arm.snapshot(.0001,'left',queue_node=True)
    arm.commit(.0001,rod=True,rod_sample_time=1e-13)
    arm.snapshot(.0001,'right',queue_node=True);arm.ack(.0001)
    original=arm.snapshot(.0001,'right')
    payload=arm.checkpoint_payload(monitor={'test':'complete-separately'},recording={'test':'complete-separately'})
    save_bundle(tmp_path/'checkpoint',kind='complete_model_checkpoint',identity={'arm':arm.identity},payload=payload,
        qualification=dict(default_adopted=False,fit_eligible=False))
    saved=load_bundle(tmp_path/'checkpoint',expected_kind='complete_model_checkpoint',expected_identity={'arm':arm.identity})
    restored=GroupTimeArm.from_checkpoint(assembly,seed,protocol,saved['payload'],mode=mode)
    assert restored.native_state.time_seconds==.0003 > restored.time_seconds==.0001
    after=restored.snapshot(.0001,'right')
    for k in original['states']:
        np.testing.assert_array_equal(original['states'][k],after['states'][k])
        np.testing.assert_array_equal(original['rhs'][k],after['rhs'][k])
    assert canonical_hash(original['aux'])==canonical_hash(after['aux'])
    assert canonical_hash(original['draws'])==canonical_hash(after['draws'])
    for target in (.0002,.0003):
        for owner in (arm,restored):
            owner.ensure_native(target)
            owner.advance_inner(owner.inner_time,target)
            owner.snapshot(target,'left',queue_node=True)
            owner.commit(target,rod=True,rod_sample_time=target-.0001+1e-13)
            owner.snapshot(target,'right',queue_node=True);owner.ack(target)
        for k in arm.inner_state:np.testing.assert_array_equal(arm.inner_state[k],restored.inner_state[k])
    assert canonical_hash(arm.spike_states)==canonical_hash(restored.spike_states)


def test_rod_channel_holds_across_true_native_frame_edge_without_snapping():
    arm=GroupTimeArm.__new__(GroupTimeArm)
    arm.protocol=dict(input_origin_seconds=0.,clock_origin_seconds=0.)
    arm.start=0.
    arm.seed=dict(native_inputs={0:np.array([10.]),1:np.array([90.])},
                  rod_inputs={0:np.array([7.]),1:np.array([55.])})
    edge=1/150
    assert arm.frame_at(edge,'left')==0 and arm.frame_at(edge,'right')==1
    np.testing.assert_array_equal(arm.rod_rate_at(edge,'right'),[7.])
    assert arm.rod_sample_at(edge,'right')==(.0066,0)
    np.testing.assert_array_equal(arm.rod_rate_at(.0067,'right'),[55.])
    np.testing.assert_array_equal(arm.rod_rate_at(np.nextafter(edge,0.),'right'),[7.])


def test_clock_rejection_is_transactional_and_events_flush_once_before_monitor():
    assembly,seed,protocol=toy()
    arm=GroupTimeArm(assembly,seed,protocol,mode='reference')
    before=canonical_hash(arm.spike_states)
    with pytest.raises(ContractError,match='skipped or repeated'):
        arm.commit(0.,rod=True,rod_sample_time=1e-13)
    assert canonical_hash(arm.spike_states)==before
    arm.initial_commit();arm.advance_inner(0.,.0001)
    pop=RGC[0]
    arm.spike_states[pop]['pending_events']=[dict(cell_index=0,time_seconds=.00003,bin_start_seconds=0.,bin_stop_seconds=.001)]
    arm.snapshot(.00005,'right',inner_state=arm.last_inner_advance.evaluate(.00005),queue_node=False)
    assert arm.spike_states[pop]['pending_events']==[]
    assert len(arm.emitted_events_unrecorded[pop])==1
    arm.snapshot(.00005,'left',inner_state=arm.last_inner_advance.evaluate(.00005),queue_node=False)
    assert len(arm.emitted_events_unrecorded[pop])==1
