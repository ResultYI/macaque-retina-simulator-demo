"""Ownership and strict mathematical/event parity, without a large network."""
from pathlib import Path
import json
from unittest.mock import patch
import numpy as np
import pytest
from examples.full_network_fixture import full_fixture
from retina_sim.hybrid import HybridRunner
from retina_sim.contracts import SolverConfig
from retina_sim import solver_lifetime
from tools.probe_solver_lifetime import BASE,load_baseline

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('after,generation',[( (10,1,0),0),((11,1,0),1),((11,2,0),2),((10,1,1),2)])
def test_collection_covers_possible_live_solver_promotion(after,generation,monkeypatch):
    collected=[]
    monkeypatch.setattr(solver_lifetime,'collection_epoch',lambda:after)
    monkeypatch.setattr(solver_lifetime.gc,'collect',lambda gen:collected.append(gen) or 7)
    row=solver_lifetime.release_solver_cycles((10,1,0))
    assert collected==[generation] and row['generation']==generation and row['collected_objects']==7
    assert row['wall_seconds']>=0 and row['CPU_seconds']>=0


@pytest.mark.parametrize('branch',['endpoint','sampled'])
def test_actual_repeated_slice_probe_releases_arrays_without_changing_steps(branch):
    old=json.loads((BASE/('baseline_'+branch+'.json')).read_bytes())
    new=json.loads((BASE/('current_'+branch+'.json')).read_bytes())
    assert old['state_hash_per_slice']==new['state_hash_per_slice']
    assert old['accepted_times']==new['accepted_times']
    assert old['returned_rows_per_slice']==new['returned_rows_per_slice']
    for key in ('solver_calls','nfev','njev','nlu'):
        assert old['metrics'][key]==new['metrics'][key]
    assert old['live_solvers_per_slice']==list(range(1,13))
    assert new['live_solvers_per_slice']==[0]*12
    assert new['retained_K_bytes']==0
    assert new['final_owner_bytes']==new['final_vector_bytes']
    assert new['metrics']['solver_lifetime_gc_wall_seconds']>=0


def test_small_network_outputs_callbacks_events_clocks_rng_and_restore_identical(record_property):
    compiled,channels,_=full_fixture(ROOT)
    original=load_baseline().HybridRunner._integrate
    old=HybridRunner(compiled,compute_backend='python');new=HybridRunner(compiled,compute_backend='python')
    config=SolverConfig(rtol=1e-8,atol=1e-11,max_step=.0001,input_interpolation='analytic')
    a=old.initialize(config);b=new.initialize(config)
    samples=np.array([0.,.00011,.00047,.00073,.00103,.0013])
    first=[];second=[]
    def observer(rows):
        return lambda t,y,bundle:rows.append((float(t),y.tobytes()))
    with patch.object(HybridRunner,'_integrate',original):
        a,left=old.advance(a,channels,.0013,sample_times=samples,state_observer=observer(first))
    b,right=new.advance(b,channels,.0013,sample_times=samples,state_observer=observer(second))
    assert first==second
    assert left.outputs_with_units.keys()==right.outputs_with_units.keys()
    for key,value in left.outputs_with_units.items():
        other=right.outputs_with_units[key]
        assert value.values.tobytes()==other.values.tobytes(),key
        assert (value.unit,value.meaning,value.evidence_tag)==(other.unit,other.meaning,other.evidence_tag)
    assert old.checkpoint(a)==new.checkpoint(b)
    assert json.dumps(a.spike_states,sort_keys=True)==json.dumps(b.spike_states,sort_keys=True)
    assert a.component_tick_indices==b.component_tick_indices
    assert a.delay_states==b.delay_states and a.link_delay_states==b.link_delay_states
    assert a.spike_states and a.delay_states  # Exercise actual RNG and native observation-delay payloads.
    for key in ('solver_calls','nfev','njev','nlu'):
        assert old.integration_metrics[key]==new.integration_metrics[key]
    restored=new.restore(new.checkpoint(b))
    c,_=new.advance(b,channels,.0017);d,_=new.advance(restored,channels,.0017)
    assert new.checkpoint(c)==new.checkpoint(d)
    record_property('callback_observations_identical',len(first))
    record_property('solver_calls_identical',old.integration_metrics['solver_calls'])
    record_property('gc_wall_seconds_visible',new.integration_metrics['solver_lifetime_gc_wall_seconds'])
