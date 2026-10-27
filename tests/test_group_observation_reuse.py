"""CPU-only same-snapshot reuse and owned monitor workspaces."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.group_inner_dynamics import RODS, MEMBRANE
from retina_sim.group_time_monitor import GroupTimeMonitor
from tests.test_group_inner_dynamics import fixture_model
from tests.test_group_time_monitor import source_fixture, evidence


def assert_arrays_identical(a, b):
    assert set(a)==set(b)
    for key in a:
        assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape
        assert a[key].tobytes()==b[key].tobytes(), key


def test_combined_witness_reuses_only_one_call_and_never_a_prior_snapshot(monkeypatch):
    model,state=fixture_model()
    source=model.outputs
    calls=[]
    def outputs(*args,**kwargs):
        calls.append('outputs')
        return source(*args,**kwargs)
    monkeypatch.setattr(model,'outputs',outputs)
    light=np.full(model.counts['rods'],100.)
    release=np.full(model.counts['cones'],.2)
    def rod_source():
        calls.append('rod_source')
        return light
    for i in range(3):
        state['rod_BC',MEMBRANE,'V']+=.0001*i
        release[:]+=.01*i
        before=canonical_hash(state)
        expected_out=model.outputs(0.,state,release)
        expected=model.pure_transition_witness(state,release,light,time_seconds=0.)
        calls.clear()
        out,pure,timing=model.observation_witness(0.,state,release,rod_source)
        assert calls==['outputs','rod_source']
        assert_arrays_identical(expected_out,out)
        assert_arrays_identical(expected,pure)
        assert canonical_hash(state)==before
        assert set(timing)=={'outputs_seconds','transition_seconds'}
        assert all(value>=0 for value in timing.values())
        assert all(not np.shares_memory(pure[k],state[k]) for k in pure)
    with pytest.raises(TypeError):
        model.commit_ticks(0.,state,_outputs=out)


@pytest.mark.parametrize('failure',['nonfinite','negative_contact','negative_rod_light','invalid_rod_state'])
def test_combined_witness_keeps_original_failure_and_input_ownership(failure):
    model,state=fixture_model()
    release=np.full(model.counts['cones'],.2)
    light=np.full(model.counts['rods'],100.)
    if failure=='nonfinite':state['rod_BC',MEMBRANE,'V'][0]=np.nan
    elif failure=='negative_contact':state['rods','individual_rod_RBC_contacts','m'][0]=-1.
    elif failure=='negative_rod_light':light[0]=-1.
    else:state['rods',RODS,'G'][0]=-100.
    before={k:v.copy() for k,v in state.items()}
    def original():
        model.outputs(0.,state,release)
        return model.pure_transition_witness(state,release,light,time_seconds=0.)
    with pytest.raises(ContractError) as expected:original()
    with pytest.raises(ContractError) as actual:
        model.observation_witness(0.,state,release,lambda:light)
    assert str(actual.value)==str(expected.value)
    assert_arrays_identical(before,state)


def test_monitor_owned_buffers_match_original_and_saved_payload_stays_owned(source_fixture):
    path=Path(__file__).resolve().parents[1]/'reports/reference_backend_milestone_20261025_v1/before/retina_sim/group_time_monitor.py'
    spec=importlib.util.spec_from_file_location('retina_sim._original_group_time_monitor',path)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    # The frozen implementation still reads the same unmodified root contract.
    original.__file__=str(Path(__file__).resolve().parents[1]/'retina_sim/group_time_monitor.py')
    assembly,seed=source_fixture
    new,point,units=evidence(source_fixture)
    old=original.GroupTimeMonitor(assembly,seed.states,parameters=seed.inner_parameters)
    saved=None;buffer_ids=None
    for i in range(4):
        states={k:v.copy() for k,v in point['states'].items()}
        rhs={k:np.full_like(v,(-1.)**i*i*1e-6) for k,v in states.items()}
        transitions={k:states[k]+i*1e-10 for k in new.transition_keys}
        for j,k in enumerate(states):states[k]+=((j%3)-1)*i*1e-9
        for monitor in (old,new):
            monitor.observe(i*.0001,'right',states,rhs,transitions=transitions,
                ports=point['ports'],aux=point['aux'],event_generation=i)
        a,b=old.payload(),new.payload()
        assert canonical_hash(a)==canonical_hash(b)
        for key in a['rows']:
            for field in ('lo','hi','rhs','defect'):
                assert a['rows'][key][field].tobytes()==b['rows'][key][field].tobytes()
        current={(k,n):id(row[n]) for k,row in new.rows.items() for n in ('lo','hi','rhs','defect')}
        if buffer_ids is None:buffer_ids=current
        else:assert current==buffer_ids
        if i==0:saved=deepcopy(b);owned=b
    assert canonical_hash(saved)==canonical_hash(owned)
    restored=GroupTimeMonitor.from_payload(assembly,seed.states,new.payload(),parameters=seed.inner_parameters)
    for monitor in (old,new,restored):
        monitor.observe(.0005,'left',states,rhs,transitions=transitions,
            ports=point['ports'],aux=point['aux'],event_generation=5)
    assert canonical_hash(old.payload())==canonical_hash(new.payload())==canonical_hash(restored.payload())
    bad={k:v.copy() for k,v in rhs.items()};bad[next(iter(bad))][0]=np.inf
    for monitor in (old,new):
        with pytest.raises(ContractError,match='Nonfinite original state/RHS'):
            monitor.observe(.0006,'right',states,bad,transitions=transitions)
    assert canonical_hash(old.payload())==canonical_hash(new.payload())
