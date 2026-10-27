"""Small CPU checks for reference execution reuse, without any CUDA run."""
from copy import copy
import importlib.util
from pathlib import Path
from threading import Lock

import numpy as np
import pytest

from retina_sim.contracts import ContractError
from retina_sim.group_inner_compiled import CompiledGroupInnerRHS
from retina_sim.group_inner_dynamics import RODS, MEMBRANE, key
from tests.test_group_inner_dynamics import fixture_model


@pytest.fixture(scope='module')
def execution_pair():
    model,state=fixture_model()
    current=CompiledGroupInnerRHS(model)
    path=Path(__file__).resolve().parents[1]/'reports/reference_backend_milestone_20261025_v1/before/retina_sim/compiled_kernel.py'
    spec=importlib.util.spec_from_file_location('retina_sim._pre_reference_execution',path)
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    previous=copy(current)
    previous._dispatch,_,_=baseline.dispatcher()
    previous._scratch=current._scratch.fork();previous._lock=Lock()
    previous.execution_profile=dict(current.execution_profile)
    return model,state,previous,current


def bits(a,b):
    assert a.dtype==b.dtype==np.float64 and a.shape==b.shape
    assert a.tobytes()==b.tobytes()


def test_merged_input_copy_keeps_full_rhs_and_private_workspace(execution_pair):
    model,state,old,new=execution_pair
    y=model.pack(state);saved=y.copy()
    for scale in (1.,1.0001,1.):
        trial=y*scale
        a,b=old(0.,trial),new(0.,trial)
        bits(a,b)
        assert not np.shares_memory(b,new._scratch.array)
        assert not np.shares_memory(b,trial)
        frozen=b.copy();new(0.,y);bits(b,frozen)
    bits(y,saved)


@pytest.mark.parametrize('section',('state_start','state_end','external'))
def test_merged_copy_same_nonfinite_failure_and_recovery(execution_pair,section):
    model,state,old,new=execution_pair
    y=model.pack(state);current=np.array([-1e-11,-2e-11])
    if section=='state_start':y[0]=np.nan
    elif section=='state_end':y[-1]=np.inf
    else:current[-1]=np.nan
    failures=[]
    for compiled in (old,new):
        with pytest.raises(ContractError) as failed:compiled(0.,y,rod_current=current)
        failures.append(str(failed.value))
    assert failures[0]==failures[1]
    valid=model.pack(state);bits(old(0.,valid),new(0.,valid))


def test_reference_held_rod_current_keeps_every_stage_dense_and_next_event(execution_pair,monkeypatch):
    model,state,old,new=execution_pair
    state={k:v.copy() for k,v in state.items()}
    state[key('rods',RODS,'G')]*=1.01
    initial=model.pack(state);g_slice=model.slices[key('rods',RODS,'G')]
    saved=model.compiled_rhs
    stage_g=[];sources=[]
    original=new.rhs_flat
    def capture(t,y,cone_release=None,rod_current=None):
        stage_g.append(y[g_slice].copy());sources.append(rod_current)
        return original(t,y,cone_release,rod_current)
    monkeypatch.setattr(new,'rhs_flat',capture)
    try:
        # A plain callable intentionally retains old per-stage rod evaluation.
        model.compiled_rhs=lambda t,y:old(t,y)
        before=model.advance_reference(state,.0009,.001,rtol=1e-6,atol=1e-9)
        model.compiled_rhs=new
        after=model.advance_reference(state,.0009,.001,rtol=1e-6,atol=1e-9)
        assert len(stage_g)==after.metrics['nfev']
        for g in stage_g:bits(g,initial[g_slice])
        assert sources and all(s is sources[0] for s in sources)
        assert not np.shares_memory(sources[0],initial)
        assert before.accepted_times==after.accepted_times
        assert before.metrics['nfev']==after.metrics['nfev']
        for t in (.0009,np.nextafter(.0009,np.inf),.00095,np.nextafter(.001,-np.inf),.001):
            side='left' if t==.001 else 'right'
            a,b=before.evaluate(t,side),after.evaluate(t,side)
            for k in model.state_keys:bits(a[k],b[k])
        committed,_,_=model.commit_ticks(.001,after.state,rod_tick=True,rod_light=np.array([100.,200.]))
        first_source=sources[0];sources.clear();stage_g.clear()
        model.advance_reference(committed,.001,.0011,rtol=1e-6,atol=1e-9)
        expected=new._rod_dispatch(model.pack(committed),new._rod_start,new._rod_count,new._rod_k,new._rod_n)
        bits(sources[0],expected)
        assert sources[0] is not first_source
        assert sources[0].tobytes()!=first_source.tobytes()
    finally:model.compiled_rhs=saved


@pytest.mark.parametrize('g',(-0.,0.,-1.))
def test_nonpositive_rod_state_keeps_generic_arithmetic_path(execution_pair,g):
    model,state,old,new=execution_pair
    y=model.pack(state);y[model.slices[key('rods',RODS,'G')]]=g
    derivative=new._reference_interval_rhs(y)
    assert derivative==new.rhs_flat
    bits(derivative(0.,y),old(0.,y))


def test_held_source_retains_first_domain_failure(execution_pair):
    model,state,old,new=execution_pair
    altered={k:v.copy() for k,v in state.items()}
    altered[key('AII_AC',MEMBRANE,'x')][:]=-.1
    failures=[];saved=model.compiled_rhs
    try:
        for compiled in (lambda t,y:old(t,y),new):
            model.compiled_rhs=compiled
            with pytest.raises(ContractError) as failed:
                model.advance_reference(altered,0.,.0001,rtol=1e-6,atol=1e-9)
            failures.append(str(failed.value))
    finally:model.compiled_rhs=saved
    assert failures[0]==failures[1]


def test_optional_execution_profile_counts_held_source_and_owned_copies(execution_pair):
    model,state,_,compiled=execution_pair
    assert not compiled.profile_enabled
    previous=dict(compiled.execution_profile);saved=model.compiled_rhs
    compiled.profile_enabled=True
    try:
        model.compiled_rhs=compiled
        result=model.advance_reference(state,0.,.0001,rtol=1e-6,atol=1e-9)
    finally:
        model.compiled_rhs=saved;compiled.profile_enabled=False
    delta={k:compiled.execution_profile[k]-previous[k] for k in previous}
    assert delta['rod_current_calls']==1
    assert delta['rhs_calls']==delta['successful_rhs_calls']==result.metrics['nfev']
    assert delta['state_copy_bytes']==delta['result_write_bytes']==model.nstate*8*result.metrics['nfev']
    assert delta['external_copy_bytes']==model.counts['rods']*8*result.metrics['nfev']
    assert 0<delta['dispatch_wall_seconds']<=delta['rhs_wall_seconds']
