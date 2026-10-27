"""Bitwise CPU comparison with the frozen pre-optimization opcode plan."""
from copy import copy,deepcopy
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest

from retina_sim.compiled_kernel import SUB,DIV,SIGMOID,FINITE,NONNEG,POSITIVE
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.group_inner_compiled import CompiledGroupInnerRHS
from retina_sim.group_inner_dynamics import MEMBRANE,BC_MECH,key
from retina_sim.spatial_group_model import RGC
from tests.test_group_inner_dynamics import fixture_model


@pytest.fixture(scope='module')
def pair():
    model,state=fixture_model()
    current=CompiledGroupInnerRHS(model)
    path=Path(__file__).with_name('fixtures')/'group_inner_pre_gate_reuse_v1.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='129b8303c39cd4a2cc21c546309e2bbf4f87a6484bfa1de109f1f2d7c1283e4c'
    before=copy(current)
    with np.load(path,allow_pickle=False) as old:
        for name in ('program','weights','indices','pointers','derivatives'):
            setattr(before,name,old[name].copy())
        before._scratch=SimpleNamespace(array=old['workspace'].copy())
    before._lock=Lock();before._views={}
    return model,state,before,current


def bits(a,b):
    assert a.dtype==b.dtype==np.float64 and a.shape==b.shape
    np.testing.assert_array_equal(a.view(np.uint64),b.view(np.uint64))


def test_only_two_duplicate_gate_chains_removed_and_barriers_preserved(pair):
    model,_,old,new=pair
    assert len(old.program)-len(new.program)==6
    for opcode in (SUB,DIV,SIGMOID):
        assert sum(old.program[:,0]==opcode)-sum(new.program[:,0]==opcode)==2
    checks=(FINITE,NONNEG,POSITIVE)
    np.testing.assert_array_equal(old.program[np.isin(old.program[:,0],checks)][:,[0,2]],
        new.program[np.isin(new.program[:,0],checks)][:,[0,2]])
    for name in ('weights','indices','pointers'):
        np.testing.assert_array_equal(getattr(old,name),getattr(new,name))
    n=sum(model.counts[p] for p in ('rod_BC','AII_AC'))
    assert old._scratch.array.nbytes-new._scratch.nbytes==8*(3*n+2)


def test_current_voltage_and_resource_rhs_match_old_plan_bitwise(pair):
    model,state,old,new=pair
    for voltage in (-1.,-.1,-.06,np.nextafter(-.045,-np.inf),-.045,np.nextafter(-.045,np.inf),-.03,0.,1.):
        for resource in (0.,.37,1.):
            changed={k:v.copy() for k,v in state.items()}
            for pop in ('rod_BC','AII_AC'):
                changed[key(pop,MEMBRANE,'V')][:]=voltage
                changed[key(pop,MEMBRANE,'x')][:]=resource
            y=model.pack(changed)
            bits(new(.00013,y),old(.00013,y))


def test_fresh_results_input_ownership_and_lock_are_unchanged(pair):
    model,state,old,new=pair
    y=model.pack(state);saved=y.copy()
    first=new(0.,y);frozen=first.copy()
    other=y.copy();other[model.slices[key('rod_BC',MEMBRANE,'V')]]+=.003
    second=new(0.,other)
    bits(first,frozen);bits(y,saved);bits(second,old(0.,other))
    assert not np.shares_memory(first,y) and not np.shares_memory(first,second)
    assert not np.shares_memory(first,new._scratch.array)
    new._lock.acquire()
    try:
        with pytest.raises(ContractError,match='already leased'):new(0.,y)
    finally:new._lock.release()
    bits(new(0.,y),old(0.,y))


@pytest.mark.parametrize('invalid',('nan_input','negative_resource','bc_domain','glycine_domain','rod_release','fp32'))
def test_same_original_failure_barrier_and_recovery(pair,invalid):
    model,state,old,new=pair
    y=model.pack(state)
    if invalid=='nan_input':y[0]=np.nan
    elif invalid=='negative_resource':y[model.slices[key('AII_AC',MEMBRANE,'x')]]=-.1
    elif invalid=='bc_domain':y[model.slices[key('midget_BC_ON',BC_MECH,'A')]]=2.
    elif invalid=='glycine_domain':y[model.slices[key('AII_AC','AII_FMB_glycine','a')]]=.1
    elif invalid=='rod_release':y[model.slices[key('rods','rod_membrane','release')]]=-.1
    else:y=y.astype(np.float32)
    barriers=[]
    for runner in (old,new):
        with pytest.raises(ContractError) as failed:runner(0.,y)
        cause=failed.value.__cause__
        if isinstance(cause,ValueError) and len(cause.args)==2 and isinstance(cause.args[1],(int,np.integer)):
            # Instruction PCs legitimately move; the original predicate ordinal must not.
            check_pcs=np.flatnonzero(np.isin(runner.program[:,0],(FINITE,NONNEG,POSITIVE)))
            ordinal=int(np.flatnonzero(check_pcs==cause.args[1])[0])
            barriers.append((type(failed.value),cause.args[0],ordinal))
        else:barriers.append((type(failed.value),str(failed.value)))
    assert barriers[0]==barriers[1]
    valid=model.pack(state)
    bits(new(0.,valid),old(0.,valid))


def test_reference_solver_dense_and_original_event_history_match_bitwise(pair):
    model,state,old,new=pair
    saved=model.compiled_rhs
    try:
        model.compiled_rhs=old
        a=model.advance_reference(state,.0009,.001,rtol=1e-6,atol=1e-9)
        model.compiled_rhs=new
        b=model.advance_reference(state,.0009,.001,rtol=1e-6,atol=1e-9)
    finally:model.compiled_rhs=saved
    assert a.metrics['nfev']==b.metrics['nfev'] and a.accepted_times==b.accepted_times
    for t in (.0009,np.nextafter(.0009,np.inf),.00095,np.nextafter(.001,-np.inf),.001):
        side='left' if t==.001 else 'right'
        av,bv=a.evaluate(t,side),b.evaluate(t,side)
        for k in model.state_keys:bits(av[k],bv[k])
    spikes={}
    for pop in RGC:
        rng=np.random.Generator(np.random.PCG64(34))
        spikes[pop]=dict(history=[[0]*model.counts[pop] for _ in range(3)],
            last_bin_counts=[1]*model.counts[pop],rng_state=deepcopy(rng.bit_generator.state),pending_events=[])
    kwargs=dict(cone_release=np.array([.9]),rod_light=np.array([100.,200.]),
        rod_tick=True,bc_tick=True,rgc_tick=True,spike_states=spikes)
    aa=model.commit_ticks(.001,a.state,**kwargs)
    bb=model.commit_ticks(.001,b.state,**kwargs)
    for k in model.state_keys:bits(aa[0][k],bb[0][k])
    assert canonical_hash(aa[1:])==canonical_hash(bb[1:])
