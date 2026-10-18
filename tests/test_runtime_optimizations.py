"""Focused numerical/contract regression for the three local optimizations."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import numpy as np
import pytest
from scipy import sparse
from retina_sim.contracts import ContractError,Quantity,SolverConfig
from retina_sim.hybrid import HybridRunner
from retina_sim.network import InputBundle,JointNetwork
from examples.full_network_fixture import full_fixture

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def fixture():return full_fixture(ROOT)

@pytest.fixture(scope='module')
def legacy_class():
    # Frozen pre-patch evaluation algorithm; imports resolve normally, without
    # replacing live modules or relaxing any production identity checks.
    path=Path(__file__).with_name('fixtures')/'runtime_network_before.py'
    spec=importlib.util.spec_from_file_location('_runtime_optimization_before',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.UncachedJointNetwork

def tiny_runner():
    runner=object.__new__(HybridRunner)
    runner.model=SimpleNamespace(parameters=None,graph=None,rhs_flat=lambda t,y,*a:-y)
    runner.integration_metrics={'solver_calls':0,'nfev':0,'njev':0,'nlu':0,'integration_wall_seconds':0.,'minimum_interval_seconds':None,'maximum_nfev_per_interval':0}
    state=SimpleNamespace(time_seconds=0.,ordered_state_arrays=(np.array([1.]),))
    def update(state,y,t):state.time_seconds=t;state.ordered_state_arrays=(y.copy(),)
    runner._replace=update
    stimulus=SimpleNamespace(interpolation='analytic',at=lambda t:InputBundle({}))
    return runner,state,stimulus

@pytest.mark.parametrize('interior',[False,True])
def test_endpoint_and_interior_samples(monkeypatch,interior):
    import retina_sim.hybrid as hybrid
    original=hybrid.solve_ivp;calls=[]
    def traced(*a,**kw):calls.append(kw['t_eval']);return original(*a,**kw)
    monkeypatch.setattr(hybrid,'solve_ivp',traced)
    runner,state,stimulus=tiny_runner();stop=.0001;samples=[.000035,stop] if interior else [stop]
    rows=runner._integrate(state,stimulus,stop,SolverConfig(max_step=.0001),samples)
    assert (calls[0] is None)==(not interior)
    assert set(rows)==set(samples) and state.time_seconds==stop
    for t,y in rows.items():np.testing.assert_allclose(y,np.exp(-t),atol=2e-15,rtol=0)
    np.testing.assert_array_equal(state.ordered_state_arrays[0],rows[stop])

@pytest.mark.parametrize('status,time',[(1,.00005),(0,.00005),(1,.0001)])
def test_successful_early_termination_is_rejected(monkeypatch,status,time):
    monkeypatch.setattr('retina_sim.hybrid.solve_ivp',lambda *a,**kw:SimpleNamespace(success=True,status=status,
        message='terminal event',t=np.array([time]),y=np.array([[.9]]),nfev=1,njev=0,nlu=0))
    runner,state,stimulus=tiny_runner()
    with pytest.raises(ContractError,match='requested endpoint'):
        runner._integrate(state,stimulus,.0001,SolverConfig(),[.0001])
    assert state.time_seconds==0 and state.ordered_state_arrays[0][0]==1

def test_public_quantity_isolation_and_internal_boundary_checks():
    a=np.array([1.,2.]);q=Quantity(a,'mV/s','absolute_voltage_derivative','e')
    a[:]=np.nan
    np.testing.assert_array_equal(q.values,[1,2]);assert not q.values.flags.writeable
    with pytest.raises(ValueError):q.values[0]=5
    for kwargs in [{'unit':'A/s'},{'meaning':'wrong'},{'shape':(3,)},{'frame':'OPL'}]:
        args=dict(unit=q.unit,meaning=q.meaning,shape=(2,),check_finite=True);args.update(kwargs)
        with pytest.raises(ContractError):q.require(**args)
    with pytest.raises(ContractError):Quantity([np.nan],'mV','absolute_voltage','e')
    # A provider that corrupts its own previously constructed Quantity is still
    # rejected by the dynamic boundary, even after restoring its readonly flag.
    q.values.setflags(write=True);q.values[0]=np.nan;q.values.setflags(write=False)
    with pytest.raises(ContractError,match='Nonfinite'):q.require(q.unit,q.meaning,check_finite=True)

def bundle(ch,t):return InputBundle({k:v.at(t) for k,v in ch.items()})

def test_single_evaluation_matches_frozen_algorithm_and_reduces_work(fixture,legacy_class,monkeypatch):
    c,ch,_=fixture;new=JointNetwork(c);old=legacy_class(c)
    y=np.concatenate([q.values for q in new.initial_conditions()]);y[0]*=1.01;t=.0002;b=bundle(ch,t)
    counts={'value':0};original=sparse.csr_matrix.__matmul__
    def counted(self,other):counts['value']+=1;return original(self,other)
    monkeypatch.setattr(sparse.csr_matrix,'__matmul__',counted)
    old_inputs,old_outputs=old.evaluate(t,y,b);old_rhs=old.rhs_flat(t,y,b,old.parameters,old.graph);before=counts['value'];counts['value']=0
    new_inputs,new_outputs=new.evaluate(t,y,b);new_rhs=new.rhs_flat(t,y,b,new.parameters,new.graph);after=counts['value']
    assert after<before
    for key in old_outputs:np.testing.assert_array_equal(old_outputs[key].values,new_outputs[key].values)
    for cid in old_inputs:
        for key in old_inputs[cid]:np.testing.assert_array_equal(old_inputs[cid][key].values,new_inputs[cid][key].values)
    np.testing.assert_array_equal(new_rhs,old_rhs)
    # The solver receives an independent buffer; no retained Quantity/state alias.
    new_rhs[:]=np.nan;assert np.all(np.isfinite(y))
    np.testing.assert_array_equal(new.rhs_flat(t,y,b,new.parameters,new.graph),old_rhs)

def test_same_time_different_state_never_reuses_context(fixture,legacy_class):
    c,ch,_=fixture;new=JointNetwork(c);old=legacy_class(c);t=.0002;b=bundle(ch,t)
    y=np.concatenate([q.values for q in new.initial_conditions()]);z=y.copy();z[0]*=1.2;z[18]*=.99
    for state in [y,z,y]:
        np.testing.assert_array_equal(new.rhs_flat(t,state,b,new.parameters,new.graph),old.rhs_flat(t,state,b,old.parameters,old.graph))
    assert not np.array_equal(new.rhs_flat(t,y,b,new.parameters,new.graph),new.rhs_flat(t,z,b,new.parameters,new.graph))

def test_native_context_requires_same_owner_and_blocks(fixture):
    c,_,_=fixture;m=c.components['outer'].native_model;blocks=tuple(q.values.copy() for q in m.initial_conditions())
    context=m.evaluation_context(blocks);changed=tuple(x.copy() for x in blocks);changed[4][:]*=.99
    for a,b in zip(m.signals(changed,context),m.signals(changed)):np.testing.assert_array_equal(a,b)
    assert any(not np.array_equal(a,b) for a,b in zip(m.signals(changed,context),context.signals))

def test_closure_trials_match_uncached_algorithm(fixture,legacy_class,monkeypatch):
    c,ch,_=full_fixture(ROOT,True,fixture[2]['displacement_operator']);new=JointNetwork(c);old=legacy_class(c)
    t=.0002;y=np.concatenate([q.values for q in new.initial_conditions()]);y[0]*=1.01;b=bundle(ch,t)
    # Record each nonlinear residual result, rather than just its final root.
    import retina_sim.network as network
    original=network.root;trials=[]
    def traced(fun,x0,**kw):
        for scale in [0.,.001,-.002]:
            trial=np.full_like(x0,scale);trials.append(fun(trial).copy())
        return original(fun,x0,**kw)
    monkeypatch.setattr(network,'root',traced)
    old_root=old.evaluate.__globals__['root'];old_trials=[]
    def trace_old(fun,x0,**kw):
        for scale in [0.,.001,-.002]:old_trials.append(fun(np.full_like(x0,scale)).copy())
        return old_root(fun,x0,**kw)
    monkeypatch.setitem(old.evaluate.__globals__,'root',trace_old)
    _,a=new.evaluate(t,y,b);_,z=old.evaluate(t,y,b)
    assert trials and len(trials)==len(old_trials)
    for x,v in zip(trials,old_trials):np.testing.assert_array_equal(x,v)
    for key in a:np.testing.assert_array_equal(a[key].values,z[key].values)
