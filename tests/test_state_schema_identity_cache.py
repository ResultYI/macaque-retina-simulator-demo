"""Schema-cache regression: identity rejection remains active after warm reuse."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from retina_sim import engine
from retina_sim.contracts import ContractError,Quantity,SolverConfig,StateSpec,canonical_hash
from retina_sim.mechanism_contracts import MechanismStateSpec


def model_and_state(spec_type=StateSpec,*,mutable_ids=False,mutable_registry=False):
    ids=['a','b'] if mutable_ids else ('a','b')
    spec=spec_type('v',ids,(2,),'mV','absolute_voltage','synthetic_test')
    specs=[spec] if mutable_registry else (spec,)
    model=SimpleNamespace(state_specs=specs,parameters=SimpleNamespace(parameter_hash='p'),
        graph=SimpleNamespace(graph_hash='g'),model_id='test',version='1',model_hash='m',
        evidence_tag='synthetic_fixture',_state_spec_hash='deliberately_stale_private_hash')
    q=Quantity([1.,2.],'mV','absolute_voltage',model.evidence_tag)
    state=engine.initialize_state(model,model.parameters,model.graph,(q,),SolverConfig())
    return model,state


@pytest.mark.parametrize('spec_type',[StateSpec,MechanismStateSpec])
def test_immutable_registry_reuses_exact_canonical_identity(monkeypatch,spec_type):
    calls=[];original=engine.canonical_hash
    def traced(value):calls.append(value);return original(value)
    monkeypatch.setattr(engine,'canonical_hash',traced)
    model,state=model_and_state(spec_type)
    for _ in range(6):engine.validate_state(state,model)
    assert len(calls)==1
    assert state.state_spec_hash==canonical_hash(model.state_specs)
    # The unrelated pre-existing model cache is never trusted.
    assert state.state_spec_hash!=model._state_spec_hash


def test_registry_replacement_rejects_old_state_and_recomputes(monkeypatch):
    model,state=model_and_state();calls=[];original=engine.canonical_hash
    monkeypatch.setattr(engine,'canonical_hash',lambda value:(calls.append(value),original(value))[1])
    model.state_specs=(replace(model.state_specs[0],state_name='other'),)
    for _ in range(2):
        with pytest.raises(ContractError,match='state_spec_hash'):engine.validate_state(state,model)
    assert len(calls)==1 and calls[0] is model.state_specs


@pytest.mark.parametrize('mutable_ids,mutable_registry',[(True,False),(False,True)])
def test_mutable_registry_is_never_cached(monkeypatch,mutable_ids,mutable_registry):
    model,state=model_and_state(mutable_ids=mutable_ids,mutable_registry=mutable_registry)
    calls=[];original=engine.canonical_hash
    monkeypatch.setattr(engine,'canonical_hash',lambda value:(calls.append(value),original(value))[1])
    engine.validate_state(state,model);engine.validate_state(state,model)
    if mutable_ids:model.state_specs[0].cell_ids[0]='changed'
    else:model.state_specs[0]=replace(model.state_specs[0],state_name='changed')
    with pytest.raises(ContractError,match='state_spec_hash'):engine.validate_state(state,model)
    assert len(calls)==3 and not hasattr(model,'_engine_state_schema_identity')


def test_warm_schema_cache_preserves_shape_and_other_identity_checks():
    model,state=model_and_state()
    bad_shape=replace(state,ordered_state_arrays=(np.zeros(1),))
    with pytest.raises(ContractError,match='State shape changed'):engine.validate_state(bad_shape,model)
    model.model_hash='changed'
    with pytest.raises(ContractError,match='model_hash'):engine.validate_state(state,model)


def test_custom_schema_with_mutable_content_keeps_full_hashing():
    model,state=model_and_state()
    spec=model.state_specs[0]
    # Custom schemas are accepted by canonical_hash only when represented as a
    # dataclass, so use a mutable dataclass with the same canonical fields.
    from dataclasses import make_dataclass
    custom=make_dataclass('MutableSpec',[(key,object) for key in vars(spec)])
    model.state_specs=(custom(**vars(spec)),)
    state=replace(state,state_spec_hash=canonical_hash(model.state_specs))
    engine.validate_state(state,model)
    model.state_specs[0].state_name='changed'
    with pytest.raises(ContractError,match='state_spec_hash'):engine.validate_state(state,model)
