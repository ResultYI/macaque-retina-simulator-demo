from dataclasses import asdict
import json
import numpy as np
from scipy.integrate import solve_ivp
from .contracts import ContractError, Quantity, StateDerivativeVector, SimulationState, canonical_hash

def _state_schema_hash(model):
    """Reuse the identical hash only for an unchanged, deeply immutable registry.

    Large populations repeat their cell-ID tuples across many state blocks.
    Registry replacement still forces hashing; mutable/custom schemas retain
    the original full validation on every call.
    """
    specs=model.state_specs
    cached=getattr(model,"_engine_state_schema_identity",None)
    if cached is not None and cached[0] is specs:return cached[1]
    digest=canonical_hash(specs)
    from .contracts import StateSpec
    from .mechanism_contracts import MechanismStateSpec
    immutable=type(specs) is tuple
    checked_ids=set()
    if immutable:
        for spec in specs:
            if (type(spec) not in (StateSpec,MechanismStateSpec)
                or any(type(getattr(spec,key)) is not str for key in
                       ("state_name","unit","meaning","initial_condition_source"))
                or type(spec.shape) is not tuple or any(type(n) is not int for n in spec.shape)
                or type(spec.cell_ids) is not tuple):
                immutable=False;break
            if id(spec.cell_ids) not in checked_ids:
                if any(type(gid) is not str for gid in spec.cell_ids):
                    immutable=False;break
                checked_ids.add(id(spec.cell_ids))
    if immutable:
        try:model._engine_state_schema_identity=(specs,digest)
        except (AttributeError,TypeError):pass
    return digest

def initialize_state(model,parameters,graph,initial_conditions,solver_config,time_seconds=0):
    if hasattr(model,"advance_author_discrete") and solver_config.method!="Chen2024AuthorDiscrete":
        raise ContractError("Chen2024 cannot silently switch to an adaptive solver")
    if parameters.parameter_hash!=model.parameters.parameter_hash or graph.graph_hash!=model.graph.graph_hash:
        raise ContractError("Initialization model/parameter/graph mismatch")
    if len(initial_conditions)!=len(model.state_specs): raise ContractError("Initial state registry mismatch")
    arrays=[]
    for q,spec in zip(initial_conditions,model.state_specs):
        q.require(spec.unit,spec.meaning,spec.shape)
        if q.evidence_tag!=model.evidence_tag: raise ContractError("Initial state evidence mismatch")
        arrays.append(q.values.copy())
    return SimulationState(float(time_seconds),tuple(arrays),_state_schema_hash(model),graph.graph_hash,parameters.parameter_hash,model.model_id,model.version,model.model_hash,solver_config.input_interpolation,solver_config,model.evidence_tag)

def validate_state(state,model):
    expected={"state_spec_hash":_state_schema_hash(model),"graph_hash":model.graph.graph_hash,"parameter_hash":model.parameters.parameter_hash,"model_hash":model.model_hash,"model_id":model.model_id,"model_version":model.version}
    for key,value in expected.items():
        if getattr(state,key)!=value: raise ContractError(f"Incompatible state/checkpoint: {key}")
    if len(state.ordered_state_arrays)!=len(model.state_specs): raise ContractError("State registry length")
    for array,spec in zip(state.ordered_state_arrays,model.state_specs):
        if array.shape!=spec.shape: raise ContractError("State shape changed")
    if state.evidence_tag!=model.evidence_tag: raise ContractError("State evidence changed")

def advance(model,state,stimulus_segment,t_stop_seconds,solver_config,sample_times=None):
    if hasattr(model,"advance_author_discrete") and solver_config.method!="Chen2024AuthorDiscrete":
        raise ContractError("Chen2024 cannot silently switch to an adaptive solver")
    if solver_config.method=="Chen2024AuthorDiscrete":
        if not hasattr(model,"advance_author_discrete"):
            raise ContractError("Model has no authorized Chen2024 discrete step")
        return model.advance_author_discrete(state,stimulus_segment,t_stop_seconds,solver_config,sample_times)
    validate_state(state,model)
    if solver_config!=state.solver_config: raise ContractError("Solver config changed across chunks; explicit reinitialization required")
    if stimulus_segment.interpolation!=solver_config.input_interpolation: raise ContractError("Stimulus interpolation contract mismatch")
    start=state.time_seconds; stop=float(t_stop_seconds)
    if stop<=start or start<stimulus_segment.start_seconds-1e-12 or stop>stimulus_segment.stop_seconds+1e-12:
        raise ContractError("Noncontiguous or out-of-segment advance")
    if stimulus_segment.evidence_tag!=state.evidence_tag: raise ContractError("Stimulus/state evidence mismatch")
    samples=np.array([start,stop] if sample_times is None else sample_times,dtype=np.float64)
    if samples.ndim!=1 or not len(samples) or not np.all(np.isfinite(samples)) or np.any(np.diff(samples)<=0) or samples[0]<start or samples[-1]>stop:
        raise ContractError("Output sample times must be sorted within advance")
    vector=np.concatenate(state.ordered_state_arrays)
    endpoints=[start]+[t for t in stimulus_segment.breakpoints_seconds if start<t<stop]+[stop]
    chunks=[]; evaluations=0
    for left,right in zip(endpoints,endpoints[1:]):
        def rhs(t,y):
            # At a ZOH break the departing interval uses its left-limit input.
            t_input=min(t,np.nextafter(right,left)) if stimulus_segment.interpolation=="zero_order_hold" else t
            derivative = model.rhs(t,y,stimulus_segment.at(t_input),model.parameters,model.graph)
            if isinstance(derivative, StateDerivativeVector):
                return derivative.flatten(model.state_specs, state.evidence_tag)
            # Compatibility for a homogeneous, explicitly declared mV kernel.
            if not all(spec.unit == "mV" for spec in model.state_specs):
                raise ContractError("Mixed state units require StateDerivativeVector")
            if derivative.evidence_tag != state.evidence_tag:
                raise ContractError("Derivative lost evidence tag")
            return derivative.require("mV/s","generator_derivative",vector.shape)
        solution=solve_ivp(rhs,(left,right),vector,method=solver_config.method,rtol=solver_config.rtol,atol=solver_config.atol,max_step=solver_config.max_step,dense_output=True)
        if not solution.success: raise ContractError("Joint solver failed: "+solution.message)
        evaluations+=solution.nfev
        selected=samples[(samples>=left)&(samples<right if right<stop else samples<=right)]
        if len(selected): chunks.append((selected,solution.sol(selected)))
        vector=solution.y[:,-1]
    times=np.concatenate([c[0] for c in chunks]); trajectory=np.concatenate([c[1] for c in chunks],axis=1)
    arrays=model.unpack(vector)
    new_state=SimulationState(stop,arrays,state.state_spec_hash,state.graph_hash,state.parameter_hash,state.model_id,state.model_version,state.model_hash,state.input_interpolation,solver_config,state.evidence_tag)
    result=model.make_result(times,trajectory,stimulus_segment.input_contract,{"solver":asdict(solver_config),"rhs_evaluations":evaluations,"sampling":"output times independent of adaptive internal steps","start_seconds":start,"stop_seconds":stop})
    return new_state,result

def checkpoint(state):
    data=asdict(state)
    data["ordered_state_arrays"]=[a.tolist() for a in state.ordered_state_arrays]
    payload={"checkpoint_version":"1.0","state":data}
    payload["payload_hash"]=canonical_hash(data)
    return json.dumps(payload,sort_keys=True,allow_nan=False)

def restore(serialized,expected_model_hashes):
    payload=json.loads(serialized)
    return restore_payload(payload,expected_model_hashes)

def restore_payload(payload,expected_model_hashes):
    """Decode the same already-parsed protocol with all original checks.

    Copy the state mapping before constructor conversion so callers retain
    their unchanged parsed document. Numeric arrays still receive the original
    independent float64 copies and SimulationState finite/domain validation.
    """
    from .contracts import SolverConfig
    if payload["checkpoint_version"]!="1.0" or canonical_hash(payload["state"])!=payload["payload_hash"]:
        raise ContractError("Checkpoint payload corrupted")
    data=dict(payload["state"])
    for key in ("state_spec_hash","graph_hash","parameter_hash","model_hash","model_id","model_version"):
        if key not in expected_model_hashes or data[key]!=expected_model_hashes[key]:
            raise ContractError(f"Incompatible checkpoint: {key}")
    data["solver_config"]=SolverConfig(**data["solver_config"])
    data["ordered_state_arrays"]=tuple(np.array(a,dtype=float) for a in data["ordered_state_arrays"])
    return SimulationState(**data)
