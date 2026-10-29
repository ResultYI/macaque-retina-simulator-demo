"""Opt-in input override on the unchanged complete held-boundary inner system.

No gap edit, isolated-neighbor clamp, new kinetic state or production factory.
Background construction reuses the exact existing initializer function code.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType,SimpleNamespace
import hashlib
from scipy import sparse
from .contracts import ContractError,canonical_hash
from .hc_lateral_interface import context
from .h1_h2_background import initialize_joint_background

def matrix_hash(m):
    a=m.tocsr()
    return canonical_hash(dict(shape=a.shape,data=a.data,indices=a.indices,indptr=a.indptr))

def make_context(base,seed,HC_configuration,input_operator,root,*,expected_original_hash):
    assembly=context(base,seed,HC_configuration,root);parent=assembly.runtime_factories['inner']
    W=sparse.csr_matrix(input_operator,copy=True);W.sort_indices()
    def inner_factory(a,parameters):
        model=parent(a,parameters);name='cones_to_S_ON_BC'
        if matrix_hash(model.mean[name])!=expected_original_hash:raise ContractError('Actual original S_ON input operator differs')
        if W.shape!=model.mean[name].shape:raise ContractError('Candidate changes the original group layout')
        model.mean=dict(model.mean,**{name:W.copy()})
        model.identity=canonical_hash(dict(parent=model.identity,S_ON_input_operator=matrix_hash(W),
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physical_parameters_unchanged=True))
        return model
    assembly.runtime_factories=dict(assembly.runtime_factories,inner=inner_factory)
    return assembly

def initialize_case(base,parent_seed,HC_configuration,W,root,protocol,*,case,input_candidate_identity,expected_original_hash):
    root=Path(root);parent_hash=canonical_hash(parent_seed);params_hash=canonical_hash(parent_seed['inner_parameters'])
    assembly=make_context(base,parent_seed,HC_configuration,W,root,expected_original_hash=expected_original_hash)
    def binding(a,seed,root_path,p):
        if canonical_hash(seed['inner_parameters'])!=params_hash:raise ContractError('Candidate changed frozen physiological parameters')
        model=a.runtime_factories['inner'](a,seed['inner_parameters'])
        descriptor=dict(schema='S_ON-coupled-input-research-binding-1',case=case,input_candidate_identity=input_candidate_identity,
            original_input_hash=expected_original_hash,actual_input_hash=matrix_hash(W),
            parameters_sha256=params_hash,states_sha256=canonical_hash(seed['states']),
            gap_matrix_sha256=matrix_hash(model.gap),gap_offset_sha256=canonical_hash(model.gap_boundary_product),
            original_reservoirs_sha256=canonical_hash(a.boundary_runtime),
            source_inner_identity=model.identity,production_adopted=False,background_qualified=False,FIT_READY=False)
        identity=canonical_hash(descriptor)
        a.parameter_binding_id=identity;seed['h1_h2_model']['parameter_binding_id']=identity
        seed['lateral_hc_interface']['binding_id']=identity
        return SimpleNamespace(identity=identity,descriptor=descriptor)
    private=dict(initialize_joint_background.__globals__)
    private.update(H1H2Native=lambda a:a.lateral_native_factory(a,bound=False),bind_parameters=binding,
        MODEL=parent_seed['h1_h2_model']['model'],INITIALIZATION='S_ON_coupled_fixed_reference_background_v1')
    initializer=FunctionType(initialize_joint_background.__code__,private,'initialize_S_ON_input_coupled_background',
        initialize_joint_background.__defaults__,initialize_joint_background.__closure__)
    initializer.__kwdefaults__=dict(initialize_joint_background.__kwdefaults__ or {})
    seed,bound=initializer(assembly,parent_seed,root,protocol)
    meta=seed['joint_background_initialization'];meta.pop('initialized_model_id')
    meta.update(case=case,input_candidate_identity=input_candidate_identity,input_operator_sha256=matrix_hash(W),
        unchanged_complete_inner=True,gap_changed=False,new_neighbor_clamp=False,
        initializer_exact_same_function_code=initializer.__code__ is initialize_joint_background.__code__,
        original_globals_modified=False,complete_checkpoint_supported=False)
    meta['initialized_model_id']=canonical_hash(meta)
    seed['lateral_hc_interface']['initialized_model_id']=meta['initialized_model_id']
    seed['S_ON_coupled_input_research']=dict(case=case,input_candidate_identity=input_candidate_identity,
        input_operator_sha256=matrix_hash(W),binding_id=bound.identity,complete_checkpoint_supported=False,
        scientific_review='PENDING_CHAT_REVIEW',production_adopted=False,background_qualified=False,FIT_READY=False)
    if canonical_hash(parent_seed)!=parent_hash:raise ContractError('Parent seed modified')
    if canonical_hash(seed['inner_parameters'])!=params_hash:raise ContractError('Frozen baseline/parameters changed')
    return assembly,seed,bound
