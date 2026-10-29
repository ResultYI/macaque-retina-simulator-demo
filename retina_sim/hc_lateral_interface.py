"""Research-only eta1 HC -> unchanged inner-network initialization interface.

The production initializer's exact function code is reused with private, explicit
native/binding dependencies. Its module globals and file are never patched.
This avoids copying the confirmed BC baseline/history and rod/AC/RGC algebra.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType,SimpleNamespace
import hashlib
import numpy as np

from .contracts import ContractError,canonical_hash
from .h1_h2_effective import H1H2GroupRunner,MECHANISM
from .hc_lateral_candidate import LateralHCCandidate
from .group_time_runner import GroupTimeArm
from .h1_h2_background import initialize_joint_background

MODEL='HC_lateral_eta1_research_interface_v1'
INITIALIZATION='HC_lateral_eta1_fixed_reference_inner_background_v1'

@dataclass(frozen=True)
class InterfaceBinding:
    identity: str
    descriptor: dict

def _digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def context(base,seed,configuration,repo_root):
    """Explicitly opt in to the already tested native provider; no registry default."""
    root=Path(repo_root);cfg=deepcopy(configuration);check=dict(cfg)
    identity=check.pop('candidate_sha256')
    if canonical_hash(check)!=identity or cfg['candidate_id']!='HC_homotypic_lateral_only_v2':
        raise ContractError('Intact selected eta1 candidate configuration required')
    if cfg['parameters']!={'effective_homotypic_rate_per_second':50.,'S_efficacy_ratio':1.}:
        raise ContractError('Only the selected eta1 research candidate is supported')
    if cfg['provider_sha256']!=_digest(root/cfg['provider']):
        raise ContractError('Selected provider changed after its saved validation')
    def native_factory(assembly,*,torch=None,device=None,plans=None,bound=True):
        if torch is not None or device is not None or plans is not None:
            raise ContractError('Research interface supports isolated CPU reference only')
        provider=LateralHCCandidate(base,coupling_rate_per_second=50.,S_efficacy_ratio=1.)
        if provider.source_version!=cfg['selected_native_execution']:
            raise ContractError('Actual provider does not match the selected native execution')
        if assembly is not None and assembly.artifact_sha256!=provider.assembly.artifact_sha256:
            raise ContractError('Independent candidate assembly identity differs')
        if bound:provider.boundary=deepcopy(seed['h1_h2_model']['boundary'])
        return provider
    provider=native_factory(None,bound=False)
    assembly=provider.assembly
    assembly.runtime_factories=dict(base.runtime_factories,native=native_factory)
    assembly.lateral_configuration=cfg
    assembly.lateral_native_factory=native_factory
    assembly.native_boundary=deepcopy(seed['h1_h2_model']['boundary'])
    return assembly

def initialize_from_native(base,old_seed,configuration,native_packed,repo_root,protocol,*,guard=lambda:None):
    """Create a distinct complete initial state using unchanged confirmed algebra.

    The callback binding is explicitly an interface identity, not a newly
    admitted ParameterResolver bundle or a physiological calibration.
    """
    assembly=context(base,old_seed,configuration,repo_root)
    root=Path(repo_root);seed=deepcopy(old_seed);parent_hash=canonical_hash(old_seed)
    parameters_hash=canonical_hash(old_seed['inner_parameters'])
    native=assembly.lateral_native_factory(assembly,bound=False)
    native.boundary=deepcopy(seed['h1_h2_model']['boundary'])
    bg=seed['states']['cones','analytic_RE','R']
    previous,_=native.construction_point(SimpleNamespace(states=seed['states'],time_seconds=seed['time_seconds']),bg,seed['time_seconds']+1.,0)
    y=np.asarray(native_packed)
    if y.dtype!=np.float64 or y.shape!=previous.shape or not np.isfinite(y).all():
        raise ContractError('Actual saved candidate native FP64 equilibrium required')
    for name in ('X','C','Vis','g_i','Vbar'):
        if not np.array_equal(y[native.slices[name]],previous[native.slices[name]]):
            raise ContractError('Native handoff changed original cone phototransduction: '+name)
    for name in ('w','z2'):seed['states']['HC',MECHANISM,name]=y[native.slices[name]].copy()
    h=y[native.slices['Vh']]
    seed['states']['HC',MECHANISM,'h1']=h[native.h1_rows].copy()
    seed['states']['HC',MECHANISM,'h2']=h[native.h2_rows].copy()
    seed['states']['cones','selected_original_z1_auxiliary','z1_aux']=y[native.slices['z1_aux']].copy()
    seed['observation_prehistory'].update(history_times=np.array([seed['time_seconds']]),
        history_values=h[None,:].copy(),prehistory=h.copy(),origin_seconds=seed['time_seconds'])
    old_meta=seed['h1_h2_model']
    seed['h1_h2_model']=dict(model=MODEL,qualification='RESEARCH_INTERFACE_ONLY',
        artifact=assembly.artifact_sha256,layout_sha256=canonical_hash(assembly.layout),
        parameters=dict(assembly.h2_parameters),boundary=deepcopy(old_meta['boundary']),
        native_configuration_sha256=configuration['candidate_sha256'],
        coupled_outer_background_qualified=False,whole_retina_background_qualified=False,
        physiological_calibration=False,FIT_READY=False)
    seed['lateral_hc_interface']=dict(schema=MODEL,parent_seed_sha256=parent_hash,
        parent_initialized_model=configuration['parent_initialized_model'],
        parent_parameter_binding=configuration['parent_binding'],
        configuration_sha256=configuration['candidate_sha256'],
        native_execution=configuration['selected_native_execution'],
        native_packed_sha256=canonical_hash(y),parameter_resolver_admitted=False,
        production_adopted=False,HC_stage_complete=False,scientific_review='PENDING_CHAT_REVIEW')
    def make_binding(a,s,root_path,p):
        if canonical_hash(s['inner_parameters'])!=parameters_hash:
            raise ContractError('Background construction changed frozen physiological parameters')
        descriptor=dict(schema='isolated-HC-inner-interface-binding-1',
            reference_parameter_binding=configuration['parent_binding'],
            frozen_inner_parameters_sha256=parameters_hash,
            candidate_configuration_sha256=configuration['candidate_sha256'],
            native_execution=configuration['selected_native_execution'],artifact=a.artifact_sha256,
            states_sha256=canonical_hash(s['states']),spike_prehistory_sha256=canonical_hash(s['spike_states']),
            held_reservoir_sha256=canonical_hash(dict(native=s['h1_h2_model']['boundary'],inner=a.boundary_runtime)),
            binding_kind='Explicit research interface composition; not global ParameterResolver admission',
            production_use_allowed=False,FIT_READY=False)
        identity=canonical_hash(descriptor)
        a.parameter_binding_id=identity;s['h1_h2_model']['parameter_binding_id']=identity
        s['lateral_hc_interface']['binding_id']=identity
        return InterfaceBinding(identity,descriptor)
    # Reuse the actual trusted function code. Only its already-required native
    # provider, binding callback and output namespace differ; no global mutation.
    private_globals=dict(initialize_joint_background.__globals__)
    private_globals.update(H1H2Native=lambda a:assembly.lateral_native_factory(a,bound=False),
        bind_parameters=make_binding,MODEL=MODEL,INITIALIZATION=INITIALIZATION)
    initializer=FunctionType(initialize_joint_background.__code__,private_globals,
        'initialize_lateral_candidate_inner',initialize_joint_background.__defaults__,initialize_joint_background.__closure__)
    initializer.__kwdefaults__=dict(initialize_joint_background.__kwdefaults__ or {})
    result,binding=initializer(assembly,seed,root,protocol,guard=guard)
    meta=result['joint_background_initialization'];meta.pop('initialized_model_id')
    meta.update(native_candidate_configuration=configuration['candidate_sha256'],
        native_execution=configuration['selected_native_execution'],
        parameter_binding_kind=binding.descriptor['binding_kind'],
        initializer_reuse=dict(source='retina_sim/h1_h2_background.py::initialize_joint_background',
            source_sha256=_digest(root/'retina_sim/h1_h2_background.py'),
            exact_same_function_code=initializer.__code__ is initialize_joint_background.__code__,
            private_dependencies_only=True,original_module_globals_modified=False,
            adapter_sha256=_digest(Path(__file__))),
        HC_stage_complete=False,scientific_review='PENDING_CHAT_REVIEW',
        fullchain_checkpoint_supported=False)
    meta['initialized_model_id']=canonical_hash(meta)
    result['lateral_hc_interface']['initialized_model_id']=meta['initialized_model_id']
    if canonical_hash(old_seed)!=parent_hash:raise ContractError('Old seed changed during independent construction')
    return assembly,result,binding

class LateralHCResearchRunner(H1H2GroupRunner):
    """Explicit CPU reference adapter; no production registration or checkpoint."""
    def __init__(self,assembly,seed,protocol,*,mode='reference',**kwargs):
        cfg=assembly.lateral_configuration;meta=seed.get('lateral_hc_interface',{})
        if mode!='reference' or kwargs.get('torch') is not None or kwargs.get('shared_plans') is not None:
            raise ContractError('Lateral research adapter supports CPU reference only')
        if (seed.get('h1_h2_model',{}).get('model')!=MODEL or meta.get('configuration_sha256')!=cfg['candidate_sha256']
            or meta.get('native_execution')!=cfg['selected_native_execution']
            or meta.get('binding_id')!=getattr(assembly,'parameter_binding_id',None)):
            raise ContractError('Independent candidate model/configuration/interface binding required')
        if canonical_hash(seed['inner_parameters'])!=seed['joint_background_initialization']['parameters_sha256']:
            raise ContractError('Frozen inner values changed')
        GroupTimeArm.__init__(self,assembly,seed,protocol,mode=mode,**kwargs)
        if self.native_physics.source_version!=cfg['selected_native_execution']:
            self.close();raise ContractError('Actual runner did not instantiate the selected lateral provider')
        self.identity=canonical_hash(dict(research_execution=self.identity,
            candidate_configuration=cfg['candidate_sha256'],interface_binding=meta['binding_id'],
            implementation=_digest(Path(__file__))))
    def checkpoint_payload(self,**kwargs):
        raise ContractError('Complete lateral-candidate checkpoint schema is not implemented; legacy relabel forbidden')
    @classmethod
    def from_checkpoint(cls,*args,**kwargs):
        raise ContractError('Lateral research adapter has no complete-checkpoint restore support')
