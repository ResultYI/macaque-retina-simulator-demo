"""69#6 frozen Gaussian input, functional H1 and constrained BC joint execution."""
from copy import deepcopy
from pathlib import Path
from types import FunctionType, SimpleNamespace
import numpy as np
from scipy import sparse
from retina_sim.contracts import canonical_hash, ContractError
from retina_sim.h1_h2_effective import H1H2Native, matrix_id
from retina_sim.group_time_storage import load_bundle
from retina_sim.h1_h2_background import initialize_joint_background
from tools.gin_hc2_bc1_context_support_v2 import ROOT, OUT, load as load_historical, protocol, read

FOLDER=OUT/'joint_gray_background'


class JointNative(H1H2Native):
    """Effective H1 space replaces H1 lateral diffusion; original H2 stays independent."""
    def __init__(self, assembly, **kwargs):
        super().__init__(assembly)
        self.coupling=assembly.joint_H2_Q
        self.degree=self.coupling.diagonal()
        self.neighbor=sparse.diags(self.degree)-self.coupling
        self.identity=dict(self.identity,joint_functional=assembly.joint_configuration,
                           H2_Q=matrix_id(self.coupling),H1_explicit_coupling=False)
        self.source_version=canonical_hash(self.identity)

    def _coefficients(self,y,t,anchor):
        aa,bb,out=super()._coefficients(y,t,anchor)
        aa=list(aa);bb=list(bb)
        aa[7]=aa[7]+50.*(self.neighbor@y[self.slices['Vh']])
        bb[7]=bb[7]+50.*self.degree
        return tuple(aa),tuple(bb),out


class JointAssembly:
    def __init__(self,parent,A,boundary,Q,configuration):
        self.parent=parent;self.joint_A=A
        self.joint_H2_Q=Q;self.joint_configuration=configuration
        self.native_boundary=deepcopy(boundary)
        self.operators=dict(parent.operators,A1=A[parent.h1_rows].tocsr(),A2=A[parent.h2_rows].tocsr())
        self.artifact_sha256=canonical_hash(dict(parent=parent.artifact_sha256,A=matrix_id(A),
            B=matrix_id(parent.matrix('routes/HC_to_cones/mean_weights.npz')),boundary=boundary,
            Q=matrix_id(Q),configuration=configuration))
        self.connection_identity=dict(parent=parent.connection_identity,joint=configuration,
                                      A=matrix_id(A),H1_explicit_coupling=False,H2_Q=matrix_id(Q))
        self.runtime_factories=dict(parent.runtime_factories,native=native_factory)
        self.lateral_native_factory=native_factory
        self.lateral_configuration=dict(parent.lateral_configuration)
        self.lateral_configuration.update(candidate_id='GIN69_FUNCTIONAL_H1_CONSTRAINED_BC',
            actual_spatial_contract=configuration,transport_schema_inherited=True)
        self.lateral_configuration.pop('candidate_sha256',None)
        self.lateral_configuration['selected_native_execution']=JointNative(self).source_version
        self.lateral_configuration['candidate_sha256']=canonical_hash(self.lateral_configuration)

    def __getattr__(self,name):return getattr(self.parent,name)

    def matrix(self,name):
        if name=='routes/cones_to_HC/mean_weights.npz':return self.joint_A
        if name.startswith('h1_h2/'):
            key=Path(name).stem
            if key in self.operators:return self.operators[key]
        return self.parent.matrix(name)


def native_factory(assembly,*,bound=True,torch=None,device=None,plans=None):
    if torch is not None or device is not None or plans is not None:
        raise ContractError('Joint source-bound CPU execution only')
    model=JointNative(assembly)
    if bound:model.boundary=deepcopy(assembly.native_boundary)
    return model


def initialize_inner(assembly,seed):
    def binding(a,s,root,p):
        descriptor=dict(artifact=a.artifact_sha256,parameters=canonical_hash(s['inner_parameters']),
                        states=canonical_hash(s['states']),joint=a.joint_configuration)
        identity=canonical_hash(descriptor);a.parameter_binding_id=identity
        s['h1_h2_model']['parameter_binding_id']=identity
        return SimpleNamespace(identity=identity,descriptor=descriptor)
    private=dict(initialize_joint_background.__globals__)
    private.update(H1H2Native=lambda a:native_factory(a,bound=False),bind_parameters=binding)
    fn=FunctionType(initialize_joint_background.__code__,private,
                   'initialize_joint_functional',initialize_joint_background.__defaults__,initialize_joint_background.__closure__)
    fn.__kwdefaults__=initialize_joint_background.__kwdefaults__
    return fn(assembly,seed,ROOT,protocol())[0]


def unbound_context():
    parent,old=load_historical()
    with np.load(FOLDER/'LOCAL_OPERATORS.npz',allow_pickle=False) as z:
        boundary={key:z[key].copy() for key in ('A_release','A_Vbar','B_h')}
    A=sparse.load_npz(FOLDER/'A_local.npz');Q=sparse.load_npz(FOLDER/'H2_Q.npz')
    cfg=read(FOLDER/'MODEL_CONFIGURATION.json')
    return JointAssembly(parent,A,boundary,Q,cfg),deepcopy(old)


def load():
    a,_=unbound_context()
    seed=load_bundle(FOLDER/'seed',expected_kind='source_construction_seed')['payload']
    if seed['h1_h2_model']['artifact']!=a.artifact_sha256:
        raise ContractError('Joint initialized seed does not match actual A/B/configuration')
    a.parameter_binding_id=seed['bc_physiology']['parameter_binding_id']
    return a,seed
