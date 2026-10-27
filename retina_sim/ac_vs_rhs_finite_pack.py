"""Opt-in removal of a second finite scan of an already checked RHS result."""
from pathlib import Path
import hashlib
import numpy as np
from .contracts import ContractError,canonical_hash
from .group_inner_dynamics import GroupInnerDynamics
from .ac_vs_model import DirectACVSRHS
from .ac_vs_rhs_readouts import RequiredOutputsACVSRHS,contract as readout_contract,install as install_readouts

KIND='ac_vs_rhs_finite_pack_v1'


def contract():
    return dict(kind=KIND,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_backend=readout_contract())


class FinitePackedACVSRHS(RequiredOutputsACVSRHS):
    def __init__(self,model,*,repo_root=None):
        super().__init__(model,repo_root=repo_root)
        if type(model).validate is not GroupInnerDynamics.validate or type(model).pack is not GroupInnerDynamics.pack:
            raise ContractError('Finite-pack adapter requires unchanged layout/finite validation and packing')
        self.identity=canonical_hash(dict(parent=self.identity,backend=contract()))
    def rhs_flat(self,t,y,cone_release=None,rod_current=None):
        state=self.model.unpack(y,copy=False)
        # The original equation body checks every returned RHS array for finiteness.
        result=GroupInnerDynamics.rhs(self.view,t,state,cone_release,rod_current)
        if set(result)!=set(self.model.state_keys):raise ContractError('Complete unchanged inner state layout required')
        for key in self.model.state_keys:
            value=np.asarray(result[key])
            if value.dtype!=np.float64 or value.shape!=(self.model.sizes[key],):
                raise ContractError('Finite original-layout FP64 inner state required: '+str(key))
        return np.concatenate([result[key] for key in self.model.state_keys])


def install(assembly,requested):
    if requested!=contract():raise ContractError('Explicit finite-pack implementation identity differs')
    if assembly.runtime_factories.get('compiled') is DirectACVSRHS:
        install_readouts(assembly,readout_contract())
    if assembly.runtime_factories.get('compiled') is not RequiredOutputsACVSRHS:
        raise ContractError('Finite-pack adapter requires the registered CPU readout backend')
    assembly.runtime_factories=dict(assembly.runtime_factories,compiled=FinitePackedACVSRHS)
    return dict(requested,scientific_artifact=assembly.artifact_sha256,no_cross_call_state_cache=True,
        finite_equation_result_checked_once=True,layout_dtype_checks_preserved=True,returned_array_independently_owned=True)
