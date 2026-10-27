"""Opt-in CPU RHS view: omit observation-only algebra, with no state cache."""
from pathlib import Path
import hashlib
import numpy as np
from .contracts import ContractError,canonical_hash
from .ac_vs_model import ACVSInner,DirectACVSRHS
from .group_inner_dynamics import GroupInnerDynamics
from .local_patch_boundary import BoundedInnerDynamics
from .spatial_group_model import RGC

KIND='ac_vs_rhs_required_outputs_v1'


def contract():
    return dict(kind=KIND,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


class _RHSView:
    def __init__(self,model):self.model=model
    def __getattr__(self,name):return getattr(self.model,name)
    def outputs(self,t,state,cone_release=None):
        out=BoundedInnerDynamics.outputs(self.model,t,state,cone_release)
        # Preserve parameter-domain checks formerly reached through derived rates.
        for pop in RGC:
            self.model.par(pop,'alpha',positive=True);self.model.par(pop,'beta',positive=True);self.model.par(pop,'mu')
            kernel=np.asarray(self.model.p[pop]['spike_history_kernel'])
            if kernel.shape!=(3,) or not np.isfinite(kernel).all():raise ContractError('Original three-bin RGC kernel required')
        return out


class RequiredOutputsACVSRHS(DirectACVSRHS):
    def __init__(self,model,*,repo_root=None):
        if type(model) is not ACVSInner or type(model).rhs is not GroupInnerDynamics.rhs:
            raise ContractError('RHS readout adapter requires the unchanged ACVS inherited equation body')
        super().__init__(model,repo_root=repo_root);self.view=_RHSView(model)
        self.identity=canonical_hash(dict(parent=self.identity,backend=contract()))
    def rhs_flat(self,t,y,cone_release=None,rod_current=None):
        state=self.model.unpack(y,copy=False)
        return self.model.pack(GroupInnerDynamics.rhs(self.view,t,state,cone_release,rod_current))


def install(assembly,requested):
    if requested!=contract():raise ContractError('Explicit RHS adapter implementation identity differs')
    factories=assembly.runtime_factories
    if factories.get('inner') is not ACVSInner or factories.get('compiled') is not DirectACVSRHS:
        raise ContractError('Only the existing direct CPU ACVS execution can select this adapter')
    assembly.runtime_factories=dict(factories,compiled=RequiredOutputsACVSRHS)
    return dict(requested,scientific_artifact=assembly.artifact_sha256,no_cross_call_state_cache=True,
        full_observation_outputs_preserved=True)
