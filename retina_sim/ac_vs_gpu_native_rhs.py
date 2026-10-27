"""Optional within-call removal of unused native analytic readouts; no state cache."""
from hashlib import sha256
from pathlib import Path

from .ac_vs_gpu import BoundedNativeGPU,bind_gpu
from .cone_hc_reduced_native import NativeEvaluation
from .contracts import canonical_hash
from .vh2005 import SOURCE_VALUES


class RequiredNativeGPU(BoundedNativeGPU):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.identity.update(required_native_outputs=sha256(Path(__file__).read_bytes()).hexdigest())
        self.source_version=canonical_hash(self.identity)

    def _required_rhs(self,y,t,anchor):
        a,b,outputs=self._coefficients(y,t,anchor)
        derivative=self._cat(tuple(ai-bi*xi for ai,bi,xi in zip(a,b,self.split(y))))
        self._require_finite(derivative,outputs['release'])
        return derivative,outputs

    def rhs(self,y,t,anchor):
        return self._required_rhs(y,t,anchor)[0]

    def evaluate(self,y,t,anchor):
        derivative,outputs=self._required_rhs(y,t,anchor)
        R,E=outputs['R'],outputs['E']
        analytic=dict(R=1000.*((anchor.light-R)/SOURCE_VALUES['tau_R']),
                      E=1000.*((R-E)/SOURCE_VALUES['tau_E']))
        return NativeEvaluation(derivative,outputs,analytic)


def bind_required_native(assembly):
    selected=bind_gpu(assembly)
    selected.runtime_factories=dict(selected.runtime_factories,native=RequiredNativeGPU)
    return selected
