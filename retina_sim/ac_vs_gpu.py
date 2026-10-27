"""Explicit V+s/CUDA binding; fixed exterior terms retain their source positions."""
from hashlib import sha256
from pathlib import Path

import numpy as np

from .ac_vs_compiled import CompiledACVSInnerRHS
from .ac_vs_model import ACVSInner, DirectACVSRHS
from .contracts import ContractError, canonical_hash
from .group_native_physics import GroupNativePhysics


def host_payload(value):
    """Lossless evidence serialization only; never modifies device state."""
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):return {k:host_payload(v) for k,v in value.items()}
    if isinstance(value, tuple):return tuple(host_payload(v) for v in value)
    if isinstance(value, list):return [host_payload(v) for v in value]
    return value


def initial_match_on_device(arm, saved):
    from tools.validate_ac_vs_equilibrium_start import initial_match
    class EvidenceView:
        def __getattr__(self, name):return getattr(arm, name)
        def initial_snapshot(self):return host_payload(arm.initial_snapshot())
        def delay_payload(self):return host_payload(arm.delay_payload())
    return initial_match(EvidenceView(), saved)


class BoundedNativeGPU(GroupNativePhysics):
    def __init__(self, assembly, *, torch=None, device=None, plans=None):
        if torch is None or torch.device(device).type != 'cuda':
            raise ContractError('Bounded native GPU requires explicit CUDA; no CPU fallback')
        self.boundary = {name: torch.as_tensor(np.array(assembly.boundary_runtime[name], copy=True),
            dtype=torch.float64, device=device) for name in ('A_Vbar', 'A_release', 'B_Vh')}
        self._pool_pointer = None
        super().__init__(assembly, torch=torch, device=device, plans=plans)
        self.identity.update(bounded_exterior=assembly.boundary_runtime['identity'],
            bounded_GPU_implementation=sha256(Path(__file__).read_bytes()).hexdigest())
        self.source_version = canonical_hash(self.identity)

    def _coefficients(self, y, t, anchor):
        # Vbar is a view of y. Release is a separately evaluated allocation.
        # Tag that operand before either pool is used in the original power.
        self._pool_pointer = self.split(y)[4].data_ptr()
        try:
            return super()._coefficients(y, t, anchor)
        finally:
            self._pool_pointer = None

    def _mv(self, matrix, value):
        result = super()._mv(matrix, value)
        if matrix is self.A:
            if self._pool_pointer is None:
                raise ContractError('Bounded GPU pool used outside coefficient evaluation')
            name = 'A_Vbar' if value.data_ptr() == self._pool_pointer else 'A_release'
            return result + self.boundary[name]
        if matrix is self.B:
            return result + self.boundary['B_Vh']
        raise ContractError('Unexpected bounded GPU native operator')


def bind_gpu(assembly):
    """Execution-only facade; graph, state, parameters and default CPU stay intact."""
    if assembly.runtime_factories.get('inner') is not ACVSInner or assembly.runtime_factories.get('compiled') is not DirectACVSRHS:
        raise ContractError('Only the frozen direct V+s assembly can select this CUDA adapter')
    selected = object.__new__(type(assembly))
    selected.__dict__.update(assembly.__dict__)
    selected.runtime_factories = dict(assembly.runtime_factories,
        compiled=CompiledACVSInnerRHS, native=BoundedNativeGPU)
    return selected
