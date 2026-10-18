"""Optional FP64 CUDA execution of the unchanged continuous vH equations.

Torch is loaded only for this explicitly selected backend. The numerical device
implementation is identified separately; it does not assert bitwise CPU parity.
"""
import numpy as np
from .contracts import ContractError
from .vh2005 import SOURCE_VALUES


class CudaNative:
    def __init__(self,model):
        import torch
        if not torch.cuda.is_available():raise ContractError('Explicit CUDA backend requires the existing CUDA device')
        if dict(model.p)!=SOURCE_VALUES:raise ContractError('CUDA lowering requires the complete unchanged native source values')
        self.torch=torch;self.p=model.p;self.nc=model.nc;self.nh=model.nh
        self.sizes=[self.nc]*8+[self.nh]*2
        def upload(matrix):
            if matrix.dtype!=np.float64:raise ContractError('Native matrix must remain float64')
            # SciPy permits unsorted CSR columns; CUDA CSR requires canonical
            # ordering. Sort an execution-only copy, preserving every weight and
            # original graph array. Duplicate columns are rejected, not merged.
            stored=matrix.copy();stored.sort_indices()
            if not stored.has_canonical_format:raise ContractError('CUDA native CSR requires distinct columns in each row')
            return torch.sparse_csr_tensor(torch.as_tensor(stored.indptr.astype(np.int64),device='cuda'),
                torch.as_tensor(stored.indices.astype(np.int64),device='cuda'),
                torch.as_tensor(stored.data,device='cuda'),size=stored.shape,
                dtype=torch.float64,device='cuda',check_invariants=True)
        self.A=upload(model.A);self.B=upload(model.B)

    def __call__(self,y,light):
        torch=self.torch;p=self.p
        if y.dtype!=torch.float64 or light.dtype!=torch.float64 or y.shape!=(sum(self.sizes),) or light.shape!=(self.nc,):
            raise ContractError('CUDA native state/input layout differs')
        R,E,X,C,V,gi,Vbar,z1,z2,h=y.split(self.sizes)
        invalid=(~torch.isfinite(y)).any() | (~torch.isfinite(light)).any() | (light<0).any()
        invalid=invalid | (gi<=0).any() | (Vbar<=0).any() | (V<0).any() | (X<0).any() | (C<0).any()
        if invalid.item():raise ContractError('Native finite/positive/nonnegative domain violated; no CUDA clipping')
        current=X**p['n_X'];adaptation=(Vbar/p['V_I'])**p['mu']
        q=1./(1.+torch.exp(-((V-torch.mv(self.B,h)-p['V_k'])/p['V_n'])))
        release=q/adaptation;pooled=(torch.mv(self.A,Vbar)/p['V_I'])**p['mu']
        alpha=1./(1.+(p['a_C']*C)**p['n_C']);beta=p['c_beta']+p['k_beta']*E
        rows=((light-R)/p['tau_R'],(R-E)/p['tau_E'],alpha-beta*X,
            (current-C)/p['tau_C'],(current/gi-V)/p['tau_m'],
            (p['a_is']*V**p['gamma']-gi)/p['tau_is'],(V-Vbar)/p['tau_a'],
            (p['g_t']*release-z1)/p['tau_1'],
            (torch.mv(self.A,z1)-z2)/(p['tau_2']*pooled),
            (z2-h)/(p['tau_h']*pooled))
        derivative=1000.*torch.cat(rows)
        if not (torch.isfinite(derivative).all() & torch.isfinite(release).all()).item():
            raise ContractError('Nonfinite native CUDA derivative/release')
        return derivative,release
