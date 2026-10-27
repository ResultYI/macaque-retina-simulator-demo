"""Opt-in large native pointwise kernels; original sparse plans and CPU inner.

No import compiles or initializes CUDA. Three installed Torch Jiterator kernels
replace only temporary pointwise tensors at the same native RHS call. Explicit
round-to-nearest intrinsics prevent contraction. All original domain/finiteness
checks remain, before publishing a derivative. No global operator registration.
"""
from pathlib import Path
import hashlib

from .contracts import ContractError,canonical_hash
from .vh2005 import SOURCE_VALUES


def _kernel(name,inputs,scalars,outputs,body):
    args=['T '+x for x in inputs]+['T '+x for x in scalars]+['T& '+x for x in outputs]
    return 'template <typename T> void '+name+'('+','.join(args)+'){'+body+'}'


class NativePointwise:
    def __init__(self,native):
        from torch.cuda.jiterator import _create_multi_output_jit_fn
        if dict(native.p)!=SOURCE_VALUES:raise ContractError('Unchanged complete native source parameters required')
        self.__dict__.update(native.__dict__)
        p=self.p
        # ATen CUDA Tensor/scalar division multiplies the CPU-rounded reciprocal.
        # Tensor/tensor division and sigmoid reciprocal retain __ddiv_rn.
        photo=dict(ir=1/p['tau_R'],ie=1/p['tau_E'],ac=p['a_C'],nc=p['n_C'],
                   cb=p['c_beta'],kb=p['k_beta'],ic=1/p['tau_C'])
        membrane=dict(ivi=1/p['V_I'],mu=p['mu'],vk=p['V_k'],ivn=1/p['V_n'],
                      ais=p['a_is'],gamma=p['gamma'],im=1/p['tau_m'],
                      iis=1/p['tau_is'],ia=1/p['tau_a'],gt=p['g_t'],i1=1/p['tau_1'])
        hc=dict(ivi=1/p['V_I'],mu=p['mu'],t2=p['tau_2'],th=p['tau_h'])
        photo_code=_kernel('resident_native_photo',['R','E','X','C','light'],photo,
            ['dR','dE','dX','dC'],
            'T alpha=__ddiv_rn(T(1),__dadd_rn(T(1),pow(__dmul_rn(ac,C),nc)));'
            'T beta=__dadd_rn(cb,__dmul_rn(kb,E));'
            'dR=__dmul_rn(__dmul_rn(__dsub_rn(light,R),ir),T(1000));'
            'dE=__dmul_rn(__dmul_rn(__dsub_rn(R,E),ie),T(1000));'
            'dX=__dmul_rn(__dsub_rn(alpha,__dmul_rn(beta,X)),T(1000));'
            'dC=__dmul_rn(__dmul_rn(__dsub_rn(X,C),ic),T(1000));')
        membrane_code=_kernel('resident_native_membrane',['V','gi','Vbar','z1','X','Bh'],membrane,
            ['dV','dgi','dVbar','dz1','release'],
            'T arg=__dmul_rn(__dsub_rn(__dsub_rn(V,Bh),vk),ivn);'
            'T q=__ddiv_rn(T(1),__dadd_rn(T(1),exp(-arg)));'
            'T adaptation=pow(__dmul_rn(Vbar,ivi),mu);release=__ddiv_rn(q,adaptation);'
            'dV=__dmul_rn(__dmul_rn(__dsub_rn(__ddiv_rn(X,gi),V),im),T(1000));'
            'dgi=__dmul_rn(__dmul_rn(__dsub_rn(__dmul_rn(ais,pow(V,gamma)),gi),iis),T(1000));'
            'dVbar=__dmul_rn(__dmul_rn(__dsub_rn(V,Vbar),ia),T(1000));'
            'dz1=__dmul_rn(__dmul_rn(__dsub_rn(__dmul_rn(gt,release),z1),i1),T(1000));')
        hc_code=_kernel('resident_native_hc',['z2','h','AVbar','Az1'],hc,['dz2','dh'],
            'T pooled=pow(__dmul_rn(AVbar,ivi),mu);'
            'dz2=__dmul_rn(__ddiv_rn(__dsub_rn(Az1,z2),__dmul_rn(t2,pooled)),T(1000));'
            'dh=__dmul_rn(__ddiv_rn(__dsub_rn(z2,h),__dmul_rn(th,pooled)),T(1000));')
        self.cuda_sources=[photo_code,membrane_code,hc_code]
        self.photo=_create_multi_output_jit_fn(photo_code,4,**photo)
        self.membrane=_create_multi_output_jit_fn(membrane_code,5,**membrane)
        self.hc=_create_multi_output_jit_fn(hc_code,2,**hc)
        self._specialization=(dict(photo),dict(membrane),dict(hc))

    def __call__(self,y,light):
        torch=self.torch
        for fn,values,source in zip((self.photo,self.membrane,self.hc),self._specialization,self.cuda_sources):
            if fn.kwargs_dict!=values or fn.code_string!=source:
                raise ContractError('Native immutable pointwise specialization changed')
        if y.dtype!=torch.float64 or light.dtype!=torch.float64 or y.shape!=(sum(self.sizes),) or light.shape!=(self.nc,):
            raise ContractError('CUDA native state/input layout differs')
        R,E,X,C,V,gi,Vbar,z1,z2,h=y.split(self.sizes)
        invalid=(~torch.isfinite(y)).any() | (~torch.isfinite(light)).any() | (light<0).any()
        invalid=invalid | (gi<=0).any() | (Vbar<=0).any() | (V<0).any() | (X<0).any() | (C<0).any()
        if invalid.item():raise ContractError('Native finite/positive/nonnegative domain violated; no CUDA clipping')
        Bh=torch.mv(self.B,h);AVbar=torch.mv(self.A,Vbar);Az1=torch.mv(self.A,z1)
        dR,dE,dX,dC=self.photo(R,E,X,C,light)
        dV,dgi,dVbar,dz1,release=self.membrane(V,gi,Vbar,z1,X,Bh)
        dz2,dh=self.hc(z2,h,AVbar,Az1)
        derivative=torch.cat((dR,dE,dX,dC,dV,dgi,dVbar,dz1,dz2,dh))
        if not (torch.isfinite(derivative).all() & torch.isfinite(release).all()).item():
            raise ContractError('Nonfinite native CUDA derivative/release')
        return derivative,release


def make_candidate(reference):
    from .resident_cusparse import make_candidate as original_candidate,_sha
    runner=original_candidate(reference)
    native=NativePointwise(runner.partition.native)
    runner.partition.native=native
    identity=dict(policy='large_native_pointwise_three_kernels_v1',
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=[hashlib.sha256(s.encode()).hexdigest() for s in native.cuda_sources],
        torch_version=str(runner.torch.__version__),fp64=True,
        nvrtc_DLL_sha256={p.name:_sha(p) for p in sorted((Path(runner.torch.__file__).parent/'lib').glob('nvrtc*.dll')) if '.alt.' not in p.name},
        round_to_nearest_intrinsics=True,checks='All original input/output predicates',
        sparse='Unchanged explicit CSR ALG2',inner='Unchanged CPU Numba')
    runner.model.model_hash=canonical_hash(dict(previous_model_hash=runner.model.model_hash,native_pointwise=identity))
    runner.model.compute_metadata=dict(runner.model.compute_metadata,native_pointwise=identity)
    runner.native_pointwise_policy=identity
    return runner
