"""Ordered FP64 CSR/contact reductions for structured Radau CUDA workspaces.

Import is inert. Construction uses the existing strict NVRTC runtime. Each
thread owns one output row and stage, preserving source order without atomics.
"""
from dataclasses import dataclass
import numpy as np
from .contracts import ContractError

SOURCE = r'''
extern "C" __global__ void radau_csr(
    const double* x, const double* weights, const long long* indices,
    const long long* pointers, const long long* shape, double* out) {
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long rows=shape[0], cols=shape[1], channels=shape[2];
    if(i>=rows*channels) return;
    const long long row=i%rows, channel=i/rows;
    double total=0.0;
    for(long long j=pointers[row];j<pointers[row+1];++j) {
        total=__dadd_rn(total,__dmul_rn(weights[j],x[channel*cols+indices[j]]));
    }
    out[i]=total;
}
extern "C" __global__ void radau_pool(
    const double* x, const long long* indices, const long long* pointers,
    const long long* shape, double* out) {
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long rows=shape[0], cols=shape[1], channels=shape[2];
    if(i>=rows*channels) return;
    const long long row=i%rows, channel=i/rows;
    double total=0.0;
    for(long long j=pointers[row];j<pointers[row+1];++j) {
        total=__dadd_rn(total,x[channel*cols+indices[j]]);
    }
    out[i]=total;
}
'''


def stable_contact_rows(post,nrows):
    """Inverse rows retain the original source contact summation order."""
    post=np.asarray(post)
    if post.dtype.kind not in 'iu' or post.ndim!=1 or type(nrows) is not int or nrows<=0 or np.any(post<0) or np.any(post>=nrows):
        raise ContractError('Original contact rows outside the declared population')
    indices=np.argsort(post,kind='stable').astype(np.int64)
    pointers=np.concatenate(([0],np.cumsum(np.bincount(post,minlength=nrows)))).astype(np.int64)
    return indices,pointers


@dataclass(frozen=True)
class OrderedCSR:
    shape: tuple
    nnz: int
    weights: object
    indices: object
    pointers: object


class OrderedDeviceReductions:
    def __init__(self,torch,device):
        from .group_gpu_runtime import CudaModule
        self.torch=torch;self.device=torch.device(device)
        self.module=CudaModule(torch,SOURCE,['radau_csr','radau_pool'])
        if self.device.type!='cuda' or self.device.index not in (None,self.module.device):
            self.module.close()
            raise ContractError('Ordered Radau reduction device differs from current CUDA device')
        self.device=torch.device('cuda',self.module.device)
        self._shapes={};self._contacts={}

    def _tensor(self,value,dtype):
        return self.torch.as_tensor(np.array(value,copy=True),dtype=dtype,device=self.device)

    def csr(self,matrix):
        csr=matrix.tocsr()
        return OrderedCSR(tuple(csr.shape),int(csr.nnz),self._tensor(csr.data,self.torch.float64),
                          self._tensor(csr.indices,self.torch.int64),self._tensor(csr.indptr,self.torch.int64))

    def _prepare(self,x,rows,cols):
        if x.dtype!=self.torch.float64 or x.device!=self.device or x.requires_grad or x.ndim<1 or x.shape[-1]!=cols:
            raise ContractError('Ordered Radau reduction requires original FP64 rows on its CUDA device')
        channels=int(np.prod(x.shape[:-1]));key=rows,cols,channels
        if key not in self._shapes:self._shapes[key]=self._tensor(key,self.torch.int64)
        out=self.torch.empty((*x.shape[:-1],rows),dtype=self.torch.float64,device=self.device)
        return x.contiguous(),self._shapes[key],out

    def matvec(self,matrix,x):
        values,shape,out=self._prepare(x,*matrix.shape)
        self.module.launch('radau_csr',out.numel(),[values,matrix.weights,matrix.indices,matrix.pointers,shape,out])
        return out

    def pool(self,name,post,nrows,x):
        if name not in self._contacts:
            indices,pointers=stable_contact_rows(post,nrows)
            self._contacts[name]=(self._tensor(indices,self.torch.int64),self._tensor(pointers,self.torch.int64))
        indices,pointers=self._contacts[name]
        values,shape,out=self._prepare(x,nrows,len(post))
        self.module.launch('radau_pool',out.numel(),[values,indices,pointers,shape,out])
        return out

    @property
    def identity(self):return dict(self.module.identity)

    def close(self):self.module.close()
