"""Opt-in native FP64 CSR ALG2 plans; no imports initialize CUDA.

Same sorted CSR tensors and original CudaNative equations. Only torch.mv(A,x)
and torch.mv(B,x) dispatch to persistent cuSPARSE descriptors/workspaces with
the explicit non-transpose CSR_ALG2 policy. No global Torch/solver flags change.
ABI constants/signatures checked against installed CUDA 13.0 cusparse.h.
This remains an unvalidated candidate until the original complete QA succeeds.
"""
from copy import copy
from pathlib import Path
from threading import RLock
import ctypes as C
import hashlib

POLICY='cusparse_csr_alg2_nontranspose_fp64_v1'
INDEX_64=3;CUDA_R_64F=1;CSR_ALG2=3;NON_TRANSPOSE=0


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
    return h.hexdigest()


class CusparseAPI:
    def __init__(self,torch):
        paths=list((Path(torch.__file__).parent/'lib').glob('cusparse64_*.dll'))
        if len(paths)!=1:raise RuntimeError('Exactly one existing Torch cuSPARSE DLL required; no fallback')
        self.path=paths[0];self.sha256=_sha(self.path);self.lib=C.CDLL(str(self.path))
        P=C.c_void_p;I=C.c_int;L=C.c_int64
        common=[P,I,P,P,P,P,P,I,I]
        signatures={
            'cusparseCreate':[C.POINTER(P)],'cusparseDestroy':[P],
            'cusparseGetVersion':[P,C.POINTER(I)],'cusparseSetStream':[P,P],
            'cusparseSetPointerMode':[P,I],
            'cusparseCreateCsr':[C.POINTER(P),L,L,L,P,P,P,I,I,I,I],
            'cusparseDestroySpMat':[P],
            'cusparseCreateDnVec':[C.POINTER(P),L,P,I],
            'cusparseDestroyDnVec':[P],'cusparseDnVecSetValues':[P,P],
            'cusparseSpMV_bufferSize':common+[C.POINTER(C.c_size_t)],
            'cusparseSpMV_preprocess':common+[P],'cusparseSpMV':common+[P],
        }
        for name,args in signatures.items():
            function=getattr(self.lib,name);function.argtypes=args;function.restype=I
        self.lib.cusparseGetErrorString.argtypes=[I];self.lib.cusparseGetErrorString.restype=C.c_char_p

    def call(self,name,*args):
        result=getattr(self.lib,name)(*args)
        if result:
            message=self.lib.cusparseGetErrorString(result)
            raise RuntimeError('%s status=%d: %s'%(name,result,message.decode() if message else 'unknown'))


class NativeCSRPlan:
    """Single-stream, serialized, immutable topology with independently owned outputs."""
    def __init__(self,matrix,api,torch):
        if (matrix.layout!=torch.sparse_csr or matrix.dtype!=torch.float64 or
                matrix.device.type!='cuda' or matrix.ndim!=2 or
                matrix.crow_indices().dtype!=torch.int64 or matrix.col_indices().dtype!=torch.int64):
            raise ValueError('Native plan requires original CUDA float64/int64 CSR tensors')
        self.matrix,self.api,self.torch=matrix,api,torch
        self.device,self.dtype,self.shape=matrix.device,matrix.dtype,matrix.shape
        self.lock=RLock();self.handle=C.c_void_p();self.a=C.c_void_p();self.x=C.c_void_p();self.y=C.c_void_p()
        self.buffer=None;self.buffer_bytes=0;self.preprocessed=False;self.closed=False
        self.alpha,self.beta=C.c_double(1.),C.c_double(0.)
        self.stream=int(torch.cuda.current_stream(self.device).cuda_stream)
        try:
            api.call('cusparseCreate',C.byref(self.handle))
            api.call('cusparseSetPointerMode',self.handle,0)
            api.call('cusparseSetStream',self.handle,C.c_void_p(self.stream))
            api.call('cusparseCreateCsr',C.byref(self.a),int(self.shape[0]),int(self.shape[1]),matrix._nnz(),
                C.c_void_p(matrix.crow_indices().data_ptr()),C.c_void_p(matrix.col_indices().data_ptr()),
                C.c_void_p(matrix.values().data_ptr()),INDEX_64,INDEX_64,0,CUDA_R_64F)
            version=C.c_int();api.call('cusparseGetVersion',self.handle,C.byref(version));self.version=version.value
        except Exception:
            self.close();raise

    @classmethod
    def __torch_function__(cls,func,types,args=(),kwargs=None):
        if len(args)==2 and isinstance(args[0],cls) and func is args[0].torch.mv and not kwargs:
            return args[0].mv(args[1])
        return NotImplemented

    def _arguments(self):
        return [self.handle,NON_TRANSPOSE,C.byref(self.alpha),self.a,self.x,C.byref(self.beta),self.y,CUDA_R_64F,CSR_ALG2]

    def mv(self,vector):
        torch=self.torch
        if (self.closed or vector.dtype!=torch.float64 or vector.device!=self.device or
                tuple(vector.shape)!=(int(self.shape[1]),) or not vector.is_contiguous() or vector.requires_grad):
            raise ValueError('Native planned SpMV vector/device/layout differs')
        if int(torch.cuda.current_stream(self.device).cuda_stream)!=self.stream:
            raise RuntimeError('Native CSR plan is bound to one CUDA stream; no silent cross-stream reuse')
        with self.lock:
            output=torch.zeros(int(self.shape[0]),dtype=torch.float64,device=self.device)
            if not self.x.value:
                self.api.call('cusparseCreateDnVec',C.byref(self.x),vector.numel(),C.c_void_p(vector.data_ptr()),CUDA_R_64F)
                self.api.call('cusparseCreateDnVec',C.byref(self.y),output.numel(),C.c_void_p(output.data_ptr()),CUDA_R_64F)
                size=C.c_size_t()
                self.api.call('cusparseSpMV_bufferSize',*self._arguments(),C.byref(size))
                self.buffer_bytes=size.value
                self.buffer=torch.empty(max(1,size.value),dtype=torch.uint8,device=self.device)
                self.api.call('cusparseSpMV_preprocess',*self._arguments(),C.c_void_p(self.buffer.data_ptr()))
                self.preprocessed=True
            else:
                self.api.call('cusparseDnVecSetValues',self.x,C.c_void_p(vector.data_ptr()))
                self.api.call('cusparseDnVecSetValues',self.y,C.c_void_p(output.data_ptr()))
            self.api.call('cusparseSpMV',*self._arguments(),C.c_void_p(self.buffer.data_ptr()))
            return output

    def close(self):
        if getattr(self,'closed',True):return
        self.closed=True
        for name,field in [('cusparseDestroyDnVec','x'),('cusparseDestroyDnVec','y'),
                           ('cusparseDestroySpMat','a'),('cusparseDestroy','handle')]:
            pointer=getattr(self,field,None)
            if pointer is not None and pointer.value:
                try:self.api.call(name,pointer)
                except Exception:pass
                pointer.value=None

    def __del__(self):
        self.close()


def make_candidate(reference, *, numerical_contract=None, purpose='numerical_qa'):
    from .resident_runner import ResidentNativeRunner
    from .contracts import canonical_hash
    runner=ResidentNativeRunner.from_runner(reference,purpose=purpose,numerical_contract=numerical_contract)
    native=copy(runner.partition.native);api=CusparseAPI(runner.torch)
    native.A=NativeCSRPlan(native.A,api,runner.torch);native.B=NativeCSRPlan(native.B,api,runner.torch)
    runner.partition.native=native
    policy=dict(policy=POLICY,source_sha256=_sha(__file__),cusparse_DLL_sha256=api.sha256,
        cusparse_version=native.A.version,torch_version=str(runner.torch.__version__),dtype='float64',
        graph_weights_and_indices='Original CudaNative sorted CSR tensors; unchanged',
        alpha=1.,beta=0.,operation='non_transpose',algorithm=CSR_ALG2,
        DOP853_and_CPU_inner_reductions='unchanged',global_flags_modified=False)
    runner.model.model_hash=canonical_hash(dict(original_resident_model_hash=runner.model.model_hash,native_sparse_plan=policy))
    runner.model.compute_metadata=dict(runner.model.compute_metadata,native_sparse_plan=policy)
    runner.native_sparse_plan_policy=policy
    return runner


def check_repeatability(runner,state,channels):
    """Three pure native evaluations, no solver/evolution/RNG or model timing."""
    torch=runner.torch;part=runner.partition;native=part.native
    light=part._light(channels[part.native_external.channel].at(state.time_seconds))
    y=state.y[:part.offset];first=None
    with torch.no_grad():
        for _ in range(3):
            rhs,release=native(y,light);bh=torch.mv(native.B,y.split(native.sizes)[9])
            now=(rhs,release,bh)
            if first is None:first=tuple(x.clone() for x in now)
            elif any(not torch.equal(a,b) for a,b in zip(first,now)):
                raise RuntimeError('Explicit native CSR ALG2 repeatability preflight failed')
    return dict(repeats=3,exact=True,pure_native_operator_only=True,
                A_workspace_bytes=native.A.buffer_bytes,B_workspace_bytes=native.B.buffer_bytes)
