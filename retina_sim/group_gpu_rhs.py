"""Million-state FP64 RHS lowered from the complete existing group typed plan.

Opt-in feasibility executor only. Every opcode/check remains. Dependencies that
cross rows form CUDA launch barriers; pointwise chains share a layer. Scatter
is inverted once into stable rows, preserving the source addition order.
"""
from bisect import bisect_right
from collections import defaultdict
from threading import Lock
import math

import numpy as np

from .contracts import ContractError, canonical_hash
from .compiled_kernel import (ADD, SUB, MUL, DIV, POW, SIGMOID, LOG, EXP,
    MINIMUM, MAXIMUM, CSR, ROW_CSR, TAKE, SCATTER, ZERO, FINITE, NONNEG, POSITIVE)
from .group_gpu_runtime import CudaModule


CHECKS = {FINITE, NONNEG, POSITIVE}
POINTWISE = {ADD, SUB, MUL, DIV, POW, SIGMOID, LOG, EXP, MINIMUM, MAXIMUM}
SENTINEL = 2147483647


def lower(compiled):
    sizes = {s.start:s.stop-s.start for s in compiled.model.slices.values()}
    sizes[compiled.nstate] = compiled._rod_count
    sizes.update({int(at):len(a) for at,a in compiled._scratch.constants})
    for code,dest,n,*_ in compiled.program:
        if int(code) not in CHECKS | {SCATTER}:
            sizes[int(dest)] = int(n)
    starts = sorted(sizes)
    def owner(at):
        j = bisect_right(starts, at)-1
        if j < 0 or at >= starts[j]+sizes[starts[j]]:
            raise ContractError('GPU lowering read outside a typed allocation')
        return starts[j]
    layers = {s:-1 for s in starts}
    written = set(s for s in starts if s < compiled.nstate+compiled._rod_count)
    written.update(int(s) for s,_ in compiled._scratch.constants)
    used = set()
    groups = defaultdict(list)
    inverse_indices, inverse_pointers = [], []
    def dependency(at, n, random=False):
        root = owner(at)
        if root not in written or at+n > root+sizes[root]:
            raise ContractError('GPU lowering reads an uninitialized or cross-allocation value')
        used.add(root)
        return layers[root]+int(random or at != root)
    for number, raw in enumerate(compiled.program):
        code,dest,n,a,an,b,bn = map(int,raw)
        if code not in POINTWISE | CHECKS | {CSR,ROW_CSR,TAKE,SCATTER,ZERO}:
            raise ContractError('Unsupported full group GPU opcode: '+str(code))
        info = dict(number=number,code=code,dest=dest,n=n,a=a,an=an,b=b,bn=bn)
        if code == ZERO:
            level = 0
        elif code in CHECKS:
            level = max(0,dependency(a,n))
        elif code in (CSR,ROW_CSR,TAKE):
            level = max(0,dependency(a,an,True))
        elif code == SCATTER:
            if dest not in written or dest in used:
                raise ContractError('Scatter must finish before consumers read its typed destination')
            level = max(0,layers[dest],dependency(a,an,True))
            count = sizes[dest]
            target = compiled.indices[b:b+n]
            if len(target) != n or np.any(target<0) or np.any(target>=count):
                raise ContractError('Scatter index outside typed destination')
            ordered = np.argsort(target,kind='stable').astype(np.int64)
            ptr = np.concatenate(([0],np.cumsum(np.bincount(target,minlength=count)))).astype(np.int64)
            info.update(ip=len(inverse_indices),pp=len(inverse_pointers),n=count)
            inverse_indices.extend(ordered.tolist())
            inverse_pointers.extend(ptr.tolist())
        else:
            # A computed scalar broadcast also needs a cross-thread barrier.
            level = max(0,dependency(a,an,an==1 and n>1),dependency(b,bn,bn==1 and n>1))
        groups[level].append(info)
        if code not in CHECKS:
            layers[dest] = level
            written.add(dest)
    signature = '(double* w,const double* weights,const long long* indices,const long long* pointers,const long long* inv,const long long* ptr,int* error)'
    sources, launches = [], []
    for level, instructions in sorted(groups.items()):
        name = 'layer_'+str(level)
        lines = ['extern "C" __global__ void '+name+signature+' {', 'int i=blockIdx.x*blockDim.x+threadIdx.x;']
        for r in instructions:
            code,dest,n,a,an,b,bn = (r[k] for k in ('code','dest','n','a','an','b','bn'))
            lines.append(f'if(i<{n}) {{')
            av = f'w[{a}'+('+i]' if an>1 else ']')
            bv = f'w[{b}'+('+i]' if bn>1 else ']')
            output = f'w[{dest}+i]'
            if code in CHECKS:
                bad = '!isfinite(v)'+(' || v<0.0' if code==NONNEG else ' || v<=0.0' if code==POSITIVE else '')
                lines.append(f'double v=w[{a}+i]; if({bad}) atomicMin(error,{r["number"]+2});')
            elif code in (CSR,ROW_CSR):
                lines.append(f'double total=0.0; for(long long j=pointers[{b}+i];j<pointers[{b}+i+1];++j) total=__dadd_rn(total,__dmul_rn(weights[{bn}+j],w[{a}+indices[{bn}+j]])); {output}=total;')
            elif code == TAKE:
                lines.append(f'{output}=w[{a}+indices[{b}+i]];')
            elif code == SCATTER:
                lines.append(f'double total={output}; for(long long j=ptr[{r["pp"]}+i];j<ptr[{r["pp"]}+i+1];++j) total=__dadd_rn(total,w[{a}+inv[{r["ip"]}+j]]); {output}=total;')
            elif code == ZERO:
                lines.append(f'{output}=0.0;')
            else:
                expressions = {
                    ADD:f'__dadd_rn({av},{bv})', SUB:f'__dsub_rn({av},{bv})',
                    MUL:f'__dmul_rn({av},{bv})', DIV:f'__ddiv_rn({av},{bv})',
                    POW:f'pow({av},{bv})', SIGMOID:f'__ddiv_rn(1.0,__dadd_rn(1.0,exp(-{av})))',
                    LOG:f'log({av})', EXP:f'exp({av})',
                    MINIMUM:f'(({bv}<{av})?{bv}:{av})', MAXIMUM:f'(({bv}>{av})?{bv}:{av})',
                }
                lines.append(output+'='+expressions[code]+';')
            lines.append('}')
        lines.append('}')
        sources.append('\n'.join(lines))
        launches.append((name,max(r['n'] for r in instructions)))
    setup = f'''extern "C" __global__ void prepare(const double* y,double* w,int* error) {{
        int i=blockIdx.x*blockDim.x+threadIdx.x;
        if(i<{compiled.nstate}) {{double v=y[i];w[i]=v;if(!isfinite(v)) atomicMin(error,0);}}
        if(i<{compiled._rod_count}) {{
            double v=__dmul_rn(__dmul_rn(-{compiled._rod_k!r},pow(y[{compiled._rod_start}+i],{compiled._rod_n!r})),1e-12);
            w[{compiled.nstate}+i]=v;if(!isfinite(v)) atomicMin(error,1);
        }}
    }}'''
    finish = ['extern "C" __global__ void finish(const double* w,double* out,int* error) {','int i=blockIdx.x*blockDim.x+threadIdx.x;']
    for dest,source,n in compiled.derivatives:
        value = '0.0' if source<0 else f'w[{source}+i]'
        finish.append(f'if(i<{n}) {{double v={value};out[{dest}+i]=v;if(!isfinite(v)) atomicMin(error,{len(compiled.program)+2});}}')
    finish.append('}')
    source = '\n'.join([setup,*sources,'\n'.join(finish)])
    return source,launches,np.asarray(inverse_indices,np.int64),np.asarray(inverse_pointers,np.int64)


class GroupGpuRHS:
    def __init__(self, compiled, torch):
        self.compiled, self.torch = compiled, torch
        self.device = torch.device('cuda',torch.cuda.current_device())
        self.nstate = compiled.nstate
        self.lock = Lock()
        self.source,self.launches,inv,ptr = lower(compiled)
        self.module = CudaModule(torch,self.source,['prepare',*[name for name,_ in self.launches],'finish'])
        self.work = torch.zeros(compiled._scratch.shape,dtype=torch.float64,device=self.device)
        for start,values in compiled._scratch.constants:
            self.work[start:start+len(values)].copy_(torch.as_tensor(values.copy(),device=self.device))
        self.weights,self.indices,self.pointers = [torch.as_tensor(a.copy(),device=self.device) for a in (compiled.weights,compiled.indices,compiled.pointers)]
        self.inverse_indices,self.inverse_pointers = [torch.as_tensor(a,device=self.device) for a in (inv,ptr)]
        self.error = torch.full((1,),SENTINEL,dtype=torch.int32,device=self.device)
        self.arguments = [self.work,self.weights,self.indices,self.pointers,self.inverse_indices,self.inverse_pointers,self.error]
        self.identity = canonical_hash(dict(schema='retipath-million-group-gpu-rhs-1',
            parent=compiled.identity,runtime=self.module.identity,scatter='stable_inverse_rows_original_sum_order'))
        self.metrics = dict(instructions=len(compiled.program),layers=len(self.launches),
            kernel_launches_per_rhs=len(self.launches)+2,
            resident_bytes=sum(t.numel()*t.element_size() for t in self.arguments),
            inner_states=self.nstate,fast_math=False,fma_contraction=False,
            full_rhs=True,all_source_domain_checks=True,scalar_status_D2H_per_rhs=1,
            legacy_small_graph_backend_used=False,scatter_reduction='serial_original_order_per_row',
            csr_reduction='serial_original_order_per_row')

    def evaluate_into(self, t, state, output):
        if self.module.closed or self.torch.cuda.current_device()!=self.module.device or int(self.torch.cuda.current_stream(self.device).cuda_stream)!=self.module.stream:
            raise ContractError('GPU module closed or device/stream changed')
        if not math.isfinite(t):
            raise ContractError('Finite original RHS time required')
        for value in (state,output):
            if value.dtype!=self.torch.float64 or value.device!=self.device or tuple(value.shape)!=(self.nstate,) or not value.is_contiguous() or value.requires_grad:
                raise ContractError('Contiguous FP64 original inner layout on bound CUDA device required')
        if state.untyped_storage().data_ptr()==output.untyped_storage().data_ptr():
            raise ContractError('GPU derivative may not alias input state')
        if not self.lock.acquire(blocking=False):
            raise ContractError('GPU RHS scratch already leased')
        try:
            self.error.fill_(SENTINEL)
            self.module.launch('prepare',self.nstate,[state,self.work,self.error])
            for name,count in self.launches:
                self.module.launch(name,count,self.arguments)
            self.module.launch('finish',max(self.compiled.derivatives[:,2]),[self.work,output,self.error])
            status = int(self.error.item())
            if status!=SENTINEL:
                raise ContractError(f'GPU original inner equation/domain failed at instruction {status-2}')
            return output
        finally:
            self.lock.release()

    def close(self):
        self.module.close()
