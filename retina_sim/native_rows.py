"""Execution-only row fusion of the verified native continuous RHS.

Every row is independent within a pass. CSR sums retain their original column
order. No fastmath, parallel reduction, equation, parameter or solver change.
Numba is imported only when the existing compiled backend is requested.
"""
import numpy as np


def _signals(work,n,layout,parameters):
    ix,iv,ig,ib,ih,oi,oa,oq,orr=[int(work[layout+j]) for j in range(9)]
    nx,vi,mu,vk,vn=work[parameters:parameters+5]
    for i in prange(n):
        current=work[ix+i]**nx
        adaptation=(work[ib+i]/vi)**mu
        q=1./(1.+np.exp(-((work[iv+i]-work[ih+i]-vk)/vn)))
        work[oi+i]=current;work[oa+i]=adaptation;work[oq+i]=q;work[orr+i]=q/adaptation


def _cone_derivatives(work,n,layout,parameters):
    r,e,x,c,v,g,vb,z,light,current,release=[int(work[layout+j]) for j in range(11)]
    dr,de,dx,dc,dv,dg,dvb,dz=[int(work[layout+11+j]) for j in range(8)]
    tr,te,ac,nc,cb,kb,tc,tm,ais,gamma,tis,ta,gt,t1=work[parameters:parameters+14]
    for i in prange(n):
        rv=work[r+i];ev=work[e+i];xv=work[x+i];cv=work[c+i]
        vv=work[v+i];gv=work[g+i];vbv=work[vb+i];iv=work[current+i]
        alpha=1./(1.+(ac*cv)**nc);beta=cb+kb*ev
        work[dr+i]=1000.*((work[light+i]-rv)/tr)
        work[de+i]=1000.*((rv-ev)/te)
        work[dx+i]=1000.*(alpha-beta*xv)
        work[dc+i]=1000.*((iv-cv)/tc)
        work[dv+i]=1000.*((iv/gv-vv)/tm)
        work[dg+i]=1000.*((ais*vv**gamma-gv)/tis)
        work[dvb+i]=1000.*((vv-vbv)/ta)
        work[dz+i]=1000.*((gt*work[release+i]-work[z+i])/t1)


def _hc_derivatives(work,n,layout,parameters):
    av,az,z,h,dz,dh=[int(work[layout+j]) for j in range(6)]
    vi,mu,t2,th=work[parameters:parameters+4]
    for i in prange(n):
        pooled=(work[av+i]/vi)**mu
        work[dz+i]=1000.*((work[az+i]-work[z+i])/(t2*pooled))
        work[dh+i]=1000.*((work[z+i]-work[h+i])/(th*pooled))


def _csr(work,n,dest,source,pointer_offset,data_offset,weights,indices,pointers):
    for i in prange(n):
        total=0.
        for j in range(pointers[pointer_offset+i],pointers[pointer_offset+i+1]):
            total+=weights[data_offset+j]*work[source+indices[data_offset+j]]
        work[dest+i]=total


_bindings=None
def compiled_bindings():
    global _bindings,prange
    import numba
    from numba import prange
    if _bindings is None:
        _bindings=tuple(numba.njit(cache=True,parallel=True,fastmath=False,boundscheck=True,error_model='numpy')(fn)
                        for fn in (_signals,_cone_derivatives,_hc_derivatives,_csr))
    return _bindings
