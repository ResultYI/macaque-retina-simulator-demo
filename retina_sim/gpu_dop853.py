"""Optional FP64 CUDA DOP853 execution with the installed SciPy coefficients.

Controller, initial step, error scale, dense polynomial and requested times
match SciPy DOP853. GPU reductions may differ in rounding; the execution route
therefore has its own persistent identity and cannot restore CPU checkpoints.
"""
import numpy as np
from scipy.integrate._ivp.rk import DOP853,SAFETY,MIN_FACTOR,MAX_FACTOR
from scipy.integrate._ivp.common import validate_tol,validate_max_step
from .contracts import ContractError


def integrate(fun,t0,y0,t_bound,*,rtol,atol,max_step,t_eval=None):
    import torch
    if y0.dtype!=torch.float64 or y0.ndim!=1 or not y0.numel() or t_bound<=t0:
        raise ContractError('CUDA DOP853 requires a nonempty forward float64 interval')
    if not np.isfinite(rtol) or rtol<=0 or not np.all(np.isfinite(atol)) or np.any(np.asarray(atol)<=0) or max_step<=0:
        raise ContractError('Invalid CUDA DOP853 numerical tolerances')
    device=y0.device;n=y0.numel();y=y0.clone();t=float(t0);nfev=0;steps=0
    rtol,atol=validate_tol(rtol,atol,n);max_step=validate_max_step(max_step)
    sample_times=np.asarray([] if t_eval is None else t_eval,dtype=float)
    if sample_times.ndim!=1 or not np.all(np.isfinite(sample_times)) or np.any(np.diff(sample_times)<=0) or np.any(sample_times<t0) or np.any(sample_times>t_bound):
        raise ContractError('CUDA DOP853 observations outside the increasing forward interval')
    def rhs(at,state):
        nonlocal nfev
        nfev+=1;result=fun(float(at),state)
        if result.shape!=state.shape or result.dtype!=torch.float64:
            raise ContractError('CUDA RHS execution layout differs')
        return result
    def rms(a):return torch.linalg.vector_norm(a).item()/n**.5
    A=torch.as_tensor(DOP853.A,dtype=torch.float64,device=device)
    B=torch.as_tensor(DOP853.B,dtype=torch.float64,device=device)
    E3=torch.as_tensor(DOP853.E3,dtype=torch.float64,device=device)
    E5=torch.as_tensor(DOP853.E5,dtype=torch.float64,device=device)
    AE=torch.as_tensor(DOP853.A_EXTRA,dtype=torch.float64,device=device)
    DD=torch.as_tensor(DOP853.D,dtype=torch.float64,device=device)
    at=torch.as_tensor(atol,dtype=torch.float64,device=device)
    f=rhs(t,y);scale=at+y.abs()*rtol;d0=rms(y/scale);d1=rms(f/scale)
    h0=1e-6 if d0<1e-5 or d1<1e-5 else .01*d0/d1
    h0=min(h0,t_bound-t0);f1=rhs(t0+h0,y+h0*f);d2=rms((f1-f)/scale)/h0
    h1=max(1e-6,h0*1e-3) if d1<=1e-15 and d2<=1e-15 else (.01/max(d1,d2))**(1/8)
    h_abs=min(100*h0,h1,t_bound-t0,max_step)
    K=torch.empty((16,n),dtype=torch.float64,device=device)
    output=[];position=0
    while position<len(sample_times) and sample_times[position]==t:
        output.append(y.cpu().numpy().copy());position+=1
    while t<t_bound:
        minimum=10*abs(np.nextafter(t,np.inf)-t)
        h_abs=min(max(h_abs,minimum),max_step);rejected=False
        while True:
            if h_abs<minimum:raise ContractError('DOP853 required step size is below floating point spacing')
            t_new=min(t+h_abs,t_bound);h=t_new-t;h_abs=abs(h);K[0].copy_(f)
            for stage in range(1,12):
                delta=torch.mv(K[:stage].T,A[stage,:stage])*h
                K[stage].copy_(rhs(t+DOP853.C[stage]*h,y+delta))
            y_new=y+h*torch.mv(K[:12].T,B);f_new=rhs(t_new,y_new);K[12].copy_(f_new)
            scale=at+torch.maximum(y.abs(),y_new.abs())*rtol
            err5=torch.mv(K[:13].T,E5)/scale;err3=torch.mv(K[:13].T,E3)/scale
            norm5=torch.linalg.vector_norm(err5).item()**2;norm3=torch.linalg.vector_norm(err3).item()**2
            error=0. if norm5==0 and norm3==0 else abs(h)*norm5/np.sqrt((norm5+.01*norm3)*n)
            if not np.isfinite(error):raise ContractError('Nonfinite DOP853 error estimate')
            if error<1:
                factor=MAX_FACTOR if error==0 else min(MAX_FACTOR,SAFETY*error**(-1/8))
                if rejected:factor=min(1,factor)
                h_abs*=factor;break
            h_abs*=max(MIN_FACTOR,SAFETY*error**(-1/8));rejected=True
        steps+=1;end=np.searchsorted(sample_times,t_new,side='right')
        if position<end:
            # solve_ivp uses dense output whenever t_eval is supplied, including
            # observations coinciding with the accepted endpoint.
            for stage in range(13,16):
                delta=torch.mv(K[:stage].T,AE[stage-13,:stage])*h
                K[stage].copy_(rhs(t+DOP853.C_EXTRA[stage-13]*h,y+delta))
            delta=y_new-y;F=torch.empty((7,n),dtype=torch.float64,device=device)
            F[0].copy_(delta);F[1].copy_(h*K[0]-delta)
            F[2].copy_(2*delta-h*(f_new+K[0]));F[3:].copy_(h*torch.mm(DD,K))
            for sampled in sample_times[position:end]:
                x=(sampled-t)/h;value=torch.zeros_like(y)
                for j,row in enumerate(reversed(F.unbind(0))):
                    value+=row;value*=x if j%2==0 else 1-x
                value+=y;output.append(value.cpu().numpy().copy())
            position=end
        t=t_new;y=y_new;f=f_new
    if position!=len(sample_times):raise ContractError('Unreturned requested DOP853 observation')
    if not torch.isfinite(y).all().item():raise ContractError('Nonfinite DOP853 final state')
    trajectory=np.asarray(output).T if output else np.empty((n,0),dtype=float)
    return {'t':sample_times,'y':trajectory,'nfev':nfev,'steps':steps,'final':y.cpu().numpy().copy()}
