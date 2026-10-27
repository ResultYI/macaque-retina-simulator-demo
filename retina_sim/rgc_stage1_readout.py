"""RGC-only native Bernoulli/history fit on identity-bound endogenous voltage."""
from copy import deepcopy
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from .contracts import ContractError
from .rgc_expectation import STATES,NEXT0,NEXT1

NAMES=('mu','history_1','history_2','history_3','log_alpha','log_beta')


def decode(theta,base):
    x=np.asarray(theta,dtype=np.float64)
    if x.shape!=(6,) or not np.isfinite(x).all():raise ContractError('Six finite declared RGC coordinates required')
    p=deepcopy(base);p['mu']=base['mu']+base['beta']*x[0]
    p['history_kernel']=np.asarray(base['history_kernel'])*(1+x[1:4])
    p['alpha']=base['alpha']*np.exp(x[4]);p['beta']=base['beta']*np.exp(x[5])
    return p


def bounds(base):
    h=np.asarray(base['history_kernel']);margin=4*base['beta']
    lo=min(base[k] for k in ['EL','EE','EI'])+float(np.minimum(h,0).sum())-margin
    hi=max(base[k] for k in ['EL','EE','EI'])+float(np.maximum(h,0).sum())+margin
    return [((lo-base['mu'])/base['beta'],(hi-base['mu'])/base['beta']),
            (-1.,1.),(-1.,1.),(-1.,1.),(np.log(.01),np.log(100.)),(np.log(.5),np.log(2.))]


def observed_past(counts,past=None):
    y=np.asarray(counts)
    if y.ndim!=2 or not np.isin(y,[0,1]).all():raise ContractError('Actual binary repeat counts required; no clipping')
    initial=np.zeros((len(y),3),dtype=np.int64) if past is None else np.asarray(past)
    if initial.shape!=(len(y),3) or not np.isin(initial,[0,1]).all():raise ContractError('Original newest-first observed prehistory required')
    extended=np.concatenate((initial[:,::-1],y),axis=1)
    return np.stack([extended[:,3-lag:3-lag+y.shape[1]] for lag in [1,2,3]],axis=-1)


def terms(theta,case,base):
    p=decode(theta,base);lo,hi=case['score_bins'];y=case['counts'][:,lo:hi]
    past=case['past'][:,lo:hi];v=case['voltage'][None,lo:hi]
    z=(v+past@p['history_kernel']-p['mu'])/p['beta'];sp=np.logaddexp(0.,z)
    a=p['alpha']*.001*sp;q=-np.expm1(-a)
    if not np.isfinite(a).all() or np.any(a<=0) or not np.isfinite(q).all():raise ContractError('Native output probability domain failed')
    common=p['alpha']*.001*expit(z)
    jac=np.stack([-common*base['beta']/p['beta'],
        *(common*past[:,:,i]*base['history_kernel'][i]/p['beta'] for i in range(3)),a,-common*z],axis=-1)
    return y,a,q,jac,sp,common


def objective(theta,cases,base):
    loss=0.;gradient=np.zeros(6);n=0
    for c in cases:
        y,a,q,jac,_,_=terms(theta,c,base)
        loss+=float(np.sum(np.where(y,-np.log(q),a)))
        derivative=np.ones_like(a);events=y==1;derivative[events]=-np.exp(-a[events])/q[events]
        gradient+=np.sum(derivative[:,:,None]*jac,axis=(0,1));n+=y.size
    if not n or not np.isfinite(loss) or not np.isfinite(gradient).all():raise ContractError('Empty/nonfinite native likelihood')
    return loss/n,gradient/n


def identifiable(cases,base,relative_threshold=.001):
    """Priority-preserving training design rank; not a confidence interval."""
    columns=[]
    for c in cases:
        _,a,_,jac,_,_=terms(np.zeros(6),c,base)
        columns.append((jac/a[:,:,None]).reshape(-1,6))
    matrix=np.vstack(columns)/np.sqrt(sum(len(v) for v in columns));selected=[];evidence=[]
    for j,name in enumerate(NAMES):
        trial=matrix[:,selected+[j]];singular=np.linalg.svd(trial,compute_uv=False)
        ratio=float(singular[-1]/singular[0]) if singular[0]>0 else 0.
        accepted=bool(singular[0]>0 and ratio>=relative_threshold)
        if accepted:selected.append(j)
        evidence.append(dict(parameter=name,accepted=accepted,relative_singular_value=ratio))
    return dict(selected_indices=selected,selected_parameters=[NAMES[j] for j in selected],per_parameter=evidence,
        threshold=relative_threshold,scope='training conditional log-hazard design only; no physiological identifiability claim')


def free_mean(voltage,parameters,initial_probability):
    pi=np.asarray(initial_probability,dtype=np.float64).copy();h=STATES@np.asarray(parameters['history_kernel']);prob=[]
    if pi.shape!=(8,) or np.any(pi<0) or abs(pi.sum()-1)>1e-12:raise ContractError('Frozen original history law required')
    for v in np.asarray(voltage):
        q=-np.expm1(-.001*parameters['alpha']*np.logaddexp(0.,(v+h-parameters['mu'])/parameters['beta']))
        prob.append(float(np.sum(pi*q)));nxt=np.zeros(8)
        for j in range(8):nxt[NEXT0[j]]+=pi[j]*(1-q[j]);nxt[NEXT1[j]]+=pi[j]*q[j]
        pi=nxt
    if abs(pi.sum()-1)>1e-12:raise ContractError('RGC free-history mass conservation failed')
    return np.asarray(prob)


def fit(cases,base,*,callback=None,guard=None):
    if not cases or any(c['split']!='train' for c in cases):raise ContractError('Only original admitted training cases may select RGC parameters')
    rank=identifiable(cases,base);indices=rank['selected_indices'];initial=np.zeros(6);trace=[]
    if not indices:return dict(status='NO_IDENTIFIABLE_OUTPUT_PARAMETER',theta=initial.tolist(),rank=rank,trace=trace)
    def evaluate(x):
        if guard:guard()
        full=np.zeros(6);full[indices]=x;loss,g=objective(full,cases,base)
        return loss,g[indices]
    def accepted(x):
        full=np.zeros(6);full[indices]=x;loss,g=objective(full,cases,base)
        row=dict(iteration=len(trace)+1,theta=full.tolist(),training_NLL_per_bin=loss);trace.append(row)
        if callback:callback(row)
    b=bounds(base);result=minimize(evaluate,np.zeros(len(indices)),jac=True,bounds=[b[j] for j in indices],method='L-BFGS-B',callback=accepted,
        options=dict(maxiter=100,maxfun=1000,maxls=30,ftol=1e-10,gtol=1e-6))
    theta=np.zeros(6);theta[indices]=result.x
    return dict(status='CONDITIONAL_RGC_OUTPUT_TRAINING_COMPLETE',theta=theta.tolist(),rank=rank,trace=trace,
        success=bool(result.success),message=str(result.message),iterations=int(result.nit),evaluations=int(result.nfev),training_NLL_per_bin=float(result.fun),
        values={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in decode(theta,base).items()},internal_parameters_changed=False)
