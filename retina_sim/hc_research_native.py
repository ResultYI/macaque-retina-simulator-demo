"""Research-only affine boundary extension of the pinned H1/H2 provider."""
from pathlib import Path
import hashlib
import numpy as np
from scipy.special import expit
from scipy.sparse.linalg import LinearOperator, gmres
from .contracts import ContractError, canonical_hash
from .h1_h2_effective import H1H2Native, cone_fixed_point, owned, csr
from .cone_hc_reduced_native import STATE_NAMES
from .vh2005 import SOURCE_VALUES

def joint_equilibrium(A, B, h1_rows, h2_rows, light, parameters, *, guard=lambda:None, boundary=None):
    """All H1 and H2 feedback enters one nonlinear fixed-point residual."""
    cone = cone_fixed_point(light); p = SOURCE_VALUES
    held_release, held_feedback = np.zeros(A.shape[0]), np.zeros(B.shape[0])
    if boundary is not None:
        if set(boundary) != {'A_release', 'A_Vbar', 'B_h'}:
            raise ContractError('Explicit conditioned background products required')
        for name, n in (('A_release', A.shape[0]), ('A_Vbar', A.shape[0]), ('B_h', B.shape[0])):
            value = np.asarray(boundary[name])
            if value.shape != (n,) or not np.isfinite(value).all() or np.any(value < 0):
                raise ContractError('Invalid conditioned equilibrium product: '+name)
        held_release, held_feedback = boundary['A_release'], boundary['B_h']
    scale = (cone['Vbar']/p['V_I'])**p['mu']
    gains = np.full(A.shape[0],p['g_t']); gains[h2_rows] = parameters['input_scale']
    feedback = B.copy().tocsc(); feedback[:,h2_rows] *= parameters['feedback_scale']; feedback = feedback.tocsr()
    def release(h): return expit((cone['Vis']-feedback@h-held_feedback-p['V_k'])/p['V_n'])/scale
    def residual(h):
        guard(); return h-gains*(A@release(h)+held_release)
    # Analytic Jv avoids finite-difference cancellation in the full coupled graph.
    # Linear budgets are the existing group_native_radau 1e-10/1e-14 values.
    h = gains*(A@release(np.zeros(A.shape[0]))+held_release)*.5
    history = []; nfev = 0
    for iteration in range(200):
        f = residual(h); nfev += 1; error = float(np.max(np.abs(f)))
        history.append(error)
        if error <= 1e-10: break
        q = expit((cone['Vis']-feedback@h-held_feedback-p['V_k'])/p['V_n'])
        derivative = q*(1-q)/(p['V_n']*scale)
        jac = LinearOperator((len(h),len(h)),matvec=lambda v:v+gains*(A@(derivative*(feedback@v))),dtype=np.float64)
        delta,status = gmres(jac,-f,rtol=1e-10,atol=1e-14,restart=32,maxiter=200)
        if status: raise ContractError('Joint analytic Newton linear solve failed: '+str((status,history)))
        for power in range(32):
            proposed = h+delta*(.5**power)
            if np.any(proposed < 0): continue
            candidate = residual(proposed); nfev += 1
            if np.max(np.abs(candidate)) < error:
                h = proposed; break
        else: raise ContractError('Joint Newton line search failed without clipping: '+str(history))
    else: raise ContractError('Joint fixed point failed to converge: '+str(history))
    if error > 1e-8 or not np.isfinite(h).all() or np.any(h < 0):
        raise ContractError('H1+H2 joint fixed point domain/residual failed: '+str(history))
    r = release(h); w = p['g_t']*(A[h1_rows]@r+held_release[h1_rows])
    return dict(cone=cone,h=owned(h),release=owned(r),w=owned(w),z2=owned(w),
        residual=error,nfev=nfev,newton_residual_history=history,feedback=csr(feedback))


class ConditionedH1H2Native(H1H2Native):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_version = canonical_hash(dict(parent=self.source_version,
            implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

    def boundary_at(self, t):
        return self.boundary


    def _coefficients(self,y,t,anchor):
        R,E = self._validate(y,anchor,t)
        X,C,V,gi,Vbar,w,z2,h,aux = self.split(y); p = self.p
        boundary = self.boundary_at(t)
        if boundary is None: raise ContractError('Explicit H1/H2 boundary binding required')
        bh = self.B@h+boundary['B_h']
        q = expit((V-bh-p['V_k'])/p['V_n']); aI = (Vbar/p['V_I'])**p['mu']; release = q/aI
        pooled = ((self.A@Vbar+boundary['A_Vbar'])[self.h1_rows]/p['V_I'])**p['mu']
        ar = self.A@release+boundary['A_release']
        ah,bh_decay = np.empty(self.nh),np.empty(self.nh)
        ah[self.h1_rows] = 1000.*z2/(p['tau_h']*pooled)
        bh_decay[self.h1_rows] = 1000./(p['tau_h']*pooled)
        ah[self.h2_rows] = self.parameters['input_scale']*ar[self.h2_rows]/self.parameters['tau_seconds']
        bh_decay[self.h2_rows] = 1/self.parameters['tau_seconds']
        a = (1000./(1+(p['a_C']*C)**p['n_C']),1000.*X/p['tau_C'],1000.*X/(gi*p['tau_m']),
            1000.*p['a_is']*V**p['gamma']/p['tau_is'],1000.*V/p['tau_a'],
            1000.*p['g_t']*ar[self.h1_rows]/p['tau_1'],1000.*w/(p['tau_2']*pooled),
            ah,1000.*p['g_t']*release[self._aux]/p['tau_1'])
        b = (1000.*(p['c_beta']+p['k_beta']*E),1000./p['tau_C'],1000./p['tau_m'],1000./p['tau_is'],
            1000./p['tau_a'],1000./p['tau_1'],1000./(p['tau_2']*pooled),bh_decay,1000./p['tau_1'])
        # Sparse readout carrier keeps the existing exact native->BC bridge API.
        full_w,full_z2,full_pool = np.zeros(self.nh),np.zeros(self.nh),np.ones(self.nh)
        full_w[self.h1_rows]=w;full_z2[self.h1_rows]=z2;full_pool[self.h1_rows]=pooled
        outputs = dict(zip(STATE_NAMES,(X,C,V,gi,Vbar,full_w,full_z2,h,aux)))
        outputs.update(R=R,E=E,q=q,release=release,Ios=X,Bh=bh,aI=aI,pooled_aI=full_pool)
        self.statistics['coefficient_evaluations'] += 1
        return a,b,outputs


