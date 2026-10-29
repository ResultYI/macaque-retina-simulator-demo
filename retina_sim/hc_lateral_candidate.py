"""Opt-in effective homotypic HC coupling; not an SI gap-conductance model.

No production factory registers this research-only provider. The parent H1/H2
implementation and all its pinned values remain unchanged.
"""
from pathlib import Path
import hashlib
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import expit

from .contracts import ContractError,canonical_hash
from .h1_h2_effective import H1H2Native,csr,matrix_id
from .vh2005 import SOURCE_VALUES

CANDIDATE='HC_homotypic_diffusion_S_preference_candidate_v2'

class _CandidateAssembly:
    """Bind the independent A before the immutable native provider is built."""
    def __init__(self,parent,A,A2,kappa,eta):
        self.parent=parent;self._A=A
        self.operators=dict(parent.operators,A2=A2)
        self.artifact_sha256=canonical_hash(dict(parent=parent.artifact_sha256,
            candidate=CANDIDATE,A_sha256=matrix_id(A),kappa=kappa,eta=eta))
    def __getattr__(self,name):return getattr(self.parent,name)
    def matrix(self,relative):
        if relative=='routes/cones_to_HC/mean_weights.npz':return self._A
        if relative=='h1_h2/A2.npz':return self.operators['A2']
        return self.parent.matrix(relative)

def homotypic_operator(A,counts):
    """Finite co-input overlap is a topology proxy, not measured gap anatomy.

    Relative group capacity is member count. A scalar normalization bounds the
    fastest outgoing coupling rate. Weighted mean is conserved with sealed crop.
    """
    counts=np.asarray(counts,dtype=float)
    if A.shape[0]!=len(counts) or np.any(counts<=0):raise ContractError('Positive actual HC group counts required')
    overlap=(A@A.T).tocsr();overlap.setdiag(0);overlap.eliminate_zeros()
    degree=np.asarray(overlap.sum(1)).ravel()
    maximum=float(np.max(degree/counts,initial=0.))
    if maximum==0:return csr(overlap),csr(overlap),0.
    neighbor=sparse.diags(1/(counts*maximum))@overlap
    Q=sparse.diags(degree/(counts*maximum))-neighbor
    if np.max(abs(np.asarray(Q.sum(1)).ravel()),initial=0.)>1e-12:
        raise ContractError('Homotypic coupling must preserve a constant signal')
    if np.max(abs(np.asarray(counts@Q).ravel()),initial=0.)>1e-11:
        raise ContractError('Homotypic coupling must conserve member-weighted mean')
    return csr(Q),csr(overlap),maximum

def preference_operator(A,cone_types,eta):
    """One S-efficacy ratio; preserve L:M ratio and full-support unit DC.

    Partial rows are untouched because the exterior S/LM products are not
    separately available. Complete retained rows have exactly the full original
    input, so their normalization is not crop renormalization.
    """
    if not np.isfinite(eta) or eta<1:raise ContractError('S-preference candidate requires eta>=1')
    S=np.asarray(cone_types)=='S'
    coverage=np.asarray(A.sum(1)).ravel()
    complete=np.abs(coverage-1.)<=1e-12
    smass=np.asarray(A[:,S].sum(1)).ravel()
    selected=complete&(smass>0)
    denominator=np.ones(A.shape[0]);denominator[selected]=1+(eta-1)*smass[selected]
    extra=sparse.diags(selected.astype(float))@A@sparse.diags(S.astype(float)*(eta-1))
    result=sparse.diags(1/denominator)@(A+extra)
    if not np.allclose(np.asarray(result.sum(1)).ravel(),coverage,rtol=0,atol=5e-15):
        raise ContractError('S preference changed original retained DC coverage')
    return csr(result),selected

class LateralHCCandidate(H1H2Native):
    def __init__(self,assembly,*,coupling_rate_per_second,S_efficacy_ratio):
        if not np.isfinite(coupling_rate_per_second) or coupling_rate_per_second<0:
            raise ContractError('Nonnegative declared effective coupling rate required')
        self.coupling_rate=float(coupling_rate_per_second)
        self.S_efficacy_ratio=float(S_efficacy_ratio)
        newA2,preference_rows=preference_operator(assembly.operators['A2'],assembly.types['cones'],self.S_efficacy_ratio)
        coos=[assembly.operators['A1'].tocoo(),newA2.tocoo()]
        newA=csr(sparse.coo_matrix((np.r_[coos[0].data,coos[1].data],
            (np.r_[assembly.h1_rows[coos[0].row],assembly.h2_rows[coos[1].row]],np.r_[coos[0].col,coos[1].col])),
            shape=(assembly.populations['HC'].group_count,assembly.populations['cones'].group_count)))
        candidate_assembly=_CandidateAssembly(assembly,newA,newA2,self.coupling_rate,self.S_efficacy_ratio)
        super().__init__(candidate_assembly)
        self.preference_rows=preference_rows
        self.A2=newA2
        self.Q={};self.overlap={};self.topology_normalizer={}
        for name,indices in (('H1',self.h1_rows),('H2',self.h2_rows)):
            q,w,m=homotypic_operator(assembly.operators['A1' if name=='H1' else 'A2'],
                assembly.populations['HC'].counts[indices])
            self.Q[name]=q;self.overlap[name]=w;self.topology_normalizer[name]=m
        self.Qfull=sparse.lil_matrix((self.nh,self.nh))
        for name,indices in (('H1',self.h1_rows),('H2',self.h2_rows)):
            self.Qfull[np.ix_(indices,indices)]=self.Q[name]
        self.Qfull=csr(self.Qfull.tocsr())
        self.coupling_degree=self.Qfull.diagonal()
        self.coupling_neighbor=csr(sparse.diags(self.coupling_degree)-self.Qfull)
        self.profile=dict(candidate=CANDIDATE,parent_native=self.source_version,
            coupling_rate_per_second=self.coupling_rate,S_efficacy_ratio=self.S_efficacy_ratio,
            reference_status='Engineering demonstration point; direction constrained, magnitude not fitted',
            physical_gap_conductance_S=None,HC_membrane_capacitance_F=None,
            no_new_state=True,heterotypic_direct_coupling=False,
            coupling_topology='Actual retained same-type co-input overlap; proxy for local homotypic adjacency',
            coupling_boundary='Sealed homotypic graph at crop; inherited Cone/HC exterior remains fixed separately',
            S_policy='One effective S-release efficacy ratio on complete input rows only; L:M ratio and original DC row mass unchanged',
            Q_sha256={k:matrix_id(v) for k,v in self.Q.items()},A2_sha256=matrix_id(self.A2),
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            physiologically_calibrated=False,FIT_READY=False,default_selected=False)
        self.identity=dict(self.identity,research_candidate=self.profile)
        self.source_version=canonical_hash(self.identity)

    def _coefficients(self,y,t,anchor):
        aa,bb,out=super()._coefficients(y,t,anchor)
        if self.coupling_rate:
            h=y[self.slices['Vh']]
            aa=list(aa);bb=list(bb)
            aa[7]=aa[7]+self.coupling_rate*(self.coupling_neighbor@h)
            bb[7]=bb[7]+self.coupling_rate*self.coupling_degree
            aa,bb=tuple(aa),tuple(bb)
        return aa,bb,out

    def equilibrate_native(self,y,anchor,t):
        """New local HC fixed point; original phototransduction stays unchanged.

        Candidate DC coupling is applied to absolute effective HC states. This
        is not an AC-only clamp, an acute-off baseline or an old seed relabel.
        No downstream initial condition is claimed for this native-only state.
        """
        y=np.array(y,copy=True);p=SOURCE_VALUES
        V=y[self.slices['Vis']];Vbar=y[self.slices['Vbar']]
        aI=(Vbar/p['V_I'])**p['mu']
        pooled=((self.A@Vbar+self.boundary['A_Vbar'])[self.h1_rows]/p['V_I'])**p['mu']
        tau=np.full(self.nh,self.parameters['tau_seconds']);tau[self.h1_rows]=p['tau_h']*.001*pooled
        gains=np.full(self.nh,self.parameters['input_scale']);gains[self.h1_rows]=p['g_t']
        coupling=sparse.diags(tau*self.coupling_rate)@self.Qfull
        linear=sparse.eye(self.nh,format='csr')+coupling
        def signals(h):
            q=expit((V-self.B@h-self.boundary['B_h']-p['V_k'])/p['V_n'])
            r=q/aI
            residual=linear@h-gains*(self.A@r+self.boundary['A_release'])
            return residual,r,q
        h=y[self.slices['Vh']].copy();history=[]
        for step in range(40):
            f,r,q=signals(h);error=float(max(abs(f)));history.append(error)
            if error<=1e-10:break
            derivative=q*(1-q)/(p['V_n']*aI)
            J=linear+sparse.diags(gains)@self.A@sparse.diags(derivative)@self.B
            delta=spsolve(J,-f)
            for power in range(24):
                proposed=h+delta*.5**power
                if np.all(proposed>=0) and max(abs(signals(proposed)[0]))<error:
                    h=proposed;break
            else:raise ContractError('Candidate local equilibrium line search failed')
        else:raise ContractError('Candidate local equilibrium did not converge')
        if error>1e-10:raise ContractError('Candidate equilibrium residual exceeds inherited target')
        y[self.slices['Vh']]=h
        w=p['g_t']*(self.A@r+self.boundary['A_release'])[self.h1_rows]
        y[self.slices['w']]=w;y[self.slices['z2']]=w
        y[self.slices['z1_aux']]=p['g_t']*r[self._aux]
        derivative=self.rhs(y,t,anchor)
        return y,dict(residual_history=history,maximum_full_native_rhs=float(max(abs(derivative))),
            downstream_seed_qualified=False,old_seed_overwritten=False)
