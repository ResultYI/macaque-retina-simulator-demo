"""Equation-specific FP64 Radau IIA5 stages for the complete inner graph.

Clock commits and publication remain with the runner. A failed attempt owns only
private arrays. The estimator is operational control, not a uniform certificate.
"""
from time import perf_counter
from pathlib import Path
import hashlib
import math

import numpy as np
from scipy.special import expit

from .contracts import ContractError, canonical_hash
from .group_inner_dynamics import InnerAdvance, RODS, MEMBRANE, BC_MECH, AC_MECH, RGC_MECH
from .spatial_group_model import BC, AC, RGC
from .group_radau_math import tableau, affine_stages, gmres, GMRESWorkspace
from .group_scc_schedule import FamilySpec, bind_scc_schedule


class _AttemptFailure(Exception):
    pass


class GroupInnerRadau:
    """Two half-steps, shared real parent stages, and bounded SCC solves.

    ``error_weights`` and ``budgets`` are full packed vectors or state mappings.
    They are copied once; absent weights freeze at the first input state. Budgets
    are never derived from a candidate current state. ``advance`` publishes CPU
    continuous packets even when the numerical workspace is on a torch device.
    """
    def __init__(self, inner, torch=None, device='cpu', guard=None, *,
                 error_weights=None, budgets=None, compiled=None, original_rhs=None, scc_plan=None):
        self.inner, self.torch, self.device, self.guard = inner, torch, device, guard
        self.original_rhs = original_rhs
        self._check_guard()
        if torch is None and str(device) != 'cpu':
            raise ContractError('A torch module is required for a device Radau workspace')
        self.active = tuple(k for k in inner.state_keys if self._is_active(k))
        self.held = tuple(k for k in inner.state_keys if k not in self.active)
        self.rod_v = ('rods','rod_membrane','V')
        self.plan = self._bind_plan(compiled,scc_plan)
        self._family_actions={}
        self.schedule=bind_scc_schedule(self.plan,self._family_specs())
        self._p, self._cp, self._matrices, self._gains = {}, {}, {}, {}
        self._device_reductions=None
        if torch is not None and torch.device(device).type=='cuda':
            from .group_inner_radau_kernels import OrderedDeviceReductions
            self._device_reductions=OrderedDeviceReductions(torch,device)
        self.C = {p:self._array(v) for p,v in inner.C.items()}
        self.gL = {p:self._array(v) for p,v in inner.gL.items()}
        self.members = {p:self._array(v) for p,v in inner.members.items()}
        self.contacts = {}
        for name,row in inner.contacts.items():
            self.contacts[name] = dict(row, pre=self._index(row['pre']), post=self._index(row['post']),
                                      coefficient=self._array(row['coefficient']))
        self.active = tuple(k for k in inner.state_keys if self._is_active(k))
        self.held = tuple(k for k in inner.state_keys if k not in self.active)
        self.rod_v = ('rods','rod_membrane','V')
        self._weights = self._fixed(error_weights, 'error weights')
        self._budgets = self._fixed(budgets, 'scientific budgets')
        self._A,self._b,self._c,self._gamma,self._bhat = tableau(self.C['rods'])
        self._gap = self._sparse('gap',inner.gap)
        self._gap_diagonal = self._array(inner.gap.diagonal())
        self._gap_C = self._cat([self.C[p] for p in inner.gap_names])
        self._gap_gL = self._cat([self.gL[p] for p in inner.gap_names])
        self.counters = {}
        self._gap_stage_workspace=GMRESWorkspace(self._zeros((3,self._gap_C.shape[0])),restart=32)
        self._gap_filter_workspace=GMRESWorkspace(self._zeros((self._gap_C.shape[0],)),restart=32)
        self.next_step=1e-4
        self._window_end=None
        self._window_attempts=self._window_accepts=0
        self.identity=dict(schema='group_inner_scc_radau_iia5_two_halfsteps_fp64_v1',
            model=inner.identity,device=str(device),scc_plan=self.plan.identity,schedule=self.schedule.identity,
            gmres_workspaces=tuple(dict(shape=shape,restart=32,dtype='float64',owned_bytes=bank.owned_bytes,
                host_bytes=bank.host_bytes,device_bytes=bank.device_bytes) for shape,bank in (
                    ((3,int(self._gap_C.shape[0])),self._gap_stage_workspace),
                    ((int(self._gap_C.shape[0]),),self._gap_filter_workspace))),
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ordered_reductions=None if self._device_reductions is None else self._device_reductions.identity)

    def _bind_plan(self,compiled,plan):
        from .group_inner_compiled import CompiledGroupInnerRHS
        from .group_scc_plan import build_scc_plan
        if compiled is None:
            compiled=self.inner.compiled_rhs
            if compiled is None or not hasattr(compiled,'program'):compiled=CompiledGroupInnerRHS(self.inner)
        if getattr(compiled,'model',None) is not self.inner:
            raise ContractError('Inner stage typed IR belongs to a different model')
        if plan is None:return build_scc_plan(self.inner,compiled)
        if (tuple(plan.full_keys)!=tuple(self.inner.state_keys) or plan.nstate!=self.inner.nstate
                or any(plan.full_slices[k]!=self.inner.slices[k] for k in self.inner.state_keys)):
            raise ContractError('Prebuilt inner SCC plan registry differs')
        nexternal=int(getattr(compiled.plan,'nexternal',0))
        external=() if not nexternal else ((self.inner.nstate,int(compiled._rod_start)),)
        identity=canonical_hash(dict(program=np.asarray(compiled.program),derivatives=np.asarray(compiled.derivatives),
            weights=np.asarray(compiled.weights),indices=np.asarray(compiled.indices),pointers=np.asarray(compiled.pointers),
            constants=tuple((int(k),v) for k,v in sorted(dict(compiled._scratch.constants).items())),
            source_model=self.inner.identity,source_compiled=getattr(compiled,'identity',None),external_bindings=external,
            external_source='group_inner_compiled._source_rod_current' if external else None))
        if identity!=plan.typed_ir_identity:
            raise ContractError('Prebuilt inner SCC plan typed IR identity differs')
        return plan

    def _family_specs(self):
        specs=[];model=self.inner
        ck=lambda c,n:(model.contacts[c]['route'].pre,c,n)
        def add(name,writes,dependencies,action,*args,groups=()):
            rules=tuple((target,source) for target,sources in dependencies.items() for source in sources)
            specs.append(FamilySpec(name,tuple(writes),rules,tuple(groups)))
            self._family_actions[name]=(action,args)
        vk=self.rod_v;rk=self._memkey('rods','release')
        add('rod_voltage',(vk,),{vk:(vk,('rods',RODS,'G'))},self._stage_rod_voltage)
        add('rod_release',(rk,),{rk:(vk,rk)},self._stage_rod_release)
        c='individual_rod_RBC_contacts';mk=ck(c,'m')
        add('rod_contact_trpm1',(mk,),{mk:(mk,rk)},self._stage_trpm1)
        vk=self._vkey('rod_BC')
        add('rod_bc_voltage',(vk,),{vk:(vk,mk)},self._stage_rod_bc_voltage)
        for pop in ('rod_BC','AII_AC'):
            vk=self._vkey(pop);xk=self._memkey(pop,'x');rk=self._memkey(pop,'release')
            add(pop+'_ribbon',(xk,rk),{xk:(xk,vk),rk:(rk,vk)},self._stage_ribbon,pop)
        c='RBC_AII_AMPA';ak,bk=ck(c,'a'),ck(c,'b')
        parent=(self._vkey('rod_BC'),self._memkey('rod_BC','x'))
        add('rbc_aii_ampa',(ak,bk),{ak:(ak,*parent),bk:(bk,*parent)},self._stage_ampa)
        gap_keys=tuple(self._vkey(pop) for pop in model.gap_names)
        deps={self._vkey(pop):(*gap_keys,ak,bk) if pop=='AII_AC' else (*gap_keys,self._bckey(pop,'activation')) for pop in model.gap_names}
        groups=tuple((k,np.zeros(model.sizes[k],dtype=np.int64)) for k in gap_keys)
        add('gap_voltage',gap_keys,deps,self._stage_gap,groups=groups)
        for pop,c in (('midget_BC_OFF','AII_FMB_glycine'),('diffuse_BC_OFF','AII_DB1_glycine')):
            ak,bk=ck(c,'a'),ck(c,'b');vk=self._vkey(pop)
            parent=(self._vkey('AII_AC'),self._memkey('AII_AC','x'))
            add(c+'_rise',(ak,),{ak:(ak,*parent)},self._stage_glycine_rise,c)
            add(pop+'_glycine',(vk,bk),{vk:(vk,bk,ak,self._bckey(pop,'activation')),bk:(bk,vk,*parent)},
                self._stage_off,pop,c,groups=((vk,np.arange(model.sizes[vk],dtype=np.int64)),(bk,model.contacts[c]['post'])))
        for pop in BC:
            vk=self._vkey(pop);rk=self._bckey(pop,'terminal_release');ak=self._bckey(pop,'A')
            writes=(rk,ak) if model.p[pop]['adaptation_enabled'] else (rk,)
            deps={rk:(rk,vk)}
            if len(writes)==2:deps[ak]=(ak,rk)
            add(pop+'_terminal',writes,deps,self._stage_bc_terminal,pop)
        def activation_keys(pop):
            if pop in BC:
                return (self._bckey(pop,'terminal_release'),self._bckey(pop,'A')) if model.p[pop]['adaptation_enabled'] else (self._bckey(pop,'terminal_release'),)
            if pop in AC:return ((pop,AC_MECH,'h'),)
            raise ContractError('Unregistered inner activation source: '+pop)
        for pop in AC:
            k=(pop,AC_MECH,'h');sources=[k]
            for route in model.routes:
                if route.post==pop:
                    if route.pre not in BC:raise ContractError('Unregistered AC input coupling: '+route.pre)
                    sources.extend(activation_keys(route.pre))
            add(pop+'_inhibition',(k,),{k:tuple(sources)},self._stage_ac,pop)
        for pop in RGC:
            k=(pop,RGC_MECH,'V');sources=[k]
            for route in model.routes:
                if route.post==pop:sources.extend(activation_keys(route.pre))
            add(pop+'_voltage',(k,),{k:tuple(sources)},self._stage_rgc,pop)
        return tuple(specs)

    def _is_active(self,k):
        p,m,n=k
        return (m in ('rod_membrane',MEMBRANE) or m in self.inner.contacts or
                m==BC_MECH and (n in ('V','terminal_release') or n=='A' and self.inner.p[p]['adaptation_enabled']) or
                m==AC_MECH and n=='h' or m==RGC_MECH and n=='V')

    def _check_guard(self):
        if self.guard is None:return True
        result = self.guard.require() if hasattr(self.guard,'require') else self.guard()
        if result is not None and not result:raise ContractError('Radau execution guard stopped the attempt')
        return True

    def _array(self,v):
        return np.array(v,dtype=np.float64,copy=True) if self.torch is None else self.torch.as_tensor(np.array(v,dtype=np.float64,copy=True),dtype=self.torch.float64,device=self.device)

    def _index(self,v):
        return np.asarray(v,dtype=np.int64) if self.torch is None else self.torch.as_tensor(np.array(v,dtype=np.int64,copy=True),dtype=self.torch.int64,device=self.device)

    def _cpu(self,v):
        if self.torch is None:return np.array(v,copy=True)
        if v.device.type=='cuda':self._bump('device_to_host_bytes',v.numel()*v.element_size())
        return v.detach().cpu().numpy().copy()

    def _copy(self,v):return v.copy() if self.torch is None else v.clone()
    def _cat(self,values,axis=0):return np.concatenate(values,axis=axis) if self.torch is None else self.torch.cat(values,dim=axis)
    def _stack(self,values,axis=0):return np.stack(values,axis=axis) if self.torch is None else self.torch.stack(values,dim=axis)
    def _zeros(self,shape):return np.zeros(shape,dtype=np.float64) if self.torch is None else self.torch.zeros(shape,dtype=self.torch.float64,device=self.device)
    def _ones(self,shape):return np.ones(shape,dtype=np.float64) if self.torch is None else self.torch.ones(shape,dtype=self.torch.float64,device=self.device)
    def _eye(self):return np.eye(3,dtype=np.float64) if self.torch is None else self.torch.eye(3,dtype=self.torch.float64,device=self.device)
    def _transpose(self,v):return v.T if self.torch is None else v.transpose(-1,-2)
    def _solve(self,a,b):
        self._check_guard();self._bump('local_direct_solve_calls')
        if self.torch is None:
            try:return np.linalg.solve(a,b)
            except np.linalg.LinAlgError as exc:raise _AttemptFailure('Singular local Radau block') from exc
        value,info=self.torch.linalg.solve_ex(a,b,check_errors=False)
        if not self._all(info==0) or not self._finite(value):
            raise _AttemptFailure('Singular/nonfinite local Radau block')
        return value
    def _exp(self,v):return np.exp(v) if self.torch is None else self.torch.exp(v)
    def _expm1(self,v):return np.expm1(v) if self.torch is None else self.torch.expm1(v)
    def _log(self,v):return np.log(v) if self.torch is None else self.torch.log(v)
    def _sigmoid(self,v):return expit(v) if self.torch is None else self.torch.sigmoid(v)
    def _clip(self,v,lo,hi):return np.clip(v,lo,hi) if self.torch is None else self.torch.clamp(v,lo,hi)
    def _maximum(self,a,b):return np.maximum(a,b) if self.torch is None else self.torch.maximum(a,b)
    def _scalar(self,v):
        if self.torch is not None and isinstance(v,self.torch.Tensor):
            if v.device.type=='cuda':self._bump('host_scalar_reads')
            return v.item()
        return v
    def _finite(self,v):return bool(np.isfinite(v).all()) if self.torch is None else bool(self._scalar(self.torch.isfinite(v).all()))
    def _all(self,v):return bool(self._scalar(v.all()))
    def _max(self,v):return float(np.max(v)) if self.torch is None else float(self._scalar(v.max()))
    def _sum(self,v):return float(np.sum(v)) if self.torch is None else float(self._scalar(v.sum()))
    def _bump(self,k,n=1):self.counters[k]=self.counters.get(k,0)+n

    def _fixed(self,v,name):
        if v is None:return None
        packed = self.inner.pack(v) if isinstance(v,dict) else np.asarray(v)
        if packed.dtype!=np.float64 or packed.shape!=(self.inner.nstate,) or not np.isfinite(packed).all() or np.any(packed<=0):
            raise ContractError('Positive finite original-layout FP64 '+name+' required')
        return {k:self._array(packed[s]) for k,s in self.inner.slices.items()}

    def _par(self,p,n,**kw):
        k=p,n
        if k not in self._p:self._p[k]=self._array(self.inner.par(p,n,**kw))
        return self._p[k]

    def _contact_par(self,c,n,**kw):
        k=c,n
        if k not in self._cp:self._cp[k]=self._array(self.inner.cp(c,n,**kw))
        return self._cp[k]

    def _sparse(self,name,matrix):
        if name not in self._matrices:
            csr=matrix.tocsr()
            if self._device_reductions is not None:
                self._matrices[name]=self._device_reductions.csr(csr)
                return self._matrices[name]
            self._matrices[name]=csr if self.torch is None else self.torch.sparse_csr_tensor(
                self._index(csr.indptr),self._index(csr.indices),self._array(csr.data),size=csr.shape,dtype=self.torch.float64,device=self.device)
        return self._matrices[name]

    def _mv(self,matrix,x,kind):
        stages=1 if x.ndim==1 else x.shape[0]
        self._bump(kind+'_matvecs',stages)
        self._bump(kind+'_edge_visits',stages*(matrix.nnz if self.torch is None or self._device_reductions is not None else matrix.values().numel()))
        if self._device_reductions is not None:
            self._check_guard();self._bump('ordered_cuda_reduction_launches')
            return self._device_reductions.matvec(matrix,x)
        if self.torch is None:return matrix@x if x.ndim==1 else (matrix@x.T).T
        return self.torch.sparse.mm(matrix,x[:,None])[:,0] if x.ndim==1 else self.torch.sparse.mm(matrix,x.T).T

    def _route(self,name,x):return self._mv(self._sparse(name,self.inner.mean[name]),x,'route')

    def _pool(self,c,x):
        """Keep each contact and preserve source order within every output row."""
        row=self.contacts[c];n=self.inner.counts[row['route'].post]
        if self._device_reductions is not None:
            self._check_guard();self._bump('ordered_cuda_reduction_launches')
            self._bump('contact_edge_visits',int(np.prod(x.shape)))
            return self._device_reductions.pool(c,self.inner.contacts[c]['post'],n,x)
        if x.ndim==1:
            out=self._zeros((n,))
            if self.torch is None:np.add.at(out,row['post'],x)
            else:out.index_add_(0,row['post'],x)
        else:
            out=self._zeros((*x.shape[:-1],n))
            if self.torch is None:
                for idx in np.ndindex(x.shape[:-1]):np.add.at(out[idx],row['post'],x[idx])
            else:out.index_add_(x.ndim-1,row['post'],x)
        self._bump('contact_edge_visits',int(np.prod(x.shape)))
        return out

    def _memkey(self,p,n):return (p,'rod_membrane' if p=='rods' else MEMBRANE,n)
    def _bckey(self,p,n):return (p,BC_MECH,n)
    def _ckey(self,c,n):return (self.contacts[c]['route'].pre,c,n)
    def _vkey(self,p):return self._memkey(p,'V') if p in ('rods','rod_BC','AII_AC') else self._bckey(p,'V')
    def _gate(self,p,V):return self._sigmoid((V-self._par(p,'r_half'))/self._par(p,'r_slope',positive=True))
    def _release(self,p,V):return self._sigmoid((V-self._par(p,'release_half'))/self._par(p,'release_slope',positive=True))

    def _kinetics(self,c,V=None):
        rise=self._contact_par(c,'tau_rise',positive=True)
        if c.startswith('AII_'):
            lo=self._contact_par(c,'decay_minus100',positive=True);hi=self._contact_par(c,'decay_0',positive=True)
            decay=lo+(hi-lo)*self._clip((V+.1)/.1,0.,1.)
            slope=(hi-lo)/.1*((V>-.1)&(V<0.))
        else:
            decay=self._contact_par(c,'tau_decay',positive=True);slope=0.
        if not self._all(decay>rise):raise _AttemptFailure('Physical contact decay must exceed rise')
        tp=self._log(decay/rise)/(1/rise-1/decay)
        norm=self._exp(-tp/decay)-self._exp(-tp/rise)
        derivative=self._exp(-tp/decay)*tp/(decay*decay)*slope
        return rise,decay,norm,derivative,slope

    def _conductance(self,c,a,b,V=None):
        _,_,norm,_,_=self._kinetics(c,V)
        g=self.contacts[c]['coefficient']*(b-a)/norm
        if not self._finite(g) or not self._all(g>=-1e-18):raise _AttemptFailure('Negative/nonfinite physical contact conductance; no clipping')
        return self._pool(c,g)

    def _target(self,p,y):
        return self._par(p,'Vref')+self._par(p,'native_voltage_gain',positive=True)*(y[self._bckey(p,'activation')]-self._par(p,'native_reference'))

    def _affine(self,k,y,q,decay,h,U,F):
        self._check_guard();value=affine_stages(y[k],q,decay,h,counters=self.counters)
        if not self._finite(value):raise _AttemptFailure('Nonfinite affine Radau stages')
        U[k]=value;F[k]=q-decay*value;self._bump('affine_rows',y[k].shape[0])
        return value

    def _ribbon(self,p,y,U,F,h):
        V=U[self._vkey(p)];gate=self._gate(p,V);attempts=self._par(p,'rmax',positive=True)*gate
        rec=1/self._par(p,'tau_recovery',positive=True)
        x=self._affine(self._memkey(p,'x'),y,rec,rec+self._par(p,'U',positive=True)*attempts,h,U,F)
        self._affine(self._memkey(p,'release'),y,self._release(p,V)/self._par(p,'tau_release',positive=True),1/self._par(p,'tau_release',positive=True),h,U,F)
        rate=self._par(p,'Npool',positive=True)*self._par(p,'U',positive=True)*attempts*x
        if not self._all(rate>=0):raise _AttemptFailure('Negative current ribbon event rate')
        return rate

    def _gap_stages(self,y,U,F,g,h):
        names=self.inner.gap_names;initial=self._cat([y[self._vkey(p)] for p in names])
        conductance=self._zeros((3,initial.shape[0]));forcing=self._zeros(conductance.shape)
        for p,section in self.inner.gap_slices.items():
            if p=='AII_AC':
                conductance[:,section]=g
                forcing[:,section]=self.gL[p]*self._par(p,'EL')+g*self._par(p,'EE')
            else:forcing[:,section]=self.gL[p]*self._target(p,y)
        diagonal=self._gap_gL+conductance
        C=self._gap_C
        # Solve in fixed coordinate units; the actual operator retains mass form.
        scale=self._cat([self._budgets[self._vkey(p)] for p in names])
        def raw(v):return C*v+h*(self._A@(self._mv(self._gap,v,'gap')+diagonal*v))
        blocks=C[:,None,None]*self._eye()+h*self._A[None,:,:]*self._transpose(diagonal+self._gap_diagonal)[:,None,:]
        block_inverse=self._solve(blocks,self._ones((blocks.shape[0],1,1))*self._eye())
        def pre(v):return self._transpose((block_inverse@self._transpose(C*scale*v)[...,None])[...,0])/scale
        rhs=C*initial+h*(self._A@forcing)
        correction,info=gmres(lambda z:raw(scale*z)/(C*scale), (rhs-raw(self._ones((3,1))*initial))/(C*scale),precondition=pre,
                            rtol=0.,atol=.002,restart=32,maxiter=96,guard=self._check_guard,counters=self.counters,workspace=self._gap_stage_workspace)
        if not info['converged']:raise _AttemptFailure('Gap Radau GMRES failed: '+str(info))
        value=initial+scale*correction
        residual=(raw(value)-rhs)/(C*scale)
        if self._max(abs(residual))>.01:raise _AttemptFailure('Gap stage coordinate residual budget exceeded')
        slopes=(forcing-self._mv(self._gap,value,'gap')-diagonal*value)/C
        for p,section in self.inner.gap_slices.items():U[self._vkey(p)]=value[:,section];F[self._vkey(p)]=slopes[:,section]

    def _off_terms(self,p,c,V,b,a,rate,target):
        row=self.contacts[c];post=row['post'];vp=V[:,post] if V.ndim==2 else V[post]
        _,decay,norm,normprime,decayprime=self._kinetics(c,vp)
        g=row['coefficient']*(b-a)/norm
        if not self._finite(g) or not self._all(g>=-1e-18):raise _AttemptFailure('Negative/nonfinite physical contact conductance; no clipping')
        ei=self._par(p,'EI');cap=self.C[p]
        fV=(self.gL[p]*(target-V)+self._pool(c,g)*(ei-V))/cap
        fb=rate-b/decay
        dv=-self.gL[p]/cap+self._pool(c,-g-(ei[post]-vp)*g*normprime/norm)/cap
        vb=row['coefficient']*(ei[post]-vp)/(cap[post]*norm)
        bv=b*decayprime/(decay*decay)
        return fV,fb,dv,vb,bv,decay

    def _off_stages(self,p,c,y,U,F,rate,h):
        self._check_guard();vk=self._bckey(p,'V');bk=self._ckey(c,'b');ak=self._ckey(c,'a')
        post=self.contacts[c]['post'];a=U[ak];target=self._target(p,y)
        V=self._ones((3,1))*y[vk]
        decay=self._kinetics(c,V[:,post])[1];b=affine_stages(y[bk],rate,1/decay,h,counters=self.counters)
        vs=self._budgets[vk];bs=self._budgets[bk]
        def evaluate(v,w):
            terms=self._off_terms(p,c,v,w,a,rate,target)
            rv=v-y[vk]-h*(self._A@terms[0]);rb=w-y[bk]-h*(self._A@terms[1])
            score=max(self._max(abs(rv)/vs),self._max(abs(rb)/bs))
            return terms,rv,rb,score
        for iteration in range(9):
            self._check_guard();terms,rv,rb,score=evaluate(V,b)
            if score<=.01:
                U[vk],U[bk]=V,b;F[vk],F[bk]=terms[:2];return
            if iteration==8:break
            _,_,dv,vb,bv,decay=terms
            D=self._eye()-h*self._A[None,:,:]*dv.T[:,None,:]
            E=-h*self._A[None,:,:]*vb.T[:,None,:]
            G=-h*self._A[None,:,:]*bv.T[:,None,:]
            H=self._eye()+h*self._A[None,:,:]/decay.T[:,None,:]
            HG=self._solve(H,G);Hr=self._solve(H,rb.T[...,None])[...,0]
            product=E@HG;forcing=(E@Hr[...,None])[...,0]
            # Pool stage matrix entries without deleting any contact unknown.
            entries=product.transpose(1,2,0) if self.torch is None else product.permute(1,2,0)
            pooled=self._pool(c,entries)
            Schur=D-(pooled.transpose(2,0,1) if self.torch is None else pooled.permute(2,0,1))
            right=-rv+self._pool(c,forcing.T)
            dV=self._solve(Schur,right.T[...,None])[...,0].T
            db=self._solve(H,(-rb.T-(G@dV[:,post].T[...,None])[...,0])[...,None])[...,0].T
            self._bump('off_newton_iterations');self._bump('off_contact_eliminations',b.shape[1])
            accepted=False
            for step in (1.,.5,.25,.125):
                try:trial=evaluate(V+step*dV,b+step*db)
                except _AttemptFailure:continue
                if trial[3]<score:
                    V,b=V+step*dV,b+step*db;accepted=True;break
            if not accepted:raise _AttemptFailure('OFF glycine bounded line search failed')
        raise _AttemptFailure('OFF glycine Newton iteration limit')

    def _stage_rod_voltage(self,y,h,U,F,cache):
        k=self.gL['rods']/self.C['rods'];source=-self.inner.rod['k']*y['rods',RODS,'G']**self.inner.rod['n']*1e-12
        q=(self.gL['rods']*self._par('rods','EL')-self.members['rods']*(source-self._par('rods','current_anchor')))/self.C['rods']
        U[self.rod_v]=y[self.rod_v]+(-self._expm1(-self._c[:,None]*h*k))*(q/k-y[self.rod_v]);F[self.rod_v]=q-k*U[self.rod_v]

    def _stage_rod_release(self,y,h,U,F,cache):
        self._affine(self._memkey('rods','release'),y,self._release('rods',U[self.rod_v])/self._par('rods','tau_release',positive=True),1/self._par('rods','tau_release',positive=True),h,U,F)

    def _stage_trpm1(self,y,h,U,F,cache):
        c='individual_rod_RBC_contacts';row=self.contacts[c];drive=U[self._memkey('rods','release')][:,row['pre']]
        if not self._all(drive>=0):raise _AttemptFailure('Negative original glutamate drive')
        target=self._sigmoid(-self._contact_par(c,'hill',positive=True)*self._log(self._clip(drive,np.finfo(float).tiny,float('inf'))/self._contact_par(c,'K',positive=True)))
        tau=self._contact_par(c,'tau',positive=True)
        self._affine(self._ckey(c,'m'),y,target/tau,1/tau,h,U,F)

    def _stage_rod_bc_voltage(self,y,h,U,F,cache):
        c='individual_rod_RBC_contacts';row=self.contacts[c];m=U[self._ckey(c,'m')]
        g=self._pool(c,row['coefficient']*m)
        if not self._all(row['coefficient']*m>=-1e-18):raise _AttemptFailure('Negative TRPM1 conductance')
        p='rod_BC';self._affine(self._vkey(p),y,(self.gL[p]*self._par(p,'EL')+g*self._par(p,'EE'))/self.C[p],(self.gL[p]+g)/self.C[p],h,U,F)

    def _stage_ribbon(self,y,h,U,F,cache,p):
        cache['rates'][p]=self._ribbon(p,y,U,F,h)

    def _stage_ampa(self,y,h,U,F,cache):
        c='RBC_AII_AMPA';rate=cache['rates']['rod_BC'][:,self.contacts[c]['pre']]
        rise,decay,_,_,_=self._kinetics(c)
        for n,tau in (('a',rise),('b',decay)):self._affine(self._ckey(c,n),y,rate,1/tau,h,U,F)

    def _stage_gap(self,y,h,U,F,cache):
        c='RBC_AII_AMPA';g=self._conductance(c,U[self._ckey(c,'a')],U[self._ckey(c,'b')])
        self._gap_stages(y,U,F,g,h)

    def _stage_glycine_rise(self,y,h,U,F,cache,c):
        forcing=cache['rates']['AII_AC'][:,self.contacts[c]['pre']]
        self._affine(self._ckey(c,'a'),y,forcing,1/self._contact_par(c,'tau_rise',positive=True),h,U,F)

    def _stage_off(self,y,h,U,F,cache,p,c):
        forcing=cache['rates']['AII_AC'][:,self.contacts[c]['pre']]
        self._off_stages(p,c,y,U,F,forcing,h)

    def _stage_bc_terminal(self,y,h,U,F,cache,p):
        tau=self._par(p,'tau_release',positive=True);rel=self._affine(self._bckey(p,'terminal_release'),y,self._release(p,U[self._vkey(p)])/tau,1/tau,h,U,F)
        if self.inner.p[p]['adaptation_enabled']:
            scale=self._par(p,'adaptation_scale',nonnegative=True);tau=self._par(p,'tau_a_seconds',positive=True)
            adapt=self._affine(self._bckey(p,'A'),y,scale*rel/tau,1/tau,h,U,F)
            if not self._all((adapt>=-1e-10)&(adapt<=scale+1e-10)) or not self._all(scale<=.75):raise _AttemptFailure('Original BC adaptation invariant violated')
            cache['activation'][p]=rel/(1+adapt**6)
        else:cache['activation'][p]=rel

    def _stage_ac(self,y,h,U,F,cache,p):
        drive=self._zeros((3,self.inner.counts[p]))
        for r in self.inner.routes:
            if r.post==p:drive=drive+self._route(r.name,cache['activation'][r.pre])
        if not self._all(drive>=0):raise _AttemptFailure('Negative AC drive')
        target=self._sigmoid(self._par(p,'input_slope',positive=True)*(drive-self._par(p,'reference_input')))
        tau=self._par(p,'tau_A_seconds',positive=True);value=self._affine((p,AC_MECH,'h'),y,target/tau,1/tau,h,U,F)
        if not self._all((value>=-1e-10)&(value<=1+1e-10)):raise _AttemptFailure('Original generic AC invariant violated')
        cache['activation'][p]=self._par(p,'output_scale',positive=True)*value

    def _stage_rgc(self,y,h,U,F,cache,p):
        ge,gi=self._rgc_conductance(p,cache['activation'])
        self._affine((p,RGC_MECH,'V'),y,(self.gL[p]*self._par(p,'EL')+ge*self._par(p,'EE')+gi*self._par(p,'EI'))/self.C[p],(self.gL[p]+ge+gi)/self.C[p],h,U,F)

    def _half(self,y,h):
        U,F={},{};self._bump('half_steps');cache=dict(rates={},activation={})
        handlers={name:(lambda u,f,action=action,args=args:action(y,h,u,f,cache,*args))
                  for name,(action,args) in self._family_actions.items()}
        self.schedule.execute(handlers,U,F,held_state=y)
        if set(U)!=set(self.active) or set(F)!=set(self.active):raise ContractError('Radau did not solve every active original state')
        return {k:self._copy(U[k][-1]) if k in U else y[k] for k in y},U,F

    def _rgc_conductance(self,p,activation,*,signed=False):
        shape=(*next(iter(activation.values())).shape[:-1],self.inner.counts[p]);ge=self._zeros(shape);gi=self._zeros(shape)
        for r in self.inner.routes:
            if r.post!=p:continue
            if r.name not in self._gains:
                self._gains[r.name]=self._array(np.broadcast_to(np.asarray(self.inner.parameters['route_gains'][r.name],dtype=np.float64),(self.inner.counts[p],)))
            gain=self._gains[r.name]
            if not self._finite(gain) or not self._all(gain>=0):raise ContractError('Original RGC gain domain differs')
            value=self.members[p]*gain*self._route(r.name,activation[r.pre])
            if r.pre in AC:gi=gi+value
            else:ge=ge+value
        if not signed and (not self._all(ge>=0) or not self._all(gi>=0)):raise _AttemptFailure('Negative RGC conductance')
        return ge,gi

    def _filter(self,y,d,shift):
        """Solve (I-shift*J_start)e=d by the original dependency DAG."""
        e={};e[self.rod_v]=self._zeros(y[self.rod_v].shape)
        def release(p,k):
            gate=self._release(p,y[self._vkey(p)]);tau=self._par(p,'tau_release',positive=True)
            coupling=gate*(1-gate)/self._par(p,'release_slope',positive=True)/tau
            e[k]=(d[k]+shift*coupling*e[self._vkey(p)])/(1+shift/tau)
        release('rods',self._memkey('rods','release'))
        c='individual_rod_RBC_contacts';row=self.contacts[c];drive=y[self._memkey('rods','release')][row['pre']]
        safe=self._clip(drive,np.finfo(float).tiny,float('inf'))
        hill=self._contact_par(c,'hill',positive=True)
        target=self._sigmoid(-hill*self._log(safe/self._contact_par(c,'K',positive=True)))
        derivative=-hill*target*(1-target)/safe*(drive>np.finfo(float).tiny)
        tau=self._contact_par(c,'tau',positive=True);mk=self._ckey(c,'m')
        e[mk]=(d[mk]+shift*derivative*e[self._memkey('rods','release')][row['pre']]/tau)/(1+shift/tau)
        g=self._pool(c,row['coefficient']*y[mk]);dg=self._pool(c,row['coefficient']*e[mk])
        p='rod_BC';vk=self._vkey(p)
        e[vk]=(d[vk]+shift*dg*(self._par(p,'EE')-y[vk])/self.C[p])/(1+shift*(self.gL[p]+g)/self.C[p])
        def ribbon(p):
            vk=self._vkey(p);xk=self._memkey(p,'x');x=y[xk];gate=self._gate(p,y[vk])
            derivative=gate*(1-gate)/self._par(p,'r_slope',positive=True)
            U=self._par(p,'U',positive=True);rmax=self._par(p,'rmax',positive=True)
            decay=1/self._par(p,'tau_recovery',positive=True)+U*rmax*gate
            e[xk]=(d[xk]-shift*U*rmax*derivative*x*e[vk])/(1+shift*decay)
            release(p,self._memkey(p,'release'))
            constant=self._par(p,'Npool',positive=True)*U*rmax
            return constant*gate*x,constant*(derivative*x*e[vk]+gate*e[xk])
        rate,drate=ribbon('rod_BC')
        c='RBC_AII_AMPA';row=self.contacts[c];rise,decay,norm,_,_=self._kinetics(c)
        for n,tau in (('a',rise),('b',decay)):
            k=self._ckey(c,n);e[k]=(d[k]+shift*drate[row['pre']])/(1+shift/tau)
        g=self._conductance(c,y[self._ckey(c,'a')],y[self._ckey(c,'b')])
        dg=self._pool(c,row['coefficient']*(e[self._ckey(c,'b')]-e[self._ckey(c,'a')])/norm)
        C=self._gap_C;diagonal=self._copy(self._gap_gL)
        right=C*self._cat([d[self._vkey(p)] for p in self.inner.gap_names])
        section=self.inner.gap_slices['AII_AC'];diagonal[section]=diagonal[section]+g
        right[section]=right[section]+shift*dg*(self._par('AII_AC','EE')-y[self._vkey('AII_AC')])
        scale=self._cat([self._budgets[self._vkey(p)] for p in self.inner.gap_names])
        def op(v):return C*v+shift*(self._mv(self._gap,v,'gap_filter')+diagonal*v)
        diagonal_pre=C+shift*(self._gap_diagonal+diagonal)
        value,info=gmres(lambda z:op(scale*z)/(C*scale),right/(C*scale),precondition=lambda z:C*z/diagonal_pre,
                         rtol=0.,atol=.0001,restart=32,maxiter=96,guard=self._check_guard,counters=self.counters,workspace=self._gap_filter_workspace)
        if not info['converged']:raise _AttemptFailure('Filtered gap GMRES failed: '+str(info))
        for p,section in self.inner.gap_slices.items():e[self._vkey(p)]=value[section]*scale[section]
        rate,drate=ribbon('AII_AC')
        for p,c in (('midget_BC_OFF','AII_FMB_glycine'),('diffuse_BC_OFF','AII_DB1_glycine')):
            row=self.contacts[c];post=row['post'];pre=row['pre'];ak=self._ckey(c,'a');bk=self._ckey(c,'b');vk=self._vkey(p)
            e[ak]=(d[ak]+shift*drate[pre])/(1+shift/self._contact_par(c,'tau_rise',positive=True))
            _,_,dv,vb,bv,decay=self._off_terms(p,c,y[vk],y[bk],y[ak],rate[pre],self._target(p,y))
            H=1+shift/decay;E=-shift*vb;G=-shift*bv
            rb=d[bk]+shift*drate[pre]
            rv=d[vk]-shift*self._pool(c,vb*e[ak])
            schur=1-shift*dv-self._pool(c,E*G/H)
            if not self._finite(schur) or not self._all(schur!=0):raise _AttemptFailure('Singular OFF filtered Schur system')
            e[vk]=(rv-self._pool(c,E*rb/H))/schur
            e[bk]=(rb-G*e[vk][post])/H
        activation,dactivation={},{}
        for p in BC:
            rk=self._bckey(p,'terminal_release');ak=self._bckey(p,'A');release(p,rk)
            rel=y[rk]
            if self.inner.p[p]['adaptation_enabled']:
                tau=self._par(p,'tau_a_seconds',positive=True);scale=self._par(p,'adaptation_scale',nonnegative=True)
                e[ak]=(d[ak]+shift*scale*e[rk]/tau)/(1+shift/tau)
                A=y[ak];denominator=1+A**6
                activation[p]=rel/denominator
                dactivation[p]=e[rk]/denominator-6*rel*A**5/(denominator*denominator)*e[ak]
            else:activation[p]=rel;dactivation[p]=e[rk]
        for p in AC:
            drive=self._zeros((self.inner.counts[p],));ddrive=self._zeros(drive.shape)
            for r in self.inner.routes:
                if r.post==p:
                    # One sparse pass carries both the value and its tangent.
                    both=self._route(r.name,self._stack((activation[r.pre],dactivation[r.pre])))
                    drive=drive+both[0];ddrive=ddrive+both[1]
            gate=self._sigmoid(self._par(p,'input_slope',positive=True)*(drive-self._par(p,'reference_input')))
            tau=self._par(p,'tau_A_seconds',positive=True);hk=(p,AC_MECH,'h')
            e[hk]=(d[hk]+shift*gate*(1-gate)*self._par(p,'input_slope',positive=True)*ddrive/tau)/(1+shift/tau)
            activation[p]=self._par(p,'output_scale',positive=True)*y[hk]
            dactivation[p]=self._par(p,'output_scale',positive=True)*e[hk]
        for p in RGC:
            ge,gi=self._rgc_conductance(p,activation);dge,dgi=self._rgc_conductance(p,dactivation,signed=True)
            vk=(p,RGC_MECH,'V');V=y[vk]
            forcing=(dge*(self._par(p,'EE')-V)+dgi*(self._par(p,'EI')-V))/self.C[p]
            e[vk]=(d[vk]+shift*forcing)/(1+shift*(self.gL[p]+ge+gi)/self.C[p])
        if set(e)!=set(self.active) or any(not self._finite(v) for v in e.values()):
            raise _AttemptFailure('Incomplete/nonfinite filtered inner estimate')
        self._bump('filtered_solves')
        return e

    def _fresh_rhs(self,t,y):
        self._check_guard()
        if self.original_rhs is not None:
            packed=self._cat([y[k] for k in self.inner.state_keys])
            if hasattr(self.original_rhs,'evaluate_into'):
                result=self.original_rhs.evaluate_into(t,packed,self._zeros(packed.shape))
            else:result=self.original_rhs(t,packed)
            if (type(result)!=type(packed) or result.dtype!=packed.dtype or result.shape!=packed.shape
                    or self.torch is not None and (result.device!=packed.device or result.requires_grad)
                    or not self._finite(result)):
                raise ContractError('Original RHS callback returned a different layout or nonfinite values')
            self._bump('fresh_original_rhs_calls')
            return {k:self._copy(result[self.inner.slices[k]]) for k in self.active}
        host={k:self._cpu(v) for k,v in y.items()}
        if self.inner.compiled_rhs is None:result=self.inner.rhs(t,host)
        else:result=self.inner.unpack(self.inner.compiled_rhs(t,self.inner.pack(host)))
        self._bump('fresh_original_rhs_calls')
        return {k:self._array(result[k]) for k in self.active}

    def _estimate(self,y,F,f0,h):
        d={k:h*(self._gamma*f0[k]+(self._bhat-self._b)@F[k]) for k in self.active}
        d[self.rod_v]=self._zeros(y[self.rod_v].shape)
        return self._filter(y,d,self._gamma*h)

    def _representations(self,t0,t1,y0,ym,y1,f0,fm,f1):
        from .continuous_state import QuinticHermite, DiscreteHold
        result={}
        for k in self.inner.state_keys:
            y=self._cpu(y0[k]);ids=np.arange(self.inner.slices[k].start,self.inner.slices[k].stop,dtype=np.int64)
            if k in self.held:rep=DiscreteHold(t0=t0,t1=t1,state_ids=ids,y0=y)
            else:
                rep=QuinticHermite(t0=t0,t1=t1,state_ids=ids,y0=y,ym=self._cpu(ym[k]),y1=self._cpu(y1[k]),f0=self._cpu(f0[k]),fm=self._cpu(fm[k]),f1=self._cpu(f1[k]))
            result[k]=rep
        return result

    def advance(self,state,t0,t1,*,rtol,atol):
        """Advance one original event-free interval, or fail without a commit."""
        from .continuous_state import AcceptedSeries
        self.inner.validate(state)
        if (not math.isfinite(t0) or not math.isfinite(t1) or t1<=t0 or t1-t0>1.00000001e-4
                or not np.isscalar(rtol) or not math.isfinite(rtol) or rtol<=0
                or not np.isscalar(atol) or not math.isfinite(atol) or atol<=0):
            raise ContractError('Radau requires positive scalar tolerances and an event-free interval at most 100us')
        self._check_guard();started=perf_counter();before_counts=dict(self.counters)
        y={k:self._array(v) for k,v in state.items()}
        if self._weights is None:self._weights={k:atol+rtol*abs(v) for k,v in y.items()}
        if self._budgets is None:self._budgets={k:self._copy(v) for k,v in self._weights.items()}
        # Publish the diagnostic estimate sum only after the whole call succeeds.
        previous=getattr(self,'_temporal_used',None)
        used={k:self._zeros(y[k].shape) if previous is None else self._copy(previous[k]) for k in self.active}
        f0=self._fresh_rhs(t0,y);pieces={k:[] for k in self.inner.state_keys}
        times=[float(t0)];now=float(t0);h=self.next_step;attempts=0;accepted=0;max_eta=0.;last_failure=None
        window_end=t0+1e-4 if self._window_end is None else self._window_end
        window_attempts=self._window_attempts;window_accepts=self._window_accepts
        while now<t1:
            self._check_guard()
            while now>=window_end:
                window_end+=1e-4;window_attempts=window_accepts=0
            if window_attempts>=127 or window_accepts>=64:raise ContractError('Radau original interval attempt/acceptance limit exceeded')
            attempts+=1;window_attempts+=1
            boundary=min(t1,window_end)
            remaining=boundary-now;step=min(h,remaining);end=float(boundary) if step==remaining else now+step
            step=end-now
            if now+step/2==now or now+step/2==end:raise ContractError('Radau step is below time resolution')
            try:
                middle,U0,F0=self._half(y,step/2)
                fm=self._fresh_rhs(now+step/2,middle)
                candidate,U1,F1=self._half(middle,step/2)
                f1=self._fresh_rhs(end,candidate)
                e0=self._estimate(y,F0,f0,step/2);e1=self._estimate(middle,F1,fm,step/2)
                error={k:abs(e0[k])+abs(e1[k]) for k in self.active}
                square=sum(self._sum((v/self._weights[k])**2) for k,v in error.items())
                rms=math.sqrt(square/self.inner.nstate)
                coordinate=0.
                for k in self.active:
                    coordinate=max(coordinate,self._max(error[k]/self._weights[k]))
                eta=max(rms,coordinate)
                if not math.isfinite(eta):raise _AttemptFailure('Nonfinite filtered estimator')
                if eta>1:raise _AttemptFailure('Filtered estimator coordinate/RMS budget exceeded: '+str(eta))
            except _AttemptFailure as exc:
                self._bump('rejected_steps');last_failure=str(exc)
                if step<=1e-4/64:raise ContractError('Radau bounded refinement failed: '+last_failure) from exc
                level=min(6,max(0,int(math.ceil(math.log2(1e-4/(step/2))))))
                h=1e-4/(2**level)
                continue
            segments=self._representations(now,end,y,middle,candidate,f0,fm,f1)
            for k,rep in segments.items():pieces[k].append(rep)
            for k in self.active:used[k]=used[k]+error[k]
            y=candidate;f0=f1;now=end;times.append(end);accepted+=1;window_accepts+=1;max_eta=max(max_eta,eta)
            factor=min(2.,max(.2,.8*max(eta,np.finfo(float).tiny)**(-.25)))
            desired=min(1e-4,step*factor)
            level=min(6,max(0,int(math.ceil(math.log2(1e-4/desired)))))
            h=1e-4/(2**level)
        series={k:AcceptedSeries(v) for k,v in pieces.items()}
        final={k:self._cpu(v) for k,v in y.items()}
        def evaluate(t,side):return {k:rep.evaluate(t,side) for k,rep in series.items()}
        representations={k:[(np.arange(self.inner.sizes[k]),rep)] for k,rep in series.items()}
        self._temporal_used=used
        self.next_step=h;self._window_end=window_end
        self._window_attempts=window_attempts;self._window_accepts=window_accepts
        delta_counts={k:v-before_counts.get(k,0) for k,v in self.counters.items()}
        metrics=dict(delta_counts,method='structured_inner_radau_iia5_two_halfsteps',steps=accepted,
                     attempts=attempts,wall_seconds=perf_counter()-started,precision='float64',
                     original_full_dimension=self.inner.nstate,active_dimension=sum(self.inner.sizes[k] for k in self.active),
                     held_outside_stage_workspace=True,all_contact_states_retained=True,
                     scc_schedule_identity=self.schedule.identity,execution_family_order=self.schedule.order,
                     gmres_owned_workspace_bytes=self._gap_stage_workspace.owned_bytes+self._gap_filter_workspace.owned_bytes,
                     gmres_owned_host_bytes=self._gap_stage_workspace.host_bytes+self._gap_filter_workspace.host_bytes,
                     gmres_owned_device_bytes=self._gap_stage_workspace.device_bytes+self._gap_filter_workspace.device_bytes,
                     max_step_seconds=1e-4,min_nonboundary_step_seconds=1e-4/64,
                     maximum_filtered_estimator_ratio=max_eta,last_rejected_reason=last_failure,
                     error_scale='frozen_original_state',embedded_estimate_sum_is_diagnostic=True,
                     uniform_error_certificate='not_established',algebraic_inverse_bound='not_established',
                     device=str(self.device),dense_export='owned_cpu_packets',
                     rod_voltage_stages='analytic_constant_forcing',rod_voltage_dense='quintic_hermite')
        return InnerAdvance(self.inner,float(t0),float(t1),final,evaluate,representations,metrics,tuple(times))





    def controller_state(self):
        """Owned checkpoint arrays in original packed order, on this backend."""
        def packed(values):
            if values is None:return None
            return self._cat([values[k] for k in self.inner.state_keys])
        used=getattr(self,'_temporal_used',None)
        ledger=None if used is None else self._cat([used[k] if k in used else self._zeros((self.inner.sizes[k],)) for k in self.inner.state_keys])
        return dict(identity=dict(self.identity),next_step=self.next_step,
                    window_end=self._window_end,window_attempts=self._window_attempts,
                    window_accepts=self._window_accepts,statistics=dict(self.counters),
                    error_weights=packed(self._weights),budgets=packed(self._budgets),
                    error_ledger=ledger)

    def restore_controller(self,state):
        """Validate the entire checkpoint before replacing controller ownership."""
        required={'identity','next_step','window_end','window_attempts','window_accepts',
                  'statistics','error_weights','budgets','error_ledger'}
        if not isinstance(state,dict) or set(state)!=required or state['identity']!=self.identity:
            raise ContractError('Inner Radau controller identity or fields differ')
        step=state['next_step'];end=state['window_end']
        if step not in tuple(1e-4/(2**i) for i in range(7)) or end is not None and not math.isfinite(end):
            raise ContractError('Inner Radau restored step/window differs')
        for name,limit in (('window_attempts',127),('window_accepts',64)):
            if type(state[name]) is not int or not 0<=state[name]<=limit:
                raise ContractError('Inner Radau restored attempt counters differ')
        replacements={}
        for name,current in (('error_weights',self._weights),('budgets',self._budgets)):
            value=state[name]
            if value is not None:
                if self.torch is not None and isinstance(value,self.torch.Tensor):
                    value=value.detach().cpu().numpy()
                value=self._fixed(value,name)
            if current is not None and (value is None or any(not self._all(current[k]==value[k]) for k in self.inner.state_keys)):
                raise ContractError('Inner Radau restored frozen scale differs')
            replacements[name]=value
        ledger=state['error_ledger'];used=None
        if ledger is not None:
            if self.torch is not None and isinstance(ledger,self.torch.Tensor):ledger=ledger.detach().cpu().numpy()
            ledger=np.asarray(ledger)
            if ledger.dtype!=np.float64 or ledger.shape!=(self.inner.nstate,) or not np.isfinite(ledger).all() or np.any(ledger<0):
                raise ContractError('Inner Radau restored estimator ledger differs')
            if any(np.any(ledger[self.inner.slices[k]]!=0) for k in self.held) or np.any(ledger[self.inner.slices[self.rod_v]]!=0):
                raise ContractError('Held/exact states cannot carry truncation estimates')
            used={k:self._array(ledger[self.inner.slices[k]]) for k in self.active}
        statistics=state['statistics']
        if not isinstance(statistics,dict) or any(not isinstance(k,str) or type(v) is not int or v<0 for k,v in statistics.items()):
            raise ContractError('Inner Radau restored statistics differ')
        self.next_step,self._window_end=step,end
        self._window_attempts,self._window_accepts=state['window_attempts'],state['window_accepts']
        self._weights,self._budgets=replacements['error_weights'],replacements['budgets']
        self._temporal_used=used;self.counters=dict(statistics)

    def close(self):
        """Release owned reductions; the supplied original RHS remains borrowed."""
        if self._device_reductions is not None:self._device_reductions.close()
