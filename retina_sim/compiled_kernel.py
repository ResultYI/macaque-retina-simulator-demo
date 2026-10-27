"""Float64 flat RHS for a verified subset of the existing component factories.

Only continuous evaluation is lowered. The SciPy solver, causal transitions,
delay publication, observations, PCG64 draws and checkpoint API stay in Python.
No fastmath, parallel reduction, persistent signal cache or parameter defaults.
"""
from dataclasses import dataclass
from array import array
from pathlib import Path
import hashlib,time,threading,os
import numpy as np
from .contracts import ContractError,canonical_hash
from .canonical_stream import CanonicalArray

IMPLEMENTATIONS={
    # Reviewed 2026-10-21: retained native branches plus optional BC release-
    # driven A, effective output and unchanged V/r lowering; continuous generic AC h. RGC retained.
    'vh2005.py':'60c1091971527216961df8a49b40e3dad8125f44de1d82d4d9374afcd4462858',
    'inner.py':'32c469e6a1de2c75a28b1bf6c6d9fcac8c6942f0391386d60608b95688bac0e3',
    'rod_pathway.py':'5601dd757e22177fb02ebd72a4305ea145f8ddd05aa130822fe43f95b27c862c',
    'bc_gain_control.py':'db45db6c6f107b1655ab6c26bcd96643ae0afec3d40fdbd9434de30755888e01',
    'ac_inhibition.py':'1e170f21f1617eb953116905d2fefdfdeaa9dd3b697e630eab08cf9d20301873',
    'physical.py':'6bd9ae8413388086ce28f7b55489944964b283dc66bea8ec5159f9d4838c0813'}
SUPPORTED={'VanHateren2005NativeConeH1','HistoryBipolarTerminal','ResidualAC','EffectiveInhibitoryAC',
    'ResidualBC_ON','ResidualBC_OFF','SynapticConductance','HistoryConductanceRGC',
    'Chen2024PrimateRodCurrent','ReceptorConductanceMembrane','ExplicitPhysicalUnitConversion',
    'RodContactTRPM1','DoubleExponentialChemicalSynapse'}
ADD,SUB,MUL,DIV,POW,SIGMOID,LOG,EXP,MINIMUM,MAXIMUM,CSR,TAKE,SCATTER,ZERO,FINITE,NONNEG,POSITIVE=range(17)
POSITIVE_MASK=17
NATIVE_SIGNALS,NATIVE_CONE,NATIVE_HC,ROW_CSR=18,19,20,21

class UnsupportedKernel(ContractError):pass

def _execute(y,x,work,program,weights,indices,pointers,derivatives):
    """Numba target: deterministic instruction/CSR order, fresh result buffer."""
    # Validate while copying, in the original state-then-input order. Every
    # address is overwritten on the next call, including after a failed check.
    for k in range(len(y)):
        value=y[k]
        if not np.isfinite(value):raise ValueError('Nonfinite compiled state/input')
        work[k]=value
    for k in range(len(x)):
        value=x[k]
        if not np.isfinite(value):raise ValueError('Nonfinite compiled state/input')
        work[len(y)+k]=value
    for k in range(len(program)):
        code,dest,n,a,an,b,bn=program[k]
        if code==NATIVE_SIGNALS:
            _native_signals(work,n,a,b)
        elif code==NATIVE_CONE:
            _native_cone(work,n,a,b)
        elif code==NATIVE_HC:
            _native_hc(work,n,a,b)
        elif code==ROW_CSR:
            _native_csr(work,n,dest,a,b,bn,weights,indices,pointers)
        elif code==CSR:
            for i in range(n):
                total=0.
                for j in range(pointers[b+i],pointers[b+i+1]):total+=weights[bn+j]*work[a+indices[bn+j]]
                work[dest+i]=total
        elif code==TAKE:
            for i in range(n):work[dest+i]=work[a+indices[b+i]]
        elif code==SCATTER:
            for i in range(n):work[dest+indices[b+i]]+=work[a+i]
        elif code==ZERO:
            for i in range(n):work[dest+i]=0.
        elif code==FINITE or code==NONNEG or code==POSITIVE:
            for i in range(n):
                v=work[a+i]
                if not np.isfinite(v):raise ValueError('Nonfinite compiled signal',k)
                if code==NONNEG and v<0:raise ValueError('Negative compiled signal',k)
                if code==POSITIVE and v<=0:raise ValueError('Nonpositive compiled signal',k)
        else:
            for i in range(n):
                av=work[a+(i if an>1 else 0)];bv=work[b+(i if bn>1 else 0)]
                if code==ADD:v=av+bv
                elif code==SUB:v=av-bv
                elif code==MUL:v=av*bv
                elif code==DIV:v=av/bv
                elif code==POW:v=av**bv
                elif code==SIGMOID:v=1./(1.+np.exp(-av))
                elif code==LOG:v=np.log(av)
                elif code==EXP:v=np.exp(av)
                elif code==MINIMUM:v=min(av,bv)
                elif code==MAXIMUM:v=max(av,bv)
                elif code==POSITIVE_MASK:v=1. if av>0 else 0.
                else:raise ValueError('Unknown compiled opcode')
                work[dest+i]=v
    result=np.empty_like(y)
    for dest,source,n in derivatives:
        for i in range(n):
            v=0. if source<0 else work[source+i]
            if not np.isfinite(v):raise ValueError('Nonfinite compiled derivative')
            result[dest+i]=v
    return result

_dispatcher=None
def dispatcher():
    global _dispatcher,_native_signals,_native_cone,_native_hc,_native_csr
    import numba
    from .native_rows import compiled_bindings
    _native_signals,_native_cone,_native_hc,_native_csr=compiled_bindings()
    if _dispatcher is None:
        # On this Windows runtime a cached parent referencing the independently
        # compiled row kernels can retain invalid cross-process link addresses.
        # Link the small dispatcher freshly; standalone row kernels may cache.
        _dispatcher=numba.njit(cache=False,fastmath=False,boundscheck=True,error_model='numpy')(_execute)
    signature=(numba.float64[::1],numba.float64[::1],numba.float64[::1],numba.int64[:,::1],
        numba.float64[::1],numba.int64[::1],numba.int64[::1],numba.int64[:,::1])
    tick=time.perf_counter();_dispatcher.compile(signature)
    return _dispatcher,time.perf_counter()-tick,numba.__version__

@dataclass(frozen=True)
class Ref:
    plan: object
    offset: int
    n: int
    def __add__(self,x):return self.plan.binary(ADD,self,x)
    def __radd__(self,x):return self.plan.binary(ADD,x,self)
    def __sub__(self,x):return self.plan.binary(SUB,self,x)
    def __rsub__(self,x):return self.plan.binary(SUB,x,self)
    def __mul__(self,x):return self.plan.binary(MUL,self,x)
    def __rmul__(self,x):return self.plan.binary(MUL,x,self)
    def __truediv__(self,x):return self.plan.binary(DIV,self,x)
    def __rtruediv__(self,x):return self.plan.binary(DIV,x,self)
    def __pow__(self,x):return self.plan.binary(POW,self,x)
    def __neg__(self):return self*-1.

def closure(fn):
    result={}
    for name,cell in zip(fn.__code__.co_freevars,fn.__closure__ or ()):
        try:result[name]=cell.cell_contents
        except ValueError:pass  # Unused branches of the verified factory.
    return result


class ScratchWorkspace(CanonicalArray):
    """Immutable initial constants plus a private, lazily allocated RHS buffer."""
    dtype=np.dtype(np.float64)

    def __init__(self,size,constants):
        self.shape=(size,);self.constants=tuple(constants);self._values=None
        for _,block in self.constants:block.setflags(write=False)

    @property
    def nbytes(self):return self.shape[0]*self.dtype.itemsize

    def blocks(self):
        index=0
        for start in range(0,self.shape[0],8192):
            stop=min(start+8192,self.shape[0]);block=np.zeros(stop-start,dtype=self.dtype)
            while index<len(self.constants):
                offset,values=self.constants[index];end=offset+len(values)
                if offset>=stop:break
                if end>start:
                    left,right=max(start,offset),min(stop,end)
                    block[left-start:right-start]=values[left-offset:right-offset]
                if end>stop:break
                index+=1
            yield block

    @property
    def array(self):
        if self._values is None:
            values=np.zeros(self.shape,dtype=self.dtype)
            for offset,block in self.constants:values[offset:offset+len(block)]=block
            self._values=values
        return self._values

    def fork(self):
        # Every nonconstant address is written by the original instruction
        # stream before being read. Scratch is not a persisted neural state.
        return type(self)(self.shape[0],self.constants)


class WorkspaceOwner:
    @property
    def work(self):return self._scratch.array


def fork_partition(partition):
    from copy import copy
    result=copy(partition)
    if hasattr(partition,'_scratch'):
        result._scratch=partition._scratch.fork()
    else:
        result.work=partition.work.copy()
    return result

class Plan:
    native_fusion_minimum_cells=32768
    def __init__(self,network):
        self.network=network;self.compiled=network.compiled;self.program=[]
        # Keep millions of constants/indices in their final numeric types.
        self.weights=array('d');self.indices=array('q');self.pointers=array('q');self.outputs={};self.inputs={};self.native={}
        self.states={};self.external=[];self.bc_output_expressions={}
        for cid,slices in network._local_slices.items():
            self.states[cid]=[Ref(self,s.start,s.stop-s.start) for s in slices]
        self.nstate=sum(s.shape[0] for s in network.state_specs)
        self._nvalues=self.nstate;self._values=None;self._constants=[];self._workspace=None
        for e in self.compiled.spec.external_inputs:
            port=self.compiled.components[e.target_component].inputs[e.target_port]
            r=self.allocate(len(port.cell_ids));self.inputs[(e.target_component,e.target_port)]=r;self.external.append((e,port))
        self.nexternal=self._nvalues-self.nstate
    @property
    def values(self):
        if self._values is None:
            self._values=self.workspace().array
        return self._values
    def workspace(self):
        if self._workspace is None:
            self._workspace=ScratchWorkspace(self._nvalues,self._constants)
            self._constants=[]
        return self._workspace
    def fused_native(self,cid):
        return self.states[cid][0].n>=self.native_fusion_minimum_cells
    def native_instruction(self,opcode,n,layout,parameters):
        a=self.ref(layout);b=self.ref(parameters)
        self.program.append((opcode,0,n,a.offset,a.n,b.offset,b.n))
    def allocate(self,n):
        if self._workspace is not None:raise UnsupportedKernel('Plan already materialized')
        r=Ref(self,self._nvalues,n);self._nvalues+=n;return r
    def ref(self,x):
        if isinstance(x,Ref):return x
        a=np.asarray(x,dtype=np.float64).ravel();r=self.allocate(len(a))
        self._constants.append((r.offset,a.copy()));return r
    def binary(self,op,a,b):
        a=self.ref(a);b=self.ref(b);n=max(a.n,b.n)
        if a.n not in (1,n) or b.n not in (1,n):raise UnsupportedKernel('Nonbroadcast compiled expression')
        r=self.allocate(n);self.program.append((op,r.offset,n,a.offset,a.n,b.offset,b.n));return r
    def unary(self,op,a):return self.binary(op,a,0.)
    def check(self,a,kind=FINITE):
        self.program.append((kind,0,a.n,a.offset,0,0,0));return a
    def zero(self,n):
        r=self.allocate(n);self.program.append((ZERO,r.offset,n,0,0,0,0));return r
    def csr(self,m,a,*,parallel_rows=False):
        m=m.tocsr();r=self.allocate(m.shape[0]);ip=len(self.pointers);dp=len(self.weights)
        self.pointers.frombytes(np.asarray(m.indptr,dtype=np.int64).tobytes())
        self.weights.frombytes(np.asarray(m.data,dtype=np.float64).tobytes())
        self.indices.frombytes(np.asarray(m.indices,dtype=np.int64).tobytes())
        self.program.append((ROW_CSR if parallel_rows else CSR,r.offset,r.n,a.offset,a.n,ip,dp));return r
    def take(self,a,index):
        r=self.allocate(len(index));ip=len(self.indices);self.indices.frombytes(np.asarray(index,dtype=np.int64).tobytes())
        # weights and column indices share offsets for CSR, so keep both aligned.
        self.weights.extend(array('d',[0.])*len(index));self.program.append((TAKE,r.offset,r.n,a.offset,a.n,ip,0));return r
    def scatter(self,target,a,index):
        ip=len(self.indices);self.indices.frombytes(np.asarray(index,dtype=np.int64).tobytes());self.weights.extend(array('d',[0.])*len(index))
        self.program.append((SCATTER,target.offset,len(index),a.offset,a.n,ip,0))
    def gather(self,cid,name):
        key=(cid,name)
        if key in self.inputs:return self.inputs[key]
        c=self.compiled.components[cid];target=c.inputs[name];r=self.ref(np.zeros(len(target.cell_ids)))
        links=self.network._incoming.get(key,())
        if not links:raise UnsupportedKernel('Unbound compiled input')
        for link in links:
            source=self.outputs.get((link.specification.source_component,link.specification.source_port))
            if source is None:raise UnsupportedKernel('Compiled order/source mismatch')
            r=r+self.csr(link.matrix,source)
        self.check(r)
        if target.meaning in ('effective_synaptic_drive','effective_synaptic_activation','excitatory_conductance','inhibitory_conductance'):self.check(r,NONNEG)
        self.inputs[key]=r;return r
    def native_signals(self,cid):
        if cid in self.native:return self.native[cid]
        m=self.compiled.components[cid].native_model;p=m.p
        R,E,X,C,V,gi,Vbar,z1,z2,h=self.states[cid]
        self.check(gi,POSITIVE);self.check(Vbar,POSITIVE);self.check(V,NONNEG)
        if self.fused_native(cid):
            bh=self.csr(m.B,h,parallel_rows=True)
            values=tuple(self.allocate(X.n) for _ in range(4))
            self.native_instruction(NATIVE_SIGNALS,X.n,[r.offset for r in (X,V,gi,Vbar,bh,*values)],
                                    [p[k] for k in ('n_X','V_I','mu','V_k','V_n')])
            self.native[cid]=values;return values
        aI=(Vbar/p['V_I'])**p['mu'];q=self.unary(SIGMOID,(V-self.csr(m.B,h)-p['V_k'])/p['V_n'])
        values=(X**p['n_X'],aI,q,q/aI);self.native[cid]=values;return values
    def chemical(self,cid):
        c=self.compiled.components[cid];o=closure(c.observe);d=closure(c.derivative)
        if not c.state_specs:return None,None,o,d
        k=closure(d['kinetics']);rise=self.ref(k['rise'])
        if not k['voltage_decay']:return self.ref(k['decay']),self.ref(k['Z']),o,d
        v=self.gather(cid,'postsynaptic_voltage')
        tau=self.ref(k['low'])+(self.ref(k['high'])-self.ref(k['low']))*self.binary(MINIMUM,1.,self.binary(MAXIMUM,0.,(v+.1)/.1))
        tp=self.unary(LOG,tau/rise)/(1/rise-1/tau)
        z=self.unary(EXP,-tp/tau)-self.unary(EXP,-tp/rise)
        return tau,z,o,d
    def observe(self,cid,port):
        c=self.compiled.components[cid];m=c.mechanism;s=self.states[cid];o=closure(c.observe)
        if m=='VanHateren2005NativeConeH1':
            if port in ('q','release','Ios'):return self.native_signals(cid)[{'Ios':0,'q':2,'release':3}[port]]
            return s[[x.state_name for x in c.state_specs].index(port)]
        if m=='HistoryBipolarTerminal':
            count=c.native_model.history_state_count;rel=s[count+1];control=c.native_model.bc_output_gain_control
            if port=='unadapted_release':return rel
            if port=='adaptation_activity':return s[count+2] if control.enabled else self.zero(rel.n)
            if port in ('activation','effective_release','output_gain'):
                # Share expression refs inside this plan, never gain values across
                # RHS calls: all instructions still read the current y each time.
                if cid not in self.bc_output_expressions:
                    if control.enabled:
                        self.check(s[count+2]+1e-10,NONNEG);self.check(control.scale+1e-10-s[count+2],NONNEG)
                    gain=1/(1+s[count+2]**6) if control.enabled else self.ref(np.ones(rel.n))
                    self.bc_output_expressions[cid]={'gain':gain}
                shared=self.bc_output_expressions[cid]
                if port=='output_gain':return shared['gain']
                if 'effective_release' not in shared:
                    shared['effective_release']=rel*shared['gain'] if control.enabled else rel
                return shared['effective_release']
            if port=='generator_activation':return s[o['count']-1]
            if port=='voltage':return s[count]
            if port=='glycine_conductance':return self.gather(cid,'inhibition')
            return self.ref(np.zeros(len(c.outputs[port].cell_ids)))
        if m in ('ResidualAC','ResidualBC_ON','ResidualBC_OFF'):return s[-1]
        if m=='EffectiveInhibitoryAC':
            h=s[0];self.check(h+1e-10,NONNEG);self.check(1+1e-10-h,NONNEG)
            if port=='activation':return self.ref(c.native_model.output_scale)*h
            if port=='target_inhibition':
                drive=self.gather(cid,'drive');self.check(drive,NONNEG)
                return self.unary(SIGMOID,self.ref(c.native_model.input_slope)*(drive-self.ref(c.native_model.reference_input)))
            return h
        if m=='SynapticConductance':
            a=self.gather(cid,'activation');self.check(a,NONNEG);return self.ref(o['gain'])*a
        if m=='HistoryConductanceRGC':return self.gather(cid,'excitation' if port=='gE' else 'inhibition') if port in ('gE','gI') else s[0]
        if m=='Chen2024PrimateRodCurrent':
            if port=='current':return -c.native_model.p['k']*s[2]**c.native_model.p['n']
            return s[[x.state_name for x in c.state_specs].index(port)]
        if m=='ExplicitPhysicalUnitConversion':
            a=c.inputs['input'].unit;b=c.outputs['output'].unit
            factors={('pA','A'):1e-12,('A','pA'):1e12,('mV','V'):1e-3,('V','mV'):1e3,('ms','s'):1e-3,('s','ms'):1e3,('nS','S'):1e-9,('S','nS'):1e9,('pF','F'):1e-12,('F','pF'):1e12}
            return self.gather(cid,'input')*factors[(a,b)]
        if m=='ReceptorConductanceMembrane':
            if port=='voltage':return s[0]
            if port in ('release','activation'):return s[1]
            if port=='gap_current':return self.ref(np.zeros(s[0].n))
            if port=='glycine_conductance':return self.gather(cid,'inhibition')
            if port=='event_rate':return self.ref(o['pool'])*self.ref(o['U'])*self.ref(o['rmax'])*self.unary(SIGMOID,(s[0]-self.ref(o['rh']))/self.ref(o['rs']))*s[2]
            return s[[x.state_name for x in c.state_specs].index(port)]
        if m=='RodContactTRPM1':return self.ref(o['gain'])*s[0] if port=='conductance' else s[0]
        if m=='DoubleExponentialChemicalSynapse':
            tau,z,o,d=self.chemical(cid)
            if not s or not o['enabled']:return self.ref(np.zeros(len(c.outputs[port].cell_ids)))
            value=self.ref(o['contacts'])*self.ref(o['peak'])*(s[1]-s[0])/z
            self.check(value+1e-18,NONNEG);return value
        raise UnsupportedKernel('Unsupported compiled observation: '+m)
    def derivative(self,cid,gaps):
        c=self.compiled.components[cid];m=c.mechanism;s=self.states[cid];d=closure(c.derivative)
        if m=='VanHateren2005NativeConeH1':
            p=c.native_model.p;R,E,X,C,V,gi,Vbar,z1,z2,h=s;L=self.gather(cid,'light')
            self.check(L,NONNEG);self.check(X,NONNEG);self.check(C,NONNEG)
            if self.fused_native(cid):
                I,aI,q,r=self.native_signals(cid)
                rows=[self.allocate(state.n) for state in s]
                self.native_instruction(NATIVE_CONE,R.n,[a.offset for a in (R,E,X,C,V,gi,Vbar,z1,L,I,r,*rows[:8])],
                    [p[k] for k in ('tau_R','tau_E','a_C','n_C','c_beta','k_beta','tau_C','tau_m','a_is','gamma','tau_is','tau_a','g_t','tau_1')])
                av=self.csr(c.native_model.A,Vbar,parallel_rows=True);az=self.csr(c.native_model.A,z1,parallel_rows=True)
                self.native_instruction(NATIVE_HC,h.n,[a.offset for a in (av,az,z2,h,*rows[8:])],
                    [p[k] for k in ('V_I','mu','tau_2','tau_h')])
                return rows
            I,aI,q,r=self.native_signals(cid);pooled=(self.csr(c.native_model.A,Vbar)/p['V_I'])**p['mu']
            alpha=1/(1+(p['a_C']*C)**p['n_C']);beta=p['c_beta']+p['k_beta']*E
            rows=((L-R)/p['tau_R'],(R-E)/p['tau_E'],alpha-beta*X,(I-C)/p['tau_C'],(I/gi-V)/p['tau_m'],(p['a_is']*V**p['gamma']-gi)/p['tau_is'],(V-Vbar)/p['tau_a'],(p['g_t']*r-z1)/p['tau_1'],(self.csr(c.native_model.A,z1)-z2)/(p['tau_2']*pooled),(z2-h)/(p['tau_h']*pooled))
            return [1000*v for v in rows]
        if m in ('ResidualAC','ResidualBC_ON','ResidualBC_OFF','Chen2024PrimateRodCurrent'):return [None]*len(s)
        if m=='EffectiveInhibitoryAC':
            return [(self.outputs[(cid,'target_inhibition')]-s[0])/c.native_model.tau_A_seconds]
        if m=='HistoryBipolarTerminal':
            a=lambda k:self.ref(d[k]);count=d['count'];V,rel=s[count:count+2];target=a('Vref')+a('gain')*(s[count-1]-a('aref'))
            current=a('gL')*(target-V)+gaps[cid]
            if d['polarity']=='OFF':current=current+self.gather(cid,'inhibition')*(a('EI')-V)
            rows=[None]*count+[current/a('C'),(self.unary(SIGMOID,(V-a('half'))/a('slope'))-rel)/a('tau')]
            control=c.native_model.bc_output_gain_control
            if control.enabled:rows.append((control.scale*rel-s[count+2])/control.tau_a_seconds)
            return rows
        if m=='HistoryConductanceRGC':
            p=c.native_model;V=s[0];e=self.gather(cid,'excitation');i=self.gather(cid,'inhibition')
            self.check(e,NONNEG);self.check(i,NONNEG)
            return [(self.ref(p.gL)*(self.ref(p.EL)-V)+e*(self.ref(p.EE)-V)+i*(self.ref(p.EI)-V))/self.ref(p.C)]
        if m=='ReceptorConductanceMembrane':
            a=lambda k:self.ref(d[k]);V,rel=s[:2];current=a('gL')*(a('EL')-V)+gaps[cid]
            rows=[None,(self.unary(SIGMOID,(V-a('half'))/a('slope'))-rel)/a('tau_release')]
            if d['kind']=='rod_membrane':current=current-(self.gather(cid,'current')-a('anchor'))
            elif d['kind'] in ('rod_BC_ON','AII'):
                current=current+self.gather(cid,'excitation')*(a('EE')-V)
                attempts=a('rmax')*self.unary(SIGMOID,(V-a('rh'))/a('rs'));rows.append((1-s[2])/a('rec')-a('U')*attempts*s[2])
            else:
                x=self.gather(cid,'drive');self.check(x,NONNEG)
                occupancy=self.unary(SIGMOID,a('hill')*self.unary(LOG,self.binary(MAXIMUM,x,np.finfo(float).tiny)/a('K')))
                # np.where(x==0,0,occupancy): implemented by a zero/nonzero mask.
                occupancy=occupancy*self.unary(POSITIVE_MASK,x)
                target=1-occupancy if d['kind']=='cone_BC_ON' else occupancy
                current=current+a('ge')*s[2]*(a('EE')-V);rows.append((target-s[2])/a('tau_e'))
                if d['kind']=='cone_BC_OFF':current=current+self.gather(cid,'inhibition')*(a('EI')-V)
            rows[0]=current/a('C');return rows
        if m=='RodContactTRPM1':
            target=closure(d['target']);x=self.gather(cid,'release');self.check(x,NONNEG)
            value=self.unary(SIGMOID,-self.ref(target['h'])*self.unary(LOG,self.binary(MAXIMUM,x,np.finfo(float).tiny)/self.ref(target['K'])))
            return [(value-s[0])/self.ref(d['tau'])]
        if m=='DoubleExponentialChemicalSynapse':
            if not s:return []
            rate=self.gather(cid,'event_rate');self.check(rate,NONNEG);tau,z,o,d=self.chemical(cid)
            return [-s[0]/self.ref(d['rise'])+rate,-s[1]/tau+rate]
        return []
    def build(self):
        for block in self.compiled.output_order:
            cid,port=block.nodes[0];self.outputs[(cid,port)]=self.check(self.observe(cid,port))
        for cid,c in self.compiled.components.items():
            for name in c.inputs:self.gather(cid,name)
        gaps={cid:self.zero(len(c.outputs['voltage'].cell_ids)) for cid,c in self.compiled.components.items() if hasattr(c.native_model,'membrane_capacitance')}
        for l,r,ix,iy,g in self.network._gap_plans:
            current=self.ref(g)*(self.take(self.outputs[(r,'voltage')],iy)-self.take(self.outputs[(l,'voltage')],ix))
            self.scatter(gaps[l],current,ix);self.scatter(gaps[r],-current,iy)
        result=[]
        for cid in self.compiled.components:
            rows=self.derivative(cid,gaps)
            if len(rows)!=len(self.states[cid]):raise UnsupportedKernel('Compiled derivative count mismatch')
            for state,row in zip(self.states[cid],rows):result.append((state.offset,-1 if row is None else row.offset,state.n))
        return np.array(result,dtype=np.int64).reshape(-1,3)

class CompiledRHS(WorkspaceOwner):
    def __init__(self,network):
        tick=time.perf_counter();c=network.compiled
        if any(b.cyclic for b in c.output_order):raise UnsupportedKernel('Algebraic closure retains Python reference')
        if any(x.delayed_operator is not None for x in c.links):raise UnsupportedKernel('Explicit delayed operators retain Python reference')
        missing=set(x.mechanism for x in c.components.values())-SUPPORTED
        if missing:raise UnsupportedKernel('Unsupported mechanisms: '+', '.join(sorted(missing)))
        for filename,expected in IMPLEMENTATIONS.items():
            if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()!=expected:
                raise UnsupportedKernel('Lowering must be reviewed against changed implementation: '+filename)
        plan=Plan(network);self.derivatives=plan.build();self.external=plan.external
        self._scratch=plan.workspace();self.program=np.array(plan.program,dtype=np.int64).reshape(-1,7)
        self.weights=np.asarray(plan.weights,dtype=np.float64);self.indices=np.asarray(plan.indices,dtype=np.int64);self.pointers=np.asarray(plan.pointers,dtype=np.int64)
        self.nstate=plan.nstate;self.lock=threading.Lock();self.network=network
        plan_seconds=time.perf_counter()-tick
        self.fn,compile_seconds,version=dispatcher()
        self.native_fused=bool(np.any(self.program[:,0]==NATIVE_SIGNALS))
        self.native_threads=8 if self.native_fused else None
        self.metadata={'backend':'numba_flat_rhs_v1','numba_version':version,'fastmath':False,'parallel':False,'boundscheck':True,
            'parent_dispatcher_cache':'disabled; fresh cross-kernel linking per process',
            'plan_seconds':plan_seconds,'compile_or_cache_load_seconds':compile_seconds,'instruction_count':len(self.program),
            'scratch_bytes':self._scratch.nbytes,'state_scalars':plan.nstate,'external_scalars':plan.nexternal,
            'plan_hash':canonical_hash({'program':self.program,'constants':self._scratch,'weights':self.weights,'indices':self.indices,'pointers':self.pointers,'derivatives':self.derivatives}),
            'scope':'Continuous RHS only; unchanged SciPy/event/history/observation/RNG/checkpoint machinery'}
        if self.native_fused:
            helper_sha=hashlib.sha256(Path(__file__).with_name('native_rows.py').read_bytes()).hexdigest()
            self.metadata.update(backend='numba_fused_native_rows_v1',parallel=True,parallel_scope='independent native rows only; CSR within-row reduction order unchanged',
                native_threads_maximum=self.native_threads,helper_sha256=helper_sha)
            self.metadata['plan_hash']=canonical_hash({'original_plan_hash':self.metadata['plan_hash'],'native_rows_sha256':helper_sha,'threads_maximum':self.native_threads})
        self.metadata['process_environment']={'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS')}
        self.metadata['plan_hash']=canonical_hash({'execution_plan':self.metadata['plan_hash'],'process_environment':self.metadata['process_environment']})
    def __call__(self,t,y,bundle,parameters,graph):
        if parameters is not self.network.parameters or graph is not self.network.graph:raise ContractError('Joint RHS identity mismatch')
        if not isinstance(y,np.ndarray) or y.dtype!=np.float64 or y.shape!=(self.nstate,):raise ContractError('Compiled RHS requires the original float64 state layout')
        if set(bundle.channels)!={e.channel for e,p in self.external}:raise ContractError('Compiled external channels mismatch')
        values=[]
        for external,port in self.external:
            q=bundle.channels[external.channel];port.validate(q);values.append(q.values)
        x=np.concatenate(values) if values else np.empty(0,dtype=np.float64)
        try:
            with self.lock:
                if not self.native_fused:return self.fn(y,x,self.work,self.program,self.weights,self.indices,self.pointers,self.derivatives)
                import numba
                previous=numba.get_num_threads();numba.set_num_threads(min(previous,self.native_threads))
                try:return self.fn(y,x,self.work,self.program,self.weights,self.indices,self.pointers,self.derivatives)
                finally:numba.set_num_threads(previous)
        except (ValueError,ZeroDivisionError,OverflowError) as exc:raise ContractError('Compiled RHS: '+str(exc)) from exc
