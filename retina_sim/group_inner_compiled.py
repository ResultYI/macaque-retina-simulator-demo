"""Direct group-to-existing-opcode lowering; no fine component assembly.

Only continuous CPU RHS is compiled. Original clock commits, RNG, history,
observations and checkpoints remain with GroupInnerDynamics and its runner.
"""
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import hashlib
import math
import time

import numpy as np

from .contracts import ContractError, canonical_hash
from .compiled_kernel import (Plan, Ref, dispatcher, SIGMOID, LOG, EXP, MINIMUM,
                              MAXIMUM, NONNEG, POSITIVE, FINITE, CSR, ROW_CSR,
                              TAKE, SCATTER, ZERO, ADD, SUB, MUL, DIV, POW)
from .group_time_partition import guard_source_dependencies
from .group_inner_dynamics import RODS, MEMBRANE, BC_MECH, AC_MECH, RGC_MECH
from .spatial_group_model import BC, AC, RGC


def _source_rod_current(y, start, count, k, n):
    current = np.empty(count, dtype=np.float64)
    for i in range(count):
        current[i] = -k*y[start+i]**n*1e-12
    return current


_rod_dispatcher = None


class CompiledGroupInnerRHS:
    """Owned typed instruction plan for every current inner continuous state."""
    def __init__(self, model, *, repo_root=None):
        global _rod_dispatcher
        root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
        sources = guard_source_dependencies(root)
        started = time.perf_counter()
        self.model = model
        self.nstate = model.nstate
        self._lock = Lock()
        self._calls = 0
        self._views = {}
        self.profile_enabled = False
        self.execution_profile = dict(rhs_calls=0, successful_rhs_calls=0, rhs_wall_seconds=0.,
            rod_current_calls=0, rod_current_wall_seconds=0., dispatch_wall_seconds=0.,
            state_copy_bytes=0, external_copy_bytes=0, result_write_bytes=0)
        # Plan only needs typed offsets, counts and explicit external ports;
        # it never instantiates the original factory/fine-cell runtime here.
        external = SimpleNamespace(channel='held_rod_current',target_component='group_inner',target_port='rod_current')
        port = SimpleNamespace(cell_ids=range(model.counts['rods']))
        compiled = SimpleNamespace(spec=SimpleNamespace(external_inputs=(external,)),
            components={'group_inner':SimpleNamespace(inputs={'rod_current':port})})
        network = SimpleNamespace(compiled=compiled,
            _local_slices={k:(s,) for k,s in model.slices.items()},
            state_specs=tuple(SimpleNamespace(shape=(model.sizes[k],)) for k in model.state_keys))
        p = self.plan = Plan(network)
        state = {k:p.states[k][0] for k in model.state_keys}
        constant_refs = {}
        def par(pop,name,**kw):
            k=('p',pop,name)
            if k not in constant_refs:constant_refs[k]=p.ref(model.par(pop,name,**kw))
            return constant_refs[k]
        def cp(component,name,**kw):
            k=('cp',component,name)
            if k not in constant_refs:constant_refs[k]=p.ref(model.cp(component,name,**kw))
            return constant_refs[k]
        def constant(k,value):
            if k not in constant_refs:constant_refs[k]=p.ref(value)
            return constant_refs[k]
        def mem(pop,name):return state[pop,'rod_membrane' if pop=='rods' else MEMBRANE,name]
        def bc(pop,name):return state[pop,BC_MECH,name]
        def contact(c,name):return state[model.contacts[c]['route'].pre,c,name]
        def sigmoid(x):return p.unary(SIGMOID,x)
        voltages = {pop:mem(pop,'V') for pop in ('rods','rod_BC','AII_AC')}
        voltages.update({pop:bc(pop,'V') for pop in BC})
        release = {pop:mem(pop,'release') for pop in ('rods','rod_BC','AII_AC')}
        ribbon_gates = {}
        def ribbon_gate(pop):
            # Share plan Refs only; each RHS still computes its current voltage gate.
            if pop not in ribbon_gates:
                ribbon_gates[pop]=sigmoid((voltages[pop]-par(pop,'r_half'))/par(pop,'r_slope',positive=True))
            return ribbon_gates[pop]
        rates = {}
        for pop in ('rod_BC','AII_AC'):
            rate = par(pop,'Npool',positive=True)*par(pop,'U',positive=True)*par(pop,'rmax',positive=True)*ribbon_gate(pop)*mem(pop,'x')
            rates[pop]=p.check(rate,NONNEG)
        activation = {}
        for pop in BC:
            rel,A=bc(pop,'terminal_release'),bc(pop,'A')
            if model.p[pop]['adaptation_enabled']:
                scale = model.par(pop,'adaptation_scale',nonnegative=True)
                if np.any(scale>.75):raise ContractError('Original BC adaptation scale exceeds .75')
                p.check(A+1e-10,NONNEG);p.check(p.ref(scale)+1e-10-A,NONNEG)
                activation[pop]=rel/(1+A**6)
            else:
                activation[pop]=rel
        for pop in AC:
            h=state[pop,AC_MECH,'h']
            p.check(h+1e-10,NONNEG);p.check(1+1e-10-h,NONNEG)
            activation[pop]=par(pop,'output_scale',positive=True)*h
        conductances,kinetics={},{}
        for c, row in model.contacts.items():
            coefficient=constant(('coefficient',c),row['coefficient'])
            if c=='individual_rod_RBC_contacts':
                value=coefficient*contact(c,'m')
            else:
                rise=cp(c,'tau_rise',positive=True)
                if c.startswith('AII_'):
                    V=p.take(voltages[row['route'].post],row['post'])
                    lo,hi=cp(c,'decay_minus100',positive=True),cp(c,'decay_0',positive=True)
                    decay=lo+(hi-lo)*p.binary(MINIMUM,1.,p.binary(MAXIMUM,0.,(V+.1)/.1))
                else:
                    decay=cp(c,'tau_decay',positive=True)
                p.check(decay-rise,POSITIVE)
                tp=p.unary(LOG,decay/rise)/(1/rise-1/decay)
                norm=p.unary(EXP,-tp/decay)-p.unary(EXP,-tp/rise)
                kinetics[c]=(rise,decay)
                value=coefficient*(contact(c,'b')-contact(c,'a'))/norm
            p.check(value,FINITE);p.check(value+1e-18,NONNEG)
            pooled=p.zero(model.counts[row['route'].post])
            p.scatter(pooled,value,row['post'])
            conductances[c]=pooled
        gap_vector=p.zero(sum(model.counts[n] for n in model.gap_names))
        for pop,s in model.gap_slices.items():
            p.scatter(gap_vector,voltages[pop],np.arange(s.start,s.stop,dtype=np.int64))
        gap_product=p.csr(model.gap,gap_vector)
        boundary_product=getattr(model,'gap_boundary_product',None)
        if boundary_product is not None:
            gap_product=gap_product+constant(('boundary','gap_product'),boundary_product)
        gap=-gap_product
        gap_currents={pop:Ref(p,gap.offset+s.start,s.stop-s.start) for pop,s in model.gap_slices.items()}
        derivatives={}
        for pop in ('rods','rod_BC','AII_AC'):
            V,rel=voltages[pop],release[pop]
            gL=constant(('gL',pop),model.gL[pop]);C=constant(('C',pop),model.C[pop])
            current=gL*(par(pop,'EL')-V)
            if pop in gap_currents:current=current+gap_currents[pop]
            if pop=='rods':
                source_current=p.inputs['group_inner','rod_current']
                current=current-constant(('members',pop),model.members[pop])*(source_current-par(pop,'current_anchor'))
            else:
                c='individual_rod_RBC_contacts' if pop=='rod_BC' else 'RBC_AII_AMPA'
                current=current+conductances[c]*(par(pop,'EE')-V)
                attempts=par(pop,'rmax',positive=True)*ribbon_gate(pop)
                x=mem(pop,'x')
                derivatives[pop,MEMBRANE,'x']=(1-x)/par(pop,'tau_recovery',positive=True)-par(pop,'U',positive=True)*attempts*x
            mechanism='rod_membrane' if pop=='rods' else MEMBRANE
            derivatives[pop,mechanism,'V']=current/C
            derivatives[pop,mechanism,'release']=(sigmoid((V-par(pop,'release_half'))/par(pop,'release_slope',positive=True))-rel)/par(pop,'tau_release',positive=True)
        for pop in BC:
            V,rel,A=bc(pop,'V'),bc(pop,'terminal_release'),bc(pop,'A')
            target=par(pop,'Vref')+par(pop,'native_voltage_gain',positive=True)*(bc(pop,'activation')-par(pop,'native_reference'))
            current=constant(('gL',pop),model.gL[pop])*(target-V)
            if pop in gap_currents:current=current+gap_currents[pop]
            if pop.endswith('_OFF'):
                c='AII_FMB_glycine' if pop=='midget_BC_OFF' else 'AII_DB1_glycine'
                current=current+conductances[c]*(par(pop,'EI')-V)
            derivatives[pop,BC_MECH,'V']=current/constant(('C',pop),model.C[pop])
            derivatives[pop,BC_MECH,'terminal_release']=(sigmoid((V-par(pop,'release_half'))/par(pop,'release_slope',positive=True))-rel)/par(pop,'tau_release',positive=True)
            if model.p[pop]['adaptation_enabled']:
                derivatives[pop,BC_MECH,'A']=(par(pop,'adaptation_scale',nonnegative=True)*rel-A)/par(pop,'tau_a_seconds',positive=True)
        for pop in AC:
            drive=p.zero(model.counts[pop])
            for r in model.routes:
                if r.post==pop:drive=drive+p.csr(model.mean[r.name],activation[r.pre])
            p.check(drive,NONNEG)
            target=sigmoid(par(pop,'input_slope',positive=True)*(drive-par(pop,'reference_input')))
            derivatives[pop,AC_MECH,'h']=(target-state[pop,AC_MECH,'h'])/par(pop,'tau_A_seconds',positive=True)
        for pop in RGC:
            if not model.counts[pop]:
                continue
            ge,gi=p.zero(model.counts[pop]),p.zero(model.counts[pop])
            for r in model.routes:
                if r.post!=pop:continue
                gain=np.broadcast_to(np.asarray(model.parameters['route_gains'][r.name],dtype=np.float64),(model.counts[pop],))
                if not np.isfinite(gain).all() or np.any(gain<0):raise ContractError('Original RGC gain domain differs')
                value=constant(('members',pop),model.members[pop])*constant(('gain',r.name),gain)*p.csr(model.mean[r.name],activation[r.pre])
                if r.pre in AC:gi=gi+value
                else:ge=ge+value
            p.check(ge,NONNEG);p.check(gi,NONNEG)
            V=state[pop,RGC_MECH,'V']
            derivatives[pop,RGC_MECH,'V']=(constant(('gL',pop),model.gL[pop])*(par(pop,'EL')-V)+ge*(par(pop,'EE')-V)+gi*(par(pop,'EI')-V))/constant(('C',pop),model.C[pop])
        for c,row in model.contacts.items():
            if c=='individual_rod_RBC_contacts':
                drive=p.take(release['rods'],row['pre']);p.check(drive,NONNEG)
                target=sigmoid(-cp(c,'hill',positive=True)*p.unary(LOG,p.binary(MAXIMUM,drive,np.finfo(float).tiny)/cp(c,'K',positive=True)))
                derivatives[row['route'].pre,c,'m']=(target-contact(c,'m'))/cp(c,'tau',positive=True)
            else:
                rate=p.take(rates[row['route'].pre],row['pre'])
                rise,decay=kinetics[c]
                derivatives[row['route'].pre,c,'a']=-contact(c,'a')/rise+rate
                derivatives[row['route'].pre,c,'b']=-contact(c,'b')/decay+rate
        self.program=np.ascontiguousarray(p.program,dtype=np.int64)
        self.weights=np.asarray(p.weights,dtype=np.float64)
        self.indices=np.asarray(p.indices,dtype=np.int64)
        self.pointers=np.asarray(p.pointers,dtype=np.int64)
        self.derivatives=np.ascontiguousarray([(model.slices[k].start,
            derivatives[k].offset if k in derivatives else -1,model.sizes[k]) for k in model.state_keys],dtype=np.int64)
        self._scratch=p.workspace()
        for values in (self.program,self.weights,self.indices,self.pointers,self.derivatives):values.flags.writeable=False
        self._dispatch,self.compile_seconds,self.numba_version=dispatcher()
        if _rod_dispatcher is None:
            import numba
            _rod_dispatcher=numba.njit(cache=False,fastmath=False,boundscheck=True,error_model='numpy')(_source_rod_current)
        self._rod_dispatch=_rod_dispatcher
        self._rod_start=model.slices['rods',RODS,'G'].start
        self._rod_count=model.counts['rods']
        self._rod_k,self._rod_n=float(model.rod['k']),float(model.rod['n'])
        self.identity=canonical_hash(dict(schema='retipath-compiled-group-inner-rhs-1',model=model.identity,
            source_equations=sources,implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            program=self.program,weights=self.weights,indices=self.indices,pointers=self.pointers,
            derivatives=self.derivatives,numba_version=self.numba_version,
            dispatcher='retina_sim.compiled_kernel.dispatcher',fastmath=False,parallel_reductions=False))
        self.metrics=dict(backend='compiled_group_opcode_fp64',source_model_identity=model.identity,
            instruction_count=len(self.program),workspace_bytes=self._scratch.nbytes,
            payload_bytes=sum(x.nbytes for x in (self.program,self.weights,self.indices,self.pointers,self.derivatives)),
            build_seconds=time.perf_counter()-started,jit_compile_seconds=self.compile_seconds,
            fastmath=False,discrete_events_changed=False,fine_runtime_constructed=False)

    def for_keys(self,state_keys):
        requested=tuple(state_keys)
        if (not requested or len(set(requested))!=len(requested)
                or any(k not in self.model.slices for k in requested)):
            raise ContractError('Distinct existing state keys required for compiled selective lowering')
        canonical=tuple(k for k in self.model.state_keys if k in set(requested))
        if canonical not in self._views:
            self._views[canonical]=_SelectiveRHS(self,canonical)
        return self._views[canonical]

    def _rod_current(self,y):
        started=time.perf_counter() if self.profile_enabled else 0.
        result=self._rod_dispatch(y,self._rod_start,self._rod_count,self._rod_k,self._rod_n)
        if self.profile_enabled:
            self.execution_profile['rod_current_calls']+=1
            self.execution_profile['rod_current_wall_seconds']+=time.perf_counter()-started
        return result

    def _reference_interval_rhs(self,initial):
        """Private binding for one event-free reference solve, never a state cache."""
        held=self.derivatives[list(self.model.state_keys).index(('rods',RODS,'G')),1]<0
        # Zero derivatives preserve strictly positive G at every RK/dense stage.
        # Keep the generic path for signed zero or nonphysical G, preserving its
        # original arithmetic and failure behavior rather than adding a domain.
        if not held or not np.all(initial[self._rod_start:self._rod_start+self._rod_count]>0):
            return self.rhs_flat
        current=self._rod_current(initial)
        def derivative(t,y):
            return self.rhs_flat(t,y,rod_current=current)
        return derivative

    def _run(self,t,y,cone_release,rod_current,program,derivatives,needs_rod_current=True):
        started=time.perf_counter() if self.profile_enabled else 0.
        if not math.isfinite(t):raise ContractError('Finite original RHS time required')
        y=np.asarray(y)
        if y.dtype!=np.float64 or y.shape!=(self.nstate,) or not y.flags.c_contiguous:
            raise ContractError('Contiguous FP64 original inner layout required')
        if cone_release is not None:
            light=np.asarray(cone_release)
            if light.dtype!=np.float64 or light.shape!=(self.model.counts['cones'],) or not np.isfinite(light).all() or np.any(light<0):
                raise ContractError('Original cone release layout/domain differs')
        if not needs_rod_current:
            source=np.zeros(self._rod_count,dtype=np.float64)
        elif rod_current is None:
            source=self._rod_current(y)
        else:
            source=np.asarray(rod_current)
            if source.dtype!=np.float64 or source.shape!=(self._rod_count,) or not source.flags.c_contiguous:
                raise ContractError('Contiguous FP64 original held rod current required')
        if not self._lock.acquire(blocking=False):
            raise ContractError('Compiled group scratch is already leased; use a separate instance')
        try:
            dispatched=time.perf_counter() if self.profile_enabled else 0.
            result=self._dispatch(y,source,self._scratch.array,program,self.weights,self.indices,self.pointers,derivatives)
            self._calls+=1
            if self.profile_enabled:
                self.execution_profile['successful_rhs_calls']+=1
                self.execution_profile['state_copy_bytes']+=y.nbytes
                self.execution_profile['external_copy_bytes']+=source.nbytes
                self.execution_profile['result_write_bytes']+=y.nbytes
            return result
        except (ValueError,ZeroDivisionError,FloatingPointError) as exc:
            raise ContractError('Compiled original inner equation/domain failed: '+str(exc)) from exc
        finally:
            if self.profile_enabled:
                profile=self.execution_profile
                profile['rhs_calls']+=1
                profile['rhs_wall_seconds']+=time.perf_counter()-started
                profile['dispatch_wall_seconds']+=time.perf_counter()-dispatched
            self._lock.release()

    def rhs_flat(self,t,y,cone_release=None,rod_current=None):
        return self._run(t,y,cone_release,rod_current,self.program,self.derivatives)

    __call__=rhs_flat


class _SelectiveRHS:
    """Backward-live subset of the SAME ordered instruction stream."""
    def __init__(self,parent,requested):
        self.parent,self.state_keys=parent,requested
        model=parent.model
        constants={offset for offset,_ in parent._scratch.constants}
        sizes={s.start:s.stop-s.start for s in model.slices.values()}
        sizes[model.nstate]=model.counts['rods']
        sizes.update({offset:len(values) for offset,values in parent._scratch.constants})
        definitions={}
        checks={FINITE,NONNEG,POSITIVE}
        for i,row in enumerate(parent.program):
            code,dest,n,a,an,b,bn=map(int,row)
            if code not in checks and code!=SCATTER:
                sizes[dest]=n;definitions[dest]=i
        starts=np.array(sorted(sizes),dtype=np.int64)
        def owner(offset):
            at=int(np.searchsorted(starts,offset,side='right'))-1
            if at<0:raise ContractError('Compiled work address outside typed allocation')
            start=int(starts[at])
            if not start<=offset<start+sizes[start]:raise ContractError('Compiled work address crosses allocation')
            return start
        def reads(row):
            code,dest,n,a,an,b,bn=map(int,row)
            if code==ZERO:return ()
            if code in checks:return (owner(a),)
            if code in (CSR,ROW_CSR,TAKE,SCATTER):return (owner(a),)
            return owner(a),owner(b)
        roots={owner(int(parent.derivatives[list(model.state_keys).index(k),1]))
               for k in requested if parent.derivatives[list(model.state_keys).index(k),1]>=0}
        def backward(initial):
            live=set(initial);keep=set()
            for i in range(len(parent.program)-1,-1,-1):
                row=parent.program[i];code,dest=map(int,row[:2])
                if code not in checks and owner(dest) in live:
                    keep.add(i);live.update(reads(row))
            return live,keep
        live,keep=backward(roots)
        # Keep domain predicates that are live values or simple check-only
        # arithmetic around them. Do not pull unrelated CSR chains into a block.
        arithmetic={ADD,SUB,MUL,DIV,POW,MINIMUM,MAXIMUM}
        def predicate(root,seen):
            if root in live:return True,True
            if root in constants:return True,False
            if root in seen or root not in definitions:return False,False
            i=definitions[root];row=parent.program[i]
            if int(row[0]) not in arithmetic:return False,False
            children=[predicate(x,seen|{root}) for x in reads(row)]
            return all(x[0] for x in children),any(x[1] for x in children)
        extra_checks=set()
        for i,row in enumerate(parent.program):
            if int(row[0]) in checks:
                root=owner(int(row[3]));valid,touches=predicate(root,set())
                if valid and touches:extra_checks.add(i);roots.add(root)
        live,keep=backward(roots);keep.update(extra_checks)
        self.program=np.ascontiguousarray(parent.program[sorted(keep)],dtype=np.int64)
        self.derivatives=parent.derivatives.copy()
        for i,k in enumerate(model.state_keys):
            if k not in requested:self.derivatives[i,1]=-1
        self.needs_rod_current=model.nstate in live
        required={k for k,s in model.slices.items() if s.start in live}
        if self.needs_rod_current:required.add(('rods',RODS,'G'))
        self.required_state_keys=tuple(k for k in model.state_keys if k in required)
        self.program.flags.writeable=self.derivatives.flags.writeable=False
        self.metrics=dict(parent.metrics,instruction_count=len(self.program),
            full_instruction_count=len(parent.program),selective_state_keys=requested,
            required_state_keys=self.required_state_keys,
            reused_full_typed_scratch=True,discrete_events_changed=False)

    def rhs_flat(self,t,y,cone_release=None,rod_current=None):
        return self.parent._run(t,y,cone_release,rod_current,self.program,self.derivatives,self.needs_rod_current)

    __call__=rhs_flat
