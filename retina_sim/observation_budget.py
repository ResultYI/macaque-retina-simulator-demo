"""QA observation intervals from the unchanged, source-pinned formula plan.

Only registered state allowances enter. External inputs/constants are exact.
This is a rectangular interval sensitivity calculation, not a new physiology
model, and never evaluates an integration stage for the candidate backend.
"""
from decimal import Decimal, localcontext
from functools import lru_cache
import numpy as np
from .contracts import ContractError
from .compiled_kernel import (Plan, ADD,SUB,MUL,DIV,POW,SIGMOID,LOG,EXP,
    MINIMUM,MAXIMUM,CSR,ROW_CSR,TAKE,SCATTER,ZERO,FINITE,NONNEG,POSITIVE)


@lru_cache(maxsize=65536)
def _transcendental_scalar(kind, x, exponent=0.):
    """Independently resolved high precision endpoints, outward to FP64.

    Decimal exp/ln are correctly rounded. For general Decimal power we require
    independently increasing precisions to resolve the same outward FP64 pair;
    this finite high precision check is not a universal transcendental proof.
    """
    pairs=[]
    for precision in (100,180,300):
        with localcontext() as ctx:
            ctx.prec=precision
            a=Decimal.from_float(float(x)); b=Decimal.from_float(float(exponent))
            if kind==EXP: value=a.exp()
            elif kind==LOG: value=a.ln()
            elif kind==SIGMOID: value=1/(1+(-a).exp())
            elif kind==POW: value=a**b
            elif kind=='softplus': value=a+(1+(-a).exp()).ln() if a>0 else (1+a.exp()).ln()
            elif kind=='probability': value=1-(-a).exp()
            else: raise ContractError('Unknown high precision observation primitive')
            # Covers the much smaller Decimal computation error before FP64
            # conversion. Two precision levels must give identical enclosure.
            if value==0:
                if kind=='probability' and a!=0:continue
                return (0.,0.)
            guard=abs(value)*Decimal(10)**(-precision+12)
            lo=np.nextafter(float(value-guard),-np.inf)
            hi=np.nextafter(float(value+guard),np.inf)
        pairs.append((lo,hi))
        if len(pairs)>1 and pairs[-1]==pairs[-2]:return pairs[-1]
    raise ContractError('REFERENCE_UNRESOLVED: high precision observation endpoint')


def primitive_bounds(kind, low, high, exponent=None):
    if kind==SIGMOID:
        # Preserve the implemented exp -> FP64 add -> FP64 divide sequence.
        # The exp enclosure includes the correctly-rounded result's adjacent
        # representables; only a verified CUDA standard-exp lowering may claim
        # its documented 1ULP bound. For CPU observations actual containment is
        # checked separately; this is no universal UCRT/NumPy accuracy claim.
        el,eh=primitive_bounds(EXP,-np.asarray(high),-np.asarray(low))
        dl,dh=arithmetic(ADD,1.,1.,el,eh)
        return arithmetic(DIV,1.,1.,dl,dh)
    low,high=np.broadcast_arrays(low,high)
    if exponent is None: exponent=np.zeros_like(low)
    exponent=np.broadcast_to(exponent,low.shape)
    if kind in (LOG,POW) and np.any(low<=0):
        raise ContractError('Observation interval crosses a positive primitive domain')
    pairs=[(_transcendental_scalar(kind,float(a),float(b)),
            _transcendental_scalar(kind,float(c),float(b)))
        for a,c,b in zip(low.flat,high.flat,exponent.flat)]
    lower=np.array([min(a[0],b[0]) for a,b in pairs]).reshape(low.shape)
    upper=np.array([max(a[1],b[1]) for a,b in pairs]).reshape(low.shape)
    return lower,upper


def arithmetic(code, al, ah, bl, bh):
    if code==ADD: lo,hi=al+bl,ah+bh
    elif code==SUB: lo,hi=al-bh,ah-bl
    elif code in (MUL,DIV):
        if code==DIV and np.any((bl<=0)&(bh>=0)):
            raise ContractError('Observation interval contains a zero denominator')
        op=np.multiply if code==MUL else np.divide
        corners=np.stack(np.broadcast_arrays(op(al,bl),op(al,bh),op(ah,bl),op(ah,bh)))
        lo,hi=corners.min(axis=0),corners.max(axis=0)
    elif code==MINIMUM: lo,hi=np.minimum(al,bl),np.minimum(ah,bh)
    elif code==MAXIMUM: lo,hi=np.maximum(al,bl),np.maximum(ah,bh)
    else:raise ContractError('Unknown observation interval arithmetic')
    return np.nextafter(lo,-np.inf),np.nextafter(hi,np.inf)


def firing_intervals(model, witness, voltage_allowance):
    """Propagate state V through the original frozen-history rate/probability.

    The history dot product is identical after strict integer/RNG checks. Its
    actual FP64 value is therefore a common constant in both evaluations.
    No empirical Hz floor or extra event-time allowance is introduced.
    """
    v=witness.V; radius=np.asarray(voltage_allowance)
    bias=model.history_kernel@witness.effective_history
    lo,hi=arithmetic(ADD,np.nextafter(v-radius,-np.inf),np.nextafter(v+radius,np.inf),bias,bias)
    lo,hi=arithmetic(SUB,lo,hi,model.mu,model.mu)
    lo,hi=arithmetic(DIV,lo,hi,model.beta,model.beta)
    lo,hi=primitive_bounds('softplus',lo,hi)
    rate_low,rate_high=arithmetic(MUL,lo,hi,model.alpha,model.alpha)
    hazard_low,hazard_high=arithmetic(MUL,rate_low,rate_high,witness.dt,witness.dt)
    p_low,p_high=primitive_bounds('probability',hazard_low,hazard_high)
    if np.any(witness.rate_Hz<rate_low)|np.any(witness.rate_Hz>rate_high):
        raise ContractError('REFERENCE_UNRESOLVED: original rate outside high precision formula interval')
    if np.any(witness.p<p_low)|np.any(witness.p>p_high):
        raise ContractError('REFERENCE_UNRESOLVED: original probability outside formula interval')
    return dict(rate_allowance=np.maximum(abs(rate_low-witness.rate_Hz),abs(rate_high-witness.rate_Hz)),
        probability_low=p_low,probability_high=p_high,
        analytic_rate_lipschitz=model.alpha/model.beta*radius,
        analytic_probability_lipschitz=witness.dt*model.alpha/model.beta*radius)


class ObservationBudget:
    def __init__(self, model, ports=None):
        if any(block.cyclic for block in model.compiled.output_order) or any(
                link.delayed_operator is not None for link in model.compiled.links):
            raise ContractError('Observation sensitivity does not cover algebraic/edge-delay closures')
        self.model=model
        # An observation-only plan avoids derivative scratch. Explicit native
        # formulas allow the same interval rules at any population size.
        class ObservationPlan(Plan):
            def ensure(self,cid,port):
                key=(cid,port)
                if key not in self.outputs:self.outputs[key]=self.check(self.observe(cid,port))
                return self.outputs[key]
            def gather(self,cid,name):
                if (cid,name) not in self.inputs:
                    for link in self.network._incoming.get((cid,name),()):
                        source=link.specification;self.ensure(source.source_component,source.source_port)
                return super().gather(cid,name)
        self.plan=plan=ObservationPlan(model); plan.native_fusion_minimum_cells=float('inf')
        required=[block.nodes[0] for block in model.compiled.output_order] if ports is None else list(ports)
        for cid,port in required:plan.ensure(cid,port)
        # Network.evaluate overwrites component placeholder gap currents.
        gaps={cid:plan.zero(len(c.outputs['voltage'].cell_ids))
            for cid,c in model.compiled.components.items() if hasattr(c.native_model,'membrane_capacitance')}
        for left,right,ix,iy,g in model._gap_plans:
            current=plan.ref(g)*(plan.take(plan.ensure(right,'voltage'),iy)-plan.take(plan.ensure(left,'voltage'),ix))
            plan.scatter(gaps[left],current,ix);plan.scatter(gaps[right],-current,iy)
        for cid,ref in gaps.items():plan.outputs[(cid,'gap_current')]=ref
        self.constants=np.array(plan.values,dtype=np.float64)

    def intervals(self, state, state_allowance, bundle):
        p=self.plan; y=np.asarray(state); radius=np.asarray(state_allowance)
        if y.dtype!=np.float64 or y.shape!=(p.nstate,) or radius.shape!=y.shape or np.any(radius<0):
            raise ContractError('Observation sensitivity requires all registered FP64 state allowances')
        lo,hi=self.constants.copy(),self.constants.copy()
        lo[:p.nstate]=np.nextafter(y-radius,-np.inf);hi[:p.nstate]=np.nextafter(y+radius,np.inf)
        offset=p.nstate
        for external,port in p.external:
            q=bundle.channels[external.channel];port.validate(q);n=len(q.values)
            lo[offset:offset+n]=hi[offset:offset+n]=q.values;offset+=n
        for code,dest,n,a,an,b,bn in p.program:
            if code in (FINITE,NONNEG,POSITIVE):continue  # intervals can extend beyond a state boundary
            if code==ZERO:lo[dest:dest+n]=hi[dest:dest+n]=0.;continue
            if code==TAKE:
                index=np.array(p.indices[b:b+n]);lo[dest:dest+n]=lo[a+index];hi[dest:dest+n]=hi[a+index];continue
            if code==SCATTER:
                for j,index in enumerate(p.indices[b:b+n]):
                    dl,dh=arithmetic(ADD,lo[dest+index],hi[dest+index],lo[a+j],hi[a+j])
                    lo[dest+index],hi[dest+index]=dl,dh
                continue
            if code in (CSR,ROW_CSR):
                for j in range(n):
                    lower=upper=np.float64(0.)
                    for k in range(p.pointers[b+j],p.pointers[b+j+1]):
                        weight=p.weights[bn+k];source=a+p.indices[bn+k]
                        wl,wh=arithmetic(MUL,lo[source],hi[source],weight,weight)
                        lower,upper=arithmetic(ADD,lower,upper,wl,wh)
                    lo[dest+j],hi[dest+j]=lower,upper
                continue
            al,ah=lo[a:a+(an if an>1 else 1)],hi[a:a+(an if an>1 else 1)]
            bl,bh=lo[b:b+(bn if bn>1 else 1)],hi[b:b+(bn if bn>1 else 1)]
            if code in (EXP,LOG,SIGMOID):lower,upper=primitive_bounds(code,al,ah)
            elif code==POW:
                if np.any(bl!=bh):raise ContractError('Variable-exponent observation budget absent')
                lower,upper=primitive_bounds(code,*np.broadcast_arrays(al,ah,bl)[:2],exponent=bl)
            else:lower,upper=arithmetic(code,al,ah,bl,bh)
            lo[dest:dest+n],hi[dest:dest+n]=lower,upper
        result={}
        for key,ref in p.outputs.items():
            s=slice(ref.offset,ref.offset+ref.n)
            if ref.offset+ref.n<=p.nstate:
                # A state alias has exactly its registered budget.
                result[key]=(y[s]-radius[s],y[s]+radius[s],radius[s].copy())
            else:result[key]=(lo[s].copy(),hi[s].copy(),None)
        return result

    def allowances(self, t, state, state_allowance, bundle):
        _,signals=self.model.evaluate(t,state,bundle)
        intervals=self.intervals(state,state_allowance,bundle)
        result={}
        for key in intervals:
            if key not in signals:raise ContractError('Required original observation absent: '+str(key))
            q=signals[key]
            lo,hi,alias=intervals[key]
            if alias is not None:result[key]=alias
            else:
                if np.any(q.values<lo)|np.any(q.values>hi):
                    raise ContractError('REFERENCE_UNRESOLVED: original observation outside source formula interval '+str(key))
                result[key]=np.maximum(np.abs(lo-q.values),np.abs(hi-q.values))
        return result
