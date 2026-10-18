"""Frozen pre-optimization evaluation algorithm for regression, 2026-10-15.

Only evaluation/derivative methods are retained; initialization uses the current
unchanged registry. This reference never replaces a production module.
"""
import numpy as np
from scipy.optimize import root
from retina_sim.contracts import ContractError,Quantity
from retina_sim.network import JointNetwork

class UncachedJointNetwork(JointNetwork):
    def evaluate(self,t,y,bundle,sampling=False,sampled_components=None):
        inputs={cid:{} for cid in self.compiled.components}; outputs={}
        local={cid:self.local_blocks(cid,y) for cid in self.compiled.components}
        for external in self.compiled.spec.external_inputs:
            q=bundle.channels[external.channel]; c=self.compiled.components[external.target_component]
            c.inputs[external.target_port].validate(q); inputs[external.target_component][external.target_port]=q
        def gather(cid,port,signals=None):
            signals=outputs if signals is None else signals
            c=self.compiled.components[cid]; target=c.inputs[port]
            if port in inputs[cid]: return inputs[cid][port]
            incoming=self._incoming.get((cid,port),())
            if not incoming: raise ContractError('Unbound runtime input')
            values=np.zeros(len(target.cell_ids))
            for link in incoming:
                source=signals.get((link.specification.source_component,link.specification.source_port))
                if link.delayed_operator is None:
                    if source is None:raise ContractError('Instantaneous source output unavailable')
                    values+=link.matrix@source.values
                else:
                    operator=link.delayed_operator
                    if operator.key not in bundle.delay_buffers:raise ContractError('Explicit delayed state missing')
                    query=t if bundle.delay_query_seconds is None else bundle.delay_query_seconds
                    values+=operator.apply(query,source,bundle.delay_buffers[operator.key],left_limit=bundle.delay_left_limit)
            q=Quantity(values,target.unit,target.meaning,target.evidence_tag)
            return q
        def observe(cid,port,signals,trial=False):
            c=self.compiled.components[cid]
            dependencies={name:gather(cid,name,signals) for name in c.output_dependencies[port]}
            dependencies['__sampling__']=sampling and (sampled_components is None or cid in sampled_components); dependencies['__closure_trial__']=trial
            return c.observe(port,t,local[cid],dependencies)
        for block in self.compiled.output_order:
            if not block.cyclic:
                cid,port=block.nodes[0]; q=observe(cid,port,outputs); self.compiled.components[cid].outputs[port].validate(q); outputs[(cid,port)]=q
            else:
                sizes=[len(self.compiled.components[cid].outputs[port].cell_ids) for cid,port in block.nodes]
                offsets=np.cumsum((0,)+tuple(sizes))
                def residual(values):
                    signals=dict(outputs)
                    for k,(cid,port) in enumerate(block.nodes):
                        p=self.compiled.components[cid].outputs[port]; signals[(cid,port)]=Quantity(values[offsets[k]:offsets[k+1]],p.unit,p.meaning,p.evidence_tag)
                    return np.concatenate([observe(cid,port,signals,True).values for cid,port in block.nodes])-values
                policy=self.compiled.spec.algebraic_closure
                solution=root(residual,np.zeros(sum(sizes)),method=policy['method'],options={'xtol':policy['rtol'],'maxfev':policy['max_evaluations']})
                error=np.max(np.abs(residual(solution.x)))
                if not solution.success or error>policy['atol']+policy['rtol']*max(1,float(np.max(np.abs(solution.x)))): raise ContractError('Joint zero-delay closure failed; no hidden delay or clipped fallback')
                for k,(cid,port) in enumerate(block.nodes):
                    p=self.compiled.components[cid].outputs[port]; q=Quantity(solution.x[offsets[k]:offsets[k+1]],p.unit,p.meaning,p.evidence_tag); p.validate(q); outputs[(cid,port)]=q
        for cid,c in self.compiled.components.items():
            for name in c.inputs:
                q=gather(cid,name); c.inputs[name].validate(q); inputs[cid][name]=q
                if q.meaning in ('effective_synaptic_drive','effective_synaptic_activation','excitatory_conductance','inhibitory_conductance') and np.any(q.values<0): raise ContractError('Negative physical/effective input after closure')
        gap_currents={cid:np.zeros(len(c.outputs['voltage'].cell_ids)) for cid,c in self.compiled.components.items() if hasattr(c.native_model,'membrane_capacitance')}
        for l,r,ix,iy,conductance in self._gap_plans:
            current=conductance*(outputs[(r,'voltage')].values[iy]-outputs[(l,'voltage')].values[ix])
            np.add.at(gap_currents[l],ix,current);np.add.at(gap_currents[r],iy,-current)
        for cid,current in gap_currents.items():
            tag=self.compiled.components[cid].evidence_tag;q=Quantity(current,'A','electrical_gap_current',tag)
            inputs[cid]['__gap_current__']=q;outputs[(cid,'gap_current')]=q
        return inputs,outputs
    def _checked_derivatives(self,t,y,bundle,parameters,graph):
        if parameters is not self.parameters or graph is not self.graph: raise ContractError('Joint RHS identity mismatch')
        inputs,_=self.evaluate(t,y,bundle); derivatives=[]
        for cid,c in self.compiled.components.items():
            local=c.derivative(t,self.local_blocks(cid,y),inputs[cid])
            if len(local)!=len(c.state_specs): raise ContractError('Component derivative count mismatch')
            for q,s in zip(local,c.state_specs):
                q.require(s.unit+'/s',s.meaning+'_derivative',s.shape)
                if q.evidence_tag!=c.evidence_tag: raise ContractError('Component derivative evidence mismatch')
                derivatives.append(Quantity(q.values,q.unit,q.meaning,self.evidence_tag))
        return tuple(derivatives)
    def rhs_flat(self,t,y,bundle,parameters,graph):
        # Frozen registry identity is checked at initialize/advance. Every derivative
        # still passes unit, meaning, shape and evidence checks above on every call.
        return np.concatenate([q.values for q in self._checked_derivatives(t,y,bundle,parameters,graph)])
