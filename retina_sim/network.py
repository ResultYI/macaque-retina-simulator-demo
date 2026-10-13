"""Simultaneous continuous network RHS, persistent state, recording and checkpoints."""
from dataclasses import dataclass,field
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
from scipy.optimize import root
from .contracts import ContractError,Quantity,StateSpec,StateDerivativeVector,SimulationResult,SolverConfig,canonical_hash
from .provenance import CoverageReport
from .engine import initialize_state,advance,checkpoint,restore,validate_state
from .chen2024 import author_discrete_config
from .mechanism_contracts import MechanismStateSpec

@dataclass(frozen=True)
class InputBundle:
    channels: dict
    delay_buffers: dict = field(default_factory=dict)
    delay_query_seconds: float | None = None
    delay_left_limit: bool = False

class NetworkStimulus:
    def __init__(self,compiled,channels,state,stop,interpolation,evidence_tag):
        expected={x.channel:x for x in compiled.spec.external_inputs}
        if set(channels)!=set(expected): raise ContractError('Supply exactly the compiled external input channels')
        from .delays import EdgeDelayBuffer
        self.delay_buffers={key:EdgeDelayBuffer.from_payload(value) for key,value in state.link_delay_states.items()}
        self.channels=channels; self.start_seconds=state.time_seconds; self.stop_seconds=float(stop)
        self.interpolation=interpolation; self.evidence_tag=evidence_tag
        self.input_contract={}; breaks=set()
        for name,segment in channels.items():
            external=expected[name]; c=compiled.components[external.target_component]; port=c.inputs[external.target_port]
            if (segment.unit,segment.meaning,segment.evidence_tag)!=(port.unit,port.meaning,port.evidence_tag): raise ContractError('External input port contract mismatch: '+name)
            if (segment.interpolation!=interpolation and not any(c.time_contract.kind=='author_discrete' for c in compiled.components.values())) or segment.start_seconds>self.start_seconds or segment.stop_seconds<stop: raise ContractError('Input time/interpolation coverage mismatch: '+name)
            breaks.update(t for t in segment.breakpoints_seconds if self.start_seconds<t<stop)
            self.input_contract[name]=segment.input_contract
        self.breakpoints_seconds=tuple(sorted(breaks))
    def at(self,t): return InputBundle({name:segment.at(min(t,np.nextafter(segment.stop_seconds,segment.start_seconds)) if segment.interpolation=='zero_order_hold' else t) for name,segment in self.channels.items()},self.delay_buffers)

class JointNetwork:
    version='architecture-1.0'
    def __init__(self,compiled):
        self.compiled=compiled; self.model_id=compiled.spec.model_id
        self.evidence_tag='synthetic_fixture' if all(c.evidence_tag=='synthetic_fixture' for c in compiled.components.values()) else 'composite_source_evidence'
        self.parameters=SimpleNamespace(parameter_hash=canonical_hash({k:c.parameters.parameter_hash for k,c in compiled.components.items()}))
        self.graph=SimpleNamespace(graph_hash=canonical_hash({'assembly':compiled.assembly_hash,'links':[x.specification.connections.graph_hash for x in compiled.links]}))
        self.state_specs=[]; self.slices={}; offset=0
        for cid,c in compiled.components.items():
            start=offset
            for s in c.state_specs:
                self.state_specs.append(MechanismStateSpec(cid+'.'+s.state_name,s.cell_ids,s.shape,s.unit,s.meaning,s.initial_condition_source)); offset+=s.shape[0]
            self.slices[cid]=slice(start,offset)
        self.state_specs=tuple(self.state_specs); self._state_spec_hash=canonical_hash(self.state_specs)
        self._state_slices=tuple(slice(int(a),int(b)) for a,b in zip(np.cumsum([0]+[s.shape[0] for s in self.state_specs])[:-1],np.cumsum([s.shape[0] for s in self.state_specs])))
        self._local_slices={}
        for cid,c in compiled.components.items():
            edges=np.cumsum([self.slices[cid].start]+[s.shape[0] for s in c.state_specs])
            self._local_slices[cid]=tuple(slice(int(a),int(b)) for a,b in zip(edges[:-1],edges[1:]))
        self._incoming={}
        for link in compiled.links:
            key=(link.specification.target_component,link.specification.target_port)
            self._incoming.setdefault(key,[]).append(link)
        self.model_hash=canonical_hash({'assembly':compiled.assembly_hash,'runtime_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('network.py','hybrid.py','background.py','recording.py','mechanism_contracts.py','delays.py','plugins.py','rod_pathway.py','ipl_geometry.py','spatial_profiles.py','hc_arbor.py','hc_geometry.py','hc_contact_diagnostics.py','finite_feedback_native.py')}})
        self._gap_plans=[]
        for g in compiled.spec.gap_junctions:
            li={x:i for i,x in enumerate(compiled.components[g.left_component].outputs['voltage'].cell_ids)};ri={x:i for i,x in enumerate(compiled.components[g.right_component].outputs['voltage'].cell_ids)}
            self._gap_plans.append((g.left_component,g.right_component,np.array([li[x] for x in g.left_cell_ids]),np.array([ri[x] for x in g.right_cell_ids]),g.conductance_evidence.value.values))
    def unpack(self,y):
        values=np.asarray(y);return tuple(values[s] for s in self._state_slices)
    def local_blocks(self,cid,y):
        values=np.asarray(y);return tuple(values[s] for s in self._local_slices[cid])
    def initial_conditions(self):
        result=[]
        for cid,c in self.compiled.components.items():
            initial=c.initial()
            if len(initial)!=len(c.state_specs): raise ContractError('Component initial state count mismatch')
            for q,s in zip(initial,c.state_specs):
                q.require(s.unit,s.meaning,s.shape)
                if q.evidence_tag!=c.evidence_tag: raise ContractError('Component initial evidence mismatch')
                result.append(Quantity(q.values,q.unit,q.meaning,self.evidence_tag))
        return tuple(result)
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
    def rhs(self,t,y,bundle,parameters,graph):
        return StateDerivativeVector(self._checked_derivatives(t,y,bundle,parameters,graph),self._state_spec_hash)
    def rhs_flat(self,t,y,bundle,parameters,graph):
        # Frozen registry identity is checked at initialize/advance. Every derivative
        # still passes unit, meaning, shape and evidence checks above on every call.
        return np.concatenate([q.values for q in self._checked_derivatives(t,y,bundle,parameters,graph)])
    def make_result(self,times,trajectory,input_contract,numeric_checks):
        output={}; rows={}
        # Record state directly and evaluate every declared output at its current state.
        offset=0
        for s in self.state_specs:
            q=Quantity(trajectory[offset:offset+s.shape[0]],s.unit,s.meaning,self.evidence_tag); offset+=s.shape[0]
            output['state:'+s.state_name]=q
        for i,t in enumerate(times):
            # Preserve departing ZOH value at the final endpoint for input-dependent readout.
            sample=min(t,np.nextafter(self._record_stimulus.stop_seconds,self._record_stimulus.start_seconds)) if self._record_stimulus.interpolation=='zero_order_hold' else t
            _,signals=self.evaluate(t,trajectory[:,i],self._record_stimulus.at(sample))
            for (cid,port),q in signals.items(): rows.setdefault(cid+'.'+port,[]).append(q.values)
        for key,arrays in rows.items():
            cid,port=key.rsplit('.',1); p=self.compiled.components[cid].outputs[port]
            output[key]=Quantity(np.stack(arrays,axis=1),p.unit,p.meaning,p.evidence_tag)
        output['time']=Quantity(times,'s','absolute_time',self.evidence_tag)
        metadata=execution_metadata(self.compiled)
        metadata['global_state_layout']=[{'name':s.state_name,'cell_ids':list(s.cell_ids),'unit':s.unit,'meaning':s.meaning,'size':s.shape[0]} for s in self.state_specs]
        return SimulationResult(self.compiled.run_mode,self.model_id,canonical_hash({k:c.parameters.source_manifest_hash for k,c in self.compiled.components.items()}),input_contract,output,CoverageReport(True,(),claim_limit='Executable reference architecture; no complete physiological retina claim'),numeric_checks,'numerical_reference_assembly',metadata)

def _approximation_metadata(component,selected_cell_ids):
    result=[]
    for entry in getattr(component.native_model,'approximations',[]):
        row=dict(entry)
        if selected_cell_ids is not None:
            selected=set(selected_cell_ids)
            for key,values in entry.items():
                if key.endswith('_ids'):
                    row[key]=[x for x in values if x in selected];row[key+'_total_count']=len(values)
        result.append(row)
    return result

def execution_metadata(compiled,selected_cell_ids=None):
    return {'assembly_hash':compiled.assembly_hash,'executed_components':[{'component_id':cid,'mechanism':c.mechanism,'layers':list(c.layers),'profile_id':c.parameters.profile.profile_id,'species':c.parameters.profile.species,'source_ids':list(c.source_ids),'evidence_tag':c.evidence_tag,'parameter_evidence_class':getattr(c.parameters.profile,'evidence_class','published_source_profile'),'approximations':_approximation_metadata(c,selected_cell_ids),'time_contract':{'kind':c.time_contract.kind,'fixed_step_seconds':c.time_contract.fixed_step_seconds,'update_rule':c.time_contract.update_rule},'internal_dependencies':list(c.internal_dependencies),'plugin_scope':(__import__('dataclasses').asdict(c.plugin_declaration) if hasattr(c,'plugin_declaration') else None)} for cid,c in compiled.components.items()],'populations':[p.record(selected_cell_ids) for p in compiled.populations.values()],'unimplemented_target_layers':[x for x in compiled.spec.required_mechanisms if not any(x in c.layers for c in compiled.components.values())],'gap_junction_contacts':[{'left_component':g.left_component,'right_component':g.right_component,'contact_count':len(g.left_cell_ids),'electrical_conductance_units':'S','bidirectional_conservative':True} for g in compiled.spec.gap_junctions],'physiological_validation':'NOT_RUN','claim_limit':'Executed component coverage only; declaration is not implementation'}

class Readout:
    @staticmethod
    def signal(result,component_id,port):
        key=component_id+'.'+port
        if key not in result.outputs_with_units: raise ContractError('Requested signal was not recorded: '+key)
        return result.outputs_with_units[key]

    @staticmethod
    def state(result,component_id,state_name):
        key='state:'+component_id+'.'+state_name
        if key not in result.outputs_with_units:
            key=component_id+'.'+state_name # Isolated author discrete output.
        if key not in result.outputs_with_units: raise ContractError('Requested state was not recorded')
        return result.outputs_with_units[key]

class NetworkRunner:
    def __new__(cls,compiled):
        from .hybrid import HybridRunner
        return HybridRunner(compiled)
