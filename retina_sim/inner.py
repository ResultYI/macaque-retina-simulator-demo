"""Explicit residual history cells and effective conductance/history spike backend."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
from .contracts import ContractError,Quantity,RunMode
from .mechanism_contracts import MechanismStateSpec
from .assembly import PortSpec,RuntimeComponent,TimeContract

def softplus(x): return np.logaddexp(0,np.asarray(x))
def _profile(spec,populations,sources,mode):
    if mode==RunMode.FULL_PHYSIOLOGY: raise ContractError('Inner profiles not certified as whole physiological retina')
    if spec.parameter_profile_id not in getattr(sources,'mechanism_profiles',{}): raise ContractError('Required inner numeric profile missing; no default kernel/conductance')
    p=sources.mechanism_profiles[spec.parameter_profile_id]
    if set(spec.population_roles)!={'cells'}: raise ContractError('Inner cell population role required')
    pop=populations[spec.population_roles['cells']]; tags=set(pop.cells.evidence_tags)
    expected_type={'ResidualBC_ON':'BC','ResidualBC_OFF':'BC','ResidualAC':'AC','HistoryConductanceRGC':'RGC','PassiveConeCable':'cone','QuasiSteadyTerminalActivation':'cone','EffectiveHCThresholdFeedback':'HC'}.get(spec.mechanism)
    if expected_type is not None and set(pop.cells.cell_type)!={expected_type}:raise ContractError('Mechanism requires explicit cell type '+expected_type)
    if len(tags)!=1: raise ContractError('Inner evidence class mismatch')
    tag=tags.pop()
    if p.evidence_class=='software_fixture' and tag!='synthetic_fixture':raise ContractError('Software fixture coefficients cannot be relabeled as animal/source measurements in any run mode')
    if mode==RunMode.NUMERIC_FIXTURE and (tag!='synthetic_fixture' or p.evidence_class!='software_fixture'): raise ContractError('Uncalibrated inner execution must explicitly use software_fixture profile')
    target_profile=spec.options.get('target_population_profile_id') if (spec.mechanism in ('SynapticConductance','ExplicitPhysicalUnitConversion') or
        (spec.mechanism=='DoubleExponentialChemicalSynapse' and spec.options.get('empty_contact_operator') is True)) else None
    expected_profile=target_profile or p.profile_id
    if any(x!=expected_profile for x in pop.cells.parameter_profile_id): raise ContractError('Inner population/profile mismatch')
    if any(x not in (p.species,p.profile_id) for x in pop.cells.species_profile_id): raise ContractError('Inner species profile mismatch')
    return p,pop,tag
def _hash(): return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def _port(name,unit,meaning,pop,tag): return PortSpec(name,unit,meaning,pop.cells.cell_id,tag)

class CausalHistory:
    def __init__(self,p,pop,tag,mechanism,options):
        self.tag=tag; self.n=len(pop.cells.cell_id); self.p=p
        self.kernel=p.quantity('kernel','dimensionless','causal_residual_kernel').values
        if self.kernel.ndim!=1 or not len(self.kernel): raise ContractError('Full finite causal kernel vector required')
        self.dt=float(p.quantity('sample_interval','s','history_sampling_interval',(),positive=True).values)
        self.bias=p.quantity('bias','dimensionless','activation_bias',(self.n,)).values
        self.baseline=p.quantity('baseline','dimensionless','input_baseline',(self.n,)).values
        self.initial_history=p.quantity('initial_history','dimensionless','causal_input_history',(len(self.kernel),self.n)).values
        representation=options.get('polarity_representation')
        if mechanism=='ResidualAC':
            if representation!='stored_signed': raise ContractError('AC signed kernel representation must be declared')
            self.sign=1
        elif representation=='canonical_dominant_positive':
            if self.kernel[np.argmax(np.abs(self.kernel))]<=0: raise ContractError('Canonical kernel already negative; refuse duplicate ON inversion')
            self.sign=-1 if mechanism=='ResidualBC_ON' else 1
        elif representation=='stored_signed': self.sign=1
        else: raise ContractError('Explicit polarity representation required')
        self.inhibition=options.get('presynaptic_inhibition',False)
        if type(self.inhibition) is not bool: raise ContractError('Explicit presynaptic gate choice required')
        self.state_specs=tuple(MechanismStateSpec('history_'+str(i),pop.cells.cell_id,(self.n,),'dimensionless','causal_synaptic_drive_history','Declared residual-profile causal prehistory') for i in range(len(self.kernel)))+(MechanismStateSpec('activation',pop.cells.cell_id,(self.n,),'dimensionless','effective_synaptic_activation','Activation of declared initial history'),)
    def activation(self,h,incoming):
        x=self.bias+self.sign*(self.kernel@h)
        if self.inhibition:
            a=incoming['inhibition'].values
            if np.any(a<0): raise ContractError('Presynaptic inhibitory gate must be nonnegative')
            x=x-a
        return softplus(x)
    def initial(self):
        if self.inhibition: initial_inhib=self.p.quantity('initial_inhibition','dimensionless','initial_inhibitory_gate',(self.n,),nonnegative=True)
        else: initial_inhib=Quantity(np.zeros(self.n),'dimensionless','effective_synaptic_activation',self.tag)
        h=self.initial_history
        a=self.activation(h,{'inhibition':initial_inhib})
        return tuple(Quantity(v,s.unit,s.meaning,self.tag) for v,s in zip(tuple(h)+(a,),self.state_specs))
    def candidate(self,blocks,incoming):
        drive=incoming['drive'].values
        if np.any(drive<0) and not incoming.get('__closure_trial__',False): raise ContractError('Effective synaptic input must be nonnegative; deviations are anchored explicitly')
        return np.vstack((drive-self.baseline,np.asarray(blocks[:-1])[:-1]))
    def observe(self,port,t,blocks,incoming):
        if incoming.get('__sampling__',False):
            candidate=self.candidate(blocks,incoming) if 'drive' in incoming else np.vstack((np.zeros(self.n),np.asarray(blocks[:-1])[:-1]))
            a=self.activation(candidate,incoming)
        else: a=blocks[-1]
        return Quantity(a,'dimensionless','effective_synaptic_activation',self.tag)
    def transition(self,blocks,incoming,outputs): return tuple(self.candidate(blocks,incoming))+(outputs['activation'].values,)
    def derivative(self,t,blocks,incoming):
        # These are explicitly timestamped ZOH history states, updated by transition.
        return tuple(Quantity(np.zeros(s.shape),s.unit+'/s',s.meaning+'_derivative',self.tag) for s in self.state_specs)

def history_component(spec,populations,sources,mode):
    p,pop,tag=_profile(spec,populations,sources,mode); model=CausalHistory(p,pop,tag,spec.mechanism,spec.options)
    meaning='effective_synaptic_activation' if spec.mechanism=='ResidualAC' else 'effective_synaptic_drive'
    inputs={'drive':_port('drive','dimensionless',meaning,pop,tag)}
    if model.inhibition: inputs['inhibition']=_port('inhibition','dimensionless','effective_synaptic_activation',pop,tag)
    outputs={'activation':_port('activation','dimensionless','effective_synaptic_activation',pop,tag)}
    dependencies=tuple(inputs) if model.kernel[0]!=0 or model.inhibition else ()
    # Even k0=0 must read the current input to commit history; output can be causal-only.
    layer='AC' if spec.mechanism=='ResidualAC' else ('BC_ON' if spec.mechanism.endswith('ON') else 'BC_OFF')
    return RuntimeComponent(spec.component_id,spec.mechanism,p,model,inputs,outputs,model.state_specs,model.initial,model.derivative,model.observe,{'activation':dependencies},TimeContract('causal_history',model.dt,'Hnew[0]=current drive-baseline; old history shifts; k already includes sampling convention'),(layer,),(pop,),tag,p.source_ids,_hash())

def conductance_component(spec,populations,sources,mode):
    p,pop,tag=_profile(spec,populations,sources,mode)
    gain=p.quantity('gain','S','synaptic_conductance_scale',(len(pop.cells.cell_id),),nonnegative=True).values
    kind=spec.options.get('receptor')
    if kind not in ('excitatory','inhibitory'): raise ContractError('Explicit E/I receptor required; conductance never negative')
    meaning=kind+'_conductance'
    def observe(port,t,blocks,incoming):
        a=incoming['activation'].values
        if np.any(a<0) and not incoming.get('__closure_trial__',False): raise ContractError('Negative activation cannot create conductance')
        return Quantity(gain*a,'S',meaning,tag)
    return RuntimeComponent(spec.component_id,spec.mechanism,p,SimpleNamespace(model_hash=p.parameter_hash),{'activation':_port('activation','dimensionless','effective_synaptic_activation',pop,tag)},{'conductance':_port('conductance','S',meaning,pop,tag)},(),lambda:(),lambda t,b,i:(),observe,{'conductance':('activation',)},TimeContract('continuous'),('synaptic_conductance',),(pop,),tag,p.source_ids,_hash())

class HistoryRGC:
    # Native graded V has no reset/noise/spike feedback in this mechanism. This
    # declaration permits a deterministic V check while keeping random spike
    # histories/RNG as a separate burn-in/statistical question.
    deterministic_state_independent_of_spikes = True
    def __init__(self,p,pop,tag):
        self.p=p; self.tag=tag; self.n=len(pop.cells.cell_id)
        self.C=p.quantity('C','F','membrane_capacitance',(self.n,),positive=True).values
        self.gL=p.quantity('gL','S','leak_conductance',(self.n,),positive=True).values
        self.EL=p.quantity('EL','V','leak_reversal',(self.n,)).values
        self.EE=p.quantity('EE','V','excitatory_reversal',(self.n,)).values
        self.EI=p.quantity('EI','V','inhibitory_reversal',(self.n,)).values
        self.V0=p.quantity('V0','V','initial_membrane_voltage',(self.n,)).values
        self.alpha=p.quantity('alpha','Hz','spike_intensity_scale',(self.n,),positive=True).values
        self.mu=p.quantity('mu','V','spike_voltage_offset',(self.n,)).values
        self.beta=p.quantity('beta','V','spike_voltage_scale',(self.n,),positive=True).values
        self.history_kernel=p.quantity('spike_history_kernel','V','strict_past_spike_history_kernel').values
        if self.history_kernel.ndim!=1: raise ContractError('Complete spike-history vector required')
        self.initial_spike_history=p.quantity('initial_spike_history','dimensionless','spike_count_history',(len(self.history_kernel),self.n),nonnegative=True).values
        self.dt=float(p.quantity('spike_interval','s','spike_sampling_interval',(),positive=True).values)
        self.state_specs=(MechanismStateSpec('V',pop.cells.cell_id,(self.n,),'V','physical_membrane_voltage','Declared RGC operating voltage'),)
    def initial(self): return (Quantity(self.V0,'V','physical_membrane_voltage',self.tag),)
    def derivative(self,t,blocks,incoming):
        e=incoming['excitation'].values; i=incoming['inhibition'].values
        if np.any(e<0) or np.any(i<0): raise ContractError('Conductances must be nonnegative')
        V=blocks[0]; d=(self.gL*(self.EL-V)+e*(self.EE-V)+i*(self.EI-V))/self.C
        return (Quantity(d,'V/s','physical_membrane_voltage_derivative',self.tag),)
    def observe(self,port,t,blocks,incoming):
        if port=='gE':return incoming['excitation']
        if port=='gI':return incoming['inhibition']
        return Quantity(blocks[0],'V','physical_membrane_voltage',self.tag)
    def intensity(self,V,history): return self.alpha*softplus((V+self.history_kernel@history-self.mu)/self.beta)
    def sample_spikes(self,t,V,aux):
        if not len(self.history_kernel): history=np.empty((0,self.n))
        elif aux['last_bin_counts'] is None: history=np.asarray(aux['history'])
        else: history=np.vstack((np.asarray(aux['last_bin_counts'])[None,:],np.asarray(aux['history'])[:-1]))
        rate=self.intensity(V,history)
        rng=np.random.Generator(np.random.PCG64()); rng.bit_generator.state=aux['rng_state']
        u=rng.random(self.n); p=-np.expm1(-rate*self.dt); counts=(u<p).astype(int)
        events=[{'cell_index':int(k),'time_seconds':float(t-np.log1p(-u[k])/rate[k]),'bin_start_seconds':float(t),'bin_stop_seconds':float(t+self.dt)} for k in np.flatnonzero(counts)]
        return {'history':history.tolist(),'last_bin_counts':counts.tolist(),'rng_state':rng.bit_generator.state,'conditional_rate_Hz':rate.tolist(),'pending_events':aux.get('pending_events',[])+events,'observation_scheme':'At least one event/bin Bernoulli; first-event time under frozen conditional hazard; no additional reset/refractory'},events

def rgc_component(spec,populations,sources,mode):
    p,pop,tag=_profile(spec,populations,sources,mode); model=HistoryRGC(p,pop,tag)
    if 'rng_seed' not in spec.options: raise ContractError('Explicit reproducible spike RNG seed required')
    model.rng_seed=spec.options['rng_seed']
    inputs={key:_port(key,'S',kind+'_conductance',pop,tag) for key,kind in (('excitation','excitatory'),('inhibition','inhibitory'))}
    outputs={'voltage':_port('voltage','V','physical_membrane_voltage',pop,tag),'gE':_port('gE','S','excitatory_conductance',pop,tag),'gI':_port('gI','S','inhibitory_conductance',pop,tag)}
    return RuntimeComponent(spec.component_id,spec.mechanism,p,model,inputs,outputs,model.state_specs,model.initial,model.derivative,model.observe,{'voltage':(),'gE':('excitation',),'gI':('inhibition',)},TimeContract('continuous'),('RGC_spike_history',),(pop,),tag,p.source_ids,_hash())
