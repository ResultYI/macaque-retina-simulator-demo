"""Physical cable/activation operators with required calibration, never arbitrary gain."""
import numpy as np
import hashlib
from pathlib import Path
from scipy.special import expit
from .contracts import ContractError,Quantity
from .mechanism_contracts import explicit_unit_conversion,MechanismStateSpec
from .assembly import PortSpec,RuntimeComponent,TimeContract

def _hash():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def passive_coefficients(lengths,membrane_areas,cross_areas,resistivity,specific_capacitance,specific_conductance,pairs,*,taper_endpoint_radii=None):
    L=lengths.require('m','compartment_length');A=membrane_areas.require('m^2','membrane_area',L.shape);cross=cross_areas.require('m^2','cross_section_area',L.shape)
    rho=resistivity.require('ohm m','axial_resistivity',L.shape);cm=specific_capacitance.require('F/m^2','specific_membrane_capacitance',L.shape);gm=specific_conductance.require('S/m^2','specific_membrane_conductance',L.shape)
    if L.ndim!=1 or any(np.any(a<=0) for a in (L,A,cross,rho,cm,gm)): raise ContractError('Positive source geometry/passive coefficients required')
    axial=np.zeros((len(L),len(L)))
    if taper_endpoint_radii is not None:
        radii=taper_endpoint_radii.require('m','taper_endpoint_radii',(len(L),2))
        if np.any(radii<=0): raise ContractError('Positive tapered radii required')
    for j,k in pairs:
        if j==k or not 0<=j<len(L) or not 0<=k<len(L) or axial[j,k]!=0: raise ContractError('Invalid/duplicate axial connection')
        if taper_endpoint_radii is None: resistance=rho[j]*L[j]/(2*cross[j])+rho[k]*L[k]/(2*cross[k])
        else:
            middle=radii.mean(axis=1)
            # Exact integral of rho/(pi*r(x)^2) for each linear half-taper.
            resistance=rho[j]*L[j]/(2*np.pi*middle[j]*radii[j,1])+rho[k]*L[k]/(2*np.pi*middle[k]*radii[k,0])
        axial[j,k]=axial[k,j]=1/resistance
    tag=lengths.evidence_tag
    if any(q.evidence_tag!=tag for q in (membrane_areas,cross_areas,resistivity,specific_capacitance,specific_conductance)): raise ContractError('Passive coefficient evidence mismatch')
    return {'C':Quantity(cm*A,'F','membrane_capacitance',tag),'gL':Quantity(gm*A,'S','leak_conductance',tag),'axial':Quantity(axial,'S','axial_conductance',tag),'area_convention':'Caller-provided membrane area; endcaps not silently inferred','taper_convention':'exact half-linear radius integral' if taper_endpoint_radii is not None else 'uniform half-compartment resistance'}

class PassiveCable:
    def __init__(self,C,gL,axial,injection,current_anchor,initial_deviation,voltage_anchor=None):
        self.C=C.require('F','membrane_capacitance');self.gL=gL.require('S','leak_conductance',self.C.shape)
        if self.C.ndim!=1 or np.any(self.C<=0) or np.any(self.gL<=0): raise ContractError('Positive compartment C/gL required')
        self.G=axial.require('S','axial_conductance',(len(self.C),len(self.C)))
        if np.any(self.G<0) or not np.array_equal(self.G,self.G.T) or np.any(np.diag(self.G)!=0): raise ContractError('Symmetric positive physical axial conductance required')
        self.b=injection.require('dimensionless','current_injection_distribution')
        if self.b.ndim!=2 or self.b.shape[0]!=len(self.C) or np.any(self.b<0) or not np.allclose(self.b.sum(axis=0),1,rtol=0,atol=1e-12): raise ContractError('Explicit normalized current injection required')
        self.anchor=current_anchor.require('A','absolute_outer_segment_current',(self.b.shape[1],));self.initial=initial_deviation.require('V','physical_voltage_deviation',self.b.shape)
        self.voltage_anchor=None if voltage_anchor is None else voltage_anchor.require('V','physical_membrane_voltage',self.b.shape)
        self.tag=current_anchor.evidence_tag
    def rhs(self,deviation,current):
        d=deviation.require('V','physical_voltage_deviation',self.b.shape);I=current.require('A','absolute_outer_segment_current',(self.b.shape[1],))
        currents=-self.gL[:,None]*d+self.G@d-self.G.sum(axis=1)[:,None]*d-self.b*(I-self.anchor)[None,:]
        return Quantity(currents/self.C[:,None],'V/s','physical_voltage_deviation_derivative',self.tag)
    def voltage(self,deviation):
        if self.voltage_anchor is None: raise ContractError('Absolute voltage anchor unknown; deltaV cannot enter absolute terminal activation')
        d=deviation.require('V','physical_voltage_deviation',self.b.shape)
        return Quantity(self.voltage_anchor+d,'V','physical_membrane_voltage',self.tag)
    def semi_implicit_step(self,deviation,current,dt_seconds):
        if not np.isfinite(dt_seconds) or dt_seconds<=0: raise ContractError('Positive numerical cable step required')
        d=deviation.require('V','physical_voltage_deviation',self.b.shape);I=current.require('A','absolute_outer_segment_current',(self.b.shape[1],))
        operator=np.diag(self.gL+self.G.sum(axis=1))-self.G
        matrix=np.diag(self.C)+dt_seconds*operator
        updated=np.linalg.solve(matrix,self.C[:,None]*d-dt_seconds*self.b*(I-self.anchor)[None,:])
        return Quantity(updated,'V','physical_voltage_deviation',self.tag)

def terminal_activation(voltage,Vhalf,slope,HC_shift):
    V=voltage.require('V','physical_membrane_voltage');half=Vhalf.require('V','activation_half_voltage',V.shape);k=slope.require('V','activation_voltage_slope',V.shape);shift=HC_shift.require('V','activation_threshold_shift',V.shape)
    if np.any(k<=0): raise ContractError('Positive Ca activation slope required')
    return Quantity(expit((V-half-shift)/k),'dimensionless','normalized_Ca_activation',voltage.evidence_tag)

def terminal_current(voltage,activation,gCa,E_Ca):
    V=voltage.require('V','physical_membrane_voltage');m=activation.require('dimensionless','normalized_Ca_activation',V.shape);g=gCa.require('S','maximum_Ca_conductance',V.shape);E=E_Ca.require('V','Ca_reversal_potential',V.shape)
    if np.any(g<0) or np.any(m<0) or np.any(m>1): raise ContractError('Physical Ca conductance/gate domain')
    return Quantity(g*m*(V-E),'A','absolute_terminal_Ca_current',voltage.evidence_tag)

def monotone_release_proxy(activation,*,profile_name):
    if profile_name!='quasi_steady_normalized_activation_proxy': raise ContractError('Named non-biochemical release approximation required')
    m=activation.require('dimensionless','normalized_Ca_activation')
    return Quantity(m,'dimensionless','effective_synaptic_drive',activation.evidence_tag)

def effective_HC_derivative(H,drive,tau,K,n):
    h=H.require('dimensionless','effective_HC_activation');x=drive.require('dimensionless','effective_synaptic_drive',h.shape);t=tau.require('s','effective_HC_time_constant',h.shape);k=K.require('dimensionless','HC_half_drive',h.shape);exponent=n.require('dimensionless','HC_hill_exponent',h.shape)
    if np.any(x<0) or np.any(t<=0) or np.any(k<=0) or np.any(exponent<=0): raise ContractError('Required positive HC calibration missing/domain invalid')
    activation=expit(exponent*np.log(np.maximum(x,np.finfo(float).tiny)/k));activation=np.where(x==0,0,activation)
    return Quantity((activation-h)/t,'s^-1','effective_HC_activation_derivative',H.evidence_tag)

def effective_HC_shift(H,anchor,kappa):
    h=H.require('dimensionless','effective_HC_activation');a=anchor.require('dimensionless','effective_HC_anchor',h.shape);k=kappa.require('V','HC_threshold_shift_gain',h.shape)
    if np.any(k<=0): raise ContractError('Positive explicit HC threshold calibration required')
    return Quantity(k*(h-a),'V','activation_threshold_shift',H.evidence_tag)

def cable_component(spec,populations,sources,mode):
    from .inner import _profile
    p,pop,tag=_profile(spec,populations,sources,mode);n=len(pop.cells.cell_id)
    C=p.quantity('C','F','membrane_capacitance');gL=p.quantity('gL','S','leak_conductance',C.values.shape);nc=len(C.values)
    model=PassiveCable(C,gL,p.quantity('axial','S','axial_conductance',(nc,nc)),p.quantity('injection','dimensionless','current_injection_distribution',(nc,n)),p.quantity('current_anchor','A','absolute_outer_segment_current',(n,)),p.quantity('initial_deviation','V','physical_voltage_deviation',(nc,n)),p.quantity('voltage_anchor','V','physical_membrane_voltage',(nc,n)) if spec.options.get('absolute_voltage_anchor',False) else None)
    states=tuple(MechanismStateSpec('deltaV_'+str(i),pop.cells.cell_id,(n,),'V','physical_voltage_deviation','Fixed declared cable calibration anchor') for i in range(nc))
    outputs={s.state_name:PortSpec(s.state_name,'V','physical_voltage_deviation',pop.cells.cell_id,tag) for s in states}
    if model.voltage_anchor is not None:
        outputs['terminal_voltage']=PortSpec('terminal_voltage','V','physical_membrane_voltage',pop.cells.cell_id,tag)
        outputs['voltage']=PortSpec('voltage','V','physical_membrane_voltage',pop.cells.cell_id,tag)
        outputs['gap_current']=PortSpec('gap_current','A','electrical_gap_current',pop.cells.cell_id,tag)
    terminal=spec.options.get('terminal_compartment')
    if type(terminal) is not int or not 0<=terminal<nc: raise ContractError('Explicit terminal compartment required')
    if model.voltage_anchor is not None:
        model.membrane_capacitance=np.full(n,model.C[terminal]);model.gap_state_index=terminal
    def initial():return tuple(Quantity(a,'V','physical_voltage_deviation',tag) for a in model.initial)
    def derivative(t,b,i):
        q=model.rhs(Quantity(np.array(b),'V','physical_voltage_deviation',tag),i['current'])
        values=q.values.copy()
        if '__gap_current__' in i:values[terminal]+=i['__gap_current__'].values/model.C[terminal]
        return tuple(Quantity(a,'V/s','physical_voltage_deviation_derivative',tag) for a in values)
    def observe(port,t,b,i):
        if port=='gap_current':return Quantity(np.zeros(n),'A','electrical_gap_current',tag)
        if port in ('terminal_voltage','voltage'):return Quantity(model.voltage_anchor[terminal]+b[terminal],'V','physical_membrane_voltage',tag)
        return Quantity(b[int(port.rsplit('_',1)[-1])],'V','physical_voltage_deviation',tag)
    return RuntimeComponent(spec.component_id,spec.mechanism,p,model,{'current':PortSpec('current','A','absolute_outer_segment_current',pop.cells.cell_id,tag)},outputs,states,initial,derivative,observe,{k:() for k in outputs},TimeContract('continuous'),('cone_passive_cable',),(pop,),tag,p.source_ids,_hash())

def activation_component(spec,populations,sources,mode):
    from .inner import _profile
    p,pop,tag=_profile(spec,populations,sources,mode);n=len(pop.cells.cell_id)
    half=p.quantity('Vhalf','V','activation_half_voltage',(n,));slope=p.quantity('slope','V','activation_voltage_slope',(n,),positive=True)
    if spec.options.get('approximation')!='quasi_steady_nonloading':raise ContractError('Explicit quasi-steady/nonloading approximation required; kinetics/current not inferred')
    inputs={name:PortSpec(name,'V',meaning,pop.cells.cell_id,tag) for name,meaning in [('voltage','physical_membrane_voltage'),('shift','activation_threshold_shift')]}
    def observe(port,t,b,i):
        q=terminal_activation(i['voltage'],half,slope,i['shift'])
        return q if port=='activation' else monotone_release_proxy(q,profile_name='quasi_steady_normalized_activation_proxy')
    outputs={'activation':PortSpec('activation','dimensionless','normalized_Ca_activation',pop.cells.cell_id,tag),'release':PortSpec('release','dimensionless','effective_synaptic_drive',pop.cells.cell_id,tag)}
    return RuntimeComponent(spec.component_id,spec.mechanism,p,None,inputs,outputs,(),lambda:(),lambda t,b,i:(),observe,{k:tuple(inputs) for k in outputs},TimeContract('continuous'),('terminal_activation_proxy',),(pop,),tag,p.source_ids,_hash())

def hc_feedback_component(spec,populations,sources,mode):
    from .inner import _profile
    import hashlib
    from pathlib import Path
    p,pop,tag=_profile(spec,populations,sources,mode);n=len(pop.cells.cell_id)
    tau=p.quantity('tau','s','effective_HC_time_constant',(n,),positive=True);K=p.quantity('K','dimensionless','HC_half_drive',(n,),positive=True);exponent=p.quantity('n','dimensionless','HC_hill_exponent',(n,),positive=True);gain=p.quantity('kappa','V','HC_threshold_shift_gain',(n,),positive=True);anchor=p.quantity('H_anchor','dimensionless','effective_HC_anchor',(n,));initialH=p.quantity('H_initial','dimensionless','effective_HC_activation',(n,))
    state=MechanismStateSpec('H',pop.cells.cell_id,(n,),'dimensionless','effective_HC_activation','Explicit physical-composition effective HC initial/anchor')
    def initial():return (Quantity(initialH.values,initialH.unit,initialH.meaning,tag),)
    def derivative(t,b,i):
        H=Quantity(b[0],'dimensionless','effective_HC_activation',tag);q=effective_HC_derivative(H,i['drive'],tau,K,exponent)
        return (Quantity(q.values,'dimensionless/s','effective_HC_activation_derivative',tag),)
    def observe(port,t,b,i):return effective_HC_shift(Quantity(b[0],'dimensionless','effective_HC_activation',tag),anchor,gain)
    return RuntimeComponent(spec.component_id,spec.mechanism,p,None,{'drive':PortSpec('drive','dimensionless','effective_synaptic_drive',pop.cells.cell_id,tag)},{'shift':PortSpec('shift','V','activation_threshold_shift',pop.cells.cell_id,tag)},(state,),initial,derivative,observe,{'shift':()},TimeContract('continuous'),('physical_effective_HC_feedback',),(pop,),tag,p.source_ids,hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

def unit_conversion_component(spec,populations,sources,mode):
    from .inner import _profile
    import hashlib
    from pathlib import Path
    p,pop,tag=_profile(spec,populations,sources,mode)
    required=('source_unit','target_unit','meaning')
    if any(k not in spec.options for k in required):raise ContractError('Explicit unit/semantic adapter configuration required')
    a=spec.options['source_unit'];b=spec.options['target_unit'];meaning=spec.options['meaning']
    explicit_unit_conversion(Quantity(np.zeros(len(pop.cells.cell_id)),a,meaning,tag),b)
    def observe(port,t,blocks,incoming):return explicit_unit_conversion(incoming['input'],b)
    return RuntimeComponent(spec.component_id,spec.mechanism,p,None,{'input':PortSpec('input',a,meaning,pop.cells.cell_id,tag)},{'output':PortSpec('output',b,meaning,pop.cells.cell_id,tag)},(),lambda:(),lambda t,s,i:(),observe,{'output':('input',)},TimeContract('continuous'),('explicit_SI_unit_adapter',),(pop,),tag,p.source_ids,hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
