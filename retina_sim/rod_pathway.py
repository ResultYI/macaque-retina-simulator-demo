"""Executable primary rod circuit, with explicit source/approximation profiles.

Chen phototransduction is pinned author-discrete. Rod/RBC/AII membrane and
receptor parameters are caller-supplied; no hidden mammalian numerical defaults.
Electrical contacts are assembled once as undirected conductances by the network.
"""
from pathlib import Path
from types import SimpleNamespace,MappingProxyType
import hashlib,re
import numpy as np
from scipy.special import expit
from .contracts import ContractError,Quantity,StateSpec,canonical_hash
from .assembly import RuntimeComponent,PortSpec,TimeContract
from .mechanism_contracts import MechanismStateSpec
from .inner import _profile
from .chen2024 import Chen2024PeripheralLMCurrent,checked_source_values,COMMIT,_validate_rate_input,STRICT_INPUT_POLICY

def phototransduction_equilibrium(model,rate):
    """Derived source fixed point for an explicitly supplied constant total rate."""
    from scipy.optimize import brentq
    a=np.asarray(rate,float)
    if np.any(a<0) or not np.all(np.isfinite(a)):raise ContractError('Invalid adapting rate')
    p=model.p;R=p['gamma']*a/p['sigma'];P=(R+p['eta'])/p['phi']
    G=np.array([brentq(lambda g:model.smax/(1+(model.q*p['k']*g**p['n']/p['beta']/p['kGC'])**p['m'])-pp*g,0,model.gdark,xtol=1e-12) for pp in P])
    Ca=model.q*p['k']*G**p['n']/p['beta'];S=P*G
    return tuple(Quantity(v,s.unit,s.meaning,model.evidence_tag) for v,s in zip((R,P,G,Ca,S),model.state_specs))

def implementation_hash(): return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def rod_source_values(root):
    checked_source_values(root)  # Verify both original Git blobs before extracting another branch.
    text=(Path(root)/'data/source_snapshots/chen2024_1bcf7c9a/initPhotoreceptorParams.m').read_text()
    branch=re.search(r"case 'primateRod'(.*?)case 'mouseCone'",text,re.S).group(1)
    values={k:float(v) for k,v in re.findall(r'params\.(\w+)\s*=\s*([0-9.eE+-]+)\s*;',branch)}
    values['timeStep']=float(re.search(r'params.timeStep\s*=\s*([0-9.eE+-]+)',text).group(1))
    return values

class ChenPrimateRod:
    """Same five-state author update, primateRod branch, no cone parameter borrowing."""
    _step=Chen2024PeripheralLMCurrent._step
    def __init__(self,p,pop,tag,root):
        self.p=MappingProxyType(rod_source_values(root));self.parameters=p;self.evidence_tag=tag
        for key,value in self.p.items():
            records=[r for r in p.records if r.parameter_id==key]
            if len(records)!=1 or float(records[0].value.values)!=value or records[0].source_id!='CHEN2024_initPhotoreceptorParams_m':
                raise ContractError('Exact primateRod source coefficients required: '+key)
        self.dark_current=self.p['gdark']**self.p['n']*self.p['k'];self.gdark=self.p['gdark']
        self.q=self.p['beta']*self.p['cdark']/self.dark_current
        self.smax=self.p['eta']/self.p['phi']*self.gdark*(1+(self.p['cdark']/self.p['kGC'])**self.p['m'])
        units=('s^-2','s^-1','Chen_scaled_concentration','Chen_scaled_concentration','Chen_scaled_concentration/s');meanings=('effective_opsin_activity','effective_pde_activity','source_scaled_cgmp','source_scaled_calcium','source_scaled_cgmp_synthesis')
        self.state_specs=tuple(MechanismStateSpec(k,pop.cells.cell_id,(len(pop.cells.cell_id),),u,m,'Chen pinned primateRod dark initialization; '+COMMIT) for k,u,m in zip(('R','P','G','C','S'),units,meanings))
        self.model_hash=canonical_hash({'source_commit':COMMIT,'branch':'primateRod','parameters':self.p.copy(),'implementation':implementation_hash()})
        self.approximations=[{'id':'rod_source_scope','why':'Published/model fit primate rod coefficients; not independent validation, no regeneration/bleaching. Dark initial conditions are explicit.'}]
    def initial_conditions(self):
        values=(0,self.p['eta']/self.p['phi'],self.gdark,self.p['cdark'],self.gdark*self.p['eta']/self.p['phi'])
        return tuple(Quantity(np.full(s.shape,v),s.unit,s.meaning,self.evidence_tag) for s,v in zip(self.state_specs,values))
    def transition(self,blocks,incoming):
        rate=incoming['light'].require('R*/s','photoisomerization_rate',self.state_specs[0].shape)
        if np.any(rate<0):raise ContractError('Negative rod photon rate')
        return self._step(blocks,rate)

def rod_component(spec,populations,sources,mode):
    p,pop,tag=_profile(spec,populations,sources,mode)
    if set(pop.cells.cell_type)!={'rod'}:raise ContractError('Primate rod front end needs rod cells')
    model=ChenPrimateRod(p,pop,tag,sources.root);ids=pop.cells.cell_id
    initial=model.initial_conditions
    if 'initial_background_rate' in spec.options:
        rate=spec.options['initial_background_rate'].require('R*/s','photoisomerization_rate',(len(ids),))
        initial=lambda:phototransduction_equilibrium(model,rate)
    def observe(port,t,b,i):
        if port=='current':return Quantity(-model.p['k']*b[2]**model.p['n'],'pA','absolute_outer_segment_current',tag)
        s=next(s for s in model.state_specs if s.state_name==port);return Quantity(b[model.state_specs.index(s)],s.unit,s.meaning,tag)
    outputs={s.state_name:PortSpec(s.state_name,s.unit,s.meaning,ids,tag) for s in model.state_specs}
    outputs['current']=PortSpec('current','pA','absolute_outer_segment_current',ids,tag)
    def derivative(t,b,i):return tuple(Quantity(np.zeros(s.shape),s.unit+'/s',s.meaning+'_derivative',tag) for s in model.state_specs)
    return RuntimeComponent(spec.component_id,spec.mechanism,p,model,{'light':PortSpec('light','R*/s','photoisomerization_rate',ids,tag)},outputs,model.state_specs,initial,derivative,observe,{k:() for k in outputs},TimeContract('author_discrete',model.p['timeStep'],'Pinned R/P/C/G old state Euler; S(Cnew); hold outputs until next author tick'),('rod',),(pop,),tag,p.source_ids,implementation_hash())

def membrane_component(spec,populations,sources,mode):
    """Conductance membrane plus dynamic receptor gates and voltage-dependent release.

    ON bipolar: glutamate occupancy closes mGluR6/TRPM1 gate. OFF bipolar: drives
    ionotropic excitatory gate. AII: RBC AMPA excitation, explicit Na/K adaptation.
    OFF glycine receptors have their own gates/conductance/reversal per phenotype.
    """
    p,pop,tag=_profile(spec,populations,sources,mode);n=len(pop.cells.cell_id);ids=pop.cells.cell_id
    kind=spec.options.get('cell_kind')
    if kind not in ('rod_membrane','rod_BC_ON','cone_BC_ON','cone_BC_OFF','AII'):raise ContractError('Named receptor/membrane phenotype required')
    expected='rod' if kind=='rod_membrane' else 'AC' if kind=='AII' else 'BC'
    if spec.required_layer and spec.required_layer not in ({'rod_BC'} if kind=='rod_BC_ON' else {'AII_AC'} if kind=='AII' else set()):raise ContractError('Membrane target phenotype does not match actual circuit mechanism')
    if set(pop.cells.cell_type)!={expected}:raise ContractError('Membrane phenotype/cell type mismatch')
    def a(name,u,m,positive=False):return p.quantity(name,u,m,(n,),positive=positive).values
    C=a('C','F','membrane_capacitance',True);gL=a('gL','S','leak_conductance',True);EL=a('EL','V','leak_reversal');V0=a('V0','V','initial_membrane_voltage')
    half=a('release_half','V','activation_half_voltage');slope=a('release_slope','V','activation_voltage_slope',True);tau_release=a('tau_release','s','release_time_constant',True)
    states=[('V','V','physical_membrane_voltage'),('release','dimensionless','effective_release_activation')]
    initial_values=[V0,expit((V0-half)/slope)]
    inputs={};output_meaning='effective_synaptic_activation' if kind!='rod_membrane' else 'effective_synaptic_drive'
    if kind=='rod_membrane':
        anchor=a('current_anchor','A','absolute_outer_segment_current')
        inputs['current']=PortSpec('current','A','absolute_outer_segment_current',ids,tag)
    elif kind in ('rod_BC_ON','AII'):
        EE=a('EE','V','excitatory_reversal')
        inputs['excitation']=PortSpec('excitation','S','excitatory_conductance',ids,tag)
    else:
        K=a('K','dimensionless','receptor_half_drive',True);hill=a('hill','dimensionless','receptor_hill_exponent',True);tau_e=a('tau_e','s','excitatory_receptor_time_constant',True)
        ge=a('gE','S','maximum_excitatory_conductance',True);EE=a('EE','V','excitatory_reversal')
        drive_meaning='effective_synaptic_activation' if kind=='AII' else 'effective_synaptic_drive'
        inputs['drive']=PortSpec('drive','dimensionless',drive_meaning,ids,tag)
        e0=a('e0','dimensionless','initial_receptor_gate');states.append(('e','dimensionless','excitatory_synaptic_gate'));initial_values.append(e0)
        if kind=='cone_BC_OFF':
            EI=a('EI','V','inhibitory_reversal')
            inputs['inhibition']=PortSpec('inhibition','S','inhibitory_conductance',ids,tag)
    if kind in ('rod_BC_ON','AII'):
        rmax=a('rmax','events/s','release_attempt_rate',True);pool=a('Npool','dimensionless','release_pool_size',True);U=a('U','dimensionless','release_probability',True);rec=a('tau_recovery','s','release_recovery_time_constant',True)
        rh=a('r_half','V','release_half_voltage');rs=a('r_slope','V','release_voltage_slope',True)
        if np.any(U>1):raise ContractError('Release probability exceeds1')
        r0=rmax*expit((V0-rh)/rs);states.append(('x','dimensionless','adaptation_gate'));initial_values.append(1/(1+rec*U*r0))
    specs=tuple(MechanismStateSpec(k,ids,(n,),u,m,'Explicit named profile initial state; no equilibrium claim') for k,u,m in states)
    outputs={'voltage':PortSpec('voltage','V','physical_membrane_voltage',ids,tag),'release':PortSpec('release','dimensionless',output_meaning,ids,tag),'gap_current':PortSpec('gap_current','A','electrical_gap_current',ids,tag)}
    if kind!='rod_membrane':outputs['activation']=PortSpec('activation','dimensionless','effective_synaptic_activation',ids,tag)
    if kind in ('rod_BC_ON','AII'):outputs['event_rate']=PortSpec('event_rate','events/s','expected_release_event_rate',ids,tag)
    outputs.update({s.state_name:PortSpec(s.state_name,s.unit,s.meaning,ids,tag) for s in specs if s.state_name not in outputs})
    if kind=='cone_BC_OFF':outputs['glycine_conductance']=PortSpec('glycine_conductance','S','inhibitory_conductance',ids,tag)
    def initial():return tuple(Quantity(v,s.unit,s.meaning,tag) for s,v in zip(specs,initial_values))
    def derivative(t,b,incoming):
        V,rel=b[:2];current=gL*(EL-V)+incoming.get('__gap_current__',Quantity(np.zeros(n),'A','electrical_gap_current',tag)).values
        ds=[None,(expit((V-half)/slope)-rel)/tau_release]
        if kind=='rod_membrane':current-=incoming['current'].values-anchor
        elif kind in ('rod_BC_ON','AII'):
            current+=incoming['excitation'].values*(EE-V)
            x=b[2];attempts=rmax*expit((V-rh)/rs);ds.append((1-x)/rec-U*attempts*x)
        else:
            x=incoming['drive'].values
            if np.any(x<0):raise ContractError('Negative receptor drive')
            occupancy=expit(hill*np.log(np.maximum(x,np.finfo(float).tiny)/K));occupancy=np.where(x==0,0,occupancy)
            target=1-occupancy if kind in ('rod_BC_ON','cone_BC_ON') else occupancy
            e=b[2];current+=ge*e*(EE-V);ds.append((target-e)/tau_e)
            if kind=='cone_BC_OFF':
                gly=incoming['inhibition'].values
                if np.any(gly<0):raise ContractError('Negative glycine conductance')
                current+=gly*(EI-V)
        ds[0]=current/C
        return tuple(Quantity(v,s.unit+'/s',s.meaning+'_derivative',tag) for s,v in zip(specs,ds))
    def observe(port,t,b,i):
        if port=='voltage':return Quantity(b[0],'V','physical_membrane_voltage',tag)
        if port in ('release','activation'):return Quantity(b[1],'dimensionless',outputs[port].meaning,tag)
        if port=='gap_current':return Quantity(np.zeros(n),'A','electrical_gap_current',tag)  # overwritten by explicit undirected network operator
        if port=='event_rate':return Quantity(pool*U*rmax*expit((b[0]-rh)/rs)*b[2],'events/s','expected_release_event_rate',tag)
        if port=='glycine_conductance':return i['inhibition']
        j=next(j for j,s in enumerate(specs) if s.state_name==port);s=specs[j];return Quantity(b[j],s.unit,s.meaning,tag)
    model=SimpleNamespace(membrane_capacitance=C,gap_state_index=0,model_hash=canonical_hash({'kind':kind,'profile':p.parameter_hash,'implementation':implementation_hash()}),approximations=[{'id':p.profile_id,'why':'Finite receptor/conductance and ribbon resource model; values individually classified. Passive AII membrane is a named reduction; active AII Na/Ih channels remain uncalibrated and unmodeled.'}])
    layers={'rod_membrane':('rod_membrane','rod_terminal'),'rod_BC_ON':('rod_BC',),'AII':('AII_AC',),'cone_BC_ON':('BC_ON',),'cone_BC_OFF':('BC_OFF',)}[kind]
    if spec.required_layer:layers=layers+(spec.required_layer,)
    return RuntimeComponent(spec.component_id,spec.mechanism,p,model,inputs,outputs,specs,initial,derivative,observe,{k:(('inhibition',) if k=='glycine_conductance' else ()) for k in outputs},TimeContract('continuous'),layers,(pop,),tag,p.source_ids,implementation_hash())

def install_primary_rod_mechanisms(registry):
    registry.register('Chen2024PrimateRodCurrent',rod_component)
    registry.register('ReceptorConductanceMembrane',membrane_component)
    registry.register('RodContactTRPM1',contact_component)
    registry.register('DoubleExponentialChemicalSynapse',chemical_component)
    registry.register('HistoryBipolarTerminal',history_terminal_component)

def history_terminal_component(spec,populations,sources,mode):
    """Keep the frozen BC causal kernel; add a continuous physical terminal state.

    The generator→V target gain is an explicit project calibration. Electrical
    currents enter V/C, never an arbitrary native generator variable.
    """
    from .inner import CausalHistory
    p,pop,tag=_profile(spec,populations,sources,mode);n=len(pop.cells.cell_id);ids=pop.cells.cell_id
    polarity=spec.options.get('polarity')
    if polarity not in ('ON','OFF'):raise ContractError('Explicit BC polarity required')
    allowed={'ON':{'midget_BC_ON','diffuse_BC_ON','S_ON_BC'},'OFF':{'midget_BC_OFF','diffuse_BC_OFF'}}
    if set(pop.cells.cell_type)!={'BC'} or spec.required_layer not in allowed[polarity]:raise ContractError('History/terminal BC requires matching physical cell type and polarity phenotype')
    hist=CausalHistory(p,pop,tag,'ResidualBC_'+polarity,{'polarity_representation':'canonical_dominant_positive','presynaptic_inhibition':False})
    def a(k,u,m,positive=False):return p.quantity(k,u,m,(n,),positive=positive).values
    C=a('C','F','membrane_capacitance',True);gL=a('gL','S','leak_conductance',True);Vref=a('Vref','V','terminal_voltage_reference');gain=a('native_voltage_gain','V','voltage_per_native_activation',True);aref=a('native_reference','dimensionless','native_activation_reference')
    V0=a('V0','V','initial_membrane_voltage');half=a('release_half','V','activation_half_voltage');slope=a('release_slope','V','activation_voltage_slope',True);tau=a('tau_release','s','release_time_constant',True)
    specs=hist.state_specs+(MechanismStateSpec('V',ids,(n,),'V','physical_membrane_voltage','Explicit physical terminal reference with native-to-voltage gain'),MechanismStateSpec('terminal_release',ids,(n,),'dimensionless','effective_release_activation','Monotone initial terminal activation'))
    count=len(hist.state_specs);inputs={'drive':PortSpec('drive','dimensionless','effective_synaptic_drive',ids,tag)}
    if polarity=='OFF':
        EI=a('EI','V','inhibitory_reversal');inputs['inhibition']=PortSpec('inhibition','S','inhibitory_conductance',ids,tag)
    outputs={'activation':PortSpec('activation','dimensionless','effective_synaptic_activation',ids,tag),'generator_activation':PortSpec('generator_activation','dimensionless','effective_synaptic_activation',ids,tag),'voltage':PortSpec('voltage','V','physical_membrane_voltage',ids,tag),'gap_current':PortSpec('gap_current','A','electrical_gap_current',ids,tag)}
    if polarity=='OFF':outputs['glycine_conductance']=PortSpec('glycine_conductance','S','inhibitory_conductance',ids,tag)
    def initial():return hist.initial()+(Quantity(V0,'V','physical_membrane_voltage',tag),Quantity(expit((V0-half)/slope),'dimensionless','effective_release_activation',tag))
    def transition(b,i,o):
        history=hist.candidate(b[:count],i);activation=hist.activation(history,i)
        return tuple(history)+(activation,)+tuple(b[count:])
    def derivative(t,b,i):
        V,rel=b[count:];target=Vref+gain*(b[count-1]-aref)
        current=gL*(target-V)+i.get('__gap_current__',Quantity(np.zeros(n),'A','electrical_gap_current',tag)).values
        if polarity=='OFF':current+=i['inhibition'].values*(EI-V)
        rows=[np.zeros(s.shape) for s in hist.state_specs]+[current/C,(expit((V-half)/slope)-rel)/tau]
        return tuple(Quantity(v,s.unit+'/s',s.meaning+'_derivative',tag) for v,s in zip(rows,specs))
    def observe(port,t,b,i):
        if port=='activation':return Quantity(b[-1],'dimensionless','effective_synaptic_activation',tag)
        if port=='generator_activation':return hist.observe('activation',t,b[:count],i)
        if port=='voltage':return Quantity(b[-2],'V','physical_membrane_voltage',tag)
        if port=='glycine_conductance':return i['inhibition']
        return Quantity(np.zeros(n),'A','electrical_gap_current',tag)
    native=SimpleNamespace(transition=transition,membrane_capacitance=C,gap_state_index=count,model_hash=canonical_hash({'profile':p.parameter_hash,'implementation':implementation_hash(),'polarity':polarity}),approximations=[{'id':'explicit_native_to_terminal_voltage','why':'Original causal BC kernel retained. A declared voltage gain/reference supplies the continuous terminal compartment; it is not measured macaque voltage conversion.'}])
    deps={name:(('drive',) if name=='generator_activation' else ('inhibition',) if name=='glycine_conductance' else ()) for name in outputs}
    return RuntimeComponent(spec.component_id,spec.mechanism,p,native,inputs,outputs,specs,initial,derivative,observe,deps,TimeContract('causal_history',hist.dt,'Frozen causal BC history at own ticks; physical terminal continuously integrates chemical and reciprocal electrical currents'),('BC_'+polarity,spec.required_layer),(pop,),tag,p.source_ids,implementation_hash())

def contact_component(spec,populations,sources,mode):
    """One actual mGluR6/TRPM1 gate per rod→RBC contact; nonlinear before pooling."""
    p,pop,tag=_profile(spec,populations,sources,mode);n=len(pop.cells.cell_id);ids=pop.cells.cell_id
    K=p.quantity('K','dimensionless','receptor_half_drive',(n,),positive=True).values
    h=p.quantity('hill','dimensionless','receptor_hill_exponent',(n,),positive=True).values
    tau=p.quantity('tau','s','receptor_time_constant',(n,),positive=True).values
    gain=p.quantity('gain','S','TRPM1_maximum_contact_conductance',(n,),nonnegative=True).values
    drive0=p.quantity('drive0','dimensionless','initial_release_drive',(n,),nonnegative=True).values
    target=lambda x:expit(-h*np.log(np.maximum(x,np.finfo(float).tiny)/K))
    state=MechanismStateSpec('m',ids,(n,),'dimensionless','receptor_gate','Initial receptor equilibrium at explicit rod release')
    def initial():return (Quantity(target(drive0),'dimensionless','receptor_gate',tag),)
    def derivative(t,b,i):
        x=i['release'].values
        if np.any(x<0):raise ContractError('Negative first synapse glutamate')
        return (Quantity((target(x)-b[0])/tau,'dimensionless/s','receptor_gate_derivative',tag),)
    def observe(port,t,b,i):return Quantity(gain*b[0],'S','excitatory_conductance',tag) if port=='conductance' else Quantity(b[0],'dimensionless','receptor_gate',tag)
    outputs={'conductance':PortSpec('conductance','S','excitatory_conductance',ids,tag),'gate':PortSpec('gate','dimensionless','receptor_gate',ids,tag)}
    return RuntimeComponent(spec.component_id,spec.mechanism,p,None,{'release':PortSpec('release','dimensionless','effective_synaptic_drive',ids,tag)},outputs,(state,),initial,derivative,observe,{k:() for k in outputs},TimeContract('continuous'),('individual_rod_RBC_receptor',),(pop,),tag,p.source_ids,implementation_hash())

def chemical_component(spec,populations,sources,mode):
    """Per-edge expected-vesicle release, peak-normalized causal conductance.

    No double counting of discrete events; this mode uses expected events only.
    a/b are integrated continuously with the joint solver, never author Euler.
    """
    p,pop,tag=_profile(spec,populations,sources,mode);n=len(pop.cells.cell_id);ids=pop.cells.cell_id
    if spec.options.get('empty_contact_operator') is True:
        if spec.options.get('empty_support_verified') is not True or p.records:
            raise ContractError('Empty contact sum requires verified complete finite support and no instantiated contact parameters')
        receptor=spec.options.get('receptor')
        if receptor not in ('excitatory','inhibitory'):raise ContractError('Named chemical receptor required')
        meaning=receptor+'_conductance'
        def initial():return ()
        def derivative(t,b,i):return ()
        def observe(port,t,b,i):return Quantity(np.zeros(n),'S',meaning,tag)
        native=SimpleNamespace(model_hash=canonical_hash({'empty_contact_sum':True,'profile':p.parameter_hash,'implementation':implementation_hash()}),
            approximations=[{'id':'legal_empty_finite_contact_sum','why':'Zero contribution of an empty, covered finite contact graph; no dummy contact/state and no measured zero-connectivity claim'}])
        return RuntimeComponent(spec.component_id,spec.mechanism,p,native,{},
            {'conductance':PortSpec('conductance','S',meaning,ids,tag)},(),initial,derivative,observe,
            {'conductance':()},TimeContract('continuous'),('chemical_'+receptor+'_conductance',),(pop,),tag,p.source_ids,implementation_hash())
    def a(k,u,m):return p.quantity(k,u,m,(n,),positive=True).values
    rise=a('tau_rise','s','synaptic_rise_time_constant');decay=a('tau_decay','s','synaptic_decay_time_constant')
    peak=a('g_peak','S','single_event_peak_conductance');contacts=a('ncontact','dimensionless','effective_release_site_multiplier')
    rate0=p.quantity('event_rate0','events/s','expected_release_event_rate',(n,),nonnegative=True).values
    if np.any(decay<=rise):raise ContractError('Double exponential decay must exceed rise')
    tp=np.log(decay/rise)/(1/rise-1/decay);Z=np.exp(-tp/decay)-np.exp(-tp/rise)
    specs=tuple(MechanismStateSpec(k,ids,(n,),'dimensionless','receptor_gate','Steady expected event occupancy from explicit presynaptic initial rate') for k in ('a','b'))
    receptor=spec.options.get('receptor')
    if receptor not in ('excitatory','inhibitory'):raise ContractError('Named chemical receptor required')
    meaning=receptor+'_conductance'
    enabled=spec.options.get('enabled',True)
    if type(enabled) is not bool:raise ContractError('Chemical ablation is explicit boolean')
    voltage_decay=spec.options.get('voltage_dependent_decay',False)
    inputs={'event_rate':PortSpec('event_rate','events/s','expected_release_event_rate',ids,tag)}
    if voltage_decay:
        low=a('decay_minus100','s','synaptic_decay_time_constant');high=a('decay_0','s','synaptic_decay_time_constant')
        inputs['postsynaptic_voltage']=PortSpec('postsynaptic_voltage','V','physical_membrane_voltage',ids,tag)
    def kinetics(incoming):
        if not voltage_decay:return decay,Z
        voltage=incoming['postsynaptic_voltage'].values
        # Source endpoints only; interpolation/clamping is explicitly project approximation.
        tau=low+(high-low)*np.minimum(1.,np.maximum(0.,(voltage+.1)/.1))
        tp=np.log(tau/rise)/(1/rise-1/tau);z=np.exp(-tp/tau)-np.exp(-tp/rise)
        return tau,z
    def initial():return tuple(Quantity(rate0*tau,'dimensionless','receptor_gate',tag) for tau in (rise,decay))
    def derivative(t,b,i):
        rate=i['event_rate'].values
        if np.any(rate<0):raise ContractError('Negative release event rate')
        tau,_=kinetics(i)
        return tuple(Quantity(-v/tau0+rate,'dimensionless/s','receptor_gate_derivative',tag) for v,tau0 in zip(b,(rise,tau)))
    def observe(port,t,b,i):
        _,z=kinetics(i)
        value=contacts*peak*(b[1]-b[0])/z if enabled else np.zeros(n)
        if np.any(value<-1e-18):raise ContractError('Negative chemical conductance; no physiological clipping')
        return Quantity(value,'S',meaning,tag)
    native=SimpleNamespace(model_hash=canonical_hash({'normalization':Z,'profile':p.parameter_hash,'implementation':implementation_hash()}),unit_event_peak=lambda t:contacts*peak*(np.exp(-np.asarray(t)/decay)-np.exp(-np.asarray(t)/rise))/Z)
    return RuntimeComponent(spec.component_id,spec.mechanism,p,native,inputs,{'conductance':PortSpec('conductance','S',meaning,ids,tag)},specs,initial,derivative,observe,{'conductance':(('postsynaptic_voltage',) if voltage_decay else ())},TimeContract('continuous'),('chemical_'+receptor+'_conductance',),(pop,),tag,p.source_ids,implementation_hash())
