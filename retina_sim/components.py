"""Executable adapters for pinned references; never an implicit physiological chain."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
from .contracts import ContractError,Quantity,StateSpec,RunMode,canonical_hash
from .assembly import RuntimeComponent,PortSpec,TimeContract,ComponentRegistry
from .reference import SomaratnaFreeman2025GaussianHCReference,build_somaratna_weights,_values,COMMIT as SF_COMMIT
from .chen2024 import Chen2024PeripheralLMCurrent,ConeScope,SPECIES as CHEN_SPECIES

SF_KEY='SomaratnaFreeman2025GaussianHCReference'
CHEN_KEY='Chen2024PeripheralLMCurrent'
SF_STATES=('cone','horizontal','b_off','b_on','g_off','g_on')
SF_LAYERS={'cone':'cone','horizontal':'HC','b_off':'BC_OFF','b_on':'BC_ON','g_off':'RGC_OFF','g_on':'RGC_ON'}
SF_SOURCES=('SF2025_runColFun_m','SF2025_runColTask_m')

def _binding(spec,populations,sources,mode,roles):
    if spec.options.get('explicit_opt_in') is not True: raise ContractError('Named reference requires explicit component opt-in')
    if set(spec.population_roles)!=set(roles): raise ContractError('Explicit population roles required: '+','.join(roles))
    p=sources.load_parameters(spec.parameter_profile_id,mode)
    pops={role:populations[spec.population_roles[role]] for role in roles}
    tags={tag for pop in pops.values() for tag in pop.cells.evidence_tags}
    if len(tags)!=1: raise ContractError('One explicit evidence class required per component binding')
    tag=tags.pop()
    if mode==RunMode.NUMERIC_FIXTURE and tag!='synthetic_fixture': raise ContractError('Numeric population must be synthetic_fixture')
    for pop in pops.values():
        if set(pop.cells.parameter_profile_id)!={p.profile.profile_id}: raise ContractError('Cell/parameter profile mismatch')
        if any(s not in (p.profile.species,p.profile.profile_id) for s in pop.cells.species_profile_id): raise ContractError('Cell/source species profile mismatch; no fallback')
    return p,pops,tag

def _implementation(): return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def _port(name,unit,meaning,pop,tag): return PortSpec(name,unit,meaning,pop.cells.cell_id,tag)

def chen_component(spec,populations,sources,mode):
    p,pops,tag=_binding(spec,populations,sources,mode,('cone',)); pop=pops['cone']; cells=pop.cells
    cells.eccentricity.require('deg','global_eccentricity',(len(cells.cell_id),),'functional_visual_field')
    if any(t!='cone' for t in cells.cell_type): raise ContractError('Chen scope requires cone population')
    scope=ConeScope(cells.cell_id,cells.subtype,tuple(cells.eccentricity.values),CHEN_SPECIES,tag)
    model=Chen2024PeripheralLMCurrent(p,scope,explicit_opt_in=True,source_reproduction=spec.options.get('source_reproduction',False))
    outputs={s.state_name:_port(s.state_name,s.unit,s.meaning,pop,tag) for s in model.state_specs}
    outputs['current']=_port('current','pA','absolute_outer_segment_current',pop,tag)
    def derivative(t,b,i):return tuple(Quantity(np.zeros(s.shape),s.unit+'/s',s.meaning+'_derivative',tag) for s in model.state_specs)
    def observe(port,t,b,i):
        if port=='current':return Quantity(-model.p['k']*b[2]**model.p['n'],'pA','absolute_outer_segment_current',tag)
        s=next(s for s in model.state_specs if s.state_name==port);return Quantity(b[model.state_specs.index(s)],s.unit,s.meaning,tag)
    def transition(b,i):
        rate=i['light'].require('R*/s','photoisomerization_rate',model.state_specs[0].shape)
        from .chen2024 import _validate_rate_input
        _validate_rate_input(rate,model.input_policy)
        return model._step(b,rate)
    model.transition=transition
    initial=model.initial_conditions
    if 'initial_background_rate' in spec.options:
        from .rod_pathway import phototransduction_equilibrium
        rate=spec.options['initial_background_rate'].require('R*/s','photoisomerization_rate',(len(cells.cell_id),))
        initial=lambda:phototransduction_equilibrium(model,rate)
    return RuntimeComponent(spec.component_id,spec.mechanism,p,model,{'light':_port('light','R*/s','photoisomerization_rate',pop,tag)},outputs,model.state_specs,initial,derivative,observe,{name:() for name in outputs},TimeContract('author_discrete',.0001,'Pinned source Euler order; R/P/C/G old-state, S(Cnew); batch N-1 versus stream N intervals'),('cone_outer_segment',),(pop,),tag,('CHEN2024_BiophysModel_m','CHEN2024_initPhotoreceptorParams_m'),_implementation())

def somaratna_component(spec,populations,sources,mode):
    p,pops,tag=_binding(spec,populations,sources,mode,SF_STATES)
    cone=pops['cone']; gang=pops['g_off']
    if cone.cells.rf_position is None or gang.cells.rf_position is None: raise ContractError('Reference RF coordinates missing')
    e=cone.cells.eccentricity.require('deg','global_eccentricity',frame='functional_visual_field')
    if not np.all(e==e[0]): raise ContractError('SF reference requires explicit single-eccentricity geometry')
    graph=build_somaratna_weights(cone.cells.rf_position,cone.cells.subtype,gang.cells.rf_position,Quantity(e[0],'deg','global_eccentricity',tag,'functional_visual_field'),p,tag)
    model=SomaratnaFreeman2025GaussianHCReference(p,graph,explicit_opt_in=True)
    for role,size in zip(SF_STATES,model.sizes):
        cells=pops[role].cells
        expected=cone.cells.rf_position if role in SF_STATES[:4] else gang.cells.rf_position
        if len(cells.cell_id)!=size or cells.rf_position is None or not np.array_equal(cells.rf_position.values,expected.values) or cells.rf_position.unit!=expected.unit or cells.rf_position.coordinate_frame!=expected.coordinate_frame:
            raise ContractError('SF column/population geometry must agree with pinned reference')
    states=tuple(StateSpec(s.state_name,pops[role].cells.cell_id,s.shape,s.unit,s.meaning,s.initial_condition_source) for role,s in zip(SF_STATES,model.state_specs))
    outputs={s.state_name:_port(s.state_name,s.unit,s.meaning,pops[role],tag) for role,s in zip(SF_STATES,states)}
    outputs['feedback_output']=_port('feedback_output','mV','algebraic_feedback',cone,tag)
    for branch in ('off','on'): outputs['rate_'+branch]=_port('rate_'+branch,'Hz','firing_rate',pops['g_'+branch],tag)
    def derivative(t,blocks,inputs):
        q=model.rhs(t,np.concatenate(blocks),inputs['contrast'],p,graph)
        return tuple(Quantity(a,s.unit+'/s',s.meaning+'_derivative',tag) for a,s in zip(model.unpack(q.values),states))
    def observe(port,t,blocks,inputs):
        native=model.make_result(np.array([t]),np.concatenate(blocks)[:,None],{},{}).outputs_with_units[port]
        return Quantity(native.values[:,0],native.unit,native.meaning,tag)
    return RuntimeComponent(spec.component_id,spec.mechanism,p,model,{'contrast':_port('contrast','dimensionless','cone_contrast',cone,tag)},outputs,states,model.initial_conditions,derivative,observe,{name:() for name in outputs},TimeContract('continuous'),tuple(SF_LAYERS.values()),tuple(pops.values()),tag,SF_SOURCES,_implementation(),(('cone','horizontal'),('horizontal','cone'),('cone','b_off'),('cone','b_on'),('b_off','g_off'),('b_on','g_on')))

def somaratna_layer_component(spec,populations,sources,mode):
    """Source equation decomposition, enabling simultaneous externally wired feedback."""
    layer=spec.mechanism.rsplit('.',1)[-1]
    if layer not in SF_STATES: raise ContractError('Unknown SF source layer')
    p,pops,tag=_binding(spec,populations,sources,mode,('cells',)); pop=pops['cells']; values=_values(p)
    if p.profile.profile_id!='SomaratnaFreeman2025GaussianHCReference_source_specific' or mode==RunMode.FULL_PHYSIOLOGY:
        raise ContractError('Source layer requires isolated SF reference profile')
    tau=float(values['tau']); sens=float(values['kSens']); surround=float(values['kSur']); rect=float(values['kRect'])
    rest=values['actRest']/rect; n=len(pop.cells.cell_id)
    meaning='generator_with_output_offset' if layer.startswith('g_') else 'generator_deviation'
    state=StateSpec(layer,pop.cells.cell_id,(n,),'mV',meaning,'SF2025_runColFun_m solveT 473-481 '+SF_COMMIT)
    inputs={}
    if layer=='cone':
        inputs['contrast']=_port('contrast','dimensionless','cone_contrast',pop,tag)
        inputs['horizontal']=_port('horizontal','mV','generator_deviation',pop,tag)
    else: inputs['drive']=_port('drive','mV','generator_deviation',pop,tag)
    outputs={'generator':_port('generator','mV',meaning,pop,tag)}
    if layer.startswith('g_'): outputs['rate']=_port('rate','Hz','firing_rate',pop,tag)
    offset=float(rest[0 if layer=='g_off' else 1]) if layer.startswith('g_') else 0.
    def initial(): return (Quantity(np.full(n,offset),'mV',meaning,tag),)
    def derivative(t,blocks,incoming):
        y=blocks[0]
        if layer=='cone': value=(-sens*incoming['contrast'].values-surround*incoming['horizontal'].values-y)/tau
        elif layer=='b_on': value=(-incoming['drive'].values-y)/tau
        else: value=(incoming['drive'].values+offset-y)/tau
        return (Quantity(value,'mV/s',meaning+'_derivative',tag),)
    def observe(port,t,blocks,incoming):
        return Quantity(rect*np.maximum(0,blocks[0]),'Hz','firing_rate',tag) if port=='rate' else Quantity(blocks[0],'mV',meaning,tag)
    native=SimpleNamespace(model_hash=canonical_hash({'source':SF_COMMIT,'layer':layer,'implementation':_implementation()}))
    return RuntimeComponent(spec.component_id,spec.mechanism,p,native,inputs,outputs,(state,),initial,derivative,observe,{k:() for k in outputs},TimeContract('continuous'),(SF_LAYERS[layer],),(pop,),tag,SF_SOURCES,_implementation())

def reference_component_registry():
    registry=ComponentRegistry()
    from .bc_ac_contact_components import BC_KEY as contact_bc, AC_KEY as contact_ac, STATIC_KEY, membrane_component as contact_membrane, static_component
    registry.register(contact_bc,contact_membrane)
    registry.register(contact_ac,contact_membrane)
    registry.register(STATIC_KEY,static_component)
    registry.register(CHEN_KEY,chen_component)
    registry.register(SF_KEY,somaratna_component)
    for layer in SF_STATES: registry.register('SomaratnaFreeman2025.'+layer,somaratna_layer_component)
    from .vh2005 import KEY,vh_component
    from .inner import history_component,conductance_component,rgc_component
    registry.register(KEY,vh_component)
    from .finite_feedback_native import KEY as finite_feedback_key,finite_feedback_component
    registry.register(finite_feedback_key,finite_feedback_component)
    for key in ('ResidualBC_ON','ResidualBC_OFF','ResidualAC'): registry.register(key,history_component)
    from .bc_ac_receptor_components import BC_KEY, AC_KEY, cone_bc_component, generic_ac_component
    registry.register(BC_KEY, cone_bc_component)
    registry.register(AC_KEY, generic_ac_component)
    registry.register('SynapticConductance',conductance_component)
    registry.register('HistoryConductanceRGC',rgc_component)
    from .physical import cable_component,activation_component,hc_feedback_component,unit_conversion_component
    registry.register('PassiveConeCable',cable_component)
    registry.register('QuasiSteadyTerminalActivation',activation_component)
    registry.register('EffectiveHCThresholdFeedback',hc_feedback_component)
    registry.register('ExplicitPhysicalUnitConversion',unit_conversion_component)
    from .plugins import install_plugin_boundaries
    install_plugin_boundaries(registry)
    from .rod_pathway import install_primary_rod_mechanisms
    install_primary_rod_mechanisms(registry)
    return registry
