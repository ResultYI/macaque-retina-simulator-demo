"""Engineering callback/port fixture ONLY; no rod/RBC/AII physiology is modeled.

The arbitrary first-order transport below exercises the plugin ABI. It is never
a default physiological mechanism, and is rejected outside numeric_fixture mode.
"""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from scipy import sparse
from retina_sim.contracts import Quantity,RunMode
from retina_sim.assembly import ComponentSpec,ModelSpec,ExternalInput,RetinaBuilder,PortSpec,RuntimeComponent,TimeContract
from retina_sim.mechanism_contracts import MechanismStateSpec,EvidenceValue
from retina_sim.plugins import PluginDeclaration,bind_mechanism_plugin
from retina_sim.core_sources import load_core_registry
from retina_sim.spatial import InputChannel
from examples.full_network_fixture import fixture_profile,population,q,link

PORTS={'RodMechanismPlugin':('light','rod_signal','R*/s','photoisomerization_rate','effective_synaptic_drive'),
       'RodBipolarMechanismPlugin':('rod_drive','bipolar_signal','dimensionless','effective_synaptic_drive','effective_synaptic_activation'),
       'AIIMechanismPlugin':('rod_bipolar','crossover_signal','dimensionless','effective_synaptic_activation','effective_synaptic_activation')}

def engineering_boundary_provider(spec,populations,sources,mode):
    """Arbitrary QA low-pass transport; not a source-supported retinal equation."""
    pop=populations[spec.population_roles['cells']];p=sources.mechanism_profiles[spec.parameter_profile_id];n=len(pop.cells.cell_id)
    incoming,outgoing,input_unit,input_meaning,output_meaning=PORTS[spec.mechanism]
    tau=p.quantity('tau','s','engineering_transport_time',(n,),positive=True).values
    gain=p.quantity('gain','s/R*' if input_unit=='R*/s' else 'dimensionless','engineering_transport_gain',(n,),positive=True).values
    initial=p.quantity('initial','dimensionless','receptor_gate',(n,)).values
    state=MechanismStateSpec('QA_transport',pop.cells.cell_id,(n,),'dimensionless','receptor_gate','Explicit arbitrary software transport initial state')
    def start():return (q(initial,'dimensionless','receptor_gate'),)
    def derivative(t,blocks,inputs):return (q((gain*inputs[incoming].values-blocks[0])/tau,'dimensionless/s','receptor_gate_derivative'),)
    def observe(port,t,blocks,inputs):return q(blocks[0],'dimensionless',output_meaning)
    return RuntimeComponent(spec.component_id,spec.mechanism,p,SimpleNamespace(),{incoming:PortSpec(incoming,input_unit,input_meaning,pop.cells.cell_id,'synthetic_fixture')},{outgoing:PortSpec(outgoing,'dimensionless',output_meaning,pop.cells.cell_id,'synthetic_fixture')},(state,),start,derivative,observe,{outgoing:()},TimeContract('continuous'),(),(pop,),'synthetic_fixture',(),'supplied by verified plugin binder')

def plugin_fixture(root,*,delays=(.0013,.0021),sampling=.0001):
    registry=load_core_registry(Path(root));builder=RetinaBuilder(registry);pops=[];components=[];by={}
    definitions=[('rod','RodMechanismPlugin','rod',.1),('RBC','RodBipolarMechanismPlugin','rod_BC',.2),('AII','AIIMechanismPlugin','AII_AC',.3)]
    for cid,mechanism,celltype,initial in definitions:
        pid='QA_plugin_'+cid;n=2
        fixture_profile(registry,pid,n,{'tau':(np.full(n,.002),'s','engineering_transport_time'),'gain':(np.full(n,.01 if cid=='rod' else 1),'s/R*' if cid=='rod' else 'dimensionless','engineering_transport_gain'),'initial':(np.full(n,initial),'dimensionless','receptor_gate')})
        pop=population(cid,celltype,(cid,)*n,np.array([[1,0],[1.01,0]]),('transition',)*n,pid,'Macaque_software_fixture');pops.append(pop);by[cid]=pop
        incoming,outgoing,input_unit,input_meaning,output_meaning=PORTS[mechanism]
        declaration=PluginDeclaration(mechanism,'QA_transport','software_fixture',(),'examples/plugin_boundary_fixture.py engineering_boundary_provider; arbitrary transport, no physiology','typed callback/graph/history ABI only',('rod phototransduction','rod BC biology','AII intrinsic/crossover/gap physiology'),{incoming:{'unit':input_unit,'meaning':input_meaning}},{outgoing:{'unit':'dimensionless','meaning':output_meaning}},(cid+'_plugin_boundary',))
        bind_mechanism_plugin(builder.registry,declaration,engineering_boundary_provider)
        components.append(ComponentSpec(cid,mechanism,{'cells':cid},pid,{'backend_id':'QA_transport'}))
    from dataclasses import replace
    first=link('rod','rod_signal','RBC','rod_drive',np.eye(2),by,'QA_plugin_RBC','functional_visual_field')
    delay_evidence=EvidenceValue('QA_connection_delays',q(delays,'s','connection_delay'),'software_fixture',None,'examples/plugin_boundary_fixture.py explicit software delay vector',{'fixture':'ABI/delay numerical check; no animal timing measurement'},'No biological uncertainty interpretation')
    first=replace(first,connections=replace(first.connections,delay_seconds=tuple(delays)),delay_prehistory=q([.1,.1],'dimensionless','effective_synaptic_drive'),delay_interpolation='linear',delay_sampling_seconds=sampling,delay_evidence=delay_evidence)
    second=link('RBC','bipolar_signal','AII','rod_bipolar',np.eye(2),by,'QA_plugin_AII','functional_visual_field')
    spec=ModelSpec('engineering_plugin_ABI_fixture',tuple(pops),tuple(components),(first,second),(ExternalInput('light','rod','light'),),scope='software_fixture_not_rod_model')
    compiled=builder.build(spec,RunMode.NUMERIC_FIXTURE)
    channel=InputChannel(0,.02,lambda t:q([100,100],'R*/s','photoisomerization_rate'),'R*/s','photoisomerization_rate','analytic','synthetic_fixture',{'kind':'synthetic plugin interface drive; no physiology'})
    return compiled,{'light':channel},builder,spec
