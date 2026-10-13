"""Explicit synthetic integration fixture, never a calibrated macaque retina.

Nine numerical cones exercise center/transition/periphery and L/M/S. All inner
numbers and population resolution are declared QA values. No fitting or data claim.
"""
from pathlib import Path
import numpy as np
from scipy import sparse
from retina_sim.contracts import CellTable,Quantity,RunMode
from retina_sim.spatial import Population,GazeTrajectory,gaze_driven_channel
from retina_sim.assembly import ModelSpec,ComponentSpec,ExternalInput,PortLink,RetinaBuilder
from retina_sim.graph import ConnectionTable
from retina_sim.core_sources import load_core_registry
from retina_sim.mechanism_contracts import MechanismProfile,EvidenceValue,register_mechanism_profile
from retina_sim.vh2005 import PROFILE,KEY
from retina_sim.geometry import PackerDensityField,McGregorDisplacement
from retina_sim.density import read_density_tabulation
from retina_sim.runtime_profiles import frozen_packer_table
import json

def q(values,unit,meaning,frame=None):return Quantity(values,unit,meaning,'synthetic_fixture',frame)
def fixture_profile(registry,name,n,entries):
    records=tuple(EvidenceValue(key,q(value,unit,meaning),'software_fixture',None,'Explicit QA configuration; NOT fitted/measured physiological value',{'fixture':'software integration','region':'synthetic3region'},'No biological uncertainty interpretation') for key,(value,unit,meaning) in entries.items())
    profile=MechanismProfile(name,'Macaque_software_fixture',{'fixture':'synthetic, no animal data','cell_count':n},records,(),'software_fixture')
    register_mechanism_profile(registry,profile);return profile

def population(pid,cell_type,subtypes,xy,regions,profile,species,anatomical=None,pedicles=None):
    ids=tuple(f'{pid}:{i}' for i in range(len(xy)));n=len(ids)
    cells=CellTable(ids,(cell_type,)*n,tuple(subtypes),(species,)*n,('specified_fixture',)*n,('specified_fixture',)*n,None,q(xy,'mm','functional_position','functional_visual_field'),'mm','functional_visual_field',q(np.linalg.norm(xy,axis=1),'mm','global_eccentricity','functional_visual_field'),(profile,)*n,('synthetic_fixture',)*n)
    anatomy=anatomical if anatomical is not None else np.asarray(xy)
    # QA point morphology is explicit. Cones have OPL pedicles, BC/AC terminals
    # and RGC dendrites use IPL; a RGC soma map is never a cone IS coordinate.
    pedicle=q(pedicles if pedicles is not None else anatomy,'mm','pedicle_position','OPL') if cell_type=='cone' else None
    terminal=q(anatomy,'mm','terminal_position','IPL') if cell_type in ('BC','AC') else None
    dendrite=q(anatomy,'mm','dendritic_position','OPL' if cell_type in ('BC','HC') else 'IPL') if cell_type!='cone' else None
    return Population(pid,cells,tuple(regions),q(anatomy,'mm','anatomical_position','anatomical_retina'),q(xy,'mm','functional_position','functional_visual_field'),pedicle,terminal,dendrite)

def link(pre,pre_port,post,post_port,matrix,pops_by_component,profile_id,frame):
    pre_ids=pops_by_component[pre].cells.cell_id;post_ids=pops_by_component[post].cells.cell_id
    m=sparse.coo_matrix(matrix);n=len(m.data)
    table=ConnectionTable(tuple(f'{pre}.{pre_port}>{post}.{post_port}:{i}' for i in range(n)),tuple(pre_ids[i] for i in m.col),tuple(post_ids[i] for i in m.row),('fixture_explicit_receptor',)*n,tuple(float(x) for x in m.data),('dimensionless',)*n,('nonnegative_weight_mechanism_owns_sign',)*n,(0.,)*n,(profile_id,)*n,('explicit_operator_coefficients',)*n,('synthetic_fixture',)*n)
    return PortLink(pre,pre_port,post,post_port,table,frame)

def full_fixture(root,presynaptic_loop=False,displacement=None):
    root=Path(root);registry=load_core_registry(root)
    r=np.array([.02,.021,.022,1.,1.002,1.004,7.,7.002,7.004]);xy=np.c_[r,np.tile([-.002,0,.002],3)]
    regions=tuple(x for x in ('center','transition','periphery') for _ in range(3))
    manifest=json.loads((root/'data/published_code/packer_cone_density.manifest.json').read_text())
    # Existing fixed code-tabulation identity, no invented cone-density curve.
    raw=root/'data/published_code/packer_cone_density.csv'
    import hashlib
    field=PackerDensityField(frozen_packer_table(root),central_policy='central_constant_cap',angular_policy='log_NT_cosine')
    nodes=json.loads((root/'sources/spatial/mcgregor2018_production_map_nodes.json').read_text())
    displacement=displacement or McGregorDisplacement(nodes,field,profile_name='published_rounded_anchor_composite')
    mapped=displacement.apply(q(xy,'mm','functional_position','functional_visual_field'))
    cones=population('cones','cone',('L','M','S')*3,xy,regions,PROFILE,'macaque_H1_source_unspecified',xy,mapped['pedicle'].values)
    centers=xy[1::3];anatomy=mapped['pedicle'].values[1::3]
    hcxy=np.repeat(centers,2,axis=0);hcreg=tuple(x for x in ('center','transition','periphery') for _ in range(2))
    hcs=population('hcs','HC',('H1','H2')*3,hcxy,hcreg,PROFILE,'macaque_H1_source_unspecified',np.repeat(anatomy,2,axis=0))
    A=np.zeros((6,9));B=np.zeros((9,6))
    for i in range(3):
        A[2*i,3*i:3*i+2]=.5;A[2*i+1,3*i:3*i+3]=1/3
        B[3*i:3*i+2,2*i:2*i+2]=.5;B[3*i+2,2*i+1]=1
    outer=ComponentSpec('outer',KEY,{'cone':'cones','h1':'hcs'},PROFILE,{'explicit_opt_in':True,'A':sparse.csr_matrix(A),'B':sparse.csr_matrix(B),'background_td':q(np.full(9,100.),'td','native_retinal_illuminance'),'pooling_policy':'pool_Vbar_before_aI','geometry_frame':'OPL','observation_interval_seconds':.0001,'unsupported_type_policy':'software_fixture_shared_H1_kinetics'})
    builder=RetinaBuilder(registry)
    native=builder.registry.instantiate(outer,{'cones':cones,'hcs':hcs},registry,RunMode.NUMERIC_FIXTURE)
    r0=native.observe('release',0,tuple(x.values for x in native.initial()),{}).values
    populations=[cones,hcs];components=[outer];links=[];by_comp={'outer':cones};region3=('center','transition','periphery');pool={}
    bc_defs=[('midget_ON','ResidualBC_ON',[0,1,3,4,6,7]),('midget_OFF','ResidualBC_OFF',[0,1,3,4,6,7]),('DB_ON','ResidualBC_ON',None),('DB_OFF','ResidualBC_OFF',None),('S_ON','ResidualBC_ON',[2,5,8])]
    for name,mechanism,indices in bc_defs:
        if indices is None:
            W=np.zeros((3,9))
            for j in range(3):W[j,3*j:3*j+2]=.5
            positions=centers;regs=region3;ana=anatomy
        else:
            W=np.eye(9)[indices];positions=xy[indices];regs=tuple(regions[i] for i in indices);ana=mapped['pedicle'].values[indices]
        n=len(W);pid='QA_'+name
        fixture_profile(registry,pid,n,{'kernel':([.5,.25,-.1],'dimensionless','causal_residual_kernel'),'sample_interval':(.001,'s','history_sampling_interval'),'bias':(np.zeros(n),'dimensionless','activation_bias'),'baseline':(W@r0,'dimensionless','input_baseline'),'initial_history':(np.zeros((3,n)),'dimensionless','causal_input_history'),'initial_inhibition':(np.zeros(n),'dimensionless','initial_inhibitory_gate')})
        pop=population(name,'BC',(name,)*n,positions,regs,pid,'Macaque_software_fixture',ana);populations.append(pop);by_comp[name]=pop;pool[name]=W
        components.append(ComponentSpec(name,mechanism,{'cells':name},pid,{'polarity_representation':'canonical_dominant_positive','presynaptic_inhibition':bool(presynaptic_loop and name=='DB_ON')}))
        links.append(link('outer','release',name,'drive',W,by_comp,pid,'OPL'))
    ac_defs=[('local_feedforward_ON','DB_ON'),('local_feedforward_OFF','DB_OFF'),('local_crossover_ON','DB_OFF'),('local_crossover_OFF','DB_ON'),('widefield_ON','DB_ON'),('widefield_OFF','DB_OFF')]
    for name,bc in ac_defs:
        n=3;pid='QA_'+name
        fixture_profile(registry,pid,n,{'kernel':([.4,.2,-.05],'dimensionless','causal_residual_kernel'),'sample_interval':(.001,'s','history_sampling_interval'),'bias':(np.full(n,-.3),'dimensionless','activation_bias'),'baseline':(np.full(n,np.log(2)),'dimensionless','input_baseline'),'initial_history':(np.zeros((3,n)),'dimensionless','causal_input_history')})
        pop=population(name,'AC',(name,)*3,centers,region3,pid,'Macaque_software_fixture',anatomy);populations.append(pop);by_comp[name]=pop
        components.append(ComponentSpec(name,'ResidualAC',{'cells':name},pid,{'polarity_representation':'stored_signed'}))
        W=np.full((3,3),1/3) if name.startswith('widefield') else np.eye(3)
        links.append(link(bc,'activation',name,'drive',W,by_comp,pid,'IPL'))
    if presynaptic_loop:links.append(link('local_feedforward_ON','activation','DB_ON','inhibition',np.eye(3)*.1,by_comp,'QA_DB_ON','IPL'))
    rgc_routes={'midget_ON':('midget_ON','local_crossover_ON'),'midget_OFF':('midget_OFF','local_crossover_OFF'),'parasol_ON':('DB_ON','widefield_ON'),'parasol_OFF':('DB_OFF','widefield_OFF'),'small_bistratified_S_ON':('S_ON','local_feedforward_OFF')}
    for ordinal,(name,(bc,ac)) in enumerate(rgc_routes.items()):
        cid='RGC_'+name;pid='QA_'+cid;n=3
        fixture_profile(registry,pid,n,{'C':(np.full(n,60e-12),'F','membrane_capacitance'),'gL':(np.full(n,2e-9),'S','leak_conductance'),'EL':(np.full(n,-.065),'V','leak_reversal'),'EE':(np.zeros(n),'V','excitatory_reversal'),'EI':(np.full(n,-.080),'V','inhibitory_reversal'),'V0':(np.full(n,-.065),'V','initial_membrane_voltage'),'alpha':(np.full(n,150.),'Hz','spike_intensity_scale'),'mu':(np.full(n,-.070),'V','spike_voltage_offset'),'beta':(np.full(n,.005),'V','spike_voltage_scale'),'spike_history_kernel':([-.012,-.008,-.004],'V','strict_past_spike_history_kernel'),'initial_spike_history':(np.zeros((3,n)),'dimensionless','spike_count_history'),'spike_interval':(.001,'s','spike_sampling_interval')})
        pop=population(cid,'RGC',(name,)*3,centers,region3,pid,'Macaque_software_fixture',mapped['soma'].values[1::3]);populations.append(pop);by_comp[cid]=pop
        components.append(ComponentSpec(cid,'HistoryConductanceRGC',{'cells':cid},pid,{'rng_seed':10000+ordinal}))
        for source,receptor in ((bc,'excitatory'),(ac,'inhibitory')):
            gain_id=cid+'_'+receptor;gain_pid='QA_'+gain_id
            fixture_profile(registry,gain_pid,n,{'gain':(np.full(n,1e-9 if receptor=='excitatory' else .4e-9),'S','synaptic_conductance_scale')})
            by_comp[gain_id]=pop
            components.append(ComponentSpec(gain_id,'SynapticConductance',{'cells':cid},gain_pid,{'receptor':receptor,'target_population_profile_id':pid}))
            W=np.zeros((3,len(by_comp[source].cells.cell_id)))
            if source.startswith('midget'):
                for j in range(3):W[j,2*j:2*j+2]=.5
            else:W=np.eye(3)
            links.append(link(source,'activation',gain_id,'activation',W,by_comp,gain_pid,'IPL'))
            links.append(link(gain_id,'conductance',cid,'excitation' if receptor=='excitatory' else 'inhibition',np.eye(3),by_comp,gain_pid,'IPL'))
        if name=='small_bistratified_S_ON':
            # Keep an explicit additional LM-OFF branch, not a colour subtraction vector.
            source='DB_OFF';gain_id=cid+'_LM_OFF';gain_pid='QA_'+gain_id
            fixture_profile(registry,gain_pid,3,{'gain':(np.full(3,.5e-9),'S','synaptic_conductance_scale')})
            by_comp[gain_id]=pop
            components.append(ComponentSpec(gain_id,'SynapticConductance',{'cells':cid},gain_pid,{'receptor':'excitatory','target_population_profile_id':pid}))
            links.extend([link(source,'activation',gain_id,'activation',np.eye(3),by_comp,gain_pid,'IPL'),link(gain_id,'conductance',cid,'excitation',np.eye(3),by_comp,gain_pid,'IPL')])
    closure={'method':'hybr','atol':1e-10,'rtol':1e-10,'max_evaluations':1000} if presynaptic_loop else None
    spec=ModelSpec('explicit_full_layer_software_fixture',tuple(populations),tuple(components),tuple(links),(ExternalInput('native_light','outer','light'),),scope='software_fixture',algebraic_closure=closure)
    compiled=builder.build(spec,RunMode.NUMERIC_FIXTURE)
    gaze=GazeTrajectory(q([0,.005,.02],'s','absolute_time'),q([[0,0],[.01,-.01],[.02,0]],'deg','prescribed_gaze_position','world_visual_field'),'linear','synthetic_given_gaze')
    def mapper(t,g,p):
        theta=p.cells.rf_position.values[:,0]/.223+g.values[0]
        L=100+20*np.sin(theta*5+200*t)
        return q(L,'td','native_retinal_illuminance')
    channel=gaze_driven_channel(cones,gaze,mapper,start=0,stop=.02,unit='td',meaning='native_retinal_illuminance',interpolation='analytic',evidence_tag='synthetic_fixture',mapper_source_id='synthetic_small_angle_native_scene')
    return compiled,{'native_light':channel},{'builder_estimate':builder.estimate(spec),'displacement':mapped['diagnostics'],'displacement_operator':displacement,'spec':spec,'registry':registry,'claim':'Software fixture; inner numeric coefficients/resolution not macaque fitted physiology'}
