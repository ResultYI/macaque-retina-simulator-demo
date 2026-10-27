"""Architecture tests with explicit synthetic inputs; no animal fit/validation."""
from pathlib import Path
from dataclasses import replace
import json,hashlib
import numpy as np
import pytest
from scipy import sparse
from retina_sim.contracts import Quantity,ContractError,RunMode,CellTable
from retina_sim.assembly import RetinaBuilder,ModelSpec,ComponentSpec,ExternalInput,AssemblyError
from retina_sim.core_sources import load_core_registry
from retina_sim.geometry import PackerDensityField,RetinalDomain,integrate_density,capacity_constrained_mosaic,assign_cone_types,partition_mosaic
from retina_sim.density import read_density_tabulation
from retina_sim.recording import RecordingPlan,NPZWindowRecorder,TimestampedDelay
from retina_sim.hybrid import NetworkSolverConfig
from retina_sim.physical import PassiveCable,terminal_activation,monotone_release_proxy,passive_coefficients
from retina_sim.mechanism_contracts import explicit_unit_conversion
from retina_sim.wiring import SparseRule,sparse_gaussian_contacts,private_line_contacts,wool_midget_profile,smooth_parameter_field
from retina_sim.spatial import InputChannel,Population
from retina_sim.vh2005 import SOURCE_VALUES,PROFILE,KEY
from examples.full_network_fixture import full_fixture,population,q,fixture_profile,link
from conftest import NUMERIC_METRICS

ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope='module')
def fixture():return full_fixture(ROOT)
@pytest.fixture(scope='module')
def loop_fixture(fixture):return full_fixture(ROOT,True,fixture[2]['displacement_operator'])
def flat(state):return np.concatenate(state.ordered_state_arrays)
def outer_only(fixture):
    c,ch,a=fixture;s=a['spec']
    spec=replace(s,populations=s.populations[:2],components=s.components[:1],links=())
    return RetinaBuilder(a['registry']).build(spec,RunMode.NUMERIC_FIXTURE),ch

def test_source_native_full_profile_equilibrium_and_scale(fixture):
    c,ch,a=fixture;m=c.components['outer'].native_model
    assert dict(m.p)==SOURCE_VALUES and len(SOURCE_VALUES)==22
    b=tuple(x.values for x in m.initial_conditions())
    d=m.derivative(0,b,{'light':q(np.full(9,100),'td','native_retinal_illuminance')})
    residual=max(float(np.max(np.abs(x.values))) for x in d);assert residual<1e-6
    NUMERIC_METRICS['native_equilibrium_derivative_max_source_units_per_s']=residual
    I,aI,sig,r=m.signals(b);np.testing.assert_allclose(r,sig/aI,rtol=0,atol=0)
    assert not np.allclose(r,sig) and np.all(r>0)
    b2=tuple(x.values for x in m.equilibrium(q(np.full(9,200),'td','native_retinal_illuminance')))
    assert not np.allclose(b[4],b2[4])
    np.testing.assert_allclose(b[4],np.full(9,b[4][0]),atol=1e-10)
    with pytest.raises(TypeError):m.p['tau_R']=1
    with pytest.raises(ValueError):m.A.data[0]=0
    hetero=q(np.linspace(80,200,9),'td','native_retinal_illuminance');b3=tuple(x.values for x in m.equilibrium(hetero))
    assert max(float(np.max(np.abs(x.values))) for x in m.derivative(0,b3,{'light':hetero}))<1e-6

def test_strict_S_H2_binding_rejects_software_extension_default(fixture):
    c,ch,a=fixture;s=a['spec'].components[0];options=dict(s.options);options.pop('unsupported_type_policy')
    with pytest.raises(ContractError,match='S/H2'):
        RetinaBuilder(a['registry']).registry.instantiate(replace(s,options=options),c.populations,a['registry'],RunMode.NUMERIC_FIXTURE)
    assert c.components['outer'].native_model.approximations[0]['affected_cone_ids']

def test_full_layer_run_measures_actual_components_and_coordinates(fixture):
    c,ch,a=fixture;initial=c.initialize();state,result=c.advance(initial,ch,.003,sample_times=[0,.001,.002,.003])
    assert len(c.components)==28 and sum(len(p.cells.cell_id) for p in c.populations.values())==69
    assert set(c.populations['cones'].regions)=={'center','transition','periphery'}
    assert set(c.populations['cones'].cells.subtype)=={'L','M','S'}
    assert set(c.populations['hcs'].cells.subtype)=={'H1','H2'}
    assert initial.sample_index==0 and state.sample_index==4
    assert len(result.metadata['executed_components'])==28
    NUMERIC_METRICS['executed_fixture_components']=len(result.metadata['executed_components']);NUMERIC_METRICS['fixture_cell_count']=69
    NUMERIC_METRICS['fixture_population_count']=len(c.populations);NUMERIC_METRICS['fixture_claim']='synthetic software integration; not fitted physiology'
    NUMERIC_METRICS['fixture_spike_events']=sum(len(e) for e in result.metadata['spike_events'].values())
    assert result.metadata['physiological_validation']=='NOT_RUN'
    for cid in state.spike_states:
        assert np.all(result.outputs_with_units[cid+'.gE'].values>0)
        assert np.all(result.outputs_with_units[cid+'.gI'].values>0)
        assert cid+'.conditional_rate' in result.outputs_with_units
        np.testing.assert_allclose(result.outputs_with_units[cid+'.window_rate'].values,result.outputs_with_units[cid+'.spike_count'].values/.003)
    assert 'RGC_small_bistratified_S_ON_LM_OFF' in c.components
    for p in result.metadata['populations']:
        assert len(p['cell_ids'])==p['total_cell_count']
        kind=p['cell_types'][0]
        if kind=='cone':assert p['pedicle_positions']['frame']=='OPL' and p['terminal_positions'] is None
        if kind=='BC':assert p['dendritic_positions']['frame']=='OPL' and p['terminal_positions']['frame']=='IPL'
        if kind=='HC':assert p['dendritic_positions']['frame']=='OPL' and p['pedicle_positions'] is None
        if kind=='RGC':assert p['dendritic_positions']['frame']=='IPL' and p['terminal_positions'] is None
    np.testing.assert_array_equal(c.populations['cones'].anatomical_positions.values,c.populations['cones'].receptive_field_positions.values)
    rgc=c.populations['RGC_midget_ON'];assert np.max(np.abs(rgc.anatomical_positions.values-rgc.receptive_field_positions.values))>0.1
    assert np.max(np.abs(c.populations['cones'].pedicle_positions.values-c.populations['cones'].receptive_field_positions.values))>0.1
    np.testing.assert_allclose(result.outputs_with_units['time'].values,[0,.001,.002,.003],atol=1e-12)

def test_non_grid_checkpoint_continuation_preserves_history_rng_and_gaze(fixture):
    c,ch,a=fixture;initial=c.initialize();whole,wr=c.advance(initial,ch,.0073)
    first,fr=c.advance(initial,ch,.0053);resumed=c.restore(c.checkpoint(first));split,sr=c.advance(resumed,ch,.0073)
    np.testing.assert_allclose(flat(whole),flat(split),rtol=2e-8,atol=2e-9)
    NUMERIC_METRICS['checkpoint_continuation_max_absolute_state_difference']=float(np.max(np.abs(flat(whole)-flat(split))))
    assert whole.sample_index==split.sample_index==8
    for cid in whole.spike_states:
        wa=whole.spike_states[cid];sa=split.spike_states[cid]
        assert wa['rng_state']==sa['rng_state'] and wa['last_bin_counts']==sa['last_bin_counts'] and wa['history']==sa['history']
        we=wr.metadata['spike_events'][cid];se=fr.metadata['spike_events'][cid]+sr.metadata['spike_events'][cid]
        assert [e['cell_id'] for e in we]==[e['cell_id'] for e in se]
        np.testing.assert_allclose([e['time_seconds'] for e in we],[e['time_seconds'] for e in se],atol=2e-10,rtol=0)
        assert all(e['time_seconds']>.0073 for e in wa['pending_events'])
    assert whole.assembly_hash==initial.assembly_hash==split.assembly_hash
    assert not np.array_equal(ch['native_light'].at(0).values,ch['native_light'].at(.0073).values)

def test_backend_options_and_seed_change_checkpoint_identity(fixture):
    c,ch,a=fixture;spec=a['spec'];changed=[]
    for component in spec.components:
        changed.append(replace(component,options=dict(component.options,rng_seed=9876)) if component.component_id=='RGC_midget_ON' else component)
    other=RetinaBuilder(a['registry']).build(replace(spec,components=tuple(changed)),RunMode.NUMERIC_FIXTURE)
    assert other.assembly_hash!=c.assembly_hash
    with pytest.raises(ContractError,match='identity'):other.restore(c.checkpoint(c.initialize()))

def test_software_coefficients_cannot_be_retagged_as_measurement_in_reference_mode(fixture):
    from types import SimpleNamespace
    from retina_sim.inner import _profile
    c,ch,a=fixture;component=next(x for x in a['spec'].components if x.component_id=='midget_ON');pop=c.populations['midget_ON']
    fake_cells=SimpleNamespace(**{name:getattr(pop.cells,name) for name in ('cell_type','parameter_profile_id','species_profile_id')},evidence_tags=('primary_measurement',)*len(pop.cells.cell_id))
    with pytest.raises(ContractError,match='cannot be relabeled'):_profile(component,{'midget_ON':SimpleNamespace(cells=fake_cells)},a['registry'],RunMode.REFERENCE)

def test_joint_zero_delay_closure_no_frame_delay(loop_fixture):
    c,ch,a=loop_fixture
    assert any(block.cyclic for block in c.output_order)
    state,result=c.advance(c.initialize(),ch,.0013)
    assert state.time_seconds==.0013 and np.all(np.isfinite(flat(state)))
    with pytest.raises(AssemblyError,match='algebraic cycle'):
        RetinaBuilder(a['registry']).build(replace(a['spec'],algebraic_closure=None),RunMode.NUMERIC_FIXTURE)

def test_ON_OFF_polarity_inverts_once_and_uses_past_history(fixture):
    c,ch,a=fixture
    for cid,direction in [('midget_ON',1),('midget_OFF',-1)]:
        m=c.components[cid].native_model;blocks=tuple(x.values for x in m.initial());baseline=m.baseline
        result=m.observe('activation',0,blocks,{'drive':q(baseline*.9,'dimensionless','effective_synaptic_drive'),'__sampling__':True})
        assert np.all(direction*(result.values-blocks[-1])>0)
        nextblocks=m.transition(blocks,{'drive':q(baseline*.9,'dimensionless','effective_synaptic_drive')},{'activation':result})
        np.testing.assert_allclose(nextblocks[0],-.1*baseline,atol=1e-14)
        np.testing.assert_array_equal(nextblocks[1],blocks[0])

def test_RGC_strict_initial_history_no_extra_reset_and_event_window(fixture):
    c,ch,a=fixture;m=c.components['RGC_midget_ON'].native_model
    history=np.ones_like(m.initial_spike_history);rng=np.random.Generator(np.random.PCG64(3))
    aux={'history':history.tolist(),'last_bin_counts':None,'rng_state':rng.bit_generator.state,'pending_events':[]}
    new,events=m.sample_spikes(0,m.V0,aux)
    np.testing.assert_array_equal(new['history'],history)
    np.testing.assert_allclose(new['conditional_rate_Hz'],m.intensity(m.V0,history))
    assert all(0<=e['time_seconds']<m.dt for e in events)
    assert len({e['cell_index'] for e in events})==len(events)

def test_unit_semantic_frame_and_resource_guards(fixture):
    c,ch,a=fixture;builder=RetinaBuilder(a['registry']);s=a['spec']
    with pytest.raises(AssemblyError,match='Resource estimate'):RetinaBuilder(a['registry'],max_compile_bytes=1).build(s,RunMode.NUMERIC_FIXTURE)
    bad=replace(s.links[0],geometry_frame='not_a_frame')
    with pytest.raises(AssemblyError,match='frame'):builder.build(replace(s,links=(bad,)+s.links[1:]),RunMode.NUMERIC_FIXTURE)
    bad=replace(s.links[0],source_port='Ios')
    with pytest.raises(AssemblyError,match='Incompatible ports'):builder.build(replace(s,links=(bad,)+s.links[1:]),RunMode.NUMERIC_FIXTURE)
    with pytest.raises(ContractError):explicit_unit_conversion(q([1],'vH_native_current','native_outer_segment_current'),'pA')
    np.testing.assert_allclose(explicit_unit_conversion(q([1],'pA','absolute_outer_segment_current'),'A').values,[1e-12])
    with pytest.raises(ContractError,match='Recording requires'):c.advance(c.initialize(),ch,.002,recording_plan=RecordingPlan(max_bytes=1))

def test_bounded_selected_recording_and_NPZ_stream(fixture,tmp_path):
    c,ch,a=fixture;chosen=('cones:0','RGC_midget_ON:0');plan=RecordingPlan(ports=('outer.release','RGC_midget_ON.voltage'),states=(),cell_ids=chosen,max_bytes=100000)
    state,result=c.advance(c.initialize(),ch,.0013,recording_plan=plan)
    assert set(result.outputs_with_units)=={'time','outer.release','RGC_midget_ON.voltage'}
    assert result.outputs_with_units['outer.release'].values.shape==(1,2)
    assert all(len(p['cell_ids'])<=1 for p in result.metadata['populations'])
    recorder=NPZWindowRecorder(tmp_path);end=c.stream(state,ch,.0023,.0005,recorder,recording_plan=plan)
    assert recorder.windows==2 and end.time_seconds==.0023
    data=np.load(tmp_path/'window_00000000.npz',allow_pickle=False)
    metadata=json.loads(str(data['manifest_json']))
    assert metadata['metadata']['physiological_validation']=='NOT_RUN'

def test_fixed_RK2_convergence_with_history_grid_unchanged(fixture):
    c,ch=outer_only(fixture);stop=.0037;outputs=[]
    for dt in (1e-4,5e-5,2.5e-5):
        cfg=NetworkSolverConfig('RK2',fixed_step_seconds=dt)
        s,_=c.advance(c.initialize(cfg),ch,stop);outputs.append(flat(s))
    coarse=float(np.linalg.norm(outputs[0]-outputs[1]));fine=float(np.linalg.norm(outputs[1]-outputs[2]))
    assert coarse>0 and fine<coarse*.5
    NUMERIC_METRICS['RK2_refinement_difference_ratio']=fine/coarse
    assert dict(c.components['outer'].native_model.p)==SOURCE_VALUES

def test_delay_linear_timestamp_and_bounded_prehistory():
    base=q([0],'V','physical_membrane_voltage');d=TimestampedDelay(.002,[0],[[0]],base,'linear')
    for i in range(1,11):d.append(i*.001,q([i],'V','physical_membrane_voltage'))
    assert len(d.history_times)<=4
    np.testing.assert_allclose(d.at(.01).values,[8])
    clone=TimestampedDelay.from_payload(d.payload());np.testing.assert_array_equal(d.at(.01).values,clone.at(.01).values)

def test_source_density_mosaic_capacity_and_exact_label_quota():
    raw=ROOT/'data/published_code/packer_cone_density.csv';field=PackerDensityField(read_density_tabulation(raw,hashlib.sha256(raw.read_bytes()).hexdigest()),central_policy='central_constant_cap',angular_policy='log_NT_cosine')
    domain=RetinalDomain((.99,1.01,-.01,.01),(.992,1.008,-.008,.008),lambda xy:np.ones(len(xy),bool),'synthetic_roi_mask','source_density_QA_ROI')
    mosaic=capacity_constrained_mosaic(field,domain,7,quadrature_shape=(64,64),iterations=2)
    audit=integrate_density(field,domain)
    assert len(mosaic.cell_ids)==audit['rounding_target'] and mosaic.diagnostics['maximum_capacity_relative_residual']<1e-8
    assert mosaic.diagnostics['minimum_nn_mm']>0
    NUMERIC_METRICS['source_density_mosaic_diagnostics']=mosaic.diagnostics
    left=RetinalDomain(domain.bounds_mm,(.99,1.,-.01,.01),domain.mask,domain.mask_source_id,'left_partition')
    right=RetinalDomain(domain.bounds_mm,(1.,1.01,-.01,.01),domain.mask,domain.mask_source_id,'right_partition')
    lm=partition_mosaic(mosaic,left);rm=partition_mosaic(mosaic,right)
    assert lm.cell_ids==rm.cell_ids==mosaic.cell_ids
    assert not np.any(lm.owner&rm.owner) and np.all(lm.owner|rm.owner)
    sd=lambda xy:field(xy)*.1
    labels,a=assign_cone_types(mosaic,field,sd,domain,1,'synthetic_LM_ratio_QA',7,.01)
    assert np.sum(labels=='S')==int(np.floor(audit['mass']*.1+.5))==a['Ns']
    assert a['NL']+a['NM']+a['Ns']==len(labels)
    noS,a0=assign_cone_types(mosaic,field,lambda xy:np.zeros(len(xy)),domain,1,'synthetic_no_S_assay_QA',7,.01)
    assert a0['Ns']==0 and len(noS)==len(mosaic.cell_ids) and set(noS)<={'L','M'}
    with pytest.raises(ContractError,match='exceeds total'):assign_cone_types(mosaic,field,lambda xy:field(xy)*1.1,domain,1,'synthetic_LM_ratio',7,.01)

def test_displacement_source_anchors_positive_J_and_catchup(fixture):
    d=fixture[2]['displacement_operator'];expected={'nasal':2.2633931995,'superior':2.0700821473,'inferior':2.0680198316,'temporal':1.8971211925}
    for name,theta in d.cardinal.items():
        nodes=d.nodes['production_nodes'][name]
        np.testing.assert_allclose(d.radial(np.array(nodes['r_mm']),np.full(5,theta)),nodes['soma_radius_mm'],atol=1e-12)
        assert abs(d.diagnostics[name]['catchup_mm']-expected[name])<1e-6
        assert d.diagnostics[name]['minimum_dR_dr']>0
    out=d.apply(q([[.01,0],[1,0],[7,0]],'mm','functional_position','functional_visual_field'))
    assert out['input_cell_count']==out['output_cell_count']==3 and np.all(out['jacobian']>0)
    np.testing.assert_allclose(out['soma'].values[-1],[7,0])

def test_sparse_contacts_frames_nonnegative_and_discrete_tail_bound():
    pre=q([[0,0],[.01,0],[.02,0]],'mm','pedicle_position','OPL');post=q([[0,0]],'mm','dendritic_position','OPL')
    rule=SparseRule('QA','OPL',q(.01,'mm','pooling_radius'),1e-3,'unit_pooling','synthetic_source','excitatory','synthetic_fixture',0)
    table,a=sparse_gaussian_contacts(('a','b','c'),pre,('d',),post,rule,effective_target_count=1.5,halo_complete=True)
    assert all(x>=0 for x in table.weight) and sum(table.weight)==pytest.approx(1)
    assert a['rows'][0]['degree_selection_omitted_weight']>0
    with pytest.raises(ContractError,match='halo'):sparse_gaussian_contacts(('a','b','c'),pre,('d',),post,rule)
    private=private_line_contacts(('a','b'),('BCa','BCb'),'QA','synthetic_fixture',receptor='explicit',delay_seconds=0)
    assert private.pre_cell_id==('a','b') and private.weight==(1.,1.)
    assert wool_midget_profile([.1,1])['private_line_branch'].tolist()==[True,False]
    values,_=smooth_parameter_field([0,1,2],1.,100.,0,2,positive=True,source_context_compatible=True,approximation_id='QA_field')
    np.testing.assert_allclose(values,[1,10,100])

def test_passive_sign_absolute_anchor_gate_and_taper_units():
    cable=PassiveCable(q([1e-12,1e-12],'F','membrane_capacitance'),q([1e-9,1e-9],'S','leak_conductance'),q([[0,2e-9],[2e-9,0]],'S','axial_conductance'),q([[1],[0]],'dimensionless','current_injection_distribution'),q([0],'A','absolute_outer_segment_current'),q([[0],[0]],'V','physical_voltage_deviation'))
    delta=q([[0],[0]],'V','physical_voltage_deviation');current=q([1e-12],'A','absolute_outer_segment_current')
    assert cable.rhs(delta,current).values[0,0]<0
    assert np.all(cable.semi_implicit_step(delta,current,.001).values<0)
    with pytest.raises(ContractError,match='anchor'):cable.voltage(delta)
    half=q([-.035],'V','activation_half_voltage');slope=q([.004],'V','activation_voltage_slope');shift=q([0],'V','activation_threshold_shift')
    a=terminal_activation(q([-.040],'V','physical_membrane_voltage'),half,slope,shift);b=terminal_activation(q([-.030],'V','physical_membrane_voltage'),half,slope,shift)
    assert b.values[0]>a.values[0]
    np.testing.assert_array_equal(monotone_release_proxy(a,profile_name='quasi_steady_normalized_activation_proxy').values,a.values)
    coefficients=passive_coefficients(q([1e-5,1e-5],'m','compartment_length'),q([1e-10,1e-10],'m^2','membrane_area'),q([1e-12,1e-12],'m^2','cross_section_area'),q([1,1],'ohm m','axial_resistivity'),q([.01,.01],'F/m^2','specific_membrane_capacitance'),q([1,1],'S/m^2','specific_membrane_conductance'),[(0,1)],taper_endpoint_radii=q([[1e-6,2e-6],[2e-6,1e-6]],'m','taper_endpoint_radii'))
    assert coefficients['axial'].values[0,1]>0 and np.array_equal(coefficients['axial'].values,coefficients['axial'].values.T)

def test_full_target_missing_bindings_are_not_completion(fixture):
    target=ModelSpec.read_target(ROOT/'configurations/full_target.json');builder=RetinaBuilder(fixture[2]['registry']);description=builder.describe(target)
    assert description['unbound'] and description['executed_component_ids']==[] and description['physiological_completion'] is False
    NUMERIC_METRICS['full_target_unbound']=description['unbound'];NUMERIC_METRICS['registered_operators']=description['registered_operator_names']
    with pytest.raises(AssemblyError):builder.build(target,RunMode.NUMERIC_FIXTURE)

def test_constant_background_warmup_explicit_scales_and_timeout(fixture):
    c,ch=outer_only(fixture);m=c.components['outer'].native_model
    channel=InputChannel(0,.01,lambda t:q(np.full(9,100),'td','native_retinal_illuminance'),'td','native_retinal_illuminance','analytic','synthetic_fixture',{'declared_constant_background':True,'kind':'synthetic_constant_background'})
    state,check=c.runner().pre_adapt(c.initialize(),{'native_light':channel},window_seconds=.001,max_duration_seconds=.002,scales={'outer.Vis':q(np.full(9,1),'vH_native_voltage','native_relative_voltage')},absolute_tolerance=1e-7,relative_tolerance=1e-7)
    assert check['converged'] and check['windows']==1 and state.time_seconds==.001

def test_stateless_physical_conversion_can_execute_without_dummy_state():
    registry=load_core_registry(ROOT);pid='QA_stateless';fixture_profile(registry,pid,1,{})
    pop=population('unit','cone',('L',),np.array([[1,0]]),('transition',),pid,'Macaque_software_fixture')
    component=ComponentSpec('convert','ExplicitPhysicalUnitConversion',{'cells':'unit'},pid,{'source_unit':'pA','target_unit':'A','meaning':'absolute_outer_segment_current'})
    spec=ModelSpec('stateless_QA',(pop,),(component,),external_inputs=(ExternalInput('current','convert','input'),))
    c=RetinaBuilder(registry).build(spec,RunMode.NUMERIC_FIXTURE)
    channel=InputChannel(0,.01,lambda t:q([2],'pA','absolute_outer_segment_current'),'pA','absolute_outer_segment_current','analytic','synthetic_fixture',{'kind':'synthetic_current'})
    state,result=c.advance(c.initialize(),{'current':channel},.002)
    assert state.ordered_state_arrays==()
    np.testing.assert_allclose(result.outputs_with_units['convert.output'].values,[[2e-12,2e-12]])

def reference_population(pid,role,xy,subtypes,profile,species,ecc=5):
    ids=tuple(f'{pid}:{i}' for i in range(len(xy)));n=len(ids)
    cells=CellTable(ids,(role,)*n,tuple(subtypes),(species,)*n,('specified_QA',)*n,('specified_QA',)*n,None,q(xy,'deg','functional_position','local_patch'),'deg','local_patch',q(np.full(n,ecc),'deg','global_eccentricity','functional_visual_field'),(profile,)*n,('synthetic_fixture',)*n)
    return Population(pid,cells,('transition' if ecc<20 else 'periphery',)*n)

def test_Somaratna_composite_and_six_layers_agree_in_joint_RHS():
    from retina_sim.components import SF_STATES,SF_KEY
    registry=load_core_registry(ROOT);pid='SomaratnaFreeman2025GaussianHCReference_source_specific';params=registry.load_parameters(pid,RunMode.NUMERIC_FIXTURE)
    cone_xy=np.array([[-.01,0],[0,0],[.008,.005],[.02,-.01]]);rgc_xy=np.array([[0,0],[.01,.008]])
    pops=tuple(reference_population(role,('cone' if role=='cone' else 'HC' if role=='horizontal' else 'BC' if role.startswith('b_') else 'RGC'),cone_xy if i<4 else rgc_xy,('L','M','S','L') if i<4 else ('source_reference',)*2,pid,params.profile.species) for i,role in enumerate(SF_STATES))
    composite=ComponentSpec('source',SF_KEY,{r:r for r in SF_STATES},pid,{'explicit_opt_in':True})
    builder=RetinaBuilder(registry);cspec=ModelSpec('SF_composite',(pops),(composite,),external_inputs=(ExternalInput('contrast','source','contrast'),))
    full=builder.build(cspec,RunMode.NUMERIC_FIXTURE);g=full.components['source'].native_model.graph
    components=tuple(ComponentSpec(r,'SomaratnaFreeman2025.'+r,{'cells':r},pid,{'explicit_opt_in':True}) for r in SF_STATES);by={p.population_id:p for p in pops}
    routes=[('cone','horizontal','drive',g.w_hc),('horizontal','cone','horizontal',g.w_ch),('cone','b_off','drive',np.eye(4)),('cone','b_on','drive',np.eye(4)),('b_off','g_off','drive',g.w_off),('b_on','g_on','drive',g.w_on)]
    links=tuple(link(pre,'generator',post,port,W,by,pid,'local_patch') for pre,post,port,W in routes)
    split=builder.build(ModelSpec('SF_six_layers',pops,components,links,(ExternalInput('contrast','cone','contrast'),)),RunMode.NUMERIC_FIXTURE)
    channel=InputChannel(0,.01,lambda t:q(np.full(4,.1),'dimensionless','cone_contrast'),'dimensionless','cone_contrast','analytic','synthetic_fixture',{'kind':'synthetic_SF_reference_contrast'})
    sa,ra=full.advance(full.initialize(),{'contrast':channel},.003)
    sb,rb=split.advance(split.initialize(),{'contrast':channel},.003)
    np.testing.assert_allclose(flat(sa),flat(sb),atol=2e-10,rtol=2e-10)
    assert len(rb.metadata['executed_components'])==6
    NUMERIC_METRICS['SF_source_adapter_max_difference_mV']=float(np.max(np.abs(flat(sa)-flat(sb))))

def test_Chen_isolated_author_adapter_remains_on_native_grid():
    from retina_sim.components import CHEN_KEY
    from retina_sim.chen2024 import rate_segment
    registry=load_core_registry(ROOT);pid='Chen2024PeripheralPrimateCone_source_mixed_macaca';p=registry.load_parameters(pid,RunMode.NUMERIC_FIXTURE)
    pop=reference_population('Chen_QA','cone',np.array([[0,0]]),('L',),pid,p.profile.species,ecc=25)
    component=ComponentSpec('Chen',CHEN_KEY,{'cone':'Chen_QA'},pid,{'explicit_opt_in':True})
    spec=ModelSpec('Chen_isolated_author_QA',(pop,),(component,),external_inputs=(ExternalInput('light','Chen','light'),))
    c=RetinaBuilder(registry).build(spec,RunMode.NUMERIC_FIXTURE)
    channel=rate_segment(q([[100],[110],[120],[100]],'R*/s','photoisomerization_rate'),0)
    state,result=c.advance(c.initialize(),{'light':channel},.0004)
    assert result.outputs_with_units['Chen.current'].unit=='pA' and state.time_seconds==.0004
    assert result.metadata['executed_components'][0]['time_contract']['kind']=='author_discrete'
    with pytest.raises(AssemblyError,match='isolated'):
        extra=replace(component,component_id='Chen2');RetinaBuilder(registry).build(replace(spec,components=(component,extra)),RunMode.NUMERIC_FIXTURE)

def test_verified_Maddox_records_keep_conditions_and_missing_calibration():
    table=json.loads((ROOT/'sources/maddox2024_activation_records.json').read_text())
    assert len(table['records'])==6 and table['animals']==1 and table['species']=='M. mulatta'
    assert [(r['Vhalf_mV'],r['k_mV']) for r in table['records']]==[(-33.65,3.736),(-35.19,4.12),(-42.98,3.858),(-34.4,3.226),(-34.92,3.727),(-43.96,2.728)]
    assert {r['holding_mV'] for r in table['records']}=={-90,-50}
    assert 'gCa_max' in table['not_supplied'] and table['regional_discrepancy']
    raw=(ROOT/'data/source/maddox2024/elife-94908-fig5-data1-v1.xlsx').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==table['source_asset_sha256']
