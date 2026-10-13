from dataclasses import replace
import numpy as np
import pytest
from conftest import q,NUMERIC_METRICS
from retina_sim.contracts import ContractError,UnsupportedReferenceBranch,SolverConfig,StimulusSegment
from retina_sim.reference import SomaratnaFreeman2025GaussianHCReference,build_somaratna_weights,COMMIT
from retina_sim.engine import initialize_state,advance

def constant_segment(u,start,stop):
    return StimulusSegment(start,stop,lambda t:q(u,'dimensionless','cone_contrast'),'dimensionless','cone_contrast','analytic','synthetic_fixture',{'kind':'constant_contrast','evidence_tag':'synthetic_fixture'})

def run(model,u,stop=.6,cfg=None,times=None):
    cfg=cfg or SolverConfig()
    state=initialize_state(model,model.parameters,model.graph,model.initial_conditions(),cfg)
    return advance(model,state,constant_segment(np.full(4,u),0,stop),stop,cfg,times)

def test_zero_input_rest(model):
    state,result=run(model,0)
    for a in state.ordered_state_arrays[:4]: assert np.max(np.abs(a))<1e-14
    assert np.allclose(state.ordered_state_arrays[4],model.gRest[0],rtol=0,atol=1e-14)
    assert np.allclose(state.ordered_state_arrays[5],model.gRest[1],rtol=0,atol=1e-14)
    assert np.allclose(result.outputs_with_units['rate_off'].values,21.8,rtol=0,atol=1e-13)
    assert np.allclose(result.outputs_with_units['rate_on'].values,25.4,rtol=0,atol=1e-13)

def test_polarity_initial_and_steady_state(model):
    initial=np.concatenate([a.values for a in model.initial_conditions()])
    derivative=model.rhs(0,initial,q(np.full(4,.1),'dimensionless','cone_contrast'),model.parameters,model.graph).values
    assert np.all(derivative[:4]<0)
    state,result=run(model,.1)
    c=-model.kSens*.1/(1+model.kSur)
    expected=np.concatenate([np.full(4,c),np.full(4,c),np.full(4,c),np.full(4,-c),np.full(2,model.gRest[0]+c),np.full(2,model.gRest[1]-c)])
    error=float(np.max(np.abs(np.concatenate(state.ordered_state_arrays)-expected)))
    assert error<1e-10
    assert np.all(state.ordered_state_arrays[2]<0) and np.all(state.ordered_state_arrays[3]>0)
    assert result.outputs_with_units['feedback_output'].meaning=='algebraic_feedback'
    NUMERIC_METRICS['uniform_steady_state_max_error_mV']=error

def test_full_gaussian_weights_and_s_exclusion(model,parameters):
    graph=model.graph
    a=np.exp(-np.sum((graph.cone_locations_deg[:,None]-graph.cone_locations_deg[None,:])**2,axis=2)/graph.radii['rHor']**2)
    expected=a.T/a.T.sum(axis=1,keepdims=True)
    assert np.allclose(graph.w_hc.toarray(),expected,rtol=0,atol=1e-15)
    assert np.any(graph.w_hc.toarray()[:,2]>0)
    assert np.all(graph.w_ch.toarray()[:,2]==0) and np.all(graph.w_on.toarray()[:,2]==0)
    assert np.all(graph.w_off.toarray()[:,2]>0)
    assert not np.allclose(graph.w_hc.toarray(),graph.w_ch.toarray().T)
    assert graph.w_hc.nnz==16 and graph.w_off.nnz==8
    for w in (graph.w_ch,graph.w_hc,graph.w_off,graph.w_on): assert np.max(np.abs(np.asarray(w.sum(axis=1))-1))<5e-15
    with pytest.raises(ContractError,match='Zero'):
        build_somaratna_weights(q(graph.cone_locations_deg,'deg','functional_position','local_patch'),('S',)*4,q(graph.ganglion_locations_deg,'deg','functional_position','local_patch'),q(5,'deg','global_eccentricity','functional_visual_field'),parameters,'synthetic_fixture')
    with pytest.raises(ContractError):
        build_somaratna_weights(q(graph.cone_locations_deg,'deg','anatomical_position','local_patch'),graph.cone_types,q(graph.ganglion_locations_deg,'deg','functional_position','local_patch'),q(5,'deg','global_eccentricity','functional_visual_field'),parameters,'synthetic_fixture')

def test_zoh_block_exponential_oracle_and_convergence(model):
    times=np.linspace(0,.12,121)
    u=np.array([.1,-.03,.04,.08])
    initial=np.concatenate([a.values for a in model.initial_conditions()])
    oracle=np.stack([initial if t==0 else model.zoh_oracle(initial,q(u,'dimensionless','cone_contrast'),t) for t in times],axis=1)
    errors=[]
    for cfg in (SolverConfig(rtol=1e-5,atol=1e-7,max_step=.03),SolverConfig(rtol=1e-10,atol=1e-12,max_step=.002)):
        state=initialize_state(model,model.parameters,model.graph,model.initial_conditions(),cfg)
        _,result=advance(model,state,constant_segment(u,0,.12),.12,cfg,times)
        actual=np.concatenate([result.outputs_with_units[s.state_name].values for s in model.state_specs],axis=0)
        errors.append(float(np.max(np.abs(actual-oracle))))
    assert errors[1]<1e-8 and errors[1]<errors[0]/10
    NUMERIC_METRICS['zoh_coarse_max_error_mV']=errors[0]
    NUMERIC_METRICS['zoh_fine_max_error_mV']=errors[1]
    NUMERIC_METRICS['zoh_fine_solver']={'method':'DOP853','rtol':1e-10,'atol':1e-12,'max_step_seconds':.002,'output_dt_seconds':.001,'dtype':'float64'}

def test_zoh_discontinuity_without_hidden_delay(model):
    cfg=SolverConfig(input_interpolation='zero_order_hold')
    u1=np.full(4,.1); u2=np.full(4,-.05)
    initial=initialize_state(model,model.parameters,model.graph,model.initial_conditions(),cfg)
    stimulus=StimulusSegment(0,.1,lambda t:q(u1 if t<.04 else u2,'dimensionless','cone_contrast'),'dimensionless','cone_contrast','zero_order_hold','synthetic_fixture',{'kind':'piecewise_constant','evidence_tag':'synthetic_fixture'},(.04,))
    state,_=advance(model,initial,stimulus,.1,cfg)
    expected=model.zoh_oracle(model.zoh_oracle(np.concatenate(initial.ordered_state_arrays),q(u1,'dimensionless','cone_contrast'),.04),q(u2,'dimensionless','cone_contrast'),.06)
    error=float(np.max(np.abs(np.concatenate(state.ordered_state_arrays)-expected)))
    assert error<1e-8
    NUMERIC_METRICS['zoh_breakpoint_max_error_mV']=error

def test_sine_frequency_amplitude_phase_and_settling(model):
    freq=7.; omega=2*np.pi*freq; u=np.array([.07,.04,.02,-.03])
    cfg=SolverConfig(max_step=.001,rtol=1e-10,atol=1e-12)
    times=np.linspace(0,1.4,1401)
    initial=initialize_state(model,model.parameters,model.graph,model.initial_conditions(),cfg)
    stimulus=StimulusSegment(0,1.4,lambda t:q(u*np.cos(omega*t),'dimensionless','cone_contrast'),'dimensionless','cone_contrast','analytic','synthetic_fixture',{'kind':'sinusoid','frequency_hz':freq,'absolute_phase':True,'evidence_tag':'synthetic_fixture'})
    _,result=advance(model,initial,stimulus,1.4,cfg,times)
    trajectory=np.concatenate([result.outputs_with_units[s.state_name].values for s in model.state_specs],axis=0)
    trajectory[-4:-2]-=model.gRest[0]; trajectory[-2:]-=model.gRest[1]
    harmonic=model.frequency_oracle(u,freq)
    oracle=np.real(harmonic[:,None]*np.exp(1j*omega*times)[None,:])
    selected=times>=.8
    transient_error=float(np.max(np.abs(trajectory[:,selected]-oracle[:,selected])))
    assert transient_error<1e-8  # declared numerical settling criterion, no fitted parameter
    fit=np.linalg.lstsq(np.column_stack((np.cos(omega*times[selected]),np.sin(omega*times[selected]),np.ones(sum(selected)))),trajectory[:,selected].T,rcond=None)[0]
    measured=fit[0]-1j*fit[1]
    amplitude_error=float(np.max(np.abs(np.abs(measured)-np.abs(harmonic))))
    phase_error=float(np.max(np.abs(np.angle(measured/harmonic))))
    assert amplitude_error<1e-8 and phase_error<1e-7
    NUMERIC_METRICS.update(sine_settling_max_error_mV=transient_error,sine_amplitude_max_error_mV=amplitude_error,sine_phase_max_error_rad=phase_error,sine_settling_criterion={'discard_before_seconds':.8,'criterion_max_residual_mV':1e-8,'fit_parameters_changed':False})

def test_rectification_limits_and_no_stochastic_spikes(model):
    _,result=run(model,2)
    assert np.min(result.outputs_with_units['rate_off'].values)>=0
    assert np.min(result.outputs_with_units['rate_on'].values)>=0
    assert all(result.outputs_with_units[k].unit=='Hz' for k in ('rate_off','rate_on'))
    assert not hasattr(model,'sample_spikes')
    assert result.metadata['source_commit']==COMMIT
    assert 'cross_species_limitations' in result.metadata and 'claim_limit' in result.metadata
    assert all(x.evidence_tag=='synthetic_fixture' for x in result.outputs_with_units.values())
    assert result.claim_level=='numeric_consistency_only'

def test_unsupported_branches_and_explicit_opt_in(model):
    with pytest.raises(UnsupportedReferenceBranch): SomaratnaFreeman2025GaussianHCReference(model.parameters,model.graph,explicit_opt_in=True,horExp=True)
    with pytest.raises(UnsupportedReferenceBranch): SomaratnaFreeman2025GaussianHCReference(model.parameters,model.graph,explicit_opt_in=True,adaptation='unknown')
    with pytest.raises(ContractError): SomaratnaFreeman2025GaussianHCReference(model.parameters,model.graph)
