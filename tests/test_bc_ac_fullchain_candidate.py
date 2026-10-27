"""Small binding/identity checks, no real model or optimizer execution."""
from copy import deepcopy
from types import SimpleNamespace,MethodType
import numpy as np
import pytest
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.bc_ac_fullchain_candidate import read,bind_parameters,validate_config,signed_self_rms,validate_drive,training_shape_objective
from retina_sim.ac_vs_model import ACVSInner,MECHANISM,LEGACY_S
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def fixture():
    config=read(ROOT/'configurations/calibration/bc_ac_fullchain_candidate_v1.json')
    seed=dict(inner_parameters=dict(identity='original',provenance={},populations={
        'midget_BC_ON':dict(tau_release=.005,tau_a_seconds=.1,adaptation_enabled=True),
        'pool':dict(C=5e-11,gL=2e-9,EL=-.05,EE=0.,alpha=100.,beta=100.),
        LEGACY_S:dict(tau_A_seconds=.01)},contacts={'AII':{'fixed':1}},route_gains={'fixed':1}),states={'kept':np.array([.1])},spike_states={'rng':'unchanged'})
    return config,seed

def test_default_fixed_and_time_binding_reaches_existing_AC_rhs_without_mutating_source():
    c,s=fixture();before=canonical_hash(s)
    c['parameters'].update(active_parameters=['bc.tau_release_s','ac.gate_tau_scale'],bounds={'bc.tau_release_s':[.001,.02],'ac.gate_tau_scale':[.5,3.]},BC_populations=['midget_BC_ON'],AC_populations=['pool'])
    bound,receipt=bind_parameters(s,c,{'bc.tau_release_s':.01,'ac.gate_tau_scale':2.},['pool'])
    assert canonical_hash(s)==before and bound['inner_parameters']['populations']['midget_BC_ON']['tau_release']==.01
    np.testing.assert_array_equal(bound['states']['kept'],s['states']['kept'])
    assert bound['spike_states']==s['spike_states'] and bound['inner_parameters']['contacts']==s['inner_parameters']['contacts']
    def derivative(parameters):
        m=object.__new__(ACVSInner);m.channels=[{'pool':'pool'}];m.C={'pool':5e-11};m.gL={'pool':2e-9}
        m.par=MethodType(lambda self,p,k,**kw:np.array([parameters['populations'][p][k]]),m)
        state={(LEGACY_S,'DynamicAmacrineInhibition','h'):np.array([.3]),('pool',MECHANISM,'V'):np.array([-.04]),('pool',MECHANISM,'s'):np.array([.2])}
        out={(LEGACY_S,'target_inhibition'):np.array([.3]),('pool','p'):np.array([.6]),('pool','gE'):np.array([2e-9]),('pool','s'):np.array([.2]),('pool','voltage'):np.array([-.04])}
        result={};m.ac_rhs(state,out,result);return result
    old,new=derivative(s['inner_parameters']),derivative(bound['inner_parameters'])
    np.testing.assert_allclose(new['pool',MECHANISM,'s'],old['pool',MECHANISM,'s']/2)
    np.testing.assert_array_equal(new['pool',MECHANISM,'V'],old['pool',MECHANISM,'V'])

def test_reject_extra_times_gain_and_nonfinite_without_changing_default_fixed_interface():
    c,s=fixture();bound,receipt=bind_parameters(s,c,{},['pool']);assert not receipt['changes']
    with pytest.raises(ContractError):bind_parameters(s,c,{'output_gain':2.},['pool'])
    c['parameters']['active_parameters']=['output_gain']
    with pytest.raises(ContractError,match='unidentifiable'):validate_config(c)
    c,s=fixture();c['parameters']['active_parameters']=['ac.membrane_tau_s','ac.gate_tau_scale']
    with pytest.raises(ContractError,match='At most one'):validate_config(c)

def test_signed_RMS_keeps_DC_sign_and_proves_pure_gain_unidentifiable():
    x=np.array([-3.,-2.,-1.,-4.]);a,den=signed_self_rms(x,[0,4]);b,_=signed_self_rms(9*x,[0,4])
    np.testing.assert_allclose(a,b);assert np.all(a<0) and a.mean()!=0 and den>0
    with pytest.raises(ContractError):signed_self_rms(np.zeros(4),[0,4])

def test_drive_rejects_cached_EI_and_clock_or_truncation_changes():
    seed=dict(time_seconds=4.993,states={('cones','reduced_native','Vbar'):np.ones(2),('rods','rod_membrane','V'):np.ones(1)})
    d=dict(stimulus_sha256='admitted',native_inputs={0:np.ones(2)},rod_inputs={0:np.ones(1)},model_origin_s=4.993,model_stop_s=4.993+7.1,source_origin_s=0.,source_dt_s=.0001,source_samples=71000,input_driver_identity='fixed',clock_bridge_identity='fixed',native_unit='td',rod_unit='R*/s')
    targets={'groups':[{'stimulus_sha256':'admitted'}]};p=validate_drive(d,seed,{},targets);assert p['start_seconds']==4.993
    bad=deepcopy(d);bad['cached_gI']=np.ones(2)
    with pytest.raises(ContractError,match='never cached'):validate_drive(bad,seed,{},targets)
    bad=deepcopy(d);bad['model_stop_s']-=.1
    with pytest.raises(ContractError,match='truncation'):validate_drive(bad,seed,{},targets)

def test_frozen_driver_and_frame_coverage_are_enforced():
    seed=dict(time_seconds=4.993,states={('cones','reduced_native','Vbar'):np.ones(2),('rods','rod_membrane','V'):np.ones(1)})
    frames=range(748,1814)
    d=dict(stimulus_sha256='admitted',native_inputs={i:np.ones(2) for i in frames},rod_inputs={i:np.ones(1) for i in frames},model_origin_s=4.993,model_stop_s=4.993+7.1,source_origin_s=0.,source_dt_s=.0001,source_samples=71000,input_driver_identity='driver',clock_bridge_identity='clock',native_unit='td',rod_unit='R*/s')
    c,_=fixture();c['launch']['fixed_input_driver']={'identity':'driver','stimulus_payload_sha256':{'admitted':canonical_hash(d)}};c['launch']['fixed_clock_bridge']={'identity':'clock'}
    protocol={'input_origin_seconds':0.,'input_fps':150.};targets={'groups':[{'stimulus_sha256':'admitted'}]}
    validate_drive(d,seed,protocol,targets,c)
    d['clock_bridge_identity']='different'
    with pytest.raises(ContractError,match='fixed_clock_bridge'):validate_drive(d,seed,protocol,targets,c)
    d['clock_bridge_identity']='clock';del d['native_inputs'][800];del d['rod_inputs'][800]
    c['launch']['fixed_input_driver']['stimulus_payload_sha256']['admitted']=canonical_hash(d)
    with pytest.raises(ContractError,match='coverage'):validate_drive(d,seed,protocol,targets,c)

def test_training_objective_excludes_validation_and_preserves_signed_dc():
    y=np.full(71000,-2.);y[20000:40000]=-1.
    cases=[dict(split='train',stimulus_sha256=str(i),targets={'E':y}) for i in range(4)]
    predictions={str(i):3*y for i in range(4)}
    assert training_shape_objective(predictions,cases,'E')['mean_loss']==pytest.approx(0.)
    cases[0]['split']='validation'
    with pytest.raises(ContractError,match='training'):training_shape_objective(predictions,cases,'E')
