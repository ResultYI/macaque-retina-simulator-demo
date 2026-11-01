"""Synthetic software fixtures only; no physiological validation or fitting."""
from copy import deepcopy
import json
import unittest
import numpy as np
from retina_sim import stimulus_baselines_v1 as reference
from retina_sim.online_voltage_glm_v1 import BIN_DT, NATIVE_DT, MembraneParameters, OnlineVoltageGLM, membrane_outputs


def fixture():
    theta=np.linspace(-.2,.2,30);theta[0]=-1.;theta[17:20]=[-2.,-.4,.1]
    model=dict(family='GLM',parameters=30,theta=theta.tolist(),mean=np.zeros(29).tolist(),scale=np.ones(29).tolist())
    physical=MembraneParameters(1.2e-10,4e-9,-.065,0.,-.08,'SYNTHETIC_FIXTURE',2)
    t=np.arange(3000)*NATIVE_DT;v=-.055+.002*np.sin(t*41)
    ge=np.full(len(v),1e-9);gi=np.full(len(v),2e-9)
    counts=(np.random.default_rng(19).random((4,len(v)//10))<.08).astype(float)
    return model,physical,v,ge,gi,counts


def observations(engine,counts,n):
    nodes=np.arange(engine.native_index,engine.native_index+n);edges=nodes[nodes%10==0]
    required={int(j//10-1) for j in edges if j>0}
    if len(edges) and engine.pending_bin is None:required.discard(int(edges[0]//10-1))
    return {k:counts[:,k] for k in required}


def advance(engine,v,ge,gi,counts=None):
    obs=None if counts is None else observations(engine,counts,len(v))
    return engine.advance(v,ge,gi,start_s=engine.origin_s+engine.native_index*NATIVE_DT,completed_spikes=obs)


class OnlineTests(unittest.TestCase):
    def test_conditional_probability_matches_original_features_and_normalization(self):
        model,p,v,ge,gi,y=fixture();engine=OnlineVoltageGLM(model,p,mode='conditional',paths=4,seed=91,source_id='fixture')
        records=[];begin=0
        for stop in (1,10,137,533,1499,len(v)):
            records+=advance(engine,v[begin:stop],ge[begin:stop],gi[begin:stop],y);begin=stop
        case=dict(counts=y,history_features=reference.history_features(y),stimulus_features=reference.stimulus_features(v))
        x=reference.unscaled_design(case,'GLM');z=(x-np.asarray(model['mean']))/np.asarray(model['scale'])@np.asarray(model['theta'])[1:]+model['theta'][0]
        expected=-np.expm1(-.09*np.logaddexp(0.,z));actual=np.stack([r.probability for r in records],axis=1)
        np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=1e-15)
        self.assertEqual(engine.completed_bins,len(v)//10-1)
        engine.close_final_bin(end_s=len(v)*NATIVE_DT,completed_spike=y[:,-1])
        self.assertEqual(engine.completed_bins,len(v)//10)

    def test_free_matches_original_ring_and_full_rng_state(self):
        model,p,v,ge,gi,_=fixture();engine=OnlineVoltageGLM(model,p,mode='free',paths=32,seed=91,source_id='fixture')
        records=advance(engine,v,ge,gi);features=reference.stimulus_features(v)
        theta=np.asarray(model['theta']);mean=np.asarray(model['mean']);scale=np.asarray(model['scale'])
        signal=theta[0]+((features-mean[:16])/scale[:16])@theta[1:17]-(mean[16:]/scale[16:])@theta[17:]
        kernel=reference.history_basis()@(theta[17:]/scale[16:]);expected,batches=reference.free_glm(signal,kernel,seed=91,paths=32,batches=4)
        actual=np.asarray([r.probability.reshape(4,-1).mean(axis=1) for r in records]).T
        np.testing.assert_allclose(actual,batches,rtol=1e-14,atol=1e-15)
        np.testing.assert_allclose(actual.mean(axis=0),expected,rtol=1e-14,atol=1e-15)
        rng=np.random.Generator(np.random.PCG64(91));past=[]
        for n,drive in enumerate(signal):
            z=np.full(32,drive)
            for lag,weight in enumerate(kernel,1):
                if n>=lag:z+=weight*past[n-lag]
            bits=rng.random(32)<-np.expm1(-.09*np.logaddexp(0.,z));past.append(bits)
            np.testing.assert_array_equal(bits,records[n].sampled_bin_outcome)
        self.assertEqual(rng.bit_generator.state,engine.rng.bit_generator.state)

    def test_native_phase_checkpoint_preserves_pending_bin_and_exact_tail(self):
        model,p,v,ge,gi,_=fixture();a=OnlineVoltageGLM(model,p,mode='free',paths=32,seed=91,source_id='fixture')
        advance(a,v[:137],ge[:137],gi[:137]);saved=json.loads(json.dumps(a.checkpoint()))
        self.assertEqual(a.native_index%10,7);self.assertEqual(a.pending_bin,13);self.assertEqual(a.completed_bins,13)
        b=OnlineVoltageGLM.from_checkpoint(model,p,saved,source_id='fixture')
        first=advance(a,v[137:],ge[137:],gi[137:]);second=advance(b,v[137:],ge[137:],gi[137:])
        for x,y in zip(first,second):
            np.testing.assert_array_equal(x.probability,y.probability);np.testing.assert_array_equal(x.sampled_bin_outcome,y.sampled_bin_outcome)
        a.close_final_bin(end_s=.3);b.close_final_bin(end_s=.3)
        self.assertEqual(a.checkpoint(),b.checkpoint())
        bad=deepcopy(saved);bad['payload']['native_index']+=1
        with self.assertRaises(ValueError):OnlineVoltageGLM.from_checkpoint(model,p,bad,source_id='fixture')
        with self.assertRaises(ValueError):OnlineVoltageGLM.from_checkpoint(model,p,saved,source_id='different_input')

    def test_future_voltage_or_current_bin_spike_cannot_change_earlier_predictions(self):
        model,p,v,ge,gi,y=fixture();first=OnlineVoltageGLM(model,p,mode='conditional',paths=4,seed=91,source_id='fixture')
        second=OnlineVoltageGLM(model,p,mode='conditional',paths=4,seed=91,source_id='fixture')
        altered=v.copy();altered[1500:]+=.003;changed=y.copy();changed[:,150]=1-y[:,150]
        a=advance(first,v,ge,gi,y);b=advance(second,altered,ge,gi,changed)
        np.testing.assert_array_equal(np.stack([r.probability for r in a[:150]]),np.stack([r.probability for r in b[:150]]))
        third=OnlineVoltageGLM(model,p,mode='conditional',paths=4,seed=91,source_id='fixture')
        c=advance(third,v,ge,gi,changed)
        np.testing.assert_array_equal(np.stack([r.probability for r in a[:151]]),np.stack([r.probability for r in c[:151]]))
        self.assertFalse(np.array_equal(a[151].probability,c[151].probability))

    def test_step_phase_observation_rejection_and_physical_currents(self):
        model,p,v,ge,gi,_=fixture();engine=OnlineVoltageGLM(model,p,mode='free',paths=1,seed=91,source_id='fixture')
        self.assertIsNotNone(engine.step_native(v[0],ge[0],gi[0],time_s=0.))
        with self.assertRaises(ValueError):engine.step_native(v[1],ge[1],gi[1],time_s=.0001,completed_spike=[1])
        self.assertEqual(engine.native_index,1)
        with self.assertRaises(ValueError):engine.step_native(v[1],ge[1],gi[1],time_s=.0002)
        self.assertIsNone(engine.step_native(v[1],ge[1],gi[1],time_s=.0001))
        out=membrane_outputs(p,v,ge,gi)
        np.testing.assert_array_equal(out['I_membrane_A'],out['I_L_A']+out['I_E_A']+out['I_I_A'])
        np.testing.assert_array_equal(out['dVdt_V_per_s'],out['I_membrane_A']/p.C_total_F)
        np.testing.assert_array_equal(out['tau_eff_s'],p.C_total_F/(p.gL_total_S+ge+gi))
        np.testing.assert_array_equal(out['gA_S'],0.)

    def test_terminal_boundary_uses_the_native_integer_clock_not_a_rounded_literal(self):
        model,p,_,_,_,_=fixture();engine=OnlineVoltageGLM(model,p,mode='free',paths=1,seed=91,source_id='clock_fixture')
        v=np.full(7000,-.055);advance(engine,v,np.zeros_like(v),np.zeros_like(v))
        self.assertNotEqual(.7,7000*NATIVE_DT)
        with self.assertRaises(ValueError):engine.close_final_bin(end_s=.7)
        engine.close_final_bin(end_s=7000*NATIVE_DT)
        self.assertEqual(engine.completed_bins,700)

    def test_packet_partition_preserves_fir_with_small_training_scales(self):
        model,p,_,_,_,_=fixture();v=-.055+1e-8*np.sin(np.arange(7000)*.013)
        full=reference.stimulus_features(v);model['theta'][17:]=np.zeros(13).tolist()
        model['mean'][:16]=full.mean(axis=0).tolist();model['scale'][:16]=full.std(axis=0).tolist()
        engine=OnlineVoltageGLM(model,p,mode='free',paths=1,seed=91,source_id='small_scale_fixture')
        records=[];begin=0
        for end in (173,997,2321,7000):
            records+=advance(engine,v[begin:end],np.zeros(end-begin),np.zeros(end-begin));begin=end
        z=model['theta'][0]+((full-np.asarray(model['mean'])[:16])/np.asarray(model['scale'])[:16])@np.asarray(model['theta'])[1:17]
        expected=-np.expm1(-.09*np.logaddexp(0.,z))
        np.testing.assert_allclose(np.asarray([r.probability[0] for r in records]),expected,rtol=1e-14,atol=1e-15)
        np.testing.assert_array_equal(engine.voltage_tail,(v-v[0])[-4999:])


if __name__=='__main__':unittest.main()
