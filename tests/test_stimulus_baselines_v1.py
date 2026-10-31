"""Focused code correctness: causal features, gradient and generated-history indexing."""
import unittest
import numpy as np
from retina_sim.stimulus_baselines_v1 import stimulus_features, history_features, objective, free_glm


class BaselineTests(unittest.TestCase):
    def test_future_stimulus_and_current_spike_cannot_affect_past(self):
        first=np.ones(1000);second=first.copy();second[500:]+=1
        np.testing.assert_array_equal(stimulus_features(first)[:50],stimulus_features(second)[:50])
        y=np.zeros((3,100));y[:,50]=1
        h=history_features(y)
        np.testing.assert_array_equal(h[:,:51],0.)
        np.testing.assert_array_equal(h[:,51,0],1.)
        np.testing.assert_array_equal(h[:,52,1],1.)

    def test_objective_gradient(self):
        rng=np.random.default_rng(7);x=np.column_stack((np.ones(40),rng.normal(size=(40,4))))
        y=(rng.random(40)<.1).astype(float);theta=rng.normal(size=5)*.1
        _,gradient=objective(theta,x,y,.001)
        numerical=[]
        for j in range(5):
            perturb=np.zeros(5);perturb[j]=1e-5
            numerical.append((objective(theta+perturb,x,y,.001)[0]-objective(theta-perturb,x,y,.001)[0])/2e-5)
        np.testing.assert_allclose(gradient,numerical,atol=1e-9,rtol=1e-6)

    def test_free_ring_matches_independent_generated_spike_reference(self):
        signal=np.linspace(-.5,.5,25);kernel=np.array([-2.,-.3,.1,0.,.05])
        probability,_=free_glm(signal,kernel,seed=91,paths=32,batches=4)
        rng=np.random.Generator(np.random.PCG64(91));past=[];expected=[]
        for t,drive in enumerate(signal):
            z=np.full(32,drive)
            for lag,weight in enumerate(kernel,1):
                if t>=lag:z+=weight*past[t-lag]
            q=-np.expm1(-.09*np.logaddexp(0.,z));expected.append(q.mean())
            past.append(rng.random(32)<q)
        np.testing.assert_allclose(probability,expected,atol=1e-15,rtol=1e-14)
        changed=signal.copy();changed[15:]+=10
        p2,_=free_glm(changed,kernel,seed=91,paths=32,batches=4)
        np.testing.assert_array_equal(probability[:15],p2[:15])


if __name__=='__main__':unittest.main()
