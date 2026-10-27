"""Minimal native-output/history/gradient checks; manufactured labels are test-only."""
from copy import deepcopy
import numpy as np
import pytest
from retina_sim.rgc_stage1_readout import objective,observed_past,free_mean,fit
from retina_sim.contracts import ContractError
from tests.test_ac_vs_model import new_fixture


def base():return dict(C=6e-11,gL=2e-9,EL=-.065,EE=0.,EI=-.08,alpha=90.,mu=-.053,beta=.00167,history_kernel=[-.012,-.008,-.004])


def test_six_output_gradients_and_validation_excluded_from_selection():
    y=np.zeros((2,40),dtype=int);y[0,[5,12,18,29]]=1;y[1,[4,10,20,30]]=1
    c=dict(split='train',counts=y,past=observed_past(y),voltage=np.linspace(-.055,-.049,40),score_bins=[3,40])
    theta=np.array([.1,.1,-.1,.1,.2,.1]);_,gradient=objective(theta,[c],base())
    numeric=[]
    for j in range(6):
        d=np.zeros(6);d[j]=1e-5;numeric.append((objective(theta+d,[c],base())[0]-objective(theta-d,[c],base())[0])/2e-5)
    np.testing.assert_allclose(gradient,numeric,rtol=1e-6,atol=1e-8)
    c['split']='validation'
    with pytest.raises(ContractError):fit([c],base())


def test_cached_voltage_replays_original_8state_law_and_uses_strict_past():
    _,_,model,state=new_fixture();pop='parasol_RGC_ON';p=model.p[pop]
    baseline={k:float(p[k]) for k in ['C','gL','EL','EE','EI','alpha','mu','beta']};baseline['history_kernel']=p['spike_history_kernel']
    v=np.linspace(-.055,-.045,20);initial=np.array([state[pop,'RGC_expected_history','history_probability_'+str(j)][0] for j in range(8)])
    direct=[]
    for value in v:
        state[pop,'RGC_spike_history','V'][:]=value
        direct.append(model.outputs(0.,state)[pop,'expected_rate_Hz'][0]*.001)
        state.update(model._probability_tick_updates(state))
    np.testing.assert_allclose(free_mean(v,baseline,initial),direct,rtol=0,atol=8192*np.finfo(float).eps)
    y=np.array([[1,0,1,0]]);past=observed_past(y)
    np.testing.assert_array_equal(past[0,2],[0,1,0])
