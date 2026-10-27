"""Numerical identifiability and finite train-only search, no real model."""
import math
import numpy as np
import pytest
from retina_sim.bc_ac_fullchain_fit import shape_with_error,sensitivity_report,bounded_pattern_fit,admissible_parameters
from retina_sim.contracts import ContractError


def test_normalization_error_encloses_perturbed_waveforms_and_unresolved_signal():
    x=np.array([-2.,-3.,-4.,-2.]);b=.001
    normalized,error=shape_with_error(x,b,[0,4])
    for sign in [-1,1]:
        changed,_=shape_with_error(x+sign*b,0.,[0,4])
        assert np.all(abs(normalized-changed)<=error)
    report=sensitivity_report({'E':(x,b)},{'time':{'E':(x+1e-6, b)}},{'time':math.log(1.25)},[0,4],.01)
    assert not report['all_parameters_resolved'] and not report['joint_parameters_admissible']


def test_independent_training_response_columns_and_collinear_failure():
    x=np.ones(100);a=x.copy();a[:50]+= .1;b=x.copy();b[::2]+=.1
    base={'E':(x,1e-9)};cases={'BC':{'E':(a,1e-9)},'AC':{'E':(b,1e-9)}}
    p=sensitivity_report(base,cases,{'BC':.1,'AC':.1},[0,100],.01)
    assert p['joint_parameters_admissible']
    cases['AC']=cases['BC'];assert not sensitivity_report(base,cases,{'BC':.1,'AC':.1},[0,100],.01)['independent']


def test_finite_search_stays_in_bounds_and_rejects_changes_below_numerical_error():
    seen=[]
    def evaluate(values):
        seen.append(values['time']);return {'loss':(values['time']-1.25)**2,'numerical_error':1e-8}
    p=bounded_pattern_fit(evaluate,{'time':1.},{'time':[.8,1.5]},initial_log_step=math.log(1.25),minimum_log_step=.05,maximum_sweeps=4)
    assert p['values']['time']==pytest.approx(1.25) and all(.8<=v<=1.5 for v in seen)
    p=bounded_pattern_fit(lambda v:{'loss':(v['time']-1.25)**2,'numerical_error':1.}, {'time':1.},{'time':[.8,1.5]},initial_log_step=.2,minimum_log_step=.05,maximum_sweeps=3)
    assert p['accepted_steps']==0 and not p['validation_used_for_selection']
    with pytest.raises(ContractError):bounded_pattern_fit(evaluate,{'time':0.},{'time':[.8,1.5]},initial_log_step=.2,minimum_log_step=.05,maximum_sweeps=3)


def test_only_resolved_independent_times_can_enter_fit_and_one_is_allowed():
    r=dict(selected_parameters=['BC'],parameters=[dict(parameter='BC',resolved_above_numerical_budget=True)],independent=False)
    assert admissible_parameters(r,['BC','AC'])==['BC']
    r['selected_parameters']=['BC','AC'];r['parameters'].append(dict(parameter='AC',resolved_above_numerical_budget=True))
    with pytest.raises(ContractError,match='Dependent'):admissible_parameters(r,['BC','AC'])
    r['selected_parameters']=[]
    with pytest.raises(ContractError,match='No unique'):admissible_parameters(r,['BC','AC'])
