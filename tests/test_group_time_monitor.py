from copy import deepcopy
from pathlib import Path
import importlib.util

import numpy as np
import pytest

from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.group_time_monitor import GroupTimeMonitor,GroupTimeComparator,frozen_budget,rhs_allowance


@pytest.fixture(scope='module')
def source_fixture():
    path=Path(__file__).with_name('test_group_parameter_binding.py')
    spec=importlib.util.spec_from_file_location('group_monitor_binding_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assembly=module.assembly.__wrapped__()
    for name,pop in assembly.populations.items():pop.group_ids=np.array([name+':0',name+':1'])
    seed=module.construct(assembly)
    return assembly,seed


def evidence(source_fixture):
    assembly,seed=source_fixture
    monitor=GroupTimeMonitor(assembly,seed.states,parameters=seed.inner_parameters)
    states={k:v.copy() for k,v in seed.states.items()}
    rhs={k:np.zeros_like(v) for k,v in states.items()}
    transitions={k:states[k].copy() for k in monitor.transition_keys}
    ports={('cones','q'):np.array([.2,.3]),('cones','release'):np.array([.1,.2]),
        ('AII_AC','gap_current'):np.array([1e-12,-1e-12])}
    units={('cones','q'):'dimensionless',('cones','release'):'dimensionless',('AII_AC','gap_current'):'A'}
    aux=dict(spike_states=deepcopy(seed.spike_states),sample_index=0,input_position={'frame':0},tick_indices={'rod':1,'BC':0,'RGC':0})
    point=dict(time_seconds=0.,side='right',event_generation=0,rhs_kind='original_equation',states=states,rhs=rhs,
        transitions=transitions,ports=ports,aux=aux)
    return monitor,point,units


def test_full_monitor_keeps_original_per_cell_statistics_and_restore(source_fixture):
    monitor,p,units=evidence(source_fixture)
    monitor.observe(0.,'right',p['states'],p['rhs'],transitions=p['transitions'],ports=p['ports'],aux=p['aux'])
    changed={k:v.copy() for k,v in p['states'].items()}
    key=('cones','reduced_native','X');changed[key][0]+=1e-7
    rhs={k:v.copy() for k,v in p['rhs'].items()};rhs[key][0]=.01
    monitor.observe(.0001,'left',changed,rhs,transitions=p['transitions'],ports=p['ports'],event_generation=1,aux=p['aux'])
    payload=monitor.payload()
    assert len(payload['rows'])==114 and payload['background_qualified'] is False
    np.testing.assert_array_equal(payload['rows']['cones:reduced_native:X']['rhs'],[.01,0.])
    np.testing.assert_array_equal(payload['rows']['cones:reduced_native:X']['lo'],p['states'][key])
    assembly,seed=source_fixture
    restored=GroupTimeMonitor.from_payload(assembly,seed.states,payload,parameters=seed.inner_parameters)
    assert restored.report(.0001)==monitor.report(.0001)
    modified=deepcopy(payload);modified['rows']['cones:reduced_native:X']['rhs'][0]=9.
    with pytest.raises(ContractError,match='payload identity'):
        GroupTimeMonitor.from_payload(assembly,seed.states,modified,parameters=seed.inner_parameters)


def test_frozen_budget_and_rhs_formula_exact_no_current_rescaling(source_fixture):
    monitor,p,units=evidence(source_fixture)
    key=('cones','reduced_native','X');row=monitor.rows[key]
    S=np.maximum(np.abs(p['states'][key]),1.)
    np.testing.assert_array_equal(row['budget'],.01*np.minimum(1e-9+1e-6*S,1e-4+1e-4*S))
    values=np.array([3.,4.]);expected=np.minimum(np.minimum(8192*np.finfo(float).eps*S/.0001,row['budget']/.0001)+8192*np.finfo(float).eps*values,row['budget']/.0001)
    np.testing.assert_array_equal(rhs_allowance(S,row['budget'],values,values),expected)
    comparison=GroupTimeComparator(monitor,p['ports'],units,parameters=source_fixture[1].inner_parameters)
    assert comparison.compare_pair(p,deepcopy(p))['status']=='PASS'
    other=deepcopy(p);other['states'][key][0]+=row['budget'][0]*2
    assert comparison.compare_pair(p,other)['status']=='FAIL'
    other=deepcopy(p);other['time_seconds']=np.nextafter(0.,np.inf)
    assert comparison.compare_pair(p,other)['status']=='FAIL'
    other=deepcopy(p);other['rhs_kind']='representation_derivative'
    assert comparison.compare_pair(p,other)['status']=='INCONCLUSIVE'


def test_full_coverage_and_typed_missing_budget_stop(source_fixture):
    monitor,p,units=evidence(source_fixture)
    transitions=dict(p['transitions']);transitions.pop(next(iter(transitions)))
    with pytest.raises(ContractError,match='coverage'):
        monitor.observe(0.,'right',p['states'],p['rhs'],transitions=transitions)
    ports=dict(p['ports']);ports['unknown','unknown']=np.zeros(2)
    unregistered=dict(units);unregistered['unknown','unknown']='unregistered'
    comparator=GroupTimeComparator(monitor,ports,unregistered,parameters=source_fixture[1].inner_parameters)
    changed=deepcopy(p);changed['ports']=ports
    assert comparator.compare_pair(changed,deepcopy(changed))['status']=='INCONCLUSIVE'


def test_ribbon_port_uses_exact_state_propagation_not_arbitrary_events_unit(source_fixture):
    monitor,p,units=evidence(source_fixture);parameters=source_fixture[1].inner_parameters
    pop='rod_BC';pp=parameters['populations'][pop]
    V=p['states'][pop,'ReceptorConductanceMembrane','V'];x=p['states'][pop,'ReceptorConductanceMembrane','x']
    from scipy.special import expit
    rate=pp['Npool']*pp['U']*pp['rmax']*expit((V-pp['r_half'])/pp['r_slope'])*x
    p['ports'][pop,'event_rate']=rate;units[pop,'event_rate']='events/s'
    comparator=GroupTimeComparator(monitor,p['ports'],units,parameters=parameters)
    assert comparator.compare_pair(p,deepcopy(p))['status']=='PASS'
    changed=deepcopy(p);changed['ports'][pop,'event_rate'][0]+=.001
    result=comparator.compare_pair(p,changed)
    assert result['status']=='FAIL'
    assert any('exact' in i['detail'] for i in result['issues'])


def test_original_rng_identity_and_lambda_event_time_interval(source_fixture):
    monitor,p,units=evidence(source_fixture);parameters=source_fixture[1].inner_parameters
    comparator=GroupTimeComparator(monitor,p['ports'],units,parameters=parameters)
    changed=deepcopy(p);changed['aux']['spike_states']['midget_RGC_ON']['rng_state']['state']['state']+=1
    assert comparator.compare_pair(p,changed)['status']=='FAIL'
    pop='midget_RGC_ON';pp=parameters['populations'][pop]
    rate=np.asarray(p['aux']['spike_states'][pop]['conditional_rate_Hz'])
    prob=-np.expm1(-rate*.001)
    u=np.array([prob[0]*.5,(1+prob[1])*.5]);dec=u<prob
    times=np.full(2,np.nan);times[dec]=.001-np.log1p(-u[dec])/rate[dec]
    draw=dict(u=u,rate_Hz=rate,probability=prob,event_decision=dec,event_time_seconds=times,bin_start_seconds=.001,
        history_hash=canonical_hash('history'),rng_before_hash=canonical_hash('before'),rng_after_hash=canonical_hash('after'),draw_index=np.arange(2))
    p['draws']={pop:draw};other=deepcopy(p)
    Bv=monitor.rows[pop,'RGC_spike_history','V']['budget']
    other['draws'][pop]['rate_Hz']=rate+pp['alpha']/pp['beta']*Bv*.1
    new=other['draws'][pop];new['probability']=-np.expm1(-new['rate_Hz']*.001)
    new['event_time_seconds'][dec]=.001-np.log1p(-u[dec])/new['rate_Hz'][dec]
    assert not np.array_equal(times[dec],new['event_time_seconds'][dec])
    assert comparator.compare_pair(p,other)['status']=='PASS'
    other['draws'][pop]['event_time_seconds'][dec]=.002
    assert comparator.compare_pair(p,other)['status']=='FAIL'


def test_monitor_statistic_comparison_preserves_complete_raw_rows(source_fixture):
    a,p,units=evidence(source_fixture);b,_,_=evidence(source_fixture)
    for m in (a,b):m.observe(0.,'right',p['states'],p['rhs'],transitions=p['transitions'],ports=p['ports'],aux=p['aux'])
    comparator=GroupTimeComparator(a,p['ports'],units,parameters=source_fixture[1].inner_parameters)
    result=comparator.compare_monitors(a,b,.0001)
    assert result['status']=='PASS' and len(result['per_cell'])==114*4
    assert result['reference_report']['background_qualified'] is False


def test_delay_numeric_budget_does_not_hide_changed_prehistory_or_input_values(source_fixture):
    monitor,p,units=evidence(source_fixture)
    comparator=GroupTimeComparator(monitor,p['ports'],units,parameters=source_fixture[1].inner_parameters)
    h=p['states']['HC','reduced_native','Vh']
    p['aux']['delay_states']={'HC':dict(unit='vH_native_voltage',history_times=np.array([0.]),history_values=h[None,:].copy(),
        prehistory=dict(unit='vH_native_voltage',values=h.copy()))}
    other=deepcopy(p);other['aux']['delay_states']['HC']['history_values'][0,0]+=monitor.rows['HC','reduced_native','Vh']['budget'][0]*.2
    assert comparator.compare_pair(p,other)['status']=='PASS'
    other['aux']['delay_states']['HC']['prehistory']['values'][0]+=.1
    assert comparator.compare_pair(p,other)['status']=='FAIL'


def test_nonfinite_or_wrong_type_stamp_cannot_be_accepted(source_fixture):
    monitor,p,units=evidence(source_fixture)
    comparator=GroupTimeComparator(monitor,p['ports'],units,parameters=source_fixture[1].inner_parameters)
    other=deepcopy(p);other['time_seconds']=np.nan
    assert comparator.compare_pair(other,deepcopy(other))['status']=='FAIL'
    other=deepcopy(p);other['side']='near';other['event_generation']=True
    assert comparator.compare_pair(other,deepcopy(other))['status']=='INCONCLUSIVE'
    other=deepcopy(p);p['aux']['input_position']['values']=np.array([1.]);other['aux']['input_position']['values']=np.array([2.])
    assert comparator.compare_pair(p,other)['status']=='FAIL'
