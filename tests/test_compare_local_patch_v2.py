"""Saved-v2 comparison contracts; no model runs or graph construction."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/'tools/compare_local_patch_v2.py'
spec=importlib.util.spec_from_file_location('compare_patch_v2_under_test',PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_source_identity_lookup_rejects_missing_and_handles_empty():
    assert module.locate([2,5,9],[5,9]).tolist()==[1,2]
    assert module.locate([],[]).size==0
    with pytest.raises(module.ContractError):module.locate([2,5],[6])


@pytest.mark.parametrize('unit,rate,offset,passed',[
    ('V',False,9e-6,True),('V',False,11e-6,False),
    ('S',False,9e-12,True),('S',False,11e-12,False),
    ('Hz',True,.09,True),('Hz',True,.11,False),
    ('dimensionless',False,.0009,True),('dimensionless',False,.0011,False)])
def test_frozen_engineering_budget(unit,rate,offset,passed):
    value=0. if unit in ('V','S') else 1.
    result=module.statistics([value],[value+offset],[value],unit,module.DEFAULT_BUDGET,rate=rate)
    assert result['passed']==passed


def test_empty_state_is_unmeasured_and_nonfinite_rejected():
    result=module.statistics([],[],[],'V',module.DEFAULT_BUDGET)
    assert result['passed'] is None and result['maximum_budget_norm'] is None
    with pytest.raises(module.ContractError):
        module.statistics([np.nan],[0.],[0.],'V',module.DEFAULT_BUDGET)


def test_expected_history_rate_uses_original_law_and_binary_clock():
    p='parasol_RGC_OFF'
    states={(p,'RGC_spike_history','V'):np.array([-.05]),**{
        (p,'RGC_expected_history','history_probability_'+str(i)):np.array([float(i==0)]) for i in range(8)}}
    params={'populations':{p:dict(alpha=10.,mu=-.05,beta=.001,spike_history_kernel=np.array([-.001,-.002,-.003]))}}
    values=module.rates(states,p,params,[0])
    hazard=10*np.log(2.)
    assert values['mean_history_hazard_Hz'][0]==pytest.approx(hazard)
    assert values['expected_binary_count_rate_Hz'][0]==pytest.approx(-np.expm1(-.001*hazard)/.001)
    states[p,'RGC_expected_history','history_probability_1'][0]=.1
    with pytest.raises(module.ContractError):module.rates(states,p,params,[0])


def test_performance_uses_exclusive_advance_and_reports_different_wall_scope():
    graph=SimpleNamespace(populations={'cones':SimpleNamespace(group_count=3),'HC':SimpleNamespace(group_count=2)},
        state_blocks=[SimpleNamespace(count=7)],probability_blocks=[SimpleNamespace(count=2)],
        auxiliary_z1_blocks=[],parent_contact_rows={'a':[0,1]})
    case=dict(graph=graph,result=dict(neural_metrics=dict(native_wall_seconds=2.,inner_wall_seconds=3.,
        build_seconds=.1,snapshot_wall_seconds=4.),simulated_seconds=.005,whole_worker_wall_seconds=20.))
    reference=dict(result=dict(neural_metrics=dict(native_wall_seconds=5.,inner_wall_seconds=10.)))
    value=module.performance(case,reference)
    assert value['continuous_advance_seconds']==5.
    assert value['continuous_advance_speedup_vs_full_reference']==3.
    assert value['compute_seconds_per_biological_second']==1000.
    assert value['whole_run_seconds']==20.
    assert value['logical_state_count']==9


def fake_case(name):
    p='parasol_RGC_OFF'
    populations={n:SimpleNamespace(recording_auxiliary_group_rows=np.array([],dtype=int)) for n in module.POPULATIONS}
    rows={n:np.array([0],dtype=int) if n not in module.RGC or n==p else np.array([],dtype=int) for n in module.POPULATIONS}
    routes={c:SimpleNamespace(contact_component=c,post='rod_BC',name=c) for c in module.CONTACTS}
    graph=SimpleNamespace(parent_population_rows=rows,core_population_rows={n:np.arange(len(v)) for n,v in rows.items()},
        parent_contact_rows={c:np.array([0]) for c in module.CONTACTS},parent_auxiliary_rows={'cones':np.array([],dtype=int)},
        routes=routes,populations=populations,array=lambda path:np.array([0]),
        target_population=p,target_parent_row=0,target_local_row=0,parent_artifact_sha256='same-parent')
    states={(p,'RGC_spike_history','V'):np.array([-.05]),('cones','native','X'):np.array([1.]),**{
        (p,'RGC_expected_history','history_probability_'+str(i)):np.array([float(i==0)]) for i in range(8)}}
    ports={(p,'gE'):np.array([1e-10]),(p,'gI'):np.array([2e-10])}
    aux=dict(clocks={'origin':0.,'indices':{'rod':10}},input_position={'public_native_frame':2},
        observation_delay={'delay_seconds':.00282})
    frame=dict(states=states,ports=ports,port_units={k:'S' for k in ports},time_seconds=1.,side='right',aux=aux,event_generation=10)
    params={'populations':{p:dict(alpha=10.,mu=-.05,beta=.001,spike_history_kernel=np.array([-.001,-.002,-.003]))}}
    seed=dict(states=states,inner_parameters=params,native_inputs={1:np.array([4.])},rod_inputs={1:np.array([2.])})
    return dict(name=name,graph=graph,result={'simulated_seconds':.0031},frames={1.:(frame,True)},final=frame,seed=seed,
        units={k:'V' if k[-1]=='V' else 'dimensionless' for k in states})


def test_complete_saved_core_comparison_checks_input_and_event_causality():
    reference,local=fake_case('reference'),fake_case('halo_030')
    result=module.compare_cases(reference,local,module.DEFAULT_BUDGET)
    assert result['passed'] and result['maximum_budget_norm']==0.
    assert result['input_original_rows_exact']
    local['seed']['native_inputs'][1][0]=5.
    assert not module.compare_cases(reference,local,module.DEFAULT_BUDGET)['passed']


def test_silent_saved_field_omission_is_rejected():
    reference,local=fake_case('reference'),fake_case('halo_030')
    del local['frames'][1.][0]['states']['cones','native','X']
    with pytest.raises(module.ContractError,match='silent field omission'):
        module.compare_cases(reference,local,module.DEFAULT_BUDGET)


def test_explicit_boundary_reservoir_diagnostic_is_preserved_separately():
    reference,local=fake_case('reference'),fake_case('halo_030')
    key=module.BOUNDARY_GAP_PORT
    local['final']['ports'][key]=np.array([0.])
    local['final']['port_units'][key]='A'
    result=module.compare_cases(reference,local,module.DEFAULT_BUDGET)
    diagnostic=result['frames'][0]['boundary_diagnostic']
    assert result['passed'] and diagnostic['unit']=='A'
    assert diagnostic['exterior_reservoir_current_A']==0.
    assert diagnostic['conservation_passed']
    assert not diagnostic['used_as_core_spatial_error']
    assert 'local_patch_boundary:gap_current' not in result['frames'][0]['ports']


def test_unknown_extra_port_and_wrong_reservoir_unit_are_rejected():
    reference,local=fake_case('reference'),fake_case('halo_030')
    local['final']['ports']['unknown','new_field']=np.array([0.])
    with pytest.raises(module.ContractError,match='silent field omission'):
        module.compare_cases(reference,local,module.DEFAULT_BUDGET)
    del local['final']['ports']['unknown','new_field']
    local['final']['ports'][module.BOUNDARY_GAP_PORT]=np.array([0.])
    local['final']['port_units'][module.BOUNDARY_GAP_PORT]='dimensionless'
    with pytest.raises(module.ContractError,match='current in A'):
        module.compare_cases(reference,local,module.DEFAULT_BUDGET)
