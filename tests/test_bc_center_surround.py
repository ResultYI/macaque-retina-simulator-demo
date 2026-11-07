"""Input timing and predeclared statistics only; no physiological fixtures."""
import numpy as np
import pytest
from retina_sim.contracts import ContractError
from tools.run_bc_center_surround import SpatialStepRunner, ONSET, OFFSET, STOP, response_statistics, analyze, projection_audit


def test_exact_step_sides_and_duration_without_movie_rounding():
    arm=object.__new__(SpatialStepRunner)
    assert arm.frame_at(0.,'left')==arm.frame_at(0.,'right')==0
    assert arm.frame_at(ONSET,'left')==0 and arm.frame_at(ONSET,'right')==1
    assert arm.frame_at(OFFSET,'left')==1 and arm.frame_at(OFFSET,'right')==2
    assert arm.frame_at(STOP,'right')==2
    assert arm.frame_end(1)-arm.frame_end(0)==pytest.approx(.15)
    for t in (-.001,np.nan,STOP+.001):
        with pytest.raises(ContractError):arm.frame_at(t)


def test_shared_control_subtraction_fixed_window_and_signed_peak():
    t=np.arange(261)*.001
    control=-.05+t*1e-5
    response=np.zeros_like(t);response[(t>=.03)&(t<=.11)]=-2e-6
    result=response_statistics(control+response,t,control,1e-8)
    assert result['mean_20_100ms_V']==pytest.approx(-2e-6)
    assert result['peak_signed_V']==pytest.approx(-2e-6)
    assert result['mean_polarity']=='hyperpolarizing'
    assert result['peak_latency_ms']==20
    result=response_statistics(control,t,control,1e-8)
    assert result['mean_polarity']=='unresolved' and result['peak_absolute_V']==0


def test_ratio_is_undefined_without_a_resolved_center():
    t=np.arange(261)*.001;n='midget_BC_ON'
    arrays={'background__time_seconds':t}
    for condition in ('background','center','annulus'):
        arrays[condition+'__'+n+'__V']=np.full(t.shape,-.05)
        for field in ('cones__release','cones__Bh','HC__Vh'):
            arrays[condition+'__'+field]=np.zeros((1,len(t)))
    cells={n:dict(mandatory=True,direct_cone_group_rows=[0],HC_dependencies=dict(group_rows=[0]))}
    arrays['annulus__'+n+'__V'][30:111]-=5e-7
    assert analyze(arrays,cells)[n]['Q_fixed_window'] is None
    arrays['center__'+n+'__V'][30:111]+=1e-6
    assert analyze(arrays,cells)[n]['Q_fixed_window']==pytest.approx(.5)


def test_disjoint_member_masks_do_not_confer_grouped_input_separation():
    arrays=dict(cone_group_rows=np.array([0,0,1,1]))
    arrays.update(center__fine_mask=np.array([True,False,False,False]),
        annulus__fine_mask=np.array([False,True,False,False]),
        center__group_fraction=np.array([.5,0.]),annulus__group_fraction=np.array([.5,0.]))
    audit=projection_audit(arrays)
    assert audit['member_masks_disjoint']
    assert not audit['ideal_center_region_kept_at_background']
    assert audit['center_mask_members_receiving_annulus_group_drive']==1
    assert audit['maximum_annulus_relative_drive_inside_center']==.025
