"""Small driver checks, no biological model construction or advance."""
import numpy as np
from types import SimpleNamespace
from scipy.integrate import solve_ivp
from tools.validate_ac_vs_timescale import calendar,CoreRecorder
from tools.validate_group_time_candidate import clock_events
import tools.validate_ac_vs_timescale as stage


def test_exact_calendar_keeps_all_original_ticks_and_required_sides():
    p=dict(start_seconds=4.993,stop_seconds=5.093,checkpoint_seconds=5.043,clock_origin_seconds=0.)
    old=clock_events(p);new=calendar(p)
    assert sum(v['rod'] for t,v in new)==sum(r for t,r,b in old)==1000
    assert sum(v['bc'] for t,v in new)==sum(b for t,r,b in old)==100
    assert sum(v['frame'] for t,v in new)==15
    assert p['checkpoint_seconds'] in dict(new)
    for t,r,b in old:
        assert dict(new)[t]['rod']==r and dict(new)[t]['bc']==b
    assert all(a[0]<b[0] for a,b in zip(new,new[1:]))


def test_original_floating_neighbour_interval_is_not_silently_dropped():
    t=5.02;end=np.nextafter(t,np.inf)
    result=solve_ivp(lambda x,y:np.ones_like(y),(t,end),np.array([1.]),method='DOP853',rtol=1e-6,atol=1e-9,dense_output=True)
    assert result.success and result.t[-1]==end


def test_core_recorder_owns_only_selected_rows_and_restores_cursor(tmp_path):
    spec=dict(selection_hash='fixed',fields={'states__p:m:V':dict(category='states',key=['p','m','V'],rows=[1],unit='V')})
    r=CoreRecorder(spec);full=np.arange(1000,dtype=float)
    s=dict(time_seconds=1.,side='right',event_generation=1,states={('p','m','V'):full})
    r.record(s);full[1]=99.
    assert r.values['states__p:m:V'][0].shape==(1,)
    assert r.values['states__p:m:V'][0][0]==1.
    resumed=CoreRecorder(spec,r.cursor);s['time_seconds']=2.;resumed.record(s)
    assert resumed.cursor['records']==2
    r.save(tmp_path/'prefix.npz');resumed.save(tmp_path/'suffix.npz')
    with np.load(tmp_path/'suffix.npz') as z:assert z['states__p:m:V'].shape==(1,1)


def test_empty_native_auxiliary_port_has_its_own_empty_scope(monkeypatch):
    monkeypatch.setattr(stage,'read',lambda path:{'units':{'cones:z1_aux':'vH_native_voltage','cones:release':'dimensionless'}})
    a=SimpleNamespace(core_population_rows={'cones':np.array([1])},routes={},
        state_blocks=(SimpleNamespace(population='cones',mechanism='reduced_native',name='X',unit='vH_native_precursor'),),
        probability_blocks=(),auxiliary_z1_blocks=())
    spec=stage.selection(a,{'states':{('cones','reduced_native','X'):np.ones(3)}})
    assert spec['fields']['ports__cones:z1_aux']['rows']==[]
    assert spec['fields']['ports__cones:release']['rows']==[1]
