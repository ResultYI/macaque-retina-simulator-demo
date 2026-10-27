"""The event-copy choice never changes tick laws, ownership or failure atomicity."""
from copy import deepcopy
from types import MethodType,SimpleNamespace
import numpy as np
import pytest
from retina_sim.contracts import canonical_hash
from retina_sim.group_inner_dynamics import GroupInnerDynamics,RGC_MECH
from retina_sim.spatial_group_model import RGC
from retina_sim.group_time_runner import GroupTimeArm,_snapshot_copy
from retina_sim.inner import HistoryRGC
from tests.test_group_inner_dynamics import fixture_model
from tests.test_group_time_runner import toy


def aux_state():
    return toy()[1]['spike_states']


def tick(model,t,state,aux,fast,**kwargs):
    return (model._commit_ticks(t,state,spike_states=aux,_auxiliary_copy=_snapshot_copy,**kwargs)
            if fast else model.commit_ticks(t,state,spike_states=aux,**kwargs))


def test_continuous_ticks_preserve_exact_rng_history_events_and_input_ownership():
    model,state=fixture_model();aux=aux_state();states=[deepcopy(state),deepcopy(state)];spikes=[deepcopy(aux),deepcopy(aux)]
    for point in states:
        for pop in RGC:point[pop,RGC_MECH,'V'][:]=.1  # Manufactured high hazard exercises actual event draws.
    for index in range(31):
        due=index%10==0;t=index*.0001;answers=[]
        for which in range(2):
            before=canonical_hash((states[which],spikes[which]))
            answer=tick(model,t,states[which],spikes[which],which==1,rod_tick=True,
                bc_tick=due,rgc_tick=due,rod_light=np.array([100.,200.]),cone_release=np.array([.2]))
            assert canonical_hash((states[which],spikes[which]))==before
            assert all(not np.shares_memory(answer[0][k],states[which][k]) for k in state)
            answers.append(answer)
        assert canonical_hash(answers[0])==canonical_hash(answers[1])
        if due:
            arm=SimpleNamespace(inner=model,protocol={'clock_origin_seconds':0.})
            draws=[GroupTimeArm._draw_witness(arm,t,spikes[i],answers[i][1],answers[i][2],states[i]) for i in range(2)]
            assert canonical_hash(draws[0])==canonical_hash(draws[1])
            if index==0:assert any(v['event_decision'].any() for v in draws[0].values())
        states=[v[0] for v in answers];spikes=[v[1] for v in answers]


def test_rod_tick_keeps_cross_aliases_cycles_custom_objects_and_delay_copies_owned():
    class ExtendedFloat(float):pass
    model,state=fixture_model();aux=aux_state();array=np.array([1.,2.]);history=[0,1]
    number=ExtendedFloat(2.);number.notes=[]
    extra=dict(a=array,b=array,history=[history,history],number=number,again=number,
        delay=dict(times=[1.,2.],values=array),sampler=dict(rng={'state':7}))
    extra['cycle']=extra;aux['extra']=extra
    populations=list(RGC);aux[populations[1]]['history']=aux[populations[0]]['history']
    result=tick(model,.0001,state,aux,True,rod_tick=True,rod_light=np.array([100.,200.]))[1]
    assert result[populations[0]]['history'] is result[populations[1]]['history']
    assert result[populations[0]]['history'] is not aux[populations[0]]['history']
    answer=result['extra']
    assert answer['a'] is answer['b'] is answer['delay']['values'] and not np.shares_memory(answer['a'],array)
    assert answer['history'][0] is answer['history'][1] and answer['history'][0] is not history
    assert answer['cycle'] is answer and answer['number'] is answer['again'] and answer['number'] is not number
    answer['a'][0]=9.;answer['history'][0][0]=9;answer['number'].notes.append('x');answer['sampler']['rng']['state']=9
    assert array[0]==1. and history[0]==0 and number.notes==[] and extra['sampler']['rng']['state']==7


def test_rod_only_copy_followed_by_flush_events_keeps_caller_pending_lists_owned():
    model,state=fixture_model();aux=aux_state();shared=[dict(cell_index=0,time_seconds=.00005)]
    for pop in RGC:aux[pop]['pending_events']=shared
    before=canonical_hash(aux);outputs=[]
    for fast in (False,True):
        copied=tick(model,.0001,state,aux,fast,rod_tick=True,rod_light=np.array([100.,200.]))[1]
        assert all(copied[pop]['pending_events'] is copied[RGC[0]]['pending_events'] for pop in RGC)
        arm=SimpleNamespace(spike_states=copied,inner_time=.0001,_emitted_until=0.,
            event_log={p:[] for p in RGC},emitted_events_unrecorded={p:[] for p in RGC})
        GroupTimeArm.flush_events(arm,.0001)
        assert all(not copied[pop]['pending_events'] for pop in RGC)
        assert canonical_hash(aux)==before
        outputs.append((arm.spike_states,arm.event_log,arm.emitted_events_unrecorded))
    assert canonical_hash(outputs[0])==canonical_hash(outputs[1])


def test_late_sampler_failure_never_mutates_original_rng_or_state(monkeypatch):
    model,state=fixture_model();aux=aux_state();original=HistoryRGC.sample_spikes
    before=canonical_hash((state,aux));calls=[0]
    def fail_later(self,*args):
        calls[0]+=1
        if calls[0]==2:raise RuntimeError('later sampler failed')
        return original(self,*args)
    monkeypatch.setattr(HistoryRGC,'sample_spikes',fail_later)
    for fast in (False,True):
        calls[0]=0
        with pytest.raises(RuntimeError,match='later sampler failed'):
            tick(model,.001,state,aux,fast,rgc_tick=True)
        assert calls[0]==2 and canonical_hash((state,aux))==before


def test_unknown_copy_failure_is_transactional():
    class Broken:
        def __deepcopy__(self,memo):raise RuntimeError('custom copy failed')
    model,state=fixture_model();aux=aux_state();aux['custom']=Broken()
    before={k:v.tobytes() for k,v in state.items()};rng=canonical_hash(aux[next(iter(aux))]['rng_state'])
    for fast in (False,True):
        with pytest.raises(RuntimeError,match='custom copy failed'):
            tick(model,.0001,state,aux,fast,rod_tick=True,rod_light=np.array([100.,200.]))
        assert {k:v.tobytes() for k,v in state.items()}==before
        assert canonical_hash(aux[next(iter(aux))]['rng_state'])==rng


@pytest.mark.parametrize('mode',['reference','gpu_dop853'])
def test_runner_selects_owned_copy_only_for_explicit_gpu_candidate(mode,monkeypatch):
    model,state=fixture_model();aux=aux_state();calls=[];original=model._commit_ticks
    def observed(*args,**kwargs):calls.append(kwargs.get('_auxiliary_copy'));return original(*args,**kwargs)
    monkeypatch.setattr(model,'_commit_ticks',observed)
    arm=SimpleNamespace(mode=mode,inner=model,inner_time=0.,inner_state=state,spike_states=aux,
        protocol={'clock_origin_seconds':0.},clock_indices={'rod':0,'bc':-1,'rgc':-1},
        last_commit_times={'rod':None,'bc':None,'rgc':None},last_draws={},event_generation=0,
        metrics={'event_commits':0},_cone_release=lambda *_:np.array([.2]),start=0.)
    arm._draw_witness=MethodType(GroupTimeArm._draw_witness,arm)
    GroupTimeArm.commit(arm,0.,bc=True,rgc=True)
    assert calls==([_snapshot_copy] if mode=='gpu_dop853' else [None])
    assert arm.event_generation==1 and arm.metrics['event_commits']==1


@pytest.mark.parametrize('mode',['reference','gpu_dop853'])
def test_runner_late_draw_failure_preserves_all_caller_fields(mode):
    model,state=fixture_model();aux=aux_state()
    def failed(*args):raise RuntimeError('draw witness rejected')
    arm=SimpleNamespace(mode=mode,inner=model,inner_time=0.,inner_state=state,spike_states=aux,
        protocol={'clock_origin_seconds':0.},clock_indices={'rod':0,'bc':-1,'rgc':-1},
        last_commit_times={'rod':None,'bc':None,'rgc':None},last_draws={},event_generation=0,
        metrics={'event_commits':0},_cone_release=lambda *_:np.array([.2]),start=0.,_draw_witness=failed)
    before=canonical_hash((state,aux,arm.clock_indices,arm.last_commit_times,arm.last_draws,arm.metrics))
    with pytest.raises(RuntimeError,match='draw witness rejected'):GroupTimeArm.commit(arm,0.,bc=True,rgc=True)
    assert arm.event_generation==0
    assert canonical_hash((state,aux,arm.clock_indices,arm.last_commit_times,arm.last_draws,arm.metrics))==before


def test_new_process_checkpoint_before_next_rgc_tick_preserves_complete_result(tmp_path):
    import json,os,subprocess,sys
    from pathlib import Path
    from retina_sim.group_time_storage import save_bundle
    model,state=fixture_model();aux=aux_state()
    for pop in RGC:state[pop,RGC_MECH,'V'][:]=.1
    for index in range(10):
        due=index==0
        state,aux,_=tick(model,index*.0001,state,aux,False,rod_tick=True,bc_tick=due,rgc_tick=due,
            rod_light=np.array([100.,200.]),cone_release=np.array([.2]))
    before=canonical_hash((state,aux));expected=tick(model,.001,state,aux,False,rod_tick=True,bc_tick=True,rgc_tick=True,
        rod_light=np.array([100.,200.]),cone_release=np.array([.2]))
    arm=SimpleNamespace(inner=model,protocol={'clock_origin_seconds':0.})
    draw=GroupTimeArm._draw_witness(arm,.001,aux,expected[1],expected[2],state)
    save_bundle(tmp_path/'checkpoint',kind='validation_evidence',identity={'fixture':'event-copy'},payload={'state':state,'aux':aux},
        qualification={'default_adopted':False,'fit_eligible':False})
    script=tmp_path/'resume.py'
    script.write_text('''import json,os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import numpy as np
from types import SimpleNamespace
from retina_sim.group_time_storage import load_bundle
from retina_sim.group_time_runner import GroupTimeArm,_snapshot_copy
from retina_sim.contracts import canonical_hash
from tests.test_group_inner_dynamics import fixture_model
p=load_bundle(Path(sys.argv[2])/'checkpoint',expected_kind='validation_evidence')['payload']
model,_=fixture_model();state,aux=p['state'],p['aux'];before=canonical_hash((state,aux))
answer=model._commit_ticks(.001,state,spike_states=aux,rod_tick=True,bc_tick=True,rgc_tick=True,
    rod_light=np.array([100.,200.]),cone_release=np.array([.2]),_auxiliary_copy=_snapshot_copy)
draw=GroupTimeArm._draw_witness(SimpleNamespace(inner=model,protocol={'clock_origin_seconds':0.}),.001,aux,answer[1],answer[2],state)
assert canonical_hash((state,aux))==before
(Path(sys.argv[2])/'result.json').write_text(json.dumps({'pid':os.getpid(),'before':before,'result':canonical_hash((answer,draw))}))
''',encoding='utf8')
    subprocess.run([sys.executable,'-X','utf8','-B',str(script),str(Path(__file__).resolve().parents[1]),str(tmp_path)],
        check=True,capture_output=True,text=True)
    restored=json.loads((tmp_path/'result.json').read_text(encoding='utf8'))
    assert restored['pid']!=os.getpid() and restored['before']==before
    assert restored['result']==canonical_hash((expected,draw))
