"""Synthetic observer IO/clock tests only; no real model/ODE/fit construction."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib,json
import numpy as np
from retina_sim.rgc_stage1_aligned_recording import AlignedObserver as ActualObserver,observing_arm,digest,canonical_points

POP='parasol_RGC_ON';ORIGIN=4.993;DT=.0001

def AlignedObserver(directory,contract,assembly):
    observer=ActualObserver(directory,contract,assembly)
    observer.set_canonical_points({ORIGIN+j*DT:j for j in range(21)})
    return observer


def test_exact_author_calendar_excludes_near_duplicate_record_only_points():
    times=[ORIGIN+j*DT for j in range(21)]
    events=[(t,dict(rod=True)) for t in times[1:]]
    near=np.nextafter(times[5],np.inf)
    events.append((near,dict(rod=False)))
    points=canonical_points(ORIGIN,times[-1],DT,21,sorted(events))
    assert points[times[5]]==5 and near not in points and len(points)==21


def setup():
    populations={p:SimpleNamespace(group_count=1,group_ids=np.array(['synthetic:'+p]),counts=np.array([1]),centroid_mm=np.array([[1.,0.]]),coordinate_role='rf') for p in ['cones','HC',POP]}
    a=SimpleNamespace(populations=populations)
    c=dict(synthetic_test_only=True,source_nodes=21,source_dt_s=DT,rate_bins=2,source_model_origin_s=ORIGIN,
        state_fields=[dict(key=[p,'synthetic_state','V'],count=1,unit='V',role='dynamic') for p in populations],
        RGC_populations=[POP],canonical_RGC_recording={POP:dict(rows=[0])},buffer_limit_bytes=4096,disk_remaining_reserve_bytes=0,
        provenance=dict(cache_identity='SYNTHETIC_ONLY_NOT_REAL_CELL6'))
    return a,c

class FakeArm:
    def __init__(self,assembly):
        self.assembly=assembly;self.time_seconds=ORIGIN;self.gen=0;self.last_count=np.array([0]);self.calls=0;self.checkpoints=0;self.closed=False
    def snapshot(self,t,side='right',**kwargs):
        self.calls+=1;j=round((t-ORIGIN)/DT);value=np.array([j*.001+self.gen*.0001],dtype=np.float64)
        return dict(time_seconds=t,side=side,event_generation=self.gen,states={(p,'synthetic_state','V'):value.copy() for p in self.assembly.populations},
            ports={(POP,'voltage'):value.copy(),(POP,'gE'):value*.1,(POP,'gI'):value*.2,(POP,'expected_rate_Hz'):np.array([20.+self.gen])},
            port_units={(POP,'voltage'):'V',(POP,'gE'):'S',(POP,'gI'):'S',(POP,'expected_rate_Hz'):'Hz'},
            aux=dict(clocks=dict(indices=dict(rod=j,bc=j//10,rgc=j//10)),input_position=dict(public_native_frame=0),spike_states={POP:dict(last_bin_counts=self.last_count.copy())}))
    def initial_snapshot(self):return self.snapshot(ORIGIN,'right')
    def checkpoint_payload(self):
        self.checkpoints+=1;return dict(time_seconds=self.time_seconds,gen=self.gen,last_count=self.last_count.copy())
    @classmethod
    def from_checkpoint(cls,assembly,payload):
        obj=cls(assembly);obj.time_seconds=payload['time_seconds'];obj.gen=payload['gen'];obj.last_count=payload['last_count'].copy();return obj
    def close(self):self.closed=True

def play(arm,lo=0,hi=21):
    pre=[];saved=None
    if lo==0:arm.initial_snapshot()
    for j in range(lo,hi):
        t=ORIGIN+j*DT;arm.time_seconds=t
        pre.append(arm.snapshot(t,'left' if j==10 else 'right',queue_node=True))
        arm.gen+=1;arm.last_count=np.array([(j//10)%2],dtype=np.int64);arm.snapshot(t,'right',queue_node=True)
        if j==1:
            arm.snapshot(ORIGIN+.00015,'left',queue_node=True);arm.snapshot(ORIGIN+.00015,'right',queue_node=True)
        if j==5:
            near=np.nextafter(t,np.inf)
            arm.snapshot(near,'right',queue_node=True);arm.snapshot(near,'right',queue_node=True)
        if j==10:saved=arm.checkpoint_payload()
    return pre,saved

def cache(tmp,pre):
    tmp.mkdir();values={POP+':'+key:np.vstack([s['ports'][POP,port] for s in pre]) for key,port in [('V','voltage'),('gE','gE'),('gI','gI')]}
    np.savez_compressed(tmp/'endogenous.npz',time_source_s=np.arange(21)*DT,expected_rate_base_Hz=np.vstack([pre[j]['ports'][POP,'expected_rate_Hz'] for j in [0,10]]),**values)
    (tmp/'manifest.json').write_text(json.dumps(dict(schema='SYNTHETIC_TEST_ONLY',arrays_sha256=digest(tmp/'endogenous.npz'))))
    return tmp

def test_aligned_schema_original_calls_and_bin_separation(tmp_path):
    a,c=setup();baseline=FakeArm(a);expected,_=play(baseline)
    observers=[]
    def factory(arm):
        o=AlignedObserver(tmp_path/'product',c,arm.assembly);observers.append(o);return o
    cls=observing_arm(FakeArm,factory);arm=cls(a);actual,_=play(arm);arm.close();o=observers[0]
    assert arm.calls==baseline.calls==47 and arm.gen==baseline.gen and arm.checkpoints==baseline.checkpoints==1
    for x,y in zip(expected,actual):
        for k in x['states']:assert np.array_equal(x['states'][k],y['states'][k])
    assert o.publish(cache(tmp_path/'cache',expected))
    m=json.loads((o.root/'manifest.json').read_text());assert m['status']=='SYNTHETIC_MICROCHECK_COMPLETE' and m['synthetic_test_only']
    idx=m['shared_point_index'];assert np.array_equal(np.load(o.root/idx['time_source_s']['path']),np.arange(21)*DT)
    assert np.load(o.root/idx['side']['path'])[10]==b'left'
    assert np.load(o.root/idx['phase']['path'])[0]==b'initial-precommit'
    assert all(f['field']!='expected_rate_Hz' for f in m['signals'])
    assert np.load(o.root/'rate_bins'/POP/'expected_rate_base_Hz.npy').shape==(2,1)
    assert np.array_equal(np.load(o.root/'rate_bins'/POP/'sampled_group_counts_base.npy'),np.array([[0],[1]]))
    assert m['extra_event_points']==4 and m['legacy_reference_check']['passed']
    assert m['layout']['maximum_value_buffer_bytes']<=c['buffer_limit_bytes']

def test_restore_copies_only_accepted_prefix_exactly(tmp_path):
    a,c=setup();base=FakeArm(a);expected,_=play(base);reference=cache(tmp_path/'cache',expected)
    full=AlignedObserver(tmp_path/'full',c,a);f=observing_arm(FakeArm,lambda _:full)(a);play(f);f.close();assert full.publish(reference)
    prefix=AlignedObserver(tmp_path/'prefix',c,a);p=observing_arm(FakeArm,lambda _:prefix)(a);_,saved=play(p,0,11);p.close()
    resumed=AlignedObserver(tmp_path/'resumed',c,a);r=observing_arm(FakeArm,lambda _:resumed).from_checkpoint(a,saved)
    assert not resumed.error and resumed.streams['primary'].row==11
    play(r,11,21);r.close();assert resumed.publish(reference)
    assert resumed.extra_event_points==4
    for file in (full.root/'primary').rglob('*.npy'):
        assert digest(file)==digest(resumed.root/file.relative_to(full.root))
    for file in (full.root/'post').rglob('*.npy'):
        assert digest(file)==digest(resumed.root/file.relative_to(full.root))
    assert json.loads((resumed.root/'PREFIX_IMPORT.json').read_text())['rows_copied']==11

def test_missing_prefix_never_publishes_fabricated_rows(tmp_path):
    a,c=setup();observer=AlignedObserver(tmp_path/'no_prefix',c,a);arm=observing_arm(FakeArm,lambda _:observer)(a)
    play(arm,5,21);arm.close()
    assert observer.error and not observer.publish(tmp_path/'unneeded_cache')
    assert not (observer.root/'manifest.json').exists()
    assert json.loads((observer.root/'STATUS.json').read_text())['status']=='INCOMPLETE_REQUIRES_LEGITIMATE_REPLAY'

def test_bad_checkpoint_prefix_hash_is_rejected(tmp_path):
    a,c=setup();old=AlignedObserver(tmp_path/'old',c,a);arm=observing_arm(FakeArm,lambda _:old)(a);_,saved=play(arm,0,11);arm.close()
    path=old.root/'primary/values/f0000.npy';data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    new=AlignedObserver(tmp_path/'new',c,a);observing_arm(FakeArm,lambda _:new).from_checkpoint(a,saved)
    assert new.error and 'SHA' in new.error['reason'] and not (new.root/'manifest.json').exists()

def test_entry_dispatch_is_selected_case_only_and_leaves_job_identity(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tools import run_rgc_stage1_cpu_full as original
    from tools.run_rgc_stage1_cpu_full_aligned import install,ROOT
    calls=[]
    def fake_popen(args,*a,**k):calls.append(args);return 'NO_PROCESS_CREATED'
    monkeypatch.setattr(original,'subprocess',SimpleNamespace(Popen=fake_popen,STDOUT=-2,CREATE_NO_WINDOW=0))
    monkeypatch.setattr(original,'Coordinator',original.Coordinator)
    monkeypatch.setattr(original,'WRAPPERS',list(original.WRAPPERS))
    install(dict(case_id='cell6_g3',synthetic_test_only=True))
    for case in ['cell6_g3','cell6_g4']:
        d=tmp_path/case;d.mkdir();j=d/'job.json';j.write_text(json.dumps(dict(case_id=case,computation_identity='same-original')));before=digest(j)
        assert original.subprocess.Popen(['python',str(ROOT/'tools/run_rgc_stage1_cpu_full_case.py'),'--job',str(j)])=='NO_PROCESS_CREATED'
        assert digest(j)==before
        assert (d/'aligned_record_spec.json').exists()==(case=='cell6_g3')
    assert calls[0][1].endswith('run_rgc_stage1_cpu_aligned_case.py')
    assert calls[1][1].endswith('run_rgc_stage1_cpu_full_case.py')
