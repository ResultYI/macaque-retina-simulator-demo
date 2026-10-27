import numpy as np
import json
from types import SimpleNamespace

from tools.validate_group_time_candidate import clock_events, sample_grid, event_snapshot_sides, preserve_stop_evidence
from retina_sim.group_time_storage import load_bundle


def test_input_event_uses_departing_then_incoming_side_without_ulp_snapping():
    edge=5.
    assert event_snapshot_sides(edge,{edge})==('left','right')
    for neighbor in (np.nextafter(edge,-np.inf),np.nextafter(edge,np.inf)):
        assert event_snapshot_sides(neighbor,{edge})==('right','right')


def test_short_protocol_covers_checkpoint_input_event_and_ordinary_endpoints():
    p=dict(start_seconds=4.993,stop_seconds=5.001,clock_origin_seconds=0.)
    events=clock_events(p)
    assert len(events)==80
    assert [e for e in events if e[0]==4.997]==[(4.997,True,True)]
    assert [e for e in events if e[0]==5.]==[(5.,True,True)]
    records=sample_grid(p['start_seconds'],p['stop_seconds'],.0005)
    assert len(records)==17 and records[0]==4.993 and records[-1]==5.001
    assert sum(t<=4.997 for t in records)==9


def test_guard_stop_flushes_actual_ordinary_values_and_keeps_diagnostic_not_checkpoint(tmp_path):
    values=np.array([.1,.2])
    class Recorder:
        def checkpoint_payload(self):
            np.save(tmp_path/'ordinary.npy',values,allow_pickle=False)
            return dict(index=2,values=values)
    arm=SimpleNamespace(inner_time=.0003,inner_state={('BC','membrane','V'):values})
    failure=dict(message='wall limit',validated_public_time_seconds=.0002)
    preserve_stop_evidence(tmp_path,{'test':'guard'},failure,{'candidate':Recorder()},
        {'candidate':arm},{'candidate':{'states':arm.inner_state}})
    np.testing.assert_array_equal(np.load(tmp_path/'ordinary.npy'),values)
    raw=load_bundle(tmp_path/'stopped_diagnostic',expected_kind='validation_evidence')['payload']
    assert raw['complete_checkpoint'] is False
    assert raw['accepted_inner_frontiers']['candidate']==.0003
    np.testing.assert_array_equal(raw['recording_cursors']['candidate']['values'],values)
    assert json.loads((tmp_path/'failure.json').read_text())['message']=='wall limit'


def test_diagnostic_export_failure_never_erases_original_stop_reason(tmp_path):
    class BrokenRecorder:
        def checkpoint_payload(self):raise OSError('disk unavailable')
    failure=dict(message='original stop')
    preserve_stop_evidence(tmp_path,{},failure,{'candidate':BrokenRecorder()},{},{})
    raw=json.loads((tmp_path/'failure.json').read_text())
    assert raw['message']=='original stop'
    assert raw['stop_evidence_export_error']=='disk unavailable'
