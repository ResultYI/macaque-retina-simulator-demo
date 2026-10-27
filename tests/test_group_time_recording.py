"""Manufactured recording values; no simulator or actual neural run is started."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys

import numpy as np
import pytest

from retina_sim.contracts import ContractError
from retina_sim.group_time_recording import GroupRecording, REPLAY_SCHEMA
from retina_sim.spatial_group_model import recording_contract


def layout():
    contract=recording_contract();populations={};fields=[]
    for pop,names in contract['fields'].items():
        count=2
        populations[pop]=SimpleNamespace(group_count=count,counts=np.array([2,1]),
            group_ids=np.array([pop+':group0',pop+':group1']),centroid_mm=np.array([[0.,0.],[.1,0.]]),
            fine_to_group=np.array([0,0,1]),source_positions_mm=np.array([[-.001,0.],[.001,0.],[.1,0.]]),
            coordinate_role='soma' if pop=='HC' else 'ipl_preterminal' if pop=='widefield_AC' else 'rf',
            source_table={'path':'synthetic/table.json','sha256':'fixture'},source_keys={'path':'synthetic/keys.npy','sha256':'fixture'})
        for name in names:
            primary=name==contract['primary'][pop]
            component='outer_native' if pop in ('cones','HC') else pop
            fields.append(SimpleNamespace(population=pop,name=name,key=component+'.'+name,
                component=component,unit='synthetic_native_units' if pop in ('cones','HC') else 'V' if name=='voltage' else 'dimensionless',
                meaning='manufactured recording fixture',primary=primary,
                rows=np.array([0,1] if primary else [0,0]),
                original_fine_rows=None if primary else np.array([0,1]),
                identity='representative_group' if primary else 'group_prediction_at_original_selected_cell'))
    return SimpleNamespace(populations=populations,recording_fields=fields,artifact_sha256='synthetic-group-artifact',
                           manifest={'files':{},'populations':{}})


def values(assembly,offset=0.):
    native={};inner={}
    for f in assembly.recording_fields:
        if f.population in ('cones','HC'):native[f.name]=np.array([1.,2.])+offset
        else:inner[f.population,f.name]=np.array([3.,4.])+offset
    return native,inner


def test_group_recording_45_fields_original_aux_duplicates_and_fresh_read(tmp_path):
    assembly=layout();out=tmp_path/'recording';times=np.array([0.,.0005,.001])
    recording=GroupRecording(assembly,out,times,{'model':'synthetic'},buffer_bytes=1600)
    native,inner=values(assembly)
    native['X']={'values':np.array([7.,7.]),'group_rows':np.array([0,0])}
    recording.record(0.,native,inner,delayedHC=np.array([.1,.2]),eventlogs={'fixture':[{'cell_index':0,'time_seconds':.0001}]})
    payload=recording.checkpoint_payload()
    assert payload['index']==1 and payload['window_index']==1
    restored=GroupRecording.restore(assembly,out,payload,{'model':'synthetic'})
    for t,offset in ((.0005,.5),(.001,1.)):
        native,inner=values(assembly,offset)
        restored.record(t,native,inner,delayedHC=np.array([.1,.2])+offset,eventlogs=[])
    frames=np.zeros((2,256,256),dtype=np.uint8);frames[1]=100
    receipt=restored.finish(frames,np.array([0.,1/150]),np.array([750,751]),
                            {'replay':{'status':'NUMERICAL_BUDGET_FAILED','background_status':'cold_construction_seed'}})
    assert receipt['recorded_fields']==45 and receipt['schema']==REPLAY_SCHEMA
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['demo_schema']==REPLAY_SCHEMA
    assert manifest['replay']['status']=='NUMERICAL_BUDGET_FAILED'
    assert manifest['replay']['physiological_calibration'] is False
    assert manifest['stimulus_geometry']['full_frame_modelled'] is False
    assert manifest['stimulus_geometry']['model_output_disk_radius_deg']==6.
    ports={p['key']:p for c in manifest['components'] for p in c['ports'].values()}
    auxiliary=ports['outer_native.X']
    assert auxiliary['original_fine_global_rows']==[0,1]
    assert auxiliary['group_rows']==[0,0]
    assert len(set(auxiliary['cell_ids']))==2
    assert auxiliary['group_ids'][0]==auxiliary['group_ids'][1]
    assert auxiliary['geometry']['RF']['values']==[[-.001,0.],[.001,0.]]
    with np.load(out/'windows/window_0000.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['outer_native.X'][:,0],[7.,7.])
    with np.load(out/'observations/delayed_hc_0000.npz',allow_pickle=False) as z:
        assert z['query_time'][0]==-.00282
        assert bool(z['feedback_uses_current_h'])
    result=subprocess.run([sys.executable,'-X','utf8','-B','-c',
        "import json,sys; from retina_platform.results import ResultStore; "
        "from retina_platform.reduced_replay import SavedStimulus; "
        "s=ResultStore(sys.argv[1]); v=SavedStimulus(s.path,s.manifest); "
        "print(json.dumps(dict(fields=len(s.read_window(0)['signals']),n=len(s.windows), "
        "schema=s.manifest['demo_schema'],frame=int(v.frame_at(0.)[0,0]))))",str(out)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    fresh=json.loads(result.stdout)
    assert fresh['fields']==45 and fresh['schema']==REPLAY_SCHEMA and fresh['frame']==0


def test_exact_times_and_native_selection_are_checked_before_cursor_change(tmp_path):
    assembly=layout();recording=GroupRecording(assembly,tmp_path/'r',np.array([0.,.0005]))
    native,inner=values(assembly)
    native['X']={'values':np.array([1.,2.]),'group_rows':np.array([0,1])}
    with pytest.raises(ContractError,match='exact ordered'):
        recording.record(0.,native,inner)
    assert recording.writer.index==0
    native,inner=values(assembly)
    recording.record(0.,native,inner)
    with pytest.raises(ValueError,match='unexpected'):
        recording.record(np.nextafter(.0005,0.),native,inner)
    assert recording.writer.index==1


def test_checkpoint_restore_refuses_unknown_later_files_or_tampering(tmp_path):
    assembly=layout();out=tmp_path/'r';recording=GroupRecording(assembly,out,np.array([0.,.0005]))
    recording.record(0.,*values(assembly));payload=recording.checkpoint_payload()
    extra=out/'windows/window_0001.npz';extra.write_bytes(b'preserve later failure')
    with pytest.raises(ContractError,match='later or unknown'):
        GroupRecording.restore(assembly,out,payload)
    assert extra.read_bytes()==b'preserve later failure'
    altered=deepcopy(payload);altered['layout_identity']='other'
    with pytest.raises(ContractError,match='schema/layout'):
        GroupRecording.restore(assembly,out,altered)


def test_delayed_readout_never_replaces_current_HC_primary(tmp_path):
    assembly=layout();out=tmp_path/'r';recording=GroupRecording(assembly,out,np.array([0.,.0005]))
    native,inner=values(assembly);native['Vh']=np.array([4.,5.])
    recording.record(0.,native,inner,delayedHC={'values':np.array([8.,9.]),'group_rows':np.array([0,1]),'query_time':-.00282})
    recording.checkpoint_payload()
    with np.load(out/'windows/window_0000.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['outer_native.Vh'][:,0],[4.,5.])
    with np.load(out/'observations/delayed_hc_0000.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['values'][:,0],[8.,9.])


def test_complete_sample_cursor_can_restore_for_finish_but_never_overwrite_final_result(tmp_path):
    assembly=layout();out=tmp_path/'r';recording=GroupRecording(assembly,out,np.array([0.,.0005]))
    for t in (0.,.0005):recording.record(t,*values(assembly))
    payload=recording.checkpoint_payload()
    restored=GroupRecording.restore(assembly,out,payload)
    restored.finish(np.zeros((2,256,256),dtype=np.uint8),np.array([0.,1/150]),np.array([0,1]))
    before=(out/'manifest.json').read_bytes()
    with pytest.raises(ContractError,match='finalized'):
        GroupRecording.restore(assembly,out,payload)
    assert (out/'manifest.json').read_bytes()==before
