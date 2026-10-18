"""Validate and render the completed private local video trial, one window at a time."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.disk_graph_storage import file_sha
from retina_platform.results import ResultStore


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args()
    directory=(ROOT/args.directory).resolve()
    if not directory.is_relative_to(ROOT):raise ValueError('Use the local project result directory')
    progress=json.loads((directory/'trial.json').read_bytes())
    if progress['status']!='COMPLETED':raise ValueError('The actual trial has not completed')
    store=ResultStore(directory);manifest=store.manifest;origin=manifest['runtime']['stimulus_origin_seconds']
    declared={p['key']:p for c in manifest['components'] for p in c['ports'].values()}
    scalar_count=sum(len(p['cell_ids']) for p in declared.values())
    assert scalar_count==manifest['recording']['scalar_count']
    times=[];events={};traces={};variable_counts={}
    wanted=['state:outer_native.Vis','state:outer_native.Vh','state:midget_BC_ON.terminal_release',
        'state:midget_BC_OFF.terminal_release','state:AII_AC.V','state:midget_RGC_ON.V',
        'state:rod.V','state:parasol_RGC_ON.V']
    # Select one actual state from each remaining requested component if its
    # exact registered voltage name differs; no inferred/derived state added.
    selected=[]
    for candidate in wanted:
        if candidate in declared:selected.append(candidate);continue
        prefix=candidate.rsplit('.',1)[0]+'.'
        alternatives=[key for key in declared if key.startswith(prefix)]
        if alternatives:selected.append(alternatives[0])
    selected=list(dict.fromkeys(selected))
    for index,path in enumerate(store.windows):
        with np.load(path,allow_pickle=False) as data:
            t=data['time'];meta=json.loads(data['manifest_json'].item())
            assert np.all(np.isfinite(t)) and np.all(np.diff(t)>0)
            if times:assert t[0]>times[-1]+1e-12
            times.extend(t.tolist())
            keys=set(data.files)-{'time','manifest_json'}
            assert keys==set(declared)
            for key in keys:
                value=data[key];info=declared[key]
                assert value.shape==(len(info['cell_ids']),len(t)) and np.all(np.isfinite(value))
                assert meta['metadata']['recorded_cell_ids'][key]==info['cell_ids']
                assert meta['ports'][key]['unit']==info['unit'] and meta['ports'][key]['meaning']==info['meaning']
                variable_counts[key]=value.shape[0]
                if key in selected:traces.setdefault(key,[]).extend(value[0].tolist())
            lo,hi=meta['metadata']['event_window_seconds']
            for component,rows in meta['metadata'].get('spike_events',{}).items():
                for event in rows:assert lo<event['time_seconds']<=hi
                events[component]=events.get(component,0)+len(rows)
    times=np.asarray(times);relative=times-origin
    assert len(times)==501 and np.max(np.abs(relative-np.arange(501)*.001))<1e-12
    assert file_sha(Path(manifest['config']['media_path']))==manifest['media_sha256']
    receipts={name:{'bytes':(directory/name).stat().st_size,'sha256':file_sha(directory/name)}
              for name in ('start_checkpoint.json','end_checkpoint.json','latest_checkpoint.json')}
    # Exercise the same public history/metadata reader used by the GUI.
    key=selected[0];read_times,read_values=store.history(key,0)
    np.testing.assert_array_equal(read_times,times);np.testing.assert_array_equal(read_values,np.asarray(traces[key]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(12,10),sharex=True,constrained_layout=True)
    for axis,key in zip(axes.flat,selected):
        info=declared[key];axis.plot(relative*1000,traces[key],lw=1.3,color='#176e8a')
        axis.set_title(key.removeprefix('state:'),fontsize=10);axis.set_ylabel(info['unit'],fontsize=9)
        axis.grid(alpha=.2);axis.tick_params(labelsize=9)
    for axis in list(axes.flat)[len(selected):]:axis.set_visible(False)
    for axis in axes[-1]:axis.set_xlabel('Video time (ms)')
    fig.suptitle('Local real video [0, 0.5 s) | Original HC v7 | Observed internal states at 1 ms\n'
        'One actual recorded cell per panel; mixed display/optics approximation; no new functional-HC acceptance',fontsize=12)
    fig.savefig(directory/'observed_internal_states.png',dpi=160);plt.close(fig)
    report={'status':'REAL_VIDEO_05S_RECORDED_STATES_AND_HISTORY_READ_PASS','local_private_results_do_not_publish':True,
        'neural_scope':'Original HC v7 transition graph, not new functional-HC acceptance',
        'input_clip_seconds':[0,.5],'observed_state_time_seconds':[float(relative[0]),float(relative[-1])],
        'samples':len(times),'sample_step_seconds':.001,'windows':len(store.windows),
        'registered_state_variables':len(declared),'scalar_states_per_sample':scalar_count,
        'unique_raw_state_bytes':scalar_count*len(times)*8,'recorded_native_cones':manifest['recording']['recorded_cones'],
        'recorded_HC':manifest['recording']['recorded_HC'],'variable_cell_counts':variable_counts,
        'spike_events_by_component':events,'checkpoint_files':receipts,
        'phase_seconds':progress['phase_seconds'],'memory_statistics':progress['memory_statistics'],
        'wall_seconds':progress['wall_seconds'],'input_preparation':manifest['input_assumptions'].get('phase_wall_seconds'),
        'public_ResultStore_history_bitwise_equal':True,'source_video_bytes_unchanged':True,
        'arbitrary_full_graph_or_continuous_time_replay_implemented':False,
        'figure':'observed_internal_states.png','figure_signals':selected,
        'figure_cell_ids':{key:declared[key]['cell_ids'][0] for key in selected}}
    (directory/'validation.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({key:report[key] for key in ('status','samples','registered_state_variables','scalar_states_per_sample',
        'windows','wall_seconds','phase_seconds','memory_statistics')}),flush=True)

if __name__=='__main__':main()
