"""Summarize the saved frozen4s continuous video; plotting never imports the model."""
import argparse,csv,hashlib,json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def read(path):return json.loads(path.read_bytes())
def write(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
def csv_file(path,rows):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(base):
    from tools.analyze_matched_S0_positions_v1 import numerical as compare_numerical
    from tools.external_video_records_v1 import open_response
    prep=read(base/'PREPARATION.json');rows=[];series=[];events=[];positions={};numerical=[];prefix=[]
    old=ROOT/'reports/external_first_frame_two_positions_20261029_v1'
    units={};selected_ports=['cones:release','HC:h1','HC:h2','rod_BC:voltage','AII_AC:voltage',
        'midget_BC_ON:voltage','midget_BC_OFF:voltage','diffuse_BC_ON:voltage','diffuse_BC_OFF:voltage','S_ON_BC:voltage',
        'same_ON_midget_AC_v1:voltage','same_OFF_midget_AC_v1:voltage','cross_OFF_ON_parasol_AC_v1:voltage',
        'cross_ON_OFF_parasol_AC_v1:voltage','local_feedforward_AC:activation']
    view_fields={};resources=[]
    for position in ['near','far']:
        folder=base/(position+'_run_v1');result=read(folder/'RESULT.json');identity=read(folder/'IDENTITY.json')
        if result['duration_seconds']!=4. or not result['dynamic_numerical_defect']['passed']:raise ValueError('Incomplete or failed continuous run')
        resources.append(dict(position=position,**read(folder/'RESOURCE.json')))
        units.update(read(folder/'PORT_UNITS.json'));position_events=read(folder/'GROUP_OBJECT_EVENTS.json')
        if any(e['time_seconds']<0 or e['time_seconds']>=4. for e in position_events):raise ValueError('Events outside source clip')
        events.extend(dict(position=position,**e) for e in position_events)
        data={};envelopes={}
        with open_response(folder) as z:
            t=z['time_seconds'];side=z['side'];fields=identity['fields']
            if t[0]!=0 or t[-1]!=4. or np.any(np.diff(t)<0):raise ValueError('Saved continuous time coverage differs')
            lookup={(float(a),str(b)):i for i,(a,b) in enumerate(zip(t,side))}
            index=[]
            for ms in range(4001):
                stamp=ms/1000;matches=np.flatnonzero((abs(t-stamp)<1e-12)&(side=='right'))
                if not len(matches):matches=np.flatnonzero(abs(t-stamp)<1e-12)
                if not len(matches):raise ValueError('Original1ms visual sample is unavailable')
                index.append(int(matches[-1]))
            display_time=t[index]
            full_means={}
            for key in fields:
                value=z[key];mean=value.mean(axis=0);data[key]=mean[index].tolist()
                view_fields[key]={k:v for k,v in fields[key].items() if k!='count'}
                view_fields[key]['layout']='Display mean; actual group counts are per position in types'
                full_means[key]=mean
                envelopes[key]=dict(initial=float(mean[0]),minimum=float(mean.min()),maximum=float(mean.max()),end=float(mean[-1]))
            for port in selected_ports:
                key='ports__'+port
                if key not in z.files:continue
                values=z[key]
                if values.shape[0]==0:continue
                data[key]=values.mean(axis=0)[index].tolist()
                view_fields[key]=dict(unit=units[port],meaning='Existing retained groups, equal-weight mean',population=port.split(':')[0])
            for pop,meta in identity['types'].items():
                v=z[pop+'__voltage'];r=z[pop+'__expected_binary_count_rate'];ge=z[pop+'__excitatory_conductance'];gi=z[pop+'__inhibitory_conductance']
                rows.append(dict(position=position,type=pop,groups=meta['groups'],members=sum(meta['represented_members']),
                    initial_group_mean_voltage_mV=float(v[:,0].mean()*1000),minimum_group_mean_voltage_mV=float(v.mean(axis=0).min()*1000),
                    maximum_group_mean_voltage_mV=float(v.mean(axis=0).max()*1000),end_group_mean_voltage_mV=float(v[:,-1].mean()*1000),
                    maximum_any_group_voltage_change_mV=float(np.max(abs(v-v[:,0,None]))*1000),
                    initial_group_mean_binary_rate_Hz=float(r[:,0].mean()),minimum_group_mean_binary_rate_Hz=float(r.mean(axis=0).min()),
                    maximum_group_mean_binary_rate_Hz=float(r.mean(axis=0).max()),end_group_mean_binary_rate_Hz=float(r[:,-1].mean()),
                    maximum_any_group_gE_change_S=float(np.max(abs(ge-ge[:,0,None]))),maximum_any_group_gI_change_S=float(np.max(abs(gi-gi[:,0,None]))),
                    group_events=sum(e['type']==pop for e in position_events)))
                for i,stamp in enumerate(t):
                    record=dict(position=position,type=pop,time_seconds=float(stamp),side=str(side[i]))
                    record.update({k.split('__')[1]:float(full_means[k][i]) for k in fields if k.startswith(pop+'__')});series.append(record)
            oldfolder=old/('near_run_v2' if position=='near' else 'far_run_v1');old_id=read(oldfolder/'IDENTITY.json')
            maximum=0.
            with np.load(oldfolder/'RESPONSE.npz',allow_pickle=False) as previous:
                matching=[lookup[(float(a),str(b))] for a,b in zip(previous['time_seconds'],previous['side'])]
                comparison,_=compare_numerical(old_id,previous,{key:z[key][:,matching] for key in old_id['fields']},None)
                maximum=max(r['maximum_error_over_frozen_budget'] for r in comparison)
            before=[e for e in position_events if e['time_seconds']<1/30]
            same=before==read(oldfolder/'GROUP_OBJECT_EVENTS.json')
            prefix.append(dict(position=position,first33ms_original50field_maximum_frozen_budget_ratio=maximum,
                old75_record_prefix_compared=True,group_event_prefix_identical=same,
                prefix_passed=maximum<=1 and same,no_old_events_appended=True,per_field_comparison=comparison))
            numerical.append(dict(position=position,raw_records=len(t),unique_times=len(np.unique(t)),
                display_records=len(index),display_rule='Original sampled1ms points; no interpolation; terminal optical left side',
                full_50field_raw_time_coverage_seconds=[float(t[0]),float(t[-1])],
                original_physical_history_conservation_passed=result['physical_history_conservation_passed'],
                dynamic_numerical_defect=result['dynamic_numerical_defect'],
                input_hold_edges_checked=len(result['input_hold_boundary_checks']),
                maximum_native_edge_jump=max(q['maximum_continuous_native_jump'] for q in result['input_hold_boundary_checks']),
                single_reference_only=True,no_independent_tight_convergence_claim=True))
            positions[position]=dict(time_seconds=display_time.tolist(),curves=data,envelopes=envelopes,
                types=identity['types'],group_events=position_events)
    csv_file(base/'POSITION_TYPE_SUMMARY.csv',rows);csv_file(base/'RGC_TIME_COURSES.csv',series)
    csv_file(base/'GROUP_OBJECT_EVENTS.csv',events);write(base/'POSITION_TYPE_SUMMARY.json',rows)
    write(base/'NUMERICAL_SCOPE.json',numerical);write(base/'FIRST_FRAME_CONTINUITY.json',prefix)
    write(base/'VIEW_DATA.json',dict(fields=view_fields,positions=positions,units=units,display_only=True,
        no_signal_normalization=True,physical_sources='RESPONSE_INDEX.json and lossless time-chunk NPZ arrays',fixed_source_time_seconds=[0.,4.]))
    preparation=read(base/'PREPARATION_RESOURCE.json');optics=read(base/'OPTICAL_RESOURCE.json');collection=read(base/'CAPTURE_CHECK.json')
    recovery_preflight=read(base/'recovery_v2/PREFLIGHT.json') if (base/'recovery_v2/PREFLIGHT.json').is_file() else {}
    preflight_seconds=recovery_preflight.get('preflight_seconds',0.)
    new=preparation['wall_seconds']+preflight_seconds+optics['wall_seconds']+collection['collection_wall_seconds']+sum(r['wall_seconds'] for r in resources)
    ledger=dict(historical_seconds=preparation['historical_cumulative_seconds'],new_seconds=new,
        cumulative_seconds=preparation['historical_cumulative_seconds']+new,preparation=preparation,
        optics=optics,collection_seconds=collection['collection_wall_seconds'],neural_runs=resources,recovery_preflight_seconds=preflight_seconds,
        delivery_rendering_excluded=True,old_single_frame_cost_retained=True,
        contains_conservative_interruption_cost=bool(optics.get('contains_conservative_interruption_cost')),
        newly_measured_seconds=preparation['wall_seconds']+preflight_seconds+optics.get('measured_recovery_seconds',optics['wall_seconds'])+collection['collection_wall_seconds']+sum(r['wall_seconds'] for r in resources))
    write(base/'RESOURCE_LEDGER.json',ledger)
    write(base/'FINAL_RESULT.json',dict(status='FROZEN_4S_VIDEO_TWO_POSITION_CONTINUOUS_COMPLETE',
        experiment_id=prep['experiment_id'],original_source_frames=120,source_seconds=[0.,4.],
        scientific_claim='Local continuous video functionality only; no fitting or physiological validation',
        numerical=numerical,first_frame_continuity=prefix,type_metrics=rows,ledger=ledger,
        FIT_READY=False,background_qualified=False,production_adopted=False))
    print(json.dumps(dict(status='ANALYZED',new_seconds=new,cumulative_seconds=ledger['cumulative_seconds'],events=len(events))),flush=True)


def plot(base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    view=read(base/'VIEW_DATA.json');fields=view['fields'];pops=list(view['positions']['near']['types'])
    fig,axes=plt.subplots(5,2,figsize=(12,13),sharex=True)
    for position,color in [('near','#0072B2'),('far','#D55E00')]:
        d=view['positions'][position];t=d['time_seconds']
        for i,pop in enumerate(pops):
            for j,(name,scale) in enumerate([('voltage',1000),('expected_binary_count_rate',1)]):
                value=np.asarray(d['curves'][pop+'__'+name]);axes[i,j].plot(t,(value-value[0])*scale,label=position,color=color,lw=1.2)
    for i,pop in enumerate(pops):
        axes[i,0].set_ylabel(pop.replace('_RGC','').replace('_',' ')+'\ndelta V (mV)',fontsize=9)
        axes[i,1].set_ylabel('delta expected binary rate (Hz)',fontsize=9)
        for ax in axes[i]:ax.grid(alpha=.2);ax.set_xlim(0,4);ax.axhline(0,color='#555',lw=.5)
    axes[0,0].legend();axes[0,1].legend()
    for ax in axes[-1]:ax.set_xlabel('Original video time (s)')
    fig.suptitle('Original113.AVI frames0-119: continuous local model\nChanges relative to each S0 initial state; group means; frozen exterior; no fit')
    fig.tight_layout(rect=(0,0,1,.95))
    for ext in ['png','svg','pdf']:fig.savefig(base/('RGC_video_curves.'+ext),dpi=160)
    plt.close(fig)
    chosen=[('ports__cones:release','Cone release (dimensionless)'),('ports__HC:h1','H1 (vH native voltage)'),
        ('ports__midget_BC_ON:voltage','Midget ON BC (V)'),('ports__midget_BC_OFF:voltage','Midget OFF BC (V)'),
        ('ports__same_ON_midget_AC_v1:voltage','Ordinary ON AC (V)'),('ports__AII_AC:voltage','AII AC (V)')]
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
    for ax,(key,label) in zip(axes.flat,chosen):
        for position,color in [('near','#0072B2'),('far','#D55E00')]:
            d=view['positions'][position];ax.plot(d['time_seconds'],d['curves'][key],label=position,color=color,lw=1.2)
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.set_xlim(0,4)
    axes[0,0].legend()
    for ax in axes[-1]:ax.set_xlabel('Original video time (s)')
    fig.suptitle('Actual saved upstream layer responses; original physical/native units')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(base/'upstream_video_curves.png',dpi=160);fig.savefig(base/'upstream_video_curves.svg');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--plot-only',action='store_true');a=p.parse_args();base=a.experiment.resolve()
    (plot if a.plot_only else summarize)(base)
