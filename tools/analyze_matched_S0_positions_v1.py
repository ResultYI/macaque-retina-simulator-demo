"""Data-only comparison of the existing near S0 and one new far standard/tight pair."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.group_time_monitor import frozen_budget
from retina_sim.group_time_storage import load_bundle
from retina_sim.contracts import canonical_hash
from tools.cross_position_context_v1 import read,write
from tools.analyze_cross_position_v1 import csv_write


def packet(path):
    with np.load(path,allow_pickle=False) as source:return {k:source[k] for k in source.files}


def numerical(identity,std,tight,seed):
    fields=[];budgets={}
    for name,contract in identity['fields'].items():
        unit=contract['unit'];initial=std[name][:,0]
        budget=(frozen_budget(std[contract['population']+'__voltage'][:,0],'V')[2]/.0001
                if unit=='V/s' else frozen_budget(initial,'s^-1' if unit=='Hz' else unit)[2])
        budgets[name]=budget;error=np.max(abs(std[name]-tight[name]),axis=1)
        fields.append(dict(field=name,unit=unit,maximum_absolute_error=float(error.max()),
            maximum_error_over_frozen_budget=float(np.max(error/budget)),passed=bool(np.all(error<=budget))))
    return fields,budgets


def coordinates(position,identity,center):
    artifact=ROOT/'reports/spatial_group_graph_v02_20261025_v1/artifact_r2'
    manifest=read(artifact/'manifest.json');scale=manifest['configuration']['mm_per_degree'];rows=[]
    for pop,info in identity['types'].items():
        folder=artifact/'populations'/pop;ids=np.load(folder/'group_ids.npy').astype(str)
        lookup={key:i for i,key in enumerate(ids)};selected=np.array([lookup[key] for key in info['group_ids']])
        group_xy=np.load(folder/'centroid_mm.npy')[selected]/scale;radii=np.linalg.norm(group_xy,axis=1)
        offsets=np.load(folder/'member_offsets.npy');members=np.load(folder/'member_global_rows.npy')
        fine=np.concatenate([members[offsets[i]:offsets[i+1]] for i in selected])
        table=ROOT/manifest['populations'][pop]['source_table']['path'];fine_xy=np.load(table.parent/'rf.npy')[fine]/scale
        assert len(fine)==info['total_represented_members']
        fine_r=np.linalg.norm(fine_xy,axis=1)
        rows.append(dict(position=position,type=pop,groups=len(selected),represented_members=len(fine),
            patch_center_x_deg=center[0],patch_center_y_deg=center[1],
            group_RF_ecc_min_deg=float(radii.min()),group_RF_ecc_max_deg=float(radii.max()),
            group_RF_ecc_member_weighted_mean_deg=float(np.average(radii,weights=info['represented_members'])),
            fine_member_RF_ecc_min_deg=float(fine_r.min()),fine_member_RF_ecc_max_deg=float(fine_r.max()),
            fine_member_RF_x_min_deg=float(fine_xy[:,0].min()),fine_member_RF_x_max_deg=float(fine_xy[:,0].max()),
            fine_member_RF_y_min_deg=float(fine_xy[:,1].min()),fine_member_RF_y_max_deg=float(fine_xy[:,1].max()),
            qualification='Frozen source RF coordinates; group dynamics/events do not become independent fine-cell trajectories'))
    return rows


def events(position,data,identity):
    checks=[];records=[]
    for pop,info in identity['types'].items():
        get=lambda field:data[pop+'__'+field]
        sampled=get('sampled_bin_counts');times=get('event_time_seconds');groups=get('event_group_index')
        counted=np.zeros_like(sampled);bins=np.floor(times/.001).astype(int)
        valid=bool(sampled.shape==(info['groups'],900) and np.isin(sampled,[0,1]).all()
            and np.all((times>=0)&(times<.9)) and np.all((groups>=0)&(groups<info['groups'])))
        if valid:np.add.at(counted,(groups,bins),1)
        error=float(np.max(abs(get('history_law_after_tick').sum(2)-1)))
        checks.append(dict(position=position,type=pop,events=len(times),
            expected_group_object_events=float(get('expected_bin_count_rate').sum()*.001),
            event_bin_counts_exact=bool(valid and np.array_equal(counted,sampled)),probability_mass_error=error,
            passed=bool(valid and np.array_equal(counted,sampled) and error<=1e-12)))
        records.extend(dict(position=position,type=pop,group_id=info['group_ids'][int(g)],
            group_index=int(g),time_seconds=float(t),bin_index=int(b),independent_fine_cell_spike=False)
            for t,g,b in zip(times,groups,bins))
    return checks,records


def summarize(position,data,identity,budgets,far_reference=None):
    t=data['time_seconds'];down=(t>=.1)&(t<.2);up=(t>=.4)&(t<.5);rows=[];curves=[]
    for pop,info in identity['types'].items():
        weights=np.asarray(info['represented_members'],float);mean=lambda x:np.average(x,weights=weights,axis=0)
        get=lambda field:data[pop+'__'+field]
        V=get('voltage');rate=get('expected_binary_count_rate');delta=V-V[:,:1];b=budgets[pop+'__voltage']
        mixed=pop=='small_bistratified_S_ON';sign=1 if pop.endswith('_ON') else -1
        polarity=None if mixed else bool(np.all(sign*delta[:,down].mean(1)>b) and np.all(sign*delta[:,up].mean(1)<-b))
        causal=bool(np.all(abs(delta[:,t<.05])<=b[:,None]));preprobe_error=float(np.max(abs(delta[:,t<.05])/b[:,None]))
        finite=all(np.isfinite(data[field]).all() for field in identity['fields'] if field.startswith(pop+'__'))
        inhibition=bool(np.all(get('inhibitory_conductance')>=0) and np.all(get('inhibitory_current')<=1e-25)
            and np.any(get('inhibitory_current')<0))
        if far_reference is None:resolved=None
        else:
            difference=np.max(abs(rate-far_reference[pop+'__expected_binary_count_rate']))
            resolved=bool(np.ptp(mean(rate))>max(5*difference,np.finfo(float).eps*max(1.,float(rate.max()))*8192))
        vm=mean(V);rm=mean(rate);ge=get('excitatory_conductance').sum(0)/weights.sum();gi=get('inhibitory_conductance').sum(0)/weights.sum()
        rows.append(dict(position=position,type=pop,groups=info['groups'],represented_members=info['total_represented_members'],
            baseline_V_mV=float(vm[0]*1000),baseline_rate_Hz=float(rm[0]),
            baseline_gE_nS_per_member=float(ge[0]*1e9),baseline_gI_nS_per_member=float(gi[0]*1e9),
            release_down_deltaV_mV=float((vm-vm[0])[down].mean()*1000),release_up_deltaV_mV=float((vm-vm[0])[up].mean()*1000),
            release_down_delta_rate_Hz=float((rm-rm[0])[down].mean()),release_up_delta_rate_Hz=float((rm-rm[0])[up].mean()),
            late_rod_deltaV_mV=float((vm-vm[0])[800:900].mean()*1000),
            rate_min_Hz=float(rm.min()),rate_max_Hz=float(rm.max()),finite=finite,
            preprobe50ms_stationary=causal,maximum_preprobe_voltage_error_over_frozen_budget=preprobe_error,
            polarity_passed=polarity,mixed_input_S_ON_spectral_polarity_unverified=mixed,
            actual_inhibitory_current_sign_passed=inhibition,rate_dynamic_resolved=resolved,
            sampled_group_object_events=int(get('sampled_bin_counts').sum()),
            expected_group_object_events=float(get('expected_bin_count_rate').sum()*.001)))
        for i,time_s in enumerate(t):
            curves.append(dict(position=position,type=pop,time_seconds=float(time_s),V_mV=float(vm[i]*1000),
                deltaV_local_baseline_mV=float((vm[i]-vm[0])*1000),source_rate_Hz=float(rm[i]),
                delta_rate_local_baseline_Hz=float(rm[i]-rm[0]),gE_nS_per_member=float(ge[i]*1e9),gI_nS_per_member=float(gi[i]*1e9)))
    return rows,curves


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--runs',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();experiment=args.experiment.resolve()
    base=args.runs.resolve() if args.runs else experiment/'dynamic_S0_v1'
    out=args.output.resolve() if args.output else base/'analysis';out.mkdir(exist_ok=False)
    protocol=read(base/'PROTOCOL.json');near_base=Path(protocol['near_reused_runs'])
    far_id=read(base/'standard/IDENTITY.json');tight_id=read(base/'tight/IDENTITY.json');near_id=read(near_base/'standard/IDENTITY.json')
    if far_id['RGC_identity']!=tight_id['RGC_identity']:raise ValueError('Far standard/tight identities differ')
    far=packet(base/'standard/RESPONSE.npz');tight=packet(base/'tight/RESPONSE.npz');near=packet(near_base/'standard/RESPONSE.npz')
    for data in [far,tight,near]:
        if not np.array_equal(data['time_seconds'],np.arange(901)*.001):raise ValueError('Original901-node grid required')
    far_seed=load_bundle(experiment/'background/seed',expected_kind='source_construction_seed')['payload']
    numerical_rows,budgets=numerical(far_id,far,tight,far_seed);near_budgets={}
    for name,contract in near_id['fields'].items():
        if contract['unit']=='V':near_budgets[name]=frozen_budget(near[name][:,0],'V')[2]
    far_rows,far_curves=summarize('far_S0',far,far_id,budgets,tight)
    near_rows,near_curves=summarize('near_S0',near,near_id,near_budgets)
    historical=read(near_base/'analysis/NEAR_TYPE_COMPLETION.json')
    for row in near_rows:row['rate_dynamic_resolved']=next(r['rate_dynamic_resolved'] for r in historical if r['type']==row['type'])
    endpoints=[packet(base/p/'STATE_END.npz') for p in ['standard','tight']];end_rows=[]
    for name,contract in far_id['state_fields'].items():
        initial=far_seed['states'][tuple(name.split('__'))];budget=frozen_budget(initial,contract['unit'])[2]
        error=abs(endpoints[0][name]-endpoints[1][name])
        end_rows.append(dict(field=name,unit=contract['unit'],maximum_absolute_error=float(error.max(initial=0)),
            maximum_error_over_frozen_budget=float(np.max(error/budget,initial=0)),passed=bool(np.all(error<=budget))))
    checks,records=events('far_S0',far,far_id);nc,nr=events('near_S0',near,near_id);checks+=nc;records+=nr
    snapshots=[load_bundle(base/p/'INNER_STATE_0900',expected_kind='validation_evidence')['payload'] for p in ['standard','tight']]
    histories=all(np.array_equal(snapshots[0]['spike_states'][p]['history'],snapshots[1]['spike_states'][p]['history']) for p in far_id['types'])
    rngs=all(canonical_hash(snapshots[0]['spike_states'][p]['rng_state'])==canonical_hash(snapshots[1]['spike_states'][p]['rng_state']) for p in far_id['types'])
    for row in far_rows:
        row['numerical_passed']=all(r['passed'] for r in numerical_rows if r['field'].startswith(row['type']+'__'))
    for row in near_rows:row['numerical_passed']=next(r['numerical_passed'] for r in historical if r['type']==row['type'])
    for row in far_rows+near_rows:
        event=next(r for r in checks if r['position']==row['position'] and r['type']==row['type']);row['event_contract_passed']=event['passed']
        passed=all(row[k] for k in ['finite','preprobe50ms_stationary','actual_inhibitory_current_sign_passed','rate_dynamic_resolved','numerical_passed','event_contract_passed']) and row['polarity_passed'] is not False
        row['status']=('MIXED_INPUT_FUNCTIONAL_S_ON_UNTESTED' if row['mixed_input_S_ON_spectral_polarity_unverified'] else 'CONDITIONAL_FUNCTIONAL_PASS') if passed else 'FUNCTIONAL_GAP_RECORDED'
    coord=coordinates('far_S0',far_id,protocol['far_center_deg'])+coordinates('near_S0',near_id,[-.04510050976293674,-.045844909154906355])
    for name,rows in [('POSITION_TYPE_SUMMARY.csv',near_rows+far_rows),('ALL_TYPE_TIME_COURSES.csv',near_curves+far_curves),
        ('NUMERICAL_REFERENCE.csv',numerical_rows),('FULL_INNER_ENDPOINT_REFERENCE.csv',end_rows),('RGC_COORDINATES_AND_MEMBERS.csv',coord),('GROUP_OBJECT_EVENTS.csv',records)]:csv_write(out/name,rows)
    write(out/'EVENT_HISTORY.json',dict(rows=checks,far_standard_tight_actual_history_equal=histories,far_standard_tight_PCG64_equal=rngs))
    write(out/'POSITION_TYPE_SUMMARY.json',near_rows+far_rows)
    result=dict(status='MATCHED_EXISTING_S0_TWO_POSITION_FUNCTIONAL_PAIR_COMPLETE_WITH_LIMITS',common_S0_id=protocol['common_S0_id'],
        far_RGC_identity=far_id['RGC_identity'],near_RGC_identity=near_id['RGC_identity'],
        numerical_passed=all(r['passed'] for r in numerical_rows+end_rows),
        maximum_readout_error_over_frozen_budget=max(r['maximum_error_over_frozen_budget'] for r in numerical_rows),
        maximum_inner_endpoint_error_over_frozen_budget=max(r['maximum_error_over_frozen_budget'] for r in end_rows),
        far_inner_states=far_id['full_inner_nstate'],readout_fields=len(numerical_rows),
        far_groups=sum(r['groups'] for r in far_rows),far_members=sum(r['represented_members'] for r in far_rows),
        far_object_events=sum(r['sampled_group_object_events'] for r in far_rows),near_object_events=sum(r['sampled_group_object_events'] for r in near_rows),
        far_history_and_rng_equal=histories and rngs,event_contract_passed=all(r['passed'] for r in checks),
        functional_gaps=[dict(position=r['position'],type=r['type']) for r in near_rows+far_rows if r['status']=='FUNCTIONAL_GAP_RECORDED'],
        near_rerun=False,new_S_selective_or_zero_gI_run=False,external_visual_contrast_calibration=False,
        blue_yellow_opponency_verified=False,source_frozen_geometry_not_biological_cell_registration=True,
        FIT_READY=False,background_qualified=False,production_adopted=False)
    write(out/'RESULT.json',result);print(json.dumps(result))


if __name__=='__main__':main()
