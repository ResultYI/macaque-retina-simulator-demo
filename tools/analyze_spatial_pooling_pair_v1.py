"""Read saved paired outputs, apply original budgets, and report bounded effects."""
from pathlib import Path
import csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.group_time_monitor import frozen_budget
from retina_sim.group_time_storage import load_bundle
from retina_sim.contracts import canonical_hash

OUT=ROOT/'reports/retinotopic_rf_revision/SPATIAL_POOLING_CANDIDATE_V1'
PARENT=ROOT/'reports/spatial_group_graph_v02_20261025_v1/artifact_r2'
POPS=('diffuse_BC_ON','diffuse_BC_OFF','parasol_RGC_ON','parasol_RGC_OFF')


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def packet(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def savecsv(p,rows):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    out=OUT/'analysis';out.mkdir();numeric=[];endpoints=[];metrics=[];curves=[];events=[];pairs=[]
    structure=read(OUT/'STRUCTURE_RESULT.json');summaries=structure['routes']
    for location in ['near','far']:
        responses={};tight_packets={};ids={};budgets={}
        for arm in ['legacy','candidate']:
            base=OUT/'function_v2'/location/arm
            standard=packet(base/'standard/RESPONSE.npz');tight=packet(base/'tight/RESPONSE.npz');identity=read(base/'standard/IDENTITY.json')
            assert identity['initial_seed_sha256']==read(base/'tight/IDENTITY.json')['initial_seed_sha256']
            assert np.array_equal(standard['time_seconds'],np.arange(901)*.001)
            assert np.array_equal(standard['time_seconds'],tight['time_seconds'])
            seed=load_bundle(base/'initial_seed',expected_kind='source_construction_seed')['payload']
            arm_budgets={}
            for field,contract in identity['fields'].items():
                unit=contract['unit'];initial=standard[field][:,0]
                b=(frozen_budget(standard[contract['population']+'__voltage'][:,0],'V')[2]/.0001 if unit=='V/s'
                    else frozen_budget(initial,'s^-1' if unit=='Hz' else unit)[2])
                err=np.max(abs(standard[field]-tight[field]),axis=1);arm_budgets[field]=b
                numeric.append(dict(location=location,arm=arm,field=field,unit=unit,all_recorded_nodes=901,
                    max_absolute_error=float(err.max(initial=0)),max_error_over_frozen_budget=float((err/b).max(initial=0)),passed=bool(np.all(err<=b))))
            ends=[packet(base/p/'STATE_END.npz') for p in ['standard','tight']]
            for field,c in identity['state_fields'].items():
                k=tuple(field.split('__'));initial=seed['states'][k];b=frozen_budget(initial,c['unit'])[2]
                err=abs(ends[0][field]-ends[1][field])
                endpoints.append(dict(location=location,arm=arm,field=field,count=c['count'],unit=c['unit'],
                    scope='endpoint_only',max_absolute_error=float(err.max(initial=0)),max_error_over_frozen_budget=float((err/b).max(initial=0)),passed=bool(np.all(err<=b))))
            for p in [p for p in identity['fields'] if p.endswith('__expected_binary_count_rate')]:
                pop=p.split('__')[0]
                times=standard[pop+'__event_time_seconds'];indices=standard[pop+'__event_group_index'];sampled=standard[pop+'__sampled_bin_counts']
                assert sampled.shape[1]==900
                binned=np.zeros_like(sampled);valid=bool(np.all((times>=0)&(times<.9)))
                if valid:np.add.at(binned,(indices,np.floor(times/.001+1e-10).astype(int)),1)
                pi=standard[pop+'__history_law_after_tick'];error=float(abs(pi.sum(2)-1).max(initial=0))
                events.append(dict(location=location,arm=arm,population=pop,events=len(times),
                    event_bins_match_counts=bool(valid and np.array_equal(binned,sampled)),
                    probability_mass_error=error,probability_mass_pass=error<=1e-12,
                    standard_tight_sampled_counts_equal=bool(np.array_equal(sampled,tight[pop+'__sampled_bin_counts'])),
                    standard_tight_PCG_equal=canonical_hash(read(base/'standard/END_HISTORY.json')[pop]['rng_state'])==canonical_hash(read(base/'tight/END_HISTORY.json')[pop]['rng_state']),
                    sampled_events_are_group_objects=True))
            responses[arm]=standard;tight_packets[arm]=tight;ids[arm]=identity;budgets[arm]=arm_budgets
        for pop in POPS:
            route=next(r for r in summaries if r['location']==location and r['post']==pop)
            folder=OUT/'structure'/location/route['route']
            with np.load(folder/'IDENTITIES.npz',allow_pickle=False) as z:
                selected=z['post_parent_groups'];core=z['core_parent_groups']
            mask=np.isin(selected,core);members=np.load(PARENT/'populations'/pop/'counts.npy',allow_pickle=False)[selected][mask].astype(float)
            t=responses['legacy']['time_seconds'];down=(t>=.1)&(t<.2);up=(t>=.4)&(t<.5)
            for arm,data in responses.items():
                v=data[pop+'__voltage'][mask];vbudget=budgets[arm][pop+'__voltage'][mask];dv=v-v[:,:1]
                mean=lambda x:np.average(x,weights=members,axis=0)
                vm=mean(v);sign=1 if pop.endswith('_ON') else -1
                polarity=bool(np.all(sign*dv[:,down].mean(1)>vbudget) and np.all(sign*dv[:,up].mean(1)<-vbudget))
                causal_error=float((abs(dv[:,t<.05])/vbudget[:,None]).max(initial=0))
                fields=['voltage','terminal_release'] if pop.startswith('diffuse') else ['excitatory_conductance','inhibitory_conductance','voltage','expected_binary_count_rate']
                finite=all(np.isfinite(data[pop+'__'+field]).all() for field in fields)
                rows=dict(location=location,arm=arm,type=pop,core_groups=int(mask.sum()),represented_members=int(members.sum()),
                    baseline_V_mV=float(vm[0]*1000),down_delta_V_mV=float((vm-vm[0])[down].mean()*1000),up_delta_V_mV=float((vm-vm[0])[up].mean()*1000),
                    polarity_passed=polarity,pre50ms_voltage_budget_ratio=causal_error,causal_pass=causal_error<=1,finite=finite,
                    baseline_terminal_release='',down_delta_terminal_release='',up_delta_terminal_release='',
                    baseline_gE_nS_per_member='',down_delta_gE_nS_per_member='',up_delta_gE_nS_per_member='',
                    baseline_rate_Hz='',down_delta_rate_Hz='',up_delta_rate_Hz='',gE_gI_nonnegative=True,rate_domain_pass=True)
                plotted=dict(V_mV=vm*1000)
                if pop.startswith('diffuse'):
                    rel=mean(data[pop+'__terminal_release'][mask]);plotted['terminal_release']=rel
                    rows.update(baseline_terminal_release=float(rel[0]),down_delta_terminal_release=float((rel-rel[0])[down].mean()),up_delta_terminal_release=float((rel-rel[0])[up].mean()))
                else:
                    ge=data[pop+'__excitatory_conductance'][mask].sum(0)/members.sum()*1e9
                    rate=mean(data[pop+'__expected_binary_count_rate'][mask]);plotted.update(gE_nS_per_member=ge,expected_rate_Hz=rate)
                    rows.update(baseline_gE_nS_per_member=float(ge[0]),down_delta_gE_nS_per_member=float((ge-ge[0])[down].mean()),up_delta_gE_nS_per_member=float((ge-ge[0])[up].mean()),
                        baseline_rate_Hz=float(rate[0]),down_delta_rate_Hz=float((rate-rate[0])[down].mean()),up_delta_rate_Hz=float((rate-rate[0])[up].mean()),
                        gE_gI_nonnegative=bool(np.all(data[pop+'__excitatory_conductance']>=0)&np.all(data[pop+'__inhibitory_conductance']>=0)),
                        rate_domain_pass=bool(np.all((data[pop+'__expected_binary_count_rate']>=0)&(data[pop+'__expected_binary_count_rate']<=1000+1e-10))))
                metrics.append(rows)
                for name,values in plotted.items():
                    for time_s,value in zip(t,values):curves.append(dict(location=location,arm=arm,type=pop,field=name,time_seconds=float(time_s),value=float(value)))
            fields=['voltage','terminal_release'] if pop.startswith('diffuse') else ['excitatory_conductance','voltage','expected_binary_count_rate']
            for field in fields:
                old=responses['legacy'][pop+'__'+field][mask];new=responses['candidate'][pop+'__'+field][mask]
                delta=new-old;numeric_error=max(float(abs(responses[a][pop+'__'+field]-tight_packets[a][pop+'__'+field]).max(initial=0)) for a in ['legacy','candidate'])
                pairs.append(dict(location=location,population=pop,field=field,
                    maximum_absolute_new_old_difference=float(abs(delta).max(initial=0)),
                    maximum_absolute_difference_in_delta_from_own_baseline=float(abs((new-new[:,:1])-(old-old[:,:1])).max(initial=0)),
                    numeric_reference_error=numeric_error,difference_resolved=bool(abs(delta).max(initial=0)>5*numeric_error),
                    two_routes_changed_together_no_single_route_causal_attribution=True))
    savecsv(out/'NUMERICAL_READOUT.csv',numeric);savecsv(out/'NUMERICAL_ENDPOINT.csv',endpoints)
    savecsv(out/'FUNCTIONAL_METRICS.csv',metrics);savecsv(out/'TIME_COURSES.csv',curves);savecsv(out/'NEW_OLD_DIFFERENCES.csv',pairs)
    write(out/'EVENT_CHECKS.json',events)
    passed=all(r['passed'] for r in numeric+endpoints) and all(r['polarity_passed'] and r['causal_pass'] and r['finite'] and r['gE_gI_nonnegative'] and r['rate_domain_pass'] for r in metrics) and all(r['event_bins_match_counts'] and r['probability_mass_pass'] for r in events)
    write(out/'RESULT.json',dict(status='CONDITIONAL_ENGINEERING_CANDIDATE_PASS' if passed else 'VALIDATION_GAP_STOP_RECORDED',
        numerical_readouts_pass=all(r['passed'] for r in numeric),numerical_endpoint_pass=all(r['passed'] for r in endpoints),
        readout_max_error_over_budget=max(r['max_error_over_frozen_budget'] for r in numeric),endpoint_max_error_over_budget=max(r['max_error_over_frozen_budget'] for r in endpoints),
        functional_rows=len(metrics),all_ON_OFF_polarity_pass=all(r['polarity_passed'] for r in metrics),
        core_polarity_scope=list(POPS),all_core_pre50ms_causal_pass=all(r['causal_pass'] for r in metrics),
        no_retrospective_budget_change=True,readout_sampling_nodes=901,full_inner_state_validation='endpoint_only',
        new_old_resolved_fields=sum(r['difference_resolved'] for r in pairs),new_old_fields=len(pairs),
        FIT_READY=False,background_qualified=False,production_adopted=False))
    print(json.dumps(read(out/'RESULT.json')))


if __name__=='__main__':main()
