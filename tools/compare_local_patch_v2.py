"""Compare already-saved Local Patch v2 runs; this tool never advances a model."""
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.group_time_storage import load_bundle
from retina_sim.local_patch_graph import LocalPatchAssembly
from retina_sim.rgc_expectation import STATES
from retina_sim.spatial_group_graph import digest
from retina_sim.spatial_group_model import POPULATIONS, RGC

CONTACTS = ('individual_rod_RBC_contacts', 'RBC_AII_AMPA', 'AII_FMB_glycine', 'AII_DB1_glycine')
BOUNDARY_GAP_PORT = ('local_patch_boundary', 'gap_current')
DEFAULT_BUDGET = dict(voltage_V=1e-5, conductance_S=1e-11, expected_rate_Hz=.1,
                      general_atol=1e-8, general_rtol=1e-3)


def locate(source, selected):
    source, selected = np.asarray(source), np.asarray(selected)
    if source.ndim != 1 or selected.ndim != 1:
        raise ContractError('One-dimensional original identities required')
    index = np.searchsorted(source, selected)
    if np.any(index >= len(source)) or not np.array_equal(source[index], selected):
        raise ContractError('Common original identity is absent from a saved frame')
    return index


def statistics(reference, local, initial, unit, budget, *, rate=False):
    a, b, z = (np.asarray(v, dtype=np.float64) for v in (reference, local, initial))
    if a.ndim != 1 or a.shape != b.shape or a.shape != z.shape or not all(np.isfinite(v).all() for v in (a,b,z)):
        raise ContractError('Finite matched vectors required for comparison')
    if not len(a):
        return dict(count=0, unit=unit, max_abs=None, rms=None, maximum_budget_norm=None, passed=None)
    if rate:
        allowance = np.full_like(a, budget['expected_rate_Hz'])
    elif unit == 'V':
        allowance = np.full_like(a, budget['voltage_V'])
    elif unit == 'S':
        allowance = np.full_like(a, budget['conductance_S'])
    else:
        allowance = budget['general_atol'] + budget['general_rtol']*np.maximum(np.abs(a),np.abs(z))
    residual = b-a
    norm = np.abs(residual)/allowance
    worst = int(np.argmax(norm))
    return dict(count=len(a),unit=unit,max_abs=float(np.max(np.abs(residual))),
        rms=float(np.sqrt(np.mean(residual*residual))),maximum_budget_norm=float(np.max(norm)),
        worst_vector_row=worst,reference_at_worst=float(a[worst]),local_at_worst=float(b[worst]),
        allowance_at_worst=float(allowance[worst]),passed=bool(np.max(norm)<=1.))


def rates(states, population, parameters, state_rows, parameter_rows=None):
    rows = np.asarray(state_rows,dtype=np.int64)
    prow = rows if parameter_rows is None else np.asarray(parameter_rows,dtype=np.int64)
    V = np.asarray(states[population,'RGC_spike_history','V'])[rows]
    law = np.column_stack([np.asarray(states[population,'RGC_expected_history',
        'history_probability_'+str(i)])[rows] for i in range(8)])
    p = parameters['populations'][population]
    def parameter(name):
        a = np.asarray(p[name],dtype=np.float64)
        return np.full(len(rows),float(a)) if a.ndim == 0 else a[prow]
    alpha,mu,beta = (parameter(n) for n in ('alpha','mu','beta'))
    kernel=np.asarray(p['spike_history_kernel'],dtype=np.float64)
    if (kernel.shape != (3,) or np.any(alpha<=0) or np.any(beta<=0)
            or np.any(law < -1e-13) or np.max(np.abs(law.sum(axis=1)-1),initial=0.)>1e-12):
        raise ContractError('Original finite binary-history law/parameters invalid')
    intensity=alpha[:,None]*np.logaddexp(0.,(V[:,None]+(STATES@kernel)[None,:]-mu[:,None])/beta[:,None])
    return dict(mean_history_hazard_Hz=np.sum(law*intensity,axis=1),
        expected_binary_count_rate_Hz=np.sum(law*(-np.expm1(-.001*intensity)),axis=1)/.001)


def binding(path):
    p=Path(path)
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p),bytes=p.stat().st_size)


def load_case(stage,name):
    graph=LocalPatchAssembly(stage/('graph_'+name),ROOT)
    directory=stage/('run_'+name)
    result=json.loads((directory/'result.json').read_text(encoding='utf8'))
    if result.get('status') != 'SHORT_AGGREGATE_PATCH_FUNCTIONAL_RUN_COMPLETE':
        raise ContractError('Only complete functional runs enter comparison')
    identity=result['identity']
    if identity['patch_artifact_sha256'] != graph.artifact_sha256:
        raise ContractError('Run and graph identities differ')
    seed=load_bundle(directory/'construction_seed',expected_kind='source_construction_seed',expected_identity=identity)['payload']
    if canonical_hash(seed) != identity['seed_sha256']:
        raise ContractError('Run construction seed identity differs')
    final=load_bundle(directory/'final_snapshot',expected_kind='validation_evidence',expected_identity=identity)['payload']
    if final['time_seconds'] != result['final_time_seconds']:
        raise ContractError('Final endpoint identity differs')
    frames={}
    sources=[binding(directory/'result.json'),binding(directory/'construction_seed/manifest.json'),
             binding(directory/'final_snapshot/manifest.json'),binding(stage/('graph_'+name)/'manifest.json')]
    for folder in ('observations_prefix','observations_suffix','observations_complete'):
        path=directory/folder
        if path.exists():
            data=load_bundle(path,expected_kind='validation_evidence',expected_identity=identity)['payload']
            sources.append(binding(path/'manifest.json'))
            for frame in data['observations']:
                if frame['side']=='right':frames[float(frame['time_seconds'])]=(frame,False)
    frames[float(final['time_seconds'])]=(final,True)
    if not set(result['recorded_times'])<=set(frames):
        raise ContractError('Saved core observations omit regular recording times')
    units={(b.population,b.mechanism,b.name):b.unit for b in
           (*graph.state_blocks,*graph.probability_blocks,*graph.auxiliary_z1_blocks)}
    return dict(name=name,graph=graph,directory=directory,result=result,seed=seed,final=final,
                frames=frames,units=units,sources=sources)


def identity_rows(case, frame, full):
    g=case['graph']
    pop={p:np.asarray(g.parent_population_rows[p] if full else frame['identities'][p]['parent_group_rows']) for p in POPULATIONS}
    contact={}
    for component in CONTACTS:
        route=next(r for r in g.routes.values() if r.contact_component==component)
        rows=np.arange(len(g.parent_contact_rows[component]))
        if not full:
            posts=g.array('routes/'+route.name+'/contact_post_group.npy')
            rows=rows[np.isin(posts,g.core_population_rows[route.post])]
        contact[component]=np.asarray(g.parent_contact_rows[component])[rows]
    aux=np.asarray(g.parent_auxiliary_rows['cones'])
    if not full:
        cone=g.populations['cones'].recording_auxiliary_group_rows
        aux=aux[np.isin(cone,g.core_population_rows['cones'])]
    return pop,contact,aux


def core_rows(case):
    g=case['graph']
    return {p:np.asarray(g.parent_population_rows[p])[g.core_population_rows[p]] for p in POPULATIONS}


def state_identity(key, maps):
    pop,contact,aux=maps
    if key[1] in contact:return contact[key[1]]
    if key[1]=='selected_original_z1_auxiliary':return aux
    return pop[key[0]]


def port_identity(key, maps, graph):
    pop,contact,aux=maps
    if key[0] in pop:return aux if key[-1]=='z1_aux' else pop[key[0]]
    if key[0] in contact:
        if key[-1]=='contact_conductance':return contact[key[0]]
        route=next(r for r in graph.routes.values() if r.contact_component==key[0])
        return pop[route.post]
    raise ContractError('Unregistered saved port identity: '+str(key))


def physiological_port_keys(reference, local):
    """One explicit exterior-reservoir diagnostic supplements the original ports."""
    keys, other = set(reference), set(local)
    if keys-other or other-keys-{BOUNDARY_GAP_PORT}:
        raise ContractError('Saved declarations differ; no silent field omission allowed')
    return sorted(keys)


def boundary_diagnostic(frame, units, full):
    if BOUNDARY_GAP_PORT not in frame['ports']:
        return None
    value=np.asarray(frame['ports'][BOUNDARY_GAP_PORT],dtype=np.float64)
    if units.get(BOUNDARY_GAP_PORT)!='A' or value.shape!=(1,) or not np.isfinite(value).all():
        raise ContractError('Explicit finite scalar boundary reservoir current in A required')
    result=dict(port=list(BOUNDARY_GAP_PORT),unit='A',exterior_reservoir_current_A=float(value[0]),
        physical_role='Opposite current of the omitted reciprocal gap exterior; no dynamic state equation',
        used_as_core_spatial_error=False,full_dynamic_current_observation=full,
        retained_dynamic_gap_current_sum_A=None,total_gap_current_A=None,
        original_conservation_allowance_A=None,conservation_passed=None)
    if full:
        retained=[np.asarray(v,dtype=np.float64) for k,v in frame['ports'].items()
                  if k[-1]=='gap_current' and k!=BOUNDARY_GAP_PORT]
        dynamic=sum(float(np.sum(v)) for v in retained)
        magnitude=sum(float(np.sum(np.abs(v))) for v in retained)+abs(float(value[0]))
        total=dynamic+float(value[0])
        allowance=1e-16+1e-10*magnitude
        result.update(retained_dynamic_gap_current_sum_A=dynamic,total_gap_current_A=total,
            original_conservation_allowance_A=allowance,conservation_passed=abs(total)<=allowance)
    return result


def common_core_maps(reference,local):
    rg,lg=reference['graph'],local['graph']
    cores=[core_rows(reference),core_rows(local)]
    pop={p:np.intersect1d(cores[0][p],cores[1][p]) for p in POPULATIONS}
    contact={}
    for component in CONTACTS:
        values=[]
        for g in (rg,lg):
            route=next(r for r in g.routes.values() if r.contact_component==component)
            posts=g.array('routes/'+route.name+'/contact_post_group.npy')
            parent_posts=np.asarray(g.parent_population_rows[route.post])[posts]
            values.append(np.asarray(g.parent_contact_rows[component])[np.isin(parent_posts,pop[route.post])])
        contact[component]=np.intersect1d(*values)
    values=[]
    for g in (rg,lg):
        cones=g.populations['cones'].recording_auxiliary_group_rows
        parents=np.asarray(g.parent_population_rows['cones'])[cones]
        values.append(np.asarray(g.parent_auxiliary_rows['cones'])[np.isin(parents,pop['cones'])])
    return pop,contact,np.intersect1d(*values)


def compare_cases(reference,local,budget):
    rg,lg=reference['graph'],local['graph']
    if (rg.target_population!=lg.target_population or rg.target_parent_row!=lg.target_parent_row
            or rg.parent_artifact_sha256!=lg.parent_artifact_sha256
            or reference['result']['simulated_seconds']!=local['result']['simulated_seconds']):
        raise ContractError('Reference and bounded run source, target or duration differ')
    times=sorted(set(reference['frames'])&set(local['frames']))
    if set(times)!=set(reference['frames']) or set(times)!=set(local['frames']):
        raise ContractError('Matched full scheduled observation coverage required')
    common=common_core_maps(reference,local)
    first_ref,first_full=reference['frames'][times[0]]
    first_maps=identity_rows(reference,first_ref,first_full)
    seed_maps=(reference['graph'].parent_population_rows,reference['graph'].parent_contact_rows,
               np.asarray(reference['graph'].parent_auxiliary_rows['cones']))
    evidence=[]
    for time in times:
        r,rf=reference['frames'][time];l,lf=local['frames'][time]
        rm,lm=identity_rows(reference,r,rf),identity_rows(local,l,lf)
        states,ports,expected={},{},{}
        if set(r['states'])!=set(l['states']):
            raise ContractError('Saved declarations differ; no silent field omission allowed')
        original_ports=physiological_port_keys(r['ports'],l['ports'])
        diagnostic=boundary_diagnostic(l,local['final']['port_units'],lf)
        for key in sorted(r['states']):
            parents=state_identity(key,common)
            ri,li=locate(state_identity(key,rm),parents),locate(state_identity(key,lm),parents)
            zi=locate(state_identity(key,seed_maps),parents)
            states[':'.join(key)]=statistics(np.asarray(r['states'][key])[ri],np.asarray(l['states'][key])[li],
                np.asarray(reference['seed']['states'][key])[zi],reference['units'][key],budget)
        for key in original_ports:
            parents=port_identity(key,common,rg)
            ri,li=locate(port_identity(key,rm,rg),parents),locate(port_identity(key,lm,lg),parents)
            zi=locate(port_identity(key,first_maps,rg),parents)
            unit=reference['final']['port_units'][key]
            if unit!=local['final']['port_units'][key]:raise ContractError('Saved port units differ')
            ports[':'.join(key)]=statistics(np.asarray(r['ports'][key])[ri],np.asarray(l['ports'][key])[li],
                np.asarray(first_ref['ports'][key])[zi],unit,budget)
        for p in RGC:
            parents=common[0][p]
            if not len(parents):continue
            indices=[locate(m[0][p],parents) for m in (rm,lm,first_maps)]
            pindices=[locate(c['graph'].parent_population_rows[p],parents) for c in (reference,local)]
            values=[rates(frame['states'],p,c['seed']['inner_parameters'],i,pi)
                    for frame,c,i,pi in ((r,reference,indices[0],pindices[0]),(l,local,indices[1],pindices[1]),
                        (first_ref,reference,indices[2],pindices[0]))]
            expected[p]={name:statistics(values[0][name],values[1][name],values[2][name],'Hz',budget,rate=True)
                         for name in values[0]}
        categories=list(states.values())+list(ports.values())+[s for p in expected.values() for s in p.values()]
        nonempty=[s for s in categories if s['count']]
        evidence.append(dict(time_seconds=time,states=states,ports=ports,expected_rgc_rates=expected,
            boundary_diagnostic=diagnostic,
            maximum_budget_norm=max((s['maximum_budget_norm'] for s in nonempty),default=None),
            passed=all(s['passed'] for s in nonempty)))
    input_evidence={}
    for category,pop in (('native_inputs','cones'),('rod_inputs','rods')):
        rr=locate(rg.parent_population_rows[pop],lg.parent_population_rows[pop])
        if set(reference['seed'][category])!=set(local['seed'][category]):
            raise ContractError('Reference and local input frames differ')
        input_evidence[category]={str(frame):bool(np.array_equal(np.asarray(reference['seed'][category][frame])[rr],value))
            for frame,value in local['seed'][category].items()}
    exact_input=all(v for frames in input_evidence.values() for v in frames.values())
    clock_evidence={key:dict(reference=reference['final']['aux'][key],local=local['final']['aux'][key],
        exact_match=canonical_hash(reference['final']['aux'][key])==canonical_hash(local['final']['aux'][key]))
        for key in ('clocks','input_position')}
    core_coverage={p:bool(np.array_equal(core_rows(reference)[p],common[0][p])) for p in POPULATIONS
                   if p not in RGC or p==rg.target_population}
    return dict(candidate=local['name'],input_original_rows_exact=exact_input,input_evidence=input_evidence,
        event_generation=[reference['final']['event_generation'],local['final']['event_generation']],
        clock_and_input_position=clock_evidence,
        delay_seconds=[reference['final']['aux']['observation_delay']['delay_seconds'],
                       local['final']['aux']['observation_delay']['delay_seconds']],
        required_reference_core_identity_coverage=core_coverage,
        common_core_group_counts={p:len(v) for p,v in common[0].items()},
        common_core_parent_rows={p:v.tolist() for p,v in common[0].items()},
        common_core_contact_counts={p:len(v) for p,v in common[1].items()},
        common_core_auxiliary_z1_count=len(common[2]),frames=evidence,
        passed=all(e['passed'] for e in evidence) and exact_input and all(core_coverage.values())
            and all(v['exact_match'] for v in clock_evidence.values())
            and reference['final']['event_generation']==local['final']['event_generation'],
        maximum_budget_norm=max((e['maximum_budget_norm'] for e in evidence if e['maximum_budget_norm'] is not None),default=None),
        independent_numerical_error_estimate=None,physiological_validation=False,
        omitted_reference_core_populations=[p for p in POPULATIONS if len(core_rows(reference)[p]) and not len(common[0][p])],
        stochastic_sample_sequences_used_as_spatial_error=False)


def trace(case):
    graph=case['graph'];p=graph.target_population;parent=graph.target_parent_row
    result=[]
    for time,(frame,full) in sorted(case['frames'].items()):
        maps=identity_rows(case,frame,full)
        row=int(locate(maps[0][p],np.asarray([parent]))[0])
        rr=rates(frame['states'],p,case['seed']['inner_parameters'],[row],[graph.target_local_row])
        result.append(dict(case=case['name'],time_seconds=time,side=frame['side'],population=p,
            parent_group_row=int(parent),voltage_V=float(frame['states'][p,'RGC_spike_history','V'][row]),
            gE_S=float(frame['ports'][p,'gE'][row]),gI_S=float(frame['ports'][p,'gI'][row]),
            mean_history_hazard_Hz=float(rr['mean_history_hazard_Hz'][0]),
            expected_binary_count_rate_Hz=float(rr['expected_binary_count_rate_Hz'][0])))
    return result


def worker_resources(case):
    receipts=[]
    for path in case['directory'].parent.rglob('worker_receipt.json'):
        jobpath=path.parent/'job.json'
        if not jobpath.is_file():continue
        job=json.loads(jobpath.read_text(encoding='utf8'))
        if Path(job.get('run_directory','')).resolve()!=case['directory'].resolve():continue
        raw=json.loads(path.read_text(encoding='utf8'))
        peak=raw.get('resource_peak',{})
        observations=raw.get('resource_observations',[])
        rss=[peak.get('process_peak_rss_bytes'),peak.get('peak_rss_bytes'),peak.get('rss_bytes')]
        rss += [v.get('process_peak_rss_bytes',v.get('rss_bytes')) for v in observations]
        gpu=[v.get('gpu',{}).get('used_bytes') for v in observations]
        receipts.append(dict(source=binding(path),worker_status=raw.get('status'),wall_seconds=raw.get('wall_seconds'),
            process_peak_rss_bytes=max((v for v in rss if v is not None),default=None),
            whole_card_peak_observed_used_bytes=max((v for v in gpu if v is not None),default=None),
            timings=raw.get('timings',raw.get('timing'))))
    return dict(workers=receipts,
        process_peak_rss_bytes=max((v['process_peak_rss_bytes'] for v in receipts if v['process_peak_rss_bytes'] is not None),default=None),
        whole_card_peak_observed_used_bytes=max((v['whole_card_peak_observed_used_bytes'] for v in receipts if v['whole_card_peak_observed_used_bytes'] is not None),default=None),
        model_gpu_allocation_bytes=0,model_execution='CPU FP64; whole-card usage includes external desktop activity')


def performance(case,reference=None):
    g=case['graph'];r=case['result'];m=r['neural_metrics']
    wall=float(m['native_wall_seconds'])+float(m['inner_wall_seconds'])
    seconds=float(r['simulated_seconds'])
    counts=dict(group_count=sum(p.group_count for p in g.populations.values()),
        logical_state_count=sum(b.count for b in (*g.state_blocks,*g.probability_blocks,*g.auxiliary_z1_blocks)),
        cone_groups=g.populations['cones'].group_count,HC_groups=g.populations['HC'].group_count,
        population_groups={p:v.group_count for p,v in g.populations.items()},
        contact_count=sum(len(v) for v in g.parent_contact_rows.values()))
    speed=None
    if reference is not None:
        rm=reference['result']['neural_metrics']
        speed=(float(rm['native_wall_seconds'])+float(rm['inner_wall_seconds']))/wall
    return dict(**counts,simulated_seconds=seconds,continuous_advance_seconds=wall,
        native_advance_seconds=float(m['native_wall_seconds']),inner_advance_seconds=float(m['inner_wall_seconds']),
        solver_build_seconds=m.get('build_seconds'),snapshot_seconds=m.get('snapshot_wall_seconds'),
        whole_run_seconds=r['whole_worker_wall_seconds'],compute_seconds_per_biological_second=wall/seconds,
        continuous_advance_speedup_vs_full_reference=speed,
        twofold_performance_goal_passed=None if speed is None else speed>=2.,
        time_scope='Nonoverlapping native_wall_seconds + inner_wall_seconds; includes original solver calls and guard work inside these scopes. Does not include loading/build/recording/checkpoint/snapshot.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,required=True)
    args=parser.parse_args()
    stage=args.stage.resolve()
    if not stage.is_relative_to(ROOT):raise ContractError('Explicit stage inside repository required')
    planpath=next((p for p in (stage/'acceptance_plan.json',stage.parent/'acceptance_plan.json') if p.is_file()),None)
    if planpath is None:raise ContractError('Predeclared acceptance_plan.json is required')
    plan=json.loads(planpath.read_text(encoding='utf8'))
    budget=plan.get('engineering_error_budget')
    if budget!=DEFAULT_BUDGET:raise ContractError('Exact predeclared v2 engineering error budget required')
    ref=load_case(stage,'reference')
    candidates=[]
    incomplete=[]
    for path in sorted(stage.glob('graph_halo_*')):
        name=path.name[6:]
        if (stage/('run_'+name)/'result.json').is_file():candidates.append(load_case(stage,name))
        else:incomplete.append(name)
    if not candidates:raise ContractError('No completed bounded candidate available')
    results=[compare_cases(ref,case,budget) for case in candidates]
    traces=[row for case in [ref,*candidates] for row in trace(case)]
    output=stage/'comparison.json';csvpath=stage/'target_trace.csv'
    if output.exists() or csvpath.exists():raise ContractError('Comparison output must be new')
    with csvpath.open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(traces[0]));writer.writeheader();writer.writerows(traces)
    comparison=dict(schema='local-retinal-patch-v2-saved-comparison-1',model_advanced_by_analysis=False,
        acceptance_plan=binding(planpath),engineering_error_budget=budget,
        scope='Saved right-side ordinary recording points and final endpoint, shared original core identities. No arbitrary video or physiological extrapolation.',
        reference=dict(name=ref['name'],performance=performance(ref),resources=worker_resources(ref),absolute_monitor=ref['result']['absolute_monitor'],
            background_qualified=ref['result'].get('qualified_background',False)),
        bounded={case['name']:dict(comparison=result,performance=performance(case,ref),resources=worker_resources(case),
            absolute_monitor=case['result']['absolute_monitor'],background_qualified=case['result'].get('qualified_background',False),
            checkpoint_restore=case['result'].get('restore')) for case,result in zip(candidates,results)},
        incomplete_candidates=incomplete,
        any_spatial_candidate_passed=any(r['passed'] for r in results),
        any_joint_spatial_and_twofold_speed_candidate_passed=any(r['passed'] and performance(c,ref)['twofold_performance_goal_passed'] for c,r in zip(candidates,results)),
        physiological_validation=False,independent_time_discretization_error=None,
        source_bindings=[b for c in [ref,*candidates] for b in c['sources']],target_trace=binding(csvpath),
        expected_rate_definition='lambda_s=alpha*logaddexp(0,(V+STATES@kernel-mu)/beta); hazard=sum(pi*lambda_s); expected binary count Hz=sum(pi*(-expm1(-.001*lambda_s)))/.001',
        stochastic_note='Sampled spike-history/RNG streams are not used to define spatial error. The deterministic eight-state expected-history law is compared.')
    with output.open('x',encoding='utf8') as stream:
        json.dump(comparison,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(output),spatial_pass=comparison['any_spatial_candidate_passed'],
        joint_spatial_performance_pass=comparison['any_joint_spatial_and_twofold_speed_candidate_passed'],
        candidates={c['name']:dict(maximum_budget_norm=r['maximum_budget_norm'],
            speedup=performance(c,ref)['continuous_advance_speedup_vs_full_reference']) for c,r in zip(candidates,results)})))


if __name__=='__main__':main()
