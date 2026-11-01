"""Freeze response-blind source groups and independent cell contracts for Cell2-4."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import math
import sys
import numpy as np
from scipy.io import loadmat

ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from retina_sim.contracts import canonical_hash
from tools.durable_research_runtime_v1 import atomic_json
from tools.cell6_halo_context_v1 import BASE,MODEL,TARGET,protocol
from tools.cell6_raw_intake_v1 import element,header,selected_spikes

OUT=ROOT/'reports/WP1_LATIMER_CELL2_4_V1'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False);prep=OUT/'preparation';prep.mkdir()
    accel=ROOT/'reports/CELL6_COMPILED_INNER_V1'
    pair=read(accel/'paired_guard_v2/PAIR_RESULT.json')
    formal=read(accel/'formal_g1/cache/g1/manifest.json')
    terminal=read(accel/'formal_g1/capture/g1_process_handoffs_guard_v2/TERMINAL.json')
    extra=read(accel/'FORMAL_RESULT.json')
    supplement=read(accel/'FORMAL_ACCEPTANCE_SUPPLEMENT.json')
    if not supplement['passed']:raise RuntimeError('Original sampler witness gate remains unresolved')
    if not pair['passed'] or terminal['status']!='EXITED_0' or not formal['physical_gate_passed'] or not formal['complete_research_restore']:
        raise RuntimeError('Predeclared research execution gate not passed')
    if not all(x['passed'] for x in extra['target_full_grid_comparison']+extra['inner_endpoint_comparison']+extra['auxiliary_rate_original_budget']):
        raise RuntimeError('Formal numerical comparison failed')
    gate=dict(read(accel/'formal_g1/PAIR_GATE.json'),
        required_execution_scope='Predeclared accepted50ms all-state pair plus full finite/physical/history/restore cache gates',
        formal_source=str(accel/'formal_g1/cache/g1'),formal_cache_complete=True,
        event_timestamp_supplement_sha256=sha(accel/'FORMAL_ACCEPTANCE_SUPPLEMENT.json'),
        event_timestamp_supplement='Original GroupTimeComparator._draw witness qualification recovered for every changed timestamp',
        full_duration_sampler_equivalence_claimed=False,original_failed_extra_comparison_preserved=True)
    compiled=read(accel/'formal_g1/capture/g1_part14_guard_v2/COMPILED_EXECUTION.json')
    backend_ids={entry['execution_identity'] for entry in compiled['instances']}
    if len(backend_ids)!=1:raise RuntimeError('Accepted compiled backend registry differs')
    gate['compiled_rhs_identity']=next(iter(backend_ids))
    atomic_json(OUT/'PAIR_GATE.json',gate)
    metadata=read(ROOT/'data/source_data/latimer2019_figshare_v1/article_metadata.json')
    sealed=set(read(ROOT/'reports/modeling_scope_20261020/latimer_cell6_rgc_v1/contract.json')['partition']['heldout_stimulus_sha256'])
    original=read(BASE/'SOURCE_IDENTITY.json');programs={};cells={};raw_hashes={}
    for cell in (2,3,4):
        path=ROOT/f'data/source_data/latimer2019_figshare_v1/Cell_{cell}.mat'
        official=next(x for x in metadata['files'] if x['name']==path.name)
        if path.stat().st_size!=official['size'] or hashlib.md5(path.read_bytes()).hexdigest()!=official['computed_md5']:
            raise RuntimeError('Official raw identity changed')
        raw_hashes[str(cell)]=sha(path)
        values=loadmat(path,variable_names=['SpikeStm'])['SpikeStm'];groups={}
        if values.ndim!=2 or values.shape[1]!=71000:raise RuntimeError('Original source duration changed')
        for row,v in enumerate(values):
            h=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
            if h in sealed:raise RuntimeError('Sealed historical Cell6 stimulus intersected; no observations decoded')
            groups.setdefault(h,[]).append(row);programs.setdefault(h,v.copy())
        ordered=sorted(groups);ntrain=max(3,len(ordered)//2)
        cells[str(cell)]=dict(cell=cell,raw_path=str(path.relative_to(ROOT)),raw_sha256=raw_hashes[str(cell)],
            stimulus_rows=groups,training_hashes=ordered[:ntrain],development_hashes=ordered[ntrain:],
            source_eccentricity_deg=None,source_type='ON-parasol',target=TARGET,members=2,
            proxy_coordinate_deg=[4.738358015,.056615877],
            qualification='Independent source-cell readout using explicitly shared engineering proxy, not measured per-cell anatomy',
            split_rule='Response-blind sorted full stimulus SHA256; first floor(N/2),minimum3, training; remainder fixed development; all repeats stay together',
            source_rows=int(values.shape[0]),unique_stimuli=len(ordered))
    keys=sorted(programs);cases=[None];arrays={};p=protocol()
    if len(keys)!=17:raise RuntimeError('Authorized unique source registry differs')
    from tools.cell6_matched_input_v1 import dictionary,owned_array
    seed_manifest=read(MODEL/'seed/manifest.json')
    descriptor=dictionary(dictionary(seed_manifest['payload'])['input_reference'])['native_background_td']
    background=owned_array(MODEL/'seed',seed_manifest,descriptor)
    frames={f:max(0,math.floor((f/150.)/.0001)) for f in p['input_frame_indices']}
    input_programs={}
    for group,h in enumerate(keys,1):
        raw=programs[h];arrays[f'g{group}_stimulus']=raw
        input_programs[group]=canonical_hash({f:background*(1+.05*(float(raw[j])-float(raw[0]))) for f,j in frames.items()})
        cases.append(dict(group=group,stimulus_sha256=h,source_seconds=7.1,
            cells=[int(c) for c in cells if h in cells[c]['stimulus_rows']],
            full_forward_binding=dict(target=TARGET,model_seed_sha256=sha(MODEL/'seed/manifest.json'),
                protocol=canonical_hash(p),input_payload_sha256=input_programs[group],same_background_initial_state=True,source_eccentricity_unknown=True)))
    if len(set(input_programs.values()))!=17:raise RuntimeError('Full input program deduplication requires a revised source registry')
    for c in cells.values():
        c['training_groups']=[keys.index(h)+1 for h in c['training_hashes']]
        c['development_groups']=[keys.index(h)+1 for h in c['development_hashes']]
    code_names=['wp1_latimer234_prepare_v1.py','wp1_latimer234_fit_v1.py','wp1_latimer234_queue_v1.py',
        'wp1_latimer_compiled_cache_v1.py','wp1_latimer_cache_worker_v1.py','wp1_latimer_segmented_cache_v1.py',
        'wp1_latimer_metrics_v1.py','cell6_compiled_inner_v1.py','cell6_compiled_kernel_guard_v2.py','cell6_runtime_monitor_v3.py']
    contract=dict(schema='wp1-latimer234-frozen-v1',created_utc=datetime.now(timezone.utc).isoformat(),cells=cells,cases=cases,
        source_spike_observations_decoded_at_contract_freeze=False,groups_selected_without_responses=True,
        code_pins={str((ROOT/'tools'/name).relative_to(ROOT)):sha(ROOT/'tools'/name) for name in code_names},
        original_stimulus_GLM_code_sha256=sha(ROOT/'retina_sim/stimulus_baselines_v1.py'),
        raw_hashes=raw_hashes,input_mapping=original['input_mapping'],score_window_seconds=[1,7],
        source_sample_dt_seconds=.0001,source_index_origin='one-based inherited project interpretation',
        model_seed_sha256=sha(MODEL/'seed/manifest.json'),model_proxy_is_same_for_all_sources=True,
        new_unique_forward_count=len(keys),scientific_equations_or_parameters_changed=False,
        readout='Independent V_GLM30; S_GLM30 and LN17 paired baselines; original basis,loss,GRID,optimizer,4096 freeMC and strictly causal history',
        readout_definition=dict(input_basis_count=16,input_support_ms=[0,499.9],history_basis_count=13,history_support_ms=[1,200],
            probability='1-exp(-0.09*softplus(z))',alpha_Hz=90.,likelihood='Bernoulli1ms',
            ridge_grid=[1e-5,1e-4,1e-3,1e-2],regularization='L2 on standardized coefficients except bias',
            train_only_standardization=True,optimizer='L-BFGS-B',maxiter=500,maxfun=2000,maxls=40,ftol=1e-12,gtol=1e-7,
            starts=1,tie_rule='Largest ridge within1e-8 of best mean held-training-stimulus NLL',
            free_MC_paths=4096,free_MC_batches=16,seed_rule='2026103100+cell*100+global_input_group',
            bootstrap_draws=1000,bootstrap='Own actual repeat count and30 aligned200ms blocks with replacement; descriptive within stimulus'),
        selection='Same finite ridge GRID, leave-one-training-stimulus-out across all declared training groups, original tie rule; no development tuning',
        single_repeat='No repeated-trial reliability claim; descriptive time-block bootstrap only',
        original_Cell6_test_accessed=False,FIT_READY=False,background_qualified=False,production_adopted=False)
    np.savez_compressed(prep/'SOURCE_PREPARATION.npz',**arrays)
    atomic_json(prep/'SOURCE_IDENTITY.json',dict(contract,preparation_npz_sha256=sha(prep/'SOURCE_PREPARATION.npz')))
    atomic_json(OUT/'PROTOCOL_FROZEN.json',contract)
    # Generalize only the declared MAT cell-array length; keep original numeric-cell parsing.
    import inspect
    source=inspect.getsource(selected_spikes).replace('tuple(shape)!=(1,30)','(len(shape)!=2 or int(shape[0])!=1)').replace('for row in range(30):','for row in range(int(shape[1])):')
    namespace=dict(selected_spikes.__globals__);exec(compile(source,__file__+':variable_trial_count','exec'),namespace)
    decode=namespace['selected_spikes']
    for c in cells.values():
        path=ROOT/c['raw_path'];spikes=decode(path,set(range(c['source_rows'])))
        data={}
        for h,rows in c['stimulus_rows'].items():
            group=keys.index(h)+1;counts=np.stack([np.bincount((spikes[i]-1)//10,minlength=7100) for i in rows])
            if counts.shape!=(len(rows),7100) or not np.isin(counts,(0,1)).all():raise RuntimeError('Binary1ms source support differs; no clipping')
            data[f'g{group}_counts']=counts
        target=OUT/'targets'/f'Latimer_Cell_{c["cell"]}';target.mkdir(parents=True)
        np.savez_compressed(target/'SOURCE_COUNTS.npz',**data);atomic_json(target/'SOURCE_IDENTITY.json',c)
    priority=[]
    for c in cells.values():
        for g in c['training_groups']:
            if g not in priority:priority.append(g)
    priority += [g for g in range(1,len(cases)) if g not in priority]
    atomic_json(OUT/'QUEUE_PLAN.json',dict(groups=priority,unique_forward_count=len(keys),cells=cells,
        training_immediately_when_own_training_caches_ready=True,original_Cell6_test_accessed=False))
    print(json.dumps(dict(unique_forward_count=len(keys),priority=priority,
        cells={c:dict(training=v['training_groups'],development=v['development_groups']) for c,v in cells.items()})))


if __name__=='__main__':main()
