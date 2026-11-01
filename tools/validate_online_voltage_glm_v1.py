"""Existing-cache engineering equivalence only. No retinal solve, fit or new score."""
from dataclasses import asdict
from pathlib import Path
import hashlib,json,time
import numpy as np
from retina_sim import stimulus_baselines_v1 as reference
from retina_sim.temporal_readout_v1 import stable_terms
from retina_sim.online_voltage_glm_v1 import OnlineVoltageGLM,MembraneParameters,membrane_outputs,NATIVE_DT

ROOT=Path(r'D:\macaque retinal simulator')
SOURCE=ROOT/'reports/WP1_LATIMER_CELL2_4_V1'
RTOL,ATOL=1e-14,1e-15


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def cached(group):
    folder=SOURCE/'cache'/f'g{group}';manifest=read(folder/'manifest.json')
    if not manifest['physical_gate_passed'] or not manifest['complete_research_restore'] or sha(folder/'endogenous.npz')!=manifest['arrays_sha256']:
        raise ValueError('Existing cache identity/gates differ')
    with np.load(folder/'endogenous.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in ('V','gE','gI','time_source_s')}
    if not np.array_equal(data['time_source_s'],np.arange(71001)*NATIVE_DT):raise ValueError('Native grid differs')
    return data,manifest,sha(folder/'manifest.json')


def run_packet(engine,data,start,end,counts=None):
    obs=None if counts is None else {b:counts[:,b] for b in range(engine.completed_bins,(end-1)//10)}
    return engine.advance(data['V'][start:end],data['gE'][start:end],data['gI'][start:end],
        start_s=start*NATIVE_DT,completed_spikes=obs)


def conditional_check(cell,group,model,physical,counts,data,source_id,guard):
    engine=OnlineVoltageGLM(model,physical,mode='conditional',paths=len(counts),seed=0,source_id=source_id)
    records=[];cuts=[0,*range(1000,71000,1000),71000]
    for start,end in zip(cuts,cuts[1:]):guard();records+=run_packet(engine,data,start,end,counts)
    engine.close_final_bin(end_s=71000*NATIVE_DT,completed_spike=counts[:,-1])
    features=reference.stimulus_features(data['V'][:-1])
    case=dict(counts=counts,history_features=reference.history_features(counts),stimulus_features=features)
    theta=np.asarray(model['theta']);mean=np.asarray(model['mean']);scale=np.asarray(model['scale'])
    original_z=theta[0]+((reference.unscaled_design(case,'GLM')-mean)/scale)@theta[1:]
    original_q=-np.expm1(-.09*np.logaddexp(0.,original_z))
    actual_q=np.stack([r.probability for r in records],axis=1);actual_z=np.stack([r.D+r.H for r in records],axis=1)
    np.testing.assert_allclose(actual_q,original_q,rtol=RTOL,atol=ATOL)
    actual_loss,_=stable_terms(actual_z,90.,counts);original_loss=reference.conditional(case,model)
    return dict(cell=cell,group=group,bins=7100,repeats=len(counts),full_source_seconds=7.1,
        max_probability_error=float(np.max(abs(actual_q-original_q))),max_linear_predictor_error=float(np.max(abs(actual_z-original_z))),
        max_NLL_error=float(np.max(abs(actual_loss-original_loss))),same_original_probability_tolerance=True,passed=True)


def reference_ring(signal,kernel,paths,seed):
    history=np.zeros((paths,200));weights=np.asarray([kernel[(p-1-np.arange(200))%200] for p in range(200)])
    rng=np.random.Generator(np.random.PCG64(seed))
    for n,drive in enumerate(signal):
        q=-np.expm1(-.09*np.logaddexp(0.,drive+history@weights[n%200]))
        bits=rng.random(paths)<q;history[:,n%200]=bits
        yield q,bits,rng.bit_generator.state,history


def free_check(cell,model,physical,data,source_id,out,guard):
    seed=2026103100+cell*100+1;paths=4096
    engine=OnlineVoltageGLM(model,physical,mode='free',paths=paths,seed=seed,source_id=source_id)
    features=reference.stimulus_features(data['V'][:-1]);theta=np.asarray(model['theta']);mean=np.asarray(model['mean']);scale=np.asarray(model['scale'])
    signal=theta[0]+((features-mean[:16])/scale[:16])@theta[1:17]-(mean[16:]/scale[16:])@theta[17:]
    kernel=reference.history_basis()@(theta[17:]/scale[16:]);ring=reference_ring(signal,kernel,paths,seed)
    cuts=sorted({0,31417,71000,*range(1000,71000,1000)});resumed=None;rows=[];batch_curves=[]
    max_q=0.;outcome_hash=hashlib.sha256();ring_hash=hashlib.sha256();last_rng=None;last_history=None
    for start,end in zip(cuts,cuts[1:]):
        guard();records=run_packet(engine,data,start,end)
        if resumed is not None:
            other=run_packet(resumed,data,start,end)
            if len(other)!=len(records):raise ValueError('Restored bin count differs')
            for a,b in zip(records,other):
                np.testing.assert_array_equal(a.probability,b.probability);np.testing.assert_array_equal(a.sampled_bin_outcome,b.sampled_bin_outcome)
        for r in records:
            q,bits,last_rng,last_history=next(ring)
            max_q=max(max_q,float(np.max(abs(r.probability-q))))
            np.testing.assert_allclose(r.probability,q,rtol=RTOL,atol=ATOL)
            np.testing.assert_array_equal(r.sampled_bin_outcome,bits)
            outcome_hash.update(r.sampled_bin_outcome.tobytes());ring_hash.update(bits.tobytes())
            batch_curves.append(r.probability.reshape(16,-1).mean(axis=1))
            rows.append([r.start_s,r.D,r.H[0],r.lambda_Hz[0],r.probability[0],int(r.sampled_bin_outcome[0])])
        if end==31417:
            state=engine.checkpoint();write(out/f'Cell{cell}_MID_CHECKPOINT.json',state)
            resumed=OnlineVoltageGLM.from_checkpoint(model,physical,read(out/f'Cell{cell}_MID_CHECKPOINT.json'),source_id=source_id)
            write(out/f'Cell{cell}_MID_STATE.json',engine.observable_state())
    engine.close_final_bin(end_s=71000*NATIVE_DT);resumed.close_final_bin(end_s=71000*NATIVE_DT)
    if engine.checkpoint()!=resumed.checkpoint():raise ValueError('Whole restored endpoint differs')
    if engine.rng.bit_generator.state!=last_rng:raise ValueError('RNG stream/order differs')
    np.testing.assert_array_equal(engine.history,last_history)
    guard();original_mean,original_batches=reference.predict(dict(stimulus_features=features),model,seed=seed,guard=guard)
    actual_batches=np.asarray(batch_curves).T;actual_mean=actual_batches.mean(axis=0)
    np.testing.assert_allclose(actual_batches,original_batches,rtol=RTOL,atol=ATOL)
    np.testing.assert_allclose(actual_mean,original_mean,rtol=RTOL,atol=ATOL)
    rows=np.asarray(rows)
    np.savez_compressed(out/f'Cell{cell}_READABLE_TRACE.npz',time_bin_start_s=rows[:,0],D_dimensionless=rows[:,1],H_path0_dimensionless=rows[:,2],
        lambda_path0_Hz=rows[:,3],probability_path0=rows[:,4],sampled_bin_path0=rows[:,5].astype(np.uint8),
        online_free_probability=actual_mean,original_free_probability=original_mean,online_batch_probability=actual_batches)
    write(out/f'Cell{cell}_END_STATE.json',engine.observable_state());write(out/f'Cell{cell}_END_CHECKPOINT.json',engine.checkpoint())
    return dict(cell=cell,group=1,paths=paths,batches=16,bins=7100,seed=seed,max_per_path_probability_error=max_q,
        max_batch_probability_error=float(np.max(abs(actual_batches-original_batches))),max_mean_probability_error=float(np.max(abs(actual_mean-original_mean))),
        all_sampled_bins_bitwise_equal=True,sampled_bin_sha256=outcome_hash.hexdigest(),independent_reference_bin_sha256=ring_hash.hexdigest(),
        exact_final_PCG_state=True,exact_completed_history=True,checkpoint_native_samples=31417,checkpoint_native_phase=7,
        pending_bin_preserved=True,restored_tail_probabilities_and_bins_bitwise_equal=True,full_endpoint_checkpoint_equal=True,passed=True)


def run(out,guard,phase):
    began=time.perf_counter();protocol=read(SOURCE/'PROTOCOL_FROZEN.json');dictionary=lambda value:dict(value['dictionary'])
    seed_folder=ROOT/'reports/CELL6_RAW_REBUILD_CONDITIONAL_READOUT_V1/halo_model_v2/seed';seed=read(seed_folder/'manifest.json')
    params=dictionary(dictionary(dictionary(seed['payload'])['inner_parameters'])['populations']);p=dictionary(params['parasol_RGC_ON'])
    identity=read(SOURCE/'targets/Latimer_Cell_2/SOURCE_IDENTITY.json')
    physical=MembraneParameters(p['C']*2,p['gL']*2,p['EL'],p['EE'],p['EI'],identity['target'],2)
    models={c:read(SOURCE/'targets'/f'Latimer_Cell_{c}'/'MODELS_FROZEN.json') for c in (2,3,4)}
    pins={str((SOURCE/'targets'/f'Latimer_Cell_{c}'/'MODELS_FROZEN.json').relative_to(ROOT)):sha(SOURCE/'targets'/f'Latimer_Cell_{c}'/'MODELS_FROZEN.json') for c in models}
    pins[str((ROOT/'retina_sim/stimulus_baselines_v1.py').relative_to(ROOT))]=sha(ROOT/'retina_sim/stimulus_baselines_v1.py')
    write(out/'PROTOCOL_FROZEN.json',dict(schema='online-equivalent-voltage-GLM-v1',source=str(SOURCE),models_and_code=pins,
        conditional_scope='All37 existing cell/stimulus pairs, all7100 bins, no new score or fit',free_scope='Predeclared shared trainingg1 only, all three models4096 paths/16 batches',
        probability_gate=dict(rtol=RTOL,atol=ATOL,source='Existing tests/test_stimulus_baselines_v1.py generated-history equivalence'),
        checkpoint_native_index=31417,history_commit='At bin close; current-bin observation never predicts that bin',
        stimulus_FIR='Original16 branches0..499.9ms, constant V0 before acquisition',history='Original13 bases1..200ms; synthesized kernel only as original free predictor',
        membrane=asdict(physical),membrane_solver_called=False,gA_S=0.,adaptation_enabled=False,reset_enabled=False,hard_refractory=False,
        statistical_D_H_units='dimensionless predictor, not voltage or physical threshold',
        source_exact_eccentricity_deg=None,source_population_scope='Peripheral source; original study typical~30deg, individual mapping not recovered',
        actual_model_proxy_coordinate_deg=identity['proxy_coordinate_deg'],spatial_registration_completed=False,
        physiological_validation=False,FIT_READY=False,background_qualified=False,production_adopted=False))
    conditional=[];free=[]
    for cell,frozen in models.items():
        target=SOURCE/'targets'/f'Latimer_Cell_{cell}';c=protocol['cells'][str(cell)]
        for name,h in frozen['source_pins'].items():
            if sha(ROOT/name)!=h:raise ValueError('Frozen training source changed: '+name)
        with np.load(target/'SOURCE_COUNTS.npz',allow_pickle=False) as z:counts={g:z[f'g{g}_counts'].copy() for g in c['training_groups']+c['development_groups']}
        for group in sorted(counts):
            phase('CONDITIONAL_EQUIVALENCE',cell=cell,group=group,completed_pairs=len(conditional))
            data,manifest,cache_sha=cached(group)
            if manifest['target']!=physical.group_id or manifest['members']!=2:raise ValueError('Frozen target/group differs')
            source_id=pins[str((target/'MODELS_FROZEN.json').relative_to(ROOT))]+':'+cache_sha
            conditional.append(conditional_check(cell,group,frozen['models']['V_GLM'],physical,counts[group],data,source_id,guard))
            write(out/'CONDITIONAL_PROGRESS.json',conditional)
        phase('FREE_AND_CHECKPOINT_EQUIVALENCE',cell=cell)
        data,_,cache_sha=cached(1);source_id=pins[str((target/'MODELS_FROZEN.json').relative_to(ROOT))]+':'+cache_sha
        free.append(free_check(cell,frozen['models']['V_GLM'],physical,data,source_id,out,guard))
        write(out/'FREE_PROGRESS.json',free)
    for name,h in pins.items():
        if sha(ROOT/name)!=h:raise ValueError('Original frozen source overwritten')
    data,_,_=cached(1)
    np.savez_compressed(out/'PHYSICAL_g1.npz',time_source_s=data['time_source_s'],**membrane_outputs(physical,data['V'],data['gE'],data['gI']))
    result=dict(passed=len(conditional)==37 and len(free)==3,conditional=conditional,free=free,physical_parameters=asdict(physical),
        wall_seconds=time.perf_counter()-began,original_frozen_sources_unchanged=True,retinal_network_rerun=False,fit_run=False,
        all_flags_false=True,physiological_adaptation_or_reset_tested=False,spatial_registration_completed=False)
    write(out/'RESULT.json',result);phase('EQUIVALENCE_COMPLETE',pairs=len(conditional),free_models=len(free))
    return result
