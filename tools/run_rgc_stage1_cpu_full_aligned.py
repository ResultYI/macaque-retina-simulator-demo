"""Opt-in selected-case all-layer observer; original CPU orchestration preserved."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS']})
OUT=ROOT/'reports/rgc_stage1_aligned_v1'
FILES=['retina_sim/rgc_stage1_aligned_recording.py','tools/run_rgc_stage1_cpu_aligned_case.py','tools/run_rgc_stage1_cpu_full_aligned.py']
CASE='cell6_g3'
CONTRACT_FILE='RECORDING_CONTRACT_r3.json'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare():
    from tools.fit_rgc_stage1 import verify
    from tools.validate_ac_vs_native_support import load
    from retina_sim.contracts import canonical_hash
    from retina_sim.rgc_stage1_aligned_recording import write
    spec=verify();a,seed,_=load(ROOT/spec['prepared_model']);case=next(c for c in spec['cases'] if c['case_id']==CASE)
    blocks=(*a.state_blocks,*a.probability_blocks,*a.auxiliary_z1_blocks);populations={p:dict(group_count=v.group_count,source_member_count=int(v.counts.sum()),coordinate_role=v.coordinate_role,dynamic_states_present=any(b.population==p for b in blocks)) for p,v in a.populations.items()}
    n=case['samples_100us']+1;scalars=sum(b.count for b in blocks);statebytes=scalars*8*n*2
    contract=dict(schema='cell6-aligned-observer-contract-1',product_schema='cell6-aligned-replay-1',case_id=CASE,
        selection='last originaltrainingcase, frozen beforefit; attach to its already requiredforward, not choose bestvalidation or newmovie',
        source_nodes=n,source_dt_s=.0001,rate_bins=case['samples_100us']//10,rate_bin_s=.001,source_model_origin_s=float(seed['time_seconds']),
        provenance=dict(case_id=CASE,scientific_contract_sha256=sha(ROOT/'reports/rgc_stage1_complete_v1/CONTRACT.json'),cache_identity=canonical_hash(dict(spec=spec,case=case)),
            stimulus_sha256=case['stimulus_sha256'],drive_sha256=case['drive_sha256'],source_seed_sha256=spec['source_seed_sha256'],target_global_id=spec['target']['global_id'],source_model_origin_s=float(seed['time_seconds']),
            registration_scope='conditional group representative, not anatomicalCell6 or physical qualification',artifact=a.artifact_sha256),
        state_fields=[dict(key=[b.population,b.mechanism,b.name],count=b.count,unit=b.unit,role=b.role,meaning=b.meaning) for b in blocks],
        populations=populations,RGC_populations=list(spec['recording']),canonical_RGC_recording=spec['recording'],
        primary_side_phase='exact original compactRGC observation: initialprecommit then precommit100us; actualleft/right saved',
        post_product='separate actual same100us post observations; never mix pre-RGC with post-upstream',
        sample_method='copy originalcapture snapshot(queue_nodeTrue) only; exact canonical calendar keys use original author_tick and recorded-index rule; exclude diagnostics and distinct near-grid record-only events; no new snapshot/RHS/integration/denseevaluation calls',
        rate_semantics='separate7100 native1ms bins; baseline expectedHz and sampledgroupcounts; fitted outputs use originalreadout replay only after truefit',
        buffer_limit_bytes=33554432,disk_remaining_reserve_bytes=4294967296,
        storage_estimate=dict(state_fields=len(blocks),state_scalars_per_phase=scalars,raw_state_both_phases_bytes=statebytes,
            derived_numeric_ports='actualfixedFP64schema frozen at firstsource snapshot; extra bytes calculated exactly then, no droppedstates',
            conservative_planning_reservation_bytes=68719476736,measured_copy_IO_overhead=False,buffer_values_cap_bytes=33554432),
        geometry_rule='real groupIDs/members/coordinate_role; no independentfinecell claim; inactivelegacyACs not fabricated',
        incomplete_policy='resume imports samecontract exact checkpointprefix with perfileSHA, then appends nextrow; oldattempt preserved. Missingprefix requires legitimate onecase replay; no interpolation',
        source_files={p:sha(ROOT/p) for p in FILES},UI_minimum_interface_sha256=read(OUT/CONTRACT_FILE)['UI_minimum_interface_sha256'] if (OUT/CONTRACT_FILE).exists() else sha(ROOT/'docs/cell6_replay/ALIGNED_INPUT_SCHEMA.json'),
        long_execution_started=False)
    OUT.mkdir(exist_ok=True);path=OUT/CONTRACT_FILE
    if path.exists() and read(path)!=contract:raise RuntimeError('Recording contract changed: preserve old version and register separately')
    if not path.exists():write(path,contract)
    write(OUT/'PREPARATION_RESULT_r3.json',dict(utc=datetime.now(timezone.utc).isoformat(),status='PREPARED_NO_FORWARD',contract_path=str(path),contract_sha256=sha(path),disk_free_bytes=shutil.disk_usage(OUT).free,
        product_discovery='reports/rgc_stage1_cpu_parallel_full_v1/workers/cell6_g3/aligned/attempt_*/manifest.json',schema='cell6-aligned-replay-1',old_scientific_pins_unchanged=True,
        normal_path_additional_7p1s_forwards=0,old_g0_g1_g2_fullalignment_requires_separate_legal_replay_if_requested=True,
        static_assembly_introspected=True,dynamic_arm_constructed=False,model_advanced=False))
    return contract

def install(contract):
    from types import SimpleNamespace
    from tools import run_rgc_stage1_cpu_full as original
    from retina_sim.rgc_stage1_aligned_recording import write
    popen=original.subprocess.Popen;coordinator=original.Coordinator
    def dispatch(args,*a,**kwargs):
        args=list(args)
        if '--job' in args and str(ROOT/'tools/run_rgc_stage1_cpu_full_case.py') in args:
            jobpath=Path(args[args.index('--job')+1]);job=read(jobpath)
            if job['case_id']==CASE:
                write(jobpath.parent/'aligned_record_spec.json',contract)
                args[args.index(str(ROOT/'tools/run_rgc_stage1_cpu_full_case.py'))]=str(ROOT/'tools/run_rgc_stage1_cpu_aligned_case.py')
        return popen(args,*a,**kwargs)
    class RecordingCoordinator(coordinator):
        def registry(self):
            result=super().registry();paths=sorted((original.OUT/'workers'/CASE/'aligned').glob('attempt_*/manifest.json'))
            write(OUT/'PRODUCTS.json',dict(schema='cell6-aligned-product-index-1',case_id=CASE,contract_path=str(OUT/CONTRACT_FILE),contract_sha256=sha(OUT/CONTRACT_FILE),
                complete_products=[dict(path=str(p),sha256=sha(p),status=read(p)['status']) for p in paths],
                attempt_statuses=[dict(path=str(p),status=read(p)) for p in sorted((original.OUT/'workers'/CASE/'aligned').glob('attempt_*/STATUS.json'))],
                if_missing='No fabricatedfallback. Need originalselectedcase capture or separately authorized sameidentityreplay.'))
            return result
    original.subprocess=SimpleNamespace(Popen=dispatch,STDOUT=subprocess.STDOUT,CREATE_NO_WINDOW=subprocess.CREATE_NO_WINDOW)
    original.Coordinator=RecordingCoordinator
    original.WRAPPERS=list(original.WRAPPERS)+FILES+['reports/rgc_stage1_aligned_v1/'+CONTRACT_FILE]
    return original

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--resume',action='store_true');args=p.parse_args();contract=prepare()
    if args.mode=='prepare':print(json.dumps(dict(status='PREPARED_NO_MODEL_ADVANCE',case_id=CASE,storage=contract['storage_estimate'])));return
    original=install(contract);sys.argv=[str(ROOT/'tools/run_rgc_stage1_cpu_full.py'),'run']+(['--resume'] if args.resume else [])
    original.main()
    from retina_sim.rgc_stage1_aligned_recording import write
    export_command=[sys.executable,'-B','-X','utf8','-m','retina_platform.replay_export','--project-root',str(ROOT)]
    exported=subprocess.run(export_command,cwd=ROOT,capture_output=True,text=True,encoding='utf8')
    write(OUT/'FIT_OUTPUT_EXPORT_RECEIPT.json',dict(command=export_command,exit_code=exported.returncode,stdout=exported.stdout,stderr=exported.stderr,
        original_fit_final=str(original.OUT/'FINAL_RESULT.json'),new_model_forward=False))
    if exported.returncode:raise RuntimeError('True fit finished; readout export failed, preservefit/cache and inspect exportreceipt')
    products=read(OUT/'PRODUCTS.json');complete=any(x['status']=='COMPLETE_ALIGNED' for x in products['complete_products'])
    write(OUT/'FINAL_ALIGNED_STATUS.json',dict(status='COMPLETE_ALIGNED' if complete else 'ALIGNMENT_INCOMPLETE_SCIENCE_RESULT_RETAINED',
        original_fit_final=str(original.OUT/'FINAL_RESULT.json'),actual_model_replayed_for_missing_data=False))
    if not complete:raise RuntimeError('Original fit stage finished; aligned recording incomplete: inspect separate productstatus, no fakefallback')

if __name__=='__main__':main()
