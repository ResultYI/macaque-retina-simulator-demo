"""One approved transition check; no default switch or geometry substitution."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,subprocess,sys,time,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.disk_graph_storage import file_sha,save_json

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--configuration',default='configurations/atlas_human_borrowed_disk3mm_floor_v1.json')
    parser.add_argument('--resource-profile',default='')
    parser.add_argument('--resume-background',default='');args=parser.parse_args()
    if args.attempt<1:raise ValueError('Positive explicit attempt identifier required')
    report=ROOT/'reports/human_disk3mm_midget_floor_20261017';output=ROOT/('reports/human_disk3mm_floor_transition_v'+str(args.attempt))
    result_path=report/('transition_attempt_'+str(args.attempt)+'.json')
    if output.exists() or result_path.exists():raise FileExistsError('Prior attempt retained; inspect before an authorized engineering repair rerun')
    cfg=args.configuration;media='reports/cone_master_migration_20261016/gui_final/controlled_image.png'
    if not (ROOT/cfg).resolve().is_relative_to(ROOT):raise ValueError('Configuration outside project')
    paths=[p for folder in ('retina_sim','retina_platform') for p in (ROOT/folder).glob('*.py')]
    paths += [ROOT/'tools'/name for name in ('run_human_transition.py','check_disk3mm.py','disk3mm_response_check.py',
        'export_disk_mixed_batch.m','export_disk_mixed_reference.m','export_mixed_reference.m')]
    paths += [ROOT/cfg,ROOT/media,ROOT/'configurations/iset_lcd_apple_mw_v1.json',ROOT/'configurations/bounded_uniform_background_v1.json']
    if args.resource_profile:paths.append(ROOT/args.resource_profile)
    result={'status':'RUNNING_ACTUAL_TRANSITION_CHECK','started_UTC':datetime.now(timezone.utc).isoformat(),
        'authorization':'User2026-10-17T14:57Z approved floor minimum and existing transition real input/background/response;15:01Z approved continued reversible mainline work',
        'configuration':cfg,'output':output.relative_to(ROOT).as_posix(),'region_RF_mm':[.96,1.04,-.04,.04],
        'resource_profile':args.resource_profile or None,'resume_background':None if not args.resume_background else {
            'path':args.resume_background,'sha256':file_sha(ROOT/args.resume_background)},
        'execution_code_and_inputs_sha256':{p.relative_to(ROOT).as_posix():file_sha(p) for p in paths},
        'central_HC_unresolved':True,'whole_domain_default_adopted':False,'physiological_validation':False,
        'midget_continuous_target_average_satisfied':'NOT_ESTABLISHED'}
    save_json(result_path,result);start=time.monotonic()
    try:
        with (report/('transition_'+str(args.attempt)+'.stdout.log')).open('xb') as out,(report/('transition_'+str(args.attempt)+'.stderr.log')).open('xb') as err:
            command=[sys.executable,'-B','-X','utf8',str(ROOT/'tools/check_disk3mm.py'),
                '--region','transition','--circuit-configuration',cfg,'--capture-transient','--check-pathway-response','--figure',
                '--media',str(ROOT/media),'--output',str(output)]
            if args.resource_profile:command+=['--resource-profile',args.resource_profile]
            if args.resume_background:command+=['--resume-background',args.resume_background]
            environment=dict(os.environ)
            if args.resource_profile:
                settings=json.loads((ROOT/args.resource_profile).read_bytes()).get('process_environment',{})
                if settings and settings!={'OPENBLAS_NUM_THREADS':'1'}:raise ValueError('Unsupported execution environment override')
                environment.update(settings)
            completed=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,env=environment)
        result['process_returncode']=completed.returncode
        delivery=output/'delivery.json'
        if delivery.exists():result['actual_delivery']=json.loads(delivery.read_bytes())
        status=result.get('actual_delivery',{}).get('regions',{}).get('transition',{}).get('status')
        result['status']='TRANSITION_RESPONSE_PASS_NUMERICALLY_PREPARED' if status=='ISET_NEURAL_MATCHED_RESPONSE_PASS_NUMERICALLY_PREPARED' else 'BLOCKED'
        if result['status']=='BLOCKED':result['error']='Actual checker did not establish prepared transition response; inspect original region report/logs'
    except Exception as error:
        result.update(status='BLOCKED',error=type(error).__name__+': '+str(error));raise
    finally:
        changed=[relative for relative,expected in result['execution_code_and_inputs_sha256'].items()
                 if file_sha(ROOT/relative)!=expected]
        result['execution_source_changed_during_run']=changed
        if changed:result.update(status='BLOCKED',error='Execution source changed during actual run; no mixed-version acceptance')
        result['wall_seconds']=time.monotonic()-start;result['finished_UTC']=datetime.now(timezone.utc).isoformat()
        save_json(result_path,result);print(result['status'],result['wall_seconds'],flush=True)

if __name__=='__main__':main()
