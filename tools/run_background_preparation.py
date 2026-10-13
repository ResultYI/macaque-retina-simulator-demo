"""One authorized same-ROI background preparation with a bounded wall budget.

Uses the existing GUI configuration and SimulationSession preparation/criterion
path. Sources, numerical thresholds, solver and physiological parameters stay
frozen. A stimulus is run only after the existing numerical criterion passes.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,sys,time
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_platform.config import RunConfig
from retina_platform.backend import SimulationSession


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall-budget-seconds',type=float,default=7200.)
    parser.add_argument('--prior-run',type=Path,default=ROOT/'reports/central_build_acceptance/20261011T163543160525Z/run/20261011T163543490912Z_d5256453')
    args=parser.parse_args()
    if args.wall_budget_seconds<=0:parser.error('Positive wall budget required')
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    config=RunConfig.from_dict(json.loads((args.prior_run/'config.json').read_text(encoding='utf-8')))
    config.warmup_seconds=0.
    config.background_initialization_mode='bounded_convergence'
    config.background_max_sim_seconds=10.
    config.background_max_wall_seconds=args.wall_budget_seconds
    config.background_require_converged=True
    config.validate()
    out=ROOT/'reports/background_preparation_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    profile=ROOT/'configurations/retina_contact_geometry_effective_v1.json'
    request={'schema_version':'1.0','authorization':'Parent authorized existing full background closure, approximately two-hour engineering budget',
        'configuration':config.to_dict(),'initialization':'fresh runner.initialize from declared uniform background; no stimulus checkpoint used',
        'prior_short_run':str(args.prior_run),'prior_measured_integration_seconds':.02,'prior_integration_wall_seconds':81.69726859848015,
        'minimum_background_duration_seconds':1.25,'estimated_minimum_integration_wall_seconds':1.25/.02*81.69726859848015,
        'estimate_limit':'Monitoring, geometry, checkpoints and exports add wall time; the two-hour cap is not a physiological threshold',
        'estimated_compile_MiB':309.4,'available_physical_memory_GiB_before_run':16.95,
        'engineering_wall_budget_seconds':args.wall_budget_seconds,'numerical_policy':'configurations/bounded_uniform_background_v1.json unchanged',
        'execution_profile_sha256':hashlib.sha256(profile.read_bytes()).hexdigest(),
        'physiology_validation':'NOT_RUN','stimulus_condition':'only after existing CONVERGED criterion'}
    (out/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    session=SimulationSession(config,ROOT,out/'run').start()
    print(json.dumps({'phase':'REQUESTED','job_directory':str(out),'estimated_minimum_integration_wall_seconds':request['estimated_minimum_integration_wall_seconds']}),flush=True)
    last_print=0.;last_status=None
    while session.is_alive():
        if session.status!=last_status or time.monotonic()-last_print>=55.:
            print(json.dumps({'phase':session.status,'run_directory':str(session.path),
                'absolute_time_seconds':session._sim_time,'stimulus_windows':session._windows}),flush=True)
            last_status=session.status;last_print=time.monotonic()
        session._thread.join(.2)
    run=session.path
    if run is None:raise RuntimeError('Session did not create a run directory')
    result={'execution_status':session.status,'run_directory':str(run),'request_file':str(out/'request.json'),
        'stimulus_windows':session._windows,'physiology_validation':'NOT_RUN'}
    for name in ('summary.json','background_initialization.json'):
        if (run/name).exists():
            saved=json.loads((run/name).read_text(encoding='utf-8'))
            if name=='summary.json':result['summary']=saved
            else:
                result['background']={k:saved[k] for k in ('status','converged','budget_exit','elapsed_seconds','wall_seconds','observations','dominant','timing_requirements','uncovered_states')}
                result['per_state_residual_file']=str(run/name)
    if (run/'error.txt').exists():result['error']=(run/'error.txt').read_text(encoding='utf-8')
    assert hashlib.sha256(profile.read_bytes()).hexdigest()==request['execution_profile_sha256']
    (out/'outcome.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if session.status=='error':raise RuntimeError(result.get('error','Background session failed'))


if __name__=='__main__':main()
