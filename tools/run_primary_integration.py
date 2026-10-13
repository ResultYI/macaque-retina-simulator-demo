from pathlib import Path
import sys,json,subprocess,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def main():
    started=datetime.now(timezone.utc);out=ROOT/'reports/primary_integration'/started.strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True,exist_ok=False)
    suites=['tests/test_core_architecture.py','tests/test_architecture_extensions.py','tests/test_primary_pathway.py']
    result=subprocess.run([sys.executable,'-B','-m','pytest',*suites,'-q','--tb=short','-p','no:cacheprovider'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    log=result.stdout+result.stderr;(out/'check.log').write_text(log,encoding='utf-8')
    summary=next((line for line in reversed(log.splitlines()) if ' passed' in line or ' failed' in line or ' error' in line),'See check.log')
    old=json.loads((ROOT/'reports/core_architecture_status.json').read_text(encoding='utf-8'))
    preserved={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in old['preserved_reference_files']}
    if preserved!=old['preserved_reference_files']:raise RuntimeError('Frozen source/reference files changed unexpectedly')
    status={'started_utc':started.isoformat(),'finished_utc':datetime.now(timezone.utc).isoformat(),'exit_code':result.returncode,'test_result_summary':summary,'test_suites':suites,'log_path':str((out/'check.log').relative_to(ROOT)),'implementation_modules':{f.stem:hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'retina_sim').glob('*.py')},'preserved_reference_files':preserved,'animal_fits_run':0,'independent_physiological_validation':'NOT_RUN','claim':'Numerics and executed architecture only; parameter calibration and independent animal validation separate'}
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    (ROOT/'reports/primary_pathway_status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':str(out/'status.json'),'summary':summary,'exit_code':result.returncode},ensure_ascii=False))
    return result.returncode
if __name__=='__main__':raise SystemExit(main())
