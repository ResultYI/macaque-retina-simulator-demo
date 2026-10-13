"""Aggregate A/B checks; preserve each actual attempt, including failures."""
from pathlib import Path
import ast,datetime,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
folder=ROOT/'reports'/stamp; folder.mkdir(parents=True)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',RETINA_NUMERIC_REPORT=str(folder/'numeric_metrics.json'))
commands=[['-B',str(ROOT/'tools/extract_packer_cone_density.py'),'--verify'],['-B','-m','pytest','-p','no:cacheprovider',str(ROOT/'tests/test_stage_a.py'),'-q'],['-B','-m','pytest','-p','no:cacheprovider',str(ROOT/'tests/test_stage_b.py'),'-q']]
commands.append(['-B','-m','pytest','-p','no:cacheprovider',str(ROOT/'tests/test_stage_c.py'),'-q'])
checks=[]
for number,args in enumerate(commands):
    command=[sys.executable]+args
    env['RETINA_NUMERIC_REPORT']=str(folder/f'numeric_metrics_{number}.json')
    result=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True)
    (folder/f'check_{number}.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    checks.append({'command':command,'exit_code':result.returncode,'log':str((folder/f'check_{number}.log').relative_to(ROOT)).replace('\\','/')})
    print(json.dumps(checks[-1])); print(result.stdout[-1200:])
syntax=[]
for path in list((ROOT/'retina_sim').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+list((ROOT/'tools').glob('*.py')):
    try: ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    except SyntaxError as error: syntax.append({'file':str(path),'error':str(error)})
metrics={}
for path in folder.glob('numeric_metrics_*.json'):
    metrics.update(json.loads(path.read_text())['metrics'])
report={'executed_at':stamp,'interpreter':sys.executable,'scope':'A/B engineering plus isolated source-specific linear reference','fixture_class':'synthetic_fixture','checks':checks,'syntax_errors':syntax,'lint_typecheck':'No lint/typecheck tool configured; AST parse and runtime contracts executed, no typechecker claim','numeric_metrics':metrics,'physiological_validation':'NOT_RUN','training':'NOT_RUN','author_array_reproduction':'BLOCKED_DATA_DECODE','full_physiology':'BLOCKED','attempt_directory':str(folder.relative_to(ROOT)).replace('\\','/')}
report['passed']=all(c['exit_code']==0 for c in checks) and not syntax
report['scope']='A/B engineering, isolated linear reference, and C peripheral LM nonlinear source-discrete current'
report['chen2024_original_matlab']=json.loads((ROOT/'reports/chen2024_matlab_execution.json').read_text(encoding='utf-8'))
inventory=ROOT/'reports/chen2024_data_inventory.json'
report['chen2024_source_data_reproduction']=json.loads(inventory.read_text(encoding='utf-8'))['source_data_reproduction'] if inventory.is_file() else 'NOT_RUN'
(folder/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'reports/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'aggregate_passed':report['passed'],'report':str(ROOT/'reports/verification.json'),'metrics':metrics}))
sys.exit(0 if report['passed'] else 1)
