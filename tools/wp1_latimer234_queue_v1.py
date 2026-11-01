"""Single owned sequential queue; fit as soon as each cell's training inputs exist."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback
ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.durable_research_runtime_v1 import atomic_json,process_identity
OUT=ROOT/'reports/WP1_LATIMER_CELL2_4_V1'
WORK=Path(r'C:\Users\win11-pc\Documents\Codex\2026-10-29\task\hc_revision')
SHELL=Path(r'C:\Users\win11-pc\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe')


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def complete(g):return (OUT/'cache'/f'g{g}'/'manifest.json').exists()


def main():
    plan=read(OUT/'QUEUE_PLAN.json');name=sys.argv[1] if len(sys.argv)>1 else 'queue'
    if Path(name).name!=name:raise ValueError('Local queue attempt name required')
    control=OUT/name;control.mkdir(exist_ok=False);(OUT/'capture').mkdir(exist_ok=True)
    atomic_json(control/'CONTROLLER.json',dict(identity=process_identity(os.getpid()),automatic_native_retry=False))
    started=time.perf_counter();active=None;phase='START';status='FAILED'
    def invoke(args,label,capture=None):
        nonlocal active,phase
        phase=label
        atomic_json(OUT/'QUEUE_PROGRESS.json',dict(phase=label,completed_cache_groups=[g for g in plan['groups'] if complete(g)],
            trained_cells=[int(c) for c in plan['cells'] if (OUT/'targets'/f'Latimer_Cell_{c}'/'MODELS_FROZEN.json').exists()],
            completed_cells=[int(c) for c in plan['cells'] if (OUT/'targets'/f'Latimer_Cell_{c}'/'COMPUTATION_COMPLETE.json').exists()],
            controller=process_identity(os.getpid()),wall_seconds=time.perf_counter()-started))
        with (control/(label+'.log')).open('x',encoding='utf8') as log:
            active=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            atomic_json(control/'ACTIVE.json',dict(phase=label,identity=process_identity(active.pid),command=args,capture=str(capture) if capture else None))
            code=active.wait()
        if code:raise RuntimeError(f'{label}: exit{code}; no automatic retry')
        if capture and read(capture/'TERMINAL.json')['status']!='EXITED_0':raise RuntimeError(f'{label}: native capture did not exit normally')
    def maybe_fit():
        for cell,c in plan['cells'].items():
            target=OUT/'targets'/f'Latimer_Cell_{cell}'
            for mode,groups,marker in [('train',c['training_groups'],'MODELS_FROZEN.json'),('evaluate',c['development_groups'],'COMPUTATION_COMPLETE.json')]:
                if mode=='evaluate' and not (target/'MODELS_FROZEN.json').exists():continue
                if (target/marker).exists() or not all(complete(g) for g in groups):continue
                label=f'Cell{cell}_{mode}';capture=OUT/'capture'/f'Cell{cell}_{name}_{mode}'
                invoke([str(SHELL),'-NoProfile','-File',str(WORK/'wp1_capture_latimer_fit_existing_python_v2.ps1'),'-Group',cell,'-OutputDirectory',str(capture)],label,capture)
                if mode=='evaluate':invoke([r'D:\anaconda\python.exe','-B',str(ROOT/'tools/wp1_latimer234_deliver_v1.py'),cell],f'Cell{cell}_deliver')
    try:
        maybe_fit()
        for g in plan['groups']:
            if (OUT/'REQUEST_STOP').exists():raise RuntimeError('Explicit STOP')
            if not complete(g):
                plan_path=OUT/f'RESUME_SEGMENTS_g{g}.json'
                capture=Path(read(plan_path)['parent_capture_directory']) if plan_path.exists() else OUT/'capture'/f'g{g}_process_handoffs_v1'
                if not capture.resolve().is_relative_to((OUT/'capture').resolve()):raise ValueError('Resume escaped owned capture directory')
                invoke([sys.executable,'-B',str(ROOT/'tools/wp1_latimer_segmented_cache_v1.py'),str(g),str(capture)],f'cache_g{g}')
            manifest=read(OUT/'cache'/f'g{g}'/'manifest.json')
            if not manifest['physical_gate_passed'] or not manifest['complete_research_restore']:raise RuntimeError('Required full cache gate failed')
            maybe_fit()
        if not all((OUT/'targets'/f'Latimer_Cell_{c}'/'COMPLETE.json').exists() for c in plan['cells']):raise RuntimeError('Some source-cell deliverables incomplete')
        status='COMPLETE'
        atomic_json(OUT/'COMPLETE.json',dict(status='CELL2_3_4_CONDITIONAL_PROXY_COMPLETE',completed_cells=[2,3,4],unique_neural_trajectories=len(plan['groups']),
            wall_seconds=time.perf_counter()-started,original_Cell6_test_accessed=False,FIT_READY=False,background_qualified=False,production_adopted=False))
    except BaseException as error:
        atomic_json(control/'FAILURE.json',dict(error=repr(error),phase=phase,traceback=traceback.format_exc(),automatic_retry=False));raise
    finally:atomic_json(control/'TERMINAL.json',dict(status=status,wall_seconds=time.perf_counter()-started,healthy_child_killed=False))


if __name__=='__main__':main()
