"""Continue fixed evaluations of an existing frozen fit; never train again."""
import ast
from datetime import datetime,timezone
from pathlib import Path
import time

from tools import fit_cell6_raw_readout_v1 as fitted
from tools.durable_research_runtime_v1 import inspect_identity


def resume(guard=lambda:None,*,output_root):
    fitted.EXPERIMENT=Path(output_root);fitted.HERE=fitted.EXPERIMENT/'readout';fitted.GUARD=guard
    directory=fitted.HERE;frozen=fitted.read(directory/'FROZEN_IDENTITY.json')
    for name,key in [('MODEL.json','model_sha256'),('READOUT_FIT.json','fit_sha256')]:
        if fitted.sha(directory/name)!=frozen[key]:raise RuntimeError('Frozen parameters or normalization changed')
    if fitted.sha(Path(fitted.__file__))!=frozen['fit_implementation_sha256']:raise RuntimeError('Original frozen evaluator changed')
    model=fitted.read(directory/'MODEL.json');fit=fitted.read(directory/'READOUT_FIT.json')
    if not fit['converged'] or fit['temporal_fit_count']!=1:raise RuntimeError('Exactly one converged saved fit required')
    rows=[]
    for group in range(6):
        guard();dest=directory/f'g{group}_RESULT.json'
        if dest.exists():
            result=fitted.read(dest)
            if result['model_id']!=model['model_id'] or result['row']['group']!=group:raise RuntimeError('Saved fixed evaluation identity differs')
            rows.append(result['row']);continue
        if group<3:raise RuntimeError('Completed training evaluations must be retained without rerun')
        mapping=fitted.read(fitted.EXPERIMENT/'CACHE_CAPTURE_MAP.json')
        capture=fitted.EXPERIMENT/'capture'/mapping.get('g'+str(group),'g'+str(group))
        while not (capture/'TERMINAL.json').exists():
            guard();owner=fitted.read(fitted.EXPERIMENT/'TRAIN_FIRST_OWNER.json')
            observed=inspect_identity(dict(pid=owner['pid'],creation_100ns=owner['creation_100ns'],image=owner['image']))
            if observed['state']!='ALIVE':
                raise RuntimeError('Owning continuation unavailable; frozen fit retained: '+repr(observed))
            if (fitted.EXPERIMENT/'CONTINUATION_FAILURE.json').exists():raise RuntimeError('Cache continuation failed')
            fitted.write(directory/'WAITING_FOR_CACHE.json',dict(group=group,model_frozen=True,training_resumed=False,utc=datetime.now(timezone.utc).isoformat()))
            time.sleep(5)
        if fitted.read(capture/'TERMINAL.json')['status']!='EXITED_0' or fitted.read(capture/'RESOURCE.json')['status']!='PASS':
            raise RuntimeError('Fixed evaluation requires a clean complete cache stage')
        rows.append(fitted.evaluate(group,model,fit))
    # Execute the frozen CSV/decision block verbatim, without its training loop.
    if fitted.sha(Path(fitted.__file__))!=frozen['fit_implementation_sha256']:raise RuntimeError('Frozen finalization changed')
    module=ast.parse(Path(fitted.__file__).read_text(encoding='utf8'))
    main=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    block=next(n for n in main.body if isinstance(n,ast.Try)).body
    first=next(i for i,n in enumerate(block) if isinstance(n,ast.With))
    tail=ast.Module(body=block[first:],type_ignores=[])
    if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('train','minimize') for n in ast.walk(tail)):
        raise RuntimeError('Evaluation finalization must not invoke an optimizer')
    namespace=dict(vars(fitted),rows=rows,model=model,fit=fit)
    exec(compile(tail,str(fitted.__file__)+':frozen_finalization','exec'),namespace)
