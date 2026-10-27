"""Opt-in, source-pinned sampling and selected recording execution.

No clock, causal transition, event/RNG, integration or checkpoint codec changes.
Only reviewed pure observations may be omitted when no consumer needs them.
"""
import ast,copy,hashlib,time
from pathlib import Path
import numpy as np
from .contracts import ContractError,Quantity,canonical_hash
from .compiled_kernel import IMPLEMENTATIONS,SUPPORTED
from .hybrid import HybridRunner
from .gpu_ordered_contract import NumericalOrderedCudaRunner
from .no_delay_publication import _Guard,_StripGuards,procedure_identity

_advance=None
_sample=None

def _check_sources(compiled):
    procedure_identity()
    for name,digest in IMPLEMENTATIONS.items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=digest:
            raise ContractError('Review pure observations after source change: '+name)
    if any(c.mechanism not in SUPPORTED or c.implementation_hash not in IMPLEMENTATIONS.values()
           for c in compiled.components.values()):
        raise ContractError('Selected observations require reviewed pure factories')
    if any(b.cyclic for b in compiled.output_order) or any(x.delayed_operator is not None for x in compiled.links):
        raise ContractError('Selected observation route does not cover closure/edge-delay graphs')

class SelectedCPUObservation:
    """Original NumPy/factory callbacks, original sparse order; call-local only."""
    def __init__(self,model):self.model=model;self.statistics=dict(calls=0,wall_seconds=0.,evaluated_ports=0)
    def evaluate(self,t,y,bundle,plan):
        tick=time.perf_counter();m=self.model;c=m.compiled;signals={};inputs={};contexts={}
        if y.dtype!=np.float64 or y.shape!=(sum(s.shape[0] for s in m.state_specs),) or not np.all(np.isfinite(y)):
            raise ContractError('Selected observation requires the entire finite FP64 state')
        wanted={(cid,name) for cid,comp in c.components.items() for name in comp.outputs
            if plan.ports is None or cid+'.'+name in plan.ports}
        externals={(e.target_component,e.target_port):bundle.channels[e.channel] for e in c.spec.external_inputs}
        def gather(cid,port):
            key=(cid,port);target=c.components[cid].inputs[port]
            if key not in inputs:
                if key in externals:q=externals[key]
                else:
                    links=m._incoming.get(key,())
                    if not links:raise ContractError('Unbound selected observation input')
                    values=np.zeros(len(target.cell_ids))
                    for link in links:
                        spec=link.specification;values+=link.matrix@ensure(spec.source_component,spec.source_port).values
                    q=Quantity(values,target.unit,target.meaning,target.evidence_tag)
                target.validate(q)
                if q.meaning in ('effective_synaptic_drive','effective_synaptic_activation','excitatory_conductance','inhibitory_conductance') and np.any(q.values<0):
                    raise ContractError('Negative selected physical/effective input')
                inputs[key]=q
            return inputs[key]
        def gap(cid):
            current=np.zeros(len(c.components[cid].outputs['voltage'].cell_ids))
            for left,right,ix,iy,g in m._gap_plans:
                if cid in (left,right):
                    value=g*(ensure(right,'voltage').values[iy]-ensure(left,'voltage').values[ix])
                    if cid==left:np.add.at(current,ix,value)
                    if cid==right:np.add.at(current,iy,-value)
            return Quantity(current,'A','electrical_gap_current',c.components[cid].evidence_tag)
        def ensure(cid,port):
            key=(cid,port)
            if key not in signals:
                comp=c.components[cid]
                if port=='gap_current' and hasattr(comp.native_model,'membrane_capacitance'):q=gap(cid)
                else:
                    incoming={name:gather(cid,name) for name in comp.output_dependencies[port]}
                    incoming.update(__sampling__=False,__closure_trial__=False)
                    # Native context computes global signals. State aliases do
                    # not use it, so do not compute unrelated release/current.
                    if cid in m._context_providers and port in ('q','release','Ios'):
                        if cid not in contexts:contexts[cid]=m._context_providers[cid](m.local_blocks(cid,y))
                        incoming['__evaluation_context__']=contexts[cid]
                    q=comp.observe(port,t,m.local_blocks(cid,y),incoming)
                comp.outputs[port].validate(q);signals[key]=q
            return signals[key]
        for block in c.output_order:
            key=block.nodes[0]
            if key in wanted:ensure(*key)
        result={key:signals[key] for block in c.output_order for key in block.nodes if key in wanted}
        self.statistics['calls']+=1;self.statistics['evaluated_ports']+=len(signals)
        self.statistics['wall_seconds']+=time.perf_counter()-tick
        return None,result

def _methods():
    global _advance,_sample
    if _advance is not None:return _advance,_sample
    tree=ast.parse(Path(__file__).with_name('hybrid.py').read_text(encoding='utf8'))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='HybridRunner')
    original=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='advance')
    guard=_Guard();advance=guard.visit(copy.deepcopy(original));hits=[]
    class Record(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if node.name=='record':
                for x in ast.walk(node):
                    if isinstance(x,ast.Call) and ast.unparse(x.func)=='self.model.evaluate':
                        hits.append(x);x.func=ast.parse('self._record_selected',mode='eval').body;x.args.append(ast.Name(id='plan',ctx=ast.Load()))
                return node
            return self.generic_visit(node)
    Record().visit(advance)
    if guard.count!=4 or len(hits)!=1:raise ContractError('Review selected recording after original advance changes')
    restored=copy.deepcopy(advance)
    for x in ast.walk(restored):
        if isinstance(x,ast.Call) and ast.unparse(x.func)=='self._record_selected':
            x.func=ast.parse('self.model.evaluate',mode='eval').body;x.args.pop()
    restored=_StripGuards().visit(restored)
    if ast.dump(restored)!=ast.dump(original):raise ContractError('Selected advance altered an unreviewed operation')
    original_sample=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_sample')
    sample=copy.deepcopy(original_sample);count=0
    for x in ast.walk(sample):
        if isinstance(x,ast.Assign) and isinstance(x.value,ast.Call) and ast.unparse(x.value.func)=='self.model.evaluate':
            x.value=ast.IfExp(test=ast.parse('self._needs_history_outputs(due)',mode='eval').body,
                body=x.value,orelse=ast.parse('({}, {})',mode='eval').body);count+=1
    if count!=1:raise ContractError('Review original sampling evaluation structure')
    for node in (advance,sample):ast.fix_missing_locations(node)
    namespace={};exec(compile(ast.Module(body=[advance,sample],type_ignores=[]),str(Path(__file__)),'exec'),HybridRunner.advance.__globals__,namespace)
    _advance,_sample=namespace['advance'],namespace['_sample'];return _advance,_sample

class ObservedCudaRunner(NumericalOrderedCudaRunner):
    @classmethod
    def from_runner(cls,reference,*,contract,record_backend='cpu_selected'):
        if record_backend not in ('cpu_selected','device_selected'):raise ContractError('Unknown selected recording backend')
        _check_sources(reference.compiled)
        base=NumericalOrderedCudaRunner.from_runner(reference,contract=contract)
        self=cls.__new__(cls);self.__dict__.update(base.__dict__);self.__dict__.pop('advance',None)
        self.model=copy.copy(base.model)
        identity=dict(procedure='selected_observation_execution_v1',record_backend=record_backend,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            implementation_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('runtime_observation.py','device_observation.py')},
            parent_model_hash=base.model.model_hash,factory_sources=dict(IMPLEMENTATIONS),
            sampling='Original causal callbacks; skip unused pure evaluate when no causal-history component is due',
            full_state_solver_clocks_events_RNG_checkpoint_codec='unchanged',default_enabled=False)
        self.model.model_hash=canonical_hash(identity)
        backend=dict(base.model.compute_metadata['integration_backend'],observation_execution=identity)
        self.model.compute_metadata=dict(base.model.compute_metadata,integration_backend=backend)
        self.observation_execution=identity
        self.observation_statistics=dict(causal_evaluations=0,skipped_unused_evaluations=0)
        if record_backend=='cpu_selected':self.record_observer=SelectedCPUObservation(self.model)
        else:
            from .device_observation import DeviceSelectedObservation
            self.record_observer=DeviceSelectedObservation(self.model)
        self._selected_advance,self._selected_sample=_methods()
        return self
    def _needs_history_outputs(self,due):
        needed=any(cid in due and c.time_contract.kind=='causal_history' for cid,c in self.compiled.components.items())
        self.observation_statistics['causal_evaluations' if needed else 'skipped_unused_evaluations']+=1
        return needed
    def _record_selected(self,t,y,bundle,plan):return self.record_observer.evaluate(t,y,bundle,plan)
    def _sample(self,state,stimulus):return self._selected_sample(self,state,stimulus)
    def advance(self,*args,**kw):return self._selected_advance(self,*args,**kw)
