"""Selected FP64 device readout from existing ordered expression lowering.

Record-only: no device value is fed to a transition or integration stage.
Static plan/graph caching only; all dynamic values are recomputed each record.
"""
import ast,hashlib,inspect,textwrap,time
from types import SimpleNamespace as NS
import numpy as np
from .contracts import ContractError,Quantity
from .compiled_kernel import Plan
from .gpu_device_inner import DeviceInnerPlan
from .gpu_ordered_joint import OrderedInnerPlan
from .gpu_device_variants import BatchedValidationInner,CudaGraphInner


class _ObservationPlan(Plan):
    native_fusion_minimum_cells=float('inf')
    def ensure(self,cid,port):
        key=(cid,port)
        if key not in self.outputs:self.outputs[key]=self.check(self.observe(cid,port))
        return self.outputs[key]
    def gather(self,cid,port):
        if (cid,port) not in self.inputs:
            for link in self.network._incoming.get((cid,port),()):
                s=link.specification;self.ensure(s.source_component,s.source_port)
        return super().gather(cid,port)


def _constructors():
    """Reuse original codegen, changing only its explicit output-size contract."""
    source=textwrap.dedent(inspect.getsource(DeviceInnerPlan.__init__));tree=ast.parse(source);hits=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Compare) and ast.unparse(node)=='expected != self.nstate':
            node.comparators[0]=ast.parse('reference.output_size',mode='eval').body;hits+=1
    if hits!=1:raise ContractError('Review device output export after original lowering changes')
    ast.fix_missing_locations(tree);space={};exec(compile(tree,'<observation output-size codegen>','exec'),DeviceInnerPlan.__init__.__globals__,space)
    base=space['__init__']
    tree=ast.parse(textwrap.dedent(inspect.getsource(OrderedInnerPlan.__init__)));hits=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value)=='super().__init__(reference)':
            node.value=ast.parse('_observation_device_init(self, reference)',mode='eval').body;hits+=1
    if hits!=1:raise ContractError('Review ordered observation constructor after original lowering changes')
    ast.fix_missing_locations(tree);space={};namespace=dict(OrderedInnerPlan.__init__.__globals__,_observation_device_init=base)
    exec(compile(tree,'<ordered observation codegen>','exec'),namespace,space)
    return space['__init__']


def _pack_source(source):
    """Pack original state spans; do not transfer the whole retina to read it."""
    tree=ast.parse(source);spans=[]
    class Zeros(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if ast.unparse(node.func)=='torch.zeros_like' and len(node.args)==1:
                inner=node.args[0]
                if isinstance(inner,ast.Call) and ast.unparse(inner.func)=='y.narrow':
                    n=ast.literal_eval(inner.args[2])
                    return ast.parse(f'torch.zeros([{n}], dtype=y.dtype, device=y.device)',mode='eval').body
            return node
    tree=Zeros().visit(tree)
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='y.narrow':
            axis,start,n=map(ast.literal_eval,node.args)
            if axis!=0:raise ContractError('Observation state slice must remain one-dimensional')
            spans.append((start,start+n))
    merged=[]
    for start,stop in sorted(set(spans)):
        if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(stop,merged[-1][1]))
        else:merged.append((start,stop))
    offsets=[];size=0
    for start,stop in merged:offsets.append((start,stop,size));size+=stop-start
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='y.narrow':
            start,n=map(ast.literal_eval,node.args[1:])
            match=next((x for x in offsets if x[0]<=start and start+n<=x[1]),None)
            if match is None:raise ContractError('Incomplete observation state packing')
            node.args[1]=ast.Constant(start-match[0]+match[2])
    # A raw whole-y export would bypass packing and is unsupported explicitly.
    for node in ast.walk(tree):
        if isinstance(node,ast.Return) and ast.unparse(node.value)=='y':raise ContractError('Whole-state device observation export unsupported')
    ast.fix_missing_locations(tree)
    # Retain the existing validation batcher's reviewed textual return form.
    packed_source=ast.unparse(tree).replace('    return (result, invalid)','    return result, invalid')+'\n'
    return packed_source,offsets,size


class DeviceSelectedObservation:
    def __init__(self,model):
        import torch
        self.model=model;self.torch=torch;self.plans={}
        self.statistics=dict(calls=0,wall_seconds=0.,state_H2D_bytes=0,output_D2H_bytes=0,static_plans=0)
    def _build(self,keys):
        p=_ObservationPlan(self.model)
        for cid,port in keys:
            if port=='gap_current' and hasattr(p.compiled.components[cid].native_model,'membrane_capacitance'):
                current=p.zero(len(p.compiled.components[cid].outputs['voltage'].cell_ids))
                for left,right,ix,iy,g in p.network._gap_plans:
                    if cid in (left,right):
                        value=p.ref(g)*(p.take(p.ensure(right,'voltage'),iy)-p.take(p.ensure(left,'voltage'),ix))
                        if cid==left:p.scatter(current,value,ix)
                        if cid==right:p.scatter(current,-value,iy)
                p.outputs[(cid,port)]=p.check(current)
            else:p.ensure(cid,port)
        cursor=0;derivatives=[];layout=[]
        for key in keys:
            ref=p.outputs[key];derivatives.append((cursor,ref.offset,ref.n));layout.append((key,cursor,ref.n));cursor+=ref.n
        ref=NS(inner_size=p.nstate,external=p.external,native=NS(nc=0),work=np.array(p.values,dtype=np.float64),
            program=np.asarray(p.program,dtype=np.int64).reshape(-1,7),pointers=np.array(p.pointers,dtype=np.int64),
            indices=np.array(p.indices,dtype=np.int64),weights=np.array(p.weights,dtype=np.float64),
            derivatives=np.asarray(derivatives,dtype=np.int64).reshape(-1,3),output_size=cursor)
        inner=OrderedInnerPlan.__new__(OrderedInnerPlan);_constructors()(inner,ref)
        inner.source,spans,packed=_pack_source(inner.source)
        inner.nstate=packed;inner.source_sha256=hashlib.sha256(inner.source.encode()).hexdigest()
        torch=self.torch;inner.compilation=torch.jit.CompilationUnit(inner.source)
        graph=CudaGraphInner(BatchedValidationInner(inner))
        self.statistics['static_plans']+=1
        return NS(plan=p,graph=graph,spans=spans,packed=packed,layout=layout,source=inner.source,
            source_sha256=inner.source_sha256,output_size=cursor)
    def evaluate(self,t,y,bundle,recording_plan):
        tick=time.perf_counter();m=self.model;torch=self.torch
        if y.dtype!=np.float64 or y.shape!=(sum(s.shape[0] for s in m.state_specs),) or not np.all(np.isfinite(y)):
            raise ContractError('Device observation requires the entire finite FP64 state')
        keys=tuple(key for block in m.compiled.output_order for key in block.nodes
            if recording_plan.ports is None or '.'.join(key) in recording_plan.ports)
        if not keys:return None,{}
        if keys not in self.plans:self.plans[keys]=self._build(keys)
        compiled=self.plans[keys];host=np.concatenate([y[start:stop] for start,stop,_ in compiled.spans]) if compiled.spans else np.empty(0)
        state=torch.as_tensor(host,device='cuda',dtype=torch.float64);external=[]
        for e,port in compiled.plan.external:
            q=bundle.channels[e.channel];port.validate(q);external.append(torch.as_tensor(q.values.copy(),device='cuda',dtype=torch.float64))
        external.append(torch.empty(0,device='cuda',dtype=torch.float64))
        result=compiled.graph(state,external).cpu().numpy()
        signals={}
        for (cid,port),start,n in compiled.layout:
            declaration=m.compiled.components[cid].outputs[port]
            q=Quantity(result[start:start+n],declaration.unit,declaration.meaning,declaration.evidence_tag)
            declaration.validate(q);signals[(cid,port)]=q
        self.statistics['calls']+=1;self.statistics['state_H2D_bytes']+=host.nbytes
        self.statistics['output_D2H_bytes']+=result.nbytes;self.statistics['wall_seconds']+=time.perf_counter()-tick
        return None,signals
