"""Typed-plan pointwise-run fusion using the installed PyTorch NVRTC path.

No cell equations are redefined. Transform the existing generated inner source;
CSR/gather/scatter and unsupported expressions retain their original lines.
Original intermediate domain predicates become per-row invalid flags. Shape,
FP64, current-state inputs, live outputs and final checks are preserved.
"""
import ast
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import hashlib,re
import numpy as np
from .contracts import ContractError
from .compiled_kernel import ADD,SUB,MUL,DIV,POW,SIGMOID,LOG,EXP,MINIMUM,MAXIMUM,POSITIVE_MASK,ZERO,SCATTER

POINTWISE={ADD,SUB,MUL,DIV,POW,SIGMOID,LOG,EXP,MINIMUM,MAXIMUM,POSITIVE_MASK,ZERO}
FINITE=re.compile(r'^    invalid = invalid \| torch.logical_not\(torch.isfinite\((.+)\)\).any\(\)$')
DOMAIN=re.compile(r'^    invalid = invalid \| \((.+) (<|<=) 0.0\).any\(\)$')
ASSIGN=re.compile(r'^    (v(\d+)) = (.+)$')


class NotFusible(Exception):pass


class FusedDeviceInnerPlan:
    def __init__(self,reference,program,*,maximum_operations=32):
        from torch.cuda.jiterator import _create_multi_output_jit_fn
        self.reference=reference;self.torch=reference.torch
        self.nstate=reference.nstate;self.external_sizes=reference.external_sizes
        self.constants=reference.constants;self.matrices=reference.matrices;self.indices=reference.indices
        self.statistics=dict(calls=0,dynamic_cross_stage_cache=False)
        # Only immutable plan constants are inspected/specialized in preparation.
        self.constant_arrays=[value.cpu().numpy().copy() for value in self.constants]
        self.constant_versions=[(value.data_ptr(),value._version) for value in self.constants]
        self.constant_uniform=[]
        for value in self.constant_arrays:
            self.constant_uniform.append(float(value.flat[0]) if value.size and np.all(value.view(np.uint64)==value.flat[0].view(np.uint64)) else None)
        self.original_source=reference.source;lines=reference.source.splitlines();nodes=[];shapes={};last_use={}
        for index,line in enumerate(lines):
            if line.startswith('def execute('):continue
            statement=ast.parse(line.strip()).body[0]
            value=getattr(statement,'value',None)
            if value is not None:
                for name in (item.id for item in ast.walk(value) if isinstance(item,ast.Name) and re.fullmatch(r'v\d+',item.id)):
                    last_use[name]=index
            match=ASSIGN.fullmatch(line);check=FINITE.fullmatch(line);domain=DOMAIN.fullmatch(line)
            if match:
                k=int(match[2]);op=int(program[k,0]);n=int(program[k,2]);expression=ast.parse(match[3],mode='eval').body
                if op==SCATTER:
                    if not isinstance(expression,ast.BinOp) or not isinstance(expression.left,ast.Name):raise ContractError('Unreviewed scatter generated source')
                    n=shapes[expression.left.id]
                shapes[match[1]]=n
                nodes.append(dict(index=index,line=line,kind='pointwise' if op in POINTWISE and n>0 else 'boundary',
                    name=match[1],opcode=op,shape=n,expression=expression))
            elif check or domain:
                text=check[1] if check else domain[1];expression=ast.parse(text,mode='eval').body
                try:n=self._shape(expression,shapes)
                except NotFusible:n=-1
                nodes.append(dict(index=index,line=line,kind='check',shape=n,expression=expression,
                    check='finite' if check else 'negative' if domain[2]=='<' else 'positive'))
            else:nodes.append(dict(index=index,line=line,kind='boundary',shape=-1))
        self.shapes=shapes;self.groups=[];self.functions=[];replacement={};skipped=set()
        i=0
        while i<len(nodes):
            node=nodes[i]
            if node['kind']!='pointwise':i+=1;continue
            n=node['shape'];region=[];j=i;count=0;best=None
            while j<len(nodes) and nodes[j]['kind'] in ('pointwise','check') and nodes[j]['shape'] in (1,n):
                if nodes[j]['kind']=='pointwise':
                    if nodes[j]['shape']!=n:break
                    count+=1
                if count>maximum_operations:break
                region.append(nodes[j])
                if count>=2:
                    try:proposal=self._generate(region,n,last_use,len(self.groups))
                    except NotFusible:proposal=None
                    if proposal is not None:best=(j,proposal,list(region))
                j+=1
            if best is None:i+=1;continue
            end,proposal,region=best
            self.functions.append(_create_multi_output_jit_fn(proposal['CUDA_source'],proposal['num_outputs']))
            self.groups.append({k:v for k,v in proposal.items() if k not in ('call_lines',)})
            replacement[region[0]['index']]=proposal['call_lines']
            skipped.update(row['index'] for row in region)
            i=end+1
        transformed=[]
        for index,line in enumerate(lines):
            if index==0:transformed.append('def execute(y, external, constants, matrices, indices):');continue
            if index in replacement:transformed.extend(replacement[index])
            elif index not in skipped:transformed.append(line)
        self.source='\n'.join(transformed)+'\n'
        namespace={'torch':self.torch,**{'_fusion'+str(i):fn for i,fn in enumerate(self.functions)}}
        exec(compile(self.source,'<current_typed_plan_pointwise_fusion>','exec'),namespace)
        self.compilation=SimpleNamespace(execute=namespace['execute'])
        self.source_sha256=hashlib.sha256((self.source+'\n'+'\n'.join(g['CUDA_source'] for g in self.groups)).encode()).hexdigest()
        eligible=sum(row['kind']=='pointwise' for row in nodes);fused=sum(g['pointwise_instructions'] for g in self.groups)
        self.coverage=dict(total_typed_instructions=len(program),pointwise_instructions=eligible,fused_pointwise_instructions=fused,
            fusion_groups=len(self.groups),unfused_pointwise_instructions=eligible-fused,
            fused_original_domain_predicates=sum(g['domain_predicates'] for g in self.groups),
            pointwise_coverage_fraction=fused/eligible if eligible else 0.,maximum_operations_per_group=maximum_operations,
            tensor_inputs_maximum=8,tensor_outputs_maximum=8,
            nonpointwise_boundary='Original CSR, gather and sequential-plan gap scatter source retained',
            immutable_constant_specialization_bytes=sum(a.nbytes for a in self.constant_arrays),
            kernel_count_is_static_estimate=False,source_sha256=self.source_sha256)

    def _shape(self,expression,shapes):
        if isinstance(expression,ast.Name) and expression.id in shapes:return shapes[expression.id]
        if isinstance(expression,ast.Constant):return 1
        if isinstance(expression,ast.Subscript) and isinstance(expression.value,ast.Name):
            key=ast.literal_eval(expression.slice)
            if expression.value.id=='constants':return self.constants[key].numel()
            if expression.value.id=='external':return self.external_sizes[key]
        if isinstance(expression,ast.Call) and isinstance(expression.func,ast.Attribute) and expression.func.attr=='narrow':return ast.literal_eval(expression.args[2])
        raise NotFusible('Unknown predicate shape')

    def _generate(self,rows,n,last_use,number):
        inputs=OrderedDict();local=set();statements=[];checks=0
        def scalar(value):
            value=float(value)
            if not np.isfinite(value):raise NotFusible('Nonfinite scalar constant')
            return 'T('+value.hex()+')'
        def leaf(node):
            text=ast.unparse(node)
            if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id=='constants':
                index=ast.literal_eval(node.slice);value=self.constant_uniform[index]
                if value is not None:return scalar(value)
            if text not in inputs:inputs[text]='x'+str(len(inputs))
            if len(inputs)>7:raise NotFusible('NVRTC tensor input arity limit')
            return inputs[text]
        def expr(node):
            if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)):return scalar(node.value)
            if isinstance(node,ast.Name):return node.id if node.id in local else leaf(node)
            if isinstance(node,ast.Subscript):return leaf(node)
            if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return '(-'+expr(node.operand)+')'
            if isinstance(node,ast.BinOp):
                a,b=expr(node.left),expr(node.right)
                operation={ast.Add:'__dadd_rn',ast.Sub:'__dsub_rn',ast.Mult:'__dmul_rn',ast.Div:'__ddiv_rn',ast.Pow:'pow'}.get(type(node.op))
                if operation is None:raise NotFusible('Unsupported pointwise binary')
                return operation+'('+a+','+b+')'
            if isinstance(node,ast.Compare) and len(node.ops)==1 and isinstance(node.ops[0],ast.Gt):return '('+expr(node.left)+'>'+expr(node.comparators[0])+')'
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                name=node.func.attr
                if name=='narrow':
                    if isinstance(node.func.value,ast.Name) and node.func.value.id in local:
                        if ast.literal_eval(node.args[1])!=0 or ast.literal_eval(node.args[2])!=n:raise NotFusible('Cross-row local slice')
                        return node.func.value.id
                    return leaf(node)
                if name in ('exp','log'):return name+'('+expr(node.args[0])+')'
                if name=='zeros_like':return 'T(0)'
                if name=='to' and isinstance(node.func.value,ast.Compare):return '('+expr(node.func.value)+'?T(1):T(0))'
                if name in ('minimum','maximum'):
                    a,b=expr(node.args[0]),expr(node.args[1]);fn='fmin' if name=='minimum' else 'fmax'
                    return '((isnan('+a+')||isnan('+b+'))?('+a+'+'+b+'):'+fn+'('+a+','+b+'))'
            raise NotFusible('Unsupported pointwise AST '+type(node).__name__)
        for row in rows:
            value=expr(row['expression'])
            if row['kind']=='pointwise':statements.append('T '+row['name']+'='+value+';');local.add(row['name'])
            else:
                checks+=1
                condition='!isfinite('+value+')' if row['check']=='finite' else '('+value+('<' if row['check']=='negative' else '<=')+'T(0))'
                statements.append('_invalid=_invalid||('+condition+');')
        outputs=[row['name'] for row in rows if row['kind']=='pointwise' and last_use.get(row['name'],-1)>rows[-1]['index']]
        if not outputs and not checks:raise NotFusible('No live output/predicate')
        if len(outputs)+(1 if checks else 0)>8:raise NotFusible('NVRTC output arity limit')
        arguments=['T _shape']+['T '+name for name in inputs.values()]
        arguments+=['T& _out'+str(i) for i in range(len(outputs))]
        if checks:arguments+=['T& _flag']
        body=(['bool _invalid=false;'] if checks else [])+statements
        body+=['_out'+str(i)+'='+name+';' for i,name in enumerate(outputs)]
        if checks:body+=['_flag=_invalid?T(1):T(0);']
        code='template <typename T> void current_inner_fusion_'+str(number)+'('+','.join(arguments)+'){'+''.join(body)+'}'
        arguments=['y.narrow(0,0,'+str(n)+')']+list(inputs)
        targets=list(outputs)+(['_invalid_group_'+str(number)] if checks else [])
        call='    '+', '.join(targets)+' = _fusion'+str(number)+'('+', '.join(arguments)+')'
        # Single-return Jiterator returns a Tensor rather than a singleton tuple.
        if len(targets)==1:call='    '+targets[0]+' = _fusion'+str(number)+'('+', '.join(arguments)+')'
        calls=[call]
        if checks:calls.append('    invalid = invalid | _invalid_group_'+str(number)+'.any()')
        return dict(CUDA_source=code,call_lines=calls,num_outputs=len(targets),pointwise_instructions=sum(r['kind']=='pointwise' for r in rows),
            domain_predicates=checks,first_source_line=rows[0]['index']+1,last_source_line=rows[-1]['index']+1,
            tensor_inputs=len(arguments),live_outputs=outputs,row_count=n)

    def validate_constants(self):
        if any((value.data_ptr(),value._version)!=identity for value,identity in zip(self.constants,self.constant_versions)):
            raise ContractError('Specialized immutable GPU constant changed; rebuild the fused typed plan')

    def __call__(self,y,external):
        from .gpu_device_variants import _validate_inner
        _validate_inner(self,y,external);self.validate_constants()
        result,invalid=self.compilation.execute(y,external,self.constants,self.matrices,self.indices)
        if invalid.item():raise ContractError('Fused original finite/nonnegative/positive predicate rejected')
        self.statistics['calls']+=1
        return result
