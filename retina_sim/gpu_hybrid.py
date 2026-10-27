"""Explicit optional GPU integration of the full native/inner joint system.

Only native vector arithmetic and the joint DOP853 arrays move to CUDA. Inner
components use their existing reviewed Numba plan at every identical RK stage;
the source release vector crosses that computational boundary without changing
cells, edges or weights. Histories, events, recording and the original numerical
background monitor retain the existing CPU implementation.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib,inspect,time
import numpy as np
from .contracts import ContractError,canonical_hash
from .hybrid import HybridRunner
from .compiled_kernel import Plan,dispatcher,WorkspaceOwner
from .gpu_native import CudaNative
from .gpu_dop853 import integrate


class NativeInnerPartition(WorkspaceOwner):
    def __init__(self,network,*,build_plan=True):
        import torch
        self.network=network;self.torch=torch;c=network.compiled
        ids=[cid for cid,value in c.components.items() if value.mechanism=='VanHateren2005NativeConeH1']
        if len(ids)!=1:raise ContractError('CUDA route requires exactly one verified native component')
        self.cid=cid=ids[0];component=c.components[cid]
        if network.slices[cid].start!=0:raise ContractError('CUDA route requires the original native-first state layout')
        if any(block.cyclic for block in c.output_order) or any(link.delayed_operator is not None for link in c.links):
            raise ContractError('CUDA route does not lower algebraic closures or explicit delayed operators')
        for link in c.links:
            s=link.specification
            if s.target_component==cid or (s.source_component==cid and s.source_port!='release'):
                raise ContractError('CUDA partition requires unchanged external light and release-only outgoing native links')
        if any(cid in (left,right) for left,right,*rest in network._gap_plans):
            raise ContractError('CUDA partition cannot omit native electrical junctions')
        external=[e for e in c.spec.external_inputs if e.target_component==cid]
        if len(external)!=1 or external[0].target_port!='light':raise ContractError('Explicit native-light channel required')
        self.native_external=external[0];self.native_port=component.inputs['light']
        self.native=CudaNative(component.native_model);self.offset=network.slices[cid].stop
        self.fn,_,_=dispatcher()
        if not build_plan:return
        components={key:value for key,value in c.components.items() if key!=cid}
        spec=NS(external_inputs=tuple(e for e in c.spec.external_inputs if e.target_component!=cid))
        order=tuple(block for block in c.output_order if block.nodes[0][0]!=cid)
        proxy=NS(compiled=NS(components=components,spec=spec,output_order=order),
            _local_slices={key:tuple(slice(s.start-self.offset,s.stop-self.offset) for s in values)
                           for key,values in network._local_slices.items() if key!=cid},
            state_specs=network.state_specs[len(component.state_specs):],
            _incoming=network._incoming,_gap_plans=network._gap_plans)
        plan=Plan(proxy)
        release=plan.allocate(self.native.nc);plan.outputs[(cid,'release')]=release
        plan.nexternal+=self.native.nc
        self.derivatives=plan.build();self.external=plan.external;self.inner_size=plan.nstate
        self._scratch=plan.workspace();self.program=np.asarray(plan.program,dtype=np.int64).reshape(-1,7)
        self.weights=np.asarray(plan.weights,dtype=np.float64);self.indices=np.asarray(plan.indices,dtype=np.int64)
        self.pointers=np.asarray(plan.pointers,dtype=np.int64)
        self.plan_hash=canonical_hash({'program':self.program,'constants':self._scratch,'weights':self.weights,
            'indices':self.indices,'pointers':self.pointers,'derivatives':self.derivatives,
            'native_component':cid,'native_offset':self.offset})

    def __call__(self,t,y,bundle):
        torch=self.torch
        expected={self.native_external.channel}|{e.channel for e,p in self.external}
        if set(bundle.channels)!=expected:raise ContractError('CUDA joint external channel set differs')
        q=bundle.channels[self.native_external.channel];self.native_port.validate(q)
        light=torch.as_tensor(np.asarray(q.values).copy(),dtype=torch.float64,device='cuda')
        outer,release=self.native(y[:self.offset],light)
        values=[]
        for external,port in self.external:
            value=bundle.channels[external.channel];port.validate(value);values.append(value.values)
        values.append(release.cpu().numpy())
        inner=self.fn(y[self.offset:].cpu().numpy(),np.concatenate(values),self.work,self.program,
            self.weights,self.indices,self.pointers,self.derivatives)
        result=torch.cat((outer,torch.as_tensor(inner,dtype=torch.float64,device='cuda')))
        if not torch.isfinite(result).all().item():raise ContractError('Nonfinite CUDA joint derivative')
        return result


class CudaHybridRunner(HybridRunner):
    """Named execution alternative; old backend checkpoints intentionally reject."""
    _partition_factory=NativeInnerPartition
    def __init__(self,compiled,*,compute_backend='numba'):
        if compute_backend!='numba':raise ContractError('CUDA execution requires the reviewed Numba inner lowering')
        import torch,scipy
        if not torch.cuda.is_available():raise ContractError('Explicit CUDA backend unavailable; no silent CPU fallback')
        torch.set_num_threads(1)
        # Establish the device before the Windows Numba/OpenMP compilation path.
        # The optional backend owns this initialization; CPU runs never import it.
        torch.cuda.init()
        super().__init__(compiled,compute_backend=compute_backend)
        if self.discrete:raise ContractError('CUDA route supports this continuous native system only')
        from scipy.integrate._ivp import rk,common
        torch.backends.cuda.matmul.allow_tf32=False
        self.partition=self._partition_factory(self.model)
        identity={'backend':'cuda_native_joint_dop853_float64_v1','torch_version':torch.__version__,
            'cuda_version':torch.version.cuda,'device':torch.cuda.get_device_name(),
            'device_capability':torch.cuda.get_device_capability(),'torch_cpu_threads':torch.get_num_threads(),
            'scipy_version':scipy.__version__,'scipy_rk_sha256':hashlib.sha256(inspect.getsource(rk).encode()).hexdigest(),
            'scipy_common_sha256':hashlib.sha256(inspect.getsource(common).encode()).hexdigest(),
            'implementation_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                     for name in ('gpu_native.py','gpu_dop853.py','gpu_hybrid.py')},
            'inner_plan_hash':self.partition.plan_hash,'dtype':'float64','tf32':False,
            'CUDA_CSR_storage':'Private sorted-column copy; original source arrays/weights untouched; duplicate columns rejected',
            'source_equations_parameters_graph':'unchanged','bitwise_cpu_parity':False}
        self.model.model_hash=canonical_hash({'existing_runtime':self.model.model_hash,'explicit_cuda_execution':identity})
        self.model.compute_metadata=dict(self.model.compute_metadata,integration_backend=identity,
            scope='CUDA native/joint DOP853; unchanged inner Numba plan, event/history/recording and CPU background monitor')

    def _integrate(self,state,stimulus,stop,config,samples):
        if config.method!='DOP853':raise ContractError('Named CUDA backend requires the unchanged DOP853 method')
        import torch
        start=state.time_seconds;y=self._flat(state)
        def rhs(t,values):
            return self.partition(t,values,stimulus.at(t,left_limit=bool(t>=stop)))
        requested=np.array(sorted(set(float(x) for x in samples if start<x<=stop)))
        endpoint_only=not np.any(requested<stop)
        targets=None if endpoint_only else np.array(sorted(set(requested.tolist()+[float(stop)])))
        tick=time.perf_counter()
        result=integrate(rhs,start,torch.as_tensor(y.copy(),dtype=torch.float64,device='cuda'),stop,
            rtol=config.rtol,atol=config.atol,max_step=config.max_step,t_eval=targets)
        metrics=self.integration_metrics;metrics['solver_calls']+=1;metrics['nfev']+=result['nfev']
        metrics['integration_wall_seconds']+=time.perf_counter()-tick
        metrics['minimum_interval_seconds']=min(metrics['minimum_interval_seconds'] or (stop-start),stop-start)
        metrics['maximum_nfev_per_interval']=max(metrics['maximum_nfev_per_interval'],result['nfev'])
        rows={float(stop):result['final']} if endpoint_only else {float(t):a for t,a in zip(result['t'],result['y'].T)}
        # With t_eval SciPy returns the dense endpoint column; preserve that
        # choice for the persisted state as well as for the recorded sample.
        self._replace(state,result['final'] if endpoint_only else result['y'][:,-1],stop)
        return rows
