"""Opt-in exact CUDA bridge: compact native-release transfer, full neural model.

The full native cone/HC system and the original joint DOP853/event loop remain
unchanged. Only private inner-plan column addresses and copied port values are
compacted. CPU history, monitoring and recording still use the full model.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import numpy as np
from scipy import sparse
from .contracts import ContractError,canonical_hash
from .compiled_kernel import Plan
from .gpu_hybrid import CudaHybridRunner

BACKEND='cuda_compact_native_joint_dop853_float64_v2'


def compact_release_layout(network,cid,native_count):
    """All compiled outgoing release consumers; no population-name assumptions."""
    links=[link for link in network.compiled.links if link.specification.source_component==cid]
    for link in links:
        if link.specification.source_port!='release' or link.specification.target_component==cid:
            raise ContractError('Compact bridge retains the release-only outward native contract')
        if link.matrix.shape[1]!=native_count or np.any(link.matrix.data==0):
            raise ContractError('Compact bridge requires canonical nonzero compiled release columns')
    used=np.unique(np.concatenate([link.matrix.indices for link in links])).astype(np.int64) if links else np.empty(0,dtype=np.int64)
    incoming={key:list(values) for key,values in network._incoming.items()};consumers=[]
    for link in links:
        m=link.matrix
        # Preserve every coefficient and the original within-row reduction order.
        remapped=sparse.csr_matrix((m.data.copy(),np.searchsorted(used,m.indices).astype(m.indices.dtype),m.indptr.copy()),
            shape=(m.shape[0],len(used)))
        s=link.specification;key=(s.target_component,s.target_port)
        if sum(value is link for value in incoming.get(key,()))!=1:
            raise ContractError('Compact bridge outgoing consumer differs from the compiled incoming plan')
        incoming[key]=[replace(value,matrix=remapped) if value is link else value for value in incoming[key]]
        component=network.compiled.components[s.target_component]
        consumers.append({'component':s.target_component,'mechanism':component.mechanism,'port':s.target_port,
            'time_contract':component.time_contract.kind,'edges':m.nnz,'unique_native_release_columns':len(np.unique(m.indices))})
    return used,incoming,consumers


class CompactNativeInnerPartition:
    def __init__(self,network,reference):
        self.network=network;self.torch=reference.torch;self.native=reference.native
        self.cid=cid=reference.cid;self.offset=reference.offset
        self.native_external=reference.native_external;self.native_port=reference.native_port
        c=network.compiled;used,incoming,consumers=compact_release_layout(network,cid,self.native.nc)
        used.setflags(write=False);self.used=used
        self.gather=self.torch.as_tensor(used.copy(),dtype=self.torch.int64,device='cuda')
        components={key:value for key,value in c.components.items() if key!=cid}
        spec=NS(external_inputs=tuple(e for e in c.spec.external_inputs if e.target_component!=cid))
        order=tuple(block for block in c.output_order if block.nodes[0][0]!=cid)
        proxy=NS(compiled=NS(components=components,spec=spec,output_order=order),
            _local_slices={key:tuple(slice(s.start-self.offset,s.stop-self.offset) for s in values)
                           for key,values in network._local_slices.items() if key!=cid},
            state_specs=network.state_specs[len(c.components[cid].state_specs):],_incoming=incoming,_gap_plans=network._gap_plans)
        plan=Plan(proxy);release=plan.allocate(len(used));plan.outputs[(cid,'release')]=release;plan.nexternal+=len(used)
        self.derivatives=plan.build();self.external=plan.external;self.inner_size=plan.nstate
        self.work=np.asarray(plan.values,dtype=np.float64);self.program=np.asarray(plan.program,dtype=np.int64).reshape(-1,7)
        self.weights=np.asarray(plan.weights,dtype=np.float64);self.indices=np.asarray(plan.indices,dtype=np.int64)
        self.pointers=np.asarray(plan.pointers,dtype=np.int64);self.fn=reference.fn
        self.plan_hash=canonical_hash({'program':self.program,'constants':self.work,'weights':self.weights,
            'indices':self.indices,'pointers':self.pointers,'derivatives':self.derivatives,'native_release_global_rows':used})
        self.layout={'native_release_cells':self.native.nc,'transferred_release_cells':len(used),
            'legacy_release_bytes_per_RHS':self.native.nc*8,'compact_release_bytes_per_RHS':len(used)*8,
            'native_release_rows_sha256':hashlib.sha256(used.tobytes()).hexdigest(),'consumers':consumers,
            'all_compiled_outgoing_ports_checked':True,'native_feedback_full':True,
            'history_monitor_and_recording':'Original complete model.evaluate and _sample; no compact readout substitution',
            'row_coefficients_and_reduction_order_unchanged':True}
        self.statistics={'RHS_calls':0,'release_D2H_bytes':0,'light_uploads':0,'light_H2D_bytes':0,'identical_light_hits':0}
        self._light_cpu=None;self._light_gpu=None

    def _light(self,q):
        value=np.asarray(q.values)
        if value.dtype!=np.float64:raise ContractError('Compact light buffer requires the original float64 Quantity')
        # Compare actual bits, never just object identity, time or a ZOH label.
        # Finite mutations of a reused Quantity and continuous/right-limit
        # changes therefore upload a fresh buffer; signed zero is retained.
        if self._light_cpu is not None and np.array_equal(value.view(np.uint64),self._light_cpu.view(np.uint64)):
            self.statistics['identical_light_hits']+=1;return self._light_gpu
        snapshot=value.copy();self._light_gpu=self.torch.as_tensor(snapshot,dtype=self.torch.float64,device='cuda')
        snapshot.setflags(write=False);self._light_cpu=snapshot
        self.statistics['light_uploads']+=1;self.statistics['light_H2D_bytes']+=snapshot.nbytes
        return self._light_gpu

    def __call__(self,t,y,bundle):
        torch=self.torch
        expected={self.native_external.channel}|{e.channel for e,p in self.external}
        if set(bundle.channels)!=expected:raise ContractError('CUDA joint external channel set differs')
        q=bundle.channels[self.native_external.channel];self.native_port.validate(q)
        outer,release=self.native(y[:self.offset],self._light(q))
        values=[]
        for external,port in self.external:
            value=bundle.channels[external.channel];port.validate(value);values.append(value.values)
        values.append(torch.index_select(release,0,self.gather).cpu().numpy())
        inner=self.fn(y[self.offset:].cpu().numpy(),np.concatenate(values),self.work,self.program,
            self.weights,self.indices,self.pointers,self.derivatives)
        result=torch.cat((outer,torch.as_tensor(inner,dtype=torch.float64,device='cuda')))
        if not torch.isfinite(result).all().item():raise ContractError('Nonfinite CUDA joint derivative')
        self.statistics['RHS_calls']+=1;self.statistics['release_D2H_bytes']+=len(self.used)*8
        return result


class CompactCudaHybridRunner(CudaHybridRunner):
    """Separate explicit implementation identity; no old checkpoint migration."""
    def __init__(self,compiled,*,compute_backend='numba'):
        super().__init__(compiled,compute_backend=compute_backend)
        reference=self.partition;reference_hash=self.model.model_hash
        self.partition=CompactNativeInnerPartition(self.model,reference)
        identity=dict(self.model.compute_metadata['integration_backend'],backend=BACKEND,
            compact_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            compact_plan_hash=self.partition.plan_hash,compact_release_layout=self.partition.layout,
            light_buffer_policy='Bitwise value-coherent single buffer; original per-call port validation and original time/left-limit evaluation',
            original_joint_integrator_event_history_criteria=True)
        self.model.model_hash=canonical_hash({'reference_cuda_execution':reference_hash,'compact_execution':identity})
        self.model.compute_metadata=dict(self.model.compute_metadata,integration_backend=identity)
