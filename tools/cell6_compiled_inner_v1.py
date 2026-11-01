"""Opt-in compiled continuous RHS; original solver/events/native HC stay intact."""
from copy import deepcopy
from time import perf_counter
import numpy as np
from retina_sim.ac_vs_compiled import CompiledACVSInnerRHS
from retina_sim.contracts import ContractError,canonical_hash
from tools.cell6_raw_runner_v1 import Cell6Runner

SCHEMA='cell6-compiled-acvs-inner-execution-v1'
PHYSICAL=('native','inner','spike_states','clocks','input_position','observation_delay','readout_cursor')
CLOSED_INSTANCES=[]


class CompiledCell6Runner(Cell6Runner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.direct_controller_identity=self.identity
        tick=perf_counter();candidate=CompiledACVSInnerRHS(self.inner,repo_root=self.assembly.repo_root if hasattr(self.assembly,'repo_root') else None)
        constructed=perf_counter()-tick;y=self.inner.pack(self.inner_state);before=y.copy()
        reference=self.compiled.rhs_flat(self.inner_time,y)
        tick=perf_counter();value=candidate.rhs_flat(self.inner_time,y);first=perf_counter()-tick
        np.testing.assert_array_equal(y,before)
        np.testing.assert_allclose(value,reference,rtol=5e-14,atol=2e-13)
        self.compiled=candidate;self.inner.compiled_rhs=candidate
        self.identity=canonical_hash(dict(parent=self.direct_controller_identity,compiled_inner=candidate.identity,schema=SCHEMA))
        self.compiled_build=dict(**candidate.metrics,construction_wall_seconds=constructed,first_rhs_wall_seconds=first,
            construction_plus_first_call_seconds=constructed+first,initial_rhs_max_abs_error=float(np.max(abs(value-reference))),
            rhs_parity_rtol=5e-14,rhs_parity_atol=2e-13,original_Vs_fixture_thresholds=True,
            runtime_class=type(candidate).__name__,python_direct_continuous_rhs_fallback_calls=0,
            total_inner_states=self.inner.nstate,continuous_lowered_blocks=int(np.sum(candidate.derivatives[:,1]>=0)),
            held_or_discrete_zero_derivative_blocks=[list(k) for k,row in zip(self.inner.state_keys,candidate.derivatives) if row[1]<0])

    def checkpoint_payload(self,**kwargs):
        payload=super().checkpoint_payload(**kwargs)
        payload['compiled_inner_execution']=dict(schema=SCHEMA,identity=self.compiled.identity,
            direct_controller_identity=self.direct_controller_identity,fastmath=False,precision='float64')
        return payload

    def close(self):
        if hasattr(self,'compiled_build') and not getattr(self,'_compiled_closed',False):
            CLOSED_INSTANCES.append(dict(build=self.compiled_build,compiled_rhs_calls=self.compiled._calls,
                execution_identity=self.compiled.identity,source_seconds=self.inner_time))
            self._compiled_closed=True
        super().close()

    @classmethod
    def from_checkpoint(cls,assembly,seed,protocol,payload,**kwargs):
        contract=payload.get('compiled_inner_execution',{})
        if contract.get('schema')!=SCHEMA:raise ContractError('Explicit reference-to-compiled transfer is required')
        result=super().from_checkpoint(assembly,seed,protocol,payload,**kwargs)
        if contract['identity']!=result.compiled.identity or contract['direct_controller_identity']!=result.direct_controller_identity:
            result.close();raise ContractError('Compiled checkpoint execution identity changed')
        return result

    @classmethod
    def from_reference_checkpoint(cls,assembly,seed,protocol,payload,**kwargs):
        if 'compiled_inner_execution' in payload:raise ContractError('Reference transfer requires the actual old backend')
        reference=Cell6Runner.from_checkpoint(assembly,seed,protocol,payload,**kwargs)
        try:
            old=reference.checkpoint_payload(monitor=payload['monitor'],recording=payload['recording'])
            if any(canonical_hash(old[k])!=canonical_hash(payload[k]) for k in PHYSICAL):raise ContractError('Old checkpoint restoration changed scientific state')
            preview=cls(assembly,seed,protocol,**kwargs)
            try:
                if preview.direct_controller_identity!=reference.identity:raise ContractError('Transfer model/source contract differs')
                transferred=deepcopy(payload);transferred['controller']['identity']=preview.identity
                transferred['compiled_inner_execution']=preview.checkpoint_payload()['compiled_inner_execution']
                build=preview.compiled_build
            finally:preview.close()
            result=cls.from_checkpoint(assembly,seed,protocol,transferred,**kwargs)
            copied=result.checkpoint_payload(monitor=payload['monitor'],recording=payload['recording'])
            if any(canonical_hash(copied[k])!=canonical_hash(old[k]) for k in PHYSICAL):result.close();raise ContractError('Transfer reset scientific state')
            old_control={k:v for k,v in old['controller'].items() if k!='identity'}
            new_control={k:v for k,v in copied['controller'].items() if k!='identity'}
            if canonical_hash(old_control)!=canonical_hash(new_control):result.close();raise ContractError('Transfer changes event/controller history')
            y=result.inner.pack(result.inner_state)
            np.testing.assert_allclose(result.compiled(result.inner_time,y),reference.compiled(result.inner_time,y),rtol=5e-14,atol=2e-13)
            result.transfer_evidence=dict(schema='explicit-source-to-compiled-state-transfer-v1',
                source_controller_identity=reference.identity,new_controller_identity=result.identity,
                exact_physical_categories=list(PHYSICAL),exact_controller_except_backend_identity=True,
                no_state_reset=True,compiled_build=build,transferred_at_seconds=result.inner_time)
            return result
        finally:reference.close()
