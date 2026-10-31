"""Research-only complete continuation and explicit original tighter reference."""
from types import FunctionType,MethodType
from dataclasses import replace
import numpy as np
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.hc_lateral_interface import LateralHCResearchRunner
from retina_sim import native_multirate as nm
from tools.integrated_dense_roundoff_v1 import IncrementDenseDOP853

SCHEMA='single-far-integrated-complete-continuation-v1'


class IntegratedRunner(LateralHCResearchRunner):
    def __init__(self,assembly,seed,protocol,*,precision='standard',**kwargs):
        if precision not in ('standard','tight'):raise ContractError('Declared original reference pair only')
        if precision=='tight':assembly.max_step=.00005  # Existing ACVS supported half-step.
        super().__init__(assembly,seed,protocol,**kwargs)
        self.precision=precision
        self.numeric_profile=dict(native_rtol=1e-6,native_atol=1e-9,native_max_step=.0001,
            inner_rtol=1e-6,inner_atol=1e-9,inner_max_step=.0001)
        if precision=='tight':
            self.numeric_profile.update(native_rtol=1e-7,native_atol=1e-10,native_max_step=.00005,
                inner_rtol=1e-7,inner_atol=1e-10,inner_max_step=.00005,
                native_dense_increment='accepted_h_B_K_before_rounded_state_subtraction_v1')
            # Reuse the exact original native reference function, with only its
            # private maximum step and predeclared rtol/atol tightened.
            fast=FunctionType(nm._solve_fast.__code__,dict(nm._solve_fast.__globals__,MAX_STEP=.00005,DOP853=IncrementDenseDOP853),
                              '_solve_fast_tighter',nm._solve_fast.__defaults__,nm._solve_fast.__closure__)
            fast.__kwdefaults__=nm._solve_fast.__kwdefaults__
            original=nm.NativeMultirateBackend.advance_joint_reference
            method=FunctionType(original.__code__,dict(original.__globals__,_solve_fast=fast),
                                original.__name__,original.__defaults__,original.__closure__)
            method.__kwdefaults__=original.__kwdefaults__
            backend=self.native_backend;backend.rtol=1e-7;backend.atol=backend.ops.array(np.full(self.native_physics.nstate,1e-10))
            backend.advance_joint_reference=MethodType(method,backend)
            backend.reference_identity=canonical_hash(dict(parent=backend.reference_identity,explicit_tighter_profile=self.numeric_profile))
            self.native_state=replace(self.native_state,backend_id=backend.reference_identity)
            bound=self.inner.advance_reference
            def advance(model,state,t0,t1,*,rtol,atol):return bound(state,t0,t1,rtol=1e-7,atol=1e-10)
            self.inner.advance_reference=MethodType(advance,self.inner)
        self.identity=canonical_hash(dict(parent_execution=self.identity,integration_schema=SCHEMA,numerical_profile=self.numeric_profile))

    def checkpoint_payload(self,**kwargs):
        payload=GroupTimeArm.checkpoint_payload(self,**kwargs)
        payload['integrated_research_contract']=dict(schema=SCHEMA,precision=self.precision,
            configuration=self.assembly.lateral_configuration['candidate_sha256'],
            binding=self.assembly.parameter_binding_id,numerical_profile=self.numeric_profile,
            production_checkpoint=False)
        h=self.native_state.y[self.native_physics.slices['Vh']]
        payload['native']['h1_h2_semantic_frontier']={name:h[rows].copy() for name,rows in [('h1',self.assembly.h1_rows),('h2',self.assembly.h2_rows)]}
        return payload

    @classmethod
    def from_checkpoint(cls,assembly,seed,protocol,payload,**kwargs):
        contract=payload.get('integrated_research_contract',{})
        if contract.get('schema')!=SCHEMA or contract.get('configuration')!=assembly.lateral_configuration['candidate_sha256'] or contract.get('binding')!=assembly.parameter_binding_id:
            raise ContractError('New research continuation identity required; no legacy relabel')
        result=GroupTimeArm.from_checkpoint.__func__(cls,assembly,seed,protocol,payload,precision=contract['precision'],**kwargs)
        h=result.native_state.y[result.native_physics.slices['Vh']]
        if contract['numerical_profile']!=result.numeric_profile or any(not np.array_equal(payload['native']['h1_h2_semantic_frontier'][name],h[rows]) for name,rows in [('h1',assembly.h1_rows),('h2',assembly.h2_rows)]):
            result.close();raise ContractError('Research continuation lost numeric/HC frontier')
        return result
