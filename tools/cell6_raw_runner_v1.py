"""Original integration with the proven cancellation-free native dense adapter."""
from dataclasses import replace
from types import FunctionType,MethodType
import numpy as np
from retina_sim import native_multirate as nm
from retina_sim.contracts import canonical_hash
from retina_sim.ac_vs_model import ACVSInner
from tools.integrated_far_runner_v1 import IntegratedRunner
from tools.integrated_dense_roundoff_v1 import IncrementDenseDOP853
from retina_sim.contracts import ContractError


def pack_without_state_cell(model,state):
    if not isinstance(state,dict):raise ContractError('Cell6 inner state must remain the complete mapping')
    model.validate(state)
    values=[]
    for key in model.state_keys:values.append(state[key])
    return np.concatenate(values)


class Cell6Runner(IntegratedRunner):
    def frame_at(self,t,side='right'):
        # The closed terminal witness has no successor input frame.
        return super().frame_at(t,'left' if t==self.stop and side=='right' else side)

    def rod_sample_at(self,t,side='right'):
        if t==self.stop and side=='left':
            origin=self.protocol['clock_origin_seconds']
            sampled=origin+(int(round((t-origin)/.0001))-1)*.0001
            return sampled,self.frame_at(sampled,'right')
        return super().rod_sample_at(t,side)

    def __init__(self,*args,precision='standard',**kwargs):
        super().__init__(*args,precision=precision,**kwargs)
        self.configure_native_dense(precision=='tight')
        self.numeric_profile['native_dense_increment']='accepted_h_B_K_before_rounded_state_subtraction_v1'
        self.numeric_profile['inner_dense_increment']='accepted_h_B_K_before_rounded_state_subtraction_v1'
        self.configure_inner_dense()
        self.inner.pack=MethodType(pack_without_state_cell,self.inner)
        self.identity=canonical_hash(dict(parent=self.identity,cell6_raw_dense_profile=self.numeric_profile))

    def configure_native_dense(self,tight):
        step=.00005 if tight else .0001
        fast=FunctionType(nm._solve_fast.__code__,dict(nm._solve_fast.__globals__,MAX_STEP=step,DOP853=IncrementDenseDOP853),
            nm._solve_fast.__name__,nm._solve_fast.__defaults__,nm._solve_fast.__closure__)
        fast.__kwdefaults__=nm._solve_fast.__kwdefaults__
        original=nm.NativeMultirateBackend.advance_joint_reference
        function=FunctionType(original.__code__,dict(original.__globals__,_solve_fast=fast),original.__name__,original.__defaults__,original.__closure__)
        function.__kwdefaults__=original.__kwdefaults__
        self.native_backend.advance_joint_reference=MethodType(function,self.native_backend)

    def configure_inner_dense(self):
        original=ACVSInner.advance_reference
        function=FunctionType(original.__code__,dict(original.__globals__,DOP853=IncrementDenseDOP853),original.__name__,original.__defaults__,original.__closure__)
        function.__kwdefaults__=original.__kwdefaults__
        def advance(model,state,t0,t1,*,rtol,atol):
            return function(model,state,t0,t1,rtol=self.numeric_profile['inner_rtol'],atol=self.numeric_profile['inner_atol'])
        self.inner.advance_reference=MethodType(advance,self.inner)

    def tighter_continuation(self):
        """Explicit numerical fork from an exact complete standard checkpoint."""
        lookahead=self.native_state.time_seconds
        self.assembly.max_step=.00005;self.native_backend.rtol=1e-7
        self.native_backend.atol=self.native_backend.ops.array(np.full(self.native_physics.nstate,1e-10))
        self.configure_native_dense(True)
        self.numeric_profile.update(native_rtol=1e-7,native_atol=1e-10,native_max_step=.00005,
            inner_rtol=1e-7,inner_atol=1e-10,inner_max_step=.00005)
        self.configure_inner_dense()
        self.native_backend.reference_identity=canonical_hash(dict(parent=self.native_backend.reference_identity,tighter_continuation=self.numeric_profile))
        self.native_state=replace(self.native_state,backend_id=self.native_backend.reference_identity)
        self.identity=canonical_hash(dict(parent=self.identity,tighter_continuation_at=self.inner_time,numerics=self.numeric_profile))
        return dict(common_standard_prefix_until=self.inner_time,retained_already_accepted_native_lookahead_until=lookahead,
            future_native_and_all_inner_steps_tighter=True,whole_Cell6_tight_run_claimed=False)
