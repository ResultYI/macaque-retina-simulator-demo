"""Opt-in same-RHS native CSR readout reuse; every original check still runs.

Only the already computed B@h intermediate is shared. Original NumPy/scipy
signal formulas, complete evaluate(sampling=True), transitions, validation and
gap checks remain unchanged. It is not a persistent signal/adaptation cache.
"""
from pathlib import Path
import hashlib
import numpy as np
from scipy.special import expit
from .contracts import ContractError
from .compiled_kernel import NATIVE_SIGNALS
from .vh2005 import _NativeSignalContext


class NativeContextReuse:
    def __init__(self,model):
        self.model=model;self.kernel=model._compiled_rhs;self.layout={}
        if self.kernel is None:raise ContractError('Native context reuse requires the original compiled RHS')
        for cid,c in model.compiled.components.items():
            if cid not in model._context_providers:continue
            if c.mechanism!='VanHateren2005NativeConeH1':raise ContractError('Unreviewed evaluation context provider')
            slices=model._local_slices[cid];native=c.native_model
            expected=[slices[j].start for j in (2,4,5,6)]
            matches=[]
            for row in self.kernel.program:
                if row[0]!=NATIVE_SIGNALS or row[2]!=native.nc:continue
                a=int(row[3]);layout=self.kernel.work[a:a+9]
                if np.array_equal(layout[:4],expected):matches.append(tuple(int(x) for x in layout))
            if len(matches)!=1:raise ContractError('No unique reviewed fused native CSR signal layout')
            self.layout[cid]=matches[0][4]
        if not self.layout:raise ContractError('No fused native evaluation context to reuse')
        self.identity=dict(procedure='same_RHS_native_Bh_context_reuse_v1',
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            complete_original_sampling_evaluation_and_all_checks_retained=True,
            reused_value='only same-call compiled B@h; original NumPy/scipy signal expressions',
            cross_state_or_cross_time_cache=False,enabled_after_exact_same_point_check=False)

    def evaluate(self,t,y,bundle,*,sampling=True):
        model=self.model;kernel=self.kernel
        # Same call only; a stale integration-stage buffer never becomes readout.
        with kernel.lock:
            if not np.array_equal(kernel.work[:kernel.nstate],y):raise ContractError('Native readout scratch belongs to another state')
            offset=kernel.nstate
            for external,port in kernel.external:
                q=bundle.channels[external.channel];port.validate(q);n=len(port.cell_ids)
                if not np.array_equal(kernel.work[offset:offset+n],q.values):raise ContractError('Native readout scratch belongs to another input')
                offset+=n
            local={cid:model.local_blocks(cid,y) for cid in model.compiled.components};contexts={}
            for cid,bh_offset in self.layout.items():
                native=model.compiled.components[cid].native_model;b=local[cid];p=native.p
                R,E,X,C,Vis,gi,Vbar,z1,z2,h=b
                # Exact original NativeConeH1.signals domain predicates.
                if np.any(gi<=0) or np.any(Vbar<=0) or np.any(Vis<0):raise ContractError('Native positive conductance/adaptation domain violated')
                bh=kernel.work[bh_offset:bh_offset+native.nc]
                aI=(Vbar/p['V_I'])**p['mu'];q=expit((Vis-bh-p['V_k'])/p['V_n'])
                contexts[cid]=_NativeSignalContext(native,b,(X**p['n_X'],aI,q,q/aI))
            return model.evaluate(t,y,bundle,sampling=sampling,_local=local,_contexts=contexts)


def _assert_same_evaluation(left,right):
    li,lo=left;ri,ro=right
    if set(li)!=set(ri) or set(lo)!=set(ro):raise ContractError('Native context comparison port layout differs')
    maximum=0.
    def check(a,b):
        nonlocal maximum
        if (a.unit,a.meaning,a.evidence_tag)!=(b.unit,b.meaning,b.evidence_tag):raise ContractError('Native context comparison quantity contract differs')
        if a.values.size:maximum=max(maximum,float(np.max(np.abs(a.values-b.values))))
        if not np.array_equal(a.values,b.values):raise ContractError('Native context values differ from original exact readout')
    for key in lo:check(lo[key],ro[key])
    for cid in li:
        if set(li[cid])!=set(ri[cid]):raise ContractError('Native context comparison input layout differs')
        for port in li[cid]:
            a=li[cid][port];b=ri[cid][port]
            if hasattr(a,'values'):check(a,b)
    return maximum


def enable_native_context_reuse(runner,state,bundle,*,policy=None,reference_state=None):
    """Compare the same already assembled state, then opt in; fail closed.

    No second network is assembled. An unsupported layout or any non-bitwise
    output uses the original evaluator and reports the actual rejection.
    """
    previous=getattr(runner,'background_monitor_context_reuse',None)
    try:
        if previous is not None:raise ContractError('Native context reuse is already selected')
        helper=NativeContextReuse(runner.model);y=runner._flat(state)
        runner.model.rhs_flat(state.time_seconds,y,bundle,runner.model.parameters,runner.model.graph)
        original=runner.model.evaluate(state.time_seconds,y,bundle,sampling=True)
        projected=helper.evaluate(state.time_seconds,y,bundle,sampling=True)
        maximum=_assert_same_evaluation(original,projected)
        del original,projected
        # Exercise every original monitor predicate and report computation on
        # the same point/reference, not merely selected native output ports.
        from .background import _WindowMonitor,BackgroundPolicy
        selected_policy=policy or BackgroundPolicy()
        original_monitor=_WindowMonitor(runner,reference_state or state,selected_policy)
        original_monitor.observe(state.time_seconds,y,bundle)
        original_report=original_monitor.report(selected_policy.window_seconds)
        del original_monitor
        runner.background_monitor_context_reuse=helper
        reused_monitor=_WindowMonitor(runner,reference_state or state,selected_policy)
        reused_monitor.observe(state.time_seconds,y,bundle)
        reused_report=reused_monitor.report(selected_policy.window_seconds)
        del reused_monitor
        if original_report!=reused_report:
            raise ContractError('Original monitor report differs under native context reuse')
        helper.identity.update(enabled_after_exact_same_point_check=True,
            all_original_input_and_output_quantities_bitwise_equal=True,
            complete_original_monitor_report_exactly_equal=True,
            maximum_absolute_quantity_difference=maximum,state_scalars=len(y),
            original_model_hash=runner.model.model_hash)
        runner.background_monitor_context_reuse=helper
        return helper.identity
    except (ContractError,ValueError,AssertionError) as exc:
        if previous is None:
            if hasattr(runner,'background_monitor_context_reuse'):del runner.background_monitor_context_reuse
        else:runner.background_monitor_context_reuse=previous
        return dict(procedure='original_full_monitor_evaluation',enabled_after_exact_same_point_check=False,
            reason=str(exc),checks_or_tolerances_changed=False)
