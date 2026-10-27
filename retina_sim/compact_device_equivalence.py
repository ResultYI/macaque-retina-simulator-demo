"""Frozen typed QA allowance for current compact device-inner candidates.

Never changes solver settings, model values, RNG or monitor criteria. These are
independent numerical engineering budgets, not physiology/global error proofs.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from .contracts import ContractError,canonical_hash
from .background import NUMERICAL_UNIT_SCALES,BackgroundPolicy

PATH=Path(__file__).resolve().parents[1]/'configurations/numerical_equivalence/compact_device_fp64_v2.json'
SHA256='82b35f99e0e0707ab77b659cce937f9fb7f80f5602d8bcc96321f7d60fa075b1'


class CompactDeviceEquivalence:
    def __init__(self):
        if hashlib.sha256(PATH.read_bytes()).hexdigest()!=SHA256:raise ContractError('Frozen compact device contract changed')
        self.policy=json.loads(PATH.read_bytes())
        self.roundoff=self.policy['roundoff_epsilon_multiplier']*np.finfo(np.float64).eps
        self.fraction=self.policy['state_budget_fraction']
    @property
    def identity(self):
        return dict(contract_id=self.policy['contract_id'],frozen_contract_sha256=SHA256,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            typed_unit_registry=NUMERICAL_UNIT_SCALES,policy=self.policy)
    @property
    def contract_hash(self):return canonical_hash(self.identity)
    def require_solver(self,solver):
        if any(getattr(solver,k)!=v for k,v in self.policy['solver_required'].items()):raise ContractError('This frozen contract requires the unchanged original solver')
        if solver.dtype!='float64':raise ContractError('FP64 required')
    def state_budget(self,solver,unit,initial):
        self.require_solver(solver)
        if unit not in NUMERICAL_UNIT_SCALES:raise ContractError('Missing typed numerical scale: '+unit)
        q=NUMERICAL_UNIT_SCALES[unit];scale=np.maximum(np.abs(initial),q['scale_floor'])
        return self.fraction*np.minimum(solver.atol+solver.rtol*scale,q['atol']+BackgroundPolicy().relative_tolerance*scale)
    @staticmethod
    def compare(original,candidate,allowance,*,label,unit):
        a=np.asarray(original);b=np.asarray(candidate);limit=np.asarray(allowance)
        if a.dtype!=np.float64 or b.dtype!=np.float64 or a.shape!=b.shape or limit.shape!=a.shape:raise ContractError('Typed FP64 comparison layout differs')
        if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(limit).all() or np.any(limit<=0):raise ContractError('Nonfinite state/output or invalid numerical budget')
        error=np.abs(a-b);ratio=error/limit
        return dict(label=label,unit=unit,passed=bool(np.all(ratio<=1)),max_error_over_budget=float(np.max(ratio,initial=0.)),
            max_absolute_difference=float(np.max(error,initial=0.)),min_allowance=float(np.min(limit)) if limit.size else None,
            max_allowance=float(np.max(limit,initial=0.)),bitwise_mismatches=int(np.count_nonzero(a.view(np.uint64)!=b.view(np.uint64))))
    def rhs(self,solver,unit,initial,original,candidate,label):
        q=NUMERICAL_UNIT_SCALES[unit];scale=np.maximum(np.abs(initial),q['scale_floor'])
        cap=self.state_budget(solver,unit,initial)/solver.max_step
        absolute=np.minimum(self.roundoff*scale/solver.max_step,cap)
        limit=np.minimum(absolute+self.roundoff*np.maximum(np.abs(original),np.abs(candidate)),cap)
        return self.compare(original,candidate,limit,label=label,unit=unit+'/s')
