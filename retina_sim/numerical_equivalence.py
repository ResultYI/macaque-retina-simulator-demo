"""Frozen, validation-only numerical contract; original bitwise gates survive.

These independently allocated engineering fractions are neither physiological
tolerances nor certified global ODE bounds. Background floors never scale atol.
"""
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import numpy as np
from .background import NUMERICAL_UNIT_SCALES
from .contracts import ContractError, canonical_hash

CONTRACT_ID = 'ordered_fp64_independent_contract_v1'
CONTRACT_PATH = Path(__file__).resolve().parents[1]/'configurations/numerical_equivalence/ordered_fp64_v1.json'
CONTRACT_SHA256 = 'e85c3a8f984471a34a3caad28664945a8555a978b58d3435a424c5d6db2ffeaf'


def fp64_arrays(*values):
    arrays = tuple(np.asarray(x) for x in values)
    if not arrays or any(x.dtype != np.float64 or x.shape != arrays[0].shape
            or not np.all(np.isfinite(x)) for x in arrays):
        raise ContractError('Numerical comparison requires finite matching FP64 arrays')
    return arrays


@dataclass(frozen=True)
class NumericalEquivalence:
    contract_id: str = CONTRACT_ID
    _implementation_sha256: str = field(init=False,repr=False)

    def __post_init__(self):
        if self.contract_id != CONTRACT_ID:
            raise ContractError('Unreviewed numerical contract identity')
        if hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() != CONTRACT_SHA256:
            raise ContractError('Frozen independent numerical contract hash changed')
        object.__setattr__(self,'_implementation_sha256',hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    @property
    def policy(self):
        value=CONTRACT_PATH.read_bytes()
        if hashlib.sha256(value).hexdigest()!=CONTRACT_SHA256:
            raise ContractError('Frozen independent numerical contract hash changed')
        return json.loads(value)

    @property
    def identity(self):
        return dict(contract_id=self.contract_id, frozen_contract_sha256=CONTRACT_SHA256,
            implementation_sha256=self._implementation_sha256,
            policy=self.policy, state_unit_registry=sorted(NUMERICAL_UNIT_SCALES),
            scope='Numerical engineering equivalence; no physiological completion')

    @property
    def contract_hash(self):
        return canonical_hash(self.identity)

    @staticmethod
    def require_unit(unit):
        if unit not in NUMERICAL_UNIT_SCALES and unit not in ('Hz', 'events/s'):
            raise ContractError('No registered numerical state/output unit: '+unit)

    def references(self, solver):
        if solver.method != 'DOP853' or solver.dtype != 'float64':
            raise ContractError('Frozen equivalence contract requires original FP64 DOP853')
        return (solver,
            replace(solver, rtol=solver.rtol/10, atol=solver.atol/10, max_step=solver.max_step/2),
            replace(solver, rtol=solver.rtol/100, atol=solver.atol/100, max_step=solver.max_step/4))

    def scale(self, solver, unit, original, candidate):
        """Original solver scale, in this registered state's native unit."""
        if unit not in NUMERICAL_UNIT_SCALES:
            raise ContractError('No registered numerical ODE state unit: '+unit)
        original, candidate = fp64_arrays(original, candidate)
        return solver.atol+solver.rtol*np.maximum(np.abs(original), np.abs(candidate))

    def budget(self, solver, unit, original, candidate):
        return .01*self.scale(solver, unit, original, candidate)

    def compare(self, solver, unit, original, candidate, finer, finest):
        return self.compare_with_allowance(unit, original, candidate, finer, finest,
            self.budget(solver, unit, original, candidate),
            basis='0.01*(original_atol+original_rtol*max(abs(cpu),abs(gpu)))')

    def compare_with_allowance(self, unit, original, candidate, finer, finest, allowance, *, basis):
        """Derived outputs require their own source-formula allowance.

        No Hz/S blanket atol is introduced. Zero sensitivity means equality;
        missing, negative or nonfinite allowances are errors.
        """
        self.require_unit(unit)
        original, candidate, finer, finest = fp64_arrays(original, candidate, finer, finest)
        allowance = np.asarray(allowance, dtype=np.float64)
        if allowance.shape != original.shape or np.any(allowance < 0) or not np.all(np.isfinite(allowance)):
            raise ContractError('Missing or invalid state/formula output allowance')
        separation = np.abs(candidate-original)
        increment = np.abs(finer-finest)
        original_error, candidate_error = np.abs(original-finest), np.abs(candidate-finest)
        resolved = increment <= .25*allowance
        equivalent = separation <= allowance
        nondegraded = candidate_error <= original_error+allowance
        bitwise = original.view(np.uint64) == candidate.view(np.uint64)
        def ratio(value):
            answer = np.zeros_like(value)
            np.divide(value, allowance, out=answer, where=allowance > 0)
            answer[(allowance == 0) & (value != 0)] = np.inf
            return answer
        def maximum(value):
            answer = float(np.max(value, initial=0.))
            return answer if np.isfinite(answer) else 'infinity'
        ratios = ratio(separation)
        index = int(np.argmax(ratios)) if ratios.size else None
        ref_index=int(np.argmax(ratio(increment))) if increment.size else None
        status = ('REFERENCE_UNRESOLVED' if not np.all(resolved) else
            'FAIL_NUMERICAL_EQUIVALENCE' if not np.all(equivalent & nondegraded) else 'PASS')
        return dict(status=status, unit=unit, scalars=int(original.size), budget_basis=basis,
            bitwise_equal=bool(np.all(bitwise)), bitwise_mismatches=int(np.count_nonzero(~bitwise)),
            reference_unresolved=int(np.count_nonzero(~resolved)),
            separation_violations=int(np.count_nonzero(~equivalent)),
            nondegradation_violations=int(np.count_nonzero(~nondegraded)),
            maximum_separation_over_budget=maximum(ratios),
            maximum_reference_increment_over_budget=maximum(ratio(increment)),
            maximum_original_error_over_budget=maximum(ratio(original_error)),
            maximum_candidate_error_over_budget=maximum(ratio(candidate_error)),
            maximum_original_to_ref1_over_budget=maximum(ratio(np.abs(original-finer))),
            maximum_absolute_difference=maximum(separation),
            minimum_budget=float(np.min(allowance)) if allowance.size else None,
            maximum_budget=maximum(allowance), worst_flat_index=index,
            worst_reference_flat_index=ref_index,
            worst_reference_witness=None if ref_index is None else {key:float(value.flat[ref_index]) for key,value in
                dict(original=original,candidate=candidate,ref1=finer,ref2=finest,allowance=allowance).items()},
            worst_witness=None if index is None else {key:float(value.flat[index]) for key,value in
                dict(original=original,candidate=candidate,ref1=finer,ref2=finest,allowance=allowance).items()})

    def rhs_stage_screen(self, solver, unit, original_state, candidate_state,
            original_stage_derivatives, candidate_stage_derivatives, h):
        """Same-state/same-input replay, weighted by original DOP853 b."""
        from scipy.integrate._ivp.rk import DOP853
        cpu, gpu = fp64_arrays(original_stage_derivatives, candidate_stage_derivatives)
        if cpu.ndim != 2 or cpu.shape[0] != len(DOP853.B) or not np.isfinite(h) or h <= 0:
            raise ContractError('Original accepted step and all DOP853 b stages required')
        scale = self.scale(solver, unit, original_state, candidate_state)
        if scale.shape != cpu.shape[1:]:
            raise ContractError('RHS replay state shape differs')
        drift = h*np.sum(np.abs(DOP853.B)[:,None]*np.abs(gpu-cpu), axis=0)
        ratios = drift/scale
        return dict(status='PASS' if np.all(ratios <= .01) else 'FAIL_RHS_LOCAL_SCREEN',
            maximum_weighted_drift_over_original_scale=float(np.max(ratios, initial=0.)),
            h_seconds=float(h), global_error_proof=False)


def compare_probability_interval(uniform, original_probability, candidate_probability, low, high):
    """A draw inside the outward interval cannot receive a pathwise PASS."""
    u,p,q,lo,hi = fp64_arrays(uniform, original_probability, candidate_probability, low, high)
    if np.any((u < 0)|(u >= 1)) or np.any((lo < 0)|(hi > 1)|(lo > hi)):
        raise ContractError('Invalid probability interval/witness')
    changed = (u < p) != (u < q)
    outside = (p < lo)|(p > hi)|(q < lo)|(q > hi)
    ambiguous = (u >= lo)&(u <= hi)
    return dict(status='FAIL_STOCHASTIC_DECISION' if np.any(changed|outside) else
        'EVENT_AMBIGUOUS' if np.any(ambiguous) else 'PASS',
        count_changes=int(np.count_nonzero(changed)),
        outside_probability_interval=int(np.count_nonzero(outside)),
        ambiguous_thresholds=int(np.count_nonzero(ambiguous)))
