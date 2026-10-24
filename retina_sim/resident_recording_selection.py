"""Opt-in acceleration of existing recording selection; no cells/states removed.

RecordingPlan.indices retains its original source and ordering. Its membership
operand is an immutable tuple with the same serialized values and exact set
lookup semantics for string cell IDs, instead of a repeated linear tuple scan.
"""
from dataclasses import replace
from pathlib import Path
import hashlib
from .contracts import ContractError,canonical_hash
from .recording import RecordingPlan


class IndexedCellIds(tuple):
    def __new__(cls,values):
        values=tuple(values)
        if any(not isinstance(v,str) for v in values):
            raise ContractError('Indexed recording requires explicit string cell IDs')
        result=super().__new__(cls,values)
        object.__setattr__(result,'_lookup',frozenset(values))
        return result

    def __contains__(self,value):
        try:return value in self._lookup
        except TypeError:return tuple.__contains__(self,value)

    def __setattr__(self,name,value):
        raise AttributeError('Indexed recording cell IDs are immutable')

    def __reduce_ex__(self,protocol):
        return type(self),(tuple(self),)


def indexed_recording_plan(plan):
    if type(plan) is not RecordingPlan:
        raise ContractError('Original explicit RecordingPlan required')
    if plan.cell_ids is None or isinstance(plan.cell_ids,IndexedCellIds):
        return plan
    return replace(plan,cell_ids=IndexedCellIds(plan.cell_ids))


def selection_identity(plan):
    return dict(strategy='indexed_tuple_v1' if isinstance(plan.cell_ids,IndexedCellIds) else 'original_tuple',
        logical_plan_hash=canonical_hash(plan),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
