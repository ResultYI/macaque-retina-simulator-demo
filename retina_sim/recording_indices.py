"""Fixed RecordingPlan ID-index layouts without changing the original API.

Only immutable ID tuples/selections are cached. Strong tuple references prevent
identity reuse, the cache is bounded, and returned arrays remain independent.
The legacy RecordingPlan and its model/checkpoint source identity are retained.
"""
from pathlib import Path
import hashlib
import numpy as np
from .recording import RecordingPlan


class IndexedRecordingPlan(RecordingPlan):
    def __init__(self,plan):
        super().__init__(ports=plan.ports,states=plan.states,cell_ids=plan.cell_ids,max_bytes=plan.max_bytes)
        stable=(plan.cell_ids is None or
            (type(plan.cell_ids) in (tuple,frozenset) and all(type(gid) is str for gid in plan.cell_ids)))
        object.__setattr__(self,'_stable_selection',stable)
        object.__setattr__(self,'_index_cache',{})
        object.__setattr__(self,'_index_stats',{'hits':0,'misses':0,'uncached':0,'empty':0})
    def indices(self,cell_ids):
        if self.cell_ids is not None and not len(self.cell_ids):
            self._index_stats['empty']+=1
            return np.empty(0,dtype=int)
        if not self._stable_selection or type(cell_ids) is not tuple:
            self._index_stats['uncached']+=1
            return super().indices(cell_ids)
        key=id(cell_ids);cached=self._index_cache.get(key)
        if cached is not None and cached[0] is cell_ids:
            self._index_stats['hits']+=1
            return cached[1].copy()
        if any(type(gid) is not str for gid in cell_ids):
            self._index_stats['uncached']+=1
            return super().indices(cell_ids)
        value=super().indices(cell_ids);value.setflags(write=False)
        if len(self._index_cache)>=64:self._index_cache.pop(next(iter(self._index_cache)))
        self._index_cache[key]=(cell_ids,value);self._index_stats['misses']+=1
        return value.copy()
    @property
    def procedure_identity(self):
        return {'procedure':'fixed_recording_index_layout_v1',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'statistics':dict(self._index_stats),'cache_layouts':len(self._index_cache),
            'selection_order_units_or_output_values_changed':False}


def indexed_recording_plan(plan):
    return IndexedRecordingPlan(plan) if type(plan) is RecordingPlan else plan
