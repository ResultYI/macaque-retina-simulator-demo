"""Bounded selected recording, streaming windows, and explicit timestamped delay."""
from dataclasses import dataclass,field
from pathlib import Path
import numpy as np
from .contracts import ContractError,Quantity

@dataclass(frozen=True)
class RecordingPlan:
    ports: tuple | None = None
    states: tuple | None = None
    cell_ids: tuple | None = None
    max_bytes: int = 256*1024*1024
    def __post_init__(self):
        if self.max_bytes<=0: raise ContractError('Recording budget must be positive')
    def indices(self,cell_ids):
        return np.arange(len(cell_ids)) if self.cell_ids is None else np.array([i for i,x in enumerate(cell_ids) if x in self.cell_ids],dtype=int)

class NPZWindowRecorder:
    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.windows=0
    def consume(self,result):
        import json
        filename=self.directory/f'window_{self.windows:08d}.npz'
        arrays={name:q.values for name,q in result.outputs_with_units.items()}
        manifest={name:{'unit':q.unit,'meaning':q.meaning,'evidence_tag':q.evidence_tag} for name,q in result.outputs_with_units.items()}
        arrays['manifest_json']=np.array(json.dumps({'ports':manifest,'metadata':result.metadata},sort_keys=True))
        np.savez_compressed(filename,**arrays);self.windows+=1
        return filename

@dataclass
class TimestampedDelay:
    delay_seconds: float
    history_times: list
    history_values: list
    prehistory: Quantity
    interpolation: str
    prehistory_stop_seconds: float | None = None
    def __post_init__(self):
        if self.delay_seconds<0 or self.interpolation not in ('linear','zero_order_hold'): raise ContractError('Explicit sourced delay/interpolation required')
        if self.prehistory_stop_seconds is None:
            if not self.history_times:raise ContractError('Explicit delay prehistory boundary required')
            self.prehistory_stop_seconds=float(self.history_times[0])
    def append(self,t,q):
        q.require(self.prehistory.unit,self.prehistory.meaning,self.prehistory.values.shape)
        if self.history_times and t<=self.history_times[-1]: raise ContractError('Delay history timestamps must increase')
        self.history_times.append(float(t));self.history_values.append(q.values.tolist())
        while len(self.history_times)>2 and self.history_times[1]<t-self.delay_seconds:
            self.history_times.pop(0);self.history_values.pop(0)
    def at(self,t):
        target=t-self.delay_seconds
        if target<=self.prehistory_stop_seconds:return self.prehistory
        if not self.history_times or target<self.history_times[0]:raise ContractError('Delay history was pruned for this readout; no substituted background')
        times=np.array(self.history_times); values=np.array(self.history_values)
        if target>times[-1]: raise ContractError('Delay requires current timestamp, no future extrapolation')
        if self.interpolation=='zero_order_hold': a=values[min(np.searchsorted(times,target,side='right')-1,len(times)-1)]
        else: a=np.array([np.interp(target,times,values[:,k]) for k in range(values.shape[1])])
        return Quantity(a,self.prehistory.unit,self.prehistory.meaning,self.prehistory.evidence_tag)
    def payload(self):
        return {'delay_seconds':self.delay_seconds,'history_times':self.history_times,'history_values':self.history_values,'prehistory':{'values':self.prehistory.values.tolist(),'unit':self.prehistory.unit,'meaning':self.prehistory.meaning,'evidence_tag':self.prehistory.evidence_tag},'interpolation':self.interpolation,'prehistory_stop_seconds':self.prehistory_stop_seconds}
    @classmethod
    def from_payload(cls,d): return cls(d['delay_seconds'],list(d['history_times']),list(d['history_values']),Quantity(**d['prehistory']),d['interpolation'],d['prehistory_stop_seconds'])
