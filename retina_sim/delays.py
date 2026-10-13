"""Method-of-steps edge delays with explicit prehistory and discontinuity sides."""
from dataclasses import dataclass
import numpy as np
from .contracts import ContractError,Quantity

@dataclass
class EdgeDelayBuffer:
    retention_seconds: float
    times: list
    values: list
    prehistory: Quantity
    origin_seconds: float
    interpolation: str
    jumps: list
    max_samples: int = 4096
    def __post_init__(self):
        if self.retention_seconds<=0 or self.interpolation not in ('linear','zero_order_hold'):raise ContractError('Positive explicit delay/retention and interpolation required')
        if self.prehistory.values.ndim!=1 or len(self.times)!=len(self.values) or not self.times or np.any(np.diff(self.times)<0):raise ContractError('Delay buffer time/value layout invalid')
    def append(self,t,q):
        q.require(self.prehistory.unit,self.prehistory.meaning,self.prehistory.values.shape)
        if q.evidence_tag!=self.prehistory.evidence_tag:raise ContractError('Delayed signal evidence changed')
        if t<self.times[-1]-1e-13:raise ContractError('Delay history time moved backwards')
        if abs(t-self.times[-1])<=1e-13:
            if np.array_equal(q.values,np.asarray(self.values[-1])):return
            # Duplicate timestamp preserves departing left and arriving right value.
            if len(self.times)>1 and self.times[-2]==self.times[-1]:self.values[-1]=q.values.tolist()
            else:self.times.append(float(t));self.values.append(q.values.tolist())
            if not self.jumps or self.jumps[-1]!=float(t):self.jumps.append(float(t))
        else:self.times.append(float(t));self.values.append(q.values.tolist())
        if len(self.times)>self.max_samples:raise ContractError('Explicit delay history budget exhausted; stream on declared grid or enlarge compile budget, no silent history loss')
    def query_edges(self,t,delays,columns,*,left_limit=False):
        target=float(t)-np.asarray(delays);columns=np.asarray(columns,dtype=int)
        if left_limit:target=np.nextafter(target,-np.inf)
        times=np.asarray(self.times);values=np.asarray(self.values);out=np.empty(len(target))
        old=target<self.origin_seconds;out[old]=self.prehistory.values[columns[old]]
        active=~old;query=target[active];cols=columns[active]
        if len(query):
            if np.any(query<times[0]-1e-12) or np.any(query>times[-1]+1e-12):raise ContractError('Delay history unavailable; no current/future or substituted-background fallback')
            query=np.minimum(query,times[-1]);lo=np.searchsorted(times,query,side='right')-1;lo=np.maximum(lo,0)
            result=values[lo,cols]
            if self.interpolation=='linear':
                hi=np.minimum(lo+1,len(times)-1);span=times[hi]-times[lo];weight=np.divide(query-times[lo],span,out=np.zeros_like(query),where=span>0)
                result=result+weight*(values[hi,cols]-result)
            out[active]=result
        return out
    def prune(self,t):
        cutoff=t-self.retention_seconds
        # Keep the last right-side sample before cutoff and every later sample.
        remove=max(0,int(np.searchsorted(self.times,cutoff,side='left'))-1)
        if remove:self.times=self.times[remove:];self.values=self.values[remove:]
        self.jumps=[x for x in self.jumps if x+self.retention_seconds>t+1e-13]
    def next_arrival(self,t,delays):
        future=[float(j+d) for j in self.jumps for d in np.unique(delays) if j+d>t+1e-12]
        return min(future) if future else np.inf
    def payload(self):
        return {'retention_seconds':self.retention_seconds,'times':self.times,'values':self.values,'prehistory':{'values':self.prehistory.values.tolist(),'unit':self.prehistory.unit,'meaning':self.prehistory.meaning,'evidence_tag':self.prehistory.evidence_tag},'origin_seconds':self.origin_seconds,'interpolation':self.interpolation,'jumps':self.jumps,'max_samples':self.max_samples}
    @classmethod
    def from_payload(cls,p):return cls(p['retention_seconds'],list(p['times']),[list(v) for v in p['values']],Quantity(**p['prehistory']),p['origin_seconds'],p['interpolation'],list(p['jumps']),p['max_samples'])

@dataclass(frozen=True)
class DelayedEdgeOperator:
    key: str
    rows: np.ndarray
    columns: np.ndarray
    weights: np.ndarray
    delays_seconds: np.ndarray
    shape: tuple
    sampling_seconds: float
    interpolation: str
    prehistory: Quantity
    max_samples: int
    def initialize(self,t):return EdgeDelayBuffer(float(self.delays_seconds.max()),[float(t)],[self.prehistory.values.tolist()],self.prehistory,float(t),self.interpolation,[],self.max_samples)
    def apply(self,t,current_source,buffer,*,left_limit=False):
        delayed=self.delays_seconds>0;values=np.zeros(len(self.weights))
        if np.any(delayed):values[delayed]=buffer.query_edges(t,self.delays_seconds[delayed],self.columns[delayed],left_limit=left_limit)
        if np.any(~delayed):
            if current_source is None:raise ContractError('Zero-delay source missing from algebraic closure')
            values[~delayed]=current_source.values[self.columns[~delayed]]
        result=np.zeros(self.shape[0]);np.add.at(result,self.rows,self.weights*values);return result
    @property
    def has_instantaneous_edges(self):return bool(np.any(self.delays_seconds==0))

def clock_contract(components,links,policy=None):
    """Commensurate sample clock; continuous dynamics remain joint method-of-steps."""
    periods={cid:c.time_contract.fixed_step_seconds for cid,c in components.items() if c.time_contract.kind=='causal_history'}
    periods.update({cid:c.native_model.dt for cid,c in components.items() if hasattr(c.native_model,'sample_spikes')})
    periods.update({cid:c.time_contract.fixed_step_seconds for cid,c in components.items() if c.time_contract.kind=='author_discrete'})
    event_clock=policy is not None and policy.get('kind')=='explicit_event_clock'
    if event_clock:
        if set(policy)!={'kind','base_step_seconds','sampled_output_hold'} or policy['sampled_output_hold']!='zero_order_hold' or policy['base_step_seconds']<=0:raise ContractError('Explicit event clock contract invalid')
    history_base=min(periods.values()) if periods else None
    if event_clock:history_base=min(periods.values()) if periods else None
    elif len(set(periods.values()))>1:
        if policy is None or policy.get('kind')!='commensurate_minimum_step' or policy.get('sampled_output_hold')!='zero_order_hold':raise ContractError('Multi-rate histories require explicit commensurate_minimum_step/ZOH policy')
        base=policy.get('base_step_seconds')
        if base is None or base<=0:raise ContractError('Explicit positive shared clock step required')
        history_base=float(base)
    elif policy is not None:
        if set(policy)!={'kind','base_step_seconds','sampled_output_hold'} or policy['kind']!='commensurate_minimum_step' or policy['sampled_output_hold']!='zero_order_hold' or policy['base_step_seconds']<=0:raise ContractError('Explicit shared clock contract invalid')
        history_base=float(policy['base_step_seconds'])
    strides={}
    if history_base is not None and not event_clock:
        for cid,period in periods.items():
            ratio=period/history_base;rounded=int(round(ratio))
            if rounded<1 or abs(ratio-rounded)>1e-9:raise ContractError('Unsupported noncommensurate rate; no implicit period rounding or arbitrary resampling')
            strides[cid]=rounded
    delays=[x.delayed_operator for x in links if x.delayed_operator is not None]
    execution_steps=[x.sampling_seconds for x in delays]
    minimum_delay=min((float(x.delays_seconds[x.delays_seconds>0].min()) for x in delays),default=None)
    execution_step=min(execution_steps+[minimum_delay]) if delays else None
    return {'history_base_seconds':history_base,'component_periods_seconds':periods,'component_tick_strides':strides,'delayed_source_step_seconds':execution_step,'minimum_explicit_edge_delay_seconds':minimum_delay,'sampled_output_hold':'zero_order_hold','continuous_delay_interpolation':{x.key:x.interpolation for x in delays},'scheduler_kind':'explicit_event_clock' if event_clock else 'commensurate_minimum_step','rate_scope':'Exact per-component timestamp events; author-discrete interval-start ZOH preserved' if event_clock else 'Commensurate explicit periods; author-discrete interval-start ZOH preserved'}
