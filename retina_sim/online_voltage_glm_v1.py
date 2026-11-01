"""Stateful equivalent V_GLM30 readout; no membrane integration or spike feedback.

Native samples are 100us; a probability at a bin start describes its next1ms.
Sampled bin outcomes are pending until that bin closes. They are not exact AP times.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .stimulus_baselines_v1 import history_basis, stimulus_basis

FloatArray = NDArray[np.float64]
NATIVE_DT, BIN_DT = .0001, .001


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def vector(value: ArrayLike, shape: tuple[int,...]) -> FloatArray:
    result=np.array(value,dtype=np.float64,copy=True)
    if result.shape!=shape or not np.isfinite(result).all():raise ValueError('Finite exact array layout required')
    return result


@dataclass(frozen=True)
class MembraneParameters:
    C_total_F: float
    gL_total_S: float
    EL_V: float
    EE_V: float
    EI_V: float
    group_id: str
    members: int

    def __post_init__(self) -> None:
        values=(self.C_total_F,self.gL_total_S,self.EL_V,self.EE_V,self.EI_V)
        if not all(math.isfinite(x) for x in values) or min(values[:2])<=0:
            raise ValueError('Finite source-bound SI membrane parameters required')
        if not self.group_id or type(self.members) is not int or self.members<1:raise ValueError('Actual representative group required')


def membrane_outputs(p: MembraneParameters, voltage_V: ArrayLike, gE_S: ArrayLike, gI_S: ArrayLike) -> dict[str,FloatArray]:
    v,ge,gi=np.broadcast_arrays(np.asarray(voltage_V,dtype=float),np.asarray(gE_S,dtype=float),np.asarray(gI_S,dtype=float))
    if not all(np.isfinite(x).all() for x in (v,ge,gi)) or np.any(ge<0) or np.any(gi<0):raise ValueError('Finite V/S inputs required')
    il=p.gL_total_S*(p.EL_V-v);ie=ge*(p.EE_V-v);ii=gi*(p.EI_V-v)
    total=il+ie+ii
    return dict(V_V=v.copy(),gE_S=ge.copy(),gI_S=gi.copy(),gA_S=np.zeros_like(v),I_L_A=il,I_E_A=ie,I_I_A=ii,
        I_A_A=np.zeros_like(v),I_membrane_A=total,tau_eff_s=p.C_total_F/(p.gL_total_S+ge+gi),dVdt_V_per_s=total/p.C_total_F)


@dataclass(frozen=True)
class BinReadout:
    bin_index: int
    start_s: float
    D: float
    H: FloatArray
    lambda_Hz: FloatArray
    probability: FloatArray
    sampled_bin_outcome: NDArray[np.bool_] | None
    membrane: dict[str,float]


class OnlineVoltageGLM:
    """Original16 FIRs +200ms statistical history, with inspectable delayed commits.

    `completed_spikes` is accepted only for bins ending within this packet.
    Free mode never accepts observed spikes. gA=0, no reset or hard refractory.
    """
    def __init__(self, model: Mapping[str,object], membrane: MembraneParameters, *, mode: str,
                 paths: int, seed: int, source_id: str, origin_s: float=0.) -> None:
        if mode not in ('conditional','free') or type(paths) is not int or not 1<=paths<=4096:
            raise ValueError('Explicit conditional/free mode and1..4096 statistical paths required')
        if not source_id or not math.isfinite(origin_s) or type(seed) is not int or seed<0:raise ValueError('Explicit source/clock/seed required')
        if model.get('family')!='GLM' or model.get('parameters')!=30:raise ValueError('Frozen V_GLM30 required')
        self.theta=vector(model['theta'],(30,));self.mean=vector(model['mean'],(29,));self.scale=vector(model['scale'],(29,))
        if np.any(self.scale<=0):raise ValueError('Original positive training scales required')
        self.basis=stimulus_basis();self.history_basis=history_basis()
        self.kernel=self.history_basis@(self.theta[17:]/self.scale[16:])
        self.weights=np.asarray([self.kernel[(p-1-np.arange(200))%200] for p in range(200)])
        for value in (self.theta,self.mean,self.scale,self.basis,self.history_basis,self.kernel,self.weights):value.flags.writeable=False
        self.mode,self.paths,self.seed,self.source_id,self.origin_s=mode,paths,seed,source_id,float(origin_s)
        self.membrane=membrane;self.rng=np.random.Generator(np.random.PCG64(seed))
        self.native_index=0;self.completed_bins=0;self.pending_bin: int | None=None
        self.pending_spikes: NDArray[np.bool_] | None=None
        self.initial_V: float | None=None;self.voltage_tail=np.zeros(0);self.history=np.zeros((paths,200))
        self.last_native: dict[str,float] | None=None;self.last_readout: BinReadout | None=None
        self.identity=digest(dict(model=dict(model),membrane=asdict(membrane),mode=mode,paths=paths,seed=seed,source_id=source_id,
            origin_s=origin_s,native_dt=NATIVE_DT,bin_dt=BIN_DT,adaptation_enabled=False,reset_enabled=False,hard_refractory=False,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            basis_sha256=hashlib.sha256(self.basis.tobytes()+self.history_basis.tobytes()).hexdigest()))

    def _bits(self, value: ArrayLike) -> NDArray[np.bool_]:
        a=np.asarray(value)
        if a.shape!=(self.paths,) or not np.isin(a,(0,1)).all():raise ValueError('One binary outcome per statistical path required')
        return a.astype(bool,copy=True)

    def _close(self, observed: ArrayLike | None) -> None:
        if self.pending_bin is None:raise ValueError('No open bin to close')
        if self.mode=='free':
            if observed is not None:raise ValueError('Free mode cannot use observed spikes')
            if self.pending_spikes is None:raise RuntimeError('Missing sampled pending outcome')
            bits=self.pending_spikes
        else:
            if observed is None:raise ValueError('Completed past bin observation required')
            bits=self._bits(observed)
        self.history[:,self.pending_bin%200]=bits
        self.completed_bins+=1;self.pending_bin=None;self.pending_spikes=None

    def advance(self, voltage_V: ArrayLike, gE_S: ArrayLike, gI_S: ArrayLike, *,
                start_s: float, completed_spikes: Mapping[int,ArrayLike] | None=None) -> list[BinReadout]:
        v=np.asarray(voltage_V,dtype=float);ge=np.asarray(gE_S,dtype=float);gi=np.asarray(gI_S,dtype=float)
        if v.ndim!=1 or ge.shape!=v.shape or gi.shape!=v.shape or not all(np.isfinite(x).all() for x in (v,ge,gi)):
            raise ValueError('Equal finite100us V/S vectors required')
        if np.any(ge<0) or np.any(gi<0):raise ValueError('Negative conductance')
        if start_s!=self.origin_s+self.native_index*NATIVE_DT:raise ValueError('Native node skipped, repeated or wrong phase')
        nodes=np.arange(self.native_index,self.native_index+len(v));positions=np.flatnonzero(nodes%10==0)
        needed={int(nodes[i]//10-1) for i in positions if nodes[i]>0}
        if positions.size and self.pending_bin is None:needed.discard(int(nodes[positions[0]]//10-1))
        supplied={} if completed_spikes is None else dict(completed_spikes)
        if self.mode=='free' and supplied:raise ValueError('Free mode rejects observed spike history')
        if self.mode=='conditional':
            if set(supplied)!=needed:raise ValueError('Only precisely completed past bins may be supplied')
            supplied={k:self._bits(value) for k,value in supplied.items()}
        if not len(v):return []
        if self.initial_V is None:self.initial_V=float(v[0])
        # Sum each causal FIR window in the original np.convolve dot order.
        # lfilter's block zi addition changes rounding; tiny training scales can
        # amplify that difference. No samples after the predicted node are read.
        context=np.concatenate((self.voltage_tail,v-self.initial_V));offset=len(self.voltage_tail)
        records=[]
        for i in positions:
            n=int(nodes[i]//10)
            if self.pending_bin is not None:self._close(supplied.get(n-1))
            if self.completed_bins!=n:raise RuntimeError('History/bin clock differs')
            stop=offset+i+1;size=min(5000,stop);recent=context[stop-size:stop]
            features=np.asarray([np.correlate(recent,b[:size][::-1],'valid')[0]+self.initial_V*b.sum() for b in self.basis.T])
            D=float(self.theta[0]+((features-self.mean[:16])/self.scale[:16])@self.theta[1:17])
            H=self.history@self.weights[n%200]-(self.mean[16:]/self.scale[16:])@self.theta[17:]
            lam=90.*np.logaddexp(0.,D+H);probability=-np.expm1(-BIN_DT*lam)
            if not np.isfinite(lam).all() or not np.all((probability>=0)&(probability<=1)):raise ValueError('Original probability domain failed')
            sampled=self.rng.random(self.paths)<probability if self.mode=='free' else None
            physical={k:float(value) for k,value in membrane_outputs(self.membrane,v[i],ge[i],gi[i]).items()}
            self.pending_bin=n;self.pending_spikes=sampled.copy() if sampled is not None else None
            record=BinReadout(n,self.origin_s+n*BIN_DT,D,H,lam,probability,sampled,physical)
            self.last_readout=record;records.append(record)
        self.native_index+=len(v)
        self.voltage_tail=context[-4999:].copy()
        self.last_native={k:float(value) for k,value in membrane_outputs(self.membrane,v[-1],ge[-1],gi[-1]).items()}
        self.last_native['time_s']=self.origin_s+(self.native_index-1)*NATIVE_DT
        return records

    def step_native(self, voltage_V: float, gE_S: float, gI_S: float, *, time_s: float,
                    completed_spike: ArrayLike | None=None) -> BinReadout | None:
        supplied={self.pending_bin:completed_spike} if completed_spike is not None and self.pending_bin is not None else None
        if completed_spike is not None and supplied is None:raise ValueError('No previous bin accepts this observation')
        records=self.advance([voltage_V],[gE_S],[gI_S],start_s=time_s,completed_spikes=supplied)
        return records[0] if records else None

    def close_final_bin(self, *, end_s: float, completed_spike: ArrayLike | None=None) -> None:
        if self.native_index%10 or end_s!=self.origin_s+self.native_index*NATIVE_DT:
            raise ValueError('A bin closes only at its exact1ms boundary')
        self._close(completed_spike)

    def observable_state(self, path: int=0) -> dict[str,object]:
        if type(path) is not int or not 0<=path<self.paths:raise ValueError('Existing statistical path required')
        last=self.last_readout
        prediction=None if last is None else dict(bin_index=last.bin_index,start_s=last.start_s,end_s=last.start_s+BIN_DT,
            D_dimensionless=last.D,H_dimensionless=float(last.H[path]),lambda_Hz=float(last.lambda_Hz[path]),
            probability=float(last.probability[path]),expected_bin_count_rate_Hz=float(last.probability[path]/BIN_DT),
            sampled_bin_outcome=None if last.sampled_bin_outcome is None else bool(last.sampled_bin_outcome[path]))
        order=(self.completed_bins-1-np.arange(200))%200
        return dict(identity=self.identity,source_id=self.source_id,group_id=self.membrane.group_id,members=self.membrane.members,
            statistical_paths=self.paths,selected_path=path,path_role='stochastic_trial_not_cell_member',native_index=self.native_index,
            next_native_sample_s=self.origin_s+self.native_index*NATIVE_DT,last_native=deepcopy(self.last_native),last_prediction=prediction,
            completed_bins=self.completed_bins,completed_history_newest_first=self.history[path,order].tolist(),pending_bin=self.pending_bin,
            pending_outcome_is_forecast_not_completed_observation=True,adaptation_enabled=False,gA_S=0.,reset_enabled=False,hard_refractory=False)

    def checkpoint(self) -> dict[str,object]:
        last=None
        if self.last_readout is not None:
            r=self.last_readout
            last=dict(bin_index=r.bin_index,start_s=r.start_s,D=r.D,H=r.H.tolist(),lambda_Hz=r.lambda_Hz.tolist(),
                probability=r.probability.tolist(),sampled_bin_outcome=None if r.sampled_bin_outcome is None else r.sampled_bin_outcome.tolist(),membrane=r.membrane)
        payload=dict(identity=self.identity,mode=self.mode,paths=self.paths,seed=self.seed,source_id=self.source_id,origin_s=self.origin_s,
            native_index=self.native_index,completed_bins=self.completed_bins,pending_bin=self.pending_bin,
            pending_spikes=None if self.pending_spikes is None else self.pending_spikes.tolist(),initial_V=self.initial_V,
            voltage_tail=self.voltage_tail.tolist(),history=self.history.tolist(),rng=deepcopy(self.rng.bit_generator.state),last_native=self.last_native,last_readout=last)
        return dict(schema='online-voltage-glm-checkpoint-v1',payload=payload,payload_sha256=digest(payload))

    @classmethod
    def from_checkpoint(cls, model: Mapping[str,object], membrane: MembraneParameters, checkpoint: Mapping[str,object], *,
                        source_id: str) -> 'OnlineVoltageGLM':
        if checkpoint.get('schema')!='online-voltage-glm-checkpoint-v1':raise ValueError('Checkpoint schema differs')
        p=checkpoint['payload']
        if not isinstance(p,dict) or digest(p)!=checkpoint['payload_sha256'] or p['source_id']!=source_id:raise ValueError('Checkpoint identity/integrity differs')
        result=cls(model,membrane,mode=p['mode'],paths=p['paths'],seed=p['seed'],source_id=source_id,origin_s=p['origin_s'])
        if result.identity!=p['identity']:raise ValueError('Model/basis/physical/runtime contract changed')
        n,closed,pending=p['native_index'],p['completed_bins'],p['pending_bin']
        if type(n) is not int or n<0 or type(closed) is not int or closed<0:raise ValueError('Invalid clock counter')
        computed=(n+9)//10
        if closed!=computed-int(pending is not None) or (pending is not None and (type(pending) is not int or pending!=closed)):
            raise ValueError('History and native phase differ')
        result.native_index=n;result.completed_bins=closed;result.pending_bin=pending
        result.voltage_tail=vector(p['voltage_tail'],(min(n,4999),));result.history=vector(p['history'],(result.paths,200))
        if not np.isin(result.history,(0,1)).all():raise ValueError('History must contain completed binary bins')
        result.initial_V=p['initial_V']
        if (n==0)!=(result.initial_V is None) or (n and not math.isfinite(result.initial_V)):raise ValueError('Constant prehistory differs')
        if result.mode=='free' and pending is not None:result.pending_spikes=result._bits(p['pending_spikes'])
        elif p['pending_spikes'] is not None:raise ValueError('Unexpected pending outcome')
        if p['rng'].get('bit_generator')!='PCG64':raise ValueError('Original PCG64 required')
        result.rng.bit_generator.state=p['rng'];result.last_native=p['last_native']
        last=p['last_readout']
        if last is not None:
            h=vector(last['H'],(result.paths,));lam=vector(last['lambda_Hz'],(result.paths,));q=vector(last['probability'],(result.paths,))
            if np.any(lam<0) or np.any((q<0)|(q>1)) or not math.isfinite(last['D']):raise ValueError('Invalid statistical readout')
            bits=None if last['sampled_bin_outcome'] is None else result._bits(last['sampled_bin_outcome'])
            result.last_readout=BinReadout(last['bin_index'],last['start_s'],last['D'],h,lam,q,bits,last['membrane'])
        return result
