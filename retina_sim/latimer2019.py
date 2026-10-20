"""Exact public MAT intake and conditional retained-RGC assay computations.

The file has no unit/index metadata. pA and MATLAB one-based sample indices are
explicit project interpretations, not falsely verified file declarations. The
paper's +/-70mV approximate current conversion is an observation transform,
not a new synaptic mechanism or a simultaneous voltage-clamp/spike trial.
"""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import hashlib,itertools,json
import numpy as np
from scipy.io import loadmat
from scipy.special import expit
from .contracts import ContractError,Quantity
from .mechanism_contracts import EvidenceValue,MechanismProfile
from .inner import HistoryRGC

MAT_SHA256='59e1457bc27f7392a3c458902941a234c687b26a525b26d168bc8609aae40025'
MAT_MD5='83be38d612d7e31bdf38b731cd2c3ebf'
SAMPLE_DT=.0001


def read_cell6(root):
    path=Path(root)/'data/source_data/latimer2019_figshare_v1/Cell_6.mat'
    raw=path.read_bytes()
    if len(raw)!=17007338 or hashlib.sha256(raw).hexdigest()!=MAT_SHA256 or hashlib.md5(raw).hexdigest()!=MAT_MD5:
        raise ContractError('Frozen Latimer Cell_6 source identity differs')
    data=loadmat(path)
    fields={k for k in data if not k.startswith('__')}
    if fields!={'ExcCurrents','ExcStm','InhCurrents','InhStm','SpikeStm','SpikeTimes','fn'}:
        raise ContractError('Unknown Latimer file schema')
    for key,n in (('ExcCurrents',30),('ExcStm',30),('InhCurrents',8),('InhStm',8),('SpikeStm',30)):
        if data[key].shape!=(n,71000) or data[key].dtype!=np.float64 or not np.isfinite(data[key]).all():
            raise ContractError('Source numeric array shape/type changed: '+key)
        data[key].setflags(write=False)
    if data['SpikeTimes'].shape!=(1,30):raise ContractError('Expected 30 separate source spike records')
    for a in data['SpikeTimes'].flat:
        a=np.asarray(a).reshape(-1)
        if not np.array_equal(a,np.round(a)) or np.any(np.diff(a)<=0) or np.any(a<1) or np.any(a>71000):
            raise ContractError('Spike indices are not supported increasing source sample indices')
    return data


def stimulus_groups(data,key):
    groups={}
    for j,row in enumerate(data[key]):groups.setdefault(hashlib.sha256(row.tobytes()).hexdigest(),[]).append(j)
    return groups


def paired_cases(data,*,acknowledge_conditional_units=False,index_origin=1):
    if not acknowledge_conditional_units or index_origin not in (0,1):
        raise ContractError('Explicit conditional pA and sample-index-origin interpretation required')
    groups={k:stimulus_groups(data,k) for k in ('ExcStm','InhStm','SpikeStm')}
    common=sorted(set(groups['ExcStm'])&set(groups['InhStm'])&set(groups['SpikeStm']))
    if len(common)!=8:raise ContractError('Expected 8 complete cross-mode stimulus groups')
    result=[]
    for group_id in common:
        e=groups['ExcStm'][group_id];i=groups['InhStm'][group_id];s=groups['SpikeStm'][group_id]
        u=data['SpikeStm'][s[0]]
        if any(not np.array_equal(u,data[k][j]) for k,ids in (('ExcStm',e),('InhStm',i),('SpikeStm',s)) for j in ids):
            raise ContractError('Stimulus hashes did not identify exactly equal samples')
        # Actual constant prefix/suffix, not an invented experimental event time.
        if not np.all(u[:10000]==u[0]) or not np.all(u[70000:]==u[0]):
            raise ContractError('Expected source constant first1s/last100ms context')
        ge=np.mean(data['ExcCurrents'][e],axis=0)*1e-12/(-.070)
        gi=np.mean(data['InhCurrents'][i],axis=0)*1e-12/(+.070)
        if np.any(ge<0) or np.any(gi<0):raise ContractError('Converted input contains negative conductance; no clipping')
        counts=[]
        for j in s:
            ind=np.asarray(data['SpikeTimes'].flat[j]).reshape(-1).astype(int)-index_origin
            y=np.bincount(ind//10,minlength=7100)
            if y.shape!=(7100,) or np.any(y>1):raise ContractError('Observed counts cannot be represented by the retained 1ms Bernoulli sampler')
            counts.append(y)
        result.append(dict(stimulus_sha256=group_id,exc_trial_indices_zero_based=e,inh_trial_indices_zero_based=i,
            spike_trial_indices_zero_based=s,ge_S=ge,gi_S=gi,spike_counts=np.asarray(counts),
            stimulus_values=u,simultaneous_same_trial_claim=False))
    return result,groups


def reference_rgc(root):
    path=Path(root)/'configurations/effective_dynamics_v1.json';raw=path.read_bytes()
    definitions=json.loads(raw)['profiles']['effective_parasol_RGC_ON'];records=[]
    for key,row in definitions.items():
        if key=='V0':value=np.array([definitions['EL']['value']])
        elif key=='initial_spike_history':value=np.zeros((3,1))
        elif row['binding']=='broadcast':value=np.array([row['value']])
        else:value=np.asarray(row['value'],float)
        records.append(EvidenceValue(key,Quantity(value,row['unit'],row['meaning'],'project_approximation'),
            row['evidence_class'],row['source_id'],'Unchanged effective_parasol_RGC_ON profile; '+key,
            {'calibration_boundary':'Latimer conditional RGC module only'},'Retained project approximation'))
    profile=MechanismProfile('latimer_cell6_retained_RGC_reference','Macaque_effective_project_profile',
        {'base_profile':'effective_parasol_RGC_ON','base_profile_sha256':hashlib.sha256(raw).hexdigest()},
        tuple(records),tuple(sorted({r.source_id for r in records})),'project_approximation',
        'Explicit retained-project RGC reference; no scientific validation from identity')
    pop=SimpleNamespace(cells=SimpleNamespace(cell_id=('Latimer_Cell_6_060908Bc2',)))
    return HistoryRGC(profile,pop,'project_approximation'),pop


def parameterized_native(reference,pop,alpha,mu):
    records=tuple(replace(r,value=Quantity(np.array([alpha if r.parameter_id=='alpha' else mu]),r.value.unit,
        r.value.meaning,r.value.evidence_tag)) if r.parameter_id in ('alpha','mu') else r for r in reference.p.records)
    profile=replace(reference.p,profile_id='latimer_cell6_conditional_RGC_candidate',records=records,
        conditions=dict(reference.p.conditions,fit_scope='Conditional on declared source-unit/observation assumptions; not production adopted'))
    return HistoryRGC(profile,pop,reference.tag)


def voltage_zoh(native,ge,gi,dt=SAMPLE_DT):
    """Exact passive native ODE under declared ZOH input; no reset or new terms."""
    ge,gi=np.asarray(ge,float),np.asarray(gi,float)
    if ge.shape!=gi.shape or ge.ndim!=2 or ge.shape[1]%10 or np.any(ge<0) or np.any(gi<0):
        raise ContractError('Complete nonnegative (cases,100us intervals) conductance arrays required')
    G=native.gL[0]+ge+gi
    equilibrium=(native.gL[0]*native.EL[0]+ge*native.EE[0]+gi*native.EI[0])/G
    decay=np.exp(-dt*G/native.C[0]);V=np.full(ge.shape[0],native.V0[0]);samples=[]
    for j in range(ge.shape[1]):
        if j%10==0:samples.append(V.copy())
        V=equilibrium[:,j]+(V-equilibrium[:,j])*decay[:,j]
    return np.asarray(samples).T,V


def observed_history_bias(native,counts):
    y=np.asarray(counts);history=np.zeros_like(y,dtype=float)
    for lag,k in enumerate(native.history_kernel,1):history[:,lag:]+=k*y[:,:-lag]
    return history


def conditional_score(native,voltage,counts,mask):
    y=np.asarray(counts);bias=observed_history_bias(native,y)
    z=(np.asarray(voltage)+bias-native.mu[0])/native.beta[0]
    lam=native.alpha[0]*np.logaddexp(0.,z);a=lam*native.dt
    logp=np.log(-np.expm1(-a));loss=np.where(y,-logp,a)
    return dict(nll=float(loss[:,mask].sum()),bins=int(y[:,mask].size),events=int(y[:,mask].sum()),
        observed_rate_Hz=float(y[:,mask].mean()/native.dt),
        predicted_conditional_occupancy_rate_Hz=float((-np.expm1(-a[:,mask])).mean()/native.dt),
        mean_conditional_hazard_Hz=float(lam[:,mask].mean()))


def exact_mean_events(native,voltage):
    """Marginalize the native three-bin Bernoulli history, not observed history."""
    states=np.asarray(list(itertools.product((0,1),repeat=3)));idx={tuple(h):j for j,h in enumerate(states)}
    n0=np.asarray([idx[(0,h[0],h[1])] for h in states]);n1=np.asarray([idx[(1,h[0],h[1])] for h in states])
    if len(native.history_kernel)!=3:raise ContractError('This exact expectation is for the retained three-bin history')
    pi=np.zeros(8);pi[0]=1.;probabilities=[]
    for v in voltage:
        lam=native.intensity(np.full(8,v),states.T);q=-np.expm1(-lam*native.dt)
        probabilities.append(float(pi@q))
        pi=np.bincount(n0,weights=pi*(1-q),minlength=8)+np.bincount(n1,weights=pi*q,minlength=8)
    return np.asarray(probabilities)
