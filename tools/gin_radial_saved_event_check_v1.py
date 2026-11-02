"""Reconstruct only the original sampler from saved V; never integrate a model."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from retina_sim.inner import HistoryRGC
from retina_sim.contracts import canonical_hash
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.group_time_monitor import GroupTimeComparator,frozen_budget
from retina_sim.group_time_storage import load_bundle
from retina_sim.spatial_group_model import RGC
from tools.gin_radial_context_v1 import BASE,read,write

def check():
    seed=load_bundle(BASE/'background/seed',expected_kind='source_construction_seed')['payload'];parameters=seed['inner_parameters'];pops=parameters['populations']
    comparator=object.__new__(GroupTimeComparator);comparator.parameters=parameters
    comparator.monitor=SimpleNamespace(rows={(p,'RGC_spike_history','V'):dict(budget=frozen_budget(seed['states'][p,'RGC_spike_history','V'],'V')[2]) for p in RGC})
    output={};all_witness={};issues=[];worst=0.
    def issue(status,where,message):issues.append(dict(status=status,where=where,message=message))
    def numeric(name,a,b,budget):
        nonlocal worst
        error=abs(a-b);worst=max(worst,float(np.max(error/budget,initial=0)))
        if np.any(error>budget):issue('FAIL',name,'Original sampler numeric allocation exceeded')
    for precision in ('standard','tight'):
        aux=deepcopy(seed['spike_states']);witnesses=[];events=[]
        with np.load(BASE/f'forward/{precision}/NUMERICAL_STATES.npz',allow_pickle=False) as z:
            for i in range(51):
                t=float(z['time_seconds'][i*10]);before=deepcopy(aux);by_pop={};counts={p:len(seed['states'][p,'RGC_spike_history','V']) for p in RGC}
                for pop in RGC:
                    p=pops[pop];kernel=np.asarray(p['spike_history_kernel']);alpha=np.asarray(p['alpha']);mu=np.asarray(p['mu']);beta=np.asarray(p['beta'])
                    proxy=SimpleNamespace(n=counts[pop],dt=.001,history_kernel=kernel,intensity=lambda v,h,aa=alpha,mm=mu,bb=beta,kk=kernel:aa*np.logaddexp(0.,(v+kk@h-mm)/bb))
                    V=z['STATE__'+pop+'__RGC_spike_history__V'][:,i*10]
                    aux[pop],by_pop[pop]=HistoryRGC.sample_spikes(proxy,t,V,aux[pop])
                    events.extend(dict(population=pop,**e) for e in by_pop[pop] if e['time_seconds']<.05)
                witness=GroupTimeArm._draw_witness(SimpleNamespace(protocol={'clock_origin_seconds':0.},inner=SimpleNamespace(counts=counts)),t,before,aux,by_pop,{})
                witnesses.append(witness)
                for pop in RGC:aux[pop]['pending_events']=[e for e in aux[pop]['pending_events'] if e['time_seconds']>t]
        actual=read(BASE/f'forward/{precision}/GROUP_OBJECT_EVENTS.json')
        key=lambda e:(e['population'],e['cell_index'],e['bin_start_seconds'])
        reconstructed=sorted(events,key=key);actual=sorted(actual,key=key)
        if canonical_hash(reconstructed)!=canonical_hash(actual):raise ValueError('Original saved-event reconstruction is not exact: '+precision)
        end=read(BASE/f'forward/{precision}/END_HISTORY.json')
        if canonical_hash(aux)!=canonical_hash(end):raise ValueError('Saved end PCG/history/rate/pending reconstruction differs: '+precision)
        output[precision]=dict(reconstructed_events_exact=True,reconstructed_end_PCG_history_pending_exact=True,bins=51,events=len(events))
        all_witness[precision]=witnesses
    for a,b in zip(all_witness['standard'],all_witness['tight']):
        for pop in RGC:comparator._draw(pop,a[pop],b[pop],numeric,issue)
    a=sorted(read(BASE/'forward/standard/GROUP_OBJECT_EVENTS.json'),key=lambda e:(e['population'],e['cell_index'],e['bin_start_seconds']))
    b=sorted(read(BASE/'forward/tight/GROUP_OBJECT_EVENTS.json'),key=lambda e:(e['population'],e['cell_index'],e['bin_start_seconds']))
    logical_equal=len(a)==len(b) and all({k:v for k,v in x.items() if k!='time_seconds'}=={k:v for k,v in y.items() if k!='time_seconds'} for x,y in zip(a,b))
    if not logical_equal:issue('FAIL','events','Event identity/bin differs')
    differences=[abs(x['time_seconds']-y['time_seconds']) for x,y in zip(a,b)] if logical_equal else []
    result=dict(status='PASS' if not issues else ('FAIL' if any(i['status']=='FAIL' for i in issues) else 'INCONCLUSIVE'),
        method='Replay original HistoryRGC.sample_spikes using saved per-ms membraneV, original compatible seed and source clock; exact saved events and complete end auxiliaries first established, then original GroupTimeComparator._draw',
        upstream_integration_run=False,per_precision=output,original_sampler_witness_checks=255,issues=issues,maximum_rate_probability_budget_ratio=worst,
        event_identity_bins_equal=logical_equal,event_times_bitwise_equal=not any(differences),different_event_times=sum(d>0 for d in differences),
        maximum_event_time_difference_s=max(differences,default=0.),time_gate='Unchanged original lambda-derived interval, not exact timestamp equality or a new tolerance',
        source='retina_sim/group_time_monitor.py::GroupTimeComparator._draw',full60s_or600s_claimed=False)
    write(BASE/'SAVED_EVENT_CHECK.json',result);return result
