"""Complete group monitor and the original frozen numerical comparison budgets.

Per-cell statistics stay on their supplied device. Full export is explicit.
Neither source-construction seeds nor these reports qualify a background.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math

import numpy as np

from .contracts import ContractError, canonical_hash
from .background import NUMERICAL_UNIT_SCALES
from .resident_monitor import UNIT_INTERVAL_MEANINGS, NONNEGATIVE_MEANINGS
from .resident_derived_ports import compare_cell, load_addendum
from .spatial_group_model import BC, RGC

SCHEMA = 'retipath-complete-group-time-monitor-1'
PAIR_SCHEMA = 'retipath-group-time-frozen-pair-1'
CONTRACT_SHA256 = '3cfab9768a856c13354ab38cc65b506da60661fe3e3b0f05e46c647c08981a77'


def _contract():
    path=Path(__file__).resolve().parents[1]/'configurations/numerical_equivalence/resident_native_fp64_v1.json'
    raw=path.read_bytes()
    # Bind its actual immutable allocation identity; no scenario relabelling.
    value=json.loads(raw)
    if (hashlib.sha256(raw).hexdigest()!=CONTRACT_SHA256
            or value.get('contract_id')!='resident_native_fp64_error_equivalence_v1'
            or value.get('state_budget_fraction')!=.01 or value.get('roundoff_epsilon_multiplier')!=8192
            or {k:value['solver_required'][k] for k in ('rtol','atol','max_step')}!=dict(rtol=1e-6,atol=1e-9,max_step=.0001)):
        raise ContractError('Original numerical allocation changed')
    return value,hashlib.sha256(raw).hexdigest()


def _cpu(value):
    if hasattr(value,'detach'):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def _owned(value):
    source=_cpu(value)
    if source.dtype!=np.float64:raise ContractError('Original FP64 numeric witness required; no conversion')
    result=np.array(source,copy=True)
    if not np.isfinite(result).all():raise ContractError('Finite FP64 numeric witness required')
    return result


def _name(key):return ':'.join(key)


def frozen_budget(initial,unit,relative_tolerance=.0001):
    """Exactly the old B_i formula, frozen before either numerical arm runs."""
    if unit not in NUMERICAL_UNIT_SCALES:raise ContractError('Unregistered numerical unit: '+unit)
    entry=NUMERICAL_UNIT_SCALES[unit]
    scale=np.maximum(np.abs(_owned(initial)),entry['scale_floor'])
    den=entry['atol']+relative_tolerance*scale
    budget=.01*np.minimum(1e-9+1e-6*scale,den)
    for value in (scale,den,budget):value.flags.writeable=False
    return scale,den,budget


def rhs_allowance(scale,budget,reference,candidate):
    epsilon=8192*np.finfo(np.float64).eps
    return np.minimum(np.minimum(epsilon*scale/.0001,budget/.0001)+
        epsilon*np.maximum(np.abs(reference),np.abs(candidate)),budget/.0001)


def _period(block):
    if block.mechanism=='Chen2024PrimateRodCurrent':return .0001
    if block.population in BC or block.mechanism=='RGC_expected_history':return .001
    return None


class GroupTimeMonitor:
    def __init__(self,assembly,initial_states,*,parameters=None,torch=None,relative_tolerance=.0001):
        if relative_tolerance!=.0001:raise ContractError('Original monitor relative tolerance must remain1e-4')
        self.assembly,self.parameters,self.torch=assembly,parameters,torch
        self.contract,self.contract_hash=_contract()
        self.blocks=tuple((*assembly.state_blocks,*assembly.probability_blocks,*assembly.auxiliary_z1_blocks))
        self.specs={(b.population,b.mechanism,b.name):b for b in self.blocks}
        if len(self.specs)!=len(self.blocks) or set(initial_states)!=set(self.specs):
            raise ContractError('Complete original group monitor registry and initial state required')
        self.rows={};self.missing=[]
        for key,b in self.specs.items():
            initial=_owned(initial_states[key])
            if initial.shape!=(b.count,):raise ContractError('Frozen original initial layout differs: '+_name(key))
            try:scale,den,budget=frozen_budget(initial,b.unit)
            except ContractError:
                self.missing.append(dict(state=_name(key),unit=b.unit));scale=den=budget=None
            self.rows[key]=dict(spec=b,initial=initial,scale=scale,den=den,budget=budget,
                period=_period(b),lo=None,hi=None,rhs=None,defect=None,native_tensor=False)
        self.transition_keys=tuple(k for k,r in self.rows.items() if r['period'] is not None)
        # Private scratch never appears in evidence or checkpoint statistics.
        self._cpu_work=np.empty(max((b.count for b in self.blocks),default=0),dtype=np.float64)
        self.count=0;self.stamps=[];self.domain_failures=[]
        self.identity=canonical_hash(dict(schema=SCHEMA,artifact=assembly.artifact_sha256,
            contract_hash=self.contract_hash,states={_name(k):r['initial'] for k,r in self.rows.items()},
            unit_registry=NUMERICAL_UNIT_SCALES,relative_tolerance=relative_tolerance))

    def observe(self,t,side,states,rhs,*,transitions,ports=None,event_generation=0,aux=None,rhs_kind='original_equation'):
        if (not math.isfinite(t) or side not in ('left','right') or type(event_generation) is not int
                or event_generation<0 or rhs_kind!='original_equation'):
            raise ContractError('Exact stamp/side/generation and original-equation RHS required')
        if set(states)!=set(self.specs) or set(rhs)!=set(self.specs) or set(transitions)!=set(self.transition_keys):
            raise ContractError('Full state, true RHS and pure one-step transition coverage required')
        for key,row in self.rows.items():
            b=row['spec'];a=states[key];f=rhs[key]
            tensor=hasattr(a,'detach')
            if tensor:
                if self.torch is None or a.dtype!=self.torch.float64 or tuple(a.shape)!=(b.count,) or f.dtype!=a.dtype or f.device!=a.device:
                    raise ContractError('Original GPU monitor layout/device/FP64 differs: '+_name(key))
                finite=lambda x:bool(self.torch.isfinite(x).all().item())
                any_value=lambda x:bool(x.any().item())
                maximum=self.torch.maximum;minimum=self.torch.minimum
                absolute=lambda x:x.abs()
                copy=lambda x:x.clone()
                zero=lambda:self.torch.zeros_like(a)
            else:
                a,f=np.asarray(a),np.asarray(f)
                if a.dtype!=np.float64 or f.dtype!=np.float64 or a.shape!=(b.count,) or f.shape!=a.shape:
                    raise ContractError('Original CPU monitor layout/FP64 differs: '+_name(key))
                finite=lambda x:bool(np.isfinite(x).all())
                any_value=lambda x:bool(np.any(x))
                maximum,minimum=np.maximum,np.minimum
                absolute=np.abs;copy=lambda x:x.copy();zero=lambda:np.zeros_like(a)
            if not finite(a) or not finite(f):raise ContractError('Nonfinite original state/RHS: '+_name(key))
            if row['lo'] is not None and tensor!=row['native_tensor']:
                if tensor and row.pop('restored_cpu_statistics',False):
                    for field in ('lo','hi','rhs','defect'):
                        row[field]=self.torch.as_tensor(row[field],dtype=a.dtype,device=a.device).clone()
                else:raise ContractError('Monitor state changed storage device')
            row['native_tensor']=tensor
            if tensor:
                row['lo']=copy(a) if row['lo'] is None else minimum(row['lo'],a)
                row['hi']=copy(a) if row['hi'] is None else maximum(row['hi'],a)
                row['rhs']=absolute(f) if row['rhs'] is None else maximum(row['rhs'],absolute(f))
            else:
                # Statistics own their arrays. Keep the original ufunc order
                # while avoiding fresh full-state results on every observation.
                work=self._cpu_work[:b.count]
                if row['lo'] is None:row['lo']=copy(a)
                else:np.minimum(row['lo'],a,out=row['lo'])
                if row['hi'] is None:row['hi']=copy(a)
                else:np.maximum(row['hi'],a,out=row['hi'])
                if row['rhs'] is None:row['rhs']=absolute(f)
                else:
                    np.abs(f,out=work)
                    np.maximum(row['rhs'],work,out=row['rhs'])
            if row['defect'] is None:row['defect']=zero()
            if row['period'] is not None:
                transition=transitions[key]
                if tensor:
                    transition=self.torch.as_tensor(transition,dtype=a.dtype,device=a.device)
                else:transition=np.asarray(transition)
                if tuple(transition.shape)!=tuple(a.shape) or not finite(transition):raise ContractError('Pure transition layout differs: '+_name(key))
                if tensor or transition.dtype!=np.float64:
                    row['defect']=maximum(row['defect'],absolute(transition-a)/row['period'])
                else:
                    np.subtract(transition,a,out=work)
                    np.abs(work,out=work)
                    np.divide(work,row['period'],out=work)
                    np.maximum(row['defect'],work,out=row['defect'])
            meaning=b.meaning
            if b.unit=='dimensionless' and (meaning in UNIT_INTERVAL_MEANINGS or b.mechanism=='RGC_expected_history'):
                if any_value((a < -1e-10)|(a > 1+1e-10)):self.domain_failures.append(_name(key))
            if meaning in NONNEGATIVE_MEANINGS and any_value(a < -1e-10):self.domain_failures.append(_name(key)+':negative_domain')
            if b.name=='A' and b.population in BC:
                if self.parameters is None:raise ContractError('Actual BC adaptation scale binding required by full monitor')
                scale=self.parameters['populations'][b.population]['adaptation_scale']
                if any_value(a > np.asarray(scale)+1e-10):self.domain_failures.append(_name(key)+':above_scale')
        for pop in RGC:
            values=[states[(pop,'RGC_expected_history','history_probability_'+str(i))] for i in range(8)]
            mass=sum(values)
            if bool(np.any(np.abs(_cpu(mass)-1.)>1e-12)):self.domain_failures.append(pop+':history_probability_mass')
        if ports is not None:
            currents=[_cpu(v) for k,v in ports.items() if k[-1]=='gap_current']
            if currents:
                total=sum(float(np.sum(a)) for a in currents);magnitude=sum(float(np.sum(np.abs(a))) for a in currents)
                if abs(total)>1e-16+1e-10*magnitude:self.domain_failures.append('reciprocal_gap_current_conservation')
        if aux is not None:self.domain_failures.extend(_history_failures(aux,t))
        self.count+=1;self.stamps.append(dict(time_hex=float(t).hex(),side=side,event_generation=event_generation))

    def report(self,width):
        if not math.isfinite(width) or width<0 or not self.count:raise ContractError('Observed nonnegative monitor width required')
        result=[];worst=0.;dominant=None
        for key,row in self.rows.items():
            if row['den'] is None:continue
            lo,hi,rhs,defect=(_cpu(row[n]) for n in ('lo','hi','rhs','defect'))
            den=row['den'];norms=((hi-lo)/den,width*rhs/den,width*defect/den)
            labels=('excursion','original_rhs','discrete_one_step_defect')
            maxima=[float(np.max(a,initial=0.)) for a in norms]
            result.append(dict(state=_name(key),unit=row['spec'].unit,
                maximum_observed_excursion_norm=maxima[0],original_rhs_norm=maxima[1],discrete_defect_norm=maxima[2]))
            for label,array,value in zip(labels,norms,maxima):
                if value>worst:
                    index=int(np.argmax(array));worst=value
                    ids=self.assembly.populations[key[0]].group_ids
                    # Contact states have their own route-order identities.
                    cell_id=str(ids[index]) if len(ids)==len(array) else 'contact:'+str(index)
                    dominant=dict(state=_name(key),cell_index=index,cell_id=cell_id,criterion=label,norm=value)
        return dict(schema=SCHEMA,states=result,maximum_norm=worst,dominant=dominant,observations=self.count,
            domain_failures=sorted(set(self.domain_failures)),missing_scales=self.missing,
            passed=not self.missing and worst<=1 and not self.domain_failures,
            numerical_only=True,background_qualified=False,source_construction_background_qualification=False)

    def payload(self):
        if not self.count:raise ContractError('Cannot checkpoint an unobserved monitor as complete')
        payload=dict(schema=SCHEMA,identity=self.identity,contract_hash=self.contract_hash,
            background_qualified=False,count=self.count,stamps=list(self.stamps),domain_failures=list(self.domain_failures),
            rows={_name(k):dict(unit=r['spec'].unit,period=r['period'],initial=r['initial'].copy(),
                scale=None if r['scale'] is None else r['scale'].copy(),den=None if r['den'] is None else r['den'].copy(),
                budget=None if r['budget'] is None else r['budget'].copy(),
                **{n:_owned(r[n]) for n in ('lo','hi','rhs','defect')}) for k,r in self.rows.items()})
        payload['payload_sha256']=canonical_hash(payload)
        return payload

    @classmethod
    def from_payload(cls,assembly,initial_states,payload,*,parameters,torch=None):
        raw=dict(payload);expected=raw.pop('payload_sha256',None)
        if raw.get('schema')!=SCHEMA or expected!=canonical_hash(raw):
            raise ContractError('Complete monitor checkpoint payload identity differs')
        result=cls(assembly,initial_states,parameters=parameters,torch=torch)
        if raw['identity']!=result.identity or raw['contract_hash']!=result.contract_hash or raw.get('background_qualified') is not False:
            raise ContractError('Full monitor initial/contract identity or qualification differs')
        if set(raw['rows'])!={_name(k) for k in result.rows} or type(raw['count']) is not int or raw['count']!=len(raw['stamps']) or raw['count']<1:
            raise ContractError('Full monitor checkpoint coverage/stamps differ')
        for k,row in result.rows.items():
            saved=raw['rows'][_name(k)]
            if saved['unit']!=row['spec'].unit or saved['period']!=row['period']:
                raise ContractError('Full monitor state unit/period differs')
            for name in ('initial','scale','den','budget'):
                if canonical_hash(saved[name])!=canonical_hash(row[name]):raise ContractError('Frozen monitor reference changed')
            for name in ('lo','hi','rhs','defect'):
                value=_owned(saved[name])
                if value.shape!=(row['spec'].count,):raise ContractError('Full monitor statistic shape differs')
                row[name]=value
            row['restored_cpu_statistics']=True
        result.count=raw['count'];result.stamps=list(raw['stamps']);result.domain_failures=list(raw['domain_failures'])
        return result


def _history_failures(aux,t):
    failures=[]
    spikes=aux.get('spike_states',{}) if isinstance(aux,dict) else {}
    for pop,state in spikes.items():
        h=np.asarray(state['history']);counts=state.get('last_bin_counts')
        if not np.isfinite(h).all() or np.any(h<0) or np.any(h!=np.floor(h)):failures.append(pop+':invalid_spike_history')
        if counts is not None and (not np.isfinite(counts).all() or np.any(np.asarray(counts)<0)):failures.append(pop+':invalid_bin_counts')
        if not state.get('rng_state'):failures.append(pop+':missing_rng')
        if any(e['time_seconds']<t-1e-12 for e in state.get('pending_events',[])):failures.append(pop+':past_pending_event')
    return failures


def _strip_variable_aux(aux,queue=False):
    """Exact semantic metadata; numeric rate/delay/event times checked apart."""
    if not isinstance(aux,dict):return aux
    result={}
    for name,value in aux.items():
        if name=='conditional_rate_Hz' or queue and name in ('history_values','values'):continue
        if name=='pending_events':
            result[name]=[{k:v for k,v in event.items() if k!='time_seconds'} for event in value]
        elif isinstance(value,dict):result[name]=_strip_variable_aux(value,
            False if name=='prehistory' else queue or name in ('delay_states','queues','observation_delay'))
        else:result[name]=value
    return result


class GroupTimeComparator:
    def __init__(self,monitor,initial_ports,port_units,*,parameters,evidence_sink=None):
        self.monitor,self.parameters,self.sink=monitor,parameters,evidence_sink
        self.port_rows={};self.unsupported=[];self.last_payload=None
        if set(initial_ports)!=set(port_units):raise ContractError('Complete frozen typed original port registry required')
        for key,value in initial_ports.items():
            declaration=port_units[key];unit=declaration['unit'] if isinstance(declaration,dict) else declaration
            initial=_owned(value)
            if unit=='events/s' and key[0] in ('rod_BC','AII_AC') and key[-1]=='event_rate':
                load_addendum();self.port_rows[key]=dict(unit=unit,initial=initial,derived='ribbon')
            elif unit in NUMERICAL_UNIT_SCALES:
                scale,den,budget=frozen_budget(initial,unit)
                self.port_rows[key]=dict(unit=unit,initial=initial,scale=scale,budget=budget)
            else:self.unsupported.append(dict(port=str(key),unit=unit))
        self.comparisons=0

    def compare_monitors(self,reference,candidate,width):
        """Compare full raw per-cell monitor statistics using frozen propagation."""
        a=reference.payload() if isinstance(reference,GroupTimeMonitor) else reference
        b=candidate.payload() if isinstance(candidate,GroupTimeMonitor) else candidate
        issues=[];raw={};status='PASS'
        def issue(kind,gate,detail):
            nonlocal status
            if kind=='FAIL' or status=='PASS':status=kind
            issues.append(dict(status=kind,gate=gate,detail=str(detail)))
        if (a.get('identity')!=self.monitor.identity or b.get('identity')!=self.monitor.identity
                or canonical_hash(a.get('stamps'))!=canonical_hash(b.get('stamps')) or a.get('count')!=b.get('count')):
            issue('INCONCLUSIVE','monitor','Complete frozen identity/observation stamps differ')
        if a.get('domain_failures') or b.get('domain_failures'):issue('FAIL','monitor.domain','Original domain predicate failed')
        if set(a.get('rows',{}))!={_name(k) for k in self.monitor.rows} or set(b.get('rows',{}))!=set(a.get('rows',{})):
            issue('INCONCLUSIVE','monitor','Complete per-cell statistic registry missing')
        else:
            for key,row in self.monitor.rows.items():
                if row['budget'] is None:
                    issue('INCONCLUSIVE','monitor.unit',row['spec'].unit);continue
                aa,bb=a['rows'][_name(key)],b['rows'][_name(key)]
                allowed_rhs=rhs_allowance(row['scale'],row['budget'],aa['rhs'],bb['rhs'])
                bounds=dict(lo=row['budget'],hi=row['budget'],rhs=allowed_rhs,
                    defect=2*row['budget']/row['period'] if row['period'] else np.zeros_like(row['budget']))
                for field,bound in bounds.items():
                    va,vb=_owned(aa[field]),_owned(bb[field]);difference=np.abs(vb-va)
                    raw[_name(key)+'.'+field]=dict(reference=va,candidate=vb,difference=vb-va,allowance=bound)
                    bad=np.flatnonzero(difference>bound)
                    if len(bad):issue('FAIL','monitor.'+_name(key)+'.'+field,'first_cell='+str(int(bad[0])))
        ref_report=reference.report(width) if isinstance(reference,GroupTimeMonitor) else None
        cand_report=candidate.report(width) if isinstance(candidate,GroupTimeMonitor) else None
        explained=True
        if ref_report is not None and cand_report is not None:
            if ref_report['passed']!=cand_report['passed']:issue('INCONCLUSIVE','monitor.decision','Original threshold decision differs')
            ra,cb=ref_report['dominant'],cand_report['dominant']
            if ra!=cb and ra is not None and cb is not None:
                def score_bound(witness,arm):
                    k=next(k for k in self.monitor.rows if _name(k)==witness['state']);row=self.monitor.rows[k]
                    data=arm['rows'][_name(k)];i=witness['cell_index'];label=witness['criterion'];den=row['den'][i]
                    if label=='excursion':return (data['hi'][i]-data['lo'][i])/den,2*row['budget'][i]/den
                    if label=='original_rhs':
                        other=b if arm is a else a
                        bound=rhs_allowance(row['scale'],row['budget'],data['rhs'],other['rows'][_name(k)]['rhs'])[i]
                        return width*data['rhs'][i]/den,width*bound/den
                    return width*data['defect'][i]/den,width*2*row['budget'][i]/row['period']/den if row['period'] else 0.
                ar,br=score_bound(ra,a),score_bound(cb,a)
                ac,bc=score_bound(ra,b),score_bound(cb,b)
                explained=ar[0]-br[0]<=ar[1]+br[1] and bc[0]-ac[0]<=ac[1]+bc[1]
                if not explained:issue('INCONCLUSIVE','monitor.dominant','Different dominant cell is unexplained by original propagated bounds')
        result=dict(schema=PAIR_SCHEMA,status=status,issues=issues,per_cell=raw,
            reference_report=ref_report,candidate_report=cand_report,dominant_difference_explained=explained,
            contract_hash=self.monitor.contract_hash,background_qualified=False)
        self.last_payload=result
        if self.sink is not None:self.sink(result)
        return result

    def compare_pair(self,reference,candidate):
        issues=[];raw={};rank={'PASS':0,'INCONCLUSIVE':1,'FAIL':2};status='PASS'
        def issue(kind,gate,detail):
            nonlocal status
            if rank[kind]>rank[status]:status=kind
            issues.append(dict(status=kind,gate=str(gate),detail=str(detail)))
        for arm in (reference,candidate):
            stamp=arm.get('time_seconds')
            if not isinstance(stamp,(float,int,np.floating,np.integer)) or isinstance(stamp,(bool,np.bool_)) or not math.isfinite(stamp):
                issue('FAIL','stamp','Finite original observation time required')
            if arm.get('side') not in ('left','right') or type(arm.get('event_generation')) is not int or arm.get('event_generation',-1)<0:
                issue('INCONCLUSIVE','stamp','Original side and integer generation required')
        for name in ('time_seconds','side','event_generation'):
            if name not in reference or name not in candidate:
                issue('INCONCLUSIVE','stamp','Missing '+name);continue
            a,b=reference[name],candidate[name]
            try:same=(float(a).hex()==float(b).hex()) if name=='time_seconds' else a==b
            except (ValueError,TypeError):same=False
            if not same:issue('FAIL','stamp',name+' differs')
        if reference.get('rhs_kind')!='original_equation' or candidate.get('rhs_kind')!='original_equation':
            issue('INCONCLUSIVE','rhs','Full original-equation callback witness required')
        if self.unsupported or self.monitor.missing:
            issue('INCONCLUSIVE','typed_budget',self.unsupported+self.monitor.missing)
        def numeric(gate,a,b,allowed):
            try:
                a,b=_owned(a),_owned(b)
                allowed=np.broadcast_to(allowed,a.shape)
            except (ContractError,ValueError,TypeError) as exc:
                issue('INCONCLUSIVE',gate,exc);return
            if a.shape!=b.shape:
                issue('INCONCLUSIVE',gate,'Numeric layout differs');return
            difference=np.abs(b-a);bad=np.flatnonzero(difference>allowed)
            raw[str(gate)]=dict(reference=a,candidate=b,difference=b-a,allowance=np.array(allowed,copy=True))
            if len(bad):
                i=int(bad[0]);issue('FAIL',gate,f'first_cell={i} difference={difference.ravel()[i]:.17g} allowance={np.broadcast_to(allowed,a.shape).ravel()[i]:.17g}')
        states_a,states_b=reference.get('states',{}),candidate.get('states',{})
        for field,expected in (('states',set(self.monitor.rows)),('rhs',set(self.monitor.rows)),('transitions',set(self.monitor.transition_keys))):
            a,b=reference.get(field),candidate.get(field)
            if not isinstance(a,dict) or not isinstance(b,dict) or set(a)!=expected or set(b)!=expected:
                issue('INCONCLUSIVE',field,'Complete original registry missing');continue
            for key in (self.monitor.transition_keys if field=='transitions' else self.monitor.rows):
                row=self.monitor.rows[key]
                if row['budget'] is None:continue
                allowed=row['budget']
                if field=='rhs':allowed=rhs_allowance(row['scale'],allowed,_cpu(a[key]),_cpu(b[key]))
                numeric(field+'.'+_name(key),a[key],b[key],allowed)
        ports_a,ports_b=reference.get('ports'),candidate.get('ports')
        if not isinstance(ports_a,dict) or not isinstance(ports_b,dict) or set(ports_a)!=set(self.port_rows) or set(ports_b)!=set(self.port_rows):
            issue('INCONCLUSIVE','ports','Complete typed original ports missing')
        else:
            for key,row in self.port_rows.items():
                if row.get('derived')=='ribbon':
                    try:
                        allowed,result=self._ribbon(key,ports_a[key],ports_b[key],states_a,states_b)
                        raw['ports.'+str(key)]=dict(reference=_owned(ports_a[key]),candidate=_owned(ports_b[key]),allowance=allowed)
                        if result is not None:issue('FAIL','ports.'+str(key),result)
                    except (KeyError,ValueError,TypeError) as exc:issue('INCONCLUSIVE','ports.'+str(key),exc)
                else:numeric('ports.'+str(key),ports_a[key],ports_b[key],row['budget'])
        aux_a,aux_b=reference.get('aux'),candidate.get('aux')
        if aux_a is None or aux_b is None:issue('INCONCLUSIVE','aux','Complete original clocks/RNG/history/input/queue identities required')
        elif canonical_hash(_strip_variable_aux(aux_a))!=canonical_hash(_strip_variable_aux(aux_b)):
            issue('FAIL','aux','Original clocks, history, counts, RNG, pending cell/order/bin, or input phase differ')
        else:self._variable_aux(aux_a,aux_b,reference.get('draws',{}),candidate.get('draws',{}),numeric,issue)
        for arm in (reference,candidate):
            if arm.get('domain_failures'):issue('FAIL','domain',arm['domain_failures'])
        da,db=reference.get('draws',{}),candidate.get('draws',{})
        if reference.get('event_witness_required') or candidate.get('event_witness_required'):
            if set(da)!=set(RGC) or set(db)!=set(RGC):issue('INCONCLUSIVE','draw','Original uniform witnesses must cover every sampled object')
        if set(da)!=set(db):issue('INCONCLUSIVE','draw','Draw registry differs')
        for pop in set(da)&set(db):self._draw(pop,da[pop],db[pop],numeric,issue)
        self.comparisons+=1
        self.last_payload=dict(schema=PAIR_SCHEMA,status=status,issues=issues,
            stamp={k:reference.get(k) for k in ('time_seconds','side','event_generation')},per_cell=raw,
            contract_hash=self.monitor.contract_hash,background_qualified=False,fit_eligible=False)
        if self.sink is not None:self.sink(self.last_payload)
        return self.last_payload

    def _ribbon(self,key,a,b,sa,sb):
        pop=key[0];Vkey=(pop,'ReceptorConductanceMembrane','V');xkey=(pop,'ReceptorConductanceMembrane','x')
        va,vb,xa,xb=(_owned(v) for v in (sa[Vkey],sb[Vkey],sa[xkey],sb[xkey]))
        Bv,Bx=self.monitor.rows[Vkey]['budget'],self.monitor.rows[xkey]['budget']
        n=len(va);p=self.parameters['populations'][pop]
        values={k:np.broadcast_to(np.asarray(p[k],dtype=np.float64),(n,)) for k in ('Npool','U','rmax','r_half','r_slope')}
        a,b=_owned(a),_owned(b)
        def down(v):return np.nextafter(v,-np.inf)
        # Outward-rounded lower bound only screens obvious passes. Every
        # uncertain case uses the unchanged rational source helper exactly.
        slope_up=np.nextafter(4*values['r_slope'],np.inf)
        voltage_term=np.minimum(1.,down(Bv/slope_up))
        factor=down(down(values['Npool']*values['U'])*values['rmax'])
        allowance=down(factor*down(Bx+down((1.+1e-10)*voltage_term)))
        uncertainty=(np.nextafter(np.abs(b-a),np.inf)>allowance)|(np.nextafter(np.abs(vb-va),np.inf)>Bv)|(np.nextafter(np.abs(xb-xa),np.inf)>Bx)
        for i in np.flatnonzero(uncertainty):
            r=compare_cell(pool=float(values['Npool'][i]),U=float(values['U'][i]),rmax=float(values['rmax'][i]),
                r_half=float(values['r_half'][i]),r_slope=float(values['r_slope'][i]),V_reference=float(va[i]),V_candidate=float(vb[i]),
                x_reference=float(xa[i]),x_candidate=float(xb[i]),rate_reference=float(a[i]),rate_candidate=float(b[i]),B_V=float(Bv[i]),B_x=float(Bx[i]))
            if not r['passed']:return allowance,dict(first_cell=int(i),exact=r)
        return allowance,None

    def _variable_aux(self,a,b,drawa,drawb,numeric,issue):
        for pop,sa in a.get('spike_states',{}).items():
            sb=b['spike_states'][pop]
            if 'conditional_rate_Hz' in sa and 'conditional_rate_Hz' in sb:
                pa=self.parameters['populations'][pop];Bv=self.monitor.rows[pop,'RGC_spike_history','V']['budget']
                ra,rb=_owned(sa['conditional_rate_Hz']),_owned(sb['conditional_rate_Hz'])
                allowed=pa['alpha']/pa['beta']*Bv+8192*np.finfo(float).eps*np.maximum(abs(ra),1.)
                numeric('aux.'+pop+'.rate',ra,rb,allowed)
            ea,eb=sa.get('pending_events',[]),sb.get('pending_events',[])
            if len(ea)==len(eb) and any(x['time_seconds']!=y['time_seconds'] for x,y in zip(ea,eb)) and pop not in drawa:
                issue('INCONCLUSIVE','pending_events','Changed event timestamp needs the original uniform/rate witness')
        for name in ('delay_states','queues','observation_delay'):
            qa,qb=a.get(name),b.get(name)
            if qa is None or qb is None:continue
            if 'history_values' in qa:qa={'native':qa};qb={'native':qb}
            for queue,va in qa.items():
                if not isinstance(va,dict):continue
                for field in ('history_values','values'):
                    if field not in va:continue
                    vb=qb[queue];unit=va.get('unit',va.get('prehistory',{}).get('unit') if isinstance(va.get('prehistory'),dict) else None)
                    key=('HC','reduced_native','Vh')
                    if unit!='vH_native_voltage' or key not in self.monitor.rows:
                        issue('INCONCLUSIVE','queue','Unsupported delay value propagation');continue
                    numeric('queue.'+queue+'.'+field,va[field],vb[field],self.monitor.rows[key]['budget'])

    def _draw(self,pop,a,b,numeric,issue):
        required={'u','rate_Hz','probability','event_decision','event_time_seconds','bin_start_seconds',
                  'history_hash','rng_before_hash','rng_after_hash','draw_index'}
        if pop not in RGC or set(a)!=required or set(b)!=required:
            issue('INCONCLUSIVE','draw.'+pop,'Complete original sampler witness required');return
        for name in ('u','history_hash','rng_before_hash','rng_after_hash','draw_index','event_decision','bin_start_seconds'):
            if canonical_hash(a[name])!=canonical_hash(b[name]):issue('FAIL','draw.'+pop,name+' differs')
        u,rate,other,pa,pb=(_owned(v) for v in (a['u'],a['rate_Hz'],b['rate_Hz'],a['probability'],b['probability']))
        deca,decb=np.asarray(a['event_decision']),np.asarray(b['event_decision'])
        p=self.parameters['populations'][pop];Bv=self.monitor.rows[pop,'RGC_spike_history','V']['budget']
        if (any(v.shape!=Bv.shape for v in (u,rate,other,pa,pb,deca,decb))
                or deca.dtype.kind!='b' or decb.dtype.kind!='b'):
            issue('INCONCLUSIVE','draw.'+pop,'Complete typed per-cell sampler witness required');return
        for arm in (a,b):
            for field in ('history_hash','rng_before_hash','rng_after_hash'):
                value=arm[field]
                if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                    issue('INCONCLUSIVE','draw.'+pop,'Strict-past/RNG hash is absent or malformed')
            order=np.asarray(arm['draw_index'])
            if order.dtype.kind not in 'iu' or order.shape!=u.shape or np.any(np.diff(order)!=1):
                issue('INCONCLUSIVE','draw.'+pop,'Original per-cell PCG64 draw order missing')
        eps=8192*np.finfo(float).eps
        ra=p['alpha']/p['beta']*Bv+eps*np.maximum(abs(rate),1.)
        ba=.001*p['alpha']/p['beta']*Bv+eps
        numeric('draw.'+pop+'.rate',rate,other,ra);numeric('draw.'+pop+'.probability',pa,pb,ba)
        if np.any((u<0)|(u>=1)) or np.any(rate<0) or np.any(other<0) or np.any((pa<0)|(pa>1)|(pb<0)|(pb>1)):issue('FAIL','draw.'+pop,'Sampler domain invalid')
        if np.any(np.abs(u-pa)<=ba):issue('INCONCLUSIVE','draw.'+pop,'Uniform inside original probability uncertainty interval')
        for arm,prob,decision,rate_value in ((a,pa,deca,rate),(b,pb,decb,other)):
            if not np.array_equal(prob,-np.expm1(-.001*rate_value)) or not np.array_equal(decision,u<prob):
                issue('FAIL','draw.'+pop,'Original probability/decision expression differs')
            times=np.asarray(arm['event_time_seconds'],dtype=np.float64);start=float(arm['bin_start_seconds'])
            if not math.isfinite(start):issue('FAIL','draw.'+pop,'Finite original bin start required');continue
            if times.shape!=u.shape:issue('INCONCLUSIVE','draw.'+pop,'Event-time layout differs');continue
            if np.any(decision & ((rate-ra)<=0)):
                issue('INCONCLUSIVE','draw.'+pop,'Actual event lower allowed rate is nonpositive');continue
            take=np.flatnonzero(decision)
            offset=-np.log1p(-u[take]);early=start+offset/(rate[take]+ra[take]);late=start+offset/(rate[take]-ra[take])
            if np.any(~np.isfinite(times[take])) or np.any(times[take]<early) or np.any(times[take]>late) or np.any(times[take]<start) or np.any(times[take]>start+.001):
                issue('FAIL','draw.'+pop,'Event time outside original lambda-derived interval/bin')
            if np.any(~np.isnan(times[~decision])):issue('FAIL','draw.'+pop,'Non-event has event timestamp')
