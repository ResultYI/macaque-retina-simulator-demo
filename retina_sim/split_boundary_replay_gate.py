"""Separate, versioned RGC cache-readout scope; never relabel full eight-cell QA.

Exactly reuses frozen voltage formulas and alpha/beta propagation, without a new
free tolerance. The old full-source scenario/count guards remain unchanged.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from .contracts import ContractError, canonical_hash
from .resident_contract import CONTRACT_SHA256


class CacheReadoutComparator:
    def __init__(self, prospective):
        from .background import NUMERICAL_UNIT_SCALES
        p=Path(__file__).resolve().parents[1]/'configurations/numerical_equivalence/resident_native_fp64_v1.json'
        raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=CONTRACT_SHA256:
            raise ContractError('Original frozen voltage budget changed')
        self.frozen=json.loads(raw);self.units=deepcopy(NUMERICAL_UNIT_SCALES)
        self.plan=deepcopy(prospective)
        if self.plan.get('schema')!='split-cache-rgc-readout-check-1':
            raise ContractError('Unreviewed cache readout comparison scope')
        for row in self.plan['states'].values():
            initial=np.asarray(row['initial'],dtype=np.float64)
            if row['unit']!='V' or initial.ndim!=1 or len(initial)!=len(row['cell_ids']) or not np.isfinite(initial).all():
                raise ContractError('Finite original typed RGC voltage reference required')
        for values in self.plan['rate_lipschitz_per_V'].values():
            if not np.isfinite(values).all() or np.any(np.asarray(values)<=0):
                raise ContractError('Finite positive original alpha/beta bounds required')
        self.plan['original_registered_unit_scales_sha256']=canonical_hash(self.units)
        self.plan_hash=canonical_hash(self.plan)

    @classmethod
    def for_runner(cls,runner,reference_state,policy,state,times,recording_plan):
        from .boundary_replay import RGC
        if (canonical_hash(reference_state)!=runner.binding['monitor_reference_hash']
                or state.time_seconds!=float(times[0])
                or not 0<float(times[-1])-float(times[0])<=.100000000001):
            raise ContractError('Actual frozen reference and bounded block required')
        states={};rates={};cells=[]
        for spec,initial in zip(runner.model.state_specs,reference_state.ordered_state_arrays):
            cid=spec.state_name.split('.',1)[0];component=runner.compiled.components[cid]
            if component.mechanism!=RGC:continue
            if spec.unit!='V':raise ContractError('Unreviewed RGC state unit')
            states['state:'+spec.state_name]=dict(initial=np.asarray(initial,dtype=np.float64).tolist(),
                unit=spec.unit,meaning=spec.meaning,cell_ids=list(spec.cell_ids))
            native=component.native_model
            alpha=np.broadcast_to(np.asarray(native.alpha,dtype=np.float64),len(spec.cell_ids))
            beta=np.broadcast_to(np.asarray(native.beta,dtype=np.float64),len(spec.cell_ids))
            if native.dt!=.001 or np.any(alpha<=0) or np.any(beta<=0):
                raise ContractError('Original RGC event-law parameters required')
            rates[cid]=(alpha/beta).tolist();cells.extend((cid,x) for x in spec.cell_ids)
        if len(cells)!=12 or len(set(cells))!=12:
            raise ContractError('The actual fixed 8+4 RGC cohort is required')
        return cls(dict(schema='split-cache-rgc-readout-check-1',
            original_budget_contract_sha256=CONTRACT_SHA256,model_hash=runner.model.model_hash,
            input_binding_hash=runner.binding['binding_hash'],
            frozen_reference_sha256=runner.binding['monitor_reference_hash'],states=states,
            rate_lipschitz_per_V=rates,relative_tolerance=policy.relative_tolerance,
            times=list(map(float,times)),cells=cells,recording_plan=recording_plan.__dict__,
            scope='RGC cache readout and same-realization history only',
            whole_source_qa_claim=False,fit_eligible=False))

    def budgets(self,key,pro=False):
        row=self.plan['states'][key];entry=self.units[row['unit']]
        scale=np.maximum(np.abs(np.asarray(row['initial'])),entry['scale_floor'])
        solver=self.frozen['solver_required']
        E=solver['atol']+solver['rtol']*scale
        D=entry['atol']+self.plan['relative_tolerance']*scale
        f=self.frozen['state_budget_fraction']
        return np.minimum(E,f*D) if pro else f*np.minimum(E,D)

    def compare(self,a,b,source_final,replay_final):
        if (canonical_hash(self.plan)!=self.plan_hash
                or canonical_hash(self.units)!=self.plan['original_registered_unit_scales_sha256']):
            raise ContractError('Readout plan or original unit scales mutated')
        shared=[]
        if set(a.outputs_with_units)!=set(b.outputs_with_units):shared.append('output keys differ')
        for k in ('recorded_cell_ids','event_count','event_window_seconds','clock_contract',
                  'history_interval_seconds','observation_interval_seconds','sampling_index','spike_events'):
            if canonical_hash(a.metadata.get(k))!=canonical_hash(b.metadata.get(k)):shared.append(k+' differs')
        for cid,aux in source_final.spike_states.items():
            for k in ('history','rng_state','last_bin_counts'):
                if canonical_hash(aux[k])!=canonical_hash(replay_final.spike_states[cid][k]):
                    shared.append(cid+'.'+k+' differs; same-history bounds invalid')
        verdict={};eps=self.frozen['roundoff_epsilon_multiplier']*sys.float_info.epsilon
        for pro in (False,True):
            issues=list(shared);worst=maximum=maximum_rate=0.
            for k,x in a.outputs_with_units.items():
                if k not in b.outputs_with_units:continue
                y=b.outputs_with_units[k];xv,yv=np.asarray(x.values),np.asarray(y.values)
                if (x.unit,x.meaning,x.evidence_tag)!=(y.unit,y.meaning,y.evidence_tag):issues.append(k+' declaration differs')
                if xv.shape!=yv.shape or not np.isfinite(xv).all() or not np.isfinite(yv).all():
                    issues.append(k+' finite/layout differs');continue
                if k=='time' or k.endswith(('.spike_count','.window_rate')):
                    if not np.array_equal(xv,yv):issues.append(k+' exact values differ')
                elif k in self.plan['states']:
                    if (a.metadata.get('recorded_cell_ids',{}).get(k)!=self.plan['states'][k]['cell_ids']
                            or xv.ndim!=2 or xv.shape[0]!=len(self.plan['states'][k]['cell_ids'])):
                        issues.append(k+' original complete cell order/layout differs');continue
                    error=np.abs(yv-xv);norm=float(np.max(error/self.budgets(k,pro)[:,None],initial=0.))
                    worst=max(worst,norm);maximum=max(maximum,float(np.max(error,initial=0.)))
                    if norm>1:issues.append(k+' exceeds derived frozen voltage budget')
                else:issues.append(k+' lacks a prospectively declared typed budget')
            for cid,slope in self.plan['rate_lipschitz_per_V'].items():
                k=next(k for k in self.plan['states'] if k.startswith('state:'+cid+'.'))
                x=np.asarray(a.metadata['spike_state_metadata'][cid]['conditional_rate_Hz'])
                y=np.asarray(b.metadata['spike_state_metadata'][cid]['conditional_rate_Hz'])
                limit=np.asarray(slope)*self.budgets(k,pro)+eps*np.maximum(np.abs(x),1.)
                if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.abs(y-x)>limit):
                    issues.append(cid+' rate exceeds original alpha/beta propagation')
                if x.shape==y.shape:maximum_rate=max(maximum_rate,float(np.max(np.abs(y-x),initial=0.)))
            verdict['pro_candidate' if pro else 'old_strict_budget_formula']=dict(
                status='FAIL' if issues else 'PASS',issues=issues,max_voltage_error_V=maximum,
                max_normalized_voltage_error=worst,max_rate_error_Hz=maximum_rate)
        return verdict


def check_baseline(cache,root,source_result,source_final,*,comparator,sample_times,
                   recording_plan,evidence_directory):
    from .boundary_replay import BoundaryReplay
    from .split_boundary_source import _save_selected_result
    if type(comparator) is not CacheReadoutComparator:
        raise ContractError('Prospective readout-only comparator required')
    replay=BoundaryReplay(cache,root,candidate={},compute_backend='python')
    initial=replay.initialize()
    final,result=replay.advance(initial,source_final.time_seconds,
        sample_times=sample_times,recording_plan=recording_plan)
    verdict=comparator.compare(source_result,result,source_final,final)
    _save_selected_result(result,Path(evidence_directory)/'baseline_replay_outputs.npz',first_chunk=True)
    return dict(schema='split-cache-rgc-readout-verdict-1',verdict=verdict,plan_sha256=comparator.plan_hash,
        cache_identity_sha256=cache.identity_hash,cache_content_sha256=cache.content_hash,
        original_schema12_replay=True,full_source_reference_rerun=False,whole_source_qa_claim=False,
        physiological_validation='NOT_RUN',fit_eligible=False)
