"""Versioned complete reduced monitor; original RHS and CPU predicates retained."""
from dataclasses import replace
import numpy as np

from .contracts import ContractError
from .background import NUMERICAL_UNIT_SCALES
from .resident_monitor import (ResidentWindowMonitor, UNIT_INTERVAL_MEANINGS,
                              NONNEGATIVE_MEANINGS, _check_source)
from .cone_hc_reduced_evaluation import native_registry, state_arrays


SCHEMA = 'cone-hc-reduced-full-equation-monitor-1'


class ReducedWindowMonitor(ResidentWindowMonitor):
    def __init__(self,runner,reference_state,policy):
        _check_source()
        self.runner,self.torch,self.policy = runner,runner.torch,policy
        self.device = runner.native.device
        self.native_cid = runner.partition.cid
        self.rows,self.missing = [],[]
        count = len(runner.compiled.components[self.native_cid].state_specs)
        old = dict(zip((s.state_name for s in runner.native.model.state_specs),
                       reference_state.ordered_state_arrays[:count]))
        initial = dict(old,w=runner.native.model.A@old['z1'],
                       z1_aux=old['z1'][list(runner.native.aux_indices)])
        self.registry = native_registry(runner)
        for spec in self.registry:
            self._row(replace(spec,state_name=self.native_cid+'.'+spec.state_name),
                      initial[spec.state_name],True,spec.state_name,None,None)
        offset = runner.partition.offset
        for spec,values in zip(runner.model.state_specs[count:],reference_state.ordered_state_arrays[count:]):
            cid = spec.state_name.split('.',1)[0]
            sl = slice(offset-runner.partition.offset,offset-runner.partition.offset+len(values))
            local = slice(offset-runner.model.slices[cid].start,
                          offset-runner.model.slices[cid].start+len(values))
            self._row(spec,values,False,None,sl,local)
            offset += len(values)
        if self.missing:
            raise ContractError('Unregistered reduced monitor units: '+str(self.missing))
        self.reset()

    def _row(self,spec,initial,native,name,inner,local):
        entry = NUMERICAL_UNIT_SCALES.get(spec.unit)
        if entry is None:
            self.missing.append(spec.unit)
            return
        den = entry['atol']+self.policy.relative_tolerance*np.maximum(np.abs(initial),entry['scale_floor'])
        if np.any(~np.isfinite(den)) or np.any(den<=0):
            raise ContractError('Reduced monitor requires the original positive frozen scales')
        if native:
            den = self.torch.as_tensor(den,dtype=self.torch.float64,device=self.device).clone()
        self.rows.append(dict(spec=spec,den=den,native=native,name=name,
                              inner_slice=inner,local_slice=local))

    def observe(self,t,point,bundle):
        self.count += 1
        torch = self.torch
        evidence = self.runner.evaluate(t,point,bundle,sampling=True,require_rhs=True)
        values = state_arrays(self.runner,point)
        derivatives = self.runner.native.blocks(evidence.native_rhs)
        derivatives.update(R=evidence.native_outputs['R_rhs'],E=evidence.native_outputs['E_rhs'])
        transitions = {}
        gap = [q.values for (cid,port),q in evidence.outputs.items() if port=='gap_current']
        if gap:
            total = sum(float(np.sum(a)) for a in gap)
            magnitude = sum(float(np.sum(np.abs(a))) for a in gap)
            if abs(total)>1e-16+1e-10*magnitude:
                self.domain_failures.append('reciprocal_gap_current_conservation')
        for cid,component in self.runner.compiled.components.items():
            kind = component.time_contract.kind
            if kind in ('causal_history','author_discrete'):
                local = evidence.local_blocks(cid)
                if kind=='causal_history':
                    new = component.native_model.transition(local,evidence.inputs[cid],
                        {p:evidence.outputs[(cid,p)] for p in component.outputs})
                else:
                    new = component.native_model.transition(local,evidence.inputs[cid])
                transitions[cid] = np.concatenate(new)
        for row in self.rows:
            spec = row['spec']
            cid = spec.state_name.split('.',1)[0]
            if row['native']:
                a,rhs = values[row['name']],derivatives[row['name']]
                if not torch.isfinite(a).all().item() or not torch.isfinite(rhs).all().item():
                    raise ContractError('Nonfinite reduced monitor state/equation RHS')
                row['lo'] = a.clone() if row['lo'] is None else torch.minimum(row['lo'],a)
                row['hi'] = a.clone() if row['hi'] is None else torch.maximum(row['hi'],a)
                row['rhs'] = torch.maximum(row['rhs'],rhs.abs())
                any_value = lambda v:bool(v.any().item())
            else:
                a,rhs = evidence.inner_y[row['inner_slice']],evidence.inner_rhs[row['inner_slice']]
                if not np.isfinite(a).all() or not np.isfinite(rhs).all():
                    raise ContractError('Nonfinite original CPU monitor state/RHS')
                row['lo'] = a.copy() if row['lo'] is None else np.minimum(row['lo'],a)
                row['hi'] = a.copy() if row['hi'] is None else np.maximum(row['hi'],a)
                row['rhs'] = np.maximum(row['rhs'],np.abs(rhs))
                any_value = lambda v:bool(np.any(v))
            if spec.unit=='dimensionless' and spec.meaning in UNIT_INTERVAL_MEANINGS:
                if any_value((a < -1e-10)|(a>1+1e-10)):
                    self.domain_failures.append(spec.state_name)
            if spec.meaning in NONNEGATIVE_MEANINGS and any_value(a < -1e-10):
                self.domain_failures.append(spec.state_name+':negative_domain')
            if spec.meaning=='release_adaptation_activity':
                control = self.runner.compiled.components[cid].native_model.bc_output_gain_control
                if any_value(a>control.scale+1e-10):
                    self.domain_failures.append(spec.state_name+':above_scale')
            if cid in transitions:
                row['defect'] = np.maximum(row['defect'],np.abs(
                    transitions[cid][row['local_slice']]-a)/self.runner.periods[cid])

    def report(self,width):
        report = super().report(width)
        report.update(schema=SCHEMA,full_z1_available=False,
                      auxiliary_cell_ids=list(self.runner.native.aux_cell_ids),
                      state_coverage=[r['spec'].state_name for r in self.rows],
                      rhs='true equation at each queried state and input side')
        return report
