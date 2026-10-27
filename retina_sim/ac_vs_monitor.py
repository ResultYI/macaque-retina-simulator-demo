"""New monitor categories, with the original absolute result kept intact."""
import numpy as np

from .contracts import ContractError
from .group_time_monitor import frozen_budget


class DynamicDefectMonitor:
    def __init__(self,assembly,initial):
        self.budgets = {}
        for b in (*assembly.state_blocks,*assembly.probability_blocks,*assembly.auxiliary_z1_blocks):
            k=(b.population,b.mechanism,b.name)
            self.budgets[k]=frozen_budget(initial[k],b.unit)[2]
        self.rows={}
        self.samples=0

    def observe(self,states,rhs,derivatives,*,time_seconds,side):
        if set(rhs)!=set(derivatives) or set(rhs)!=set(states):
            raise ContractError('Complete represented dynamic derivative registry required')
        for k in rhs:
            defect=np.abs(derivatives[k]-rhs[k])
            if not np.isfinite(defect).all(): raise ContractError('Nonfinite dynamic numerical defect')
            allowance=self.budgets[k]/.0001
            norm=float(np.max(defect/allowance,initial=0.))
            name=':'.join(k)
            if name not in self.rows or norm>self.rows[name]['maximum_norm']:
                self.rows[name]=dict(maximum_norm=norm,max_abs_defect=float(np.max(defect,initial=0.)),
                    time_seconds=time_seconds,side=side)
        self.samples+=1

    def report(self):
        worst=max(self.rows,key=lambda k:self.rows[k]['maximum_norm']) if self.rows else None
        maximum=0. if worst is None else self.rows[worst]['maximum_norm']
        return dict(category='DYNAMIC_NUMERICAL_DEFECT',passed=bool(self.samples and maximum<=1),
            maximum_norm=maximum,dominant=worst,samples=self.samples,per_state=self.rows,
            definition='abs(dense_reconstruction_derivative - original_RHS)/(frozen_state_budget/100us)',
            discrete_states='zero between events; original event maps/history checks separate',
            old_absolute_monitor_unchanged=True)
