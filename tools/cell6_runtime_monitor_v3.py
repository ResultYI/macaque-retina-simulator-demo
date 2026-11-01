"""Original monitor checks; typed private-buffer maximum avoids the captured ufunc path."""
import hashlib,math
from pathlib import Path
import numpy as np
from numba import njit
from retina_sim.contracts import ContractError
from tools import cell6_runtime_monitor_v2 as previous


@njit(cache=False,fastmath=False,boundscheck=True)
def _maximum_kernel(target,other):
    if target.size!=other.size:raise ValueError('Statistic array shape differs')
    for index in range(target.size):
        a=target[index];b=other[index]
        # NumPy maximum propagates NaN and chooses the second signed zero on ties.
        if math.isnan(a):target[index]=a
        elif math.isnan(b):target[index]=b
        elif a>b:target[index]=a
        else:target[index]=b


CALLS=0


def maximum_into(target,other):
    global CALLS
    if (target.dtype!=np.float64 or other.dtype!=np.float64 or target.ndim!=1 or
        target.shape!=other.shape or not target.flags.c_contiguous or not other.flags.c_contiguous or
        not target.flags.writeable or not target.flags.owndata):
        raise ContractError('Private owned FP64 monitor statistic arrays required')
    _maximum_kernel(target,other);CALLS+=1


SOURCE=previous.SOURCE
for before,after in (
    ("np.maximum(row['hi'],a,out=row['hi'])","maximum_into(row['hi'],a)"),
    ("np.maximum(row['rhs'],work,out=row['rhs'])","maximum_into(row['rhs'],work)"),
    ("np.maximum(row['defect'],work,out=row['defect'])","maximum_into(row['defect'],work)"),
):
    if SOURCE.count(before)!=1:raise RuntimeError('Exact monitor-statistic source changed')
    SOURCE=SOURCE.replace(before,after)
namespace=dict(previous.namespace,maximum_into=maximum_into)
exec(compile(SOURCE,__file__+':typed_statistic_maximum','exec'),namespace)
if namespace['observe'].__code__.co_cellvars:raise RuntimeError('Monitor unexpectedly closes over row buffers')


class Cell6Monitor(previous.Cell6Monitor):
    observe=namespace['observe']


def evidence():
    return dict(schema='private-monitor-statistic-maximum-v3',calls=CALLS,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        typed_FP64=True,fastmath=False,boundscheck=True,owned_output_required=True,
        all_original_monitor_checks_and_payload_retained=True,
        equation_or_state_or_event_changed=False,native_corruption_root_cause_proven=False)
