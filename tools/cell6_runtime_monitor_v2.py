"""Experiment-only closure-free control flow; original monitor arithmetic."""
import inspect
import textwrap
from retina_sim import group_time_monitor as original

SOURCE=textwrap.dedent(inspect.getsource(original.GroupTimeMonitor.observe))
REPLACEMENTS=(
    ('finite=lambda x:bool(self.torch.isfinite(x).all().item())',
     'finite=lambda x,torch=self.torch:bool(torch.isfinite(x).all().item())'),
    ('zero=lambda:self.torch.zeros_like(a)',
     'zero=lambda a=a,torch=self.torch:torch.zeros_like(a)'),
    ('zero=lambda:np.zeros_like(a)', 'zero=lambda a=a:np.zeros_like(a)'),
    ("values=[states[(pop,'RGC_expected_history','history_probability_'+str(i))] for i in range(8)]",
     "values=[]\n        for i in range(8):values.append(states[(pop,'RGC_expected_history','history_probability_'+str(i))])"),
)
for before,after in REPLACEMENTS:
    if SOURCE.count(before)!=1:raise RuntimeError('Frozen monitor adapter source pattern changed')
    SOURCE=SOURCE.replace(before,after)
namespace=dict(vars(original))
exec(compile(SOURCE,__file__+':generated_observe','exec'),namespace)
if namespace['observe'].__code__.co_cellvars:raise RuntimeError('Monitor still closes over mutable row/state locals')


class Cell6Monitor(original.GroupTimeMonitor):
    observe=namespace['observe']
