"""Cache identical deterministic HC quadrature rules during large construction.

The original rule/order/tolerance/partition and random generator are unchanged.
Only repeated evaluations at exactly the same bounds and order are reused.
"""
from contextlib import contextmanager

@contextmanager
def memoized_hc_rules():
    from . import hc_mosaic
    original=hc_mosaic._Quadrature;instances=[]
    class Memoized(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.rule_cache={};self.rule_cache_hits=0;instances.append(self)
        def rule(self,bounds,order):
            key=(tuple(bounds),int(order))
            if key in self.rule_cache:
                self.rule_cache_hits+=1
                return self.rule_cache[key].copy()
            result=super().rule(bounds,order);self.rule_cache[key]=result.copy();return result
    hc_mosaic._Quadrature=Memoized
    try:yield instances
    finally:hc_mosaic._Quadrature=original
