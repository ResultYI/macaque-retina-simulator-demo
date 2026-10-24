"""Same canonical input hashes with bounded, content-keyed JSON reuse.

Every call hashes every numeric input byte afresh. Object identity is never a
cache key. No time/side/unit/meaning/frame field is omitted. Delayed-link and
unknown schemas fall back to the original complete canonical encoder.
"""
from collections import OrderedDict
import hashlib


class InputEvidenceHash:
    def __init__(self,capacity=16):
        if type(capacity) is not int or capacity<=0:raise ValueError('Positive bounded cache required')
        self.capacity=capacity;self.cache=OrderedDict();self.calls=0;self.hits=0

    def __call__(self,bundle):
        import numpy as np
        from .network import InputBundle
        from .contracts import Quantity,canonical_hash
        self.calls+=1
        if (type(bundle) is not InputBundle or type(bundle.channels) is not dict or
                type(bundle.delay_buffers) is not dict or bundle.delay_buffers or
                set(bundle.__dataclass_fields__)!={'channels','delay_buffers','delay_query_seconds','delay_left_limit'}):
            return canonical_hash(bundle)
        rows={}
        for name,q in bundle.channels.items():
            if type(name) is not str or type(q) is not Quantity or set(q.__dataclass_fields__)!={'values','unit','meaning','evidence_tag','coordinate_frame'}:
                return canonical_hash(bundle)
            a=q.values
            if type(a) is not np.ndarray or a.dtype.hasobject or a.flags.writeable:
                return canonical_hash(bundle)
            original_shape=a.shape
            a=np.ascontiguousarray(a)
            value_sha=hashlib.sha256(memoryview(a).cast('B')).hexdigest() if a.nbytes else hashlib.sha256(b'').hexdigest()
            rows[name]=dict(dtype=str(a.dtype),shape=original_shape,value_sha256=value_sha,
                unit=q.unit,meaning=q.meaning,evidence_tag=q.evidence_tag,coordinate_frame=q.coordinate_frame)
        key=canonical_hash(dict(channels=rows,delay_buffers={},delay_query_seconds=bundle.delay_query_seconds,
                                delay_left_limit=bundle.delay_left_limit))
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);return self.cache[key]
        result=canonical_hash(bundle)
        self.cache[key]=result
        if len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return result
