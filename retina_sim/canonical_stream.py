"""Byte-identical canonical JSON hashing without a whole-array/list JSON copy.

The encoding matches contracts.canonical_hash's original json.dumps contract,
including ndarray dtype/shape/data, sorted string keys, enums and dataclasses.
It is a memory implementation change, not a new identity schema or evidence.
"""
from enum import Enum
import hashlib,json
import numpy as np

_OPTIONS={'sort_keys':True,'allow_nan':False,'separators':(',',':')}
# These exact NumPy scalar classes are native JSON float/str subclasses. Block
# encoding emits the identical scalar tokens and comma separators, avoiding a
# Python encoder call for every element of immutable observation-history rows.
_PRIMITIVES=(type(None),bool,int,float,str,np.float64,np.str_)

class CanonicalSequence:
    """Re-iterable JSON list whose records are constructed one at a time."""
    def __init__(self, factory):
        if not callable(factory):raise TypeError('A fresh iterator factory is required')
        self.factory=factory


class CanonicalArray:
    """Explicit one-dimensional array source for bounded canonical hashing.

    Implementations supply dtype, shape and fresh ordered numeric blocks. This
    uses the existing ndarray JSON representation, not a new identity format.
    """
    def blocks(self):
        raise NotImplementedError


def _stream_sequence(sequence,cache=None):
    yield '['
    iterator=iter(sequence.factory());first=True
    while True:
        try:value=next(iterator)
        except StopIteration:break
        if not first:yield ','
        yield from _chunks(value,cache)
        # Release the completed record before constructing the next one.
        del value
        first=False
    yield ']'


def _array_data(array,cache=None):
    if isinstance(array,CanonicalArray):
        yield '['
        count=0
        for block in array.blocks():
            if block.ndim!=1 or block.dtype!=array.dtype:
                raise ValueError('Canonical array block layout differs')
            if not len(block):continue
            if count:yield ','
            yield json.dumps(block.tolist(),**_OPTIONS)[1:-1]
            count+=len(block)
        if array.shape!=(count,):raise ValueError('Canonical array length differs')
        yield ']'
        return
    if array.ndim==0:
        yield json.dumps(array.tolist(),**_OPTIONS);return
    yield '['
    if array.ndim==1:
        previous_bytes=previous_json=None
        numeric=type(array) in (np.ndarray,np.memmap) and array.dtype.kind in 'biuf'
        for start in range(0,len(array),8192):
            if start:yield ','
            block=array[start:start+8192]
            if numeric:
                # Read every current byte. The immutable byte snapshot also
                # supplies the encoded values, so no mutable-array identity or
                # hidden old contents can satisfy a later block's comparison.
                raw=block.tobytes(order='C')
                if raw!=previous_bytes:
                    previous_json=(json.dumps(np.frombuffer(raw,dtype=array.dtype).tolist(),**_OPTIONS)[1:-1]
                                   if cache is None else cache.encode(array.dtype,raw))
                    previous_bytes=raw
                yield previous_json
            else:
                yield json.dumps(block.tolist(),**_OPTIONS)[1:-1]
    else:
        for index,row in enumerate(array):
            if index:yield ','
            yield from _array_data(row,cache)
    yield ']'

def _sequence(values,cache=None):
    yield '['
    for start in range(0,len(values),8192):
        block=values[start:start+8192]
        if start:yield ','
        if all(type(value) in _PRIMITIVES for value in block):
            yield json.dumps(block,**_OPTIONS)[1:-1]
        else:
            for index,value in enumerate(block):
                if index:yield ','
                yield from _chunks(value,cache)
    yield ']'

def _mapping(value,cache=None):
    # Preserve the original str(key) conversion, including last-key collision
    # semantics, before the JSON encoder sorts the resulting string keys.
    values={str(key):item for key,item in value.items()}
    yield '{'
    for index,key in enumerate(sorted(values)):
        if index:yield ','
        yield json.dumps(key,**_OPTIONS);yield ':';yield from _chunks(values[key],cache)
    yield '}'

def _chunks(value,cache=None):
    if isinstance(value,CanonicalSequence):yield from _stream_sequence(value,cache)
    elif isinstance(value,(np.ndarray,CanonicalArray)):
        yield '{"data":';yield from _array_data(value,cache)
        yield ',"dtype":';yield json.dumps(str(value.dtype),**_OPTIONS)
        yield ',"shape":';yield from _sequence(list(value.shape));yield '}'
    elif isinstance(value,Enum):yield from _chunks(value.value,cache)
    elif hasattr(value,'__dataclass_fields__'):
        yield from _mapping({key:getattr(value,key) for key in value.__dataclass_fields__},cache)
    elif isinstance(value,dict):yield from _mapping(value,cache)
    elif isinstance(value,(list,tuple)):yield from _sequence(value,cache)
    else:yield json.dumps(value,**_OPTIONS)

def stream_canonical_hash(value,*,block_cache=None):
    digest=hashlib.sha256()
    for chunk in _chunks(value,block_cache):digest.update(chunk.encode())
    return digest.hexdigest()
