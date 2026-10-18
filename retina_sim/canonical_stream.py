"""Byte-identical canonical JSON hashing without a whole-array/list JSON copy.

The encoding matches contracts.canonical_hash's original json.dumps contract,
including ndarray dtype/shape/data, sorted string keys, enums and dataclasses.
It is a memory implementation change, not a new identity schema or evidence.
"""
from enum import Enum
import hashlib,json
import numpy as np

_OPTIONS={'sort_keys':True,'allow_nan':False,'separators':(',',':')}
_PRIMITIVES=(type(None),bool,int,float,str)

def _array_data(array):
    if array.ndim==0:
        yield json.dumps(array.tolist(),**_OPTIONS);return
    yield '['
    if array.ndim==1:
        for start in range(0,len(array),8192):
            if start:yield ','
            yield json.dumps(array[start:start+8192].tolist(),**_OPTIONS)[1:-1]
    else:
        for index,row in enumerate(array):
            if index:yield ','
            yield from _array_data(row)
    yield ']'

def _sequence(values):
    yield '['
    for start in range(0,len(values),8192):
        block=values[start:start+8192]
        if start:yield ','
        if all(type(value) in _PRIMITIVES for value in block):
            yield json.dumps(block,**_OPTIONS)[1:-1]
        else:
            for index,value in enumerate(block):
                if index:yield ','
                yield from _chunks(value)
    yield ']'

def _mapping(value):
    # Preserve the original str(key) conversion, including last-key collision
    # semantics, before the JSON encoder sorts the resulting string keys.
    values={str(key):item for key,item in value.items()}
    yield '{'
    for index,key in enumerate(sorted(values)):
        if index:yield ','
        yield json.dumps(key,**_OPTIONS);yield ':';yield from _chunks(values[key])
    yield '}'

def _chunks(value):
    if isinstance(value,np.ndarray):
        yield '{"data":';yield from _array_data(value)
        yield ',"dtype":';yield json.dumps(str(value.dtype),**_OPTIONS)
        yield ',"shape":';yield from _sequence(list(value.shape));yield '}'
    elif isinstance(value,Enum):yield from _chunks(value.value)
    elif hasattr(value,'__dataclass_fields__'):
        yield from _mapping({key:getattr(value,key) for key in value.__dataclass_fields__})
    elif isinstance(value,dict):yield from _mapping(value)
    elif isinstance(value,(list,tuple)):yield from _sequence(value)
    else:yield json.dumps(value,**_OPTIONS)

def stream_canonical_hash(value):
    digest=hashlib.sha256()
    for chunk in _chunks(value):digest.update(chunk.encode())
    return digest.hexdigest()
