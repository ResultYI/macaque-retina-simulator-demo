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

def _array_data(array):
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
                    previous_json=json.dumps(np.frombuffer(raw,dtype=array.dtype).tolist(),**_OPTIONS)[1:-1]
                    previous_bytes=raw
                yield previous_json
            else:
                yield json.dumps(block.tolist(),**_OPTIONS)[1:-1]
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
