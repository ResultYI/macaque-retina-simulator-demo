"""Lossless failure-only oracle evidence; no acceptance decision is changed."""
from copy import deepcopy
import hashlib
from itertools import zip_longest
from pathlib import Path

import numpy as np

from .canonical_block_cache import CanonicalBlockCache
from .canonical_stream import _chunks
from .group_time_storage import save_bundle


def freeze(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):return value.copy()
    if isinstance(value, dict):return {key:freeze(item) for key,item in value.items()}
    if isinstance(value, list):return [freeze(item) for item in value]
    if isinstance(value, tuple):return tuple(freeze(item) for item in value)
    return deepcopy(value)


def canonical_byte_comparison(value):
    """Compare every original/cached UTF-8 chunk, retaining original SHA identity."""
    old=hashlib.sha256();cached=hashlib.sha256();count=size=0;first=None
    for a,b in zip_longest(_chunks(value),_chunks(value,CanonicalBlockCache())):
        left=b'' if a is None else a.encode('utf8')
        right=b'' if b is None else b.encode('utf8')
        old.update(left);cached.update(right)
        if first is None and (a is None or b is None or left!=right):
            at=next((i for i,(x,y) in enumerate(zip(left,right)) if x!=y),min(len(left),len(right)))
            first=dict(chunk=count,byte_offset=size+at,original=left[max(0,at-24):at+24].decode('utf8','replace'),
                       cached=right[max(0,at-24):at+24].decode('utf8','replace'))
        count+=1;size+=len(left)
    return dict(equal=first is None,original_sha256=old.hexdigest(),cached_sha256=cached.hexdigest(),
                original_bytes=size,chunks=count,first_difference=first)


def first_difference(expected,actual,path='$'):
    if isinstance(expected,np.ndarray) and isinstance(actual,np.ndarray):
        if expected.dtype!=actual.dtype or expected.shape!=actual.shape:
            return dict(path=path,kind='array_layout',expected_dtype=str(expected.dtype),actual_dtype=str(actual.dtype),
                        expected_shape=list(expected.shape),actual_shape=list(actual.shape))
        a=np.ascontiguousarray(expected);b=np.ascontiguousarray(actual)
        unequal=np.flatnonzero(a.reshape(-1).view('V'+str(a.dtype.itemsize))!=b.reshape(-1).view('V'+str(b.dtype.itemsize)))
        if not len(unequal):return None
        index=int(unequal[0]);x=a.reshape(-1)[index];y=b.reshape(-1)[index]
        result=dict(path=path,kind='array_value',flat_index=index,differing_elements=len(unequal),
                    expected=repr(x.item()),actual=repr(y.item()),expected_bytes=x.tobytes().hex(),actual_bytes=y.tobytes().hex())
        if a.dtype.kind in 'fiu':result['absolute_difference']=float(abs(float(x)-float(y)))
        return result
    if isinstance(expected,dict) and isinstance(actual,dict):
        a={str(k):v for k,v in expected.items()};b={str(k):v for k,v in actual.items()}
        if set(a)!=set(b):return dict(path=path,kind='mapping_keys',missing=sorted(set(a)-set(b)),extra=sorted(set(b)-set(a)))
        for key in sorted(a):
            difference=first_difference(a[key],b[key],path+'.'+key)
            if difference is not None:return difference
        return None
    if isinstance(expected,(list,tuple)) and isinstance(actual,(list,tuple)):
        if len(expected)!=len(actual):return dict(path=path,kind='sequence_length',expected=len(expected),actual=len(actual))
        for index,(x,y) in enumerate(zip(expected,actual)):
            difference=first_difference(x,y,path+'['+str(index)+']')
            if difference is not None:return difference
        return None
    a=''.join(_chunks(expected));b=''.join(_chunks(actual))
    return None if a==b else dict(path=path,kind='scalar',expected=repr(expected),actual=repr(actual),
                                 expected_canonical=a,actual_canonical=b)


def capture_mismatch(output,identity,index,expected_row,actual_row,evidence,*,monitors=None,arms=None,expected_evidence=None):
    from tools.validate_group_time_candidate import checkpoint_hash
    from .group_scc_protocol import write_json
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(diagnostic_only=True,index=index,expected_row=expected_row,actual_row=actual_row,
                decision='Original mismatch still fails; diagnostic never changes acceptance',modes={})
    write_json(output/'summary.json',report)  # Preserve the actual row before expensive capture.
    current=freeze(evidence)
    # Persist full actual values before optional owner/export/encoding diagnostics.
    # Their failure must not discard the primary mismatch evidence again.
    report['actual_bundle']=save_bundle(output/'actual_raw',kind='validation_evidence',identity=identity,
        payload=dict(actual=current,expected_row=expected_row,actual_row=actual_row),
        qualification=dict(default_adopted=False,fit_eligible=False))
    write_json(output/'summary.json',report)
    payload=dict(actual=current)
    if monitors is not None:payload['monitors']={mode:freeze(monitor.payload()) for mode,monitor in monitors.items()}
    if arms is not None:
        payload['owners']={mode:dict(time_seconds=arm.time_seconds,inner_time=arm.inner_time,event_generation=arm.event_generation,
            emitted_until=arm._emitted_until,clock_indices=freeze(arm.clock_indices),last_commit_times=freeze(arm.last_commit_times),
            spike_states=freeze(arm.spike_states),observation_delay=freeze(arm.delay_payload())) for mode,arm in arms.items()}
    if expected_evidence is not None:payload['expected']=freeze(expected_evidence)
    for mode,value in current.items():
        proof=canonical_byte_comparison(value)
        details=dict(canonical_bytes=proof,rehash_matches_original_actual_row=proof['original_sha256']==actual_row['hashes'][mode],
            stamp={key:value[key] for key in ('time_seconds','side','event_generation')},
            fields={key:checkpoint_hash(item,reuse_arrays=False) for key,item in value.items()})
        if expected_evidence is not None:
            baseline=payload['expected'][mode]
            baseline_hash=checkpoint_hash(baseline,reuse_arrays=False)
            details['expected_witness_matches_oracle']=baseline_hash==expected_row['hashes'][mode]
            details['first_difference']=first_difference(baseline,value) if details['expected_witness_matches_oracle'] else dict(kind='untrusted_expected_witness')
        report['modes'][mode]=details
        write_json(output/'summary.json',report)
    report['bundle']=save_bundle(output/'raw',kind='validation_evidence',identity=identity,payload=payload,
        qualification=dict(default_adopted=False,fit_eligible=False))
    write_json(output/'summary.json',report)
    return report
