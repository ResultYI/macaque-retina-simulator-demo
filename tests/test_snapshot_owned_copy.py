"""Copied snapshots retain old values, aliases, isolation and fresh events."""
from copy import deepcopy

import numpy as np

from retina_sim.contracts import canonical_hash
from retina_sim.group_time_runner import _snapshot_copy


def test_sampler_lists_keep_original_digest_and_owned_mutable_containers():
    row = [0., 1., 2.]
    state = dict(history=[row, row], last_bin_counts=[0, 1, 0], conditional_rate_Hz=[1., 2., 3.],
                 rng_state=dict(state=dict(state=5, inc=7)), pending_events=[dict(time_seconds=4.997)],
                 extra=dict(array=np.array([1., 2.])))
    original = dict(cell=state)
    result = _snapshot_copy(original)
    assert canonical_hash(result) == canonical_hash(deepcopy(original))
    assert result['cell']['history'][0] is result['cell']['history'][1]
    assert result['cell']['history'][0] is not row
    result['cell']['history'][0][0] = 99.
    result['cell']['pending_events'][0]['time_seconds'] = 5.
    result['cell']['extra']['array'][0] = 8.
    assert row[0] == 0.
    assert state['pending_events'][0]['time_seconds'] == 4.997
    assert state['extra']['array'][0] == 1.
    state['last_bin_counts'][0] = 1
    assert _snapshot_copy(original)['cell']['last_bin_counts'][0] == 1
    assert result['cell']['last_bin_counts'][0] == 0


def test_nested_unknown_mutables_cycles_and_numpy_types_keep_deepcopy_semantics():
    original = [np.float64(1.), {'key': (bytearray(b'x'), np.array([2.]))}]
    result = _snapshot_copy(original)
    assert canonical_hash(result[0]) == canonical_hash(original[0])
    result[1]['key'][0][0] = 2
    result[1]['key'][1][0] = 3.
    assert original[1]['key'][0] == bytearray(b'x')
    assert original[1]['key'][1][0] == 2.
    cycle = []
    cycle.append(cycle)
    copied = _snapshot_copy(cycle)
    assert copied[0] is copied and copied is not cycle


def test_exact_builtin_fast_path_does_not_treat_mutable_subclasses_as_atomic():
    class ExtendedFloat(float):
        pass
    number=ExtendedFloat(1.)
    number.notes=[]
    original={'values':[0.,1,True,None,'text',b'data',number], 'number':number}
    copied=_snapshot_copy(original)
    assert copied['values'][-1] is copied['number'] and copied['number'] is not number
    copied['number'].notes.append('changed')
    assert number.notes==[]


def test_fresh_content_under_same_stamp_side_generation_dense_and_restored_objects():
    value=dict(time_hex=float(4.9931).hex(),side='right',event_generation=4,
               states={'dense':np.array([1.,2.])},history=[[0.,1.]],pending=[{'time_seconds':4.994}])
    first=_snapshot_copy(value)
    previous=canonical_hash(first)
    for change in (lambda:value['states']['dense'].__setitem__(0,3.),
                   lambda:value['history'][0].__setitem__(0,1.),
                   lambda:value.update(side='left'),lambda:value.update(event_generation=5)):
        change()
        current=_snapshot_copy(value)
        assert canonical_hash(current)==canonical_hash(deepcopy(value))!=previous
        previous=canonical_hash(current)
    restored=deepcopy(value)
    assert canonical_hash(_snapshot_copy(restored))==previous
    restored['states']['dense'][0]=7.
    assert canonical_hash(_snapshot_copy(restored))!=previous
    assert first['states']['dense'][0]==1. and first['history'][0][0]==0.
