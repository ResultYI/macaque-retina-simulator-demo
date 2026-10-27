import json

import numpy as np

from retina_sim.contracts import canonical_hash
from retina_sim.group_time_storage import save_bundle, load_bundle


def test_loader_transfers_owned_arrays_but_repeated_references_remain_independent(tmp_path, monkeypatch):
    target = tmp_path/'bundle'
    save_bundle(target, kind='validation_evidence', identity={}, payload=[np.arange(4.)],
                qualification=dict(default_adopted=False, fit_eligible=False))
    path = target/'manifest.json'
    raw = json.loads(path.read_text())
    raw.pop('manifest_hash')
    raw['payload'].append(dict(raw['payload'][0]))
    raw['manifest_hash'] = canonical_hash(raw)
    path.write_text(json.dumps(raw))
    actual = np.load
    seen = []
    def track(*args, **kwargs):
        a = actual(*args, **kwargs)
        seen.append(a)
        return a
    monkeypatch.setattr(np, 'load', track)
    values = load_bundle(target, expected_kind='validation_evidence')['payload']
    assert values[0] is seen[0]
    assert not np.shares_memory(values[0], values[1])
    values[0][0] = 99.
    assert values[1][0] == 0.
    assert actual(target/'array_00000.npy')[0] == 0.


def test_fortran_and_numeric_list_keep_original_layout_and_types(tmp_path):
    a = np.asfortranarray(np.arange(12.).reshape(3, 4))
    payload = dict(fortran=a, lists=[list(range(1100))])
    save_bundle(tmp_path/'bundle', kind='validation_evidence', identity={}, payload=payload,
                qualification=dict(default_adopted=False, fit_eligible=False))
    result = load_bundle(tmp_path/'bundle', expected_kind='validation_evidence')['payload']
    assert result['fortran'].flags.c_contiguous
    np.testing.assert_array_equal(result['fortran'], a)
    assert result['lists'] == payload['lists']
