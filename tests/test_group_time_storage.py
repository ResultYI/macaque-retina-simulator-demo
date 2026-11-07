import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from retina_sim.contracts import ContractError
from retina_sim.group_time_storage import save_bundle, load_bundle, COMPLETE_FIELDS


def test_full_checkpoint_write_exit_new_process_read_and_hash_rejection(tmp_path):
    payload = dict.fromkeys(COMPLETE_FIELDS)
    payload['native'] = {'time': .004, 'y': np.array([1., 2.]), 'future_stop': .0045}
    payload['inner'] = {('pop','mechanism','V'): np.array([-.055])}
    payload['spike_states'] = {'history': [[0]*1100], 'rng_state': {'state': 123456789123456789123456789}}
    folder = tmp_path/'checkpoint'
    identity = {'graph': 'same-frozen-group', 'parameters': 'same-source'}
    save_bundle(folder, kind='complete_model_checkpoint', identity=identity, payload=payload,
                qualification={'default_adopted': False, 'fit_eligible': False, 'coupled_background_qualified': False})
    restored = load_bundle(folder, expected_kind='complete_model_checkpoint', expected_identity=identity)
    np.testing.assert_array_equal(restored['payload']['inner']['pop','mechanism','V'], [-.055])
    result = subprocess.run([sys.executable,'-B','-c',
        "import json,sys;from retina_sim.group_time_storage import load_bundle; x=load_bundle(sys.argv[1],expected_kind='complete_model_checkpoint'); print(json.dumps(dict(y=x['payload']['native']['y'].tolist(), future=x['payload']['native']['future_stop'])))", str(folder)],
        capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict(y=[1.,2.],future=.0045)
    with pytest.raises(ContractError, match='identity'):
        load_bundle(folder, expected_kind='complete_model_checkpoint', expected_identity={'graph':'wrong'})
    with next(folder.glob('*.npy')).open('ab') as stream:stream.write(b'tampered')
    with pytest.raises(ContractError, match='identity'):
        load_bundle(folder, expected_kind='complete_model_checkpoint')


def test_partial_or_old_payload_cannot_claim_complete_restore(tmp_path):
    with pytest.raises(ContractError, match='Complete group checkpoint'):
        save_bundle(tmp_path/'bad',kind='complete_model_checkpoint',identity={},payload={'native':np.ones(1)},
                    qualification={'default_adopted':False,'fit_eligible':False})


def test_failed_encoding_removes_only_this_attempt(tmp_path):
    unrelated = tmp_path/'other.writing-existing'
    unrelated.mkdir()
    marker = unrelated/'evidence.txt'
    marker.write_text('retain')
    with pytest.raises(ContractError, match='nonfinite'):
        save_bundle(tmp_path/'bad', kind='validation_evidence', identity={},
                    payload=[np.arange(3.), np.array([np.nan])],
                    qualification={'default_adopted':False,'fit_eligible':False})
    assert list(tmp_path.iterdir()) == [unrelated]
    assert marker.read_text() == 'retain'


def test_cleanup_failure_preserves_original_write_error(tmp_path, monkeypatch):
    import retina_sim.group_time_storage as storage
    def locked_directory(path):
        raise PermissionError('directory is locked')
    monkeypatch.setattr(storage.shutil, 'rmtree', locked_directory)
    with pytest.raises(ContractError, match='nonfinite') as caught:
        save_bundle(tmp_path/'bad', kind='validation_evidence', identity={},
                    payload=[np.arange(3.), np.array([np.nan])],
                    qualification={'default_adopted':False,'fit_eligible':False})
    staging, = tmp_path.iterdir()
    assert caught.value.__notes__ == [
        f'Could not remove incomplete bundle {staging}: directory is locked']


def test_failed_publish_keeps_concurrent_target_and_removes_staging(tmp_path, monkeypatch):
    import retina_sim.group_time_storage as storage
    target = tmp_path/'checkpoint'
    def failed_publish(source, destination):
        target.mkdir()
        (target/'evidence.txt').write_text('concurrent result')
        raise OSError('publication failed')
    monkeypatch.setattr(storage.os, 'replace', failed_publish)
    with pytest.raises(OSError, match='publication failed'):
        save_bundle(target, kind='validation_evidence', identity={}, payload=np.arange(3.),
                    qualification={'default_adopted':False,'fit_eligible':False})
    assert list(tmp_path.iterdir()) == [target]
    assert (target/'evidence.txt').read_text() == 'concurrent result'
    with pytest.raises(ContractError, match='Fresh group bundle'):
        save_bundle(target, kind='validation_evidence', identity={}, payload=np.arange(3.),
                    qualification={'default_adopted':False,'fit_eligible':False})
    assert (target/'evidence.txt').read_text() == 'concurrent result'

