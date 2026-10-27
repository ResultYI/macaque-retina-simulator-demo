import json

import pytest

from retina_sim.contracts import ContractError, canonical_hash
from tools.validate_group_reference_backend import protocol, enforce_wall, restore_session
from tools.validate_group_time_candidate import clock_events, sample_grid


def test_reference_protocol_retains_all_original_limits_and_boundaries():
    p = protocol()
    assert p['execution_mode'] == 'reference'
    assert p['runtime_limits']['paired_test_wall_seconds'] == 180
    assert p['runtime_limits']['minimum_available_bytes'] == 4*1024**3
    assert len(clock_events(p)) == 80
    assert (4.997, True, True) in clock_events(p)
    assert (5., True, True) in clock_events(p)
    assert len(sample_grid(4.993, 5.001, .0005)) == 17


def test_wall_clock_is_not_reset_across_process_restore():
    enforce_wall(100., now=280.)
    with pytest.raises(ContractError, match='180s'):
        enforce_wall(100., now=280.00001)


def test_resume_rejects_stop_failure_identity_or_expired_budget(tmp_path, monkeypatch):
    import tools.validate_group_reference_backend as driver
    monkeypatch.setattr(driver.time, 'time', lambda: 200.)
    identity = dict(version=1)
    session = dict(identity=identity, test_started_unix=100.)
    (tmp_path/'session_checkpoint.json').write_text(json.dumps(dict(session, session_hash=canonical_hash(session))))
    assert restore_session(tmp_path, identity) == session
    for file in ('REQUEST_STOP','failure.json','result.json'):
        path = tmp_path/file
        path.write_text('preserve')
        with pytest.raises(ContractError, match='cannot be resumed'):
            restore_session(tmp_path, identity)
        path.unlink()
    with pytest.raises(ContractError, match='identity'):
        restore_session(tmp_path, dict(version=2))
    monkeypatch.setattr(driver.time, 'time', lambda: 281.)
    with pytest.raises(ContractError, match='180s'):
        restore_session(tmp_path, identity)
