"""Exercise the worker's live lease guard without constructing a model."""
from pathlib import Path

import pytest

from retina_sim.group_scc_protocol import AcceptanceFailure, write_json
from tools.validate_group_scc_radau import require_live_budget_lease


def test_transient_sharing_lock_rechecks_the_actual_committed_budget(tmp_path, monkeypatch):
    path = tmp_path/'budget_ledger.json'
    lease = dict(active='owner', stopped=False, used=dict(cold=1., paired=2.))
    write_json(path, lease)
    read = Path.read_bytes
    calls = 0

    def shared(value):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError('transient Windows sharing violation')
        return read(value)

    monkeypatch.setattr(Path, 'read_bytes', shared)
    require_live_budget_lease(path, 'owner', 'paired')
    assert calls == 2
    write_json(path, dict(lease, stopped=True))
    with pytest.raises(AcceptanceFailure, match='stopped or changed'):
        require_live_budget_lease(path, 'owner', 'paired')


def test_persistent_sharing_lock_and_invalid_json_still_fail_closed(tmp_path, monkeypatch):
    path = tmp_path/'budget_ledger.json'
    path.write_text('{invalid', encoding='utf8')
    with pytest.raises(ValueError):
        require_live_budget_lease(path, 'owner', 'paired')

    def locked(value):
        raise PermissionError('persistent Windows sharing violation')

    monkeypatch.setattr(Path, 'read_bytes', locked)
    with pytest.raises(PermissionError, match='persistent'):
        require_live_budget_lease(path, 'owner', 'paired')


@pytest.mark.parametrize('stop_during_read',[False,True])
def test_worker_ownership_transient_read_still_checks_current_stop(tmp_path,monkeypatch,stop_during_read):
    import tools.validate_group_scc_radau as harness
    own = dict(pid=9,creation_ticks=20)
    job = dict(token='owner',supervisor_process=dict(pid=8,creation_ticks=10))
    monkeypatch.setattr(harness,'process_identity',lambda _:own)
    monkeypatch.setattr(harness,'verified_ancestry',lambda *args:[])
    monkeypatch.setattr(harness,'process_alive',lambda *args:True)
    receipt = tmp_path/'restored_worker_ownership.json'
    write_json(receipt,dict(token='owner',worker=own,label='restored',job_kill_on_close=True))
    original = Path.read_bytes
    reads = 0
    def shared(path):
        nonlocal reads
        if path == receipt:
            reads += 1
            if reads == 1:
                if stop_during_read:(tmp_path/'REQUEST_STOP').write_text('stop')
                raise PermissionError('transient ownership sharing violation')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',shared)
    if stop_during_read:
        with pytest.raises(ValueError,match='No live supervisor'):
            harness.worker_handshake(tmp_path,job,'restored')
    else:
        assert harness.worker_handshake(tmp_path,job,'restored')['worker'] == own
    assert reads == 2


@pytest.mark.parametrize('failure',['persistent_permission','invalid_json','wrong_token'])
def test_worker_ownership_persistent_and_identity_failures_remain_errors(tmp_path,monkeypatch,failure):
    import tools.validate_group_scc_radau as harness
    own = dict(pid=9,creation_ticks=20)
    job = dict(token='owner',supervisor_process=dict(pid=8,creation_ticks=10))
    monkeypatch.setattr(harness,'process_identity',lambda _:own)
    monkeypatch.setattr(harness,'verified_ancestry',lambda *args:[])
    monkeypatch.setattr(harness,'process_alive',lambda *args:True)
    receipt = tmp_path/'restored_worker_ownership.json'
    write_json(receipt,dict(token='other',worker=own,label='restored',job_kill_on_close=True))
    if failure == 'persistent_permission':
        def locked(_):raise PermissionError('persistent ownership sharing violation')
        monkeypatch.setattr(Path,'read_bytes',locked)
        error = PermissionError
    else:
        if failure == 'invalid_json':receipt.write_text('{bad',encoding='utf8')
        error = ValueError
    with pytest.raises(error):harness.worker_handshake(tmp_path,job,'restored')
