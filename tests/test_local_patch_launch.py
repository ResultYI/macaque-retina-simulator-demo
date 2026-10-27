"""Fail-closed launch protection tests; never build a graph or start a worker."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / 'tools/run_local_patch.py'
    spec = importlib.util.spec_from_file_location('local_patch_launch_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from retina_sim import background_checkpoint
    memory = SimpleNamespace(last=dict(rss_bytes=1024, available_physical_bytes=8*1024**3),
        stop_reason=None, check=lambda: True)
    monkeypatch.setattr(background_checkpoint, 'MemoryGuard', lambda *args: memory)
    monkeypatch.setattr(module, 'process_alive', lambda pid, ticks: True)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'gpu_memory', lambda: dict(uuid='test-device', used_bytes=1024,
        total_bytes=16*1024**3))
    return module, memory


def authorization(module):
    return dict(deadline_unix=module.time.time()+1200., source_pins={})


@pytest.mark.parametrize('gate', ['stage_stop', 'run_stop', 'phase_deadline', 'worker_deadline'])
def test_stop_and_wall_deadlines_are_checked_even_during_resource_throttle(launcher, tmp_path, gate):
    module, memory = launcher
    stage = tmp_path/'new_stage'; stage.mkdir()
    run = tmp_path/'run'; run.mkdir()
    auth = authorization(module)
    started = module.time.monotonic()
    if gate == 'stage_stop': (stage/'REQUEST_STOP').write_text('stop', encoding='utf8')
    elif gate == 'run_stop': (run/'REQUEST_STOP').write_text('stop', encoding='utf8')
    elif gate == 'phase_deadline': auth['deadline_unix'] = module.time.time()-1.
    elif gate == 'worker_deadline': started -= 301.
    guard = module.Guard(stage, auth, worker_started=started, extra_stop=run)
    guard.last = module.time.monotonic()
    with pytest.raises(RuntimeError, match='STOP|deadline|300s'):
        guard()
    assert guard.observations == []


def test_parent_pid_reuse_fails_before_resource_throttle(launcher, monkeypatch, tmp_path):
    module, _ = launcher
    checked = []
    def reused(pid, ticks):
        checked.append((pid, ticks))
        return False
    monkeypatch.setattr(module, 'process_alive', reused)
    guard = module.Guard(tmp_path, authorization(module), parent=dict(pid=123, creation_ticks=1))
    guard.last = module.time.monotonic()
    with pytest.raises(RuntimeError, match='Supervisor process identity changed'):
        guard()
    assert checked == [(123, 1)]


def test_ram_measurement_failure_is_not_substituted_with_a_pass(launcher, monkeypatch, tmp_path):
    module, memory = launcher
    memory.stop_reason = 'memory_peak_rss_limit'
    memory.check = lambda: False
    monkeypatch.setattr(module, 'gpu_memory', lambda: pytest.fail('RAM failure must stop immediately'))
    with pytest.raises(RuntimeError, match='RAM gate'):
        module.Guard(tmp_path, authorization(module))()


@pytest.mark.parametrize('used,total', [(13*1024**3,16*1024**3), (8*1024**3,8*1024**3)])
def test_whole_card_cap_and_free_reserve_remain_hard_gates(launcher, monkeypatch, tmp_path, used, total):
    module, _ = launcher
    monkeypatch.setattr(module, 'gpu_memory', lambda: dict(uuid='test-device',used_bytes=used,total_bytes=total))
    with pytest.raises(Exception, match='GPU|VRAM'):
        module.Guard(tmp_path, authorization(module))()


def test_unmeasurable_gpu_and_source_mutation_fail_closed(launcher, monkeypatch, tmp_path):
    module, _ = launcher
    def unavailable(): raise RuntimeError('GPU measurement unavailable')
    monkeypatch.setattr(module, 'gpu_memory', unavailable)
    with pytest.raises(RuntimeError, match='measurement unavailable'):
        module.Guard(tmp_path, authorization(module))()
    monkeypatch.setattr(module, 'gpu_memory', lambda: dict(uuid='test-device',used_bytes=0,total_bytes=16*1024**3))
    source = tmp_path/'source.py'; source.write_text('original', encoding='utf8')
    auth = authorization(module); auth['source_pins'] = {'source.py': module.sha(source)}
    guard = module.Guard(tmp_path, auth); guard()
    assert len(guard.observations) == 1
    source.write_text('changed', encoding='utf8'); guard.last = 0.
    with pytest.raises(RuntimeError, match='Source identity changed'):
        guard()
    assert len(guard.observations) == 1


@pytest.mark.parametrize('returncode,stdout,stderr', [(1,'','driver failure'), (0,'',''),
    (0,'gpu-a, 1, 16000\ngpu-b, 2, 16000\n','')])
def test_gpu_query_failure_or_ambiguous_devices_is_rejected(launcher, monkeypatch, returncode, stdout, stderr):
    module, _ = launcher
    # Restore the original function hidden by the shared fixture.
    path = Path(__file__).resolve().parents[1] / 'tools/run_local_patch.py'
    spec = importlib.util.spec_from_file_location('local_patch_query_under_test', path)
    query_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(query_module)
    monkeypatch.setattr(query_module.subprocess, 'run', lambda *a, **k:
        SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr))
    with pytest.raises(RuntimeError): query_module.gpu_memory()


def test_existing_stage_and_artifact_are_never_overwritten(launcher, tmp_path):
    module, _ = launcher
    stage = tmp_path/'closed_stage'; stage.mkdir()
    sentinel = stage/'REQUEST_STOP'; sentinel.write_text('historical stop', encoding='utf8')
    with pytest.raises(FileExistsError):
        module.demo(SimpleNamespace(output=stage, small=tmp_path/'small.json', large=tmp_path/'large.json'))
    assert list(stage.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding='utf8') == 'historical stop'
    artifact = tmp_path/'existing.json'; artifact.write_text('{"preserved":true}', encoding='utf8')
    with pytest.raises(FileExistsError): module.write(artifact, {'replacement': True})
    assert artifact.read_text(encoding='utf8') == '{"preserved":true}'


def test_preparation_deadline_is_cooperative_and_precedes_resource_throttle(launcher, tmp_path):
    module, _ = launcher
    guard = module.Guard(tmp_path, authorization(module))
    guard.last = module.time.monotonic()
    guard.preparation_deadline = module.time.monotonic()-1.
    with pytest.raises(RuntimeError, match='Preparation 300s'):
        guard()


@pytest.mark.parametrize('closed_by', ['deadline', 'complete', 'stop', 'active', 'worker_budget'])
def test_repair_cannot_revive_a_closed_expired_or_exhausted_stage(launcher, tmp_path, closed_by):
    module, _ = launcher
    stage = tmp_path/'prior_stage'; stage.mkdir()
    auth = authorization(module)
    ledger = dict(status='FAILED_PRESERVED', active_worker=None, used_worker_seconds=9., workers=[])
    if closed_by == 'deadline': auth['deadline_unix'] = module.time.time()-1.
    elif closed_by == 'complete': ledger['status'] = 'COMPLETE'
    elif closed_by == 'stop': (stage/'REQUEST_STOP').write_text('keep stopped', encoding='utf8')
    elif closed_by == 'active': ledger['active_worker'] = dict(pid=123)
    elif closed_by == 'worker_budget': ledger['used_worker_seconds'] = 900.
    module.write(stage/'AUTHORIZATION.json', auth); module.write(stage/'ledger.json', ledger)
    before = {p.name: p.read_bytes() for p in stage.iterdir()}
    args = SimpleNamespace(continue_stage=stage, repair_reason='ordinary implementation fix',
        small=tmp_path/'small.json', large=tmp_path/'large.json')
    with pytest.raises(RuntimeError, match='Closed, stopped, exhausted or active'):
        module.demo(args)
    assert {p.name: p.read_bytes() for p in stage.iterdir()} == before


def test_repair_requires_explicit_false_model_forward_evidence(launcher, tmp_path):
    module, _ = launcher
    stage = tmp_path/'current_stage'; stage.mkdir()
    logdir = stage/'small_prefix'; logdir.mkdir()
    run = stage/'run_small'; run.mkdir()
    module.write(stage/'AUTHORIZATION.json', authorization(module))
    module.write(stage/'ledger.json', dict(status='FAILED_PRESERVED', active_worker=None,
        used_worker_seconds=9., workers=[dict(name='small_prefix')]))
    module.write(logdir/'job.json', dict(run_directory=str(run), worker_log_directory=str(logdir), resume=False))
    module.write(run/'failure.json', dict(error='IndexError'))
    module.write(logdir/'worker_receipt.json', dict(status='FAILED', resource_final_measurement_error=None))
    args = SimpleNamespace(continue_stage=stage, repair_reason='ordinary implementation fix',
        small=tmp_path/'small.json', large=tmp_path/'large.json')
    with pytest.raises(RuntimeError, match='Only a preserved pre-forward'):
        module.demo(args)
    assert not list(stage.glob('AUTHORIZATION_REPAIR_*.json'))


@pytest.mark.parametrize('status,resource_error', [('RESOURCE_CHECK_FAILED', None),
    ('FAILED', 'available_memory_reserve')])
def test_repair_cannot_reclassify_a_resource_failure(launcher, tmp_path, status, resource_error):
    module, _ = launcher
    stage = tmp_path/'resource_failed_stage'; stage.mkdir()
    logdir = stage/'small_prefix'; logdir.mkdir()
    run = stage/'run_small'; run.mkdir()
    module.write(stage/'AUTHORIZATION.json', authorization(module))
    module.write(stage/'ledger.json', dict(status='FAILED_PRESERVED', active_worker=None,
        used_worker_seconds=9., workers=[dict(name='small_prefix')]))
    module.write(logdir/'job.json', dict(run_directory=str(run), worker_log_directory=str(logdir), resume=False))
    module.write(run/'failure.json', dict(error='IndexError', model_forward_started=False))
    module.write(logdir/'worker_receipt.json', dict(status=status,
        resource_final_measurement_error=resource_error))
    args = SimpleNamespace(continue_stage=stage, repair_reason='ordinary implementation fix',
        small=tmp_path/'small.json', large=tmp_path/'large.json')
    with pytest.raises(RuntimeError, match='resource|Resource'):
        module.demo(args)
    assert not list(stage.glob('AUTHORIZATION_REPAIR_*.json'))


@pytest.mark.parametrize('rejection', ['different_parent_error', 'nonzero_exit', 'incomplete_receipt', 'resource_error'])
def test_postrun_ownership_repair_exception_has_a_narrow_scope(launcher, tmp_path, rejection):
    module, _ = launcher
    stage = tmp_path/'old_unqualified_stage'; stage.mkdir()
    logdir = stage/'small_prefix'; logdir.mkdir()
    run = stage/'run_small'; run.mkdir()
    auth = authorization(module)
    ledger = dict(status='FAILED_PRESERVED', active_worker=None, used_worker_seconds=49.,
        error="RuntimeError('Worker identity or completion receipt differs')",
        workers=[dict(name='small_prefix', exit_code=0)])
    receipt = dict(status='COMPLETE', resource_final_measurement_error=None)
    if rejection == 'different_parent_error': ledger['error'] = "RuntimeError('Source identity changed')"
    elif rejection == 'nonzero_exit': ledger['workers'][0]['exit_code'] = 1
    elif rejection == 'incomplete_receipt': receipt['status'] = 'FAILED'
    elif rejection == 'resource_error': receipt['resource_final_measurement_error'] = 'memory_peak_rss_limit'
    module.write(stage/'AUTHORIZATION.json', auth); module.write(stage/'ledger.json', ledger)
    module.write(logdir/'job.json', dict(run_directory=str(run), worker_log_directory=str(logdir), resume=False))
    module.write(logdir/'worker_receipt.json', receipt)
    before = (stage/'ledger.json').read_bytes()
    args = SimpleNamespace(continue_stage=stage, repair_reason='fix Windows launcher bookkeeping',
        small=tmp_path/'small.json', large=tmp_path/'large.json')
    with pytest.raises((RuntimeError, FileNotFoundError)):
        module.demo(args)
    assert not list(stage.glob('AUTHORIZATION_REPAIR_*.json'))
    assert (stage/'ledger.json').read_bytes() == before


@pytest.mark.parametrize('mismatch', ['worker_token', 'stage_token', 'job_sha',
    'ancestry_supervisor', 'ancestry_worker', 'missing_wrapper'])
def test_unqualified_old_result_needs_intact_saved_identity_evidence_for_a_fresh_rerun(launcher, tmp_path, mismatch):
    module, _ = launcher
    stage = tmp_path/'old_identity_stage'; stage.mkdir()
    logdir = stage/'small_prefix'; logdir.mkdir()
    run = stage/'run_small'; run.mkdir()
    supervisor = dict(pid=300, creation_ticks=1)
    worker = dict(pid=100, creation_ticks=3)
    wrapper = dict(pid=200, creation_ticks=2)
    job = dict(run_directory=str(run), worker_log_directory=str(logdir), resume=False,
        supervisor=supervisor, worker_token='old-worker-token', stage_token='same-stage')
    module.write(logdir/'job.json', job)
    handshake = dict(**worker, worker_token=job['worker_token'], stage_token=job['stage_token'],
        job_sha256=module.sha(logdir/'job.json'), ancestry=[worker, wrapper, supervisor])
    if mismatch == 'worker_token': handshake['worker_token'] = 'wrong'
    elif mismatch == 'stage_token': handshake['stage_token'] = 'wrong'
    elif mismatch == 'job_sha': handshake['job_sha256'] = '0'*64
    elif mismatch == 'ancestry_supervisor': handshake['ancestry'][-1] = dict(pid=301, creation_ticks=1)
    elif mismatch == 'ancestry_worker': handshake['ancestry'][0] = dict(pid=101, creation_ticks=3)
    elif mismatch == 'missing_wrapper': handshake['ancestry'] = [worker, supervisor]
    module.write(logdir/'handshake.json', handshake)
    module.write(logdir/'worker_receipt.json', dict(status='COMPLETE', resource_final_measurement_error=None))
    module.write(stage/'AUTHORIZATION.json', authorization(module))
    module.write(stage/'ledger.json', dict(status='FAILED_PRESERVED', active_worker=None,
        used_worker_seconds=49., error="RuntimeError('Worker identity or completion receipt differs')",
        workers=[dict(name='small_prefix', exit_code=0)]))
    before = (stage/'ledger.json').read_bytes()
    args = SimpleNamespace(continue_stage=stage, repair_reason='fresh rerun after supervisor implementation repair',
        small=tmp_path/'small.json', large=tmp_path/'large.json')
    with pytest.raises(RuntimeError, match='tokens/source job/ancestry differ'):
        module.demo(args)
    assert not list(stage.glob('AUTHORIZATION_REPAIR_*.json'))
    assert not list(stage.glob('IDENTITY_UNQUALIFIED_*.json'))
    assert (stage/'ledger.json').read_bytes() == before
