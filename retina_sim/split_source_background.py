"""Explicit split background preparation with original qualification predicates.

No integration, device import or allocation occurs on module import. Runtime
entry requires approved actual input/graph receipts and a current useful permit.
The binary checkpoint remains diagnostic; qualification is earned separately by
the unchanged original background report and strict envelope checks.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import time
import uuid

from .contracts import ContractError, canonical_hash
from .resident_background import (ResidentBackgroundCheckpointStore, _scope,
    _origin, _execution_record, prepare_resident_background)
from .cone_hc_split_checkpoint import save_complete, load_complete
from .split_chunk_store import write_json, receipt, read_receipt, verify_checkpoint_files


class SplitBackgroundCheckpointStore(ResidentBackgroundCheckpointStore):
    """Same qualification validator, complete binary storage, new owned root."""
    def __init__(self, root, runner, reference_state, light_context, policy,
                 *, directory, create=False):
        from .cone_hc_split_runner import ConeHCSplitRunner
        if type(runner) is not ConeHCSplitRunner:
            raise ContractError('Binary split background store requires the explicit split runner')
        super().__init__(root, runner, reference_state, light_context, policy)
        self.binding['split_background_store_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.binding_hash = canonical_hash(self.binding)
        self.directory = Path(directory).resolve()
        workspace = Path(root).resolve()
        if self.directory == workspace or not self.directory.is_relative_to(workspace):
            raise ContractError('Split background session must stay within this project')
        if create:
            if self.directory.exists() or not self.directory.parent.is_dir():
                raise ContractError('Background session must be new with an existing parent')
            self.directory.mkdir()
            write_json(self.directory/'owner.json', receipt(dict(owner_uuid=uuid.uuid4().hex,
                binding_hash=self.binding_hash, schema='split-qualified-background-store-1')))
        self.owner = read_receipt(self.directory/'owner.json')
        if self.owner['binding_hash'] != self.binding_hash:
            raise ContractError('Owned background store binding differs')

    def save(self, path, state, report=None, *, origin_seconds=None, safety=None):
        self.runner._check_runtime()
        self._require_current_geometry()
        path = Path(path).resolve()
        if path.parent != self.directory or path.exists():
            raise ContractError('Background snapshot must be absent in its owned session')
        origin = _origin(self.runner, state, origin_seconds)
        if any(state.input_position.get(k) != v for k,v in self.expected_context.items()):
            raise ContractError('Only the actual bound uniform-background phase can qualify')
        value = dict(schema='split-binary-background-envelope-1', binding=self.binding,
            binding_hash=self.binding_hash, absolute_time_seconds=state.time_seconds,
            preparation_origin_seconds=origin, report=report, report_hash=canonical_hash(report), safety=safety,
            owner_uuid=self.owner['owner_uuid'])
        if report is not None:
            if report.get('resident_execution') != _execution_record(self.runner):
                raise ContractError('Background report is from a different execution')
            if report.get('converged'):
                self.require_convergence(value)
        staging = path.with_name(path.name+'.writing-'+uuid.uuid4().hex)
        staging.mkdir()
        checkpoint = save_complete(self.runner, state, staging/'checkpoint')
        # Binary receipt is not substituted for a numerical qualification report.
        value['complete_checkpoint_hash'] = checkpoint['manifest_hash']
        value = receipt(value)
        write_json(staging/'qualification.json', value)
        os.replace(staging, path)
        write_json(self.directory/'head.json', dict(snapshot=path.name,
            receipt_sha256=value['receipt_sha256']), replace=True)
        self._retire_snapshots(path)
        return value

    def _retire_snapshots(self, newest):
        candidates = []
        for p in self.directory.glob('snapshot-*'):
            if not p.is_dir() or '.writing-' in p.name:
                continue
            r = read_receipt(p/'qualification.json')
            if r.get('owner_uuid') != self.owner['owner_uuid'] or r.get('binding_hash') != self.binding_hash:
                raise ContractError('Unowned background snapshot encountered; no cleanup')
            candidates.append((r['absolute_time_seconds'], p, r))
        candidates.sort(key=lambda x:(x[0], x[1].name))
        for _, p, r in candidates[-2:]:
            verify_checkpoint_files(p/'checkpoint', r['complete_checkpoint_hash'])
        for _, p, r in candidates[:-2]:
            target = (p/'checkpoint').resolve()
            if not target.exists():
                continue
            if p.resolve().parent != self.directory or target.parent != p.resolve() or target.name != 'checkpoint':
                raise ContractError('Background retirement escaped owned snapshot')
            # Keep the immutable report, manifest and an explicit retirement log.
            if not (p/'retired_checkpoint_manifest.json').exists():
                write_json(p/'retired_checkpoint_manifest.json', json.loads((target/'manifest.json').read_bytes()))
            if not (p/'retirement.json').exists():
                write_json(p/'retirement.json', dict(manifest_hash=r['complete_checkpoint_hash'],
                    retained_newest=newest.name, reason='two_newer_owned_background_snapshots'))
            shutil.rmtree(target)

    def load(self, path, *, require_converged=False, decode_mode=None, diagnostic_path=None):
        self.runner._check_runtime()
        self._require_current_geometry()
        path = Path(path).resolve()
        if path.parent != self.directory or '.writing-' in path.name:
            raise ContractError('Uncommitted/out-of-session split background path')
        value = read_receipt(path/'qualification.json')
        if (value.get('schema') != 'split-binary-background-envelope-1'
                or value.get('binding') != self.binding or value.get('binding_hash') != self.binding_hash
                or value.get('owner_uuid') != self.owner['owner_uuid']
                or value.get('report_hash') != canonical_hash(value.get('report'))):
            raise ContractError('Background graph/input/report/content identity differs')
        if require_converged:
            self.require_convergence(value)
        state = load_complete(self.runner, path/'checkpoint',
            expected_manifest_hash=value['complete_checkpoint_hash'], expected_input_context=self.expected_context)
        if state.time_seconds != value['absolute_time_seconds']:
            raise ContractError('Qualified report/checkpoint time differs')
        _origin(self.runner, state, value['preparation_origin_seconds'])
        return state, value


def prepare_split_source_background(store, state, channels, *, reference_state,
                                    guard, resume_path=None, checkpoint_wall_seconds=90.,
                                    max_prepare_advance_seconds=2., on_progress=None):
    """One bounded preparation, partial progress never marked qualified.

    Original consecutive-window streak is reset on resume. The same frozen
    monitor reference/policy/graph must be reconstructed before calling this API.
    """
    if type(store) is not SplitBackgroundCheckpointStore:
        raise ContractError('Explicit binary split background store required')
    runner, policy = store.runner, store.policy
    _scope(runner, policy)
    from .useful_runtime_budget import goal_only_authorization
    goal_only=goal_only_authorization(runner.contract.runtime_authorization or {})
    preparation_limit=policy.max_sim_seconds if goal_only else 2.
    if not 0 < max_prepare_advance_seconds <= preparation_limit or checkpoint_wall_seconds <= 0:
        raise ContractError('First useful preparation is bounded to <=2 simulated seconds')
    runner.contract.require_input_policy(channels, phase='uniform_background')
    runner.runtime_guard = guard.check
    if resume_path is not None:
        state, old = store.load(resume_path)
        if (old.get('report') or {}).get('converged'):
            store.require_convergence(old)
            return state, old['report'], Path(resume_path)
    origin = _origin(runner, state)
    # A segment budget is not a new scientific max_sim policy/binding. Keeping
    # the original policy allows later explicitly authorized continuation.
    segment_limit = state.time_seconds+max_prepare_advance_seconds
    last_saved = time.monotonic()
    last_progress=last_saved
    last_window=None
    latest_path = None

    def save(current, report=None):
        nonlocal last_saved, latest_path
        guard.require(force=True)
        latest_path = store.directory/('snapshot-'+str(time.time_ns()))
        store.save(latest_path, current, report, origin_seconds=origin, safety=guard.last)
        last_saved = time.monotonic()

    save(state)

    def progress(current):
        nonlocal last_progress
        if not guard.check():
            return False
        if on_progress is not None and time.monotonic()-last_progress>=10.:
            on_progress(current,last_window)
            last_progress=time.monotonic()
        # Original driver batches at most 10 monitoring grid intervals. Stop
        # before the next batch could cross the segment budget, never skip a
        # monitor/event or alter its solver schedule to reach an artificial cap.
        if current.time_seconds+10*policy.monitor_step_seconds > segment_limit+1e-12:
            return False
        if time.monotonic()-last_saved >= checkpoint_wall_seconds:
            save(current)
        return True

    def window(current, row):
        nonlocal last_window
        last_window=row
        # Stream summaries, no full network time series. Preserve complete rows.
        write_json(store.directory/('window-'+str(time.time_ns())+'.json'), row)

    try:
        state, report = prepare_resident_background(runner, state, channels, policy,
            reference_state=reference_state, on_chunk=progress, on_window=window,
            preparation_origin_seconds=origin)
        if guard.stop_reason:
            report['budget_exit'] = guard.stop_reason
        # Record even an exhausted budget's non-qualification report, without
        # attempting a further state export or running beyond that budget.
        write_json(store.directory/('final-report-'+str(time.time_ns())+'.json'), report)
        save(state, report)
        return state, report, latest_path
    except Exception as exc:
        write_json(store.directory/('failure-'+str(time.time_ns())+'.json'), dict(
            exception_type=type(exc).__name__, message=str(exc),
            last_complete_snapshot=None if latest_path is None else latest_path.name,
            resource_sample=dict(guard.last), qualification='NOT_GRANTED_BY_FAILURE'))
        raise
