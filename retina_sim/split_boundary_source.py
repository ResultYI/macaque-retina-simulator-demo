"""Qualified split source -> original schema 1.2 caches, one atomic chunk each.

Opt-in API, no default registration. Constructing a session performs explicit
state/checkpoint I/O; advance_one is a model run and requires separate runtime
authorization. A prepared API or an engineering fixture is not fit eligibility.
"""
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import time

import numpy as np

from .contracts import ContractError, canonical_hash
from .boundary_replay import BoundaryRecorder, BoundaryExtractor, BoundaryCache, _PreparedOrigin, RGC
from .cone_hc_split_checkpoint import save_complete, load_complete, file_sha
from .recording import RecordingPlan
from .split_chunk_store import SplitChunkStore, write_json


def _checkpoint_record(runner, state, path, expected_hash):
    """One coherent CPU export per chunk; never a full-state observer export."""
    from .hybrid import HybridRunner
    public = runner.export_state(state)
    facade = copy(runner.original)
    facade.compiled, facade.model = runner.compiled, copy(runner.model)
    facade.periods = runner.periods
    facade.event_clock, facade.observation_dt = runner.event_clock, runner.observation_dt
    serialized = HybridRunner.checkpoint(facade, public)
    manifest = json.loads((Path(path)/'manifest.json').read_bytes())
    digest = manifest.pop('manifest_hash')
    if (digest != expected_hash or canonical_hash(manifest) != digest
            or manifest['time_seconds'] != state.time_seconds
            or manifest['input_binding_hash'] != runner.binding['binding_hash']
            or hashlib.sha256(serialized.encode('utf8')).hexdigest() != manifest['files']['public_state.json']['sha256']):
        raise ContractError('Boundary source public state differs from the committed full checkpoint')
    # Original BoundaryRecorder checks the prepared source twice. Reuse the exact
    # validated codec bytes, not a second/third full JSON encoding.
    facade.checkpoint = lambda s: serialized if s is public else HybridRunner.checkpoint(facade, s)
    return facade, public, canonical_hash(json.loads(serialized))


def _inner_extractor(runner, channels):
    offset = runner.partition.offset
    # Reject unexpected native boundary dependencies; no silent full-state copy.
    for _, cid, _ in channels:
        if runner.model.slices[cid].start < offset:
            raise ContractError('Source boundary unexpectedly retains native states')
    def local(cid, y):
        slices = runner.model._local_slices[cid]
        if any(s.start < offset for s in slices):
            raise ContractError('Boundary extraction reached outside the CPU inner block')
        return tuple(y[s.start-offset:s.stop-offset] for s in slices)
    proxy = SimpleNamespace(compiled=runner.compiled,
        model=SimpleNamespace(_incoming=runner.model._incoming, local_blocks=local))
    return BoundaryExtractor(proxy, channels)


class _QualifiedLineage:
    """Internal capability earned by a validated session, never a bool flag."""
    def __init__(self, session):
        self.session_hash = session.session_hash
        self.preparation = deepcopy(session.session['binding']['qualification'])
        if self.preparation.get('status') != 'verified_original_background_contract':
            raise ContractError('Source session does not descend from a qualified background')


def _make_source_recorder(runner, state, checkpoint_path, checkpoint_hash, *, background_channels,
                         identity, registry, qualification, preceding_commit=None):
    """Private workflow adapter, only verified qualification earns this origin."""
    if type(qualification) is not _QualifiedLineage:
        raise ContractError('A diagnostic/engineering source cannot create a qualified BoundaryRecorder')
    facade, public, public_hash = _checkpoint_record(runner, state, checkpoint_path, checkpoint_hash)
    preparation = dict(qualification.preparation, split_complete_checkpoint_hash=checkpoint_hash,
        preceding_commit_sha256=preceding_commit, conditioned_continuation=True)
    source_identity = dict(identity, split_scientific_contract_sha256=runner.contract.sha256,
        split_complete_checkpoint_hash=checkpoint_hash, source_input_binding_hash=runner.binding['binding_hash'],
        preceding_commit_sha256=preceding_commit, scientific_fit_validation='NOT_RUN', fit_eligible=False,
        source_status='EXPLICIT_SPLIT_SOURCE_CANDIDATE_QUALIFIED_BACKGROUND')
    recorder = BoundaryRecorder(facade, public, background_channels, identity=source_identity,
        mode='rgc_only', registry=registry, _prepared_origin=_PreparedOrigin(public_hash, preparation))
    recorder.extractor = _inner_extractor(runner, recorder.channels)
    return recorder


def _endpoint(cache, index):
    # Time belongs to the chain separately; exact values and derivatives must
    # agree on both sides of an ordinary chunk seam.
    return canonical_hash(dict(values=cache.values[index], derivatives=cache.derivatives[index]))


def _save_selected_result(result, path, *, first_chunk):
    arrays = {k:q.values for k,q in result.outputs_with_units.items()}
    times = np.asarray(result.outputs_with_units['time'].values)
    # Each stored numerical trace contains its starting support point. This mask
    # declares logical sample ownership without deleting or moving that point.
    arrays['logical_sample_owned'] = np.ones(len(times), dtype=bool)
    if not first_chunk:
        arrays['logical_sample_owned'][0] = False
    declarations = {k:dict(unit=q.unit, meaning=q.meaning, evidence_tag=q.evidence_tag)
                    for k,q in result.outputs_with_units.items()}
    arrays['manifest_json'] = np.array(json.dumps(dict(ports=declarations, metadata=result.metadata,
        event_ownership='(chunk_start,chunk_stop] as original ResidentRecording.flush_events',
        support_sample_ownership='first chunk includes start; later start support is unowned'), allow_nan=False))
    np.savez_compressed(path, **arrays)


class SplitSourceSession:
    """Holds one current state/monitor, never the list of past caches/results."""
    def __init__(self, runner, state, store, *, reference_state, monitor_policy,
                 background_channels, channels, registry, identity, qualification,
                 guard, current_checkpoint, checkpoint_hash):
        self.runner, self.state, self.store = runner, state, store
        self.reference, self.policy = reference_state, monitor_policy
        self.background_channels, self.channels = background_channels, channels
        self.registry, self.identity = registry, identity
        self.qualification = _QualifiedLineage(store)
        self.guard = guard
        self.current_checkpoint = Path(current_checkpoint)
        self.checkpoint_hash = checkpoint_hash
        self.monitor = runner.make_monitor(reference_state, monitor_policy)
        runner.runtime_guard = guard.check

    @classmethod
    def from_qualified_background(cls, background_store, qualified_path, *, directory,
            reference_state, background_channels, movie_channels, channel_data_hashes,
            identity, registry, guard):
        from .split_source_background import SplitBackgroundCheckpointStore
        if type(background_store) is not SplitBackgroundCheckpointStore:
            raise ContractError('Original validated split background envelope required')
        runner = background_store.runner
        guard.require(force=True)
        state, envelope = background_store.load(qualified_path, require_converged=True)
        runner.contract.require_input_policy(movie_channels, phase='movie')
        # Preserve absolute phase/history and explicitly discard only the old
        # background lookahead, exactly as the existing continuation interface.
        state = runner.bind_continuation_inputs(state, movie_channels, channel_data_hashes, phase='movie')
        qualification = dict(status='verified_original_background_contract',
            binding_hash=envelope['binding_hash'], report_hash=envelope['report_hash'],
            complete_checkpoint_hash=envelope['complete_checkpoint_hash'],
            absolute_time_seconds=envelope['absolute_time_seconds'], policy_id=background_store.policy.profile_id,
            time_policy='unchanged_absolute_continuation', history_policy='unchanged_conditioned_preparation_history')
        binding = dict(model_hash=runner.model.model_hash, science_sha256=runner.contract.sha256,
            source_input_binding=runner.binding, identity=identity, qualification=qualification,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            target_graph_receipt_sha256=runner.contract.runtime_authorization['target_graph_receipt_sha256'],
            input_policy_receipt_sha256=runner.contract.runtime_authorization['input_policy_receipt_sha256'],
            original_teacher_background_frames=750, original_teacher_background_seconds=5.,
            source_time_origin_seconds=state.time_seconds, default_backend_changed=False, fitting_run=False)
        store = SplitChunkStore.create(directory, binding)
        checkpoint = save_complete(runner, state, store.root/'origin_checkpoint')
        write_json(store.root/'origin_receipt.json', checkpoint)
        return cls(runner, state, store, reference_state=reference_state, monitor_policy=background_store.policy,
            background_channels=background_channels, channels=movie_channels, registry=registry,
            identity=identity, qualification=qualification, guard=guard,
            current_checkpoint=store.root/'origin_checkpoint', checkpoint_hash=checkpoint['manifest_hash'])

    @classmethod
    def resume_from_qualified_background(cls, background_store, qualified_path, *, directory,
            expected_session_hash, reference_state, background_channels, movie_channels,
            channel_data_hashes, registry, guard):
        """Cold recipe: verify background, rebuild the exact phase binding, load head.

        No numerical time is advanced here. This cold-process path remains to be
        exercised as part of the first actual continuation, not a separate run.
        """
        from .split_source_background import SplitBackgroundCheckpointStore
        if type(background_store) is not SplitBackgroundCheckpointStore:
            raise ContractError('Verified split background store required for cold reconstruction')
        state, _ = background_store.load(qualified_path, require_converged=True)
        runner = background_store.runner
        runner.bind_continuation_inputs(state, movie_channels, channel_data_hashes, phase='movie')
        del state
        return cls.resume(runner, directory, expected_session_hash=expected_session_hash,
            reference_state=reference_state, monitor_policy=background_store.policy,
            background_channels=background_channels, channels=movie_channels, registry=registry, guard=guard)

    @classmethod
    def resume(cls, runner, directory, *, expected_session_hash, reference_state,
               monitor_policy, background_channels, channels, registry, guard):
        """Runner must already reconstruct the SAME movie binding, never relabel."""
        runner._check_runtime()
        guard.require(force=True)
        store = SplitChunkStore(directory, expected_session_hash=expected_session_hash)
        b = store.session['binding']
        if (b['model_hash'] != runner.model.model_hash or b['science_sha256'] != runner.contract.sha256
                or b['source_input_binding'] != runner.binding
                or b['implementation_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                or b['input_policy_receipt_sha256'] != runner.contract.runtime_authorization['input_policy_receipt_sha256']
                or b['target_graph_receipt_sha256'] != runner.contract.runtime_authorization['target_graph_receipt_sha256']):
            raise ContractError('Cold source restore requires the exact scientific/input/target binding')
        last = store.latest()
        if last is None:
            info = json.loads((store.root/'origin_receipt.json').read_bytes())
            path, digest = store.root/'origin_checkpoint', info['manifest_hash']
        else:
            path, digest = store.root/f"chunk-{last['index']:06d}"/'checkpoint', last['checkpoint_manifest_sha256']
            cache = BoundaryCache.load(path.parent/'cache', expected_identity_hash=last['cache_identity_sha256'])
            if cache.content_hash != last['cache_content_sha256']:
                raise ContractError('Last committed original boundary loader content differs')
            del cache
        state = load_complete(runner, path, expected_manifest_hash=digest)
        if last is not None and state.time_seconds != last['stop_seconds']:
            raise ContractError('Last checkpoint/time chain differs; no replay of completed chunks')
        runner.contract.validate_channels(channels, runner.binding)
        return cls(runner, state, store, reference_state=reference_state, monitor_policy=monitor_policy,
            background_channels=background_channels, channels=channels, registry=registry,
            identity=b['identity'], qualification=b['qualification'], guard=guard,
            current_checkpoint=path, checkpoint_hash=digest)

    def advance_one(self, stop, *, comparator_factory):
        """Actual forward entry. Caller chooses one explicit canonical stop."""
        if not callable(comparator_factory):
            raise ContractError('Useful capture requires its prospectively frozen cache replay comparator')
        runner, start = self.runner, self.state.time_seconds
        runner._check_runtime()
        self.guard.require(force=True)
        if stop <= start or stop-start > .100000000001:
            raise ContractError('First useful source chunks are bounded to <=100ms')
        stage, last = self.store.stage()
        if last is not None and last['stop_seconds'] != start:
            raise ContractError('A newer chunk is already committed; restore its head before any further forward')
        try:
            recorder = _make_source_recorder(runner, self.state, self.current_checkpoint, self.checkpoint_hash,
                background_channels=self.background_channels, identity=self.identity, registry=self.registry,
                qualification=self.qualification, preceding_commit=None if last is None else last['receipt_sha256'])
            self.monitor.reset()
            width = sum(len(runner.compiled.components[cid].inputs[port].cell_ids)
                        for _,cid,port in recorder.channels)
            row_limit = min(100000, (64*1024**2)//max(1,16*width))
            def observe(t,y,bundle):
                self.monitor.observe(t,y,bundle)  # complete original predicates
                if self.monitor.domain_failures:
                    raise ContractError('Source full monitor reported a domain failure')
                if len(recorder.rows) >= row_limit:
                    raise ContractError('Current source boundary block exceeded its bounded row storage')
                inner = y[runner.partition.offset:].detach().cpu().numpy()
                recorder.observe(t,inner,bundle)
            # Preserve all original 100us queue knots/sample times; split itself
            # adds input-side/event/native accepted endpoints to the observer.
            dt, origin = runner.observation_dt, self.state.grid_origin_seconds
            first = int(np.floor((start-origin)/dt+1e-8))+1
            last_tick = int(np.floor((stop-origin)/dt+1e-8))
            times = [start]+[origin+i*dt for i in range(first,last_tick+1) if origin+i*dt < stop]+[stop]
            states = tuple(s.state_name for s in runner.model.state_specs
                if runner.compiled.components[s.state_name.split('.',1)[0]].mechanism == RGC)
            ids = tuple(cell for c in runner.compiled.components.values() if c.mechanism == RGC
                        for cell in c.state_specs[0].cell_ids)
            plan = RecordingPlan(ports=(), states=states, cell_ids=ids, max_bytes=16*1024**2)
            comparator = comparator_factory(self.state, times, plan)
            from .split_boundary_replay_gate import CacheReadoutComparator
            if type(comparator) is not CacheReadoutComparator:
                raise ContractError('Source comparator must be sealed before forward execution')
            write_json(stage/'cache_readout_plan.json',dict(plan=comparator.plan,plan_sha256=comparator.plan_hash))
            tick = time.perf_counter()
            current, result = runner.advance(self.state, self.channels, stop,
                sample_times=times, recording_plan=plan, state_observer=observe)
            hot = time.perf_counter()-tick
            summary = self.monitor.report(stop-start)
            summary['interpretation'] = 'full scientific monitor; fixed-point norm is descriptive during movie, not qualification'
            breaks = sorted({t for ch in self.channels.values() for t in ch.breakpoints_seconds if start <= t <= stop})
            cache = recorder.finish(current, stimulus_breakpoints=breaks, background_status='verified_original_background_contract')
            cache.save(stage/'cache')
            checked = BoundaryCache.load(stage/'cache', expected_identity_hash=cache.identity_hash)
            if checked.content_hash != cache.content_hash:
                raise ContractError('Original schema1.2 loader rejected source content')
            from .split_boundary_replay_gate import check_baseline
            replay_gate = check_baseline(checked,Path(__file__).resolve().parents[1],result,current,
                comparator=comparator,sample_times=times,recording_plan=plan,evidence_directory=stage)
            write_json(stage/'baseline_replay.json',replay_gate)
            if any(v['status']!='PASS' for v in replay_gate['verdict'].values()):
                raise ContractError('Useful cache baseline replay old/B gate failed or incomplete; no commit')
            del checked
            _save_selected_result(result, stage/'selected_outputs.npz', first_chunk=last is None)
            del result
            write_json(stage/'monitor.json', summary)
            self.guard.require(force=True)
            checkpoint = save_complete(runner, current, stage/'checkpoint')
            write_json(stage/'execution.json', dict(advance_wall_seconds=hot,
                resource_sample=dict(self.guard.last), numerical_contract_sha256=runner.contract.sha256,
                permit_sha256=runner.contract._authorization_sha256,
                production_fit_validation='NOT_RUN', event_ownership='original (start,stop]',
                cache_is_numerical_boundary_approximation=True))
            row = self.store.commit(stage, start=start, stop=stop,
                first_boundary_hash=_endpoint(cache,0), last_boundary_hash=_endpoint(cache,-1),
                checkpoint_receipt=checkpoint, cache_identity=cache.identity_hash, cache_content=cache.content_hash)
            del cache, recorder
            self.state = current
            self.current_checkpoint = self.store.root/f"chunk-{row['index']:06d}"/'checkpoint'
            self.checkpoint_hash = row['checkpoint_manifest_sha256']
            self.store.retire_owned_checkpoints()
            self.guard.require(force=True, chunk_boundary=True)
            return row
        except Exception as exc:
            # Small witness only; last valid committed state remains recoverable.
            write_json(self.store.root/('failure-'+str(time.time_ns())+'.json'), dict(
                exception_type=type(exc).__name__, message=str(exc), start_seconds=start,
                requested_stop_seconds=stop, uncommitted_directory=stage.name,
                last_commit=self.store.latest(verify_files=False), resource_sample=dict(self.guard.last)))
            raise
