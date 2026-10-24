"""Explicit observation-only queue nodes; no solver-endpoint epsilon matching.

The legacy split implementation stays byte-exact for authentic checkpoint load.
The explicit migration below changes only future queue membership and identity.
"""
from copy import copy
from pathlib import Path
from threading import RLock
import hashlib

VERSION = 'cone-hc-explicit-observation-queue-1'


def queue_contract():
    return dict(version=VERSION,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        nodes='exact requested samples and original outer interval ends',
        near_grid_solver_endpoints=False, inherited_history='preserve verbatim',
        interpolation_pruning_delay='unchanged', feedback='current h, never observation queue')


def is_queue_node(t, samples, interval_stop):
    return t in samples or t == interval_stop


def upgrade_split_reference(runner, state):
    """Explicit fork AFTER strict legacy restore; never relabel a saved receipt."""
    from .contracts import ContractError, canonical_hash
    from .cone_hc_split_runner import ConeHCSplitRunner, ConeHCSplitState
    from .resident_runner import ResidentNativeRunner

    if type(runner) is not ConeHCSplitRunner or type(state) is not ConeHCSplitState:
        raise ContractError('Queue migration requires an authentic legacy split runner/state')

    class ExactQueueSplitRunner(ConeHCSplitRunner):
        def _update_observation_delays(self, state, t, y):
            if t > state.time_seconds:
                raise ContractError('Native lookahead cannot populate the public observation delay queue')
            if is_queue_node(t, self._queue_sample_times, self._queue_interval_stop):
                ResidentNativeRunner._update_observation_delays(self, state, t, y)

    with runner._lock:
        runner._check_runtime()
        state.validate(runner)
        other = ExactQueueSplitRunner.__new__(ExactQueueSplitRunner)
        other.__dict__ = dict(runner.__dict__)
        # Old diagnostic instance hooks must not bypass the corrected policy.
        other.__dict__.pop('_update_observation_delays', None)
        other.model, other.partition = copy(runner.model), copy(runner.partition)
        other.partition.work = runner.partition.work.copy()
        other.partition.statistics = dict(runner.partition.statistics)
        other.partition._light_cpu = other.partition._light_gpu = None
        other.workspace = other.native_workspace = None
        other._lock = RLock()
        other.integration_metrics = dict(runner.integration_metrics)
        other.split_metrics = dict(runner.split_metrics)
        other.evaluator = copy(runner.evaluator)
        other.evaluator.runner = other
        identity = dict(legacy_model_hash=runner.model.model_hash, observation_queue=queue_contract())
        other.model.model_hash = canonical_hash(identity)
        other.model.compute_metadata = dict(runner.model.compute_metadata, observation_queue_upgrade=identity)
        migrated = state.fork()
        migrated.metadata['model_hash'] = other.model.model_hash
        migrated.input_position['observation_queue_upgrade'] = dict(identity, time_seconds=state.time_seconds)
        migrated.validate(other)
        return other, migrated


def queue_time_evidence(delay_states, *, time_seconds):
    """CPU timestamp metadata only; no full HC values or device transfer."""
    return dict(schema='cone-hc-queue-time-evidence-1', time_seconds=float(time_seconds),
        observation_queue=queue_contract(), components={cid:dict(
            seconds=[float(t) for t in delay.history_times],
            hex=[float(t).hex() for t in delay.history_times])
            for cid, delay in sorted(delay_states.items())})


def compare_queue_times(reference, candidate):
    """Keep complete timestamp lists and an exact first-difference witness."""
    rows = {}
    for cid in sorted(set(reference['components']) | set(candidate['components'])):
        a, b = reference['components'].get(cid), candidate['components'].get(cid)
        ah, bh = ([] if a is None else a['hex']), ([] if b is None else b['hex'])
        equal = a is not None and b is not None and ah == bh
        first = next((i for i in range(max(len(ah), len(bh)))
                      if i >= len(ah) or i >= len(bh) or ah[i] != bh[i]), None)
        def node(value, index):
            return (None if value is None or index is None or index >= len(value['hex']) else
                    dict(seconds=value['seconds'][index], hex=value['hex'][index]))
        rows[cid] = dict(status='PASS' if equal else 'FAIL', reference_count=len(ah), candidate_count=len(bh),
            first_difference=None if equal else dict(index=first, reference=node(a,first), candidate=node(b,first),
                missing_component='reference' if a is None else 'candidate' if b is None else None))
    same_time = float(reference['time_seconds']).hex() == float(candidate['time_seconds']).hex()
    same_contract = reference['observation_queue'] == candidate['observation_queue']
    return dict(schema='cone-hc-queue-time-comparison-1',
        status='PASS' if same_time and same_contract and all(r['status']=='PASS' for r in rows.values()) else 'FAIL',
        same_public_time=same_time, same_queue_contract=same_contract,
        reference=reference, candidate=candidate, components=rows)
