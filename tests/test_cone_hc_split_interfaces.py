"""Written for future authorized regression; NOT EXECUTED in implementation turn.

These cases require no graph construction/CUDA. They do not certify numerical
accuracy, events or performance. CUDA solver cases are separately opt-in.
"""
from copy import copy
from types import SimpleNamespace as NS
import unittest
import numpy as np

from retina_sim.contracts import ContractError
from retina_sim.cone_hc_split_contract import ConeHCSplitContract, guard_block_triangular
from retina_sim.cone_hc_split_readout import AcceptedNativeCursor
from retina_sim.cone_hc_split_runner import ConeHCSplitRunner, next_native_input_bound, native_input_at
from retina_sim.cone_hc_split_runner import ConeHCSplitState


class SplitInterfaceTests(unittest.TestCase):
    def test_checked_in_manifest_refuses_execution_before_runner_or_device_access(self):
        c = ConeHCSplitContract()
        self.assertFalse(c.candidate['execution_enabled'])
        self.assertFalse(c.candidate['acceptance']['proposed_budget_enabled'])
        self.assertIsNone(c.candidate['acceptance']['approved_contract_sha256'])
        with self.assertRaisesRegex(ContractError, 'PENDING_APPROVAL'):
            ConeHCSplitRunner.from_resident(object(), purpose='numerical_qa', numerical_contract=c)

    @staticmethod
    def graph():
        native, inner = 'full_native', 'inner'
        return NS(partition=NS(cid=native, offset=100), nstate=110, periods={inner: .0001},
                  compiled=NS(components={native: NS(time_contract=NS(kind='continuous'))},
                              links=[], output_order=[],
                              spec=NS(external_inputs=[NS(target_component=native, target_port='light')])),
                  model=NS(slices={native: slice(0, 100)}, _context_providers={native: object()}, _gap_plans=[]))

    def test_actual_topology_rejects_inner_to_native_and_cross_gap(self):
        r = self.graph()
        self.assertTrue(guard_block_triangular(r)['no_inner_to_native'])
        link = NS(specification=NS(source_component='inner', target_component='full_native', source_port='release'),
                  delayed_operator=None)
        r.compiled.links = [link]
        with self.assertRaisesRegex(ContractError, 'incoming native'):
            guard_block_triangular(r)
        r.compiled.links = []
        r.model._gap_plans = [('full_native', 'inner', None, None, None)]
        with self.assertRaisesRegex(ContractError, 'gap feedback'):
            guard_block_triangular(r)

    def test_native_bound_ignores_downstream_ticks_and_chunks(self):
        channel = NS(breakpoints_seconds=(.006, .012), stop_seconds=.02)
        self.assertEqual(next_native_input_bound(channel, .0001), (.006, True))
        self.assertEqual(next_native_input_bound(channel, .006), (.012, True))
        self.assertEqual(next_native_input_bound(channel, .012), (.02, False))
        with self.assertRaises(ContractError):
            next_native_input_bound(channel, .02)

    def test_native_zoh_left_side_uses_no_forward_epsilon(self):
        seen = []
        channel = NS(interpolation='zero_order_hold', start_seconds=0., stop_seconds=.02,
                     at=lambda t: seen.append(t) or t)
        t = .006
        self.assertEqual(native_input_at(channel, t, left_limit=True), np.nextafter(t, -np.inf))
        self.assertEqual(native_input_at(channel, t), t)
        channel.interpolation = 'linear'
        self.assertEqual(native_input_at(channel, t, left_limit=True), t)

    def test_observation_queue_rejects_native_lookahead(self):
        runner = ConeHCSplitRunner.__new__(ConeHCSplitRunner)
        runner.observation_dt = .0001
        with self.assertRaisesRegex(ContractError, 'lookahead cannot populate'):
            runner._update_observation_delays(NS(time_seconds=.0001), .0005, None)

    def test_original_requested_dense_queue_knots_are_retained_but_new_block_points_are_not(self):
        from unittest.mock import patch
        from retina_sim.resident_runner import ResidentNativeRunner
        runner = ConeHCSplitRunner.__new__(ConeHCSplitRunner)
        runner.observation_dt = .0001
        runner._queue_sample_times = frozenset((.000031,))
        runner._queue_interval_stop = .0001
        state = NS(time_seconds=.0001, grid_origin_seconds=0.)
        with patch.object(ResidentNativeRunner, '_update_observation_delays') as old:
            runner._update_observation_delays(state, .000031, None)
            runner._update_observation_delays(state, .000037, None)
            self.assertEqual(old.call_count, 1)

    def test_input_break_observes_left_then_right_without_event_or_rng(self):
        runner = ConeHCSplitRunner.__new__(ConeHCSplitRunner)
        stimulus = NS(channels={'native': NS(breakpoints_seconds=(.006,))},
                      at=lambda t, left_limit=False: (t, left_limit))
        calls = []
        runner._observe(lambda t, y, q: calls.append(q), stimulus, .006, object())
        self.assertEqual(calls, [(.006, True), (.006, False)])

    def test_state_fork_does_not_share_mutable_rng_or_history(self):
        state = ConeHCSplitState(y=object(), time_seconds=0., metadata={}, assembly_hash='assembly',
                                grid_origin_seconds=0., sample_index=0,
                                spike_states={'rgc': {'history': [1, 0, 0], 'rng_state': {'state': [42]}}},
                                delay_states={}, input_position={}, link_delay_states={},
                                component_tick_indices={'rgc': 0})
        child = state.fork()
        child.spike_states['rgc']['history'][0] = 0
        child.spike_states['rgc']['rng_state']['state'][0] = 99
        self.assertEqual(state.spike_states['rgc']['history'], [1, 0, 0])
        self.assertEqual(state.spike_states['rgc']['rng_state']['state'], [42])


class FakeSolver:
    def __init__(self):
        self.t, self.y = 0., None

    def fork(self):
        return copy(self)


class FakeSegment:
    def __init__(self, start, stop):
        self.start, self.stop = start, stop


class CursorOwnershipTests(unittest.TestCase):
    def cursor(self):
        return AcceptedNativeCursor(FakeSolver(), NS(project=lambda s: s), 'input_identity')

    def test_native_ahead_does_not_advance_observers_or_prune_their_history(self):
        c = self.cursor()
        c.solver.t = .0005
        c.append(FakeSegment(0., .0005))
        c.acknowledge('inner', .0005)
        self.assertEqual(len(c.segments), 1)
        self.assertEqual(c.consumed['observation'], 0.)
        for name in ('monitor', 'recording', 'observation'):
            c.acknowledge(name, .0005)
        self.assertEqual(c.segments, [])

    def test_fork_owns_controller_and_watermarks_but_shares_immutable_segments(self):
        c = self.cursor()
        c.solver.t = .0005
        c.append(FakeSegment(0., .0005))
        other = c.fork()
        other.solver.t = .001
        other.acknowledge('inner', .0005)
        self.assertEqual(c.solver.t, .0005)
        self.assertEqual(c.consumed['inner'], 0.)
        self.assertIs(other.segments[0], c.segments[0])

    def test_no_unaccepted_future_query(self):
        c = self.cursor()
        c.solver.t = .0005
        c.append(FakeSegment(0., .0005))
        with self.assertRaisesRegex(ContractError, 'outside retained accepted'):
            c.segment(.0006)

    def test_boundary_side_selects_old_or_new_accepted_epoch(self):
        c = self.cursor()
        c.solver.t = .0005
        a = FakeSegment(0., .0005)
        c.append(a)
        c.solver.t = .001
        b = FakeSegment(.0005, .001)
        c.append(b)
        self.assertIs(c.segment(.0005, left_limit=True)[0], a)
        self.assertIs(c.segment(.0005, left_limit=False)[0], b)


if __name__ == '__main__':
    unittest.main()
