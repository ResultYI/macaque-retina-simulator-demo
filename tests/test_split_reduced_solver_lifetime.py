"""Completed inner slices release their workspace without changing integration."""
import gc
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch
import weakref

import numpy as np
import pytest
from scipy import integrate

REPORT = Path(__file__).resolve().parents[1]/'reports/cone_hc_reduced_foveal_6deg_run_20261024_v1'


def original(kind):
    path = REPORT/('baseline_before_lifetime_cone_hc_'+kind+'_runner.py')
    expected = {'split':'a14e872b9972951a36495f6f72178c4b11a17f248966c07a78e79db1930118d3',
                'reduced':'04d578a51d92102a8acf9e1f3cf5ef7cb652412600a8b4c0f0cfec1331e20ebc'}[kind]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location('retina_sim._lifetime_baseline_'+kind, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, 'ConeHCSplitRunner' if kind == 'split' else 'ReducedConeHCRunner')


def tracker():
    refs, accepted = [], []
    class Tracked(integrate.DOP853):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            refs.append((weakref.ref(self), weakref.ref(self.K_extended)))
        def step(self):
            result = super().step()
            accepted.append(float(self.t))
            return result
    return Tracked, refs, accepted


def plain(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key:plain(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def collect_result(probe):
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()  # Expose ownership; production global GC settings never change.
    try:
        result, refs, accepted = probe()
        return result, [bool(a() is not None) for a,b in refs], [bool(b() is not None) for a,b in refs], accepted
    finally:
        gc.collect()
        if was_enabled:
            gc.enable()


def test_reduced_actual_advance_events_ULP_queue_and_checkpoint_payload_unchanged():
    from retina_sim.cone_hc_reduced_runner import ReducedConeHCRunner
    from retina_sim.cone_hc_reduced_checkpoint import _payload
    from retina_sim.contracts import canonical_hash
    from tests.test_cone_hc_reduced_interfaces import test_added_native_endpoints_do_not_move_original_event_ticks_by_one_ULP
    methods = [original('reduced').advance, ReducedConeHCRunner.advance]
    outputs = []
    for method in methods:
        def probe():
            tracked, refs, accepted = tracker()
            result = {}
            def capture(self, *args, **kwargs):
                state = method(self, *args, **kwargs)
                result.update(payload=canonical_hash(plain(_payload(self, state))),
                              nfev=self.metrics['inner_nfev'])
                return state
            with patch.object(integrate, 'DOP853', tracked), patch.object(ReducedConeHCRunner, 'advance', capture):
                test_added_native_endpoints_do_not_move_original_event_ticks_by_one_ULP()
            return result, refs, accepted
        outputs.append(collect_result(probe))
    assert outputs[0][0] == outputs[1][0]
    assert outputs[0][3] == outputs[1][3] and outputs[1][3]
    assert all(outputs[0][1]) and all(outputs[0][2])
    assert not any(outputs[1][1]) and not any(outputs[1][2])


def test_split_actual_integrate_dense_rows_clocks_and_evaluations_unchanged():
    import torch
    from retina_sim.cone_hc_split_runner import ConeHCSplitRunner
    from retina_sim.contracts import SolverConfig
    outputs = []
    for method in (original('split')._integrate, ConeHCSplitRunner._integrate):
        def probe():
            tracked, refs, accepted = tracker()
            acknowledgements, rows = [], []
            cursor = NS(segments=[], state_at=lambda t:torch.ones(1, dtype=torch.float64),
                        acknowledge=lambda *args:acknowledgements.append(args))
            runner = NS(partition=NS(offset=1), torch=torch, _check_runtime=lambda:None,
                _accept_native_until=lambda *args:cursor, _inner_rhs=lambda c,s,e,t,y:-1000.*y,
                split_metrics=dict(inner_nfev=0, inner_accepts=0), integration_metrics=dict(
                    solver_calls=0, nfev=0, integration_wall_seconds=0., accepted_steps=0,
                    minimum_interval_seconds=None, maximum_nfev_per_interval=0))
            state = NS(y=torch.ones(65, dtype=torch.float64), time_seconds=0.)
            def commit(y,t,r):
                state.y, state.time_seconds = y, t
            state.commit = commit
            with patch.object(integrate, 'DOP853', tracked):
                for end in (.0001, .0002, .0003):
                    returned = method(runner, state, None, end, SolverConfig(), [(state.time_seconds+end)/2, end])
                    rows.append(tuple((t,y.numpy().tobytes()) for t,y in sorted(returned.items())))
            return dict(rows=rows, nfev=runner.split_metrics['inner_nfev'],
                        accepts=runner.split_metrics['inner_accepts'], time=state.time_seconds,
                        acknowledgements=acknowledgements), refs, accepted
        outputs.append(collect_result(probe))
    assert outputs[0][0] == outputs[1][0] and outputs[0][3] == outputs[1][3]
    assert all(outputs[0][1]) and all(outputs[0][2])
    assert not any(outputs[1][1]) and not any(outputs[1][2])
