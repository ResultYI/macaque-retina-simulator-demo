"""Refinement and typed history contracts; fixtures are not physiology data."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from retina_sim.bc_independent import member_incidence, temporal_profile
from retina_sim.contracts import ContractError
from retina_sim.group_inner_dynamics import BC_MECH, GroupInnerDynamics
from test_group_inner_dynamics import fixture_model


def configuration():
    return json.loads((Path(__file__).resolve().parents[1]/'configurations/research/BC_macaque_fullchain.json').read_bytes())


def test_original_members_are_not_merged_or_reselected():
    parent = SimpleNamespace(group_count=2, fine_to_group=np.array([-1, 0, 1, 0]))
    child = SimpleNamespace(group_count=3, member_global_rows=np.array([3, 1, 2]))
    matrix = member_incidence(parent, child)
    np.testing.assert_array_equal(matrix @ np.array([2., 5., 11.]), [7., 11.])
    child.member_global_rows[0] = 0
    with pytest.raises(ContractError): member_incidence(parent, child)


def test_typed_profile_is_constant_but_not_a_shared_mutable_object():
    cfg = configuration()
    p = temporal_profile(cfg, 'midget_BC_ON', [0., 6.])
    p['kernel'][0] = 17.
    assert temporal_profile(cfg, 'midget_BC_ON', 3.)['kernel'] == [.5, .25, -.1]
    for e in (-.001, 6.001, np.nan):
        with pytest.raises(ContractError): temporal_profile(cfg, 'midget_BC_ON', e)
    cfg['temporal_profiles']['midget_BC_ON']['tau_release_seconds'] = float('nan')
    with pytest.raises(ContractError): temporal_profile(cfg, 'midget_BC_ON', 3.)


def test_distinct_typed_kernels_reach_original_clock_without_cross_type_changes():
    model, state = fixture_model()
    parameters = deepcopy(model.parameters)
    parameters['populations']['midget_BC_ON']['kernel'] = [.8, .1, -.05]
    changed = GroupInnerDynamics(model.assembly, parameters)
    before = {k:v.copy() for k,v in state.items()}
    old, _, _ = model.commit_ticks(.001, state, cone_release=np.array([.9]), bc_tick=True)
    new, _, _ = changed.commit_ticks(.001, state, cone_release=np.array([.9]), bc_tick=True)
    key = ('midget_BC_ON', BC_MECH, 'activation')
    assert not np.array_equal(old[key], new[key])
    for k in old:
        if k != key: np.testing.assert_array_equal(old[k], new[k])
    for k in state: np.testing.assert_array_equal(state[k], before[k])


