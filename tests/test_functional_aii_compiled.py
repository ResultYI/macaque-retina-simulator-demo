"""The optional execution backend must preserve both current and legacy equations."""
from copy import deepcopy

import numpy as np
import pytest

from retina_sim.ac_vs_compiled import CompiledACVSInnerRHS
from retina_sim.aii_functional import S_KEY
from test_aii_functional import reference_fixture


@pytest.mark.parametrize('functional', [False, True])
def test_compiled_equations_and_held_states_match_reference(functional):
    old, original, new, current = reference_fixture()
    model, state = (new, current) if functional else (old, original)
    compiled = CompiledACVSInnerRHS(model)
    for gate in (0., .37, 1.):
        probe = deepcopy(state)
        for key in probe:
            if key[-1] == 'V':
                probe[key] += np.linspace(-.001,.001,len(probe[key]))
        if functional:
            probe[S_KEY][:] = gate
            probe['rod_BC','ReceptorConductanceMembrane','release'] *= .9
        value = model.pack(probe)
        before = value.copy()
        actual = compiled.rhs_flat(0.,value)
        expected = model.pack(model.rhs(0.,probe))
        np.testing.assert_allclose(actual,expected,rtol=5e-14,atol=2e-13)
        np.testing.assert_array_equal(value,before)
        for key, section in model.slices.items():
            if key[-1].startswith('history_') or key[-1]=='activation':
                np.testing.assert_array_equal(actual[section],expected[section])
