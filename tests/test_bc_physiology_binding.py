"""Check the conductance decomposition against the retained current equation."""
import numpy as np

from retina_sim.group_parameter_binding import bc_glycine_dc_conductance
from retina_sim.group_inner_dynamics import BC_MECH
from test_group_inner_dynamics import fixture_model


def test_glycine_dc_slope_includes_voltage_dependent_steady_kinetics():
    model, state = fixture_model()
    for component in ('AII_FMB_glycine', 'AII_DB1_glycine'):
        c = model.contacts[component]
        pop = c['route'].post
        voltage = state[pop, BC_MECH, 'V']
        chord, slope = bc_glycine_dc_conductance(model, state, component, voltage)
        eps = 1e-7
        gp, _ = bc_glycine_dc_conductance(model, state, component, voltage+eps)
        gm, _ = bc_glycine_dc_conductance(model, state, component, voltage-eps)
        force = model.par(pop, 'EI')-voltage
        actual = -(gp*(force-eps)-gm*(force+eps))/(2*eps)
        np.testing.assert_allclose(slope, actual, rtol=2e-9, atol=1e-21)
        assert np.all(slope > chord)
