from pathlib import Path
import importlib.util

import numpy as np
import pytest

from retina_sim.contracts import ContractError
from retina_sim.group_inner_compiled import CompiledGroupInnerRHS
from retina_sim.group_inner_dynamics import key,BC_MECH,AC_MECH,MEMBRANE


@pytest.fixture(scope='module')
def compiled_model():
    path=Path(__file__).with_name('test_group_inner_dynamics.py')
    spec=importlib.util.spec_from_file_location('group_inner_cpu_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model,state=module.fixture_model()
    return model,state,CompiledGroupInnerRHS(model)


def test_compiled_full_equations_match_independent_numpy_reference(compiled_model):
    model,state,compiled=compiled_model
    altered={k:v.copy() for k,v in state.items()}
    altered[key('AII_AC',MEMBRANE,'V')][:]=[-.04,-.07]
    altered[key('midget_BC_ON',BC_MECH,'V')][:]=[-.06,-.05]
    altered[key('midget_BC_OFF',BC_MECH,'V')][:]=-.08
    altered[key('local_feedforward_AC',AC_MECH,'h')][:]=.3
    for c,row in model.contacts.items():
        if c!='individual_rod_RBC_contacts':
            altered[key(row['route'].pre,c,'a')][:]=.01
            altered[key(row['route'].pre,c,'b')][:]=.1
    y=model.pack(altered)
    expected=model.pack(model.rhs(.00013,altered))
    actual=compiled(.00013,y)
    np.testing.assert_allclose(actual,expected,rtol=5e-14,atol=2e-13)
    # History, probabilities and source rod author states remain exactly held.
    for k in model.state_keys:
        if k[1] in ('Chen2024PrimateRodCurrent','RGC_expected_history') or k[2].startswith('history_') or k[2]=='activation':
            np.testing.assert_array_equal(actual[model.slices[k]],0.)
    assert compiled._dispatch.nopython_signatures
    assert compiled.metrics['fine_runtime_constructed'] is False


def test_explicit_rod_current_and_failed_domain_do_not_mutate_state(compiled_model):
    model,state,compiled=compiled_model
    y=model.pack(state);saved=y.copy();current=np.array([-1e-11,-2e-11])
    reference=model.pack(model.rhs(0.,state,rod_current=current))
    np.testing.assert_allclose(compiled(0.,y,rod_current=current),reference,rtol=5e-14,atol=2e-13)
    np.testing.assert_array_equal(y,saved)
    invalid=y.copy();invalid[model.slices[key('local_feedforward_AC',AC_MECH,'h')]]=2.
    with pytest.raises(ContractError,match='domain'):
        compiled(0.,invalid)
    np.testing.assert_array_equal(y,saved)
    with pytest.raises(ContractError,match='FP64'):
        compiled(0.,y.astype(np.float32))


def test_selective_program_preserves_ordered_gap_glycine_and_scatter(compiled_model):
    model,state,compiled=compiled_model
    requested=(key('midget_BC_OFF',BC_MECH,'V'),key('AII_AC','AII_FMB_glycine','b'))
    selected=compiled.for_keys(requested)
    assert selected is compiled.for_keys(requested[::-1])
    assert selected.metrics['instruction_count']<compiled.metrics['instruction_count']
    for scale in (1.,1.1):
        current={k:v.copy() for k,v in state.items()}
        current[key('AII_AC','AII_FMB_glycine','a')][:]=.01
        current[key('AII_AC','AII_FMB_glycine','b')][:]=.1*scale
        current[key('AII_AC',MEMBRANE,'V')][:]=[-.04,-.06]
        y=model.pack(current);full=compiled(0.,y);partial=selected(0.,y)
        for k in model.state_keys:
            np.testing.assert_array_equal(partial[model.slices[k]],full[model.slices[k]] if k in requested else 0.)
    assert key('midget_BC_OFF',BC_MECH,'V') in selected.required_state_keys
    assert key('AII_AC',MEMBRANE,'x') in selected.required_state_keys
    assert not any(k[0].startswith('parasol_RGC') for k in selected.required_state_keys)


def test_selective_RGC_does_not_evaluate_other_contacts_or_histories(compiled_model):
    model,state,compiled=compiled_model
    requested=(('midget_RGC_ON','RGC_spike_history','V'),)
    selected=compiled.for_keys(requested)
    assert not any(k[1] in ('RBC_AII_AMPA','AII_FMB_glycine','AII_DB1_glycine') for k in selected.required_state_keys)
    y=model.pack(state)
    expected=compiled(0.,y)[model.slices[requested[0]]]
    np.testing.assert_array_equal(selected(0.,y)[model.slices[requested[0]]],expected)
    assert selected.metrics['instruction_count']<compiled.metrics['instruction_count']/2
