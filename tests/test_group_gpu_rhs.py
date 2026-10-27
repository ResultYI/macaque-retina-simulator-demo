from pathlib import Path
import importlib.util

import numpy as np
import pytest

from retina_sim.contracts import ContractError
from retina_sim.group_inner_compiled import CompiledGroupInnerRHS
from retina_sim.group_gpu_rhs import GroupGpuRHS
from retina_sim.group_inner_dynamics import AC_MECH, MEMBRANE


@pytest.fixture(scope='module')
def execution():
    import torch
    path=Path(__file__).with_name('test_group_inner_dynamics.py')
    spec=importlib.util.spec_from_file_location('gpu_group_fixture',path)
    fixture=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    model,state=fixture.fixture_model()
    compiled=CompiledGroupInnerRHS(model)
    gpu=GroupGpuRHS(compiled,torch)
    yield model,state,compiled,gpu,torch
    gpu.close()


def compare(compiled,gpu,torch,t,y):
    reference=compiled(t,y)
    device=torch.as_tensor(y.copy(),device='cuda')
    output=torch.empty_like(device)
    gpu.evaluate_into(t,device,output)
    np.testing.assert_array_equal(device.cpu().numpy(),y)
    np.testing.assert_allclose(output.cpu().numpy(),reference,rtol=5e-14,atol=2e-13)
    return output.cpu().numpy()


def test_full_equations_contacts_gap_and_held_states(execution):
    model,state,compiled,gpu,torch=execution
    altered={k:v.copy() for k,v in state.items()}
    altered['AII_AC',MEMBRANE,'V'][:]=[-.04,-.07]
    altered['local_feedforward_AC',AC_MECH,'h'][:]=.3
    for c,row in model.contacts.items():
        if c!='individual_rod_RBC_contacts':
            altered[row['route'].pre,c,'a'][:]=.01
            altered[row['route'].pre,c,'b'][:]=.1
    y=model.pack(altered)
    actual=compare(compiled,gpu,torch,.00013,y)
    np.testing.assert_allclose(actual,model.pack(model.rhs(.00013,altered)),rtol=5e-14,atol=2e-13)
    for dest,source,n in compiled.derivatives:
        if source<0: np.testing.assert_array_equal(actual[dest:dest+n],np.zeros(n))


def test_failed_domain_and_nonfinite_state_recover_without_mutation(execution):
    model,state,compiled,gpu,torch=execution
    y=model.pack(state)
    for kind in ('domain','nonfinite'):
        invalid=y.copy()
        if kind=='domain': invalid[model.slices['local_feedforward_AC',AC_MECH,'h']]=2.
        else: invalid[0]=np.nan
        for fn in (lambda:compiled(0.,invalid),lambda:gpu.evaluate_into(0.,torch.as_tensor(invalid,device='cuda'),torch.empty(y.shape,dtype=torch.float64,device='cuda'))):
            with pytest.raises(ContractError): fn()
        compare(compiled,gpu,torch,0.,y)


def test_fp64_layout_alias_and_stream_guards(execution):
    model,state,compiled,gpu,torch=execution
    y=torch.as_tensor(model.pack(state),device='cuda')
    with pytest.raises(ContractError,match='FP64'): gpu.evaluate_into(0.,y.float(),torch.empty_like(y))
    with pytest.raises(ContractError,match='alias'): gpu.evaluate_into(0.,y,y)
    with pytest.raises(ContractError,match='Finite'): gpu.evaluate_into(float('nan'),y,torch.empty_like(y))
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(ContractError,match='stream'): gpu.evaluate_into(0.,y,torch.empty_like(y))
    compare(compiled,gpu,torch,0.,model.pack(state))


def test_bitwise_repeatability_and_original_intermediate_workspace(execution):
    model,state,compiled,gpu,torch=execution
    y=model.pack(state)
    first=compare(compiled,gpu,torch,0.,y)
    second=compare(compiled,gpu,torch,0.,y)
    np.testing.assert_array_equal(first,second)
    # Every allocation is retained, including all checks and scatter targets.
    np.testing.assert_allclose(gpu.work.cpu().numpy(),compiled._scratch.array,rtol=5e-14,atol=2e-13)
