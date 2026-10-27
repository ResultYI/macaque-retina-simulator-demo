"""Focused new-layout lowering check; original equations remain the oracle."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from retina_sim.ac_vs_compiled import CompiledACVSInnerRHS
from retina_sim.ac_vs_model import MECHANISM, LEGACY_S
from retina_sim.contracts import ContractError
from retina_sim.group_gpu_rhs import GroupGpuRHS
from retina_sim.group_inner_dynamics import BC_MECH


def fixture():
    spec=importlib.util.spec_from_file_location('new_ac_fixture',Path(__file__).with_name('test_ac_vs_model.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    _,_,model,state=module.new_fixture()
    for i,c in enumerate(model.channels):
        state[c['pool'],MECHANISM,'V'][:]=-.045+i*.003
        state[c['pool'],MECHANISM,'s'][:]=.13+i*.11
        state[c['pre'],BC_MECH,'terminal_release'][:]=.2+i*.1
    return model,state


def test_complete_typed_Vs_matches_original_numpy():
    model,state=fixture();compiled=CompiledACVSInnerRHS(model)
    y=model.pack(state);before=y.copy()
    np.testing.assert_allclose(compiled(0.,y),model.pack(model.rhs(0.,state)),rtol=5e-14,atol=2e-13)
    np.testing.assert_array_equal(y,before)
    for c in model.channels:
        assert compiled.derivatives[list(model.state_keys).index((c['pool'],MECHANISM,'V')),1]>=0
        assert compiled.derivatives[list(model.state_keys).index((c['pool'],MECHANISM,'s')),1]>=0


def test_CUDA_same_Vs_full_layout_and_domain():
    import torch
    model,state=fixture();compiled=CompiledACVSInnerRHS(model);gpu=GroupGpuRHS(compiled,torch)
    try:
        y=model.pack(state);device=torch.as_tensor(y.copy(),dtype=torch.float64,device='cuda');result=torch.empty_like(device)
        gpu.evaluate_into(0.,device,result)
        np.testing.assert_allclose(result.cpu().numpy(),model.pack(model.rhs(0.,state)),rtol=5e-14,atol=2e-13)
        np.testing.assert_array_equal(device.cpu().numpy(),y)
        bad=y.copy();bad[model.slices[model.channels[0]['pool'],MECHANISM,'s']]=1.01
        with pytest.raises(ContractError):gpu.evaluate_into(0.,torch.as_tensor(bad,device='cuda'),result)
        gpu.evaluate_into(0.,device,result)
        np.testing.assert_allclose(result.cpu().numpy(),compiled(0.,y),rtol=5e-14,atol=2e-13)
    finally:gpu.close()
