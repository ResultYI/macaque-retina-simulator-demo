"""Only the newly introduced binding; no repeat whole-model validation suite."""
from types import SimpleNamespace
import numpy as np
import pytest
from retina_sim.bc_effective_time_candidate import EffectiveTimeAssembly, CAPACITANCE
from retina_sim.contracts import ContractError


class Source:
    artifact_sha256='parent'
    runtime_factories={}
    manifest={'files':{CAPACITANCE:{'source':'retained'}},'gap':{'unchanged':True}}
    populations={'diffuse_BC_ON':SimpleNamespace(counts=np.array([1,3]))}
    def array(self,name):
        assert name==CAPACITANCE
        return np.array([20e-12,60e-12])
    def matrix(self,name):
        return name


def seed():
    return dict(states={('x','y','z'):np.array([4.])},
        spike_states={'PCG64':'kept'},observation_prehistory={'values':[1,2]},
        inner_parameters=dict(identity='original',populations={'diffuse_BC_ON':dict(C=20e-12,gL=1e-9),
            'diffuse_BC_OFF':dict(C=20e-12,gL=1e-9)},route_gains={'E':1e-9}))


def test_effective_c_is_real_total_and_other_fields_fixed():
    original=seed();a=EffectiveTimeAssembly(Source(),original,.010)
    assert a.seed['inner_parameters']['populations']['diffuse_BC_ON']['C']==original['inner_parameters']['populations']['diffuse_BC_ON']['gL']*.010
    assert np.array_equal(a.array(CAPACITANCE),original['inner_parameters']['populations']['diffuse_BC_ON']['gL']*.010*np.array([1,3]))
    assert a.seed['inner_parameters']['populations']['diffuse_BC_OFF']==original['inner_parameters']['populations']['diffuse_BC_OFF']
    assert a.seed['inner_parameters']['route_gains']==original['inner_parameters']['route_gains']
    assert a.seed['spike_states']==original['spike_states']
    assert a.seed['observation_prehistory']==original['observation_prehistory']
    assert a.manifest['files']==Source.manifest['files']
    assert original['inner_parameters']['populations']['diffuse_BC_ON']['C']==20e-12
    assert a.artifact_sha256!=Source.artifact_sha256


def test_baseline_bytes_and_parameter_limits():
    from copy import copy
    a=EffectiveTimeAssembly(Source(),seed(),.020)
    assert a.seed['inner_parameters']['populations']['diffuse_BC_ON']['C']==20e-12
    assert np.array_equal(a.array(CAPACITANCE),Source().array(CAPACITANCE))
    b=copy(a)
    assert b.array(CAPACITANCE) is a.total_c
    for tau in (.0049,.0301,float('nan')):
        with pytest.raises(ContractError):EffectiveTimeAssembly(Source(),seed(),tau)
