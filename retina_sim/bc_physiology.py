"""Explicit new BC research construction over the existing source-bound network."""
from pathlib import Path
import json

from .contracts import ContractError, canonical_hash
from .group_parameter_binding import bind_bc_physiology
from .group_time_runner import GroupTimeArm
from .h1_h2_effective import H1H2GroupRunner


DEFAULT_CONFIGURATION = 'configurations/research/BC_PHYSIOLOGY_2026-11-02.json'


class BCPhysiologyModel:
    def __init__(self, repo_root, configuration=DEFAULT_CONFIGURATION):
        from tools.gin_hc2_bc1_context_support_v2 import load
        self.root = Path(repo_root).resolve()
        path = self.root/configuration
        self.configuration = json.loads(path.read_bytes())
        if self.configuration.get('source_context') != 'GIN_69_6_HC2_BC1_SUPPORT_V2':
            raise ContractError('BC source context is not approved for this construction')
        assembly, original = load()
        if assembly.repo_root.resolve() != self.root:
            raise ContractError('BC research root differs from source context')
        self.assembly, self.seed = bind_bc_physiology(assembly, original, self.root, path)
        self.identity = canonical_hash(dict(artifact=self.assembly.artifact_sha256,
            binding=self.seed['bc_physiology'], parameters=self.seed['inner_parameters']))

    def runner(self, protocol):
        return BCPhysiologyRunner(self.assembly, self.seed, protocol)


class BCPhysiologyRunner(H1H2GroupRunner):
    """Original CPU scheduler, history events and complete checkpoint transport."""
    def __init__(self, assembly, seed, protocol, *, mode='reference', **kwargs):
        binding = seed.get('bc_physiology', {})
        if (mode != 'reference' or kwargs.get('torch') is not None
                or binding.get('parameter_binding_id') != getattr(assembly, 'parameter_binding_id', None)
                or canonical_hash(seed['inner_parameters']) != seed['joint_background_initialization']['parameters_sha256']):
            raise ContractError('BC runner requires its new source-bound CPU parameter identity')
        GroupTimeArm.__init__(self, assembly, seed, protocol, mode=mode, **kwargs)
        self.identity = canonical_hash(dict(parent=self.identity, bc_binding=binding))

    def checkpoint_payload(self, **kwargs):
        payload = GroupTimeArm.checkpoint_payload(self, **kwargs)
        payload['controller']['bc_parameter_binding'] = self.seed['bc_physiology']['parameter_binding_id']
        return payload

    @classmethod
    def from_checkpoint(cls, assembly, seed, protocol, payload, **kwargs):
        if payload.get('controller', {}).get('bc_parameter_binding') != seed.get('bc_physiology', {}).get('parameter_binding_id'):
            raise ContractError('Checkpoint BC parameters differ')
        return GroupTimeArm.from_checkpoint.__func__(cls, assembly, seed, protocol, payload, **kwargs)
