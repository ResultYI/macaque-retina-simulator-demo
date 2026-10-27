"""Explicit numerical-contract identity for the active ordered CUDA route.

Construction does not assert acceptance and does not register a GUI/default
backend. Original bitwise candidate identities and witnesses remain unchanged.
"""
from copy import copy
from pathlib import Path
import hashlib

from .contracts import canonical_hash
from .gpu_ordered_joint import OrderedJointCudaRunner
from .numerical_equivalence import NumericalEquivalence

BACKEND = 'cuda_ordered_joint_dop853_float64_numerical_contract_v1'


class NumericalOrderedCudaRunner(OrderedJointCudaRunner):
    def __init__(self, compiled, *, contract, compute_backend='numba', graph=True):
        from .gpu_hybrid import CudaHybridRunner
        reference=CudaHybridRunner(compiled,compute_backend=compute_backend)
        self.__dict__.update(type(self).from_runner(reference,contract=contract,graph=graph).__dict__)

    @classmethod
    def from_runner(cls, reference, *, contract, graph=True):
        if not isinstance(contract, NumericalEquivalence):
            raise TypeError('An explicit versioned numerical contract is required')
        self = super().from_runner(reference, graph=graph)
        self.model = copy(self.model)
        identity = dict(self.model.compute_metadata['integration_backend'], backend=BACKEND,
            numerical_contract=contract.identity, numerical_contract_hash=contract.contract_hash,
            contract_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            acceptance_status='REQUIRES_EXTERNAL_VALIDATION', default_enabled=False,
            numerical_equivalence_budget='Frozen independent contract '+contract.identity['frozen_contract_sha256'])
        self.model.model_hash = canonical_hash(dict(ordered_execution=self.model.model_hash, numerical_contract=identity))
        self.model.compute_metadata = dict(self.model.compute_metadata, integration_backend=identity)
        self.numerical_contract = contract
        return self

    def initialize(self, solver_config=None, time_seconds=0):
        state = super().initialize(solver_config, time_seconds)
        state.input_position['numerical_contract_hash'] = self.numerical_contract.contract_hash
        return state
