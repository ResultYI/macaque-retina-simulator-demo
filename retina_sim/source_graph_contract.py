"""Approved source graph binding; deliberately not a numerical QA certificate."""
from pathlib import Path
import hashlib
import json

from .contracts import ContractError, canonical_hash
from .resident_contract import ResidentNativeContract


def source_rf_mm(population):
    position = population.cells.rf_position
    if position is None:
        raise ContractError('Approved source requires explicit existing RF coordinates')
    position.require('mm', 'functional_position', (len(population.cells.cell_id), 2), 'functional_visual_field')
    return position.values


class SourceGraphContract(ResidentNativeContract):
    def __init__(self, reference, *, configuration_path, configuration_sha256):
        super().__init__()
        self.reference_contract_sha256 = self.sha256
        path = Path(configuration_path).resolve()
        root = Path(__file__).resolve().parents[1]
        if not path.is_relative_to(root):
            raise ContractError('Approved source configuration must be in this project')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != configuration_sha256:
            raise ContractError('Approved source target configuration changed')
        self.configuration = json.loads(raw)
        self.configuration_path = str(path)
        self.configuration_sha256 = configuration_sha256
        self.source_graph_binding = self._binding(reference)
        self.sha256 = canonical_hash(dict(scope='approved_useful_source_graph_v1',
            frozen_numerical_formula_contract=self.reference_contract_sha256,
            graph_binding=self.source_graph_binding, numerical_QA_pass_claim=False))

    def _binding(self, reference):
        import numpy as np
        scope = self.record['scenario_required']
        identity = reference.model.compute_metadata.get('integration_backend', {})
        native = reference.partition.native
        if (identity.get('backend') != scope['reference_backend'] or
                native.nc != scope['native_cones'] or native.nh != scope['native_HC']):
            raise ContractError('Source graph requires the complete original native closure/backend')
        rows = reference.compiled.global_disk_view['selected_rows']
        if not np.array_equal(rows['cones'], np.arange(native.nc)):
            raise ContractError('Approved optical/native ordering differs')
        expected = self.configuration['owner_global_rows']
        cohort = {}
        for cid, required in expected.items():
            if not np.array_equal(np.sort(rows[cid]), np.sort(required)):
                raise ContractError('Source target rows differ from the approved 8+4 configuration')
            pop = reference.compiled.populations[cid]
            cohort[cid] = dict(rows=np.asarray(rows[cid]), cell_ids=tuple(pop.cells.cell_id),
                               rf_mm=source_rf_mm(pop))
        outputs = sum(len(c.state_specs[0].cell_ids) for c in reference.compiled.components.values()
                      if hasattr(c.native_model, 'sample_spikes'))
        if outputs != 12 or sum(len(v) for v in expected.values()) != outputs:
            raise ContractError('Approved source requires exactly the original8 plus comparison4 outputs')
        model = reference.model
        return dict(configuration_path=self.configuration_path,
            configuration_sha256=self.configuration_sha256,
            compact_model_hash=model.model_hash, assembly_hash=reference.compiled.assembly_hash,
            graph_hash=model.graph.graph_hash, parameter_hash=model.parameters.parameter_hash,
            state_spec_hash=model._state_spec_hash, target_cohort_sha256=canonical_hash(cohort),
            components=len(reference.compiled.components), state_scalars=sum(s.shape[0] for s in model.state_specs),
            native_cones=native.nc, native_HC=native.nh, RGC_outputs=outputs,
            numerical_QA_pass_claim=False)

    def validate_reference(self, reference):
        if self._binding(reference) != self.source_graph_binding:
            raise ContractError('Actual approved source graph identity changed')

    @classmethod
    def from_binding(cls, reference, binding):
        obj = cls(reference, configuration_path=binding['configuration_path'],
                  configuration_sha256=binding['configuration_sha256'])
        if obj.source_graph_binding != binding:
            raise ContractError('Useful split graph differs from its prospective source binding')
        return obj
