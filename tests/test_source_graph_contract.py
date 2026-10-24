"""Graph-binding tests only: no assembly, solver, CUDA or JIT."""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import json
import unittest
import numpy as np

from retina_sim.contracts import ContractError, Quantity
from retina_sim.resident_contract import ResidentNativeContract
from retina_sim.source_graph_contract import SourceGraphContract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'tools/vr_teacher_handoff_v1/fit_integration_v1/capture_preparation_v2/capture_configuration.json'


class SourceGraphTests(unittest.TestCase):
    def setUp(self):
        scope = ResidentNativeContract().record['scenario_required']
        cfg = json.loads(CONFIG.read_bytes())
        rows = {k: np.array(v) for k, v in cfg['owner_global_rows'].items()}
        rows['cones'] = np.arange(scope['native_cones'])
        populations = {k: NS(cells=NS(cell_id=tuple('id'+str(x) for x in v),
                            rf_position=Quantity(np.zeros((len(v), 2)), 'mm',
                            'functional_position', 'CPU fixture', 'functional_visual_field')))
                       for k, v in cfg['owner_global_rows'].items()}
        components = {k: NS(native_model=NS(sample_spikes=True),
                            state_specs=[NS(cell_ids=pop.cells.cell_id)])
                      for k, pop in populations.items()}
        self.reference = NS(partition=NS(native=NS(nc=scope['native_cones'], nh=scope['native_HC'])),
            compiled=NS(global_disk_view={'selected_rows': rows}, populations=populations,
                        components=components, assembly_hash='assembly'),
            model=NS(compute_metadata={'integration_backend': {'backend': scope['reference_backend']}},
                     model_hash='model', graph=NS(graph_hash='graph'),
                     parameters=NS(parameter_hash='parameters'), _state_spec_hash='state',
                     state_specs=[NS(shape=(100,))]))
        self.kw = dict(configuration_path=CONFIG,
                       configuration_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest())

    def test_new_source_binding_does_not_pass_old_scenario(self):
        contract = SourceGraphContract(self.reference, **self.kw)
        contract.validate_reference(self.reference)
        self.assertFalse(contract.source_graph_binding['numerical_QA_pass_claim'])
        with self.assertRaises(ContractError):
            ResidentNativeContract().validate_reference(self.reference)

    def test_changed_graph_or_target_is_rejected(self):
        contract = SourceGraphContract(self.reference, **self.kw)
        self.reference.model.graph.graph_hash = 'changed'
        with self.assertRaises(ContractError):
            contract.validate_reference(self.reference)
        self.reference.compiled.global_disk_view['selected_rows']['midget_RGC_ON'][0] += 1
        with self.assertRaises(ContractError):
            SourceGraphContract(self.reference, **self.kw)

    def test_binding_and_config_digest_cannot_be_relabelled(self):
        contract = SourceGraphContract(self.reference, **self.kw)
        binding = dict(contract.source_graph_binding, state_spec_hash='changed')
        with self.assertRaises(ContractError):
            SourceGraphContract.from_binding(self.reference, binding)
        with self.assertRaises(ContractError):
            SourceGraphContract(self.reference, **dict(self.kw, configuration_sha256='0'*64))
