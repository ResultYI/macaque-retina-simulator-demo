"""Read actual saved patch states; never reconstruct or integrate a missing cell."""
from pathlib import Path
import math

import numpy as np

from .contracts import ContractError
from .group_time_storage import load_bundle
from .spatial_group_model import CONTACT_STATES


class SavedPatchObservation:
    """A source-row query over one identity-bound complete saved snapshot."""
    def __init__(self, assembly, snapshot_directory):
        self.assembly = assembly
        self.directory = Path(snapshot_directory).resolve()
        bundle = load_bundle(self.directory, expected_kind='validation_evidence')
        if bundle['identity'].get('patch_artifact_sha256') != assembly.artifact_sha256:
            raise ContractError('Saved observation belongs to a different patch graph')
        self.snapshot = bundle['payload']
        self.bundle_identity = bundle['identity']
        self.bundle_manifest_hash = bundle['manifest_hash']
        required = {'time_seconds', 'side', 'event_generation', 'states', 'rhs', 'ports', 'port_units'}
        if (not isinstance(self.snapshot, dict) or not required <= set(self.snapshot)
                or not math.isfinite(self.snapshot['time_seconds'])
                or self.snapshot['side'] not in ('left', 'right')):
            raise ContractError('Complete saved snapshot with exact time/side required')
        self.specs = {(block.population, block.mechanism, block.name): block
            for block in (*assembly.state_blocks, *assembly.probability_blocks, *assembly.auxiliary_z1_blocks)}

    def _identity(self, population, original_row, allow_representative_group):
        if population not in self.assembly.populations:
            raise ContractError('Unknown saved population: ' + str(population))
        if (isinstance(original_row, (bool, np.bool_)) or not isinstance(original_row, (int, np.integer))
                or type(allow_representative_group) is not bool):
            raise ContractError('Integer original source row and explicit Boolean representation flag required')
        layout = self.assembly.populations[population]
        original_row = int(original_row)
        if not 0 <= original_row < len(layout.fine_to_group):
            raise ContractError('Original source row lies outside the population asset')
        local_row = int(layout.fine_to_group[original_row])
        if local_row < 0:
            raise ContractError('Original source row was not computed in this patch')
        count = int(layout.counts[local_row])
        if count > 1 and not allow_representative_group:
            raise ContractError('This source row belongs to a multi-member representative group; an independent cell state was not computed')
        return dict(population=population, requested_original_source_row=original_row,
            local_group_row=local_row,
            parent_group_row=int(self.assembly.parent_population_rows[population][local_row]),
            group_id=str(layout.group_ids[local_row]), member_count=count,
            representation='single_member_source_model_cell' if count == 1 else 'representative_group_prediction',
            independent_cell_state_computed=count == 1,
            biological_recording_registration_claimed=False)

    def _auxiliary_row(self, population, original_row):
        if population != 'cones':
            raise ContractError('Selected original z1 auxiliary belongs to cones')
        rows = np.asarray(self.assembly.populations['cones'].recording_auxiliary_fine_rows)
        matches = np.flatnonzero(rows == original_row)
        if len(matches) != 1:
            raise ContractError('This exact original source row has no saved z1 auxiliary; no group-based reconstruction is allowed')
        return int(matches[0])

    @staticmethod
    def _scalar(value, index, expected_count, role):
        value = np.asarray(value)
        if value.dtype != np.float64 or value.shape != (expected_count,) or not np.isfinite(value[index]):
            raise ContractError('Saved FP64 observation layout/domain differs: ' + role)
        return float(value[index])

    def _stamp(self):
        return dict(time_seconds=float(self.snapshot['time_seconds']), side=self.snapshot['side'],
            event_generation=int(self.snapshot['event_generation']),
            snapshot_manifest_hash=self.bundle_manifest_hash,
            patch_artifact_sha256=self.assembly.artifact_sha256,
            query_source='actual_saved_snapshot; no model evaluation')

    def read_state(self, population, original_row, mechanism, state, *, allow_representative_group=False):
        identity = self._identity(population, original_row, allow_representative_group)
        key = population, mechanism, state
        if mechanism in CONTACT_STATES:
            raise ContractError('A physical contact state requires a contact identity; a cell row alone is ambiguous')
        if key not in self.specs or key not in self.snapshot['states'] or key not in self.snapshot['rhs']:
            raise ContractError('Requested state/RHS was not saved with this patch state registry')
        block = self.specs[key]
        if mechanism == 'selected_original_z1_auxiliary':
            index = self._auxiliary_row(population, int(original_row))
            identity.update(local_auxiliary_row=index,
                auxiliary_identity='exact original selected auxiliary node; drive retains its source-group precision')
        else:
            index = identity['local_group_row']
        result = dict(self._stamp(), identity=identity, state_key=key, unit=block.unit,
            value=self._scalar(self.snapshot['states'][key], index, block.count, 'state'),
            rhs=self._scalar(self.snapshot['rhs'][key], index, block.count, 'original-equation RHS'),
            rhs_time_unit='s')
        transitions = self.snapshot.get('transitions', {})
        result['pure_transition'] = (self._scalar(transitions[key], index, block.count, 'pure discrete transition')
            if key in transitions else None)
        return result

    def read_port(self, population, original_row, name, *, allow_representative_group=False):
        identity = self._identity(population, original_row, allow_representative_group)
        key = population, name
        if key not in self.snapshot['ports'] or key not in self.snapshot['port_units']:
            raise ContractError('Requested effective port was not saved')
        if key == ('cones', 'z1_aux'):
            index = self._auxiliary_row(population, int(original_row))
            count = len(self.assembly.populations['cones'].recording_auxiliary_fine_rows)
            identity.update(local_auxiliary_row=index,
                auxiliary_identity='exact original selected auxiliary node; drive retains its source-group precision')
        else:
            index = identity['local_group_row']
            count = self.assembly.populations[population].group_count
        return dict(self._stamp(), identity=identity, port_key=key,
            unit=self.snapshot['port_units'][key],
            value=self._scalar(self.snapshot['ports'][key], index, count, 'effective port'))
