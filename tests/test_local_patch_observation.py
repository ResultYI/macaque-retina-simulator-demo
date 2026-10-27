"""Saved observation identity/query tests only; no model evaluation."""
from types import SimpleNamespace

import numpy as np
import pytest

from retina_sim.contracts import ContractError
from retina_sim.group_time_storage import save_bundle
from retina_sim.local_patch_observation import SavedPatchObservation


@pytest.fixture
def saved(tmp_path):
    # Original rows1/2 share a computed representative group; row4 is absent.
    cone=SimpleNamespace(fine_to_group=np.array([0,1,1,2,-1]),counts=np.array([1,2,1]),
        group_ids=np.array(['cone:10','cone:20','cone:30']),group_count=3,
        recording_auxiliary_fine_rows=np.array([0,1]),recording_auxiliary_group_rows=np.array([0,1]))
    state=SimpleNamespace(population='cones',mechanism='native',name='Vis',count=3,unit='vH_native_voltage')
    auxiliary=SimpleNamespace(population='cones',mechanism='selected_original_z1_auxiliary',name='z1_aux',count=2,unit='vH_native_voltage')
    assembly=SimpleNamespace(artifact_sha256='patch-identity',populations={'cones':cone},
        parent_population_rows={'cones':np.array([10,20,30])},state_blocks=(state,),
        probability_blocks=(),auxiliary_z1_blocks=(auxiliary,))
    keys=[('cones','native','Vis'),('cones','selected_original_z1_auxiliary','z1_aux')]
    snapshot=dict(time_seconds=4.9941,side='right',event_generation=13,
        states={keys[0]:np.array([1.,2.,3.]),keys[1]:np.array([7.,8.])},
        rhs={keys[0]:np.array([.1,.2,.3]),keys[1]:np.array([.7,.8])},
        ports={('cones','release'):np.array([.4,.5,.6]),('cones','z1_aux'):np.array([7.,8.])},
        port_units={('cones','release'):'dimensionless',('cones','z1_aux'):'vH_native_voltage'})
    path=tmp_path/'snapshot'
    save_bundle(path,kind='validation_evidence',identity=dict(patch_artifact_sha256=assembly.artifact_sha256),
        payload=snapshot,qualification=dict(default_adopted=False,fit_eligible=False))
    return assembly,path


def test_single_member_cell_reads_saved_state_rhs_unit_and_original_identity(saved):
    observation=SavedPatchObservation(*saved)
    result=observation.read_state('cones',0,'native','Vis')
    assert result['value']==1. and result['rhs']==.1 and result['unit']=='vH_native_voltage'
    assert result['identity']['parent_group_row']==10 and result['identity']['group_id']=='cone:10'
    assert result['identity']['member_count']==1 and result['identity']['independent_cell_state_computed']
    assert result['time_seconds']==4.9941 and result['query_source'].startswith('actual_saved_snapshot')
    assert observation.read_port('cones',0,'release')['value']==.4


def test_multi_member_source_row_cannot_be_silently_reported_as_independent_cell(saved):
    observation=SavedPatchObservation(*saved)
    with pytest.raises(ContractError,match='multi-member representative group'):
        observation.read_state('cones',2,'native','Vis')
    result=observation.read_state('cones',2,'native','Vis',allow_representative_group=True)
    assert result['value']==2. and result['identity']['member_count']==2
    assert result['identity']['representation']=='representative_group_prediction'
    assert result['identity']['independent_cell_state_computed'] is False
    assert result['identity']['requested_original_source_row']==2
    with pytest.raises(ContractError,match='multi-member representative group'):
        observation.read_port('cones',1,'release')


@pytest.mark.parametrize('row',[True,-1,5,4,1.])
def test_invalid_or_uncomputed_original_rows_are_rejected(saved,row):
    observation=SavedPatchObservation(*saved)
    with pytest.raises(ContractError):observation.read_state('cones',row,'native','Vis')


def test_missing_state_or_population_is_not_reconstructed(saved):
    observation=SavedPatchObservation(*saved)
    with pytest.raises(ContractError,match='not saved'):
        observation.read_state('cones',0,'native','missing')
    with pytest.raises(ContractError,match='Unknown saved population'):
        observation.read_state('unknown',0,'native','Vis')
    with pytest.raises(ContractError,match='not saved'):
        observation.read_port('cones',0,'missing')


def test_z1_requires_the_exact_recorded_original_auxiliary_row(saved):
    observation=SavedPatchObservation(*saved)
    result=observation.read_state('cones',0,'selected_original_z1_auxiliary','z1_aux')
    assert result['value']==7. and result['identity']['local_auxiliary_row']==0
    assert observation.read_port('cones',0,'z1_aux')['value']==7.
    # Row2 shares row1's computed group, but has no independent saved auxiliary.
    for query in (
        lambda:observation.read_state('cones',2,'selected_original_z1_auxiliary','z1_aux',allow_representative_group=True),
        lambda:observation.read_port('cones',2,'z1_aux',allow_representative_group=True),
        lambda:observation.read_port('cones',3,'z1_aux')):
        with pytest.raises(ContractError,match='exact original source row'):
            query()


def test_snapshot_from_another_patch_is_rejected(saved):
    assembly,path=saved
    assembly.artifact_sha256='different-patch'
    with pytest.raises(ContractError,match='different patch graph'):
        SavedPatchObservation(assembly,path)
