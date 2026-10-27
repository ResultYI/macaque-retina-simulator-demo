"""Manufactured CPU lifecycle regression for the separately identified B arm."""
from copy import deepcopy
import numpy as np
from retina_sim.contracts import canonical_hash
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.group_time_storage import save_bundle,load_bundle
from tests.test_group_time_runner import toy


def accepted(arm,t0,t1):
    arm.ensure_native(t1,'left')
    result=arm.advance_inner(t0,t1)
    arm.snapshot(t1,'left',queue_node=True)
    arm.commit(t1,rod=True,rod_sample_time=max(0.,t1-.0001+1e-13))
    arm.snapshot(t1,'right',queue_node=True)
    arm.ack(t1)
    return result


def test_radau_complete_checkpoint_and_identical_continuation(tmp_path):
    assembly,seed,protocol=toy()
    arm=GroupTimeArm(assembly,seed,protocol,mode='radau')
    restored=None
    try:
        arm.initial_snapshot();arm.initial_commit()
        accepted(arm,0.,.0001)
        payload=deepcopy(arm.checkpoint_payload(monitor={'test':'complete'},recording={'test':'complete'}))
        oracle=canonical_hash(payload)
        save_bundle(tmp_path/'radau',kind='complete_model_checkpoint',identity={'arm':arm.identity},
            payload=payload,qualification={'default_adopted':False,'fit_eligible':False})
        saved=load_bundle(tmp_path/'radau',expected_kind='complete_model_checkpoint',expected_identity={'arm':arm.identity})
        restored=GroupTimeArm.from_checkpoint(assembly,seed,protocol,saved['payload'],mode='radau')
        assert canonical_hash(payload)==oracle
        assert canonical_hash(restored.checkpoint_payload(monitor={'test':'complete'},recording={'test':'complete'}))==oracle
        accepted(arm,.0001,.0002);accepted(restored,.0001,.0002)
        a=arm.snapshot(.0002,'right');b=restored.snapshot(.0002,'right')
        for field in ('states','rhs','transitions','ports'):
            assert set(a[field])==set(b[field])
            for key in a[field]:np.testing.assert_array_equal(a[field][key],b[field][key])
        assert canonical_hash(a['aux'])==canonical_hash(b['aux'])
        assert canonical_hash(a['draws'])==canonical_hash(b['draws'])
    finally:
        arm.close()
        if restored is not None:restored.close()


def test_quintic_selected_readout_packet_uses_actual_accepted_owner():
    assembly,seed,protocol=toy()
    arm=GroupTimeArm(assembly,seed,protocol,mode='radau')
    try:
        arm.initial_commit();interval=arm.ensure_native(.0001,'left')
        window=arm._window_readout(interval)
        t=.000037
        actual=window.at(t,'right')
        direct=interval.outputs_at(t,'right')['release'][arm.selection.cone_rows]
        np.testing.assert_allclose(actual,direct,rtol=3e-14,atol=1e-14)
        assert all(rep.kind=='quintic_hermite' for packet in window.packets for rep in packet._representations.values() if rep is not None)
    finally:
        arm.close()


def test_restore_rejects_readout_from_foreign_source():
    import pytest
    from retina_sim.contracts import ContractError
    assembly,seed,protocol=toy()
    arm=GroupTimeArm(assembly,seed,protocol,mode='radau')
    try:
        arm.initial_commit()
        interval=arm.ensure_native(.0001,'left')
        arm._window_readout(interval)
        accepted(arm,0.,.0001)
        payload=deepcopy(arm.checkpoint_payload())
        assert payload['readout_cursor']['windows']
        payload['readout_cursor']['windows'][0]['packets'][0]['metadata']['source_version']='foreign_numerical_backend'
        with pytest.raises(ContractError,match='accepted native owner'):
            GroupTimeArm.from_checkpoint(assembly,seed,protocol,payload,mode='radau')
    finally:
        arm.close()
