"""Whole-group exclusion must protect unilluminated central members."""
import numpy as np
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.run_bc_center_isolation import isolate_mask, result_identity, save_states


def test_protection_removes_entire_group_without_renormalizing_survivors():
    xy=np.array([[0.,0.],[2.,0.],[3.,0.],[4.,0.],[1.,0.],[5.,0.]])
    groups=np.array([0,0,1,1,2,2]); counts=np.array([2,2,2])
    candidate=np.array([False,True,True,False,False,True])
    protected, protected_groups, mask, fraction=isolate_mask(xy,np.zeros(2),1.,groups,counts,candidate)
    np.testing.assert_array_equal(protected_groups,[0,2])
    np.testing.assert_array_equal(mask,[False,False,True,False,False,False])
    np.testing.assert_array_equal(fraction,[0.,.5,0.])
    assert np.all(fraction[groups[protected]]==0.)


def test_no_surviving_groups_is_not_amplified_or_replaced():
    xy=np.array([[0.,0.],[2.,0.]])
    _,_,mask,fraction=isolate_mask(xy,np.zeros(2),1.,np.array([0,0]),np.array([2]),np.array([False,True]))
    assert not mask.any() and not fraction.any()


def test_byte_cell_ids_roundtrip_and_identity_without_json_conversion():
    arrays={'cone_global_ids':np.array([b'cone:1',b'cone:2']), 'V':np.array([-.05,-.049])}
    stream=BytesIO()
    save_states(stream,arrays,{'status':'INTEGRATION_COMPLETED_ANALYSIS_PENDING'})
    stream.seek(0)
    with np.load(stream,allow_pickle=False) as z:
        for key,value in arrays.items():
            np.testing.assert_array_equal(z[key],value)
    identity=result_identity(arrays,'model',{'stop':.26})
    assert identity==result_identity(dict(reversed(list(arrays.items()))),'model',{'stop':.26})
    changed=dict(arrays,cone_global_ids=np.array([b'cone:1',b'cone:3']))
    assert identity!=result_identity(changed,'model',{'stop':.26})


def test_disk_snapshot_survives_postsave_serialization_failure():
    arrays={'cone_global_ids':np.array([b'cone:1']), 'time_seconds':np.array([0.,.001]),
            'V':np.array([-.05,-.049])}
    with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tmp') as directory:
        path=Path(directory)/'snapshot.npz'
        save_states(path,arrays,{'status':'INTEGRATION_COMPLETED_ANALYSIS_PENDING'})
        original_bytes=path.read_bytes()
        try:
            save_states(path,arrays,{'invalid_metadata':b'bytes'})
        except TypeError:
            pass
        else:
            raise AssertionError('Expected deliberately invalid metadata to fail')
        assert path.read_bytes()==original_bytes
        assert not path.with_suffix('.npz.writing').exists()
        with np.load(path,allow_pickle=False) as z:
            for key,value in arrays.items():
                np.testing.assert_array_equal(z[key],value)
