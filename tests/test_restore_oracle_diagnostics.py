import numpy as np

from retina_sim.restore_oracle_diagnostics import canonical_byte_comparison,capture_mismatch,first_difference,freeze
from retina_sim.group_time_storage import load_bundle
from tools.validate_group_time_candidate import checkpoint_hash


def test_first_difference_preserves_signed_zero_layout_and_semantic_paths():
    a={'aux':{'rng_state':{'state':5}},'states':np.array([0.,1.])}
    b=freeze(a);b['states'][0]=-0.
    diff=first_difference(a,b)
    assert diff['path']=='$.states' and diff['flat_index']==0 and diff['absolute_difference']==0.
    b=freeze(a);b['aux']['rng_state']['state']=6
    assert first_difference(a,b)['path']=='$.aux.rng_state.state'
    assert first_difference([1.,2.],(1.,2.)) is None
    assert first_difference(np.array([1.]),np.array([1.],dtype=np.float32))['kind']=='array_layout'


def test_canonical_proof_checks_every_byte_and_freeze_owns_inputs():
    x=np.resize(np.array([-0.,0.,1.e-300,1.e300]),20000)
    value={('tuple','key'):[x,x[::-1]],'metadata':['中文字',1,None,True]}
    frozen=freeze(value);proof=canonical_byte_comparison(frozen)
    assert proof['equal'] and proof['original_sha256']==proof['cached_sha256']==checkpoint_hash(value,reuse_arrays=False)
    assert proof['original_bytes']>100000
    x[0]=2.
    assert checkpoint_hash(frozen)!=checkpoint_hash(value)


def test_capture_saves_actual_row_raw_witness_and_numeric_first_difference(tmp_path):
    expected={'reference':dict(time_seconds=1.,side='right',event_generation=3,states=np.array([1.,2.]),aux={'history':[0,1]})}
    actual=freeze(expected);actual['reference']['states'][1]=np.nextafter(2.,3.)
    row=lambda evidence:dict(time_hex=float(1).hex(),side='right',hashes={mode:checkpoint_hash(v) for mode,v in evidence.items()})
    result=capture_mismatch(tmp_path/'failure',{'diagnostic':True},15,row(expected),row(actual),actual,expected_evidence=expected)
    detail=result['modes']['reference']
    assert detail['canonical_bytes']['equal'] and detail['rehash_matches_original_actual_row']
    assert detail['expected_witness_matches_oracle'] and detail['first_difference']['flat_index']==1
    saved=load_bundle(tmp_path/'failure/raw',expected_kind='validation_evidence')['payload']
    assert checkpoint_hash(saved['actual'])==checkpoint_hash(actual)
    assert saved['actual']['reference']['states'][1]!=saved['expected']['reference']['states'][1]


def test_full_actual_is_already_durable_when_optional_monitor_export_fails(tmp_path):
    import json,pytest
    class Broken:
        def payload(self):raise RuntimeError('optional monitor export failed')
    actual={'reference':dict(time_seconds=1.,side='right',event_generation=1,states=np.array([1.,2.]))}
    row=dict(time_hex=float(1).hex(),side='right',hashes={'reference':checkpoint_hash(actual['reference'])})
    with pytest.raises(RuntimeError,match='optional monitor export failed'):
        capture_mismatch(tmp_path/'failure',{'diagnostic':True},15,row,row,actual,monitors={'reference':Broken()})
    saved=load_bundle(tmp_path/'failure/actual_raw',expected_kind='validation_evidence')['payload']
    assert checkpoint_hash(saved['actual'])==checkpoint_hash(actual)
    assert saved['actual_row']==row and saved['expected_row']==row
    report=json.loads((tmp_path/'failure/summary.json').read_text(encoding='utf8'))
    assert report['actual_row']==row and report['actual_bundle']
