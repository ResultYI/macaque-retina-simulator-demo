"""Import preserves old evidence, scientific generation, and new job ownership."""
import ast
from copy import deepcopy
from pathlib import Path

import pytest

from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.group_scc_protocol import read_json,sha256,write_json
from retina_sim.group_time_storage import COMPLETE_FIELDS,save_bundle
from retina_sim.restore_import import (DRIVER,SOURCE_FILES,_generation_parts,bind_identity,
    copy_recording_prefix,inspect_source,verify_source_pins)

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'reports/overnight_20261026_v1/workbench/r5_import_source_driver.py'


def case(tmp_path):
    parent='reports/stage';source=tmp_path/parent/'old'
    source.mkdir(parents=True)
    p=dict(start_seconds=4.993,stop_seconds=5.001,checkpoint_seconds=4.997)
    old=dict(schema='test',candidate='gpu_dop853',execution_class='A',protocol=canonical_hash(p),
        group_manifest='g',seed_manifest='s',historical_pins={},implementation={DRIVER:sha256(FROZEN),'physics.py':'same'})
    base=deepcopy(old);base['implementation'][DRIVER]=sha256(ROOT/DRIVER)
    base['implementation']['retina_sim/restore_import.py']='new'
    job=dict(identity=old,candidate='gpu_dop853',phase='paired',token='original')
    session=dict(identity=old,candidate='gpu_dop853',public_time_seconds=4.997,hot_started_unix=10.,writer_pid=1)
    first=dict(status='PAIRED_UNINTERRUPTED_COMPLETE',scope_seconds=[4.993,5.001],
        observation_fields=45,ordinary_recording_samples=17,
        dense_certificates=[dict(status='PASS',complete_coverage=True)]*80,
        **dict.fromkeys(('numerical_status','domain_status','event_status','monitor_status','dense_status'),'PASS'))
    for name in SOURCE_FILES:
        (source/name).parent.mkdir(parents=True,exist_ok=True)
        write_json(source/name,{})
    write_json(source/'job.json',job);write_json(source/'protocol.json',p)
    write_json(source/'session_checkpoint.json',dict(session,session_hash=canonical_hash(session)))
    write_json(source/'uninterrupted_result.json',first)
    write_json(source/'uninterrupted_oracle.json',dict(callbacks=[{}]*138))
    frozen=tmp_path/'reports/overnight_20261026_v1/workbench/r5_import_source_driver.py'
    frozen.parent.mkdir(parents=True);frozen.write_bytes(FROZEN.read_bytes())
    target=tmp_path/DRIVER;target.parent.mkdir(parents=True);target.write_bytes((ROOT/DRIVER).read_bytes())
    return source,p,base,parent


def test_import_binds_both_identities_and_detects_source_mutation(tmp_path):
    source,p,base,parent=case(tmp_path)
    receipt=inspect_source(tmp_path,source,p,base,parent)
    bound=bind_identity(base,receipt)
    assert bound['restore_import']==canonical_hash(receipt)
    assert receipt['source_token']=='original' and receipt['original_failure_preserved']
    assert base['historical_pins']=={} and bound['historical_pins']==receipt['input_pins']
    verify_source_pins(tmp_path,receipt)
    write_json(source/'REQUEST_STOP',dict(changed=True))
    with pytest.raises(ContractError,match='Imported restore input changed'):
        verify_source_pins(tmp_path,receipt)


@pytest.mark.parametrize('change',['science','added','session','partial','oracle','protocol'])
def test_import_rejects_changed_science_or_incomplete_source(tmp_path,change):
    source,p,base,parent=case(tmp_path)
    if change=='science':base['implementation']['physics.py']='changed'
    if change=='added':base['implementation']['unexpected.py']='new'
    if change=='session':write_json(source/'session_checkpoint.json',{})
    if change=='partial':
        first=read_json(source/'uninterrupted_result.json');first['dense_certificates'].pop()
        write_json(source/'uninterrupted_result.json',first)
    if change=='oracle':write_json(source/'uninterrupted_oracle.json',dict(callbacks=[{}]*137))
    if change=='protocol':p=dict(p,diagnostic_scope_only=True)
    with pytest.raises((ContractError,KeyError)):
        inspect_source(tmp_path,source,p,base,parent)


def test_successful_observation_and_advance_ast_unchanged_but_math_change_rejected():
    old=FROZEN.read_text(encoding='utf8');current=(ROOT/DRIVER).read_text(encoding='utf8')
    assert _generation_parts(old)==_generation_parts(current)
    changed=current.replace('begin = arms[\'reference\'].inner_time','begin = 0.')
    assert _generation_parts(old)!=_generation_parts(changed)


def test_recording_import_copies_only_committed_hash_bound_files(tmp_path):
    source=tmp_path/'old';source.mkdir();out=tmp_path/'new';out.mkdir()
    for mode in ('reference','gpu_dop853'):
        folder=source/'restored'/mode;folder.mkdir(parents=True)
        (folder/'committed.bin').write_bytes(b'prefix')
        (folder/'later.bin').write_bytes(b'not committed')
        save_bundle(source/(mode+'_checkpoint'),kind='complete_model_checkpoint',identity={},
            payload=dict(dict.fromkeys(COMPLETE_FIELDS),recording={'files':{'committed.bin':sha256(folder/'committed.bin')}}),
            qualification=dict(default_adopted=False,fit_eligible=False))
    copy_recording_prefix(tmp_path,dict(source=str(source),input_pins={}),out)
    for mode in ('reference','gpu_dop853'):
        assert [p.name for p in (out/'restored'/mode).iterdir()]==['committed.bin']
        assert (out/'restored'/mode/'committed.bin').read_bytes()==b'prefix'


def test_mismatch_is_preserved_even_when_both_capture_and_fallback_write_fail(monkeypatch,tmp_path):
    import retina_sim.restore_oracle_diagnostics as diagnostics
    from retina_sim.group_scc_protocol import AcceptanceFailure
    def broken(*args,**kwargs):raise OSError('disk failed')
    monkeypatch.setattr(diagnostics,'capture_mismatch',broken)
    tree=ast.parse((ROOT/DRIVER).read_text(encoding='utf8'))
    observe=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='observe')
    branch=next(n for n in ast.walk(observe) if isinstance(n,ast.If) and isinstance(n.test,ast.BoolOp)
        and ast.dump(n.test).startswith("BoolOp(op=Or(), values=[Compare(left=Name(id='oracle_index'"))
    code=compile(ast.fix_missing_locations(ast.Module(body=branch.body,type_ignores=[])),'failure_branch','exec')
    scope=dict(oracle_index=0,oracle=[{}],row={'actual':1},restore_import=None,output=tmp_path,
        identity={},evidence={},monitors={},arms={},timed=lambda m,n,fn:fn(),write_json=broken,
        AcceptanceFailure=AcceptanceFailure)
    with pytest.raises(AcceptanceFailure,match='RESTORE_INCONCLUSIVE.*callback 0'):
        exec(code,scope)
