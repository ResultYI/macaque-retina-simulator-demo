"""Explicit read-only import of a completed main trajectory for a new restore job."""
import ast
from copy import deepcopy
from pathlib import Path
import shutil

from .contracts import ContractError,canonical_hash
from .group_scc_protocol import read_json,sha256
from .group_time_storage import load_bundle


ADDED_DIAGNOSTICS={'retina_sim/restore_oracle_diagnostics.py','retina_sim/restore_import.py'}
DRIVER='tools/validate_group_scc_radau.py'
SOURCE_FILES=('job.json','protocol.json','session_checkpoint.json','uninterrupted_result.json',
              'uninterrupted_oracle.json','environment.json','paired_entry_gate.json',
              'reference_checkpoint/manifest.json','gpu_dop853_checkpoint/manifest.json','initial_ports/manifest.json',
              'failure.json','REQUEST_STOP')


def _generation_parts(text):
    worker=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='run_worker')
    observe=deepcopy(next(n for n in ast.walk(worker) if isinstance(n,ast.FunctionDef) and n.name=='observe'))
    for node in ast.walk(observe):
        if isinstance(node,ast.If) and ast.dump(node.test).startswith('BoolOp(op=Or(), values=[Compare(left=Name(id=\'oracle_index\''):
            # Failure-only diagnostics precede the unchanged unconditional raise.
            node.body=[n for n in node.body if isinstance(n,ast.Raise)]
    events=next(n for n in worker.body if isinstance(n,ast.Try)).body
    loop=next(n for n in events if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='event_index')
    return ast.dump(observe,include_attributes=False),ast.dump(loop,include_attributes=False)


def inspect_source(root,source,protocol,current_identity,report_parent):
    root=Path(root).resolve();source=Path(source).resolve()
    if not source.is_dir() or source.parent!=(root/report_parent).resolve():
        raise ContractError('Restore source must be an existing immediate child of this accounting stage')
    job=read_json(source/'job.json',permission_retry_seconds=.25)
    old=job['identity'];session=read_json(source/'session_checkpoint.json',permission_retry_seconds=.25)
    first=read_json(source/'uninterrupted_result.json',permission_retry_seconds=.25)
    oracle=read_json(source/'uninterrupted_oracle.json',permission_retry_seconds=.25)
    if (job.get('candidate')!='gpu_dop853' or job.get('phase')!='paired'
            or protocol.get('diagnostic_scope_only') or read_json(source/'protocol.json')!=protocol
            or first.get('status')!='PAIRED_UNINTERRUPTED_COMPLETE'
            or any(first.get(name)!='PASS' for name in ('numerical_status','domain_status','event_status','monitor_status','dense_status'))
            or first.get('scope_seconds')!=[4.993,5.001] or first.get('observation_fields')!=45
            or first.get('ordinary_recording_samples')!=17 or len(first.get('dense_certificates',[]))!=80
            or any(row.get('status')!='PASS' or row.get('complete_coverage') is not True for row in first['dense_certificates'])
            or len(oracle['callbacks'])!=138):
        raise ContractError('Restore import requires the unchanged complete8ms main trajectory and every original output/gate')
    witness=session.pop('session_hash',None)
    if (witness!=canonical_hash(session) or session['identity']!=old or session.get('candidate')!='gpu_dop853'
            or session['public_time_seconds']!=protocol['checkpoint_seconds']):
        raise ContractError('Original checkpoint session integrity differs')
    for key in ('schema','candidate','execution_class','protocol','group_manifest','seed_manifest','historical_pins'):
        if old.get(key)!=current_identity.get(key):raise ContractError('Original scientific/protocol lineage changed: '+key)
    changed=[name for name,value in old['implementation'].items() if current_identity['implementation'].get(name)!=value]
    if changed!=[DRIVER] and changed!=[]:raise ContractError('Scientific/generation implementation changed: '+str(changed))
    if not set(current_identity['implementation']).difference(old['implementation']).issubset(ADDED_DIAGNOSTICS):
        raise ContractError('Unreviewed implementation added to restore import')
    # Bind the exact prior driver, then independently prove its successful-path
    # observation and advance loop match; only the failure capture is ignored.
    frozen=root/'reports/overnight_20261026_v1/workbench/r5_import_source_driver.py'
    if not frozen.is_file() or sha256(frozen)!=old['implementation'][DRIVER]:
        raise ContractError('Original frozen driver bytes required for restore import')
    if _generation_parts(frozen.read_text(encoding='utf8'))!=_generation_parts((root/DRIVER).read_text(encoding='utf8')):
        raise ContractError('Observation/advance generation changed; a new main trajectory is required')
    files={str((source/name).relative_to(root)).replace('\\','/'):sha256(source/name) for name in SOURCE_FILES}
    return dict(source=str(source),source_relative=source.relative_to(root).as_posix(),source_identity=old,
        source_token=job['token'],source_hot_started_unix=session['hot_started_unix'],source_writer_pid=session['writer_pid'],
        entry_gate=read_json(source/'paired_entry_gate.json'),input_pins=files,
        scientific_generation_unchanged=True,original_failure_preserved=(source/'failure.json').exists(),
        purpose='New owned restore-only attempt; does not resume or reclassify original failed job')


def bind_identity(base,receipt):
    return dict(base,restore_import=canonical_hash(receipt),
                historical_pins=dict(base['historical_pins'],**receipt['input_pins']))


def verify_source_pins(root,receipt):
    for relative,expected in receipt['input_pins'].items():
        if sha256(Path(root)/relative)!=expected:raise ContractError('Imported restore input changed: '+relative)


def copy_recording_prefix(root,receipt,output):
    source=Path(receipt['source']);output=Path(output)
    for mode in ('reference','gpu_dop853'):
        payload=load_bundle(source/(mode+'_checkpoint'),expected_kind='complete_model_checkpoint')['payload']
        destination=output/'restored'/mode;destination.mkdir(parents=True,exist_ok=False)
        for relative,expected in payload['recording']['files'].items():
            target=(destination/relative).resolve()
            if not target.is_relative_to(destination.resolve()):raise ContractError('Unsafe recording prefix path')
            candidates=[(source/folder/mode/relative).resolve() for folder in ('restored','uninterrupted')]
            original=next((p for p in candidates if p.is_relative_to(source) and p.is_file() and sha256(p)==expected),None)
            if original is None:raise ContractError('No intact original recording prefix file: '+relative)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,target)
        del payload
    verify_source_pins(root,receipt)
