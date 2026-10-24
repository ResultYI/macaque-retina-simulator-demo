"""Atomic source chunks and a bounded *new-session-only* checkpoint journal.

This module is CPU/file I/O only. It never assembles or advances a model. Cache
files and numerical evidence are immutable; only owned old recovery snapshots
may be retired after two newer complete commits exist.
"""
from pathlib import Path
import json
import math
import os
import shutil
import uuid

from .contracts import ContractError, canonical_hash
from .cone_hc_split_checkpoint import file_sha


def write_json(path, value, *, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise ContractError('Refusing to replace an existing research artifact: '+str(path))
    tmp = path.with_name(path.name+'.writing-'+uuid.uuid4().hex)
    with tmp.open('xb') as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode('utf8'))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_receipt(path):
    value = json.loads(Path(path).read_bytes())
    digest = value.pop('receipt_sha256')
    if canonical_hash(value) != digest:
        raise ContractError('Committed receipt changed: '+str(path))
    return dict(value, receipt_sha256=digest)


def receipt(value):
    return dict(value, receipt_sha256=canonical_hash(value))


def verify_checkpoint_files(directory, expected_hash):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_bytes())
    digest = manifest.pop('manifest_hash')
    if canonical_hash(manifest) != digest or digest != expected_hash:
        raise ContractError('Retained complete checkpoint manifest changed')
    for name, item in manifest.get('files', {}).items():
        path = (directory/name).resolve()
        if (Path(name).name != name or path.parent != directory or not path.is_file()
                or path.stat().st_size != item['bytes'] or file_sha(path) != item['sha256']):
            raise ContractError('Retained checkpoint content changed; refusing retirement')
    return manifest


class SplitChunkStore:
    SCHEMA = 'qualified-split-source-chunks-1'

    def __init__(self, directory, *, expected_session_hash):
        self.root = Path(directory).resolve()
        self.session = read_receipt(self.root/'session.json')
        self.session_hash = self.session['receipt_sha256']
        if (self.session.get('schema') != self.SCHEMA or self.session_hash != expected_session_hash
                or not self.session.get('owner_uuid')):
            raise ContractError('Explicit source session identity differs')

    @classmethod
    def create(cls, directory, binding):
        root = Path(directory).resolve()
        workspace = Path(__file__).resolve().parents[1]
        if root.exists() or not root.parent.is_dir() or root == workspace or not root.is_relative_to(workspace):
            raise ContractError('New source session must be absent inside the project with an existing parent')
        root.mkdir()
        value = receipt(dict(schema=cls.SCHEMA, owner_uuid=uuid.uuid4().hex, binding=binding))
        write_json(root/'session.json', value)
        return cls(root, expected_session_hash=value['receipt_sha256'])

    def latest(self, *, verify_files=True):
        """Rebuild head from contiguous commits, ignoring visible .writing dirs."""
        directories = sorted(p for p in self.root.glob('chunk-*') if p.is_dir() and '.writing-' not in p.name)
        previous = None
        for index, path in enumerate(directories):
            if path.name != f'chunk-{index:06d}' or path.resolve().parent != self.root:
                raise ContractError('Committed source chunk sequence has a gap or unsafe path')
            row = read_receipt(path/'commit.json')
            if (row['session_sha256'] != self.session_hash or row['index'] != index
                    or row['previous_commit_sha256'] != (None if previous is None else previous['receipt_sha256'])
                    or not math.isfinite(row['start_seconds']) or not math.isfinite(row['stop_seconds'])
                    or row['stop_seconds'] <= row['start_seconds']
                    or (previous is not None and row['start_seconds'] != previous['stop_seconds'])
                    or (previous is not None and row['first_boundary_sha256'] != previous['last_boundary_sha256'])):
                raise ContractError('Chunk time/endpoint/identity chain differs')
            if verify_files:
                for name, info in row['files'].items():
                    item = (path/name).resolve()
                    if not item.is_relative_to(path.resolve()) or not item.is_file() or file_sha(item) != info['sha256']:
                        raise ContractError('Committed source output changed: '+name)
            previous = row
        return previous

    def stage(self):
        last = self.latest(verify_files=False)
        index = 0 if last is None else last['index']+1
        stage = self.root/f'chunk-{index:06d}.writing-{uuid.uuid4().hex}'
        stage.mkdir()
        return stage, last

    def commit(self, stage, *, start, stop, first_boundary_hash, last_boundary_hash,
               checkpoint_receipt, cache_identity, cache_content):
        stage = Path(stage).resolve()
        if stage.parent != self.root or '.writing-' not in stage.name:
            raise ContractError('Unowned source staging directory')
        last = self.latest(verify_files=False)
        index = 0 if last is None else last['index']+1
        if (not stage.name.startswith(f'chunk-{index:06d}.writing-') or not math.isfinite(start)
                or not math.isfinite(stop) or stop <= start or checkpoint_receipt['time_seconds'] != stop
                or (last is not None and (start != last['stop_seconds'] or first_boundary_hash != last['last_boundary_sha256']))):
            raise ContractError('Source commit does not continue its exact preceding endpoint')
        # Binary arrays are separately guarded by the complete checkpoint codec.
        manifest = json.loads((stage/'checkpoint/manifest.json').read_bytes())
        digest = manifest.pop('manifest_hash')
        if canonical_hash(manifest) != digest or digest != checkpoint_receipt['manifest_hash']:
            raise ContractError('Source chunk checkpoint is not the supplied complete receipt')
        b = self.session['binding']
        if (manifest.get('actual_complete_checkpoint') is not True or manifest.get('time_seconds') != stop
                or (b.get('model_hash') is not None and manifest.get('model_hash') != b['model_hash'])
                or (b.get('science_sha256') is not None and manifest.get('scientific_contract_sha256') != b['science_sha256'])
                or (b.get('source_input_binding') is not None and
                    manifest.get('input_binding_hash') != b['source_input_binding']['binding_hash'])):
            raise ContractError('Source checkpoint is incomplete or belongs to another model/input/science version')
        files = {}
        for item in sorted(stage.rglob('*')):
            if item.is_file() and not item.is_relative_to(stage/'checkpoint'):
                name = item.relative_to(stage).as_posix()
                with item.open('r+b') as stream:
                    os.fsync(stream.fileno())
                files[name] = dict(sha256=file_sha(item), bytes=item.stat().st_size)
        row = receipt(dict(session_sha256=self.session_hash, owner_uuid=self.session['owner_uuid'],
            index=index, previous_commit_sha256=None if last is None else last['receipt_sha256'],
            start_seconds=float(start), stop_seconds=float(stop), first_boundary_sha256=first_boundary_hash,
            last_boundary_sha256=last_boundary_hash, checkpoint_manifest_sha256=digest,
            cache_identity_sha256=cache_identity, cache_content_sha256=cache_content, files=files))
        write_json(stage/'commit.json', row)
        target = self.root/f'chunk-{index:06d}'
        if target.exists():
            raise ContractError('Source chunk already committed; no replacement/replay')
        os.replace(stage, target)
        write_json(self.root/'head.json', dict(index=index, commit_sha256=row['receipt_sha256']), replace=True)
        return row

    def retire_owned_checkpoints(self):
        """Keep two committed recovery states, never cache/output/research assets.

        This method only removes checkpoint subdirectories of this explicitly
        created UUID-bound session. A manifest and retirement receipt remain.
        """
        last = self.latest()
        if last is None or last['index'] < 2:
            return
        # Never prune unless both retained snapshots still have their manifests.
        for index in (last['index']-1, last['index']):
            chunk = self.root/f'chunk-{index:06d}'
            r = read_receipt(chunk/'commit.json')
            verify_checkpoint_files(chunk/'checkpoint', r['checkpoint_manifest_sha256'])
        for index in range(last['index']-1):
            chunk = self.root/f'chunk-{index:06d}'
            row = read_receipt(chunk/'commit.json')
            target = chunk/'checkpoint'
            if not target.exists():
                continue
            resolved = target.resolve()
            if (row['owner_uuid'] != self.session['owner_uuid'] or row['session_sha256'] != self.session_hash
                    or chunk.resolve().parent != self.root or resolved.parent != chunk.resolve()
                    or resolved.name != 'checkpoint'):
                raise ContractError('Refusing checkpoint retirement outside owned new session')
            manifest = json.loads((target/'manifest.json').read_bytes())
            digest = manifest.pop('manifest_hash')
            if canonical_hash(manifest) != digest or digest != row['checkpoint_manifest_sha256']:
                raise ContractError('Old checkpoint identity changed before retirement')
            if not (chunk/'retired_checkpoint_manifest.json').exists():
                write_json(chunk/'retired_checkpoint_manifest.json', dict(manifest, manifest_hash=digest))
            if not (chunk/'retirement.json').exists():
                write_json(chunk/'retirement.json', dict(checkpoint_manifest_sha256=digest,
                    retained_head_commit_sha256=last['receipt_sha256'], reason='two_newer_owned_complete_commits'))
            shutil.rmtree(resolved)
