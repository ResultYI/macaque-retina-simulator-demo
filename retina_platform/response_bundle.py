"""Portable, bounded JSON/NPZ result packages. No scripts or pickle are executed."""
from pathlib import Path, PurePosixPath
from dataclasses import dataclass
import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
import zipfile

MAX_TOTAL = 8 * 2**30
MAX_FILE = 2 * 2**30
MAX_FILES = 4096


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name):
    parts = PurePosixPath(name).parts
    if not parts or '\\' in name or ':' in name or name.startswith('/') or any(p in ('..', '.', '') for p in name.split('/')):
        raise ValueError('包内路径非法：' + name)
    if any(p.rstrip(' .') != p or p.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))} for p in parts):
        raise ValueError('包内Windows路径非法：' + name)
    return name


@dataclass
class ResultSource:
    manifest: dict
    files: dict
    temporary: object = None
    origin: Path | None = None

    def close(self):
        if self.temporary is not None:
            self.temporary.cleanup()
            self.temporary = None


def validate_manifest(manifest):
    if manifest.get('schema') != 'retina-response-bundle-1':
        raise ValueError('不支持的响应包版本')
    if manifest.get('status') != 'complete':
        raise ValueError('这是partial/失败结果，不能作为完整响应回放')
    if not manifest.get('positions') or not manifest.get('run_id'):
        raise ValueError('缺少位置或运行身份')
    descriptors = manifest.get('files', {})
    if not descriptors or len(descriptors) > MAX_FILES or sum(x['bytes'] for x in descriptors.values()) > MAX_TOTAL:
        raise ValueError('包清单为空或超过8GiB/4096文件限制')
    for name, row in descriptors.items():
        safe_name(name)
        if not 0 <= row['bytes'] <= MAX_FILE or len(row.get('sha256', '')) != 64:
            raise ValueError('文件大小或SHA清单非法：' + name)
    required = [manifest['stimulus']['frames']]
    for position in manifest['positions'].values():
        required += [position['geometry'], position['connections'], position['response_prefix'] + 'RESPONSE_INDEX.json',
                     position['response_prefix'] + 'IDENTITY.json', position['response_prefix'] + 'GROUP_OBJECT_EVENTS.json']
    if any(name not in descriptors for name in required):
        raise ValueError('完整响应包缺少必要文件')


def open_package(path):
    path = Path(path).resolve()
    temporary = tempfile.TemporaryDirectory(prefix='retina_response_')
    directory = Path(temporary.name)
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [safe_name(entry.filename) for entry in entries]
            if len(names) > MAX_FILES + 1 or len(set(n.casefold() for n in names)) != len(names):
                raise ValueError('包内文件过多或存在重复/大小写冲突路径')
            if sum(entry.file_size for entry in entries) > MAX_TOTAL or any(entry.file_size > MAX_FILE for entry in entries):
                raise ValueError('解包大小超限（总8GiB、单文件2GiB）')
            if shutil.disk_usage(directory).free < sum(entry.file_size for entry in entries) + 256 * 2**20:
                raise ValueError('临时目录磁盘空间不足，未开始解包')
            if any(entry.flag_bits & 1 or stat.S_ISLNK(entry.external_attr >> 16) for entry in entries):
                raise ValueError('不支持加密文件或符号链接')
            info = archive.getinfo('manifest.json')
            if info.file_size > 8 * 2**20: raise ValueError('manifest过大')
            manifest = json.loads(archive.read(info))
            validate_manifest(manifest)
            expected = manifest['files']
            if set(names) != set(expected) | {'manifest.json'}:
                raise ValueError('归档文件与manifest不一致')
            for entry in entries:
                name = entry.filename
                target = directory / name
                if not target.resolve().is_relative_to(directory): raise ValueError('包路径逃逸')
                if name != 'manifest.json' and entry.file_size != expected[name]['bytes']:
                    raise ValueError('文件大小与manifest不一致：' + name)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(); size = 0
                with archive.open(entry) as stream, target.open('xb') as output:
                    while chunk := stream.read(2**20):
                        size += len(chunk)
                        if size > min(entry.file_size, MAX_FILE): raise ValueError('解包实际大小超限')
                        output.write(chunk); digest.update(chunk)
                if name != 'manifest.json' and digest.hexdigest() != expected[name]['sha256']:
                    raise ValueError('文件完整性失败：' + name)
        return ResultSource(manifest, {name: directory / name for name in expected}, temporary, path)
    except BaseException:
        temporary.cleanup()
        raise


def open_result(path, root):
    path = Path(path).resolve()
    if path.suffix.lower() == '.zip': return open_package(path)
    base = path if path.is_dir() else path.parent
    if (base / 'manifest.json').is_file():
        manifest = read(base / 'manifest.json')
        if manifest.get('schema') == 'retina-response-bundle-1':
            validate_manifest(manifest)
            files = {}
            for name, row in manifest['files'].items():
                member = (base / safe_name(name)).resolve()
                if not member.is_relative_to(base) or not member.is_file() or member.stat().st_size != row['bytes'] or sha(member) != row['sha256']:
                    raise ValueError('结果目录文件完整性或路径失败：' + name)
                files[name] = member
            return ResultSource(manifest, files, origin=base)
    if not (base / 'PREPARATION.json').is_file() and (base.parent / 'PREPARATION.json').is_file():base=base.parent
    if not (base / 'PREPARATION.json').is_file():
        raise ValueError('请选择完整响应ZIP或包含PREPARATION.json的运行目录')
    from .response_metadata import metadata
    temporary = tempfile.TemporaryDirectory(prefix='retina_metadata_')
    try:
        manifest, files = metadata(Path(root), base, Path(temporary.name))
        return ResultSource(manifest, files, temporary, base)
    except BaseException:
        temporary.cleanup()
        raise


def export_package(source, target, progress=lambda text: None):
    if source.manifest.get('status') != 'complete':raise ValueError('partial结果不能导出为完整响应包')
    target = Path(target).resolve()
    if target.exists(): raise FileExistsError('保留已有结果包，请选择新文件名：' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(source.manifest)
    manifest['files'] = {}
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.partial')
    total = sum(path.stat().st_size for path in source.files.values())
    if total > MAX_TOTAL: raise ValueError('结果包超过8GiB限制')
    try:
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for index, (name, path) in enumerate(source.files.items()):
                safe_name(name)
                size = path.stat().st_size
                if size > MAX_FILE: raise ValueError('单文件超过2GiB：' + name)
                archive.write(path, name)
                category = name.split('/')[0]
                manifest['files'][name] = dict(bytes=size, sha256=sha(path), category=category,
                    purpose={'stimulus':'原始播放帧与时间', 'responses':'完整已保存逐组响应', 'cells':'坐标/组身份/真实连接',
                             'provenance':'参数和来源', 'reports':'报告和指标', 'logs':'执行日志'}[category])
                progress(f'打包 {index+1}/{len(source.files)} · {category}')
            validate_manifest(manifest)
            archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
        with temporary.open('r+b') as stream: os.fsync(stream.fileno())
        # Windows rename is atomic and refuses to replace an existing target.
        os.rename(temporary, target)
    except BaseException:
        if temporary.exists(): temporary.unlink()
        raise
    return target
