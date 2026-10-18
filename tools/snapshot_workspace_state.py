"""Save reversible workspace code/configuration plus frozen dependency identities.

Large immutable generated assets remain in place and are verified by hash.
No Git index mutation, cleanup, generation or checkpoint overwrite is performed.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys,platform

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    import numpy,scipy
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=ROOT/'reports'/('workspace_recovery_'+stamp);output.mkdir(exist_ok=False)
    files={};dependencies={};pending=[];unavailable=[]
    def copy(path):
        relative=path.relative_to(ROOT).as_posix()
        if relative in files:return
        raw=path.read_bytes();target=output/'workspace'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        expected=hashlib.sha256(raw).hexdigest()
        if sha(target)!=expected or sha(path)!=expected:raise RuntimeError('Snapshot source changed while copying: '+relative)
        files[relative]={'sha256':expected,'bytes':len(raw)}
    for directory in ('retina_sim','retina_platform','tools','tests','examples','configurations','schemas','sources','docs'):
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo'):
                copy(path)
    for path in ROOT.iterdir():
        if path.is_file():copy(path)
    def bindings(value):
        if isinstance(value,dict):
            relative=value.get('path',value.get('artifact'));expected=value.get('sha256')
            if isinstance(relative,str) and isinstance(expected,str) and len(expected)==64:pending.append((relative,expected))
            asset=value.get('asset_path');expected_asset=value.get('expected_asset_sha256')
            if isinstance(asset,str) and isinstance(expected_asset,str):pending.append((asset,expected_asset))
            for item in value.values():bindings(item)
        elif isinstance(value,list):
            for item in value:bindings(item)
    maincfg=json.loads((ROOT/'configurations/retina_platform_v1.json').read_bytes());bindings(maincfg)
    paths=[ROOT/'configurations'/name for name in ('atlas_isetbio_source_stop_shared_v1.json','atlas_human_borrowed_disk3mm_v1.json',
        'atlas_human_borrowed_disk3mm_floor_v1.json','retina_atlas_human_borrowed_disk3mm_v1.json','atlas_hc_disk3mm_v1.json')]
    for folder in ('human_disk3mm_build_20261017','human_disk3mm_midget_floor_20261017'):
        paths += [p for p in (ROOT/'reports'/folder).glob('*.json') if p.stat().st_size<4*1024**2]
    for path in paths:
        if path.exists():copy(path);bindings(json.loads(path.read_bytes()))
    while pending:
        relative,expected=pending.pop();path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT):continue
        relative=path.relative_to(ROOT).as_posix()
        if relative in dependencies:
            if dependencies[relative]['sha256']!=expected:unavailable.append({'path':relative,'reason':'conflicting historical binding','expected':expected})
            continue
        if not path.is_file():unavailable.append({'path':relative,'reason':'declared historical asset unavailable','expected':expected});continue
        actual=sha(path)
        if actual!=expected:unavailable.append({'path':relative,'reason':'declared historical hash differs','expected':expected,'actual':actual});continue
        dependencies[relative]={'sha256':actual,'bytes':path.stat().st_size,'storage':'immutable original retained in workspace'}
        if path.suffix=='.json' and path.stat().st_size<=4*1024**2:
            copy(path);bindings(json.loads(path.read_bytes()))
    for name,args in [('git_diff.patch',['diff','--binary','--no-ext-diff']),('git_status.z',['status','--porcelain=v1','-z']),
                      ('git_index.z',['ls-files','-s','-z']),('git_HEAD.txt',['rev-parse','HEAD'])]:
        (output/name).write_bytes(subprocess.check_output(['git',*args],cwd=ROOT))
    paused=ROOT/'reports/disk3mm_build_20261016/macaque_paused_20261017T140036Z/manifest.json'
    receipt={'schema':'workspace_recovery_snapshot_v1','created_UTC':stamp,'workspace':str(ROOT),
        'authorization':'2026-10-17T15:01Z Sentinel_4eb30bb2cc1481919b6ca9e8eabe4093: preserve current state and continue reversible mainline work',
        'python':sys.executable,'Python_version':platform.python_version(),'NumPy':numpy.__version__,'SciPy':scipy.__version__,
        'files':files,'verified_dependencies':dependencies,'unavailable_historical_bindings':unavailable,
        'all_snapshot_file_copies_byte_verified':True,'source_code_configuration_coverage':'all files in source/config/tests/docs/tools/examples/schemas/sources plus root files, including uncommitted/untracked local code',
        'large_assets_policy':'not copied or mutated; immutable originals retained and recorded by verified SHA256',
        'inflight_work':'midget-floor frozen graph build may still be completing; this is not a neural state checkpoint',
        'macaque5855_snapshot':{'path':paused.parent.relative_to(ROOT).as_posix(),'manifest_sha256':sha(paused),'modified':False}}
    (output/'manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'RESTORE.txt').write_text('Stop active model workers before rollback. Verify manifest hashes. Restore files under workspace/ to the same project paths; original large assets remain in place and must match verified_dependencies. Git HEAD/index/diff evidence is included. No automatic deletion, history rewrite or macaque-generator resume is part of recovery.\n',encoding='utf-8')
    check=json.loads((output/'manifest.json').read_bytes());assert len(check['files'])==len(files)
    print(json.dumps({'snapshot':output.relative_to(ROOT).as_posix(),'manifest_sha256':sha(output/'manifest.json'),
        'copied_files':len(files),'copied_bytes':sum(v['bytes'] for v in files.values()),'verified_asset_dependencies':len(dependencies),
        'unavailable_historical_bindings':len(unavailable),'byte_verification':True,'macaque5855_snapshot_modified':False}),flush=True)

if __name__=='__main__':main()
