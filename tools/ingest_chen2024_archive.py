"""Safely ingest only verified published MAT/README; never execute members."""
from pathlib import Path,PurePosixPath,PureWindowsPath
import datetime,hashlib,json,stat,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
EXPECTED={'Figure2-SourceData1.mat':(41490462,'6724e2edd8e8708456f32900a14efae88192a5ba597365046d0031eb543d1331'),'README.md':(1357,'f25cb72716327f5fc08cfe26cc423bc3364f8f331900dd1972eeb664c1911d03')}
def ingest(archive):
    archive=Path(archive).resolve(strict=True)
    out=ROOT/'data/source/chen2024_dryad_v3_325663';out.mkdir(parents=True,exist_ok=True)
    before=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
    inventory={'archive_path':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':before,'ingested_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset_doi':'10.5061/dryad.q2bvq83vg','version_id':325663,'members':[],'extractions':[],'policy':'Only exact published MAT/README; verify SHA before writing; no code execution; archive unchanged'}
    with zipfile.ZipFile(archive) as z:
        names=set()
        for member in z.infolist():
            name=member.filename;p=PurePosixPath(name);w=PureWindowsPath(name)
            mode=member.external_attr>>16
            if name in names or p.is_absolute() or w.is_absolute() or w.drive or '..' in p.parts or '\\' in name or ':' in name or stat.S_ISLNK(mode):
                raise ValueError('Unsafe/duplicate archive member: '+name)
            names.add(name)
            inventory['members'].append({'name':name,'bytes':member.file_size,'compressed_bytes':member.compress_size,'crc32':f'{member.CRC:08x}','external_attr':member.external_attr,'is_dir':member.is_dir()})
        for name,(size,digest) in EXPECTED.items():
            info=z.getinfo(name)
            if info.file_size!=size: raise ValueError('Published size mismatch: '+name)
            raw=z.read(info);actual=hashlib.sha256(raw).hexdigest()
            if len(raw)!=size or actual!=digest: raise ValueError('Published identity mismatch: '+name)
            target=(out/name).resolve()
            if out.resolve() not in target.parents: raise ValueError('Extraction escapes source directory')
            if target.exists() and target.read_bytes()!=raw: raise ValueError('Existing source asset differs; refusing overwrite')
            target.write_bytes(raw)
            inventory['extractions'].append({'member':name,'path':target.relative_to(ROOT).as_posix(),'bytes':len(raw),'actual_sha256':actual,'published_sha256':digest,'identity':'VERIFIED'})
    after=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
    if before!=after: raise ValueError('Source archive changed during ingestion')
    inventory['source_archive_unchanged']=True
    (ROOT/'reports/chen2024_archive_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return inventory
if __name__=='__main__':
    result=ingest(sys.argv[1]);print(json.dumps(result,ensure_ascii=True))
