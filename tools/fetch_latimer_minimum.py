"""Fetch one smallest primary-data MAT plus public source metadata, with caps."""
from pathlib import Path
import hashlib,json,urllib.request,urllib.parse
ROOT=Path(__file__).resolve().parents[1]


def fetch(url,path,cap):
    if path.exists():raise FileExistsError(path)
    request=urllib.request.Request(url,headers={'User-Agent':'retina-source-audit/1.0'})
    with urllib.request.urlopen(request,timeout=45) as r:
        if int(r.headers.get('Content-Length','0'))>cap:raise ValueError('Declared asset exceeds cap')
        raw=r.read(cap+1)
        if len(raw)>cap:raise ValueError('Asset exceeds byte cap')
        resolved=urllib.parse.urlsplit(r.url)
        result=dict(requested_url=url,resolved_url=urllib.parse.urlunsplit((resolved.scheme,resolved.netloc,resolved.path,'','')),
            bytes=len(raw),content_type=r.headers.get('Content-Type'),
            sha256=hashlib.sha256(raw).hexdigest(),md5=hashlib.md5(raw).hexdigest())
    path.write_bytes(raw);return result


def main():
    d=ROOT/'data/source_data/latimer2019_figshare_v1';d.mkdir(parents=True,exist_ok=True)
    meta=json.loads((d/'article_metadata.json').read_bytes());chosen=min(meta['files'],key=lambda f:f['size'])
    if meta['version']!=1 or chosen['name']!='Cell_6.mat' or chosen['size']!=17007338:
        raise ValueError('Expected reviewed article v1 smallest record identity')
    out=dict(article_id=meta['id'],article_version=meta['version'],selection='Smallest file by declared bytes, before looking at observations',
        complete_bundle_downloaded=False,files={})
    row=fetch(chosen['download_url'],d/chosen['name'],25*1024**2)
    if row['bytes']!=chosen['size'] or row['md5']!=chosen['computed_md5']:raise ValueError('Figshare asset identity mismatch')
    out['files'][chosen['name']]=row
    for name,url,cap in (
        ('latimer2019_PMC6989090.xml','https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6989090/fullTextXML',2*1024**2),
        ('author_repository_commit.json','https://api.github.com/repos/pillowlab/CBEM/commits/master',256*1024),
        ('author_data_directory.json','https://api.github.com/repos/pillowlab/CBEM/contents/Data',256*1024)):
        try:out['files'][name]=fetch(url,d/name,cap)
        except Exception as exc:out['files'][name]=dict(error_type=type(exc).__name__,error=str(exc))
    (d/'download_manifest.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':main()
