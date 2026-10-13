"""Register verified data/README/paper assets; change no model parameters."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retina_sim.provenance import Registry
ledger_path=ROOT/'sources/registry.json'
ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
parameter_bytes=json.dumps(ledger['parameters'],sort_keys=True)
profile_bytes=json.dumps(ledger['profiles'],sort_keys=True)
inventory=json.loads((ROOT/'reports/chen2024_data_inventory.json').read_text(encoding='utf-8'))
assert inventory['identity_and_structure']=='VERIFIED'
archive=json.loads((ROOT/'reports/chen2024_archive_inventory.json').read_text(encoding='utf-8'))
new=[]
for sid,name,kind,location,level,license in [
 ('CHEN2024_Dryad_SourceData','Figure2-SourceData1.mat','data_table','PrimateConeResponses{1:6}, exact fields and array inventory reports/chen2024_data_inventory.json','numeric_table_verified','CC0; actual official dataset metadata'),
 ('CHEN2024_Dryad_README','README.md','supplement','Primate and mouse cone responses paragraphs: 0.1ms, R*/sec, normalized repeat means','identity_verified','CC0; actual official dataset metadata'),
 ('CHEN2024_Paper_93795','elife-93795-v1.pdf','paper','pp5/15/18-19: fitting, scope, recordings and model methods','identity_verified','CC BY; article copyright statement p1')]:
    path=ROOT/'data/source/chen2024_dryad_v3_325663'/name
    new.append({'source_id':sid,'title':'Chen et al 2024 verified public source '+name,'authors':['Chen Q','Ingram NT','Baudin J','Angueyra JM','Sinha R','Rieke F'],'publication_year':2024,'doi_or_canonical_url':'https://cdn.elifesciences.org/articles/93795/elife-93795-v1.pdf' if kind=='paper' else 'https://datadryad.org/dataset/doi:10.5061/dryad.q2bvq83vg','artifact_kind':kind,'repository_commit':None,'source_path':path.relative_to(ROOT).as_posix(),'source_locator':location,'asset_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'source_blob_sha1':None,'license':license,'verification_level':level,'provenance_notes':'Actual asset bytes verified. MAT/README published hashes match user-provided archive. Data are source repeat means, not independent validation; no per-cell biology or preprocessing invented. No parameter record derived or retuned.','retrieved_at':archive['ingested_at_utc'],'upstream_sources_checked':{'actual_bytes':'archive inventory and file SHA256','structure':'actual MATLAB cell/field/shape/dtype inventory','units':'actual README, paper and pinned code checked','input_semantics_history':'unresolved negative rates, range and prehistory; model comparison NOT_RUN'}})
ids={s['source_id'] for s in new}
ledger['sources']=[s for s in ledger['sources'] if s['source_id'] not in ids]+new
assert json.dumps(ledger['parameters'],sort_keys=True)==parameter_bytes
assert json.dumps(ledger['profiles'],sort_keys=True)==profile_bytes
ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
Registry.read(ledger_path)
print(json.dumps({'sources':len(ledger['sources']),'parameters':len(ledger['parameters']),'profiles':len(ledger['profiles']),'parameters_changed':False,'profiles_changed':False}))
