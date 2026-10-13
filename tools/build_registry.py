"""Materialize schema, ledger and coverage from pinned actual source files."""
from dataclasses import fields
import datetime, hashlib, json, re, shutil, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from retina_sim.provenance import SourceRecord, ParameterRecord, Profile, Registry

ROOT=Path(__file__).resolve().parents[1]
COMMIT='5406d1533dae1678f94687379ac078a612f6170f'
AUTHOR=ROOT/'data/source_snapshots/somaratna_freeman_5406d153'
asset_manifest=json.loads((ROOT/'sources/author_asset_manifest.json').read_text())
statically_audited={'runColFun.m','setColLit.m','runColTask.m'}
for asset in asset_manifest['assets']:
    asset['verification_level']='code_verified' if asset['name'] in statically_audited else 'identity_verified'
(ROOT/'sources/author_asset_manifest.json').write_text(json.dumps(asset_manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
retrieved=asset_manifest['retrieved_at']
sources=[]
for a in asset_manifest['assets']:
    name=a['name']
    sources.append(dict(source_id='SF2025_'+name.replace(' ','_').replace('.','_'),title='Somaratna & Freeman 2025 author asset: '+name,authors=['Somaratna MA','Freeman AW'],publication_year=2025,doi_or_canonical_url='https://github.com/AlanFreeman/Colour/blob/'+COMMIT+'/'+name.replace(' ','%20'),artifact_kind='data_table' if name.endswith('.mat') else 'code',repository_commit=COMMIT,source_path=a['path'],source_locator={'runColFun.m':'calConv 52-65; calDeriv 67-107; gang 266-298; solveF 413-469; solveT 471-510; weight 512-550','setColLit.m':'literature-model constants and fit coefficients, entire file','runColTask.m':'resp: output rectification line 666'}.get(name,'entire source file; binary MCOS table not numerically decoded' if name.endswith('.mat') else 'entire source file'),asset_sha256=a['asset_sha256'],source_blob_sha1=a['source_blob_sha1'],license='CC BY 4.0; License.txt',verification_level=a['verification_level'],provenance_notes='Actual fixed-commit file downloaded and hashed. Code source, not primary measurement. MAT identity is not numerical table verification.',retrieved_at=retrieved,upstream_sources_checked={'author_code':'actual file statically read','upstream_papers':'not independently reread in this execution; preserve author citation labels and supplied audit limitations'}))

packer=ROOT/'data/source_snapshots/isetmacaque_cf22ac9d/PackerConeRc.m'
sources.append(dict(source_id='ISETMacaque_PackerConeRc',title='ISETMacaque published_code_tabulation',authors=['ISETMacaque contributors'],publication_year=2026,doi_or_canonical_url='https://github.com/isetbio/ISETMacaque/blob/cf22ac9d681482ca1fcc365a423ef0c2be7916a3/toolbox/%2Bsimulator/%2Bload/PackerConeRc.m',artifact_kind='code',repository_commit='cf22ac9d681482ca1fcc365a423ef0c2be7916a3',source_path=str(packer.relative_to(ROOT)).replace('\\','/'),source_locator='macaqueConeDensity: data 59-73,78-90,103-121,126-138; scaling 75,93,123,142; Rc 14-21,30-31',asset_sha256=hashlib.sha256(packer.read_bytes()).hexdigest(),source_blob_sha1=None,license='MIT; retained LICENSE',verification_level='numeric_table_verified',provenance_notes='Source-code table literals and scaling verified, not original measured cell counts. Exact species not inferred from repository name. Publication year 2026 is asset audit year, not underlying Packer paper year.',retrieved_at=retrieved,upstream_sources_checked={'Packer original measurement':'pending','local repository':'HEAD checked; whole working tree cleanliness not verified'}))
watson_repo=Path(r'D:\retipath\experiments\retipath_foveal_v1\exploratory_v2\vendor_pinned\isetbio_d2924c2f')
watson_file=watson_repo/'calculators/@WatsonRGCModel/WatsonRGCModel.m'
watson=ROOT/'data/source_snapshots/isetbio_d2924c2f'
watson.mkdir(exist_ok=True)
shutil.copyfile(watson_file,watson/'WatsonRGCModel.m'); shutil.copyfile(watson_repo/'LICENSE',watson/'LICENSE')
sources.append(dict(source_id='ISETBio_WatsonSpacing',title='ISETBio exact hexagonal-spacing relation',authors=['ISETBio contributors'],publication_year=2026,doi_or_canonical_url='https://github.com/isetbio/isetbio/blob/d2924c2fe1d41c7dc5af1006d8081735b08f1b35/calculators/%40WatsonRGCModel/WatsonRGCModel.m',artifact_kind='code',repository_commit='d2924c2fe1d41c7dc5af1006d8081735b08f1b35',source_path='data/source_snapshots/isetbio_d2924c2f/WatsonRGCModel.m',source_locator='lines 113-117; spacingFromDensity=sqrt(2/(sqrt(3)*density))',asset_sha256=hashlib.sha256(watson_file.read_bytes()).hexdigest(),source_blob_sha1=None,license='See retained ISETBio LICENSE (project-specific terms)',verification_level='code_verified',provenance_notes='Mathematical perfect-hex-mosaic conversion used by PackerConeRc; not macaque anatomical spacing measurement. Audit year is not paper publication year.',retrieved_at=retrieved,upstream_sources_checked={'Watson 2014 Eq A4':'author code comment only; not visually inspected paper equation'}))

text=(AUTHOR/'setColLit.m').read_text()
units={'tau':'s','kSens':'mV/contrast-unit','kSur':'dimensionless','kRect':'Hz/mV','actRest':'Hz','magRet':'deg/mm','radOptCoef':'polynomial coefficients for radius_deg(e_deg), descending powers','radHorCoef':'polynomial coefficients for log(radius_deg)(e_deg), descending powers','radGangCoef':'polynomial coefficients for radius_deg(e_deg), descending powers'}
params=[]
for name,unit in units.items():
    match=re.search(r'm\.p\.'+re.escape(name)+r'\s*=\s*([^;]+);',text)
    if not match: raise ValueError('Missing actual author parameter '+name)
    literal=match.group(1).strip()
    value=[float(x.strip()) for x in literal.strip('[]').split(',')] if literal.startswith('[') else float(literal)
    line=text.count('\n',0,match.start())+1
    params.append(dict(parameter_id='SF2025_'+name,name=name,value=value,unit=unit,shape=[len(value)] if isinstance(value,list) else [],source_id='SF2025_setColLit_m',source_locator=f'setColLit.m line {line}; literal {literal}',derivation='identity: parsed actual pinned author literal',evidence_kind='source_fit' if name.startswith('rad') else 'author_assumption',species='source_specific_mixed',sample_context='Author literature model; original sample conditions not comprehensively audited',retinal_region='local source-specific patch; complete physiological region validity not established',meridian='not assigned without decoded author arrays',coordinate_frame='functional_visual_field',support_domain=None,conditions={'branch':'Gaussian-HC','input':'cone_contrast','background_adaptation':'absent'},uncertainty='Unknown author fit/measurement uncertainty; engineering test tolerance is separate',allowed_modes=['component_reference','numeric_fixture'],status='usable_reference',assumption_approval_reference=None))
profile=dict(profile_id='SomaratnaFreeman2025GaussianHCReference_source_specific',species='source_specific_mixed',kind='named_reference',parameter_ids=[p['parameter_id'] for p in params],conditions={'branch':'Gaussian-HC','input':'cone_contrast','background_adaptation':'absent'},coordinate_frame='functional_visual_field',explicit_opt_in=True,claim_limit='Cross-species author-code linear reference only; no main physiology defaults, adaptation or author-array reproduction')
registry={'schema_version':'2.0','species_policy':{'policy':'common framework; parameters kept separately by macaque species and source; no unjustified averaging; human/cat excluded from macaque defaults','approval_reference':'Sentinel_601db2e51e7c81919e962a4cc23a6f39','named_reference':'explicit opt-in, mixed provenance disclosed, not main model'},'sources':sources,'parameters':params,'profiles':[profile]}
for source in sources:
    if source['source_id'].startswith('SF2025_'):
        source['upstream_sources_checked']['author_code']='Relevant functions/parameter literals statically audited' if source['verification_level']=='code_verified' else 'Downloaded identity/bytes verified; no numerical decoding or complete code audit claimed'
    if source['source_id'].startswith('ISET'):
        source['publication_year']=None
        source['upstream_sources_checked']['publication_year']='Unknown original publication year; 2026 refers only to this asset audit'
(ROOT/'sources/registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def obj(cls,properties):
    return {'type':'object','additionalProperties':False,'required':[f.name for f in fields(cls)],'properties':properties}
string={'type':'string','minLength':1}; nullable={'type':['string','null']}
s_props={f.name:string for f in fields(SourceRecord)}
s_props.update(authors={'type':'array','minItems':1,'items':string},publication_year={'type':['integer','null']},repository_commit=nullable,source_blob_sha1=nullable,upstream_sources_checked={'type':'object'},asset_sha256={'type':'string','pattern':'^[0-9a-f]{64}$'},verification_level={'enum':['identity_verified','code_verified','numeric_table_verified','primary_measurement_verified']})
p_props={f.name:string for f in fields(ParameterRecord)}
p_props.update(value={'type':['number','array','null']},shape={'type':'array','items':{'type':'integer','minimum':0}},source_id=nullable,derivation={'type':['string','object']},support_domain={'type':['object','null']},conditions={'type':'object'},allowed_modes={'type':'array','minItems':1,'items':{'enum':['full_physiology','component_reference','numeric_fixture']}},assumption_approval_reference=nullable,evidence_kind={'enum':['measured','source_fit','author_assumption','project_assumption','synthetic_fixture','unknown']},status={'enum':['usable_reference','usable_physiology','blocked']})
profile_props={f.name:string for f in fields(Profile)}
profile_props.update(parameter_ids={'type':'array','uniqueItems':True,'items':string},conditions={'type':'object'},explicit_opt_in={'type':'boolean'},kind={'enum':['macaque_main','named_reference','synthetic_fixture']})
profile_schema=obj(Profile,profile_props)
profile_schema['allOf']=[{'if':{'properties':{'kind':{'const':'macaque_main'}}},'then':{'properties':{'species':{'type':'string','pattern':'^M\\.'}}}},{'if':{'properties':{'kind':{'const':'named_reference'}}},'then':{'properties':{'explicit_opt_in':{'const':True}}}}]
schema={'$schema':'https://json-schema.org/draft/2020-12/schema','title':'来源账本及分物种 profile v2','type':'object','additionalProperties':False,'required':['schema_version','species_policy','sources','parameters','profiles'],'properties':{'schema_version':{'const':'2.0'},'species_policy':{'type':'object','required':['policy','approval_reference','named_reference']},'sources':{'type':'array','items':obj(SourceRecord,s_props)},'parameters':{'type':'array','items':obj(ParameterRecord,p_props)},'profiles':{'type':'array','items':profile_schema},},'description':'Structural validation is separate from semantic loader asset/species/domain/condition checks. Author mixture only for explicitly selected named reference; not main defaults.'}
(ROOT/'schemas/source_registry.schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
Registry.read(ROOT/'sources/registry.json')
print(json.dumps({'sources':len(sources),'parameters':len(params),'profiles':1,'actual_source_literals_parsed':True}))
