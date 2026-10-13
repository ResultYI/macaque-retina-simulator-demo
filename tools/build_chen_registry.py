"""Extend, never replace, stage A/B ledger using actual historical author literals."""
from pathlib import Path
import hashlib,json,re,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from retina_sim.parameter_contracts import CHEN_PARAMETER_SPECS
ROOT=Path(__file__).resolve().parents[1]
COMMIT='1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2'
PROFILE='Chen2024PeripheralPrimateCone_source_mixed_macaca'
SPECIES='source_mixed_Macaca_fascicularis_nemestrina_mulatta'
ledger=json.loads((ROOT/'sources/registry.json').read_text(encoding='utf-8'))
manifest=json.loads((ROOT/'sources/chen2024_asset_manifest.json').read_text())
for key in ('sources','parameters','profiles'):
    identity={'sources':'source_id','parameters':'parameter_id','profiles':'profile_id'}[key]
    known_code_sources={'CHEN2024_'+Path(a['path']).name.replace('.','_') for a in manifest['assets']}
    ledger[key]=[r for r in ledger[key] if (r[identity] not in known_code_sources if key=='sources' else not r[identity].startswith(('CHEN2024_','Chen2024')))]
for asset in manifest['assets']:
    raw=(ROOT/asset['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=asset['asset_sha256'] or hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=asset['source_blob_sha1']: raise ValueError('Source identity changed')
    name=Path(asset['path']).name
    asset['verification_level']='code_verified' if name in ('BiophysModel.m','initPhotoreceptorParams.m') else 'identity_verified'
    ledger['sources'].append(dict(source_id='CHEN2024_'+name.replace('.','_'),title='Chen et al 2024 historical author '+name,authors=['Chen Q','Rieke F','author repository contributors'],publication_year=2024,doi_or_canonical_url=asset['url'],artifact_kind='code',repository_commit=COMMIT,source_path=asset['path'],source_locator={'BiophysModel.m':'nonlinear branch lines 33-72; old-state R/P/C/G; S from Cnew, G from Sold','initPhotoreceptorParams.m':'timeStep line16; peripheralPrimateCone lines19-31; darkCurrent line69'}.get(name,'identity/license only; demo never executed'),asset_sha256=asset['asset_sha256'],source_blob_sha1=asset['source_blob_sha1'],license='MIT; Copyright 2024 chrischen2; retained LICENSE',verification_level=asset['verification_level'],provenance_notes='Actual pinned original bytes and Git blob verified. Nonlinear source code, not primary animal measurement or independent validation.',retrieved_at=manifest['retrieved_at'],upstream_sources_checked={'author_files':'actual static inspection of nonlinear/parameter functions only','paper_scope':'provided independent audit; >20deg L/M, <50k R*/s, three Macaca sources','upstream_individual_measurements':'not independently audited; no per-cell metadata fabricated'}))
text=(ROOT/'data/source_snapshots/chen2024_1bcf7c9a/initPhotoreceptorParams.m').read_text()
branch=re.search(r"case 'peripheralPrimateCone'(.*?)case 'primateRod'",text,re.S)
assert branch
conditions={'module':'outer_segment_current_only','region':'>20deg','cone_types':['L','M'],'source_species':['M. fascicularis','M. nemestrina','M. mulatta'],'public_input':'total_R*/s','time_step_seconds':.0001,'no_bleaching_or_regeneration':True}
for parameter_id,(unit,shape) in CHEN_PARAMETER_SPECS.items():
    name=parameter_id.removeprefix('CHEN2024_')
    block=text if name=='timeStep' else branch.group(1)
    match=re.search(r'params\.'+name+r'\s*=\s*([^;]+);([^\n]*)',block)
    if not match: raise ValueError('Missing source parameter '+name)
    actual_index=match.start()+(0 if name=='timeStep' else branch.start(1))
    line=text.count('\n',0,actual_index)+1
    value=float(match.group(1))
    ledger['parameters'].append(dict(parameter_id=parameter_id,name=name,value=value,unit=unit,shape=list(shape),source_id='CHEN2024_initPhotoreceptorParams_m',source_locator=f'initPhotoreceptorParams.m line{line}, peripheralPrimateCone; literal {match.group(1).strip()}',derivation={'value_transform':'identity actual source literal','source_comment':match.group(2).strip(),'interface_unit_basis':'Source normalized effective activities; P formally s^-1 via dG=S-PG; R/eta formally s^-2 via dP=R+eta-phiP; gamma formally s^-2/R*. These are dimensional derivations, not measured molecule units.' if name in ('eta','gamma') else 'Source/paper convention preserved; timeStep is source numerical setting, not physiology constant'},evidence_kind='author_assumption',species=SPECIES,sample_context='Published consensus from mixed three-Macaca source recordings; individual identities and uncertainty not available here',retinal_region='peripheral >20deg only',meridian='not reported per cell',coordinate_frame='outer_segment',support_domain={'eccentricity':{'unit':'deg','lower':20,'upper':None,'lower_closed':False,'upper_closed':False},'total_rate':{'unit':'R*/s','lower':0,'upper':50000,'lower_closed':True,'upper_closed':False}},conditions=conditions,uncertainty='Source consensus; per-cell and fit uncertainty not independently established',allowed_modes=['component_reference','numeric_fixture'],status='usable_reference',assumption_approval_reference=None))
ledger['profiles'].append(dict(profile_id=PROFILE,species=SPECIES,kind='named_reference',parameter_ids=list(CHEN_PARAMETER_SPECS),conditions=conditions,coordinate_frame='outer_segment',explicit_opt_in=True,claim_limit='Isolated peripheral LM nonlinear outer-segment current; mixed Macaca sources; no foveal/S/V/HC/bleaching or independent physiology claim'))
ledger['species_policy']['chen2024_approval_reference']='Sentinel_db5d1bccdbbc8191bed02a864c13f5f4'
(ROOT/'sources/registry.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'sources/chen2024_asset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'added_sources':4,'added_parameters':12,'profiles':len(ledger['profiles']),'values_parsed_from_actual_source':True}))
