"""Pinned ISET MAT capture and an explicit, fixed LMS reference-equivalent vH bridge.

This is an engineering input adapter. It neither validates macaque calibration
nor changes native cone/H1 kinetics. Raw capture remains available in the MAT.
"""
from pathlib import Path
import hashlib,json,re,math
import numpy as np
from scipy.io import loadmat,whosmat
from .contracts import ContractError,canonical_hash

CAPTURE_SCHEMA='isetbio_cone_capture_v1'
REFERENCE_SCHEMA='fixed_LMS_reference_equivalent_vh_v1'

def project_rf_equals_is_binding(population,*,reference_id):
    """Explicit selection of the existing atlas project transfer, never measured IS."""
    rf=population.receptive_field_positions or population.cells.rf_position
    rf.require('mm','functional_position',(len(population.cells.cell_id),2),'functional_visual_field')
    source=Path(__file__).with_name('source_atlas.py')
    _require("rf_transfer='rf_equals_inner_segment_mm_v1'" in source.read_text(encoding='utf8'),'Existing atlas transfer is unavailable')
    return {'transfer_id':'rf_equals_inner_segment_mm_v1','evidence_tag':'project_approximation',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'reference_id':reference_id,
        'cell_ids':list(population.cells.cell_id),'positions_mm':rf.values.copy()}

def _require(condition,message):
    if not condition:raise ContractError('ISET_INPUT: '+message)

def _sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None

def reference_equivalent_drive(rates,types,profile,*,capture_scope_sha256):
    """q/alpha, alpha=q_ref/(L_v_ref*pupil area); one fixed coefficient per LMS class.

    Class coefficients preserve differences between cells' apertures/positions.
    For changing spectra the result is reference-equivalent drive, not photopic td.
    """
    _require(isinstance(profile,dict) and profile.get('schema')==REFERENCE_SCHEMA,'Explicit versioned fixed LMS reference required')
    for key in ('profile_id','source_id','reference_conditions','scope','evidence_tag'):
        _require(isinstance(profile.get(key),str) and bool(profile[key].strip()),'Missing reference '+key)
    _require(profile['scope'] in ('engineering_fixture','explicit_calibration_candidate','approved_mixed_reference'),'Reference scope is not physiological validation')
    _require(_sha(capture_scope_sha256) and profile.get('capture_scope_sha256')==capture_scope_sha256,'Reference capture/optics scope differs')
    for key in ('reference_scene_sha256','reference_capture_sha256'):
        _require(_sha(profile.get(key)),'Missing frozen '+key)
    lv=profile.get('reference_luminance_cd_m2');pupil=profile.get('pupil_diameter_mm')
    for value in (lv,pupil):
        _require(isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value) and value>0,'Positive explicit reference luminance/pupil required')
    qref=profile.get('reference_rates_rstar_s_LMS')
    _require(isinstance(qref,dict) and set(qref)=={'L','M','S'},'Exactly three fixed LMS reference rates required; no cellwise normalization')
    _require(all(isinstance(v,(int,float)) and not isinstance(v,bool) and np.isfinite(v) and v>0 for v in qref.values()),'Positive finite LMS reference rates required')
    _require(all(x in qref for x in types),'Unknown cone type in reference bridge')
    rates=np.asarray(rates,dtype=float)
    _require(rates.ndim==2 and rates.shape[1]==len(types) and np.all(np.isfinite(rates)) and np.all(rates>=0),'Finite nonnegative time/cell capture rates required')
    iref=float(lv)*np.pi*(float(pupil)/2)**2
    alpha={kind:float(qref[kind])/iref for kind in ('L','M','S')}
    drive=rates/np.array([alpha[kind] for kind in types])[None,:]
    _require(np.all(np.isfinite(drive)),'Reference conversion overflow')
    return drive,{'profile_id':profile['profile_id'],'profile_sha256':canonical_hash(profile),
        'reference_retinal_illuminance_td':iref,'alpha_rstar_s_per_reference_td_LMS':alpha,
        'coefficient_policy':'fixed_LMS_reference_classes_not_per_cell_or_per_frame',
        'capture_scope_sha256':capture_scope_sha256,'evidence_tag':profile['evidence_tag'],'scope':profile['scope'],
        'meaning':'reference_equivalent_cone_drive_not_general_photopic_td',
        'physiological_calibration_claim':False}

def read_isetbio_capture(path,*,max_bytes=128*1024**2):
    """Read bounded v7 MAT numeric arrays, keeping time/cell axes even for one cone."""
    path=Path(path)
    _require(path.is_file() and path.suffix.lower()=='.mat','A local v7 MAT capture is required')
    _require(isinstance(max_bytes,int) and not isinstance(max_bytes,bool) and max_bytes>0,'Positive byte budget required')
    identity=(path.stat().st_size,path.stat().st_mtime_ns)
    _require(identity[0]<=max_bytes,'Capture file exceeds budget')
    rows=whosmat(path);required={'manifest_json','expected_counts','window_edges_seconds','is_mm','rf_mm','wave_nm','retinal_photons','spatial_support_m'}
    _require(len({r[0] for r in rows})==len(rows) and {r[0] for r in rows}==required,'Unknown/missing MAT variables')
    _require(all(kind in ('double','char') for _,_,kind in rows),'Only plain numeric/string MAT arrays are allowed')
    _require(sum(math.prod(shape)* (8 if kind=='double' else 4) for _,shape,kind in rows)<=max_bytes,'Declared MAT arrays exceed budget')
    _require(next(math.prod(shape) for name,shape,_ in rows if name=='manifest_json')<=1024**2,'Capture JSON exceeds budget')
    raw=loadmat(path,chars_as_strings=True,squeeze_me=False,struct_as_record=True,mat_dtype=True)
    manifest=raw['manifest_json']
    _require(manifest.dtype.kind=='U' and manifest.size==1,'Scalar JSON capture manifest required')
    from .offline_input import unique_json
    meta=json.loads(str(manifest.item()),object_pairs_hook=unique_json,
        parse_constant=lambda value:(_require(False,'Nonfinite JSON '+value)))
    _require(isinstance(meta,dict) and meta.get('schema')==CAPTURE_SCHEMA,'Unknown capture schema')
    _require(meta.get('data_axes')==['time','cell'] and meta.get('unit')=='R*/window' and meta.get('value_semantics')=='expected','Expected time/cell window counts required')
    _require(meta.get('isetbio_commit')=='d2924c2fe1d41c7dc5af1006d8081735b08f1b35' and
        meta.get('isetcam_commit')=='17146a9f0470cc89b58d583851f680187268506d','Different pinned toolbox version')
    _require(_sha(meta.get('capture_scope_sha256')),'Frozen capture scope required')
    _require(meta.get('scope') in ('engineering_fixture','explicit_calibration_candidate'),'Explicit capture scope required')
    _require(meta.get('mechanism_ownership')=={'optics':'ISETCam','photopigment_capture':'ISETBio',
        'phototransduction':'consumer','photon_noise':'none','neural_noise':'consumer'},'Capture/transduction/noise ownership differs')
    for key in required-{'manifest_json'}:
        value=raw[key];_require(value.dtype.kind=='f' and np.all(np.isfinite(value)),'Finite numeric '+key+' required')
    cones=meta.get('cones',{});ids=cones.get('cell_ids',[]);types=cones.get('types',[])
    _require(isinstance(ids,list) and ids and all(isinstance(x,str) and x for x in ids) and len(set(ids))==len(ids),'Unique frozen IDs required')
    _require(isinstance(types,list) and len(types)==len(ids) and set(types)<={'L','M','S'},'Frozen LMS types required')
    counts=raw['expected_counts'];edges=raw['window_edges_seconds'].reshape(-1)
    _require(counts.shape==(len(edges)-1,len(ids)) and np.all(counts>=0),'Count dimensions/values differ')
    _require(len(edges)>=2 and edges[0]==0 and np.all(np.diff(edges)>0) and np.allclose(np.diff(edges),np.diff(edges)[0],rtol=0,atol=1e-12),'All integration windows need uniform positive spacing from zero')
    _require(meta.get('time',{}).get('window_edges_seconds')==edges.tolist(),'Time declaration differs from actual window edges')
    for key in ('is_mm','rf_mm'):_require(raw[key].shape==(len(ids),2),'Coordinates must be cell x 2')
    wave=raw['wave_nm'].reshape(-1);photons=raw['retinal_photons']
    _require(len(wave)>=3 and np.all(wave>0) and np.all(np.diff(wave)>0),'Increasing positive wavelengths required')
    _require(photons.ndim in (3,4),'Retinal irradiance needs row/column/wavelength/window axes')
    expected=(photons.shape[0],photons.shape[1],len(wave),len(edges)-1)
    if len(edges)==2 and photons.ndim==3:photons=photons[:,:,:,None]
    _require(photons.shape==expected and np.all(photons>=0),'Retinal irradiance dimensions/values differ')
    _require(raw['spatial_support_m'].shape==photons.shape[:2]+(2,),'Retinal spatial support shape differs')
    _require((path.stat().st_size,path.stat().st_mtime_ns)==identity,'Capture changed during read')
    return {'manifest':meta,'counts':counts.copy(),'edges':edges.copy(),'is_mm':raw['is_mm'].copy(),
        'rf_mm':raw['rf_mm'].copy(),'wave_nm':wave.copy(),'retinal_photons':photons.copy(),
        'spatial_support_m':raw['spatial_support_m'].copy(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def convert_isetbio_capture(capture_path,template_path,output_path,compiled,light,*,reference_id,reference_profile,cone_is_binding):
    """Replace only one cone channel in an already valid offline packet; rods are preserved.

    The explicit template supplies background/prehistory and all other channels.
    The returned packet must still pass the existing population/clock contracts.
    """
    from .offline_input import load_offline_inputs,unique_json
    template=load_offline_inputs(template_path,compiled,light,reference_id=reference_id)
    capture=read_isetbio_capture(capture_path);meta=capture['manifest'];cones=meta['cones']
    targets=[e for e in compiled.spec.external_inputs if compiled.components[e.target_component].inputs[e.target_port].unit=='td']
    _require(len(targets)==1,'Exactly one native cone channel expected')
    name=targets[0].channel;pop=compiled.populations.get(cones['population_id'])
    _require(pop is not None and set(pop.cells.cell_id)==set(cones['cell_ids']),'Capture IDs/population differ from the owned cone population')
    order=[cones['cell_ids'].index(x) for x in pop.cells.cell_id]
    _require(tuple(cones['types'][i] for i in order)==tuple(pop.cells.subtype),'Capture types differ from frozen population')
    _require(isinstance(cone_is_binding,dict) and cone_is_binding.get('reference_id')==reference_id and
        cone_is_binding.get('cell_ids')==list(pop.cells.cell_id) and _sha(cone_is_binding.get('source_sha256')),
        'Explicit owned IS binding/reference/source required')
    ispos=np.asarray(cone_is_binding.get('positions_mm'),dtype=float)
    _require(ispos.shape==(len(order),2) and np.all(np.isfinite(ispos)),'Explicit finite owned IS coordinates required')
    _require(meta.get('coordinates',{}).get('is_binding_id')==cone_is_binding.get('transfer_id') and
        meta.get('coordinates',{}).get('is_binding_source_sha256')==cone_is_binding['source_sha256'],'Capture IS binding identity differs')
    _require(np.allclose(capture['is_mm'][order],ispos,rtol=0,atol=1e-12),'Capture IS coordinates differ from owned anatomical positions')
    coordinates=meta.get('coordinates',{})
    _require(coordinates.get('reference_id')==reference_id and coordinates.get('is_role')=='inner_segment' and
        coordinates.get('is_frame')=='anatomical_retina' and coordinates.get('rf_role')=='rf' and
        coordinates.get('rf_frame')=='functional_visual_field','Capture coordinate roles/reference differ')
    rates=capture['counts']/np.diff(capture['edges'])[:,None]
    _,bridge=reference_equivalent_drive(rates,cones['types'],reference_profile,capture_scope_sha256=meta['capture_scope_sha256'])
    _require(reference_profile['pupil_diameter_mm']==meta['optics']['pupil_diameter_mm'],'Reference pupil differs from optics')
    _require(reference_profile['scope']==meta['scope'],'Reference/capture scopes differ')
    with np.load(template_path,allow_pickle=False) as data:
        manifest=json.loads(data['manifest_json'].item(),object_pairs_hook=unique_json)
        arrays={k:data[k].copy() for k in data.files if k!='manifest_json'}
    d=manifest['channels'][name]
    _require(meta['cones']['evidence_tag']==d['evidence_tag'],'Capture metadata cannot upgrade the consumer evidence')
    _require(meta['time']['source_origin_seconds']==d['time']['source_origin_seconds'] and
        capture['edges'][-1]==d['time']['stop_seconds'],'Explicit source epoch/duration differ from template')
    d.update(cell_ids=cones['cell_ids'],unit='R*/window',input_stage='reference_equivalent_cone_drive',
        value_semantics='expected',coordinate_registration=coordinates['registration'],
        reference_profile=reference_profile,capture_scope_sha256=meta['capture_scope_sha256'],
        capture_sha256=capture['sha256'],capture_types=cones['types'],capture_pupil_diameter_mm=meta['optics']['pupil_diameter_mm'],
        capture_scope=meta['scope'],photon_noise={'owner':'none','model':'none'},
        mechanism_ownership={'photopigment_capture':'upstream','phototransduction':'consumer','neural_noise':'consumer'})
    d['time'].update(representation='integration_windows',interpolation='zero_order_hold',source_grid_step_seconds=float(np.diff(capture['edges'])[0]))
    arrays[d['positions_key']]=capture['rf_mm'];arrays[d['values_key']]=capture['counts'];arrays[d['time']['axis_key']]=capture['edges']
    manifest['producer']={'name':'Pinned ISETCam/ISETBio capture adapter','version':'1.0','source_id':meta['source_id'],
        'capture_sha256':capture['sha256'],'capture_manifest':meta,'fixed_reference_bridge':bridge,
        'preserved_other_channels_from_sha256':template.source_summary['source_sha256']}
    output=Path(output_path);_require(not output.exists(),'Output exists; choose a fresh path')
    with output.open('xb') as stream:np.savez_compressed(stream,manifest_json=np.array(json.dumps(manifest,allow_nan=False)),**arrays)
    return load_offline_inputs(output,compiled,light,reference_id=reference_id)
