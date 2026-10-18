"""Versioned NPZ receptor drives, bound to existing populations and InputChannel.

manifest_json is a scalar JSON string. Each channel declares its array keys;
values are time x cell, positions are cell x 2 in the project's RF frame.
No optical calibration, gain, noise realization or phototransduction is added.
"""
from dataclasses import dataclass,replace
from pathlib import Path
import hashlib,json,zipfile,math
import numpy as np
from .contracts import ContractError,Quantity,canonical_hash
from .spatial import InputChannel
from .photometry import photoisomerization_window_mean_rate

SCHEMA='offline_receptor_input_v1'
ROUND_OFF_SECONDS=1e-12

class OfflineInputError(ContractError):
    def __init__(self,code,detail):
        self.code=code;super().__init__(code+': '+detail)

def fail(code,detail):raise OfflineInputError('OFFLINE_'+code,detail)

def text(value,label):
    if not isinstance(value,str) or not value.strip():fail('METADATA',label+' requires a nonempty string')
    return value

def object_value(value,label):
    if not isinstance(value,dict):fail('METADATA',label+' requires an object')
    return value

def number(value,label):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):fail('METADATA',label+' requires a finite number')
    return float(value)

def unique_json(pairs):
    result={}
    for key,value in pairs:
        if key in result:fail('METADATA','Duplicate JSON key '+key)
        result[key]=value
    return result

def numeric(data,key,label):
    key=text(key,label+' array key')
    if key not in data:fail('ARRAY',label+' array missing: '+key)
    try:value=np.asarray(data[key])
    except ValueError:fail('ARRAY',label+' cannot contain pickled objects')
    if value.dtype.kind not in 'fiu' or not np.all(np.isfinite(value)):fail('ARRAY',label+' must contain finite real numbers')
    return value.astype(np.float64,copy=True)

@dataclass(frozen=True)
class OfflineInputs:
    channels: dict
    source_summary: dict

def _read_channel(data,declared,external,compiled,light,reference_id):
    name=external.channel;component=compiled.components[external.target_component];port=component.inputs[external.target_port]
    d=object_value(declared,name);stage=text(d.get('input_stage'),name+' input_stage');unit=text(d.get('unit'),name+' unit')
    if stage=='photocurrent':fail('ALREADY_PHOTOTRANSDUCED',name+' cannot feed ISET photocurrent into phototransduction again')
    bridge_stage=stage=='reference_equivalent_cone_drive'
    if stage not in ('retinal_illuminance','photoisomerization','reference_equivalent_cone_drive'):fail('STAGE',name+' unsupported input stage; optics/calibration must be explicit upstream')
    if port.unit=='td' and unit in ('R*/s','R*/window') and not bridge_stage:
        fail('VH_TD_MAPPING_UNAVAILABLE','Absolute photoisomerization to vH td is not calibrated; no gain or route substitution')
    if (stage,unit) not in (('retinal_illuminance','td'),('photoisomerization','R*/s'),('photoisomerization','R*/window'),('reference_equivalent_cone_drive','R*/s'),('reference_equivalent_cone_drive','R*/window')):
        fail('UNIT',name+' stage/unit mismatch')
    target_unit='td' if bridge_stage else 'R*/s' if unit=='R*/window' else unit
    if target_unit!=port.unit or port.meaning not in ('native_retinal_illuminance','photoisomerization_rate'):fail('UNIT',name+' does not match its existing consumer port')
    pop_id=text(d.get('population_id'),name+' population_id')
    if pop_id not in compiled.populations:fail('IDS',name+' population does not exist')
    pop=compiled.populations[pop_id]
    if tuple(pop.cells.cell_id)!=tuple(port.cell_ids):fail('IDS',name+' population differs from consumer cell IDs')
    ids=d.get('cell_ids')
    if not isinstance(ids,list) or not ids or any(not isinstance(v,str) or not v for v in ids) or len(set(ids))!=len(ids):
        fail('IDS',name+' requires unique explicit string cell IDs')
    if set(ids)!=set(port.cell_ids):fail('IDS',name+' IDs differ from the existing population; no nearest/row-index substitute')
    source_index={value:i for i,value in enumerate(ids)}
    order=np.array([source_index[v] for v in port.cell_ids],dtype=int)
    if d.get('data_axes')!=['time','cell']:fail('ARRAY',name+' v1 needs explicit time/cell axes; select and record a trial upstream')
    if d.get('coordinate_role')!='rf':fail('COORDINATES',name+' v1 requires RF binding, not a pedicle/soma substitute')
    if d.get('coordinate_unit')!='mm' or d.get('coordinate_frame')!='functional_visual_field':fail('COORDINATES',name+' coordinate unit/frame mismatch')
    if d.get('reference_id')!=reference_id:fail('COORDINATES',name+' global reference differs')
    registration=text(d.get('coordinate_registration'),name+' coordinate_registration')
    rf=pop.receptive_field_positions if pop.receptive_field_positions is not None else pop.cells.rf_position
    if rf is None:fail('COORDINATES',name+' has no supplied RF coordinates')
    rf.require('mm','functional_position',(len(ids),2),'functional_visual_field')
    positions=numeric(data,d.get('positions_key'),name+' positions')
    if positions.shape!=(len(ids),2) or not np.allclose(positions[order],rf.values,rtol=0,atol=1e-12):fail('COORDINATES',name+' positions do not match owned IDs')
    tag=text(d.get('evidence_tag'),name+' evidence_tag')
    if tag!=port.evidence_tag:fail('EVIDENCE',name+' evidence differs from the existing consumer; metadata cannot upgrade the model')
    mechanisms=object_value(d.get('mechanism_ownership'),name+' mechanism_ownership')
    capture='consumer' if stage=='retinal_illuminance' else 'upstream'
    if mechanisms.get('photopigment_capture')!=capture or mechanisms.get('phototransduction')!='consumer' or mechanisms.get('neural_noise')!='consumer':
        fail('MECHANISM_OWNERSHIP',name+' capture/transduction/neural noise responsibility is inconsistent')
    semantics=d.get('value_semantics')
    if semantics not in ('deterministic','expected','sampled'):fail('NOISE',name+' value semantics must be deterministic, expected or sampled')
    noise=object_value(d.get('photon_noise'),name+' photon_noise')
    if semantics=='sampled':
        if noise.get('owner')!='upstream' or not isinstance(noise.get('model'),str) or noise['model']=='none' or not noise['model'].strip():
            fail('NOISE',name+' sampled input must name upstream noise')
        text(noise.get('realization_id'),name+' realization_id')
    elif noise.get('owner')!='none' or noise.get('model')!='none':fail('NOISE',name+' expected/deterministic input cannot hide an applied noise realization')
    if unit=='R*/window' and semantics not in ('expected','sampled'):fail('COUNT_SEMANTICS',name+' integrated count must be expected or sampled')
    background=object_value(d.get('background'),name+' background')
    if background.get('includes_background') is not True or background.get('unit')!=port.unit:fail('BACKGROUND',name+' requires declared total input in consumer units')
    base=(light.native_background_td if port.unit=='td' else light.rod_background_rstar_s if set(pop.cells.cell_type)=={'rod'} else light.cone_background_rstar_s)
    if number(background.get('value'),name+' background value')!=base or background.get('source_id')!=light.background_source:
        fail('BACKGROUND',name+' background differs from current initialization; no baseline addition or adaptation reset')
    time=object_value(d.get('time'),name+' time')
    if time.get('reference')!='stimulus_relative_seconds':fail('TIME',name+' v1 requires explicitly stimulus-relative seconds')
    start=number(time.get('start_seconds'),name+' start');stop=number(time.get('stop_seconds'),name+' stop')
    source_origin=number(time.get('source_origin_seconds'),name+' source time origin')
    if start!=0 or stop<=start:fail('TIME',name+' must cover a positive interval from stimulus origin zero')
    prehistory=object_value(d.get('prehistory'),name+' prehistory')
    if prehistory.get('kind')!='declared_uniform_background' or number(prehistory.get('stop_seconds'),name+' prehistory boundary')!=start:
        fail('PREHISTORY',name+' must declare its actual uniform-background boundary; no guessed experimental prehistory')
    representation=time.get('representation');interpolation=time.get('interpolation')
    if representation not in ('samples','integration_windows') or interpolation not in ('linear','zero_order_hold'):fail('TIME',name+' explicit supported time representation/interpolation required')
    axis=numeric(data,time.get('axis_key'),name+' time')
    if axis.ndim!=1 or len(axis)<2 or axis[0]!=start or abs(axis[-1]-stop)>ROUND_OFF_SECONDS or np.any(np.diff(axis)<=0):fail('TIME',name+' time axis must increase and match declared boundaries')
    values=numeric(data,d.get('values_key'),name+' values')
    if values.ndim!=2 or values.shape[1]!=len(ids) or np.any(values<0):fail('VALUES',name+' requires nonnegative time-by-cell total input')
    values=values[:,order]
    conversion=None
    if representation=='integration_windows':
        if unit!='R*/window' or interpolation!='zero_order_hold':fail('COUNT_SEMANTICS',name+' windows require count units and held means; linear interpolation changes the integral')
        meaning=('expected' if semantics=='expected' else 'sampled')+'_photoisomerization_count'
        try:rates=photoisomerization_window_mean_rate(Quantity(values,unit,meaning,tag),Quantity(axis,'s','integration_window_edges',tag)).values
        except ContractError as error:fail('COUNT_SEMANTICS',str(error))
        if not np.allclose(rates*np.diff(axis)[:,None],values,rtol=2e-15,atol=0):fail('CONSERVATION',name+' count/rate round-trip failed')
        values=rates
        conversion={'source_unit':unit,'target_unit':'R*/s','meaning':'piecewise_constant_window_mean_not_instantaneous_rate',
            'integration_window_edges_seconds':axis.tolist(),'count_integral_preserved':True}
    elif unit=='R*/window' or len(values)!=len(axis):fail('TIME',name+' samples require one rate/illuminance row per timestamp')
    reference_bridge=None
    if bridge_stage:
        from .isetbio_input import reference_equivalent_drive
        if component.mechanism!='VanHateren2005NativeConeH1' or set(pop.cells.cell_type)!={'cone'}:
            fail('REFERENCE','Reference-equivalent drive is scoped to existing native cone/H1')
        expected_types={v:kind for v,kind in zip(port.cell_ids,pop.cells.subtype)}
        if d.get('capture_types')!=[expected_types[v] for v in ids]:fail('REFERENCE','Capture cone types differ')
        profile=object_value(d.get('reference_profile'),name+' reference_profile')
        if profile.get('pupil_diameter_mm')!=d.get('capture_pupil_diameter_mm') or profile.get('scope')!=d.get('capture_scope'):
            fail('REFERENCE','Capture/reference pupil or scope differs')
        try:values,reference_bridge=reference_equivalent_drive(values,pop.cells.subtype,profile,capture_scope_sha256=d.get('capture_scope_sha256'))
        except ContractError as error:fail('REFERENCE',str(error))
    clock=component.time_contract;step=clock.fixed_step_seconds if clock.kind=='author_discrete' else None
    if step is not None:
        if interpolation!='zero_order_hold' or not np.allclose(axis/step,np.round(axis/step),rtol=0,atol=1e-8):
            fail('CLOCK',name+' author input requires held data and boundaries aligned to its existing tick')
    declared_step=number(time.get('source_grid_step_seconds'),name+' source_grid_step_seconds')
    if declared_step<=0 or not np.allclose(np.diff(axis),declared_step,rtol=0,atol=ROUND_OFF_SECONDS):
        fail('TIME',name+' source grid declaration differs from actual sample/window spacing')
    for array in (axis,values):array.setflags(write=False)
    def evaluate(t):
        t=float(np.clip(t,start,stop));index=min(max(int(np.searchsorted(axis,t,side='right')-1),0),len(values)-1)
        if interpolation=='linear' and index<len(values)-1:
            fraction=(t-axis[index])/(axis[index+1]-axis[index]);result=(1-fraction)*values[index]+fraction*values[index+1]
        else:result=values[index]
        return Quantity(result,port.unit,port.meaning,tag)
    summary={'channel':name,'population_id':pop_id,'consumer':external.target_component+'.'+external.target_port,
        'consumer_mechanism':component.mechanism,'source_unit':unit,'consumer_unit':port.unit,'input_stage':stage,
        'cell_count':len(ids),'cell_ids_hash':canonical_hash(tuple(port.cell_ids)),
        'source_columns_reordered':tuple(ids)!=tuple(port.cell_ids),'coordinate_role':'rf','reference_id':reference_id,'coordinate_registration':registration,
        'start_seconds':start,'stop_seconds':stop,'source_grid_step_seconds':declared_step,'representation':representation,
        'time_axis_end_roundoff_seconds':float(axis[-1]-stop),
        'source_origin_seconds':source_origin,'value_semantics':semantics,'photon_noise':noise,'mechanism_ownership':mechanisms,'background':background,'conversion':conversion,
        'reference_bridge':reference_bridge,
        'prehistory':'Declared uniform background via existing rebase_stimulus_channels; no inferred experimental history',
        'structural_validation':'PASS_NOT_PHYSIOLOGICAL_CALIBRATION'}
    contract=dict(summary,includes_background=True,grid_step_seconds=step or declared_step)
    return InputChannel(start,stop,evaluate,port.unit,port.meaning,interpolation,tag,contract,tuple(float(v) for v in axis[1:-1])),summary

def load_offline_inputs(path,compiled,light_context,*,reference_id,max_bytes=128*1024**2):
    """Return existing InputChannels in consumer ID order, without changing assembly/state."""
    path=Path(path);text(reference_id,'expected reference_id')
    if not path.is_file() or path.suffix.lower()!='.npz':fail('FORMAT','A local NPZ file is required')
    if isinstance(max_bytes,bool) or not isinstance(max_bytes,int) or max_bytes<=0:fail('BUDGET','Positive integer input memory budget required')
    original_stat=path.stat()
    try:
        with zipfile.ZipFile(path) as archive:
            infos=archive.infolist()
            if len({x.filename for x in infos})!=len(infos) or sum(x.file_size for x in infos)>max_bytes:fail('BUDGET','Duplicate members or decompressed input exceeds declared memory budget')
            for info in infos:
                if not info.filename.endswith('.npy'):fail('FORMAT','NPZ must contain only array members')
                with archive.open(info) as stream:
                    version=np.lib.format.read_magic(stream)
                    if version==(1,0):shape,_,dtype=np.lib.format.read_array_header_1_0(stream)
                    elif version==(2,0):shape,_,dtype=np.lib.format.read_array_header_2_0(stream)
                    else:fail('FORMAT','Unsupported NPY header version')
                if dtype.hasobject:fail('FORMAT','Pickled object arrays are forbidden')
                if math.prod(shape)*dtype.itemsize>max_bytes:fail('BUDGET','Declared array allocation exceeds input memory budget')
        with np.load(path,allow_pickle=False) as data:
            if 'manifest_json' not in data:fail('METADATA','manifest_json is required')
            raw=data['manifest_json']
            if raw.shape!=() or raw.dtype.kind!='U' or len(raw.item().encode('utf8'))>1024**2:fail('METADATA','manifest_json must be one bounded Unicode JSON string')
            manifest=json.loads(raw.item(),object_pairs_hook=unique_json,parse_constant=lambda v:fail('METADATA','Nonfinite JSON '+v))
            if not isinstance(manifest,dict) or manifest.get('schema')!=SCHEMA or manifest.get('version')!='1.0':fail('VERSION','Unknown offline receptor input version')
            producer=object_value(manifest.get('producer'),'producer')
            for key in ('name','version','source_id'):text(producer.get(key),'producer '+key)
            if manifest.get('background_context_hash')!=light_context.background_context_hash:fail('BACKGROUND','Input context differs from the declared initialization')
            geometry=_validate_geometry_binding(manifest.get('geometry_identity'),compiled)
            declared=object_value(manifest.get('channels'),'channels');external=compiled.spec.external_inputs
            if len({e.channel for e in external})!=len(external) or set(declared)!={e.channel for e in external}:fail('CHANNELS','File must match the existing assembly external channel set')
            channels={};summaries=[]
            for binding in external:
                channel,summary=_read_channel(data,declared[binding.channel],binding,compiled,light_context,reference_id)
                channels[binding.channel]=channel;summaries.append(summary)
            if len({(c.start_seconds,c.stop_seconds) for c in channels.values()})!=1:fail('TIME','All channels must share explicit input boundaries')
    except OfflineInputError:raise
    except (OSError,ValueError,KeyError,zipfile.BadZipFile,TypeError) as error:fail('FORMAT',type(error).__name__+' in offline NPZ; no guessed fields or pickle fallback')
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    final_stat=path.stat()
    if (original_stat.st_size,original_stat.st_mtime_ns)!=(final_stat.st_size,final_stat.st_mtime_ns):fail('IDENTITY','Input file changed while reading')
    summary={'schema':SCHEMA,'version':'1.0','source_file':str(path.resolve()),'source_sha256':digest,'producer':producer,
        'background_context_hash':light_context.background_context_hash,'channels':summaries,
        'geometry_identity':geometry,'geometry_binding_status':'ACTUAL_TYPED_GEOMETRY_BOUND' if geometry else 'LEGACY_CHANNEL_BINDINGS_ONLY',
        'source_identity_scope':'File/contract identity only; exporter statements are not primary physiological verification',
        'automatic_optics_gain_noise_or_phototransduction':False}
    for channel in channels.values():channel.input_contract.update(source_sha256=digest,producer=producer)
    return OfflineInputs(channels,summary)

def _validate_geometry_binding(declared,compiled):
    if declared is None:
        if getattr(compiled,'input_geometry_binding_required',False):
            fail('GEOMETRY','This geometry requires an actual typed geometry/configuration binding; old packets cannot be reused')
        return None
    from .geometry_identity import require_geometry_identity
    try:return require_geometry_identity(declared,compiled)
    except ContractError as error:fail('GEOMETRY',str(error))

def rebase_offline_inputs(bundle,compiled,light_context,*,stimulus_origin_seconds,grid_origin_seconds):
    """Reuse background rebasing and coalesce only clock-roundoff-equivalent knots."""
    from .background import constant_background_channels,rebase_stimulus_channels
    origin=number(stimulus_origin_seconds,'runner stimulus origin');grid_origin=number(grid_origin_seconds,'runner grid origin')
    if origin<grid_origin:fail('TIME','Stimulus origin precedes existing runner grid')
    if bundle.source_summary['background_context_hash']!=light_context.background_context_hash:fail('BACKGROUND','Rebase background differs from validated input')
    _validate_geometry_binding(bundle.source_summary.get('geometry_identity'),compiled)
    stop=origin+next(iter(bundle.channels.values())).stop_seconds
    requested_stop=stop
    periods=[]
    for component in compiled.components.values():
        if component.time_contract.fixed_step_seconds is not None:periods.append(component.time_contract.fixed_step_seconds)
        observation=getattr(getattr(component,'native_model',None),'observation_interval_seconds',None)
        if observation is not None:periods.append(observation)
    if periods:
        tick=min(periods)
        nearest=grid_origin+round((stop-grid_origin)/tick)*tick
        if abs(nearest-stop)<=ROUND_OFF_SECONDS:stop=nearest
    background=constant_background_channels(compiled,light_context,stop)
    result=rebase_stimulus_channels(bundle.channels,background,original_origin_seconds=0.,stimulus_origin_seconds=origin,stop_seconds=stop)
    if periods:
        for name,channel in result.items():
            knots=[];snapped=0
            for boundary in channel.breakpoints_seconds:
                nearest=grid_origin+round((boundary-grid_origin)/tick)*tick
                if abs(nearest-boundary)<=ROUND_OFF_SECONDS:
                    snapped+=nearest!=boundary;boundary=nearest
                if channel.start_seconds<boundary<channel.stop_seconds:knots.append(float(boundary))
            result[name]=replace(channel,breakpoints_seconds=tuple(sorted(set(knots))),
                input_contract=dict(channel.input_contract,clock_roundoff_snapped_knots=snapped,
                    clock_roundoff_end_adjustment_seconds=stop-requested_stop,
                    integration_windows_unchanged=True,existing_runner_grid_origin_seconds=grid_origin))
    return result
