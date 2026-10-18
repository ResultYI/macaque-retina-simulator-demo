"""ISET capture on fixed global optical tiles and one shared LMS reference."""
from pathlib import Path
import hashlib,json,math,os,subprocess,time,uuid
import numpy as np
from scipy.io import savemat,loadmat,whosmat
from scipy.ndimage import map_coordinates
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.disk_graph_storage import file_sha,save_json
from retina_sim.disk_input import join_captures
from retina_sim.geometry_identity import compiled_geometry_identity
from retina_sim.isetbio_input import REFERENCE_SCHEMA,reference_equivalent_drive
from retina_sim.offline_input import SCHEMA,load_offline_inputs
from .iset_frontend import MixedReferenceFrontend,source_signature,PROFILE,BUDGET,digest

def read_disk_capture(path,signature):
    path=Path(path)
    required={'manifest_json','expected_counts','background_counts','q_ref_LMS','window_edges_seconds','diameters_um','fixed_radiance_gain_numeric'}
    if path.stat().st_size>BUDGET:raise ContractError('Global tile MAT exceeds existing128MiB capture budget')
    info=whosmat(path)
    if {name for name,_,_ in info}!=required or any(kind not in ('double','char') for _,_,kind in info):raise ContractError('Unknown global tile MAT variables/types')
    if sum(math.prod(shape)*8 for _,shape,_ in info)>BUDGET:raise ContractError('Global tile declared arrays exceed capture budget')
    raw=loadmat(path,mat_dtype=True,squeeze_me=False,chars_as_strings=True);meta=json.loads(raw['manifest_json'].item())
    if meta.get('schema')!='approved_mixed_capture_v1' or meta.get('request_signature')!=signature:raise ContractError('Global tile request identity differs')
    for name in required-{'manifest_json'}:
        if raw[name].dtype.kind!='f' or not np.all(np.isfinite(raw[name])):raise ContractError('Nonfinite global tile '+name)
    if np.any(raw['expected_counts']<0) or np.any(raw['background_counts']<=0) or np.any(raw['q_ref_LMS']<=0):raise ContractError('Invalid global tile capture/reference counts')
    meta['fixed_radiance_gain']=float(raw['fixed_radiance_gain_numeric'].item())
    if meta['fixed_radiance_gain']<=0 or np.any(raw['diameters_um']<=0):raise ContractError('Positive frozen global gain/apertures required')
    return raw,meta

def _run_batch(root,profile,jobs,directory,on_status,on_boundary):
    if not jobs:return
    directory=Path(directory);batch=directory/('batch_'+uuid.uuid4().hex+'.json');save_json(batch,jobs)
    quote=lambda path:str(Path(path).resolve()).replace("'","''")
    expression="addpath('"+quote(root/'tools')+"');export_disk_mixed_batch('"+quote(batch)+"');"
    pref=directory/'matlab_preferences';temporary=directory/'matlab_temp';pref.mkdir(exist_ok=True);temporary.mkdir(exist_ok=True)
    environment=dict(os.environ,MATLAB_PREFDIR=str(pref),TEMP=str(temporary),TMP=str(temporary))
    with (directory/('stdout_'+batch.stem+'.log')).open('wb') as out,(directory/('stderr_'+batch.stem+'.log')).open('wb') as err:
        process=subprocess.Popen([profile['matlab_executable'],'-batch',expression],env=environment,stdout=out,stderr=err,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        last_progress=time.monotonic();last_status=last_progress;done=0
        while process.poll() is None:
            if on_boundary is not None and on_boundary() is False:
                process.terminate();process.wait();raise ContractError('Global ISET capture stopped at requested boundary; completed tile files retained')
            completed=sum(Path(job['output']+'.ready').is_file() for job in jobs)
            if completed>done:done=completed;last_progress=time.monotonic()
            if time.monotonic()-last_progress>profile['maximum_export_wall_seconds']:
                process.terminate();process.wait();raise ContractError('Global ISET tile exceeded original per-export wall budget; completed tiles retained')
            if time.monotonic()-last_status>=15:
                if on_status:on_status('ISET fixed global tiles %d/%d'%(done,len(jobs)))
                last_status=time.monotonic()
            time.sleep(.2)
        if process.returncode:raise ContractError('Pinned global MATLAB/ISET capture failed; no RGB cone fallback: '+str(directory))

def _capture_request(root,profile,directory,data,export,*,on_status=None,on_boundary=None,defer_jobs=None):
    """One create-once cache entry; completed pending exports are recoverable."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    capture=directory/'capture.mat';binding=directory/'binding.json';signature=export['request_signature']
    if capture.exists():
        if not binding.exists():raise ContractError('Unregistered global capture retained for inspection')
        recorded=json.loads(binding.read_bytes())
        if recorded['capture_sha256']!=file_sha(capture) or recorded['input_mat_sha256']!=file_sha(directory/'input.mat'):
            raise ContractError('Global capture cache/input identity differs; retained')
        if 'export_sha256' in recorded and recorded['export_sha256']!=file_sha(directory/'export.json'):
            raise ContractError('Global cached export configuration changed')
        return read_disk_capture(capture,signature),False
    for marker in directory.glob('capture_pending_*.mat.ready'):
        pending=Path(str(marker)[:-6])
        raw,meta=read_disk_capture(pending,signature)
        os.replace(pending,capture);save_json(binding,{'capture_sha256':file_sha(capture),'request_signature':signature,
            'input_mat_sha256':file_sha(directory/'input.mat'),'recovered_completed_pending_capture':True})
        return (raw,meta),False
    source=directory/'input.mat'
    if source.exists():
        # Compare actual arrays and keep the existing create-once input bytes.
        previous=loadmat(source,squeeze_me=False)
        if any(key not in previous or not np.array_equal(np.asarray(value),previous[key].reshape(np.asarray(value).shape)) for key,value in data.items()):
            raise ContractError('Existing global tile input data differs; retained')
    else:savemat(source,data,do_compression=True)
    export=dict(export,input_mat=str(source.resolve()));config=directory/'export.json'
    if config.exists():
        if json.loads(config.read_bytes())!=export:raise ContractError('Global export configuration differs; retained')
    else:save_json(config,export)
    pending=directory/('capture_pending_'+uuid.uuid4().hex+'.mat')
    job={'configuration':str(config.resolve()),'output':str(pending.resolve())}
    if defer_jobs is not None:
        defer_jobs.append(job)
        return (None,{'pending_path':str(pending),'directory':str(directory),'signature':signature}),True
    _run_batch(root,profile,[job],directory,on_status,on_boundary)
    return _finish_capture(directory,pending,signature),True

def _finish_capture(directory,pending,signature):
    directory=Path(directory);pending=Path(pending);capture=directory/'capture.mat'
    raw,meta=read_disk_capture(pending,signature);os.replace(pending,capture)
    save_json(directory/'binding.json',{'capture_sha256':file_sha(capture),'request_signature':signature,
        'input_mat_sha256':file_sha(directory/'input.mat'),'export_sha256':file_sha(directory/'export.json')})
    return raw,meta

class DiskMixedReferenceFrontend(MixedReferenceFrontend):
    def __init__(self,config,compiled,media,gaze,root,*,input_path,on_status=None,on_boundary=None):
        self.config=config;self.compiled=compiled;self.root=Path(root).resolve();self.input_path=Path(input_path).resolve()
        if self.input_path.exists():raise FileExistsError('Run input snapshot already exists')
        circuit=compiled.global_disk_circuit;view=compiled.global_disk_view
        from retina_sim.atlas_population_network import circuit_configuration
        configuration=circuit_configuration(self.root,config.circuit_configuration_path)
        binding=configuration['input_plan'];path=self.root/binding['path']
        if file_sha(path)!=binding['sha256']:raise ContractError('Global optical plan identity differs')
        plan=json.loads(path.read_bytes());cone=circuit.table('cones')
        if plan['cone_table']!=cone.binding or plan['regenerate_on_ROI'] is not False:raise ContractError('Global optical plan belongs to different cone geometry')
        if plan.get('exporter_sha256')!=file_sha(self.root/'tools/export_disk_mixed_reference.m'):
            raise ContractError('Frozen optical plan exporter identity differs; create an explicit input-plan revision')
        profile,sources=source_signature(self.root)
        for relative in ('tools/export_disk_mixed_reference.m','tools/export_disk_mixed_batch.m','retina_sim/disk_input.py','retina_platform/disk_frontend.py'):
            sources[str((self.root/relative).resolve())]=file_sha(self.root/relative)
        reference=dict(plan['reference']);scope=canonical_hash({'plan':binding,'profile':profile,'sources':sources,
            'reference_linear_u':config.reference_linear_value,'fixed_reference_definition':reference})
        reference['plan_definition_sha256']=reference['definition_sha256'];reference['definition_sha256']=scope
        u=config.reference_linear_value;digital=12.92*u if u<=.0031308 else 1.055*u**(1/2.4)-.055
        color=np.full(3,digital);focal=.223e-3/np.tan(np.pi/180)
        def fov(samples):return float(np.rad2deg(2*np.arctan(samples*.001e-3/(2*focal))))
        self.reference_directory=self.root/'reports/iset_disk_input_cache'/scope/'reference'
        data={'rgb_frames':np.full((193,193,3,1),digital,dtype=np.float32),'window_edges_seconds':np.array([0.,.0001]),
            'frame_indices':np.array([1.]),'reference_rgb':color,'is_mm':np.zeros((3,2)),'types':np.array([1.,2.,3.]),
            'frozen_apertures_um':np.full(3,reference['reference_diameter_um']),'image_center_deg':np.zeros(2)}
        export={'profile':profile,'global_reference':reference,'patch_fov_deg':fov(193),'request_signature':scope,'background_signature':scope}
        (refraw,refmeta),ref_actual=_capture_request(self.root,profile,self.reference_directory,data,export,on_status=on_status,on_boundary=on_boundary)
        if refmeta.get('global_reference_definition_sha256')!=scope:raise ContractError('Captured global LMS reference belongs to a different source scope')
        reference.update(fixed_radiance_gain=refmeta['fixed_radiance_gain'],q_ref_LMS_rstar_s=refraw['q_ref_LMS'].reshape(-1).tolist(),
            reference_energy_spectrum_hex=refmeta['reference_energy_spectrum_hex'])
        ticks=int(round(config.duration_seconds/.0001));edges=np.arange(ticks+1)*.0001
        active=np.asarray(view['selected_rows']['cones']);pop=compiled.populations['cones'];ids=list(pop.cells.cell_id);types=list(pop.cells.subtype)
        if ticks<1 or (ticks+1)*(len(active)+len(compiled.populations['rod'].cells.cell_id))*8>BUDGET:raise ContractError('Global active capture exceeds128MiB time/cell budget')
        poses=np.asarray([gaze(float(t)) for t in edges[:-1]])
        media_identity=file_sha(config.media_path);request=canonical_hash({'reference_scope':scope,'media':media_identity,
            'start':config.start_seconds,'duration':config.duration_seconds,'field_width_deg':config.field_width_deg,'gaze':poses})
        assembled=canonical_hash({'global_capture':request,'active_cone_rows':active,'geometry':compiled_geometry_identity(compiled)})
        self.directory=self.root/'reports/iset_disk_input_cache'/assembled;self.directory.mkdir(parents=True,exist_ok=True)
        captures=[];rows=[];tile_manifest=[];actual=ref_actual;lookup={int(row):i for i,row in enumerate(active)};jobs=[];tasks=[]
        width,height=media.info['width'],media.info['height'];ppd=width/config.field_width_deg
        for tile in plan['tiles']:
            row_path=self.root/tile['global_rows']['path']
            if file_sha(row_path)!=tile['global_rows']['sha256']:raise ContractError('Fixed optical tile ownership changed')
            global_rows=np.load(row_path,allow_pickle=False);kept=np.array([i for i,row in enumerate(global_rows) if int(row) in lookup],dtype=np.int64)
            if not len(kept):continue
            samples=tile['scene_samples'];center=np.asarray(tile['center_mm']);axis=np.arange(samples)-(samples-1)/2
            xx,yy=np.meshgrid((axis*.001+center[0])/.223,(-axis*.001+center[1])/.223)
            frames=[];mapping={};indices=[]
            for t,pose in zip(edges[:-1],poses):
                media_time=config.start_seconds+float(t)
                frame_index=0 if media.info['kind']=='image' else min(int(media_time*media.info['fps']+1e-8),media.info['frame_count']-1)
                key=(frame_index,tuple(pose))
                if key not in mapping:
                    rgb=media.frame_at(media_time).astype(float)/255.;coords=[height/2+ppd*(yy+pose[1]),width/2+ppd*(xx+pose[0])]
                    frames.append(np.stack([map_coordinates(rgb[:,:,k],coords,order=1,mode='constant',cval=digital,prefilter=False) for k in range(3)],axis=2).astype(np.float32))
                    mapping[key]=len(frames)
                    if sum(frame.nbytes for frame in frames)>BUDGET:raise ContractError('Fixed global tile RGB frames exceed128MiB budget')
                indices.append(mapping[key])
            tile_sig=canonical_hash({'global_capture':request,'tile':tile,'reference_scope':scope})
            directory=self.root/'reports/iset_disk_input_cache'/scope/'tiles'/tile_sig
            data={'rgb_frames':np.stack(frames,axis=3),'window_edges_seconds':edges,'frame_indices':np.asarray(indices,dtype=float),
                'reference_rgb':color,'is_mm':cone.coordinates['rf'][global_rows],'types':np.array([{'L':1.,'M':2.,'S':3.}[x.decode('ascii')] for x in cone.extras['subtype'][global_rows]]),
                'frozen_apertures_um':cone.extras['aperture_diameter_um'][global_rows],'image_center_deg':center/.223}
            export={'profile':profile,'global_reference':reference,'patch_fov_deg':fov(samples),'request_signature':tile_sig,'background_signature':scope}
            (raw,meta),called=_capture_request(self.root,profile,directory,data,export,on_status=on_status,on_boundary=on_boundary,defer_jobs=jobs)
            actual|=called
            tasks.append((raw,meta,kept,np.asarray([lookup[int(global_rows[i])] for i in kept],dtype=np.int64),directory,tile['tile_index']))
            if on_status:on_status('ISET fixed global tiles prepared %d'%len(tasks))
        _run_batch(self.root,profile,jobs,self.directory,on_status,on_boundary)
        for raw,meta,kept,local_rows,directory,tile_index in tasks:
            if raw is None:raw,meta=_finish_capture(directory,meta['pending_path'],meta['signature'])
            selected_raw=dict(raw,expected_counts=raw['expected_counts'][:,kept],background_counts=raw['background_counts'].reshape(-1)[kept],
                diameters_um=raw['diameters_um'].reshape(-1)[kept])
            captures.append((selected_raw,meta));rows.append(local_rows)
            tile_manifest.append({'tile_index':tile_index,'capture_sha256':file_sha(directory/'capture.mat'),'directory':str(directory)})
            if on_status:on_status('ISET global tiles captured %d'%len(captures))
        joined=join_captures(rows,captures,len(active),reference_identity=scope)
        raw={'expected_counts':joined['expected_counts'],'background_counts':joined['background_counts'][None,:],
            'q_ref_LMS':joined['q_ref_LMS'][None,:],'window_edges_seconds':edges[None,:],
            'diameters_um':joined['diameters_um'][None,:],'fixed_radiance_gain_numeric':np.array([[joined['fixed_radiance_gain']]])}
        meta=dict(refmeta,request_signature=assembled,background_signature=scope,global_reference_definition_sha256=scope,
            tile_captures=tile_manifest,finite_optical_support='fixed canonical40um optical halos; no infinite-PSF/global calibration claim',
            reference_scene_directory=str(self.reference_directory))
        capture=self.directory/'capture.mat'
        if capture.exists():read_disk_capture(capture,assembled)
        else:savemat(capture,dict(raw,manifest_json=json.dumps(meta)),do_compression=True)
        self.geometry_identity=compiled_geometry_identity(compiled);self.cache_reused=not actual
        self._bind(config,compiled,media,gaze,pop,ids,types,circuit.record['reference_id'],raw,meta,edges,assembled,scope,sources)

    def _bind(self,config,compiled,media,gaze,pop,ids,types,ref,raw,meta,edges,request,background_signature,source_hashes):
        from retina_sim.disk_input_packet import write_global_input,load_global_input
        from .backend import _MediaChannels
        geometry=compiled_geometry_identity(compiled)
        if geometry!=self.geometry_identity:raise ContractError('Global compiled geometry changed during ISET capture')
        profile={'schema':REFERENCE_SCHEMA,'profile_id':PROFILE+'_global_disk_fixed_LMS_reference','source_id':PROFILE,
            'scope':'approved_mixed_reference','evidence_tag':'project_approximation',
            'reference_conditions':'One fixed foveal LMS triplet, complete-network mean aperture/blur and LCD radiance gain; no per-tile or ROI refit',
            'capture_scope_sha256':background_signature,'reference_scene_sha256':digest(self.reference_directory/'input.mat'),
            'reference_capture_sha256':digest(self.reference_directory/'capture.mat'),'reference_luminance_cd_m2':meta['reference_luminance_cd_m2'],
            'pupil_diameter_mm':3.,'reference_rates_rstar_s_LMS':dict(zip(('L','M','S'),raw['q_ref_LMS'].reshape(-1).tolist()))}
        background,bridge=reference_equivalent_drive(raw['background_counts']/.0001,types,profile,capture_scope_sha256=background_signature)
        config.frontend_signature=canonical_hash({'global_reference':background_signature,'reference':profile['reference_rates_rstar_s_LMS'],
            'background_values':background,'fixed_gain':meta['fixed_radiance_gain']})
        config.native_background_values=tuple(background[0]);light=config.light_context()
        rod_source=_MediaChannels(config,compiled,media,gaze,0.);rod_channels=rod_source.window(0.,config.duration_seconds)
        circuit=compiled.global_disk_circuit;view=compiled.global_disk_view
        arrays={'native_xy':pop.receptive_field_positions.values,'native_counts':raw['expected_counts'],
            'native_bgcounts':raw['background_counts'],'native_rows':view['selected_rows']['cones'],
            'native_types':np.asarray(types,dtype='S1'),'edges':edges};declarations={}
        for external in compiled.spec.external_inputs:
            port=compiled.components[external.target_component].inputs[external.target_port];name=external.channel;native=name=='native_light'
            pid='cones' if native else 'rod';global_name='cones' if native else 'rods';owned=compiled.populations[pid]
            if not native:
                arrays[name+'_values']=np.stack([rod_channels[name].at(float(t)).values for t in edges])
                arrays[name+'_xy']=owned.cells.rf_position.values;arrays[name+'_rows']=view['selected_rows'][global_name]
            declared={'population_id':pid,'global_population':global_name,'population_binding':circuit.record['tables'][global_name]['binding'],
                'global_rows_key':'native_rows' if native else name+'_rows','data_axes':['time','cell'],
                'coordinate_role':'rf','coordinate_unit':'mm','coordinate_frame':'functional_visual_field','reference_id':ref,
                'coordinate_registration':'frozen global RF=IS project chart' if native else 'existing nominal rod retinal/RF chart; no cone Henle transfer',
                'positions_key':'native_xy' if native else name+'_xy','values_key':'native_counts' if native else name+'_values',
                'unit':'R*/window' if native else 'R*/s','input_stage':'reference_equivalent_cone_drive' if native else 'photoisomerization',
                'evidence_tag':port.evidence_tag,'value_semantics':'expected' if native else 'deterministic',
                'photon_noise':{'owner':'none','model':'none'},'mechanism_ownership':{'photopigment_capture':'upstream','phototransduction':'consumer','neural_noise':'consumer'},
                'background':{'includes_background':True,'unit':port.unit,'value':100. if native else config.rod_background_rstar_s,'source_id':light.background_source},
                'prehistory':{'kind':'declared_uniform_background','stop_seconds':0.,'spatial_uniformity':'uniform reference scene; actual per-cell captures retained'},
                'time':{'representation':'integration_windows' if native else 'samples','reference':'stimulus_relative_seconds','source_origin_seconds':0.,
                    'start_seconds':0.,'stop_seconds':config.duration_seconds,'axis_key':'edges','interpolation':'zero_order_hold','source_grid_step_seconds':.0001}}
            if native:declared.update(reference_profile=profile,capture_scope_sha256=background_signature,capture_types_key='native_types',
                background_counts_key='native_bgcounts',capture_pupil_diameter_mm=3.,capture_scope='approved_mixed_reference')
            declarations[name]=declared
        producer={'name':'Fixed global LCD-Apple/MW/ISETBio frontend','version':'1.0','source_id':PROFILE,
            'capture_sha256':digest(self.directory/'capture.mat'),'request_signature':request,'frontend_signature':config.frontend_signature,
            'capture':meta,'fixed_reference_bridge':bridge,'source_hashes':source_hashes,
            'rod_input_scope':'existing effective RGB-weighted project rod drive; no ISET rod capture or physiological calibration claim',
            'rod_input':rod_source.last_input_summary}
        write_global_input(self.input_path,compiled,light,reference_id=ref,arrays=arrays,declarations=declarations,producer=producer)
        self.bundle=load_global_input(self.input_path,compiled,light,reference_id=ref)
        self.summary=dict(self.bundle.source_summary,cache_directory=str(self.directory),actual_ISET_call=not self.cache_reused,
            cache_reused=self.cache_reused,capture_producer='real pinned ISET APIs on fixed global optical tiles',
            global_reference_origin_mm=[0.,0.],global_graph=circuit.binding,frontend_signature=config.frontend_signature,
            background_drive_range_reference_td=[float(background.min()),float(background.max())])
        self.last_input_summary=self.summary;self.channels=None
