"""Approved LCD-Apple/MW production input; an ISET failure never selects RGB cones."""
from pathlib import Path
import hashlib,json,math,os,subprocess,time,uuid
import numpy as np
from scipy.io import savemat,loadmat,whosmat
from scipy.ndimage import map_coordinates
from retina_sim.contracts import canonical_hash,ContractError
from retina_sim.isetbio_input import REFERENCE_SCHEMA,reference_equivalent_drive,project_rf_equals_is_binding
from retina_sim.offline_input import SCHEMA,load_offline_inputs,rebase_offline_inputs
from retina_sim.geometry_identity import compiled_geometry_identity

PROFILE='iset_lcd_apple_mw_v1'
BUDGET=128*1024**2

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def source_signature(root):
    root=Path(root);path=root/'configurations'/ (PROFILE+'.json');profile=json.loads(path.read_bytes())
    if profile.get('profile_id')!=PROFILE or profile.get('scope')!='approved_mixed_reference':raise ContractError('Invalid approved ISET profile')
    files=[path,root/'tools/export_mixed_reference.m',Path(__file__)]
    for key in ('isetcam','isetbio'):
        folder=Path(profile[key+'_root']);head=(folder/'.git/HEAD').read_text().strip()
        if head.startswith('ref: '):head=(folder/'.git'/head[5:]).read_text().strip()
        if head!=profile[key+'_commit']:raise ContractError('Pinned '+key+' version differs; no fallback')
    cam=Path(profile['isetcam_root']);bio=Path(profile['isetbio_root'])
    files.extend([cam/'data/displays/LCD-Apple.mat',cam/'displays/displayCreate.m',cam/'displays/displayGet.m',
        cam/'opticalimage/oiCreate.m',cam/'opticalimage/oiCompute.m',cam/'opticalimage/optics/opticsCreate.m',
        bio/'cones/@cPhotoPigment/cPhotoPigment.m',bio/'cones/@Macular/Macular.m',bio/'opticalimage/@Lens/Lens.m'])
    files.extend((bio/'cones/cmosaic/@cMosaic').glob('*.m'))
    files.extend((bio/'ganglioncells/+RGCmodels/+Watson/+convert').glob('positionsToSpacings.m'))
    for name in ('T_log10coneabsorbance_ss.mat','macularPigment.mat','lensPokornySmith.mat'):files.extend(bio.rglob(name))
    return profile,{str(p.resolve()):digest(p) for p in files}

def read_capture(path,request):
    path=Path(path)
    if path.stat().st_size>BUDGET:raise ContractError('ISET capture exceeds 128 MiB budget')
    shapes=whosmat(path);required={'manifest_json','expected_counts','background_counts','q_ref_LMS','window_edges_seconds','diameters_um'}
    if {name for name,_,_ in shapes}!=required or any(kind not in ('double','char') for _,_,kind in shapes):
        raise ContractError('ISET MAT variables/types differ from the production contract')
    if sum(math.prod(s)*8 for _,s,_ in shapes)>BUDGET:raise ContractError('ISET declared arrays exceed memory budget')
    raw=loadmat(path,mat_dtype=True,squeeze_me=False,chars_as_strings=True)
    meta=json.loads(raw['manifest_json'].item())
    if meta.get('schema')!='approved_mixed_capture_v1' or meta.get('request_signature')!=request:
        raise ContractError('ISET cache request identity mismatch')
    for name in required-{'manifest_json'}:
        if raw[name].dtype.kind!='f' or not np.all(np.isfinite(raw[name])):raise ContractError('Nonfinite ISET '+name)
    if np.any(raw['expected_counts']<0) or np.any(raw['background_counts']<=0) or np.any(raw['q_ref_LMS']<=0):
        raise ContractError('Invalid ISET capture/reference rates')
    return raw,meta

def run_matlab(profile,root,config,output,on_status,on_boundary):
    executable=Path(profile['matlab_executable'])
    if not executable.is_file():raise ContractError('缺少配置的 MATLAB；不会回退 RGB cone 输入')
    quote=lambda p:str(Path(p).resolve()).replace("'","''")
    expression="addpath('"+quote(root/'tools')+"');export_mixed_reference('"+quote(config)+"','"+quote(output)+"');"
    # Keep launch-time writes inside this capture's workspace, including when
    # the desktop user's roaming settings directory is read-only in a sandbox.
    preferences=config.parent/'matlab_preferences';temporary=config.parent/'matlab_temp'
    preferences.mkdir(exist_ok=True);temporary.mkdir(exist_ok=True)
    environment=dict(os.environ,MATLAB_PREFDIR=str(preferences.resolve()),TEMP=str(temporary.resolve()),TMP=str(temporary.resolve()))
    with (config.parent/'matlab_stdout.log').open('wb') as stdout,(config.parent/'matlab_stderr.log').open('wb') as stderr:
        process=subprocess.Popen([str(executable),'-batch',expression],stdout=stdout,stderr=stderr,
            env=environment,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        started=time.monotonic();last_status=started
        while process.poll() is None:
            if on_boundary is not None and on_boundary() is False:
                process.terminate();process.wait();raise ContractError('ISET 输入导出已停止')
            if time.monotonic()-started>profile['maximum_export_wall_seconds']:
                process.terminate();process.wait();raise ContractError('ISET 输入导出达到明确时间预算；不会回退 RGB')
            if time.monotonic()-last_status>=15:
                if on_status:on_status('MATLAB/ISET 导出中')
                last_status=time.monotonic()
            time.sleep(.2)
    if process.returncode or not output.is_file():
        raise ContractError('MATLAB/ISET 输入失败；无 RGB fallback。详见 '+str(config.parent/'matlab_stdout.log'))

class MixedReferenceFrontend:
    def __init__(self,config,compiled,media,gaze,root,*,input_path,on_status=None,on_boundary=None):
        self.config=config;self.compiled=compiled;self.root=Path(root);self.input_path=Path(input_path).resolve()
        if self.input_path.exists():raise FileExistsError('Run input snapshot already exists: '+str(self.input_path))
        profile,source_hashes=source_signature(self.root)
        from retina_sim.atlas_population_network import load_population_record
        record,graph_binding=load_population_record(self.root,owner_bounds_mm=config.owner_bounds(),halo_mm=config.halo_mm,
            execution_profile='atlas_isetbio_geometry_v1' if config.execution_profile=='atlas_isetbio_geometry_v1' else 'atlas_population_revision_v1',
            configuration_path=config.circuit_configuration_path or None)
        pop=compiled.populations['cones'];ids=list(pop.cells.cell_id);types=list(pop.cells.subtype)
        ref=record['atlas']['reference_id'];is_binding=project_rf_equals_is_binding(pop,reference_id=ref)
        from retina_sim.isetbio_geometry import input_geometry_support
        positions=is_binding['positions_mm'];support_ids,support_xy=input_geometry_support(self.root,record)
        index={gid:i for i,gid in enumerate(support_ids)};active=np.array([index[x] for x in ids],dtype=int)
        if not np.array_equal(support_xy[active],positions):raise ContractError('Frozen full-support geometry differs from active cones')
        self.geometry_identity=compiled_geometry_identity(compiled)
        background_signature=canonical_hash({'profile':profile,'sources':source_hashes,'graph':graph_binding['sha256'],
            'compiled_geometry_identity':self.geometry_identity,
            'cell_ids':ids,'types':types,'IS_mm':positions,'support_ids':support_ids,'support_mm':support_xy,
            'RF_IS_transfer':is_binding['source_sha256'],'halo_mm':config.halo_mm,'reference_u':config.reference_linear_value})
        ticks=int(round(config.duration_seconds/.0001));edges=np.arange(ticks+1,dtype=float)*.0001
        if ticks<1 or (ticks+1)*(len(ids)+len(compiled.populations['rod'].cells.cell_id))*8>BUDGET:
            raise ContractError('ISET 离线输入超过128 MiB；请缩短记录时长，不会减少细胞')
        poses=np.array([gaze(float(t)) for t in edges[:-1]])
        request=canonical_hash({'background_signature':background_signature,'media_sha256':digest(config.media_path),
            'start_seconds':config.start_seconds,'duration_seconds':config.duration_seconds,
            'field_width_deg':config.field_width_deg,'gaze':poses,'seed':config.seed})
        directory=self.root/'reports/iset_input_cache'/request;directory.mkdir(parents=True,exist_ok=True)
        self.directory=directory;capture=directory/'capture.mat';binding_file=directory/'binding.json'
        self.cache_reused=capture.exists()
        if capture.exists():
            if not binding_file.is_file() or json.loads(binding_file.read_bytes()).get('capture_sha256')!=digest(capture):
                raise ContractError('ISET 缓存身份不匹配；保留文件供检查，不会回退 RGB')
        else:
            self._export(config,media,profile,directory,positions,support_xy,active,types,poses,edges,
                request,background_signature,on_status,on_boundary)
        raw,meta=read_capture(capture,request)
        if meta.get('background_signature')!=background_signature:raise ContractError('ISET background identity differs')
        self._bind(config,compiled,media,gaze,pop,ids,types,ref,raw,meta,edges,request,background_signature,source_hashes)

    def _export(self,config,media,profile,directory,positions,support_xy,active,types,poses,edges,
            request,background_signature,on_status,on_boundary):
        center=(positions.min(axis=0)+positions.max(axis=0))/2
        extent=(positions.max(axis=0)-positions.min(axis=0))+2*config.halo_mm
        cols=int(np.ceil(extent[0]*1e3/profile['scene_sampling_um']))+1
        rows=int(np.ceil(extent[1]*1e3/profile['scene_sampling_um']))+1
        if max(rows,cols)>profile['maximum_scene_samples_per_axis']:raise ContractError('ISET 光学ROI超过明确网格预算')
        pitch=max(extent[0]/(cols-1),extent[1]/(rows-1));patch_width=cols*pitch
        patch_fov=np.rad2deg(2*np.arctan((patch_width*1e-3)/(2*(.223e-3/np.tan(np.pi/180)))))
        x=(np.arange(cols)-(cols-1)/2)*pitch+center[0]
        y=((rows-1)/2-np.arange(rows))*pitch+center[1]
        xx,yy=np.meshgrid(x/.223,y/.223)
        u=config.reference_linear_value
        digital_ref=12.92*u if u<=.0031308 else 1.055*u**(1/2.4)-.055
        reference_rgb=np.full(3,digital_ref)
        frames=[];frame_map={};indices=[];timestamps=[];frame_bytes=0
        width,height=media.info['width'],media.info['height'];ppd=width/config.field_width_deg
        for t,pose in zip(edges[:-1],poses):
            media_time=config.start_seconds+float(t)
            frame_index=0 if media.info['kind']=='image' else min(int(media_time*media.info['fps']+1e-8),media.info['frame_count']-1)
            key=(frame_index,tuple(pose))
            if key not in frame_map:
                rgb=media.frame_at(media_time).astype(float)/255.
                coords=[height/2+ppd*(yy+pose[1]),width/2+ppd*(xx+pose[0])]
                patch=np.stack([map_coordinates(rgb[:,:,k],coords,order=1,mode='constant',cval=digital_ref,prefilter=False) for k in range(3)],axis=2)
                frames.append(patch.astype(np.float32));frame_map[key]=len(frames);frame_bytes+=frames[-1].nbytes
                if frame_bytes>BUDGET:raise ContractError('ISET RGB帧缓存超过128 MiB；请缩短记录时长')
                timestamps.append({'frame_index':frame_index,'requested_media_seconds':media_time,
                    'decoded_timestamp_seconds':media.last_timestamp_seconds,'gaze_deg':pose.tolist()})
            indices.append(frame_map[key])
        input_mat=directory/'input.mat'
        savemat(input_mat,{'rgb_frames':np.stack(frames,axis=3),'window_edges_seconds':edges,
            'frame_indices':np.array(indices,dtype=float),'reference_rgb':reference_rgb,
            'support_is_mm':support_xy,'active_indices':active.astype(float)+1,'is_mm':positions,
            'types':np.array([{'L':1,'M':2,'S':3}[x] for x in types],dtype=float),'image_center_deg':center/.223},do_compression=True)
        export={'profile':profile,'input_mat':str(input_mat.resolve()),'patch_fov_deg':float(patch_fov),
            'request_signature':request,'background_signature':background_signature}
        config_path=directory/'export.json';config_path.write_text(json.dumps(export,allow_nan=False),encoding='utf8')
        pending=directory/('capture_pending_'+uuid.uuid4().hex+'.mat')
        run_matlab(profile,self.root,config_path,pending,on_status,on_boundary)
        read_capture(pending,request)
        binding={'capture_sha256':digest(pending),'input_mat_sha256':digest(input_mat),
            'background_signature':background_signature,'profile':export['profile'],
            'frame_timestamps':timestamps,'media_timestamp_fallback_used':media.info.get('timestamp_fallback_used',False)}
        binding_path=directory/'binding.json';temporary=binding_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(binding,allow_nan=False),encoding='utf8');os.replace(temporary,binding_path)
        os.replace(pending,directory/'capture.mat')

    def _bind(self,config,compiled,media,gaze,pop,ids,types,ref,raw,meta,edges,request,background_signature,source_hashes):
        geometry=compiled_geometry_identity(compiled)
        if hasattr(self,'geometry_identity') and self.geometry_identity!=geometry:
            raise ContractError('Actual typed geometry/configuration changed while capturing input; no stale capture binding')
        counts=raw['expected_counts'];bgcounts=raw['background_counts'];ticks=len(edges)-1
        if counts.shape!=(ticks,len(ids)) or bgcounts.shape!=(1,len(ids)) or not np.array_equal(raw['window_edges_seconds'].reshape(-1),edges):
            raise ContractError('ISET capture time/cell dimensions differ from frozen request')
        reference_profile={'schema':REFERENCE_SCHEMA,'profile_id':PROFILE+'_fixed_LMS_reference','source_id':PROFILE,
            'scope':'approved_mixed_reference','evidence_tag':'project_approximation',
            'reference_conditions':'Existing reference RGB; same LCD/MW, mean aperture and standard capture at fixed patch-center LMS reference cones',
            'capture_scope_sha256':background_signature,'reference_scene_sha256':digest(self.directory/'input.mat'),
            'reference_capture_sha256':digest(self.directory/'capture.mat'),'reference_luminance_cd_m2':meta['reference_luminance_cd_m2'],
            'pupil_diameter_mm':3.,'reference_rates_rstar_s_LMS':dict(zip(('L','M','S'),raw['q_ref_LMS'].reshape(-1).tolist()))}
        background_values,bridge=reference_equivalent_drive(bgcounts/.0001,types,reference_profile,capture_scope_sha256=background_signature)
        config.frontend_signature=canonical_hash({'background_signature':background_signature,
            'reference':reference_profile['reference_rates_rstar_s_LMS'],'background_values':background_values,'fixed_gain':meta['fixed_radiance_gain']})
        config.native_background_values=tuple(background_values[0]);light=config.light_context()
        from .backend import _MediaChannels
        rods=_MediaChannels(config,compiled,media,gaze,0.).window(0.,config.duration_seconds)
        arrays={'native_xy':pop.receptive_field_positions.values,'native_counts':counts,'edges':edges};declarations={}
        for external in compiled.spec.external_inputs:
            port=compiled.components[external.target_component].inputs[external.target_port]
            pid=next(k for k,v in compiled.populations.items() if tuple(v.cells.cell_id)==tuple(port.cell_ids));owned=compiled.populations[pid]
            native=external.channel=='native_light';key=external.channel
            if not native:
                if key not in rods:raise ContractError('No explicitly preserved rod input for '+key)
                arrays[key+'_values']=np.stack([rods[key].at(float(t)).values for t in edges])
                arrays[key+'_xy']=owned.cells.rf_position.values
            declarations[key]={'population_id':pid,'cell_ids':list(port.cell_ids),'data_axes':['time','cell'],
                'coordinate_role':'rf','coordinate_unit':'mm','coordinate_frame':'functional_visual_field','reference_id':ref,
                'coordinate_registration':'frozen_RF_IS_project_transfer_and_explicit_scene_row_registration',
                'positions_key':'native_xy' if native else key+'_xy','values_key':'native_counts' if native else key+'_values',
                'unit':'R*/window' if native else 'R*/s','input_stage':'reference_equivalent_cone_drive' if native else 'photoisomerization',
                'evidence_tag':port.evidence_tag,'value_semantics':'expected' if native else 'deterministic',
                'photon_noise':{'owner':'none','model':'none'},'mechanism_ownership':{'photopigment_capture':'upstream','phototransduction':'consumer','neural_noise':'consumer'},
                'background':{'includes_background':True,'unit':port.unit,'value':100. if native else config.rod_background_rstar_s,'source_id':light.background_source},
                'prehistory':{'kind':'declared_uniform_background','stop_seconds':0.,'spatial_uniformity':'uniform reference scene; per-cell capture retained'},
                'time':{'representation':'integration_windows' if native else 'samples','reference':'stimulus_relative_seconds','source_origin_seconds':0.,
                    'start_seconds':0.,'stop_seconds':config.duration_seconds,'axis_key':'edges','interpolation':'zero_order_hold','source_grid_step_seconds':.0001}}
            if native:declarations[key].update(reference_profile=reference_profile,capture_scope_sha256=background_signature,
                capture_types=types,capture_pupil_diameter_mm=3.,capture_scope='approved_mixed_reference')
        manifest={'schema':SCHEMA,'version':'1.0','producer':{'name':'Active LCD-Apple/MW/ISETBio frontend','version':'1.0','source_id':PROFILE,
            'capture_sha256':digest(self.directory/'capture.mat'),'request_signature':request,'frontend_signature':config.frontend_signature,
            'capture':meta,'fixed_reference_bridge':bridge,'source_hashes':source_hashes},
            'background_context_hash':light.background_context_hash,'channels':declarations,
            'geometry_identity':geometry}
        packet=self.input_path
        # The capture cache is shared; each validated Run binding is create-once.
        with packet.open('xb') as f:np.savez_compressed(f,manifest_json=np.array(json.dumps(manifest,allow_nan=False)),**arrays)
        self.bundle=load_offline_inputs(packet,compiled,light,reference_id=ref)
        self.summary=dict(self.bundle.source_summary,cache_directory=str(self.directory),actual_ISET_call=not self.cache_reused,
            capture_producer='real pinned ISET APIs',cache_reused=self.cache_reused,
            input_binding_policy='validated_create_once_run_owned_snapshot',
            frontend_signature=config.frontend_signature,background_drive_range_reference_td=[float(background_values.min()),float(background_values.max())])
        self.last_input_summary=self.summary;self.channels=None

    def set_origin(self,origin,grid_origin):
        self.channels=rebase_offline_inputs(self.bundle,self.compiled,self.config.light_context(),
            stimulus_origin_seconds=origin,grid_origin_seconds=grid_origin)

    def window(self,start,stop):
        if self.channels is None:raise ContractError('ISET source has no prepared stimulus origin')
        return self.channels
