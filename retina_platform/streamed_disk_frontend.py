"""Opt-in bounded disk input using the unchanged memoryless ISET exporter.

One capture per distinct (decoded frame, gaze pose) retains 100-us integration
time. All tile jobs use the original batch runner and recoverable cache. There
is no framewise neural reset, new optical approximation, or sampling reduction.
"""
from dataclasses import replace
from pathlib import Path
import json,time
import numpy as np
from scipy.io import savemat
from scipy.ndimage import map_coordinates
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.disk_graph_storage import file_sha, save_json
from retina_sim.disk_input import join_captures
from retina_sim.geometry_identity import compiled_geometry_identity
from retina_sim.offline_input import rebase_offline_inputs
from retina_sim.streamed_input import SCHEMA, BUDGET, load_streamed_input
from .disk_frontend import (DiskMixedReferenceFrontend, _capture_request,
                            _finish_capture, read_disk_capture)
from .bounded_iset_batch import run_batch as _run_batch
from .iset_frontend import source_signature


def exposure_index(config, media, gaze, edges):
    """Retain every original bin and its exact first occurrence time/pose."""
    keys, exposures, mapping = {}, [], []
    for t in edges[:-1]:
        pose = np.asarray(gaze(float(t)), dtype=float)
        if pose.shape != (2,) or not np.all(np.isfinite(pose)):
            raise ContractError('Finite two-dimensional prescribed gaze required')
        time = config.start_seconds + float(t)
        frame = 0 if media.info['kind'] == 'image' else min(int(time*media.info['fps']+1e-8), media.info['frame_count']-1)
        key = (frame, tuple(pose))
        if key not in keys:
            keys[key] = len(exposures)
            exposures.append((float(t), frame, pose))
        mapping.append([keys[key], 0])
    return exposures, mapping


def save_shard(directory, name, values):
    path = Path(directory) / name
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.nbytes > BUDGET or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ContractError('Bounded finite nonnegative capture shard required')
    if path.exists():
        if not np.array_equal(np.load(path, allow_pickle=False), array):
            raise ContractError('Run-owned capture shard changed; retained')
    else:
        with path.open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
    return {'path':name, 'sha256':file_sha(path), 'shape':list(array.shape)}


class StreamedDiskMixedReferenceFrontend(DiskMixedReferenceFrontend):
    def __init__(self, config, compiled, media, gaze, root, *, input_path,
                 on_status=None, on_boundary=None):
        began=time.perf_counter();phases={}
        self.config, self.compiled, self.root = config, compiled, Path(root).resolve()
        self.input_path = Path(input_path).resolve()
        if self.input_path.exists():
            raise FileExistsError('Run-owned streamed input already exists; use explicit strict reload')
        self.input_path.parent.mkdir(parents=True, exist_ok=True)
        directory = self.input_path.parent / (self.input_path.stem + '_shards')
        directory.mkdir(exist_ok=False)
        circuit, view = compiled.global_disk_circuit, compiled.global_disk_view
        from retina_sim.atlas_population_network import circuit_configuration
        configuration = circuit_configuration(self.root, config.circuit_configuration_path)
        binding = configuration['input_plan']
        plan_path = self.root / binding['path']
        if file_sha(plan_path) != binding['sha256']:
            raise ContractError('Frozen global optical plan differs')
        plan = json.loads(plan_path.read_bytes())
        cone = circuit.table('cones')
        if (plan['cone_table'] != cone.binding or plan['regenerate_on_ROI'] is not False or
                plan['exporter_sha256'] != file_sha(self.root/'tools/export_disk_mixed_reference.m')):
            raise ContractError('Frozen optical geometry/exporter differs')
        profile, sources = source_signature(self.root)
        for relative in ('tools/export_disk_mixed_reference.m','tools/export_disk_mixed_batch.m',
                         'retina_sim/disk_input.py','retina_platform/disk_frontend.py'):
            sources[str((self.root/relative).resolve())] = file_sha(self.root/relative)
        reference = dict(plan['reference'])
        scope = canonical_hash({'plan':binding,'profile':profile,'sources':sources,
            'reference_linear_u':config.reference_linear_value,'fixed_reference_definition':reference})
        reference['plan_definition_sha256'] = reference['definition_sha256']
        reference['definition_sha256'] = scope
        u = config.reference_linear_value
        digital = 12.92*u if u <= .0031308 else 1.055*u**(1/2.4)-.055
        color = np.full(3, digital)
        focal = .223e-3/np.tan(np.pi/180)
        def fov(samples):
            return float(np.rad2deg(2*np.arctan(samples*.001e-3/(2*focal))))
        self.reference_directory = self.root/'reports/iset_disk_input_cache'/scope/'reference'
        data = {'rgb_frames':np.full((193,193,3,1),digital,dtype=np.float32),
            'window_edges_seconds':np.array([0.,.0001]),'frame_indices':np.array([1.]),
            'reference_rgb':color,'is_mm':np.zeros((3,2)),'types':np.array([1.,2.,3.]),
            'frozen_apertures_um':np.full(3,reference['reference_diameter_um']),'image_center_deg':np.zeros(2)}
        export = {'profile':profile,'global_reference':reference,'patch_fov_deg':fov(193),
                  'request_signature':scope,'background_signature':scope}
        tick=time.perf_counter()
        reference_jobs=[]
        (refraw, refmeta), actual = _capture_request(self.root,profile,self.reference_directory,data,export,
            on_status=on_status,on_boundary=on_boundary,defer_jobs=reference_jobs)
        if refraw is None:
            _run_batch(self.root,profile,reference_jobs,self.reference_directory,on_status,on_boundary)
            refraw,refmeta=_finish_capture(self.reference_directory,refmeta['pending_path'],refmeta['signature'])
        phases['reference_capture_or_strict_cache_read']=time.perf_counter()-tick
        if refmeta.get('global_reference_definition_sha256') != scope:
            raise ContractError('Captured fixed LMS reference differs')
        reference.update(fixed_radiance_gain=refmeta['fixed_radiance_gain'],
            q_ref_LMS_rstar_s=refraw['q_ref_LMS'].reshape(-1).tolist(),
            reference_energy_spectrum_hex=refmeta['reference_energy_spectrum_hex'])
        ticks = int(round(config.duration_seconds/.0001))
        if ticks < 1 or abs(ticks*.0001-config.duration_seconds) > 1e-12:
            raise ContractError('Stream duration must retain complete original 100-us windows')
        edges = np.arange(ticks+1)*.0001
        exposures, mapping = exposure_index(config,media,gaze,edges)
        media_identity = file_sha(config.media_path)
        request = canonical_hash({'reference_scope':scope,'media':media_identity,
            'start':config.start_seconds,'duration':config.duration_seconds,'field_width_deg':config.field_width_deg,
            'gaze':np.asarray([gaze(float(t)) for t in edges[:-1]])})
        active = np.asarray(view['selected_rows']['cones'])
        lookup = {int(row):i for i,row in enumerate(active)}
        if len(active)*8*6 > BUDGET:
            raise ContractError('Single-exposure global capture needs an explicit larger storage revision')
        tiles = []
        for tile in plan['tiles']:
            path = self.root/tile['global_rows']['path']
            if file_sha(path) != tile['global_rows']['sha256']:
                raise ContractError('Frozen optical tile rows changed')
            rows = np.load(path,allow_pickle=False)
            kept = np.array([i for i,row in enumerate(rows) if int(row) in lookup],dtype=np.int64)
            if len(kept):
                tiles.append((tile,rows,kept,np.asarray([lookup[int(rows[i])] for i in kept],dtype=np.int64)))
        width,height = media.info['width'],media.info['height']
        ppd = width/config.field_width_deg
        jobs, tasks = [], [[] for _ in exposures]
        tick=time.perf_counter();decode_seconds=0.
        for exposure,(t,frame_index,pose) in enumerate(exposures):
            decode_tick=time.perf_counter()
            rgb = media.frame_at(config.start_seconds+t).astype(float)/255.
            decode_seconds+=time.perf_counter()-decode_tick
            for tile,rows,kept,local in tiles:
                if on_boundary is not None and on_boundary() is False:
                    raise ContractError('Stream preparation stopped; completed fixed tile captures retained')
                samples = tile['scene_samples'];center = np.asarray(tile['center_mm'])
                axis = np.arange(samples)-(samples-1)/2
                xx,yy = np.meshgrid((axis*.001+center[0])/.223,(-axis*.001+center[1])/.223)
                coords = [height/2+ppd*(yy+pose[1]),width/2+ppd*(xx+pose[0])]
                patch = np.stack([map_coordinates(rgb[:,:,k],coords,order=1,mode='constant',
                    cval=digital,prefilter=False) for k in range(3)],axis=2).astype(np.float32)
                # Same one-window request as the original eager frontend at this actual time/pose.
                one_request = canonical_hash({'reference_scope':scope,'media':media_identity,
                    'start':config.start_seconds+t,'duration':.0001,'field_width_deg':config.field_width_deg,
                    'gaze':np.asarray([pose])})
                signature = canonical_hash({'global_capture':one_request,'tile':tile,'reference_scope':scope})
                cache = self.root/'reports/iset_disk_input_cache'/scope/'tiles'/signature
                data = {'rgb_frames':patch[:,:,:,None],'window_edges_seconds':np.array([0.,.0001]),
                    'frame_indices':np.array([1.]),'reference_rgb':color,'is_mm':cone.coordinates['rf'][rows],
                    'types':np.array([{'L':1.,'M':2.,'S':3.}[x.decode('ascii')] for x in cone.extras['subtype'][rows]]),
                    'frozen_apertures_um':cone.extras['aperture_diameter_um'][rows],'image_center_deg':center/.223}
                export = {'profile':profile,'global_reference':reference,'patch_fov_deg':fov(samples),
                          'request_signature':signature,'background_signature':scope}
                (raw,meta),called = _capture_request(self.root,profile,cache,data,export,
                    on_status=on_status,on_boundary=on_boundary,defer_jobs=jobs)
                actual |= called
                # Do not retain loaded tile arrays or RGB across exposures.
                tasks[exposure].append((cache,signature,meta.get('pending_path'),kept,local,tile['tile_index']))
                del raw
            if on_status:
                on_status('ISET bounded exposures prepared %d/%d'%(exposure+1,len(exposures)))
        phases['decode_crop_and_prepare_tile_requests']=time.perf_counter()-tick
        phases['media_frame_decode_or_cache_copy_and_float_conversion']=decode_seconds
        tick=time.perf_counter()
        _run_batch(self.root,profile,jobs,directory,on_status,on_boundary)
        phases['MATLAB_tile_batch']=time.perf_counter()-tick
        tick=time.perf_counter()
        count_shards, exposure_records, first_raw, first_meta, background = [], [], None, None, None
        for exposure,entries in enumerate(tasks):
            captures, rows, evidence = [], [], []
            for cache,signature,pending,kept,local,tile_index in entries:
                raw,meta = (_finish_capture(cache,pending,signature) if pending is not None
                            else read_disk_capture(cache/'capture.mat',signature))
                captures.append((dict(raw,expected_counts=raw['expected_counts'][:,kept],
                    background_counts=raw['background_counts'].reshape(-1)[kept],
                    diameters_um=raw['diameters_um'].reshape(-1)[kept]),meta))
                rows.append(local)
                evidence.append({'tile_index':tile_index,'capture_sha256':file_sha(cache/'capture.mat'),'directory':str(cache)})
            joined = join_captures(rows,captures,len(active),reference_identity=scope)
            if background is None:
                background = joined['background_counts'].copy()
            elif not np.array_equal(background,joined['background_counts']):
                raise ContractError('Per-exposure capture changed the fixed initialized background')
            count_shards.append(save_shard(directory,'counts_%06d.npy'%exposure,joined['expected_counts']))
            if exposure == 0:
                first_raw = {'expected_counts':joined['expected_counts'],
                    'background_counts':background[None,:],'q_ref_LMS':joined['q_ref_LMS'][None,:],
                    'window_edges_seconds':np.array([[0.,.0001]]),'diameters_um':joined['diameters_um'][None,:],
                    'fixed_radiance_gain_numeric':np.array([[joined['fixed_radiance_gain']]])}
                first_meta = dict(refmeta,request_signature=request,background_signature=scope,
                    global_reference_definition_sha256=scope,tile_captures=evidence,
                    reference_scene_directory=str(self.reference_directory))
            evidence_path = directory/('exposure_%06d.json'%exposure)
            save_json(evidence_path,{'time_seconds':exposures[exposure][0],
                'frame_index':exposures[exposure][1],'gaze_deg':exposures[exposure][2].tolist(),'tile_captures':evidence})
            exposure_records.append({'path':evidence_path.name,'sha256':file_sha(evidence_path)})
            if on_status:
                on_status('ISET bounded exposures joined %d/%d'%(exposure+1,len(exposures)))
        self.directory = directory
        savemat(directory/'capture.mat',dict(first_raw,manifest_json=json.dumps(first_meta)),do_compression=True)
        phases['join_validate_and_write_count_shards']=time.perf_counter()-tick
        # Use the unchanged strict global packet binder for geometry/units/reference/background.
        one_config = replace(config,duration_seconds=.0001)
        contract_path = directory/'contract.npz'
        self.input_path,self.geometry_identity,self.cache_reused = contract_path,compiled_geometry_identity(compiled),not actual
        tick=time.perf_counter()
        self._bind(one_config,compiled,media,gaze,compiled.populations['cones'],
            list(compiled.populations['cones'].cells.cell_id),list(compiled.populations['cones'].cells.subtype),
            circuit.record['reference_id'],first_raw,first_meta,np.array([0.,.0001]),request,scope,sources)
        phases['strict_one_window_contract_bind']=time.perf_counter()-tick
        config.frontend_signature = one_config.frontend_signature
        config.native_background_values = one_config.native_background_values
        from .backend import _MediaChannels
        rod_source = _MediaChannels(config,compiled,media,gaze,0.)
        rod_shards,rod_mapping,rod_evidence = [],np.empty((ticks+1,2),dtype=np.int64),[]
        tick=time.perf_counter()
        for lo in range(0,ticks,20):
            hi = min(ticks,lo+20)
            channels = rod_source.window(float(edges[lo]),float(edges[hi]))
            if set(channels) != {'rod_light'}:
                raise ContractError('Streamed disk route requires the unchanged effective rod input channel')
            values = np.stack([channels['rod_light'].at(float(t)).values for t in edges[lo:hi+1]])
            index = len(rod_shards)
            rod_shards.append(save_shard(directory,'rods_%06d.npy'%index,values))
            rod_mapping[lo:hi+1,0] = index
            rod_mapping[lo:hi+1,1] = np.arange(hi-lo+1)
            rod_evidence.append(rod_source.last_input_summary)
        phases['original_rod_mapping_in_bounded_chunks']=time.perf_counter()-tick
        producer = dict(self.summary['producer'],version='1.0+bounded_storage',
            storage_source_sha256={str(Path(__file__).resolve()):file_sha(__file__),
                'retina_sim/streamed_input.py':file_sha(self.root/'retina_sim/streamed_input.py'),
                'retina_platform/bounded_iset_batch.py':file_sha(Path(__file__).with_name('bounded_iset_batch.py'))},
            unique_optical_exposures=len(exposures),original_window_count=ticks,
            tile_export_jobs_this_run=len(jobs),active_optical_tiles=len(tiles),
            temporal_history='Unchanged exporter captures one memoryless OI; phototransduction/noise absent. Neural state is continuous across all frames.',
            rod_input_chunks=rod_evidence)
        manifest = {'schema':SCHEMA,'version':'1.0','global_graph':circuit.binding,
            'geometry_identity':self.geometry_identity,'background_context_hash':config.light_context().background_context_hash,
            'reference_id':circuit.record['reference_id'],'edges_seconds':edges.tolist(),
            'contract_packet':{'path':'contract.npz','sha256':file_sha(contract_path)},'producer':producer,
            'exposure_records':exposure_records,
            'channels':{'native_light':{'semantics':'expected_Rstar_per_original_window','shards':count_shards,'row_map':mapping},
                        'rod_light':{'semantics':'sampled_original_rod_Rstar_s','shards':rod_shards,'row_map':rod_mapping.tolist()}}}
        # Manifest beside shards keeps all references inside a single run-owned directory.
        save_json(directory/'manifest.json',manifest)
        self.input_path = directory/'manifest.json'
        tick=time.perf_counter()
        self.bundle,self.stores = load_streamed_input(self.input_path,compiled,config.light_context(),
            reference_id=circuit.record['reference_id'])
        phases['strict_stream_manifest_and_all_shard_validation']=time.perf_counter()-tick
        phases['total_input_preparation']=time.perf_counter()-began
        self.summary = dict(self.bundle.source_summary,actual_ISET_call=actual,cache_reused=not actual,
            frontend_signature=config.frontend_signature,manifest_path=str(self.input_path),
            phase_wall_seconds=phases,timing_interpretation='Decode is included in preparation; phase times are not all disjoint')
        self.last_input_summary,self.channels = self.summary,None
        save_json(Path(input_path),{'schema':'bounded_global_input_pointer_v1','manifest_path':str(self.input_path),
            'manifest_sha256':file_sha(self.input_path)})

    def set_origin(self, origin, grid_origin):
        self.channels = rebase_offline_inputs(self.bundle,self.compiled,self.config.light_context(),
            stimulus_origin_seconds=origin,grid_origin_seconds=grid_origin)
