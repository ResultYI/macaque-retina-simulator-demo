"""Explicit original-HC v7 video trial; bounded ISET, 1-ms observed states.

No new background is prepared. Strict convergence is the default. Transient
initialization requires explicit --background-mode transient and --background-source.
This route does not qualify the new functional-HC graph or full physiology.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,os,sys,time,shutil,subprocess,gc
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def save(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n',encoding='utf8')
    os.replace(temporary,path)


def pin_v7_execution_environment():
    """Reproduce the recorded process setting before NumPy initializes BLAS.

    This field participates in the existing compiled execution identity. The
    old checkpoint is never relabelled to the caller's different environment.
    """
    source=ROOT/'reports/performance_real_prepared_A_v4/legacy_bridge_case/case.json'
    record=json.loads(source.read_bytes())
    manifest=ROOT/'reports/human_disk3mm_floor_transition_v7/transition/manifest.json'
    import hashlib
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=record['source_manifest_sha256']:
        raise ValueError('Retained v7 execution environment belongs to a different source manifest')
    expected=record['integration_metadata']['process_environment']['OPENBLAS_NUM_THREADS']
    if expected!='1':raise ValueError('Unexpected retained v7 BLAS process setting; review required')
    if os.environ.get('OPENBLAS_NUM_THREADS')!=expected:
        if 'numpy' in sys.modules:
            raise ValueError('Launch a fresh video process so the recorded BLAS setting precedes NumPy initialization')
        os.environ['OPENBLAS_NUM_THREADS']=expected
    return {'OPENBLAS_NUM_THREADS':expected,'source':str(source.relative_to(ROOT))}


def checkpoint_file(runner,state,path):
    # Existing tested streamed legacy encoder: all dynamic arrays, RNG, clocks,
    # event/delay histories and input position; same public checkpoint protocol.
    from retina_sim.checkpoint_streaming import _checkpoint,_chunks
    tick=time.perf_counter();path=Path(path);temporary=path.with_name(path.name+'.writing')
    with temporary.open('w',encoding='utf8') as stream:
        for chunk in _chunks(_checkpoint(runner,state)):stream.write(chunk)
        stream.write('\n')
    os.replace(temporary,path)
    return {'path':str(path),'bytes':path.stat().st_size,'seconds':time.perf_counter()-tick}


def recording_cohort(compiled):
    import numpy as np
    selected=set()
    for name,pop in compiled.populations.items():
        if name not in ('cones','HC'):selected.update(pop.cells.cell_id)
    columns=[link.matrix.indices for link in compiled.links if link.specification.source_component=='outer_native']
    cone_rows=np.unique(np.concatenate(columns));native=compiled.components['outer_native'].native_model
    hc_rows=np.unique(native.B[cone_rows].indices)
    selected.update(compiled.populations['cones'].cells.cell_id[i] for i in cone_rows)
    selected.update(compiled.populations['HC'].cells.cell_id[i] for i in hc_rows)
    return frozenset(selected),cone_rows,hc_rows


def archive_uncommitted_windows(output, committed):
    """Preserve results newer than the last atomic full checkpoint on resume."""
    output=Path(output).resolve();windows=sorted((output/'windows').glob('window_*.npz'))
    if not isinstance(committed,int) or committed<0 or committed>len(windows):
        raise ValueError('Checkpoint committed window count differs from saved results')
    for index,path in enumerate(windows):
        if path.name!='window_%08d.npz'%index:raise ValueError('Saved recording sequence is incomplete')
    pending=windows[committed:]
    if not pending:return
    destination=output/('recovered_uncommitted_'+str(time.time_ns()))
    destination.mkdir()
    for path in pending:
        target=(destination/path.name).resolve()
        if not path.resolve().is_relative_to(output) or not target.is_relative_to(output):
            raise ValueError('Recovery archive must stay inside this output')
        os.replace(path,target)
    save(destination/'receipt.json',{'reason':'Results after the last full checkpoint; retained while deterministic replay resumes',
        'committed_windows':committed,'preserved_files':[path.name for path in pending]})


def state_manifest(compiled,runner,plan):
    from retina_platform.backend import _port_cells
    records={cid:{'id':cid,'mechanism':c.mechanism,'layers':list(c.layers),'ports':{}}
             for cid,c in compiled.components.items()}
    for spec in runner.model.state_specs:
        indices=plan.indices(spec.cell_ids)
        if not len(indices):continue
        ids=tuple(spec.cell_ids[i] for i in indices);cid,name=spec.state_name.split('.',1)
        records[cid]['ports']['state:'+name]={'key':'state:'+spec.state_name,'unit':spec.unit,
            'meaning':spec.meaning,'evidence_tag':runner.model.evidence_tag,'temporal_kind':'sampled',
            **_port_cells(compiled,ids)}
    for record in records.values():
        main=next(iter(record['ports'].values()),{})
        record.update({key:main.get(key,[] if key!='population' and key!='geometry' else {} if key=='geometry' else None)
            for key in ('population','cell_ids','cell_types','subtypes','regions','geometry')})
    return list(records.values())


def main():
    from tools.video_background_start import (
        add_background_arguments, select_background_source, restore_background_start, begin_media_stimulus)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media',type=Path,required=True);parser.add_argument('--start',type=float,default=0.)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='Strictly resume this output and its latest full checkpoint')
    parser.add_argument('--group-optical-frames',action='store_true',help='Explicit grouped_tiles_v1 optical requests; original exporter and 100-us mapping retained')
    parser.add_argument('--prepared-input-from',type=Path,help='Strictly reuse the exact input packet of a completed same-video/configuration/model run')
    parser.add_argument('--fast-lossless-storage',action='store_true',help='Same checkpoint JSON bytes and decoded NPZ data; encode checkpoint subtrees once and store NPZ without DEFLATE')
    parser.add_argument('--skip-empty-link-publication',action='store_true',help='Explicit equivalent procedure: avoid unused full-state packing when link-delay buffers are empty')
    add_background_arguments(parser)
    args=parser.parse_args();output=(ROOT/args.output).resolve();media_path=args.media.resolve()
    if not output.is_relative_to(ROOT) or args.start<0:raise ValueError('Project output and nonnegative clip start required')
    if media_path.name in {'synthetic_rgb_50fps.avi','local_spot_8Hz.avi'}:
        raise ValueError('This authorized trial requires real footage covering the requested 0.5 seconds')
    execution_environment=pin_v7_execution_environment()
    global np
    import numpy as np
    executable=shutil.which('ffprobe')
    if executable is None:raise ValueError('Existing ffprobe is required for the bounded metadata preflight')
    probe=subprocess.run([executable,'-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,avg_frame_rate,r_frame_rate,duration,nb_frames:format=duration,size','-of','json',str(media_path)],
        capture_output=True,text=True,encoding='utf8',check=True)
    probe=json.loads(probe.stdout);stream=probe['streams'][0]
    media_info={'kind':'video','width':int(stream['width']),'height':int(stream['height']),
        'fps':float(Fraction(stream['avg_frame_rate'])),'duration_seconds':float(probe['format']['duration']),
        'frame_count':int(stream['nb_frames']),'probe':'ffprobe metadata only; no full-frame decoder before resource admission'}
    if (media_info['kind']!='video' or
            args.start+.5>media_info['duration_seconds']+1e-9):
        raise ValueError('This authorized trial requires real footage covering the requested 0.5 seconds')
    if output.exists() != args.resume:
        raise ValueError('Use a fresh output or explicitly resume an existing output')
    # Select actual archived compiler bytes before factory imports; never relabel an old checkpoint.
    from tools.benchmark_compiler_snapshot import use_v7_compiler
    compiler=use_v7_compiler()
    from retina_sim.disk_graph_storage import file_sha
    from retina_sim.background_checkpoint import MemoryGuard,BackgroundCheckpointStore
    from tools.benchmark_execution_procedures import available_windows_commit_bytes
    from retina_platform.config import RunConfig
    from retina_sim.atlas_population_network import atlas_population_network
    from retina_sim.execution_backends import make_runner
    from retina_sim.contracts import SolverConfig
    from retina_sim.recording import RecordingPlan,NPZWindowRecorder
    write_checkpoint,recorder_type=checkpoint_file,NPZWindowRecorder
    if args.fast_lossless_storage:
        from retina_sim.lossless_video_storage import checkpoint_file_spooled,StoredNPZWindowRecorder
        write_checkpoint,recorder_type=checkpoint_file_spooled,StoredNPZWindowRecorder
    from retina_sim.recording_indices import IndexedRecordingPlan
    from retina_sim.offline_input import rebase_offline_inputs
    from retina_sim.streamed_input import load_streamed_input
    from retina_platform.streamed_disk_frontend import StreamedDiskMixedReferenceFrontend
    from retina_sim.checkpoint_loading import restore_retina_payload
    from tools.disk3mm_response_check import aligned_response_times
    guard=MemoryGuard(14336*1024**2,8192*1024**2)
    memory_statistics={'peak_python_RSS_bytes':0,'minimum_system_available_physical_bytes':None}
    def boundary():
        admissible=guard.check()
        memory_statistics['peak_python_RSS_bytes']=max(memory_statistics['peak_python_RSS_bytes'],guard.peak_rss_bytes)
        available=guard.last['available_physical_bytes']
        old=memory_statistics['minimum_system_available_physical_bytes']
        memory_statistics['minimum_system_available_physical_bytes']=available if old is None else min(old,available)
        torch=sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            memory_statistics['GPU_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        if not admissible:
            raise MemoryError('Original memory reserve reached: '+json.dumps(guard.last))
        return not (output/'STOP').exists()
    boundary();commit=available_windows_commit_bytes()
    print('RESOURCE_ADMISSION',json.dumps(dict(guard.last,available_commit_bytes=commit)),flush=True)
    if guard.last['available_physical_bytes']<15*1024**3 or commit<7*1024**3:
        raise MemoryError('Original startup admission requires 15-GiB free physical and 7-GiB commit')
    wall=time.perf_counter();phases={};output.mkdir(exist_ok=args.resume)
    globals()['_run_output']=output
    save(output/'media_probe.json',media_info)
    original=ROOT/'reports/human_disk3mm_floor_transition_v7/transition'
    source=json.loads((original/'manifest.json').read_bytes())
    configuration,background_checkpoint,background_identity=select_background_source(
        ROOT,original,source['configuration'],args.background_mode,args.background_source)
    cfg=replace(RunConfig.from_dict(configuration),media_path=str(media_path),start_seconds=args.start,
        duration_seconds=.5,sample_seconds=.001,window_seconds=.002)
    if args.restore_diagnostics:
        import hashlib
        from retina_sim.background_checkpoint import checkpoint_hash_diagnostics
        diagnostic_tick=time.perf_counter();serialized=background_checkpoint.read_bytes()
        snapshot=checkpoint_hash_diagnostics(json.loads(serialized),hashlib.sha256(serialized).hexdigest())
        save(output/'restore_before_backend.json',snapshot)
        if any(snapshot[key]['expected']!=snapshot[key]['actual'] for key in ('complete','auxiliary','report')):
            raise ValueError('Source checkpoint hash rejection before backend setup; no assembly performed')
        del serialized,snapshot;gc.collect()
        phases['restore_pre_backend_diagnostics']=time.perf_counter()-diagnostic_tick;boundary()
    identity={'media_sha256':file_sha(media_path),'start_seconds':args.start,'duration_seconds':.5,
        'source_manifest_sha256':file_sha(original/'manifest.json'),'compiler':compiler,
        'execution_environment':execution_environment,
        'route':'original_HC_v7_only','new_functional_HC_acceptance':False}
    if background_identity is not None:
        identity['background_start']=dict(background_identity,driver_sha256=file_sha(Path(__file__)),
            selector_sha256=file_sha(ROOT/'tools/video_background_start.py'))
    if args.group_optical_frames and args.prepared_input_from is None:
        from retina_platform.grouped_disk_capture import SCHEMA as group_schema
        identity['optical_grouping']={'schema':group_schema,
            'planner_sha256':file_sha(ROOT/'retina_platform/grouped_disk_capture.py'),
            'frontend_sha256':file_sha(ROOT/'retina_platform/streamed_disk_frontend.py')}
    prepared_binding=None
    if args.prepared_input_from is not None:
        from tools.prepared_video_input import prepared_input_binding
        admission_tick=time.perf_counter()
        prepared_binding=prepared_input_binding(ROOT,args.prepared_input_from,cfg,identity)
        phases['prepared_input_admission']=time.perf_counter()-admission_tick
        identity['prepared_input']=prepared_binding
    if args.fast_lossless_storage:
        identity['storage_execution']={'procedure':'once_encoded_checkpoint_and_stored_NPZ_v1',
            'source_sha256':file_sha(ROOT/'retina_sim/lossless_video_storage.py'),
            'checkpoint_builder_sha256':file_sha(ROOT/'retina_sim/checkpoint_streaming.py'),
            'checkpoint_protocol_and_numeric_record_values':'unchanged'}
    if args.skip_empty_link_publication:
        from retina_sim.no_delay_publication import procedure_identity
        identity['advance_execution']=procedure_identity()
    previous=json.loads((output/'trial.json').read_bytes()) if args.resume else None
    if previous and previous['identity']!=identity:raise ValueError('Resume media/clip/source/compiler differs')
    initialization=None
    def progress(phase,**values):
        row={'status':phase,'worker_pid':os.getpid(),'identity':identity,'phase_seconds':phases,'wall_seconds':time.perf_counter()-wall,
             'memory':guard.last,'memory_statistics':memory_statistics,**values}
        if initialization is not None:row['background_initialization']=initialization
        save(output/'trial.json',row);print(json.dumps({k:v for k,v in row.items() if k not in ('identity','memory')}),flush=True)
    progress('ASSEMBLY');tick=time.perf_counter()
    construction=replace(cfg,native_background_values=None,frontend_signature='')
    compiled,_,audit=atlas_population_network(ROOT,duration=.0001,light_context=construction.light_context(),
        owner_bounds_mm=cfg.owner_bounds(),halo_mm=.04,max_spatial_cells=cfg.max_spatial_cells,
        seed=cfg.seed,execution_profile=cfg.execution_profile,configuration_path=cfg.circuit_configuration_path)
    phases['assembly']=time.perf_counter()-tick;boundary()
    # Existing production audit compaction retains all actual parameter arrays
    # and selected geometry, then releases duplicated construction-only JSON.
    tick=time.perf_counter()
    from retina_platform.disk_run_audit import prepare_global_audit
    definition=prepare_global_audit(compiled,audit,output)
    save(output/'compiled_definition.json',definition)
    solver_definition=dict(audit['resolved_execution_profile']['solver'])
    del audit,definition;gc.collect()
    phases['construction_metadata_export_and_release']=time.perf_counter()-tick;boundary()
    progress('RUNNER_AND_STRICT_BACKGROUND');tick=time.perf_counter()
    runner=make_runner(compiled,cfg.neural_integration_backend)
    solver=SolverConfig(**solver_definition);reference=runner.initialize(solver)
    phases['runner_and_reference']=time.perf_counter()-tick;tick=time.perf_counter()
    if args.background_mode=='transient':
        from retina_sim.execution_backends import verify_cuda_execution,CUDA_BACKENDS
        if cfg.neural_integration_backend in CUDA_BACKENDS:
            progress('CPU_GPU_100US_QUALIFICATION')
            qualification=verify_cuda_execution(runner,solver,cfg.light_context())
            phases['CPU_GPU_100us_qualification']=time.perf_counter()-tick
            save(output/'execution_backend_check.json',qualification);boundary();tick=time.perf_counter()
    frozen_background_context=cfg.light_context().background_context_hash
    store=BackgroundCheckpointStore(ROOT,runner,reference,cfg.light_context(),cfg.background_policy())
    phases['strict_binding_construction']=time.perf_counter()-tick
    if not args.resume:
        tick=time.perf_counter()
        state,envelope,initialization=restore_background_start(store,background_checkpoint,args.background_mode,
            diagnostic_path=output/'restore_inside_loader.json' if args.restore_diagnostics else None)
        save(output/'background_initialization.json',initialization)
        del envelope
        origin=state.time_seconds
        phases['strict_restore']=time.perf_counter()-tick;boundary()
        progress('TRANSIENT_BACKGROUND_RESTORED' if args.background_mode=='transient' else 'STRICT_BACKGROUND_RESTORED',
            absolute_background_time_seconds=origin,background_initialization=initialization,
            model_hash=runner.model.model_hash,background_fallback_performed=False)
    else:
        initialization_path=output/'background_initialization.json'
        if initialization_path.is_file():
            initialization=json.loads(initialization_path.read_bytes())
        elif args.background_mode=='strict' and background_identity is None:
            initialization={'mode':'strict','status':None,'test_scope':'Legacy strict video resume; original state metadata retained'}
        else:raise ValueError('Explicit background-start resume requires its saved initialization metadata')
    from retina_platform.media import MediaSource
    with MediaSource(media_path) as media:
        gaze=lambda t:np.array([cfg.gaze_x_deg,cfg.gaze_y_deg])
        if cfg.gaze_csv:raise ValueError('Original v7 trial expects its fixed gaze; no trajectory substitution')
        progress('BOUNDED_ISET_INPUT');tick=time.perf_counter()
        if args.resume:
            saved=json.loads((output/'manifest.json').read_bytes());cfg=RunConfig.from_dict(saved['config'])
        if prepared_binding is not None:
            from tools.prepared_video_input import load_prepared_input
            packet,stores,input_summary=load_prepared_input(prepared_binding,compiled,runner.model.model_hash,cfg)
            save(output/'input_stream.json',{'schema':'bounded_global_input_pointer_v1',
                'manifest_path':packet.source_summary['source_file'],
                'manifest_sha256':packet.source_summary['source_sha256'],'prepared_input':prepared_binding})
        elif args.resume:
            packet,stores=load_streamed_input(output/'input_stream_shards/manifest.json',compiled,cfg.light_context(),
                reference_id=compiled.global_disk_circuit.record['reference_id'])
            input_summary=packet.source_summary
        else:
            frontend=StreamedDiskMixedReferenceFrontend(cfg,compiled,media,gaze,ROOT,input_path=output/'input_stream.json',
                on_status=lambda text:progress('BOUNDED_ISET_INPUT',input_status=text),on_boundary=boundary,
                group_frames=args.group_optical_frames)
            packet,input_summary=frontend.bundle,frontend.summary
        phases['ISET_input_or_strict_input_cache_load']=time.perf_counter()-tick;boundary()
        if cfg.light_context().background_context_hash!=frozen_background_context:
            raise ValueError('Actual video input reference/background differs from the strictly restored v7 background')
        if args.resume:
            tick=time.perf_counter()
            checkpoint=json.loads((output/'latest_checkpoint.json').read_bytes())
            state,_=restore_retina_payload(runner,checkpoint,expected_input_context={
                'phase':'media_stimulus','input_source_sha256':packet.source_summary['source_sha256']})
            origin=state.input_position['stimulus_origin_seconds'];del checkpoint
            archive_uncommitted_windows(output,state.input_position['recorded_window_count'])
            phases['strict_restore']=time.perf_counter()-tick
        begin_media_stimulus(state,packet.source_summary['source_sha256'],origin,initialization)
        boundary()
        selected,cone_rows,hc_rows=recording_cohort(compiled)
        plan=IndexedRecordingPlan(RecordingPlan(ports=(),states=None,cell_ids=selected,max_bytes=64*1024**2))
        scalar_count=sum(len(plan.indices(s.cell_ids)) for s in runner.model.state_specs)
        channels=rebase_offline_inputs(packet,compiled,cfg.light_context(),stimulus_origin_seconds=origin,
            grid_origin_seconds=state.grid_origin_seconds)
        times,alignment=aligned_response_times(runner,state.grid_origin_seconds,origin,.5,.001)
        manifest={'schema':'retina-platform-1','config':cfg.to_dict(),'media':dict(media.info),
            'media_sha256':identity['media_sha256'],'assembly_hash':compiled.assembly_hash,
            'required_layers':list(compiled.spec.required_mechanisms),'components':state_manifest(compiled,runner,plan),
            'input_assumptions':input_summary,'runtime':{'stimulus_origin_seconds':origin,'grid_origin_seconds':state.grid_origin_seconds,
                'integration_backend':cfg.neural_integration_backend,'model_hash':runner.model.model_hash},
            'recording':{'scope':'Formal observation cohort: all inner states plus feedforward cone/HC support; not the full dependency graph',
                'scalar_count':scalar_count,'samples':501,'sample_seconds':.001,'unique_raw_state_bytes':scalar_count*501*8,
                'recorded_cones':len(cone_rows),'recorded_HC':len(hc_rows),'states':'all registered internal states in cohort',
                'timestamped_spikes':'window metadata','full_graph_arbitrary_time_replay':'not implemented','alignment':alignment},
            'identity':identity,'background_initialization':initialization,
            'physiological_calibration':'NOT_COMPLETE','source_graph':'original HC v7, not functional HC acceptance'}
        save(output/'manifest.json',manifest)
        if not args.resume:
            receipt=write_checkpoint(runner,state,output/'start_checkpoint.json')
            phases['start_checkpoint_write']=receipt['seconds']
            phases['initial_latest_checkpoint_write']=write_checkpoint(runner,state,output/'latest_checkpoint.json')['seconds']
        advance=runner.advance
        if args.skip_empty_link_publication:
            from retina_sim.no_delay_publication import no_empty_link_advance
            advance,procedure=no_empty_link_advance(runner)
            if procedure!=identity['advance_execution']:raise ValueError('Advance execution procedure changed after admission')
        recorder=recorder_type(output/'windows');recorder.windows=len(list((output/'windows').glob('window_*.npz')))
        phases['advance_and_record']=0.;phases['record_files_write']=0.;phases['rolling_checkpoint_write']=0.
        last_checkpoint=time.perf_counter()
        try:
            while state.time_seconds<times[-1]-1e-12:
                if not boundary():break
                completed=int(round((state.time_seconds-origin)/.001));end=min(500,completed+2)
                samples=times[completed if completed==0 else completed+1:end+1]
                tick=time.perf_counter()
                state,result=advance(state,channels,float(times[end]),sample_times=samples,recording_plan=plan)
                phases['advance_and_record']+=time.perf_counter()-tick;tick=time.perf_counter()
                temporary=recorder_type(output/'window_pending').consume(result)
                destination=recorder.directory/('window_%08d.npz'%recorder.windows)
                os.replace(temporary,destination);recorder.windows+=1
                state.input_position['recorded_window_count']=recorder.windows
                phases['record_files_write']+=time.perf_counter()-tick
                if time.perf_counter()-last_checkpoint>=60:
                    phases['rolling_checkpoint_write']+=write_checkpoint(runner,state,output/'latest_checkpoint.json')['seconds']
                    last_checkpoint=time.perf_counter()
                progress('ADVANCING',stimulus_seconds=state.time_seconds-origin,windows=recorder.windows,
                    integration_metrics=dict(runner.integration_metrics))
        except BaseException:
            write_checkpoint(runner,state,output/'latest_checkpoint.json')
            progress('STOPPED_WITH_FULL_CHECKPOINT',stimulus_seconds=state.time_seconds-origin)
            raise
        phases['end_checkpoint_write']=write_checkpoint(runner,state,output/'end_checkpoint.json')['seconds']
        phases['final_latest_checkpoint_write']=write_checkpoint(runner,state,output/'latest_checkpoint.json')['seconds']
        progress('COMPLETED' if state.time_seconds>=times[-1]-1e-12 else 'STOPPED_WITH_FULL_CHECKPOINT',
            stimulus_seconds=state.time_seconds-origin,windows=recorder.windows,integration_metrics=dict(runner.integration_metrics),
            recording=manifest['recording'],background_fallback_performed=False)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        output=globals().get('_run_output')
        if output is not None and output.is_dir():
            previous=json.loads((output/'trial.json').read_bytes()) if (output/'trial.json').exists() else {}
            previous.update(status='FAILED_RECOVERABLE_ARTIFACTS_RETAINED',error_type=type(error).__name__,error=str(error))
            save(output/'failure.json',previous);save(output/'trial.json',previous)
        raise
