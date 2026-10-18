"""Read-only0.5s video preflight: no ISET, model assembly or neural execution."""
from pathlib import Path
import argparse,hashlib,json,os,re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
KNOWN_SYNTHETIC={'synthetic_rgb_50fps.avi','local_spot_8Hz.avi'}


def inspect_project_media():
    files=[];references=set();skip={'.git','.aws','.codex','.agents','.venv','node_modules','__pycache__'}
    extensions={'.avi','.mp4','.mov','.mkv','.webm','.m4v','.mpeg','.mpg'}
    for base,dirs,names in os.walk(ROOT,followlinks=False):
        dirs[:]=[name for name in dirs if name not in skip and (Path(base)/name).resolve().is_relative_to(ROOT)]
        for name in names:
            path=Path(base)/name
            if path.suffix.lower() in extensions:
                files.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,
                    'known_synthetic':path.name in KNOWN_SYNTHETIC})
            if name not in ('config.json','request.json','retina_platform_v1.json'):continue
            try:
                with path.open('rb') as stream:text=stream.read(65536).decode('utf-8-sig',errors='ignore')
                for match in re.finditer(r'"media_path"\s*:\s*("(?:[^"\\]|\\.)*")',text):
                    value=json.loads(match.group(1))
                    if Path(value).suffix.lower() in extensions:references.add(value)
            except (OSError,ValueError):pass
    return files,[{'path':path,'exists':(Path(path) if Path(path).is_absolute() else ROOT/path).is_file()}
                  for path in sorted(references)]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--media',type=Path)
    parser.add_argument('--start',type=float,default=0.);parser.add_argument('--duration',type=float,default=.5)
    parser.add_argument('--output',type=Path,default=Path('reports/video_05s_preflight.json'))
    args=parser.parse_args();output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT):raise ValueError('Preflight output must remain in project')
    if args.start<0 or args.duration<=0:raise ValueError('Positive clip duration and nonnegative start required')
    original=ROOT/'reports/human_disk3mm_floor_transition_v7/transition'
    profile=json.loads((ROOT/'reports/performance_profile_schema_cache_v1/profile.json').read_bytes())
    cfg=json.loads((ROOT/'configurations/atlas_human_borrowed_disk3mm_floor_input_v2.json').read_bytes())
    plan=json.loads((ROOT/cfg['input_plan']['path']).read_bytes())
    with np.load(original/'global_definition/exact_view_rows.npz',allow_pickle=False) as rows:
        cones=len(rows['cones']);HC=len(rows['HC']);rods=len(rows['rods'])
        cone_recorded=len(rows['recorded_native_local_rows']);HC_recorded=len(rows['recorded_HC_local_rows'])
        active=rows['cones']
        graph=json.loads((ROOT/cfg['global_graph']['path']).read_bytes())
        mask=np.zeros(graph['tables']['cones']['count'],dtype=bool);mask[active]=True
        tiles=sum(bool(np.any(mask[np.load(ROOT/tile['global_rows']['path'],allow_pickle=False)])) for tile in plan['tiles'])
    native_states=8*cones+2*HC;inner_states=profile['state_scalars']-native_states
    selected_states=8*cone_recorded+2*HC_recorded+inner_states
    points=int(round(args.duration/.001))+1;bins=int(round(args.duration/.0001))
    checkpoint=ROOT/profile['checkpoint_source']
    with checkpoint.open('rb') as stream:actual_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual_sha!=profile['source_checkpoint_sha256']:raise ValueError('Known converged checkpoint file changed')
    files,references=inspect_project_media();blockers=[];media_info=None
    if args.media:
        media_path=args.media.resolve()
        from retina_platform.media import MediaSource
        with MediaSource(media_path) as media:media_info=dict(media.info)
        if media_info['kind']!='video':blockers.append('PROVIDE_A_VIDEO_NOT_A_STILL_IMAGE')
        elif args.start+args.duration>media_info['duration_seconds']+1e-9:blockers.append('VIDEO_TOO_SHORT_FOR_REQUESTED_CLIP')
        if media_path.name in KNOWN_SYNTHETIC:blockers.append('KNOWN_SYNTHETIC_QA_IS_NOT_THE_REQUESTED_REAL_VIDEO')
    else:blockers.append('REAL_VIDEO_PATH_REQUIRED; existing project videos are generated QA/spot stimuli')
    buffer=(bins+1)*(cones+rods)*8;budget=128*1024**2
    qa_path=ROOT/'reports/streamed_input_actual_ISET_QA_v2/result.json'
    qa=json.loads(qa_path.read_bytes()) if qa_path.exists() else {}
    streaming_ready=(qa.get('status')=='ALL_VALUES_AND_BOUNDARIES_BITWISE_EQUAL' and qa.get('real_ISET') is True
        and (ROOT/'tools/run_video_05s.py').is_file() and (ROOT/'retina_platform/streamed_disk_frontend.py').is_file())
    if buffer>budget and not streaming_ready:blockers.append('BOUNDED_INPUT_DRIVER_NOT_QUALIFIED; original eager path exceeds128MiB')
    bg=json.loads((original/'matched_response/background_initialization.json').read_bytes())
    report={'status':'BLOCKED_NO_VIDEO_COMPUTATION_STARTED' if blockers else 'PREFLIGHT_PASS_ONLY',
        'requested_clip':{'start_seconds':args.start,'duration_seconds':args.duration,'media':media_info},
        'blockers':blockers,'workspace_media':files,'project_referenced_videos':references,
        'graph':{'configuration':'configurations/atlas_human_borrowed_disk3mm_floor_input_v2.json','global_graph':cfg['global_graph'],
            'scope':'Original-HC v7 transition, not new functional-HC validation','owner_RF_bounds_mm':[.96,1.04,-.04,.04],
            'cones':cones,'HC':HC,'rods':rods,'full_registered_states':profile['state_scalars'],
            'required_old_compiler':'reports/current_wip_review/frozen_v7_compiler/provenance.json'},
        'background':{'checkpoint':str(checkpoint.relative_to(ROOT)),'file_sha256':actual_sha,
            'model_hash':profile['model_hash'],'strict_restored_in_latest_test':True,'converged':bg['converged'],
            'absolute_time_seconds':2.099999999999879,'no_background_fallback':'Stop on mismatch; never start a new long background'},
        'optical_workload':{'integration_bins_100us':bins,'full_clip_buffer_bytes':buffer,'full_clip_buffer_GiB':buffer/1024**3,
            'guard_is_conservative_not_an_allocated_array':True,
            'native_expected_counts':{'shape':[bins,cones],'dtype':'float64','bytes':bins*cones*8},
            'rod_rate_samples':{'shape':[bins+1,rods],'dtype':'float64','bytes':(bins+1)*rods*8},
            'input_step_source':'retina_platform/disk_frontend.py:130 hardcoded0.0001s; not a temporal ISET requirement',
            'actual_interpolation':'zero_order_hold window means; original widths retained before fixed LMS bridge',
            'bounded_input_ready':streaming_ready,'bounded_input_entry':'tools/run_video_05s.py',
            'bounded_capture_LRU_bytes':64*1024**2,'actual_ISET_small_QA':str(qa_path.relative_to(ROOT)),
            'single_capture_budget_bytes':budget,'max_current_single_capture_seconds':(budget//((cones+rods)*8)-1)*.0001,
            'active_optical_tiles':tiles,'example_50fps_frames_for_05s':25,'example_50fps_framewise_tile_jobs':25*tiles,
            'existing_implementation_naive_2ms_chunks_tile_jobs':int(np.ceil(args.duration/.002))*tiles,
            'display':'Existing LCD-Apple / human-MW mixed reference, assumed sRGB; not measured user-display calibration',
            'framewise_rate_hold':'Potential exact reuse for fixed gaze, deterministic static ISET with no phototransduction/noise; streaming driver and boundary equivalence still required. Do not just loop the synthetic video.'},
        'recording_proposal':{'sample_seconds':.001,'sample_points_including_endpoints':points,
            'registered_internal_states':'All model.state_specs within the existing formal recorded cohort; states=None, fixed cell_ids',
            'recorded_cones':cone_recorded,'recorded_HC':HC_recorded,'other_populations':'All selected inner BC/AC/RGC/rod/contact populations per existing prepare_global_audit cohort',
            'selected_internal_state_scalars_per_sample':selected_states,'raw_internal_state_bytes':selected_states*points*8,
            'raw_internal_state_MiB':selected_states*points*8/1024**2,
            'checked_variable_names':bg['checked_states'],
            'checkpoint_start_end_estimate_bytes':checkpoint.stat().st_size*2,'one_atomic_rolling_recovery_estimate_bytes':checkpoint.stat().st_size,
            'event_history':'Keep original timestamped spike events; starting/ending full RNG/history/delay states; recovery checkpoints. Do not promise per-time full RNG/delay buffers without replay.',
            'full_graph_raw_states_if_100us_bytes':profile['state_scalars']*(bins+1)*8,
            'direct_view_scope':'Video trial entry records the formal cohort at1ms; state keys are declared for ResultStore/GUI browsing. Not the full dependency graph or arbitrary continuous time.',
            'current_capabilities':'Recorder supports internal state arrays; current GUI main path sets states=() and lists output ports. Full internal-state selection/view metadata is not enabled by default. Arbitrary-full-graph checkpoint replay viewer is not implemented.'},
        'precompute_answer':'Cache the same-video/same-input forward result for instant later viewing; reuse a strictly matching background and compiled structure for later videos. Different video input still requires new forward simulation.',
        'required_stage_timing':['media decode','ISET cold compute versus validated cache reuse','graph assembly and backend setup',
            'strict background restore','0.5s network advance plus1ms ROI recording','checkpoint/result serialization and disk writes'],
        'previous_v7_20ms':'Static controlled orange PNG, fixed gaze, one100us real ISET expected-rate capture repeated; two20ms background/stimulus branches.668.282s whole delivery included setup/restores/save, not a video throughput measurement.',
        'execution_started':False,'long_background_started':False,'default_changed':False}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'blockers':blockers,'native_rate_buffer_GiB':buffer/1024**3,
        'recorded_state_MiB':report['recording_proposal']['raw_internal_state_MiB'],'active_optical_tiles':tiles,
        'example_50fps_framewise_tile_jobs':25*tiles,'report':str(output.relative_to(ROOT))},indent=2))


if __name__=='__main__':main()
