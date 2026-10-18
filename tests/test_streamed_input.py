from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from types import SimpleNamespace
from scipy.io import savemat,loadmat
from retina_platform.config import RunConfig
from retina_platform.media import MediaSource
from retina_platform.backend import _MediaChannels
from retina_platform.streamed_disk_frontend import exposure_index, save_shard
from retina_sim.contracts import ContractError
from retina_sim.disk_graph_storage import file_sha
from retina_sim.offline_input import _read_channel, OfflineInputs, rebase_offline_inputs
from retina_sim.isetbio_input import REFERENCE_SCHEMA, reference_equivalent_drive
from retina_sim.streamed_input import RowStore, held_channel
from test_offline_input import fixture, REF

ROOT = Path(__file__).resolve().parents[1]


def reference():
    return {'schema':REFERENCE_SCHEMA,'profile_id':'software_storage_QA','source_id':'synthetic_QA',
        'reference_conditions':'Synthetic rates only; storage equivalence, not optics validation',
        'scope':'engineering_fixture','evidence_tag':'synthetic_fixture','capture_scope_sha256':'a'*64,
        'reference_scene_sha256':'b'*64,'reference_capture_sha256':'c'*64,
        'reference_luminance_cd_m2':100/(np.pi*1.5**2),'pupil_diameter_mm':3.,
        'reference_rates_rstar_s_LMS':{'L':123.,'M':271.,'S':37.}}


def make_native(fixture, edges, counts):
    compiled,light,manifest,arrays = fixture
    d = manifest['channels']['native_light']
    d.update(unit='R*/window',input_stage='reference_equivalent_cone_drive',
        capture_types=['L','M'],reference_profile=reference(),capture_scope_sha256='a'*64,
        capture_pupil_diameter_mm=3.,capture_scope='engineering_fixture')
    d['mechanism_ownership']['photopigment_capture'] = 'upstream'
    d['time'].update(stop_seconds=float(edges[-1]),representation='integration_windows',interpolation='zero_order_hold')
    arrays['native_light_time'],arrays['native_light_values'] = edges,counts
    channel,_ = _read_channel(arrays,d,compiled.spec.external_inputs[0],compiled,light,REF)
    _,bridge = reference_equivalent_drive(np.zeros((1,2)),['L','M'],reference(),capture_scope_sha256='a'*64)
    alpha = [bridge['alpha_rstar_s_per_reference_td_LMS'][kind] for kind in ('L','M')]
    return channel,alpha


def test_all_qa_video_bins_boundaries_and_rebase_exact(tmp_path,fixture):
    """Actual existing decoded QA frames; synthetic counts isolate storage only."""
    compiled,light,manifest,arrays = fixture
    config = RunConfig(media_path=str(ROOT/'examples/platform_media/synthetic_rgb_50fps.avi'),
        duration_seconds=.0413,warmup_seconds=0.,execution_profile='atlas_population_revision_v1')
    edges = np.arange(414)*.0001
    with MediaSource(config.media_path) as media:
        exposures,mapping = exposure_index(config,media,lambda t:np.zeros(2),edges)
        assert [row[1] for row in exposures] == [0,1,2]
        counts = [np.asarray([[media.frame_at(t).mean()/100.,media.frame_at(t).std()/100.+.1]])
                  for t,_,_ in exposures]
        shards = [save_shard(tmp_path,'counts_%d.npy'%i,value) for i,value in enumerate(counts)]
        expanded = np.vstack([counts[i][0] for i,_ in mapping])
        eager,alpha = make_native(fixture,edges,expanded)
        store = RowStore(tmp_path,shards,mapping,2,cache_bytes=32)
        streamed = held_channel(eager,edges,store,counts=True,alpha=alpha)
        rod_source = _MediaChannels(config,compiled,media,lambda t:np.zeros(2),0.)
        original_rods = rod_source.window(0.,config.duration_seconds)['rod_light']
        rod_values = np.stack([original_rods.at(float(t)).values for t in edges])
        rod_decl = manifest['channels']['rod_light']
        rod_decl.update(unit='R*/s',value_semantics='deterministic')
        rod_decl['time'].update(stop_seconds=config.duration_seconds,representation='samples')
        arrays['rod_light_values'],arrays['rod_light_time'] = rod_values,edges
        rod_eager,_ = _read_channel(arrays,rod_decl,compiled.spec.external_inputs[1],compiled,light,REF)
        rod_map,rod_shards = np.empty((len(edges),2),dtype=np.int64),[]
        for lo in range(0,len(edges)-1,20):
            hi = min(len(edges)-1,lo+20)
            chunk = rod_source.window(float(edges[lo]),float(edges[hi]))['rod_light']
            values = np.stack([chunk.at(float(t)).values for t in edges[lo:hi+1]])
            # Chunking does not reset or alter the original effective rod calculation.
            np.testing.assert_array_equal(values,rod_values[lo:hi+1])
            index = len(rod_shards)
            rod_shards.append(save_shard(tmp_path,'rods_%d.npy'%index,values))
            rod_map[lo:hi+1,0],rod_map[lo:hi+1,1] = index,np.arange(hi-lo+1)
    rods = held_channel(rod_eager,edges,RowStore(tmp_path,rod_shards,rod_map,2,cache_bytes=672))
    times = sorted(set(edges.tolist()+((edges[:-1]+edges[1:])/2).tolist()+
        np.nextafter(edges[1:-1],-np.inf).tolist()+np.nextafter(edges[1:-1],np.inf).tolist()))
    for t in times:
        for old,new in ((eager,streamed),(rod_eager,rods)):
            np.testing.assert_array_equal(old.at(t).values.view(np.uint64),new.at(t).values.view(np.uint64))
    assert streamed.breakpoints_seconds == eager.breakpoints_seconds
    assert rods.breakpoints_seconds == rod_eager.breakpoints_seconds
    assert store.peak_resident_bytes <= 32
    # This catches the tempting but incorrect count/nominal-dt preconversion.
    assert np.any(np.diff(edges) != .0001)
    summaries = {'background_context_hash':light.background_context_hash,'geometry_identity':None}
    before = rebase_offline_inputs(OfflineInputs({'native_light':eager,'rod_light':rod_eager},summaries),
        compiled,light,stimulus_origin_seconds=2.1-1.21e-13,grid_origin_seconds=0.)
    after = rebase_offline_inputs(OfflineInputs({'native_light':streamed,'rod_light':rods},summaries),
        compiled,light,stimulus_origin_seconds=2.1-1.21e-13,grid_origin_seconds=0.)
    for name in before:
        assert before[name].breakpoints_seconds == after[name].breakpoints_seconds
        for t in (2.0999,2.1,2.11999999999,2.12,2.1413):
            np.testing.assert_array_equal(before[name].at(t).values.view(np.uint64),after[name].at(t).values.view(np.uint64))


def test_gaze_is_part_of_exposure_key_and_original_grid_retained(tmp_path):
    cfg = RunConfig(duration_seconds=.0004)
    media = type('Media',(),{'info':{'kind':'video','fps':50.,'frame_count':6}})()
    edges = np.arange(5)*.0001
    exposures,mapping = exposure_index(cfg,media,lambda t:np.array([t,0.]),edges)
    assert len(exposures) == 4 and mapping == [[0,0],[1,0],[2,0],[3,0]]
    assert [x[0] for x in exposures] == edges[:-1].tolist()


@pytest.mark.parametrize('failure',['hash','dtype','negative','map','outside','budget'])
def test_shards_fail_closed(tmp_path,failure):
    shard = save_shard(tmp_path,'counts.npy',np.array([[1.,2.]]))
    mapping = [[0,0]]
    budget = 64
    if failure == 'hash':
        shard['sha256'] = '0'*64
    elif failure in ('dtype','negative'):
        np.save(tmp_path/'counts.npy',np.array([[1.,-2.]]) if failure=='negative' else np.array([[1,2]],dtype='int64'))
        shard['sha256'] = file_sha(tmp_path/'counts.npy')
    elif failure == 'map':
        mapping = [[0,1]]
    elif failure == 'outside':
        shard['path'] = '../outside.npy'
    elif failure == 'budget':
        budget = 8
    with pytest.raises(ContractError):
        RowStore(tmp_path,[shard],mapping,2,cache_bytes=budget)


def exercise_frontend_pair(tmp_path,fixture,monkeypatch,real_iset=False):
    """Real preparation/binding/readers; deterministic mock optical values only."""
    import retina_platform.disk_frontend as old
    import retina_platform.streamed_disk_frontend as new
    from retina_sim.streamed_input import load_streamed_input
    root = tmp_path
    for relative in ('tools/export_disk_mixed_reference.m','tools/export_disk_mixed_batch.m',
                     'retina_sim/disk_input.py','retina_platform/disk_frontend.py',
                     'retina_sim/streamed_input.py'):
        path = root/relative;path.parent.mkdir(exist_ok=True,parents=True)
        path.write_bytes((ROOT/relative).read_bytes())
    compiled = fixture[0]
    for c in compiled.components.values():
        c.outputs = {};c.parameters = SimpleNamespace(parameter_hash='fixture')
        c.binding_options_hash=c.implementation_hash='fixture'
    compiled.geometry_source_binding = 'fixture'
    compiled.global_disk_circuit = SimpleNamespace(binding={'sha256':'fixture'},
        record={'reference_id':REF,'tables':{'cones':{'binding':'cone_fixture'},'rods':{'binding':'rod_fixture'}}})
    compiled.global_disk_view = {'selected_rows':{'cones':np.array([0,1]),'rods':np.array([0,1])}}
    compiled.global_spatial_asset = SimpleNamespace(binding='fixture_view')
    cone = SimpleNamespace(binding='cone_fixture',coordinates={'rf':compiled.populations['cones'].cells.rf_position.values},
        extras={'subtype':np.array([b'L',b'M']),'aperture_diameter_um':np.array([2.,2.])})
    compiled.global_disk_circuit.table = lambda name:cone
    np.save(root/'rows.npy',np.array([0,1]))
    plan = {'cone_table':'cone_fixture','regenerate_on_ROI':False,
        'exporter_sha256':file_sha(root/'tools/export_disk_mixed_reference.m'),
        'reference':{'definition_sha256':'fixture','reference_diameter_um':2.,'blur_um':2.},
        'tiles':[{'global_rows':{'path':'rows.npy','sha256':file_sha(root/'rows.npy')},
                  'scene_samples':193,'center_mm':[.35,0.],'tile_index':[0,0]}]}
    (root/'plan.json').write_text(json.dumps(plan),encoding='utf8')
    (root/'configuration.json').write_text(json.dumps({'profile_id':'atlas_isetbio_geometry_v1',
        'input_plan':{'path':'plan.json','sha256':file_sha(root/'plan.json')}}),encoding='utf8')
    if real_iset:
        profile,sources=old.source_signature(ROOT)
    else:
        profile,sources={'profile_id':'iset_lcd_apple_mw_v1'},{}
    monkeypatch.setattr(old,'source_signature',lambda root:(profile,dict(sources)))
    monkeypatch.setattr(new,'source_signature',lambda root:(profile,dict(sources)))
    calls=[]
    def export(root,profile,jobs,directory,on_status,on_boundary):
        calls.append(len(jobs))
        for job in jobs:
            c=json.loads(Path(job['configuration']).read_bytes())
            data=loadmat(c['input_mat']);n=len(data['types'].reshape(-1))
            types=data['types'].reshape(-1).astype(int)-1
            qref=np.array([123.,271.,37.]);dt=np.diff(data['window_edges_seconds'].reshape(-1))[0]
            frames=data['rgb_frames'];unique=np.array([frames[:,:,:,k].mean() for k in range(frames.shape[3])])
            counts=np.stack([qref[types]*dt*(1+unique[int(i)-1]) for i in data['frame_indices'].reshape(-1)])
            meta={'schema':'approved_mixed_capture_v1','request_signature':c['request_signature'],
                'global_reference_definition_sha256':c['global_reference']['definition_sha256'],
                'fixed_radiance_gain':1.,'reference_energy_spectrum_hex':['fixture'],
                'reference_luminance_cd_m2':100/(np.pi*1.5**2)}
            savemat(job['output'],{'manifest_json':json.dumps(meta),'expected_counts':counts,
                'background_counts':(qref[types]*dt)[None,:],'q_ref_LMS':qref[None,:],
                'window_edges_seconds':data['window_edges_seconds'],
                'diameters_um':data['frozen_apertures_um'],'fixed_radiance_gain_numeric':np.array([[1.]])})
            Path(job['output']+'.ready').touch()
    if not real_iset:
        monkeypatch.setattr(old,'_run_batch',export);monkeypatch.setattr(new,'_run_batch',export)
    cfg=RunConfig(media_path=str(ROOT/'examples/platform_media/synthetic_rgb_50fps.avi'),duration_seconds=.0203,
        warmup_seconds=0.,execution_profile='atlas_population_revision_v1',circuit_configuration_path='configuration.json')
    other=replace(cfg)
    with MediaSource(cfg.media_path) as media:
        eager=old.DiskMixedReferenceFrontend(cfg,compiled,media,lambda t:np.zeros(2),root,input_path=root/'eager.npz')
        streamed=new.StreamedDiskMixedReferenceFrontend(other,compiled,media,lambda t:np.zeros(2),root,input_path=root/'stream.json')
    if not real_iset:
        assert calls==[1,1,2]  # reference, eager tile, one batch containing two streamed exposures
    assert cfg.frontend_signature==other.frontend_signature
    assert cfg.light_context().background_context_hash==other.light_context().background_context_hash
    reloaded,stores=load_streamed_input(streamed.input_path,compiled,other.light_context(),reference_id=REF)
    edges=np.arange(204)*.0001
    for name,channel in eager.bundle.channels.items():
        for t in np.concatenate((edges,(edges[:-1]+edges[1:])/2,np.nextafter(edges[1:-1],-np.inf),np.nextafter(edges[1:-1],np.inf))):
            np.testing.assert_array_equal(channel.at(float(t)).values.view(np.uint64),
                reloaded.channels[name].at(float(t)).values.view(np.uint64))
    for source in (eager,streamed):source.set_origin(2.1-1.21e-13,0.)
    for name in eager.channels:
        assert eager.channels[name].breakpoints_seconds==streamed.channels[name].breakpoints_seconds
        for t in (2.0999,2.1,2.12,2.1203):
            np.testing.assert_array_equal(eager.channels[name].at(t).values.view(np.uint64),streamed.channels[name].at(t).values.view(np.uint64))
    bad=json.loads(streamed.input_path.read_bytes());bad['global_graph']={'sha256':'other'}
    bad_path=streamed.input_path.with_name('bad_graph.json');bad_path.write_text(json.dumps(bad),encoding='utf8')
    with pytest.raises(ContractError,match='graph'):
        load_streamed_input(bad_path,compiled,other.light_context(),reference_id=REF)
    return {'status':'ALL_VALUES_AND_BOUNDARIES_BITWISE_EQUAL','real_ISET':real_iset,
        'duration_seconds':cfg.duration_seconds,'input_windows':len(edges)-1,'cones':2,'rods':2,
        'unique_optical_exposures':len(stores['native_light'].records),'source_video':cfg.media_path,
        'scope':'Synthetic QA geometry/video; input storage equivalence only, no neural or physiological acceptance',
        'eager_packet':str(root/'eager.npz'),'stream_manifest':str(streamed.input_path),
        'frontend_signature_equal':True,'background_context_equal':True}


def test_complete_frontend_packet_batch_and_reload_matches_eager(tmp_path,fixture,monkeypatch):
    result=exercise_frontend_pair(tmp_path,fixture,monkeypatch)
    assert result['status']=='ALL_VALUES_AND_BOUNDARIES_BITWISE_EQUAL'


def test_video_recovery_retains_uncommitted_results(tmp_path):
    from tools.run_video_05s import archive_uncommitted_windows
    directory=tmp_path/'windows';directory.mkdir()
    for index in range(4):
        (directory/('window_%08d.npz'%index)).write_bytes(bytes([index]))
    archive_uncommitted_windows(tmp_path,2)
    assert len(list(directory.glob('window_*.npz')))==2
    archive=next(tmp_path.glob('recovered_uncommitted_*'))
    assert (archive/'window_00000002.npz').read_bytes()==b'\x02'
    assert (archive/'window_00000003.npz').read_bytes()==b'\x03'
    archive_uncommitted_windows(tmp_path,2)
    with pytest.raises(ValueError):archive_uncommitted_windows(tmp_path,3)


def test_video_checkpoint_writer_retains_original_full_protocol(tmp_path):
    from examples.full_network_fixture import full_fixture
    from retina_sim.hybrid import HybridRunner
    from tools.run_video_05s import checkpoint_file
    compiled,_,_=full_fixture(ROOT);runner=HybridRunner(compiled,compute_backend='python')
    state=runner.initialize();state.input_position.update(phase='media_stimulus',recorded_window_count=0)
    path=tmp_path/'full.json';checkpoint_file(runner,state,path)
    # Whitespace differs from the public dumps default; all nested payloads and
    # hashes retain the original public protocol exactly.
    assert json.loads(path.read_bytes())==json.loads(runner.checkpoint(state))
    restored=runner.restore(path.read_text(encoding='utf8'))
    assert restored.input_position==state.input_position
    for a,b in zip(state.ordered_state_arrays,restored.ordered_state_arrays):
        np.testing.assert_array_equal(a.view(np.uint64),b.view(np.uint64))
    from retina_sim.background import constant_background_channels
    from retina_sim.runtime_profiles import LightContext
    from retina_sim.recording import RecordingPlan,NPZWindowRecorder
    from retina_sim.recording_indices import IndexedRecordingPlan
    from retina_platform.results import ResultStore
    from tools.run_video_05s import state_manifest
    selected=frozenset(gid for pop in compiled.populations.values() for gid in pop.cells.cell_id)
    plan=IndexedRecordingPlan(RecordingPlan(ports=(),states=None,cell_ids=selected))
    channels=constant_background_channels(compiled,LightContext(),.002,
        interpolation=state.solver_config.input_interpolation)
    _,result=runner.advance(state,channels,.002,sample_times=np.array([0.,.001,.002]),recording_plan=plan)
    (tmp_path/'manifest.json').write_text(json.dumps({'schema':'retina-platform-1',
        'components':state_manifest(compiled,runner,plan)}),encoding='utf8')
    NPZWindowRecorder(tmp_path/'windows').consume(result)
    store=ResultStore(tmp_path)
    for name,value in result.outputs_with_units.items():
        if name=='time':continue
        info=store.port_info(name)
        assert info['unit']==value.unit and len(info['cell_ids'])==value.values.shape[0]
        times,values=store.history(name,0)
        np.testing.assert_array_equal(times,[0.,.001,.002])
        np.testing.assert_array_equal(values,value.values[0])
