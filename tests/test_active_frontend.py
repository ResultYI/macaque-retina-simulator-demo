from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pytest
from scipy.io import savemat
from PIL import Image
from retina_platform.config import RunConfig
from retina_platform.backend import _MediaChannels
from retina_platform.media import MediaSource
from retina_platform.iset_frontend import MixedReferenceFrontend,digest,read_capture,PROFILE
from retina_sim.background import constant_background_channels
from retina_sim.contracts import ContractError
from retina_sim.offline_input import load_offline_inputs
from test_offline_input import fixture,REF

ROOT=Path(__file__).resolve().parents[1]

def test_unique_default_and_historical_rgb_run_rejected(tmp_path):
    image=tmp_path/'input.png';Image.new('RGB',(8,8),(128,128,128)).save(image)
    cfg=RunConfig.project_default(ROOT);cfg.media_path=str(image)
    assert cfg.rgb_adapter==PROFILE and cfg.background_checkpoint_path==''
    cfg.validate()
    for old in ('contrast_reference_v1','legacy_rgb_proxy_v1'):
        with pytest.raises(ValueError,match='历史结果'):replace(cfg,rgb_adapter=old).validate()
    with pytest.raises(ValueError,match='100 td'):replace(cfg,native_background_td=200.).validate()

def test_per_cell_capture_background_is_preserved_and_signed(fixture):
    compiled,light,_,_=fixture
    actual=replace(light,native_background_values=(80.,120.))
    assert actual.background_context_hash!=light.background_context_hash
    channels=constant_background_channels(compiled,actual,.001)
    np.testing.assert_array_equal(channels['native_light'].at(.0002).values,[80.,120.])
    np.testing.assert_array_equal(channels['rod_light'].at(.0002).values,[5.,5.])
    with pytest.raises(ContractError,match='IDs/count'):
        constant_background_channels(compiled,replace(light,native_background_values=(100.,)),.001)

def test_active_rgb_helper_only_preserves_original_rod_mapping(tmp_path,fixture):
    image=tmp_path/'input.png';Image.new('RGB',(8,8),(220,120,30)).save(image)
    compiled=fixture[0];cfg=RunConfig(media_path=str(image),duration_seconds=.0002,warmup_seconds=0.,execution_profile='atlas_population_revision_v1')
    with MediaSource(image) as media:
        active=_MediaChannels(cfg,compiled,media,lambda t:np.zeros(2),0.).window(0.,.0002)
        historical=_MediaChannels(replace(cfg,rgb_adapter='contrast_reference_v1'),compiled,media,lambda t:np.zeros(2),0.).window(0.,.0002)
    assert set(active)=={'rod_light'}
    for t in (0.,.00005,.0001,.0002):np.testing.assert_array_equal(active['rod_light'].at(t).values,historical['rod_light'].at(t).values)

def capture_file(tmp_path):
    path=tmp_path/'capture.mat'
    arrays={'manifest_json':json.dumps({'schema':'approved_mixed_capture_v1','request_signature':'known_request'}),
        'expected_counts':np.array([[1.,2.],[2.,4.]]),'background_counts':np.array([[.1,.2]]),
        'q_ref_LMS':np.array([[100.,200.,300.]]),'window_edges_seconds':np.array([[0.,.0001,.0002]]),
        'diameters_um':np.array([[2.,3.]])}
    savemat(path,arrays);return path,arrays

def test_mat_identity_values_and_object_failures(tmp_path):
    path,arrays=capture_file(tmp_path);raw,_=read_capture(path,'known_request')
    assert raw['expected_counts'].shape==(2,2)
    with pytest.raises(ContractError,match='identity'):read_capture(path,'old_RGB_request')
    arrays['expected_counts'][0,0]=-1.;savemat(path,arrays)
    with pytest.raises(ContractError,match='rates'):read_capture(path,'known_request')
    arrays['expected_counts']=np.array([[{}]],dtype=object);savemat(path,arrays)
    with pytest.raises(ContractError,match='variables/types'):read_capture(path,'known_request')

@pytest.mark.parametrize('second_rod_background',[5.,10.],ids=['same_config_reuse','changed_rod_background'])
def test_run_owned_input_snapshot_survives_shared_capture_reuse(tmp_path,fixture,second_rod_background):
    compiled=fixture[0];image=tmp_path/'input.png';Image.new('RGB',(8,8),(220,120,30)).save(image)
    shared=tmp_path/'capture_cache';shared.mkdir();capture_file(shared)
    (shared/'input.mat').write_bytes(b'synthetic reference scene identity')
    legacy=shared/'input.npz';legacy.write_bytes(b'preserved historical shared packet')
    raw,_=read_capture(shared/'capture.mat','known_request');edges=raw['window_edges_seconds'].reshape(-1)
    meta={'reference_luminance_cd_m2':100./(np.pi*1.5**2),'fixed_radiance_gain':1.}
    pop=compiled.populations['cones'];ids=list(pop.cells.cell_id);types=list(pop.cells.subtype)
    def bind(frontend,cfg,media):
        frontend._bind(cfg,compiled,media,lambda t:np.zeros(2),pop,ids,types,REF,raw,meta,edges,
            'known_request','a'*64,{'synthetic_fixture':'b'*64})
    runs=[]
    with MediaSource(image) as media:
        for name,rod_background in [('first',5.),('second',second_rod_background)]:
            cfg=RunConfig(media_path=str(image),duration_seconds=.0002,warmup_seconds=0.,
                execution_profile='atlas_population_revision_v1',rod_background_rstar_s=rod_background)
            directory=tmp_path/name;directory.mkdir()
            frontend=MixedReferenceFrontend.__new__(MixedReferenceFrontend)
            frontend.config=cfg;frontend.compiled=compiled;frontend.root=ROOT
            frontend.directory=shared;frontend.input_path=directory/'input.npz';frontend.cache_reused=True
            bind(frontend,cfg,media);runs.append((frontend,cfg))
            if name=='first':first_bytes=frontend.input_path.read_bytes();first_sha=frontend.summary['source_sha256']
        first,cfg=runs[0];second,_=runs[1]
        with pytest.raises(FileExistsError):bind(first,cfg,media)
    assert first.input_path!=second.input_path
    assert first.summary['source_file']==str(first.input_path.resolve())
    assert second.summary['source_file']==str(second.input_path.resolve())
    assert first.input_path.read_bytes()==first_bytes and digest(first.input_path)==first_sha
    restored=load_offline_inputs(first.input_path,compiled,cfg.light_context(),reference_id=REF)
    assert restored.source_summary['source_sha256']==first_sha
    for channel in ('native_light','rod_light'):
        factor=second_rod_background/5. if channel=='rod_light' else 1.
        np.testing.assert_array_equal(second.bundle.channels[channel].at(.0001).values,
            first.bundle.channels[channel].at(.0001).values*factor)
    assert first.summary['producer']['capture_sha256']==second.summary['producer']['capture_sha256']
    assert first.summary['cache_reused'] and second.summary['cache_reused']
    assert legacy.read_bytes()==b'preserved historical shared packet'

def test_approved_profile_has_explicit_human_conventions():
    p=json.loads((ROOT/'configurations/iset_lcd_apple_mw_v1.json').read_bytes())
    assert p['optics']=='human mw' and p['pupil_diameter_mm']==3 and p['microns_per_degree']==223
    assert p['optical_density_LMS']==[.5,.5,.4] and p['macular_density']==.35 and p['lens_density']==1
    assert p['aperture_diameter_to_spacing']==.79 and p['reference_retinal_illuminance_td']==100

def test_background_checkpoint_rejects_old_frontend_context(tmp_path):
    from examples.full_network_fixture import full_fixture
    from retina_sim.background_checkpoint import BackgroundCheckpointStore
    from retina_sim.background import BackgroundPolicy
    from retina_sim.runtime_profiles import LightContext
    compiled,_,_=full_fixture(ROOT);runner=compiled.runner();state=runner.initialize();policy=BackgroundPolicy()
    old=LightContext();path=tmp_path/'old_background.json'
    BackgroundCheckpointStore(ROOT,runner,state,old,policy).save(path,state)
    count=len(compiled.components[compiled.spec.external_inputs[0].target_component].inputs[compiled.spec.external_inputs[0].target_port].cell_ids)
    new=replace(old,calibration='approved_mixed_reference:new',native_background_values=tuple(np.linspace(80.,120.,count)))
    store=BackgroundCheckpointStore(ROOT,runner,runner.initialize(),new,policy)
    with pytest.raises(ContractError,match='identity mismatch; no fallback'):store.load(path)

def test_candidate_background_binds_actual_geometry_without_advancing_time(tmp_path):
    from copy import copy
    from examples.full_network_fixture import full_fixture
    from retina_sim.background_checkpoint import BackgroundCheckpointStore
    from retina_sim.background import BackgroundPolicy
    from retina_sim.hybrid import HybridRunner
    compiled,_,_=full_fixture(ROOT);runner=HybridRunner(compiled,compute_backend='python')
    policy=BackgroundPolicy();light=RunConfig().light_context();state=runner.initialize()
    path=tmp_path/'legacy_background.json'
    legacy=BackgroundCheckpointStore(ROOT,runner,state,light,policy);legacy.save(path,state)
    assert 'typed_geometry_identity' not in legacy.binding
    compiled.input_geometry_binding_required=True
    store=BackgroundCheckpointStore(ROOT,runner,runner.initialize(),light,policy)
    with pytest.raises(ContractError,match='identity mismatch; no fallback'):store.load(path)
    path=tmp_path/'geometry_bound_t0.json';store.save(path,store.reference)
    restored,_=store.load(path)
    assert restored.time_seconds==0. and store.reference.time_seconds==0.
    # Mutation of a native spatial weight is not hidden by the assembly hash.
    changed=copy(compiled);changed.components=dict(compiled.components)
    outer=copy(changed.components['outer']);outer.native_model=copy(outer.native_model)
    outer.native_model.A=outer.native_model.A.copy();outer.native_model.A.data[0]*=.99
    changed.components['outer']=outer
    r=HybridRunner(changed,compute_backend='python')
    mismatch=BackgroundCheckpointStore(ROOT,r,r.initialize(),light,policy)
    assert mismatch.binding['typed_geometry_identity']!=store.binding['typed_geometry_identity']
    with pytest.raises(ContractError,match='identity mismatch; no fallback'):mismatch.load(path)
    compiled.components['outer']=outer
    with pytest.raises(ContractError,match='identity mismatch'):store.load(path)
    with pytest.raises(ContractError,match='identity mismatch'):store.save(tmp_path/'stale.json',store.reference)

def test_capture_cannot_be_relabelled_after_geometry_change(fixture):
    from dataclasses import replace
    from retina_sim.geometry_identity import compiled_geometry_identity
    from retina_sim.contracts import Quantity
    c=fixture[0];frontend=MixedReferenceFrontend.__new__(MixedReferenceFrontend)
    frontend.geometry_identity=compiled_geometry_identity(c)
    old=c.populations['cones'];rf=old.receptive_field_positions
    c.populations['cones']=replace(old,pedicle_positions=Quantity(rf.values,'mm','pedicle_position',rf.evidence_tag,'OPL'))
    with pytest.raises(ContractError,match='changed while capturing'):
        frontend._bind(None,c,None,None,None,None,None,None,None,None,None,None,None,None)

def test_historical_gui_view_keeps_active_frontend_and_uses_stimulus_origin():
    import tkinter as tk
    from types import SimpleNamespace
    from retina_platform.app import RetinaApp
    root=tk.Tk();root.withdraw();app=RetinaApp(root,ROOT);errors=[];app._error=lambda error:errors.append(str(error))
    try:
        old=RunConfig.project_default(ROOT).to_dict()
        old.update(rgb_adapter='contrast_reference_v1',media_path=str(ROOT/'examples/platform_media/synthetic_rgb_50fps.avi'),
            start_seconds=.0199,duration_seconds=.0002,warmup_seconds=0.)
        store=SimpleNamespace(manifest={'config':old,'runtime':{'stimulus_origin_seconds':1.3}},windows=[1],
            read_window=lambda i:{'time':np.array([1.3,1.3002]),'signals':{}})
        app._restore_result_input(store)
        assert not errors and app.config.rgb_adapter==PROFILE
        assert app.params['rgb_adapter'].get()=='LCD-Apple / human-MW'
        assert app.preview_var.get()==pytest.approx(.0201)
    finally:app.close()
