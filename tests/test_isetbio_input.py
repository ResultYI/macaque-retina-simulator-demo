import json
import numpy as np
import pytest
from scipy.io import savemat
from retina_sim.contracts import ContractError
from retina_sim.isetbio_input import reference_equivalent_drive,read_isetbio_capture,REFERENCE_SCHEMA
from retina_sim.offline_input import load_offline_inputs,OfflineInputError
from test_offline_input import fixture,write,REF

def reference():
    return {'schema':REFERENCE_SCHEMA,'profile_id':'fixed_test_reference','source_id':'synthetic_fixture',
        'reference_conditions':'Explicit constant test spectrum, same optics/pigment, one reference aperture per LMS class',
        'scope':'engineering_fixture','evidence_tag':'synthetic_fixture','capture_scope_sha256':'a'*64,
        'reference_scene_sha256':'b'*64,'reference_capture_sha256':'c'*64,
        'reference_luminance_cd_m2':100/np.pi,'pupil_diameter_mm':2.,
        'reference_rates_rstar_s_LMS':{'L':1000.,'M':500.,'S':250.}}

def test_fixed_class_conversion_preserves_cells_and_windows():
    rates=np.array([[1000.,2000.,500.,250.],[1100.,2200.,550.,275.]])
    drive,summary=reference_equivalent_drive(rates,['L','L','M','S'],reference(),capture_scope_sha256='a'*64)
    np.testing.assert_allclose(drive,[[100.,200.,100.,100.],[110.,220.,110.,110.]],rtol=2e-15)
    assert drive[0,1]/drive[0,0]==2 and summary['coefficient_policy'].startswith('fixed_LMS')
    assert not summary['physiological_calibration_claim']
    edges=np.array([0.,.001,.002]);counts=rates*np.diff(edges)[:,None]
    np.testing.assert_allclose(np.sum(drive*np.diff(edges)[:,None],axis=0),
        np.sum(counts,axis=0)/np.array([10.,10.,5.,2.5]),rtol=2e-15)

@pytest.mark.parametrize('change',[
    {'schema':'unversioned'}, {'capture_scope_sha256':'d'*64}, {'pupil_diameter_mm':None},
    {'reference_luminance_cd_m2':0.}, {'reference_capture_sha256':''},
    {'reference_rates_rstar_s_LMS':{'cone:a':1000.,'cone:b':500.}},
    {'reference_rates_rstar_s_LMS':{'L':1000.,'M':500.,'S':float('nan')}}])
def test_missing_or_cellwise_reference_rejected(change):
    profile=reference();profile.update(change)
    with pytest.raises(ContractError):reference_equivalent_drive([[1000.]],['L'],profile,capture_scope_sha256='a'*64)

def test_reference_bridge_reader_keeps_native_model_and_background(tmp_path,fixture):
    compiled,light,m,a=fixture;d=m['channels']['native_light']
    d.update(input_stage='reference_equivalent_cone_drive',unit='R*/window',reference_profile=reference(),
        capture_scope_sha256='a'*64,capture_types=['L','M'],capture_pupil_diameter_mm=2.,capture_scope='engineering_fixture')
    d['mechanism_ownership']['photopigment_capture']='upstream'
    d['time'].update(representation='integration_windows',interpolation='zero_order_hold')
    a[d['values_key']]=np.array([[.1,.05],[.11,.055]])
    result=load_offline_inputs(write(tmp_path,m,a),compiled,light,reference_id=REF)
    native=result.channels['native_light']
    np.testing.assert_allclose(native.at(.00005).values,[100.,100.],rtol=2e-15)
    np.testing.assert_allclose(native.at(.00015).values,[110.,110.],rtol=2e-15)
    assert native.unit=='td' and compiled.components['outer_native'].mechanism=='VanHateren2005NativeConeH1'
    assert native.input_contract['reference_bridge']['meaning']=='reference_equivalent_cone_drive_not_general_photopic_td'
    assert native.input_contract['background']['value']==light.native_background_td
    d['capture_types']=['M','L']
    with pytest.raises(OfflineInputError,match='REFERENCE'):load_offline_inputs(write(tmp_path,m,a),compiled,light,reference_id=REF)

def mat_capture(tmp_path,*,cones=1,windows=1):
    edges=(np.arange(windows+1)*.001).tolist();ids=['cone:'+str(i) for i in range(cones)]
    meta={'schema':'isetbio_cone_capture_v1','scope':'engineering_fixture','data_axes':['time','cell'],'unit':'R*/window','value_semantics':'expected',
        'isetbio_commit':'d2924c2fe1d41c7dc5af1006d8081735b08f1b35','isetcam_commit':'17146a9f0470cc89b58d583851f680187268506d',
        'capture_scope_sha256':'a'*64,'cones':{'cell_ids':ids,'types':['L']*cones},'time':{'window_edges_seconds':edges},
        'mechanism_ownership':{'optics':'ISETCam','photopigment_capture':'ISETBio','phototransduction':'consumer','photon_noise':'none','neural_noise':'consumer'}}
    arrays={'manifest_json':json.dumps(meta),'expected_counts':np.ones((windows,cones)),
        'window_edges_seconds':np.array(edges),'is_mm':np.ones((cones,2)), 'rf_mm':np.ones((cones,2)),
        'wave_nm':np.array([400.,500.,600.]),'retinal_photons':np.ones((3,4,3,windows)),'spatial_support_m':np.ones((3,4,2))}
    path=tmp_path/'capture.mat';savemat(path,arrays);return path,arrays

@pytest.mark.parametrize('cones,windows',[(1,1),(3,1),(1,2),(3,2)])
def test_mat_singleton_axes_are_preserved(tmp_path,cones,windows):
    path,_=mat_capture(tmp_path,cones=cones,windows=windows);result=read_isetbio_capture(path)
    assert result['counts'].shape==(windows,cones) and result['retinal_photons'].shape==(3,4,3,windows)

def test_mat_bad_shapes_counts_budget_and_objects(tmp_path):
    path,a=mat_capture(tmp_path)
    with pytest.raises(ContractError,match='budget'):read_isetbio_capture(path,max_bytes=2)
    a['expected_counts'][0,0]=-1.;savemat(path,a)
    with pytest.raises(ContractError,match='Count'):read_isetbio_capture(path)
    a['expected_counts']=np.array([[{}]],dtype=object);savemat(path,a)
    with pytest.raises(ContractError,match='numeric/string'):read_isetbio_capture(path)
