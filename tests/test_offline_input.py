import json,io,zipfile
from types import SimpleNamespace
import numpy as np
import pytest
from retina_sim.contracts import CellTable,Quantity,ContractError
from retina_sim.spatial import Population,InputChannel
from retina_sim.assembly import PortSpec,TimeContract,ExternalInput
from retina_sim.runtime_profiles import LightContext
from retina_sim.offline_input import load_offline_inputs,rebase_offline_inputs,OfflineInputError,SCHEMA
from retina_sim.photometry import (photoisomerization_window_mean_rate,photon_irradiance_to_energy,
    integrate_cone_photoisomerization_rate,Aperture,QuantumEfficiencyProfile)

TAG='synthetic_fixture';REF='fixture_owned_RF_reference'
def q(v,u,m,frame=None):return Quantity(v,u,m,TAG,frame)

@pytest.fixture
def fixture():
    pops={}
    for pid,kind in [('cones','cone'),('rod','rod')]:
        ids=(pid+':a',pid+':b');xy=q([[.34,0],[.35,.001]],'mm','functional_position','functional_visual_field')
        cells=CellTable(ids,(kind,)*2,('L','M') if kind=='cone' else ('rod',)*2,('fixture',)*2,
            ('N',)*2,('nasal',)*2,None,xy,'mm','functional_visual_field',
            q([.34,.35],'mm','global_eccentricity','functional_visual_field'),('fixture',)*2,(TAG,)*2)
        pops[pid]=Population(pid,cells,('center',)*2,receptive_field_positions=xy)
    components={
        'outer_native':SimpleNamespace(inputs={'light':PortSpec('light','td','native_retinal_illuminance',pops['cones'].cells.cell_id,TAG)},
            time_contract=TimeContract('continuous'),mechanism='VanHateren2005NativeConeH1',populations=(pops['cones'],)),
        'rod':SimpleNamespace(inputs={'light':PortSpec('light','R*/s','photoisomerization_rate',pops['rod'].cells.cell_id,TAG)},
            time_contract=TimeContract('author_discrete',.0001),mechanism='Chen2024PrimateRodCurrent',populations=(pops['rod'],))}
    compiled=SimpleNamespace(populations=pops,components=components,spec=SimpleNamespace(external_inputs=(
        ExternalInput('native_light','outer_native','light'),ExternalInput('rod_light','rod','light')),gap_junctions=()),
        assembly_hash='synthetic_fixture',links=(),clock={'fixture':True})
    for cid,c in components.items():c.identity='synthetic_fixture:'+cid;c.native_model=None
    light=LightContext();arrays={};channels={}
    for name,pid,unit,stage,base in [('native_light','cones','td','retinal_illuminance',100.),('rod_light','rod','R*/window','photoisomerization',5.)]:
        arrays[name+'_xy']=pops[pid].cells.rf_position.values.copy();arrays[name+'_time']=np.array([0.,.0001,.0002])
        arrays[name+'_values']=np.array([[100.,105.],[110.,120.],[90.,100.]]) if unit=='td' else np.array([[.0005,.0007],[.0008,.001]])
        channels[name]={'population_id':pid,'cell_ids':list(pops[pid].cells.cell_id),'data_axes':['time','cell'],
            'coordinate_role':'rf','coordinate_unit':'mm','coordinate_frame':'functional_visual_field','reference_id':REF,
            'coordinate_registration':'fixture_project_RF_used_verbatim','positions_key':name+'_xy','values_key':name+'_values',
            'unit':unit,'input_stage':stage,'evidence_tag':TAG,'value_semantics':'expected',
            'mechanism_ownership':{'photopigment_capture':'consumer' if unit=='td' else 'upstream','phototransduction':'consumer','neural_noise':'consumer'},
            'photon_noise':{'owner':'none','model':'none'},'background':{'includes_background':True,'unit':'td' if unit=='td' else 'R*/s','value':base,'source_id':light.background_source},
            'prehistory':{'kind':'declared_uniform_background','stop_seconds':0.},
            'time':{'representation':'samples' if unit=='td' else 'integration_windows','reference':'stimulus_relative_seconds',
                'source_origin_seconds':3.2,'start_seconds':0.,'stop_seconds':.0002,'axis_key':name+'_time',
                'interpolation':'linear' if unit=='td' else 'zero_order_hold','source_grid_step_seconds':.0001}}
    manifest={'schema':SCHEMA,'version':'1.0','producer':{'name':'unit fixture','version':'1','source_id':'invented_software_fixture'},
        'background_context_hash':light.background_context_hash,'channels':channels}
    return compiled,light,manifest,arrays

def write(tmp_path,manifest,arrays):
    path=tmp_path/'input.npz';np.savez(path,manifest_json=np.array(json.dumps(manifest)),**arrays);return path

def load(tmp_path,fixture):
    compiled,light,manifest,arrays=fixture
    return load_offline_inputs(write(tmp_path,manifest,arrays),compiled,light,reference_id=REF)

def test_legacy_values_id_reorder_and_window_conservation(tmp_path,fixture):
    compiled,light,manifest,arrays=fixture
    original=arrays['native_light_values'].copy();counts=arrays['rod_light_values'].copy()
    for name in manifest['channels']:
        manifest['channels'][name]['cell_ids'].reverse()
        arrays[name+'_xy']=arrays[name+'_xy'][::-1];arrays[name+'_values']=arrays[name+'_values'][:,::-1]
    result=load(tmp_path,fixture);native=result.channels['native_light'];rod=result.channels['rod_light']
    legacy=InputChannel(0,.0002,lambda t:q([np.interp(t,arrays['native_light_time'],original[:,k]) for k in range(2)],'td','native_retinal_illuminance'),
        'td','native_retinal_illuminance','linear',TAG,{'old_input':True})
    for t in [0.,.00005,.0001,.00015,.0002]:np.testing.assert_allclose(native.at(t).values,legacy.at(t).values,rtol=0,atol=1e-13)
    for i,t in enumerate([.00005,.00015]):np.testing.assert_allclose(rod.at(t).values*.0001,counts[i],rtol=2e-15,atol=0)
    assert rod.input_contract['conversion']['integration_window_edges_seconds']==[0.,.0001,.0002]
    assert rod.input_contract['grid_step_seconds']==.0001 and rod.breakpoints_seconds==(.0001,)
    assert all(x['source_columns_reordered'] for x in result.source_summary['channels'])
    assert result.source_summary['channels'][0]['source_origin_seconds']==3.2
    for t in [float('nan'),float('inf'),-.01,.01]:
        with pytest.raises(ContractError):native.at(t)

@pytest.mark.parametrize('field,value,code',[
    ('input_stage','photocurrent','ALREADY_PHOTOTRANSDUCED'),('unit','pA','UNIT'),
    ('cell_ids',['cones:a','wrong'],'IDS'),('cell_ids',['cones:a','cones:a'],'IDS'),
    ('coordinate_role','opl_pedicle','COORDINATES'),('reference_id','different_eye','COORDINATES'),
    ('data_axes',['trial','time','cone'],'ARRAY'),('evidence_tag','measurement','EVIDENCE')])
def test_bad_native_contract_rejected(tmp_path,fixture,field,value,code):
    fixture[2]['channels']['native_light'][field]=value
    with pytest.raises(OfflineInputError,match='OFFLINE_'+code):load(tmp_path,fixture)

def test_absolute_iset_cone_rate_cannot_become_td(tmp_path,fixture):
    d=fixture[2]['channels']['native_light'];d['input_stage']='photoisomerization';d['unit']='R*/s'
    with pytest.raises(OfflineInputError,match='VH_TD_MAPPING_UNAVAILABLE'):load(tmp_path,fixture)

@pytest.mark.parametrize('failure,code',[
    ('negative','VALUES'),('coordinates','COORDINATES'),('noise','NOISE'),('ownership','MECHANISM_OWNERSHIP'),
    ('background','BACKGROUND'),('context','BACKGROUND'),('prehistory','PREHISTORY'),('version','VERSION'),
    ('channels','CHANNELS'),('nonfinite','ARRAY'),('clock','CLOCK'),('spacing','TIME'),('linear_counts','COUNT_SEMANTICS')])
def test_other_failures(tmp_path,fixture,failure,code):
    _,_,m,a=fixture;rod=m['channels']['rod_light']
    if failure=='negative':a['rod_light_values'][0,0]=-1
    elif failure=='coordinates':a['native_light_xy'][0,0]+=.01
    elif failure=='noise':rod['photon_noise']={'owner':'upstream','model':'Poisson'}
    elif failure=='ownership':rod['mechanism_ownership']['phototransduction']='upstream'
    elif failure=='background':rod['background']['value']=6.
    elif failure=='context':m['background_context_hash']='different_background'
    elif failure=='prehistory':rod['prehistory']['kind']='unknown_experimental_history'
    elif failure=='version':m['version']='2'
    elif failure=='channels':del m['channels']['rod_light']
    elif failure=='nonfinite':a['rod_light_values'][0,0]=np.nan
    elif failure=='clock':a['rod_light_time'][1]=.00015
    elif failure=='spacing':a['rod_light_time'][-1]=.0003;rod['time']['stop_seconds']=.0003
    elif failure=='linear_counts':rod['time']['interpolation']='linear'
    with pytest.raises(OfflineInputError,match='OFFLINE_'+code):load(tmp_path,fixture)

def test_sampled_counts_record_noise_without_resampling(tmp_path,fixture):
    rod=fixture[2]['channels']['rod_light'];rod['value_semantics']='sampled'
    rod['photon_noise']={'owner':'upstream','model':'exporter_declared_gaussian_rounded_count_approximation','realization_id':'known_fixture_trial_0'}
    fixture[3]['rod_light_values']=np.array([[1.,0.],[2.,3.]])
    result=load(tmp_path,fixture);np.testing.assert_array_equal(result.channels['rod_light'].at(.00005).values,[10000.,0.])
    assert result.channels['rod_light'].input_contract['photon_noise']==rod['photon_noise']
    fixture[3]['rod_light_values'][0,0]=1.5
    with pytest.raises(OfflineInputError,match='COUNT_SEMANTICS'):load(tmp_path,fixture)

def test_budget_and_pickle_rejected(tmp_path,fixture):
    compiled,light,m,a=fixture;path=write(tmp_path,m,a)
    with pytest.raises(OfflineInputError,match='BUDGET'):load_offline_inputs(path,compiled,light,reference_id=REF,max_bytes=10)
    a['unsafe_object']=np.array([{'not_data':1}],dtype=object)
    with pytest.raises(OfflineInputError,match='FORMAT'):load(tmp_path,fixture)

def test_forged_array_shape_rejected_before_allocation(tmp_path,fixture):
    header=io.BytesIO();np.lib.format.write_array_header_1_0(header,{'descr':'<f8','fortran_order':False,'shape':(10**12,)})
    path=tmp_path/'forged.npz'
    with zipfile.ZipFile(path,'w') as archive:archive.writestr('forged.npy',header.getvalue())
    with pytest.raises(OfflineInputError,match='BUDGET'):load_offline_inputs(path,fixture[0],fixture[1],reference_id=REF)

def test_rebase_reuses_background_and_existing_clock(tmp_path,fixture):
    compiled,light,_,_=fixture;bundle=load(tmp_path,fixture);origin=1.3-3e-14
    channels=rebase_offline_inputs(bundle,compiled,light,stimulus_origin_seconds=origin,grid_origin_seconds=0.)
    assert channels['rod_light'].breakpoints_seconds==(13001*.0001,)
    assert channels['rod_light'].input_contract['clock_roundoff_snapped_knots']==1
    np.testing.assert_array_equal(channels['native_light'].at(origin-.0001).values,[100.,100.])
    np.testing.assert_array_equal(channels['rod_light'].at(origin-.0001).values,[5.,5.])
    np.testing.assert_allclose(channels['rod_light'].at(origin+.00005).values,[5.,7.],rtol=2e-15,atol=0)
    assert channels['rod_light'].input_contract['integration_windows_unchanged']
    with pytest.raises(OfflineInputError,match='BACKGROUND'):
        rebase_offline_inputs(bundle,compiled,LightContext(rod_background_rstar_s=6.),stimulus_origin_seconds=origin,grid_origin_seconds=0.)

def test_rebased_end_covers_exact_gui_author_tick_without_extending_real_gaps(tmp_path,fixture):
    from retina_sim.network import NetworkStimulus
    from dataclasses import replace
    compiled,light,_,_=fixture;bundle=load(tmp_path,fixture);origin=2.1-1.21e-13
    channels=rebase_offline_inputs(bundle,compiled,light,stimulus_origin_seconds=origin,grid_origin_seconds=0.)
    requested_stop=21002*.0001
    assert all(c.stop_seconds==requested_stop for c in channels.values())
    state=SimpleNamespace(time_seconds=origin,link_delay_states={})
    NetworkStimulus(compiled,channels,state,requested_stop,'linear',TAG)
    np.testing.assert_array_equal(channels['native_light'].at(requested_stop).values,[90.,100.])
    channels['native_light']=replace(channels['native_light'],stop_seconds=requested_stop-.0001,breakpoints_seconds=())
    with pytest.raises(ContractError,match='coverage mismatch'):NetworkStimulus(compiled,channels,state,requested_stop,'linear',TAG)

def test_declared_decimal_duration_and_author_grid_endpoint_roundoff(tmp_path,fixture):
    _,_,manifest,arrays=fixture;axis=np.arange(13)*.0001
    for name,d in manifest['channels'].items():
        d['time']['stop_seconds']=.0012;arrays[d['time']['axis_key']]=axis.copy()
        arrays[d['values_key']]=np.full((13 if name=='native_light' else 12,2),100. if name=='native_light' else .0005)
    result=load(tmp_path,fixture)
    assert result.channels['native_light'].stop_seconds==.0012
    assert result.channels['rod_light'].input_contract['conversion']['integration_window_edges_seconds']==axis.tolist()
    np.testing.assert_allclose(result.channels['rod_light'].at(.00115).values,[5.,5.],rtol=3e-15,atol=0)
    manifest['channels']['native_light']['time']['stop_seconds']=.0011
    with pytest.raises(OfflineInputError,match='TIME'):load(tmp_path,fixture)

def test_nonuniform_count_mean_and_photon_energy_conserve():
    edges=q([0.,.001,.004],'s','integration_window_edges');counts=q([[.3,.6],[.9,1.8]],'R*/window','expected_photoisomerization_count')
    rate=photoisomerization_window_mean_rate(counts,edges)
    np.testing.assert_allclose(rate.values*np.diff(edges.values)[:,None],counts.values,rtol=2e-15,atol=0)
    photons=q([[[2.,3.]],[[4.,5.]]],'photons/(s m^2 nm)','retinal_spectral_photon_irradiance');wave=q([400.,700.],'nm','wavelength')
    energy=photon_irradiance_to_energy(photons,wave)
    aperture=Aperture('dimensionless_mask',q([[1.],[1.]],'dimensionless','aperture_mask'),q([[2.],[2.]],'m^2','spatial_quadrature_weight'),q([2.,2.],'m^2','effective_collection_area'),'fixture','fixture')
    efficiency=QuantumEfficiencyProfile(q([.5,.25],'dimensionless','photoisomerization_efficiency'),'fixture','fixture',('fixture_quantum_efficiency',),(),'Fixture only, no animal calibration')
    result=integrate_cone_photoisomerization_rate(energy,wave,q([10.,20.],'nm','spectral_quadrature_weight'),aperture,efficiency)
    np.testing.assert_allclose(result.values,np.sum(photons.values[:,0,:]*[.5,.25]*[10.,20.],axis=1)*2,rtol=2e-15,atol=0)

def test_new_geometry_requires_actual_binding_and_rechecks_rebase(tmp_path,fixture):
    from dataclasses import replace
    from retina_sim.geometry_identity import compiled_geometry_identity
    compiled,light,manifest,_=fixture
    legacy=load(tmp_path,fixture)
    assert legacy.source_summary['geometry_binding_status']=='LEGACY_CHANNEL_BINDINGS_ONLY'
    compiled.input_geometry_binding_required=True
    with pytest.raises(OfflineInputError,match='OFFLINE_GEOMETRY'):load(tmp_path,fixture)
    with pytest.raises(OfflineInputError,match='OFFLINE_GEOMETRY'):
        rebase_offline_inputs(legacy,compiled,light,stimulus_origin_seconds=0.,grid_origin_seconds=0.)
    manifest['geometry_identity']=compiled_geometry_identity(compiled)
    bound=load(tmp_path,fixture)
    assert bound.source_summary['geometry_binding_status']=='ACTUAL_TYPED_GEOMETRY_BOUND'
    old=compiled.populations['cones'];rf=old.receptive_field_positions
    # Same IDs and RF, different typed OPL geometry must still fail.
    compiled.populations['cones']=replace(old,pedicle_positions=q(rf.values+.001,'mm','pedicle_position','OPL'))
    with pytest.raises(OfflineInputError,match='OFFLINE_GEOMETRY'):load(tmp_path,fixture)
    with pytest.raises(OfflineInputError,match='OFFLINE_GEOMETRY'):
        rebase_offline_inputs(bound,compiled,light,stimulus_origin_seconds=0.,grid_origin_seconds=0.)
