from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json
import shutil
import numpy as np
import pytest
from retina_sim.contracts import ContractError,Quantity
from retina_sim.runtime_profiles import LightContext,RuntimeProfile,frozen_packer_table
from retina_sim.mechanism_contracts import EvidenceValue,MechanismProfile,validate_mechanism_evidence,validate_population_evidence_domain
from retina_sim.spatial_profiles import bounded_matrix
from retina_sim.core_sources import load_core_registry

ROOT=Path(__file__).resolve().parents[1]

def test_profile_replaces_real_constants_and_checks_units():
    p=RuntimeProfile(ROOT)
    row=p.bind('effective_rod_membrane',{'C':(np.full(7,999.),'F','membrane_capacitance')})
    assert np.array_equal(row['C'][0],np.full(7,5e-12))
    with pytest.raises(ContractError,match='unit/type'):
        p.bind('effective_rod_membrane',{'C':(np.ones(7),'pF','membrane_capacitance')})

def test_packer_frozen_hash_is_independent_of_current_csv(tmp_path):
    for f in ['data/published_code/packer_cone_density.csv','data/published_code/packer_cone_density.manifest.json',
              'data/source_snapshots/isetmacaque_cf22ac9d/PackerConeRc.m']:
        dest=tmp_path/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    assert len(frozen_packer_table(tmp_path).rows)==60
    with (tmp_path/'data/published_code/packer_cone_density.csv').open('ab') as stream:stream.write(b'\n')
    with pytest.raises(ContractError,match='hash'):frozen_packer_table(tmp_path)

def test_identity_does_not_allow_dynamic_measurement_and_domain_is_enforced():
    registry=load_core_registry(ROOT)
    record=EvidenceValue('C',Quantity([5e-12],'F','membrane_capacitance','synthetic_fixture'),
        'measurement','VH2005_source_supplement','Extracted coefficient',{'cell_types':['cone']},'unverified')
    profile=MechanismProfile('invalid','M. mulatta',{},(record,),('VH2005_source_supplement',),'measurement')
    with pytest.raises(ContractError,match='primary measurement'):validate_mechanism_evidence(registry,profile)
    pop=SimpleNamespace(cells=SimpleNamespace(cell_type=('rod',)))
    with pytest.raises(ContractError,match='cell type'):validate_population_evidence_domain(profile,[pop])

def test_hard_support_reports_deficit_without_distant_nearest():
    W,audit=bounded_matrix([[0.,0.],[1.,0.]],[[0.,0.],[.5,0.]],3,support_mm=.01)
    assert W.toarray().tolist()==[[1.,0.],[0.,0.]]
    assert [r['missing_degree'] for r in audit]==[2,3]

@pytest.fixture(scope='module')
def changed_background_network():
    from examples.full_target_network import full_target_network
    light=LightContext(250.,2000.,15.)
    return full_target_network(ROOT,effective_cones=64,effective_rods=96,duration=.008,
        adaptation_seconds=.002,light_context=light,execution_profile='source_constrained_approximation')

def test_background_build_initialization_and_drive_share_context(changed_background_network):
    c,ch,a=changed_background_network
    options={s.component_id:s.options for s in c.spec.components}
    for component,key,value in [('outer_native','background_td',250.),('physical_OS','initial_background_rate',2000.),('rod','initial_background_rate',15.)]:
        assert np.all(options[component][key].values==value)
    for name,value in [('native_light',250.),('physical_cone_light',2000.),('rod_light',15.)]:
        assert np.all(ch[name].at(0.).values==value)
    assert 'NOT_steady_state_verified' in a['light_context']['adaptation_status']

def test_source_roi_has_unscaled_density_independent_hc_and_native_route(changed_background_network):
    c,ch,a=changed_background_network
    assert a['effective_density_scale']==1. and a['rod_geometry']['effective_density_scale']==1.
    assert abs(a['effective_cones']-a['cone_density_source_mass'])<=.5
    assert a['cone_route']=='native_shared_H1_approximation'
    assert not any(x.source_component=='physical_terminal' and x.target_component.endswith('BC_ON') for x in c.spec.links)
    hc=c.populations['HC']
    assert hc.cells.subtype.count('H1')==27 and hc.cells.subtype.count('H2')==9
    assert all(row['maximum_distance_mm'] is None or row['maximum_distance_mm']<=row['hard_support_mm']+1e-12
        for audit in a['connectivity_support'] for row in audit.get('rows',[]))
    assert len(c.components)==39
    for binding in a['regional_support']['component_bindings']:
        assert binding['source_ids']==list(c.components[binding['component_id']].source_ids)

def test_rgb_prerun_is_uniform_background(changed_background_network):
    from retina_platform.backend import _MediaChannels
    from retina_platform.config import RunConfig
    from retina_platform.media import MediaSource
    c,_,_=changed_background_network
    cfg=RunConfig(media_path=str(ROOT/'examples/platform_media/synthetic_rgb.png'),
        native_background_td=250.,cone_background_rstar_s=2000.,rod_background_rstar_s=15.)
    with MediaSource(cfg.media_path) as media:
        channels=_MediaChannels(cfg,c,media,lambda t:np.zeros(2)).window(0.,.001)
        for name,value in [('native_light',250.),('physical_cone_light',2000.),('rod_light',15.)]:
            assert np.all(channels[name].at(.0005).values==value)

def test_all_maddox_pairs_drive_actual_ca_activation_without_release_inference():
    from retina_sim.source_operators import maddox_ca_activation
    registry=load_core_registry(ROOT)
    records=json.loads((ROOT/'sources/maddox2024_activation_records.json').read_text(encoding='utf-8'))['records']
    for row in records:
        name='cell%d_hold%d' % (row['source_cell_ordinal'],row['holding_mV'])
        output=maddox_ca_activation(Quantity([row['Vhalf_mV']*.001],'V','physical_membrane_voltage','synthetic_fixture'),
            Quantity([0.],'V','activation_threshold_shift','synthetic_fixture'),registry,name)
        assert np.allclose(output.values,[.5],rtol=0,atol=1e-14)
        assert registry.mechanism_profiles['maddox_activation_'+name].records[0].evidence_class=='published_fit'
    with pytest.raises(ContractError,match='exact Maddox'):
        maddox_ca_activation(Quantity([-.03],'V','physical_membrane_voltage','synthetic_fixture'),
            Quantity([0.],'V','activation_threshold_shift','synthetic_fixture'),registry,'average')

def test_legacy_config_remains_preview():
    from retina_platform.config import RunConfig
    assert RunConfig().execution_profile=='retina_contact_geometry_effective_v1'
    old=RunConfig.from_dict({'duration_seconds':.006})
    assert (old.execution_profile,old.adaptation_policy)==('engineering_preview','legacy_first_frame')

def test_central_private_line_uses_parent_id_even_when_index_order_differs():
    from scipy import sparse
    from retina_sim.wiring import enforce_midget_parent_matrix
    W=sparse.csr_matrix([[1.,0.],[0.,1.]])
    actual=enforce_midget_parent_matrix(W,('BC:a','BC:b'),('BC:b','BC:a'),[1.,.5])
    assert actual.toarray().tolist()==[[0.,1.],[1.,0.]]
    with pytest.raises(ContractError,match='outside declared contact support'):
        enforce_midget_parent_matrix(W,('BC:a','BC:b'),('BC:b','BC:a'),[1.,.5],
            pre_positions=[[0.,0.],[1.,0.]],post_positions=[[0.,0.],[1.,0.]],support_mm=.04)

def test_local_initial_balances_use_actual_upstream_profiles(changed_background_network):
    from retina_sim.initialization import initial_event_rate
    c,_,a=changed_background_network
    rbc=c.components['rod_BC']
    profile=rbc.parameters.profile
    nu=initial_event_rate(profile)
    chemical=c.components['RBC_AII_AMPA']
    assert np.allclose(chemical.parameters.profile.quantity('event_rate0','events/s','expected_release_event_rate').values,nu[0])
    ac=c.components['local_feedforward_AC'].parameters.profile
    assert np.all(ac.quantity('baseline','dimensionless','input_baseline').values==.5)
    p=rbc.parameters.profile
    V=p.quantity('V0','V','initial_membrane_voltage').values
    gL=p.quantity('gL','S','leak_conductance').values
    EL=p.quantity('EL','V','leak_reversal').values
    assert np.allclose(gL*(EL-V)+(2e-9/17)*(0-V),0.,rtol=0,atol=1e-23)

def test_actual_initial_rhs_has_no_local_reference_jump(changed_background_network):
    from retina_sim.network import InputBundle
    c,channels,_=changed_background_network
    runner=c.runner();model=runner.model
    y=np.concatenate([q.values for q in model.initial_conditions()])
    inputs,_=model.evaluate(0.,y,InputBundle({name:channel.at(0.) for name,channel in channels.items()}))
    # Full AII/cone/gap equilibrium is deliberately unsolved. Check only the
    # independent or conditional closures constructed from real upstream ports.
    for cid in ('rod_membrane','individual_rod_RBC_contacts','rod_BC','RBC_AII_AMPA','physical_HC',
                'midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON'):
        derivatives=c.components[cid].derivative(0.,model.local_blocks(cid,y),inputs[cid])
        assert max(np.max(np.abs(q.values)) for q in derivatives)<1e-10,cid

def test_rgb_black_reference_and_outside_reference_use_the_declared_background(changed_background_network):
    from retina_platform.backend import _MediaChannels
    from retina_platform.config import RunConfig
    c,_,_=changed_background_network
    cfg=RunConfig(warmup_seconds=0.,duration_seconds=.001,reference_linear_value=.25,
        native_background_td=250.,cone_background_rstar_s=2000.,rod_background_rstar_s=15.)
    # Floating point decoded samples isolate adapter behavior from 8-bit rounding.
    class Media:
        info={'width':128,'height':128,'kind':'image'}
        last_timestamp_seconds=0.
        def __init__(self,value):self.value=value
        def frame_at(self,t):return np.full((128,128,3),self.value)
    srgb=.25**(1/2.4)*1.055-.055
    for pixel,pose,multiplier in [(0.,[-4.6/.223,0.],0.),(255*srgb,[-4.6/.223,0.],1.),(0.,[1e6,0.],1.)]:
        channels=_MediaChannels(cfg,c,Media(pixel),lambda t:np.asarray(pose)).window(0.,.001)
        for name,value in [('native_light',250.),('physical_cone_light',2000.),('rod_light',15.)]:
            assert np.allclose(channels[name].at(.0005).values,value*multiplier,rtol=1e-14,atol=1e-14)


def test_last_media_tick_covers_origin_plus_duration_roundoff():
    from types import SimpleNamespace
    from retina_platform.backend import _MediaChannels
    from retina_platform.config import RunConfig, AUTHOR_STEP
    cfg=RunConfig(warmup_seconds=0.,duration_seconds=.020)
    cells=SimpleNamespace(cell_id=('c0',),subtype=('L',),
        rf_position=SimpleNamespace(values=np.array([[.35,0.]])))
    compiled=SimpleNamespace(populations={'cones':SimpleNamespace(cells=cells)})
    class Media:
        info={'width':16,'height':16,'kind':'image'}
        last_timestamp_seconds=0.
        def frame_at(self,t):return np.full((16,16,3),128.)
    source=_MediaChannels(cfg,compiled,Media(),lambda t:np.zeros(2),.0002)
    # This endpoint is one ULP above the floating sum .0002 + .020.
    stop=202*AUTHOR_STEP
    channel=source.window(182*AUTHOR_STEP,stop)['native_light']
    assert channel.stop_seconds>=stop and np.isfinite(channel.at(stop).values).all()
    with pytest.raises(ValueError,match='outside declared'):
        source.window(stop,stop+AUTHOR_STEP)
