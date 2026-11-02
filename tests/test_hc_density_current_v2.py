"""Software contracts only: no cell generation, neural simulation or fitting."""
from pathlib import Path
import json
import numpy as np
import pytest
from scipy.integrate import quad
from retina_sim.contracts import ContractError,Quantity
from retina_sim.hc_current import JOINT_ID,current_hc,file_hash,require_current_hc_asset
from retina_sim.hc_joint_v2 import JointV2HCDensityProfile
from retina_sim.hc_mosaic import HCDensityProfile
from retina_sim.atlas_hc import AtlasHCProvider,make_hc_soma_density
from retina_sim.atlas_hc_master import FixedJointHCMaster,materialize_hc_master
from retina_sim.disk_hc_functional import eccentric_projection

ROOT=Path(__file__).resolve().parents[1]

def profile():return HCDensityProfile(ROOT)

def test_default_factory_and_current_atlas_call_v2():
    for p in (profile(),make_hc_soma_density(ROOT),AtlasHCProvider(ROOT).density):
        assert isinstance(p,JointV2HCDensityProfile) and p.profile_id==JOINT_ID
        assert p.metadata['current_hc_identity']==current_hc(ROOT)['identity']
        assert p.metadata['physiological_validation'] is False

def test_supplied_coefficients_constraints_and_count_without_refit():
    p=profile();np.testing.assert_array_equal(p.coefficients,[45245.30264390676,-48597.44237836773,3602.1397344609636])
    np.testing.assert_array_equal(p.lambdas,[-.8616128661131702,-3.112593835972845,-.03494994340253653])
    np.testing.assert_allclose(p.radial([0,.6]),[250,23000],rtol=0,atol=1e-8)
    assert abs(float(p.derivative(.6)))<1e-7 and p.derivative(0)!=0
    assert np.all(p.radial(np.linspace(0,1.338,101))>0)
    count=p.expected_disk_count(1.338)
    assert count==pytest.approx(113291.53030094731,rel=1e-12)
    assert count==pytest.approx(quad(lambda r:2*np.pi*r*float(p.radial(r)),0,1.338,epsabs=1e-7,epsrel=1e-12)[0],rel=1e-12)

def test_H1_H2_mass_and_explicit_soma_units():
    p=profile();xy=np.c_[np.linspace(0,1.338,101),np.zeros(101)];total=p.total(xy)
    np.testing.assert_array_equal(p.type_density(xy,'H1')+p.type_density(xy,'H2'),total)
    np.testing.assert_allclose(p.type_density(xy,'H2'),.2*total,rtol=1e-15)
    assert p.density_at_soma_eccentricity_deg(3.88)==p.radial(.86524)
    assert p.density_at_soma_eccentricity_deg(6)==p.radial(1.338)
    with pytest.raises(ContractError):p.density_at_soma_eccentricity_deg(6.00001)
    with pytest.raises(ContractError):p.total(Quantity([[.8,0]],'deg','anatomical_position','fixture','anatomical_retina'))
    with pytest.raises(ContractError):p.total(Quantity([[.8,0]],'mm','functional_position','fixture','functional_visual_field'))

def test_contact_rows_use_attachment_values_and_authorized_inner_C1_join():
    p=profile();rows=p.config['geometry']['rows']
    for row in rows:
        for kind in ('H1','H2'):
            assert float(p.contact_radius_mm(row['r_mm'],kind)*1000)==pytest.approx(row[kind+'_radius_um'],abs=2e-11)
    for kind,central in [('H1',.00804),('H2',.01222)]:
        assert p.contact_radius_mm(.58,kind)==central and p.contact_radius_derivative_mm(.58,kind)==0
        for boundary in (.58,.6):
            left=np.nextafter(boundary,0);right=np.nextafter(boundary,1)
            assert p.contact_radius_mm(left,kind)==pytest.approx(p.contact_radius_mm(right,kind),abs=1e-16)
            assert p.contact_radius_derivative_mm(left,kind)==pytest.approx(p.contact_radius_derivative_mm(right,kind),abs=1e-13)
        assert np.all(p.contact_radius_mm(np.linspace(.58,.6,101),kind)>0)
    with pytest.raises(ContractError):p.radial(1.338001)
    with pytest.raises(ContractError):p.contact_radius_mm(1.338001,'H1')

def test_real_provider_dispatch_geometry_units_and_outer_rejection():
    provider=AtlasHCProvider(ROOT);points=Quantity([[.2,0],[.6,0],[1.338,0],[1.339,0]],'mm','arbor_anchor_position','fixture','OPL')
    packet=provider.footprints(('a','b','c','d'),('H1','H1','H2','H1'),points,mode='connection_preview',enable_central_candidate=True)
    assert packet['manifest']['profile_id']==JOINT_ID and not packet['manifest']['physiological_validation']
    assert packet['footprints']['by_hc_id']['a']['radius_mm']==.00804
    assert packet['footprints']['by_hc_id']['b']['kernel']=={'kind':'uniform'}
    assert packet['blocked_records'][0]['hc_id']=='d' and 'OUTSIDE' in packet['blocked_records'][0]['reason']
    with pytest.raises(ContractError):provider.footprints(('a',),('H1',),Quantity([[.8,0]],'mm','anatomical_position','fixture','functional_visual_field'))

def test_existing_disk_membership_eligibility_and_raw_unit_weights_unchanged():
    points=np.array([[.75,0],[.75,.03]]);kinds=np.array(['H1','H2']);r=profile().contact_radius_mm(np.linalg.norm(points,axis=1),'H1')
    r[1]=profile().contact_radius_mm(np.linalg.norm(points[1]),'H2')
    cones=np.concatenate([point+[[.001,0],[.002,0],[.003,0]] for point in points]);labels=np.tile(['L','S','M'],2)
    raw,audit=eccentric_projection(cones,labels,points,kinds,np.ones(6,bool),1.1,r)
    assert raw[:,0].nnz==2 and raw[:,1].nnz==3 and raw[labels=='S'][:,kinds=='H1'].nnz==0
    np.testing.assert_array_equal(raw.data,np.ones(raw.nnz));assert not audit['outside_radius_fallback']

def test_source_hash_changes_identity_and_old_assets_are_rejected(tmp_path):
    selected=current_hc(ROOT)
    with pytest.raises(ContractError,match='STALE'):require_current_hc_asset(ROOT,{})
    require_current_hc_asset(ROOT,{'hc_density_geometry_identity':selected['identity']})
    (tmp_path/'configurations').mkdir();asset=tmp_path/'parameters.json';asset.write_text('{"version":1}',encoding='utf8')
    config=dict(selected_profile_id=JOINT_ID,bindings={'parameters':dict(path='parameters.json',sha256=file_hash(asset))})
    path=tmp_path/'configurations/hc_density_geometry_current.json';path.write_text(json.dumps(config),encoding='utf8');first=current_hc(tmp_path)['identity']
    asset.write_text('{"version":2}',encoding='utf8')
    with pytest.raises(ContractError,match='IDENTITY_CHANGED'):current_hc(tmp_path)
    config['bindings']['parameters']['sha256']=file_hash(asset);path.write_text(json.dumps(config),encoding='utf8')
    assert current_hc(tmp_path)['identity']!=first

def test_old_explicit_profile_preserved_and_new_master_not_silently_built(monkeypatch):
    old=HCDensityProfile(ROOT,profile_path='configurations/hc_joint_synthetic_v1.json');assert type(old) is HCDensityProfile
    assert old.support_radius_mm==7.1 and old.radial(0)==pytest.approx(250.)
    monkeypatch.setattr('retina_sim.atlas_hc_master._configuration',lambda *args,**kwargs:({'construction_domain_pending':True},{}))
    for operation in (lambda:FixedJointHCMaster(ROOT,{},profile()),lambda:materialize_hc_master(ROOT,{},profile())):
        with pytest.raises(ContractError,match='HC_V2_MASTER_NOT_BUILT'):operation()
