from dataclasses import replace
from pathlib import Path
from decimal import Decimal
import hashlib,json
import numpy as np
import pytest
from conftest import ROOT,q,NUMERIC_METRICS
from retina_sim.contracts import ContractError,MissingCalibrationError,Quantity,RunMode,CellTable,SolverConfig,StimulusSegment
from retina_sim.provenance import GateError,validate_parameters,validate_runnable,Registry
from retina_sim.schema import validate_schema
from retina_sim.graph import ConnectionTable,row_normalize
from retina_sim.density import read_density_tabulation,interpolate_meridian_density,evaluate_packer_rc_reference,MagnificationProfile,density_per_degree_squared
from retina_sim.photometry import Aperture,QuantumEfficiencyProfile,integrate_cone_photoisomerization_rate,radiance_to_retinal_irradiance,rgb_to_photoisomerization_rate,PLANCK_J_S,LIGHT_M_S
from retina_sim.engine import initialize_state,advance,checkpoint,restore

@pytest.mark.parametrize('mutation',[{'source_id':None},{'unit':''},{'value':None},{'evidence_kind':'unknown'},{'shape':(2,)},{'species':'cat'},{'conditions':{'different_background':True}},{'coordinate_frame':'anatomical_retina'}])
def test_bad_parameter_rejected(registry,parameters,mutation):
    bad=replace(parameters.records[0],**mutation)
    report=validate_parameters((bad,),[bad.parameter_id],registry.sources,ROOT,RunMode.NUMERIC_FIXTURE,parameters.profile)
    assert not report.allowed
    assert all(b.parameter_id==bad.parameter_id for b in report.blockers)
    assert all(b.blocked_module and b.needed_evidence for b in report.blockers)

def test_hash_and_source_identity(registry,parameters):
    source=registry.sources[parameters.records[0].source_id]
    with pytest.raises(ContractError,match='SHA256'): Registry(ROOT).register_source(replace(source,asset_sha256='0'*64))
    with pytest.raises(ContractError,match='SHA1'): replace(source,source_blob_sha1='0'*40).verify(ROOT)
    with pytest.raises(ContractError,match='commit'): replace(source,repository_commit=None).verify(ROOT)

@pytest.mark.parametrize('evidence,approval',[('synthetic_fixture',None),('project_assumption',None),('unknown',None)])
def test_fixture_and_unapproved_assumption_never_full(registry,parameters,evidence,approval):
    p=replace(parameters.records[0],evidence_kind=evidence,assumption_approval_reference=approval,species='M. mulatta',allowed_modes=('full_physiology',),status='usable_physiology')
    profile=replace(parameters.profile,kind='macaque_main',species='M. mulatta',parameter_ids=(p.parameter_id,))
    report=validate_parameters((p,),profile.parameter_ids,registry.sources,ROOT,RunMode.FULL_PHYSIOLOGY,profile)
    assert not report.allowed

def test_code_not_primary_measurement(registry,parameters):
    p=replace(parameters.records[0],evidence_kind='measured')
    assert not validate_parameters((p,),[],registry.sources,ROOT,RunMode.REFERENCE,parameters.profile).allowed

def test_profile_isolation_and_schema(registry):
    data=json.loads((ROOT/'sources/registry.json').read_text(encoding='utf-8'))
    schema=json.loads((ROOT/'schemas/source_registry.schema.json').read_text(encoding='utf-8'))
    validate_schema(data,schema)
    data['profiles'][0]['explicit_opt_in']=False
    with pytest.raises(ContractError): validate_schema(data,schema)
    data['profiles'][0].update(kind='macaque_main',species='human',explicit_opt_in=True)
    with pytest.raises(ContractError): validate_schema(data,schema)
    with pytest.raises(GateError): registry.load_parameters(data['profiles'][0]['profile_id'],RunMode.FULL_PHYSIOLOGY)
    with pytest.raises(ContractError,match='default'): registry.load_parameters('missing_default',RunMode.NUMERIC_FIXTURE)

@pytest.fixture
def density():
    manifest=json.loads((ROOT/'data/published_code/packer_cone_density.manifest.json').read_text())
    return read_density_tabulation(ROOT/'data/published_code/packer_cone_density.csv',manifest['csv_sha256'])

def test_density_literal_nodes_and_identity(density):
    assert len(density.rows)==60
    for row in density.rows:
        result=interpolate_meridian_density(density,float(row['eccentricity_mm']),row['meridian'],'computational_linear_interpolation')
        assert result.exact_source_literal==row['cone_density_cones_per_mm2']
        assert Decimal(result.exact_source_literal)==Decimal(row['source_density_literal'])*Decimal('1e3')
        assert result.bracketing_source_rows[0]['source_line']==row['source_line']
    with pytest.raises(ContractError,match='hash'): read_density_tabulation(ROOT/'data/published_code/packer_cone_density.csv','0'*64)

@pytest.mark.parametrize('e,m',[(0,'temporal'),(-1,'nasal'),(16,'temporal'),(20,'nasal'),(.1,'superior'),(float('nan'),'temporal')])
def test_density_out_of_support(density,e,m):
    with pytest.raises(ContractError): interpolate_meridian_density(density,e,m,'computational_linear_interpolation')

def test_density_interpolation_and_rc_order(density):
    rows=[r for r in density.rows if r['meridian']=='temporal']
    x1,x2=[float(r['eccentricity_mm']) for r in rows[:2]]
    d1,d2=[float(r['cone_density_cones_per_mm2']) for r in rows[:2]]
    e=(x1+x2)/2
    mag=MagnificationProfile(1000/210,'deg/mm','ISETMacaque_PackerConeRc','line 14','source-specific code reference; not universal physiology')
    result=interpolate_meridian_density(density,e,'temporal','computational_linear_interpolation')
    assert result.value==pytest.approx((d1+d2)/2,rel=1e-15)
    rc=evaluate_packer_rc_reference(density,e*mag.value,'temporal',mag)
    spacing=lambda d: (2/(3**.5*d))**.5
    expected=(spacing(d1)+spacing(d2))/2*mag.value*.204*2**.5
    assert rc.value==pytest.approx(expected,rel=2e-15)
    wrong=spacing((d1+d2)/2)*mag.value*.204*2**.5
    assert abs(rc.value-wrong)>1e-6
    assert density_per_degree_squared(100,mag)==pytest.approx(100/mag.value**2)
    with pytest.raises(ContractError): interpolate_meridian_density(density,e,'temporal',None)
    NUMERIC_METRICS['rc_interpolation_order_error_deg']=abs(rc.value-expected)

@pytest.mark.parametrize('source,target',[('cone_contrast','photoisomerization_rate'),('generator_deviation','absolute_voltage'),('generator_with_output_offset','generator_deviation')])
def test_quantity_semantic_mismatch(source,target):
    with pytest.raises(ContractError): q([1],'mV',source).require('mV',target)

def test_unit_and_coordinate_types():
    with pytest.raises(ContractError): q([1],'mV','generator_deviation').require('mV/s','generator_derivative')
    with pytest.raises(ContractError): q([[0,0]],'mm','anatomical_position','retinal_mm').require('mm','functional_position',frame='retinal_mm')
    with pytest.raises(ContractError): q([[0,0]],'deg','functional_position','local_patch').require('deg','functional_position',frame='functional_visual_field')
    cells=CellTable(('c1',),('cone',),('L',),('unresolved_macaque',),('unknown',),('temporal',),None,q([[.1,0]],'mm','functional_position','retinal_mm'),'mm','retinal_mm',q([5],'deg','global_eccentricity','functional_visual_field'),('source_specific',),('synthetic_fixture',))
    assert cells.soma_position is None  # no fabricated displacement
    with pytest.raises(ContractError): replace(cells,soma_position=cells.rf_position)

def test_sparse_direction_and_no_implicit_edges():
    edges=ConnectionTable(('e0',),('c0',),('c1',),('fixture',),(2.,),('dimensionless',),('positive post drive',),(None,),('synthetic_fixture',),('none',),('synthetic_fixture',))
    a=edges.to_csr(('c0','c1')).toarray()
    assert np.array_equal(a,[[0,0],[2,0]])
    assert edges.delay_seconds==(None,)
    with pytest.raises(ContractError): edges.to_csr(('c0','other'))
    with pytest.raises(ContractError): row_normalize([[0,0]])

def calibration(area=2e-12,convention='normalized_mean',area_unit='m^2'):
    area_value=area if area_unit=='m^2' else area*1e6
    # Three spatial nodes, normalized mean, all values are synthetic.
    da=q(np.full(3,area_value/3),area_unit,'spatial_quadrature_weight')
    effective=q(area_value,area_unit,'effective_collection_area')
    w=q(np.full(3,1/area),'m^-2','normalized_aperture_weight') if convention=='normalized_mean' else q(np.ones(3),'dimensionless','aperture_mask')
    ap=Aperture(convention,w,da,effective,'synthetic_fixture','synthetic_fixture')
    eta=QuantumEfficiencyProfile(q([.2,.4],'dimensionless','photoisomerization_efficiency'),'synthetic_fixture','synthetic_fixture',('absorption','isomerization'),('ocular_transmission',),'Explicit synthetic absorption/isomerization probability; no repeated media transmission')
    return ap,eta

def photo(ap,eta):
    return integrate_cone_photoisomerization_rate(q(np.full((3,2),2.),'W/(m^2 nm)','retinal_spectral_irradiance'),q([500,600],'nm','wavelength'),q([10,20],'nm','spectral_quadrature_weight'),ap,eta)

def test_photometry_analytic_area_units_and_conventions():
    ap,eta=calibration(); rate=photo(ap,eta)
    expected=2*2e-12*(.2*500e-9*10+.4*600e-9*20)/(PLANCK_J_S*LIGHT_M_S)
    assert rate.unit=='R*/s' and rate.evidence_tag=='synthetic_fixture'
    assert rate.values[0]==pytest.approx(expected,rel=1e-14)
    ap2,eta2=calibration(4e-12); assert photo(ap2,eta2).values[0]==pytest.approx(2*expected,rel=1e-14)
    mask,eta=calibration(convention='dimensionless_mask'); assert photo(mask,eta).values[0]==pytest.approx(expected,rel=1e-14)
    mm,eta=calibration(area_unit='mm^2'); assert photo(mm,eta).values[0]==pytest.approx(expected,rel=1e-14)
    NUMERIC_METRICS['photo_analytic_relative_error']=abs(rate.values[0]/expected-1)

@pytest.mark.parametrize('mutation',['no_efficiency','bad_efficiency','double_factor','bad_normalization','bad_shape','negative_spectrum','wrong_unit'])
def test_photometry_gate(mutation):
    ap,eta=calibration()
    E=q(np.ones((3,2)),'W/(m^2 nm)','retinal_spectral_irradiance')
    lam=q([500,600],'nm','wavelength'); dl=q([10,20],'nm','spectral_quadrature_weight')
    if mutation=='no_efficiency': eta=None
    if mutation=='bad_efficiency': eta=replace(eta,efficiency=q([1.1,.4],'dimensionless','photoisomerization_efficiency'))
    if mutation=='double_factor': eta=replace(eta,included_factors=('ocular_transmission',))
    if mutation=='bad_normalization': ap=replace(ap,spatial_weights=q([1,1,1],'m^-2','normalized_aperture_weight'))
    if mutation=='bad_shape': dl=q([1],'nm','spectral_quadrature_weight')
    if mutation=='negative_spectrum': E=q(-np.ones((3,2)),'W/(m^2 nm)','retinal_spectral_irradiance')
    if mutation=='wrong_unit': E=q(np.ones((3,2)),'W/(m^2 sr nm)','spectral_radiance')
    with pytest.raises(ContractError): integrate_cone_photoisomerization_rate(E,lam,dl,ap,eta)

def test_missing_optics_and_rgb_never_guessed():
    with pytest.raises(MissingCalibrationError): radiance_to_retinal_irradiance(None)
    with pytest.raises(MissingCalibrationError,match='ABSOLUTE'): rgb_to_photoisomerization_rate(np.zeros((2,2,3)))

def test_persistent_state_chunk_checkpoint_and_time(model):
    cfg=SolverConfig(max_step=.001)
    start=initialize_state(model,model.parameters,model.graph,model.initial_conditions(),cfg,time_seconds=1.3)
    stimulus=StimulusSegment(1.3,1.55,lambda t:q(np.full(4,.1*np.cos(2*np.pi*7*t)),'dimensionless','cone_contrast'),'dimensionless','cone_contrast','analytic','synthetic_fixture',{'kind':'absolute-phase sinusoid','evidence_tag':'synthetic_fixture'})
    full,_=advance(model,start,stimulus,1.55,cfg)
    part,_=advance(model,start,stimulus,1.43,cfg)
    hashes={key:getattr(part,key) for key in ('state_spec_hash','graph_hash','parameter_hash','model_hash','model_id','model_version')}
    resumed=restore(checkpoint(part),hashes)
    continued,result=advance(model,resumed,stimulus,1.55,cfg)
    error=float(np.max(np.abs(np.concatenate(full.ordered_state_arrays)-np.concatenate(continued.ordered_state_arrays))))
    assert error<1e-8
    assert continued.time_seconds==1.55
    assert result.outputs_with_units['time'].values[0]==1.43
    assert all(qv.evidence_tag=='synthetic_fixture' for qv in result.outputs_with_units.values())
    NUMERIC_METRICS['chunk_checkpoint_max_error_mV']=error
    for key in ('graph_hash','parameter_hash','model_hash','state_spec_hash'):
        with pytest.raises(ContractError): restore(checkpoint(part),dict(hashes,**{key:'changed'}))
    with pytest.raises(ContractError): advance(model,part,stimulus,1.4,cfg)
    wrong_initial=list(model.initial_conditions()); wrong_initial[0]=q(np.zeros(4),'mV','absolute_voltage')
    with pytest.raises(ContractError): initialize_state(model,model.parameters,model.graph,wrong_initial,cfg)

def test_full_mode_all_blockers_and_preserved_requirements():
    requirements=json.loads((ROOT/'requirements.json').read_text(encoding='utf-8'))
    assert {r['requirement_id'] for r in requirements}>=set(['S'+f'{i:02d}' for i in range(1,15)]+['P01','E01','E02','A01','A02','B01'])
    report=validate_runnable({'mode':'full_physiology','explicit_component':'SomaratnaFreeman2025GaussianHCReference'},requirements)
    assert not report.allowed and len(report.blockers)>=14
    assert 'S06' in {b.parameter_id for b in report.blockers}
    assert any('BLOCKED_DATA_DECODE' in b.missing for b in report.blockers)
    assert all(r['status'] in ('未开始','局部实现','来源待补','数值验证','生理验证') for r in requirements)
    assert not validate_runnable({'mode':'numeric_fixture'},requirements).allowed

@pytest.mark.parametrize('domain',[{}, {'e':None}, {'e':{'unit':'deg','lower':float('nan'),'upper':10,'lower_closed':True,'upper_closed':True}}, {'e':{'unit':'','lower':0,'upper':10,'lower_closed':True,'upper_closed':True}}, {'e':{'unit':'deg','lower':0,'upper':10,'lower_closed':1,'upper_closed':True}}, {'e':{'unit':'deg','lower':10,'upper':0,'lower_closed':True,'upper_closed':True}}])
def test_malformed_support_domain_rejected(registry,parameters,domain):
    bad=replace(parameters.records[0],support_domain=domain)
    report=validate_parameters((bad,),[],registry.sources,ROOT,RunMode.NUMERIC_FIXTURE,parameters.profile)
    assert not report.allowed

def test_execution_preflight_raises_all_blockers():
    from retina_sim.provenance import require_runnable
    requirements=json.loads((ROOT/'requirements.json').read_text(encoding='utf-8'))
    with pytest.raises(GateError) as caught:
        require_runnable({'mode':'full_physiology'},requirements)
    assert len(caught.value.report.blockers)==len(validate_runnable({'mode':'full_physiology'},requirements).blockers)

def test_mixed_unit_state_joint_solver_and_checkpoint():
    from types import SimpleNamespace
    from retina_sim.contracts import StateSpec,StateDerivativeVector,SimulationResult,canonical_hash
    from retina_sim.provenance import CoverageReport
    specs=(StateSpec('a',('a0',),(1,),'dimensionless','activity','synthetic_fixture'),StateSpec('v',('v0',),(1,),'mV','generator_deviation','synthetic_fixture'))
    model=SimpleNamespace(state_specs=specs,model_id='synthetic_mixed_unit_ODE',version='1',model_hash=canonical_hash('synthetic_mixed_unit_ODE_v1'),evidence_tag='synthetic_fixture',parameters=SimpleNamespace(parameter_hash='synthetic_parameters'),graph=SimpleNamespace(graph_hash='synthetic_graph'))
    def rhs(t,y,stimulus,parameters,graph):
        # Synthetic engineering fixture, not a biological mechanism or parameter.
        return StateDerivativeVector((q([-y[0]],'dimensionless/s','activity_derivative'),q([y[0]-y[1]],'mV/s','generator_deviation_derivative')),canonical_hash(specs))
    model.rhs=rhs; model.unpack=lambda y:(y[:1],y[1:])
    model.make_result=lambda times,trajectory,contract,checks:SimulationResult(RunMode.NUMERIC_FIXTURE,model.model_id,'synthetic_manifest',contract,{'activity':q(trajectory[0],'dimensionless','activity'),'voltage':q(trajectory[1],'mV','generator_deviation')},CoverageReport(True,()),checks,'numeric_consistency_only')
    cfg=SolverConfig(max_step=.02)
    state=initialize_state(model,model.parameters,model.graph,(q([1],'dimensionless','activity'),q([1],'mV','generator_deviation')),cfg)
    stimulus=StimulusSegment(0,.6,lambda t:q([0],'dimensionless','cone_contrast'),'dimensionless','cone_contrast','analytic','synthetic_fixture',{'evidence_tag':'synthetic_fixture'})
    first,_=advance(model,state,stimulus,.2,cfg)
    hashes={key:getattr(first,key) for key in ('state_spec_hash','graph_hash','parameter_hash','model_hash','model_id','model_version')}
    final,_=advance(model,restore(checkpoint(first),hashes),stimulus,.6,cfg)
    expected=np.array([np.exp(-.6),1.6*np.exp(-.6)])
    error=np.max(np.abs(np.concatenate(final.ordered_state_arrays)-expected))
    assert error<1e-9
    NUMERIC_METRICS['mixed_unit_synthetic_solver_max_scaled_error']=float(error)

@pytest.mark.parametrize('mutation',['unit','meaning','evidence','registry'])
def test_derivative_blocks_preserve_units_semantics_and_evidence(mutation):
    from retina_sim.contracts import StateSpec,StateDerivativeVector,canonical_hash
    specs=(StateSpec('a',('a0',),(1,),'dimensionless','activity','synthetic_fixture'),)
    derivative=q([1],'dimensionless/s','activity_derivative')
    h=canonical_hash(specs)
    if mutation=='unit': derivative=replace(derivative,unit='mV/s')
    if mutation=='meaning': derivative=replace(derivative,meaning='absolute_voltage_derivative')
    if mutation=='evidence': derivative=replace(derivative,evidence_tag='unlabelled')
    if mutation=='registry': h='changed'
    with pytest.raises(ContractError): StateDerivativeVector((derivative,),h).flatten(specs,'synthetic_fixture')

@pytest.mark.parametrize('parameter_id,wrong_unit',[('SF2025_tau','mm'),('SF2025_kSur','mV'),('SF2025_actRest','dimensionless'),('SF2025_radOptCoef','deg')])
def test_nonempty_wrong_parameter_unit_rejected(registry,parameters,model,parameter_id,wrong_unit):
    from retina_sim.reference import SomaratnaFreeman2025GaussianHCReference
    original=registry.parameters[parameter_id]
    bad=replace(original,unit=wrong_unit)
    registry.parameters[parameter_id]=bad
    with pytest.raises(GateError): registry.load_parameters(parameters.profile.profile_id,RunMode.NUMERIC_FIXTURE)
    replaced=replace(parameters,records=tuple(bad if p.parameter_id==parameter_id else p for p in parameters.records))
    with pytest.raises(ContractError,match='unit/shape'):
        SomaratnaFreeman2025GaussianHCReference(replaced,model.graph,explicit_opt_in=True)
