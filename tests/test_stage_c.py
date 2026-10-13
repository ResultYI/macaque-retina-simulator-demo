from dataclasses import replace
import hashlib,json,sys
import numpy as np
import pytest
from scipy.io import loadmat
from conftest import ROOT,q,NUMERIC_METRICS
from retina_sim.contracts import Quantity,ContractError,RunMode,SolverConfig
from retina_sim.engine import advance,checkpoint,restore,initialize_state
from retina_sim.provenance import Registry,GateError
from retina_sim.chen2024 import Chen2024PeripheralLMCurrent,ConeScope,SPECIES,COMMIT,load_chen_parameters,author_discrete_config,rate_segment,checked_source_values,baseline_subtracted_current,STRICT_INPUT_POLICY,AUTHOR_INPUT_POLICY,HighLightApplicabilityWarning
sys.path.insert(0,str(ROOT/'tools'))
from chen2024_scalar_source_oracle import scalar_source_oracle

# Thresholds fixed before running this suite. These are engineering tolerances.
MATLAB_ABS_PA=1e-8
MATLAB_REL=1e-10
STATE_ABS=1e-8

@pytest.fixture(scope='module')
def oracle():
    mat=loadmat(ROOT/'reports/chen2024_matlab_oracle.mat',simplify_cells=True)
    assert mat['metadata']['commit']==COMMIT and mat['metadata']['evidence_tag']=='synthetic_fixture'
    assert mat['metadata']['branch']=='nonlinear' and mat['metadata']['dt_seconds']==.0001
    return mat

@pytest.fixture
def cparams(): return load_chen_parameters(RunMode.NUMERIC_FIXTURE)

@pytest.fixture
def cone(cparams):
    return Chen2024PeripheralLMCurrent(cparams,ConeScope(('synthetic_L',),('L',),(25.,),SPECIES,'synthetic_fixture'),explicit_opt_in=True)

@pytest.fixture
def author_cone():
    return Chen2024PeripheralLMCurrent(load_chen_parameters(RunMode.REFERENCE),ConeScope(('synthetic_L',),('L',),(25.,),SPECIES,'synthetic_fixture'),explicit_opt_in=True,source_reproduction=True)

def test_actual_source_identity_and_parameters(cparams,oracle):
    manifest=json.loads((ROOT/'sources/chen2024_asset_manifest.json').read_text())
    assert manifest['commit']==COMMIT and manifest['license']=='MIT'
    assert len(manifest['assets'])==4
    for asset in manifest['assets']:
        raw=(ROOT/asset['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==asset['asset_sha256']
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==asset['source_blob_sha1']
    values=checked_source_values()
    assert len(cparams.records)==12
    assert all(float(oracle['initial'][name])==value for name,value in values.items())
    assert all(p.species==SPECIES and p.evidence_kind=='author_assumption' for p in cparams.records)
    assert all(p.support_domain['eccentricity']['lower']==20 and not p.support_domain['eccentricity']['lower_closed'] for p in cparams.records)

def test_rate_segment_zero_order_hold_between_grid_points():
    rates=q(np.array([[10.],[20.]]),'R*/s','photoisomerization_rate')
    for origin in (0.,1.3):
        segment=rate_segment(rates,origin)
        boundary=origin+.0001
        assert segment.at(origin+.000075).values[0]==10
        assert segment.at(np.nextafter(boundary,-np.inf)).values[0]==10
        assert segment.at(boundary).values[0]==20
        with pytest.raises(ContractError): segment.at(np.nextafter(origin,-np.inf))
        with pytest.raises(ContractError): segment.at(origin+.0002)

@pytest.mark.parametrize('name',['dark','background','weak_flash','step_flash','high_rate'])
def test_unmodified_matlab_current_and_scalar_state_parity(cone,oracle,name):
    fixture=oracle['oracle'][name]
    assert fixture['evidence_tag']=='synthetic_fixture'
    rate=np.asarray(fixture['rates_Rstar_per_s'])
    result=cone.forward_kernel_reference(q(rate,'R*/s','photoisomerization_rate'))
    actual=result.outputs_with_units['current'].values[0]
    original=np.asarray(fixture['absolute_current_pA'])
    abs_error=float(np.max(np.abs(actual-original)))
    rel_error=float(np.max(np.abs(actual-original)/np.maximum(np.abs(original),1e-12)))
    assert abs_error<=MATLAB_ABS_PA and rel_error<=MATLAB_REL
    expected=scalar_source_oracle(rate,checked_source_values())
    state=np.array([result.outputs_with_units[key].values[0] for key in ('R','P','G','C','S')])
    state_error=float(np.max(np.abs(state-expected)))
    assert state_error<=STATE_ABS
    assert np.all(result.outputs_with_units['current'].values<0)
    assert all(v.evidence_tag=='synthetic_fixture' for v in result.outputs_with_units.values())
    assert result.metadata['source_commit']==COMMIT and result.metadata['voltage_coupling']=='BLOCKED'
    assert 'last sample not consumed' in result.metadata['author_sample_semantics']
    NUMERIC_METRICS['chen2024_'+name]={'matlab_absolute_max_error_pA':abs_error,'matlab_relative_max_error':rel_error,'scalar_all_state_max_error_in_declared_units':state_error,'samples':len(rate),'evidence_tag':'synthetic_fixture'}

def test_dark_baseline_and_units(cone):
    result=cone.forward_kernel_reference(q(np.zeros(1000),'R*/s','photoisomerization_rate'))
    out=result.outputs_with_units
    err=float(np.max(np.abs(out['current'].values+428.75)))
    assert err<1e-9
    assert out['current'].unit=='pA' and out['current'].meaning=='absolute_outer_segment_current'
    assert out['R'].unit=='s^-2' and out['P'].unit=='s^-1'
    assert out['G'].unit==out['C'].unit=='uM' and out['S'].unit=='uM/s'
    assert 'voltage' not in out
    NUMERIC_METRICS['chen2024_dark_absolute_error_pA']=err

def test_background_settles_and_flash_sign(cone,oracle):
    bg=cone.forward_kernel_reference(q(oracle['oracle']['background']['rates_Rstar_per_s'],'R*/s','photoisomerization_rate'))
    arrays=np.array([bg.outputs_with_units[key].values[0] for key in ('R','P','G','C','S')])
    differences=np.abs(np.diff(arrays[:,-1000:],axis=1))/np.maximum(np.abs(arrays[:,-1000:-1]),1)
    residual=float(np.max(differences))
    assert residual<1e-8
    rates=oracle['oracle']['weak_flash']['rates_Rstar_per_s']
    flash=cone.forward_kernel_reference(q(rates,'R*/s','photoisomerization_rate')).outputs_with_units['current'].values[0]
    assert np.max(flash[6000:10000])>flash[0]+1
    NUMERIC_METRICS['chen2024_background_last_1000_scaled_step_residual']=residual
    NUMERIC_METRICS['chen2024_background_settling_criterion']={'duration_seconds':3.9999,'last_samples':1000,'max_scaled_step_change_threshold':1e-8}

def test_nonlinear_flash_response_is_not_scaled_linear_filter(cone,oracle):
    rates=oracle['oracle']['weak_flash']['rates_Rstar_per_s']
    one=cone.forward_kernel_reference(q(rates,'R*/s','photoisomerization_rate')).outputs_with_units['current'].values[0]
    two=cone.forward_kernel_reference(q(rates*2,'R*/s','photoisomerization_rate')).outputs_with_units['current'].values[0]
    residual=float(np.max(np.abs((two-two[0])-2*(one-one[0]))))
    assert residual>1e-4
    NUMERIC_METRICS['chen2024_synthetic_flash_nonlinearity_residual_pA']=residual

def test_stream_chunks_checkpoint_preserve_source_cache(cone,oracle):
    rates=oracle['oracle']['step_flash']['rates_Rstar_per_s'][:-1,None]
    stimulus=rate_segment(q(rates,'R*/s','photoisomerization_rate'),1.3)
    cfg=author_discrete_config(); initial=cone.initialize(1.3)
    end,full=advance(cone,initial,stimulus,stimulus.stop_seconds,cfg)
    state=initial; pieces=[]
    for endpoint in [1.3+1234*.0001,1.3+11000*.0001,stimulus.stop_seconds]:
        state,part=advance(cone,state,stimulus,endpoint,cfg)
        pieces.append(part)
        expected={k:getattr(state,k) for k in ('state_spec_hash','graph_hash','parameter_hash','model_hash','model_id','model_version')}
        state=restore(checkpoint(state),expected)
    for key in ('R','P','G','C','S','current'):
        reconstructed=np.concatenate([p.outputs_with_units[key].values if i==0 else p.outputs_with_units[key].values[:,1:] for i,p in enumerate(pieces)],axis=1)
        assert np.array_equal(reconstructed,full.outputs_with_units[key].values)
    assert all(np.array_equal(a,b) for a,b in zip(state.ordered_state_arrays,end.ordered_state_arrays))
    assert state.time_seconds==stimulus.stop_seconds
    NUMERIC_METRICS['chen2024_chunk_checkpoint_max_error']=0.0
    for key in ('graph_hash','parameter_hash','model_hash','state_spec_hash'):
        with pytest.raises(ContractError): restore(checkpoint(state),dict(expected,**{key:'wrong'}))

def test_last_author_sample_not_consumed(cone):
    rates=np.zeros(100); rates[-1]=49999
    first=cone.forward_kernel_reference(q(rates,'R*/s','photoisomerization_rate'))
    second=cone.forward_kernel_reference(q(np.zeros(100),'R*/s','photoisomerization_rate'))
    assert np.array_equal(first.outputs_with_units['current'].values,second.outputs_with_units['current'].values)
    assert first.outputs_with_units['time'].values[-1]==pytest.approx(.0099)

@pytest.mark.parametrize('parameter,mutation',[('sigma',{'unit':'mm'}),('eta',{'unit':'s^-1'}),('gamma',{'unit':'dimensionless'}),('k',{'value':.02}),('gdark',{'value':20.5}),('phi',{'source_id':None})])
def test_parameter_unit_identity_and_no_mixing(cone,cparams,parameter,mutation):
    records=tuple(replace(p,**mutation) if p.parameter_id=='CHEN2024_'+parameter else p for p in cparams.records)
    with pytest.raises(ContractError): Chen2024PeripheralLMCurrent(replace(cparams,records=records),cone.scope,explicit_opt_in=True)
    if 'unit' in mutation:
        registry=Registry.read(ROOT/'sources/registry.json')
        registry.parameters['CHEN2024_'+parameter]=next(p for p in records if p.parameter_id=='CHEN2024_'+parameter)
        with pytest.raises(GateError): registry.load_parameters(cparams.profile.profile_id,RunMode.NUMERIC_FIXTURE)

@pytest.mark.parametrize('types,ecc,species',[(('S',),(25,),SPECIES),(('L',),(20,),SPECIES),(('M',),(5,),SPECIES),(('L',),(float('nan'),),SPECIES),(('L',),(25,),'M. mulatta'),(('L',),(25,),'human')])
def test_scope_rejection(types,ecc,species):
    with pytest.raises(ContractError): ConeScope(('fixture',),types,ecc,species,'synthetic_fixture')

@pytest.mark.parametrize('value,unit,meaning',[([-1,0],'R*/s','photoisomerization_rate'),([50000,0],'R*/s','photoisomerization_rate'),([1,1],'R*/step','photoisomerization_count'),([1,1],'dimensionless','cone_contrast'),([[1,2],[1,2]],'R*/s','photoisomerization_rate')])
def test_public_rate_rejections(cone,value,unit,meaning):
    with pytest.raises(ContractError): cone.forward_kernel_reference(q(value,unit,meaning))

def test_nonfinite_dt_type_solver_and_output_grid(cone):
    with pytest.raises(ContractError): q([0,float('nan')],'R*/s','photoisomerization_rate')
    with pytest.raises(ContractError): rate_segment(q(np.zeros((10,1)),'R*/s','photoisomerization_rate'),0,.001)
    with pytest.raises(ContractError): SolverConfig(method='Chen2024AuthorDiscrete',input_interpolation='zero_order_hold',fixed_step_seconds=.001)
    with pytest.raises(ContractError): initialize_state(cone,cone.parameters,cone.graph,cone.initial_conditions(),SolverConfig())
    segment=rate_segment(q(np.zeros((10,1)),'R*/s','photoisomerization_rate'),0)
    with pytest.raises(ContractError): advance(cone,cone.initialize(),segment,.001,SolverConfig())
    with pytest.raises(ContractError): advance(cone,cone.initialize(),segment,.00015,author_discrete_config())
    with pytest.raises(ContractError): advance(cone,cone.initialize(),segment,float('nan'),author_discrete_config())
    with pytest.raises(ContractError): advance(cone,cone.initialize(),segment,.001,author_discrete_config(),sample_times=[.00015])

def test_baseline_explicit_and_no_voltage_or_inverse(cone,cparams):
    current=q([-420,-410],'pA','absolute_outer_segment_current')
    baseline=q(-428.75,'pA','absolute_outer_segment_current')
    diff=baseline_subtracted_current(current,baseline)
    assert np.array_equal(diff.values,[8.75,18.75]) and diff.meaning=='baseline_subtracted_outer_segment_current'
    with pytest.raises(ContractError): baseline_subtracted_current(current,q(1,'mV','absolute_voltage'))
    for kwargs in ({'branch':'linear'},{'branch':'inverse'},{'voltage_gain':1},{'explicit_opt_in':False}):
        options={'explicit_opt_in':True,**kwargs}
        with pytest.raises(ContractError): Chen2024PeripheralLMCurrent(cparams,cone.scope,**options)
    with pytest.raises(ContractError): Chen2024PeripheralLMCurrent(replace(cparams,run_mode=RunMode.FULL_PHYSIOLOGY),cone.scope,explicit_opt_in=True)

def test_real_data_inventory_never_replaced_by_synthetic_mat():
    inventory=json.loads((ROOT/'reports/chen2024_data_inventory.json').read_text())
    record=next(f for f in inventory['files'] if f['name']=='Figure2-SourceData1.mat')
    assert record['published_sha256']=='6724e2edd8e8708456f32900a14efae88192a5ba597365046d0031eb543d1331'
    if not record['exists']:
        assert inventory['status']=='BLOCKED_DOWNLOAD' and inventory['observed_mat_variables'] is None and inventory['observed_cells_and_groups'] is None
        assert inventory['source_data_reproduction']=='NOT_RUN'
    else:
        assert record['actual_sha256']==record['published_sha256']
    assert 'NOT independent' in inventory['data_usage']

def test_verified_real_curves_preserved_and_no_invented_error_metrics():
    inventory=json.loads((ROOT/'reports/chen2024_data_inventory.json').read_text())
    file=next(f for f in inventory['files'] if f['name']=='Figure2-SourceData1.mat')
    raw=(ROOT/file['path']).read_bytes()
    assert len(raw)==41490462 and hashlib.sha256(raw).hexdigest()==file['published_sha256']
    source=loadmat(ROOT/file['path'],simplify_cells=True)['PrimateConeResponses']
    metrics=json.loads((ROOT/'reports/chen2024_real_data/all_cell_metrics.json').read_text())
    full=np.load(ROOT/'reports/chen2024_real_data/full_published_traces.npz',allow_pickle=False)
    assert len(source)==metrics['cell_count']==6 and metrics['model_run_count']==0
    assert metrics['comparison_status']=='BLOCKED_INPUT_SEMANTICS_AND_HISTORY'
    for index,(record,cell) in enumerate(zip(metrics['cells'],source),1):
        assert record['cell_array_index_matlab_1based']==index and record['biological_cell_id'] is None
        assert np.array_equal(full[f'cell_{index}_stimulus_Rstar_per_s'],cell['AdaptingStepStim'])
        assert np.array_equal(full[f'cell_{index}_response_normalized'],cell['AdaptingStepResponse'])
        assert record['stimulus_negative_samples']==int(np.sum(cell['AdaptingStepStim']<0))==1026
        assert record['stimulus_gte50000_samples']==int(np.sum(cell['AdaptingStepStim']>=50000))==12538
        assert record['model_ran'] is False and record['fitted_parameters'] is False
        assert all(record[key] is None for key in ('model_rmse_normalized','model_mae_normalized','model_max_abs_error_normalized','model_variance_explained','model_pearson_r','raw_trial_noise_ceiling'))
        quantity=Quantity(cell['AdaptingStepStim'][:,None],'R*/s','photoisomerization_rate','published_source_repeat_mean')
        with pytest.raises(ContractError,match='Rate outside supported'): rate_segment(quantity,0)
        with pytest.raises(ContractError,match='Rate outside supported'): rate_segment(quantity,0,input_policy=AUTHOR_INPUT_POLICY)

@pytest.mark.parametrize('name,symlink',[('../Figure2-SourceData1.mat',False),('C:/outside.mat',False),('Figure2-SourceData1.mat',True)])
def test_source_archive_rejects_unsafe_member_before_extraction(tmp_path,monkeypatch,name,symlink):
    import zipfile
    import ingest_chen2024_archive as module
    archive=tmp_path/'unsafe.zip'
    with zipfile.ZipFile(archive,'w') as z:
        entry=zipfile.ZipInfo(name)
        if symlink: entry.external_attr=0o120777<<16
        z.writestr(entry,b'not actual source data')
    monkeypatch.setattr(module,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='Unsafe/duplicate'): module.ingest(archive)
    assert not any((tmp_path/'data').rglob('*.mat'))

def test_gate_correction_preserves_physiology_and_before_trajectories(cone,author_cone):
    import inspect
    from retina_sim.contracts import canonical_hash
    before=ROOT/'reports/chen2024_gate_correction_before'
    record=json.loads((before/'fingerprints_before.json').read_text())
    for name,digest in record['methods_sha256'].items():
        assert hashlib.sha256(inspect.getsource(getattr(Chen2024PeripheralLMCurrent,name)).encode()).hexdigest()==digest
    text=(ROOT/'retina_sim/chen2024.py').read_text(encoding='utf-8')
    block=text[text.index('        self.dark_current='):text.index('    @property')]
    assert hashlib.sha256(block.encode()).hexdigest()==record['constructor_physiology_block_sha256']
    for path,digest in record['immutable_file_sha256'].items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    assert cone.parameters.parameter_hash==record['parameter_hash_numeric_fixture']
    assert author_cone.parameters.parameter_hash==record['parameter_hash_component_reference']
    assert cone.graph.graph_hash==record['graph_hash'] and canonical_hash(cone.state_specs)==record['state_spec_hash']
    assert canonical_hash(cone.initial_conditions())==record['initial_conditions_hash']
    assert checked_source_values()==record['source_values']
    assert {k:getattr(cone,k) for k in ('gdark','dark_current','q','smax')}==record['derived_constants']
    old=np.load(before/'in_domain_trajectories_before.npz',allow_pickle=False)
    maximum=0.
    for name in ('dark','background','step'):
        rate=q(old[name+'_rate'],'R*/s','photoisomerization_rate')
        for model in (cone,author_cone):
            out=model.forward_kernel_reference(rate).outputs_with_units
            for key in ('R','P','G','C','S','current','time'):
                assert np.array_equal(out[key].values,old[name+'_'+key])
                maximum=max(maximum,float(np.max(np.abs(out[key].values-old[name+'_'+key]))))
    NUMERIC_METRICS['chen2024_gate_in_domain_before_after_max_error']=maximum
    assert cone.model_hash!=record['model_hash_before'] # Whole code/policy identity changed, biology did not.
    assert cone.model_hash!=author_cone.model_hash

def test_explicit_author_positive_peaks_warn_and_keep_source_equations(author_cone):
    rate=np.full(2000,5000.);rate[1000:1020]=100000.
    with pytest.warns(HighLightApplicabilityWarning,match='no bleaching/regeneration') as notices:
        result=author_cone.forward_kernel_reference(q(rate,'R*/s','photoisomerization_rate'))
    assert len(notices)==1
    exposure=result.metadata['input_exposure']
    assert result.run_mode==RunMode.REFERENCE and result.metadata['input_policy']==AUTHOR_INPUT_POLICY
    assert exposure['peak_rate']==100000 and exposure['minimum_rate']==5000
    assert exposure['samples_gte50000_by_cell']==[20] and exposure['intervals_consumed_per_cell']==1999
    assert exposure['cumulative_photoisomerizations_Rstar_by_cell']==[float(rate[:-1].sum()*.0001)]
    assert exposure['exposure_conditions']['bleaching_regeneration_in_model'] is False
    assert exposure['biological_validity']==result.metadata['biological_validity']=='NOT_ESTABLISHED'
    warning=exposure['warnings'][0]
    assert warning['annotation_is_hard_source_cap'] is False and warning['source_url'].endswith('elife-93795-v1.pdf')
    expected=scalar_source_oracle(rate,checked_source_values())
    states=np.array([result.outputs_with_units[k].values[0] for k in ('R','P','G','C','S')])
    assert float(np.max(np.abs(states-expected)))<=STATE_ABS
    assert np.all(result.outputs_with_units['current'].values<0)
    NUMERIC_METRICS['chen2024_author_positive_high_peak_scalar_state_max_error']=float(np.max(np.abs(states-expected)))

@pytest.mark.parametrize('mode',[RunMode.NUMERIC_FIXTURE,RunMode.FULL_PHYSIOLOGY])
def test_author_policy_cannot_enter_other_modes(cparams,cone,mode):
    with pytest.raises(ContractError,match='component_reference only'):
        Chen2024PeripheralLMCurrent(replace(cparams,run_mode=mode),cone.scope,explicit_opt_in=True,source_reproduction=True)

def test_default_reference_and_cross_policy_state_remain_closed(author_cone,cone):
    reference=Chen2024PeripheralLMCurrent(load_chen_parameters(RunMode.REFERENCE),author_cone.scope,explicit_opt_in=True)
    high=q(np.full(50,100000.),'R*/s','photoisomerization_rate')
    for model in (reference,cone):
        with pytest.raises(ContractError): model.forward_kernel_reference(high)
    permissive=rate_segment(q(np.full((50,1),100000.),'R*/s','photoisomerization_rate'),0,input_policy=AUTHOR_INPUT_POLICY)
    for model in (reference,cone):
        with pytest.raises(ContractError,match='policy mismatch'): advance(model,model.initialize(),permissive,permissive.stop_seconds,author_discrete_config())
    strict=rate_segment(q(np.zeros((50,1)),'R*/s','photoisomerization_rate'),0)
    with pytest.raises(ContractError,match='policy mismatch'): advance(author_cone,author_cone.initialize(),strict,strict.stop_seconds,author_discrete_config())
    with pytest.raises(ContractError,match='model_hash'): advance(author_cone,reference.initialize(),permissive,permissive.stop_seconds,author_discrete_config())
    with pytest.raises(ContractError,match='Unknown'): rate_segment(q(np.zeros((50,1)),'R*/s','photoisomerization_rate'),0,input_policy='unchecked')

@pytest.mark.parametrize('value',[-1,float('nan'),float('inf')])
def test_author_reproduction_keeps_negative_and_nonfinite_rejections(author_cone,value):
    with pytest.raises(ContractError): author_cone.forward_kernel_reference(q([value,1000.],'R*/s','photoisomerization_rate'))

def test_author_high_peak_stream_chunking_and_exposure(author_cone):
    rate=q(np.full((128,1),100000.),'R*/s','photoisomerization_rate')
    segment=rate_segment(rate,0,input_policy=AUTHOR_INPUT_POLICY)
    with pytest.warns(HighLightApplicabilityWarning): single,out=advance(author_cone,author_cone.initialize(),segment,.0128,author_discrete_config())
    with pytest.warns(HighLightApplicabilityWarning): part,first=advance(author_cone,author_cone.initialize(),segment,.0064,author_discrete_config())
    with pytest.warns(HighLightApplicabilityWarning): chunk,second=advance(author_cone,part,segment,.0128,author_discrete_config())
    assert all(np.array_equal(a,b) for a,b in zip(single.ordered_state_arrays,chunk.ordered_state_arrays))
    assert np.array_equal(np.r_[first.outputs_with_units['current'].values[0],second.outputs_with_units['current'].values[0,1:]],out.outputs_with_units['current'].values[0])
    assert out.metadata['input_exposure']['cumulative_photoisomerizations_Rstar_by_cell']==[1280.]
    assert first.metadata['input_exposure']['cumulative_photoisomerizations_Rstar_by_cell']==[640.]
    assert second.metadata['input_exposure']['cumulative_photoisomerizations_Rstar_by_cell']==[640.]
    assert second.metadata['input_exposure']['start_seconds']==.0064
