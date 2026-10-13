"""MIT-derived Chen2024 historical nonlinear forward kernel, isolated pA port.

Attribution: chrischen2/photoreceptorLinearization, commit 1bcf7c9a..., MIT.
Changes: explicit public rate units, persistent state/cache, strict scope and
checkpoint identity. Source Euler order retained; no linear/inverse branch.
"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace,MappingProxyType
import hashlib,re,warnings
import numpy as np
from .contracts import ContractError,Quantity,StateSpec,SolverConfig,StimulusSegment,SimulationState,SimulationResult,RunMode,canonical_hash
from .parameter_contracts import CHEN_PARAMETER_SPECS
from .provenance import Registry,CoverageReport

COMMIT='1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2'
PROFILE='Chen2024PeripheralPrimateCone_source_mixed_macaca'
SPECIES='source_mixed_Macaca_fascicularis_nemestrina_mulatta'
SOURCE_BLOBS={'BiophysModel.m':'8117c9f157fc8768993e41c521efe23aff8fc7dc','initPhotoreceptorParams.m':'086e324f5f5a38f04249c63b1650f68fea4d3586'}
STRICT_INPUT_POLICY='strict_project_reference'
AUTHOR_INPUT_POLICY='author_source_reproduction'
PAPER_URL='https://cdn.elifesciences.org/articles/93795/elife-93795-v1.pdf'
LIMITS=('Peripheral >20deg L/M only; source recordings from three Macaca species, not a species-specific default','No bleaching/regeneration or all slow adaptation; paper <50000 applicability text is not an instantaneous source-code cap','Absolute negative outer-segment pA; no cone voltage, terminal calcium, neurotransmission or HC/BC coupling','Synthetic source parity is numerical validation; source fitting data are not independent external validation')
ROOT=Path(__file__).resolve().parents[1]

class HighLightApplicabilityWarning(UserWarning): pass

def _validate_rate_input(a,input_policy):
    if input_policy not in (STRICT_INPUT_POLICY,AUTHOR_INPUT_POLICY): raise ContractError('Unknown Chen2024 input policy')
    if not np.all(np.isfinite(a)) or np.any(a<0): raise ContractError('Rate outside supported nonnegative finite domain')
    if input_policy==STRICT_INPUT_POLICY and np.any(a>=50000): raise ContractError('Rate outside supported strict project [0,50000) R*/s envelope')

def _exposure_metadata(a,dt,input_policy,start,stop):
    a=np.asarray(a,dtype=float)
    high=bool(np.any(a>=50000))
    notices=[]
    if high:
        notices=[{'code':'CHEN2024_HIGH_LIGHT_APPLICABILITY_UNCERTAIN','kind':'biological_applicability_warning','source_url':PAPER_URL,'source_locator':'Fig3 p7 / Fig4 p8; Discussion p15 Photopigment bleaching','source_commit':COMMIT,'annotation_level_Rstar_per_s':50000,'annotation_is_hard_source_cap':False,'statement':'High peaks allowed only for explicit numerical author reproduction. No bleaching/regeneration; biological validity not established. Paper gives no universal instantaneous cap or mechanically applicable whole-segment mean cap.'}]
    return {'input_policy':input_policy,'unit':'R*/s','minimum_rate':float(a.min()),'peak_rate':float(a.max()),'minimum_rate_by_cell':a.min(axis=0).tolist(),'peak_rate_by_cell':a.max(axis=0).tolist(),'mean_rate_by_cell_descriptive_only':a.mean(axis=0).tolist(),'intervals_consumed_per_cell':a.shape[0],'cell_count':a.shape[1],'start_seconds':float(start),'stop_seconds':float(stop),'duration_seconds':a.shape[0]*dt,'cumulative_photoisomerizations_Rstar_by_cell':(a.sum(axis=0)*dt).tolist(),'samples_gte50000_by_cell':(a>=50000).sum(axis=0).tolist(),'exposure_conditions':{'includes_background':True,'rate_semantics':'ZOH total rate including background; R*/s, not per-step counts','illumination_history_before_this_advance':'not inferred; external provenance required','physical_delivery_and_bleaching_status':'not measured or inferred from rates','bleaching_regeneration_in_model':False},'biological_validity':'NOT_ESTABLISHED','warnings':notices}

def checked_source_values(root=ROOT):
    folder=Path(root)/'data/source_snapshots/chen2024_1bcf7c9a'
    for name,expected in SOURCE_BLOBS.items():
        raw=(folder/name).read_bytes()
        if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=expected:
            raise ContractError('Pinned Chen2024 source identity mismatch: '+name)
    text=(folder/'initPhotoreceptorParams.m').read_text(encoding='utf-8')
    branch=re.search(r"case 'peripheralPrimateCone'(.*?)case 'primateRod'",text,re.S).group(1)
    result={}
    for key in CHEN_PARAMETER_SPECS:
        name=key.removeprefix('CHEN2024_')
        match=re.search(r'params\.'+name+r'\s*=\s*([^;]+);',text if name=='timeStep' else branch)
        result[name]=float(match.group(1))
    return result

def load_chen_parameters(run_mode,root=ROOT):
    return Registry.read(Path(root)/'sources/registry.json').load_parameters(PROFILE,run_mode)

@dataclass(frozen=True)
class ConeScope:
    cell_ids: tuple
    cone_types: tuple
    eccentricity_functional_deg: tuple
    species_source_profile: str
    evidence_tag: str

    def __post_init__(self):
        n=len(self.cell_ids)
        if not n or len(set(self.cell_ids))!=n or len(self.cone_types)!=n or len(self.eccentricity_functional_deg)!=n:
            raise ContractError('Cone scope IDs/shape invalid')
        if any(x not in ('L','M') for x in self.cone_types): raise ContractError('S cone / unknown cone type unsupported')
        if any(not np.isfinite(e) or e<=20 for e in self.eccentricity_functional_deg): raise ContractError('Only source peripheral >20deg supported')
        if self.species_source_profile!=SPECIES or not self.evidence_tag: raise ContractError('Mixed-Macaca source scope must be explicit')

def author_discrete_config():
    return SolverConfig(method='Chen2024AuthorDiscrete',input_interpolation='zero_order_hold',fixed_step_seconds=.0001)

def rate_segment(rates,start_seconds,dt_seconds=.0001,*,input_policy=STRICT_INPUT_POLICY):
    a=rates.require('R*/s','photoisomerization_rate')
    if a.ndim!=2 or not a.shape[0] or not a.shape[1] or dt_seconds!=.0001: raise ContractError('Rate grid requires (intervals,cells) and author dt')
    _validate_rate_input(a,input_policy)
    boundaries=start_seconds+np.arange(a.shape[0]+1)*dt_seconds
    def evaluate(t):
        index=int(np.searchsorted(boundaries,t,side='right')-1)
        if not 0<=index<a.shape[0]: raise ContractError('Rate evaluated outside supplied intervals')
        return Quantity(a[index],'R*/s','photoisomerization_rate',rates.evidence_tag)
    return StimulusSegment(float(start_seconds),float(start_seconds+a.shape[0]*dt_seconds),evaluate,'R*/s','photoisomerization_rate','zero_order_hold',rates.evidence_tag,{'unit':'R*/s','meaning':'photoisomerization_rate','includes_background':True,'grid_step_seconds':dt_seconds,'interval_count':a.shape[0],'evidence_tag':rates.evidence_tag,'chen2024_input_policy':input_policy})

class Chen2024PeripheralLMCurrent:
    model_id='Chen2024PeripheralLMCurrent'
    version='0.1.0'
    def __init__(self,parameters,scope,*,explicit_opt_in=False,branch='nonlinear',voltage_gain=None,source_reproduction=False):
        if type(source_reproduction) is not bool or (source_reproduction and parameters.run_mode!=RunMode.REFERENCE):
            raise ContractError('Author source reproduction requires explicit boolean selection in component_reference only')
        self.input_policy=AUTHOR_INPUT_POLICY if source_reproduction else STRICT_INPUT_POLICY
        if branch!='nonlinear' or voltage_gain is not None: raise ContractError('Linear/inverse/pA-to-mV branches forbidden')
        if not explicit_opt_in or parameters.run_mode==RunMode.FULL_PHYSIOLOGY or parameters.profile.profile_id!=PROFILE or parameters.profile.kind!='named_reference':
            raise ContractError('Explicit isolated Chen2024 component selection required')
        if parameters.profile.species!=SPECIES or scope.species_source_profile!=SPECIES:
            raise ContractError('Cannot substitute species-specific or non-macaque profile')
        if parameters.run_mode==RunMode.NUMERIC_FIXTURE and scope.evidence_tag!='synthetic_fixture': raise ContractError('Numerical geometry/state must be synthetic_fixture')
        expected=checked_source_values(); values={}
        for p in parameters.records:
            if p.parameter_id not in CHEN_PARAMETER_SPECS: raise ContractError('Unknown Chen2024 parameter')
            unit,shape=CHEN_PARAMETER_SPECS[p.parameter_id]
            if p.unit!=unit or tuple(p.shape)!=shape or np.asarray(p.value).shape!=shape or not np.isfinite(p.value): raise ContractError('Chen2024 parameter unit/shape mismatch')
            name=p.parameter_id.removeprefix('CHEN2024_')
            if name in values or p.value!=expected[name]: raise ContractError('Fixed author parameter changed; no source-parity tuning allowed')
            if p.source_id!='CHEN2024_initPhotoreceptorParams_m' or p.species!=SPECIES or p.conditions!=parameters.profile.conditions or parameters.run_mode.value not in p.allowed_modes:
                raise ContractError('Parameter source/species/conditions/mode mismatch')
            values[name]=float(p.value)
        if set(values)!=set(expected): raise ContractError('Missing Chen2024 parameter; no fallback')
        self.p=MappingProxyType(values); self.parameters=parameters; self.scope=scope; self.evidence_tag=scope.evidence_tag
        self.graph=SimpleNamespace(graph_hash=canonical_hash({'scope':scope,'connections':None,'component':'isolated_outer_segment'}))
        self.dark_current=values['gdark']**values['n']*values['k']
        self.gdark=(self.dark_current/values['k'])**(1/values['n'])
        self.q=values['beta']*values['cdark']/self.dark_current
        self.smax=values['eta']/values['phi']*self.gdark*(1+(values['cdark']/values['kGC'])**values['m'])
        units=('s^-2','s^-1','uM','uM','uM/s')
        meanings=('effective_opsin_activity','effective_pde_activity','cgmp_concentration','outer_segment_calcium','cgmp_synthesis_cache')
        self.state_specs=tuple(StateSpec(name,scope.cell_ids,(len(scope.cell_ids),),unit,meaning,'BiophysModel.m nonlinear initialization 39-51; commit '+COMMIT) for name,unit,meaning in zip(('R','P','G','C','S'),units,meanings))
    @property
    def model_hash(self):
        return canonical_hash({'id':self.model_id,'version':self.version,'source':COMMIT,'source_blobs':SOURCE_BLOBS,'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'numerical_scheme':'source explicit old-state Euler; S(Cnew), G(Sold)','limits':LIMITS,'input_policy':self.input_policy})
    def initial_conditions(self):
        p=self.p
        values=(0,p['eta']/p['phi'],self.gdark,p['cdark'],self.gdark*p['eta']/p['phi'])
        return tuple(Quantity(np.full(spec.shape,value),spec.unit,spec.meaning,self.evidence_tag) for spec,value in zip(self.state_specs,values))
    def initialize(self,time_seconds=0):
        from .engine import initialize_state
        return initialize_state(self,self.parameters,self.graph,self.initial_conditions(),author_discrete_config(),time_seconds)
    def unpack(self,y): return tuple(np.split(np.asarray(y),5))
    def _step(self,arrays,rate):
        R,P,G,C,S=arrays; p=self.p; dt=p['timeStep']
        # Preserve two additions in source R; no new-state substitution.
        new_R=R+dt*(-p['sigma']*R)
        new_R=new_R+p['gamma']*(rate*dt)
        new_P=P+dt*(R+p['eta']-p['phi']*P)
        new_C=C+dt*(self.q*p['k']*G**p['n']-p['beta']*C)
        new_S=self.smax/(1+(new_C/p['kGC'])**p['m'])
        new_G=G+dt*(S-P*G)
        new=(new_R,new_P,new_G,new_C,new_S)
        if any(not np.all(np.isfinite(v)) for v in new) or np.any(new_R<0) or any(np.any(v<=0) for v in new[1:]):
            raise ContractError('Source discrete trajectory invalid; no clipping/stabilization invented')
        return new
    def advance_author_discrete(self,state,segment,stop,config,sample_times=None):
        from .engine import validate_state
        validate_state(state,self)
        if self.input_policy==AUTHOR_INPUT_POLICY and self.parameters.run_mode!=RunMode.REFERENCE: raise ContractError('Author reproduction cannot enter another run mode')
        if segment.input_contract.get('chen2024_input_policy',STRICT_INPUT_POLICY)!=self.input_policy: raise ContractError('Stimulus/model input policy mismatch')
        if config!=state.solver_config or config!=author_discrete_config(): raise ContractError('Author fixed-step config mismatch')
        if (segment.unit,segment.meaning,segment.interpolation)!=('R*/s','photoisomerization_rate','zero_order_hold') or segment.evidence_tag!=state.evidence_tag:
            raise ContractError('Current model requires tagged total photoisomerization-rate ZOH')
        dt=self.p['timeStep']; start=state.time_seconds
        if not np.isfinite(stop): raise ContractError('Nonfinite stream stop')
        steps=int(round((stop-start)/dt))
        offset=round((start-segment.start_seconds)/dt)
        if not np.isfinite(stop) or steps<=0 or abs((stop-start)-steps*dt)>1e-10 or abs(start-segment.start_seconds-offset*dt)>1e-10 or start<segment.start_seconds-1e-12 or stop>segment.stop_seconds+1e-12:
            raise ContractError('Noncontiguous / off-grid author stream interval')
        if segment.input_contract.get('grid_step_seconds')!=dt or segment.input_contract.get('includes_background') is not True:
            raise ContractError('Source total-rate sampling contract required')
        rates=[]
        for i in range(steps):
            a=segment.at(segment.start_seconds+(offset+i)*dt).require('R*/s','photoisomerization_rate',(len(self.scope.cell_ids),))
            _validate_rate_input(a,self.input_policy)
            rates.append(a)
        exposure=_exposure_metadata(rates,dt,self.input_policy,start,stop)
        if exposure['warnings']:
            warnings.warn('CHEN2024_HIGH_LIGHT_APPLICABILITY_UNCERTAIN: peak '+str(exposure['peak_rate'])+' R*/s; no bleaching/regeneration; numerical author reproduction is not biological validation. '+PAPER_URL,HighLightApplicabilityWarning,stacklevel=2)
        arrays=tuple(a.copy() for a in state.ordered_state_arrays)
        trajectory=np.empty((5,len(self.scope.cell_ids),steps+1)); trajectory[:,:,0]=arrays
        for i,rate in enumerate(rates):
            arrays=self._step(arrays,rate); trajectory[:,:,i+1]=arrays
        times=start+np.arange(steps+1)*dt
        if sample_times is not None:
            samples=np.asarray(sample_times,dtype=float)
            indices=np.rint((samples-start)/dt).astype(int)
            if samples.ndim!=1 or not len(samples) or not np.all(np.isfinite(samples)) or np.any(np.diff(samples)<=0) or np.any(indices<0) or np.any(indices>steps) or not np.allclose(samples,times[indices],rtol=0,atol=1e-10):
                raise ContractError('Source outputs cannot silently resample off-grid times')
            times=times[indices]; trajectory=trajectory[:,:,indices]
        new=SimulationState(float(stop),arrays,state.state_spec_hash,state.graph_hash,state.parameter_hash,state.model_id,state.model_version,state.model_hash,state.input_interpolation,config,state.evidence_tag)
        output={spec.state_name:Quantity(trajectory[i],spec.unit,spec.meaning,self.evidence_tag) for i,spec in enumerate(self.state_specs)}
        output['time']=Quantity(times,'s','absolute_time',self.evidence_tag)
        output['current']=Quantity(-self.p['k']*trajectory[2]**self.p['n'],'pA','absolute_outer_segment_current',self.evidence_tag)
        output['J']=Quantity(self.p['k']*trajectory[2]**self.p['n'],'pA','outer_segment_current_magnitude',self.evidence_tag)
        for key,value,unit,meaning in [('q',self.q,'uM/(pA s)','source_current_to_calcium_coefficient'),('Smax',self.smax,'uM/s','maximum_cgmp_synthesis'),('dark_current_magnitude',self.dark_current,'pA','source_default_dark_current')]:
            output[key]=Quantity(value,unit,meaning,self.evidence_tag)
        result=SimulationResult(self.parameters.run_mode,self.model_id,self.parameters.source_manifest_hash,segment.input_contract,output,CoverageReport(True,(),claim_limit='; '.join(LIMITS)),{'method':config.method,'dt_seconds':dt,'dtype':'float64','steps_consumed':steps,'rtol_atol':'not used by source fixed-step update','sampling':'one interval input -> one update; output includes initial node'},'numeric_source_parity_only' if self.evidence_tag=='synthetic_fixture' else 'isolated_source_component',{'source_commit':COMMIT,'scope':self.scope.__dict__,'source_parameter_hash':self.parameters.parameter_hash,'claim_limit':list(LIMITS),'primary_dynamic_states':['R','P','G','C'],'algebraic_source_cache':'S','baseline_subtraction':'not applied','voltage_coupling':'BLOCKED'})
        result.metadata['input_policy']=self.input_policy
        result.metadata['input_exposure']=exposure
        result.metadata['biological_validity']='NOT_ESTABLISHED'
        result.metadata['parameter_support_domain_interpretation']='Ledger total_rate [0,50000) retained as strict project envelope; not source-code numerical cap; explicit author reproduction records applicability warnings'
        return new,result
    def forward_kernel_reference(self,sample_rates):
        a=sample_rates.require('R*/s','photoisomerization_rate')
        if a.ndim==1: a=a[:,None]
        if a.ndim!=2 or a.shape[0]<2 or a.shape[1]!=len(self.scope.cell_ids): raise ContractError('Author batch sample grid/domain invalid')
        if self.input_policy==AUTHOR_INPUT_POLICY and self.parameters.run_mode!=RunMode.REFERENCE: raise ContractError('Author reproduction cannot enter another run mode')
        _validate_rate_input(a,self.input_policy)
        if sample_rates.evidence_tag!=self.evidence_tag: raise ContractError('Batch input evidence mismatch')
        from .engine import advance
        intervals=Quantity(a[:-1],'R*/s','photoisomerization_rate',self.evidence_tag)
        segment=rate_segment(intervals,0,input_policy=self.input_policy)
        _,result=advance(self,self.initialize(),segment,segment.stop_seconds,author_discrete_config())
        result.metadata['author_sample_semantics']='N input samples, N output nodes; last sample not consumed, exactly as MATLAB source'
        result.metadata['provided_batch_sample_range']={'minimum_rate':float(a.min()),'peak_rate':float(a.max()),'sample_count':a.shape[0],'last_sample_consumed':False,'unit':'R*/s'}
        return result

def baseline_subtracted_current(current,baseline):
    a=current.require('pA','absolute_outer_segment_current')
    b=baseline.require('pA','absolute_outer_segment_current')
    if current.evidence_tag!=baseline.evidence_tag: raise ContractError('Baseline evidence mismatch')
    try: result=a-b
    except ValueError as e: raise ContractError('Explicit baseline shape mismatch') from e
    return Quantity(result,'pA','baseline_subtracted_outer_segment_current',current.evidence_tag)
