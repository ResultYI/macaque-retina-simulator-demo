"""Additional typed mechanisms without altering the frozen legacy reference contracts."""
from dataclasses import dataclass
import numpy as np
from .contracts import ContractError,Quantity,canonical_hash

STATE_UNITS={'native_illuminance':'td','native_precursor':'vH_native_precursor','native_os_current':'vH_native_current','native_relative_voltage':'vH_native_voltage','native_conductance':'vH_native_conductance','effective_transmitter_drive':'vH_native_voltage','receptor_gate':'dimensionless','effective_release_activation':'dimensionless','adaptation_gate':'dimensionless','excitatory_synaptic_gate':'dimensionless','inhibitory_synaptic_gate':'dimensionless','physical_membrane_voltage':'V','physical_voltage_deviation':'V'}
STATE_UNITS.update({'causal_synaptic_drive_history':'dimensionless','effective_synaptic_activation':'dimensionless'})
STATE_UNITS['effective_HC_activation']='dimensionless'
STATE_UNITS.update({'source_scaled_cgmp':'Chen_scaled_concentration','source_scaled_calcium':'Chen_scaled_concentration','source_scaled_cgmp_synthesis':'Chen_scaled_concentration/s'})

@dataclass(frozen=True)
class MechanismStateSpec:
    state_name: str
    cell_ids: tuple
    shape: tuple
    unit: str
    meaning: str
    initial_condition_source: str
    def __post_init__(self):
        if self.shape!=(len(self.cell_ids),) or not self.cell_ids or len(set(self.cell_ids))!=len(self.cell_ids): raise ContractError('Mechanism state cell layout mismatch')
        if self.meaning in STATE_UNITS:
            if self.unit!=STATE_UNITS[self.meaning]: raise ContractError('Mechanism state semantic/unit mismatch')
        else:
            from .contracts import StateSpec
            StateSpec(self.state_name,self.cell_ids,self.shape,self.unit,self.meaning,self.initial_condition_source)
        if not self.initial_condition_source: raise ContractError('Initial condition source required')

@dataclass(frozen=True)
class EvidenceValue:
    parameter_id: str
    value: Quantity
    evidence_class: str
    source_id: str | None
    locator: str
    supported_domain: dict
    uncertainty: str
    def __post_init__(self):
        if self.evidence_class not in ('measurement','published_fit','author_assumption','project_approximation','software_fixture'): raise ContractError('Unknown parameter evidence class')
        if not self.parameter_id or not self.locator or not self.uncertainty or not self.supported_domain: raise ContractError('Parameter context/units/evidence required')
        if self.evidence_class!='software_fixture' and not self.source_id: raise ContractError('Sourced/approximation parameters require evidence source')
        if self.evidence_class=='software_fixture' and self.value.evidence_tag!='synthetic_fixture': raise ContractError('Fixture parameter tag required')
        if not isinstance(self.supported_domain,dict): raise ContractError('Evidence supported_domain must be an object')

@dataclass(frozen=True)
class MechanismProfile:
    profile_id: str
    species: str
    conditions: dict
    records: tuple
    source_ids: tuple
    evidence_class: str
    approval_reference: str | None = None
    kind: str = 'named_reference'
    @property
    def profile(self): return self
    @property
    def parameter_hash(self): return canonical_hash(self)
    @property
    def source_manifest_hash(self): return canonical_hash({'sources':self.source_ids,'records':self.records})
    def quantity(self,name,unit,meaning,shape=None,positive=False,nonnegative=False):
        matches=[r for r in self.records if r.parameter_id==name]
        if len(matches)!=1: raise ContractError('Required parameter missing/duplicated: '+name)
        q=matches[0].value; a=q.require(unit,meaning,shape)
        if positive and np.any(a<=0) or nonnegative and np.any(a<0): raise ContractError('Parameter domain invalid: '+name)
        return q

def register_mechanism_profile(registry,profile):
    if profile.profile_id in registry.profiles: raise ContractError('Duplicate profile')
    if profile.evidence_class=='project_approximation' and not profile.approval_reference: raise ContractError('Named project approximation requires approval reference')
    if profile.species in ('human','cat'): raise ContractError('Human/cat not accepted in macaque mechanism registry')
    for source in profile.source_ids:
        if source not in registry.sources: raise ContractError('Unknown profile source '+source)
    validate_mechanism_evidence(registry,profile)
    if not hasattr(registry,'mechanism_profiles'): registry.mechanism_profiles={}
    registry.mechanism_profiles[profile.profile_id]=profile
    registry.profiles[profile.profile_id]=profile

def validate_mechanism_evidence(registry,profile):
    """Asset identity gates evidence classes; this does not establish physiology."""
    if len({r.parameter_id for r in profile.records}) != len(profile.records):
        raise ContractError('Duplicate dynamic parameter ID')
    for record in profile.records:
        if record.source_id is None:
            if record.evidence_class != 'software_fixture':raise ContractError('Missing dynamic source identity')
            continue
        if record.source_id not in registry.sources:raise ContractError('Unknown parameter source')
        source=registry.sources[record.source_id]
        from .provenance import validate_evidence_identity
        validate_evidence_identity(source,registry.root,record.evidence_class,record.locator,
            parameter_id=record.parameter_id,value=record.value.values,unit=record.value.unit,context=record.supported_domain)
        if record.evidence_class=='published_fit' and source.verification_level not in ('code_verified','numeric_table_verified','primary_measurement_verified'):
            raise ContractError(record.parameter_id+': published fit needs verified code/table or primary source')
        domain=record.supported_domain
        if not isinstance(domain,dict) or not domain:raise ContractError('Dynamic parameter domain object required')
        for axis,limits in domain.items():
            if isinstance(limits,dict) and {'unit','lower','upper'} <= set(limits):
                lo,hi=limits['lower'],limits['upper']
                if not limits['unit'] or any(x is not None and (type(x) not in (int,float) or not np.isfinite(x)) for x in (lo,hi)):
                    raise ContractError('Invalid dynamic source domain '+axis)
                if any(k in limits and type(limits[k]) is not bool for k in ('lower_closed','upper_closed')):
                    raise ContractError('Source domain closure must be boolean '+axis)
                if lo is not None and hi is not None and lo>=hi:raise ContractError('Empty dynamic source domain '+axis)
        if record.evidence_class=='measurement' and not any(k in domain for k in ('type','species','cell_types','recording_context')):
            raise ContractError('Measurement type/recording domain is required')

def validate_population_evidence_domain(profile,populations):
    """Structured applicability constraints are executable, unlike legacy free text."""
    for record in profile.records:
        domain=record.supported_domain
        for pop in populations:
            if 'cell_types' in domain:
                allowed=domain['cell_types']
                if not isinstance(allowed,(list,tuple)) or not allowed or not set(pop.cells.cell_type)<=set(allowed):
                    raise ContractError(record.parameter_id+': cell type outside source domain')
            if 'subtypes' in domain:
                allowed=domain['subtypes']
                if not isinstance(allowed,(list,tuple)) or not allowed or not set(pop.cells.subtype)<=set(allowed):
                    raise ContractError(record.parameter_id+': subtype outside source domain')
            if 'eccentricity_deg' in domain:
                limits=domain['eccentricity_deg']
                if limits.get('unit')!='deg':raise ContractError('Eccentricity source domain must declare deg')
                values=pop.cells.eccentricity.require('deg','global_eccentricity')
                for side,comparison in (('lower',np.less),('upper',np.greater)):
                    bound=limits.get(side)
                    if bound is not None and np.any(comparison(values,bound)):
                        raise ContractError(record.parameter_id+': eccentricity outside source domain')
                    if bound is not None and not limits.get(side+'_closed',True) and np.any(values==bound):
                        raise ContractError(record.parameter_id+': open source-domain endpoint')

def explicit_unit_conversion(q,target_unit):
    factors={('pA','A'):1e-12,('A','pA'):1e12,('mV','V'):1e-3,('V','mV'):1e3,('ms','s'):1e-3,('s','ms'):1e3,('nS','S'):1e-9,('S','nS'):1e9,('pF','F'):1e-12,('F','pF'):1e12}
    if (q.unit,target_unit) not in factors: raise ContractError('No declared physical unit conversion; native/physical or contrast/light bridge needs calibration')
    return Quantity(q.values*factors[(q.unit,target_unit)],target_unit,q.meaning,q.evidence_tag,q.coordinate_frame)
