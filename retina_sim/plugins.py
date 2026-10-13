"""Typed missing-physiology plugin boundaries; no invented rod/BC/AII equations."""
from dataclasses import dataclass
from pathlib import Path
import inspect,hashlib
from .contracts import ContractError,RunMode,canonical_hash

BOUNDARIES={
    'RodMechanismPlugin':{'cell_type':'rod','input':'light','output':'rod_signal','unmodeled':'Rod transduction, adaptation, native/physical output calibration and species/context parameters'},
    'RodBipolarMechanismPlugin':{'cell_type':'rod_BC','input':'rod_drive','output':'bipolar_signal','unmodeled':'Rod bipolar synaptic mechanism, kinetics, gain and anatomical convergence'},
    'AIIMechanismPlugin':{'cell_type':'AII_AC','input':'rod_bipolar','output':'crossover_signal','unmodeled':'AII intrinsic dynamics, cone crossover polarity and physical gap-junction mechanisms'},
}
@dataclass(frozen=True)
class PluginDeclaration:
    boundary_name: str
    backend_id: str
    evidence_class: str
    source_ids: tuple
    mechanism_locator: str
    implemented_scope: str
    unmodeled_scope: tuple
    input_contracts: dict
    output_contracts: dict
    layers: tuple
    def __post_init__(self):
        if self.boundary_name not in BOUNDARIES or not self.backend_id or not self.mechanism_locator or not self.implemented_scope or not self.unmodeled_scope or not self.layers:raise ContractError('Complete source/fixture plugin declaration required')
        if self.evidence_class not in ('published_fit','author_assumption','project_approximation','software_fixture'):raise ContractError('Explicit plugin evidence class required')
        if self.evidence_class!='software_fixture' and not self.source_ids:raise ContractError('Physiology plugin requires verified mechanism evidence; no cone substitution/default rod equation')
        boundary=BOUNDARIES[self.boundary_name]
        if boundary['input'] not in self.input_contracts or boundary['output'] not in self.output_contracts:raise ContractError('Required rod pathway plugin ports are not declared')
        for value in tuple(self.input_contracts.values())+tuple(self.output_contracts.values()):
            if set(value)!={'unit','meaning'} or not value['unit'] or not value['meaning']:raise ContractError('Plugin port units/semantics cannot be unknown or inferred')

def install_plugin_boundaries(registry):
    registry._physiology_plugins={}
    for name in BOUNDARIES:
        def factory(spec,populations,sources,mode,name=name):
            backend=spec.options.get('backend_id')
            if (name,backend) not in registry._physiology_plugins:raise ContractError(name+': required mechanism backend is unbound; '+BOUNDARIES[name]['unmodeled'])
            declaration,provider,provider_hash=registry._physiology_plugins[(name,backend)]
            if set(spec.population_roles)!={'cells'}:raise ContractError('Plugin requires explicit cells population role')
            pop=populations[spec.population_roles['cells']]
            if set(pop.cells.cell_type)!={BOUNDARIES[name]['cell_type']}:raise ContractError('Plugin cell type mismatch; no cone/BC/AII renaming')
            profile=getattr(sources,'mechanism_profiles',{}).get(spec.parameter_profile_id)
            if profile is None:raise ContractError('Plugin numerical mechanism profile unbound')
            if declaration.evidence_class=='software_fixture':
                if mode!=RunMode.NUMERIC_FIXTURE or profile.evidence_class!='software_fixture' or set(pop.cells.evidence_tags)!={'synthetic_fixture'}:raise ContractError('Engineering plugin is synthetic-only, not a physiological/source profile')
            elif profile.evidence_class=='software_fixture':raise ContractError('Software coefficients cannot satisfy physiological plugin evidence')
            for source in declaration.source_ids:
                if source not in sources.sources:raise ContractError('Plugin mechanism source is not registered/verified')
                sources.sources[source].verify(sources.root)
            c=provider(spec,populations,sources,mode)
            if c.parameters.parameter_hash!=profile.parameter_hash or (len(c.populations)!=1 or c.populations[0] is not pop):raise ContractError('Plugin profile/population identity mismatch')
            if set(c.inputs)!=set(declaration.input_contracts) or set(c.outputs)!=set(declaration.output_contracts):raise ContractError('Plugin implementation ports differ from declared boundary')
            for ports,declared in ((c.inputs,declaration.input_contracts),(c.outputs,declaration.output_contracts)):
                for port,contract in declared.items():
                    actual=ports[port]
                    if (actual.unit,actual.meaning,actual.cell_ids)!=(contract['unit'],contract['meaning'],pop.cells.cell_id):raise ContractError('Plugin implemented port semantics/units/cells mismatch')
            if c.evidence_tag!=pop.cells.evidence_tags[0]:raise ContractError('Plugin output evidence inconsistent')
            c.layers=declaration.layers;c.source_ids=declaration.source_ids
            c.implementation_hash=canonical_hash({'provider_source_sha256':provider_hash,'declaration':declaration,'boundary_implementation':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
            c.plugin_declaration=declaration
            return c
        registry.register(name,factory)

def bind_mechanism_plugin(registry,declaration,provider):
    if not isinstance(declaration,PluginDeclaration) or not callable(provider):raise ContractError('Actual executable plugin provider and declaration required')
    key=(declaration.boundary_name,declaration.backend_id)
    if key in registry._physiology_plugins:raise ContractError('Duplicate mechanism plugin backend')
    try:source=inspect.getsource(provider)
    except (OSError,TypeError):raise ContractError('Plugin provider must have inspectable source; opaque callbacks are rejected')
    registry._physiology_plugins[key]=(declaration,provider,hashlib.sha256(source.encode()).hexdigest())

def boundary_status(registry):
    return {name:{'boundary_implemented':True,'default_physiology_mechanism':None,'reason':contract['unmodeled'],'bound_backends':[backend for boundary,backend in registry._physiology_plugins if boundary==name]} for name,contract in BOUNDARIES.items()}
