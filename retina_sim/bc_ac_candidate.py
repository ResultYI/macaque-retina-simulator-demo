"""Source-bound candidate parameters and topology-preserving BC/AC migration.

The distributed profile intentionally contains no numerical coefficients. It
cannot run until values, provenance and operating conditions are supplied.
"""
from pathlib import Path
from dataclasses import replace
import copy
import hashlib
import json
import numpy as np
from .contracts import ContractError, MissingCalibrationError, Quantity, RunMode, canonical_hash
from .assembly import RetinaBuilder
from .mechanism_contracts import EvidenceValue, MechanismProfile, register_mechanism_profile
from .bc_ac_receptor_components import ROUTE, BC_KEY, AC_KEY, BC_TYPES, AC_TYPES, POOLING

COMMON = {
    'C': ('F', 'membrane_capacitance'), 'gL': ('S', 'leak_conductance'),
    'EL': ('V', 'leak_reversal'), 'V0': ('V', 'initial_membrane_voltage'),
    'K': ('dimensionless', 'receptor_half_drive'),
    'hill': ('dimensionless', 'receptor_hill_exponent'),
    'tau_e': ('s', 'excitatory_receptor_time_constant'),
    'gE': ('S', 'maximum_excitatory_conductance'), 'EE': ('V', 'excitatory_reversal'),
    'e0': ('dimensionless', 'initial_receptor_gate'),
}
BC_PARAMETERS = dict(COMMON, release_half=('V', 'activation_half_voltage'),
    release_slope=('V', 'activation_voltage_slope'), tau_release=('s', 'release_time_constant'))
AC_PARAMETERS = dict(COMMON, activation_half=('V', 'effective_output_half_voltage'),
    activation_slope=('V', 'effective_output_voltage_slope'))


def required_parameters(cid):
    if cid in BC_TYPES:
        return dict(BC_PARAMETERS, **({'EI': ('V', 'inhibitory_reversal')} if BC_TYPES[cid] == 'OFF' else {}))
    if cid in AC_TYPES:
        return dict(AC_PARAMETERS)
    raise ContractError('Unknown approved BC/AC candidate target: ' + cid)


class CandidateParametersMissing(MissingCalibrationError):
    def __init__(self, missing):
        self.missing = missing
        super().__init__('BC_AC_CANDIDATE_UNBOUND: ' + '; '.join(
            name + ': ' + ', '.join(fields) for name, fields in missing.items()) +
            '. No guessed coefficients, fallback or default-route change.')


class CandidatePoolingUnresolved(ContractError):
    pass


class CandidateProfile:
    def __init__(self, path):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes(); self.sha256 = hashlib.sha256(raw).hexdigest()
        self.data = json.loads(raw)
        self._data_hash=canonical_hash(self.data)
        from .contact_first_receptor import ROUTE as V2, POOLING as V2_POOL, AGGREGATION
        self.contact_first=self.data.get('equation_route')==V2
        self.route=V2 if self.contact_first else ROUTE
        self.pooling=V2_POOL if self.contact_first else POOLING
        if self.contact_first and self.data.get('aggregation_contract')!=AGGREGATION:
            raise ContractError('Contact-first fixed-weight/shared-tau aggregation contract required')
        if (self.data.get('schema_version') != ('bc-ac-candidate-2.0' if self.contact_first else 'bc-ac-candidate-1.0') or
                self.data.get('equation_route') != self.route or
                self.data.get('pooling_order') != self.pooling or
                self.data.get('glutamate_concentration_claim') is not False or
                self.data.get('default_enabled') is not False or
                type(self.data.get('test_only')) is not bool or not self.data.get('profile_id')):
            raise ContractError('Explicit candidate profile identity, effective-drive and nondefault policy required')
        for cid, family in self.data['families'].items():
            definitions = self.definitions(cid)
            if set(family['parameters']) != set(definitions):
                raise ContractError('Candidate coefficient set differs from approved equations: ' + cid)
            for key, row in family['parameters'].items():
                if (row.get('unit'), row.get('meaning')) != definitions[key]:
                    raise ContractError('Candidate parameter unit/meaning mismatch: ' + cid + '.' + key)
                if row.get('binding') not in ('broadcast', 'per_cell', 'derived_background'):
                    raise ContractError('Candidate parameter needs explicit broadcast/per_cell binding')
                if row['binding']=='per_cell' and key not in ('V0','e0'):
                    raise ContractError('First candidate allows per-cell initialization only; no unmeasured per-cell coefficient freedom')
                if row['binding']=='derived_background' and (not self.contact_first or key not in ('K','V0','e0','release_half','activation_half')):
                    raise ContractError('Unsupported background-derived candidate binding')

    @property
    def test_only(self):
        return self.data['test_only']

    def definitions(self,cid):
        result=required_parameters(cid)
        if self.contact_first:
            result['gE']=('S','conductance_per_weighted_receptor_gate')
            result['e0']=('dimensionless','initial_weighted_receptor_gate')
        return result

    def profile_id(self, cid):
        return self.data['profile_id'] + '::' + cid

    def missing(self, targets):
        result = {}
        for cid in targets:
            definitions = self.definitions(cid)
            family = self.data['families'].get(cid, {})
            if not isinstance(family.get('conditions'), dict) or not family['conditions']:
                result[cid] = ['experimental_conditions']
            for key in definitions:
                row = family.get('parameters', {}).get(key, {})
                needed = ((row.get('value') is None and row.get('binding')!='derived_background') or not row.get('evidence_class') or
                    not row.get('locator') or not row.get('supported_domain') or not row.get('uncertainty') or
                    (not row.get('source_id') and row.get('evidence_class') != 'software_fixture'))
                if needed:
                    result.setdefault(cid, []).append(key)
        return result

    def require_ready(self, targets):
        if (canonical_hash(self.data)!=self._data_hash or
                hashlib.sha256(self.path.read_bytes()).hexdigest()!=self.sha256):
            raise ContractError('Candidate profile changed after its source identity was read')
        missing = self.missing(targets)
        if missing:
            raise CandidateParametersMissing(missing)
        for cid in targets:
            for key, row in self.data['families'][cid]['parameters'].items():
                if self.data.get('hypothesis_baseline') and (row.get('classification') not in
                        ('measured','borrowed','derived','effective_assumption','fitted') or
                        not row.get('rationale') or not row.get('sensitivity')):
                    raise ContractError('Hypothesis parameters require explicit classification, rationale and sensitivity')
                if self.data.get('hypothesis_baseline'):
                    if row['classification']=='measured' and row['evidence_class']!='measurement':
                        raise ContractError('Measured classification requires the existing primary-measurement evidence gate')
                    if row['classification']=='borrowed' and not row.get('reference_sources'):
                        raise ContractError('Borrowed classification requires its actual reference source')
                    if row['classification']=='fitted' and self.data['fitting_policy'].get('fit_performed') is not True:
                        raise ContractError('Fitted classification requires an actual matched-data fit, not this hypothesis baseline')
                if row['evidence_class'] == 'software_fixture' and not self.test_only:
                    raise ContractError('Synthetic coefficients cannot enter a non-test candidate: ' + cid + '.' + key)
                if row['binding']=='derived_background':
                    if not row.get('derivation'):raise ContractError('Background binding requires an explicit derivation')
                    continue
                value = np.asarray(row['value'])
                if value.dtype.kind not in 'iuf' or not np.all(np.isfinite(value)):
                    raise ContractError('Candidate coefficient must be finite numeric data: ' + cid + '.' + key)
        if not self.test_only and not self.contact_first:
            raise CandidatePoolingUnresolved('CANDIDATE_POOLING_UNRESOLVED: aggregate f(Wu) is a synthetic example only; '
                'contact-first Wf(u) is not generally equivalent. Production/source parameter binding requires '
                'a revised, explicitly approved aggregation contract; no additional contact states were inferred.')

    def bind(self, registry, component, population, mode, background_values=None):
        cid = component.component_id; self.require_ready((cid,))
        if self.test_only and (mode != RunMode.NUMERIC_FIXTURE or
                set(population.cells.evidence_tags) != {'synthetic_fixture'}):
            raise ContractError('SYNTHETIC_CANDIDATE_REQUIRES_SOFTWARE_FIXTURE: no animal/project-population relabelling')
        n = len(population.cells.cell_id)
        old = registry.mechanism_profiles[component.parameter_profile_id]
        tag = next(iter(set(population.cells.evidence_tags)))
        if len(set(population.cells.evidence_tags)) != 1:
            raise ContractError('Candidate population evidence tags must be explicit and uniform')
        records = []
        for key, row in self.data['families'][cid]['parameters'].items():
            value = np.asarray((background_values or {}).get(key) if row['binding']=='derived_background' else row['value'], dtype=float)
            if row['binding']=='derived_background':
                if value.shape==():value=np.full(n,float(value))
                if value.shape!=(n,) or not np.all(np.isfinite(value)):
                    raise ContractError('Explicit matching background context required for '+cid+'.'+key)
            elif row['binding'] == 'broadcast':
                if value.shape != ():
                    raise ContractError('Broadcast candidate coefficient must be scalar: ' + cid + '.' + key)
                value = np.full(n, float(value))
            elif value.shape != (n,):
                raise ContractError('Per-cell candidate coefficient order/shape mismatch: ' + cid + '.' + key)
            if self.test_only and row['evidence_class'] != 'software_fixture':
                raise ContractError('Synthetic candidate coefficients must remain software_fixture')
            records.append(EvidenceValue(key, Quantity(value, row['unit'], row['meaning'], tag),
                row['evidence_class'], row['source_id'], row['locator'], row['supported_domain'], row['uncertainty']))
        conditions = dict(self.data['families'][cid]['conditions'], equation_route=self.route,
            candidate_profile_sha256=self.sha256, initialization_parameters=['V0', 'e0'],
            effective_drive_not_concentration=True, pooling_order=self.pooling, test_only=self.test_only,
            background_context=(background_values or {}).get('_context'),
            parameter_provenance={k:{field:r.get(field) for field in ('classification','rationale','derivation','sensitivity','reference_sources')}
                for k,r in self.data['families'][cid]['parameters'].items()})
        profile = MechanismProfile(self.profile_id(cid), old.species, conditions, tuple(records),
            tuple(sorted({r.source_id for r in records if r.source_id})),
            'software_fixture' if self.test_only else 'project_approximation',
            approval_reference=self.data.get('approval','Sentinel_27c9730906a08191998a122fcc59e601: structure only; coefficients require their own provenance'),
            kind='unvalidated_receptor_candidate')
        register_mechanism_profile(registry, profile)
        return profile


def migrate_bc_ac(parent, registry, candidate, *, targets=None, background_inputs=None, coefficient_reference=None):
    """Replace only selected BC/AC equations; retain every connection and cell ID."""
    candidate = candidate if isinstance(candidate, CandidateProfile) else CandidateProfile(candidate)
    by_id = {c.component_id: c for c in parent.spec.components}
    targets = tuple(targets) if targets is not None else tuple(
        cid for cid in (*BC_TYPES, *AC_TYPES) if cid in by_id)
    if not targets or len(targets) != len(set(targets)) or any(cid not in by_id for cid in targets):
        raise ContractError('Explicit nonempty, unique existing candidate targets required')
    candidate.require_ready(targets)
    isolated = copy.copy(registry)
    isolated.sources = dict(registry.sources)
    isolated.profiles = dict(registry.profiles)
    isolated.mechanism_profiles = dict(registry.mechanism_profiles)
    background_values={}
    if candidate.data.get('hypothesis_baseline'):
        from .bc_ac_hypothesis import register_hypothesis_source, resolve_background_bindings
        register_hypothesis_source(isolated,candidate)
        background_values=resolve_background_bindings(parent,candidate,targets,background_inputs,coefficient_reference)
    pops = dict(parent.populations); specs = []
    for component in parent.spec.components:
        cid = component.component_id
        if cid not in targets:
            specs.append(component); continue
        required_parameters(cid)
        if component.mechanism not in (('HistoryBipolarTerminal', 'ResidualBC_ON', 'ResidualBC_OFF')
                if cid in BC_TYPES else ('ResidualAC',)):
            raise ContractError('Candidate migration requires an original retained BC/AC equation: ' + cid)
        pop_id = component.population_roles['cells']; population = pops[pop_id]
        profile = candidate.bind(isolated, component, population, parent.run_mode, background_values.get(cid))
        cells = replace(population.cells, parameter_profile_id=(profile.profile_id,) * len(population.cells.cell_id))
        pops[pop_id] = replace(population, cells=cells)
        contract = dict(route=candidate.route, profile_sha256=candidate.sha256, parameter_profile_id=profile.profile_id,
            pooling_order=candidate.pooling, glutamate_concentration_claim=False, test_only=candidate.test_only,
            input='native_cone_effective_drive' if cid in BC_TYPES else 'BC_effective_terminal_activation',
            per_contact_biological_gate_claim=False, physiology_validated=False)
        if cid in background_values:
            contract['background_initial_values']={k:profile.quantity(k,*candidate.definitions(cid)[k]).values.tolist()
                for k in ('V0','e0')}
            contract['hypothesis_baseline']=True
        # Old FIR options/coefficients are not carried into the new state equations.
        options = {key: value for key, value in component.options.items() if key in ('parent_cone_ids',)}
        specs.append(replace(component, mechanism=BC_KEY if cid in BC_TYPES else AC_KEY,
            parameter_profile_id=profile.profile_id, required_layer=cid,
            options=dict(options, candidate_dynamics=contract)))
    for link in parent.spec.links:
        if link.source_component in targets and link.source_port == 'generator_activation':
            raise ContractError('A consumer still requires the removed FIR diagnostic generator_activation')
    spec = replace(parent.spec, model_id=parent.spec.model_id + '.' + candidate.route,
        populations=tuple(pops[p.population_id] for p in parent.spec.populations), components=tuple(specs),
        scope=candidate.route)
    reduction=None
    if candidate.contact_first:
        from .bc_ac_contact_migration import contact_first_spec
        # Preserve the already rebound populations while transforming signal order.
        transformed,reduction=contact_first_spec(parent,specs,targets,candidate)
        spec=replace(spec,components=transformed.components,links=transformed.links,
                     external_inputs=transformed.external_inputs)
    result = RetinaBuilder(isolated).build(spec, parent.run_mode)
    if candidate.contact_first:
        from .bc_ac_contact_migration import validate_retained_tables
        validate_retained_tables(parent,result)
    elif canonical_hash((spec.links, spec.gap_junctions, spec.external_inputs)) != canonical_hash(
            (parent.spec.links, parent.spec.gap_junctions, parent.spec.external_inputs)):
        raise ContractError('Candidate migration changed frozen circuit connections')
    for key, population in parent.populations.items():
        if result.populations[key].cells.cell_id != population.cells.cell_id:
            raise ContractError('Candidate migration changed cell identity/order')
    for name in ('geometry_source_binding', 'input_geometry_binding_required', 'typed_view_domain',
                 'frozen_typed_geometry_binding', 'global_disk_circuit', 'global_disk_view', 'canonical_spatial_recipe'):
        if hasattr(parent, name):
            setattr(result, name, getattr(parent, name))
    result.candidate_dynamics = dict(equation_route=candidate.route, profile_path=str(candidate.path),
        profile_sha256=candidate.sha256, targets=list(targets), test_only=candidate.test_only,
        compute_policy='python_reference_only_unlowered_candidate', original_default_unchanged=True,
        connectivity_and_cell_ids_preserved=True, pooling_order=candidate.pooling,
        reduction=reduction, extra_contact_dynamic_states=0,
        activation_scale='new bounded effective output; old downstream gains remain uncalibrated',
        background='New state layout; old checkpoint/background is not a compatible adapted state',
        physiology_validated=False)
    if candidate.data.get('hypothesis_baseline'):
        from collections import Counter
        result.candidate_dynamics.update(hypothesis_baseline=True,parameter_fit_performed=False,
            classification_counts=dict(Counter(r['classification'] for cid in targets
                for r in candidate.data['families'][cid]['parameters'].values())),
            fitting_policy=candidate.data['fitting_policy'],
            initial_state_policy=candidate.data['initialization_policy'])
    return result, isolated, result.candidate_dynamics
