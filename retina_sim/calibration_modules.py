"""Clamped experiments using the same factories, parameters and runner as the circuit.

No fitting algorithm or new cell dynamics live here. All time arrays are explicit
bin edges in seconds; no frame-rate inference, resampling or current reinterpretation.
"""
from dataclasses import dataclass, replace, asdict, is_dataclass
from pathlib import Path
import copy
import json
import numpy as np
from .assembly import ModelSpec, ExternalInput, RetinaBuilder
from .contracts import ContractError, Quantity, canonical_hash
from .spatial import InputChannel

MECHANISMS = {
    'cone_hc_joint': {'VanHateren2005NativeConeH1'},
    'bc_clamped': {'HistoryBipolarTerminal', 'ResidualBC_ON', 'ResidualBC_OFF','CandidateConeBipolarReceptorMembrane','ContactFirstConeBipolarReceptorMembraneV2'},
    'generic_ac_clamped': {'ResidualAC','CandidateGenericAmacrineReceptorMembrane','ContactFirstGenericAmacrineReceptorMembraneV2'},
    'rgc': {'HistoryConductanceRGC'},
}
MODULE_MODES = {'rgc': ('conductance_clamp', 'current_clamp'),
    'cone_hc_joint': ('native_light_drive',), 'bc_clamped': ('declared_input_clamps',),
    'generic_ac_clamped': ('declared_input_clamps',)}

# A local fit is evidence for its experiment, never automatic whole-circuit
# acceptance. Legacy snapshots remain identity-only; fitted proposals are opt-in.
SYSTEM_INTEGRATION_POLICY = dict(
    schema_version='calibration-system-integration-1.0',
    shared_equations_parameters_units_required=True,
    boundary_changes_require_explicit_review=['interface_scale', 'input_background', 'adaptation_state', 'feedback'],
    required_reassembly_checks=['same_condition_inputs_and_prepared_background', 'upstream_layer_responses',
        'changed_layer_responses', 'downstream_layer_responses', 'all_assembled_RGC_targets'],
    criteria='Predeclare experimental conditions, metrics and tolerances; none are inferred from a local fit',
    local_fit_is_system_acceptance=False, arbitrary_other_layer_compensation_allowed=False,
    system_checks_status='not_performed', automatic_fitting_enabled=False)


def system_compatibility(parent):
    """Freeze full-network bindings without claiming a prepared background run."""
    return dict(schema_version='calibration-system-context-1.0',
        parent_assembly_hash=parent.assembly_hash,
        component_identities={cid:c.identity for cid,c in parent.components.items()},
        parameter_hashes={cid:c.parameters.parameter_hash for cid,c in parent.components.items()},
        interface_hashes={cid:canonical_hash((c.inputs,c.outputs)) for cid,c in parent.components.items()},
        binding_options_hashes={cid:c.binding_options_hash for cid,c in parent.components.items()},
        connectivity_hash=canonical_hash((parent.spec.links,parent.spec.gap_junctions,parent.spec.external_inputs)),
        population_hashes={pid:p._population_hash for pid,p in parent.populations.items()},
        layer_inventory={cid:list(c.layers) for cid,c in parent.components.items() if c.layers},
        assembled_RGC_targets=[cid for cid,c in parent.components.items() if c.mechanism=='HistoryConductanceRGC'],
        external_input_and_adapted_background='Not supplied or validated by module construction; require same-condition re-preparation and checks',
        integration_policy=copy.deepcopy(SYSTEM_INTEGRATION_POLICY))


def gap_boundary_id(gap):
    return canonical_hash(gap)


def affected_closure(spec, changed):
    """Directed chemical dependencies plus reciprocal electrical feedback."""
    reached = set(changed)
    edges = [(l.source_component, l.target_component) for l in spec.links]
    edges += [(a, b) for g in spec.gap_junctions for a, b in
              ((g.left_component, g.right_component), (g.right_component, g.left_component))]
    while True:
        expanded = reached | {b for a, b in edges if a in reached}
        if expanded == reached:
            return sorted(reached)
        reached = expanded


@dataclass
class CalibrationModule:
    compiled: object
    registry: object
    manifest: dict
    baselines: dict

    def inputs(self, traces):
        cid = self.manifest['component_id']
        component = self.compiled.components[cid]
        if set(traces) != set(component.inputs):
            raise ContractError('Supply exactly every declared module boundary port')
        channels = {}
        intervals = set()
        for name, port in component.inputs.items():
            trace = traces[name]
            if trace.time_origin != self.manifest['conditions']['time_origin']:
                raise ContractError('Module input time origin differs from experiment')
            channels['module_' + name] = trace.channel(port, self.baselines[name])
            intervals.add((float(trace.edges_seconds[0]), float(trace.edges_seconds[-1])))
        if len(intervals) != 1:
            raise ContractError('Module traces require the same explicit time coverage')
        return channels


def build_module(parent, registry, component_id, *, kind, mode, conditions, baselines,
                 open_gap_boundaries=()):
    """Open and declare every external boundary; retain internal Cone-HC feedback.

    `baselines` contains typed per-cell quantities for every input. The parent's
    initial states are retained; they are never relabelled an adapted steady state.
    For causal BC/AC, only the derived input reference is rebound to the supplied
    baseline. No fitted kernel, membrane, release or spike coefficient is changed.
    """
    if kind not in MECHANISMS or component_id not in parent.components:
        raise ContractError('Unknown calibration module/component')
    original = next(c for c in parent.spec.components if c.component_id == component_id)
    if original.mechanism not in MECHANISMS[kind]:
        raise ContractError('Module kind does not match the retained component factory')
    if mode not in MODULE_MODES[kind]:
        raise ContractError('Unsupported experimental boundary mode')
    required = ('source_id', 'species', 'retinal_region', 'baseline_source', 'time_origin', 'adaptation_status')
    if any(not isinstance(conditions.get(k), str) or not conditions[k] for k in required):
        raise ContractError('Explicit experimental source, region, baseline, adaptation and time origin required')
    # The existing native factory stores H1 and H2 in the same `h1` HC table.
    if kind == 'cone_hc_joint' and not {'cone', 'h1'} <= set(original.population_roles):
        raise ContractError('Cone-HC calibration must keep the complete joint component')
    gaps = [g for g in parent.spec.gap_junctions if component_id in (g.left_component, g.right_component)]
    gap_ids = {gap_boundary_id(g) for g in gaps}
    if set(open_gap_boundaries) != gap_ids:
        raise ContractError('Explicitly acknowledge every opened electrical gap boundary; no silent feedback removal')
    runtime = parent.components[component_id]
    ports = dict(runtime.inputs)
    if mode == 'current_clamp':
        from .assembly import PortSpec
        ids = runtime.outputs['voltage'].cell_ids
        ports['injected_current'] = PortSpec('injected_current', 'A',
            'injected_current_positive_inward', ids, runtime.evidence_tag)
    if set(baselines) != set(ports):
        raise ContractError('Typed baselines required for every module input, including zero clamps')
    for name, port in ports.items():
        baselines[name].require(port.unit, port.meaning, (len(port.cell_ids),))
        if port.unit in ('S', 'td', 'R*/s') and np.any(baselines[name].values < 0):
            raise ContractError('Conductance/light baselines must be nonnegative')
    isolated_registry = copy.copy(registry)
    isolated_registry.profiles = dict(registry.profiles)
    isolated_registry.mechanism_profiles = dict(registry.mechanism_profiles)
    parameter_profile = isolated_registry.mechanism_profiles.get(original.parameter_profile_id)
    if kind in ('bc_clamped', 'generic_ac_clamped') and parameter_profile is not None:
        records = tuple(replace(r, value=Quantity(baselines['drive'].values, r.value.unit,
            r.value.meaning, r.value.evidence_tag), locator=r.locator + '; declared isolated input baseline')
            if r.parameter_id == 'baseline' else r for r in parameter_profile.records)
        parameter_profile = replace(parameter_profile, records=records)
        isolated_registry.mechanism_profiles[parameter_profile.profile_id] = parameter_profile
        isolated_registry.profiles[parameter_profile.profile_id] = parameter_profile
    boundary = dict(mode=mode, parent_assembly_hash=parent.assembly_hash, ports=sorted(ports),
                    opened_gap_boundaries=sorted(gap_ids), conditions=copy.deepcopy(conditions))
    options = dict(original.options, module_boundary=boundary, current_clamp_boundary=mode == 'current_clamp')
    if kind == 'cone_hc_joint':
        # Reuse the existing factory's background initialization, with the actual
        # declared module light reference rather than a different parent light.
        if np.any(baselines['light'].values <= 0):
            raise ContractError('Native Cone-HC initialization requires a positive declared background')
        options['background_td'] = baselines['light']
    component = replace(original, options=options)
    factories = None
    if mode == 'current_clamp':
        from .components import reference_component_registry
        from .calibration_current_clamp import KEY, current_clamp_component
        factories = reference_component_registry()
        factories.register(KEY, current_clamp_component)
        component = replace(component, mechanism=KEY)
    populations = tuple(p for p in parent.spec.populations if p.population_id in original.population_roles.values())
    spec = ModelSpec(parent.spec.model_id + '.module.' + component_id, populations, (component,),
        external_inputs=tuple(ExternalInput('module_' + port, component_id, port) for port in ports),
        required_mechanisms=(original.required_layer,) if original.required_layer else (),
        scope='isolated_module_clamped_v1', clock_policy=parent.spec.clock_policy)
    compiled = RetinaBuilder(isolated_registry, components=factories).build(spec, parent.run_mode)
    isolated = compiled.components[component_id]
    if runtime.implementation_hash != getattr(isolated, 'base_implementation_hash', isolated.implementation_hash):
        raise ContractError('Independent module and full circuit must use the same implementation')
    manifest = dict(schema_version='calibration-module-1.0', kind=kind, mode=mode,
        component_id=component_id, mechanism=isolated.mechanism, base_mechanism=original.mechanism,
        parent_assembly_hash=parent.assembly_hash, module_assembly_hash=compiled.assembly_hash,
        implementation_hash=isolated.implementation_hash, base_implementation_hash=runtime.implementation_hash,
        compute_policy='python_reference_only' if mode == 'current_clamp' or 'candidate_dynamics' in original.options else 'existing_reviewed_backend_selection',
        conditions=copy.deepcopy(conditions),
        boundary=boundary, equation_revision=getattr(parent, 'modeling_scope', {}).get(
            'equation_revision', 'historical_retained_component_equations'),
        initial_state=('existing joint factory at declared native light reference; background adaptation not verified'
            if kind == 'cone_hc_joint' else 'retained factory state; background adaptation not verified'),
        feedback='joint internal Cone-HC retained' if kind == 'cone_hc_joint' else 'external feedback opened and inputs explicitly clamped',
        free_fit_parameters=[], automatic_fitting_enabled=False,
        system_compatibility=system_compatibility(parent),
        invalidation={'affected_neural_components': affected_closure(parent.spec, (component_id,)),
            'background':'Affected feedback closure must be prepared again; old background is only a candidate initial state',
            'optical_input':'Reuse only if optics, display, stimulus and receptor geometry identities are unchanged'})
    compiled.calibration_module = manifest
    if hasattr(parent,'candidate_dynamics'):
        manifest['equation_revision']=parent.candidate_dynamics['equation_route']
        manifest['candidate_dynamics']=parent.candidate_dynamics
        if manifest['equation_revision']=='bc_ac_contact_first_candidate_v2':
            manifest['drive_boundary']='Post-contact weighted occupancy Wf(u); not raw presynaptic u. Use contact_first_receptor.aggregate_occupancy with original W/K/hill.'
    return CalibrationModule(compiled, isolated_registry, manifest, dict(baselines))


@dataclass(frozen=True)
class PortTrace:
    edges_seconds: object
    values: object
    unit: str
    meaning: str
    cell_ids: tuple
    baseline: object
    recording_kind: str
    source_id: str
    time_origin: str

    def channel(self, port, required_baseline):
        edges = np.asarray(self.edges_seconds, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if (edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or
                np.any(np.diff(edges) <= 0) or values.shape != (len(edges) - 1, len(port.cell_ids)) or
                not np.all(np.isfinite(values))):
            raise ContractError('Explicit increasing bin edges and finite bin-by-cell values required')
        if (self.unit, self.meaning, tuple(self.cell_ids)) != (port.unit, port.meaning, tuple(port.cell_ids)):
            raise ContractError('Module trace unit/meaning/cell order mismatch; no implicit conversion')
        expected_kind = 'synaptic_conductance' if port.unit == 'S' else (
            'injected_current' if port.meaning == 'injected_current_positive_inward' else
            'native_light' if port.unit in ('td', 'R*/s') else 'effective_drive')
        if self.recording_kind != expected_kind or not self.source_id or not self.time_origin:
            raise ContractError('Recording boundary type mismatch; voltage-clamp current is not injected current or conductance')
        if self.unit in ('S', 'td', 'R*/s') and np.any(values < 0):
            raise ContractError('Negative conductance/light values require source review; no silent rectification')
        if not np.array_equal(np.asarray(self.baseline), required_baseline.values):
            raise ContractError('Trace baseline differs from declared module operating reference')
        # Copy arrays: a caller changing its original data cannot alter a live run.
        edges = edges.copy(); values = values.copy()
        def evaluate(t):
            i = min(max(int(np.searchsorted(edges, t, side='right')) - 1, 0), len(values) - 1)
            return Quantity(values[i], self.unit, self.meaning, port.evidence_tag)
        return InputChannel(float(edges[0]), float(edges[-1]), evaluate, self.unit, self.meaning,
            'zero_order_hold', port.evidence_tag,
            dict(source_id=self.source_id, recording_kind=self.recording_kind, time_origin=self.time_origin,
                 baseline=np.asarray(self.baseline).tolist(), preprocessing='none; explicit source bin edges'),
            tuple(float(t) for t in edges[1:-1]))


def voltage_clamp_conductance(current_A, *, holding_V, reversal_V, receptor,
                              leak_subtracted, isolated_receptor_current, source_id):
    """Explicit I_outward=g*(Vhold-Erev) conversion; no fitted reversal or clipping."""
    if (receptor not in ('excitatory', 'inhibitory') or leak_subtracted is not True or
            isolated_receptor_current is not True or not source_id):
        raise ContractError('Require isolated, leak-subtracted receptor current and a source')
    if not isinstance(current_A, Quantity):
        raise ContractError('Voltage-clamp current requires an explicitly typed A quantity')
    current = current_A.require('A', 'voltage_clamp_current_positive_outward')
    hold = np.asarray(holding_V, dtype=float); reversal = np.asarray(reversal_V, dtype=float)
    if (not np.all(np.isfinite(current)) or not np.all(np.isfinite(hold)) or
            not np.all(np.isfinite(reversal)) or np.any(hold == reversal)):
        raise ContractError('Finite explicit voltages and a nonzero driving force required')
    try:
        conductance = current / (hold - reversal)
    except ValueError as error:
        raise ContractError('Voltage/current array shapes do not broadcast') from error
    if np.any(conductance < 0):
        raise ContractError('Current sign/driving force produces negative conductance; no silent rectification')
    return Quantity(conductance, 'S', receptor + '_conductance', 'project_approximation'), {
        'source_id': source_id, 'input_current_sign': 'positive_outward', 'holding_V': hold.tolist(),
        'reversal_V': reversal.tolist(), 'conversion': 'I_outward/(Vhold-Erev)',
        'assumptions': 'isolated receptor current, leak subtracted; effective voltage clamp, no series/space-clamp correction inferred'}


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _parameter_records(parameters, definitions):
    """Keep native ParameterSet records and dynamic EvidenceValue records distinct."""
    records, derived = {}, {}
    native = not hasattr(parameters.profile, 'records')
    for row in parameters.records:
        initialization=parameters.profile.conditions.get('initialization_parameters',())
        target = derived if definitions.get(row.parameter_id, {}).get('binding') == 'derived' or row.parameter_id in initialization else records
        if native:
            # Native source records have scalar values, evidence_kind and a full
            # source locator/domain, not a Quantity or invented semantic meaning.
            target[row.parameter_id] = _plain(row)
        else:
            target[row.parameter_id] = dict(value=row.value.values.tolist(), unit=row.value.unit,
                meaning=row.value.meaning, evidence_class=row.evidence_class, source_id=row.source_id,
                locator=row.locator, supported_domain=row.supported_domain, uncertainty=row.uncertainty)
    return records, derived, 'native_ParameterSet' if native else 'mechanism_EvidenceValue'


def parameter_package(module, *, version, source_id, applicability):
    """Save an unfitted, mergeable coefficient snapshot; initial states stay separate."""
    if not version or not source_id or not isinstance(applicability, dict) or not applicability:
        raise ContractError('Parameter version, source and applicability required')
    cid = module.manifest['component_id']; component = module.compiled.components[cid]
    runtime = getattr(module.registry, 'runtime_profile', None)
    definitions = runtime.data['profiles'].get(component.parameters.profile.profile_id, {}) if runtime else {}
    records, derived, record_format = _parameter_records(component.parameters, definitions)
    package = dict(schema_version='module-parameters-1.1', version=version, source_id=source_id,
        status='unfitted_snapshot', applicability=copy.deepcopy(applicability),
        component_id=cid, profile_id=component.parameters.profile.profile_id,
        implementation_hash=component.implementation_hash,
        base_implementation_hash=module.manifest['base_implementation_hash'],
        parameter_record_format=record_format,
        native_profile_metadata=_plain(component.parameters.profile) if record_format == 'native_ParameterSet' else None,
        native_parameter_set_identity=dict(parameter_hash=component.parameters.parameter_hash,
            source_manifest_hash=component.parameters.source_manifest_hash) if record_format == 'native_ParameterSet' else None,
        base_profile_identity=dict(profile_id=runtime.data['profile_id'], sha256=runtime.sha256,
            scope_sha256=runtime.modeling_scope.sha256 if runtime.modeling_scope else None) if runtime else None,
        equation_revision=module.manifest['equation_revision'],
        parent_assembly_hash=module.manifest['parent_assembly_hash'], records=records,
        system_compatibility=copy.deepcopy(module.manifest['system_compatibility']),
        initialization_not_transferable=derived, invalidation=module.manifest['invalidation'])
    return dict(package, content_hash=canonical_hash(package))


def save_parameter_package(package, path):
    # Exclusive creation keeps old versions and user edits intact.
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(package, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def merge_parameter_package(parent, registry, package, *, applicability, allowed_fields=None, proposal_path=None):
    """Rebuild identity snapshots, or an explicit restricted proposal revision.

    Fitted proposals additionally require an exact caller whitelist and persisted
    artifact. Neither path constitutes whole-system physiological acceptance.
    """
    if package.get('schema_version')=='rgc-source-reference-candidate-1.0':
        from .source_reference_candidate import merge_source_reference_candidate
        return merge_source_reference_candidate(parent,registry,package,applicability=applicability,
            allowed_fields=allowed_fields,proposal_path=proposal_path)
    if package.get('schema_version')=='module-fitted-proposal-1.0':
        from .calibration_proposals import merge_fitted_proposal
        return merge_fitted_proposal(parent,registry,package,applicability=applicability,
            allowed_fields=allowed_fields,proposal_path=proposal_path)
    payload = {k: v for k, v in package.items() if k != 'content_hash'}
    cid = package['component_id']
    component = parent.components.get(cid)
    if (package.get('schema_version') not in ('module-parameters-1.0','module-parameters-1.1') or
            canonical_hash(payload) != package.get('content_hash') or
            package['status'] != 'unfitted_snapshot' or component is None or
            package['parent_assembly_hash'] != parent.assembly_hash or
            package.get('base_implementation_hash', package['implementation_hash']) != component.implementation_hash or
            package['profile_id'] != component.parameters.profile.profile_id or
            package['applicability'] != applicability):
        raise ContractError('Parameter package identity, version, state or applicability mismatch')
    if package.get('schema_version')=='module-parameters-1.1':
        if canonical_hash(package.get('system_compatibility'))!=canonical_hash(system_compatibility(parent)):
            raise ContractError('Full-system interface/parameter/background-binding/feedback contract mismatch')
    expected = package['records']
    runtime = getattr(registry, 'runtime_profile', None)
    if runtime is not None and package['base_profile_identity'] != dict(
            profile_id=runtime.data['profile_id'], sha256=runtime.sha256,
            scope_sha256=runtime.modeling_scope.sha256 if runtime.modeling_scope else None):
        raise ContractError('Parameter package base profile identity mismatch')
    definitions = runtime.data['profiles'].get(package['profile_id'], {}) if runtime else {}
    original, _, record_format = _parameter_records(component.parameters, definitions)
    if package.get('parameter_record_format', record_format) != record_format or set(expected) != set(original):
        raise ContractError('Parameter package must contain the complete coefficient set')
    if (canonical_hash(expected) != canonical_hash(original) or
            (record_format == 'native_ParameterSet' and canonical_hash(package['native_profile_metadata']) !=
             canonical_hash(_plain(component.parameters.profile)))):
        raise ContractError('Changed coefficients require a source-bound calibrated profile/scope revision')
    if record_format == 'native_ParameterSet' and package['native_parameter_set_identity'] != dict(
            parameter_hash=component.parameters.parameter_hash,
            source_manifest_hash=component.parameters.source_manifest_hash):
        raise ContractError('Native parameter/source manifest identity mismatch')
    result = RetinaBuilder(registry).build(parent.spec, parent.run_mode)
    if hasattr(parent,'candidate_dynamics'):result.candidate_dynamics=parent.candidate_dynamics
    result.parameter_package_merge = dict(content_hash=package['content_hash'],
        status='unfitted_snapshot_identity_merge', initialization_transferred=False,
        background_validated=False, invalidation=package['invalidation'],
        integration_contract_status=('full_system_bindings_matched_checks_not_performed'
            if package['schema_version']=='module-parameters-1.1' else 'legacy_identity_snapshot_no_system_acceptance'),
        integration_policy=copy.deepcopy(SYSTEM_INTEGRATION_POLICY), system_acceptance=False)
    return result
