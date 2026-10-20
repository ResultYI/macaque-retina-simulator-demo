"""Opt-in initial-state seeds using existing local background equilibria.

This module does not change reference calibration, component constructors,
anchors, profiles or runtime identity.  A seed is not an adapted checkpoint:
histories, queues, event clocks and random generators are copied unchanged and
must undergo the original background preparation.  No adaptation cache is used.
"""
from copy import copy, deepcopy
from dataclasses import dataclass, replace
import hashlib
import numpy as np

from .background import (
    BackgroundPolicy, NUMERICAL_UNIT_SCALES, prepare_background,
    timing_requirements,
)
from .contracts import ContractError, Quantity, canonical_hash
from .engine import validate_state
from .hybrid import RetinaState
from .network import NetworkStimulus
from .rod_pathway import ChenPrimateRod, phototransduction_equilibrium
from .vh2005 import NativeConeH1


RETAINED_MINIMUM_SECONDS = 1.25  # Authorized engineering floor, not physiology.


def _fingerprint(state):
    """Hash arrays without materializing a large JSON list of state values."""
    d = state.dynamic
    return canonical_hash({
        'dynamic': {key: getattr(d, key) for key in (
            'time_seconds', 'state_spec_hash', 'graph_hash', 'parameter_hash',
            'model_id', 'model_version', 'model_hash', 'input_interpolation',
            'solver_config', 'evidence_tag', 'optional_rng_state')},
        'arrays': [hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
                   for a in d.ordered_state_arrays],
        'assembly': state.assembly_hash, 'grid_origin': state.grid_origin_seconds,
        'sample_index': state.sample_index, 'spike_states': state.spike_states,
        'delay_states': state.delay_states, 'input_position': state.input_position,
        'link_delay_states': state.link_delay_states,
        'component_tick_indices': state.component_tick_indices,
    })


def _fork(state, arrays=None):
    # SimulationState validates/copies all numeric arrays; auxiliary payloads
    # also receive deep copies.  The runner, native models and graph are shared.
    dynamic = replace(state.dynamic, ordered_state_arrays=(
        state.ordered_state_arrays if arrays is None else tuple(arrays)))
    return RetinaState(dynamic, state.assembly_hash, state.grid_origin_seconds,
        state.sample_index, deepcopy(state.spike_states), deepcopy(state.delay_states),
        deepcopy(state.input_position), deepcopy(state.link_delay_states),
        deepcopy(state.component_tick_indices))


def _check_initial_state(runner, state):
    if not isinstance(state, RetinaState) or state.assembly_hash != runner.compiled.assembly_hash:
        raise ContractError('Background seed requires a matching RetinaState')
    if runner.discrete is not None:
        raise ContractError('Background seed requires a compiled JointNetwork')
    validate_state(state.dynamic, runner.model)
    if state.time_seconds != state.grid_origin_seconds or state.sample_index != 0:
        raise ContractError('Background seed is only for an unadvanced initial state')
    expected = {cid: (1 if c.time_contract.kind == 'author_discrete' else 0)
        for cid, c in runner.compiled.components.items() if cid in runner.periods}
    if state.component_tick_indices != expected:
        raise ContractError('Background seed must preserve the original initial tick phase')
    for a in state.ordered_state_arrays:
        if a.dtype != np.float64 or not np.all(np.isfinite(a)):
            raise ContractError('Background seed requires finite FP64 registered state')


def _constant_inputs(runner, state, channels):
    if not channels or any(ch.input_contract.get('declared_constant_background') is not True
                           for ch in channels.values()):
        raise ContractError('Background seed requires explicitly constant runtime channels')
    stop = min(ch.stop_seconds for ch in channels.values())
    if stop <= state.time_seconds:
        raise ContractError('Background seed needs input coverage after the initial time')
    # Reuse original port, evidence, interpolation and coverage validation.
    stimulus = NetworkStimulus(runner.compiled, channels, state, stop,
        state.input_interpolation, runner.model.evidence_tag)
    values = {}
    for name, ch in channels.items():
        first = ch.at(state.time_seconds)
        if ch.breakpoints_seconds:
            raise ContractError('Constant background seed cannot have input breakpoints')
        for t in ((state.time_seconds + stop) * .5,
                  np.nextafter(stop, state.time_seconds)):
            other = ch.at(t)
            if (first.unit, first.meaning, first.evidence_tag) != (
                    other.unit, other.meaning, other.evidence_tag) or not np.array_equal(
                    first.values, other.values):
                raise ContractError('Declared constant background changed within its interval')
        values[name] = first
    digest = canonical_hash({'contracts': stimulus.input_contract, 'values': values})
    return values, digest


@dataclass(frozen=True)
class BackgroundSeed:
    state: RetinaState
    reference_state: RetinaState
    audit: dict
    reference_fingerprint: str
    seeded_fingerprint: str
    input_fingerprint: str
    numerical_unit_scales: dict


def seed_background_state(runner, state, channels, *, component_ids=None):
    """Return isolated candidate/reference states; never modify ``state``.

    ``channels`` must be the actual constant runtime background (for ISET this
    means already captured per-cone rates converted by the existing contract).
    This function neither captures light nor substitutes a nominal 100td value.
    Only the existing native cone/HC and Chen rod equilibria are supported.
    Other state blocks, including causal-history arrays, remain unchanged.
    """
    _check_initial_state(runner, state)
    inputs, input_hash = _constant_inputs(runner, state, channels)
    supported = {cid for cid, c in runner.compiled.components.items()
        if isinstance(c.native_model, (NativeConeH1, ChenPrimateRod))}
    selected = supported if component_ids is None else set(component_ids)
    if not selected or selected - supported:
        raise ContractError('Select only existing native cone/HC or Chen rod equilibria')
    specs = runner.model.state_specs
    index = {s.state_name: i for i, s in enumerate(specs)}
    if len(index) != len(specs):
        raise ContractError('Registered state names must be unique for seed injection')
    before = _fingerprint(state)
    arrays = list(state.ordered_state_arrays)
    rows = []
    for cid, component in runner.compiled.components.items():
        if cid not in selected:
            continue
        externals = [e for e in runner.compiled.spec.external_inputs
            if e.target_component == cid and e.target_port == 'light']
        if len(externals) != 1:
            raise ContractError('Seeded component needs exactly one external light channel: ' + cid)
        light = inputs[externals[0].channel]
        component.inputs['light'].validate(light)
        if np.any(light.values < 0):
            raise ContractError('Background seed requires nonnegative total light: ' + cid)
        # FiniteContactConeH1.equilibrium writes an audit attribute.  Its shallow
        # solver copy shares immutable coefficients/CSR but isolates that write.
        native = copy(component.native_model)
        if isinstance(native, NativeConeH1):
            initial = native.equilibrium(light)
            method = 'existing_native_equilibrium_actual_vector'
        else:
            rate = light.require('R*/s', 'photoisomerization_rate',
                                 component.state_specs[0].shape, check_finite=True)
            initial = phototransduction_equilibrium(native, rate)
            method = 'existing_Chen_primateRod_constant_rate_fixed_point'
        if len(initial) != len(component.state_specs):
            raise ContractError('Equilibrium returned an incomplete registered state: ' + cid)
        changed = []
        for q, local_spec in zip(initial, component.state_specs):
            name = cid + '.' + local_spec.state_name
            if name not in index:
                raise ContractError('Equilibrium state missing from joint registry: ' + name)
            i = index[name]
            joint = specs[i]
            if (joint.unit, joint.meaning, joint.shape, joint.cell_ids) != (
                    local_spec.unit, local_spec.meaning, local_spec.shape, local_spec.cell_ids):
                raise ContractError('Equilibrium component/joint state schema mismatch: ' + name)
            if not isinstance(q, Quantity) or q.evidence_tag != component.evidence_tag:
                raise ContractError('Equilibrium state evidence mismatch: ' + name)
            q.require(joint.unit, joint.meaning, joint.shape, check_finite=True)
            arrays[i] = q.values
            if not np.array_equal(q.values, state.ordered_state_arrays[i]):
                changed.append(name)
        rows.append({'component': cid, 'method': method, 'changed_states': changed,
            'input_channel': externals[0].channel, 'input_unit': light.unit,
            'input_values_sha256': hashlib.sha256(light.values.view(np.uint8)).hexdigest(),
            'input_range': [float(light.values.min()), float(light.values.max())],
            'local_solver_audit': deepcopy(getattr(native, 'background_equilibrium_audit', None))})
    reference = _fork(state)
    candidate = _fork(state, arrays)
    validate_state(candidate.dynamic, runner.model)
    if _fingerprint(state) != before:
        raise ContractError('Background seed unexpectedly mutated its original state')
    audit = {'schema_version': 'background_seed_v1', 'status': 'UNADAPTED_SEED',
        'assembly_hash': state.assembly_hash, 'model_hash': state.model_hash,
        'state_spec_hash': state.state_spec_hash, 'components': rows,
        'unchanged_components': [cid for cid in runner.compiled.components if cid not in selected],
        'reference_calibration_and_output_anchors_changed': False,
        'runtime_identity_changed': False, 'histories_delays_events_rng_reset': False,
        'actual_input_requirement': 'same explicitly constant runtime channels; no nominal-vector fallback',
        'input_cell_order_contract': 'compiled external port cell_ids, in the same order consumed by runtime',
        'constant_input_witness': 'declared constant contract plus start/middle/end-left equality',
        'monitor_reference': 'independent unchanged original initial state',
        'retained_minimum_seconds': RETAINED_MINIMUM_SECONDS,
        'whole_network_steady_state_claim': False, 'seed_alone_wash_in_completed': False,
        'physiological_validation': 'NOT_RUN', 'adaptation_cache_enabled': False}
    return BackgroundSeed(candidate, reference, audit, _fingerprint(reference),
        _fingerprint(candidate), input_hash, deepcopy(NUMERICAL_UNIT_SCALES))


def prepare_seeded_background(runner, seed, channels, policy=None, *, on_chunk=None, on_window=None):
    """Use original RHS/ticks/monitor with unchanged reference denominators.

    The original acceptance criteria and at least the authorized 1.25s floor
    remain mandatory. Short budgeted QA runs therefore return NOT_REACHED.
    Return a new prepared state/report; the seed and its reference stay intact.
    """
    if not isinstance(seed, BackgroundSeed):
        raise ContractError('Explicit BackgroundSeed required')
    _check_initial_state(runner, seed.state)
    validate_state(seed.reference_state.dynamic, runner.model)
    if (_fingerprint(seed.reference_state) != seed.reference_fingerprint or
            _fingerprint(seed.state) != seed.seeded_fingerprint):
        raise ContractError('Seed or frozen monitor reference changed before preparation')
    _, input_hash = _constant_inputs(runner, seed.state, channels)
    if input_hash != seed.input_fingerprint:
        raise ContractError('Seed and preparation background inputs differ')
    if canonical_hash(seed.numerical_unit_scales) != canonical_hash(NUMERICAL_UNIT_SCALES):
        raise ContractError('Original numerical unit scales must be retained')
    policy = policy or BackgroundPolicy()
    original = BackgroundPolicy()
    for key in ('window_seconds', 'stable_windows', 'known_tau_multiplier',
                'delay_multiplier', 'monitor_step_seconds', 'relative_tolerance',
                'stochastic_burnin_seconds'):
        if getattr(policy, key) != getattr(original, key):
            raise ContractError('Seed prototype must retain original background criterion: ' + key)
    if (policy.minimum_seconds < original.minimum_seconds or
            timing_requirements(runner, policy)['minimum_required_seconds'] < RETAINED_MINIMUM_SECONDS):
        raise ContractError('Seed prototype retains the original 1.25-second minimum preparation')
    result, report = prepare_background(runner, _fork(seed.state), channels, policy,
        unit_scales=deepcopy(seed.numerical_unit_scales),
        reference_state=seed.reference_state, on_chunk=on_chunk, on_window=on_window)
    if _fingerprint(seed.reference_state) != seed.reference_fingerprint:
        raise ContractError('Frozen monitor reference changed during preparation')
    report['background_seed'] = deepcopy(seed.audit)
    report['background_seed']['preparation_status'] = report['status']
    return result, report
