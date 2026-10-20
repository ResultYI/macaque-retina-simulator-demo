"""Approved candidate equations; effective drives are not glutamate concentrations.

BC reuses the existing receptor/membrane/release factory. AC has only a receptor
gate and a passive membrane, with an instantaneous effective voltage output.
Both isolated modules and circuits instantiate these same factories.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
from scipy.special import expit
from .assembly import RuntimeComponent, PortSpec, TimeContract
from .contracts import Quantity, ContractError, canonical_hash
from .mechanism_contracts import MechanismStateSpec

ROUTE = 'bc_ac_receptor_candidate_v1'
BC_KEY = 'CandidateConeBipolarReceptorMembrane'
AC_KEY = 'CandidateGenericAmacrineReceptorMembrane'
BC_TYPES = {'midget_BC_ON': 'ON', 'midget_BC_OFF': 'OFF',
            'diffuse_BC_ON': 'ON', 'diffuse_BC_OFF': 'OFF', 'S_ON_BC': 'ON'}
AC_TYPES = ('local_feedforward_AC', 'local_crossover_AC', 'widefield_AC')
POOLING = 'frozen_connection_pool_then_single_effective_receptor_gate'


def _contract(spec):
    contract = spec.options.get('candidate_dynamics', {})
    if (contract.get('route') != ROUTE or contract.get('pooling_order') != POOLING or
            contract.get('glutamate_concentration_claim') is not False or
            not contract.get('profile_sha256')):
        raise ContractError('Candidate receptor equations require explicit route, source profile and effective-drive contract')
    return contract


def cone_bc_component(spec, populations, sources, mode):
    contract = _contract(spec)
    polarity = BC_TYPES.get(spec.component_id)
    if polarity is None or spec.required_layer not in (None, spec.component_id):
        raise ContractError('Candidate cone BC requires one of the five approved target identities')
    from .rod_pathway import membrane_component
    # The old factory already implements e_inf, its receptor gate, membrane
    # currents and terminal release. Its historical target whitelist is bypassed
    # only inside this separately checked cone-BC wrapper, never modified globally.
    base = membrane_component(replace(spec, mechanism='ReceptorConductanceMembrane',
        required_layer=None, options=dict(spec.options,
            cell_kind='cone_BC_' + polarity)), populations, sources, mode)
    e0=base.parameters.quantity('e0','dimensionless','initial_receptor_gate').values
    if np.any((e0<0)|(e0>1)):
        raise ContractError('Candidate receptor initial gate must be within [0,1]')
    model = SimpleNamespace(**vars(base.native_model))
    model.approximations = [dict(id=ROUTE,
        why='Single effective BC receptor gate after frozen pooling; no concentration calibration or fitted cone-BC subtype claim',
        input_contract=contract, release='Retained first-order effective voltage-dependent release; no new Ca/resource mechanism')]
    def derivative(t,blocks,incoming):
        n=len(base.inputs['drive'].cell_ids)
        incoming['drive'].require('dimensionless','effective_synaptic_drive',(n,))
        if polarity=='OFF':incoming['inhibition'].require('S','inhibitory_conductance',(n,))
        return base.derivative(t,blocks,incoming)
    result = replace(base, mechanism=BC_KEY, native_model=model,
        derivative=derivative,
        layers=base.layers + (spec.component_id,), implementation_hash=canonical_hash({
            'base': base.implementation_hash,
            'candidate': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))
    result.reused_membrane_implementation_hash = base.implementation_hash
    return result


def generic_ac_component(spec, populations, sources, mode):
    contract = _contract(spec)
    if spec.component_id not in AC_TYPES or spec.required_layer not in (None, spec.component_id):
        raise ContractError('Candidate AC is one of the three generic effective functional units')
    from .inner import _profile
    p, pop, tag = _profile(replace(spec, mechanism='ResidualAC'), populations, sources, mode)
    n = len(pop.cells.cell_id); ids = pop.cells.cell_id
    def a(name, unit, meaning, positive=False):
        return p.quantity(name, unit, meaning, (n,), positive=positive).values
    C = a('C', 'F', 'membrane_capacitance', True)
    gL = a('gL', 'S', 'leak_conductance', True)
    EL = a('EL', 'V', 'leak_reversal'); V0 = a('V0', 'V', 'initial_membrane_voltage')
    K = a('K', 'dimensionless', 'receptor_half_drive', True)
    hill = a('hill', 'dimensionless', 'receptor_hill_exponent', True)
    tau = a('tau_e', 's', 'excitatory_receptor_time_constant', True)
    gE = a('gE', 'S', 'maximum_excitatory_conductance', True)
    EE = a('EE', 'V', 'excitatory_reversal')
    e0 = a('e0', 'dimensionless', 'initial_receptor_gate')
    half = a('activation_half', 'V', 'effective_output_half_voltage')
    slope = a('activation_slope', 'V', 'effective_output_voltage_slope', True)
    if np.any((e0 < 0) | (e0 > 1)):
        raise ContractError('Candidate receptor initial gate must be within [0,1]')
    states = (MechanismStateSpec('V', ids, (n,), 'V', 'physical_membrane_voltage',
                    'Explicit candidate initial voltage; not an adapted-state claim'),
              MechanismStateSpec('e', ids, (n,), 'dimensionless', 'excitatory_synaptic_gate',
                    'Explicit candidate receptor initial gate'))
    def port(name, unit, meaning):
        return PortSpec(name, unit, meaning, ids, tag)
    inputs = {'drive': port('drive', 'dimensionless', 'effective_synaptic_activation')}
    outputs = {'activation': port('activation', 'dimensionless', 'effective_synaptic_activation'),
               'voltage': port('voltage', 'V', 'physical_membrane_voltage'),
               'receptor_gate': port('receptor_gate', 'dimensionless', 'excitatory_synaptic_gate')}
    def initial():
        return (Quantity(V0, 'V', states[0].meaning, tag), Quantity(e0, 'dimensionless', states[1].meaning, tag))
    def derivative(t, blocks, incoming):
        V, e = blocks
        u = incoming['drive'].require('dimensionless', 'effective_synaptic_activation', (n,))
        if np.any(u < 0):
            raise ContractError('Candidate AC requires nonnegative effective drive; no rectification')
        occupied = expit(hill * np.log(np.maximum(u, np.finfo(float).tiny) / K))
        occupied = np.where(u == 0, 0., occupied)
        dV = (gL * (EL - V) + gE * e * (EE - V)) / C
        return (Quantity(dV, 'V/s', states[0].meaning + '_derivative', tag),
                Quantity((occupied-e)/tau, 'dimensionless/s', states[1].meaning + '_derivative', tag))
    def observe(port_name, t, blocks, incoming):
        V, e = blocks
        if port_name == 'activation':
            return Quantity(expit((V-half)/slope), 'dimensionless', 'effective_synaptic_activation', tag)
        return Quantity(V if port_name == 'voltage' else e, outputs[port_name].unit, outputs[port_name].meaning, tag)
    model = SimpleNamespace(model_hash=canonical_hash({'profile': p.parameter_hash, 'route': ROUTE}),
        approximations=[dict(id=ROUTE, why='Generic AC two-state effective model; no identified subtype, active channels or release resources',
                             input_contract=contract)])
    return RuntimeComponent(spec.component_id, AC_KEY, p, model, inputs, outputs, states,
        initial, derivative, observe, {name: () for name in outputs}, TimeContract('continuous'),
        ('AC', spec.component_id), (pop,), tag, p.source_ids,
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
