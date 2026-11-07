"""Fixed-background functional AII reduction on the current AC/BC graph.

Effective gains preserve reference conductances, not receptor frequency response.
Historical receptor equations remain available only through historical builders.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import expit

from .ac_vs_model import ACVSInner, DirectACVSRHS, MECHANISM as AC_VS, LEGACY_S, equilibrium
from .contracts import ContractError, canonical_hash
from .cone_hc_eccentric import EccentricHCFullChainRunner
from .group_inner_dynamics import MEMBRANE, BC_MECH, AC_MECH, RGC_MECH
from .spatial_group_graph import digest
from .spatial_group_model import BC, RGC, RecordingField, StateBlock


SCHEMA = 'macaque-functional-ac-aii'
DEFAULT_CONFIGURATION = 'configurations/research/AC_functional.json'
COMPONENTS = ('RBC_AII_AMPA', 'AII_FMB_glycine', 'AII_DB1_glycine')
V_KEY = ('AII_AC', MEMBRANE, 'V')
S_KEY = ('AII_AC', MEMBRANE, 's')


def reference_mapping(model, state):
    """Map analytic steady contact conductances using one verified fixed point."""
    model.validate(state)
    rhs = model.rhs(0., state)
    residual = max(float(np.max(abs(v), initial=0.)) for v in rhs.values())
    if residual > 1e-8:
        raise ContractError('AII gain mapping requires a fixed reference background')
    outputs = model.outputs(0., state)
    gate = expit((state[V_KEY]-model.par('AII_AC', 'release_half'))/
                 model.par('AII_AC', 'release_slope', positive=True))
    matrices, edges, ledger = {}, {}, {}
    for component in COMPONENTS:
        c = model.contacts[component]
        source = state['rod_BC', MEMBRANE, 'release'] if component == COMPONENTS[0] else gate
        denominator = source[c['pre']]
        rate = outputs[c['route'].pre, 'event_rate'][c['pre']]
        rise, decay, norm = model._kinetics(state, component)
        reference = c['coefficient']*rate*(decay-rise)/norm
        if (not np.isfinite(reference).all() or np.any(reference < 0)
                or np.any(denominator <= 0) or not np.isfinite(denominator).all()):
            raise ContractError('Blocked effective conductance mapping: '+component)
        np.testing.assert_allclose(outputs[component, 'contact_conductance'], reference, rtol=1e-10, atol=1e-24)
        gain = reference/denominator
        if not np.isfinite(gain).all():
            raise ContractError('Nonfinite effective gain: '+component)
        gain.flags.writeable = False
        edges[component] = gain
        matrices[component] = sparse.csr_matrix((gain, (c['post'], c['pre'])),
            shape=(model.counts[c['route'].post], model.counts[c['route'].pre]))
        ledger[component] = dict(pre=c['route'].pre, post=c['route'].post, unit='S',
            reference_conductance_S=reference, reference_source_gate=denominator,
            gain_S=gain, pre_rows=c['pre'], post_rows=c['post'],
            interpretation='physical grouped contact total; no additional member multiplier')
    return matrices, edges, dict(reference_state_sha256=canonical_hash(state),
        reference_parameter_identity=model.parameters['identity'], maximum_reference_rhs=residual,
        method='analytic steady receptor conductance / dimensionless source gate',
        fixed_across_stimuli=True, connections=ledger)


class FunctionalAIIAssembly:
    def __init__(self, parent, matrices, edges, mapping):
        self.parent = parent
        self.aii_matrices, self.aii_edges, self.aii_mapping = matrices, edges, mapping
        blocks = [b for b in parent.state_blocks if b.mechanism not in COMPONENTS
                  and not (b.population == 'AII_AC' and b.name in ('release', 'x'))]
        blocks.append(StateBlock(*S_KEY, 'dimensionless', 'effective inhibitory output gate',
                                 parent.populations['AII_AC'].group_count, 0))
        self.state_blocks, offset = [], 0
        for block in blocks:
            self.state_blocks.append(replace(block, offset=offset)); offset += block.count
        self.state_blocks = tuple(self.state_blocks)
        self.state_scalar_count = offset
        self.manifest = deepcopy(parent.manifest)
        self.manifest['functional_aii'] = dict(schema=SCHEMA, mapping=canonical_hash(mapping),
            state_contract='AII V and s; no AII resource or receptor states')
        self.artifact_sha256 = canonical_hash(dict(parent=parent.artifact_sha256,
            functional_aii=self.manifest['functional_aii'], states=[vars(b) for b in self.state_blocks],
            implementation=digest(Path(__file__))))
        self.runtime_factories = dict(parent.runtime_factories, inner=FunctionalAIIInner, compiled=DirectACVSRHS)
        self.populations = dict(parent.populations)
        fields = [f for f in parent.recording_fields if f.population in self.populations
                  and f.population not in ('local_crossover_AC', 'widefield_AC', 'AII_AC')]
        # Typed HC selection retained H1 rows but inherited the unsliced fine IDs.
        for i, field in enumerate(fields):
            if field.population == 'HC' and field.name in ('w', 'z2') and field.original_fine_rows is not None:
                mapped = self.populations['HC'].fine_to_group[field.original_fine_rows]
                selected = np.isin(mapped, field.rows)
                fields[i] = replace(field, rows=mapped[selected].copy(),
                                    original_fine_rows=field.original_fine_rows[selected].copy())
        existing = {(f.population, f.name) for f in fields}
        def add(pop, name, unit, meaning, identity='representative_group'):
            if (pop, name) in existing:
                return
            layout = self.populations[pop]
            fields.append(RecordingField(pop, pop, name, unit, meaning, np.arange(layout.group_count),
                True, layout.coordinate_role, identity=identity))
            existing.add((pop, name))
        for channel in parent.ac_config['channels']:
            pop = channel['pool']
            layout = self.populations[pop]
            self.populations[pop] = SimpleNamespace(**vars(layout),
                source_table=dict(kind='effective_pool', target_population=channel['target']),
                source_keys=dict(semantics='one functional pool per target representative; not anatomical AC cells'))
            for name, unit in (('voltage', 'V'), ('s', 'dimensionless'), ('gE', 'S')):
                add(pop, name, unit, 'ordinary AC '+name, 'effective_functional_pool')
            for prefix, unit in (('gI_', 'S'), ('I_', 'A')):
                add(channel['target'], prefix+channel['relation'], unit, 'ordinary AC contribution to target RGC')
        for name, unit, meaning in (('voltage', 'V', 'effective AII membrane voltage'),
                ('s', 'dimensionless', 'effective AII output gate'),
                ('gE_RBC', 'S', 'RBC effective excitation; physical group total'),
                ('gap_current', 'A', 'net local and held-exterior reciprocal gap current')):
            add('AII_AC', name, unit, meaning)
        for pop in ('midget_BC_OFF', 'diffuse_BC_OFF'):
            add(pop, 'gI_AII', 'S', 'AII effective inhibitory conductance', 'independently_solved_cell')
        for pop in parent.manifest['gap']['population_order']:
            if pop != 'AII_AC':
                add(pop, 'gap_current', 'A', 'reciprocal AII gap current', 'independently_solved_cell')
        self.recording_fields = tuple(fields)
        self.recording_field_count = len(fields)

    def __getattr__(self, name):
        return getattr(self.parent, name)


class FunctionalAIIInner(ACVSInner):
    membrane_populations = ('rods', 'rod_BC')
    dynamic_contact_components = ('individual_rod_RBC_contacts',)
    port_unit_overrides = dict(ACVSInner.port_unit_overrides, gE_RBC='S', gI_AII='S')

    def __init__(self, assembly, parameters):
        super().__init__(assembly, parameters)
        self.K = assembly.aii_matrices
        self.edge_gain = assembly.aii_edges
        self.par('AII_AC', 'tau_release', positive=True)
        self.par('AII_AC', 'release_slope', positive=True)
        self.identity = canonical_hash(dict(parent=self.identity, schema=SCHEMA,
            mapping=canonical_hash(assembly.aii_mapping), implementation=digest(Path(__file__))))

    def contact_conductance(self, state, component):
        if component not in COMPONENTS:
            return super().contact_conductance(state, component)
        source = state['rod_BC', MEMBRANE, 'release'] if component == COMPONENTS[0] else state[S_KEY]
        if np.any(source < 0) or np.any(source > 1+1e-10):
            raise ContractError('Effective AII input must be a dimensionless fraction')
        return self.edge_gain[component]*source[self.contacts[component]['pre']]

    def outputs(self, t, state, cone_release=None):
        s = state[S_KEY]
        if np.any(s < -1e-10) or np.any(s > 1+1e-10):
            raise ContractError('Functional AII s outside [0,1]; no clipping')
        out = super().outputs(t, state, cone_release)
        out['AII_AC', 'voltage'] = state[V_KEY]
        out['AII_AC', 's'] = out['AII_AC', 'activation'] = state[S_KEY]
        out['AII_AC', 'gE_RBC'] = out[COMPONENTS[0], 'conductance']
        for component in COMPONENTS[1:]:
            out[self.contacts[component]['route'].post, 'gI_AII'] = out[component, 'conductance']
        return out

    def rhs(self, t, state, cone_release=None, rod_current=None):
        result = super().rhs(t, state, cone_release, rod_current)
        V, s = state[V_KEY], state[S_KEY]
        voltages = np.concatenate([V if pop == 'AII_AC' else state[pop, BC_MECH, 'V'] for pop in self.gap_names])
        gap = -(self.gap @ voltages)[self.gap_slices['AII_AC']]
        gE = self.K[COMPONENTS[0]] @ state['rod_BC', MEMBRANE, 'release']
        result[V_KEY] = (self.gL['AII_AC']*(self.par('AII_AC', 'EL')-V)
            +gE*(self.par('AII_AC', 'EE')-V)+gap)/self.C['AII_AC']
        result[S_KEY] = (expit((V-self.par('AII_AC', 'release_half'))/
            self.par('AII_AC', 'release_slope', positive=True))-s)/self.par('AII_AC', 'tau_release', positive=True)
        return result


def initialize_functional_background(assembly, original):
    """Solve the coupled gap system with frozen BC baselines and exterior current."""
    seed = deepcopy(original)
    p = seed['inner_parameters']
    p['populations']['AII_AC'] = {k:v for k,v in p['populations']['AII_AC'].items()
        if k in ('C', 'gL', 'EL', 'EE', 'release_half', 'release_slope', 'tau_release')}
    p['provenance']['functional_aii'] = dict(mapping_sha256=canonical_hash(assembly.aii_mapping),
        interpretation='effective reference mapping and inherited engineering membrane/gate parameters; uncalibrated')
    p['identity'] = canonical_hash(dict(parent=p['identity'], functional_aii=p['provenance']['functional_aii']))
    model = FunctionalAIIInner(assembly, p)
    state = seed['states'] = {k:v for k,v in seed['states'].items()
        if k[1] not in COMPONENTS and not (k[0] == 'AII_AC' and k[2] in ('release', 'x'))}
    target = {pop:model.par(pop, 'Vref')+model.par(pop, 'native_voltage_gain', positive=True)*
        (state[pop, BC_MECH, 'activation']-model.par(pop, 'native_reference')) for pop in BC}
    ge = model.K[COMPONENTS[0]] @ state['rod_BC', MEMBRANE, 'release']
    diagonal, right = [], []
    for pop in model.gap_names:
        e = ge if pop == 'AII_AC' else np.zeros(model.counts[pop])
        v = model.par(pop, 'EL') if pop == 'AII_AC' else target[pop]
        diagonal.append(model.gL[pop]+e)
        right.append(model.gL[pop]*v+e*(model.par(pop, 'EE') if pop == 'AII_AC' else 0.))
    voltage = spsolve(model.gap.matrix+sparse.diags(np.concatenate(diagonal)),
                      np.concatenate(right)-model.gap_boundary_product)
    for pop, section in model.gap_slices.items():
        state[pop, MEMBRANE if pop == 'AII_AC' else BC_MECH, 'V'] = voltage[section].copy()
    state[S_KEY] = expit((state[V_KEY]-model.par('AII_AC', 'release_half'))/
                         model.par('AII_AC', 'release_slope', positive=True))
    for component in COMPONENTS[1:]:
        pop = model.contacts[component]['route'].post
        gi = model.K[component] @ state[S_KEY]
        state[pop, BC_MECH, 'V'] = (model.gL[pop]*target[pop]+gi*model.par(pop, 'EI'))/(model.gL[pop]+gi)
    for pop in BC:
        rel = expit((state[pop, BC_MECH, 'V']-model.par(pop, 'release_half'))/model.par(pop, 'release_slope', positive=True))
        state[pop, BC_MECH, 'terminal_release'] = rel
        state[pop, BC_MECH, 'A'] = model.par(pop, 'adaptation_scale')*rel if model.p[pop]['adaptation_enabled'] else np.zeros_like(rel)
    inner = lambda: {k:state[k] for k in model.state_keys}
    out = model.outputs(0., inner())
    state[LEGACY_S, AC_MECH, 'h'] = out[LEGACY_S, 'target_inhibition'].copy()
    for channel in model.channels:
        pop = channel['pool']; V, _, s = equilibrium(out[pop, 'gE'], model.p[pop])
        state[pop, AC_VS, 'V'], state[pop, AC_VS, 's'] = V, s
    out = model.outputs(0., inner())
    for pop in RGC:
        ge, gi, g = out[pop, 'gE'], out[pop, 'gI'], model.gL[pop]
        state[pop, RGC_MECH, 'V'] = (g*model.par(pop, 'EL')+ge*model.par(pop, 'EE')+gi*model.par(pop, 'EI'))/(g+ge+gi)
    # Reference continuity permits retaining the existing spike law and RNG.
    maximum_change = max(float(np.max(abs(v-original['states'][k]), initial=0.))
                         for k,v in state.items() if k in original['states'])
    if maximum_change > 1e-10:
        raise ContractError('Functional mapping failed fixed-background continuity')
    residual = max(float(np.max(abs(v), initial=0.)) for v in model.rhs(0., inner()).values())
    if residual > 1e-8:
        raise ContractError('Functional BC-AII background is not balanced')
    seed['h1_h2_model']['artifact'] = assembly.artifact_sha256
    seed['functional_aii'] = dict(schema=SCHEMA, artifact=assembly.artifact_sha256,
        mapping_sha256=canonical_hash(assembly.aii_mapping), maximum_background_rhs=residual,
        maximum_retained_state_change=maximum_change, background_solved=True,
        old_checkpoint_compatible=False, physiological_calibration=False)
    seed['joint_background_initialization'] = dict(seed['joint_background_initialization'],
        initialized_model=SCHEMA, parameters_sha256=canonical_hash(p),
        initial_state_sha256=canonical_hash(state), parent_seed_sha256=canonical_hash(original),
        parent_artifact=assembly.artifact_sha256, functional_aii=seed['functional_aii'])
    seed['joint_background_initialization']['initialized_model_id'] = canonical_hash(seed['joint_background_initialization'])
    return seed


class FunctionalACModel:
    def __init__(self, repo_root, configuration=DEFAULT_CONFIGURATION):
        from .assembly import RetinaBuilder
        root = Path(repo_root).resolve()
        path = Path(configuration)
        if not path.is_absolute(): path = root/path
        self.config = json.loads(path.read_bytes())
        if (self.config['schema'] != SCHEMA or self.config['research_only'] is not True
                or self.config['physiologically_calibrated'] is not False):
            raise ContractError('Explicit uncalibrated functional AC configuration required')
        self.upstream = RetinaBuilder.build_research_bc(root, self.config['bc_configuration'])
        parent = self.upstream
        old_hash = canonical_hash(parent.seed)
        old = parent.assembly.runtime_factories['inner'](parent.assembly, parent.seed['inner_parameters'])
        state = {k:parent.seed['states'][k] for k in old.state_keys}
        matrices, edges, self.mapping = reference_mapping(old, state)
        self.assembly = FunctionalAIIAssembly(parent.assembly, matrices, edges, self.mapping)
        self.seed = initialize_functional_background(self.assembly, parent.seed)
        self.protocol = deepcopy(parent.protocol)
        self.identity = canonical_hash(dict(artifact=self.assembly.artifact_sha256, seed=self.seed, config=self.config))
        if canonical_hash(parent.seed) != old_hash:
            raise ContractError('Functional AII construction mutated its BC parent')

    def runner(self, protocol=None):
        return FunctionalACRunner(self.assembly, self.seed, self.protocol if protocol is None else protocol)


class FunctionalACRunner(EccentricHCFullChainRunner):
    def __init__(self, assembly, seed, protocol, **kwargs):
        if seed.get('functional_aii', {}).get('artifact') != assembly.artifact_sha256:
            raise ContractError('Functional AII requires its reconstructed background')
        super().__init__(assembly, seed, protocol, **kwargs)

    def checkpoint_payload(self, **kwargs):
        payload = super().checkpoint_payload(**kwargs)
        payload['controller']['functional_aii'] = self.seed['functional_aii']
        return payload

    @classmethod
    def from_checkpoint(cls, assembly, seed, protocol, payload, **kwargs):
        if canonical_hash(payload.get('controller', {}).get('functional_aii')) != canonical_hash(seed.get('functional_aii')):
            raise ContractError('Old or mismatched AII checkpoint; reconstruct the functional background')
        return super().from_checkpoint(assembly, seed, protocol, payload, **kwargs)
