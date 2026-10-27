"""State-level, source-guarded time partition for the frozen v0.2 group graph.

This module compiles declarations only. It never initializes a neural state,
selects a numerical backend, consumes RNG, or advances an event clock. Spatial
leaf width does not occur in any time classification rule.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import hashlib

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .contracts import ContractError
from .spatial_group_graph import digest
from .spatial_group_model import BC, AC, RGC, CONTACT_STATES, REQUIRED_ROUTE_PAIRS

SCHEMA = 'retipath-group-time-partition-1'

# Audited equations, not hashes dynamically blessed from arbitrary current code.
# A change requires revisiting the dependency/affine rules and a new contract.
EQUATION_SOURCE_SHA256 = MappingProxyType({
    'retina_sim/cone_hc_reduced_native.py': 'eb1150227598307c5f066ea3ed496e5550de20f31eaf05a7666975017944ae3a',
    'retina_sim/vh2005.py': '60c1091971527216961df8a49b40e3dad8125f44de1d82d4d9374afcd4462858',
    'retina_sim/rod_pathway.py': '5601dd757e22177fb02ebd72a4305ea145f8ddd05aa130822fe43f95b27c862c',
    'retina_sim/inner.py': '32c469e6a1de2c75a28b1bf6c6d9fcac8c6942f0391386d60608b95688bac0e3',
    'retina_sim/ac_inhibition.py': '1e170f21f1617eb953116905d2fefdfdeaa9dd3b697e630eab08cf9d20301873',
    'retina_sim/bc_gain_control.py': 'db45db6c6f107b1655ab6c26bcd96643ae0afec3d40fdbd9434de30755888e01',
    'retina_sim/rgc_expectation.py': '70a9e35ff30f1faebc46312e183598f42c5919083edc2e610d64b97e732886d0',
    'retina_sim/chen2024.py': '3eda0b1855ec28ca8307e3c02ee5fd1a355f36d8f47ef0541da2fd33b719a031',
    'retina_sim/disk_circuit.py': '479a3325989aa9852e46aa2071e5b4dd29f2a585b2d61fcb38d1c697c342c691',
    'data/source_snapshots/chen2024_1bcf7c9a/BiophysModel.m': 'fb5ff0a6880d5c78421990981dcf31932b0d415d9d216deb73a64753e5f37937',
    'data/source_snapshots/chen2024_1bcf7c9a/initPhotoreceptorParams.m': 'dbff286bc0bda0fbc0aa87775f83285f5dc93d5028559d19efb64651b9ef4de0',
})


def guard_source_dependencies(root):
    root = Path(root).resolve()
    for relative, expected in EQUATION_SOURCE_SHA256.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise ContractError('Time partition equation dependency changed; re-audit required: '+relative)
    return dict(EQUATION_SOURCE_SHA256)


def _readonly(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def _state_id(block):
    return ':'.join((block.population, block.mechanism, block.name))


@dataclass(frozen=True)
class StateDependency:
    state_id: str
    population: str
    mechanism: str
    state_name: str
    unit: str
    count: int
    storage_role: str
    rhs_dependencies: tuple
    commit_events: tuple
    commit_dependencies: tuple


@dataclass(frozen=True)
class PortDependency:
    port_id: str
    dependencies: tuple
    expression: str
    nonlinear_order: str


@dataclass(frozen=True)
class RowPartition:
    state_id: str
    category: str
    rows: np.ndarray
    coupled_block: str | None
    reason: str


@dataclass(frozen=True)
class TimePartition:
    artifact_sha256: str
    nodes: tuple
    ports: tuple
    row_partitions: tuple
    gap_component_labels: object
    gap_component_sizes: np.ndarray
    source_sha256: object
    core_state_scalars: int
    probability_state_scalars: int
    selected_z1_state_scalars: int

    def describe(self):
        categories = {}
        for row in self.row_partitions:
            categories[row.category] = categories.get(row.category, 0)+len(row.rows)
        largest = int(self.gap_component_sizes.max(initial=0))
        return dict(schema=SCHEMA, artifact_sha256=self.artifact_sha256,
            structural_partition_complete=True, numerical_backend_enabled=False,
            model_forward_started=False, B_numerical_change_status='PENDING_APPROVAL',
            equation_source_sha256=dict(self.source_sha256),
            scalar_counts=dict(core=self.core_state_scalars, probability_history=self.probability_state_scalars,
                selected_original_z1=self.selected_z1_state_scalars, by_category=categories),
            nodes=[dict(state_id=n.state_id, population=n.population, mechanism=n.mechanism,
                state_name=n.state_name, unit=n.unit, count=n.count, storage_role=n.storage_role,
                rhs_dependencies=list(n.rhs_dependencies), commit_events=list(n.commit_events),
                commit_dependencies=list(n.commit_dependencies)) for n in self.nodes],
            ports=[dict(port_id=p.port_id, dependencies=list(p.dependencies), expression=p.expression,
                nonlinear_order=p.nonlinear_order) for p in self.ports],
            row_partitions=[dict(state_id=p.state_id, category=p.category, count=len(p.rows),
                row_sha256=hashlib.sha256(p.rows.tobytes()).hexdigest(), coupled_block=p.coupled_block,
                reason=p.reason) for p in self.row_partitions],
            aggregate_gap=dict(nodes=sum(len(a) for a in self.gap_component_labels.values()),
                components=len(self.gap_component_sizes), nontrivial_components=int(np.count_nonzero(self.gap_component_sizes>1)),
                largest_component_nodes=largest, inherits_original_42_node_bound=False,
                isolated_node_count=int(np.count_nonzero(self.gap_component_sizes == 1))),
            dependency_guards=dict(HC_feedback_only_release=True, BC_A_driven_by_unadapted_release=True,
                generic_AC_nonlinearity_after_BC_pooling=True, RGC_voltage_independent_of_spikes=True,
                glycine_decay_and_peak_normalization_use_current_postsynaptic_voltage=True,
                resource_event_rate_uses_V_and_x_not_filtered_release=True),
            required_future_acceptance=['B-class numerical contract approval', 'Parameter and initial/history binding',
                'Complete RHS monitor and representation error allocation', 'Same-graph paired numerical/causal/restore acceptance'],
            clocks=dict(rod='100 us author recurrence; no skipped ticks', BC='1 ms common-snapshot history commit',
                RGC='1 ms strict-past probability/history and original PCG64 commit',
                observation='existing 0.5 ms plus full monitor/event requirements',
                input='actual 150 Hz boundaries', HC_readout_delay='existing designated 2.82 ms readout only'),
            solver_step_seconds=None,
            fast_block_candidate='FP64 DOP853 max100us only after B approval; no step selected by this compiler')


def compile_time_partition(assembly, repo_root):
    """Classify every declared scalar without running or importing model builders."""
    source_hashes = guard_source_dependencies(repo_root)
    routes = tuple(assembly.routes.values())
    if {(r.pre, r.post) for r in routes} != set(REQUIRED_ROUTE_PAIRS) or len(routes) != len(REQUIRED_ROUTE_PAIRS):
        raise ContractError('Unknown group route dependency invalidates audited time partition')
    contact_routes = {r.contact_component: r for r in routes if r.contact_component}
    if set(contact_routes) != set(CONTACT_STATES):
        raise ContractError('All four actual contact mechanisms are required')
    contact_pairs = {('rods','rod_BC'):'individual_rod_RBC_contacts',
        ('rod_BC','AII_AC'):'RBC_AII_AMPA', ('AII_AC','midget_BC_OFF'):'AII_FMB_glycine',
        ('AII_AC','diffuse_BC_OFF'):'AII_DB1_glycine'}
    gap_pairs = {('AII_AC','AII_AC'), *(pair for n in ('midget_BC_ON','diffuse_BC_ON','S_ON_BC')
                  for pair in (('AII_AC',n),(n,'AII_AC')))}
    for route in routes:
        pair = (route.pre,route.post)
        expected_kind = ('gap' if pair in gap_pairs else 'native' if pair in (('cones','HC'),('HC','cones'))
                         else 'primary_chemical' if pair in contact_pairs else 'inner_chemical')
        if route.kind != expected_kind or route.contact_component != contact_pairs.get(pair):
            raise ContractError('Changed route mechanism/port role requires dependency re-audit: '+route.name)
    # Recompute the NEW aggregate graph. Fine-graph component labels are not used.
    gap_names = tuple(assembly.manifest['gap']['population_order'])
    gap_size = sum(assembly.populations[n].group_count for n in gap_names)
    lap = assembly.matrix('gap/laplacian_S.npz')
    if lap.shape != (gap_size, gap_size):
        raise ContractError('Aggregate gap component layout differs')
    adjacency = lap.copy()
    adjacency.setdiag(0.)
    adjacency.eliminate_zeros()
    if adjacency.data.size:
        adjacency.data[:] = 1.
    _, labels = connected_components(adjacency, directed=False)
    component_sizes = np.bincount(labels)
    gap_labels, gap_coupled = {}, {}
    offset = 0
    for name in gap_names:
        count = assembly.populations[name].group_count
        local = labels[offset:offset+count]
        gap_labels[name] = _readonly(local, np.int64)
        gap_coupled[name] = component_sizes[local] > 1
        offset += count
    gly_coupled = {}
    for component in ('AII_FMB_glycine', 'AII_DB1_glycine'):
        route = contact_routes[component]
        coefficients = assembly.matrix('routes/'+route.name+'/contact_coefficient_S.npz')
        gly_coupled[route.post] = np.asarray(coefficients.getnnz(axis=1)) > 0

    core = tuple(assembly.state_blocks)
    probability = tuple(assembly.probability_blocks)
    auxiliary = tuple(assembly.auxiliary_z1_blocks)
    all_blocks = core+probability+auxiliary
    ids = {_state_id(b): b for b in all_blocks}
    if len(ids) != len(all_blocks):
        raise ContractError('Repeated logical state identity')
    lookup = {(b.population, b.mechanism, b.name): _state_id(b) for b in all_blocks}
    def state(pop, mechanism, name):
        try:
            return lookup[pop, mechanism, name]
        except KeyError as exc:
            raise ContractError('Required state dependency missing: '+str(exc)) from exc
    def cone(name): return state('cones', 'analytic_RE' if name in ('R', 'E') else 'reduced_native', name)
    def hc(name): return state('HC', 'reduced_native', name)
    def bc(pop, name): return state(pop, 'HistoryBipolarTerminal', name)
    def membrane(pop, name): return state(pop, 'rod_membrane' if pop == 'rods' else 'ReceptorConductanceMembrane', name)
    def contact(component, name):
        return state(contact_routes[component].pre, component, name)
    def port(pop, name): return 'port:'+pop+':'+name

    ports = [PortDependency(port('cones','q'), (cone('Vis'), hc('Vh')),
                 'sigmoid((Vis-B@Vh-Vk)/Vn)', 'B@Vh before sigmoid; current feedback'),
             PortDependency(port('cones','release'), (port('cones','q'), cone('Vbar')),
                 'q/(Vbar/VI)^mu', 'No pooling/nonlinearity exchange'),
             PortDependency(port('HC','p'), (cone('Vbar'),),
                 '(A@Vbar/VI)^mu', 'A@Vbar before power; never frozen p'),
             PortDependency(port('rods','current'), (state('rods','Chen2024PrimateRodCurrent','G'),),
                 '-k*G^n pA; explicit original A conversion and background anchor', 'Author tick hold')]
    for pop in BC:
        ports.append(PortDependency(port(pop,'effective_release'), (bc(pop,'terminal_release'), bc(pop,'A')),
            'terminal_release/(1+A^6)', 'A is driven by raw release and affects downstream output only'))
    for pop in AC:
        ports.append(PortDependency(port(pop,'activation'), (state(pop,'DynamicAmacrineInhibition','h'),),
            'output_scale*h', 'drive target is sigmoid AFTER weighted BC pooling'))
    for pop in ('rod_BC','AII_AC'):
        ports.append(PortDependency(port(pop,'event_rate'), (membrane(pop,'V'), membrane(pop,'x')),
            'Npool*U*rmax*sigmoid((V-r_half)/r_slope)*x', 'Not the separately filtered release state'))
    for component, route in contact_routes.items():
        deps = (contact(component,'m'),) if component == 'individual_rod_RBC_contacts' else (contact(component,'a'),contact(component,'b'))
        if component.startswith('AII_'):
            deps += (bc(route.post,'V'),)
        ports.append(PortDependency(port(component,'conductance'), deps,
            'gain*m' if component == 'individual_rod_RBC_contacts' else 'coefficient*(b-a)/Z(current_postsynaptic_V)' if component.startswith('AII_') else 'coefficient*(b-a)/Z',
            'TRPM1 nonlinear target per contact before pooling; glycine tau and Z use current voltage'))

    incoming = {p: [] for p in assembly.populations}
    for route in routes:
        if route.kind in ('gap', 'native'):
            continue
        value = port(route.contact_component,'conductance') if route.contact_component else port('cones','release') if route.pre == 'cones' else port(route.pre,'activation') if route.pre in AC else port(route.pre,'effective_release')
        incoming[route.post].append(value)
    nodes, partitions = [], []

    def emit(block, dependencies, category, coupled=None, reason='', events=(), mask=None):
        sid = _state_id(block)
        dependencies = tuple(dict.fromkeys(dependencies))
        is_discrete = category in ('author_discrete', 'discrete_history')
        nodes.append(StateDependency(sid, block.population, block.mechanism, block.name, block.unit,
            block.count, block.role, () if is_discrete else dependencies, tuple(events),
            dependencies if is_discrete else ()))
        if mask is None:
            partitions.append(RowPartition(sid, category, _readonly(np.arange(block.count), np.int64), coupled, reason))
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (block.count,):
                raise ContractError('Temporal row split differs from state shape')
            for selected, cat, group, why in ((mask,'fast_joint',coupled,reason),
                    (~mask,'analytic_affine',None,'Fixed leak and held BC generator; no active gap/glycine feedback on these rows')):
                if np.any(selected):
                    partitions.append(RowPartition(sid, cat, _readonly(np.flatnonzero(selected),np.int64), group, why))

    for block in all_blocks:
        pop, mechanism, name = block.population, block.mechanism, block.name
        own = _state_id(block)
        if pop == 'cones' and mechanism == 'analytic_RE' and name in ('R','E'):
            emit(block, (own,'input:cones:light') if name=='R' else (own,cone('R')),
                 'analytic_zoh', 'cone_RE', 'Existing exact two-stage ZOH anchor; source-ms to seconds', ('input_boundary',))
        elif pop == 'cones' and mechanism == 'reduced_native':
            if name in ('X','C'):
                emit(block, (cone('X'),cone('C'),cone('E')) if name=='X' else (cone('X'),cone('C')), 'fast_joint','cone_X_C',
                     'Nonlinear X/C feedback retained jointly; no HC feedback enters either equation')
            elif name in ('Vis','g_i'):
                emit(block, (cone('Vis'),cone('g_i'),cone('X')) if name=='Vis' else (cone('Vis'),cone('g_i')),
                     'waveform_coupled','cone_Vis_gi','Full continuous g_i at fast stages; accepted-waveform self-consistency required')
            elif name == 'Vbar':
                emit(block,(own,cone('Vis')),'exponential_forced',None,'Feedforward fixed tau_a; no return into X/C/Vis/gi')
            else:
                raise ContractError('Unknown cone equation state: '+name)
        elif pop == 'HC' and mechanism == 'reduced_native' and name in ('w','z2','Vh'):
            deps = (own,port('cones','release')) if name=='w' else (own,hc('w'),port('HC','p')) if name=='z2' else (own,hc('z2'),port('HC','p'))
            emit(block,deps,'fast_joint','HC_release_feedback','Current B@h -> release -> A@release closes loop; p(t) remains dynamic')
        elif pop == 'rods' and mechanism == 'Chen2024PrimateRodCurrent' and name in ('R','P','G','C','S'):
            emit(block, tuple(state('rods',mechanism,n) for n in ('R','P','G','C','S'))+('input:rods:light',),
                 'author_discrete','rod_author_recurrence','Preserve original old/new state order, no skipped ticks',('rod_tick',))
        elif pop == 'rods' and mechanism == 'rod_membrane' and name in ('V','release'):
            emit(block,(own,port('rods','current')) if name=='V' else (own,membrane('rods','V')),
                 'analytic_affine' if name=='V' else 'exponential_forced', None,
                 'Held current and fixed leak with audited current sign/anchor' if name=='V' else 'Voltage sigmoid drives fixed release decay',('rod_tick',) if name=='V' else ())
        elif pop in ('rod_BC','AII_AC') and mechanism == 'ReceptorConductanceMembrane' and name in ('V','release','x'):
            deps=(own,*incoming[pop]) if name=='V' else (own,membrane(pop,'V'))
            if name=='V' and pop in gap_labels:
                deps += tuple(state(n,'HistoryBipolarTerminal','V') if n in BC else membrane(n,'V') for n in gap_names)
            emit(block,deps,'fast_joint','AII_ONBC_gap' if pop=='AII_AC' else 'RBC_membrane_release_resource',
                 'V, separate release and resource retained; resource depletion depends on current V and x')
        elif pop in BC and mechanism == 'HistoryBipolarTerminal':
            if name in ('history_0','history_1','history_2','activation'):
                commits = (port('cones','release'),) if name=='history_0' else (bc(pop,'history_'+str(int(name[-1])-1)),) if name.startswith('history_') else (bc(pop,'history_0'),bc(pop,'history_1'),port('cones','release'))
                emit(block,commits,'discrete_history','BC_common_snapshot',
                     'Zero continuous derivative; original FIR/activation common-snapshot commit',('BC_tick',))
            elif name=='V':
                mask=gap_coupled.get(pop,gly_coupled.get(pop,np.zeros(block.count,bool)))
                feedback=tuple(port(r.contact_component,'conductance') for r in routes
                               if r.post==pop and r.contact_component)
                deps=(own,bc(pop,'activation'),*feedback)
                if pop in gap_labels:
                    deps += tuple(state(n,'HistoryBipolarTerminal','V') if n in BC else membrane(n,'V') for n in gap_names)
                emit(block,deps,'fast_joint','AII_ONBC_gap' if pop in gap_labels else pop+'_glycine_feedback',
                     'Actual aggregate gap/glycine feedback rows stay coupled; isolated rows may use held affine membrane',mask=mask)
            elif name in ('terminal_release','A'):
                emit(block,(own,bc(pop,'V')) if name=='terminal_release' else (own,bc(pop,'terminal_release')),
                     'exponential_forced',None,'A driven by unadapted terminal release; downstream gain does not feed this RHS')
            else:
                raise ContractError('Unknown BC equation state: '+name)
        elif pop in AC and mechanism == 'DynamicAmacrineInhibition' and name=='h':
            emit(block,(own,*incoming[pop]),'exponential_forced',None,
                 '10ms fixed-decay current generic AC; nonlinear target after actual weighted pooling')
        elif pop in RGC and mechanism == 'RGC_spike_history' and name=='V':
            emit(block,(own,*incoming[pop]),'integrating_factor',None,
                 'Current time-varying gE/gI; no spike/history feedback into V')
        elif mechanism in CONTACT_STATES and name in CONTACT_STATES[mechanism]:
            route=contact_routes[mechanism]
            if mechanism=='individual_rod_RBC_contacts':
                deps=(own,membrane('rods','release'))
            else:
                deps=(own,port(route.pre,'event_rate'))
            if mechanism.startswith('AII_') and name=='b':
                deps+=(bc(route.post,'V'),)
                emit(block,deps,'fast_joint',route.post+'_glycine_feedback',
                     'Current postsynaptic V controls decay; normalization also reads V at output')
            else:
                emit(block,deps,'exponential_forced',None,'Fixed tau branch after true upstream waveform accepted')
        elif pop in RGC and mechanism=='RGC_expected_history' and name in tuple('history_probability_'+str(i) for i in range(8)):
            emit(block, tuple(state(pop,mechanism,'history_probability_'+str(i)) for i in range(8))+
                 (state(pop,'RGC_spike_history','V'),), 'discrete_history','RGC_strict_past_history',
                 'Unchanged probability law; real PCG64/pending-event state remains separate',('RGC_tick',))
        elif pop=='cones' and mechanism=='selected_original_z1_auxiliary' and name=='z1_aux':
            emit(block,(own,port('cones','release')),'exponential_forced',None,
                 'Only frozen original selected nodes; cannot reconstruct other z1 history from w')
        else:
            raise ContractError('Unknown state invalidates time partition: '+own)

    declared={n.state_id for n in nodes}; port_ids={p.port_id for p in ports}
    for node in nodes:
        unknown=[d for d in node.rhs_dependencies+node.commit_dependencies if d not in declared|port_ids and not d.startswith('input:')]
        if unknown:
            raise ContractError('Undeclared state/port dependencies: '+str(unknown))
        selected=[r.rows for r in partitions if r.state_id==node.state_id]
        if not np.array_equal(np.sort(np.concatenate(selected)),np.arange(node.count)):
            raise ContractError('Temporal row partitions omitted or duplicated state scalars: '+node.state_id)
    for algebraic in ports:
        if any(d not in declared|port_ids for d in algebraic.dependencies):
            raise ContractError('Undeclared algebraic output dependency: '+algebraic.port_id)
    counts=tuple(sum(b.count for b in x) for x in (core,probability,auxiliary))
    expected=(assembly.state_scalar_count,assembly.probability_scalar_count,assembly.auxiliary_z1_scalar_count)
    if counts!=expected:
        raise ContractError('Time partition changed logical state/observation counts')
    return TimePartition(assembly.artifact_sha256,tuple(nodes),tuple(ports),tuple(partitions),
        MappingProxyType(gap_labels),_readonly(component_sizes,np.int64),MappingProxyType(source_hashes),*counts)
