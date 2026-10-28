"""Source-rederived group references and explicit, unwarmed construction IVP.

No fine-state projection or temporal preparation occurs here. Construction
seeds are legal initial-value problems, not qualified coupled backgrounds.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import math

import numpy as np
from scipy import sparse
from scipy.special import expit

from .contracts import ContractError, Quantity, canonical_hash
from .spatial_group_graph import digest, verified_path
from .spatial_group_model import BC, AC, RGC
from .group_time_partition import guard_source_dependencies
from .vh2005 import NativeConeH1, SOURCE_VALUES


NATIVE = 'reduced_native'
ROD = 'Chen2024PrimateRodCurrent'
MEMBRANE = 'ReceptorConductanceMembrane'
BC_MECH = 'HistoryBipolarTerminal'
RGC_MECH = 'RGC_spike_history'
AC_MECH = 'DynamicAmacrineInhibition'
SCHEMA = 'retipath-source-group-construction-seed-1'


def _owned(value):
    a = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(a).all():
        raise ContractError('Nonfinite group binding/initial state')
    a.flags.writeable = False
    return a


def _profile(definitions):
    return {k: _owned(v['value']) if isinstance(v.get('value'), list) else v['value']
            for k, v in definitions.items() if v['binding'] != 'derived'}


def _vector(value, count, name):
    try:
        a = np.broadcast_to(np.asarray(value, dtype=np.float64), (count,)).copy()
    except (TypeError, ValueError) as exc:
        raise ContractError('Construction field shape differs: '+name) from exc
    if not np.isfinite(a).all():
        raise ContractError('Nonfinite construction field: '+name)
    return a


class _CheckedCSR(sparse.csr_matrix):
    def __matmul__(self, other):
        if getattr(self, '_source_guard', None) is not None:
            self._source_guard()
        return super().__matmul__(other)


def source_native_equilibrium(A, B, background_td, *, guard=None):
    """Call the unchanged source equilibrium on GROUP A/B, no fine runtime."""
    A, B = _CheckedCSR(A), _CheckedCSR(B)
    A._source_guard = B._source_guard = guard
    nc, nh = A.shape[1], A.shape[0]
    background = _vector(background_td, nc, 'native actual background')
    if B.shape != (nc, nh) or np.any(background < 0):
        raise ContractError('Group native topology/background differs')
    for matrix in (A, B):
        if (not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)
                or not np.allclose(np.asarray(matrix.sum(1)).ravel(), 1., rtol=0, atol=1e-12)):
            raise ContractError('Source native construction requires original unit-DC operators; no renormalization')
    names = ('R','E','X','C','Vis','g_i','Vbar','z1','z2','Vh')
    units = ('td','td','vH_native_precursor','vH_native_current','vH_native_voltage',
             'vH_native_conductance','vH_native_voltage','vH_native_voltage','vH_native_voltage','vH_native_voltage')
    model = SimpleNamespace(p=SOURCE_VALUES, A=A, B=B, nc=nc, nh=nh, tag='project_approximation',
        state_specs=tuple(SimpleNamespace(unit=u, meaning=n) for n,u in zip(names, units)))
    if guard is not None:
        guard()
    q = Quantity(background, 'td', 'native_retinal_illuminance', 'project_approximation')
    values = NativeConeH1.equilibrium(model, q)
    return {n: _owned(v.values) for n,v in zip(names, values)}


def source_rod_equilibrium(root, rate, count):
    """Unchanged primateRod source fixed point at an explicit total light rate."""
    from .rod_pathway import rod_source_values, phototransduction_equilibrium
    p = rod_source_values(root)
    dark = p['gdark']**p['n']*p['k']
    q = p['beta']*p['cdark']/dark
    smax = p['eta']/p['phi']*p['gdark']*(1+(p['cdark']/p['kGC'])**p['m'])
    rate = _vector(rate, count, 'actual rod background')
    if np.any(rate < 0):
        raise ContractError('Rod construction requires nonnegative total background rate')
    # Repeated identical rates use the exact same source scalar solve.
    unique, inverse = np.unique(rate, return_inverse=True)
    model = SimpleNamespace(p=p, q=q, smax=smax, gdark=p['gdark'], evidence_tag='project_approximation',
        state_specs=tuple(SimpleNamespace(unit='source', meaning=n) for n in ('R','P','G','C','S')))
    source = phototransduction_equilibrium(model, unique)
    states = {n: _owned(v.values[inverse]) for n,v in zip(('R','P','G','C','S'), source)}
    return dict(p, q=q, smax=smax), states


@dataclass(frozen=True)
class GroupConstructionSeed:
    inner_parameters: dict
    states: dict
    spike_states: dict
    observation_prehistory: dict
    input_reference: dict
    identity: dict
    time_seconds: float
    source_construction_legal: bool = True
    coupled_background_qualified: bool = False
    resolved_parameters: object | None = None
    separate_state_bundle: object | None = None

    def require_qualified_background(self):
        raise ContractError('Source construction seed is not a qualified/warmed coupled background')


def build_group_construction_seed(assembly, repo_root, *, input_policy_binding, rng_seed,
                                  time_seconds=0., guard=None):
    """Re-evaluate original construction rules on frozen GROUP operators.

    The current run's approved input policy supplies rod exterior rate; the
    native reference is the actual cached per-group background vector. Every
    source parameter remains per original member; the inner model sums C/gL.
    """
    import json
    from retina_platform.config import RunConfig
    from .rgc_defaults import _validate_adoption
    from .source_reference_candidate import read_source_candidate
    root = Path(repo_root).resolve()
    if (not math.isfinite(time_seconds) or type(rng_seed) is not int or rng_seed < 0
            or rng_seed != RunConfig.project_default(root).seed):
        raise ContractError('Finite origin and unchanged explicit configured PCG64 seed required')
    source_hashes = guard_source_dependencies(root)
    main_path, dynamics_path = root/'configurations/retina_platform_v1.json', root/'configurations/effective_dynamics_v1.json'
    expected = assembly.manifest['parameter_sources']
    if (digest(main_path) != expected['main_configuration_sha256']
            or digest(dynamics_path) != expected['effective_profile_sha256']):
        raise ContractError('Group parameter/main source identity differs from frozen artifact')
    main = json.loads(main_path.read_text(encoding='utf8'))
    profiles = json.loads(dynamics_path.read_text(encoding='utf8'))['profiles']
    policy_path = verified_path(root, input_policy_binding)
    required_policy = (root/assembly.manifest['configuration']['source_run']/'input_policy.json').resolve()
    if policy_path != required_policy:
        raise ContractError('Exact current run input policy required, not an old proposal')
    input_policy = json.loads(policy_path.read_text(encoding='utf8'))
    if input_policy.get('approval_status') != 'USER_AUTHORIZED_CENTERED_ENGINEERING_DEMO_20261024':
        raise ContractError('Actual authorized centered-input policy required')
    rod_rate = float(input_policy['rod']['outside_rate_Rstar_per_s'])
    bg = np.asarray(assembly.array('input/group_background_native_td.npy'), dtype=np.float64)
    nc = assembly.populations['cones'].group_count
    if bg.shape != (nc,) or not np.isfinite(bg).all() or np.any(bg < 0):
        raise ContractError('Actual cached group native background required; no scalar substitute')
    source_hashes.update({str(p.relative_to(root)).replace('\\','/'): digest(p)
        for p in (main_path, dynamics_path, policy_path)})
    for relative in ('retina_sim/atlas_runtime.py','retina_sim/atlas_primary_rod.py',
            'retina_sim/rgc_defaults.py','retina_sim/hybrid.py','retina_platform/config.py'):
        source_hashes[relative] = digest(root/relative)
    if guard is not None:
        guard()
    A = assembly.matrix('routes/cones_to_HC/mean_weights.npz')
    B = assembly.matrix('routes/HC_to_cones/mean_weights.npz')
    native = source_native_equilibrium(A, B, bg, guard=guard)
    release_reference = expit((native['Vis']-B@native['Vh']-SOURCE_VALUES['V_k'])/SOURCE_VALUES['V_n'])/(native['Vbar']/SOURCE_VALUES['V_I'])**SOURCE_VALUES['mu']
    parameters = dict(populations={}, contacts={}, route_gains={}, rod_phototransduction={})
    pops, states = parameters['populations'], {}
    counts = {n: p.group_count for n,p in assembly.populations.items()}
    members = {n: np.asarray(p.counts, dtype=np.float64) for n,p in assembly.populations.items()}
    def put(pop, mechanism, name, value):
        states[pop, mechanism, name] = _owned(value)
    for name in ('X','C','Vis','g_i','Vbar'):
        put('cones',NATIVE,name,native[name])
    for name in ('R','E'):
        put('cones','analytic_RE',name,native[name])
    w = A@native['z1']
    for name,value in (('w',w),('z2',native['z2']),('Vh',native['Vh'])):
        put('HC',NATIVE,name,value)
    aux = assembly.populations['cones'].recording_auxiliary_group_rows
    put('cones','selected_original_z1_auxiliary','z1_aux',native['z1'][aux])
    p_rod, rod = source_rod_equilibrium(root, rod_rate, counts['rods'])
    parameters['rod_phototransduction'] = p_rod
    for name,value in rod.items():
        put('rods',ROD,name,value)
    rod_ref = verified_path(root, main['rod_reference'])
    dark = json.loads(rod_ref.read_text(encoding='utf8'))
    if dark.get('schema_version') != 'fixed-rod-dark-reference-1.0' or dark.get('current_anchor_A') != 0. or dark.get('dark_voltage_V') != -.04:
        raise ContractError('Existing approved fixed rod dark-reference contract required')
    source_hashes[rod_ref.relative_to(root).as_posix()] = digest(rod_ref)
    pops['rods'] = _profile(profiles['effective_rod_membrane'])
    current = -p_rod['k']*rod['G']**p_rod['n']*1e-12
    pops['rods']['current_anchor'] = float(dark['current_anchor_A'])
    pops['rods']['V0'] = pops['rods']['EL']-(current-pops['rods']['current_anchor'])/pops['rods']['gL']
    rod_release = expit((pops['rods']['V0']-pops['rods']['release_half'])/pops['rods']['release_slope'])
    put('rods','rod_membrane','V',pops['rods']['V0'])
    put('rods','rod_membrane','release',rod_release)
    contact_route = {r.contact_component:r for r in assembly.routes.values() if r.contact_component}
    trpm = _profile(profiles['effective_individual_TRPM1'])
    parameters['contacts']['individual_rod_RBC_contacts'] = trpm
    gate_per_rod = expit(-trpm['hill']*np.log(np.maximum(rod_release,np.finfo(float).tiny)/trpm['K']))
    route = contact_route['individual_rod_RBC_contacts']
    conductance = assembly.matrix('routes/'+route.name+'/contact_coefficient_S.npz')@gate_per_rod
    for pop in ('rod_BC','AII_AC'):
        p = pops[pop] = _profile(profiles['effective_'+pop])
        if pop == 'rod_BC':
            total_leak = members[pop]*p['gL']
            p['V0'] = (total_leak*p['EL']+conductance*p['EE'])/(total_leak+conductance)
        V0 = _vector(p['V0'], counts[pop], pop+'.V0')
        put(pop,MEMBRANE,'V',V0)
        put(pop,MEMBRANE,'release',expit((V0-p['release_half'])/p['release_slope']))
        r0 = p['rmax']*expit((V0-p['r_half'])/p['r_slope'])
        put(pop,MEMBRANE,'x',1/(1+p['tau_recovery']*p['U']*r0))
    gc = main['bc_output_gain_control']
    gc_path = verified_path(root,gc['profile'])
    gc_data = json.loads(gc_path.read_text(encoding='utf8'))
    if gc.get('enabled') is not True or gc_data.get('driver') != 'unadapted_terminal_release':
        raise ContractError('Frozen enabled BC raw-release adaptation required')
    source_hashes[gc_path.relative_to(root).as_posix()] = digest(gc_path)
    bc_release = {}
    for pop in BC:
        p = pops[pop] = _profile(profiles['effective_'+pop])
        p['baseline'] = assembly.matrix('routes/cones_to_'+pop+'/mean_weights.npz')@release_reference
        p.update(adaptation_enabled=True, tau_a_seconds=gc_data['tau_a']['value'], adaptation_scale=gc_data['scale']['value'])
        history = np.full((len(p['kernel']),counts[pop]),p['initial_history'],dtype=np.float64)
        if len(p['kernel']) != 3:
            raise ContractError('Frozen three-tap BC history required')
        sign = -1 if pop.endswith('ON') or pop == 'S_ON_BC' else 1
        for i,h in enumerate(history):put(pop,BC_MECH,'history_'+str(i),h)
        activation = np.logaddexp(0.,p['bias']+sign*np.asarray(p['kernel'])@history)
        put(pop,BC_MECH,'activation',activation)
        V0 = _vector(p['V0'],counts[pop],pop+'.V0')
        bc_release[pop] = expit((V0-p['release_half'])/p['release_slope'])
        put(pop,BC_MECH,'V',V0)
        put(pop,BC_MECH,'terminal_release',bc_release[pop])
        put(pop,BC_MECH,'A',p['adaptation_scale']*bc_release[pop])
    ac_policy = main['generic_ac_inhibition']
    ac_path = verified_path(root,ac_policy['profile'])
    ac_control = json.loads(ac_path.read_text(encoding='utf8'))
    if ac_policy.get('enabled') is not True or ac_control.get('mapping') != 'fixed_reference_value_and_DC_slope':
        raise ContractError('Frozen generic AC fixed-reference replacement required')
    source_hashes[ac_path.relative_to(root).as_posix()] = digest(ac_path)
    ac_activation = {}
    for pop in AC:
        p = pops[pop] = _profile(profiles['effective_'+pop])
        pre = 'diffuse_BC_OFF' if pop == 'local_crossover_AC' else 'diffuse_BC_ON'
        baseline = assembly.matrix('routes/'+pre+'_to_'+pop+'/mean_weights.npz')@bc_release[pre]
        value = np.logaddexp(0.,p['bias']); slope = float(np.sum(p['kernel']))*expit(p['bias'])
        p.update(baseline=baseline,reference_input=baseline,output_scale=2*value,
            input_slope=2*slope/value,tau_A_seconds=ac_control['type_time_constants'][pop]['value'])
        put(pop,AC_MECH,'h',np.full(counts[pop],.5))
        ac_activation[pop] = np.full(counts[pop],value)
    gains = parameters['route_gains']
    totals = {pop:{'excitatory':np.zeros(counts[pop]),'inhibitory':np.zeros(counts[pop])} for pop in RGC}
    for r in assembly.routes.values():
        if r.post not in RGC:
            continue
        receptor = 'inhibitory' if r.pre in AC else 'excitatory'
        definitions = profiles['effective_'+r.post+'_'+receptor+'_'+r.pre]
        gain = float(definitions['gain']['value'])
        gains[r.name] = gain
        initial = ac_activation[r.pre] if r.pre in AC else bc_release[r.pre]
        # Match original source construction order: V0 is bound before the
        # later output-adaptation rebuild, so this uses RAW BC release.
        totals[r.post][receptor] += gain*(assembly.matrix('routes/'+r.name+'/sum_weights.npz')@initial)
    adoptions = main['rgc_output_defaults']['accepted_types']
    spike_states = {}
    for ordinal,pop in enumerate(RGC):
        p = pops[pop] = _profile(profiles['effective_'+pop])
        rule = adoptions[pop]
        recipe,path = read_source_candidate(root,root/rule['recipe_path'])
        if digest(path) != rule['recipe_sha256']:
            raise ContractError('Current adopted RGC recipe identity differs')
        acceptance_path = verified_path(root,rule['acceptance'])
        accepted = json.loads(acceptance_path.read_text(encoding='utf8'))
        _validate_adoption(accepted,pop,recipe)
        for binding in accepted.get('evidence_files',[]):verified_path(root,binding)
        for name,row in recipe['changes'].items():p[name] = float(row['value'])
        source_hashes[path.relative_to(root).as_posix()] = digest(path)
        source_hashes[acceptance_path.relative_to(root).as_posix()] = digest(acceptance_path)
        leak = members[pop]*p['gL'];e,i = totals[pop]['excitatory'],totals[pop]['inhibitory']
        V0 = (leak*p['EL']+e*p['EE']+i*p['EI'])/(leak+e+i)
        p['V0'] = V0
        put(pop,RGC_MECH,'V',V0)
        h = np.full((len(p['spike_history_kernel']),counts[pop]),p['initial_spike_history'],dtype=np.float64)
        if h.shape[0] != 3 or np.any(h != 0):
            raise ContractError('Existing declared all-zero three-bin history required')
        for j in range(8):put(pop,'RGC_expected_history','history_probability_'+str(j),np.full(counts[pop],1. if j==0 else 0.))
        rng = np.random.Generator(np.random.PCG64(rng_seed+100+ordinal))
        rate = p['alpha']*np.logaddexp(0.,(V0+np.asarray(p['spike_history_kernel'])@h-p['mu'])/p['beta'])
        spike_states[pop] = dict(history=h,last_bin_counts=None,rng_state=deepcopy(rng.bit_generator.state),
            pending_events=[],conditional_rate_Hz=rate)
    for component,r in contact_route.items():
        post = assembly.array('routes/'+r.name+'/contact_post_group.npy')
        pre = assembly.array('routes/'+r.name+'/contact_pre_group.npy')
        if component == 'individual_rod_RBC_contacts':
            put(r.pre,component,'m',gate_per_rod[pre])
            continue
        p = parameters['contacts'][component] = _profile(profiles['effective_'+component])
        ppre = pops[r.pre]
        V = states[r.pre,MEMBRANE,'V']
        x = states[r.pre,MEMBRANE,'x']
        event_rate = ppre['Npool']*ppre['U']*ppre['rmax']*expit((V-ppre['r_half'])/ppre['r_slope'])*x
        if component.startswith('AII_'):
            Vpost = states[r.post,BC_MECH,'V'][post]
            decay = p['decay_minus100']+(p['decay_0']-p['decay_minus100'])*np.clip((Vpost+.1)/.1,0,1)
            p['tau_decay'] = _owned(decay)
        else:
            decay = p['tau_decay']
        put(r.pre,component,'a',event_rate[pre]*p['tau_rise'])
        put(r.pre,component,'b',event_rate[pre]*decay)
    expected_blocks = (*assembly.state_blocks,*assembly.probability_blocks,*assembly.auxiliary_z1_blocks)
    if set(states) != {(b.population,b.mechanism,b.name) for b in expected_blocks}:
        raise ContractError('Source construction omitted a declared group state')
    for b in expected_blocks:
        if states[b.population,b.mechanism,b.name].shape != (b.count,):
            raise ContractError('Source construction state layout differs: '+b.population+'.'+b.name)
    if guard is not None:guard()
    initial_hash = canonical_hash({':'.join(k):v for k,v in states.items()})
    identity = dict(schema=SCHEMA,artifact_sha256=assembly.artifact_sha256,
        input_policy=dict(input_policy_binding),source_sha256=source_hashes,
        initial_state_sha256=initial_hash,parameter_sha256=canonical_hash(parameters),
        time_seconds=float(time_seconds),rng_seed=rng_seed,
        source_construction_legal=True,coupled_background_qualified=False,
        classification='SOURCE_CONSTRUCTION_IVP_NOT_PREPARED_BACKGROUND',
        parent_checkpoint_used=False,neural_model_advanced=False,
        reference_rule='Actual cached group native background and approved rod exterior background; no implicit100/5',
        parameters_rule='Original broadcast/literal values and source derivations re-evaluated on exact group mean/sum matrices; never average incompatible fine parameters')
    observation = dict(delay_seconds=SOURCE_VALUES['observation_delay']*.001,
        origin_seconds=float(time_seconds),history_times=np.array([time_seconds]),
        history_values=states['HC',NATIVE,'Vh'][None,:],
        prehistory=states['HC',NATIVE,'Vh'],unit='vH_native_voltage',meaning='native_relative_voltage',
        interpolation='linear',prehistory_kind='explicit source construction constant for t<origin',
        feedback_uses_current_state=True)
    result = GroupConstructionSeed(parameters,states,spike_states,observation,
        dict(native_background_td=_owned(bg),rod_background_rstar_s=rod_rate),identity,float(time_seconds))
    from .parameter_migration import bind_constructed_group
    return bind_constructed_group(assembly,result,root)
