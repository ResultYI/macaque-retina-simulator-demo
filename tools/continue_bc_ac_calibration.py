"""Qualify existing AC observations and bind the frozen model to GIN69 gray input."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import time

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.linalg import LinearOperator, gmres, spsolve
from scipy.special import expit

from retina_sim.assembly import RetinaBuilder
from retina_sim.contracts import canonical_hash, ContractError
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.hc_eccentric import spatial_parameters
from retina_sim.hc_biexponential import _exponential_weights
from retina_sim.h2_priority import complete_priority
from retina_sim.h1_h2_effective import cone_fixed_point, matrix_id, MECHANISM
from retina_sim.group_time_storage import save_bundle
from retina_sim.group_inner_dynamics import BC_MECH, MEMBRANE, RGC_MECH, AC_MECH
from retina_sim.aii_functional import V_KEY, S_KEY, COMPONENTS
from retina_sim.ac_vs_model import equilibrium, MECHANISM as AC_VS, LEGACY_S
from retina_sim.spatial_group_model import BC, RGC
from retina_sim.rgc_expectation import stationary_history_rates, STATES
from retina_sim.spatial_group_graph import digest
from retina_sim.vh2005 import SOURCE_VALUES as P
from tools.prepare_gin69_joint_hc import guard
from tools.durable_research_runtime_v1 import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2'
AC = ROOT/'reports/ac_vs_redesign_v1'
DEST = OUT/'current_ac_source'


def read(path):
    return json.loads(Path(path).read_bytes())


def qualify():
    started = time.perf_counter()
    registry = read(ROOT/'configurations/regional_observations_v1.json')
    accepted = read(AC/'FUNCTIONAL_ACCEPTANCE.json')
    for path, expected in accepted['implementation_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ContractError('Accepted AC implementation changed: '+path)
    channels = read(ROOT/'configurations/ac_vs_functional_v1.json')['channels']
    targets = []
    for channel in channels:
        parasol = 'parasol' in channel['target']
        sources = ['Crook2014', 'Cafaro2013'] if parasol else ['Crook2011', 'Cafaro2013']
        targets.append(dict(pool=channel['pool'], target=channel['target'], relation=channel['relation'],
            sources=sources, direct_matched_time_series=False, fit_eligible=False,
            observation_class='published population summaries and qualitative circuit constraints',
            reason='No registered independent stimulus-aligned gI waveform with matching type, region, light, units and repeat split'))
    targets.extend([
        dict(pool=LEGACY_S, sources=['Crook2009'], direct_matched_time_series=False, fit_eligible=False,
             reason='Published RGC output latency/opponency is not a separate S-ON AC conductance waveform'),
        dict(pool='AII_AC', sources=['Percival2022'], direct_matched_time_series=False, fit_eligible=False,
             status='AII_FUNCTIONAL_CONSTRAINED', reason='No independent AII/RBC/rod-driven BC time series; puff summary is not a per-edge gain')])
    result = dict(status='AC_CALIBRATION_NO_VALID_TARGET', model_identity=accepted['model_identity'],
        BC=dict(fixed=True, qualification='conditional source-constrained DC anchors plus engineering acceptance; no matched dynamic fit',
                S_ON='engineering parameters retained', OFF_DC_caveat='functional AII retains chord conductance, not old glycine differential slope'),
        pools=targets, parameters_fitted=[], optimizer_calls=0, AC_update_accepted=False,
        metrics=dict(baseline_train=None, candidate_train=None, baseline_development=None, candidate_development=None,
                     reason='No eligible independent target; no sensitivity or optimizer run is justified'),
        AII=dict(status='AII_FUNCTIONAL_CONSTRAINED', unchanged=True,
                 retained=['Rod-RBC-AII', 'AII-AII gap', 'AII-ON BC bidirectional gap', 'AII-OFF BC inhibition']),
        excluded_sources=dict(Latimer2019=0, Trong_Rieke=0, Cell6='methodology only; not AC physiological teacher'),
        source_observation_audit={k:dict(material=registry['source_materials'].get(k),
            constraints=registry['source_mechanism_constraints'].get(k),
            raw_time_sampling=None, synchronized_stimulus_arrays=None, independent_train_development_split=None,
            missing_fields_not_inferred=True) for k in ['Crook2014','Crook2011','Crook2009','Cafaro2013','Percival2022']},
        input_sha256={p:digest(ROOT/p) for p in ['configurations/regional_observations_v1.json',
            'data/source_data/bc_physiology_targets.json', 'reports/ac_vs_redesign_v1/FUNCTIONAL_ACCEPTANCE.json',
            'reports/BC_PHYSIOLOGY_IMPLEMENTATION_2026-11-02/CALIBRATION_METRICS.json']},
        accepted_engineering_repeated=False, wall_seconds=time.perf_counter()-started,
        next_stage='RGC69_6 frozen30parameter voltage GLM; source background and current neural cache required')
    atomic_json(AC/'LAYERED_CALIBRATION.json', result)
    return result


def source_outer(model):
    """Reweight cached gray products with the frozen current full fine-source rows."""
    began = time.perf_counter()
    a = model.assembly
    folder = OUT/'executions/assemble_native_01'
    rec = read(folder/'NATIVE_ASSEMBLY_RESULT.json')
    cones, hcs = [FrozenDiskTable(ROOT, rec[k]) for k in ('cone_table','hc_table')]
    ct, ht = cones.extras['subtype'].astype('U1'), hcs.extras['subtype'].astype('U2')
    with np.load(folder/'MEMBERSHIP.npz') as z:
        mem = {k:z[k].copy() for k in z.files}
    with np.load(OUT/'joint_gray_background/STATIC_NATIVE_SOURCES.npz') as z:
        gray = {k:z[k].copy() for k in z.files}
    with np.load(OUT/'joint_gray_background/SOURCE_GEOMETRY.npz') as z:
        geometry = {k:z[k].copy() for k in ('candidate_rows','cone_types','pedicle_mm')}
    np.testing.assert_array_equal(gray['candidate_rows'], geometry['candidate_rows'])
    source = ROOT/'reports/CONE_HC_OUTER_SOURCE_EXTENSION_20261102_v3/cones_01'
    official = np.load(source/'official_source_rows.npy')
    xy = np.load(source/'cone_pedicle_mm.npy', mmap_mode='r')
    np.testing.assert_array_equal(xy[gray['candidate_rows']], geometry['pedicle_mm'])
    with np.load(ROOT/'reports/CONE_HC_MINIMAL_REAL_DIAGNOSTIC_20261102_v6/run_02/SOURCE_LABELS.npz') as z:
        labels = z['cone_candidate_rows']; types = z['cone_types'].astype('U1')
    lm = labels[types != 'S']
    if not np.all(np.isin(lm, gray['candidate_rows'])):
        raise ContractError('Current complete LM source lacks frozen optical gray capture')
    index = np.full(len(official), -1, dtype=int); index[gray['candidate_rows']] = np.arange(len(gray['candidate_rows']))
    lm_gray = index[lm]
    np.testing.assert_array_equal(types[types != 'S'], geometry['cone_types'][lm_gray])
    cone_ids = np.array([int(k.decode('ascii').split(':')[-1]) for k in cones.keys])
    candidate = np.searchsorted(official, cone_ids)
    np.testing.assert_array_equal(official[candidate], cone_ids)
    parent_map = np.full(len(official), -1, dtype=int)
    parent_map[candidate] = mem['global_cone_fine_to_group']
    maps = parent_map[lm]
    hcmap, hcounts, ccounts = mem['fine_to_group'], mem['counts'], mem['cone_counts']
    nh, nc = len(hcounts), len(ccounts)
    raw = sparse.load_npz(folder/'RAW_HC_BY_CONE.npz').tocsr()
    h2fine = np.flatnonzero(ht == 'H2')
    h2 = complete_priority(raw[h2fine], ct, read(ROOT/'configurations/research/H2_S_priority.json'))
    required = np.unique(h2.indices)
    if np.any(mem['global_cone_fine_to_group'][required] < 0):
        raise ContractError('H2 source is outside captured parent cone background')
    group = mem['global_cone_fine_to_group']; valid = np.flatnonzero(group >= 0)
    projection = sparse.csr_matrix((np.ones(len(valid)), (valid,group[valid])), shape=(len(ct),nc))
    hp = sparse.csr_matrix((1/hcounts[hcmap[h2fine]], (hcmap[h2fine],np.arange(len(h2fine)))), shape=(nh,len(h2fine)))
    A = (hp@h2@projection).toarray()
    ar, av = np.zeros(nh), np.zeros(nh)
    par, _, _ = spatial_parameters(ROOT,hcs.coordinates['soma'],ht,read(ROOT/'configurations/research/HC_eccentric_biexponential.json'))
    tree = cKDTree(xy[lm])
    for j in np.flatnonzero(ht == 'H1'):
        guard()
        ids = np.asarray(tree.query_ball_point(hcs.coordinates['soma'][j],par['support_radius_mm'][j],return_sorted=True),int)
        d = np.linalg.norm(xy[lm[ids]]-hcs.coordinates['soma'][j],axis=1)
        w = .15*_exponential_weights(d,par['lambda_short_mm'][j])+.85*_exponential_weights(d,par['lambda_long_mm'][j])
        inside = maps[ids] >= 0; g = hcmap[j]; n = hcounts[g]
        A[g] += np.bincount(maps[ids[inside]],weights=w[inside],minlength=nc)/n
        outside = lm_gray[ids[~inside]]
        ar[g] += np.dot(w[~inside],gray['release'][outside])/n
        av[g] += np.dot(w[~inside],gray['Vbar'][outside])/n
    A = sparse.csr_matrix(A)
    B = a.global_B
    hr, cr = a.native_rows['HC'], a.native_rows['cones']
    localA, localB = A[hr][:,cr], B[cr][:,hr]
    errors = dict(A=float(np.max(abs((localA-a.matrix('routes/cones_to_HC/mean_weights.npz')).data),initial=0.)),
                  B=float(np.max(abs((localB-a.matrix('routes/HC_to_cones/mean_weights.npz')).data),initial=0.)))
    if max(errors.values()) > 2e-14:
        raise ContractError('Source background would change frozen local matrices: '+str(errors))
    native = a.runtime_factories['native'](a,bound=False)
    q = native.Qfull.tocoo()
    Q = sparse.csr_matrix((q.data,(hr[q.row],hr[q.col])),shape=(nh,nh))
    linear = sparse.eye(nh)+native.coupling_rate*P['tau_h']*Q/1000.
    light = a.global_light.copy(); cone = cone_fixed_point(light)
    scale = (cone['Vbar']/P['V_I'])**P['mu']
    def release(h):
        return expit((cone['Vis']-B@h-P['V_k'])/P['V_n'])/scale
    def residual(h):
        return linear@h-P['g_t']*(A@release(h)+ar)
    h = P['g_t']*(A@release(np.zeros(nh))+ar)*.5
    errors_bg = []
    for iteration in range(200):
        guard(); f = residual(h); error = float(np.max(abs(f))); errors_bg.append(error)
        if error <= 1e-10:
            break
        q = expit((cone['Vis']-B@h-P['V_k'])/P['V_n']); derivative = q*(1-q)/(P['V_n']*scale)
        J = LinearOperator((nh,nh),matvec=lambda v:linear@v+P['g_t']*(A@(derivative*(B@v))),dtype=float)
        delta,status = gmres(J,-f,rtol=1e-10,atol=1e-14,restart=32,maxiter=200)
        if status:
            raise ContractError('Frozen gray Newton GMRES budget failed')
        for power in range(32):
            proposed = h+delta*.5**power
            if np.all(proposed>=0) and np.max(abs(residual(proposed))) < error:
                h = proposed; break
        else:
            raise ContractError('Frozen gray Newton line search failed')
    else:
        raise ContractError('Frozen gray equilibrium failed')
    r = release(h)
    outside_c = np.ones(nc,bool); outside_c[cr] = False
    outside_h = np.ones(nh,bool); outside_h[hr] = False
    boundary = dict(A_release=A[hr][:,outside_c]@r[outside_c]+ar[hr],
                    A_Vbar=A[hr][:,outside_c]@cone['Vbar'][outside_c]+av[hr],
                    B_h=B[cr][:,outside_h]@h[outside_h])
    # Bind only this newly built object; frozen builders and saved diagnostics are untouched.
    a.native_boundary = boundary
    native = a.runtime_factories['native'](a,bound=False); native.boundary = boundary
    seed = deepcopy(model.seed); state = seed['states']
    for name,value in cone.items():
        state['cones','analytic_RE' if name in ('R','E') else 'reduced_native',name] = value[cr].copy()
    w = P['g_t']*(A[hr[a.h1_rows]]@r+ar[hr[a.h1_rows]])
    for name,value in [('w',w),('z2',w),('h1',h[hr[a.h1_rows]]),('h2',h[hr[a.h2_rows]])]:
        state['HC',MECHANISM,name] = value.copy()
    state['cones','selected_original_z1_auxiliary','z1_aux'] = P['g_t']*r[cr][np.asarray(native.aux_indices,dtype=int)]
    seed['input_reference']['native_background_td'] = light[cr].copy()
    seed['observation_prehistory'].update(history_times=np.array([0.]),history_values=h[hr][None,:],prehistory=h[hr].copy(),origin_seconds=0.)
    seed['h1_h2_model']['boundary'] = deepcopy(boundary)
    y,anchor = native.construction_point(SimpleNamespace(states=state,time_seconds=0.),light[cr],1.,0)
    maxrhs = float(np.max(abs(native.rhs(y,0.,anchor))))
    if maxrhs > 1e-8:
        raise ContractError('Current gray native equilibrium fails existing RHS gate: '+str(maxrhs))
    return seed, native, dict(parent_A_sha256=matrix_id(A),local_operator_error=errors,
        native_rhs=maxrhs,outer_residuals=errors_bg,gray_native_range=[float(light.min()),float(light.max())],
        static_LM_sources=len(gray['candidate_rows']), optical_recomputed=False,
        source_capture_sha256=digest(OUT/'joint_gray_background/STATIC_NATIVE_SOURCES.npz'),
        boundary_sha256=canonical_hash(boundary),wall_seconds=time.perf_counter()-began)


def source_inner(model, seed, native):
    state = seed['states']; a = model.assembly
    inner = a.runtime_factories['inner'](a,seed['inner_parameters'])
    y,anchor = native.construction_point(SimpleNamespace(states=state,time_seconds=0.),seed['input_reference']['native_background_td'],1.,0)
    release = native.evaluate(y,0.,anchor).outputs['release']
    target = {}
    for pop in BC:
        h = a.matrix('routes/cones_to_'+pop+'/mean_weights.npz')@release-inner.par(pop,'baseline')
        for i in range(3):
            state[pop,BC_MECH,'history_'+str(i)] = h.copy()
        sign = 1 if pop.endswith('_OFF') else -1
        activation = np.logaddexp(0.,inner.par(pop,'bias')+sign*np.sum(inner.p[pop]['kernel'])*h)
        state[pop,BC_MECH,'activation'] = activation
        target[pop] = inner.par(pop,'Vref')+inner.par(pop,'native_voltage_gain')*(activation-inner.par(pop,'native_reference'))
    ge = inner.K[COMPONENTS[0]]@state['rod_BC',MEMBRANE,'release']
    diagonal,right = [],[]
    for pop in inner.gap_names:
        e = ge if pop=='AII_AC' else np.zeros(inner.counts[pop])
        v = inner.par(pop,'EL') if pop=='AII_AC' else target[pop]
        diagonal.append(inner.gL[pop]+e)
        right.append(inner.gL[pop]*v+e*(inner.par(pop,'EE') if pop=='AII_AC' else 0.))
    voltage = spsolve(inner.gap.matrix+sparse.diags(np.concatenate(diagonal)),np.concatenate(right)-inner.gap_boundary_product)
    for pop,section in inner.gap_slices.items():
        state[pop,MEMBRANE if pop=='AII_AC' else BC_MECH,'V'] = voltage[section].copy()
    state[S_KEY] = expit((state[V_KEY]-inner.par('AII_AC','release_half'))/inner.par('AII_AC','release_slope'))
    for component in COMPONENTS[1:]:
        pop = inner.contacts[component]['route'].post
        gi = inner.K[component]@state[S_KEY]
        state[pop,BC_MECH,'V'] = (inner.gL[pop]*target[pop]+gi*inner.par(pop,'EI'))/(inner.gL[pop]+gi)
    for pop in BC:
        rel = expit((state[pop,BC_MECH,'V']-inner.par(pop,'release_half'))/inner.par(pop,'release_slope'))
        state[pop,BC_MECH,'terminal_release'] = rel
        state[pop,BC_MECH,'A'] = inner.par(pop,'adaptation_scale')*rel if inner.p[pop]['adaptation_enabled'] else np.zeros_like(rel)
    current = lambda:{k:state[k] for k in inner.state_keys}
    outputs = inner.outputs(0.,current(),release)
    state[LEGACY_S,AC_MECH,'h'] = outputs[LEGACY_S,'target_inhibition'].copy()
    for channel in inner.channels:
        pop = channel['pool']; v,_,s = equilibrium(outputs[pop,'gE'],inner.p[pop])
        state[pop,AC_VS,'V'],state[pop,AC_VS,'s'] = v,s
    outputs = inner.outputs(0.,current(),release)
    for pop in RGC:
        ge,gi,g = outputs[pop,'gE'],outputs[pop,'gI'],inner.gL[pop]
        v = (g*inner.par(pop,'EL')+ge*inner.par(pop,'EE')+gi*inner.par(pop,'EI'))/(g+ge+gi)
        state[pop,RGC_MECH,'V'] = v
        law = stationary_history_rates(SimpleNamespace(history_kernel=np.asarray(inner.p[pop]['spike_history_kernel']),dt=.001,
            deterministic_state_independent_of_spikes=True,**{n:inner.par(pop,n) for n in ('alpha','mu','beta')}),v,cold_start_bins=1)
        pi = law['history_probabilities']
        for i in range(8):
            state[pop,'RGC_expected_history','history_probability_'+str(i)] = pi[:,i]
        rng = np.random.Generator(np.random.PCG64()); rng.bit_generator.state = deepcopy(seed['spike_states'][pop]['rng_state'])
        picks = np.sum(rng.random(len(v))[:,None]>np.cumsum(pi,axis=1),axis=1)
        if np.any(picks>=8) or seed['spike_states'][pop]['pending_events']:
            raise ContractError('Invalid stationary model prehistory')
        history = STATES[picks].T.astype(float)
        seed['spike_states'][pop].update(history=history,last_bin_counts=None,rng_state=deepcopy(rng.bit_generator.state),
            conditional_rate_Hz=inner.par(pop,'alpha')*np.logaddexp(0.,(v+np.asarray(inner.p[pop]['spike_history_kernel'])@history-inner.par(pop,'mu'))/inner.par(pop,'beta')))
    inner.validate(current())
    residual = max(float(np.max(abs(v),initial=0.)) for v in inner.rhs(0.,current(),release).values())
    if residual>1e-8:
        raise ContractError('Current functional inner gray background fails RHS gate: '+str(residual))
    if canonical_hash(seed['inner_parameters']) != canonical_hash(model.seed['inner_parameters']):
        raise ContractError('Source background changed frozen physiology parameters')
    seed['joint_background_initialization'].update(initial_state_sha256=canonical_hash(state),
        held_reservoir_sha256=canonical_hash(dict(native=seed['h1_h2_model']['boundary'],inner=a.boundary_runtime)),
        source_gray_rebinding=True, parameters_sha256=canonical_hash(seed['inner_parameters']))
    seed['functional_aii'] = dict(seed['functional_aii'],source_gray_rebinding=True,maximum_background_rhs=residual)
    return residual


def prepare():
    started = time.perf_counter(); qualify()
    DEST.mkdir(exist_ok=True)
    if (DEST/'seed').exists():
        raise FileExistsError('Retain completed current-source seed')
    model = RetinaBuilder.build_research_ac(ROOT)
    if model.identity != read(AC/'FUNCTIONAL_ACCEPTANCE.json')['model_identity']:
        raise ContractError('Current accepted AC model identity differs')
    original_hash = canonical_hash(model.seed)
    seed,native,evidence = source_outer(model)
    residual = source_inner(model,seed,native)
    inputs = read(OUT/'FORMAL_INPUT_LOCATION.json'); folder = Path(inputs['path']); receipt = read(folder/'RESULT.json')
    if digest(folder/'RESULT.json')!=inputs['result_sha256'] or digest(folder/'NATIVE_INPUTS.npz')!=receipt['array_sha256']:
        raise ContractError('Frozen real-source input changed')
    if receipt['split_sha256'] != digest(OUT/'SPLIT_CONTRACT.json'):
        raise ContractError('Frozen time split changed')
    with np.load(folder/'NATIVE_INPUTS.npz') as z:
        drive = z['drive'].copy()
        np.testing.assert_array_equal(z['local_cone_rows'],model.assembly.parent_population_rows['cones'])
        np.testing.assert_array_equal(z['frame_start_seconds'],np.arange(6750)/150.)
    assert drive.shape==(6750,model.assembly.populations['cones'].group_count)
    seed['native_inputs'] = {i:drive[i] for i in range(6750)}
    rod = next(iter(seed['rod_inputs'].values()))
    seed['rod_inputs'] = {i:rod.copy() for i in range(6750)}
    assert canonical_hash(model.seed)==original_hash
    identity = canonical_hash(dict(model=model.identity,seed=seed,input=receipt['array_sha256'],split=receipt['split_sha256']))
    save_bundle(DEST/'seed',kind='source_construction_seed',identity=identity,payload=seed,
        qualification=dict(default_adopted=False,fit_eligible=False))
    result = dict(status='CURRENT_AC_SOURCE_BACKGROUND_READY',identity=identity,model_identity=model.identity,
        source_outer=evidence,inner_rhs=residual,parameters_sha256=canonical_hash(seed['inner_parameters']),
        AII_mapping_sha256=canonical_hash(model.assembly.aii_mapping),source_native_input_sha256=receipt['array_sha256'],
        split_sha256=receipt['split_sha256'],seed_manifest_sha256=digest(DEST/'seed/manifest.json'),
        no_parameter_updates=True,old_checkpoint_reused=False,wall_seconds=time.perf_counter()-started,
        source_code_sha256=digest(Path(__file__)),FIT_READY=False,background_qualified=False,production_adopted=False)
    atomic_json(DEST/'BACKGROUND.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    prepare()
