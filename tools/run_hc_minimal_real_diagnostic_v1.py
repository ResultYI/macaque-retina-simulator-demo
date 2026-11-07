from retina_sim.hc_research_native import joint_equilibrium
"""Three conditioned Cone/H1 experiments on immutable source cells, no fitting."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import argparse
import copy
import ctypes
import hashlib
import json
import os
import time
import traceback

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from scipy.special import expit

from retina_sim.contracts import canonical_hash
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.source_atlas import SourceDensityAtlas
from retina_sim.atlas_s_partition import EffectiveMulattaSPartition
from retina_sim.hc_biexponential import support_radius
from retina_sim.hc_minimal_diagnostic import bind_diagnostic, require_diagnostic_native
from retina_sim.h1_h2_effective import H1H2Native, cone_fixed_point, matrix_id
from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
from retina_sim.hc_expansion_geometry import native_td_step
from retina_sim.vh2005 import SOURCE_VALUES


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data): path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def peak_memory_bytes():
    if os.name != 'nt':
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)]+[
            (key, ctypes.c_size_t) for key in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.windll.kernel32; kernel.GetCurrentProcess.restype = ctypes.c_void_p
    func = ctypes.windll.psapi.GetProcessMemoryInfo
    func.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not func(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise RuntimeError('Peak working-set measurement failed')
    return int(counters.PeakWorkingSetSize)


def extend_labels(root, positions_is, official_rows, unknown, labels, source_record):
    if not len(unknown): return dict(new_labels=0)
    atlas = SourceDensityAtlas(root, profile_id='retina_atlas_human_borrowed_disk3mm_v1')
    expanded = copy.copy(atlas.cone_master)
    official = root/'vendor_pinned/isetbio/data/sdr/isetbio-mosaics/cone_lattices/right/nominal-fov-58deg.mat'
    if sha(official) != source_record['official_source_sha256']: raise RuntimeError('Source changed')
    obj = loadmat(official, simplify_cells=True)
    if float(obj['params']['radius'])/1000 < 6.5: raise RuntimeError('Label macros not covered')
    all_xy = -np.asarray(obj['rfPositions'], float)/1000
    context = np.flatnonzero(np.linalg.norm(all_xy, axis=1) <= 6.5)
    expanded.positions = all_xy[context]
    expanded.source_keys = tuple('sdr_human_54eb0a9ef905_row:'+str(r) for r in context)
    expanded.bounds = (-6.5, 6.5, -6.5, 6.5); expanded._tiles = {}; expanded._tile_indices = {}
    original_tile = expanded.tile
    def tile(tx, ty, size):
        corners = np.array([[tx, ty], [tx+1, ty], [tx, ty+1], [tx+1, ty+1]])*size
        if np.linalg.norm(corners, axis=1).max() > 6.5: raise RuntimeError('Unsupported label macro')
        return original_tile(tx, ty, size)
    expanded.tile = tile; atlas.cone_master = expanded
    keys = tuple('sdr_human_54eb0a9ef905_row:'+str(r) for r in official_rows[unknown])
    values, audit = EffectiveMulattaSPartition(atlas).assign(atlas._table('cones', keys, positions_is[unknown]))
    labels[unknown] = values
    return dict(new_labels=len(unknown), provenance=audit)


def native_model(binding, hc_rows, cone_rows):
    nc, nh = binding['B'].shape
    cones = SimpleNamespace(group_count=nc, group_ids=cone_rows,
        recording_auxiliary_group_rows=np.array([], int), recording_auxiliary_fine_rows=np.array([], int))
    p = SOURCE_VALUES
    pars = dict(tau_seconds=p['tau_h']*.001, input_scale=p['g_t'], feedback_scale=1.)
    assembly = SimpleNamespace(populations={'cones': cones,
        'HC': SimpleNamespace(group_count=nh, group_ids=hc_rows)},
        matrix=lambda path: binding['A'] if 'cones_to_HC' in path else binding['B'],
        artifact_sha256=binding['identity'], h1_rows=np.arange(nh), h2_rows=np.array([], int),
        h2_parameters=pars, layout={'scope': 'minimal_real_graph_diagnostic', 'HC_parent_rows': hc_rows},
        connection_identity={'identity': binding['identity']})
    native = H1H2Native(assembly); native.boundary = binding['boundary']
    return native, pars


def integrate(native, y0, light, stimulus, *, tightened=False):
    times = np.arange(61)*.002
    full = []
    rtol, atol, step = (1e-9, 1e-12, .0001) if tightened else (1e-7, 1e-10, .0002)
    anchor = AnalyticREAnchor(0., light, light, light, .02, 0)
    nfev = 0; clock = time.perf_counter()
    for start, end, selected in ((0., .02, times <= .02), (.02, .12, times > .02)):
        sol = solve_ivp(lambda t, y: native.rhs(y, t, anchor), (start, end), y0,
                        method='DOP853', rtol=rtol, atol=atol, max_step=step, t_eval=times[selected])
        if not sol.success or not np.isfinite(sol.y).all(): raise RuntimeError(sol.message)
        full.append(sol.y); nfev += sol.nfev
        y0 = sol.y[:, -1]
        if end == .02:
            R, E = anchor.at(.02); anchor = AnalyticREAnchor(.02, R, E, stimulus, .12, 1)
    y = np.column_stack(full)
    if y.shape != (native.nstate, 61): raise RuntimeError('Incomplete 120ms trajectory')
    return times, y, dict(nfev=nfev, wall_seconds=time.perf_counter()-clock,
                          rtol=rtol, atol=atol, max_step_seconds=step, success=True)


def readout(n, t, y, target, core, gaps):
    p = SOURCE_VALUES
    h = y[n.slices['Vh']]
    feedback = n.B@h+n.boundary['B_h'][:, None]
    q = expit((y[n.slices['Vis']]-feedback-p['V_k'])/p['V_n'])
    release = q/(y[n.slices['Vbar']]/p['V_I'])**p['mu']
    pool = n.A@release+n.boundary['A_release'][:, None]
    if not all(np.isfinite(a).all() for a in (y, release, pool, feedback)):
        raise RuntimeError('Nonfinite readout')
    return dict(t_seconds=t, y=y, H1_target=h[target], release_core=release[core],
        pooled_release_target=pool[target], feedback_core=feedback[core],
        release_unsupported=release[gaps], feedback_unsupported=feedback[gaps],
        q_core=q[core], sigmoid_slope_core=q[core]*(1-q[core])/p['V_n'],
        q_unsupported=q[gaps], release_all=release, H1_all=h)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); root = Path(__file__).resolve().parents[1]
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat(); clock = time.perf_counter()
    result = dict(started_utc=started, status='RUNNING', scope='minimal_real_graph_diagnostic',
                  production_adopted=False, boundary_convergence_qualified=False)
    try:
        v5 = root/'reports/CONE_HC_FINITE_DYNAMIC_HALO_20261102_v5'
        frozen = json.loads((v5/'preflight_01/FROZEN_PROTOCOL.json').read_bytes())['protocol']
        protocol = dict(schema='minimal_real_graph_diagnostic_v1', input=frozen['stimulus'],
            background_td=1000., duration_seconds=.12, solver=frozen['solver'],
            numerical_budget=frozen['convergence'], refinement_condition='B_biexponential_once',
            refinement=dict(rtol=1e-9, atol=1e-12, max_step_seconds=.0001),
            dynamic_domain='OPL bounding rectangle of target seven contacts and known M gap, padded 5um',
            dynamic_HC='all existing H1 B partners of every dynamic LM; include target row2891',
            no_new_cells=True, grouping='fine identity, no spatial approximation',
            exterior='known full A column mass times homogeneous 1000Td compatible background per condition',
            exterior_limit='external stimulus changes and propagated perturbations are omitted',
            permission='user-authorized registered zero B rows retain true input and full Cone dynamics',
            strict_structure_gate_bypassed_only_by_independent_diagnostic_permission=True)
        write(out/'FROZEN_PROTOCOL.json', dict(frozen_utc=started, protocol=protocol))
        base = root/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/executions/assemble_native_01'
        rec = json.loads((base/'NATIVE_ASSEMBLY_RESULT.json').read_bytes())
        cones, hcs = FrozenDiskTable(root, rec['cone_table']), FrozenDiskTable(root, rec['hc_table'])
        raw = sparse.load_npz(base/'RAW_HC_BY_CONE.npz').tocsr()
        ct, ht = cones.extras['subtype'].astype('U1'), hcs.extras['subtype'].astype('U2')
        cx, hx = cones.coordinates['opl_pedicle'], hcs.coordinates['soma']
        core = raw[2891].indices; known_gap = 166128
        seeds = np.r_[core, known_gap]; lo = cx[seeds].min(0)-.005; hi = cx[seeds].max(0)+.005
        ci = np.flatnonzero(np.all((cx >= lo) & (cx <= hi), axis=1) & (ct != 'S'))
        partners = np.unique(raw[:, ci].tocoo().row)
        hj = np.union1d(partners[ht[partners] == 'H1'], [2891])
        all_B = raw[ht == 'H1'].T.tocsr()
        full_B_count = np.asarray(all_B.sum(1)).ravel()[ci]
        B = sparse.diags(np.divide(1., full_B_count, out=np.zeros(len(ci)), where=full_B_count > 0))@raw[hj][:, ci].T
        B = B.tocsr(); zero = np.flatnonzero(full_B_count == 0)
        if ci[zero].tolist() != [known_gap]: raise RuntimeError('New zero B inventory requires explicit separate audit')
        master = json.loads((root/'data/atlas/GIN_69_6_HC2_BC1_SUPPORT_V2/HC_master.json').read_bytes())
        bounds = master['boundary']['canonical_bounds_mm']
        edge = np.minimum.reduce([cx[ci, 0]-bounds[0], bounds[1]-cx[ci, 0], cx[ci, 1]-bounds[2], bounds[3]-cx[ci, 1]])
        prior = json.loads((v5/'preflight_01/RESULT.json').read_bytes())
        if edge.min() <= prior['fixed_region']['conservative_max_contact_mm']:
            raise RuntimeError('Unknown exterior HC partners at domain edge')
        core_local = np.searchsorted(ci, core); target = int(np.flatnonzero(hj == 2891)[0])
        source = root/'reports/CONE_HC_OUTER_SOURCE_EXTENSION_20261102_v3/cones_01'
        source_record = json.loads((source/'RESULT.json').read_bytes())
        names = ['cone_inner_segment_mm.npy', 'cone_pedicle_mm.npy', 'official_source_rows.npy']
        for name in names:
            if sha(source/name) != source_record['files'][name]: raise RuntimeError('Cone source changed')
        isxy, ped, source_rows = (np.load(source/name) for name in names)
        old_source_rows = np.array([int(k.decode().split(':')[-1]) for k in cones.keys])
        old_to_source = np.searchsorted(source_rows, old_source_rows)
        np.testing.assert_array_equal(source_rows[old_to_source], old_source_rows)
        dyn_source = old_to_source[ci]
        np.testing.assert_array_equal(ped[dyn_source], cx[ci])
        labels = np.full(len(ped), '', dtype='U1'); labels[old_to_source] = ct
        artifact_hashes = {key.replace('\\', '/'): value for key, value in
                          json.loads((v5/'EVIDENCE_MANIFEST.json').read_bytes()).items()}
        for name in ('TARGET_A_MEMBERS.npz', 'TARGET_A_ONLY.npz'):
            path = v5/'preflight_01'/name
            if sha(path) != artifact_hashes[path.relative_to(root).as_posix()]: raise RuntimeError('v5 target asset changed')
        with np.load(v5/'preflight_01/TARGET_A_MEMBERS.npz', allow_pickle=False) as z:
            saved_source = z['cone_candidate_rows']; saved_labels = z['cone_types']
            common = labels[saved_source] != ''
            np.testing.assert_array_equal(labels[saved_source][common], saved_labels[common])
            labels[saved_source] = saved_labels
        cfg_path = root/'configurations/research/HC_biexponential_vh2007_reference_v1.json'
        cfg = json.loads(cfg_path.read_bytes()); pspace = cfg['H1']
        radius = support_radius(pspace['lambda_l_um']/1000, cfg['tail_mass_budget'])
        tree = cKDTree(ped)
        envelope = radius+np.linalg.norm(hx[hj]-hx[2891], axis=1).max()
        needed = np.asarray(tree.query_ball_point(hx[2891], envelope), int)
        unknown = needed[labels[needed] == '']
        print('Domain:', len(ci), 'cones,', len(hj), 'H1; additional source labels:', len(unknown), flush=True)
        write(out/'ADDITIONAL_LABELS.json', extend_labels(root, isxy, source_rows, unknown, labels, source_record))
        np.savez_compressed(out/'SOURCE_LABELS.npz', cone_candidate_rows=needed, cone_types=labels[needed])
        # Stream complete fine A rows; store dynamic entries and exact held column sums only.
        op_start = time.perf_counter(); nc, nh = len(ci), len(hj)
        mats = {'uniform': np.zeros((nh, nc)), 'biexponential': np.zeros((nh, nc))}
        held = {k: np.zeros(nh) for k in mats}
        dyn_map = np.full(len(ped), -1, int); dyn_map[dyn_source] = np.arange(nc)
        stimulated = np.linalg.norm(isxy-np.array(protocol['input']['center_mm']), axis=1)
        stimulated = (stimulated >= .12) & (stimulated < .32)
        target_audit = {}
        for j, parent_h in enumerate(hj):
            old_ids = raw[parent_h].indices; local_ids = old_to_source[old_ids]
            coeff = raw[parent_h].data
            if not len(local_ids) or np.any(labels[local_ids] == 'S'): raise RuntimeError('Bad H1 local input')
            dist = np.linalg.norm(ped[local_ids]-hx[parent_h], axis=1)
            short = coeff*np.exp(-(dist-dist.min())/(pspace['lambda_s_um']/1000)); short /= short.sum()
            uniform = coeff/coeff.sum()
            long_ids = np.asarray(sorted(tree.query_ball_point(hx[parent_h], radius)), int)
            if np.any(labels[long_ids] == ''): raise RuntimeError('Unlabelled parent source')
            long_ids = long_ids[labels[long_ids] != 'S']
            dist_long = np.linalg.norm(ped[long_ids]-hx[parent_h], axis=1)
            long = np.exp(-(dist_long-dist_long.min())/(pspace['lambda_l_um']/1000)); long /= long.sum()
            for mode, ids, weights in [('uniform', local_ids, uniform),
                    ('biexponential', local_ids, pspace['w_s']*short),
                    ('biexponential', long_ids, (1-pspace['w_s'])*long)]:
                inside = dyn_map[ids] >= 0
                np.add.at(mats[mode][j], dyn_map[ids[inside]], weights[inside])
                held[mode][j] += weights[~inside].sum()
            if parent_h == 2891:
                computed = np.zeros(len(ped))
                computed[local_ids] += pspace['w_s']*short
                computed[long_ids] += (1-pspace['w_s'])*long
                saved = sparse.load_npz(v5/'preflight_01/TARGET_A_ONLY.npz').toarray()[0]
                np.testing.assert_allclose(computed[saved_source], saved, rtol=2e-14, atol=1e-17)
                target_audit = dict(saved_A_max_abs_difference=float(abs(computed[saved_source]-saved).max()),
                    full_stimulated_LM_source_count=int(np.sum(stimulated & (computed > 0))),
                    held_stimulated_LM_source_count=int(np.sum(stimulated & (computed > 0) & (dyn_map < 0))),
                    held_stimulated_target_A_mass=float(computed[stimulated & (dyn_map < 0)].sum()),
                    dynamic_target_A_mass=float(computed[dyn_source].sum()))
            if (j+1) % 50 == 0: print('Exact A rows:', j+1, '/', nh, flush=True)
        matrices = {k: sparse.csr_matrix(v) for k, v in mats.items()}
        for k in matrices:
            np.testing.assert_allclose(np.asarray(matrices[k].sum(1)).ravel()+held[k], 1., rtol=0, atol=2e-14)
            sparse.save_npz(out/('A_'+k+'_dynamic.npz'), matrices[k])
        sparse.save_npz(out/'B_local_on.npz', B)
        op_seconds = time.perf_counter()-op_start
        np.savez_compressed(out/'GEOMETRY.npz', cone_parent_rows=ci, hc_parent_rows=hj,
            cone_keys=cones.keys[ci], hc_keys=hcs.keys[hj], cone_IS_mm=isxy[dyn_source],
            cone_pedicle_mm=ped[dyn_source], hc_anchor_mm=hx[hj], cone_types=ct[ci],
            core_cone_parent_rows=core, core_local=core_local, target_hc_local=target,
            unsupported_local=zero, unsupported_feedback=np.isin(np.arange(nc), zero),
            A_uniform_held_mass=held['uniform'], A_biexponential_held_mass=held['biexponential'])
        registry = [dict(cone_parent_row=int(ci[i]), key=cones.keys[ci[i]].decode(),
            status='unsupported_feedback', geometry_complete=True, B_row_sum=0.,
            approximation='retain true time-varying light and complete Cone states; zero feedback, no invented contact') for i in zero]
        domain = dict(OPL_bounds_mm=[lo[0], hi[0], lo[1], hi[1]], cones=nc, H1=nh,
            directly_stimulated_dynamic_cones=int(stimulated[dyn_source].sum()),
            fixed_region_edge_clearance_min_mm=float(edge.min()), unsupported_feedback=registry,
            target_A_audit=target_audit, H2_scope='excluded by previously verified directed causal separation',
            full_original_annulus_simulated=False, fine_identity=True)
        write(out/'DOMAIN_AND_BOUNDARY.json', domain)
        source_files = [Path(__file__), cfg_path, base/'RAW_HC_BY_CONE.npz', base/'NATIVE_ASSEMBLY_RESULT.json',
            v5/'preflight_01/TARGET_A_ONLY.npz', v5/'preflight_01/TARGET_A_MEMBERS.npz',
            root/'retina_sim/hc_minimal_diagnostic.py', root/'retina_sim/h1_h2_effective.py',
            root/'retina_sim/cone_hc_reduced_native.py', root/'retina_sim/hc_biexponential.py',
            root/'retina_sim/hc_support_closure.py', root/'retina_sim/vh2005.py']
        source_ids = {str(f.relative_to(root)): sha(f) for f in source_files}
        write(out/'SOURCE_IDENTITIES.json', source_ids)
        source_identity = canonical_hash(dict(files=source_ids, cone_rows=ci, hc_rows=hj,
            A={k: matrix_id(v) for k, v in matrices.items()}, B=matrix_id(B), protocol=protocol,
            cone_types=ct[ci], positions=ped[dyn_source]))
        np.savez_compressed(out/'HELD_MASS.npz', **held)
        light = np.full(nc, 1000.)
        spec = dict(protocol['input']); spec.pop('coordinate_frame')
        stimulus = native_td_step(isxy[dyn_source], 1000., time_seconds=.04, **spec)
        np.testing.assert_array_equal(stimulus, np.load(source/'annulus_on_td.npy')[dyn_source])
        result.update(domain=domain, source_identity=source_identity, operator_build_seconds=op_seconds,
            parameters=dict(SOURCE_VALUES), experiments={})
        outputs = {}; advance_seconds = 0.; p = SOURCE_VALUES
        for name, mode, off in [('A_uniform', 'uniform', False), ('B_biexponential', 'biexponential', False),
                                 ('C_feedback_off', 'biexponential', True)]:
            print('Background and integration:', name, flush=True)
            bgstart = time.perf_counter(); cone0 = cone_fixed_point(np.array([1000.]))
            v0 = float(cone0['Vis'][0]); ai0 = (v0/p['V_I'])**p['mu']
            if off:
                r_ext = float(expit((v0-p['V_k'])/p['V_n'])/ai0); h_ext = p['g_t']*r_ext
            else:
                h_ext = brentq(lambda h: h-p['g_t']*expit((v0-h-p['V_k'])/p['V_n'])/ai0,
                               0., p['g_t']/ai0, xtol=1e-12)
                r_ext = h_ext/p['g_t']
            binding = bind_diagnostic(matrices[mode], B, cone_parent_rows=ci,
                registered_unsupported_rows=ci[zero], A_held_mass=held[mode], B_held_mass=np.zeros(nc),
                exterior_release=r_ext, exterior_Vbar=v0, exterior_h=h_ext,
                source_identity=source_identity, feedback_off=off)
            n, pars = native_model(binding, hj, ci)
            eq = joint_equilibrium(n.A, n.B, np.arange(nh), np.array([], int), light, pars, boundary=n.boundary)
            y0 = np.concatenate([eq['cone'][k] for k in ('X', 'C', 'Vis', 'g_i', 'Vbar')]
                +[eq['w'], eq['z2'], eq['h'], np.array([])])
            anchor = AnalyticREAnchor(0., light, light, light, .02, 0)
            rhs = float(np.max(np.abs(n.rhs(y0, 0., anchor))))
            permit = require_diagnostic_native(binding, input_available=np.ones(nc, bool),
                feedback_geometry_complete=np.ones(nc, bool), background_max_rhs=rhs)
            bgseconds = time.perf_counter()-bgstart
            np.savez_compressed(out/(name+'_BACKGROUND.npz'), y0=y0, release0=eq['release'],
                light=light, stimulus=stimulus, **binding['boundary'])
            t, y, run = integrate(n, y0, light, stimulus); advance_seconds += run['wall_seconds']
            data = readout(n, t, y, target, core_local, zero); outputs[name] = data
            np.savez_compressed(out/(name+'_TRAJECTORY.npz'), **data)
            metrics = dict(identity=binding['identity'], native_source_version=n.source_version,
                permit=permit, background_max_rhs=rhs, background_solve_seconds=bgseconds,
                exterior=dict(release0=r_ext, Vbar0=v0, h0=h_ext, rule='homogeneous compatible exterior'),
                integration=run, H1_baseline=float(data['H1_target'][0]),
                H1_delta_final=float(data['H1_target'][-1]-data['H1_target'][0]),
                core_release_baseline=data['release_core'][:, 0].tolist(),
                core_release_delta_final=(data['release_core'][:, -1]-data['release_core'][:, 0]).tolist(),
                core_q0=data['q_core'][:, 0].tolist(), core_sigmoid_slope0=data['sigmoid_slope_core'][:, 0].tolist(),
                unsupported_release0=data['release_unsupported'][:, 0].tolist(),
                unsupported_release_delta_final=(data['release_unsupported'][:, -1]-data['release_unsupported'][:, 0]).tolist(),
                unsupported_q0=data['q_unsupported'][:, 0].tolist(), all_finite=True,
                sample_count=len(t), mass_ledger={k: v.tolist() for k, v in binding['mass_ledger'].items()})
            result['experiments'][name] = metrics; write(out/(name+'_RESULT.json'), metrics)
            if name == 'B_biexponential':
                print('One tightened B integration', flush=True)
                tt, yy, refined = integrate(n, y0, light, stimulus, tightened=True)
                advance_seconds += refined['wall_seconds']
                check = readout(n, tt, yy, target, core_local, zero)
                np.savez_compressed(out/'B_biexponential_REFINED.npz', **check)
                budget = protocol['numerical_budget']; gates = {}
                for key, absolute in [('H1_target', budget['H1_native_absolute_floor']),
                                       ('release_core', budget['release_native_absolute_floor']),
                                       ('release_unsupported', budget['release_native_absolute_floor'])]:
                    delta = check[key]-check[key][..., :1]
                    allowed = budget['solver_refinement_fraction_of_budget']*(absolute+budget['relative_budget']*np.max(abs(delta), axis=-1))
                    error = np.max(abs(check[key]-data[key]), axis=-1)
                    gates[key] = dict(max_abs_error=np.asarray(error).tolist(), allowed=np.asarray(allowed).tolist(),
                                      passed=bool(np.all(error <= allowed)))
                result['solver_refinement'] = dict(condition=name, integration=refined, gates=gates,
                    passed=all(g['passed'] for g in gates.values()), scope='key B outputs only; A/C not independently refined')
        differences = {}
        for label, one, two in [('biexp_minus_uniform', 'B_biexponential', 'A_uniform'),
                                ('feedback_on_minus_off', 'B_biexponential', 'C_feedback_off')]:
            data = {}
            for key in ('H1_target', 'release_core', 'pooled_release_target', 'feedback_core', 'release_unsupported'):
                a, b = outputs[one][key], outputs[two][key]
                data[key+'_absolute'] = a-b
                data[key+'_delta'] = (a-a[..., :1])-(b-b[..., :1])
            np.savez_compressed(out/(label+'.npz'), t_seconds=outputs[one]['t_seconds'], **data)
            differences[label] = {k: dict(peak_abs=float(np.max(abs(v))), final=np.asarray(v[..., -1]).tolist())
                                  for k, v in data.items()}
        result.update(status='PASS_CONDITIONAL_REAL_GRAPH_DIAGNOSTIC' if result['solver_refinement']['passed'] else 'FAIL_NUMERICAL_REFINEMENT',
            differences=differences, model_advance_wall_seconds=advance_seconds,
            physiological_qualification=False, formal_2_1_3_0_halo_test='NOT_RUN', downstream='NOT_RUN')
    except Exception as error:
        result.update(status='FAILED_RETAINED', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(ended_utc=datetime.now(timezone.utc).isoformat(), total_wall_seconds=time.perf_counter()-clock,
                      peak_process_working_set_bytes=peak_memory_bytes())
        write(out/'RESULT.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'total_wall_seconds', 'model_advance_wall_seconds',
                                            'peak_process_working_set_bytes')}, indent=2), flush=True)


if __name__ == '__main__': main()
