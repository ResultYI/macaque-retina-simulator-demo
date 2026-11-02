"""Rebuild this handout's finite HC disk with the selected density and dynamics."""
from pathlib import Path
import argparse
import json
import sys
import time
import traceback
import zipfile

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim import hc_mosaic
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
from retina_sim.contracts import canonical_hash
from retina_sim.disk_graph_storage import file_sha
from retina_sim.disk_hc_functional import eccentric_projection
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.geometry import RetinalDomain
from retina_sim.hc_current import current_hc
from retina_sim.hc_group_projection import project_hc_contacts
from retina_sim.hc_joint_v2 import JointV2HCDensityProfile
from retina_sim.h1_h2_effective import joint_equilibrium, matrix_id
from retina_sim.spatial_group_model import _verify_artifact
from retina_sim.vh2005 import SOURCE_VALUES
from tools.durable_research_runtime_v1 import DurableRun, atomic_json, utc, inspect_run
from tools.run_fullfield_hc_preview import make_assembly, SparseFullFieldNative, member_mean
from hc_generation_acceleration import screened_point_energy

OUT = ROOT / 'reports/HC_JOINT_V2_FIRST_FRAME_0_6DEG_20261101_v1'
VIDEO = ROOT / 'reports/cone_video_113_4s_eccentric_aperture_0_6deg_20261029_v1'
PARENT = ROOT / 'reports/spatial_group_graph_v02_20261025_v1/artifact_r2'
CANDIDATE = ROOT / 'reports/retinotopic_rf_revision/native_validation_v1/CANDIDATE_MODEL.json'
RADIUS = 1.338
MM_PER_DEGREE = .223
GUARD = lambda: None
VISUAL_SUPPORT_REPAIR = False
ORIGINAL_RUN = OUT


def select_visual_support_repair():
    global OUT, RADIUS, VISUAL_SUPPORT_REPAIR
    VISUAL_SUPPORT_REPAIR = True
    OUT = ROOT / 'reports/HC_VISUAL6DEG_SUPPORT_REPAIR_20261101_v1'
    RADIUS = 1.75


class DiskProfile(JointV2HCDensityProfile):
    """Bounds over the declared circular tissue; no out-of-domain field query."""
    def __init__(self, root):
        super().__init__(root)
        if VISUAL_SUPPORT_REPAIR:
            original_support = self.support_radius_mm
            self.support_radius_mm = RADIUS
            self.metadata.update(radial_support_mm=[0., RADIUS],
                original_application_support_mm=original_support,
                extension='Explicit local support-halo continuation of the same density, subtype split and contact equations; engineering approximation',
                observation_visual_radius_deg=6.)
            self.profile_hash = canonical_hash(dict(base_profile_hash=self.profile_hash,
                support_radius_mm=RADIUS, extension=self.metadata['extension']))

    def bounds_for_rectangle(self, bounds):
        lo, hi = hc_mosaic._radial_rectangle_range(bounds)
        # Only rectangle intersection with the disk contains physical tissue.
        lo, hi = min(lo, RADIUS), min(hi, RADIUS)
        values = self.radial([lo, hi] + ([.6] if lo <= .6 <= hi else []))
        return dict(total_min=float(values.min()), total_max=float(values.max()),
                    H1_min=float(values.min()*.8), H1_max=float(values.max()*.8),
                    H2_min=float(values.min()*.2), H2_max=float(values.max()*.2))

    def type_density(self, xy, cell_type):
        GUARD()
        return super().type_density(xy, cell_type)


class DiskQuadrature(hc_mosaic._Quadrature):
    """Integrate rectangle/disk intersections using curved integration limits."""
    def rule(self, bounds, order):
        GUARD()
        x0, x1, y0, y1 = bounds
        x0, x1 = max(x0, -RADIUS), min(x1, RADIUS)
        if x0 >= x1:
            return np.zeros(2)
        breaks = [x0, x1]
        for y in (y0, y1):
            if abs(y) < RADIUS:
                edge = np.sqrt(RADIUS**2-y*y)
                breaks.extend(x for x in (-edge, edge) if x0 < x < x1)
        nodes, weights = self.rules[order]
        answer = np.zeros(2)
        for a, b in zip(sorted(set(breaks))[:-1], sorted(set(breaks))[1:]):
            x = (a+b)/2 + nodes*(b-a)/2
            edge = np.sqrt(np.maximum(0., RADIUS**2-x*x))
            low, high = np.maximum(y0, -edge), np.minimum(y1, edge)
            keep = high > low
            if not keep.any():
                continue
            x, low, high, wx = x[keep], low[keep], high[keep], weights[keep]*(b-a)/2
            y = (low[:, None]+high[:, None])/2 + nodes*(high-low)[:, None]/2
            points = np.c_[np.broadcast_to(x[:, None], y.shape).ravel(), y.ravel()]
            w = (wx[:, None]*weights*(high-low)[:, None]/2).ravel()
            self.evaluated_points += len(points)
            if self.evaluated_points > self.max_points:
                raise RuntimeError('Declared circular quadrature budget exceeded')
            if not self.domain.contains(points).all():
                raise RuntimeError('Circular quadrature escaped the declared tissue')
            answer += [np.dot(w, self.profile.type_density(points, t)) for t in ('H1', 'H2')]
        return answer

    def integrate(self, bounds, depth=0, absolute_tolerance=None):
        if tuple(bounds) == (-RADIUS, RADIUS, -RADIUS, RADIUS):
            exact = self.profile.expected_disk_count(RADIUS)*np.array([.8, .2])
            return exact, np.full(2, 1e-8)
        return super().integrate(bounds, depth, absolute_tolerance)


def session(stage):
    journal = DurableRun(OUT/(stage+'_runtime'), 'current_HC_'+stage)
    memory = MemoryGuard(24*1024**3, 8*1024**3)
    start, last = time.perf_counter(), -1e9
    def guard():
        nonlocal last
        now = time.perf_counter()
        if now-last > 5:
            if now-start > 3600:
                raise TimeoutError('Declared one-hour stage bound exceeded')
            if not memory.check():
                raise MemoryError(str(memory.last))
            journal.heartbeat(stage=stage, elapsed_seconds=now-start, memory=memory.last)
            last = now
    return journal, memory, start, guard


def load_cones():
    candidate = json.loads(CANDIDATE.read_bytes())
    cones = FrozenDiskTable(ROOT, candidate['fine_source_model']['tables']['cones'])
    parent, manifest, parent_id = _verify_artifact(PARENT)
    meta = manifest['populations']['cones']
    assert meta['source_table'] == cones.binding
    def arr(key):
        return np.load(parent/f'populations/cones/{key}.npy', allow_pickle=False)
    mapping, counts, keys = arr('fine_to_group'), arr('counts'), arr('keys')
    types = np.asarray(meta['subtype_labels'])[keys[:, meta['key_columns'].index('subtype')]]
    assert np.array_equal(np.bincount(mapping[mapping >= 0], minlength=len(counts)), counts)
    return cones, mapping, counts, types, parent_id


def build(resume_geometry=False):
    global GUARD
    stage = 'geometry'
    if OUT.exists():
        if not resume_geometry or (OUT/'HC_GEOMETRY.npz').exists():
            raise FileExistsError('Existing v2 run retained; inspect its receipts before continuation')
        previous = sorted(OUT.glob('geometry*_runtime'))
        ownership = [inspect_run(p) for p in previous]
        assert all(o['decision'] == 'TERMINAL_RECORDED' for o in ownership)
        assert not any(p['state'] in ('ALIVE','UNKNOWN') for o in ownership for p in o['observations'])
        attempt = len(previous)+1
        assert (OUT/'GEOMETRY_FAILURE.json').exists()
        for name in ('GEOMETRY_FAILURE.json','GEOMETRY_CONTRACT.json'):
            (OUT/(Path(name).stem+f'_BEFORE_{attempt}.json')).write_bytes((OUT/name).read_bytes())
        atomic_json(OUT/f'GEOMETRY_RECOVERY_{attempt}.json', dict(ownership=ownership,
            reason='Preserve density, integration tolerance and pair mechanics; use the recorded depth budget and verified zero-energy candidate screening'))
        stage = f'geometry_recovery_{attempt}'
    else:
        OUT.mkdir()
    archive = HERE/'before_current_density_v2.zip'
    if not archive.exists():
        names = ['main.tex', 'main.pdf', 'QA.json', 'README_ZH.md']
        names += [p.name for p in HERE.glob('hc_distribution_density.*')]
        names += [p.name for p in HERE.glob('hc_fullfield_response_preview.*')]
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for name in names:
                z.write(HERE/name, name)
    journal, memory, started, GUARD = session(stage)
    try:
        selection = current_hc(ROOT)
        profile = DiskProfile(ROOT)
        assert profile.metadata['current_hc_identity'] == selection['identity']
        bounds = (-RADIUS, RADIUS, -RADIUS, RADIUS)
        domain = RetinalDomain(bounds, bounds,
            lambda xy: np.linalg.norm(xy, axis=1) <= RADIUS,
            'HC_v2_explicit_finite_disk_1p338mm', 'HC_v2_disk_curved_quadrature_v1')
        options = json.loads((ROOT/'configurations/hc_joint_v2_generator.json').read_bytes())['generator']
        options['max_depth'] = 36
        atomic_json(OUT/'GEOMETRY_CONTRACT.json', dict(started_utc=utc(),
            current_hc_identity=selection['identity'], selected_configuration=selection,
            implementation_sha256=file_sha(__file__), radius_mm=RADIUS, seed=20261010,
            generator_options=options, domain_adapter='Curved disk quadrature and disk-intersection bounds; original pair energy after conservative zero-contribution screening',
            energy_screening_sha256=file_sha(HERE/'hc_generation_acceleration.py'),
            finite_open_boundary=True, external_HC_created=VISUAL_SUPPORT_REPAIR,
            profile_metadata=profile.metadata, physiological_validation=False))
        original = hc_mosaic._Quadrature
        original_energy = hc_mosaic._point_energy
        hc_mosaic._Quadrature = DiskQuadrature
        hc_mosaic._point_energy = screened_point_energy
        try:
            print('Generating current joint HC mosaic in disk radius', RADIUS, 'mm', flush=True)
            mosaic = hc_mosaic.generate_joint_hc(profile, domain, 20261010, 600000, options=options)
        finally:
            hc_mosaic._Quadrature = original
            hc_mosaic._point_energy = original_energy
        xy, ht = mosaic.positions.values, mosaic.types
        assert np.all(np.linalg.norm(xy, axis=1) <= RADIUS)
        n_by_type = {t: int(np.sum(ht == t)) for t in ('H1', 'H2')}
        expected = profile.expected_disk_count(RADIUS)*np.array([.8, .2])
        assert np.max(np.abs(np.array(list(n_by_type.values()))-expected)) < 1.1
        radii = np.array([float(profile.contact_radius_mm(r, t))
                          for r, t in zip(np.linalg.norm(xy, axis=1), ht)])
        np.savez_compressed(OUT/'HC_GEOMETRY.npz', xy_mm=xy, subtype=ht,
                            cell_ids=np.asarray(mosaic.cell_ids), radius_mm=radii,
                            current_hc_identity=selection['identity'], mm_per_degree=MM_PER_DEGREE)
        atomic_json(OUT/'MOSAIC_DIAGNOSTICS.json', mosaic.diagnostics)
        print('New HC counts:', n_by_type, flush=True)
        finish_connections(xy, ht, radii, n_by_type, expected, selection, memory, started, journal)
    except BaseException as e:
        atomic_json(OUT/'GEOMETRY_FAILURE.json', dict(error=repr(e), traceback=traceback.format_exc()))
        journal.finish('FAILED', error=repr(e))
        raise


def finish_connections(xy, ht, radii, n_by_type, expected, selection, memory, started, journal):
    cones, cm, cc, ct, parent_id = load_cones()
    cp = cones.coordinates['opl_pedicle']
    safe = np.linalg.norm(cp, axis=1)+radii.max() <= RADIUS
    print('Building contacts to all', cones.count, 'existing canonical cone pedicles', flush=True)
    raw, audit = eccentric_projection(cp, cones.extras['subtype'].astype('U1'), xy, ht,
                                      safe, RADIUS, radii)
    raw = raw.T.tocsr()
    sparse.save_npz(OUT/'RAW_HC_BY_CONE.npz', raw)
    hm, hc = np.arange(len(ht)), np.ones(len(ht), dtype=np.int64)
    projected = project_hc_contacts(raw, cones.extras['subtype'].astype('U1'), ht, cm, hm, cc, hc)
    sparse.save_npz(OUT/'A.npz', projected['A'])
    sparse.save_npz(OUT/'B.npz', projected['typed_B'])
    mass = np.asarray(raw.sum(1)).ravel()
    centers = np.divide(raw@cones.coordinates['rf'], mass[:, None],
        out=np.full_like(xy, np.nan), where=mass[:, None] > 0)
    np.savez_compressed(OUT/'DISPLAY_GEOMETRY.npz', soma_xy_mm=xy, input_center_mm=centers,
        subtype=ht, cone_counts=cc, cone_types=ct, cone_map=cm,
        HC_supported=mass > 0, cone_complete_HC_neighborhood=safe)
    degrees = {t: dict(HC_count=n_by_type[t], zero_input=int(np.sum((ht == t)&(mass == 0))),
        mean_convergence=float(mass[ht == t].mean()),
        diameter_um_min_max=(2e3*radii[ht == t]).min().item()) for t in ('H1', 'H2')}
    for t in degrees:
        degrees[t]['diameter_um_min_max'] = [float(2e3*radii[ht == t].min()), float(2e3*radii[ht == t].max())]
    assert current_hc(ROOT)['identity'] == selection['identity']
    result = dict(status='GEOMETRY_READY', started_utc=json.loads((OUT/'GEOMETRY_CONTRACT.json').read_bytes())['started_utc'], ended_utc=utc(),
        current_hc_identity=selection['identity'], counts=n_by_type,
        expected_counts=expected.tolist(), finite_HC_radius_mm=RADIUS,
        cones=cones.count, cone_groups=len(cc), HC_groups=len(ht), contacts=raw.nnz,
        typed_statistics=degrees, parent_cone_artifact=parent_id, cone_table=cones.binding,
        A=matrix_id(projected['A']), B=matrix_id(projected['typed_B']),
        geometry_sha256=file_sha(OUT/'HC_GEOMETRY.npz'),
        wall_seconds=time.perf_counter()-started, peak_rss_bytes=memory.peak_rss_bytes,
        original_HC_geometry_reused=False, no_parameter_refit=True,
        complete_HC_neighborhood_cones=int(safe.sum()), physiological_validation=False)
    atomic_json(OUT/'BUILD_RESULT.json', result)
    journal.finish('GEOMETRY_READY', result='BUILD_RESULT.json')
    print(json.dumps(result), flush=True)


def connect_existing():
    global GUARD
    accepted = json.loads((OUT/'MOSAIC_ACCEPTANCE.json').read_bytes())
    assert accepted['status'] == 'COMPLETE_GEOMETRY_CHECKPOINT_ACCEPTED'
    assert accepted['resource_PASS']
    assert accepted['current_hc_identity'] == current_hc(ROOT)['identity']
    assert accepted['geometry_sha256'] == file_sha(OUT/'HC_GEOMETRY.npz')
    if (OUT/'BUILD_RESULT.json').exists() or (OUT/'connection_runtime').exists():
        raise FileExistsError('Existing connection stage retained')
    journal, memory, started, GUARD = session('connection')
    try:
        with np.load(OUT/'HC_GEOMETRY.npz',allow_pickle=False) as d:
            xy, ht, radii = [d[k] for k in ('xy_mm','subtype','radius_mm')]
        profile = DiskProfile(ROOT)
        expected = profile.expected_disk_count(RADIUS)*np.array([.8,.2])
        finish_connections(xy, ht, radii, accepted['counts'], expected, current_hc(ROOT), memory, started, journal)
    except BaseException as e:
        atomic_json(OUT/'CONNECTION_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()))
        journal.finish('FAILED',error=repr(e))
        raise


def respond():
    global GUARD
    build = json.loads((OUT/'BUILD_RESULT.json').read_bytes())
    assert build['status'] == 'GEOMETRY_READY'
    assert build['current_hc_identity'] == current_hc(ROOT)['identity']
    if (OUT/'response_runtime').exists():
        raise FileExistsError('Existing response ownership retained; no duplicate run')
    journal, memory, started, GUARD = session('response')
    try:
        d = np.load(OUT/'DISPLAY_GEOMETRY.npz', allow_pickle=False)
        active = d['HC_supported']
        A = sparse.load_npz(OUT/'A.npz')[active].tocsr()
        B = sparse.load_npz(OUT/'B.npz')[:, active].tocsr()
        cc, ct, cm, ht = [d[k] for k in ('cone_counts', 'cone_types', 'cone_map', 'subtype')]
        cones, check_map, _, _, parent_id = load_cones()
        assert np.array_equal(check_map, cm)
        with np.load(VIDEO/'geometry.npz', allow_pickle=False) as g:
            rows, source_types = g['source_table_rows'], g['types']
            assert np.array_equal(g['xy_mm'], cones.coordinates['rf'][rows])
            assert np.array_equal(source_types, cones.extras['subtype'][rows].astype('U1'))
        capture = loadmat(VIDEO/'first_frame/capture.mat')
        completion = json.loads((VIDEO/'first_frame/capture_complete.json').read_bytes())
        assert json.loads(capture['manifest_json'].item())['request_signature'] == completion['request_signature']
        bridge_path = ROOT/'reports/external_video_two_positions_4s_20261029_v1/far/BINDING.json'
        bridge = json.loads(bridge_path.read_bytes())['conversion']
        alpha = np.array([bridge['alpha_rstar_s_per_reference_td_LMS'][t] for t in source_types])
        duration = float(np.diff(capture['window_edges_seconds'].ravel())[0])
        assert duration == 1/30 and np.all(cm[rows] >= 0)
        background = member_mean(capture['background_counts'].ravel()/duration/alpha, cm[rows], cc)
        light = member_mean(capture['expected_counts'].ravel()/duration/alpha, cm[rows], cc)
        source_names = ('retina_sim/vh2005.py','retina_sim/h1_h2_effective.py','retina_sim/hc_lateral_candidate.py',
                        'retina_sim/hc_joint_v2.py','retina_sim/hc_current.py')
        hashes = {n: file_sha(ROOT/n) for n in source_names}
        identity = dict(current_hc_identity=build['current_hc_identity'], build_sha256=file_sha(OUT/'BUILD_RESULT.json'),
            parent_cone_artifact=parent_id, A=matrix_id(A), B=matrix_id(B), source_hashes=hashes)
        assembly = make_assembly(A, B, cc, np.ones(active.sum(), dtype=np.int64), ct, ht[active], identity)
        native = SparseFullFieldNative(assembly)
        native.boundary = dict(A_release=np.zeros(native.nh), A_Vbar=np.zeros(native.nh), B_h=np.zeros(native.nc))
        contract = dict(started_utc=utc(), identity=identity,
            geometry=dict(fine_cones=cones.count, cone_groups=native.nc, fine_HC=len(ht), active_HC=native.nh),
            stimulus_duration_seconds=duration, captured_cones=len(rows),
            HC_boundary=f'Finite open anatomical disk, soma radius<={RADIUS}mm; observation visual radius6deg',
            cone_representation='Existing complete cone-group mapping retained; all new HC have independent states',
            physiological_validation=False, FIT_READY=False, production_adopted=False)
        atomic_json(OUT/'CONTRACT.json', contract)
        print('Solving new background:', contract['geometry'], flush=True)
        eq = joint_equilibrium(A, B, native.h1_rows, np.array([], dtype=int), background,
                              assembly.h2_parameters, guard=GUARD)
        y = np.concatenate([eq['cone'][n] for n in ('X','C','Vis','g_i','Vbar')]
                           + [eq['w'], eq['z2'], eq['h'], np.array([], dtype=float)])
        baseline = AnalyticREAnchor(0., background, background, background, duration, 0)
        y, receipt = native.equilibrate_native(y, baseline, 0.)
        assert receipt['maximum_full_native_rhs'] <= 1e-7
        np.savez_compressed(OUT/'BACKGROUND.npz', y=y, background=background, light=light)
        atomic_json(OUT/'BACKGROUND.json', dict(receipt, direct_residual=eq['residual']))
        print('Background RHS:', receipt['maximum_full_native_rhs'], flush=True)
        stimulus = AnalyticREAnchor(0., background, background, light, duration, 1)
        times = np.r_[np.arange(0., .033, .001), duration]
        results, receipts = {}, {}
        for label, rtol, atol, step in [('standard',1e-6,1e-9,1e-4),('tight',1e-7,1e-10,5e-5)]:
            begin = time.perf_counter()
            def rhs(t, state):
                GUARD()
                return native.rhs(state, t, stimulus)
            print('Integrating all current finite HC:', label, flush=True)
            sol = solve_ivp(rhs, (0., duration), y, method='DOP853', rtol=rtol, atol=atol, max_step=step, t_eval=times)
            assert sol.success and np.isfinite(sol.y).all() and np.array_equal(sol.t, times)
            results[label] = sol.y
            np.savez_compressed(OUT/(label.upper()+'.npz'), time_seconds=sol.t,
                HC=sol.y[native.slices['Vh']], HC_delta=sol.y[native.slices['Vh']]-y[native.slices['Vh'],None])
            receipts[label] = dict(nfev=sol.nfev, wall_seconds=time.perf_counter()-begin,
                response_sha256=file_sha(OUT/(label.upper()+'.npz')))
            atomic_json(OUT/(label.upper()+'_RECEIPT.json'), receipts[label])
            print(label, receipts[label], flush=True)
        error = abs(results['standard']-results['tight'])
        ratio = float(np.max(error/(1e-9+1e-6*np.maximum(abs(results['standard']),abs(results['tight'])))))
        hc_error = float(np.max(error[native.slices['Vh']]))
        assert ratio <= 1. and hc_error <= 1e-6
        delta = np.full(len(ht), np.nan)
        delta[active] = results['standard'][native.slices['Vh'],-1]-y[native.slices['Vh']]
        np.savez_compressed(OUT/'HC_RESPONSE.npz', delta_h=delta, supported=active, subtype=ht,
            input_center_mm=d['input_center_mm'], soma_xy_mm=d['soma_xy_mm'], duration_seconds=duration)
        assert current_hc(ROOT)['identity'] == build['current_hc_identity']
        assert all(file_sha(ROOT/n) == h for n, h in hashes.items())
        result = dict(status='COMPLETE_VALIDATED', accepted=True, started_utc=contract['started_utc'], ended_utc=utc(),
            current_hc_identity=build['current_hc_identity'], geometry=contract['geometry'],
            numerical_maximum_all_state_budget_ratio=ratio, numerical_maximum_HC_difference=hc_error,
            background_rhs=receipt['maximum_full_native_rhs'], receipts=receipts,
            wall_seconds=time.perf_counter()-started, peak_rss_bytes=memory.peak_rss_bytes,
            physiological_validation=False, FIT_READY=False, production_adopted=False, complete_4s_movie=False)
        atomic_json(OUT/'RESULT.json', result)
        journal.finish('COMPLETE_VALIDATED', result='RESULT.json')
        print(json.dumps(result), flush=True)
    except BaseException as e:
        atomic_json(OUT/'RESPONSE_FAILURE.json', dict(error=repr(e), traceback=traceback.format_exc()))
        journal.finish('FAILED', error=repr(e))
        raise


def audit_visual_coverage():
    cones, _, _, _, _ = load_cones()
    raw = sparse.load_npz(OUT/'RAW_HC_BY_CONE.npz').tocsr()
    with np.load(VIDEO/'geometry.npz', allow_pickle=False) as g:
        rows = g['source_table_rows']
    with np.load(OUT/'HC_GEOMETRY.npz', allow_pickle=False) as g:
        radii = g['radius_mm']
    distance = np.linalg.norm(cones.coordinates['opl_pedicle'][rows], axis=1)
    complete = distance + radii.max() <= RADIUS
    if not complete.all():
        raise RuntimeError('Captured Cone contact neighborhoods exceed generated HC tissue')
    contacted = np.asarray(raw[:, rows].sum(axis=0)).ravel() > 0
    eccentricity = np.linalg.norm(cones.coordinates['rf'][rows], axis=1)/.223
    old = sparse.load_npz(ORIGINAL_RUN/'RAW_HC_BY_CONE.npz')
    old_contacted = np.asarray(old[:, rows].sum(axis=0)).ravel() > 0
    bands = []
    for a, b in [(0,4),(4,5),(5,5.25),(5.25,5.5),(5.5,5.75),(5.75,6.00000001)]:
        mask = (eccentricity >= a) & (eccentricity < b)
        n = int(mask.sum())
        hit = int(np.sum(contacted & mask))
        bands.append(dict(lower_deg=a, upper_deg=min(b,6), captured_cones=n,
            contacted=hit, fraction=hit/n, prior_contacted=int(np.sum(old_contacted & mask))))
    outer = eccentricity >= 5.5
    if contacted[outer].mean() < .98:
        raise RuntimeError('Outer visual annulus failed the predefined98percent contact coverage gate')
    result = dict(status='PASS', captured_cones=len(rows), captured_contacted=int(contacted.sum()),
        complete_contact_neighborhoods=int(complete.sum()), anatomical_HC_radius_mm=RADIUS,
        maximum_captured_pedicle_radius_mm=float(distance.max()),
        maximum_HC_contact_radius_mm=float(radii.max()), minimum_contact_margin_mm=float((RADIUS-distance-radii.max()).min()),
        annuli=bands, all_captured_geometry_complete=True, outer_contact_fraction=float(contacted[outer].mean()),
        no_artificial_partners=True, no_spatial_rescaling=True)
    atomic_json(OUT/'VISUAL_COVERAGE.json', result)
    print(json.dumps(result), flush=True)


def boundary_check():
    global OUT, RADIUS
    parent = OUT
    large = json.loads((parent/'RESULT.json').read_bytes())
    assert large['status'] == 'COMPLETE_VALIDATED'
    check = parent/'boundary_check_1p65mm'
    if check.exists():
        raise FileExistsError('Existing boundary check retained')
    with np.load(parent/'DISPLAY_GEOMETRY.npz', allow_pickle=False) as z:
        d = {k:z[k] for k in z.files}
    mask = np.linalg.norm(d['soma_xy_mm'],axis=1) <= 1.65
    raw = sparse.load_npz(parent/'RAW_HC_BY_CONE.npz')[mask].tocsr()
    cones, cm, cc, ct, _ = load_cones()
    ht = d['subtype'][mask]
    projected = project_hc_contacts(raw, cones.extras['subtype'].astype('U1'), ht,
        cm, np.arange(mask.sum()), cc, np.ones(mask.sum(),dtype=np.int64))
    check.mkdir()
    for name, value in [('A',projected['A']),('B',projected['typed_B'])]:
        sparse.save_npz(check/(name+'.npz'), value)
    for name in ('soma_xy_mm','input_center_mm','subtype','HC_supported'):
        d[name] = d[name][mask]
    np.savez_compressed(check/'DISPLAY_GEOMETRY.npz', **d)
    atomic_json(check/'BUILD_RESULT.json', dict(status='GEOMETRY_READY',
        current_hc_identity=large['current_hc_identity'], parent_build_sha256=file_sha(parent/'BUILD_RESULT.json'),
        parent_HC_rows=np.flatnonzero(mask).tolist(), anatomical_HC_radius_mm=1.65))
    OUT, RADIUS = check, 1.65
    try:
        respond()
    finally:
        OUT, RADIUS = parent, 1.75
    with np.load(parent/'HC_RESPONSE.npz') as a, np.load(check/'HC_RESPONSE.npz') as b:
        view = np.linalg.norm(a['input_center_mm'][mask],axis=1) <= 6*.223
        selected = view & a['supported'][mask] & b['supported']
        error = abs(a['delta_h'][mask][selected]-b['delta_h'][selected])
        maximum = float(error.max())
    result = dict(status='PASS' if maximum <= 1e-3 else 'FAIL',
        radius_mm=[1.65,1.75], shared_HC_geometry=True, compared_HC=int(selected.sum()),
        maximum_observation_HC_difference=maximum, tolerance_native_units=1e-3,
        background_recomputed_for_each_domain=True)
    atomic_json(parent/'BOUNDARY_CONVERGENCE.json', result)
    if result['status'] != 'PASS':
        raise RuntimeError('Observation HC responses depend on the finite support boundary')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('build','connect','respond','audit','boundary'))
    parser.add_argument('--visual-support', action='store_true')
    parser.add_argument('--resume-geometry', action='store_true')
    args = parser.parse_args()
    if args.visual_support:
        select_visual_support_repair()
    if args.stage == 'build': build(args.resume_geometry)
    elif args.stage == 'connect': connect_existing()
    elif args.stage == 'audit': audit_visual_coverage()
    elif args.stage == 'boundary': boundary_check()
    else: respond()
