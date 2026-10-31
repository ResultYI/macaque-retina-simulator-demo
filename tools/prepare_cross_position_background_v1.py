"""One constant optical background and independent local initialization."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.cross_position_context_v1 import patch, typed, read, write
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.spatial_group_graph import digest
from retina_sim.group_time_storage import save_bundle
from retina_sim.cone_aperture_eccentricity import resolve_apertures
from retina_sim.isetbio_input import REFERENCE_SCHEMA, reference_equivalent_drive
from retina_sim.background_checkpoint import MemoryGuard

REF = ROOT / 'reports/iset_disk_input_cache/79384a1477b0b3c11fdce71169c4cda04ed23c0340d5576ef3469039de34f6d7/reference/capture.mat'


def optical_setup(prepared, out, scene_anchor=None):
    base, local = patch(prepared)
    table = ROOT / base.manifest['populations']['cones']['source_table']['path']
    group = base.root / 'populations/cones'
    offsets = np.load(group / 'member_offsets.npy')
    members = np.load(group / 'member_global_rows.npy')
    rows = local.parent_population_rows['cones']
    fine = np.concatenate([members[offsets[r]:offsets[r + 1]] for r in rows])
    xy = np.load(table.parent / 'rf.npy')[fine]
    types = np.load(table.parent / 'subtype.npy')[fine].astype('U1')
    ids = tuple('actual_fine_cone_row_' + str(int(i)) for i in fine)
    profile = read(ROOT / 'configurations/research/macaque_cone_aperture_eccentricity_v1.json')
    apertures = resolve_apertures(ids, types, xy, profile)
    savemat(out / 'geometry.mat', dict(is_mm=xy, types=np.array([{'L': 1, 'M': 2, 'S': 3}[t] for t in types]),
        reference_diameter_um=apertures['reference_diameter_um'], zone_diameter_um=apertures['zone_diameter_um']), do_compression=True)
    np.savez_compressed(out / 'APERTURE_GEOMETRY.npz', **apertures, fine_rows=fine,
                        local_group_counts=local.populations['cones'].counts)
    anchor=read(scene_anchor) if scene_anchor else None
    reference=Path(anchor['reference_capture_path']) if anchor else REF
    if anchor and digest(reference)!=anchor['source_reference_sha256']:
        raise ContractError('Frozen S0 reference capture differs')
    ref = loadmat(reference, variable_names=['manifest_json', 'q_ref_LMS'])
    meta = json.loads(ref['manifest_json'].item())
    center = (xy.min(0) + xy.max(0)) / 2
    # A single bounded optical field, actual support plus existing aperture margin.
    samples = int(np.ceil(np.max(xy.max(0) - xy.min(0)) * 1000 + 100))
    samples += (samples % 2 == 0)
    if samples > 1024:
        raise ContractError('Single local optical field would exceed existing sample bound')
    p = deepcopy(anchor['profile']) if anchor else read(ROOT / 'configurations/iset_lcd_apple_mw_v1.json')
    p.update(isetcam_root=str(ROOT / 'vendor_pinned/isetcam'), isetbio_root=str(ROOT / 'vendor_pinned/isetbio'))
    f = .223e-3 / np.tan(np.pi / 180)
    config = dict(profile=p, geometry_mat=str(out / 'geometry.mat'), center_mm=center,
        samples=samples, patch_fov_deg=float(np.degrees(2 * np.arctan(samples * .001e-3 / (2 * f)))),
        fixed_radiance_gain=meta['fixed_radiance_gain'], reference_rgb=meta['reference_rgb'],
        reference_energy_spectrum_hex=meta['reference_energy_spectrum_hex'], reference_capture_path=str(reference),
        q_ref_LMS=ref['q_ref_LMS'].reshape(-1).tolist(), aperture_profile=canonical_hash(profile),
        source_reference_sha256=digest(reference), new_position_dynamic_normalization=False,
        input_scope='Existing fixed global neutral reference; new local aperture capture. Layer probes are relative to this local background; no matched physical eccentricity calibration claimed.')
    if anchor:
        keys=('fixed_radiance_gain','reference_rgb','reference_energy_spectrum_hex','q_ref_LMS','aperture_profile','source_reference_sha256')
        if any(canonical_hash(config[key])!=canonical_hash(anchor[key]) for key in keys):
            raise ContractError('Requested position would change a frozen S0 physical/bridge/aperture parameter')
        config['scene_anchor']=dict(path=str(Path(scene_anchor).relative_to(ROOT)),sha256=digest(scene_anchor))
        config['input_scope']='Frozen existing near-center S0, independently sampled at requested position; no paired result or global equilibrium claim'
    write(out / 'OPTICAL_CONFIG.json', config)


def initialize(prepared, out, guard, source=None):
    from retina_sim.h1_h2_background import initialize_joint_background
    from retina_sim.hc_lateral_candidate import LateralHCCandidate
    from retina_sim.cone_aperture_hc_bridge import prepare_aperture_background
    from retina_sim.h1_h2_effective import save_artifact
    source = Path(source or out)
    raw = loadmat(source / 'capture.mat', variable_names=['rates_rstar_s', 'manifest_json'])
    optical = read(source / 'OPTICAL_CONFIG.json')
    with np.load(source / 'APERTURE_GEOMETRY.npz', allow_pickle=False) as z:
        types, counts = z['cone_types'].astype('U1'), z['local_group_counts']
    scope = canonical_hash(optical)
    bridge = dict(schema=REFERENCE_SCHEMA, profile_id='cross_position_fixed_global_LMS_v1',
        source_id='existing_global_LCD_MW_reference', reference_conditions='Frozen existing global foveal neutral reference, one fixed LMS triplet and gain; no local refit',
        scope='engineering_fixture', evidence_tag='project_approximation', capture_scope_sha256=scope,
        reference_scene_sha256=digest(Path(optical['reference_capture_path']).with_name('input.mat')),
        reference_capture_sha256=optical['source_reference_sha256'],
        reference_luminance_cd_m2=100 / (np.pi * 1.5**2), pupil_diameter_mm=3.,
        reference_rates_rstar_s_LMS=dict(zip(('L', 'M', 'S'), optical['q_ref_LMS'])))
    drive, bridge_result = reference_equivalent_drive(raw['rates_rstar_s'].reshape(1, -1), types, bridge, capture_scope_sha256=scope)
    cuts = np.r_[0, np.cumsum(counts)]
    bg = np.array([np.mean(drive[0, cuts[i]:cuts[i + 1]]) for i in range(len(counts))])
    write(out / 'FIXED_LMS_BRIDGE.json', dict(profile=bridge, result=bridge_result,
        capture_file_sha256=digest(source / 'capture.mat'), background_range=[float(bg.min()), float(bg.max())],
        local_group_mean_rule='Actual fine members, arithmetic mean once; no per-position normalization'))
    if source != out:
        with np.load(source / 'NEAR_OPTICAL_BACKGROUND.npz', allow_pickle=False) as saved:
            if not np.array_equal(bg, saved['native_background']):
                raise ContractError('Previously completed optical background differs')
        write(out / 'OPTICAL_REUSE.json', dict(source=str(source.relative_to(ROOT)),
            capture_sha256=digest(source / 'capture.mat'), optical_run_repeated=False,
            background_bitwise_equal=True))
    guard()
    h, old, labels = typed(prepared, guard=guard)
    outer = h.seed(old, guard=guard)
    protocol = dict(schema='cross-position-held-background-initialization-1', clock_origin_seconds=0.,
        input_origin_seconds=0., stop_seconds=.006, BC_RGC_period_seconds=.001, rod_step_seconds=.0001,
        initial_time_seconds=0., recording_step_seconds=.001)
    joint, binding = initialize_joint_background(h, outer, ROOT, protocol, guard=guard)
    native = LateralHCCandidate(h, coupling_rate_per_second=50., S_efficacy_ratio=1.)
    cfg = deepcopy(read(ROOT / 'configurations/research/HC_homotypic_lateral_only_v2.json'))
    cfg.pop('candidate_sha256')
    cfg.update(parent_initialized_model=joint['joint_background_initialization']['initialized_model_id'],
        parent_binding=binding.identity, selected_native_execution=native.source_version,
        binding_scope='Same frozen HC parameters; independent requested-position geometry and initialization; not the historical4.73 identity')
    cfg['candidate_sha256'] = canonical_hash(cfg)
    a, seed, interface, y, equilibrium = prepare_aperture_background(h, joint, cfg, bg, ROOT, protocol,
        input_identity=dict(mode='new', optical_capture_id=digest(source / 'capture.mat'),
            aperture_profile=optical['aperture_profile'], bridge_profile=canonical_hash(bridge)))
    model = a.runtime_factories['inner'](a, seed['inner_parameters'])
    state = {k: np.array(seed['states'][k], copy=True) for k in model.state_keys}
    provider = a.lateral_native_factory(a)
    from types import SimpleNamespace
    native_y, anchor = provider.construction_point(SimpleNamespace(states=seed['states'], time_seconds=0.), bg, 1., 0)
    release = provider.evaluate(native_y, 0., anchor).outputs['release']
    rhs = model.rhs(0., state, release)
    after, _, _ = model.commit_ticks(0., state, cone_release=release, bc_tick=True)
    history = {':'.join(k): float(np.max(abs(after[k] - v), initial=0.)) for k, v in state.items()
               if k[1] == 'HistoryBipolarTerminal' and k[2].startswith('history_')}
    if any(history.values()) or any(not np.isfinite(v).all() for v in rhs.values()):
        raise ContractError('New local background finite/first-history check failed')
    write(out / 'HC_POSITION_BINDING.json', cfg)
    save_artifact(h, labels, out / 'typed_artifact')
    save_bundle(out / 'inner_boundary', kind='validation_evidence', identity=dict(artifact=h.artifact_sha256),
                payload=h.boundary_runtime, qualification=dict(default_adopted=False, fit_eligible=False,
                    background_qualified=False, FIT_READY=False))
    save_bundle(out / 'seed', kind='source_construction_seed', identity=seed['joint_background_initialization'],
                payload=seed, qualification=dict(default_adopted=False, fit_eligible=False,
                    background_qualified=False, FIT_READY=False, production_adopted=False))
    np.savez_compressed(out / 'BACKGROUND.npz', native_background=bg, cone_release=release, native_packed=native_y)
    write(out / 'INITIALIZATION_CHECK.json', dict(finite=True, first_BC_history_changes=history,
        maximum_inner_RHS=max(float(np.max(abs(v), initial=0.)) for v in rhs.values()),
        nstate=model.nstate, equilibrium=equilibrium, local_seed_independent=True, old473_seed_used=False,
        physical_gap_unchanged=True, exterior_gap_terms=model.gap_boundary_product,
        inherited_parameters=True, FIT_READY=False, background_qualified=False, production_adopted=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['optical_setup', 'initialize'], required=True)
    parser.add_argument('--input-background', type=Path)
    parser.add_argument('--scene-anchor',type=Path)
    parser.add_argument('--max-rss-gib', type=int, choices=[8, 24], default=8)
    parser.add_argument('--historical-seconds', type=float, default=4263.204762000069)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.mode == 'optical_setup' or args.input_background:
        out.mkdir(exist_ok=False)
    start = time.perf_counter()
    memory = MemoryGuard(args.max_rss_gib * 1024**3, 8 * 1024**3)
    def guard():
        if time.perf_counter() - start > 170:
            raise TimeoutError('Short initialization 170-second guard')
        if not memory.check():
            raise MemoryError('Initialization protection: ' + str(memory.stop_reason))
    status = 'FAILED'
    try:
        guard()
        if args.mode == 'optical_setup': optical_setup(args.prepared,out,args.scene_anchor)
        else: initialize(args.prepared, out, guard, args.input_background)
        status = 'PASS'
    except BaseException as exc:
        import traceback
        write(out / (args.mode + '_FAILURE.json'), dict(error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        write(out / (args.mode + '_RESOURCE.json'), dict(status=status, mode=args.mode,
            wall_seconds=time.perf_counter() - start, memory=memory.last, neural_forward_seconds=0.,
            historical_seconds=args.historical_seconds, stop_reason=memory.stop_reason,
            recorded_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__': main()
