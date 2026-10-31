"""Read-only asset preflight; does not construct or advance the retinal model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    'reports/cone_aperture_eccentricity/HC_new_v1/seed/manifest.json',
    'reports/cone_aperture_eccentricity/HC_new_v1/BACKGROUND.npz',
    'reports/layerwise_physiology/BC_functional_v1/PROTOCOL.json',
    'reports/h1_h2_joint_background_init_20261028_v1/validation/seed/manifest.json',
    'reports/h2_s_on_bc_defect_20261028_v1/diagnose.py',
    'reports/h2_s_on_bc_defect_20261028_v1/CHANGE_MANIFEST.json',
    'reports/h2_integration_20261028_v1/delivery/typed_artifact/manifest.json',
    'reports/ac_vs_native_support_v1/pre_registration_v1.json',
    'reports/ac_vs_native_support_v1/prepared_v1/initial_seed/manifest.json',
    'reports/ac_vs_native_support_v1/prepared_v1/boundary/manifest.json',
    'reports/ac_vs_redesign_v1/functional_static_halo_030_v1/selection.npz',
    'reports/ac_vs_redesign_v1/functional_static_halo_060_v1/selection.npz',
    'reports/spatial_group_graph_v02_20261025_v1/artifact_r2/manifest.json',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = ROOT / 'configurations/research/RGC_FUNCTIONAL_V1_candidate.json'
    checks = [dict(path=p, present=(ROOT / p).is_file()) for p in REQUIRED]
    result = dict(
        status='BLOCKED_MISSING_FROZEN_INPUTS' if any(not p['present'] for p in checks)
        else 'FILES_PRESENT_MODEL_IDENTITY_NOT_VALIDATED',
        root=str(ROOT), python=sys.version, checks=checks,
        checked_dependency_surface_is_exhaustive=False,
        configuration_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest(),
        full_network_run_completed=False, model_constructed=False,
        simulated_seconds=0, model_wall_seconds=0,
        production_adopted=False, background_qualified=False, FIT_READY=False)
    if args.output.exists():
        raise FileExistsError('Preserve existing preflight output: ' + str(args.output))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result))
    return 2 if result['status'] == 'BLOCKED_MISSING_FROZEN_INPUTS' else 0


if __name__ == '__main__':
    raise SystemExit(main())
