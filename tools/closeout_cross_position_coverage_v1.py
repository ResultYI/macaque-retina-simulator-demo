"""Save blocked two-position status and a data-only geometry figure."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/layerwise_physiology/cross_position_functional_coverage_v1_20261029'
REFERENCE = ROOT / 'reports/layerwise_physiology/RGC_functional_v1_laptop_20261029'


def write(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    start = time.perf_counter()
    static = json.loads((OUT / 'STATIC_COVERAGE_RESULT.json').read_bytes())
    geometry = json.loads((OUT / 'GEOMETRY_SELECTION.json').read_bytes())
    evidence = json.loads((OUT / 'REQUIRED_SOURCE_EVIDENCE.json').read_bytes())
    ledger = json.loads((OUT / 'RESOURCE_LEDGER.json').read_bytes())
    old = json.loads((REFERENCE / 'source_coupled_v1/analysis/TYPE_COMPLETION.json').read_bytes())
    artifact = ROOT / 'reports/spatial_group_graph_v02_20261025_v1/artifact_r2'
    manifest = json.loads((artifact / 'manifest.json').read_bytes())
    center = np.asarray(geometry['center_deg'])
    scale = manifest['configuration']['mm_per_degree']
    fig, ax = plt.subplots(figsize=(8.3, 7.4), constrained_layout=True)
    for row in old:
        pop = row['type']
        xy = np.load(artifact / 'populations' / pop / 'centroid_mm.npy') / scale
        xy = xy[np.linalg.norm(xy - center, axis=1) <= .72]
        label = pop.replace('_RGC', '').replace('small_bistratified_S_ON', 'S-bistratified')
        ax.scatter(xy[:, 0], xy[:, 1], s=14, alpha=.6, label=label)
    for radius, style in ((.15, '-'), (.3, '--'), (.6, ':')):
        ax.add_patch(Circle(center, radius, fill=False, color='black', linestyle=style, linewidth=1.2))
    ax.plot(*center, 'kx', markersize=10)
    ax.plot(0, 0, 'k+', markersize=8)
    ax.set(xlim=(center[0] - .75, center[0] + .75), ylim=(center[1] - .75, center[1] + .75),
           aspect='equal', xlabel='Parent x (deg)', ylabel='Parent y (deg)',
           title=f"Frozen RGC group centroids: center e={geometry['actual_eccentricity_deg']:.5f} deg\n"
                 'Core 0.15 / inner 0.30 / native 0.60 deg; new dynamics blocked')
    ax.legend(loc='upper left', fontsize=8)
    ax.text(.02, .02, 'Centroids represent frozen groups, not individual cell locations.\n'
            'Ancestry can extend beyond circles; exterior coupling remains explicit.',
            transform=ax.transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
    fig.savefig(OUT / 'near_center_geometry.png', dpi=170)
    fig.savefig(OUT / 'near_center_geometry.svg')
    plt.close(fig)

    missing = [r for r in evidence if not r['passed']]
    result = dict(schema='cross-position-functional-coverage-closeout-1',
        status='INCOMPLETE_SOURCE_EVIDENCE_BLOCKED', completed_utc=datetime.now(timezone.utc).isoformat(),
        near_center_eccentricity_deg=geometry['actual_eccentricity_deg'], near_center_deg=geometry['center_deg'],
        reference_eccentricity_deg=4.73, reference_dynamic_results_reused=True,
        near_structure_ready=True, near_local_background_ready=False, near_dynamic_run_complete=False,
        two_position_dynamic_comparison_complete=False, source_forward_seconds=0.,
        source_evidence_passed=sum(r['passed'] for r in evidence), source_evidence_total=len(evidence),
        unresolved_evidence=missing,
        block=dict(action='build_group_construction_seed on unchanged parent graph',
                   target='new near-center source construction and local background',
                   reason='Exact SHA-bound original RGC engineering adoption evidence files absent',
                   outcome='FAILED_GUARD_BEFORE_NEW_SEED', approval_denied=False, uncertain=False),
        additional_dependency_closure_confirmed=False,
        next_required_step='Restore these exact originals and verify SHA; then resume original source construction, new local input/background and matched dynamics. Further dependencies remain untested.',
        no_seed_substitution=True, no_sha_guard_bypass=True, no_scientific_parameter_change=True,
        FIT_READY=False, background_qualified=False, production_adopted=False,
        static_result='STATIC_COVERAGE_RESULT.json', comparison='TWO_POSITION_COVERAGE.csv',
        existing_reference_curves='../RGC_functional_v1_laptop_20261029/source_coupled_v1/analysis/index.html')
    write(OUT / 'RESULT.json', result)
    rows = []
    for row in old:
        pop = row['type']
        rows.append(f"| {pop} | {static['core_counts'][pop]} | {static['represented_members'][pop]} | blocked / no new background | {row['groups']} | {row['baseline_source_rate_Hz']:.6f} |")
    needs = '\n'.join(f"- `{r['path']}`\n  SHA256 `{r['expected_sha256']}`" for r in missing)
    text = f'''# Cross-position functional coverage v1 — incomplete

The frozen graph covers a near-center candidate at **{geometry['actual_eccentricity_deg']:.14f} degrees**, center {geometry['center_deg']} degrees. It is original parasol OFF group row {geometry['selected_parent_group_row']}; it is not an exact-zero or single fine-cell position. Selection used minimum RGC centroid radius before response evaluation, with deterministic type/row tie breaks. Explicit `centroid_mm.npy / {scale}` fixes the coordinate units; no response-based position search occurred.

Near-center geometry and static native060/inner030 selection are ready. New source construction stopped at the original SHA-bound RGC evidence gate. **No new local seed, background equilibrium, 900 ms response, numerical reference, or two-position dynamic comparison exists.** Empty near-position response cells in the CSV are unavailable data, not zero responses. Existing 4.73-degree results are referenced without recomputation or per-position normalization.

| Existing RGC type | Near core groups | Near represented members | Near function | 4.73 groups | 4.73 baseline source rate Hz |
|---|---:|---:|---|---:|---:|
{chr(10).join(rows)}

These are group-object counts and represented membership totals. They are not counts of independent event trains. The near core contains 55 RGC group objects and 2173 represented members. Current source-bound parameter adoption retains its engineering qualification; it has not been verified for a new initial state.

The geometric core has no Rod, rod BC or AII centroid. True chemical ancestry retains {static['counts']['rods']} Rod, {static['counts']['rod_BC']} rod BC and {static['counts']['AII_AC']} AII groups. Typed native support retains {static['counts']['cones']} Cone and {static['counts']['HC']} HC groups ({static['H1_groups']} H1 / {static['H2_groups']} H2). The gap matrix has {static['internal_gap_nnz']} internal nonzero entries (including diagonal terms) and {static['exterior_gap_nnz']} exterior nonzero terms. Exterior products have not been bound to a new background and must not be set to zero. No reciprocal edge was removed to make an isolated model. Core labels for HC/widefield AC follow the existing feedback/route contract and do not mean soma-centroid membership in a 0.15-degree disk.

![Geometry only](near_center_geometry.png)

The predeclared 900 ms protocol is in `PROTOCOL.json`: Cone release x0.9 at 50–200 ms, x1.1 at 350–500 ms, Rod input x2 at 600–750 ms. The near Rod response remains unrun, rather than structurally declared absent. Input/background must be independently bound and solved at this position; the old periphery seed cannot stand in for it. There is no new optical capture qualification.

## Exact current blocker

Action: `build_group_construction_seed` on the unchanged frozen parent graph. It failed before producing a seed because these original engineering-adoption evidence files are missing (not present with a different hash):

{needs}

`REQUIRED_SOURCE_EVIDENCE.json` records all {len(evidence)} unique adoption evidence paths; {sum(r['passed'] for r in evidence)} existing originals pass. `PREPARATION_FAILURE.json` preserves the exact failing guard. This is a missing-source blocker, not an approval denial. Restore original bytes, verify their hashes, then resume. This list is not a closed dependency package: later input/background dependencies have not yet been exercised. Do not fabricate equivalent JSON or weaken checks.

## Reproduction and retained artifacts

- `tools/prepare_cross_position_coverage_v1.py --output <fresh reports directory>` fixes geometry, attempts original algebraic source construction, and saves independent static selection after the source guard succeeds. Parent construction is not a new background equilibrium.
- `tools/finish_cross_position_static_v1.py --output <preserved failed directory> --attempt <new static directory name>` provides static geometry/typed projection only. A previous failed source-construction record is required. It is not a resumable runtime or an alternative seed.
- `static_only_v2/selection.npz`, `STATIC_GRAPH_DESCRIPTOR.json` and `typed_artifact/` preserve static rows, types and existing operators. Their descriptor explicitly has `runtime_ready=false`, no valid new seed binding and no solved background.
- `STATIC_COVERAGE_RESULT.json`, `TWO_POSITION_COVERAGE.csv` and `REFERENCE_473_REUSE.json` preserve the measured static counts and original 4.73-degree references. [Existing 4.73 curves](../RGC_functional_v1_laptop_20261029/source_coupled_v1/analysis/index.html) retain their original conditional qualification and S-bistratified limitations.
- `STATIC_PREPARATION_FAILURE.json` preserves an initial static adapter omission (`runtime_factories`); corrected using the existing bounded factories in the second attempt. No scientific equation or parameter changed.

The parent declares a 6-degree output radius; source ancestors extend farther and all bounds are in `GEOMETRY_SELECTION.json`. The parent logical graph identity is `{static['source_artifact']}`; the new static typed selection identity is `{static['selected_artifact']}`. The selection descriptor retains an unresolved old input-policy reference only to validate geometric schema; it does not bind or certify a new runtime.

`RESOURCE_LEDGER.json` carries forward historical {ledger['historical_model_wall_seconds']:.9f} s, conservatively adds this preparation/static work, and records zero new neural forward time. Production defaults and the old ledger are unchanged. `FIT_READY=false`, `background_qualified=false`, `production_adopted=false`. No Git commit, push or reset was performed. Stop here pending exact source restoration; do not expand to continuous eccentricity fitting.
'''
    with (OUT / 'README.md').open('x', encoding='utf8') as stream:
        stream.write(text)
    ledger['status'] = result['status']
    ledger['reporting_wall_seconds'] = time.perf_counter() - start
    ledger['closed_utc'] = result['completed_utc']
    ledger['new_output_bytes'] = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    (OUT / 'RESOURCE_LEDGER.json').write_text(json.dumps(ledger, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(result=result['status'], near_eccentricity_deg=result['near_center_eccentricity_deg'],
                          missing_originals=len(missing), ledger=ledger)), flush=True)


if __name__ == '__main__':
    main()
