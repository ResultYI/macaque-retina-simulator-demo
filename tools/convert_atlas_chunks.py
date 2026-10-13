"""Convert supplied frozen local masters and verify row/query identity once.

This tool never generates cells or executes dynamics. Outputs are new files;
source JSON and shared runtime/configuration files are left intact.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retina_sim.atlas import AtlasROI
from retina_sim.atlas_chunks import ChunkedAtlasReader, convert_frozen_master, file_sha256, read_frozen_master
from retina_sim.atlas_sources import PackerFourMeridianDensity, MERIDIANS
from retina_sim.contracts import ContractError, Quantity
from retina_sim.hc_mosaic import HCDensityProfile
from retina_sim.population_revision import ApprovedPackerRodDensity, ApprovedParasolRFDensity


def verify_one(path, output_root):
    sha = file_sha256(path)
    table, types, bounds, role, halo, metadata = read_frozen_master(path, expected_source_sha256=sha)
    receipt = convert_frozen_master(path, output_root / (path.stem + "_" + sha[:12]), expected_source_sha256=sha)
    reader = ChunkedAtlasReader(receipt["path"], expected_source_sha256=sha,
                                expected_manifest_hash=receipt["storage_manifest_hash"])
    assert reader.metadata_record() == metadata
    assert reader.full_table_hash == table.table_hash
    x0, x1, y0, y1 = bounds
    sx, sy = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    point = table.coordinate(role).values[len(types) // 2]
    cut = float(point[0])
    cases = {
        "full_fixed_rows": (bounds, bounds, False),
        "overlap_A": ((cx - .04*sx, cx + .04*sx, cy - .04*sy, cy + .04*sy),
                      (cx - .20*sx, cx + .20*sx, cy - .20*sy, cy + .20*sy), True),
        "overlap_B": ((cx + .06*sx, cx + .14*sx, cy - .04*sy, cy + .04*sy),
                      (cx - .10*sx, cx + .30*sx, cy - .20*sy, cy + .20*sy), True),
        "left_of_exact_cell_boundary": ((x0, cut, y0, y1), (x0, cut, y0, y1), False),
        "right_of_exact_cell_boundary": ((cut, x1, y0, y1), (cut, x1, y0, y1), False),
    }
    selected = {}
    query_report = {}
    for name, (owner, query, pair_check) in cases.items():
        roi = AtlasROI(owner, query, table.reference_id, coordinate_role=role)
        expected = table.query(roi)
        if name == "overlap_A":
            tracemalloc.start()
        start = time.perf_counter()
        actual = reader.query(roi, max_cells=len(types), require_pair_context=pair_check)
        seconds = time.perf_counter() - start
        if name == "overlap_A":
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        assert actual.global_cell_ids == expected.global_cell_ids
        assert actual.source_keys == expected.source_keys
        assert np.array_equal(actual.row_indices, expected.row_indices)
        assert np.array_equal(actual.owner, expected.owner)
        assert actual.owner_status == expected.owner_status
        assert actual.cell_types == tuple(types[i] for i in expected.row_indices)
        for r in table.coordinates:
            # Byte equality includes the exact stored float64 representation.
            assert actual.coordinate(r).values.tobytes() == expected.coordinate(r).values.tobytes()
        if name == "full_fixed_rows":
            assert actual.table.table_hash == table.table_hash
        selected[name] = set(actual.global_cell_ids)
        query_report[name] = dict(actual.io_stats, measured_wall_seconds=seconds)
    assert selected["overlap_A"] & selected["overlap_B"]
    assert not selected["left_of_exact_cell_boundary"] & selected["right_of_exact_cell_boundary"]
    assert selected["left_of_exact_cell_boundary"] | selected["right_of_exact_cell_boundary"] == set(table.global_cell_ids)
    assert table.global_cell_ids[len(types) // 2] in selected["right_of_exact_cell_boundary"]
    outside = (x0 - .001, x1, y0, y1)
    try:
        reader.query(AtlasROI(bounds, outside, table.reference_id, coordinate_role=role), max_cells=len(types))
    except ContractError:
        outside_rejected = True
    else:
        raise AssertionError("Out-of-master query was accepted")
    if halo:
        try:
            reader.query(AtlasROI(bounds, bounds, table.reference_id, coordinate_role=role), max_cells=len(types))
        except ContractError:
            pass
        else:
            raise AssertionError("HC missing pair context was accepted")
    assert file_sha256(path) == sha
    return {**receipt, "source_json": str(path.relative_to(ROOT)), "source_json_bytes": path.stat().st_size,
            "population": table.population, "full_table_hash": table.table_hash,
            "index_role": role, "canonical_bounds_mm": list(bounds), "pair_context_mm": halo,
            "identity_verification": "PASS; full/overlap/exact-cell-boundary IDs, source keys, types, row order, coordinates and owner flags equal",
            "source_json_unchanged": True, "out_of_master_rejected": outside_rejected,
            "source_metadata_equal": True, "query_results": query_report,
            "measured_overlap_query_peak_traced_bytes": peak,
            "memory_measurement_scope": "tracemalloc peak of overlap_A query after reader initialization; not process RSS, not source JSON/conversion/graph memory",
            "block_file_bytes": sum(e["file_bytes"] for e in reader.manifest["blocks"]),
            "largest_block_decoded_array_bytes": max(e["decoded_array_bytes"] for e in reader.manifest["blocks"])}


def density_budget(local):
    """Density-only planning quadrature on a radius-3 disk, not cell generation."""
    z, w = np.polynomial.legendre.leggauss(64)
    r, theta = 1.5*(z + 1), np.pi*(z + 1)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    xy = np.column_stack((rr.ravel()*np.cos(tt.ravel()), rr.ravel()*np.sin(tt.ravel())))
    area_weights = r[:, None]*(1.5*w[:, None])*(np.pi*w[None, :])
    integrate = lambda field: float(np.sum(np.asarray(field(xy)).reshape(64, 64)*area_weights))
    rod = ApprovedPackerRodDensity(ROOT)
    parasol = ApprovedParasolRFDensity("ON")
    cone = PackerFourMeridianDensity(ROOT, "all_cones", central_policy="legacy_Packer_published_code_central_bridge_v1",
                                    rf_transfer="physical_retinal_mm_equals_planar_RF_mm_project_transfer_v1")
    hc = HCDensityProfile(ROOT)
    cone_field = lambda points: cone(Quantity(points, "mm", "functional_position", "project_approximation", "functional_visual_field"))
    masses = {"cones": integrate(cone_field), "rods": integrate(rod), "rod_BC": integrate(lambda p: rod(p)/18),
              "AII_AC": 5000*np.pi*3**2, "parasol_RGC_ON": integrate(parasol), "parasol_RGC_OFF": integrate(parasol),
              "HC": integrate(hc)}
    measured = {r["population"]: r for r in local}
    populations = {}
    for name, mass in masses.items():
        n = int(np.floor(mass + .5))
        estimate = {"estimated_density_mass": mass, "rounded_count_estimate": n,
                    "domain": "radius-3 disk in each field's declared plane; no connection/physical guard included",
                    "coordinate_float64_xy_bytes_lower_bound": 16*n}
        if name not in ("HC", "cones"):
            side = max(64, int(np.ceil(np.sqrt(8*n))))
            generation = side*side*80 + n*96
            estimate.update(equal_mass_quadrature_side_estimate=side, equal_mass_generation_core_bytes_estimate=generation,
                            exceeds_current_512MiB_generation_cap=(generation > 512*1024**2),
                            exceeds_current_60000_population_budget=(n > 60000))
        if name == "HC":
            estimate.update(joint_two_sweep_proposals_estimate=2*n, partition_target_mass8_scale_estimate=n/8,
                            exceeds_current_500000_proposals=(2*n > 500000), core_arrays_128bytes_per_cell_estimate=128*n)
        if name in measured:
            reference = measured[name]
            estimate.update(compressed_storage_bytes_estimate_from_local_ratio=int(np.ceil(n*reference["storage_bytes"]/reference["count"])),
                            extrapolation_reference=reference["source_json"],
                            local_measured_total_storage_bytes_per_cell=reference["storage_bytes"]/reference["count"])
        populations[name] = estimate
    boundaries = {
        "Packer_rods": {"cardinal_digitized_node_ranges_mm": {m: [float(rod.rows[m][0]["eccentricity_mm"]), float(rod.rows[m][-1]["eccentricity_mm"])] for m in MERIDIANS},
                         "central_completion": "approved quadratic below each actual first node; source map digitization and nominal RF registration remain approximations; no exterior completion"},
        "Packer_cones": {"cardinal_digitized_node_ranges_mm": {m: [float(cone.rows[m][0]["eccentricity_mm"]), float(cone.rows[m][-1]["eccentricity_mm"])] for m in MERIDIANS},
                         "central_completion": "existing approved published-code bridge; no new density or L/M/S source map"},
        "parasol_RF": {"radial_application_mm": [0, 3], "guard_beyond_3mm": "UNSUPPORTED; no clamping/extrapolation; [-3,3]^2 corners are not supported"},
        "RBC": "existing rod/18 effective transfer only; physical support domain must be derived from actual targets",
        "AII": "existing 5000/mm2 effective physical transfer only; RF disk does not bound actual displaced IPL targets",
        "HC": {"density_radial_application_mm": [0, hc.support_radius_mm], "generation": "approved joint master, H1/H2 marks and pair interaction retained; no per-block generators",
                "frozen_current_bounds_mm": measured["HC"]["canonical_bounds_mm"], "pair_context_mm": measured["HC"]["pair_context_mm"],
                "morphology": "central/peripheral direct arbor curves remain missing; no morphology or density changes authorized"},
        "Henle_and_IPL": "RF0-3 must be mapped before choosing physical HC/rod/RBC/AII generation bounds; bounded source bridges and directional limits stay active",
        "other_inner_populations": "midget/diffuse/S/AC/WAC populations derive from actual cone parents and typed sites, not independently measured global density; no justified full-population count supplied",
        "S": "existing effective S route remains separately labelled; do not fill missing species/direction/central evidence",
        "graph_boundary": "finite support/HC pair context required; recurrent exterior closure remains uncertified",
    }
    return {"status": "ESTIMATES_ONLY_NO_MASTER_GENERATED", "method": "64x64 polar Gauss-Legendre density-only quadrature except exact constant AII area",
            "numeric_error_bound": "not measured; engineering order-of-magnitude planning, not final quotas",
            "populations": populations, "partial_total_cells_estimate": sum(v["rounded_count_estimate"] for v in populations.values()),
            "partial_total_excludes": "other inner typed populations, displacement guards, edges, dynamic states, checkpoints, backgrounds and backups",
            "space_index_square_tile_upper_estimate_per_population": int(np.ceil(6/.1 + 2))**2,
            "disk_storage_estimates": "local total bytes/cell extrapolation includes local block/metadata overhead; compression and global index overhead may differ; not a guarantee",
            "generation_memory_estimates": "existing spatial_profiles side rule and geometry core-array formula only; Python/JSON/solver/graph and peak RSS excluded",
            "conversion_memory_limit": "converter currently fully reads an existing JSON/table once; large JSON conversion needs separately budgeted memory or a streaming producer; storage reader remains lazy",
            "generation_time": "NOT_MEASURED; local timers do not predict global equal-mass or joint time",
            "source_boundaries": boundaries,
            "version_and_cache_policy": "repartitioning unchanged rows preserves source JSON SHA/table hash; generating a larger domain changes recipe/canonical namespace/master version and invalidates background/checkpoint caches bound to old populations and edges; never reuse a local cache as global background",
            "configuration_policy": "one shared configuration; storage layout is internal and adds no user version/menu"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ContractError("Validation report already exists; overwrite refused")
    paths = sorted((ROOT/"data/atlas/population_masters").glob("*.json")) + [ROOT/"data/atlas/hc_joint_central_master_v2.json"]
    local = [verify_one(path, args.output_root) for path in paths]
    report = {"status": "PASS", "schema": "frozen_atlas_spatial_chunks_validation_v1",
              "scope": "one batch, existing frozen local masters only; no sampling, model training or dynamics",
              "local_masters": local, "total_original_json_bytes": sum(r["source_json_bytes"] for r in local),
              "total_chunk_storage_bytes": sum(r["storage_bytes"] for r in local),
              "full_RF_0_3_planning": density_budget(local)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "masters": len(local), "original_bytes": report["total_original_json_bytes"],
                      "chunk_bytes": report["total_chunk_storage_bytes"], "report": str(args.report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
