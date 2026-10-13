"""Fixed global effective rod/RBC/AII geometry using the actual rod atlas source.

RBC ratio and AII density are existing project transfers, not new measurements.
No cone Henle map, central rod extrapolation, kinetics or conductance is added.
"""
from pathlib import Path
import hashlib
import numpy as np

from .atlas import AtlasROI, CanonicalCellTable
from .contracts import ContractError, Quantity, canonical_hash
from .geometry import keyed_rng
from .runtime_profiles import execution_profile
from .spatial_profiles import bounded_matrix, finite_support_coverage


RECIPE = "fixed_global_effective_rod_inner_tiles_v1"
PROFILE = "retina_contact_geometry_effective_v1"
PROFILE_SHA256 = "ab63ff04001615a6a1e48bdf8ceb9af5da01596361dc273ab735d39d3db9f893"


def _quantity(points, meaning, frame, tag="project_approximation"):
    return Quantity(np.asarray(points, dtype=float).reshape(-1, 2), "mm", meaning, tag, frame)


def _record(quantity):
    return {"values": quantity.values.tolist(), "unit": quantity.unit,
            "meaning": quantity.meaning, "frame": quantity.coordinate_frame,
            "evidence_tag": quantity.evidence_tag}


def _fixed_points(atlas, bounds, population, density, namespace, max_cells):
    size, seed = atlas.config["tile_size_mm"], atlas.config["seed"]
    x0, x1, y0, y1 = bounds
    ix = range(int(np.floor(x0 / size)), int(np.ceil(x1 / size)))
    iy = range(int(np.floor(y0 / size)), int(np.ceil(y1 / size)))
    if len(ix) * len(iy) > max_cells:
        raise ContractError(f"{population} effective tile budget exceeded; no density thinning")
    keys, positions, generated = [], [], 0
    for tx in ix:
        for ty in iy:
            center = _quantity([[(tx + .5) * size, (ty + .5) * size]],
                               "functional_position", "functional_visual_field")
            rho = float(density(center)[0])
            if not np.isfinite(rho) or rho < 0:
                raise ContractError("Effective density must remain finite nonnegative")
            rng = keyed_rng(seed, namespace + "|" + population + "|" + str(tx) + "|" + str(ty))
            count = int(rng.poisson(rho * size * size))
            generated += count
            if generated > max_cells:
                raise ContractError(f"{population} effective cell budget exceeded; density unchanged")
            xy = (np.array([tx, ty]) + rng.random((count, 2))) * size
            selected = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
            keys.extend(f"tile:{tx}:{ty}:cell:{i}" for i in np.flatnonzero(selected))
            positions.extend(xy[selected])
    return tuple(keys), np.asarray(positions, dtype=float).reshape(-1, 2), generated


def build_effective_rod_inner(source_atlas, roi, *, max_cells=60000):
    """Return JSON-ready finite geometry, retaining unsupported source absence.

    ``roi`` is the rod physical-chart query expressed in the atlas RF reference.
    Its same numeric bounds are explicitly transferred to effective OPL/IPL.
    It does not generate the displaced cone BC domain or certify exterior loops.
    """
    if (not isinstance(roi, AtlasROI) or roi.coordinate_role != "rf" or
            roi.reference_id != source_atlas.config["reference_id"]):
        raise ContractError("Rod inner geometry requires the declared RF/mm global reference")
    if not isinstance(max_cells, int) or isinstance(max_cells, bool) or max_cells < 1:
        raise ContractError("Positive integer per-population cell budget required")
    profile = execution_profile(source_atlas.root, PROFILE)
    geometry = profile["geometry"]
    ratio, aii_density = geometry["RBC_to_rod_density_ratio"], geometry["AII_density_cells_per_mm2"]
    if (profile["profile_file_sha256"] != PROFILE_SHA256 or ratio != 1 / 18 or
            aii_density != 5000 or geometry["hard_support_mm"] != .04):
        raise ContractError("Frozen approved rod-inner effective geometry profile changed")
    corners = np.array([[x, y] for x in roi.owner_bounds_mm[:2] for y in roi.owner_bounds_mm[2:]])
    radius = float(np.linalg.norm(corners, axis=1).max())
    phase = "phase1" if radius <= source_atlas.config["phase1_rf_radius_mm"][1] else "phase2"
    if radius > source_atlas.config["phase2_rf_radius_mm"][1]:
        raise ContractError("Owner exceeds the declared atlas phase domain")
    common = {"atlas_id": source_atlas.config["atlas_id"],
              "atlas_version": source_atlas.config["atlas_version"],
              "reference_id": source_atlas.config["reference_id"],
              "tile_size_mm": source_atlas.config["tile_size_mm"],
              "seed": source_atlas.config["seed"], "recipe": RECIPE, "numpy_version": np.__version__}
    # AII sampling is independent of rod source, rod count and requested ROI.
    aii_namespace = canonical_hash(dict(common, population="AII_AC", density_cells_mm2=aii_density,
                                       transfer=geometry["AII_transfer"]))
    tables, typed_points, ownership, contacts, blockers = {}, {}, {}, {}, []

    def store(population, keys, points, namespace, *, rf=False, aii=False):
        coordinates = {"soma": _quantity(points, "anatomical_position", "anatomical_retina")}
        if rf:
            coordinates["rf"] = _quantity(points, "functional_position", "functional_visual_field")
        if population != "rods":
            coordinates["ipl_preterminal"] = _quantity(points, "terminal_position", "IPL")
        if aii:
            coordinates["ipl_post_arbor"] = _quantity(points, "dendritic_position", "IPL")
        table = CanonicalCellTable(common["atlas_id"], common["atlas_version"], namespace,
                                   population, keys, coordinates, common["reference_id"])
        tables[population] = table
        selected = table.query(AtlasROI(roi.owner_bounds_mm, roi.query_bounds_mm,
                                        roi.reference_id, coordinate_role="soma"))
        ownership[population] = {"owner": selected.owner.tolist(), "owner_status": selected.owner_status,
                                "owner_count": int(selected.owner.sum())}
        return table

    aii_keys, aii_points, aii_generated = _fixed_points(source_atlas, roi.query_bounds_mm, "AII_AC",
        lambda q: np.full(len(q.values), aii_density), aii_namespace, max_cells)
    aii = store("AII_AC", aii_keys, aii_points, aii_namespace, aii=True)
    rod_source_error, rod_manifest, rbc_generated = None, None, None
    try:
        rod_patch = source_atlas.query(roi, population="rods", max_cells=max_cells, phase=phase)
    except ContractError as error:
        if "Packer rod curve unsupported at " not in str(error):
            raise
        rod_source_error = str(error)
        blockers.append({"populations": ["rods", "rod_BC"], "status": "unsupported",
            "reason": rod_source_error, "scope": "requested canonical rod tile/support envelope",
            "not_biological_absence": True, "legacy_rod_ring_not_used": True})
    else:
        rod_manifest = rod_patch.manifest
        selection = rod_patch.selection
        rod_points = selection.coordinate("rf").values
        rod = store("rods", selection.source_keys, rod_points, selection.table.source_version, rf=True)
        typed_points["rods"] = {"OPL_spherule": _record(_quantity(rod_points, "terminal_position", "OPL")),
            "transfer": "old_rod_RF_OPL_spherule_colocation_project_v1", "cone_Henle_applied": False,
            "cone_pedicle_label_used": False}
        rbc_namespace = canonical_hash(dict(common, population="rod_BC", density_ratio=ratio,
            rod_mosaic_namespace=source_atlas.mosaic_namespace,
            density_rule=source_atlas.config["density_rule"], transfer=geometry["RBC_transfer"]))
        rbc_keys, rbc_points, rbc_generated = _fixed_points(source_atlas, roi.query_bounds_mm,
            "rod_BC", lambda q: ratio * source_atlas.rods(q), rbc_namespace, max_cells)
        rbc = store("rod_BC", rbc_keys, rbc_points, rbc_namespace, rf=True)
        typed_points["rod_BC"] = {"OPL_dendrite": _record(_quantity(rbc_points, "dendritic_position", "OPL")),
                                  "transfer": "old_RBC_OPL_to_IPL_point_colocation_effective_v1"}

        def contact(pre, post, pre_points, post_points, budget, frame):
            matrix, rows = bounded_matrix(pre_points.values, post_points.values, budget, support_mm=.04)
            coverage = finite_support_coverage(post_points.values, .04, [roi.query_bounds_mm])
            weights = matrix.tocoo()
            route = pre.population + "->" + post.population
            for j, row in enumerate(rows):
                row.update(post_id=post.global_cell_ids[j], support_envelope_complete=bool(coverage[j]),
                    row_status="FINITE_QUERY_INCOMPLETE" if not coverage[j] else
                        "EMPTY_SAMPLE_FINITE_SUPPORT" if row["actual_degree"] == 0 else "FINITE_EFFECTIVE_POOL",
                    exact_budget_enforcement=False, not_measured_biological_zero=True)
                if not coverage[j] or row["actual_degree"] == 0:
                    blockers.append({"route": route, "post_id": post.global_cell_ids[j],
                                     "reason": row["row_status"], "is_owner": ownership[post.population]["owner"][j]})
            contacts[route] = {"frame": frame, "selection_budget": budget, "hard_support_mm": .04,
                "pre_positions": _record(pre_points), "post_positions": _record(post_points),
                "edges": [{"pre_id": pre.global_cell_ids[i], "post_id": post.global_cell_ids[j],
                    "weight": float(value), "unit": "dimensionless"}
                    for j, i, value in zip(weights.row, weights.col, weights.data)],
                "rows": rows, "evidence_class": "project_approximation", "kinetics_included": False}

        contact(rod, rbc, _quantity(rod_points, "terminal_position", "OPL"),
                _quantity(rbc_points, "dendritic_position", "OPL"), 40, "OPL")
        contacts["rods->rod_BC"]["operator_placement"] = "existing per-rod nonlinear TRPM1 before RBC pooling; geometry only"
        contact(rbc, aii, rbc.coordinate("ipl_preterminal"), aii.coordinate("ipl_post_arbor"), 3, "IPL")

    return {"tables": {name: table.record() for name, table in tables.items()},
        "typed_points": typed_points, "contacts": contacts, "blockers": blockers,
        "metadata": {"recipe_id": RECIPE, "profile_id": PROFILE, "profile_sha256": PROFILE_SHA256,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_atlas_version": source_atlas.source_version, "rod_source_manifest": rod_manifest,
            "source_status": "unsupported" if rod_source_error else "source_informed_project_geometry",
            "rod_source_error": rod_source_error, "phase": phase,
            "RBC_to_rod_density_ratio": ratio, "RBC_density_formula": "2*rho_rod/36",
            "RBC_density_basis_degree": 36, "rod_RBC_selection_budget": 40,
            "RBC_AII_selection_budget": 3, "AII_density_cells_per_mm2": aii_density,
            "density_evidence": "existing effective source-anchor transfer; not measured regional RBC/AII density",
            "AII_independent_of_rod_count": True, "AII_generated_cell_count": aii_generated,
            "RBC_generated_cell_count": rbc_generated, "population_ownership": ownership,
            "owner_domain": list(roi.owner_bounds_mm), "query_support": list(roi.query_bounds_mm),
            "coordinate_transfer": "explicit old physical rod/AII point chart; not cone Henle or measured RF/soma identity",
            "canonical_sampling": "fixed global piecewise-constant Poisson tiles/keyedRNG; no ROI seed",
            "max_cells": max_cells, "budget_scope": "per population full-tile generated cells and tile work",
            "density_scale": 1., "boundary_mode": "finite_open_boundary", "recurrent_closure": "NOT_CERTIFIED",
            "geometry_only": True, "dynamics_run": False, "physiological_validation": "NOT_RUN"}}
