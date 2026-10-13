"""Finite HC atlas candidates; effective disks are never measured branch anatomy."""
from copy import deepcopy
from collections.abc import Mapping
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from scipy.interpolate import PchipInterpolator

from .contracts import ContractError, Quantity
from .hc_arbor import validate_footprints
from .hc_mosaic import HCDensityProfile


_MERIDIANS = ("nasal", "superior", "temporal", "inferior")
_ANGLES = {name: i * math.pi / 2 for i, name in enumerate(_MERIDIANS)}


def _configuration(root):
    root = Path(root)
    path = root / "configurations/atlas_hc_candidate_v1.json"
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != "1.0" or data.get("profile_id") != "atlas_hc_candidate_v1":
        raise ContractError("Unknown HC atlas candidate profile")
    for row in data["source_assets"]:
        if hashlib.sha256((root / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ContractError("HC atlas source identity changed: " + row["path"])
    return data, hashlib.sha256(raw).hexdigest()


def make_hc_soma_density(root):
    """Reuse the existing total+pH1 field, verifying its original interval values.

    Verification of six numeric pairs does not turn the midpoint interpolation,
    central shape or directional extension into measured density everywhere.
    """
    config, _ = _configuration(root)
    profile = HCDensityProfile(root)
    rows = json.loads((Path(root) / config["verified_density_rows_path"]).read_bytes())["rows"]
    table2 = [row for row in rows if row["source_object"] == "Table 2"]
    original_rows = []
    for old in profile.source_rows:
        interval = tuple(old["eccentricity_interval_mm"])
        matches = [row for row in table2 if
                   (row["eccentricity_min_mm"], row["eccentricity_max_mm"]) == interval]
        if len(matches) != 2 or any(not any(row["cell_type"] == subtype and
                row["density_cells_per_mm2"] == old[subtype] for row in matches) for subtype in ("H1", "H2")):
            raise ContractError("Existing HC field does not match the verified Table2 interval values")
        original_rows.extend(deepcopy(matches))
    profile.metadata.update(verified_Table2_original_rows=original_rows,
        verified_Table2_asset=config["verified_density_rows_path"],
        original_intervals_not_replaced_with_measured_midpoints=True,
        numeric_source_binding="six existing density pairs match verified primary table; field interpolation remains project_approximation",
        physiological_validation=False)
    return profile


class AtlasHCProvider:
    """OPL anchor queries return per-ID footprints and explicit blocked records.

    A density callback is used at finite peripheral anchor nodes only. It must
    return a Quantity in cells/mm^2 or cones/mm^2, population_density, OPL.
    The alternative source-reported density transfer is explicitly named.
    The result's H1/H2 defaults are None: callers must inspect blocked_records
    and use per-ID overrides, rather than filling unsupported cells centrally.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.config, self.profile_sha256 = _configuration(root)
        self.density = make_hc_soma_density(root)
        self.verified_density_rows = tuple(deepcopy(json.loads(
            (self.root / self.config["verified_density_rows_path"]).read_bytes())["rows"]))
        peripheral = self.config["peripheral"]
        # Reported-density bound; actual-density/morphology callers plan from returned finite envelopes.
        self.maximum_envelope_mm = max(*self.config["central"]["radius_mm"].values(),
            *(math.sqrt(count / (math.pi * peripheral["reported_cone_density_cells_mm2"]))
              for count in peripheral["project_selected_convergence"].values()))

    def footprints(self, hc_ids, hc_types, anchor_positions, *, pedicle_density=None,
                   density_mode=None, mode="central_reference", enable_central_candidate=False,
                   morphology_by_hc_id=None):
        ids, types = tuple(hc_ids), tuple(hc_types)
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids) or len(types) != len(ids):
            raise ContractError("HC footprint query requires unique IDs and matching subtype count")
        if not isinstance(anchor_positions, Quantity):
            raise ContractError("HC anchor positions require a typed OPL Quantity")
        if anchor_positions.meaning not in ("anatomical_position", "arbor_anchor_position"):
            raise ContractError("HC anchor meaning must be explicit anatomical_position or arbor_anchor_position")
        xy = anchor_positions.require("mm", anchor_positions.meaning, frame="OPL")
        if xy.ndim != 2 or xy.shape[0] != len(ids) or xy.shape[1] not in (2, 3):
            raise ContractError("HC anchors require matching xy or explicit xyz coordinates")
        if mode not in ("central_reference", "connection_preview") or not isinstance(enable_central_candidate, bool):
            raise ContractError("Unknown HC candidate mode or candidate flag")
        if enable_central_candidate and mode != "connection_preview":
            raise ContractError("Inward central candidate is available only in explicit connection_preview mode")
        density_mode = density_mode or ("actual_local_pedicle_density" if pedicle_density is not None
                                        else "source_reported_local_density_project_transfer")
        if density_mode not in ("actual_local_pedicle_density", "source_reported_local_density_project_transfer"):
            raise ContractError("Unknown equal-area anchor density mode")
        if pedicle_density is not None and density_mode != "actual_local_pedicle_density":
            raise ContractError("A supplied density callback cannot silently be ignored or used as fallback")
        morphology = {} if morphology_by_hc_id is None else morphology_by_hc_id
        if not isinstance(morphology, Mapping) or set(morphology) - set(ids):
            raise ContractError("Morphology overrides require IDs present in this HC query")
        footprints = {"H1": None, "H2": None, "by_hc_id": {}}
        coverage, blocked, curves = [], [], {}
        central = self.config["central"]
        outer = self.config["peripheral"]

        def curve(subtype, meridian):
            key = (subtype, meridian)
            if key not in curves:
                radii = self.config["interpolation"]["project_knots_mm"]
                peripheral_r = radii[1:]
                theta = _ANGLES[meridian]
                points = np.c_[np.cos(theta) * np.asarray(peripheral_r),
                               np.sin(theta) * np.asarray(peripheral_r)]
                rho = None
                error = None
                if density_mode == "source_reported_local_density_project_transfer":
                    rho = np.full(len(points), outer["reported_cone_density_cells_mm2"])
                elif pedicle_density is None:
                    error = "PEDICLE_DENSITY_CALLBACK_MISSING"
                else:
                    try:
                        q = pedicle_density(Quantity(points, "mm", "anatomical_position",
                            "project_approximation", "OPL"))
                        if not isinstance(q, Quantity) or q.unit not in ("cells/mm^2", "cones/mm^2"):
                            raise ContractError("callback requires typed cells/mm^2 or cones/mm^2")
                        rho = q.require(q.unit, "population_density", (len(points),), "OPL")
                        if np.any(rho <= 0):
                            error = "PEDICLE_DENSITY_NONPOSITIVE_AT_ANCHOR"
                    except (ContractError, ValueError, TypeError) as problem:
                        error = "PEDICLE_DENSITY_UNSUPPORTED_OR_UNTYPED: " + str(problem)
                if error is not None:
                    curves[key] = {"error": error, "meridian": meridian, "cell_type": subtype,
                                   "anchor_positions_mm": points.tolist()}
                else:
                    count = outer["project_selected_convergence"][subtype]
                    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                        radius = np.r_[central["radius_mm"][subtype], np.sqrt(count / (np.pi * rho))]
                    if not np.all(np.isfinite(radius)) or np.any(radius <= 0):
                        curves[key] = {"error": "EQUAL_AREA_ANCHOR_RADIUS_NONFINITE", "meridian": meridian, "cell_type": subtype}
                    else:
                        curves[key] = {"interpolator": PchipInterpolator(radii, np.log(radius), extrapolate=False),
                            "radius_nodes_mm": radius.tolist(), "radial_knots_mm": radii,
                            "maximum_radius_mm": float(radius.max()), "meridian": meridian, "cell_type": subtype,
                            "peripheral_density_values_cells_mm2": rho.tolist(),
                            "density_mode": density_mode}
            return curves[key]

        for cid, subtype, point in zip(ids, types, xy):
            r = float(np.linalg.norm(point[:2]))
            row = {"hc_id": cid, "cell_type": subtype, "coordinate_frame": "OPL", "unit": "mm",
                   "position_meaning": anchor_positions.meaning, "anchor_position_mm": point.tolist(),
                   "radius_from_fovea_mm": r, "source_domain": deepcopy(self.config["source_domains"]),
                   "application_domain": [0 if enable_central_candidate else .5,
                                          outer["source_interval_mm"][1] if mode == "connection_preview" else 1],
                   "query_support": "complete finite footprint containment is checked separately by contact builder"}
            failure = None
            footprint = None
            if subtype not in ("H1", "H2"):
                failure = "HC_SUBTYPE_UNSUPPORTED"
            elif cid in morphology:
                original = deepcopy(morphology[cid])
                if original.get("kind") != "morphology_nodes":
                    failure = "DIRECT_MORPHOLOGY_OVERRIDE_REQUIRES_BRANCH_NODES"
                else:
                    band = original.get("application_anchor_band_mm", original.get("radial_support_mm"))
                    if not isinstance(band, (tuple, list)) or len(band) != 2 or not np.all(np.isfinite(band)) or not 0 <= band[0] <= r <= band[1]:
                        failure = "MORPHOLOGY_APPLICATION_DOMAIN_UNSUPPORTED"
                    elif (np.asarray(original.get("nodes_mm")).ndim == 2 and
                          np.asarray(original["nodes_mm"]).shape[1] == 3 and
                          original.get("node_coordinates") == "relative_to_anchor" and len(point) != 3):
                        failure = "MORPHOLOGY_XYZ_REQUIRES_XYZ_ANCHOR"
                    else:
                        footprint = validate_footprints({"by_hc_id": {cid: original}}, require_configured=False)["by_hc_id"][cid]
                        row.update(representation="branch_morphology", application_domain=list(band),
                                   source_domain=original.get("source_domain", original.get("radial_support_mm")),
                                   coverage_status="source_supported" if footprint["evidence_class"] in ("measurement", "published_fit") else "project_approximation")
            elif r < .5:
                if not enable_central_candidate:
                    failure = "CENTRAL_INWARD_CANDIDATE_NOT_SELECTED"
                else:
                    radius = central["radius_mm"][subtype]
                    row.update(candidate_id="central_constant_footprint_prior_v1",
                               evidence_status="EXTRAPOLATED_UNCALIBRATED", low_evidence=True)
            elif r <= 1:
                radius = central["radius_mm"][subtype]
                row.update(candidate_id="central_OPL_equal_area_W2000", evidence_status="SOURCE_ANCHORED_EFFECTIVE_DISK")
            elif mode != "connection_preview":
                failure = "PERIPHERAL_INTERPOLATION_NOT_SELECTED"
            elif r > outer["source_interval_mm"][1]:
                failure = "HC_FOOTPRINT_OUTSIDE_BOUNDED_ANCHOR_APPLICATION"
            else:
                theta = float(np.mod(np.arctan2(point[1], point[0]), 2*np.pi))
                sector = theta / (np.pi / 2)
                index = int(np.floor(sector)) % 4
                fraction = sector - np.floor(sector)
                weighted = [(_MERIDIANS[index], 1-fraction)]
                if fraction > 1e-12:
                    weighted.append((_MERIDIANS[(index+1) % 4], fraction))
                selected = [(curve(subtype, name), weight) for name, weight in weighted]
                failures = [c for c, _ in selected if "error" in c]
                if failures:
                    failure = "NO_SUPPORTED_FINITE_EQUAL_AREA_ANCHOR"
                    row["anchor_failures"] = deepcopy(failures)
                else:
                    radius = float(np.exp(sum(weight * float(c["interpolator"](r)) for c, weight in selected)))
                    bound = max(c["maximum_radius_mm"] for c, _ in selected)
                    if not np.isfinite(radius) or radius <= 0 or radius > bound * (1 + 1e-12):
                        failure = "POSITIVE_INTERPOLATION_BOUND_VIOLATION"
                    row.update(candidate_id="HC_positive_bounded_equal_area_interpolation_v1",
                        evidence_status="INTERPOLATED_UNCALIBRATED", radial_anchor_domain_mm=[1, 6],
                        meridian_weights=weighted, maximum_anchor_radius_mm=bound,
                        density_mode=density_mode, radius_is_inverse_density_at_query=False)
            if failure is None and footprint is None:
                footprint = {"kind": "disk", "radius_mm": radius, "unit": "mm", "coordinate_frame": "OPL",
                    "source_profile_id": self.config["profile_id"], "evidence_class": "project_approximation",
                    "approximation_id": row["candidate_id"], "source_locator": "Wassle2000 printed p599; explicit bounded project transfer/interpolation",
                    "cross_animal_transfer": True, "cross_species_transfer": "quoted convergence cohort unresolved; target composite macaque",
                    "meridian_transfer": self.config["interpolation"]["meridian_policy"],
                    "representation": "equivalent_area_effective_disk_not_measured_branch_arbor",
                    "application_anchor_band_mm": row["application_domain"],
                    "source_domain": row["source_domain"], "kernel": {"kind": "uniform"}}
                row.update(representation=footprint["representation"], coverage_status="project_approximation",
                           radius_mm=radius, density_basis="all cone pedicles; no unknown S/LM fraction correction")
            if failure is not None:
                row.update(coverage_status="unsupported", reason=failure)
                blocked.append(deepcopy(row))
            else:
                footprints["by_hc_id"][cid] = footprint
            coverage.append(row)
        validated = validate_footprints(footprints, require_configured=False)
        curve_manifest = [{k: v for k, v in c.items() if k != "interpolator"}
                          for c in curves.values()]
        return {"footprints": validated, "blocked_records": blocked, "per_hc_coverage": coverage,
                "manifest": {"profile_id": self.config["profile_id"], "profile_sha256": self.profile_sha256,
                    "mode": mode, "central_candidate_explicitly_enabled": enable_central_candidate,
                    "density_mode": density_mode, "finite_anchor_curves": curve_manifest,
                    "maximum_generated_envelope_mm": max((row["envelope_mm"] for row in validated["by_hc_id"].values()), default=None),
                    "source_domains": deepcopy(self.config["source_domains"]), "source_assets": deepcopy(self.config["source_assets"]),
                    "source_intervals_preserved": True, "per_cone_integer_divergence_imposed": False,
                    "geometry_ready": not blocked, "physiological_validation": False,
                    "inward_candidate_completes_coverage": False,
                    "HC_gap": "off; no dynamic gap state or conductance inferred by footprint provider",
                    "type_eligibility": "H1 S exclusion requires independently resolved cone types; no density fraction guessed",
                    "boundary_claim": "finite footprint only; query containment and recurrent exterior closure remain separate",
                    "claim": self.config["claim"]}}
