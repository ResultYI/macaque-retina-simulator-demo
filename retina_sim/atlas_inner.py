"""Existing typed effective inner anchors attached to supplied atlas cones.

This is geometry/ancestry, not new subtype density or a calibrated inner model.
New OPL points never enter the old McGregor total-displacement/soma split.
"""
from dataclasses import dataclass
from types import MappingProxyType
import re

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .atlas import CanonicalCellTable
from .contracts import ContractError, Quantity
from .ipl_geometry import local_typed_sites, widefield_grid
from .spatial_profiles import bounded_matrix
from .wiring import enforce_midget_parent_matrix, wool_midget_profile


ADAPTER = "atlas_supplied_OPL_effective_IPL_v1"
EVIDENCE = "project_approximation"


def _quantity(points, meaning, frame):
    return Quantity(np.asarray(points, dtype=float).reshape(-1, 2), "mm", meaning, EVIDENCE, frame)


@dataclass(frozen=True)
class EffectiveInnerLayout:
    tables: object
    parent_ids: object
    contacts: object
    blockers: tuple
    metadata: dict

    def record(self):
        contacts = {}
        for route, record in self.contacts.items():
            matrix = record["matrix"].tocoo()
            contacts[route] = {key: value for key, value in record.items()
                               if key not in ("matrix", "pre_positions", "post_positions")}
            contacts[route]["edges"] = [{"pre_id": record["pre_ids"][i],
                "post_id": record["post_ids"][j], "weight": float(weight),
                "unit": "dimensionless"} for j, i, weight in zip(matrix.row, matrix.col, matrix.data)]
            for name in ("pre_positions", "post_positions"):
                quantity = record[name]
                contacts[route][name] = {"values": quantity.values.tolist(), "unit": quantity.unit,
                    "meaning": quantity.meaning, "frame": quantity.coordinate_frame,
                    "evidence_tag": quantity.evidence_tag}
        return {"tables": {name: table.record() for name, table in self.tables.items()},
                "parent_ids": dict(self.parent_ids), "contacts": contacts,
                "blockers": list(self.blockers), "metadata": self.metadata}


def build_effective_inner(cones, *, configuration, cone_subtypes=None,
                          temporal_equivalent_mm=None, temporal_equivalence_source_id=None,
                          diffuse_seed_cone_ids=None, include_widefield_ac=False,
                          max_effective_cells=8192):
    """Build midget/S ancestry and independent local AC effective sites.

    Temporal equivalence must be supplied by a named source/adapter; RF radius
    is never silently substituted. Unknown labels retain structural candidates
    but cannot contribute typed weights. Diffuse seeds are explicitly supplied;
    their sampling count is effective resolution, not a measured BC density.
    """
    if not isinstance(cones, CanonicalCellTable):
        raise ContractError("Inner adapter requires supplied canonical cone anatomy")
    rf = cones.coordinate("rf").values
    pedicle = cones.coordinate("opl_pedicle").values
    count = len(cones.global_cell_ids)
    digest = configuration.get("profile_sha256", "")
    if (configuration.get("hard_support_mm") != .04 or
            not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None):
        raise ContractError("Existing typed IPL profile and unchanged 40um hard support required")
    labels = np.asarray(["UNKNOWN"] * count if cone_subtypes is None else cone_subtypes, dtype=object)
    if labels.shape != (count,):
        raise ContractError("Cone subtype array must match canonical cone rows")
    labels = np.array(["UNKNOWN" if label is None else label for label in labels], dtype=object)
    if any(label not in ("L", "M", "S", "UNKNOWN") for label in labels):
        raise ContractError("Cone subtypes must be L/M/S/UNKNOWN; no implicit non-S typing")
    if temporal_equivalent_mm is None:
        equivalent = np.full(count, np.nan)
    else:
        equivalent = np.asarray(temporal_equivalent_mm, dtype=float)
        if equivalent.shape != (count,):
            raise ContractError("Explicit temporal-equivalent mm array must match cone rows")
        if (not isinstance(temporal_equivalence_source_id, str) or
                not temporal_equivalence_source_id.strip()):
            raise ContractError("Temporal equivalence requires a named source/project adapter")
    tables, parents, contacts, blockers = {}, {}, {}, []
    source_version = cones.source_version + "|" + ADAPTER + "|" + digest
    unknown_points = pedicle[labels == "UNKNOWN"]
    unknown_tree = cKDTree(unknown_points) if len(unknown_points) else None

    def make_table(population, keys, points_rf, points_ipl, *, post=False):
        role, meaning = ("ipl_post_arbor", "dendritic_position") if post else ("ipl_preterminal", "terminal_position")
        coordinates = {role: _quantity(points_ipl, meaning, "IPL")}
        if points_rf is not None:
            coordinates["rf"] = _quantity(points_rf, "functional_position", "functional_visual_field")
        table = CanonicalCellTable(cones.atlas_id, cones.atlas_version, source_version, population,
                                   tuple(keys), coordinates, cones.reference_id)
        tables[population] = table
        return table

    def put_contact(pre, post, matrix, frame, pre_positions, post_positions, statuses, rows):
        route = pre + "->" + post
        matrix = matrix.tolil()
        pre_ids = cones.global_cell_ids if pre == cones.population else tables[pre].global_cell_ids
        for j, status in enumerate(statuses):
            rows[j].update(selected_candidate_pre_ids=[pre_ids[i] for i in matrix.rows[j]],
                           candidate_weights_before_status_gate=list(matrix.data[j]))
            if status.startswith("BLOCKED") or status.startswith("INCOMPLETE"):
                matrix[j, :] = 0
                blockers.append({"route": route, "post_id": tables[post].global_cell_ids[j], "reason": status})
        contacts[route] = {"matrix": matrix.tocsr(), "pre_ids": pre_ids,
            "post_ids": tables[post].global_cell_ids, "frame": frame,
            "pre_positions": pre_positions, "post_positions": post_positions,
            "row_status": tuple(statuses), "rows": rows,
            "evidence_class": EVIDENCE, "boundary_mode": "finite_supplied_candidates",
            "query_support_complete": "NOT_ESTABLISHED_BY_THIS_ADAPTER"}

    def unknown_near(points, support):
        if unknown_tree is None:
            return np.zeros(len(points), dtype=bool)
        radii = np.broadcast_to(support, (len(points),))
        return np.array([bool(unknown_tree.query_ball_point(point, radius))
                         for point, radius in zip(points, radii)], dtype=bool)

    def pool_contact(pre, post, pre_positions, post_positions, budget, *, frame="IPL",
                     gaussian_scale=None, upstream_statuses=None, block_unknown=False,
                     seed_statuses=None):
        # A known S-seeded diffuse placeholder is not a functional diffuse BC.
        # Remove only that known ineligible candidate before the finite query;
        # incomplete/unknown upstream rows still retain their explicit blockers.
        eligible=np.array([i for i in range(len(pre_positions.values)) if upstream_statuses is None or
            upstream_statuses[i]!='BLOCKED_DIFFUSE_SEED_IS_S'],dtype=int)
        matrix, rows = bounded_matrix(pre_positions.values[eligible], post_positions.values, budget,
                                     support_mm=.04, gaussian_scale=gaussian_scale)
        coo=matrix.tocoo()
        matrix=sparse.csr_matrix((coo.data,(coo.row,eligible[coo.col])),shape=(len(post_positions.values),len(pre_positions.values)))
        uncertain = unknown_near(post_positions.values, .04) if block_unknown else np.zeros(len(rows), bool)
        statuses = []
        for j, row in enumerate(rows):
            selected = matrix.getrow(j).indices
            status = "READY_PROJECT_FINITE_POOL" if len(selected) else "INCOMPLETE_NO_INPUT_WITHIN_40UM"
            if uncertain[j]:
                status = "BLOCKED_UNKNOWN_CONE_SUBTYPE_IN_INPUT_SUPPORT"
            elif upstream_statuses is not None and any(not upstream_statuses[i].startswith("READY") for i in selected):
                status = "BLOCKED_UPSTREAM_TYPED_DRIVE"
            if seed_statuses is not None and seed_statuses[j] is not None:
                status = seed_statuses[j]
            row.update(selection_budget=budget, exact_budget_enforcement=budget == 1,
                       source_rule="existing effective pooling; deficits retained, no distant fill")
            statuses.append(status)
        put_contact(pre, post, matrix, frame, pre_positions, post_positions, statuses, rows)

    if diffuse_seed_cone_ids is not None:
        seed_ids = tuple(diffuse_seed_cone_ids)
        index = {cell: i for i, cell in enumerate(cones.global_cell_ids)}
        if len(set(seed_ids)) != len(seed_ids) or any(cell not in index for cell in seed_ids):
            raise ContractError("Diffuse effective seeds require unique supplied canonical cone IDs")
        seed_ids = tuple(sorted(seed_ids))
        seeds = np.array([index[cell] for cell in seed_ids], dtype=int)
        lm = np.flatnonzero((labels == "L") | (labels == "M"))
        for polarity in ("ON", "OFF"):
            name = "diffuse_BC_" + polarity
            bc = make_table(name, seed_ids, rf[seeds], pedicle[seeds])
            # Sampling ancestry is not a private synaptic parent for diffuse BC.
            pre = _quantity(pedicle[lm], "pedicle_position", "OPL")
            post = _quantity(pedicle[seeds], "dendritic_position", "OPL")
            matrix, rows = bounded_matrix(pre.values, post.values, 8, support_mm=.04,
                                          gaussian_scale=.08 + .04 * np.linalg.norm(rf[seeds], axis=1))
            complete = sparse.lil_matrix((len(seeds), count))
            complete[:, lm] = matrix
            uncertain = unknown_near(post.values, .04)
            statuses = []
            for j, row in enumerate(rows):
                status = ("BLOCKED_DIFFUSE_SEED_IS_S" if labels[seeds[j]] == "S" else
                    "BLOCKED_UNKNOWN_CONE_SUBTYPE_IN_INPUT_SUPPORT" if uncertain[j] else
                    "READY_PROJECT_FINITE_POOL" if row["actual_degree"] else "INCOMPLETE_NO_INPUT_WITHIN_40UM")
                statuses.append(status)
                row.update(sampling_seed_cone_id=seed_ids[j], selection_budget=8,
                    exact_budget_enforcement=False, seed_is_not_private_synaptic_parent=True,
                    gaussian_scale_mm=float(.08 + .04 * np.linalg.norm(rf[seeds[j]])))
            put_contact(cones.population, name, complete, "OPL", cones.coordinate("opl_pedicle"), post, statuses, rows)

    for prefix, indices in (("midget", np.flatnonzero(labels != "S")),
                            ("S", np.flatnonzero(labels == "S"))):
        polarities = ("ON", "OFF") if prefix == "midget" else ("ON",)
        for polarity in polarities:
            bc_name = f"midget_BC_{polarity}" if prefix == "midget" else "S_ON_BC"
            cone_parents = tuple(cones.global_cell_ids[i] for i in indices)
            bc = make_table(bc_name, cone_parents, rf[indices], pedicle[indices])
            parents[bc_name] = cone_parents
            known = labels[indices] != "UNKNOWN"
            opl_supported = np.linalg.norm(pedicle[indices], axis=1) <= 10 if prefix == "midget" else np.ones(len(indices), bool)
            direct = sparse.csr_matrix((np.ones(len(indices)), (np.arange(len(indices)), indices)),
                                       shape=(len(indices), count))
            statuses = ["READY_PROJECT_PRIVATE_PARENT" if typed and supported else
                        "BLOCKED_UNKNOWN_CONE_SUBTYPE" if not typed else "BLOCKED_MIDGET_OPL_ABOVE_10MM"
                        for typed, supported in zip(known, opl_supported)]
            direct_rows = [{"parent_cone_id": cone, "distance_mm": 0.,
                "rule": "qualified source identity; no nearest substitution"} for cone in cone_parents]
            put_contact(cones.population, bc_name, direct, "OPL",
                cones.coordinate("opl_pedicle"), _quantity(pedicle[indices], "dendritic_position", "OPL"),
                statuses, direct_rows)

            rgc_name = f"midget_RGC_{polarity}" if prefix == "midget" else "small_bistratified_S_ON"
            rgc = make_table(rgc_name, bc.global_cell_ids, rf[indices], pedicle[indices], post=True)
            parents[rgc_name] = bc.global_cell_ids
            n = len(indices)
            if prefix == "S":
                matrix, rows = bounded_matrix(pedicle[indices], pedicle[indices], 8, support_mm=.04)
                uncertain = unknown_near(pedicle[indices], .04)
                statuses = ["BLOCKED_UNKNOWN_S_PARTITION_IN_SUPPORT" if blocked else
                    "READY_PROJECT_S_RELAY_MINIMUM_ONE" if row["actual_degree"] >= 1 else
                    "INCOMPLETE_S_RELAY_ZERO_INPUT" for blocked, row in zip(uncertain, rows)]
                for row in rows:
                    row.update(minimum_selected_inputs=1, exact_target_enforcement=False,
                               legacy_selection_budget=8, interpretation="effective relay, not biological BC convergence")
            else:
                matrix = sparse.lil_matrix((n, n))
                rows = [{"post_index": j} for j in range(n)]
                statuses = ["BLOCKED_UNKNOWN_CONE_SUBTYPE" if not typed else
                            "BLOCKED_WOOL_TEMPORAL_EQUIVALENCE_MISSING_OR_UNSUPPORTED" for typed in known]
                valid = known & opl_supported & np.isfinite(equivalent[indices]) & (equivalent[indices] >= 0) & (equivalent[indices] <= 10)
                selected = np.flatnonzero(valid)
                if len(selected):
                    convergence = wool_midget_profile(equivalent[indices][selected])
                    radii = np.where(convergence["private_line_branch"], .001, convergence["center_radius_mm"])
                    support = np.minimum(.04, radii * np.sqrt(np.log(100.)))
                    ready_pre = np.flatnonzero(known & opl_supported)
                    candidate, audits = bounded_matrix(pedicle[indices][ready_pre], pedicle[indices][selected],
                        convergence["effective_contact_target"], support_mm=support, gaussian_scale=radii)
                    parent_ids = tuple(bc.global_cell_ids[j] for j in selected)
                    candidate = enforce_midget_parent_matrix(candidate,
                        tuple(bc.global_cell_ids[j] for j in ready_pre), parent_ids,
                        convergence["effective_contact_target"], pre_positions=pedicle[indices][ready_pre],
                        post_positions=pedicle[indices][selected], support_mm=support)
                    uncertain = unknown_near(pedicle[indices][selected], support)
                    for k, j in enumerate(selected):
                        target = convergence["effective_contact_target"][k]
                        rows[j] = dict(audits[k], parent_bc_id=bc.global_cell_ids[j],
                            temporal_equivalent_mm=float(equivalent[indices][j]),
                            source_domain="0.25..10 temporal-equivalent mm",
                            private_or_minimum_one_convention=bool(target <= 1))
                        matrix[j, ready_pre] = candidate.getrow(k).toarray()
                        statuses[j] = ("READY_PROJECT_PRIVATE_OR_MINIMUM_ONE" if target <= 1 else
                            "BLOCKED_UNKNOWN_CONE_SUBTYPE_IN_CONVERGENCE_SUPPORT" if uncertain[k] else
                            "INCOMPLETE_MIDGET_CONVERGENCE_FINITE_SUPPORT" if audits[k]["missing_degree"] else
                            "READY_PROJECT_WOOL_CONVERGENCE")
                for j in np.flatnonzero(~opl_supported):
                    statuses[j] = "BLOCKED_MIDGET_OPL_ABOVE_10MM"
            put_contact(bc_name, rgc_name, matrix, "IPL", bc.coordinate("ipl_preterminal"),
                        rgc.coordinate("ipl_post_arbor"), statuses, rows)

    if count:
        lo, hi = pedicle.min(0) - .04, pedicle.max(0) + .04
        bounds = (lo[0], hi[0], lo[1], hi[1])
        sites = [("cone_site:" + cell, point_rf, point_p)
                 for cell, point_rf, point_p in zip(cones.global_cell_ids, rf, pedicle)]
        for ac_name in ("local_feedforward_AC", "local_crossover_AC"):
            _, ac_rf, points, site_ids = local_typed_sites(sites, ac_name, bounds, profile_hash=digest)
            ac = make_table(ac_name, site_ids, ac_rf, points)
            coordinates = dict(ac.coordinates, ipl_post_arbor=_quantity(points, "dendritic_position", "IPL"))
            tables[ac_name] = CanonicalCellTable(ac.atlas_id, ac.atlas_version, ac.source_version,
                ac.population, ac.source_keys, coordinates, ac.reference_id)
    if include_widefield_ac:
        if count:
            keys, points, _ = widefield_grid(bounds, max_cells=max_effective_cells)
        else:
            keys, points = (), np.empty((0, 2))
        ac = make_table("widefield_AC", keys, None, points)
        tables["widefield_AC"] = CanonicalCellTable(ac.atlas_id, ac.atlas_version, ac.source_version,
            ac.population, ac.source_keys, dict(ac.coordinates,
                ipl_post_arbor=_quantity(points, "dendritic_position", "IPL")), ac.reference_id)

    for ac_name, bc_name, budget in (("local_feedforward_AC", "diffuse_BC_ON", 4),
                                     ("local_crossover_AC", "diffuse_BC_OFF", 4),
                                     ("widefield_AC", "diffuse_BC_ON", 16)):
        if ac_name not in tables:
            continue
        if bc_name not in tables:
            blockers.append({"population": ac_name, "reason": "EFFECTIVE_DIFFUSE_BC_DRIVE_NOT_SUPPLIED",
                "geometry_status": "COMPUTABLE_PROJECT_TYPED_SITES"})
            continue
        input_region = "ON" if bc_name.endswith("ON") else "OFF"
        if input_region not in configuration["strata"][ac_name]["input"]:
            raise ContractError("Existing effective BC->AC IPL stratum mismatch")
        pool_contact(bc_name, ac_name, tables[bc_name].coordinate("ipl_preterminal"),
                     tables[ac_name].coordinate("ipl_post_arbor"), budget,
                     upstream_statuses=contacts[cones.population + "->" + bc_name]["row_status"])

    for rgc_name, ac_name in (("midget_RGC_ON", "local_crossover_AC"),
                              ("midget_RGC_OFF", "local_feedforward_AC")):
        if ac_name not in tables:
            continue
        output_region = "ON" if rgc_name.endswith("ON") else "OFF"
        if output_region not in configuration["strata"][ac_name]["output"]:
            raise ContractError("Existing effective AC->midget RGC IPL stratum mismatch")
        input_route = ("diffuse_BC_OFF" if ac_name == "local_crossover_AC" else "diffuse_BC_ON") + "->" + ac_name
        upstream = contacts[input_route]["row_status"] if input_route in contacts else (
            "BLOCKED_EFFECTIVE_AC_DRIVE_NOT_SUPPLIED",) * len(tables[ac_name].global_cell_ids)
        pool_contact(ac_name, rgc_name, tables[ac_name].coordinate("ipl_preterminal"),
                     tables[rgc_name].coordinate("ipl_post_arbor"), 1, upstream_statuses=upstream)

    if diffuse_seed_cone_ids is None:
        blockers.append({"populations": ["diffuse_BC_ON", "diffuse_BC_OFF"],
            "reason": "EXPLICIT_CANONICAL_EFFECTIVE_DIFFUSE_SAMPLING_NOT_SUPPLIED"})
    blockers.append({"populations": ["parasol_RGC_ON", "parasol_RGC_OFF"],
        "reason": "INDEPENDENT_CANONICAL_EFFECTIVE_RGC_SEEDS_NOT_SUPPLIED"})
    metadata = {"adapter_id": ADAPTER, "IPL_profile_sha256": digest, "cone_table_hash": cones.table_hash,
        "evidence_class": EVIDENCE, "coordinate_rule": "supplied OPL pedicle co-location as effective IPL point; not measured morphology",
        "soma": "ABSENT; old McGregor soma/split/inverse is not applied to supplied new pedicles",
        "temporal_equivalence_source_id": temporal_equivalence_source_id,
        "Wool_scope": "source fit .25..10 temporal-equivalent mm; central private-line and minimum1 are project conventions",
        "unknown_cone_ids": [cell for cell, label in zip(cones.global_cell_ids, labels) if label == "UNKNOWN"],
        "diffuse_sampling_seed_cone_ids": [] if diffuse_seed_cone_ids is None else list(seed_ids),
        "diffuse_sampling_rule": "explicit outer-provider effective sampling IDs; no subtype density inferred",
        "effective_pooling_budgets": {"cone_to_diffuse_BC": 8, "BC_to_local_AC": 4,
                                     "BC_to_widefield_AC": 16, "local_AC_to_midget_inhibition": 1},
        "widefield_sampling": "fixed global IPL grid40um/3, origin0; RF and soma absent" if include_widefield_ac else "NOT_REQUESTED",
        "subtypes": labels.tolist(), "HC_gap": "NOT_IMPLEMENTED_HERE", "dynamics": "NOT_RUN_OR_CHANGED",
        "boundary": "finite supplied candidates; support completeness and recurrent exterior closure not established",
        "global_coverage_claim": False}
    return EffectiveInnerLayout(MappingProxyType(tables), MappingProxyType(parents),
                                MappingProxyType(contacts), tuple(blockers), metadata)
