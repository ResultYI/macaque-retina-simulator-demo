"""Explicit owner/output extent, independent of retained dependency assets.

An output limit never licenses cutting a recurrent graph or renormalizing it.
Historical geometries remain readable. Only new runtime owner requests are gated.
"""
from pathlib import Path
import hashlib
import json
import math

from .contracts import ContractError


PROFILE = "configurations/natural_movie_region_scope_v1.json"


def read_owner_scope(root):
    root = Path(root).resolve()
    path = root / PROFILE
    if not path.is_file():
        return None
    raw = path.read_bytes()
    scope = json.loads(raw)
    if scope.get("schema_version") != "natural-movie-region-scope-1.0":
        raise ContractError("Unknown explicit owner spatial-scope schema")
    radius = scope.get("maximum_owner_eccentricity_deg")
    scale = scope.get("project_mm_per_degree")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or v <= 0 for v in (radius, scale)):
        raise ContractError("Owner scope requires positive finite degree/mm values")
    if (scope.get("coordinate_role") != "rf" or
            scope.get("coordinate_frame") != "functional_visual_field" or
            scope.get("crop_recurrent_dependencies") is not False):
        raise ContractError("RF owner scope must retain actual recurrent dependencies")
    expected = radius * scale
    if not math.isclose(scope.get("maximum_owner_radius_mm", -1), expected,
                        rel_tol=0., abs_tol=1e-12):
        raise ContractError("Owner degree/mm limits disagree")
    binding = {"path": PROFILE, "sha256": hashlib.sha256(raw).hexdigest()}
    return dict(scope, binding=binding)


def enforce_owner_scope(root, bounds):
    """Reject a new recording rectangle beyond the declared output disk.

    The rectangle must be wholly supported. Dependency/halo positions are not
    falsely presented as additional output coverage and are never clipped here.
    """
    scope = read_owner_scope(root)
    if scope is None:
        return None
    if len(bounds) != 4 or any(not math.isfinite(float(v)) for v in bounds):
        raise ContractError("Owner bounds require four finite retinal-mm values")
    x0, x1, y0, y1 = map(float, bounds)
    if x1 <= x0 or y1 <= y0:
        raise ContractError("Owner recording rectangle must have positive area")
    requested = max(math.hypot(x, y) for x in (x0, x1) for y in (y0, y1))
    if requested > scope["maximum_owner_radius_mm"] + 1e-12:
        raise ContractError(
            "OWNER_SPATIAL_SCOPE_EXCEEDED: requested %.9g mm; maximum %.9g mm "
            "(%.9g degrees under the explicit project conversion); retained "
            "source assets do not expand the active output domain" %
            (requested, scope["maximum_owner_radius_mm"],
             scope["maximum_owner_eccentricity_deg"]))
    return {"scope": scope["binding"],
            "maximum_owner_eccentricity_deg": scope["maximum_owner_eccentricity_deg"],
            "maximum_owner_radius_mm": scope["maximum_owner_radius_mm"],
            "requested_owner_max_radius_mm": requested,
            "dependency_positions_clipped": False,
            "central_parameter_defaults_changed": False}
