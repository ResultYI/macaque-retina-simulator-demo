"""Schein's digitized IS-to-pedicle curves, without an unsupported central/tail fill."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .contracts import ContractError, Quantity, finite_array


SCHEIN_SOURCE_PACKAGE_SHA256 = "3bcbf83d595a2dd6cc08fe5e3832c6351d93f881e0e8ef6fdfaef4a1c1ce55f6"
SCHEIN_CURVE_SHA256 = "b1444893bcb7a4f811718180609f53acce16e1192030ef08e6d756faf6e27323"
_SOURCE_DIR = Path("data/source_packages/20261012/Schein1988_Henle_source_verified")
_FILES = {
    "schein1988_fig15_digitized_curves.json": SCHEIN_CURVE_SHA256,
    "provenance.json": "25623970569d3f978b80102585521533a93ad9ca25f5204f51510819cd2829bd",
    "figure15_original_crop.png": "5a76d75f3cae74ce733c1935e3185fad435f549735e437df6860874d490b2aba",
    "textual_anchors.json": "5288d094cbe232bc11c16a2082309aa6b63b6200f23704b4a5ed735d9b208d76",
}
_MERIDIANS = ("nasal", "superior", "temporal", "inferior")
_RF_TRANSFER = "rf_equals_inner_segment_mm_v1"


@dataclass(frozen=True)
class HenleCurve:
    inner_segment_radius_mm: np.ndarray
    displacement_mm: np.ndarray

    @property
    def domain_mm(self):
        return (float(self.inner_segment_radius_mm[0]), float(self.inner_segment_radius_mm[-1]))


class ScheinHenleDisplacement:
    """An opt-in single-eye curve provider; axes are +x nasal and +y superior.

    The four cardinal curves are author-derived, then digitized. Linear radial
    interpolation and interpolation between retinal meridians are project rules.
    RF-to-IS equality requires the named transfer; no soma relation is supplied.
    """

    def __init__(self, root, *, angular_policy="piecewise_linear_cardinal", rf_transfer=None):
        if angular_policy != "piecewise_linear_cardinal":
            raise ContractError("Schein requires explicit piecewise_linear_cardinal angular policy")
        if rf_transfer not in (None, _RF_TRANSFER):
            raise ContractError(f"Unknown Schein RF transfer: {rf_transfer!r}")
        self.angular_policy = angular_policy
        self.rf_transfer = rf_transfer
        self.source_directory = Path(root) / _SOURCE_DIR
        raw_files = {}
        for name, expected in _FILES.items():
            try:
                raw = (self.source_directory / name).read_bytes()
            except OSError as exc:
                raise ContractError(f"Schein source asset unavailable: {name}: {exc}") from exc
            actual = hashlib.sha256(raw).hexdigest()
            if actual != expected:
                raise ContractError(f"Schein source identity mismatch for {name}: {actual} != {expected}")
            raw_files[name] = raw
        payload = json.loads(raw_files["schein1988_fig15_digitized_curves.json"])
        self._provenance = json.loads(raw_files["provenance.json"])
        self._textual_anchors = json.loads(raw_files["textual_anchors.json"])
        if (not payload["author_derived"] or not payload["digitized"]
                or payload["raw_cell_measurement"] or len(payload["data"]) != 343):
            raise ContractError("Schein must retain its digitized author-derived curve identity")
        curves = {}
        for meridian in _MERIDIANS:
            rows = [row for row in payload["data"] if row["meridian"] == meridian]
            r = finite_array([row["is_eccentricity_mm"] for row in rows])
            d = finite_array([row["henle_radial_displacement_mm"] for row in rows])
            p = finite_array([row["pedicle_eccentricity_mm"] for row in rows])
            if (len(r) < 2 or np.any(np.diff(r) <= 0) or np.any(d < 0)
                    or np.any(np.diff(p) <= 0) or not np.allclose(p, r + d, rtol=0, atol=1e-12)):
                raise ContractError(f"Invalid or folded Schein {meridian} IS-to-pedicle curve")
            curves[meridian] = HenleCurve(r, d)
        self.curves = MappingProxyType(curves)

    @property
    def provenance(self):
        return {
            "source_id": "Schein1988",
            "source_version": "fig15_digitization_20261012",
            "package_sha256": SCHEIN_SOURCE_PACKAGE_SHA256,
            "verified_asset_hashes": dict(_FILES),
            "source": dict(self._provenance["source"]),
            "species": self._provenance["species"],
            "specimen": self._provenance["model_specimen"],
            "evidence_status": "author_derived_then_digitized",
            "raw_cell_measurement": False,
            "curve_point_count": 343,
            "reference_axes": {"positive_x": "nasal", "positive_y": "superior", "origin": "foveal_center"},
            "source_domain": {name: list(curve.domain_mm) for name, curve in self.curves.items()},
            "source_frame": "anatomical_retina",
            "source_coordinate": "cone_inner_segment_position",
            "application_domain": "intersection of required cardinal source domains; no extrapolation",
            "output_frame": "OPL",
            "output_coordinate": "pedicle_position",
            "unit": "mm",
            "radial_interpolation_rule": "linear_between_digitized_curve_samples_v1",
            "angular_interpolation_rule": self.angular_policy,
            "rf_transfer": self.rf_transfer,
            "uncertainty": dict(self._provenance["uncertainty"]),
            "coordinate_roles": dict(self._provenance["coordinate_roles"]),
            "textual_anchors_not_curve_fill": [dict(row) for row in self._textual_anchors],
            "source_statement": "Fig. 15: pedicle radius minus cone inner-segment radius, derived by equating cumulative counts; retinal-plane radial displacement.",
            "source_statement_kind": "extraction_description_not_verbatim_article_quote",
            "discrete_mapping_count_preserved": True,
            "discrete_mapping_jacobian_applied": False,
            "physiological_validation": False,
        }

    def _input_xy(self, positions):
        if not isinstance(positions, Quantity):
            raise ContractError("Schein input requires a typed Quantity")
        if positions.meaning == "functional_position":
            if self.rf_transfer != _RF_TRANSFER:
                raise ContractError(f"Schein RF coordinates require explicit project transfer {_RF_TRANSFER}")
            xy = positions.require("mm", "functional_position", frame="functional_visual_field")
            transfer = self.rf_transfer
        else:
            xy = positions.require("mm", "cone_inner_segment_position", frame="anatomical_retina")
            transfer = None
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ContractError("Schein positions must have shape (n, 2)")
        return xy, transfer

    @staticmethod
    def _weights(xy):
        angle = np.mod(np.arctan2(xy[1], xy[0]), 2 * np.pi)
        quadrant = angle / (np.pi / 2)
        nearest = int(np.rint(quadrant))
        if abs(quadrant - nearest) <= 1e-12:
            return ((_MERIDIANS[nearest % 4], 1.0),)
        lower = int(np.floor(quadrant))
        fraction = float(quadrant - lower)
        return ((_MERIDIANS[lower], 1 - fraction), (_MERIDIANS[(lower + 1) % 4], fraction))

    def coverage(self, positions):
        """Per-point support, including missing data separately from a numerical zero."""
        xy, transfer = self._input_xy(positions)
        points = []
        for i, position in enumerate(xy):
            r = float(np.linalg.norm(position))
            weights = self._weights(position)
            missing = [name for name, _ in weights if not self.curves[name].domain_mm[0] - 1e-12 <= r <= self.curves[name].domain_mm[1] + 1e-12]
            point = {"index": i, "inner_segment_radius_mm": r,
                     "meridian_weights": dict(weights), "rf_transfer": transfer}
            if missing:
                point.update(status="unsupported", displacement_mm=None,
                             reason="missing digitized curve support: " + ", ".join(missing))
            else:
                d = sum(weight * float(np.interp(r, self.curves[name].inner_segment_radius_mm,
                                               self.curves[name].displacement_mm)) for name, weight in weights)
                radial_exact = all(np.any(np.isclose(self.curves[name].inner_segment_radius_mm,
                                                    r, rtol=0, atol=1e-12)) for name, _ in weights)
                exact = len(weights) == 1 and radial_exact
                point.update(status="source_supported" if exact and transfer is None else "project_approximation",
                             displacement_mm=d, reason="digitized cardinal sample" if exact else "project numerical interpolation",
                             radial_interpolation=not radial_exact, angular_interpolation=len(weights) > 1)
            points.append(point)
        return {"provenance": self.provenance, "points": points,
                "supported": all(point["status"] != "unsupported" for point in points),
                "input_cell_count": len(xy), "query_frame": positions.coordinate_frame,
                "query_coordinate": positions.meaning}

    @staticmethod
    def _require_supported(report):
        missing = [p for p in report["points"] if p["status"] == "unsupported"]
        if missing:
            details = "; ".join(f"point {p['index']} r={p['inner_segment_radius_mm']:.6g} mm: {p['reason']}" for p in missing[:5])
            raise ContractError(f"Schein IS-to-pedicle mapping unsupported ({len(missing)} points): {details}; no central bridge or tail fill")

    def apply(self, positions):
        xy, _ = self._input_xy(positions)
        report = self.coverage(positions)
        self._require_supported(report)
        r = np.linalg.norm(xy, axis=1)
        d = np.array([point["displacement_mm"] for point in report["points"]])
        mapped = xy * ((r + d) / r)[:, None]
        evidence = ("Schein1988_digitized_author_derived" if all(p["status"] == "source_supported" for p in report["points"])
                    else "Schein1988_digitized_author_derived_project_interpolation_or_RF_transfer")
        return Quantity(mapped, "mm", "pedicle_position", evidence, "OPL")

    def inverse(self, positions):
        """Invert the same monotone radial map, preserving the input point count."""
        if not isinstance(positions, Quantity):
            raise ContractError("Schein inverse requires a typed Quantity")
        xy = positions.require("mm", "pedicle_position", frame="OPL")
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ContractError("Schein pedicle positions must have shape (n, 2)")
        mapped = np.empty_like(xy)
        for i, position in enumerate(xy):
            weights = self._weights(position)
            lo = max(self.curves[name].domain_mm[0] for name, _ in weights)
            hi = min(self.curves[name].domain_mm[1] for name, _ in weights)
            knots = np.unique(np.r_[lo, hi, np.concatenate([
                self.curves[name].inner_segment_radius_mm for name, _ in weights])])
            knots = knots[(knots >= lo) & (knots <= hi)]
            pedicle_r = knots + sum(weight * np.interp(knots, self.curves[name].inner_segment_radius_mm,
                                                      self.curves[name].displacement_mm) for name, weight in weights)
            r = float(np.linalg.norm(position))
            if not pedicle_r[0] - 1e-12 <= r <= pedicle_r[-1] + 1e-12:
                raise ContractError(f"Schein inverse point {i} r={r:.6g} mm outside mapped source domain [{pedicle_r[0]:.6g}, {pedicle_r[-1]:.6g}] mm")
            is_r = float(np.interp(r, pedicle_r, knots))
            mapped[i] = position * (is_r / r)
        return Quantity(mapped, "mm", "cone_inner_segment_position",
                        "Schein1988_project_inverse_of_digitized_author_derived_curve", "anatomical_retina")
