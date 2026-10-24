"""Input-only scene multipliers for the proposed fixed-reference interpretation.

No ISET, optical image, native model, GPU, or fitted gain is constructed here.
Cone scene multipliers MUST pass through the existing ISET optics/capture: they
are not native cone drive and must never be substituted for captured R*/window.
The independent rod rate has only the explicitly selected engineering meaning.
"""
import numpy as np
from .handoff import approved_policy


def _sample_codes(frame, project_xy_mm, registration, outside_code):
    frame = np.asarray(frame)
    xy = np.asarray(project_xy_mm, dtype=float)
    if (frame.dtype != np.uint8 or frame.shape != (256, 256)
            or xy.ndim < 2 or xy.shape[-1] != 2 or not np.isfinite(xy).all()):
        raise ValueError("Stored uint8[256,256] and finite project x/y scene-grid coordinates required")
    pixels = (xy / registration["project_mm_per_degree"] - np.asarray(registration["teacher_rf_deg"]))
    pixels = pixels * registration["pixels_per_degree"] + np.asarray(registration["patch_center_xy"])
    inside = np.all((pixels >= 0) & (pixels <= 255), axis=-1)
    # Clipping only protects array indices. Entire outside samples use the
    # explicitly proposed exterior code; no edge replication into the exterior.
    clipped = np.clip(pixels, 0, 255)
    x, y = clipped[..., 0], clipped[..., 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, 255), np.minimum(y0 + 1, 255)
    a, b = x - x0, y - y0
    value = ((1-a)*(1-b)*frame[y0, x0] + a*(1-b)*frame[y0, x1]
             + (1-a)*b*frame[y1, x0] + a*b*frame[y1, x1])
    return np.where(inside, value, outside_code)


class LinearGrayScene:
    """Future input adapter, gated on the exact externally recorded approval."""
    def __init__(self, project_root, approval_receipt):
        _, objects = approved_policy(project_root, approval_receipt)
        self.policy = objects["input_policy"]
        self.registration = self.policy["spatial_registration"]

    def reference_multipliers(self, stored_frame, scene_grid_project_xy_mm):
        """Dimensionless factors multiplying the ONE existing reference spectrum.

        Grid points describe the ISET scene support, not only the owned cones;
        optics, apertures and the original cone capture must still be applied.
        Original 150 Hz frame selection is supplied by Gray8Chunks, without a
        separate gamma decode, temporal interpolation, or background replacement.
        """
        codes = _sample_codes(stored_frame, scene_grid_project_xy_mm,
                              self.registration, self.policy["outside"]["proposed_gray8_code"])
        return codes / 255. / self.policy["spectrum"]["reference_linear_value"]

    def independent_rod_rates(self, stored_frame, rod_project_xy_mm):
        factor = self.reference_multipliers(stored_frame, rod_project_xy_mm)
        return factor * self.policy["rod"]["proposed_reference_rate_Rstar_per_s"]
