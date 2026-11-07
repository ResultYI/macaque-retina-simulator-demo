"""Native ISETBio apertures bound to actual Cone identities and complete support."""
import numpy as np
from scipy.spatial import cKDTree
from .contracts import ContractError, canonical_hash
from .native_cone_apertures import PROFILE, native_apertures

SCHEMA = PROFILE['schema']

def resolve_apertures(cone_ids, cone_types, positions_mm, profile, *, support_positions_mm):
    if profile != PROFILE:
        raise ContractError('Use the native ISETBio aperture profile; custom diameter curves are retired')
    ids=tuple(cone_ids);types=tuple(cone_types);xy=np.asarray(positions_mm,float)
    if (not ids or len(set(ids))!=len(ids) or len(types)!=len(ids) or set(types)-{'L','M','S'}
            or xy.shape!=(len(ids),2) or not np.isfinite(xy).all()):
        raise ContractError('Cone identity/type/position mismatch')
    support=np.asarray(support_positions_mm,float)
    distance,rows=cKDTree(support).query(xy,k=1)
    if not np.array_equal(support[rows],xy):
        raise ContractError('Every selected Cone must exist exactly in the complete aperture support')
    diameters,areas=native_apertures(support)
    diameters=diameters[rows];areas=areas[rows]
    return dict(cone_ids=np.asarray(ids),cone_types=np.asarray(types),positions_mm=xy.copy(),
        eccentricity_degrees=np.linalg.norm(xy,axis=1)*1000/223,
        reference_diameter_um=diameters,effective_area_m2=areas,
        zone_diameter_um=diameters.copy(),
        evidence_class=np.full(len(ids),'native_ISETBio_human_borrowed'),
        source_id=np.full(len(ids),'cMosaic.computeConeApertures'),
        profile_id=PROFILE['profile_id'],profile_sha256=canonical_hash(PROFILE),
        anatomical_diameter_is_independently_measured=False,
        unit_integral_kernel=True,area_applied_once=True)
