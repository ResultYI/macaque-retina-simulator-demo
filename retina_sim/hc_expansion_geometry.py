"""Batch evaluation of the existing bounded Henle rule; no new displacement law."""
import numpy as np
from .contracts import ContractError


from .disk_source_geometry import schein_array as bounded_pedicles


def native_td_step(positions_is_mm, background_td, *, center_mm, inner_radius_mm,
                   outer_radius_mm, contrast, time_seconds, onset_seconds, offset_seconds):
    """White native-Td retinal diagnostic stimulus, not a display/optics capture.

    Every supplied cone gets its background, including the dynamic halo.
    Positions are IS, never pedicles; annuli use [inner, outer).
    """
    xy=np.asarray(positions_is_mm,float);center=np.asarray(center_mm,float)
    if xy.ndim!=2 or xy.shape[1]!=2 or center.shape!=(2,) or not np.isfinite(xy).all() or not np.isfinite(center).all():
        raise ContractError('Explicit finite IS stimulus coordinates required')
    values=[background_td,inner_radius_mm,outer_radius_mm,contrast,time_seconds,onset_seconds,offset_seconds]
    if not np.isfinite(values).all() or background_td<=0 or not 0<=inner_radius_mm<outer_radius_mm or contrast < -1 or not onset_seconds<offset_seconds:
        raise ContractError('Invalid native-Td stimulus parameters')
    output=np.full(len(xy),float(background_td))
    if onset_seconds<=time_seconds<offset_seconds:
        r=np.linalg.norm(xy-center,axis=1)
        output[(r>=inner_radius_mm)&(r<outer_radius_mm)]*=1+contrast
    return output
