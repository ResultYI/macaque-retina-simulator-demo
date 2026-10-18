"""Disk masks on the same immutable global atlas; no per-ROI generation."""
from dataclasses import dataclass,replace
import numpy as np
from .atlas import AtlasROI,_inside
from .contracts import ContractError

@dataclass(frozen=True)
class DiskAtlasROI(AtlasROI):
    owner_radius_mm: float = 3.
    query_radius_mm: float = 3.

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.owner_radius_mm) or not np.isfinite(self.query_radius_mm) or not 0<self.owner_radius_mm<=self.query_radius_mm:
            raise ContractError('Finite positive disk owner/query radii required')
    @staticmethod
    def _mask(points,bounds,radius):
        xy=np.asarray(points);mask=_inside(xy,bounds)
        # The complete closed disk includes its cardinal boundary points.
        if tuple(bounds)==(-radius,radius,-radius,radius):
            mask=(np.abs(xy[:,0])<=radius)&(np.abs(xy[:,1])<=radius)
        return mask&(np.sum(xy**2,axis=1)<=radius**2)
    def owner_mask(self,points):return self._mask(points,self.owner_bounds_mm,self.owner_radius_mm)
    def query_mask(self,points):return self._mask(points,self.query_bounds_mm,self.query_radius_mm)
    def maximum_owner_radius(self):return min(super().maximum_owner_radius(),self.owner_radius_mm)
    def with_query_bounds(self,bounds):return replace(self,query_bounds_mm=tuple(bounds))

