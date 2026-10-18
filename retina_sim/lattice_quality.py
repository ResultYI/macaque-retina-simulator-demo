"""Static diagnostics of saved coordinates, with the pinned mesh-q definition.

Fresh SciPy/Qhull triangulation is not the MATLAB history triangulation and
does not recreate its termination state. No relaxation or neural solve occurs.
"""
import numpy as np
from scipy.spatial import Delaunay
from .contracts import ContractError

def triangle_quality(points,triangles):
    p=np.asarray(points,dtype=float)[np.asarray(triangles,dtype=int)]
    a=np.linalg.norm(p[:,0]-p[:,1],axis=1);b=np.linalg.norm(p[:,0]-p[:,2],axis=1);c=np.linalg.norm(p[:,1]-p[:,2],axis=1)
    if np.any(a*b*c<=0):raise ContractError('Duplicate triangle vertices in frozen geometry')
    return (b+c-a)*(c+a-b)*(a+b-c)/(a*b*c)

def frozen_lattice_quality(points_mm,regions):
    xy=np.asarray(points_mm,dtype=float)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<3 or not np.all(np.isfinite(xy)):
        raise ContractError('At least three finite saved 2D positions required')
    triangles=Delaunay(xy).simplices;q=triangle_quality(xy,triangles);centers=xy[triangles].mean(1)
    results=[]
    for label,bounds in [('saved_point_convex_hull',None)]+list(regions.items()):
        if bounds is None:values=q
        else:
            x0,x1,y0,y1=bounds
            values=q[(centers[:,0]>=x0)&(centers[:,0]<x1)&(centers[:,1]>=y0)&(centers[:,1]<y1)]
        results.append({'region':label,'centroid_bounds_mm':bounds,'triangles':len(values),
            'q_percentiles':dict(zip(('0.8','3','7','15','50','99.8'),np.percentile(values,[.8,3,7,15,50,99.8]).tolist())) if len(values) else {},
            'q_min':float(values.min()) if len(values) else None})
    return {'method':'Fresh SciPy Delaunay on immutable full saved points; q=2r/R from pinned meshQuality.m',
        'regional_rule':'Select triangle centroids after full triangulation; never triangulate an isolated cropped ROI',
        'upstream_log_quality_is_not_recomputed_coordinate_quality':True,
        'upstream_history_triangulation_or_stop_state_reproduced':False,
        'upstream_ellipse_centroid_filter_applied':False,
        'algorithm_threshold_changed':False,'physiological_acceptance':False,'regions':results}
