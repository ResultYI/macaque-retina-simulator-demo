"""Rasterize fixed physical cone apertures once; never change capture values."""
from pathlib import Path
import numpy as np
from scipy.io import savemat


def prepare_raster(path, xy_mm, diameters_um, bounds_mm, samples=512, supersampling=4):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    xy=np.asarray(xy_mm);diameters=np.asarray(diameters_um).reshape(-1)
    assert xy.shape==(len(diameters),2) and len(xy)<2**32-1
    assert np.all(np.isfinite(xy)) and np.all(np.isfinite(diameters)) and np.all(diameters>0)
    bounds=np.asarray(bounds_mm).reshape(-1);size=samples*supersampling
    assert bounds.shape==(4,) and bounds[1]>bounds[0] and bounds[3]>bounds[2]
    labels=np.zeros((size,size),dtype=np.uint16 if len(xy)<65535 else np.uint32)
    dx=(bounds[1]-bounds[0])/size;dy=(bounds[3]-bounds[2])/size
    for i,(position,diameter) in enumerate(zip(xy,diameters)):
        radius=diameter/2000
        left=max(0,int(np.ceil((position[0]-radius-bounds[0])/dx-.5)))
        right=min(size-1,int(np.floor((position[0]+radius-bounds[0])/dx-.5)))
        top=max(0,int(np.ceil((bounds[3]-position[1]-radius)/dy-.5)))
        bottom=min(size-1,int(np.floor((bounds[3]-position[1]+radius)/dy-.5)))
        if left>right or top>bottom:continue
        x=bounds[0]+(np.arange(left,right+1)+.5)*dx-position[0]
        y=bounds[3]-(np.arange(top,bottom+1)+.5)*dy-position[1]
        mask=y[:,None]**2+x[None,:]**2<=radius**2
        labels[top:bottom+1,left:right+1][mask]=i+1
    assert int(labels.max())<=len(xy) and np.any(labels==0)
    savemat(path,dict(cone_ids=labels,supersampling=float(supersampling),
                      bounds_mm=bounds,diameters_um=diameters),do_compression=True)
