"""Explicit approved static exterior extension; original <=6 degree table is frozen."""
import csv
from pathlib import Path
import numpy as np
from .contracts import ContractError, canonical_hash
from .gin69_frozen_aperture import resolve_apertures


def outer_apertures(root, xy, profile):
    xy=np.asarray(xy,dtype=np.float64)
    ecc=np.linalg.norm(xy,axis=1)*1000/float(profile['coordinate_binding']['microns_per_degree'])
    diameter=np.empty(len(xy));rows=list(csv.DictReader((Path(root)/'data/published_code/packer_cone_density.csv').open()))
    for mer,keep in [('nasal',xy[:,0]>=0),('temporal',xy[:,0]<0)]:
        knots=profile['meridians'][mer]['knots']
        xx=np.array([k['eccentricity_degrees'] for k in knots]);yy=np.array([k['reference_diameter_um'] for k in knots])
        inside=keep&(ecc<=6)
        diameter[inside]=np.interp(ecc[inside],xx,yy)
        table=[r for r in rows if r['meridian']==mer]
        rx=np.array([float(r['eccentricity_mm']) for r in table]);rho=np.array([float(r['cone_density_cones_per_mm2']) for r in table])
        outside=keep&(ecc>6);rr=.223*ecc[outside]
        if np.any(rr<rx[0]) or np.any(rr>rx[-1]):raise ContractError('Exterior exceeds frozen Packer density support')
        spacing=lambda rr:1000*np.sqrt(2/(np.sqrt(3)*np.interp(rr,rx,rho)))
        diameter[outside]=yy[-1]+.79*(spacing(rr)-spacing(6*.223))
    sigma=.204*diameter
    return dict(reference_diameter_um=diameter,zone_diameter_um=np.round(diameter/.05)*.05,
                effective_area_m2=np.pi*(np.sqrt(2)*sigma*1e-6)**2,eccentricity_deg=ecc,
                identity=canonical_hash(dict(profile=profile,extension='D6+.79*(s(e)-s6);linear density;static only')))
