"""Explicit anatomical reference diameter, sampling kernel and collecting area.

No lattice generation or optical PSF is performed here. Profile knots carry
their own measured/derived/engineering status; spacing is never read as IS
diameter by this adapter. A profile may explicitly declare an engineering
spacing-based constraint, but must supply that derivation and identity.
"""
import numpy as np
from .contracts import ContractError,canonical_hash

SCHEMA='macaque-cone-aperture-eccentricity-1'

def resolve_apertures(cone_ids,cone_types,positions_mm,profile):
    if profile.get('schema')!=SCHEMA:raise ContractError('Explicit macaque aperture profile required')
    ids=tuple(cone_ids);types=tuple(cone_types);xy=np.asarray(positions_mm,float)
    if (not ids or len(set(ids))!=len(ids) or len(types)!=len(ids) or set(types)-{'L','M','S'}
            or xy.shape!=(len(ids),2) or not np.isfinite(xy).all()):raise ContractError('Cone identity/type/position mismatch')
    mag=float(profile['coordinate_binding']['microns_per_degree'])
    if not np.isfinite(mag) or mag<=0:raise ContractError('Explicit positive project retinal magnification required')
    ecc=np.linalg.norm(xy,axis=1)*1000/mag
    if np.any(ecc<0) or np.any(ecc>float(profile['supported_eccentricity_degrees'][1])+1e-12):
        raise ContractError('Cone outside declared functional aperture range')
    meridian=np.degrees(np.arctan2(xy[:,1],xy[:,0]))
    diam=np.empty(len(ids));tags=[];sources=[]
    for i,e in enumerate(ecc):
        key='nasal' if xy[i,0]>=0 else 'temporal'
        data=profile['meridians'][key];knots=data['knots']
        xx=np.array([k['eccentricity_degrees'] for k in knots],float);yy=np.array([k['reference_diameter_um'] for k in knots],float)
        if (len(xx)<2 or not np.all(np.diff(xx)>0) or np.any(yy<=0) or not np.isfinite(yy).all()
                or xx[0]>e or xx[-1]<e):raise ContractError('Explicit bracketing diameter knots required')
        diam[i]=np.interp(e,xx,yy)
        j=min(max(int(np.searchsorted(xx,e,side='right'))-1,0),len(xx)-2)
        tags.append('engineering' if any('engineering' in knots[k]['evidence_class'] for k in (j,j+1)) else 'source_derived_interpolation')
        sources.append('|'.join(sorted(set(knots[k]['source_id'] for k in (j,j+1)))))
    ratio=float(profile['kernel']['sigma_per_reference_diameter'])
    if profile['kernel']['shape']!='Gaussian' or not np.isfinite(ratio) or ratio<=0:
        raise ContractError('Explicit Gaussian width convention required')
    sigma=ratio*diam;radius=np.sqrt(2)*sigma
    # ISETBio Gaussian convention: characteristic-radius area, multiplied once.
    area=np.pi*(radius*1e-6)**2
    zone_width=float(profile['kernel']['zone_diameter_width_um'])
    if not np.isfinite(zone_width) or zone_width<=0:raise ContractError('Explicit zone width required')
    zone_diam=np.round(diam/zone_width)*zone_width
    if np.any(zone_diam<=0):raise ContractError('Nonpositive quantized aperture')
    return dict(cone_ids=np.asarray(ids),cone_types=np.asarray(types),positions_mm=xy.copy(),
        eccentricity_degrees=ecc,meridian_degrees=meridian,reference_diameter_um=diam,
        sigma_um=sigma,characteristic_radius_um=radius,FWHM_diameter_um=2*np.sqrt(2*np.log(2))*sigma,
        effective_area_m2=area,anatomical_reference_disk_area_m2=np.pi*(diam*.5e-6)**2,
        zone_diameter_um=zone_diam,evidence_class=np.array(tags),source_id=np.array(sources),
        profile_id=profile['profile_id'],profile_sha256=canonical_hash(profile),
        anatomical_diameter_is_independently_measured=False,
        unit_integral_kernel=True,area_applied_once=True)
