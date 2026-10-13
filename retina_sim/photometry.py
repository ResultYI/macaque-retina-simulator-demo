from dataclasses import dataclass
import numpy as np
from .contracts import ContractError, MissingCalibrationError, Quantity

PLANCK_J_S = 6.62607015e-34  # exact SI defining constant, not physiological parameter
LIGHT_M_S = 299792458.0     # exact SI defining constant

@dataclass(frozen=True)
class Aperture:
    convention: str
    spatial_weights: Quantity
    area_quadrature: Quantity
    effective_area: Quantity
    source_id: str
    scope: str

@dataclass(frozen=True)
class QuantumEfficiencyProfile:
    efficiency: Quantity
    source_id: str
    scope: str
    included_factors: tuple
    already_in_irradiance: tuple
    definition: str

def _area(q, meaning):
    if q.meaning != meaning or q.unit not in ("m^2","mm^2"):
        raise ContractError(f"Expected area/{meaning}")
    return q.values*(1 if q.unit=="m^2" else 1e-6)

def integrate_cone_photoisomerization_rate(irradiance,wavelength_nm,quadrature_weight_nm,aperture,quantum_efficiency_profile):
    if aperture is None or quantum_efficiency_profile is None:
        raise MissingCalibrationError("Explicit aperture/A_eff and quantum-efficiency profile required")
    if not aperture.source_id or not aperture.scope or not quantum_efficiency_profile.source_id or not quantum_efficiency_profile.scope:
        raise MissingCalibrationError("Profiles require source identity and scope")
    if aperture.scope!=quantum_efficiency_profile.scope:
        raise MissingCalibrationError("Calibration profile scope mismatch")
    if not quantum_efficiency_profile.definition or not quantum_efficiency_profile.included_factors:
        raise MissingCalibrationError("Efficiency chain must be explicitly defined")
    if set(quantum_efficiency_profile.included_factors)&set(quantum_efficiency_profile.already_in_irradiance):
        raise MissingCalibrationError("Efficiency double-counts factors already included in irradiance")
    E=irradiance.require("W/(m^2 nm)","retinal_spectral_irradiance")
    lam=wavelength_nm.require("nm","wavelength")
    dlam=quadrature_weight_nm.require("nm","spectral_quadrature_weight")
    if E.ndim==2: E=E[None,:,:]
    if E.ndim!=3 or lam.ndim!=1 or dlam.shape!=lam.shape or E.shape[-1]!=len(lam):
        raise ContractError("Spectrum/wavelength/quadrature shape mismatch")
    if np.any(E<0) or np.any(lam<=0) or np.any(dlam<=0) or np.any(np.diff(lam)<=0):
        raise ContractError("Nonnegative spectrum and strictly increasing positive wavelengths/weights required")
    nc,ns,nl=E.shape
    dA=_area(aperture.area_quadrature,"spatial_quadrature_weight")
    area=_area(aperture.effective_area,"effective_collection_area")
    weights=aperture.spatial_weights.values
    try:
        dA=np.broadcast_to(dA,(nc,ns)); weights=np.broadcast_to(weights,(nc,ns)); area=np.broadcast_to(area,(nc,))
    except ValueError as e: raise ContractError("Aperture shape mismatch") from e
    if np.any(dA<=0) or np.any(area<=0) or np.any(weights<0): raise ContractError("Positive areas and nonnegative aperture required")
    if aperture.convention=="normalized_mean":
        if aperture.spatial_weights.meaning!="normalized_aperture_weight" or aperture.spatial_weights.unit not in ("m^-2","mm^-2"):
            raise ContractError("Normalized weights must have inverse-area units")
        weights=weights*(1 if aperture.spatial_weights.unit=="m^-2" else 1e6)
        if not np.allclose(np.sum(weights*dA,axis=1),1,rtol=1e-12,atol=1e-12):
            raise ContractError("Normalized aperture integral != 1")
        P=np.sum(E*weights[:,:,None]*dA[:,:,None],axis=1)*area[:,None]
    elif aperture.convention=="dimensionless_mask":
        aperture.spatial_weights.require("dimensionless","aperture_mask")
        if np.any(weights>1): raise ContractError("Collection-mask fraction outside [0,1]")
        if not np.allclose(np.sum(weights*dA,axis=1),area,rtol=1e-12,atol=0):
            raise ContractError("Mask integral != declared A_eff")
        P=np.sum(E*weights[:,:,None]*dA[:,:,None],axis=1)
    else: raise ContractError("Explicit aperture convention required")
    eta=quantum_efficiency_profile.efficiency.require("dimensionless","photoisomerization_efficiency")
    try: eta=np.broadcast_to(eta,(nc,nl))
    except ValueError as e: raise ContractError("Efficiency spectrum shape mismatch") from e
    if np.any(eta<0) or np.any(eta>1): raise ContractError("Efficiency outside [0,1]")
    quantities=(wavelength_nm,quadrature_weight_nm,aperture.spatial_weights,aperture.area_quadrature,aperture.effective_area,quantum_efficiency_profile.efficiency)
    if any(q.evidence_tag!=irradiance.evidence_tag for q in quantities):
        raise ContractError("Input evidence tags inconsistent")
    rate=np.sum(P*eta*(lam[None,:]*1e-9)/(PLANCK_J_S*LIGHT_M_S)*dlam[None,:],axis=1)
    return Quantity(rate,"R*/s","photoisomerization_rate",irradiance.evidence_tag)

def radiance_to_retinal_irradiance(*args,**kwargs):
    raise MissingCalibrationError("Optics/pupil/geometry/spectral profile unavailable; radiance cannot be forced into irradiance")

def rgb_to_photoisomerization_rate(*args,**kwargs):
    raise MissingCalibrationError("ERROR_ABSOLUTE_PHOTOMETRY_UNAVAILABLE: no RGB/display calibration")
