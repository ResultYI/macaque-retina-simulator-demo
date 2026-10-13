from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import csv, hashlib
import numpy as np
from .contracts import ContractError

@dataclass(frozen=True)
class MeridianDensityTable:
    rows: tuple
    asset_sha256: str
    evidence_class: str = "published_code_tabulation"

@dataclass(frozen=True)
class DensityQueryResult:
    value: float
    unit: str
    meridian: str
    bracketing_source_rows: tuple
    interpolation_provenance: dict
    exact_source_literal: str | None
    evidence_class: str = "published_code_tabulation"

@dataclass(frozen=True)
class MagnificationProfile:
    value: float
    unit: str
    source_id: str
    source_locator: str
    scope: str
    def __post_init__(self):
        if not np.isfinite(self.value) or self.value <= 0 or self.unit != "deg/mm" or not self.source_id or not self.source_locator or not self.scope:
            raise ContractError("Explicit sourced positive magnification in deg/mm required")

def read_density_tabulation(asset,expected_hash):
    raw=Path(asset).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_hash: raise ContractError("Density asset hash mismatch")
    rows=tuple(csv.DictReader(raw.decode().splitlines()))
    if len(rows)!=60 or sum(r["meridian"]=="temporal" for r in rows)!=28 or sum(r["meridian"]=="nasal" for r in rows)!=32:
        raise ContractError("Density source counts mismatch")
    counts={(m,s):sum(r["meridian"]==m and r["source_segment"]==s for r in rows) for m in ("temporal","nasal") for s in ("foveal","peripheral")}
    if counts!={("temporal","foveal"):13,("temporal","peripheral"):15,("nasal","foveal"):13,("nasal","peripheral"):19}:
        raise ContractError("Density segment counts mismatch")
    for r in rows:
        if r["evidence_class"]!="published_code_tabulation" or Decimal(r["eccentricity_mm"])!=Decimal(r["source_eccentricity_literal"]) or Decimal(r["cone_density_cones_per_mm2"])!=Decimal(r["source_density_literal"])*Decimal(r["density_multiplier"]):
            raise ContractError("Density literal/transform mismatch")
    for m in ("temporal","nasal"):
        e=[Decimal(r["eccentricity_mm"]) for r in rows if r["meridian"]==m]
        if e[0]<=0 or any(a>=b for a,b in zip(e,e[1:])): raise ContractError("Invalid density support")
    return MeridianDensityTable(rows,expected_hash)

def _bracket(table,e,meridian):
    if meridian not in ("temporal","nasal"): raise ContractError("Unsupported meridian; no 2-D inference")
    rows=[r for r in table.rows if r["meridian"]==meridian]
    x=np.array([float(r["eccentricity_mm"]) for r in rows])
    if not np.isfinite(e) or e<x[0] or e>x[-1]: raise ContractError("Outside tabulated support; no e=0 or extrapolation")
    k=int(np.searchsorted(x,e))
    if k<len(x) and x[k]==e: return rows,x,k,k
    return rows,x,k-1,k

def interpolate_meridian_density(table,eccentricity_mm,meridian,method):
    if method!="computational_linear_interpolation": raise ContractError("Explicit computational interpolation method required")
    rows,x,i,j=_bracket(table,float(eccentricity_mm),meridian)
    y=np.array([float(r["cone_density_cones_per_mm2"]) for r in rows])
    f=0 if i==j else (eccentricity_mm-x[i])/(x[j]-x[i])
    value=float(y[i] if i==j else y[i]*(1-f)+y[j]*f)
    return DensityQueryResult(value,"cones/mm^2",meridian,(rows[i],rows[j]),{"method":method,"asset_sha256":table.asset_sha256,"fraction":float(f),"claim":"computational interpolation only"},rows[i]["cone_density_cones_per_mm2"] if i==j else None)

def density_per_degree_squared(density_per_mm2,magnification_profile):
    return np.asarray(density_per_mm2)/magnification_profile.value**2

def evaluate_packer_rc_reference(table,eccentricity_deg,meridian,magnification_profile):
    # The authored order is density -> triangular-lattice spacing -> Rc,
    # THEN interp1 on Rc. Do not interpolate density first.
    e_mm=float(eccentricity_deg)/magnification_profile.value
    rows,x,i,j=_bracket(table,e_mm,meridian)
    density=np.array([float(r["cone_density_cones_per_mm2"]) for r in rows])
    spacing_mm=np.sqrt(2/(np.sqrt(3)*density))
    rc_deg=spacing_mm*magnification_profile.value*0.204*np.sqrt(2)
    f=0 if i==j else (e_mm-x[i])/(x[j]-x[i])
    return DensityQueryResult(float(rc_deg[i] if i==j else (1-f)*rc_deg[i]+f*rc_deg[j]),"deg",meridian,(rows[i],rows[j]),{"method":"computational_linear_interpolation","order":"density->spacing->Rc->interpolate_Rc","spacing_geometry":"triangular lattice; WatsonRGCModel.spacingFromDensity","magnification":magnification_profile.__dict__,"coefficient_source":"PackerConeRc.m lines 20-21","asset_sha256":table.asset_sha256},None)
