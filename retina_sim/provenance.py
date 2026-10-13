from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from .contracts import ContractError, RunMode, canonical_hash
from .parameter_contracts import SF_PARAMETER_SPECS, CHEN_PARAMETER_SPECS

@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    title: str
    authors: tuple
    publication_year: int | None
    doi_or_canonical_url: str
    artifact_kind: str
    repository_commit: str | None
    source_path: str
    source_locator: str
    asset_sha256: str
    source_blob_sha1: str | None
    license: str
    verification_level: str
    provenance_notes: str
    retrieved_at: str
    upstream_sources_checked: dict

    def verify(self, root):
        if any(not getattr(self,n) for n in ("source_id","title","authors","doi_or_canonical_url","source_path","source_locator","license","provenance_notes","retrieved_at")):
            raise ContractError(f"{self.source_id}: incomplete SourceRecord")
        if self.artifact_kind not in ("paper","code","data_table","supplement"):
            raise ContractError(f"{self.source_id}: unknown artifact kind")
        if self.artifact_kind == "code" and (not self.repository_commit or len(self.repository_commit) != 40):
            raise ContractError(f"{self.source_id}: fixed code commit required")
        if self.verification_level not in ("identity_verified","code_verified","numeric_table_verified","primary_measurement_verified"):
            raise ContractError(f"{self.source_id}: unknown verification level")
        path = Path(self.source_path)
        if not path.is_absolute(): path = Path(root)/path
        if not path.is_file(): raise ContractError(f"{self.source_id}: missing asset {path}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.asset_sha256:
            raise ContractError(f"{self.source_id}: bad asset SHA256")
        if self.source_blob_sha1 is not None:
            blob = hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
            if blob != self.source_blob_sha1: raise ContractError(f"{self.source_id}: bad Git blob SHA1")

@dataclass(frozen=True)
class ParameterRecord:
    parameter_id: str
    name: str
    value: object
    unit: str
    shape: tuple
    source_id: str | None
    source_locator: str
    derivation: object
    evidence_kind: str
    species: str
    sample_context: str
    retinal_region: str
    meridian: str
    coordinate_frame: str
    support_domain: dict | None
    conditions: dict
    uncertainty: str
    allowed_modes: tuple
    status: str
    assumption_approval_reference: str | None = None

@dataclass(frozen=True)
class Profile:
    profile_id: str
    species: str
    kind: str
    parameter_ids: tuple
    conditions: dict
    coordinate_frame: str
    explicit_opt_in: bool
    claim_limit: str

@dataclass(frozen=True)
class ParameterSet:
    records: tuple
    profile: Profile
    run_mode: RunMode
    source_manifest_hash: str

    @property
    def parameter_hash(self): return canonical_hash(self)

@dataclass(frozen=True)
class Blocker:
    parameter_id: str
    missing: str
    blocked_module: str
    needed_evidence: str

@dataclass(frozen=True)
class CoverageReport:
    allowed: bool
    blockers: tuple
    requirements: tuple = ()
    claim_limit: str = "No full physiological claim"

    def to_dict(self): return asdict(self)

class GateError(ContractError):
    def __init__(self, report):
        self.report = report
        super().__init__("; ".join(f"{b.parameter_id}: {b.missing} [{b.blocked_module}]" for b in report.blockers))

def validate_evidence_identity(source,root,evidence_class,locator,*,parameter_id=None,value=None,unit=None,context=None):
    """Shared static/dynamic identity gate; no automatic physiological validation."""
    source.verify(root)
    if not locator:raise ContractError('Concrete source locator required')
    if evidence_class in ('measurement','measured'):
        if source.verification_level!='primary_measurement_verified':
            raise ContractError('Asset identity is not primary measurement verification')
        verification=source.upstream_sources_checked.get('primary_verification',{})
        if not isinstance(verification,dict) or verification.get('verified') is not True or not all(verification.get(k) for k in ('material_path','material_sha256','locator','condition')):
            raise ContractError('Primary measurement verification material/locator/condition missing')
        material=Path(root)/verification['material_path']
        if not material.is_file() or hashlib.sha256(material.read_bytes()).hexdigest()!=verification['material_sha256']:
            raise ContractError('Primary verification material identity mismatch')
        proof=verification.get('parameters',{}).get(parameter_id,{})
        if proof.get('locator')!=locator or proof.get('unit')!=unit or proof.get('context')!=context or 'value' not in proof:
            raise ContractError('Primary verification does not cover this parameter locator/unit/condition')
        expected=np.asarray(proof['value'],dtype=float);actual=np.asarray(value,dtype=float)
        try: expected=np.broadcast_to(expected,actual.shape)
        except ValueError:raise ContractError('Primary verification value shape mismatch')
        if not np.array_equal(expected,actual):raise ContractError('Primary verification value mismatch')

def validate_parameters(parameters, requirements, sources, root, mode, profile):
    blockers=[]
    def block(p, reason):
        blockers.append(Blocker(p.parameter_id, reason, p.name, "Supply independently checked source, units, context and permitted profile"))
    if profile.kind not in ("macaque_main", "named_reference", "synthetic_fixture"):
        blockers.append(Blocker(profile.profile_id,"Unknown profile kind","profile","Declare mode and species policy"))
    if profile.kind == "named_reference" and not profile.explicit_opt_in:
        blockers.append(Blocker(profile.profile_id,"Named reference requires explicit opt-in","profile","Explicit component selection"))
    if mode == RunMode.FULL_PHYSIOLOGY and (profile.kind != "macaque_main" or not profile.species.startswith("M.")):
        blockers.append(Blocker(profile.profile_id,"Full mode requires a source-separated macaque species profile","profile","Verified species-specific profile"))
    if profile.kind == "macaque_main" and profile.species in ("human","cat","source_specific_mixed","unknown"):
        blockers.append(Blocker(profile.profile_id,"Non-macaque/mixed defaults forbidden","profile","Separate macaque profiles"))
    known={p.parameter_id:p for p in parameters}
    for requirement in requirements:
        if requirement not in known:
            blockers.append(Blocker(requirement,"Missing required parameter","profile","Register required sourced parameter"))
    for p in parameters:
        if p.value is None or p.evidence_kind == "unknown" or p.status == "blocked": block(p,"Unknown/null/blocked parameter")
        if not p.unit: block(p,"Missing unit")
        interfaces={**SF_PARAMETER_SPECS,**CHEN_PARAMETER_SPECS}
        if p.parameter_id in interfaces:
            expected_unit, expected_shape = interfaces[p.parameter_id]
            if p.unit != expected_unit or tuple(p.shape) != expected_shape:
                block(p,"Parameter interface unit/shape mismatch")
        if p.evidence_kind not in ("measured","source_fit","author_assumption","project_assumption","synthetic_fixture","unknown"):
            block(p,"Unknown evidence kind")
        if p.value is not None:
            try:
                a=np.asarray(p.value,dtype=float)
                if a.shape != tuple(p.shape) or not np.all(np.isfinite(a)): block(p,"Invalid value/shape")
            except (ValueError,TypeError): block(p,"Numeric parameter expected")
        if not p.source_id or p.source_id not in sources: block(p,"Missing source")
        else:
            try: validate_evidence_identity(sources[p.source_id],root,p.evidence_kind,p.source_locator,
                    parameter_id=p.parameter_id,value=p.value,unit=p.unit,context=p.conditions)
            except ContractError as e: block(p,str(e))
            if p.evidence_kind == "measured" and sources[p.source_id].verification_level != "primary_measurement_verified":
                block(p,"Code/asset identity is not primary-measurement verification")
        if not p.source_locator or not p.derivation or not p.uncertainty or not p.sample_context or not p.retinal_region or not p.meridian:
            block(p,"Missing source locator/context/derivation/uncertainty")
        if mode.value not in p.allowed_modes: block(p,"Mode not permitted")
        if p.coordinate_frame != profile.coordinate_frame: block(p,"Coordinate frame mismatch")
        if p.conditions != profile.conditions: block(p,"Condition mismatch")
        if p.species != profile.species: block(p,"Species mismatch; averaging or silent substitution forbidden")
        if p.support_domain is not None:
            if not isinstance(p.support_domain, dict) or not p.support_domain:
                block(p,"Invalid empty/non-object support domain")
            else:
                for axis,domain in p.support_domain.items():
                    if not isinstance(domain,dict) or set(domain) != {"unit","lower","upper","lower_closed","upper_closed"}:
                        block(p,f"Invalid domain {axis}"); continue
                    limits = (domain["lower"], domain["upper"])
                    if not isinstance(domain["unit"],str) or not domain["unit"] or any(type(domain[k]) is not bool for k in ("lower_closed","upper_closed")):
                        block(p,f"Missing units/closure semantics for domain {axis}")
                    if any(x is not None and (type(x) not in (int,float) or not np.isfinite(x)) for x in limits):
                        block(p,f"Nonfinite/nonnumeric domain limits {axis}")
                    elif all(x is not None for x in limits) and limits[0] >= limits[1]:
                        block(p,f"Invalid domain limits {axis}")
        if mode == RunMode.FULL_PHYSIOLOGY:
            if p.evidence_kind == "synthetic_fixture": block(p,"Synthetic fixture forbidden in physiology")
            if p.status != "usable_physiology": block(p,"Parameter not permitted for physiology")
            if p.evidence_kind == "project_assumption" and not p.assumption_approval_reference: block(p,"Unapproved project assumption")
            if p.species in ("human","cat","source_specific_mixed","unknown"): block(p,"Non-macaque/mixed parameter forbidden in main profile")
            if p.support_domain is None: block(p,"Unknown support domain")
        if profile.kind == "macaque_main" and p.species != profile.species: block(p,"Main profile must separate species and source")
    return CoverageReport(not blockers,tuple(blockers))

class Registry:
    def __init__(self, root):
        self.root=Path(root); self.sources={}; self.parameters={}; self.profiles={}
    def register_source(self, record):
        record.verify(self.root)
        if record.source_id in self.sources: raise ContractError("Duplicate source ID")
        self.sources[record.source_id]=record
    def register_parameter(self, record):
        if record.parameter_id in self.parameters: raise ContractError("Duplicate parameter ID")
        self.parameters[record.parameter_id]=record
    def register_profile(self, profile):
        if profile.profile_id in self.profiles: raise ContractError("Duplicate profile ID")
        self.profiles[profile.profile_id]=profile
    def load_parameters(self, profile_id, mode):
        mode=RunMode(mode)
        if profile_id not in self.profiles: raise ContractError("Unknown profile ID; no default fallback")
        profile=self.profiles[profile_id]
        records=tuple(self.parameters[i] for i in profile.parameter_ids if i in self.parameters)
        report=validate_parameters(records,profile.parameter_ids,self.sources,self.root,mode,profile)
        if not report.allowed: raise GateError(report)
        return ParameterSet(records,profile,mode,canonical_hash(self.sources))
    @classmethod
    def read(cls,path,root=None):
        path=Path(path); data=json.loads(path.read_text(encoding="utf-8"))
        if data["schema_version"] != "2.0": raise ContractError("Unsupported registry version")
        registry=cls(root or path.parent.parent)
        for source in data["sources"]:
            source["authors"]=tuple(source["authors"]); registry.register_source(SourceRecord(**source))
        for p in data["parameters"]:
            p["shape"]=tuple(p["shape"]); p["allowed_modes"]=tuple(p["allowed_modes"]); registry.register_parameter(ParameterRecord(**p))
        for p in data["profiles"]:
            p["parameter_ids"]=tuple(p["parameter_ids"]); registry.register_profile(Profile(**p))
        return registry

def validate_runnable(request, requirements):
    mode=RunMode(request["mode"])
    blockers=[]
    if mode == RunMode.FULL_PHYSIOLOGY:
        for r in requirements:
            if r["gate"] not in ("PHYSIOLOGY_VALIDATED","POLICY_CONFIRMED","ENGINEERING_VALIDATED"):
                blockers.append(Blocker(r["requirement_id"],r["gate"]+": "+"; ".join(r["missing_evidence"]),r["required_module"],r["claim_limit"]))
    elif not request.get("explicit_component"):
        blockers.append(Blocker("component","Explicit component selection required","run","Choose isolated reference or numeric fixture"))
    return CoverageReport(not blockers,tuple(blockers),tuple(requirements),"Engineering numerical fixtures / named linear reference only")

def require_runnable(request, requirements):
    """Execution preflight raises with every blocker, without fallback to reference."""
    report = validate_runnable(request, requirements)
    if not report.allowed:
        raise GateError(report)
    return report
