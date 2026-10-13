"""Somaratna & Freeman 2025 Gaussian-HC source-specific linear kernel.

Adapted from AlanFreeman/Colour commit 5406d153..., CC BY 4.0.
Changes: Python/float64, algebraic feedback output, explicit streaming state,
fail-closed semantics and synthetic geometry tests. Not author-array reproduction.
"""
from pathlib import Path
import hashlib
import numpy as np
from scipy.linalg import expm
from .contracts import ContractError, UnsupportedReferenceBranch, Quantity, StateSpec, RunMode, SimulationResult, canonical_hash
from .graph import ReferenceGraph,row_normalize
from .provenance import CoverageReport
from .parameter_contracts import SF_PARAMETER_SPECS

COMMIT="5406d1533dae1678f94687379ac078a612f6170f"
LIMITS=("Source-specific cross-species literature reference; not a macaque main default",
        "Fixed-tau linear deviations with output rectification; no luminance-dependent phototransduction/adaptation",
        "H1/midget ON/OFF only; no AC, other RGC types, full L/M/S rules, anatomical displacement or continuous 2-D whole retina",
        "Author MAT MCOS tables not decoded; author arrays and paper reproduction not completed",
        "No spike-time/Poisson/Bernoulli/refractory mechanism")

def _values(parameters):
    values={}; seen=set()
    for p in parameters.records:
        if p.parameter_id not in SF_PARAMETER_SPECS:
            raise ContractError("Unknown author parameter interface: "+p.parameter_id)
        unit,shape=SF_PARAMETER_SPECS[p.parameter_id]
        a=np.asarray(p.value,dtype=np.float64)
        if p.unit!=unit or tuple(p.shape)!=shape or a.shape!=shape or not np.all(np.isfinite(a)):
            raise ContractError("Author parameter unit/shape mismatch: "+p.parameter_id)
        if p.parameter_id in seen: raise ContractError("Duplicate author parameter")
        seen.add(p.parameter_id)
        values[p.parameter_id.removeprefix("SF2025_")]=a
    return values

def build_somaratna_weights(cone_locations_deg,cone_types,ganglion_locations_deg,eccentricity_functional_deg,profile,evidence_tag):
    if profile.profile.kind!="named_reference" or profile.run_mode==RunMode.FULL_PHYSIOLOGY:
        raise ContractError("Author weights require isolated named reference profile")
    loc=cone_locations_deg.require("deg","functional_position",frame="local_patch")
    gang=ganglion_locations_deg.require("deg","functional_position",frame="local_patch")
    e=eccentricity_functional_deg.require("deg","global_eccentricity",(),"functional_visual_field").item()
    if loc.ndim!=2 or loc.shape[1]!=2 or gang.ndim!=2 or gang.shape[1]!=2 or len(cone_types)!=len(loc):
        raise ContractError("Reference geometry shape mismatch")
    if not np.isfinite(e) or e<0 or any(x not in ("L","M","S") for x in cone_types):
        raise ContractError("Scalar functional eccentricity and L/M/S labels required")
    if any(q.evidence_tag!=evidence_tag for q in (cone_locations_deg,ganglion_locations_deg,eccentricity_functional_deg)):
        raise ContractError("Graph geometry evidence mismatch")
    if profile.run_mode==RunMode.NUMERIC_FIXTURE and evidence_tag!="synthetic_fixture":
        raise ContractError("Numerical fixtures must be labeled synthetic_fixture")
    p=_values(profile)
    radii={"rOpt":float(np.polyval(p["radOptCoef"],e)),"rHor":float(np.exp(np.polyval(p["radHorCoef"],e))),"rGang":float(np.polyval(p["radGangCoef"],e))}
    if any(not np.isfinite(r) or r<=0 for r in radii.values()): raise ContractError("Nonpositive author radius; unsupported evaluation")
    def attenuation(post,pre,r):
        d2=np.sum((post[:,None,:]-pre[None,:,:])**2,axis=2)
        a=np.exp(-d2/r**2)
        if np.any(a==0): raise ContractError("Gaussian underflow would remove full-kernel edges; no implicit cutoff permitted")
        return a
    a=attenuation(loc,loc,radii["rHor"])
    excluded=np.asarray(cone_types)=="S"
    ch=a.copy(); ch[:,excluded]=0
    off=attenuation(gang,loc,radii["rGang"])
    on=off.copy(); on[:,excluded]=0
    return ReferenceGraph(row_normalize(ch),row_normalize(a.T),row_normalize(off),row_normalize(on),tuple(cone_types),loc,gang,e,radii,evidence_tag,profile.profile.profile_id,COMMIT)

class SomaratnaFreeman2025GaussianHCReference:
    model_id="SomaratnaFreeman2025GaussianHCReference"
    version="0.1.0"
    def __init__(self,parameters,graph,*,explicit_opt_in=False,horExp=False,adaptation=None):
        if horExp: raise UnsupportedReferenceBranch("horExp=true time/frequency branches are not equivalent; Gaussian-HC only")
        if adaptation is not None: raise UnsupportedReferenceBranch("No adaptation option supported by this fixed-tau reference")
        if not explicit_opt_in or parameters.profile.kind!="named_reference":
            raise ContractError("Explicit isolated named-reference opt-in required")
        if parameters.run_mode==RunMode.FULL_PHYSIOLOGY: raise ContractError("Cross-species reference forbidden as full macaque model")
        if graph.profile_id!=parameters.profile.profile_id or graph.source_commit!=COMMIT:
            raise ContractError("Reference graph/profile/source identity mismatch")
        self.parameters=parameters; self.graph=graph; self.evidence_tag=graph.evidence_tag
        self.p=_values(parameters)
        for name in ("tau","kSens","kSur","kRect","actRest","radOptCoef","radHorCoef","radGangCoef"):
            if name not in self.p: raise ContractError(f"Missing author parameter {name}")
        self.tau=float(self.p["tau"]); self.kSens=float(self.p["kSens"]); self.kSur=float(self.p["kSur"]); self.kRect=float(self.p["kRect"])
        if any(x<=0 for x in (self.tau,self.kSens,self.kSur,self.kRect)) or self.p["actRest"].shape!=(2,):
            raise ContractError("Invalid reference scalars/offsets")
        self.gRest=self.p["actRest"]/self.kRect
        n=len(graph.cone_types); m=len(graph.ganglion_locations_deg)
        self.sizes=(n,n,n,n,m,m)
        self.state_specs=tuple(StateSpec(name,tuple(f"{name}:{i}" for i in range(size)),(size,),"mV","generator_with_output_offset" if name.startswith("g_") else "generator_deviation","runColFun.m solveT lines 473-481; fixed commit "+COMMIT) for name,size in zip(("cone","horizontal","b_off","b_on","g_off","g_on"),self.sizes))

    @property
    def model_hash(self):
        return canonical_hash({"id":self.model_id,"version":self.version,"source_commit":COMMIT,"implementation_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"limits":LIMITS,"branch":"Gaussian-HC"})

    def unpack(self,y):
        return tuple(np.split(np.asarray(y),np.cumsum(self.sizes)[:-1]))

    def initial_conditions(self):
        return tuple(Quantity(np.full(size,self.gRest[i-4]) if i>=4 else np.zeros(size),"mV",spec.meaning,self.evidence_tag) for i,(spec,size) in enumerate(zip(self.state_specs,self.sizes)))

    def rhs(self,t_seconds,state_vector,stimulus,parameters,graph):
        if parameters is not self.parameters or graph is not self.graph: raise ContractError("RHS parameter/graph mismatch")
        u=stimulus.require("dimensionless","cone_contrast",(self.sizes[0],))
        if stimulus.evidence_tag!=self.evidence_tag: raise ContractError("RHS stimulus provenance mismatch")
        y=np.asarray(state_vector,dtype=float)
        if y.shape!=(sum(self.sizes),) or not np.all(np.isfinite(y)): raise ContractError("RHS state shape/nonfinite")
        c,h,bo,bn,go,gn=self.unpack(y)
        derivatives=((-self.kSens*u-self.kSur*graph.w_hc@h-c)/self.tau,
                     (graph.w_ch@c-h)/self.tau,(c-bo)/self.tau,(-c-bn)/self.tau,
                     (graph.w_off@bo+self.gRest[0]-go)/self.tau,
                     (graph.w_on@bn+self.gRest[1]-gn)/self.tau)
        return Quantity(np.concatenate(derivatives),"mV/s","generator_derivative",self.evidence_tag)

    def make_result(self,times,trajectory,input_contract,numeric_checks):
        blocks=np.split(trajectory,np.cumsum(self.sizes)[:-1],axis=0)
        output={spec.state_name:Quantity(a,spec.unit,spec.meaning,self.evidence_tag) for spec,a in zip(self.state_specs,blocks)}
        output["time"]=Quantity(times,"s","absolute_time",self.evidence_tag)
        output["feedback_output"]=Quantity(self.kSur*(self.graph.w_hc@blocks[1]),"mV","algebraic_feedback",self.evidence_tag)
        output["rate_off"]=Quantity(self.kRect*np.maximum(0,blocks[4]),"Hz","firing_rate",self.evidence_tag)
        output["rate_on"]=Quantity(self.kRect*np.maximum(0,blocks[5]),"Hz","firing_rate",self.evidence_tag)
        return SimulationResult(self.parameters.run_mode,self.model_id,self.parameters.source_manifest_hash,input_contract,output,CoverageReport(True,(),claim_limit="; ".join(LIMITS)),numeric_checks,"numeric_consistency_only" if self.evidence_tag=="synthetic_fixture" else "isolated_source_specific_reference",{"source_commit":COMMIT,"cross_species_limitations":list(LIMITS),"graph_evidence_tag":self.evidence_tag,"parameter_profile":self.parameters.profile.profile_id,"claim_limit":"Not a complete or physiologically validated macaque retina"})

    def oracle_matrices(self):
        # Independent block assembly in shifted z=g-gRest coordinates.
        n,n2,_,_,mo,mn=self.sizes; total=sum(self.sizes)
        offsets=np.cumsum((0,)+self.sizes)
        A=np.zeros((total,total)); B=np.zeros((total,n))
        for k,size in enumerate(self.sizes): A[offsets[k]:offsets[k+1],offsets[k]:offsets[k+1]]=-np.eye(size)
        A[offsets[0]:offsets[1],offsets[1]:offsets[2]]=-self.kSur*self.graph.w_hc.toarray()
        A[offsets[1]:offsets[2],offsets[0]:offsets[1]]=self.graph.w_ch.toarray()
        A[offsets[2]:offsets[3],offsets[0]:offsets[1]]=np.eye(n)
        A[offsets[3]:offsets[4],offsets[0]:offsets[1]]=-np.eye(n)
        A[offsets[4]:offsets[5],offsets[2]:offsets[3]]=self.graph.w_off.toarray()
        A[offsets[5]:offsets[6],offsets[3]:offsets[4]]=self.graph.w_on.toarray()
        B[:n,:]=-self.kSens*np.eye(n)
        return A/self.tau,B/self.tau

    def zoh_oracle(self,y,contrast,dt_seconds):
        if not np.isfinite(dt_seconds) or dt_seconds<=0: raise ContractError("Positive oracle step required")
        u=contrast.require("dimensionless","cone_contrast",(self.sizes[0],))
        A,B=self.oracle_matrices(); size=A.shape[0]; nc=B.shape[1]
        augmented=np.zeros((size+nc,size+nc)); augmented[:size,:size]=A; augmented[:size,size:]=B
        discrete=expm(dt_seconds*augmented)
        shifted=np.array(y,dtype=float,copy=True)
        shifted[-sum(self.sizes[-2:]): -self.sizes[-1]]-=self.gRest[0]; shifted[-self.sizes[-1]:]-=self.gRest[1]
        z=discrete[:size,:size]@shifted+discrete[:size,size:]@u
        z[-sum(self.sizes[-2:]): -self.sizes[-1]]+=self.gRest[0]; z[-self.sizes[-1]:]+=self.gRest[1]
        return z

    def frequency_oracle(self,u_hat,frequency_hz):
        if not np.isfinite(frequency_hz) or frequency_hz<0: raise ContractError("Nonnegative frequency required")
        u=np.asarray(u_hat,dtype=complex)
        if u.shape!=(self.sizes[0],): raise ContractError("Frequency stimulus shape")
        C=1/(1+1j*self.tau*2*np.pi*frequency_hz)
        c=np.linalg.solve(np.eye(self.sizes[0])+self.kSur*C**2*(self.graph.w_hc@self.graph.w_ch).toarray(),-self.kSens*C*u)
        h=C*(self.graph.w_ch@c); bo=C*c; bn=-C*c
        return np.concatenate((c,h,bo,bn,C*(self.graph.w_off@bo),C*(self.graph.w_on@bn)))
