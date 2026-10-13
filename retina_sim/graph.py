from dataclasses import dataclass
import numpy as np
from scipy import sparse
from .contracts import ContractError, canonical_hash, finite_array

@dataclass(frozen=True)
class ConnectionTable:
    edge_id: tuple
    pre_cell_id: tuple
    post_cell_id: tuple
    synapse_type: tuple
    weight: tuple
    weight_unit: tuple
    sign_convention: tuple
    delay_seconds: tuple
    source_profile_id: tuple
    normalization_rule: tuple
    evidence_tags: tuple

    def __post_init__(self):
        n = len(self.edge_id)
        if len(set(self.edge_id)) != n: raise ContractError("Duplicate edge ID")
        for name in self.__dataclass_fields__:
            if len(getattr(self, name)) != n: raise ContractError(f"Edge field length mismatch: {name}")
        if any(not np.isfinite(w) for w in self.weight): raise ContractError("Nonfinite edge weight")
        if any(d is not None and (not np.isfinite(d) or d < 0) for d in self.delay_seconds):
            raise ContractError("Invalid sourced delay")
        if any(not x for name in ("synapse_type", "weight_unit", "sign_convention", "source_profile_id", "normalization_rule", "evidence_tags") for x in getattr(self, name)):
            raise ContractError("Explicit edge semantics and provenance required")

    def to_csr(self, cell_ids):
        if len(set(cell_ids)) != len(cell_ids): raise ContractError("Duplicate cell IDs")
        index = {v: i for i, v in enumerate(cell_ids)}
        if any(i not in index for i in self.pre_cell_id+self.post_cell_id):
            raise ContractError("Edge references unknown cell ID")
        pairs = list(zip(self.post_cell_id, self.pre_cell_id))
        if len(set(pairs)) != len(pairs):
            raise ContractError("Multiple edges at same pair require separate synapse matrices")
        matrix = sparse.coo_matrix((self.weight, ([index[x] for x in self.post_cell_id], [index[x] for x in self.pre_cell_id])), shape=(len(index),len(index))).tocsr()
        return matrix  # rows=postsynaptic, columns=presynaptic; no new edges/cutoff

    @property
    def graph_hash(self): return canonical_hash(self)

def row_normalize(matrix):
    a = np.array(matrix, dtype=np.float64, copy=True)
    if a.ndim != 2 or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ContractError("Weights must be finite, nonnegative, 2-D")
    totals = a.sum(axis=1)
    if np.any(totals <= 0): raise ContractError("Zero effective presynaptic row")
    return a/totals[:,None]

@dataclass(frozen=True)
class ReferenceGraph:
    w_ch: object
    w_hc: object
    w_off: object
    w_on: object
    cone_types: tuple
    cone_locations_deg: np.ndarray
    ganglion_locations_deg: np.ndarray
    eccentricity_functional_deg: float
    radii: dict
    evidence_tag: str
    profile_id: str
    source_commit: str

    def __post_init__(self):
        n = len(self.cone_types)
        locations = finite_array(self.cone_locations_deg, (n,2))
        gang = finite_array(self.ganglion_locations_deg)
        if gang.ndim != 2 or gang.shape[1] != 2 or not len(gang): raise ContractError("Ganglion locations shape")
        object.__setattr__(self, "cone_locations_deg", locations)
        object.__setattr__(self, "ganglion_locations_deg", gang)
        for name, shape in (("w_ch",(n,n)),("w_hc",(n,n)),("w_off",(len(gang),n)),("w_on",(len(gang),n))):
            a = sparse.csr_matrix(getattr(self,name), dtype=np.float64, copy=True)
            if a.shape != shape or np.any(a.data < 0) or not np.all(np.isfinite(a.data)):
                raise ContractError(f"Invalid {name} shape/weights")
            if not np.allclose(np.asarray(a.sum(axis=1)).ravel(),1,rtol=0,atol=5e-15):
                raise ContractError(f"{name} rows not normalized")
            object.__setattr__(self,name,a)
        if not self.evidence_tag: raise ContractError("Graph evidence tag required")

    @property
    def graph_hash(self):
        return canonical_hash({"matrices":{name:getattr(self,name).toarray() for name in ("w_ch","w_hc","w_off","w_on")},"cone_types":self.cone_types,"cone_locations":self.cone_locations_deg,"ganglion_locations":self.ganglion_locations_deg,"e":self.eccentricity_functional_deg,"radii":self.radii,"profile":self.profile_id,"evidence":self.evidence_tag,"commit":self.source_commit})
