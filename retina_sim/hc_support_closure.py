"""Fine-parent Cone/HC dependency closure; never normalize a cropped operator."""
import hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from .contracts import ContractError, canonical_hash
from .h1_h2_effective import matrix_id


def native_dependency_closure(A, B, seed_cones, seed_hcs, *,
                              hc_support_complete, cone_feedback_complete,
                              cone_input_available):
    """Trace C -> local B partners -> effective A inputs to a fixed point.

    Certificates are per-fine-cell masks supplied by the geometry/input owner.
    A zero B row is missing feedback, never an implicit feedback-off control.
    This certifies the finite declared graph only, not the infinite retina.
    """
    A, B = (sparse.csr_matrix(m, dtype=float, copy=True) for m in (A, B))
    if B.shape != A.shape[::-1]:
        raise ContractError('A/B parent dimensions differ')
    for m in (A, B):
        m.sum_duplicates(); m.eliminate_zeros(); m.sort_indices()
        if not np.isfinite(m.data).all() or np.any(m.data < 0):
            raise ContractError('Finite nonnegative parent weights required')
    nh, nc = A.shape
    def mask(value, n):
        value = np.asarray(value)
        if value.shape != (n,) or value.dtype != np.bool_:
            raise ContractError('Explicit per-fine-cell boolean certificate required')
        return value.copy()
    hs = mask(hc_support_complete, nh)
    cs = mask(cone_feedback_complete, nc)
    inputs = mask(cone_input_available, nc)
    def selection(rows, n):
        rows = np.asarray(rows)
        if rows.ndim != 1 or rows.dtype.kind not in 'iu' or np.any(rows < 0) or np.any(rows >= n):
            raise ContractError('Explicit valid parent row IDs required')
        result = np.zeros(n, dtype=bool); result[rows] = True
        return result
    cones, hcs = selection(seed_cones, nc), selection(seed_hcs, nh)
    if not cones.any() and not hcs.any():
        raise ContractError('At least one requested output row required')
    initial_c, initial_h = cones.copy(), hcs.copy()
    visited_c, visited_h = np.zeros(nc, bool), np.zeros(nh, bool)
    history = []
    while True:
        frontier_c = cones & ~visited_c
        hcs[B[frontier_c].indices] = True
        visited_c |= frontier_c
        frontier_h = hcs & ~visited_h
        cones[A[frontier_h].indices] = True
        visited_h |= frontier_h
        history.append(dict(cones=int(cones.sum()), HC=int(hcs.sum())))
        if np.array_equal(cones, visited_c) and np.array_equal(hcs, visited_h):
            break
    ci, hi = np.flatnonzero(cones), np.flatnonzero(hcs)
    amass, bmass = np.asarray(A.sum(1)).ravel(), np.asarray(B.sum(1)).ravel()
    missing = dict(HC_support=hi[~hs[hi]], cone_feedback=ci[~cs[ci]],
        cone_input=ci[~inputs[ci]],
        HC_nonunit_fine_row=hi[~np.isclose(amass[hi], 1., rtol=0, atol=2e-14)],
        cone_nonunit_fine_row=ci[~np.isclose(bmass[ci], 1., rtol=0, atol=2e-14)])
    identity = canonical_hash(dict(A=matrix_id(A), B=matrix_id(B), seeds_c=initial_c,
        seeds_h=initial_h, hc_support=hs, cone_feedback=cs, cone_input=inputs,
        source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    return dict(cone_rows=ci, hc_rows=hi, A=A[hi][:,ci].tocsr(), B=B[ci][:,hi].tocsr(),
        missing=missing, history=history, identity=identity,
        ready=not any(len(v) for v in missing.values()),
        scope='exact finite-parent fine graph; physical/input certificates external',
        crop_renormalized=False)


def require_closed_native(result):
    """Reject incomplete support before constructing a background or runner."""
    if not result['ready'] or any(len(v) for v in result['missing'].values()):
        counts = {k: len(v) for k,v in result['missing'].items() if len(v)}
        raise ContractError('Native support/feedback/input closure incomplete: '+str(counts))
    return result


def require_finite_boundary_native(binding, *, dynamic_feedback_complete,
                                   dynamic_input_available, background_max_rhs,
                                   background_tolerance):
    """Separate experimental admission from strict closure and response qualification.

    Exterior A/B products permit held background, but cannot conceal a missing
    local feedback contact of a dynamic cone. Passing permits a convergence
    experiment only; it never grants production or physiological qualification.
    """
    nc = binding['B'].shape[0]
    for name, value in (('feedback', dynamic_feedback_complete), ('input', dynamic_input_available)):
        a = np.asarray(value)
        if a.shape != (nc,) or a.dtype != np.bool_ or not a.all():
            raise ContractError('Finite boundary dynamic '+name+' incomplete')
    if np.any(binding['mass_ledger']['B_missing_source_mass'] > 0):
        raise ContractError('Finite boundary cannot replace missing dynamic local feedback')
    if (not np.isfinite([background_max_rhs, background_tolerance]).all()
            or background_tolerance <= 0 or not 0 <= background_max_rhs <= background_tolerance):
        raise ContractError('Finite boundary background residual failed')
    return dict(binding_identity=binding['identity'], experimental_admission=True,
                strict_closure=False, response_convergence_qualified=False,
                production_adopted=False)
